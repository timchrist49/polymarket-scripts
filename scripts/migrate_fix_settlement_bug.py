#!/usr/bin/env python3
"""
Migration: Fix settlement bug - Recalculate outcomes with correct price_to_beat.

CRITICAL BUG: parse_market_start() was returning CLOSE time instead of START time,
causing price_to_beat to store the END price. This led to settlement comparing
END vs END, detecting no movement, and assigning inverted win/loss outcomes.

This script:
1. Recalculates correct START prices for all trades
2. Re-determines outcomes based on correct price comparison
3. Updates is_win and profit_loss for all settled trades
"""

import asyncio
import sqlite3
import re
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from polymarket.config import get_settings
from polymarket.trading.btc_price import BTCPriceService


def parse_market_window(slug: str) -> tuple[int, int]:
    """
    Extract start and close timestamps from market slug.

    Returns:
        (start_timestamp, close_timestamp)
    """
    match = re.search(r'(\d{10})$', slug)
    if not match:
        return None, None

    close_timestamp = int(match.group(1))
    start_timestamp = close_timestamp - 900  # 15 minutes before

    return start_timestamp, close_timestamp


def determine_outcome(btc_close_price: float, btc_start_price: float) -> str:
    """Determine which outcome won (YES or NO)."""
    if btc_close_price > btc_start_price:
        return "YES"  # UP won
    else:
        return "NO"   # DOWN won (includes tie)


def calculate_profit_loss(
    action: str,
    actual_outcome: str,
    position_size: float,
    executed_price: float
) -> tuple[float, bool]:
    """Calculate profit/loss for a settled trade."""
    shares = position_size / executed_price

    if (action == "YES" and actual_outcome == "YES") or \
       (action == "NO" and actual_outcome == "NO"):
        # Win - each share worth $1
        payout = shares * 1.00
        profit_loss = payout - position_size
        is_win = True
    else:
        # Loss - shares worth $0
        profit_loss = -position_size
        is_win = False

    return profit_loss, is_win


async def fix_settlement_outcomes(db_path: str = "data/performance.db", dry_run: bool = False):
    """
    Fix settlement outcomes for all trades affected by the bug.

    Args:
        db_path: Path to SQLite database
        dry_run: If True, show what would change without updating database
    """

    print("=" * 80)
    print("SETTLEMENT BUG FIX - Recalculating Outcomes")
    print("=" * 80)
    print(f"\nDatabase: {db_path}")
    print(f"Mode: {'DRY RUN (no changes)' if dry_run else 'LIVE (will update database)'}")
    print()

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Initialize BTC price service
    settings = get_settings()
    btc_service = BTCPriceService(settings)

    # Get all settled trades (those with is_win set)
    cursor.execute("""
        SELECT
            id, market_slug, action, position_size, executed_price,
            price_to_beat, actual_outcome, is_win, profit_loss
        FROM trades
        WHERE action IN ('YES', 'NO')
          AND is_win IS NOT NULL
        ORDER BY id
    """)

    trades = cursor.fetchall()

    if not trades:
        print("⚠️ No settled trades found.")
        return

    print(f"Found {len(trades)} settled trades to check\n")

    stats = {
        'total': len(trades),
        'incorrect': 0,
        'correct': 0,
        'failed_price_fetch': 0,
        'updated': 0
    }

    changes = []

    for trade in trades:
        trade_id, slug, action, pos_size, exec_price, old_price_to_beat, old_outcome, old_is_win, old_pnl = trade

        # Parse market window
        start_ts, close_ts = parse_market_window(slug)

        if not start_ts or not close_ts:
            print(f"⚠️ Trade #{trade_id}: Failed to parse slug: {slug}")
            stats['failed_price_fetch'] += 1
            continue

        # Fetch correct prices
        try:
            btc_start = await btc_service.get_price_at_timestamp(start_ts)
            btc_close = await btc_service.get_price_at_timestamp(close_ts)

            if not btc_start or not btc_close:
                print(f"⚠️ Trade #{trade_id}: Failed to fetch BTC prices")
                stats['failed_price_fetch'] += 1
                continue

            btc_start = float(btc_start)
            btc_close = float(btc_close)

        except Exception as e:
            print(f"⚠️ Trade #{trade_id}: Error fetching prices: {e}")
            stats['failed_price_fetch'] += 1
            continue

        # Calculate correct outcome
        new_outcome = determine_outcome(btc_close, btc_start)
        new_pnl, new_is_win = calculate_profit_loss(action, new_outcome, pos_size, exec_price)

        # Check if outcome changed
        if old_outcome != new_outcome or old_is_win != new_is_win:
            stats['incorrect'] += 1

            change = {
                'id': trade_id,
                'slug': slug,
                'action': action,
                'old_outcome': old_outcome,
                'new_outcome': new_outcome,
                'old_is_win': old_is_win,
                'new_is_win': new_is_win,
                'old_pnl': old_pnl,
                'new_pnl': new_pnl,
                'old_price_to_beat': old_price_to_beat,
                'new_price_to_beat': btc_start,
                'btc_start': btc_start,
                'btc_close': btc_close,
                'movement': btc_close - btc_start
            }
            changes.append(change)

            print(f"❌ Trade #{trade_id} | {slug}")
            print(f"   Action: {action}")
            print(f"   BTC: ${btc_start:.2f} → ${btc_close:.2f} ({change['movement']:+.2f})")
            print(f"   OLD: outcome={old_outcome}, is_win={old_is_win}, P&L=${old_pnl:.2f}")
            print(f"   NEW: outcome={new_outcome}, is_win={new_is_win}, P&L=${new_pnl:.2f}")
            print()

            if not dry_run:
                # Update database
                cursor.execute("""
                    UPDATE trades
                    SET price_to_beat = ?,
                        actual_outcome = ?,
                        is_win = ?,
                        profit_loss = ?
                    WHERE id = ?
                """, (btc_start, new_outcome, new_is_win, new_pnl, trade_id))
                stats['updated'] += 1
        else:
            stats['correct'] += 1

    if not dry_run and stats['updated'] > 0:
        conn.commit()

    conn.close()

    # Print summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total trades checked: {stats['total']}")
    print(f"Correct outcomes:     {stats['correct']} ✓")
    print(f"Incorrect outcomes:   {stats['incorrect']} ❌")
    print(f"Failed price fetch:   {stats['failed_price_fetch']}")

    if not dry_run:
        print(f"\nUpdated trades:       {stats['updated']} ✓")
        print("\n✅ Database updated successfully!")
    else:
        print(f"\nDRY RUN - Would update: {stats['incorrect']} trades")
        print("\n⚠️ Run without --dry-run to apply changes")

    print("=" * 80)

    return changes, stats


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Fix settlement bug in database")
    parser.add_argument('--dry-run', action='store_true',
                       help='Show changes without updating database')
    parser.add_argument('--db', default='data/performance.db',
                       help='Path to database (default: data/performance.db)')
    args = parser.parse_args()

    # Change to project root
    import os
    os.chdir(Path(__file__).parent.parent)

    asyncio.run(fix_settlement_outcomes(args.db, args.dry_run))


if __name__ == "__main__":
    main()
