#!/usr/bin/env python3
"""Test settlement on existing unsettled trades."""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from polymarket.config import Settings
from polymarket.performance.database import PerformanceDatabase
from polymarket.performance.tracker import PerformanceTracker
from polymarket.performance.settler import TradeSettler
from polymarket.trading.btc_price import BTCPriceService


async def main():
    """Run settlement dry run."""
    print("=== Trade Settlement Dry Run ===\n")

    # Initialize
    settings = Settings()
    db = PerformanceDatabase("data/performance.db")
    tracker = PerformanceTracker(db=db)
    btc_service = BTCPriceService(settings)

    settler = TradeSettler(db, btc_service)
    settler._tracker = tracker

    # Show current state
    cursor = db.conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM trades WHERE is_win IS NULL AND action IN ('YES', 'NO')")
    unsettled_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM trades WHERE is_win IS NOT NULL")
    settled_count = cursor.fetchone()[0]

    print(f"📊 Current Status:")
    print(f"   Unsettled trades: {unsettled_count}")
    print(f"   Already settled: {settled_count}")
    print()

    if unsettled_count == 0:
        print("✅ No trades need settlement!")
        return

    # Run settlement
    print(f"🔄 Running settlement on up to {unsettled_count} trades...\n")

    stats = await settler.settle_pending_trades(batch_size=100)

    # Show results
    print("\n=== Settlement Results ===")
    print(f"✅ Settled: {stats['settled_count']}")
    print(f"🏆 Wins: {stats['wins']}")
    print(f"❌ Losses: {stats['losses']}")
    print(f"💰 Total Profit: ${stats['total_profit']:.2f}")
    print(f"⏳ Still Pending: {stats['pending_count']}")

    if stats['errors']:
        print(f"\n⚠️  Errors: {len(stats['errors'])}")
        for error in stats['errors'][:5]:
            print(f"   - {error}")

    # Show final state
    cursor.execute("SELECT COUNT(*) FROM trades WHERE is_win IS NULL AND action IN ('YES', 'NO')")
    remaining = cursor.fetchone()[0]

    print(f"\n📊 Final Status:")
    print(f"   Remaining unsettled: {remaining}")

    if remaining == 0:
        print("\n🎉 All trades settled successfully!")
    else:
        print(f"\n⚠️  {remaining} trades could not be settled (likely BTC price unavailable)")

    # Close
    await btc_service.close()
    db.close()


if __name__ == "__main__":
    asyncio.run(main())
