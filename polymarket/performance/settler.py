"""Trade settlement service for determining win/loss outcomes."""

import re
import structlog
from datetime import datetime, timedelta
from typing import Dict, Optional
from decimal import Decimal

from polymarket.performance.database import PerformanceDatabase

logger = structlog.get_logger()


class TradeSettler:
    """Settles trades by comparing BTC prices at market close."""

    def __init__(self, db: PerformanceDatabase, btc_fetcher):
        """
        Initialize trade settler.

        Args:
            db: Performance database
            btc_fetcher: BTC price service (from auto_trade.py)
        """
        self.db = db
        self.btc_fetcher = btc_fetcher

    def _parse_market_close_timestamp(self, market_slug: str) -> Optional[int]:
        """
        Extract Unix timestamp from market slug.

        Args:
            market_slug: Format "btc-updown-15m-1770828300" or variations

        Returns:
            Unix timestamp or None if parsing fails
        """
        if not market_slug:
            return None

        # Pattern: any text ending with a 10-digit Unix timestamp
        # Unix timestamps are 10 digits for dates between 2001-2286
        match = re.search(r'(\d{10})$', market_slug)

        if match:
            return int(match.group(1))

        logger.warning(
            "Failed to parse timestamp from market slug",
            market_slug=market_slug
        )
        return None

    def _determine_outcome(
        self,
        btc_close_price: float,
        price_to_beat: float
    ) -> str:
        """
        Determine which outcome won (YES or NO).

        Args:
            btc_close_price: BTC price at market close
            price_to_beat: Baseline BTC price from cycle start

        Returns:
            "YES" if UP won, "NO" if DOWN won
        """
        if btc_close_price > price_to_beat:
            return "YES"  # UP won
        else:
            return "NO"   # DOWN won (includes tie)

    def _calculate_profit_loss(
        self,
        action: str,
        actual_outcome: str,
        position_size: float,
        executed_price: float
    ) -> tuple[float, bool]:
        """
        Calculate profit/loss for a settled trade.

        Polymarket binary mechanics:
        - Shares bought: position_size / executed_price
        - If win: Payout = shares × $1.00
        - If loss: Payout = $0

        Args:
            action: "YES" or "NO"
            actual_outcome: "YES" or "NO"
            position_size: Dollar amount invested
            executed_price: Price paid per share

        Returns:
            (profit_loss, is_win)
        """
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

    def _get_unsettled_trades(self, batch_size: int = 50) -> list[dict]:
        """
        Query unsettled trades from database.

        Args:
            batch_size: Maximum number of trades to return

        Returns:
            List of trade records as dicts
        """
        cursor = self.db.conn.cursor()

        # Query trades that:
        # 1. Have action YES or NO (not HOLD)
        # 2. Are not yet settled (is_win IS NULL)
        # 3. Are old enough (>15 minutes old)
        cursor.execute("""
            SELECT
                id, timestamp, market_slug, action,
                position_size, executed_price, price_to_beat
            FROM trades
            WHERE action IN ('YES', 'NO')
              AND is_win IS NULL
              AND datetime(timestamp) < datetime('now', '-15 minutes')
            ORDER BY timestamp ASC
            LIMIT ?
        """, (batch_size,))

        # Convert to list of dicts
        trades = []
        for row in cursor.fetchall():
            trades.append({
                'id': row[0],
                'timestamp': row[1],
                'market_slug': row[2],
                'action': row[3],
                'position_size': row[4],
                'executed_price': row[5],
                'price_to_beat': row[6]
            })

        return trades
