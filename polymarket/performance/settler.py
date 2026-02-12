"""Trade settlement service for determining win/loss outcomes."""

import re
import asyncio
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

    async def _get_btc_price_at_timestamp(self, timestamp: int) -> Optional[float]:
        """
        Fetch BTC price at specific timestamp.

        Args:
            timestamp: Unix timestamp

        Returns:
            BTC price as float, or None if unavailable
        """
        try:
            price_decimal = await self.btc_fetcher.get_price_at_timestamp(timestamp)

            if price_decimal is None:
                return None

            return float(price_decimal)

        except Exception as e:
            logger.error(
                "Failed to fetch BTC price at timestamp",
                timestamp=timestamp,
                error=str(e)
            )
            return None

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

    async def settle_pending_trades(self, batch_size: int = 50) -> Dict:
        """
        Settle all pending trades that have closed.

        Args:
            batch_size: Max trades to process per cycle

        Returns:
            Settlement statistics
        """
        stats = {
            "success": True,
            "settled_count": 0,
            "wins": 0,
            "losses": 0,
            "total_profit": 0.0,
            "pending_count": 0,
            "errors": []
        }

        try:
            # Get unsettled trades
            trades = self._get_unsettled_trades(batch_size)

            logger.info(
                "Starting settlement cycle",
                pending_trades=len(trades)
            )

            for trade in trades:
                try:
                    # Parse close timestamp
                    close_timestamp = self._parse_market_close_timestamp(trade['market_slug'])

                    if close_timestamp is None:
                        error_msg = f"Failed to parse timestamp from {trade['market_slug']}"
                        logger.error(error_msg, trade_id=trade['id'])
                        stats['errors'].append(error_msg)
                        # Mark as UNKNOWN but don't count in win rate
                        if hasattr(self, '_tracker'):
                            self._tracker.update_trade_outcome(
                                trade_id=trade['id'],
                                actual_outcome="UNKNOWN",
                                profit_loss=0.0,
                                is_win=False
                            )
                        continue

                    # Fetch BTC price at close
                    btc_close_price = await self.btc_fetcher.get_price_at_timestamp(close_timestamp)

                    if btc_close_price is None:
                        # Skip - will retry next cycle
                        logger.warning(
                            "BTC price unavailable, will retry",
                            trade_id=trade['id'],
                            timestamp=close_timestamp
                        )
                        stats['pending_count'] += 1
                        continue

                    # Convert Decimal to float for comparison
                    btc_close_price = float(btc_close_price)

                    # Determine outcome
                    actual_outcome = self._determine_outcome(
                        btc_close_price=btc_close_price,
                        price_to_beat=trade['price_to_beat']
                    )

                    # Calculate profit/loss
                    profit_loss, is_win = self._calculate_profit_loss(
                        action=trade['action'],
                        actual_outcome=actual_outcome,
                        position_size=trade['position_size'],
                        executed_price=trade['executed_price']
                    )

                    # Update database (via tracker if available, otherwise direct)
                    if hasattr(self, '_tracker'):
                        self._tracker.update_trade_outcome(
                            trade_id=trade['id'],
                            actual_outcome=actual_outcome,
                            profit_loss=profit_loss,
                            is_win=is_win
                        )
                    else:
                        # Direct update for testing
                        cursor = self.db.conn.cursor()
                        cursor.execute("""
                            UPDATE trades
                            SET actual_outcome = ?,
                                profit_loss = ?,
                                is_win = ?
                            WHERE id = ?
                        """, (actual_outcome, profit_loss, is_win, trade['id']))
                        self.db.conn.commit()

                    # Update stats
                    stats['settled_count'] += 1
                    if is_win:
                        stats['wins'] += 1
                    else:
                        stats['losses'] += 1
                    stats['total_profit'] += profit_loss

                    logger.info(
                        "Trade settled",
                        trade_id=trade['id'],
                        action=trade['action'],
                        outcome=actual_outcome,
                        is_win=is_win,
                        profit_loss=f"${profit_loss:.2f}"
                    )

                except Exception as e:
                    error_msg = f"Failed to settle trade {trade.get('id', '?')}: {str(e)}"
                    logger.error(error_msg)
                    stats['errors'].append(error_msg)
                    continue
                finally:
                    # Rate limit: 2 second delay between each trade settlement
                    # to prevent API hammering
                    await asyncio.sleep(2)

        except Exception as e:
            logger.error("Settlement cycle failed", error=str(e))
            stats['success'] = False
            stats['errors'].append(str(e))

        return stats
