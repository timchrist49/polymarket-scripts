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

    def __init__(self, db: PerformanceDatabase, btc_fetcher, order_verifier=None):
        """
        Initialize trade settler.

        Args:
            db: Performance database
            btc_fetcher: BTC price service (from auto_trade.py)
            order_verifier: OrderVerifier instance for order verification (optional)
        """
        self.db = db
        self.btc_fetcher = btc_fetcher
        self.order_verifier = order_verifier

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
            # The slug contains the market START timestamp.
            # Detect market duration from slug: btc-updown-5m-* = 300s, else 900s (15m).
            duration = 300 if "-5m-" in market_slug else 900
            return int(match.group(1)) + duration

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
    ) -> tuple[float, bool, float]:
        """
        Calculate profit/loss for a settled trade.

        Polymarket binary mechanics:
        - Shares bought: position_size / executed_price
        - If win: Payout = shares × $1.00, minus 2% fee on winnings
        - If loss: Payout = $0

        Args:
            action: "YES" or "NO"
            actual_outcome: "YES" or "NO"
            position_size: Dollar amount invested
            executed_price: Price paid per share

        Returns:
            (profit_loss, is_win, fee_paid)
        """
        shares = position_size / executed_price

        if (action == "YES" and actual_outcome == "YES") or \
           (action == "NO" and actual_outcome == "NO"):
            # Win - each share worth $1
            payout = shares * 1.00
            gross_profit = payout - position_size

            # Polymarket charges 2% fee on winnings only
            fee_paid = gross_profit * 0.02 if gross_profit > 0 else 0.0
            profit_loss = gross_profit - fee_paid
            is_win = True
        else:
            # Loss - shares worth $0
            profit_loss = -position_size
            fee_paid = 0.0
            is_win = False

        return profit_loss, is_win, fee_paid

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
        # 3. Were actually executed (not skipped) - NEW: prevents phantom trades
        # 4. Are old enough (>15 minutes old)
        cursor.execute("""
            SELECT
                id, timestamp, market_slug, action,
                position_size, executed_price, price_to_beat,
                order_id, verification_status
            FROM trades
            WHERE action IN ('YES', 'NO')
              AND is_win IS NULL
              AND execution_status = 'executed'
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
                'price_to_beat': row[6],
                'order_id': row[7],
                'verification_status': row[8]
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
            "verification_failures": 0,
            "price_discrepancies": 0,
            "partial_fills": 0,
            "errors": []
        }

        try:
            # Get unsettled trades
            trades = self._get_unsettled_trades(batch_size)

            logger.info(
                "Starting settlement cycle with verification",
                pending_trades=len(trades),
                verifier_enabled=self.order_verifier is not None
            )

            for trade in trades:
                try:
                    # NEW: Verify order execution BEFORE calculating P&L
                    actual_price = trade['executed_price']
                    actual_size = trade['position_size']
                    tx_hash = None

                    if self.order_verifier and trade.get('order_id'):
                        verification = await self.order_verifier.verify_order_full(
                            trade['order_id']
                        )

                        if not verification['verified']:
                            # Order never filled - mark as failed
                            logger.warning(
                                "Order verification failed - trade not filled",
                                trade_id=trade['id'],
                                order_id=trade['order_id'],
                                status=verification['status']
                            )

                            self._mark_trade_failed(trade['id'], verification)
                            stats['verification_failures'] += 1
                            continue  # Skip P&L calculation

                        # Use verified data
                        actual_price = verification['fill_price']
                        actual_size = verification['fill_amount']
                        tx_hash = verification['transaction_hash']

                        # Calculate discrepancy
                        estimated_price = trade['executed_price']
                        price_discrepancy_pct = self.order_verifier.calculate_price_discrepancy(
                            estimated_price, actual_price
                        )

                        # Alert on large discrepancies
                        if abs(price_discrepancy_pct) > 5.0:
                            logger.warning(
                                "Large price discrepancy detected",
                                trade_id=trade['id'],
                                estimated=f"${estimated_price:.3f}",
                                actual=f"${actual_price:.3f}",
                                discrepancy=f"{price_discrepancy_pct:+.2f}%"
                            )
                            stats['price_discrepancies'] += 1

                        # Track partial fills
                        if verification['partial_fill']:
                            logger.info(
                                "Partial fill detected",
                                trade_id=trade['id'],
                                filled=verification['fill_amount'],
                                expected=verification['original_size'],
                                fill_pct=f"{(verification['fill_amount'] / verification['original_size'] * 100):.1f}%"
                            )
                            stats['partial_fills'] += 1

                        # Store verification data
                        self._update_verification_data(
                            trade_id=trade['id'],
                            verification=verification,
                            price_discrepancy_pct=price_discrepancy_pct
                        )

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

                    # Calculate profit/loss using VERIFIED data
                    profit_loss, is_win, fee_paid = self._calculate_profit_loss(
                        action=trade['action'],
                        actual_outcome=actual_outcome,
                        position_size=actual_size,  # Use verified size
                        executed_price=actual_price  # Use verified price
                    )

                    # Update database (via tracker if available, otherwise direct)
                    if hasattr(self, '_tracker'):
                        self._tracker.update_trade_outcome(
                            trade_id=trade['id'],
                            actual_outcome=actual_outcome,
                            profit_loss=profit_loss,
                            is_win=is_win,
                            fee_paid=fee_paid
                        )
                    else:
                        # Direct update for testing
                        cursor = self.db.conn.cursor()
                        cursor.execute("""
                            UPDATE trades
                            SET actual_outcome = ?,
                                profit_loss = ?,
                                is_win = ?,
                                fee_paid = ?
                            WHERE id = ?
                        """, (actual_outcome, profit_loss, is_win, fee_paid, trade['id']))
                        self.db.conn.commit()

                    # Update stats
                    stats['settled_count'] += 1
                    if is_win:
                        stats['wins'] += 1
                    else:
                        stats['losses'] += 1
                    stats['total_profit'] += profit_loss

                    logger.info(
                        "Trade settled with verification",
                        trade_id=trade['id'],
                        action=trade['action'],
                        outcome=actual_outcome,
                        is_win=is_win,
                        profit_loss=f"${profit_loss:.2f}",
                        verified=self.order_verifier is not None and trade.get('order_id') is not None
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

    def _mark_trade_failed(self, trade_id: int, verification: dict) -> None:
        """Mark a trade as failed due to verification failure."""
        cursor = self.db.conn.cursor()
        cursor.execute("""
            UPDATE trades
            SET verification_status = 'failed',
                verification_timestamp = ?,
                skip_reason = ?
            WHERE id = ?
        """, (
            int(datetime.now().timestamp()),
            f"Order not filled: {verification['status']}",
            trade_id
        ))
        self.db.conn.commit()

    def _update_verification_data(
        self,
        trade_id: int,
        verification: dict,
        price_discrepancy_pct: float
    ) -> None:
        """Update trade with verification data."""
        cursor = self.db.conn.cursor()
        cursor.execute("""
            UPDATE trades
            SET verified_fill_price = ?,
                verified_fill_amount = ?,
                transaction_hash = ?,
                fill_timestamp = ?,
                partial_fill = ?,
                verification_status = 'verified',
                verification_timestamp = ?,
                price_discrepancy_pct = ?
            WHERE id = ?
        """, (
            verification['fill_price'],
            verification['fill_amount'],
            verification['transaction_hash'],
            verification['fill_timestamp'],
            verification['partial_fill'],
            int(datetime.now().timestamp()),
            price_discrepancy_pct,
            trade_id
        ))
        self.db.conn.commit()
