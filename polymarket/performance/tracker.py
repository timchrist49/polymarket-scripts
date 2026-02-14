# polymarket/performance/tracker.py
"""Performance tracking service."""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Optional
import structlog

from polymarket.performance.database import PerformanceDatabase
from polymarket.models import TradingDecision, BTCPriceData, TechnicalIndicators, AggregatedSentiment, Market

logger = structlog.get_logger()


@dataclass
class TestModeMetrics:
    """Aggregated metrics for test mode performance."""
    total_trades: int
    executed_trades: int
    execution_rate: float
    wins: int
    losses: int
    win_rate: float
    total_pnl: Decimal
    avg_arbitrage_edge: float
    avg_confidence: float
    timeframe_alignment_stats: dict  # {alignment_type: count}


class PerformanceTracker:
    """Tracks trading performance and stores to database."""

    def __init__(self, db_path: str = "data/performance.db", db: Optional[PerformanceDatabase] = None):
        """
        Initialize performance tracker.

        Args:
            db_path: Path to SQLite database (':memory:' for testing)
            db: Existing database instance (for testing with shared DB)
        """
        self.db = db if db is not None else PerformanceDatabase(db_path)
        self._owns_db = (db is None)
        logger.info("Performance tracker initialized")

    async def log_decision(
        self,
        market: Market,
        decision: TradingDecision,
        btc_data: BTCPriceData,
        technical: TechnicalIndicators,
        aggregated: AggregatedSentiment,
        price_to_beat: Optional[Decimal] = None,
        time_remaining_seconds: Optional[int] = None,
        is_end_phase: bool = False,
        actual_probability: float | None = None,
        arbitrage_edge: float | None = None,
        arbitrage_urgency: str | None = None,
        timeframe_analysis: Optional['TimeframeAnalysis'] = None,
        is_test_mode: bool = False
    ) -> int:
        """
        Log a trading decision to the database.

        Args:
            market: Market object
            decision: Trading decision
            btc_data: BTC price data
            technical: Technical indicators
            aggregated: Aggregated sentiment
            price_to_beat: Baseline price for comparison
            time_remaining_seconds: Time until market closes
            is_end_phase: Whether in end-of-market phase

        Returns:
            Trade ID
        """
        try:
            # Extract market slug
            market_slug = market.slug or market.id

            # Extract timeframe data if available
            tf_15m_dir = None
            tf_1h_dir = None
            tf_4h_dir = None
            tf_alignment = None
            tf_modifier = None

            if timeframe_analysis:
                tf_15m_dir = timeframe_analysis.tf_15m.direction
                tf_1h_dir = timeframe_analysis.tf_1h.direction
                tf_4h_dir = timeframe_analysis.tf_4h.direction
                tf_alignment = timeframe_analysis.alignment_score
                tf_modifier = timeframe_analysis.confidence_modifier

            # Build trade data dict
            trade_data = {
                "timestamp": datetime.now(),
                "market_slug": market_slug,
                "market_id": market.id,

                # Decision
                "action": decision.action,
                "confidence": decision.confidence,
                "position_size": float(decision.position_size),
                "reasoning": decision.reasoning,

                # Market Context
                "btc_price": float(btc_data.price),
                "price_to_beat": float(price_to_beat) if price_to_beat else None,
                "time_remaining_seconds": time_remaining_seconds,
                "is_end_phase": is_end_phase,

                # Signals
                "social_score": aggregated.social.score,
                "market_score": aggregated.market.score,
                "final_score": aggregated.final_score,
                "final_confidence": aggregated.final_confidence,
                "signal_type": aggregated.signal_type,

                # Technical
                "rsi": technical.rsi,
                "macd": technical.macd_value,
                "trend": technical.trend,

                # Pricing
                "yes_price": market.best_ask,
                "no_price": 1 - market.best_bid if market.best_bid else 0.5,
                "executed_price": market.best_ask if decision.action == "YES"
                                else (1 - market.best_bid if market.best_bid else 0.5) if decision.action == "NO"
                                else None,

                # Arbitrage tracking
                "actual_probability": actual_probability,
                "arbitrage_edge": arbitrage_edge,
                "arbitrage_urgency": arbitrage_urgency,

                # Timeframe analysis
                "timeframe_15m_direction": tf_15m_dir,
                "timeframe_1h_direction": tf_1h_dir,
                "timeframe_4h_direction": tf_4h_dir,
                "timeframe_alignment": tf_alignment,
                "confidence_modifier": tf_modifier,

                # Test mode flag
                "is_test_mode": 1 if is_test_mode else 0
            }

            trade_id = self.db.log_trade(trade_data)

            logger.info(
                "Decision logged to database",
                trade_id=trade_id,
                action=decision.action,
                market_slug=market_slug
            )

            return trade_id

        except Exception as e:
            logger.error("Failed to log decision", error=str(e))
            # Don't block trading on logging failure
            return -1

    def _extract_market_slug(self, market: dict) -> str:
        """Extract market slug from market data."""
        # Try explicit slug field first
        if "slug" in market:
            return market["slug"]

        # Fall back to question-based slug
        question = market.get("question", "unknown")
        # Simplified slug generation
        slug = question.lower().replace(" ", "-").replace("?", "")[:50]
        return slug

    async def update_execution_metrics(
        self,
        trade_id: int,
        analysis_price: float,
        execution_price: Optional[float] = None,
        price_staleness_seconds: Optional[int] = None,
        price_movement_favorable: Optional[bool] = None,
        skipped_unfavorable_move: bool = False,
        actual_position_size: Optional[float] = None,
        filled_via: Optional[str] = None,
        limit_order_timeout: Optional[int] = None
    ) -> None:
        """
        Update trade record with execution metrics from JIT price fetching.

        Args:
            trade_id: Trade record ID to update
            analysis_price: Price from cycle start (when analysis was done)
            execution_price: Fresh price at execution time (None if skipped)
            price_staleness_seconds: Time between analysis and execution
            price_movement_favorable: Whether price moved favorably
            skipped_unfavorable_move: Whether trade was skipped due to safety check
            actual_position_size: Actual position size after risk management (overrides AI suggestion)
            filled_via: How the order was filled ('market', 'limit', 'limit_partial')
            limit_order_timeout: Timeout used for limit orders (seconds)
        """
        try:
            # Calculate slippage if we have both prices
            price_slippage_pct = None
            if execution_price is not None and analysis_price is not None:
                price_slippage_pct = ((execution_price - analysis_price) / analysis_price) * 100

            # Update the database record
            cursor = self.db.conn.cursor()

            # Build SQL based on whether position_size needs updating
            if actual_position_size is not None:
                cursor.execute("""
                    UPDATE trades
                    SET analysis_price = ?,
                        price_staleness_seconds = ?,
                        price_slippage_pct = ?,
                        price_movement_favorable = ?,
                        skipped_unfavorable_move = ?,
                        position_size = ?,
                        filled_via = ?,
                        limit_order_timeout = ?
                    WHERE id = ?
                """, (
                    analysis_price,
                    price_staleness_seconds,
                    price_slippage_pct,
                    price_movement_favorable,
                    skipped_unfavorable_move,
                    actual_position_size,
                    filled_via,
                    limit_order_timeout,
                    trade_id
                ))
            else:
                cursor.execute("""
                    UPDATE trades
                    SET analysis_price = ?,
                        price_staleness_seconds = ?,
                        price_slippage_pct = ?,
                        price_movement_favorable = ?,
                        skipped_unfavorable_move = ?,
                        filled_via = ?,
                        limit_order_timeout = ?
                    WHERE id = ?
                """, (
                    analysis_price,
                    price_staleness_seconds,
                    price_slippage_pct,
                    price_movement_favorable,
                    skipped_unfavorable_move,
                    filled_via,
                    limit_order_timeout,
                    trade_id
                ))

            self.db.conn.commit()

            logger.debug(
                "Execution metrics updated",
                trade_id=trade_id,
                staleness_seconds=price_staleness_seconds,
                slippage_pct=f"{price_slippage_pct:+.2f}%" if price_slippage_pct else None,
                skipped=skipped_unfavorable_move
            )

        except Exception as e:
            logger.error("Failed to update execution metrics", trade_id=trade_id, error=str(e))

    def update_trade_outcome(
        self,
        trade_id: int,
        actual_outcome: str,
        profit_loss: float,
        is_win: bool
    ) -> None:
        """
        Update trade record with settlement outcome.

        Args:
            trade_id: Trade ID to update
            actual_outcome: "YES" or "NO"
            profit_loss: Dollar profit/loss
            is_win: Whether trade won
        """
        cursor = self.db.conn.cursor()

        cursor.execute("""
            UPDATE trades
            SET actual_outcome = ?,
                profit_loss = ?,
                is_win = ?
            WHERE id = ?
        """, (actual_outcome, profit_loss, is_win, trade_id))

        self.db.conn.commit()

        logger.info(
            "Trade outcome updated",
            trade_id=trade_id,
            outcome=actual_outcome,
            is_win=is_win,
            profit_loss=f"${profit_loss:.2f}"
        )

    async def update_trade_status(
        self,
        trade_id: int,
        execution_status: str,
        skip_reason: str = None,
        skip_type: str = None
    ) -> None:
        """
        Update trade execution status.

        Args:
            trade_id: Trade ID to update
            execution_status: 'pending', 'executed', 'skipped', 'failed'
            skip_reason: Optional reason for skipping
            skip_type: Optional type of skip (e.g., 'validation', 'risk_check')
        """
        try:
            cursor = self.db.conn.cursor()

            # Just update execution_status for now
            # (skip_reason and skip_type could be added as columns later if needed)
            cursor.execute("""
                UPDATE trades
                SET execution_status = ?
                WHERE id = ?
            """, (execution_status, trade_id))

            self.db.conn.commit()

            logger.debug(
                "Trade status updated",
                trade_id=trade_id,
                status=execution_status,
                reason=skip_reason
            )

        except Exception as e:
            logger.error("Failed to update trade status", trade_id=trade_id, error=str(e))

    def calculate_test_mode_metrics(self, last_n_trades: int = 20) -> Optional[TestModeMetrics]:
        """Calculate aggregated metrics for last N trades.

        Args:
            last_n_trades: Number of recent trades to analyze

        Returns:
            TestModeMetrics with aggregated statistics
        """
        cursor = self.db.conn.cursor()

        # Get last N trades
        cursor.execute("""
            SELECT
                execution_status,
                is_win,
                profit_loss,
                arbitrage_edge,
                confidence,
                timeframe_alignment
            FROM trades
            WHERE is_test_mode = 1
            ORDER BY timestamp DESC
            LIMIT ?
        """, (last_n_trades,))

        trades = cursor.fetchall()

        if not trades:
            return None

        total_trades = len(trades)
        executed = sum(1 for t in trades if t[0] in ['filled', 'FILLED'])
        execution_rate = executed / total_trades if total_trades > 0 else 0.0

        # Only count settled trades for win rate
        settled = [t for t in trades if t[1] is not None]
        wins = sum(1 for t in settled if t[1] == 1)
        losses = len(settled) - wins
        win_rate = wins / len(settled) if settled else 0.0

        # Total P&L from settled trades
        total_pnl = sum((Decimal(str(t[2])) for t in settled if t[2] is not None), Decimal('0'))

        # Average metrics
        avg_edge = sum(t[3] or 0 for t in trades) / total_trades
        avg_confidence = sum(t[4] or 0 for t in trades) / total_trades

        # Timeframe alignment breakdown
        alignment_stats = {}
        for t in trades:
            alignment = t[5] or "UNKNOWN"
            alignment_stats[alignment] = alignment_stats.get(alignment, 0) + 1

        return TestModeMetrics(
            total_trades=total_trades,
            executed_trades=executed,
            execution_rate=execution_rate,
            wins=wins,
            losses=losses,
            win_rate=win_rate,
            total_pnl=total_pnl,
            avg_arbitrage_edge=avg_edge,
            avg_confidence=avg_confidence,
            timeframe_alignment_stats=alignment_stats
        )

    def close(self):
        """Close database connection."""
        if self._owns_db:
            self.db.close()
