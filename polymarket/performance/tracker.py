# polymarket/performance/tracker.py
"""Performance tracking service."""

from datetime import datetime
from decimal import Decimal
from typing import Optional
import structlog

from polymarket.performance.database import PerformanceDatabase
from polymarket.models import TradingDecision, BTCPriceData, TechnicalIndicators, AggregatedSentiment, Market

logger = structlog.get_logger()


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
        is_end_phase: bool = False
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
                                else None
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
        skipped_unfavorable_move: bool = False
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
        """
        try:
            # Calculate slippage if we have both prices
            price_slippage_pct = None
            if execution_price is not None and analysis_price is not None:
                price_slippage_pct = ((execution_price - analysis_price) / analysis_price) * 100

            # Update the database record
            cursor = self.db.conn.cursor()
            cursor.execute("""
                UPDATE trades
                SET analysis_price = ?,
                    price_staleness_seconds = ?,
                    price_slippage_pct = ?,
                    price_movement_favorable = ?,
                    skipped_unfavorable_move = ?
                WHERE id = ?
            """, (
                analysis_price,
                price_staleness_seconds,
                price_slippage_pct,
                price_movement_favorable,
                skipped_unfavorable_move,
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

    def close(self):
        """Close database connection."""
        if self._owns_db:
            self.db.close()
