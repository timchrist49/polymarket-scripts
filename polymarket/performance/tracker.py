# polymarket/performance/tracker.py
"""Performance tracking service."""

from datetime import datetime
from decimal import Decimal
from typing import Optional
import structlog

from polymarket.performance.database import PerformanceDatabase
from polymarket.models import TradingDecision, BTCPriceData, TechnicalIndicators, AggregatedSentiment

logger = structlog.get_logger()


class PerformanceTracker:
    """Tracks trading performance and stores to database."""

    def __init__(self, db_path: str = "data/performance.db"):
        """
        Initialize performance tracker.

        Args:
            db_path: Path to SQLite database (':memory:' for testing)
        """
        self.db = PerformanceDatabase(db_path)
        logger.info("Performance tracker initialized")

    async def log_decision(
        self,
        market: dict,
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
            market: Market data dict
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
            # Extract market slug from question or ID
            market_slug = self._extract_market_slug(market)

            # Build trade data dict
            trade_data = {
                "timestamp": datetime.now(),
                "market_slug": market_slug,
                "market_id": market.get("id"),

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
                "yes_price": market.get("best_ask"),
                "no_price": 1 - market.get("best_bid", 0.5),
                "executed_price": market.get("best_ask") if decision.action == "YES"
                                else 1 - market.get("best_bid", 0.5) if decision.action == "NO"
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

    def close(self):
        """Close database connection."""
        self.db.close()
