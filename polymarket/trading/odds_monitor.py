"""Real-time odds monitoring for event-driven trading cycles."""

from typing import Callable, Optional, Dict
import asyncio
from datetime import datetime
import structlog

from polymarket.trading.realtime_odds_streamer import RealtimeOddsStreamer
from polymarket.trading.market_validator import MarketValidator

logger = structlog.get_logger(__name__)


class OddsMonitor:
    """Monitors real-time odds and triggers trading cycles on opportunities."""

    def __init__(
        self,
        streamer: RealtimeOddsStreamer,
        validator: MarketValidator,
        on_opportunity_detected: Callable[[str, str, float], None],
        threshold_percentage: float = 70.0,
        sustained_duration_seconds: float = 5.0,
        cooldown_seconds: float = 30.0
    ):
        """Initialize the OddsMonitor.

        Args:
            streamer: RealtimeOddsStreamer instance for getting current odds
            validator: MarketValidator instance for validating market activity
            on_opportunity_detected: Callback when opportunity detected (market_slug, direction, odds)
            threshold_percentage: Odds threshold as percentage (default: 70.0)
            sustained_duration_seconds: How long odds must be above threshold (default: 5.0)
            cooldown_seconds: Cooldown between opportunities (default: 30.0)
        """
        self._streamer = streamer
        self._validator = validator
        self._on_opportunity_detected = on_opportunity_detected
        self._threshold = threshold_percentage / 100.0  # Convert to decimal
        self._sustained_duration = sustained_duration_seconds
        self._cooldown_duration = cooldown_seconds

        # Runtime state
        self._is_running = False
        self._monitor_task: Optional[asyncio.Task] = None
        self._threshold_start_time: Dict[str, datetime] = {}
        self._last_trigger_time: Dict[str, datetime] = {}

        # Market tracking (set by AutoTrader during initialization)
        self._market_id: Optional[str] = None
        self._market_slug: Optional[str] = None

        logger.info(
            "OddsMonitor initialized",
            threshold_percentage=threshold_percentage,
            threshold_decimal=self._threshold,
            sustained_duration_seconds=sustained_duration_seconds,
            cooldown_seconds=cooldown_seconds
        )

    async def start(self) -> None:
        """Start the odds monitoring loop."""
        if self._is_running:
            logger.warning("OddsMonitor already running, ignoring start request")
            return

        logger.info("Starting OddsMonitor")
        self._is_running = True

        # Create the monitor task (loop implementation in Task 7)
        self._monitor_task = asyncio.create_task(self._monitor_loop())

    async def stop(self) -> None:
        """Stop the odds monitoring loop."""
        if not self._is_running:
            logger.warning("OddsMonitor not running, ignoring stop request")
            return

        logger.info("Stopping OddsMonitor")
        self._is_running = False

        # Cancel the monitor task
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                logger.debug("Monitor task cancelled successfully")

    async def _check_opportunities(self) -> Optional[dict]:
        """Check if current odds present an opportunity.

        Returns:
            Dict with market_slug, direction, odds if opportunity found, else None
        """
        try:
            # Check if market is configured
            if not self._market_id or not self._market_slug:
                logger.debug("No market configured for monitoring")
                return None

            # Get current odds from streamer
            snapshot = self._streamer.get_current_odds(self._market_id)

            if not snapshot:
                logger.debug("No current odds available")
                return None

            # Validate market is active
            if not self._validator.is_market_active(self._market_slug):
                logger.debug(
                    "Market not active, skipping",
                    market_slug=self._market_slug
                )
                return None

            # Check if YES odds exceed threshold
            if snapshot.yes_odds >= self._threshold:
                logger.info(
                    "Opportunity detected: YES",
                    market_slug=self._market_slug,
                    odds=snapshot.yes_odds,
                    threshold=self._threshold
                )
                return {
                    "market_slug": self._market_slug,
                    "direction": "YES",
                    "odds": snapshot.yes_odds
                }

            # Check if NO odds exceed threshold
            if snapshot.no_odds >= self._threshold:
                logger.info(
                    "Opportunity detected: NO",
                    market_slug=self._market_slug,
                    odds=snapshot.no_odds,
                    threshold=self._threshold
                )
                return {
                    "market_slug": self._market_slug,
                    "direction": "NO",
                    "odds": snapshot.no_odds
                }

            # No opportunity found
            logger.debug(
                "No opportunity - odds below threshold",
                market_slug=self._market_slug,
                yes_odds=snapshot.yes_odds,
                no_odds=snapshot.no_odds,
                threshold=self._threshold
            )
            return None

        except Exception as e:
            logger.error(
                "Error checking opportunities",
                error=str(e),
                exc_info=True
            )
            return None

    async def _monitor_loop(self) -> None:
        """Monitor loop placeholder (implemented in Task 7)."""
        try:
            while self._is_running:
                # Placeholder: sleep 1 second
                await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            logger.debug("Monitor loop cancelled")
            raise
