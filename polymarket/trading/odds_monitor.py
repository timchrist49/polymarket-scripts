"""Real-time odds monitoring for event-driven trading cycles."""

from typing import Callable, Optional, Dict
import asyncio
from datetime import datetime, timezone
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
            # Get current market ID from streamer (always up-to-date after transitions)
            current_market_id = self._streamer._current_market_id

            if not current_market_id:
                logger.debug("No active market in streamer")
                return None

            # Get current odds from streamer (using live market ID)
            snapshot = self._streamer.get_current_odds(current_market_id)

            if not snapshot:
                logger.debug("No current odds available")
                return None

            # Get current market slug from streamer
            current_market_slug = self._streamer._current_market_slug

            # Check staleness
            age = (datetime.now(timezone.utc) - snapshot.timestamp).total_seconds()
            if age > 120:  # 2 minutes
                logger.warning(
                    "Odds data too stale, skipping",
                    age_seconds=age,
                    market_id=current_market_id,
                    market_slug=current_market_slug
                )
                return None

            # Validate market is active
            if current_market_slug and not self._validator.is_market_active(current_market_slug):
                logger.debug(
                    "Market not active, skipping",
                    market_slug=current_market_slug
                )
                return None

            # Check if YES odds exceed threshold
            if snapshot.yes_odds >= self._threshold:
                logger.info(
                    "Opportunity detected: YES",
                    market_slug=current_market_slug,
                    odds=snapshot.yes_odds,
                    threshold=self._threshold
                )
                return {
                    "market_slug": current_market_slug or "unknown",
                    "direction": "YES",
                    "odds": snapshot.yes_odds
                }

            # Check if NO odds exceed threshold
            if snapshot.no_odds >= self._threshold:
                logger.info(
                    "Opportunity detected: NO",
                    market_slug=current_market_slug,
                    odds=snapshot.no_odds,
                    threshold=self._threshold
                )
                return {
                    "market_slug": current_market_slug or "unknown",
                    "direction": "NO",
                    "odds": snapshot.no_odds
                }

            # No opportunity found
            logger.debug(
                "No opportunity - odds below threshold",
                market_slug=current_market_slug,
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

    async def _check_and_handle_opportunity(self) -> None:
        """Check for opportunities and handle sustained threshold + cooldown logic."""
        # Check for opportunity
        opportunity = await self._check_opportunities()

        # Get current market ID from streamer
        market_id = self._streamer._current_market_id

        if not opportunity:
            # No opportunity - clear threshold tracking for this market
            if market_id and market_id in self._threshold_start_time:
                logger.debug(
                    "Opportunity lost, clearing threshold tracking",
                    market_id=market_id
                )
                self._threshold_start_time.pop(market_id, None)
            return

        # Extract opportunity details
        market_slug = opportunity["market_slug"]
        direction = opportunity["direction"]
        odds = opportunity["odds"]

        # Check cooldown
        if market_id in self._last_trigger_time:
            time_since_last = (datetime.now(timezone.utc) - self._last_trigger_time[market_id]).total_seconds()
            if time_since_last < self._cooldown_duration:
                logger.debug(
                    "In cooldown period, skipping",
                    market_id=market_id,
                    time_since_last=time_since_last,
                    cooldown_duration=self._cooldown_duration
                )
                return

        # Track sustained threshold
        now = datetime.now(timezone.utc)

        if market_id not in self._threshold_start_time:
            # First time above threshold
            logger.debug(
                "Threshold exceeded, starting sustained timer",
                market_id=market_id,
                direction=direction,
                odds=odds
            )
            self._threshold_start_time[market_id] = now
            return

        # Check if sustained for required duration
        sustained_duration = (now - self._threshold_start_time[market_id]).total_seconds()

        if sustained_duration >= self._sustained_duration:
            # Sustained threshold met - trigger opportunity
            logger.info(
                "Sustained threshold met, triggering opportunity",
                market_slug=market_slug,
                market_id=market_id,
                direction=direction,
                odds=odds,
                sustained_duration=sustained_duration
            )

            # Call the callback
            self._on_opportunity_detected(market_slug, direction, odds)

            # Update tracking
            self._last_trigger_time[market_id] = now
            self._threshold_start_time.pop(market_id, None)
        else:
            logger.debug(
                "Threshold sustained but not long enough yet",
                market_id=market_id,
                sustained_duration=sustained_duration,
                required_duration=self._sustained_duration
            )

    async def _monitor_loop(self) -> None:
        """Monitor loop that checks for opportunities every second."""
        logger.info("Monitor loop started")

        try:
            while self._is_running:
                try:
                    # Check for opportunities and handle sustained/cooldown logic
                    await self._check_and_handle_opportunity()
                except Exception as e:
                    # Log error but continue monitoring
                    logger.error(
                        "Error in monitor loop iteration",
                        error=str(e),
                        exc_info=True
                    )

                # Sleep 1 second before next check
                await asyncio.sleep(1.0)

        except asyncio.CancelledError:
            logger.info("Monitor loop cancelled")
            raise

        logger.info("Monitor loop stopped")
