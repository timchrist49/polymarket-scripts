"""Real-time odds monitoring for event-driven trading cycles."""

from typing import Callable, Optional
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
        self._last_opportunity_time: Optional[datetime] = None

        logger.info(
            "OddsMonitor initialized",
            threshold_percentage=threshold_percentage,
            threshold_decimal=self._threshold,
            sustained_duration_seconds=sustained_duration_seconds,
            cooldown_seconds=cooldown_seconds
        )
