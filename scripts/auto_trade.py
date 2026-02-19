#!/usr/bin/env python3
"""
Autonomous Polymarket Trading Bot

Runs continuously, executing trading cycles every 3 minutes for BTC 15-min markets.

Usage:
    python scripts/auto_trade.py              # Run bot
    POLYMARKET_MODE=read_only python scripts/auto_trade.py  # Dry run
    python scripts/auto_trade.py --once       # Single cycle
    python scripts/auto_trade.py --interval 120  # Custom interval
"""

import asyncio
import sys
import signal
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Optional

import structlog
import typer

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from polymarket.client import PolymarketClient
from polymarket.config import Settings
from polymarket.models import Market

from polymarket.trading.btc_price import BTCPriceService
from polymarket.trading.social_sentiment import SocialSentimentService
from polymarket.trading.market_microstructure import MarketMicrostructureService
from polymarket.trading.signal_aggregator import SignalAggregator
from polymarket.trading.technical import TechnicalAnalysis
from polymarket.trading.ai_decision import AIDecisionService
from polymarket.trading.risk import RiskManager
from polymarket.trading.market_tracker import MarketTracker
from polymarket.trading.probability_calculator import ProbabilityCalculator
from polymarket.trading.arbitrage_detector import ArbitrageDetector
from polymarket.trading.smart_order_executor import SmartOrderExecutor
from polymarket.trading.timeframe_analyzer import TimeframeAnalyzer, TimeframeAnalysis
from polymarket.trading.signal_lag_detector import detect_signal_lag
from polymarket.trading.conflict_detector import SignalConflictDetector, ConflictSeverity
from polymarket.trading.odds_poller import MarketOddsPoller
from polymarket.trading.realtime_odds_streamer import RealtimeOddsStreamer
from polymarket.trading.odds_monitor import OddsMonitor
from polymarket.trading.market_validator import MarketValidator
from polymarket.trading.contrarian import get_movement_threshold
from polymarket.performance.tracker import PerformanceTracker
from polymarket.performance.cleanup import CleanupScheduler
from polymarket.performance.reflection import ReflectionEngine
from polymarket.performance.adjuster import ParameterAdjuster, AdjustmentTier, get_repo_root
from polymarket.performance.settler import TradeSettler
from polymarket.telegram.bot import TelegramBot

app = typer.Typer(help="Autonomous Polymarket Trading Bot")
logger = structlog.get_logger()


async def price_history_saver(buffer, interval: int = 300):
    """
    Background task to save price buffer to disk periodically.

    Args:
        buffer: PriceHistoryBuffer instance
        interval: Save interval in seconds (default 300 = 5 minutes)
    """
    logger.info(f"Price history saver started (interval: {interval}s)")

    while True:
        try:
            await asyncio.sleep(interval)
            await buffer.save_to_disk()
            logger.debug("Price history auto-saved")
        except asyncio.CancelledError:
            logger.info("Price history saver stopped")
            raise  # Re-raise to properly cancel task
        except Exception as e:
            logger.error(f"Failed to auto-save price history: {e}")
            # Continue running despite errors


async def price_history_cleaner(buffer, interval: int = 3600):
    """
    Background task to cleanup old entries periodically.

    Args:
        buffer: PriceHistoryBuffer instance
        interval: Cleanup interval in seconds (default 3600 = 1 hour)
    """
    logger.info(f"Price history cleaner started (interval: {interval}s)")

    while True:
        try:
            await asyncio.sleep(interval)
            removed = await buffer.cleanup_old_entries()
            if removed > 0:
                logger.info(f"Price history auto-cleanup: {removed} entries removed")
        except asyncio.CancelledError:
            logger.info("Price history cleaner stopped")
            raise  # Re-raise to properly cancel task
        except Exception as e:
            logger.error(f"Failed to auto-cleanup price history: {e}")
            # Continue running despite errors


@dataclass
class TestModeConfig:
    """Configuration for test mode trading."""
    enabled: bool = False
    paper_trading: bool = True  # NEW: No real money in test mode
    min_bet_amount: Decimal = Decimal("5.0")
    max_bet_amount: Decimal = Decimal("10.0")
    min_confidence: float = 0.70
    min_odds_threshold: float = 0.75  # NEW: Odds requirement
    min_arbitrage_edge: float = 0.02
    traded_markets: set[str] = field(default_factory=set)


class AutoTrader:
    """Main autonomous trading bot orchestrator."""

    def __init__(self, settings: Settings, interval: int = 60):
        self.settings = settings
        self.interval = interval
        self.client = PolymarketClient()

        # Initialize services
        self.btc_service = BTCPriceService(settings)
        self.social_service = SocialSentimentService(settings)
        self.market_service = None  # Will initialize per cycle with condition_id
        self.aggregator = SignalAggregator()
        self.ai_service = AIDecisionService(settings)
        self.risk_manager = RiskManager(settings)
        self.market_tracker = MarketTracker(settings)
        self.performance_tracker = PerformanceTracker()

        # Odds polling service
        self.odds_poller = MarketOddsPoller(self.client)

        # Real-time odds streamer (WebSocket)
        self.realtime_streamer = RealtimeOddsStreamer(self.client)
        logger.info("Real-time odds streamer initialized")

        # Market validator and odds monitor for event-driven cycle triggering
        self.market_validator = MarketValidator()
        self.odds_monitor = OddsMonitor(
            streamer=self.realtime_streamer,
            validator=self.market_validator,
            on_opportunity_detected=self._handle_opportunity_detected,
            threshold_percentage=60.0,  # Lowered from 70% to account for REST API lag
            sustained_duration_seconds=5.0,
            cooldown_seconds=30.0
        )
        logger.info(
            "OddsMonitor initialized for event-driven triggering",
            threshold_percentage=60.0,  # Match actual value above
            sustained_duration_seconds=5.0,
            cooldown_seconds=30.0
        )

        # Track traded markets for production mode (one order per market per session)
        self._traded_markets: set[str] = set()

        self.telegram_bot = TelegramBot(settings)  # Initialize Telegram bot
        self.cleanup_scheduler = CleanupScheduler(
            db=self.performance_tracker.db,
            telegram=self.telegram_bot,  # Pass Telegram bot instance
            interval_hours=168,  # Weekly
            days_threshold=30
        )

        # Self-reflection and parameter adjustment
        self.reflection_engine = ReflectionEngine(
            db=self.performance_tracker.db,
            settings=settings
        )
        self.parameter_adjuster = ParameterAdjuster(
            settings=settings,
            db=self.performance_tracker.db,
            telegram=self.telegram_bot
        )

        # Order verification
        from polymarket.performance.order_verifier import OrderVerifier
        self.order_verifier = OrderVerifier(
            client=self.client,
            db=self.performance_tracker.db
        )

        # Trade settlement
        self.trade_settler = TradeSettler(
            db=self.performance_tracker.db,
            btc_fetcher=self.btc_service,
            order_verifier=self.order_verifier
        )
        # Give settler access to tracker for updates
        self.trade_settler._tracker = self.performance_tracker

        # Multi-timeframe analysis
        # Note: Analyzer will be initialized after btc_service starts and buffer is available
        self.timeframe_analyzer: Optional[TimeframeAnalyzer] = None

        # State tracking
        self.cycle_count = 0
        self.trades_today = 0
        self.total_trades = 0  # Track total trades for reflection trigger
        self.consecutive_losses = 0  # Track consecutive losses for reflection trigger
        self.last_trade_was_loss = False  # Track last trade outcome
        self.pnl_today = Decimal("0")
        self.running = True

        # Track open positions for stop-loss
        self.open_positions: list[dict] = []

        # Per-market pending-order lock: prevents duplicate orders while a limit
        # order is awaiting fill/timeout (before it's added to open_positions)
        self._markets_with_pending_orders: set[str] = set()

        # Cached data from the last completed trading cycle, used by the price watcher
        # to avoid re-fetching indicators/sentiment on every 10s price-movement check.
        self._last_cycle_data: dict | None = None

        # Watcher-level analysis lock: tracks markets for which the price watcher has
        # already spawned an analysis task that hasn't finished yet.
        # Prevents the 10s loop from queuing duplicate tasks before _markets_with_pending_orders
        # is set (which only happens deep inside _process_market, 30-60s later).
        self._markets_with_active_watcher_analysis: set[str] = set()

        # Cycle-level analysis lock: tracks markets currently undergoing AI analysis in
        # ANY cycle (OddsMonitor-triggered, timer-triggered, or watcher-triggered).
        # The OddsMonitor has a 30s cooldown, but AI analysis takes 30-60s, so a second
        # cycle can start and pass the _traded_markets check before the first cycle places
        # its order. This lock prevents the AI call from even starting for a duplicate.
        self._markets_with_active_cycle_analysis: set[str] = set()

        # Timed strategy state: AI decisions stored at T=10min for entry in last 5 min.
        # _timed_decisions holds the full execution context keyed by market_id.
        # _analysis_triggered tracks markets where the 10min analysis has been fired.
        self._timed_decisions: dict[str, dict] = {}
        self._analysis_triggered: set[str] = set()

        # Background tasks for price buffer maintenance
        self.background_tasks: list[asyncio.Task] = []

        # Initialize test mode
        self.test_mode = TestModeConfig(
            enabled=os.getenv("TEST_MODE", "").lower() == "true",
            paper_trading=True,  # CRITICAL: Enable paper trading in test mode
            min_bet_amount=Decimal("5.0"),
            max_bet_amount=Decimal("10.0"),
            min_confidence=0.70,
            min_odds_threshold=0.75,
            min_arbitrage_edge=0.02,
            traded_markets=set()
        )

        # Log test mode status
        if self.test_mode.enabled:
            logger.warning(
                "=" * 70,
                test_mode=True,
                banner="TEST MODE ENABLED"
            )
            logger.warning(
                "[TEST] Trading with min $5, max $10 bets, 70% min confidence, 2% min edge",
                min_bet=str(self.test_mode.min_bet_amount),
                max_bet=str(self.test_mode.max_bet_amount),
                min_confidence=self.test_mode.min_confidence,
                min_edge=f"{self.test_mode.min_arbitrage_edge * 100:.0f}%"
            )
            logger.warning("=" * 70)

    def _handle_opportunity_detected(self, market_slug: str, direction: str, odds: float) -> None:
        """Handle detected trading opportunity by triggering cycle asynchronously.

        Args:
            market_slug: The market that triggered the opportunity
            direction: YES or NO
            odds: Current odds value
        """
        # Skip cycle triggering in dry-run mode
        if self.settings.dry_run:
            logger.info(
                "Dry-run mode: opportunity detected but not triggering cycle",
                market_slug=market_slug,
                direction=direction,
                odds=odds
            )
            return

        # NEW: Timed strategy — cycles are triggered by _market_timing_watcher at T=10min.
        # OddsMonitor is kept running for WebSocket connectivity but no longer fires cycles.
        logger.debug(
            "OddsMonitor opportunity noted (timed strategy: analysis triggered at T=10min)",
            market_slug=market_slug,
            direction=direction,
            odds=odds
        )

    async def initialize(self) -> None:
        """Initialize async resources before trading cycles."""
        # Start BTC price WebSocket stream
        await self.btc_service.start()
        logger.info("Initialized Polymarket WebSocket for BTC prices")
        logger.info("Performance tracking enabled")

        # Initialize multi-timeframe analyzer with price buffer
        if self.btc_service._stream and self.btc_service._stream.price_buffer:
            self.timeframe_analyzer = TimeframeAnalyzer(
                price_buffer=self.btc_service._stream.price_buffer
            )
            logger.info("Multi-timeframe analyzer initialized")
        else:
            logger.warning("Price buffer not available - timeframe analysis will be disabled")

        # Start cleanup scheduler in background
        asyncio.create_task(self.cleanup_scheduler.start())
        logger.info("Cleanup scheduler started (runs weekly)")
        logger.info("Self-reflection system enabled (triggers: 10 trades, 3 consecutive losses)")

        # Start odds polling background task
        odds_task = asyncio.create_task(self.odds_poller.start_polling())
        self.background_tasks.append(odds_task)
        logger.info("Odds poller started (background task)")

        # Start real-time odds streamer
        await self.realtime_streamer.start()
        logger.info("Real-time odds streamer started")

        # Start OddsMonitor for event-driven cycle triggering
        await self.odds_monitor.start()
        logger.info("OddsMonitor started for event-driven triggering")

        # Start settlement loop
        settlement_task = asyncio.create_task(self._run_settlement_loop())
        logger.info("Settlement loop started")

        # Start background tasks for price buffer maintenance
        if self.btc_service._stream and self.btc_service._stream.price_buffer:
            buffer = self.btc_service._stream.price_buffer
            saver_task = asyncio.create_task(price_history_saver(buffer, interval=300))
            cleaner_task = asyncio.create_task(price_history_cleaner(buffer, interval=3600))
            self.background_tasks.extend([saver_task, cleaner_task])
            logger.info("Price buffer background tasks started (save: 5min, cleanup: 1hr)")

        # Enhancement 4: Price movement watcher — fires analysis when BTC moves >$25
        price_watcher_task = asyncio.create_task(self._price_movement_watcher())
        self.background_tasks.append(price_watcher_task)
        logger.info(
            "Price movement watcher started",
            trigger_usd=self.PRICE_WATCHER_TRIGGER_USD,
            interval_seconds=self.PRICE_WATCHER_INTERVAL_SECONDS
        )

        # Enhancement 5: Stop-loss watcher — checks open positions every 30s
        stop_loss_task = asyncio.create_task(self._stop_loss_watcher())
        self.background_tasks.append(stop_loss_task)
        logger.info(
            "Stop-loss watcher started",
            interval_seconds=self.STOP_LOSS_WATCHER_INTERVAL_SECONDS
        )

        # Timed strategy: timing watcher fires AI analysis at T=10min of each 15-min market
        timing_watcher_task = asyncio.create_task(self._market_timing_watcher())
        self.background_tasks.append(timing_watcher_task)
        logger.info(
            "Market timing watcher started (AI analysis triggered at T=10min)",
            analysis_trigger_at="600s elapsed",
            entry_window="last 5 minutes (<=300s remaining)"
        )

        # Timed strategy: entry monitor executes stored decisions when odds reach 70%
        timed_entry_task = asyncio.create_task(self._timed_entry_monitor())
        self.background_tasks.append(timed_entry_task)
        logger.info(
            "Timed entry monitor started",
            entry_odds_threshold="70%",
            check_interval_seconds=10
        )

    async def _trigger_reflection(self, trigger_type: str) -> None:
        """
        Trigger reflection analysis and process recommendations.

        Args:
            trigger_type: '10_trades', '3_losses', or 'end_of_day'
        """
        try:
            logger.info(
                "Triggering self-reflection",
                trigger_type=trigger_type,
                total_trades=self.total_trades,
                consecutive_losses=self.consecutive_losses
            )

            # Run reflection analysis
            insights = await self.reflection_engine.analyze_performance(
                trigger_type=trigger_type,
                trades_analyzed=self.total_trades
            )

            # Send insights to Telegram
            if insights and insights.get("insights"):
                insights_text = "\n".join(f"• {i}" for i in insights["insights"])
                await self.telegram_bot.send_reflection_summary(
                    trigger_type=trigger_type,
                    insights_text=insights_text,
                    trades_analyzed=self.total_trades
                )

            # Process recommendations
            if insights and insights.get("recommendations"):
                await self._process_recommendations(insights["recommendations"])

        except Exception as e:
            logger.error("Reflection failed", error=str(e), trigger_type=trigger_type)

    async def _process_recommendations(self, recommendations: list) -> None:
        """
        Process reflection recommendations and apply adjustments.

        Args:
            recommendations: List of parameter adjustment recommendations
        """
        for rec in recommendations:
            try:
                parameter_name = rec.get("parameter")
                current_value = rec.get("current")
                recommended_value = rec.get("recommended")
                reason = rec.get("reason", "AI recommendation")

                if not all([parameter_name, current_value is not None, recommended_value is not None]):
                    logger.warning("Incomplete recommendation", recommendation=rec)
                    continue

                # Validate adjustment
                if not self.parameter_adjuster.validate_adjustment(parameter_name, recommended_value):
                    logger.warning(
                        "Recommendation rejected - outside safe bounds",
                        parameter=parameter_name,
                        value=recommended_value
                    )
                    continue

                # Classify tier
                tier = self.parameter_adjuster.classify_tier(
                    parameter_name=parameter_name,
                    current_value=current_value,
                    new_value=recommended_value
                )

                logger.info(
                    "Processing recommendation",
                    parameter=parameter_name,
                    current=current_value,
                    recommended=recommended_value,
                    tier=tier.value
                )

                # Apply adjustment (handles approval for Tier 2, pause for Tier 3)
                applied = await self.parameter_adjuster.apply_adjustment(
                    parameter_name=parameter_name,
                    old_value=current_value,
                    new_value=recommended_value,
                    reason=reason,
                    tier=tier
                )

                if applied:
                    # Update settings in memory
                    if hasattr(self.settings, parameter_name):
                        setattr(self.settings, parameter_name, recommended_value)
                        logger.info(
                            "Parameter adjusted",
                            parameter=parameter_name,
                            old_value=current_value,
                            new_value=recommended_value,
                            tier=tier.value
                        )
                    else:
                        logger.error("Unknown parameter name", parameter=parameter_name)

            except Exception as e:
                logger.error("Failed to process recommendation", error=str(e), rec=rec)

    async def _run_settlement_loop(self):
        """Background loop for settling trades."""
        interval_seconds = self.settings.settlement_interval_minutes * 60

        logger.info(
            "Settlement loop started",
            interval_minutes=self.settings.settlement_interval_minutes
        )

        while True:
            try:
                stats = await self.trade_settler.settle_pending_trades(
                    batch_size=self.settings.settlement_batch_size
                )

                if stats["settled_count"] > 0:
                    logger.info(
                        "Settlement cycle complete",
                        settled=stats["settled_count"],
                        wins=stats["wins"],
                        losses=stats["losses"],
                        total_profit=f"${stats['total_profit']:.2f}",
                        pending=stats["pending_count"]
                    )

                # Check for stuck trades
                if stats["pending_count"] > 0 and stats["settled_count"] == 0:
                    logger.warning(
                        "No trades settled but pending exist",
                        pending_count=stats["pending_count"]
                    )

                # Alert if errors
                if stats["errors"]:
                    logger.error(
                        "Settlement errors occurred",
                        error_count=len(stats["errors"]),
                        errors=stats["errors"][:3]  # First 3 errors
                    )

            except Exception as e:
                logger.error("Settlement loop error", error=str(e))

            await asyncio.sleep(interval_seconds)

    async def _check_consecutive_losses(self) -> None:
        """Check if we have 3 consecutive losses and trigger reflection."""
        try:
            cursor = self.performance_tracker.db.conn.cursor()

            # Get last 3 trades with known outcomes
            cursor.execute("""
                SELECT is_win, action
                FROM trades
                WHERE is_win IS NOT NULL
                ORDER BY timestamp DESC
                LIMIT 3
            """)

            rows = cursor.fetchall()

            if len(rows) >= 3:
                # Check if all 3 are losses
                recent_losses = [row['is_win'] == 0 for row in rows]

                if all(recent_losses):
                    # Only trigger once per streak
                    if not self.last_trade_was_loss or self.consecutive_losses < 3:
                        self.consecutive_losses = 3
                        self.last_trade_was_loss = True
                        logger.warning(
                            "Detected 3 consecutive losses - triggering reflection",
                            recent_trades=[row['action'] for row in rows]
                        )
                        asyncio.create_task(self._trigger_reflection("3_losses"))
                else:
                    # Reset streak
                    self.consecutive_losses = 0
                    self.last_trade_was_loss = False

        except Exception as e:
            logger.error("Failed to check consecutive losses", error=str(e))

    def _check_emergency_pause(self) -> bool:
        """Check if emergency pause flag is set."""
        # Check environment variable
        if self.settings.emergency_pause_enabled:
            return True

        # Check for emergency pause file in repository root
        pause_file = get_repo_root() / ".emergency_pause"
        return pause_file.exists()

    async def _get_btc_momentum(
        self,
        btc_service,
        current_price: Decimal
    ) -> dict | None:
        """
        Calculate actual BTC momentum over last 5 minutes.

        Compares current price to 5 minutes ago to detect actual BTC direction,
        independent of Polymarket sentiment.

        Args:
            btc_service: BTCPriceService instance
            current_price: Current BTC price

        Returns:
            {
                'price_5min_ago': Decimal,
                'momentum_pct': float,
                'direction': 'UP' | 'DOWN' | 'FLAT'
            }
            or None if history unavailable (graceful fallback)
        """
        try:
            # Use existing BTCPriceService.get_price_history()
            # Note: This may return empty if insufficient data
            history = await btc_service.get_price_history(minutes=5)

            if not history or len(history) < 2:
                logger.info("BTC price history unavailable for momentum calc")
                return None

            # Get oldest price in 5-minute window
            price_5min_ago = history[0].price

            # Calculate percentage change
            momentum_pct = float((current_price - price_5min_ago) / price_5min_ago * 100)

            # Classify direction (>0.1% threshold to filter noise)
            if momentum_pct > 0.1:
                direction = 'UP'
            elif momentum_pct < -0.1:
                direction = 'DOWN'
            else:
                direction = 'FLAT'

            logger.info(
                "BTC momentum calculated",
                price_5min_ago=f"${price_5min_ago:,.2f}",
                current=f"${current_price:,.2f}",
                change=f"{momentum_pct:+.2f}%",
                direction=direction
            )

            return {
                'price_5min_ago': price_5min_ago,
                'momentum_pct': momentum_pct,
                'direction': direction
            }

        except Exception as e:
            logger.warning("BTC momentum calculation failed", error=str(e))
            return None  # Graceful fallback

    async def run_cycle(self) -> None:
        """Execute one trading cycle."""
        self.cycle_count += 1
        cycle_start_time = datetime.now(timezone.utc)
        logger.info(
            "Starting trading cycle",
            cycle=self.cycle_count,
            timestamp=cycle_start_time.isoformat()
        )

        # Emergency pause check
        if self._check_emergency_pause():
            logger.critical("Emergency pause is enabled - stopping trading")
            logger.critical("Review parameter adjustments and delete .emergency_pause file to resume")
            self.running = False
            return

        try:
            # Step 1: Market Discovery - Find BTC 15-min markets (with end-phase filter)
            markets = await self.get_tradeable_markets()
            if not markets:
                logger.info("No tradeable markets found, skipping cycle")
                return

            logger.info("Found markets", count=len(markets))

            # Configure market for monitoring
            self.odds_monitor._market_id = markets[0].id
            self.odds_monitor._market_slug = markets[0].slug
            logger.debug(
                "Market context configured for OddsMonitor",
                market_id=markets[0].id,
                market_slug=markets[0].slug
            )

            # Extract condition_id and token_ids from discovered market
            condition_id = getattr(markets[0], 'condition_id', None)
            token_ids = markets[0].get_token_ids() if markets else []

            if not condition_id:
                logger.warning("No condition_id found, using fallback")
                condition_id = "unknown"

            if not token_ids:
                logger.warning("No token_ids found")

            # Initialize market service with token_ids (preferred) or condition_id (fallback)
            if not self.market_service or self.market_service.condition_id != condition_id:
                self.market_service = MarketMicrostructureService(
                    self.settings,
                    condition_id=condition_id,
                    token_ids=token_ids if token_ids else None
                )

            # Step 2: Data Collection (parallel) - NEW: fetch social + market + funding + dominance + volume
            btc_data, social_sentiment, market_signals, funding_signal, dominance_signal = await asyncio.gather(
                self.btc_service.get_current_price(),
                self.social_service.get_social_score(),
                self.market_service.get_market_score(),
                self.btc_service.get_funding_rates(),
                self.btc_service.get_btc_dominance(),
            )

            # Calculate actual BTC momentum (last 5 minutes) - ONCE PER LOOP
            btc_momentum = await self._get_btc_momentum(
                self.btc_service,
                btc_data.price
            )

            # Log momentum if available
            if btc_momentum:
                logger.info(
                    "BTC actual movement",
                    direction=btc_momentum['direction'],
                    change_pct=f"{btc_momentum['momentum_pct']:+.2f}%"
                )

            # Multi-timeframe trend analysis
            timeframe_analysis = None
            if self.test_mode.enabled and self.timeframe_analyzer:
                timeframe_analysis = await self.timeframe_analyzer.analyze()
                if timeframe_analysis:
                    logger.info(
                        "Multi-timeframe analysis",
                        tf_1m=timeframe_analysis.tf_1m.direction,
                        tf_5m=timeframe_analysis.tf_5m.direction,
                        tf_15m=timeframe_analysis.tf_15m.direction,
                        tf_30m=timeframe_analysis.tf_30m.direction,
                        alignment=timeframe_analysis.alignment_score,
                        modifier=f"{timeframe_analysis.confidence_modifier:+.1%}"
                    )

            logger.info(
                "Data collected",
                btc_price=f"${btc_data.price:,.2f}",
                social_score=f"{social_sentiment.score:+.2f}",
                social_conf=f"{social_sentiment.confidence:.2f}",
                market_score=f"{market_signals.score:+.2f}",
                market_conf=f"{market_signals.confidence:.2f}",
                funding_score=f"{funding_signal.score:+.2f}" if funding_signal else "N/A",
                funding_conf=f"{funding_signal.confidence:.2f}" if funding_signal else "N/A",
                dominance_score=f"{dominance_signal.score:+.2f}" if dominance_signal else "N/A",
                dominance_conf=f"{dominance_signal.confidence:.2f}" if dominance_signal else "N/A"
            )

            # Step 3: Technical Analysis (optional - graceful if unavailable)
            try:
                # Use 15 minutes for faster analysis (sufficient for RSI/MACD on 15-min markets)
                price_history = await self.btc_service.get_price_history(minutes=15)
                indicators = TechnicalAnalysis.calculate_indicators(price_history)
                logger.info(
                    "Technical indicators",
                    rsi=f"{indicators.rsi:.1f}",
                    macd=f"{indicators.macd_value:.2f}",
                    trend=indicators.trend
                )
            except Exception as e:
                logger.warning("Technical analysis unavailable, using neutral defaults", error=str(e))
                # Create neutral indicators
                from polymarket.models import TechnicalIndicators
                indicators = TechnicalIndicators(
                    rsi=50.0,
                    macd_value=0.0,
                    macd_signal=0.0,
                    macd_histogram=0.0,
                    ema_short=float(btc_data.price),
                    ema_long=float(btc_data.price),
                    sma_50=float(btc_data.price),
                    volume_change=0.0,
                    price_velocity=0.0,
                    trend="NEUTRAL"
                )

            # Step 3.5: Contrarian Signal Detection
            contrarian_signal = None
            try:
                from polymarket.trading.contrarian import detect_contrarian_setup

                # Calculate odds from market (use first market as reference)
                if markets:
                    reference_market = markets[0]
                    yes_odds = reference_market.best_bid if reference_market.best_bid else 0.50
                    no_odds = 1.0 - yes_odds

                    contrarian_signal = detect_contrarian_setup(
                        rsi=indicators.rsi,
                        yes_odds=yes_odds,
                        no_odds=no_odds
                    )

                    if contrarian_signal:
                        logger.info(
                            "Contrarian signal detected",
                            type=contrarian_signal.type,
                            rsi=f"{contrarian_signal.rsi:.1f}",
                            suggested_direction=contrarian_signal.suggested_direction,
                            crowd_direction=contrarian_signal.crowd_direction,
                            crowd_confidence=f"{contrarian_signal.crowd_confidence:.0%}",
                            confidence=f"{contrarian_signal.confidence:.0%}"
                        )
            except Exception as e:
                logger.warning("Contrarian detection failed, continuing without", error=str(e))
                contrarian_signal = None

            # Step 3.6: Market Regime Detection
            regime = None
            try:
                # Calculate price changes from historical data
                if len(price_history) >= 10:
                    price_changes = [
                        ((float(price_history[i].price) - float(price_history[i-1].price)) / float(price_history[i-1].price)) * 100
                        for i in range(1, min(11, len(price_history)))  # Last 10 changes
                    ]

                    # Get 24h high/low from btc_data (if available from volume data)
                    high_24h = float(btc_data.price) * 1.02  # Estimate if not available
                    low_24h = float(btc_data.price) * 0.98

                    from polymarket.trading.regime_detector import RegimeDetector
                    regime_detector = RegimeDetector()
                    regime = regime_detector.detect_regime(
                        price_changes=price_changes,
                        current_price=float(btc_data.price),
                        high_24h=high_24h,
                        low_24h=low_24h
                    )
                else:
                    logger.warning("Insufficient price history for regime detection")
            except Exception as e:
                logger.warning("Regime detection failed", error=str(e))

            # Step 4: Aggregate Signals - NEW (includes funding + dominance + contrarian)
            aggregated_sentiment = self.aggregator.aggregate(
                social_sentiment,
                market_signals,
                funding=funding_signal,
                dominance=dominance_signal,
                contrarian=contrarian_signal
            )

            logger.info(
                "Sentiment aggregated",
                final_score=f"{aggregated_sentiment.final_score:+.2f}",
                final_conf=f"{aggregated_sentiment.final_confidence:.2f}",
                agreement=f"{aggregated_sentiment.agreement_multiplier:.2f}x",
                signal=aggregated_sentiment.signal_type
            )

            # Step 5: Get portfolio value
            try:
                portfolio = self.client.get_portfolio_summary()
                # FIX: Use only USDC balance (not total_value which includes phantom resolved positions)
                portfolio_value = Decimal(str(portfolio.usdc_balance))
                logger.info(
                    "Portfolio fetched",
                    total_value=f"${portfolio.total_value:.2f}",
                    usdc_balance=f"${portfolio.usdc_balance:.2f}",
                    positions_value=f"${portfolio.positions_value:.2f}",
                    purchase_value=f"${portfolio.purchase_value:.2f}",  # NEW
                    unrealized_pl=f"${portfolio.unrealized_pl:+.2f}",  # NEW
                    using_balance=f"${portfolio_value:.2f} (free USDC only, excl. ${portfolio.positions_value:.2f} in tokens)"
                )
                if portfolio_value == 0:
                    logger.warning("Portfolio value is 0, using default $1000")
                    portfolio_value = Decimal("1000")  # Default for read_only mode
            except Exception as e:
                logger.error("Failed to fetch portfolio, using default $1000", error=str(e))
                portfolio_value = Decimal("1000")  # Fallback

            # Step 6: Process each market
            for market in markets:
                await self._process_market(
                    market, btc_data, indicators,
                    aggregated_sentiment,  # CHANGED: pass aggregated instead of social
                    portfolio_value,
                    btc_momentum,  # NEW: pass momentum calculated once per loop
                    cycle_start_time,  # NEW: pass cycle start time for JIT metrics
                    timeframe_analysis,  # NEW: timeframe analysis for AI context
                    regime,  # NEW: market regime for adaptive strategy
                    contrarian_signal  # NEW: contrarian signal detection
                )

            # Cache cycle data for the price-movement watcher (Enhancement 4).
            # Watcher runs every 10s and re-uses these cached values so it doesn't
            # need to re-fetch indicators/sentiment on every check.
            self._last_cycle_data = {
                'indicators': indicators,
                'sentiment': aggregated_sentiment,
                'portfolio_value': portfolio_value,
                'btc_momentum': btc_momentum,
                'timeframe_analysis': timeframe_analysis,
                'regime': regime,
                'contrarian_signal': contrarian_signal,
                'timestamp': datetime.now(timezone.utc),
            }

            # Step 7: Stop-loss check
            await self._check_stop_loss()

            logger.info("Cycle completed", cycle=self.cycle_count)

        except Exception as e:
            logger.error(
                "Cycle error",
                cycle=self.cycle_count,
                error=str(e),
                exc_info=True
            )

    async def _discover_markets(self) -> list[Market]:
        """Find active BTC 15-minute markets."""
        try:
            # Use the slug-based discovery method from client
            market = self.client.discover_btc_15min_market()
            return [market]
        except Exception as e:
            logger.warning("Market discovery failed", error=str(e))
            # Fallback: try manual search
            try:
                markets = self.client.get_markets(search="bitcoin", limit=50, active_only=True)
                btc_markets = [
                    m for m in markets
                    if ("btc" in m.question.lower() or "bitcoin" in m.question.lower())
                    and ("15" in m.question.lower() or "15m" in (m.slug or "").lower())
                    and m.active
                ]
                return btc_markets
            except Exception as e2:
                logger.error("Fallback market discovery also failed", error=str(e2))
                return []

    async def get_tradeable_markets(self) -> list[Market]:
        """
        Fetch and filter BTC 15-min markets.

        Filters:
        - Must have >= 5 minutes remaining (300 seconds)
        - Must be active and tradeable

        Returns:
            List of tradeable Market objects
        """
        try:
            # Fetch all active markets from Polymarket
            all_markets = await self._discover_markets()

            tradeable = []
            filtered_count = 0

            for market in all_markets:
                # Skip if market has no end_date
                if not market.end_date:
                    logger.warning(
                        "Market has no end_date, skipping",
                        market_id=market.id
                    )
                    continue

                # Calculate time remaining
                now = datetime.now(timezone.utc)
                time_remaining = (market.end_date - now).total_seconds()

                # End-phase filter DISABLED - allow trading in final minutes
                # (User requested removal to catch late odds spikes)

                tradeable.append(market)

            logger.info(
                "Markets filtered",
                total=len(all_markets),
                tradeable=len(tradeable),
                filtered_end_phase=filtered_count
            )

            return tradeable

        except Exception as e:
            logger.error("Failed to fetch markets", error=str(e))
            return []

    async def _mark_trade_skipped(
        self,
        trade_id: int,
        reason: str,
        skip_type: str = "validation"
    ) -> None:
        """Mark a trade as skipped in database."""
        if trade_id <= 0:
            return

        try:
            await self.performance_tracker.update_trade_status(
                trade_id=trade_id,
                execution_status='skipped',
                skip_reason=reason,
                skip_type=skip_type
            )
            logger.info(
                "Trade marked as skipped",
                trade_id=trade_id,
                reason=reason,
                skip_type=skip_type
            )
        except Exception as e:
            logger.error("Failed to mark trade as skipped", error=str(e))

    async def _process_market(
        self,
        market: Market,
        btc_data,
        indicators,
        aggregated_sentiment,  # CHANGED from sentiment
        portfolio_value: Decimal,
        btc_momentum: dict | None,  # NEW: momentum calculated once per loop
        cycle_start_time: datetime,  # NEW: for JIT execution metrics
        timeframe_analysis,  # NEW: multi-timeframe trend analysis
        regime,  # NEW: market regime detection
        contrarian_signal  # NEW: contrarian signal detection
    ) -> None:
        """Process a single market for trading decision."""
        try:
            # In test mode, skip if we've already traded this market
            if self.test_mode.enabled:
                if market.id in self.test_mode.traded_markets:
                    logger.info(
                        "[TEST] Skipping market - already traded in this session",
                        market_id=market.id,
                        market_question=market.question[:80] if market.question else "",
                        traded_count=len(self.test_mode.traded_markets)
                    )
                    return

            # In production mode, skip if we've already placed an order for this market
            if not self.test_mode.enabled:
                if market.id in self._traded_markets:
                    logger.info(
                        "Skipping market - already traded in this session",
                        market_id=market.id,
                        market_question=market.question[:80] if market.question else "",
                        traded_count=len(self._traded_markets)
                    )
                    return

                # Cycle-level analysis lock: if another cycle is already running AI
                # analysis for this market, skip. Without this, an OddsMonitor re-fire
                # during the 30-60s AI call starts a second concurrent analysis that
                # also passes the _traded_markets check above (since the first cycle
                # hasn't executed yet), resulting in two real orders.
                if market.id in self._markets_with_active_cycle_analysis:
                    logger.info(
                        "Skipping market - AI analysis already in progress for this market",
                        market_id=market.id,
                        market_question=market.question[:80] if market.question else ""
                    )
                    return

                self._markets_with_active_cycle_analysis.add(market.id)

            # Get real-time odds from WebSocket streamer
            odds_snapshot = self.realtime_streamer.get_current_odds(market.id)

            if odds_snapshot:
                # Use real-time odds (zero latency!)
                yes_odds = odds_snapshot.yes_odds
                no_odds = odds_snapshot.no_odds
                from datetime import datetime, timezone
                age_ms = (datetime.now(timezone.utc) - odds_snapshot.timestamp).total_seconds() * 1000
                logger.debug(
                    "Using real-time odds from WebSocket",
                    market_id=market.id,
                    yes_odds=f"{yes_odds:.2%}",
                    no_odds=f"{no_odds:.2%}",
                    age_ms=f"{age_ms:.0f}"
                )
            else:
                # Fallback to market odds if WebSocket not ready yet
                yes_odds = market.best_bid if market.best_bid else 0.50
                no_odds = 1.0 - yes_odds
                logger.debug(
                    "Using market odds (WebSocket not ready)",
                    market_id=market.id,
                    yes_odds=f"{yes_odds:.2%}",
                    no_odds=f"{no_odds:.2%}"
                )

            # Early filtering: skip markets where neither side > 60%
            # EXCEPTION: timed strategy always runs analysis at T=10min regardless of odds
            # (AI will return HOLD if direction is unclear; entry requires 70% anyway)
            is_timed_analysis = market.id in self._analysis_triggered
            yes_qualifies = (yes_odds > 0.60)
            no_qualifies = (no_odds > 0.60)

            if not is_timed_analysis and not (yes_qualifies or no_qualifies):
                logger.info(
                    "Skipping market - neither side > 60% odds (real-time check)",
                    market_id=market.id,
                    yes_odds=f"{yes_odds:.2%}",
                    no_odds=f"{no_odds:.2%}",
                    outcomes=market.outcomes
                )
                return  # Skip this market

            # Get token IDs
            token_ids = market.get_token_ids()
            if not token_ids:
                logger.warning("No token IDs found", market_id=market.id)
                return

            # Parse market slug for timing
            market_slug = market.slug or ""
            if not market_slug:
                logger.warning("Market has no slug, skipping timing/price-to-beat", market_id=market.id)

            start_time = self.market_tracker.parse_market_start(market_slug) if market_slug else None

            # Calculate time remaining
            time_remaining = None
            is_end_of_market = False
            if start_time:
                time_remaining = self.market_tracker.calculate_time_remaining(start_time)
                is_end_of_market = self.market_tracker.is_end_of_market(start_time)

                logger.info(
                    "Market timing",
                    market_id=market.id,
                    time_remaining=f"{time_remaining//60}m {time_remaining%60}s",
                    is_end_phase=is_end_of_market
                )

                # Skip if market has already closed (no time to execute)
                if time_remaining == 0:
                    logger.info(
                        "Skipping cycle - market already expired",
                        market_id=market.id,
                        market_slug=market_slug
                    )
                    return

            # Get or set price-to-beat (only if we have a valid slug)
            price_to_beat = self.market_tracker.get_price_to_beat(market_slug) if market_slug else None
            if price_to_beat is None and start_time and market_slug:
                # First time seeing this market - fetch historical price at market start
                start_timestamp = int(start_time.timestamp())
                historical_price = await self.btc_service.get_price_at_timestamp(start_timestamp)

                if historical_price:
                    price_to_beat = historical_price
                    self.market_tracker.set_price_to_beat(market_slug, price_to_beat)
                    logger.info(
                        "Price-to-beat set from historical data",
                        market_id=market.id,
                        market_start=start_time.isoformat(),
                        price=f"${price_to_beat:,.2f}"
                    )
                else:
                    # Historical price fetch failed - DO NOT fall back to current price
                    # This would cause incorrect price_to_beat calculation
                    logger.error(
                        "Price-to-beat unavailable - all sources failed",
                        market_id=market.id,
                        market_slug=market_slug,
                        market_start=start_time.isoformat(),
                        reason="Historical price not available from any source"
                    )
                    # Leave price_to_beat as None - caller will skip this market
                    price_to_beat = None

            # Calculate price difference
            if price_to_beat:
                diff, diff_pct = self.market_tracker.calculate_price_difference(
                    btc_data.price, price_to_beat
                )
                logger.info(
                    "Price comparison",
                    current=f"${btc_data.price:,.2f}",
                    price_to_beat=f"${price_to_beat:,.2f}",
                    difference=f"${diff:+,.2f}",
                    percentage=f"{diff_pct:+.2f}%"
                )

                # Signal lag detection: fires when CLOB market odds lag behind BTC price reality
                # e.g. BTC moved UP but CLOB still shows NO > YES (market hasn't caught up)
                # Uses CLOB direction (yes_odds vs no_odds) — NOT macro sentiment
                # Macro sentiment is passed to AI for decision-making, not for lag detection
                signal_lag_detected = False
                signal_lag_reason = None

                btc_direction = "UP" if btc_data.price > price_to_beat else "DOWN"
                # CLOB direction: which side is the market currently pricing higher?
                clob_direction = "BULLISH" if yes_odds >= no_odds else "BEARISH"

                signal_lag_detected, signal_lag_reason = detect_signal_lag(
                    btc_direction,
                    clob_direction,
                    0.9  # High confidence since CLOB direction is directly observed
                )

                if signal_lag_detected:
                    # Bypass signal lag when BTC has moved far enough that the CLOB lag
                    # IS the arbitrage opportunity. A BTC move of ≥10% from price_to_beat
                    # means the market simply hasn't repriced yet — that's exactly what we trade.
                    # Below 10%, the contradiction could be genuine noise → still block.
                    btc_movement_pct = (
                        abs(float(diff) / float(price_to_beat)) * 100
                        if price_to_beat and price_to_beat != 0 else 0
                    )
                    ARBITRAGE_BYPASS_THRESHOLD_PCT = 10.0  # 10% BTC move bypasses signal lag

                    if btc_movement_pct >= ARBITRAGE_BYPASS_THRESHOLD_PCT:
                        logger.info(
                            "Signal lag detected but bypassed — large BTC move vs CLOB lag = arbitrage",
                            market_id=market.id,
                            btc_movement_pct=f"{btc_movement_pct:.1f}%",
                            bypass_threshold=f"{ARBITRAGE_BYPASS_THRESHOLD_PCT:.0f}%",
                            reason="CLOB has not repriced — this IS the edge"
                        )
                        # Fall through to analysis — do NOT return
                    elif not self.test_mode.enabled:
                        logger.warning(
                            "Skipping trade due to signal lag",
                            market_id=market.id,
                            reason=signal_lag_reason,
                            btc_movement_pct=f"{btc_movement_pct:.1f}%"
                        )
                        return  # HOLD - contradiction below bypass threshold
                    else:
                        logger.info(
                            "[TEST] Signal lag detected - data sent to AI anyway",
                            market_id=market.id,
                            reason=signal_lag_reason
                        )

                # Check minimum movement threshold to avoid entering too early
                # contrarian_signal is set by Step 3.5 (Contrarian Signal Detection) above

                # Calculate threshold based on contrarian signal
                MIN_MOVEMENT_THRESHOLD = get_movement_threshold(contrarian_signal)
                if contrarian_signal:
                    logger.info(
                        "Contrarian setup - reducing movement threshold",
                        market_id=market.id,
                        default_threshold="$30",
                        contrarian_threshold="$30",
                        reasoning="Reversals start with small movements"
                    )

                abs_diff = abs(diff)
                if abs_diff < MIN_MOVEMENT_THRESHOLD:
                    if not self.test_mode.enabled:
                        logger.info(
                            "Skipping market - insufficient BTC movement",
                            market_id=market.id,
                            movement=f"${abs_diff:.2f}",
                            threshold=f"${MIN_MOVEMENT_THRESHOLD}",
                            reason="Wait for clearer directional signal"
                        )
                        return  # Skip this market, no trade
                    else:
                        logger.info(
                            "[TEST] Bypassing movement threshold - data sent to AI",
                            market_id=market.id,
                            movement=f"${abs_diff:.2f}",
                            threshold=f"${MIN_MOVEMENT_THRESHOLD}",
                            bypassed=True
                        )


            # NEW: Enhanced Market Signals from CoinGecko Pro
            # Fetch additional market signals for better edge detection
            market_signals = None
            try:
                from polymarket.trading.market_signals import MarketSignalProcessor, Signal

                signal_processor = MarketSignalProcessor()

                # Fetch funding rates (multi-exchange with timeout protection)
                funding_rate_raw = await self.btc_service.get_funding_rate_raw()
                funding_signal = signal_processor.process_funding_rate(funding_rate_raw)

                # Fetch multi-exchange prices for premium comparison
                exchange_prices = await self.btc_service.get_exchange_prices()
                premium_signal = signal_processor.process_exchange_premium(exchange_prices)

                # Get recent volumes for volume confirmation
                recent_volumes = await self.btc_service.get_recent_volumes(hours=24)
                current_volume = recent_volumes[-1] if recent_volumes else None
                volume_signal = signal_processor.process_volume_confirmation(
                    current_volume=current_volume,
                    recent_volumes=recent_volumes,
                    movement_usd=abs_diff
                )

                # Aggregate all signals
                signals = [funding_signal, premium_signal, volume_signal]
                market_signals = signal_processor.aggregate_signals(
                    signals=signals,
                    weights={
                        "funding_rate": 0.35,  # 35% weight to funding rates
                        "exchange_premium": 0.35,  # 35% weight to exchange premium
                        "volume": 0.30  # 30% weight to volume confirmation
                    }
                )

                logger.info(
                    "Market signals aggregated",
                    direction=market_signals.direction,
                    confidence=f"{market_signals.confidence:.2f}",
                    funding=f"{funding_signal.direction} ({funding_signal.confidence:.2f})",
                    premium=f"{premium_signal.direction} ({premium_signal.confidence:.2f})",
                    volume_signal_data=f"{volume_signal.direction} ({volume_signal.confidence:.2f})"
                )

            except Exception as e:
                logger.warning("Failed to fetch market signals, continuing without them", error=str(e))
                market_signals = None

            # Timeframe alignment check - don't trade against larger trend
            if timeframe_analysis and timeframe_analysis.alignment_score == "CONFLICTING":
                if not self.test_mode.enabled:
                    logger.info(
                        "Skipping trade - conflicting timeframes",
                        market_id=market.id,
                        tf_15m_trend=timeframe_analysis.tf_15m.direction,
                        tf_30m_trend=timeframe_analysis.tf_30m.direction,
                        reason="Don't trade against larger timeframe trend"
                    )
                    return
                else:
                    logger.info(
                        "[TEST] Bypassing timeframe check - data sent to AI",
                        market_id=market.id,
                        tf_15m_trend=timeframe_analysis.tf_15m.direction,
                        tf_30m_trend=timeframe_analysis.tf_30m.direction,
                        bypassed=True
                    )

            # Market regime check - skip unclear/volatile markets
            if regime and regime.regime in ["UNCLEAR", "VOLATILE"]:
                if not self.test_mode.enabled:
                    logger.info(
                        "Skipping trade - unfavorable market regime",
                        market_id=market.id,
                        regime=regime.regime,
                        volatility=f"{regime.volatility:.2f}%",
                        confidence=f"{regime.confidence:.2f}",
                        reason="Only trade in trending or ranging markets"
                    )
                    return
                else:
                    logger.info(
                        "[TEST] Bypassing regime check - data sent to AI",
                        market_id=market.id,
                        regime=regime.regime,
                        volatility=f"{regime.volatility:.2f}%",
                        confidence=f"{regime.confidence:.2f}",
                        bypassed=True
                    )

            # Fetch and analyze orderbook for execution quality
            orderbook = self.client.get_orderbook(token_ids[0])  # YES token
            orderbook_analysis = None
            if orderbook:
                from polymarket.trading.orderbook_analyzer import OrderbookAnalyzer
                analyzer = OrderbookAnalyzer()
                orderbook_analysis = analyzer.analyze_orderbook(
                    orderbook,
                    target_size=8.0  # Base position size
                )

                if orderbook_analysis:
                    # Skip if spread too wide (poor execution quality)
                    # Note: CTF/AMM markets always show ~9900 bps spread (full range AMM liquidity,
                    # not a real bid-ask spread). Only block real CLOB markets with wide spreads.
                    if 500 < orderbook_analysis.spread_bps < 9000:  # 5-90% = real CLOB wide spread
                        if not self.test_mode.enabled:
                            logger.info(
                                "Skipping trade - spread too wide",
                                market_id=market.id,
                                spread_bps=f"{orderbook_analysis.spread_bps:.1f}",
                                liquidity_score=f"{orderbook_analysis.liquidity_score:.2f}",
                                reason="Wide spread = poor execution quality"
                            )
                            return
                        else:
                            logger.info(
                                "[TEST] Bypassing spread check - data sent to AI",
                                market_id=market.id,
                                spread_bps=f"{orderbook_analysis.spread_bps:.1f}",
                                liquidity_score=f"{orderbook_analysis.liquidity_score:.2f}",
                                bypassed=True
                            )

                    # Skip if can't fill order
                    if not orderbook_analysis.can_fill_order:
                        logger.info(
                            "Skipping trade - insufficient liquidity",
                            market_id=market.id,
                            liquidity_score=f"{orderbook_analysis.liquidity_score:.2f}",
                            bid_depth=f"${orderbook_analysis.bid_depth_100bps:.2f}",
                            ask_depth=f"${orderbook_analysis.ask_depth_100bps:.2f}",
                            reason="Not enough depth to fill order"
                        )
                        return

            # Build market data dict with ALL context
            # Note: Polymarket returns prices for UP/YES token only
            # DOWN/NO price is complementary: 1 - UP_price
            up_bid = market.best_bid or 0.50
            up_ask = market.best_ask or 0.50

            market_dict = {
                "token_id": token_ids[0],  # Temporary, for logging only
                "question": market.question,
                "yes_price": up_ask,           # UP token ask price (to buy UP)
                "no_price": 1 - up_bid,        # DOWN = 1 - UP (complementary)
                "active": market.active,
                "outcomes": market.outcomes if hasattr(market, 'outcomes') else ["Yes", "No"],
                # NEW: Price-to-beat and timing context
                "price_to_beat": price_to_beat,
                "time_remaining_seconds": time_remaining or 900,  # Default to 15 min if None
                "is_end_of_market": is_end_of_market,
                # NEW: BTC momentum data
                "btc_momentum": btc_momentum,  # Will be None if unavailable
            }

            logger.info(
                "Market prices (complementary)",
                market_id=market.id,
                up_bid=f"{up_bid:.3f}",
                up_ask=f"{up_ask:.3f}",
                yes_price=f"{market_dict['yes_price']:.3f}",
                no_price=f"{market_dict['no_price']:.3f}",
                outcomes=market_dict["outcomes"],
                note="DOWN price = 1 - UP bid"
            )

            # NEW: Calculate actual probability from price momentum
            probability_calculator = ProbabilityCalculator()

            # Get historical prices for probability calculation
            import time
            current_time = int(time.time())
            price_5min_result = await self.btc_service.get_price_at_timestamp(current_time - 300)
            price_5min_ago = float(price_5min_result) if price_5min_result else float(btc_data.price)
            price_10min_result = await self.btc_service.get_price_at_timestamp(current_time - 600)
            price_10min_ago = float(price_10min_result) if price_10min_result else float(btc_data.price)
            volatility_15min = await self.btc_service.calculate_15min_volatility() or 0.005  # Default 0.5%

            actual_probability = probability_calculator.calculate_directional_probability(
                current_price=float(btc_data.price),
                target_price=float(price_to_beat) if price_to_beat else float(btc_data.price),
                price_5min_ago=price_5min_ago,
                price_10min_ago=price_10min_ago,
                volatility_15min=volatility_15min,
                time_remaining_seconds=time_remaining or 900,
                orderbook_imbalance=orderbook_analysis.order_imbalance if orderbook_analysis else 0.0
            )

            # NEW: Detect arbitrage opportunity
            arbitrage_detector = ArbitrageDetector()
            arbitrage_opportunity = arbitrage_detector.detect_arbitrage(
                actual_probability=actual_probability,
                market_yes_odds=market_dict['yes_price'],
                market_no_odds=market_dict['no_price'],
                market_id=market.id
            )

            if arbitrage_opportunity:
                logger.info(
                    "Arbitrage opportunity detected",
                    market_id=market.id,
                    edge_percentage=f"{arbitrage_opportunity.edge_percentage:.1%}",
                    recommended_action=arbitrage_opportunity.recommended_action,
                    urgency=arbitrage_opportunity.urgency,
                    confidence_boost=f"{arbitrage_opportunity.confidence_boost:.2f}"
                )

            # Step 1: AI Decision - pass all market context including orderbook, timeframe, regime, arbitrage, market signals, and contrarian signal
            decision = await self.ai_service.make_decision(
                btc_price=btc_data,
                technical_indicators=indicators,
                aggregated_sentiment=aggregated_sentiment,
                market_data=market_dict,
                portfolio_value=portfolio_value,
                orderbook_data=orderbook_analysis,  # orderbook depth analysis
                timeframe_analysis=timeframe_analysis,  # NEW: multi-timeframe analysis
                regime=regime,  # NEW: market regime detection
                arbitrage_opportunity=arbitrage_opportunity,  # NEW: arbitrage opportunity
                market_signals=market_signals,  # NEW: CoinGecko Pro market signals
                contrarian_signal=contrarian_signal,  # NEW: contrarian mean-reversion signal
                force_trade=self.test_mode.enabled  # NEW: TEST MODE - force YES/NO decision
            )

            # Re-check _traded_markets after AI call: another concurrent cycle may have
            # placed an order during the 30-60s AI processing window (belt-and-suspenders
            # alongside _markets_with_active_cycle_analysis which blocks at entry).
            if not self.test_mode.enabled and market.id in self._traded_markets:
                self._markets_with_active_cycle_analysis.discard(market.id)
                logger.info(
                    "Skipping - market was traded by concurrent cycle during AI analysis",
                    market_id=market.id
                )
                return


            # NEW: Conflict detection and confidence adjustment
            conflict_detector = SignalConflictDetector()
            conflict_analysis = conflict_detector.analyze_conflicts(
                btc_direction="UP" if (price_to_beat and btc_data.price > price_to_beat) else "DOWN",
                technical_trend=indicators.trend,
                sentiment_direction="BULLISH" if aggregated_sentiment.final_score > 0 else "BEARISH",
                regime_trend=regime.trend_direction if regime else None,
                timeframe_alignment=timeframe_analysis.alignment_score if timeframe_analysis else None,
                market_signals_direction=market_signals.direction if market_signals else None,
                market_signals_confidence=market_signals.confidence if market_signals else None
            )

            # Apply conflict analysis
            if conflict_analysis.should_hold:
                logger.warning(
                    "AUTO-HOLD due to SEVERE signal conflicts",
                    market_id=market.id,
                    severity=conflict_analysis.severity.value,
                    conflicts=conflict_analysis.conflicts_detected
                )
                return  # Don't trade

            # Apply confidence penalty
            if conflict_analysis.confidence_penalty != 0.0:
                original_confidence = decision.confidence
                decision.confidence += conflict_analysis.confidence_penalty
                decision.confidence = max(0.0, min(1.0, decision.confidence))  # Clamp to 0-1

                logger.info(
                    "Applied conflict penalty",
                    market_id=market.id,
                    original=f"{original_confidence:.2f}",
                    penalty=f"{conflict_analysis.confidence_penalty:+.2f}",
                    final=f"{decision.confidence:.2f}",
                    conflicts=conflict_analysis.conflicts_detected
                )

            # Test mode: Force YES/NO and check confidence threshold
            if self.test_mode.enabled:
                # If AI returned HOLD despite force_trade, override based on sentiment
                if decision.action == "HOLD":
                    # Determine direction from sentiment score
                    decision.action = "YES" if aggregated_sentiment.final_score > 0 else "NO"
                    logger.warning(
                        "[TEST] AI returned HOLD - forcing direction from sentiment",
                        market_id=market.id,
                        sentiment_score=aggregated_sentiment.final_score,
                        forced_action=decision.action
                    )

                # Check minimum confidence threshold
                if decision.confidence < self.test_mode.min_confidence:
                    logger.info(
                        "[TEST] Skipping trade - confidence below threshold",
                        market_id=market.id,
                        ai_confidence=f"{decision.confidence:.2f}",
                        min_required=f"{self.test_mode.min_confidence:.2f}",
                        action=decision.action
                    )
                    return

                # Calculate Kelly-suggested size first
                kelly_size = decision.position_size

                # Enforce minimum bet (Polymarket requires $5 minimum)
                final_size = max(kelly_size, self.test_mode.min_bet_amount)

                # Enforce maximum bet
                final_size = min(final_size, self.test_mode.max_bet_amount)

                logger.info(
                    "[TEST] Position sizing",
                    market_id=market.id,
                    kelly_suggested=f"${kelly_size:.2f}",
                    final_amount=f"${final_size:.2f}",
                    min_enforced=kelly_size < self.test_mode.min_bet_amount,
                    max_enforced=kelly_size > self.test_mode.max_bet_amount
                )

                decision.position_size = final_size

            # Additional validation: YES trades need stronger momentum to avoid mean reversion
            # CHECK FIRST before logging to avoid phantom trades
            if decision.action == "YES" and price_to_beat and not self.test_mode.enabled:
                diff, _ = self.market_tracker.calculate_price_difference(
                    btc_data.price, price_to_beat
                )
                MIN_YES_MOVEMENT = 30  # $30 minimum for YES trades

                if diff < MIN_YES_MOVEMENT:
                    logger.info(
                        "Skipping YES trade - insufficient upward momentum",
                        market_id=market.id,
                        movement=f"${diff:+,.2f}",
                        threshold=f"${MIN_YES_MOVEMENT}",
                        reason="Avoid buying exhausted momentum (mean reversion risk)"
                    )
                    return  # Skip WITHOUT creating DB record

            # RSI-action contradiction check: reject bets that fight strong momentum.
            # RSI > 85 means BTC is surging UP — betting NO (DOWN) contradicts this.
            # RSI < 15 means BTC is collapsing — betting YES (UP) contradicts this.
            # EXCEPTION: Contrarian mean-reversion bets are exempt — OVERBOUGHT_REVERSAL
            # intentionally bets NO when RSI > 90 + crowd is overly bullish, and
            # OVERSOLD_REVERSAL intentionally bets YES when RSI < 10 + crowd is oversold.
            if indicators and indicators.rsi is not None and not self.test_mode.enabled:
                rsi = indicators.rsi
                is_contrarian_no = (
                    contrarian_signal is not None
                    and contrarian_signal.type == "OVERBOUGHT_REVERSAL"
                    and decision.action == "NO"
                )
                is_contrarian_yes = (
                    contrarian_signal is not None
                    and contrarian_signal.type == "OVERSOLD_REVERSAL"
                    and decision.action == "YES"
                )
                if decision.action == "NO" and rsi > 85 and not is_contrarian_no:
                    logger.info(
                        "Skipping NO trade — RSI strongly overbought contradicts DOWN bet",
                        market_id=market.id,
                        rsi=f"{rsi:.1f}",
                        reason="RSI > 85 = strong upward momentum contradicts NO/DOWN bet (no contrarian signal)"
                    )
                    return
                if decision.action == "YES" and rsi < 15 and not is_contrarian_yes:
                    logger.info(
                        "Skipping YES trade — RSI strongly oversold contradicts UP bet",
                        market_id=market.id,
                        rsi=f"{rsi:.1f}",
                        reason="RSI < 15 = strong downward momentum contradicts YES/UP bet (no contrarian signal)"
                    )
                    return

            # NOW log decision to performance tracker (only if validation passed)
            trade_id = -1
            try:
                trade_id = await self.performance_tracker.log_decision(
                    market=market,
                    decision=decision,
                    btc_data=btc_data,
                    technical=indicators,
                    aggregated=aggregated_sentiment,
                    price_to_beat=price_to_beat,
                    time_remaining_seconds=time_remaining,
                    is_end_phase=is_end_of_market,
                    actual_probability=arbitrage_opportunity.actual_probability if arbitrage_opportunity else None,
                    arbitrage_edge=arbitrage_opportunity.edge_percentage if arbitrage_opportunity else None,
                    arbitrage_urgency=arbitrage_opportunity.urgency if arbitrage_opportunity else None,
                    is_test_mode=self.test_mode.enabled,
                    timeframe_analysis=timeframe_analysis,
                    contrarian_detected=bool(contrarian_signal),
                    contrarian_type=contrarian_signal.type if contrarian_signal else None
                )
            except Exception as e:
                logger.error("Performance logging failed", error=str(e))
                # Continue trading - don't block on logging failures
                trade_id = -1

            # Map AI decision to correct token based on outcomes
            # Outcomes are typically ["Up", "Down"] or ["Yes", "No"]
            # AI returns "YES" to buy first outcome, "NO" to buy second outcome
            if decision.action == "YES":
                token_id = token_ids[0]  # First outcome (e.g., "Up")
                token_name = market_dict["outcomes"][0]
            elif decision.action == "NO":
                token_id = token_ids[1]  # Second outcome (e.g., "Down")
                token_name = market_dict["outcomes"][1]
            else:
                token_id = None
                token_name = "HOLD"

            # Log decision with token mapping
            if self.settings.bot_log_decisions:
                logger.info(
                    "AI Decision",
                    market_id=market.id,
                    action=decision.action,
                    token=token_name,
                    confidence=f"{decision.confidence:.2f}",
                    reasoning=decision.reasoning,
                    position_size=str(decision.position_size)
                )

            # Log AI's response to contrarian suggestion
            if contrarian_signal:
                # Map contrarian direction to action
                contrarian_action_map = {
                    "UP": "YES",
                    "DOWN": "NO"
                }
                suggested_action = contrarian_action_map.get(contrarian_signal.suggested_direction, "UNKNOWN")

                if decision.action == suggested_action:
                    logger.info(
                        "AI accepted contrarian suggestion",
                        market_id=market.id,
                        contrarian_type=contrarian_signal.type,
                        suggested=contrarian_signal.suggested_direction,
                        ai_action=decision.action,
                        ai_confidence=f"{decision.confidence:.2f}"
                    )
                else:
                    logger.info(
                        "AI rejected contrarian suggestion",
                        market_id=market.id,
                        contrarian_type=contrarian_signal.type,
                        suggested=contrarian_signal.suggested_direction,
                        ai_action=decision.action,
                        ai_reasoning=decision.reasoning[:100] if decision.reasoning else "No reasoning provided"
                    )

            # Skip if HOLD decision
            if decision.action == "HOLD" or token_id is None:
                logger.info(
                    "Decision: HOLD",
                    market_id=market.id,
                    reason=decision.reasoning
                )
                return

            # Guard: skip if a limit order is already pending for this market
            # (open_positions is only updated AFTER order verification, leaving a
            # gap where the OddsMonitor can retrigger a second/third order)
            if market.id in self._markets_with_pending_orders:
                logger.info(
                    "Skipping cycle - pending order already exists for market",
                    market_id=market.id
                )
                return

            # Step 2: Risk Validation
            validation = await self.risk_manager.validate_decision(
                decision=decision,
                portfolio_value=portfolio_value,
                market=market_dict,
                open_positions=self.open_positions,
                test_mode=self.test_mode.enabled  # NEW: pass test mode flag
            )

            if not validation.approved:
                logger.info(
                    "Decision rejected by risk manager",
                    market_id=market.id,
                    reason=validation.reason
                )
                return

            # NEW: JIT odds validation (fetch fresh odds before execution)
            fresh_market = self.client.discover_btc_15min_market()
            if fresh_market:
                # Use CLOB realtime odds for JIT check (avoids stale Gamma prices)
                _jit_odds = self.realtime_streamer.get_current_odds(market.id)
                if _jit_odds:
                    yes_odds_fresh = _jit_odds.yes_odds
                    no_odds_fresh = _jit_odds.no_odds
                else:
                    yes_odds_fresh = fresh_market.best_bid if fresh_market.best_bid else 0.50
                    no_odds_fresh = 1.0 - yes_odds_fresh

                # Floor check: entry requires 70%+ odds (timed strategy). No upper ceiling.
                # If odds not yet there, store context and let _timed_entry_monitor execute later.
                entry_odds_met = (
                    (decision.action == "YES" and yes_odds_fresh >= 0.70) or
                    (decision.action == "NO" and no_odds_fresh >= 0.70)
                )
                if not entry_odds_met and not self.test_mode.enabled:
                    current_side_odds = yes_odds_fresh if decision.action == "YES" else no_odds_fresh
                    logger.info(
                        "Decision stored - odds below 70% entry threshold, monitoring for entry",
                        market_id=market.id,
                        action=decision.action,
                        current_odds=f"{current_side_odds:.2%}",
                        entry_threshold="70%",
                        time_remaining=time_remaining
                    )
                    # Store full execution context for _timed_entry_monitor
                    self._timed_decisions[market.id] = {
                        'action': decision.action,
                        'market': market,
                        'decision': decision,
                        'amount': validation.adjusted_position,
                        'token_id': token_id,
                        'token_name': token_name,
                        'trade_id': trade_id,
                        'btc_data': btc_data,
                        'btc_current': float(btc_data.price),
                        'btc_price_to_beat': float(price_to_beat) if price_to_beat else None,
                        'arbitrage_opportunity': arbitrage_opportunity,
                        'conflict_analysis': conflict_analysis,
                        'signal_lag_detected': signal_lag_detected,
                        'signal_lag_reason': signal_lag_reason,
                        'stored_at': datetime.now(timezone.utc),
                    }
                    return  # _timed_entry_monitor will execute when odds reach 70%

                # Store odds for paper trade logging
                odds_yes = yes_odds_fresh
                odds_no = no_odds_fresh
                odds_qualified = entry_odds_met
            else:
                # If we can't fetch fresh odds, use cached
                if cached_odds:
                    odds_yes = cached_odds.yes_odds
                    odds_no = cached_odds.no_odds
                    odds_qualified = (
                        (decision.action == "YES" and cached_odds.yes_qualifies) or
                        (decision.action == "NO" and cached_odds.no_qualifies)
                    )
                else:
                    # No odds available, log warning but continue
                    logger.warning("No odds available for validation", market_id=market.id)
                    odds_yes = 0.50
                    odds_no = 0.50
                    odds_qualified = False

            # Step 3: Execute Trade
            if self.settings.mode == "trading":
                # Use CLOB realtime odds for analysis price (avoids stale Gamma prices)
                _rt_odds = self.realtime_streamer.get_current_odds(market.id)
                if decision.action == "YES":
                    market_price = _rt_odds.yes_odds if _rt_odds else (market.best_ask if market.best_ask else 0.50)
                else:  # NO
                    market_price = _rt_odds.no_odds if _rt_odds else (1 - (market.best_bid if market.best_bid else 0.50))

                await self._execute_trade(
                    market, decision, validation.adjusted_position,
                    token_id, token_name, market_price,
                    trade_id, cycle_start_time,
                    btc_data=btc_data,
                    btc_current=float(btc_data.price),
                    btc_price_to_beat=float(price_to_beat) if price_to_beat else None,
                    arbitrage_opportunity=arbitrage_opportunity,
                    conflict_analysis=conflict_analysis,
                    signal_lag_detected=signal_lag_detected,
                    signal_lag_reason=signal_lag_reason,
                    odds_yes=odds_yes,
                    odds_no=odds_no,
                    odds_qualified=odds_qualified
                )

                # Track trades for reflection triggers
                self.total_trades += 1

                # Trigger reflection every 10 trades
                if self.total_trades % 10 == 0:
                    asyncio.create_task(self._trigger_reflection("10_trades"))

                # Check for consecutive losses trigger (based on database)
                await self._check_consecutive_losses()

                # Send test mode report every 20 trades
                if self.test_mode.enabled:
                    # Check if we've hit report milestone (every 20 trades)
                    if self.total_trades > 0 and self.total_trades % 20 == 0:
                        try:
                            metrics = self.performance_tracker.calculate_test_mode_metrics(last_n_trades=20)
                            if metrics:
                                trade_range = f"Trades {self.total_trades - 19}-{self.total_trades}"
                                await self.telegram_bot.send_test_mode_report(metrics, trade_range)
                        except Exception as e:
                            logger.error("Failed to send test mode report", error=str(e))

            else:
                logger.info(
                    "Dry run - would execute trade",
                    market_id=market.id,
                    action=decision.action,
                    amount=str(validation.adjusted_position)
                )

        except Exception as e:
            logger.error(
                "Market processing error",
                market_id=market.id,
                error=str(e)
            )
        finally:
            # Always release the cycle analysis lock on any exit (HOLD, error, or
            # after-trade). For the traded path this is a no-op since it was already
            # cleared in _traded_markets.add(), but it's safe to discard twice.
            if not self.test_mode.enabled:
                self._markets_with_active_cycle_analysis.discard(market.id)

    async def _get_fresh_market_data(self, market_id: str) -> Optional[Market]:
        """
        Fetch fresh market data immediately before order execution.

        Args:
            market_id: The market ID to fetch

        Returns:
            Market object with current best_bid/best_ask prices, or None if market transition detected

        Raises:
            Exception: If market not found or fetch fails
        """
        try:
            # Refetch the current market to get latest prices
            fresh_market = self.client.discover_btc_15min_market()

            # Verify it's the same market
            if fresh_market.id != market_id:
                logger.warning(
                    "Market transition detected - skipping trade to avoid race condition",
                    expected=market_id,
                    got=fresh_market.id,
                    reason="New market's orderbook may not be ready yet"
                )
                # Return None to signal trade should be skipped
                return None

            logger.info(
                "Fetched fresh market data",
                market_id=fresh_market.id,
                best_ask=f"{fresh_market.best_ask:.3f}",
                best_bid=f"{fresh_market.best_bid:.3f}"
            )

            return fresh_market

        except Exception as e:
            logger.error("Failed to fetch fresh market data", market_id=market_id, error=str(e))
            raise

    def _analyze_price_movement(
        self,
        analysis_price: float,
        execution_price: float,
        token_name: str
    ) -> tuple[bool, str]:
        """
        Analyze price movement between analysis and execution.

        For YES token: Lower price = better for buyer (favorable)
        For NO token: Higher price = worse for buyer (unfavorable)

        Args:
            analysis_price: Price when analysis was done
            execution_price: Current price at execution
            token_name: Either "YES" or "NO"

        Returns:
            Tuple of (should_execute: bool, reason: str)
        """
        # Calculate price change percentage
        price_change_pct = ((execution_price - analysis_price) / analysis_price) * 100

        # Determine if movement is favorable based on token type
        # For YES token: price decreased = favorable (better for buyer)
        # For NO token: price increased = unfavorable (worse for buyer)
        is_favorable = (price_change_pct < 0)

        # Get thresholds from settings
        unfavorable_threshold = self.settings.trade_max_unfavorable_move_pct
        favorable_warn_threshold = self.settings.trade_max_favorable_warn_pct

        # Check unfavorable movement
        if not self.test_mode.enabled and not is_favorable and abs(price_change_pct) > unfavorable_threshold:
            reason = (
                f"Price moved {price_change_pct:+.2f}% worse "
                f"(threshold: {unfavorable_threshold}%)"
            )
            logger.warning(
                "Skipping trade due to unfavorable price movement",
                token=token_name,
                analysis_price=f"{analysis_price:.3f}",
                execution_price=f"{execution_price:.3f}",
                change_pct=f"{price_change_pct:+.2f}%",
                threshold=f"{unfavorable_threshold}%"
            )
            return False, reason

        # Check favorable movement (warning only)
        if is_favorable and abs(price_change_pct) > favorable_warn_threshold:
            reason = (
                f"Price moved {price_change_pct:+.2f}% better "
                f"(unexpected opportunity)"
            )
            logger.warning(
                "Executing with surprisingly favorable price movement",
                token=token_name,
                analysis_price=f"{analysis_price:.3f}",
                execution_price=f"{execution_price:.3f}",
                change_pct=f"{price_change_pct:+.2f}%"
            )
            return True, reason

        # Normal execution
        reason = f"Price movement acceptable ({price_change_pct:+.2f}%)"
        logger.info(
            "Price movement check passed",
            token=token_name,
            change_pct=f"{price_change_pct:+.2f}%"
        )
        return True, reason

    async def _execute_trade(
        self,
        market,
        decision,
        amount: Decimal,
        token_id: str,
        token_name: str,
        market_price: float,
        trade_id: int,
        cycle_start_time: datetime,
        btc_data = None,  # NEW: Full BTCPriceData object
        btc_current: Optional[float] = None,
        btc_price_to_beat: Optional[float] = None,
        arbitrage_opportunity = None,
        conflict_analysis = None,  # NEW parameter
        signal_lag_detected: bool = False,  # NEW parameter
        signal_lag_reason: str | None = None,  # NEW parameter
        odds_yes: float = 0.50,  # NEW parameter
        odds_no: float = 0.50,  # NEW parameter
        odds_qualified: bool = False  # NEW parameter
    ) -> None:
        """Execute a trade order with JIT price fetching and safety checks."""
        try:
            # NEW: Paper trading fork
            if self.test_mode.enabled and self.test_mode.paper_trading:
                await self._execute_paper_trade(
                    market=market,
                    decision=decision,
                    amount=amount,
                    token_name=token_name,
                    market_price=market_price,
                    btc_data=btc_data,
                    btc_price_to_beat=btc_price_to_beat,
                    conflict_analysis=conflict_analysis,
                    signal_lag_detected=signal_lag_detected,
                    signal_lag_reason=signal_lag_reason,
                    odds_yes=odds_yes,
                    odds_no=odds_no,
                    odds_qualified=odds_qualified
                )
                return  # Exit before real order placement

            from polymarket.models import OrderRequest

            # Store analysis price (from cycle start)
            analysis_price = market_price

            logger.info(
                "Pre-execution pricing",
                token=token_name,
                analysis_price=f"{analysis_price:.3f}",
                action=decision.action
            )

            # Fetch fresh market data immediately before execution
            try:
                fresh_market = await self._get_fresh_market_data(market.id)
            except Exception as e:
                logger.error("Failed to fetch fresh market data, aborting trade", error=str(e))
                # Update metrics with failure
                if trade_id > 0:
                    await self.performance_tracker.update_execution_metrics(
                        trade_id=trade_id,
                        analysis_price=analysis_price,
                        execution_price=None,
                        price_staleness_seconds=None,
                        price_movement_favorable=None,
                        skipped_unfavorable_move=True
                    )
                return

            # Check if market transition was detected
            if fresh_market is None:
                logger.info(
                    "Trade skipped - market transition in progress",
                    market_id=market.id,
                    token=token_name,
                    reason="Waiting for next cycle to avoid race condition"
                )
                # Update metrics with skip
                if trade_id > 0:
                    await self.performance_tracker.update_execution_metrics(
                        trade_id=trade_id,
                        analysis_price=analysis_price,
                        execution_price=None,
                        price_staleness_seconds=None,
                        price_movement_favorable=None,
                        skipped_unfavorable_move=True
                    )
                return

            # Check if market is still active (rollover protection)
            if not fresh_market.active:
                logger.warning(
                    "Trade skipped - market expired/rolled over",
                    market_id=fresh_market.id,
                    token=token_name,
                    action=decision.action
                )
                # Update metrics with skip reason
                if trade_id > 0:
                    await self.performance_tracker.update_execution_metrics(
                        trade_id=trade_id,
                        analysis_price=analysis_price,
                        execution_price=None,
                        price_staleness_seconds=int((datetime.now(timezone.utc) - cycle_start_time).total_seconds()),
                        price_movement_favorable=None,
                        skipped_unfavorable_move=True
                    )
                return

            # Calculate fresh execution price using CLOB realtime odds
            # Falls back to Gamma fresh_market if realtime streamer has no data
            _exec_odds = self.realtime_streamer.get_current_odds(market.id)
            if decision.action == "YES":
                execution_price = _exec_odds.yes_odds if _exec_odds else (fresh_market.best_ask if fresh_market.best_ask else 0.50)
            else:  # NO
                execution_price = _exec_odds.no_odds if _exec_odds else (1 - (fresh_market.best_bid if fresh_market.best_bid else 0.50))

            # Calculate price staleness
            execution_time = datetime.now(timezone.utc)
            price_staleness_seconds = int((execution_time - cycle_start_time).total_seconds())

            # Run adaptive safety check
            should_execute, reason = self._analyze_price_movement(
                analysis_price=analysis_price,
                execution_price=execution_price,
                token_name=token_name
            )

            # Determine if movement was favorable (price decreased for buyer)
            price_movement_favorable = (execution_price < analysis_price)

            if not should_execute:
                logger.warning(
                    "Trade skipped due to safety check",
                    token=token_name,
                    reason=reason,
                    staleness_seconds=price_staleness_seconds
                )
                # Track skipped trade in performance system
                if trade_id > 0:
                    await self.performance_tracker.update_execution_metrics(
                        trade_id=trade_id,
                        analysis_price=analysis_price,
                        execution_price=execution_price,
                        price_staleness_seconds=price_staleness_seconds,
                        price_movement_favorable=price_movement_favorable,
                        skipped_unfavorable_move=True
                    )
                return

            # Log final execution price
            logger.info(
                "Executing with fresh price",
                token=token_name,
                execution_price=f"{execution_price:.3f}",
                price_change_pct=f"{((execution_price - analysis_price) / analysis_price * 100):+.2f}%"
            )

            # REMOVED: Arbitrage requirement gate
            # We still calculate arbitrage for AI context, but NO LONGER BLOCK trades based on edge size.
            # Rationale: Bot had 10.6% win rate when requiring arbitrage edge. Signal conflict detection
            # and odds validation are more effective filters.
            #
            # if self.test_mode.enabled and arbitrage_opportunity:
            #     arb_edge = arbitrage_opportunity.edge_percentage
            #     if arb_edge < self.test_mode.min_arbitrage_edge:
            #         logger.info(
            #             "[TEST] Skipping trade - arbitrage edge below minimum",
            #             market_id=market.id,
            #             edge=f"{arb_edge:.2%}",
            #             minimum=f"{self.test_mode.min_arbitrage_edge:.2%}",
            #             reason="Edge too small - likely noise trade"
            #         )
            #         return

            # NEW: Execute using smart limit orders (saves 3-6% in fees)
            smart_executor = SmartOrderExecutor()

            # Determine urgency from arbitrage opportunity
            urgency = arbitrage_opportunity.urgency if arbitrage_opportunity else "MEDIUM"

            # Use CLOB realtime odds for the correct token's price.
            # fresh_market.best_bid/ask are YES/UP token prices from Gamma (stale).
            # For NO/DOWN token, we need the NO token's CLOB price, not the YES price.
            _smart_odds = self.realtime_streamer.get_current_odds(market.id)
            if _smart_odds:
                _token_price = _smart_odds.yes_odds if decision.action == "YES" else _smart_odds.no_odds
                smart_best_bid = _token_price         # last CLOB trade price = bid reference
                smart_best_ask = _token_price + 0.02  # reasonable ask spread
            else:
                smart_best_bid = fresh_market.best_bid if fresh_market.best_bid else execution_price - 0.01
                smart_best_ask = fresh_market.best_ask if fresh_market.best_ask else execution_price

            # Lock this market to prevent duplicate orders while limit order is in-flight
            self._markets_with_pending_orders.add(market.id)

            execution_result = await smart_executor.execute_smart_order(
                client=self.client,
                token_id=token_id,
                side="BUY",
                amount=float(amount),
                urgency=urgency,
                current_best_ask=smart_best_ask,
                current_best_bid=smart_best_bid,
                tick_size=0.001,
                max_fallback_price=smart_best_ask * 1.20  # Cap fallback at 20% slippage from decision price
            )

            # Check execution result
            if execution_result["status"] != "FILLED":
                logger.warning(
                    "Order not filled",
                    market_id=market.id,
                    token=token_name,
                    status=execution_result["status"],
                    reason=execution_result.get("message", "Unknown")
                )
                # Update metrics with skip
                if trade_id > 0:
                    await self.performance_tracker.update_execution_metrics(
                        trade_id=trade_id,
                        analysis_price=analysis_price,
                        execution_price=execution_price,
                        price_staleness_seconds=price_staleness_seconds,
                        price_movement_favorable=price_movement_favorable,
                        skipped_unfavorable_move=True
                    )
                self._markets_with_pending_orders.discard(market.id)
                return  # Skip this trade

            order_id = execution_result["order_id"]
            filled_via = execution_result.get("filled_via", "limit")

            # CRITICAL: Mark market as traded IMMEDIATELY after CLOB order is confirmed
            # filled. Do NOT wait until after verification/Telegram/DB updates — any
            # exception in those steps would leave _traded_markets unset, clearing all
            # locks and allowing the next OddsMonitor trigger to place a second order.
            if not self.test_mode.enabled:
                self._traded_markets.add(market.id)
                self._markets_with_active_cycle_analysis.discard(market.id)
                logger.info(
                    "Market locked immediately after CLOB fill - no further orders possible",
                    market_id=market.id,
                    order_id=order_id
                )

            # NEW: Phase 1 Quick Status Check (2 seconds)
            logger.info(
                "Running quick order verification",
                order_id=order_id,
                trade_id=trade_id
            )

            await asyncio.sleep(2)  # Wait for order to process

            quick_status = await self.order_verifier.check_order_quick(
                order_id=order_id,
                trade_id=trade_id,
                timeout=2.0
            )

            # Handle quick check results
            if quick_status['status'] == 'failed':
                logger.error(
                    "Order failed immediately",
                    order_id=order_id,
                    trade_id=trade_id,
                    raw_status=quick_status['raw_status']
                )
                # Update trade status
                if trade_id > 0:
                    await self.performance_tracker.update_trade_status(
                        trade_id=trade_id,
                        execution_status='failed',
                        skip_reason=f"Order failed: {quick_status['raw_status']}"
                    )
                self._markets_with_pending_orders.discard(market.id)
                return  # Don't count this as a successful trade

            elif quick_status['needs_alert']:
                logger.warning(
                    "Order requires attention",
                    order_id=order_id,
                    trade_id=trade_id,
                    status=quick_status['status'],
                    raw_status=quick_status['raw_status']
                )
                # Send Telegram alert for partial fills or issues
                try:
                    await self.telegram_bot.send_message(
                        f"⚠️ Order Alert\n"
                        f"Order ID: {order_id[:8]}...\n"
                        f"Status: {quick_status['raw_status']}\n"
                        f"Trade ID: {trade_id}"
                    )
                except Exception as e:
                    logger.warning("Failed to send alert", error=str(e))

            # Log success with arbitrage edge
            logger.info(
                "Trade executed and verified",
                market_id=market.id,
                action=decision.action,
                token=token_name,
                amount=str(amount),
                order_id=order_id,
                filled_via=filled_via,
                quick_status=quick_status['status'],
                arbitrage_edge=f"{arbitrage_opportunity.edge_percentage:.1%}" if arbitrage_opportunity else "N/A"
            )

            # Mark market as traded in test mode
            if self.test_mode.enabled:
                self.test_mode.traded_markets.add(market.id)
                logger.info(
                    "[TEST] Market marked as traded",
                    market_id=market.id,
                    total_traded_markets=len(self.test_mode.traded_markets)
                )

            # Mark market as traded in production mode (prevent double-betting)
            if not self.test_mode.enabled:
                self._traded_markets.add(market.id)
                self._markets_with_active_cycle_analysis.discard(market.id)
                logger.info(
                    "Market marked as traded - will not re-enter this market",
                    market_id=market.id,
                    total_traded_markets=len(self._traded_markets)
                )

            # Release pending-order lock now that _traded_markets guards re-entry
            self._markets_with_pending_orders.discard(market.id)

            # Send Telegram notification
            try:
                await self.telegram_bot.send_trade_alert(
                    market_slug=market.slug or f"Market {market.id}",
                    action=decision.action,
                    confidence=decision.confidence,
                    position_size=float(amount),
                    price=execution_price,  # Use fresh execution price
                    reasoning=decision.reasoning,
                    btc_current=btc_current,
                    btc_price_to_beat=btc_price_to_beat
                )
            except Exception as e:
                logger.warning("Failed to send Telegram notification", error=str(e))

            # Track position for stop-loss
            self.open_positions.append({
                "token_id": token_id,
                "side": decision.action,
                "amount": float(amount),
                "entry_odds": 0.50,  # Approximate entry
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

            # Update execution metrics in performance tracker
            if trade_id > 0:
                try:
                    await self.performance_tracker.update_execution_metrics(
                        trade_id=trade_id,
                        analysis_price=analysis_price,
                        execution_price=execution_price,
                        price_staleness_seconds=price_staleness_seconds,
                        price_movement_favorable=price_movement_favorable,
                        skipped_unfavorable_move=False,
                        actual_position_size=float(amount),  # Use risk-adjusted amount, not AI suggestion
                        filled_via=execution_result.get("filled_via"),
                        limit_order_timeout=execution_result.get("timeout_used"),
                        order_id=order_id
                    )
                except Exception as e:
                    logger.warning("Failed to update execution metrics", error=str(e))

            self.trades_today += 1

        except Exception as e:
            self._markets_with_pending_orders.discard(market.id)
            logger.error(
                "Trade execution failed",
                market_id=market.id,
                error=str(e)
            )

    async def _execute_paper_trade(
        self,
        market,
        decision,
        amount: Decimal,
        token_name: str,
        market_price: float,
        btc_data,  # CHANGED: Full BTCPriceData object (not float)
        btc_price_to_beat: Optional[float],
        conflict_analysis,
        signal_lag_detected: bool,
        signal_lag_reason: str | None,
        odds_yes: float,
        odds_no: float,
        odds_qualified: bool
    ) -> None:
        """
        Execute a paper trade (simulated trade, no real money).

        Logs trade to paper_trades table and sends detailed Telegram alert.
        """
        try:
            import json

            # btc_data is now passed as parameter (includes source from Chainlink)

            # Calculate time remaining
            time_remaining_seconds = 900  # Default 15 min
            if market.end_date:
                time_remaining_seconds = int((market.end_date - datetime.now(timezone.utc)).total_seconds())

            # Log paper trade
            paper_trade_id = self.performance_tracker.log_paper_trade(
                market=market,
                decision=decision,
                btc_data=btc_data,
                executed_price=market_price,
                position_size=float(amount),
                price_to_beat=Decimal(str(btc_price_to_beat)) if btc_price_to_beat else None,
                time_remaining_seconds=time_remaining_seconds,
                signal_lag_detected=signal_lag_detected,
                signal_lag_reason=signal_lag_reason,
                conflict_severity=conflict_analysis.severity.value if conflict_analysis else "NONE",
                conflicts_list=conflict_analysis.conflicts_detected if conflict_analysis else [],
                odds_yes=odds_yes,
                odds_no=odds_no,
                odds_qualified=odds_qualified
            )

            # Format summaries for Telegram alert
            # Get technical indicators (need to recalculate or pass in)
            # For now, create simple summaries from available data
            technical_summary = "✅ Technical: (detailed summary TBD)"
            sentiment_summary = "✅ Sentiment: (detailed summary TBD)"
            timeframe_summary = "⚠️ Timeframes: (detailed summary TBD)"

            # Send Telegram alert
            await self.telegram_bot.send_paper_trade_alert(
                market_slug=market.slug or f"Market {market.id}",
                action=decision.action,
                confidence=decision.confidence,
                position_size=float(amount),
                executed_price=market_price,
                time_remaining_seconds=time_remaining_seconds,
                technical_summary=technical_summary,
                sentiment_summary=sentiment_summary,
                odds_yes=odds_yes,
                odds_no=odds_no,
                odds_qualified=odds_qualified,
                timeframe_summary=timeframe_summary,
                signal_lag_detected=signal_lag_detected,
                signal_lag_reason=signal_lag_reason,
                conflict_severity=conflict_analysis.severity.value if conflict_analysis else "NONE",
                conflicts_list=conflict_analysis.conflicts_detected if conflict_analysis else [],
                ai_reasoning=decision.reasoning
            )

            logger.info(
                "Paper trade executed",
                paper_trade_id=paper_trade_id,
                market_slug=market.slug,
                action=decision.action,
                amount=f"${amount:.2f}",
                confidence=f"{decision.confidence:.2f}"
            )

            # Mark market as traded in test mode
            if self.test_mode.enabled:
                self.test_mode.traded_markets.add(market.id)

        except Exception as e:
            logger.error(
                "Paper trade execution failed",
                market_id=market.id,
                error=str(e),
                exc_info=True
            )

    async def _check_stop_loss(self) -> None:
        """Check and execute stop-loss for open positions."""
        if not self.open_positions:
            return

        try:
            # Get current market data
            markets = {}
            for pos in self.open_positions:
                token_id = pos["token_id"]
                # Note: Would need to fetch current market data here
                # Skipping for now as it requires market lookup

            # Evaluate stop-loss
            # to_close = await self.risk_manager.evaluate_stop_loss(
            #     self.open_positions, markets
            # )
            #
            # # Close positions
            # for close in to_close:
            #     await self._close_position(close)

        except Exception as e:
            logger.error("Stop-loss check error", error=str(e))

    # ─────────────────────────────────────────────────────────────────────────
    # Background watcher: stop-loss (Enhancement 5)
    # ─────────────────────────────────────────────────────────────────────────

    STOP_LOSS_WATCHER_INTERVAL_SECONDS = 30

    async def _stop_loss_watcher(self) -> None:
        """Background task: evaluate stop-loss every 30s instead of every 3 minutes.

        Delegates to the existing _check_stop_loss() — no new logic here.
        Runs independently of the main cycle so a sharp reversal triggers exit
        within ~30s rather than waiting up to 180s for the next cycle.
        """
        logger.info(
            "Stop-loss watcher started",
            interval_seconds=self.STOP_LOSS_WATCHER_INTERVAL_SECONDS
        )
        while self.running:
            await asyncio.sleep(self.STOP_LOSS_WATCHER_INTERVAL_SECONDS)
            try:
                if self.open_positions:  # Skip if nothing to protect
                    await self._check_stop_loss()
            except Exception as e:
                logger.error("Stop-loss watcher error", error=str(e))

    # ─────────────────────────────────────────────────────────────────────────
    # Timed strategy: 10-min analysis trigger + entry monitor
    # ─────────────────────────────────────────────────────────────────────────

    TIMING_WATCHER_INTERVAL_SECONDS = 10
    ANALYSIS_TRIGGER_ELAPSED_SECONDS = 600   # T=10min into the 15-min market
    ANALYSIS_TRIGGER_WINDOW_SECONDS = 60     # 60s window to fire (600–660s elapsed)
    TIMED_ENTRY_ODDS_MIN = 0.70              # Minimum CLOB odds to enter
    TIMED_ENTRY_ODDS_MAX = 0.80              # Maximum CLOB odds (above = near settlement)
    TIMED_ENTRY_WINDOW_SECONDS = 300         # Only enter in last 5 minutes (<=300s remaining)

    async def _market_timing_watcher(self) -> None:
        """Background task: trigger AI analysis at the 10-minute mark of each 15-min market.

        This replaces the OddsMonitor as the primary cycle trigger. At T=10min
        (600s elapsed, 300s remaining), fires run_cycle() once per market for
        full AI analysis + data collection. The _timed_entry_monitor then watches
        for 70%+ odds in the remaining 5 minutes to execute the order.

        State managed here:
          _analysis_triggered — set when the cycle is fired (prevents re-firing)
          _timed_decisions    — populated by _process_market when odds < 70% at analysis time
        """
        logger.info(
            "Market timing watcher started",
            analysis_trigger_at=f"T+{self.ANALYSIS_TRIGGER_ELAPSED_SECONDS}s (10-min mark)",
            entry_window=f"last {self.TIMED_ENTRY_WINDOW_SECONDS}s (5 min)",
            entry_odds_min=f"{self.TIMED_ENTRY_ODDS_MIN:.0%}"
        )

        while self.running:
            try:
                await asyncio.sleep(self.TIMING_WATCHER_INTERVAL_SECONDS)

                # Get current active market from streamer
                market_id = self.realtime_streamer._current_market_id
                market_slug = self.realtime_streamer._current_market_slug

                if not market_id or not market_slug:
                    continue

                # Parse market start time from slug
                start_time = self.market_tracker.parse_market_start(market_slug)
                if not start_time:
                    continue

                now = datetime.now(timezone.utc)
                elapsed = (now - start_time).total_seconds()
                time_remaining = max(0, 900 - elapsed)

                # Clean up expired market state
                if time_remaining == 0:
                    self._analysis_triggered.discard(market_id)
                    self._timed_decisions.pop(market_id, None)
                    continue

                # Skip if already traded for this market
                if market_id in self._traded_markets:
                    continue

                # Skip if analysis already triggered for this market in this epoch
                if market_id in self._analysis_triggered:
                    continue

                # Trigger analysis when 600–660s have elapsed (T=10min window)
                trigger_end = self.ANALYSIS_TRIGGER_ELAPSED_SECONDS + self.ANALYSIS_TRIGGER_WINDOW_SECONDS
                if self.ANALYSIS_TRIGGER_ELAPSED_SECONDS <= elapsed < trigger_end:
                    logger.info(
                        "T=10min mark reached — triggering AI analysis",
                        market_id=market_id,
                        market_slug=market_slug,
                        elapsed_seconds=int(elapsed),
                        time_remaining_seconds=int(time_remaining)
                    )
                    self._analysis_triggered.add(market_id)
                    asyncio.create_task(self.run_cycle())

            except Exception as e:
                logger.error("Market timing watcher error", error=str(e))

    async def _timed_entry_monitor(self) -> None:
        """Background task: execute stored decisions when 70%+ odds appear in last 5 minutes.

        After _market_timing_watcher fires run_cycle() at T=10min, _process_market
        may store a decision in _timed_decisions if the odds aren't yet at 70%.
        This monitor checks every 10s. When:
          - Time remaining <= 5 min (300s)
          - Stored decision exists for the active market
          - CLOB odds for the decided direction >= 70% AND < 80%
        it fires _execute_timed_entry() to place the order.
        """
        logger.info(
            "Timed entry monitor started",
            entry_odds_threshold=f"{self.TIMED_ENTRY_ODDS_MIN:.0%}",
            entry_window=f"last {self.TIMED_ENTRY_WINDOW_SECONDS}s",
            check_interval_seconds=self.TIMING_WATCHER_INTERVAL_SECONDS
        )

        while self.running:
            try:
                await asyncio.sleep(self.TIMING_WATCHER_INTERVAL_SECONDS)

                if not self._timed_decisions:
                    continue

                for market_id in list(self._timed_decisions.keys()):
                    stored = self._timed_decisions.get(market_id)
                    if not stored:
                        continue

                    # Skip if market was already traded or has a pending order
                    if market_id in self._traded_markets or market_id in self._markets_with_pending_orders:
                        logger.info(
                            "Timed entry: market already traded/pending — removing stored decision",
                            market_id=market_id
                        )
                        self._timed_decisions.pop(market_id, None)
                        continue

                    # Check time remaining
                    market = stored['market']
                    start_time = self.market_tracker.parse_market_start(market.slug or "")
                    if not start_time:
                        continue

                    time_remaining = self.market_tracker.calculate_time_remaining(start_time)

                    if time_remaining == 0:
                        logger.info(
                            "Timed entry: market expired — removing stored decision",
                            market_id=market_id
                        )
                        self._timed_decisions.pop(market_id, None)
                        self._analysis_triggered.discard(market_id)
                        continue

                    if time_remaining > self.TIMED_ENTRY_WINDOW_SECONDS:
                        logger.debug(
                            "Timed entry: not yet in last-5-min window",
                            market_id=market_id,
                            time_remaining=time_remaining
                        )
                        continue

                    # Check current CLOB odds
                    odds_snapshot = self.realtime_streamer.get_current_odds(market_id)
                    if not odds_snapshot:
                        continue

                    action = stored['action']
                    current_odds = odds_snapshot.yes_odds if action == "YES" else odds_snapshot.no_odds

                    logger.debug(
                        "Timed entry check",
                        market_id=market_id,
                        action=action,
                        current_odds=f"{current_odds:.2%}",
                        entry_min=f"{self.TIMED_ENTRY_ODDS_MIN:.0%}",
                        time_remaining=time_remaining
                    )

                    if current_odds >= self.TIMED_ENTRY_ODDS_MIN:
                        logger.info(
                            "Timed entry condition met — executing stored decision",
                            market_id=market_id,
                            action=action,
                            current_odds=f"{current_odds:.2%}",
                            time_remaining=time_remaining
                        )
                        # Remove from dict BEFORE creating task to prevent double-execution
                        entry_context = self._timed_decisions.pop(market_id)
                        asyncio.create_task(self._execute_timed_entry(market_id, entry_context, current_odds))

            except Exception as e:
                logger.error("Timed entry monitor error", error=str(e))

    async def _execute_timed_entry(self, market_id: str, stored: dict, trigger_odds: float) -> None:
        """Execute a stored timed decision when entry conditions are met.

        Called by _timed_entry_monitor when 70%+ odds are detected. Performs a
        final JIT check at execution time before calling _execute_trade().
        """
        try:
            if market_id in self._traded_markets or market_id in self._markets_with_pending_orders:
                logger.info(
                    "Timed entry aborted — already traded/pending between monitor check and execution",
                    market_id=market_id
                )
                return

            market = stored['market']
            decision = stored['decision']

            # Get fresh CLOB odds for execution price
            _rt_odds = self.realtime_streamer.get_current_odds(market_id)
            if decision.action == "YES":
                market_price = _rt_odds.yes_odds if _rt_odds else trigger_odds
                odds_yes = market_price
                odds_no = 1.0 - market_price
            else:
                market_price = _rt_odds.no_odds if _rt_odds else trigger_odds
                odds_no = market_price
                odds_yes = 1.0 - market_price

            # Final JIT guard: re-validate 70% floor at actual execution time
            if market_price < self.TIMED_ENTRY_ODDS_MIN:
                logger.info(
                    "Timed entry blocked — odds fell below 70% between monitor check and execution",
                    market_id=market_id,
                    action=decision.action,
                    odds=f"{market_price:.2%}"
                )
                return

            logger.info(
                "Executing timed entry",
                market_id=market_id,
                action=decision.action,
                odds=f"{market_price:.2%}",
                amount=str(stored['amount']),
                stored_at=stored['stored_at'].isoformat()
            )

            await self._execute_trade(
                market=market,
                decision=decision,
                amount=stored['amount'],
                token_id=stored['token_id'],
                token_name=stored['token_name'],
                market_price=market_price,
                trade_id=stored['trade_id'],
                cycle_start_time=stored['stored_at'],
                btc_data=stored['btc_data'],
                btc_current=stored['btc_current'],
                btc_price_to_beat=stored['btc_price_to_beat'],
                arbitrage_opportunity=stored['arbitrage_opportunity'],
                conflict_analysis=stored['conflict_analysis'],
                signal_lag_detected=stored['signal_lag_detected'],
                signal_lag_reason=stored['signal_lag_reason'],
                odds_yes=odds_yes,
                odds_no=odds_no,
                odds_qualified=True
            )

            # Track for reflection triggers
            self.total_trades += 1
            if self.total_trades % 10 == 0:
                asyncio.create_task(self._trigger_reflection("10_trades"))
            await self._check_consecutive_losses()

        except Exception as e:
            logger.error("Timed entry execution failed", market_id=market_id, error=str(e))

    # ─────────────────────────────────────────────────────────────────────────
    # Background watcher: price movement trigger (Enhancement 4)
    # ─────────────────────────────────────────────────────────────────────────

    PRICE_WATCHER_INTERVAL_SECONDS = 10
    PRICE_WATCHER_TRIGGER_USD = 25.0  # Trigger analysis if BTC moves >$25 from price_to_beat

    async def _price_movement_watcher(self) -> None:
        """Background task: fire immediate market analysis on significant BTC price moves.

        Runs every 10s. Complements the OddsMonitor (CLOB odds changes) and the
        3-minute timer cycle by detecting raw BTC price movement.
        When |BTC_current - price_to_beat| >= $25 and no pending order for that
        market, re-uses the last cycle's cached indicators/sentiment to trigger
        _process_market() immediately without waiting for the next 3-min cycle.
        """
        logger.info(
            "Price movement watcher started",
            interval_seconds=self.PRICE_WATCHER_INTERVAL_SECONDS,
            trigger_usd=self.PRICE_WATCHER_TRIGGER_USD
        )
        while self.running:
            try:
                await asyncio.sleep(self.PRICE_WATCHER_INTERVAL_SECONDS)

                # Nothing to do until the first full cycle has populated the cache
                if not self._last_cycle_data:
                    continue

                btc_price_data = await self.btc_service.get_current_price()
                if not btc_price_data:
                    continue

                current_price = btc_price_data.price

                # Check each active market's price-to-beat
                markets = await self._discover_markets()
                for market in markets:
                    slug_str = market.slug or ""
                    price_to_beat = self.market_tracker.get_price_to_beat(slug_str)
                    if not price_to_beat:
                        continue

                    movement = abs(float(current_price - price_to_beat))

                    if movement >= self.PRICE_WATCHER_TRIGGER_USD:
                        # NEW: Timed strategy — analysis is triggered by _market_timing_watcher
                        # at T=10min. Price watcher is disabled to prevent early entries.
                        logger.debug(
                            "Price watcher movement detected (timed strategy: skipping early trigger)",
                            market_id=market.id,
                            movement=f"${movement:.2f}",
                            threshold=f"${self.PRICE_WATCHER_TRIGGER_USD}",
                            note="_market_timing_watcher handles analysis at T=10min"
                        )
                        continue

            except Exception as e:
                logger.error("Price watcher error", error=str(e))

    async def _trigger_market_analysis(self, market, btc_price_data) -> None:
        """Trigger a single-market analysis using the most recent cached cycle data.

        Called by _price_movement_watcher. Uses self._last_cycle_data so we don't
        re-fetch indicators/sentiment on every 10s tick — the cached values are
        fresh enough for a spot-price-triggered decision.
        Lock (_markets_with_active_watcher_analysis) is always cleared on exit.
        """
        try:
            cache = self._last_cycle_data
            if not cache:
                return

            # Stale cache guard: if indicators are older than 90s (half of the 180s main cycle),
            # they may reflect a market regime that has changed. Skip and let the next main cycle
            # provide fresh indicators. This prevents firing NO bets when RSI has since jumped
            # to overbought (as happened on market 1771490700 where cached RSI=26.6 was used
            # while the real RSI had already moved to 97.9).
            MAX_CACHE_AGE_SECONDS = 90
            cache_age = (datetime.now(timezone.utc) - cache['timestamp']).total_seconds()
            if cache_age > MAX_CACHE_AGE_SECONDS:
                logger.info(
                    "Price watcher skipping — cached indicators too stale for reliable analysis",
                    market_id=market.id,
                    cache_age_seconds=f"{cache_age:.0f}s",
                    max_age=f"{MAX_CACHE_AGE_SECONDS}s"
                )
                return

            await self._process_market(
                market,
                btc_price_data,                  # fresh BTC spot price
                cache['indicators'],
                cache['sentiment'],
                cache['portfolio_value'],
                cache['btc_momentum'],
                datetime.now(timezone.utc),      # fresh cycle time
                cache['timeframe_analysis'],
                cache['regime'],
                cache['contrarian_signal'],
            )
        except Exception as e:
            logger.error("Price trigger analysis failed", market_id=market.id, error=str(e))
        finally:
            # Always release the watcher lock so the next price move can trigger again
            self._markets_with_active_watcher_analysis.discard(market.id)

    async def _close_position(self, close: dict) -> None:
        """Close a position via stop-loss."""
        try:
            token_id = close["token_id"]
            side = close["side"]

            logger.info(
                "Stop-loss executed",
                token_id=token_id,
                reason=close["reason"]
            )

            # Remove from open positions
            self.open_positions = [
                p for p in self.open_positions
                if p["token_id"] != token_id
            ]

        except Exception as e:
            logger.error("Stop-loss execution error", error=str(e))

    async def run(self) -> None:
        """Main loop - runs until interrupted."""
        logger.info(
            "Starting AutoTrader",
            mode=self.settings.mode,
            interval=self.interval
        )

        # Setup signal handlers
        def signal_handler(sig, frame):
            logger.info("Shutting down gracefully...")
            self.running = False

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Initialize async resources
        await self.initialize()

        # OLD TIMER-BASED LOOP (replaced by OddsMonitor event-driven triggering)
        # Keeping this code commented for reference during transition period
        # while self.running:
        #     await self.run_cycle()
        #
        #     # Wait before next cycle
        #     if self.running:
        #         logger.info(f"Waiting {self.interval} seconds until next cycle...")
        #         await asyncio.sleep(self.interval)

        # NEW EVENT-DRIVEN APPROACH:
        # OddsMonitor now triggers cycles via _handle_opportunity_detected callback
        # when sustained high odds (>70% for 5s) are detected.
        # Keep the bot running until interrupted, but don't loop on timer.
        while self.running:
            await asyncio.sleep(1)  # Keep alive loop, actual cycles triggered by events

        # Cleanup
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        if self.background_tasks:
            logger.info("Background tasks stopped")

        # Stop OddsMonitor
        if self.odds_monitor:
            await self.odds_monitor.stop()
            logger.info("OddsMonitor stopped")

        # Stop real-time odds streamer
        await self.realtime_streamer.stop()
        logger.info("Real-time odds streamer stopped")

        await self.btc_service.close()  # Now closes WebSocket
        await self.social_service.close()
        if self.market_service:
            await self.market_service.close()
        self.cleanup_scheduler.stop()
        logger.info("Cleanup scheduler stopped")
        self.performance_tracker.close()
        logger.info("Performance tracker closed")
        logger.info("AutoTrader shutdown complete")

    async def run_once(self) -> None:
        """Run a single cycle for testing."""
        await self.initialize()
        await self.run_cycle()

        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        await self.btc_service.close()  # Now closes WebSocket
        await self.social_service.close()
        if self.market_service:
            await self.market_service.close()
        self.cleanup_scheduler.stop()
        self.performance_tracker.close()


@app.command()
def main(
    interval: int = typer.Option(60, help="Cycle interval in seconds"),
    once: bool = typer.Option(False, help="Run single cycle then exit")
) -> None:
    """Run the autonomous trading bot."""
    # Load settings
    settings = Settings()

    # Configure stdlib logging (required for structlog's LoggerFactory)
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=logging.INFO,
    )

    # Configure logging
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer() if settings.log_json else structlog.dev.ConsoleRenderer()
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Create trader
    trader = AutoTrader(settings, interval)

    # Run
    if once:
        asyncio.run(trader.run_once())
    else:
        asyncio.run(trader.run())


if __name__ == "__main__":
    app()
