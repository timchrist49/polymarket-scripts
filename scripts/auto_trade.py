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
        cycle_start_time = datetime.now()
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
            btc_data, social_sentiment, market_signals, funding_signal, dominance_signal, volume_data = await asyncio.gather(
                self.btc_service.get_current_price(),
                self.social_service.get_social_score(),
                self.market_service.get_market_score(),
                self.btc_service.get_funding_rates(),
                self.btc_service.get_btc_dominance(),
                self.btc_service.get_volume_data(),  # NEW: Volume confirmation for breakouts
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
                        tf_15m=timeframe_analysis.tf_15m.direction,
                        tf_1h=timeframe_analysis.tf_1h.direction,
                        tf_4h=timeframe_analysis.tf_4h.direction,
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

            # Step 3.5: Market Regime Detection
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

            # Step 4: Aggregate Signals - NEW (includes funding + dominance)
            aggregated_sentiment = self.aggregator.aggregate(
                social_sentiment,
                market_signals,
                funding=funding_signal,
                dominance=dominance_signal
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
                    using_balance=f"${portfolio_value:.2f}"
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
                    volume_data,  # NEW: volume data for AI context
                    timeframe_analysis,  # NEW: timeframe analysis for AI context
                    regime  # NEW: market regime for adaptive strategy
                )

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

                # Filter: Require >= 5 minutes remaining
                if time_remaining < 300:
                    filtered_count += 1
                    logger.debug(
                        "Filtered end-phase market",
                        market_id=market.id,
                        time_remaining_sec=int(time_remaining)
                    )
                    continue

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
        volume_data,  # NEW: volume data for breakout confirmation
        timeframe_analysis,  # NEW: multi-timeframe trend analysis
        regime  # NEW: market regime detection
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

            # NEW: Early odds filtering (background poll check)
            cached_odds = await self.odds_poller.get_odds(market.id)
            if cached_odds:
                if not (cached_odds.yes_qualifies or cached_odds.no_qualifies):
                    logger.info(
                        "Skipping market - neither side > 75% odds",
                        market_id=market.id,
                        yes_odds=f"{cached_odds.yes_odds:.2%}",
                        no_odds=f"{cached_odds.no_odds:.2%}"
                    )
                    return  # Skip this market
            else:
                logger.debug("No cached odds available (polling may not have run yet)")

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
                    # Fallback to current price if historical fetch fails
                    price_to_beat = btc_data.price
                    self.market_tracker.set_price_to_beat(market_slug, price_to_beat)
                    logger.warning(
                        "Price-to-beat fallback to current price",
                        market_id=market.id,
                        reason="Historical price fetch failed",
                        price=f"${price_to_beat:,.2f}"
                    )

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

                # NEW: Signal lag detection
                signal_lag_detected = False
                signal_lag_reason = None

                btc_direction = "UP" if btc_data.price > price_to_beat else "DOWN"
                sentiment_direction = "BULLISH" if aggregated_sentiment.final_score > 0 else "BEARISH"

                signal_lag_detected, signal_lag_reason = detect_signal_lag(
                    btc_direction,
                    sentiment_direction,
                    aggregated_sentiment.final_confidence
                )

                if signal_lag_detected:
                    if not self.test_mode.enabled:
                        logger.warning(
                            "Skipping trade due to signal lag",
                            market_id=market.id,
                            reason=signal_lag_reason
                        )
                        return  # HOLD - don't trade contradictions
                    else:
                        logger.info(
                            "[TEST] Signal lag detected - data sent to AI anyway",
                            market_id=market.id,
                            reason=signal_lag_reason
                        )

                # Check minimum movement threshold to avoid entering too early
                MIN_MOVEMENT_THRESHOLD = 100  # $100 minimum BTC movement
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

                # Volume confirmation for large moves (breakout detection)
                if abs_diff > 200 and volume_data:  # $200+ move = potential breakout
                    if not volume_data.is_high_volume:
                        if not self.test_mode.enabled:
                            logger.info(
                                "Skipping large move without volume confirmation",
                                market_id=market.id,
                                movement=f"${diff:+,.2f}",
                                volume_ratio=f"{volume_data.volume_ratio:.2f}x",
                                reason="Breakouts require volume > 1.5x average"
                            )
                            return  # Skip low-volume breakouts
                        else:
                            logger.info(
                                "[TEST] Bypassing volume confirmation - data sent to AI",
                                market_id=market.id,
                                movement=f"${diff:+,.2f}",
                                volume_ratio=f"{volume_data.volume_ratio:.2f}x",
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
                        tf_1h_trend=timeframe_analysis.tf_1h.direction,
                        tf_4h_trend=timeframe_analysis.tf_4h.direction,
                        reason="Don't trade against larger timeframe trend"
                    )
                    return
                else:
                    logger.info(
                        "[TEST] Bypassing timeframe check - data sent to AI",
                        market_id=market.id,
                        tf_1h_trend=timeframe_analysis.tf_1h.direction,
                        tf_4h_trend=timeframe_analysis.tf_4h.direction,
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
                    if orderbook_analysis.spread_bps > 500:  # 5% spread
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

            # Step 1: AI Decision - pass all market context including orderbook, volume, timeframe, regime, arbitrage, and market signals
            decision = await self.ai_service.make_decision(
                btc_price=btc_data,
                technical_indicators=indicators,
                aggregated_sentiment=aggregated_sentiment,
                market_data=market_dict,
                portfolio_value=portfolio_value,
                orderbook_data=orderbook_analysis,  # orderbook depth analysis
                volume_data=volume_data,  # NEW: volume confirmation
                timeframe_analysis=timeframe_analysis,  # NEW: multi-timeframe analysis
                regime=regime,  # NEW: market regime detection
                arbitrage_opportunity=arbitrage_opportunity,  # NEW: arbitrage opportunity
                market_signals=market_signals,  # NEW: CoinGecko Pro market signals
                force_trade=self.test_mode.enabled  # NEW: TEST MODE - force YES/NO decision
            )

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
                MIN_YES_MOVEMENT = 200  # $200 minimum for YES trades (higher threshold)

                if diff < MIN_YES_MOVEMENT:
                    logger.info(
                        "Skipping YES trade - insufficient upward momentum",
                        market_id=market.id,
                        movement=f"${diff:+,.2f}",
                        threshold=f"${MIN_YES_MOVEMENT}",
                        reason="Avoid buying exhausted momentum (mean reversion risk)"
                    )
                    return  # Skip WITHOUT creating DB record

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
                    timeframe_analysis=timeframe_analysis
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

            # Skip if HOLD decision
            if decision.action == "HOLD" or token_id is None:
                logger.info(
                    "Decision: HOLD",
                    market_id=market.id,
                    reason=decision.reasoning
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
            fresh_market = self.client.get_market_by_slug(market.slug)
            if fresh_market:
                yes_odds_fresh = fresh_market.best_bid if fresh_market.best_bid else 0.50
                no_odds_fresh = 1.0 - yes_odds_fresh

                # Check if AI's chosen side still qualifies
                if decision.action == "YES" and yes_odds_fresh <= 0.75:
                    logger.info(
                        "Skipping trade - YES odds below threshold at execution time",
                        market_id=market.id,
                        odds=f"{yes_odds_fresh:.2%}",
                        threshold="75%"
                    )
                    return
                elif decision.action == "NO" and no_odds_fresh <= 0.75:
                    logger.info(
                        "Skipping trade - NO odds below threshold at execution time",
                        market_id=market.id,
                        odds=f"{no_odds_fresh:.2%}",
                        threshold="75%"
                    )
                    return

                # Store odds for paper trade logging
                odds_yes = yes_odds_fresh
                odds_no = no_odds_fresh
                odds_qualified = (
                    (decision.action == "YES" and yes_odds_fresh > 0.75) or
                    (decision.action == "NO" and no_odds_fresh > 0.75)
                )
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
                # Determine market price based on which token we're buying
                # UP/YES token: use ask price from market
                # DOWN/NO token: use complement (1 - UP bid)
                if decision.action == "YES":
                    market_price = market.best_ask if market.best_ask else 0.50  # UP ask
                else:  # NO
                    market_price = 1 - (market.best_bid if market.best_bid else 0.50)  # DOWN = 1 - UP

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
                        price_staleness_seconds=int((datetime.now() - cycle_start_time).total_seconds()),
                        price_movement_favorable=None,
                        skipped_unfavorable_move=True
                    )
                return

            # Calculate fresh execution price from fresh market
            if decision.action == "YES":
                execution_price = fresh_market.best_ask if fresh_market.best_ask else 0.50
            else:  # NO
                execution_price = 1 - (fresh_market.best_bid if fresh_market.best_bid else 0.50)

            # Calculate price staleness
            execution_time = datetime.now()
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

            execution_result = await smart_executor.execute_smart_order(
                client=self.client,
                token_id=token_id,
                side="BUY",
                amount=float(amount),
                urgency=urgency,
                current_best_ask=fresh_market.best_ask if fresh_market.best_ask else execution_price,
                current_best_bid=fresh_market.best_bid if fresh_market.best_bid else execution_price - 0.01,
                tick_size=0.001
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
                return  # Skip this trade

            order_id = execution_result["order_id"]
            filled_via = execution_result.get("filled_via", "limit")

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
                "timestamp": datetime.now().isoformat()
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

        while self.running:
            await self.run_cycle()

            # Wait before next cycle
            if self.running:
                logger.info(f"Waiting {self.interval} seconds until next cycle...")
                await asyncio.sleep(self.interval)

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
