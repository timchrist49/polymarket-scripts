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
from datetime import datetime
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
from polymarket.performance.tracker import PerformanceTracker
from polymarket.performance.cleanup import CleanupScheduler
from polymarket.performance.reflection import ReflectionEngine
from polymarket.performance.adjuster import ParameterAdjuster, AdjustmentTier
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


class AutoTrader:
    """Main autonomous trading bot orchestrator."""

    def __init__(self, settings: Settings, interval: int = 180):
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

        # Trade settlement
        self.trade_settler = TradeSettler(
            db=self.performance_tracker.db,
            btc_fetcher=self.btc_service
        )
        # Give settler access to tracker for updates
        self.trade_settler._tracker = self.performance_tracker

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

    async def initialize(self) -> None:
        """Initialize async resources before trading cycles."""
        # Start BTC price WebSocket stream
        await self.btc_service.start()
        logger.info("Initialized Polymarket WebSocket for BTC prices")
        logger.info("Performance tracking enabled")

        # Start cleanup scheduler in background
        asyncio.create_task(self.cleanup_scheduler.start())
        logger.info("Cleanup scheduler started (runs weekly)")
        logger.info("Self-reflection system enabled (triggers: 10 trades, 3 consecutive losses)")

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

        # Check for emergency pause file
        pause_file = Path(__file__).parent.parent / ".emergency_pause"
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
            # Step 1: Market Discovery - Find BTC 15-min markets
            markets = await self._discover_markets()
            if not markets:
                logger.info("No BTC markets found, skipping cycle")
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

            # Step 2: Data Collection (parallel) - NEW: fetch social + market + funding + dominance
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
                    cycle_start_time  # NEW: pass cycle start time for JIT metrics
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

    async def _process_market(
        self,
        market: Market,
        btc_data,
        indicators,
        aggregated_sentiment,  # CHANGED from sentiment
        portfolio_value: Decimal,
        btc_momentum: dict | None,  # NEW: momentum calculated once per loop
        cycle_start_time: datetime  # NEW: for JIT execution metrics
    ) -> None:
        """Process a single market for trading decision."""
        try:
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

            # Step 1: AI Decision - CHANGED: pass aggregated_sentiment
            decision = await self.ai_service.make_decision(
                btc_price=btc_data,
                technical_indicators=indicators,
                aggregated_sentiment=aggregated_sentiment,  # CHANGED
                market_data=market_dict,
                portfolio_value=portfolio_value
            )

            # Log decision to performance tracker
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
                    is_end_phase=is_end_of_market
                )
            except Exception as e:
                logger.error("Performance logging failed", error=str(e))
                # Continue trading - don't block on logging failures

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
                open_positions=self.open_positions
            )

            if not validation.approved:
                logger.info(
                    "Decision rejected by risk manager",
                    market_id=market.id,
                    reason=validation.reason
                )
                return

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
                    btc_current=float(btc_data.price),
                    btc_price_to_beat=float(price_to_beat) if price_to_beat else None
                )

                # Track trades for reflection triggers
                self.total_trades += 1

                # Trigger reflection every 10 trades
                if self.total_trades % 10 == 0:
                    asyncio.create_task(self._trigger_reflection("10_trades"))

                # Check for consecutive losses trigger (based on database)
                await self._check_consecutive_losses()

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
        if not is_favorable and abs(price_change_pct) > unfavorable_threshold:
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
        btc_current: Optional[float] = None,
        btc_price_to_beat: Optional[float] = None
    ) -> None:
        """Execute a trade order with JIT price fetching and safety checks."""
        try:
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

            # Create order request with fresh execution price
            order_request = OrderRequest(
                token_id=token_id,
                side="BUY",  # Always BUY for our decision (YES or NO token)
                price=execution_price,  # Use fresh execution price
                size=float(amount),
                order_type="market"
            )

            result = self.client.create_order(order_request, dry_run=False)

            # Only log success if order was actually placed
            if result and result.order_id:
                logger.info(
                    "Trade executed",
                    market_id=market.id,
                    action=decision.action,
                    token=token_name,
                    amount=str(amount),
                    order_id=result.order_id
                )
            else:
                logger.warning(
                    "Order placement failed",
                    market_id=market.id,
                    action=decision.action,
                    token=token_name,
                    amount=str(amount),
                    result=str(result) if result else "None"
                )
                return  # Abort trade execution

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
                        actual_position_size=float(amount)  # Use risk-adjusted amount, not AI suggestion
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
    interval: int = typer.Option(180, help="Cycle interval in seconds"),
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
