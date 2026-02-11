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

app = typer.Typer(help="Autonomous Polymarket Trading Bot")
logger = structlog.get_logger()


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

        # State tracking
        self.cycle_count = 0
        self.trades_today = 0
        self.pnl_today = Decimal("0")
        self.running = True

        # Track open positions for stop-loss
        self.open_positions: list[dict] = []

    async def initialize(self) -> None:
        """Initialize async resources before trading cycles."""
        # Start BTC price WebSocket stream
        await self.btc_service.start()
        logger.info("Initialized Polymarket WebSocket for BTC prices")

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
        logger.info(
            "Starting trading cycle",
            cycle=self.cycle_count,
            timestamp=datetime.now().isoformat()
        )

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

            # Step 2: Data Collection (parallel) - NEW: fetch social + market
            btc_data, social_sentiment, market_signals = await asyncio.gather(
                self.btc_service.get_current_price(),
                self.social_service.get_social_score(),
                self.market_service.get_market_score(),
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
                market_conf=f"{market_signals.confidence:.2f}"
            )

            # Step 3: Technical Analysis (optional - graceful if unavailable)
            try:
                price_history = await self.btc_service.get_price_history(minutes=60)
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

            # Step 4: Aggregate Signals - NEW
            aggregated_sentiment = self.aggregator.aggregate(social_sentiment, market_signals)

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
                    btc_momentum  # NEW: pass momentum calculated once per loop
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
        btc_momentum: dict | None  # NEW: momentum calculated once per loop
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
            market_dict = {
                "token_id": token_ids[0],  # Temporary, for logging only
                "question": market.question,
                "yes_price": market.best_bid or 0.50,
                "no_price": market.best_ask or 0.50,
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
                "Market prices",
                market_id=market.id,
                yes_price=f"{market_dict['yes_price']:.3f}",
                no_price=f"{market_dict['no_price']:.3f}",
                outcomes=market_dict["outcomes"]
            )

            # Step 1: AI Decision - CHANGED: pass aggregated_sentiment
            decision = await self.ai_service.make_decision(
                btc_price=btc_data,
                technical_indicators=indicators,
                aggregated_sentiment=aggregated_sentiment,  # CHANGED
                market_data=market_dict,
                portfolio_value=portfolio_value
            )

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
                # Determine market price based on decision
                # For BUY orders, use ask price (what sellers want)
                market_price = market.best_ask if market.best_ask else 0.50
                await self._execute_trade(market, decision, validation.adjusted_position, token_id, token_name, market_price)
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

    async def _execute_trade(self, market, decision, amount: Decimal, token_id: str, token_name: str, market_price: float) -> None:
        """Execute a trade order."""
        try:
            from polymarket.models import OrderRequest

            logger.info(
                "Order pricing",
                token=token_name,
                market_price=f"{market_price:.3f}",
                action=decision.action
            )

            # Create order request
            order_request = OrderRequest(
                token_id=token_id,
                side="BUY",  # Always BUY for our decision (YES or NO token)
                price=market_price,  # Use actual market price
                size=float(amount),
                order_type="market"
            )

            result = self.client.create_order(order_request, dry_run=False)

            logger.info(
                "Trade executed",
                market_id=market.id,
                action=decision.action,
                token=token_name,
                amount=str(amount),
                order_id=result.order_id
            )

            # Track position for stop-loss
            self.open_positions.append({
                "token_id": token_id,
                "side": decision.action,
                "amount": float(amount),
                "entry_odds": 0.50,  # Approximate entry
                "timestamp": datetime.now().isoformat()
            })

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
        await self.btc_service.close()  # Now closes WebSocket
        await self.social_service.close()
        if self.market_service:
            await self.market_service.close()
        logger.info("AutoTrader shutdown complete")

    async def run_once(self) -> None:
        """Run a single cycle for testing."""
        await self.initialize()
        await self.run_cycle()
        await self.btc_service.close()  # Now closes WebSocket
        await self.social_service.close()
        if self.market_service:
            await self.market_service.close()


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
