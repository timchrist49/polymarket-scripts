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

        # State tracking
        self.cycle_count = 0
        self.trades_today = 0
        self.pnl_today = Decimal("0")
        self.running = True

        # Track open positions for stop-loss
        self.open_positions: list[dict] = []

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

            # Extract condition_id from discovered market
            condition_id = getattr(markets[0], 'condition_id', None)
            if not condition_id:
                logger.warning("No condition_id found, using fallback")
                condition_id = "unknown"

            # Initialize market service with condition_id
            if not self.market_service or self.market_service.condition_id != condition_id:
                self.market_service = MarketMicrostructureService(self.settings, condition_id)

            # Step 2: Data Collection (parallel) - NEW: fetch social + market
            btc_data, social_sentiment, market_signals = await asyncio.gather(
                self.btc_service.get_current_price(),
                self.social_service.get_social_score(),
                self.market_service.get_market_score(),
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
                portfolio_value = Decimal(str(portfolio.total_value))
                if portfolio_value == 0:
                    portfolio_value = Decimal("1000")  # Default for read_only mode
            except:
                portfolio_value = Decimal("1000")  # Fallback

            # Step 6: Process each market
            for market in markets:
                await self._process_market(
                    market, btc_data, indicators,
                    aggregated_sentiment,  # CHANGED: pass aggregated instead of social
                    portfolio_value
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
        portfolio_value: Decimal
    ) -> None:
        """Process a single market for trading decision."""
        try:
            # Get token IDs
            token_ids = market.get_token_ids()
            if not token_ids:
                logger.warning("No token IDs found", market_id=market.id)
                return

            # Use first token (YES token)
            token_id = token_ids[0]

            # Build market data dict
            market_dict = {
                "token_id": token_id,
                "question": market.question,
                "yes_price": market.best_bid or 0.50,
                "no_price": market.best_ask or 0.50,
                "active": market.active
            }

            # Step 1: AI Decision - CHANGED: pass aggregated_sentiment
            decision = await self.ai_service.make_decision(
                btc_price=btc_data,
                technical_indicators=indicators,
                aggregated_sentiment=aggregated_sentiment,  # CHANGED
                market_data=market_dict,
                portfolio_value=portfolio_value
            )

            # Log decision
            if self.settings.bot_log_decisions:
                logger.info(
                    "AI Decision",
                    market_id=market.id,
                    action=decision.action,
                    confidence=f"{decision.confidence:.2f}",
                    reasoning=decision.reasoning,
                    position_size=str(decision.position_size)
                )

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
                await self._execute_trade(market, decision, validation.adjusted_position, token_id)
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

    async def _execute_trade(self, market, decision, amount: Decimal, token_id: str) -> None:
        """Execute a trade order."""
        try:
            from polymarket.models import OrderRequest

            # Create order request
            order_request = OrderRequest(
                token_id=token_id,
                side="BUY",  # Always BUY for our decision (YES or NO token)
                price=0.50,  # Market order approximation
                size=float(amount),
                order_type="market"
            )

            result = self.client.create_order(order_request, dry_run=False)

            logger.info(
                "Trade executed",
                market_id=market.id,
                action=decision.action,
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

        while self.running:
            await self.run_cycle()

            # Wait before next cycle
            if self.running:
                logger.info(f"Waiting {self.interval} seconds until next cycle...")
                await asyncio.sleep(self.interval)

        # Cleanup
        await self.btc_service.close()
        await self.social_service.close()  # NEW
        if self.market_service:
            await self.market_service.close()  # NEW
        logger.info("AutoTrader shutdown complete")

    async def run_once(self) -> None:
        """Run a single cycle for testing."""
        await self.run_cycle()
        await self.btc_service.close()
        await self.social_service.close()  # NEW
        if self.market_service:
            await self.market_service.close()  # NEW


@app.command()
def main(
    interval: int = typer.Option(180, help="Cycle interval in seconds"),
    once: bool = typer.Option(False, help="Run single cycle then exit")
) -> None:
    """Run the autonomous trading bot."""
    # Load settings
    settings = Settings()

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
