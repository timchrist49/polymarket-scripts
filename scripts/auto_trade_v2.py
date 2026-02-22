#!/usr/bin/env python3
"""
V2 Multi-Asset Trading Bot

Uses V2 AI (ai_analysis_v2) as the sole decision engine.
Monitors 8 markets: BTC / ETH / SOL / XRP × 5m / 15m.
Executes ONLY when V2 AI confidence >= 0.90.

Usage:
    python3 -u scripts/auto_trade_v2.py            # Live trading
    DRY_RUN=true python3 -u scripts/auto_trade_v2.py  # Dry-run

No MARKET_TYPE env var needed — all 8 markets run simultaneously.
Current V1 bot (auto_trade.py) continues running unchanged alongside this.
"""

import asyncio
import logging
import os
import re
import signal
import sys
import time
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Optional

import structlog
import typer

sys.path.insert(0, str(Path(__file__).parent.parent))

from polymarket.client import (
    PolymarketClient,
    ASSET_COINGECKO_IDS,
    ASSET_NAMES,
    ASSET_SYMBOLS,
)
from polymarket.config import Settings
from polymarket.models import Market, TradingDecision
from polymarket.performance.database import PerformanceDatabase
from polymarket.performance.tracker import PerformanceTracker
from polymarket.performance.order_verifier import OrderVerifier
from polymarket.telegram.bot import TelegramBot
from polymarket.trading.btc_price import BTCPriceService
from polymarket.trading.asset_price_service import AssetPriceService
from polymarket.trading.risk import RiskManager
from polymarket.trading.smart_order_executor import SmartOrderExecutor

from AI_analysis_upgrade.ai_analysis_v2 import run_analysis, V2Prediction
from AI_analysis_upgrade.trend_analyzer import AssetTrendAnalyzer
from AI_analysis_upgrade import config as ai_config

app = typer.Typer(help="V2 Multi-Asset Trading Bot")
logger = structlog.get_logger()

# ── Configuration ──────────────────────────────────────────────────────────────

# 8 markets: (asset_key, timeframe)
MARKETS = [
    ("btc", "5m"),
    ("btc", "15m"),
    ("eth", "5m"),
    ("eth", "15m"),
    ("sol", "5m"),
    ("sol", "15m"),
    ("xrp", "5m"),
    ("xrp", "15m"),
]

# Market duration in seconds
DURATION = {"5m": 300, "15m": 900}

# Seconds elapsed since market start when V2 AI should fire
# 5m: T+3min = 180s (2 min remaining)
# 15m: T+12min = 720s (3 min remaining)
AI_TRIGGER_ELAPSED = {"5m": 180, "15m": 720}

# Execute only when V2 AI confidence >= this threshold
V2_CONFIDENCE_THRESHOLD = 0.90

# Minimum absolute USD gap required per asset before trading.
# Prevents "inflated gap_z" false signals on flat markets (e.g. SOL $0.10
# gap producing gap_z=-2.40 because realized_vol≈0 → 92% false confidence).
# Trade 1555 (SOL gap=$0.10, gap_z=-2.40) would have been blocked.
MIN_GAP_USD_REQUIRED: dict[str, float] = {
    "btc": 15.0,   # $15 minimum gap  ($67k asset, ~0.02%)
    "eth": 2.00,   # $2  minimum gap  ($1900 asset, ~0.10%)
    "sol": 0.25,   # $0.25 minimum gap ($84 asset,  ~0.30%)
    "xrp": 0.03,   # $0.03 minimum gap ($1.40 asset, ~2.1%)
}

# CLOB contradiction veto: skip when CLOB strongly disagrees with V2 direction
# and gap_z is not conclusive.  Catches the case where clob=77% YES but V2
# outputs NO@92% on a $0.10 gap (which was trade 1555's CLOB signal).
CLOB_VETO_THRESHOLD = 0.75         # CLOB vs V2 disagreement threshold
CLOB_VETO_GAP_Z_OVERRIDE = 2.5    # |gap_z| above which CLOB veto is ignored

# Seconds to wait between settlement checks
SETTLEMENT_INTERVAL_SECONDS = 120

# Extra buffer after market end before looking for the next market (seconds)
POST_MARKET_BUFFER = 30

# Seconds to sleep between market discovery retries
DISCOVERY_RETRY_INTERVAL = 15


# ── V2 Settlement ──────────────────────────────────────────────────────────────

class V2Settler:
    """Lightweight settlement for V2 multi-asset trades.

    Reads unsettled V2 trades from the trades table (signal_type='V2_AI'),
    looks up the close price for the correct asset, and records the outcome.
    """

    def __init__(
        self,
        db: PerformanceDatabase,
        tracker: PerformanceTracker,
        btc_service: BTCPriceService,
        asset_services: dict[str, AssetPriceService],
    ):
        self.db = db
        self.tracker = tracker
        self.btc_service = btc_service
        self.asset_services = asset_services  # {asset_key: AssetPriceService}

    async def _get_close_price(self, market_slug: str, close_ts: int) -> Optional[float]:
        """Get the asset's USD price at market close timestamp."""
        for asset_key, svc in self.asset_services.items():
            if market_slug.startswith(f"{asset_key}-"):
                price = await svc.get_price_at_timestamp(float(close_ts))
                return float(price) if price is not None else None
        # Default: BTC
        result = await self.btc_service.get_price_at_timestamp(close_ts)
        return float(result) if result is not None else None

    async def settle_pending(self) -> dict:
        """Settle all pending V2 trades whose markets have closed."""
        stats = {"settled": 0, "wins": 0, "losses": 0, "pending": 0, "errors": 0}
        try:
            cursor = self.db.conn.cursor()
            cursor.execute("""
                SELECT id, market_slug, action, executed_price, position_size, price_to_beat
                FROM trades
                WHERE is_win IS NULL
                  AND signal_type = 'V2_AI'
                  AND executed_price IS NOT NULL
            """)
            trades = [dict(r) for r in cursor.fetchall()]

            for trade in trades:
                try:
                    slug = trade["market_slug"] or ""
                    # Parse market start timestamp from slug (last 10 digits)
                    m = re.search(r"(\d{10})$", slug)
                    if not m:
                        logger.warning("V2 settle: cannot parse slug", slug=slug)
                        stats["errors"] += 1
                        continue

                    duration = 300 if "-5m-" in slug else 900
                    close_ts = int(m.group(1)) + duration

                    # Skip if market not yet closed
                    if time.time() < close_ts + 10:  # +10s grace
                        stats["pending"] += 1
                        continue

                    close_price = await self._get_close_price(slug, close_ts)
                    if close_price is None:
                        logger.warning("V2 settle: close price unavailable", slug=slug)
                        stats["pending"] += 1
                        continue

                    ptb = trade["price_to_beat"]
                    if ptb is None:
                        logger.warning("V2 settle: no PTB stored", trade_id=trade["id"])
                        stats["errors"] += 1
                        continue

                    actual_outcome = "YES" if close_price > float(ptb) else "NO"
                    is_win = trade["action"] == actual_outcome

                    # P&L: stake × (1/odds - 1) for wins, -stake for losses
                    ep = float(trade["executed_price"] or 0.5)
                    ps = float(trade["position_size"] or 0)
                    pnl = ps * (1.0 / ep - 1.0) if is_win and ep > 0 else -ps

                    self.tracker.update_trade_outcome(
                        trade_id=trade["id"],
                        actual_outcome=actual_outcome,
                        profit_loss=pnl,
                        is_win=is_win,
                    )
                    stats["settled"] += 1
                    if is_win:
                        stats["wins"] += 1
                    else:
                        stats["losses"] += 1

                    logger.info(
                        "V2 trade settled",
                        trade_id=trade["id"],
                        slug=slug,
                        action=trade["action"],
                        actual=actual_outcome,
                        is_win=is_win,
                        pnl=f"${pnl:.2f}",
                    )
                except Exception as e:
                    logger.error("V2 settle trade error", trade_id=trade.get("id"), error=str(e))
                    stats["errors"] += 1

        except Exception as e:
            logger.error("V2 settle_pending error", error=str(e))
            stats["errors"] += 1

        return stats


# ── Per-Market Monitor ─────────────────────────────────────────────────────────

class MarketMonitor:
    """Monitors one (asset, timeframe) market and fires V2 AI at trigger time.

    Lifecycle per market:
      1. Discover active market via Polymarket Gamma API
      2. Sleep until AI trigger time (elapsed = AI_TRIGGER_ELAPSED[timeframe])
      3. Gather: current price, PTB, realized vol, trend, CLOB odds
      4. Run V2 AI analysis
      5. Log to ai_analysis_log
      6. If confidence >= 0.90 AND dry_run=False: execute trade via SmartOrderExecutor
      7. Log trade to trades table; send Telegram alert
      8. Sleep until market end + POST_MARKET_BUFFER; go to 1
    """

    def __init__(
        self,
        asset: str,
        timeframe: str,
        client: PolymarketClient,
        btc_service: BTCPriceService,
        asset_services: dict[str, AssetPriceService],
        trend_analyzers: dict[str, AssetTrendAnalyzer],
        settings: Settings,
        db: PerformanceDatabase,
        tracker: PerformanceTracker,
        telegram: TelegramBot,
        risk_manager: RiskManager,
        order_verifier: OrderVerifier,
        dry_run: bool = False,
    ):
        self.asset = asset
        self.timeframe = timeframe
        self.duration = DURATION[timeframe]
        self.ai_trigger_elapsed = AI_TRIGGER_ELAPSED[timeframe]
        self.label = f"{asset.upper()}-{timeframe}"

        self.client = client
        self.btc_service = btc_service
        self.asset_services = asset_services
        self.trend_analyzers = trend_analyzers
        self.settings = settings
        self.db = db
        self.tracker = tracker
        self.telegram = telegram
        self.risk_manager = risk_manager
        self.order_verifier = order_verifier
        self.dry_run = dry_run

        # Track markets we've already traded this session (no double-entry)
        self._traded_market_ids: set[str] = set()

        # Per-market state: reset at each market discovery
        # _market_open_price: live price captured at T=0 (used as PTB)
        # _current_market_trade_id: trade_id logged for the active market
        self._market_open_price: Optional[float] = None
        self._current_market_trade_id: Optional[int] = None

    def _get_price_service(self):
        """Return the correct price service for this monitor's asset."""
        if self.asset == "btc":
            return self.btc_service
        return self.asset_services[self.asset]

    async def _get_ptb(self, market_start_ts: float) -> Optional[float]:
        """Return the price-to-beat for this market.

        Prefers the live price captured at market open (T=0), which is fetched
        immediately at market discovery and most closely matches the Chainlink
        oracle price Polymarket uses for resolution.

        Falls back to buffer-based timestamp lookup when the open-time capture
        is unavailable (e.g. bot restarted mid-market).
        """
        if self._market_open_price is not None:
            logger.debug(
                f"{self.label} PTB from open-time capture",
                price=f"{self._market_open_price:.4f}",
            )
            return self._market_open_price
        # Fallback: buffer lookup
        if self.asset == "btc":
            result = await self.btc_service.get_price_at_timestamp(int(market_start_ts))
            return float(result) if result is not None else None
        svc = self.asset_services[self.asset]
        return await svc.get_price_at_timestamp(market_start_ts)

    async def _get_current_price(self) -> Optional[float]:
        """Fetch latest price for this asset."""
        if self.asset == "btc":
            try:
                btc_data = await self.btc_service.get_current_price()
                return float(btc_data.price)
            except Exception as e:
                logger.warning(f"{self.label} BTC price fetch failed", error=str(e))
                return None
        svc = self.asset_services[self.asset]
        return await svc.get_current_price()

    def _get_realized_vol(self) -> float:
        """Return realized vol/min from the rolling CoinGecko buffer.

        All assets (including BTC) use AssetPriceService for this.
        Falls back to 15.0 $/min for BTC if the service has no data yet.
        """
        svc = self.asset_services.get(self.asset)
        if svc is None:
            return 15.0 if self.asset == "btc" else 0.0
        vol = svc.get_realized_vol_per_min()
        # If buffer is too short to compute vol, use a BTC-specific default
        if vol == 0.0 and self.asset == "btc":
            return 15.0
        return vol

    def _get_clob_yes_odds(self, market: Market) -> float:
        """Get YES token CLOB midpoint from orderbook REST API.

        Falls back to market.best_bid if orderbook is unavailable.
        """
        try:
            token_ids = market.get_token_ids()
            if not token_ids:
                raise ValueError("No token IDs")
            yes_token_id = token_ids[0]
            book = self.client.get_orderbook(yes_token_id, depth=5)
            if not book:
                raise ValueError("Empty orderbook")
            bids = book.get("bids", [])
            asks = book.get("asks", [])
            # CLOB returns dicts {"price": "0.62", "size": ...}, sorted:
            #   bids ascending (worst→best), use bids[-1] for best bid
            #   asks descending (worst→best), use asks[-1] for best ask
            if bids and asks:
                bid_p = bids[-1]["price"] if isinstance(bids[-1], dict) else bids[-1][0]
                ask_p = asks[-1]["price"] if isinstance(asks[-1], dict) else asks[-1][0]
                return (float(bid_p) + float(ask_p)) / 2.0
            if bids:
                b = bids[-1]["price"] if isinstance(bids[-1], dict) else bids[-1][0]
                return float(b)
        except Exception as e:
            logger.debug(f"{self.label} orderbook failed, using best_bid", error=str(e))
        # Fallback
        if market.best_bid:
            return float(market.best_bid)
        return 0.5

    async def _discover_market(self) -> Optional[Market]:
        """Discover the current active market. Returns None on failure."""
        try:
            market = await asyncio.to_thread(self.client.discover_market, self.asset, self.timeframe)
            return market
        except Exception as e:
            logger.warning(f"{self.label} market discovery failed", error=str(e))
            return None

    async def _settle_trade_now(self, trade_id: int, close_price: float) -> None:
        """Settle a trade immediately using the live close price at market end.

        Called ~5s after market end so the price matches the Chainlink oracle
        reading used for resolution.  The V2Settler (running every 120s) will
        naturally skip this trade since is_win will already be set.
        """
        try:
            cursor = self.db.conn.cursor()
            cursor.execute(
                "SELECT action, executed_price, position_size, price_to_beat, is_win "
                "FROM trades WHERE id = ?",
                (trade_id,),
            )
            row = cursor.fetchone()
            if not row:
                logger.warning(f"{self.label} settle_now: trade not found", trade_id=trade_id)
                return
            if row["is_win"] is not None:
                logger.debug(f"{self.label} settle_now: already settled", trade_id=trade_id)
                return

            action = row["action"]
            ep = float(row["executed_price"] or 0.5)
            ps = float(row["position_size"] or 0)
            ptb = float(row["price_to_beat"] or 0)

            actual_outcome = "YES" if close_price > ptb else "NO"
            is_win = action == actual_outcome
            pnl = ps * (1.0 / ep - 1.0) if is_win and ep > 0 else -ps

            self.tracker.update_trade_outcome(
                trade_id=trade_id,
                actual_outcome=actual_outcome,
                profit_loss=pnl,
                is_win=is_win,
            )
            logger.info(
                f"{self.label} trade settled at market close",
                trade_id=trade_id,
                close_price=f"{close_price:.4f}",
                ptb=f"{ptb:.4f}",
                gap=f"{close_price - ptb:+.4f}",
                actual=actual_outcome,
                is_win=is_win,
                pnl=f"${pnl:.2f}",
            )
        except Exception as e:
            logger.error(f"{self.label} settle_now failed", trade_id=trade_id, error=str(e))

    async def _log_ai_analysis(
        self,
        market: Market,
        prediction: V2Prediction,
        current_price: float,
        ptb: float,
    ) -> int:
        """Log V2 AI prediction to ai_analysis_log. Returns row id."""
        try:
            gap_usd = current_price - ptb
            row_id = self.db.log_ai_analysis(
                market_slug=market.slug or market.id,
                market_id=market.id,
                bot_type=f"v2_{self.asset}_{self.timeframe}",
                action=prediction.action,
                confidence=prediction.confidence,
                reasoning=prediction.reasoning,
                btc_price=current_price,
                ptb_price=ptb,
                btc_movement=gap_usd,
            )
            return row_id
        except Exception as e:
            logger.warning(f"{self.label} log_ai_analysis failed", error=str(e))
            return -1

    async def _execute_trade(
        self,
        market: Market,
        prediction: V2Prediction,
        current_price: float,
        ptb: float,
        clob_yes: float,
        time_remaining: int,
    ) -> None:
        """Execute a trade via SmartOrderExecutor and log to DB + Telegram."""
        token_ids = market.get_token_ids()
        if not token_ids or len(token_ids) < 2:
            logger.error(f"{self.label} no token IDs on market", market_id=market.id)
            return

        # YES = token[0], NO = token[1]
        if prediction.action == "YES":
            token_id = token_ids[0]
            token_name = "Yes"
            execution_price = clob_yes
        else:
            token_id = token_ids[1]
            token_name = "No"
            execution_price = 1.0 - clob_yes

        # Skip if already traded this market
        if market.id in self._traded_market_ids:
            logger.info(f"{self.label} already traded market {market.id}, skipping")
            return

        # Position sizing via RiskManager
        try:
            portfolio = await asyncio.to_thread(self.client.get_portfolio_summary)
            portfolio_value = Decimal(str(portfolio.usdc_balance))
        except Exception:
            portfolio_value = Decimal("500")  # Safe fallback

        decision = TradingDecision(
            action=prediction.action,
            confidence=prediction.confidence,
            position_size=Decimal("0"),  # RiskManager will compute
            reasoning=prediction.reasoning or "V2 AI",
            token_id=token_id,
            stop_loss_threshold=0.0,  # V2 bot does not use stop-loss
        )

        validation = await self.risk_manager.validate_decision(
            decision=decision,
            portfolio_value=portfolio_value,
            market={"active": True, "best_bid": clob_yes, "best_ask": clob_yes},
            open_positions=[],
            test_mode=False,
        )

        if not validation.approved or not validation.adjusted_position:
            logger.info(
                f"{self.label} risk check rejected",
                reason=validation.reason,
                confidence=f"{prediction.confidence:.2f}",
            )
            return

        amount = validation.adjusted_position

        if self.dry_run:
            logger.info(
                f"[DRY-RUN] {self.label} would execute",
                action=prediction.action,
                confidence=f"{prediction.confidence:.0%}",
                amount=str(amount),
                price=f"{execution_price:.3f}",
            )
            return

        # Log to trades table BEFORE executing (get trade_id)
        trade_data = {
            "timestamp": datetime.now(),
            "market_slug": market.slug or market.id,
            "market_id": market.id,
            "action": prediction.action,
            "confidence": prediction.confidence,
            "position_size": float(amount),
            "reasoning": prediction.reasoning,
            "btc_price": current_price,
            "price_to_beat": ptb,
            "time_remaining_seconds": time_remaining,
            "signal_type": "V2_AI",
            "yes_price": clob_yes,
            "no_price": 1.0 - clob_yes,
            "executed_price": execution_price,
        }
        try:
            trade_id = self.db.log_trade(trade_data)
        except Exception as e:
            logger.error(f"{self.label} log_trade failed", error=str(e))
            trade_id = -1

        # Track trade_id so _run() can settle it immediately at market close
        self._current_market_trade_id = trade_id

        # Execute via SmartOrderExecutor
        smart_executor = SmartOrderExecutor()
        token_price = execution_price
        try:
            execution_result = await smart_executor.execute_smart_order(
                client=self.client,
                token_id=token_id,
                side="BUY",
                amount=float(amount),
                urgency="HIGH",
                current_best_ask=token_price + 0.02,
                current_best_bid=token_price,
                tick_size=0.001,
                max_fallback_price=min(token_price * 1.05, 0.97),
            )
        except Exception as e:
            logger.error(f"{self.label} execute_smart_order failed", error=str(e))
            return

        if execution_result.get("status") != "FILLED":
            logger.warning(
                f"{self.label} order not filled",
                status=execution_result.get("status"),
                reason=execution_result.get("message"),
            )
            return

        order_id = execution_result.get("order_id", "")
        self._traded_market_ids.add(market.id)

        logger.info(
            f"{self.label} trade executed",
            action=prediction.action,
            confidence=f"{prediction.confidence:.0%}",
            amount=str(amount),
            price=f"{execution_price:.3f}",
            order_id=order_id,
        )

        # Update trade record with order_id
        if trade_id > 0:
            try:
                cursor = self.db.conn.cursor()
                cursor.execute(
                    "UPDATE trades SET order_id = ? WHERE id = ?",
                    (order_id, trade_id),
                )
                self.db.conn.commit()
            except Exception:
                pass

        # Telegram alert
        try:
            await self.telegram.send_trade_alert(
                market_slug=market.slug or market.id,
                action=prediction.action,
                confidence=prediction.confidence,
                position_size=float(amount),
                price=execution_price,
                reasoning=f"[V2 {self.label}] {prediction.reasoning}",
                btc_current=current_price,
                btc_price_to_beat=ptb,
            )
        except Exception as e:
            logger.warning(f"{self.label} Telegram alert failed", error=str(e))

    async def run_loop(self) -> None:
        """Infinite loop: discover → wait → analyse → maybe execute → wait → repeat."""
        logger.info(f"MarketMonitor started", monitor=self.label)

        while True:
            try:
                # ── 1. Discover current market ─────────────────────────────
                market = None
                while market is None:
                    market = await self._discover_market()
                    if market is None:
                        await asyncio.sleep(DISCOVERY_RETRY_INTERVAL)

                # end_date is reliable from Gamma API; start_date is not (can be wrong).
                # Derive start from end_date - duration to get correct timing.
                if market.end_date:
                    market_end_ts = market.end_date.timestamp()
                    market_start_ts = market_end_ts - self.duration
                elif market.start_date:
                    market_start_ts = market.start_date.timestamp()
                    market_end_ts = market_start_ts + self.duration
                else:
                    market_start_ts = time.time()
                    market_end_ts = market_start_ts + self.duration
                ai_fire_ts = market_start_ts + self.ai_trigger_elapsed

                logger.info(
                    f"{self.label} market discovered",
                    slug=market.slug,
                    market_id=market.id,
                    ai_fires_in=f"{max(0, ai_fire_ts - time.time()):.0f}s",
                    ends_in=f"{max(0, market_end_ts - time.time()):.0f}s",
                )

                # ── 1.5 Reset per-market state; capture live price as PTB ──
                # Fetched immediately at discovery (T≈0) so it matches the
                # Chainlink oracle reference price as closely as possible.
                self._market_open_price = None
                self._current_market_trade_id = None
                _open_price = await self._get_current_price()
                if _open_price is not None:
                    self._market_open_price = _open_price
                    logger.info(
                        f"{self.label} PTB captured at market open",
                        price=f"{_open_price:.4f}",
                        asset=self.asset,
                    )

                # ── 2. If AI trigger has already passed, skip this market ──
                now = time.time()
                if now >= ai_fire_ts:
                    time_to_end = max(0, market_end_ts - now) + POST_MARKET_BUFFER
                    logger.info(
                        f"{self.label} AI trigger already passed, skipping market",
                        slug=market.slug,
                        wait=f"{time_to_end:.0f}s",
                    )
                    await asyncio.sleep(time_to_end)
                    continue

                # ── 3. Sleep until AI trigger time ─────────────────────────
                wait = ai_fire_ts - time.time()
                if wait > 0:
                    logger.debug(f"{self.label} sleeping {wait:.0f}s until AI trigger")
                    await asyncio.sleep(wait)

                # ── 4. Gather inputs ───────────────────────────────────────
                now = time.time()
                time_remaining = max(0, int(market_end_ts - now))

                current_price = await self._get_current_price()
                if current_price is None:
                    logger.warning(f"{self.label} current price unavailable, skipping")
                    await asyncio.sleep(max(0, market_end_ts - time.time()) + POST_MARKET_BUFFER)
                    continue

                ptb = await self._get_ptb(market_start_ts)
                if ptb is None:
                    logger.warning(f"{self.label} PTB unavailable, using current price")
                    ptb = current_price

                realized_vol = self._get_realized_vol()
                clob_yes = self._get_clob_yes_odds(market)

                trend_analyzer = self.trend_analyzers[self.asset]
                try:
                    trend = await trend_analyzer.compute()
                except Exception as e:
                    logger.warning(f"{self.label} trend compute failed", error=str(e))
                    from AI_analysis_upgrade.trend_analyzer import TrendResult
                    trend = TrendResult(trend_score=0.0, fear_greed=50, timeframe_scores={})

                gap_usd = current_price - ptb

                logger.info(
                    f"{self.label} V2 AI analysis starting",
                    current=f"{current_price:.4f}",
                    ptb=f"{ptb:.4f}",
                    gap=f"{gap_usd:+.4f}",
                    clob_yes=f"{clob_yes:.2f}",
                    time_remaining=f"{time_remaining}s",
                    trend=f"{trend.trend_score:+.2f}",
                )

                # ── 5. Run V2 AI ───────────────────────────────────────────
                prediction = await run_analysis(
                    market_slug=market.slug or market.id,
                    gap_usd=gap_usd,
                    clob_yes=clob_yes,
                    time_remaining_seconds=time_remaining,
                    btc_price=current_price,
                    ptb=ptb,
                    realized_vol_per_min=realized_vol,
                    trend=trend,
                    market_duration_seconds=self.duration,
                    asset_name=ASSET_NAMES.get(self.asset, self.asset.upper()),
                    asset_symbol=ASSET_SYMBOLS.get(self.asset, self.asset.upper()),
                )

                logger.info(
                    f"{self.label} V2 AI result",
                    action=prediction.action,
                    confidence=f"{prediction.confidence:.0%}",
                    phase=prediction.phase,
                    gap_z=f"{prediction.gap_z:.2f}",
                    reasoning=prediction.reasoning[:100] if prediction.reasoning else "",
                )

                # ── 6. Log AI analysis ─────────────────────────────────────
                await self._log_ai_analysis(market, prediction, current_price, ptb)

                # ── 7. Execute if confidence >= threshold ──────────────────
                if prediction.action in ("YES", "NO") and prediction.confidence >= V2_CONFIDENCE_THRESHOLD:
                    logger.info(
                        f"{self.label} CONFIDENCE THRESHOLD MET — checking filters",
                        action=prediction.action,
                        confidence=f"{prediction.confidence:.0%}",
                    )

                    # ── Layer 1: Minimum absolute USD gap ──────────────────
                    # Prevents inflated gap_z from near-zero realized_vol
                    # dividing a tiny USD gap (the trade 1555 failure mode).
                    gap_usd_abs = abs(gap_usd)
                    min_gap = MIN_GAP_USD_REQUIRED.get(self.asset, 0.0)
                    if gap_usd_abs < min_gap:
                        logger.info(
                            f"{self.label} SKIP — gap_usd below noise floor",
                            gap_usd=f"{gap_usd:+.4f}",
                            min_required=f"{min_gap:.4f}",
                            gap_z=f"{prediction.gap_z:.2f}",
                            asset=self.asset,
                        )

                    # ── Layer 2: CLOB contradiction veto ───────────────────
                    # If CLOB strongly disagrees with our direction and gap_z
                    # is not conclusively large, defer to market consensus.
                    elif (
                        (prediction.action == "NO"
                         and clob_yes > CLOB_VETO_THRESHOLD
                         and abs(prediction.gap_z) < CLOB_VETO_GAP_Z_OVERRIDE)
                        or
                        (prediction.action == "YES"
                         and clob_yes < (1.0 - CLOB_VETO_THRESHOLD)
                         and abs(prediction.gap_z) < CLOB_VETO_GAP_Z_OVERRIDE)
                    ):
                        logger.info(
                            f"{self.label} SKIP — CLOB contradicts V2 direction",
                            v2_action=prediction.action,
                            clob_yes=f"{clob_yes:.2f}",
                            gap_z=f"{prediction.gap_z:.2f}",
                            veto_threshold=f"{CLOB_VETO_THRESHOLD:.2f}",
                        )

                    else:
                        logger.info(
                            f"{self.label} ALL FILTERS PASSED — executing",
                            action=prediction.action,
                            confidence=f"{prediction.confidence:.0%}",
                            gap_usd=f"{gap_usd:+.4f}",
                            gap_z=f"{prediction.gap_z:.2f}",
                        )
                        await self._execute_trade(
                            market=market,
                            prediction=prediction,
                            current_price=current_price,
                            ptb=ptb,
                            clob_yes=clob_yes,
                            time_remaining=time_remaining,
                        )
                else:
                    logger.info(
                        f"{self.label} confidence below threshold — no trade",
                        action=prediction.action,
                        confidence=f"{prediction.confidence:.0%}",
                        threshold=f"{V2_CONFIDENCE_THRESHOLD:.0%}",
                    )

                # ── 8. Wait for market close; capture close price and settle ─
                # Sleep until the market ends, then immediately fetch the live
                # price to use as the close price.  This matches the Chainlink
                # oracle's reference time as closely as possible, minimising the
                # systematic CoinGecko ↔ Chainlink delta.
                time_to_close = max(0, market_end_ts - time.time())
                if time_to_close > 0:
                    logger.debug(f"{self.label} waiting {time_to_close:.0f}s for market close")
                    await asyncio.sleep(time_to_close)

                # Fetch live close price right at market end (~5s grace)
                close_price = await self._get_current_price()
                if close_price is not None:
                    logger.info(
                        f"{self.label} close price at market end",
                        price=f"{close_price:.4f}",
                        asset=self.asset,
                    )
                    if self._current_market_trade_id and self._current_market_trade_id > 0:
                        await self._settle_trade_now(self._current_market_trade_id, close_price)

                # Buffer before looking for the next market
                await asyncio.sleep(POST_MARKET_BUFFER)

            except asyncio.CancelledError:
                logger.info(f"{self.label} monitor cancelled")
                return
            except Exception as e:
                logger.error(f"{self.label} monitor error", error=str(e))
                await asyncio.sleep(DISCOVERY_RETRY_INTERVAL)


# ── Main Orchestrator ──────────────────────────────────────────────────────────

class AutoTraderV2:
    """V2 multi-asset trading bot orchestrator."""

    def __init__(self, settings: Settings, dry_run: bool = False):
        self.settings = settings
        self.dry_run = dry_run
        self.running = True

        self.client = PolymarketClient()
        self.btc_service = BTCPriceService(settings)
        self.telegram = TelegramBot(settings)
        self.risk_manager = RiskManager(settings)

        # Per-asset price services (all 4 assets from CoinGecko).
        # BTC uses this for realized-vol computation; BTCPriceService (Chainlink) is
        # still used for PTB and current-price where Chainlink gives better accuracy.
        cg_key = os.getenv("COINGECKO_API_KEY", getattr(ai_config, "COINGECKO_API_KEY", ""))
        self.asset_services: dict[str, AssetPriceService] = {
            asset: AssetPriceService(coin_id=ASSET_COINGECKO_IDS[asset], api_key=cg_key)
            for asset in ("btc", "eth", "sol", "xrp")
        }

        # One trend analyzer per asset (shared between 5m and 15m monitors)
        self.trend_analyzers: dict[str, AssetTrendAnalyzer] = {
            asset: AssetTrendAnalyzer(
                coin_id=ASSET_COINGECKO_IDS[asset],
                coingecko_api_key=cg_key,
            )
            for asset in ("btc", "eth", "sol", "xrp")
        }

        # Performance tracking + settlement
        self.tracker = PerformanceTracker()
        self.db = self.tracker.db
        self.order_verifier = OrderVerifier(client=self.client, db=self.db)

        self.settler = V2Settler(
            db=self.db,
            tracker=self.tracker,
            btc_service=self.btc_service,
            asset_services=self.asset_services,
        )

        self.background_tasks: list[asyncio.Task] = []

    async def _initialize(self) -> None:
        """Start all async services."""
        logger.info("Initializing V2 bot services...")

        # Start BTC price service (Chainlink WebSocket)
        await self.btc_service.start()
        logger.info("BTC price service started")

        # Start CoinGecko price services for all assets (BTC, ETH, SOL, XRP)
        # (used for realized-vol computation and as fallback PTB)
        for asset, svc in self.asset_services.items():
            await svc.start()
            logger.info(f"{asset.upper()} CoinGecko price service started", price=svc._current_price)

        logger.info(
            "V2 bot initialized",
            markets=len(MARKETS),
            dry_run=self.dry_run,
            confidence_threshold=f"{V2_CONFIDENCE_THRESHOLD:.0%}",
        )

    async def _settlement_loop(self) -> None:
        """Periodic settlement of V2 trades."""
        logger.info("V2 settlement loop started", interval=SETTLEMENT_INTERVAL_SECONDS)
        while self.running:
            await asyncio.sleep(SETTLEMENT_INTERVAL_SECONDS)
            try:
                stats = await self.settler.settle_pending()
                if stats["settled"] > 0:
                    logger.info(
                        "V2 settlement cycle",
                        settled=stats["settled"],
                        wins=stats["wins"],
                        losses=stats["losses"],
                    )
            except Exception as e:
                logger.error("V2 settlement loop error", error=str(e))

    async def run(self) -> None:
        """Run the V2 bot until interrupted."""
        def _signal_handler(sig, frame):
            logger.info("V2 bot shutting down gracefully...")
            self.running = False

        signal.signal(signal.SIGINT, _signal_handler)
        signal.signal(signal.SIGTERM, _signal_handler)

        await self._initialize()

        # Launch one monitor per (asset, timeframe)
        for asset, timeframe in MARKETS:
            monitor = MarketMonitor(
                asset=asset,
                timeframe=timeframe,
                client=self.client,
                btc_service=self.btc_service,
                asset_services=self.asset_services,
                trend_analyzers=self.trend_analyzers,
                settings=self.settings,
                db=self.db,
                tracker=self.tracker,
                telegram=self.telegram,
                risk_manager=self.risk_manager,
                order_verifier=self.order_verifier,
                dry_run=self.dry_run,
            )
            task = asyncio.create_task(monitor.run_loop(), name=f"v2_{asset}_{timeframe}")
            self.background_tasks.append(task)

        # Settlement loop
        settle_task = asyncio.create_task(self._settlement_loop(), name="v2_settler")
        self.background_tasks.append(settle_task)

        logger.info(
            "V2 bot running",
            monitors=len(MARKETS),
            assets=["BTC", "ETH", "SOL", "XRP"],
            timeframes=["5m", "15m"],
        )

        # Keep alive
        while self.running:
            await asyncio.sleep(1)

        # Cleanup
        for task in self.background_tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        # Close price services
        await self.btc_service.close()
        for svc in self.asset_services.values():
            await svc.stop()
        for analyzer in self.trend_analyzers.values():
            await analyzer.close()

        self.tracker.close()
        logger.info("V2 bot shutdown complete")


# ── Entry Point ────────────────────────────────────────────────────────────────

@app.command()
def main() -> None:
    """Run the V2 multi-asset trading bot."""
    settings = Settings()

    dry_run = os.getenv("DRY_RUN", "false").lower() == "true" or settings.dry_run

    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=logging.INFO)
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer()
            if settings.log_json
            else structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    if dry_run:
        logger.warning("V2 bot starting in DRY-RUN mode (no real trades)")

    bot = AutoTraderV2(settings, dry_run=dry_run)
    asyncio.run(bot.run())


if __name__ == "__main__":
    app()
