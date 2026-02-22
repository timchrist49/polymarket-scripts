#!/usr/bin/env python3
"""
Quant Shadow Service

Monitors 8 markets (BTC/ETH/SOL/XRP × 5m/15m) in pure shadow mode.
No orders are placed. At analysis time, fires logistic-regression prediction,
logs features + edge to quant_shadow_log. After settlement sends Telegram
comparing quant prediction vs V2 decision vs actual outcome.

Usage:
    python3 -u scripts/quant_shadow.py
"""

import asyncio
import os
import re
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import structlog

sys.path.insert(0, str(Path(__file__).parent.parent))

from polymarket.client import PolymarketClient, ASSET_COINGECKO_IDS
from polymarket.config import Settings
from polymarket.models import Market
from polymarket.performance.database import PerformanceDatabase
from polymarket.telegram.bot import TelegramBot
from polymarket.trading.btc_price import BTCPriceService
from polymarket.trading.asset_price_service import AssetPriceService
from AI_analysis_upgrade.trend_analyzer import AssetTrendAnalyzer
from AI_analysis_upgrade.ai_analysis_v2 import compute_gap_z, get_phase
from quant_shadow.model import load_model, predict as quant_predict

logger = structlog.get_logger()

# ── Config ───────────────────────────────────────────────────────────────────

MARKETS = [
    ("btc", "5m"), ("btc", "15m"),
    ("eth", "5m"), ("eth", "15m"),
    ("sol", "5m"), ("sol", "15m"),
    ("xrp", "5m"), ("xrp", "15m"),
]
DURATION = {"5m": 300, "15m": 900}
AI_TRIGGER_ELAPSED = {"5m": 180, "15m": 720}
SETTLEMENT_CHECK_INTERVAL = 10    # seconds between close-price retry attempts
SETTLEMENT_MAX_ATTEMPTS = 12      # up to 2 minutes of retries
POST_MARKET_BUFFER = 30
DISCOVERY_RETRY_INTERVAL = 15


# ── Per-market monitor ───────────────────────────────────────────────────────

class QuantShadowMonitor:
    """Monitors one (asset, timeframe) pair. Fires quant prediction at trigger time."""

    def __init__(
        self,
        asset: str,
        timeframe: str,
        client: PolymarketClient,
        btc_service: BTCPriceService,
        asset_services: dict,
        trend_analyzers: dict,
        db: PerformanceDatabase,
        telegram: Optional[TelegramBot],
        model,
        scaler,
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
        self.db = db
        self.telegram = telegram
        self.model = model
        self.scaler = scaler

        self._analyzed_markets: set[str] = set()

    # ── Price helpers ────────────────────────────────────────────────────────

    async def _get_current_price(self) -> Optional[float]:
        try:
            if self.asset == "btc":
                data = await self.btc_service.get_current_price()
                return float(data.price) if data else None
            svc = self.asset_services.get(self.asset)
            return await svc.get_current_price() if svc else None
        except Exception as e:
            logger.warning(f"{self.label} current price failed", error=str(e))
            return None

    async def _get_price_at(self, ts: float) -> Optional[float]:
        try:
            if self.asset == "btc":
                r = await self.btc_service.get_price_at_timestamp(int(ts))
                return float(r) if r else None
            svc = self.asset_services.get(self.asset)
            return await svc.get_price_at_timestamp(ts) if svc else None
        except Exception as e:
            logger.warning(f"{self.label} price_at failed", error=str(e))
            return None

    def _get_realized_vol(self) -> float:
        svc = self.asset_services.get(self.asset)
        if svc:
            v = svc.get_realized_vol_per_min()
            if v > 0:
                return v
        return 15.0 if self.asset == "btc" else 5.0

    def _get_clob_yes(self, market: Market) -> float:
        try:
            token_ids = market.get_token_ids()
            if token_ids:
                book = self.client.get_orderbook(token_ids[0], depth=5)
                if book:
                    bids = book.get("bids", [])
                    asks = book.get("asks", [])
                    if bids and asks:
                        return (float(bids[0][0]) + float(asks[0][0])) / 2.0
        except Exception:
            pass
        return float(market.best_bid) if market.best_bid else 0.5

    async def _get_trend_score(self) -> float:
        try:
            analyzer = self.trend_analyzers.get(self.asset)
            if analyzer:
                result = await asyncio.to_thread(analyzer.analyze)
                return float(result.trend_score) if result else 0.0
        except Exception:
            pass
        return 0.0

    # ── Main lifecycle ───────────────────────────────────────────────────────

    async def run(self) -> None:
        """Main loop: discover market → predict → settle → repeat."""
        while True:
            try:
                await self._run_one_market()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"{self.label} run loop error", error=str(e))
                await asyncio.sleep(DISCOVERY_RETRY_INTERVAL)

    async def _discover_market(self) -> Market:
        while True:
            try:
                market = await asyncio.to_thread(
                    self.client.discover_market, self.asset, self.timeframe
                )
                return market
            except Exception as e:
                logger.warning(f"{self.label} discovery failed", error=str(e))
                await asyncio.sleep(DISCOVERY_RETRY_INTERVAL)

    async def _run_one_market(self) -> None:
        market = await self._discover_market()

        # Derive timestamps (end_date is reliable; start_date may be stale)
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
        now = time.time()

        # Sleep until trigger time
        if now < ai_fire_ts:
            wait = ai_fire_ts - now
            logger.info(
                f"{self.label} waiting",
                fires_in=f"{wait:.0f}s",
                ends_in=f"{market_end_ts - now:.0f}s",
                slug=market.slug,
            )
            await asyncio.sleep(wait)

        # Skip if already analyzed this market window
        if market.id in self._analyzed_markets:
            await asyncio.sleep(max(0, market_end_ts - time.time() + POST_MARKET_BUFFER))
            return

        # Gather features
        now = time.time()
        time_remaining = int(market_end_ts - now)
        if time_remaining <= 0:
            self._analyzed_markets.add(market.id)
            return

        current_price = await self._get_current_price()
        ptb = await self._get_price_at(market_start_ts)
        if current_price is None or ptb is None:
            logger.warning(f"{self.label} missing price data — skipping prediction")
            return

        vol = self._get_realized_vol()
        gap_usd = current_price - ptb
        gap_z = compute_gap_z(gap_usd, vol, time_remaining / 60.0)
        phase = get_phase(time_remaining, self.duration)
        clob_yes = await asyncio.to_thread(self._get_clob_yes, market)
        trend_score = await self._get_trend_score()

        # Quant prediction
        p_yes, action, edge = quant_predict(
            self.model, self.scaler, gap_z, clob_yes, trend_score, phase
        )

        logger.info(
            f"{self.label} QUANT prediction",
            slug=market.slug,
            gap_z=f"{gap_z:+.2f}",
            clob=f"{clob_yes:.2f}",
            trend=f"{trend_score:+.2f}",
            phase=phase,
            p_yes=f"{p_yes:.2f}",
            action=action,
            edge=f"{edge:+.3f}",
        )

        # Persist to DB
        row_id = self.db.log_quant_shadow(
            market_slug=market.slug or market.id,
            market_id=market.id,
            asset=self.asset,
            timeframe=self.timeframe,
            gap_usd=gap_usd,
            gap_z=gap_z,
            clob_yes=clob_yes,
            realized_vol_per_min=vol,
            trend_score=trend_score,
            time_remaining_seconds=time_remaining,
            phase=phase,
            quant_p_yes=p_yes,
            quant_action=action,
            quant_edge=edge,
        )

        self._analyzed_markets.add(market.id)

        # Wait for market to close (+ small grace period for price propagation)
        sleep_secs = max(5, market_end_ts - time.time() + 15)
        await asyncio.sleep(sleep_secs)

        # Settle and alert
        await self._settle_and_alert(
            row_id, market, ptb, p_yes, action, edge,
            gap_z, clob_yes, vol, trend_score, time_remaining, phase, market_end_ts,
        )

        await asyncio.sleep(POST_MARKET_BUFFER)

    async def _settle_and_alert(
        self, row_id, market, ptb, p_yes, action, edge,
        gap_z, clob_yes, vol, trend_score, tte, phase, market_end_ts,
    ) -> None:
        """Fetch close price, compute outcome, compare to V2, send Telegram."""
        close_price = None
        for _ in range(SETTLEMENT_MAX_ATTEMPTS):
            close_price = await self._get_price_at(market_end_ts)
            if close_price is not None:
                break
            await asyncio.sleep(SETTLEMENT_CHECK_INTERVAL)

        if close_price is None:
            logger.warning(f"{self.label} settlement failed: close price unavailable", slug=market.slug)
            return

        actual_outcome = "YES" if close_price > ptb else "NO"
        quant_correct = None
        if action != "HOLD":
            quant_correct = 1 if action == actual_outcome else 0

        # Look up V2's decision for this market
        v2_action = v2_confidence = v2_correct = None
        try:
            cursor = self.db.conn.cursor()
            cursor.execute(
                """
                SELECT action, confidence FROM ai_analysis_log
                WHERE market_slug = ? AND bot_type LIKE 'v2_%'
                ORDER BY id DESC LIMIT 1
                """,
                (market.slug or market.id,),
            )
            row = cursor.fetchone()
            if row:
                v2_action = row[0]
                v2_confidence = row[1]
                v2_correct = 1 if v2_action == actual_outcome else 0
        except Exception as e:
            logger.warning(f"{self.label} V2 lookup failed", error=str(e))

        # Persist settlement
        self.db.settle_quant_shadow(
            row_id, actual_outcome, quant_correct, v2_action, v2_confidence, v2_correct
        )

        logger.info(
            f"{self.label} settled",
            actual=actual_outcome,
            close=f"{close_price:,.2f}",
            ptb=f"{ptb:,.2f}",
            quant_action=action,
            quant_correct=quant_correct,
            v2_action=v2_action,
            v2_correct=v2_correct,
        )

        await self._send_telegram_alert(
            market.slug or market.id, actual_outcome, close_price, ptb,
            p_yes, action, edge, quant_correct,
            v2_action, v2_confidence, v2_correct,
            gap_z, clob_yes, vol, trend_score, tte, phase,
        )

    async def _send_telegram_alert(
        self,
        slug, actual_outcome, close_price, ptb,
        p_yes, action, edge, quant_correct,
        v2_action, v2_conf, v2_correct,
        gap_z, clob_yes, vol, trend_score, tte, phase,
    ) -> None:
        if not self.telegram:
            return

        def result_icon(pred_action, correct):
            if pred_action is None or pred_action == "HOLD":
                return "⏭️"
            return "✅" if correct else "❌"

        actual_icon = "✅" if actual_outcome == "YES" else "❌"
        quant_icon = result_icon(action, quant_correct)
        v2_icon = result_icon(v2_action, v2_correct)

        edge_str = f"{edge:+.1%}" if action != "HOLD" else "N/A"
        v2_conf_str = f"{v2_conf:.0%}" if v2_conf else "N/A"
        asset_sym = self.asset.upper()

        msg = (
            f"🔬 Quant Shadow: {self.label} settled\n"
            f"{'─' * 32}\n"
            f"Outcome: {actual_icon} {actual_outcome} "
            f"({asset_sym} ${close_price:,.0f} vs PTB ${ptb:,.0f})\n\n"
            f"Quant: {action} (p={p_yes:.2f}, edge={edge_str}) {quant_icon}\n"
            f"V2:    {v2_action or 'N/A'} ({v2_conf_str}) {v2_icon}\n\n"
            f"Features:\n"
            f"  gap_z={gap_z:+.2f} | clob={clob_yes:.2f} | vol={vol:.1f}/min\n"
            f"  trend={trend_score:+.2f} | phase={phase} | tte={tte}s"
        )

        try:
            await self.telegram.send_message(msg)
        except Exception as e:
            logger.warning(f"{self.label} Telegram alert failed", error=str(e))


# ── Stats loop ───────────────────────────────────────────────────────────────

async def stats_loop(db: PerformanceDatabase) -> None:
    """Log quant shadow accuracy summary every 10 minutes."""
    while True:
        await asyncio.sleep(600)
        try:
            cursor = db.conn.cursor()
            cursor.execute("""
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN quant_action != 'HOLD' THEN 1 ELSE 0 END) as acted,
                    SUM(CASE WHEN quant_correct = 1 THEN 1 ELSE 0 END) as quant_wins,
                    SUM(CASE WHEN v2_correct = 1 THEN 1 ELSE 0 END) as v2_wins
                FROM quant_shadow_log
                WHERE actual_outcome IS NOT NULL
            """)
            r = cursor.fetchone()
            if r and r[0] > 0:
                logger.info(
                    "Quant shadow stats",
                    total_settled=r[0],
                    acted=r[1],
                    quant_wins=r[2] or 0,
                    v2_wins=r[3] or 0,
                    quant_win_rate=f"{(r[2] or 0) / max(r[1] or 1, 1):.1%}",
                    v2_win_rate=f"{(r[3] or 0) / r[0]:.1%}",
                )
        except Exception as e:
            logger.warning("Stats query failed", error=str(e))


# ── Entry point ──────────────────────────────────────────────────────────────

async def main() -> None:
    logger.info("Quant Shadow Service starting", markets=len(MARKETS))

    settings = Settings()
    client = PolymarketClient()
    db = PerformanceDatabase()

    telegram = None
    try:
        telegram = TelegramBot(settings)
        await telegram.send_message("🔬 Quant Shadow Service started — monitoring 8 markets")
    except Exception:
        logger.warning("Telegram not configured or unavailable")

    # Load logistic regression model (trains if model file not found)
    model, scaler = load_model()

    # Price services
    btc_service = BTCPriceService(settings)
    cg_key = os.getenv("COINGECKO_API_KEY", getattr(settings, "COINGECKO_API_KEY", ""))
    asset_services: dict[str, AssetPriceService] = {
        asset: AssetPriceService(coin_id=ASSET_COINGECKO_IDS[asset], api_key=cg_key)
        for asset in ("btc", "eth", "sol", "xrp")
    }

    # Trend analyzers (one per unique asset)
    trend_analyzers: dict[str, AssetTrendAnalyzer] = {
        asset: AssetTrendAnalyzer(
            coin_id=ASSET_COINGECKO_IDS[asset],
            coingecko_api_key=cg_key,
        )
        for asset in ("btc", "eth", "sol", "xrp")
    }

    # Start background price services
    for svc in asset_services.values():
        asyncio.create_task(svc.start())
    asyncio.create_task(btc_service.start())

    # Give services 5s to warm up before first monitor fires
    await asyncio.sleep(5)

    # Launch 8 market monitors + stats loop
    tasks = [
        asyncio.create_task(
            QuantShadowMonitor(
                asset=asset,
                timeframe=tf,
                client=client,
                btc_service=btc_service,
                asset_services=asset_services,
                trend_analyzers=trend_analyzers,
                db=db,
                telegram=telegram,
                model=model,
                scaler=scaler,
            ).run(),
            name=f"quant-{asset}-{tf}",
        )
        for asset, tf in MARKETS
    ]
    tasks.append(asyncio.create_task(stats_loop(db), name="stats"))

    # Graceful shutdown on SIGTERM / SIGINT
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, lambda: [t.cancel() for t in tasks])

    logger.info("Quant Shadow running", n_monitors=len(MARKETS))
    try:
        await asyncio.gather(*tasks, return_exceptions=True)
    finally:
        logger.info("Quant Shadow shutting down")
        if telegram:
            try:
                await telegram.send_message("🔬 Quant Shadow Service stopped")
            except Exception:
                pass


if __name__ == "__main__":
    asyncio.run(main())
