"""AI Analysis Upgrade Service — main entry point.

Run: python3 -m AI_analysis_upgrade.service

Polls production DB every 30s for:
  1. New v1 predictions → runs v2 analysis and saves to ai_analysis_v2
  2. Settled markets not yet compared → sends Telegram comparison alert
"""
import asyncio
import sqlite3
import structlog
import aiohttp

from AI_analysis_upgrade import config
from AI_analysis_upgrade.database import UpgradeDB
from AI_analysis_upgrade.trend_analyzer import BTCTrendAnalyzer
from AI_analysis_upgrade.coingecko import CoinGeckoSignals
from AI_analysis_upgrade.clob_reader import fetch_clob_odds
from AI_analysis_upgrade.ai_analysis_v2 import run_analysis
from AI_analysis_upgrade.telegram_alerts import format_comparison_alert, send_comparison_alert

logger = structlog.get_logger()


async def analysis_loop(
    db: UpgradeDB,
    trend_analyzer: BTCTrendAnalyzer,
    cg: CoinGeckoSignals,
    session: aiohttp.ClientSession,
):
    """Find new v1 predictions and run v2 analysis on them."""
    rows = db.get_unanalyzed_v1_rows()
    if not rows:
        return

    # Fetch trend + CoinGecko signals concurrently (shared across all markets in this batch)
    trend, cg_signals = await asyncio.gather(
        trend_analyzer.compute(),
        cg.fetch_all(),
    )
    logger.info(
        "Signals fetched",
        trend=f"{trend.trend_score:+.2f}",
        prior=f"{trend.p_yes_prior:.2f}",
        fg=trend.fear_greed,
        cpi=f"{cg_signals['cpi']:+.4f}" if cg_signals.get("cpi") is not None else "n/a",
        vol_spike=f"{cg_signals.get('volume_spike', 1.0):.2f}",
    )

    cpi = cg_signals.get("cpi")
    volume_spike = cg_signals.get("volume_spike")
    funding_rate = cg_signals.get("funding_rate")

    for row in rows:
        market_slug = row["market_slug"]
        logger.info("Running v2 analysis", market=market_slug, v1=row.get("action"))

        # Fetch real-time CLOB odds
        clob = await fetch_clob_odds(market_slug, session)
        clob_yes = clob["yes"]

        btc_price = float(row.get("btc_price") or 0)
        ptb = float(row.get("ptb") or btc_price)

        # gap_usd = btc_price - ptb: positive = BTC ABOVE PTB = YES territory
        gap_usd = btc_price - ptb

        # Estimated time remaining: conservative 3 minutes
        # (v2 fires as soon as v1 fires; both happen ~3-5 min before close)
        time_remaining_seconds = 180

        # Realized vol: default $15/min (typical for 5-15 min BTC windows)
        realized_vol_per_min = 15.0

        market_duration = 900 if row.get("bot_type") == "15m" else 300

        prediction = await run_analysis(
            market_slug=market_slug,
            gap_usd=gap_usd,
            clob_yes=clob_yes,
            time_remaining_seconds=time_remaining_seconds,
            btc_price=btc_price,
            ptb=ptb,
            realized_vol_per_min=realized_vol_per_min,
            trend=trend,
            market_duration_seconds=market_duration,
            cpi=cpi,
            volume_spike=volume_spike,
            funding_rate=funding_rate,
        )

        db.save_v2_prediction(
            v1_id=row["id"],
            market_slug=market_slug,
            bot_type=row.get("bot_type", "unknown"),
            v1_action=row.get("action", ""),
            v1_confidence=float(row.get("confidence") or 0),
            v2_action=prediction.action,
            v2_confidence=prediction.confidence,
            v2_reasoning=prediction.reasoning,
            trend_score=trend.trend_score,
            p_yes_prior=trend.p_yes_prior,
            fear_greed=trend.fear_greed,
            gap_usd=gap_usd,
            gap_z=prediction.gap_z,
            phase=prediction.phase,
            clob_yes=clob_yes,
            cpi=cpi,
            volume_spike=volume_spike,
            funding_rate=funding_rate,
        )
        logger.info(
            "V2 prediction saved",
            market=market_slug,
            v1=row.get("action"),
            v2=prediction.action,
            confidence=f"{prediction.confidence:.2f}",
            rule=prediction.hard_rule or "AI",
        )


async def alert_loop(db: UpgradeDB, session: aiohttp.ClientSession):
    """Check production DB for newly settled markets and sync outcomes + send alerts."""
    # Sync outcomes: check v1 rows that now have actual_outcome set
    conn = sqlite3.connect(db.db_path)
    conn.row_factory = sqlite3.Row
    newly_settled = conn.execute("""
        SELECT l.id, l.actual_outcome
        FROM ai_analysis_log l
        JOIN ai_analysis_v2 v ON v.v1_id = l.id
        WHERE l.actual_outcome IS NOT NULL
          AND v.actual_outcome IS NULL
    """).fetchall()
    conn.close()

    for row in newly_settled:
        db.mark_outcome(v1_id=row["id"], actual_outcome=row["actual_outcome"])

    # Send comparison alerts for settled rows
    unsent = db.get_unsent_settled_rows()
    if not unsent:
        return

    score = db.get_running_score()

    for row in unsent:
        msg = format_comparison_alert(
            market_slug=row["market_slug"],
            v1_action=row["v1_action"] or "?",
            v1_confidence=float(row["v1_confidence"] or 0),
            v2_action=row["v2_action"] or "?",
            v2_confidence=float(row["v2_confidence"] or 0),
            actual_outcome=row["actual_outcome"],
            trend_score=float(row["trend_score"] or 0),
            fear_greed=int(row["fear_greed"] or 50),
            gap_usd=float(row["gap_usd"] or 0),
            gap_z=float(row["gap_z"] or 0),
            phase=row["phase"] or "unknown",
            v1_total_wins=score["v1_wins"] or 0,
            v2_total_wins=score["v2_wins"] or 0,
            total_markets=score["total"] or 0,
            hard_rule=None,
        )
        sent = await send_comparison_alert(msg, session)
        if sent:
            logger.info("Comparison alert sent", market=row["market_slug"])
        # Mark sent even when Telegram isn't configured — prevents re-alerting every cycle
        db.mark_comparison_sent(row["v1_id"])


async def main():
    logger.info("AI Analysis Upgrade Service starting", db=config.PRODUCTION_DB_PATH)
    db = UpgradeDB()
    trend_analyzer = BTCTrendAnalyzer()
    cg = CoinGeckoSignals()

    try:
        async with aiohttp.ClientSession() as session:
            while True:
                try:
                    await analysis_loop(db, trend_analyzer, cg, session)
                except Exception as e:
                    logger.error("analysis_loop error", error=str(e))

                try:
                    await alert_loop(db, session)
                except Exception as e:
                    logger.error("alert_loop error", error=str(e))

                await asyncio.sleep(config.POLL_INTERVAL_SECONDS)
    finally:
        await trend_analyzer.close()
        await cg.close()


if __name__ == "__main__":
    asyncio.run(main())
