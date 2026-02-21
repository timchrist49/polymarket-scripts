"""Format and send Telegram comparison alerts: v1 vs v2 after market settlement."""
import aiohttp
import structlog
from typing import Optional

from AI_analysis_upgrade import config

logger = structlog.get_logger()


def format_comparison_alert(
    market_slug: str,
    v1_action: str,
    v1_confidence: float,
    v2_action: str,
    v2_confidence: float,
    actual_outcome: str,
    trend_score: float,
    fear_greed: int,
    gap_usd: float,
    gap_z: float,
    phase: str,
    v1_total_wins: int,
    v2_total_wins: int,
    total_markets: int,
    hard_rule: Optional[str] = None,
) -> str:
    """Format the comparison Telegram message."""
    v1_correct = v1_action == actual_outcome
    v2_correct = v2_action == actual_outcome

    v1_mark = "✅" if v1_correct else "❌"
    v2_mark = "✅" if v2_correct else "❌"

    trend_dir = "↑ Bull" if trend_score > 0.1 else ("↓ Bear" if trend_score < -0.1 else "→ Neutral")
    fg_label = "Greed" if fear_greed > 55 else ("Fear" if fear_greed < 45 else "Neutral")

    v1_pct = int(v1_total_wins / total_markets * 100) if total_markets else 0
    v2_pct = int(v2_total_wins / total_markets * 100) if total_markets else 0

    leader = "🏆 V2 WINNING" if v2_total_wins > v1_total_wins else (
        "🏆 V1 WINNING" if v1_total_wins > v2_total_wins else "🤝 TIED"
    )

    hard_rule_note = f"\n  Hard rule: {hard_rule}" if hard_rule else ""

    return (
        f"🔬 AI Analysis Comparison\n"
        f"Market: {market_slug}\n"
        f"Outcome: {'🟢 YES' if actual_outcome == 'YES' else '🔴 NO'}\n"
        f"\n"
        f"V1 (current): {v1_action} ({v1_confidence:.0%}) {v1_mark}\n"
        f"V2 (new):     {v2_action} ({v2_confidence:.0%}) {v2_mark}{hard_rule_note}\n"
        f"\n"
        f"Context: {trend_dir} | F&G {fear_greed}/100 ({fg_label}) | gap ${gap_usd:+.0f} | z={gap_z:.1f} | {phase}\n"
        f"\n"
        f"📊 Running Score ({total_markets} markets)\n"
        f"V1: {v1_total_wins}/{total_markets} ({v1_pct}%) | V2: {v2_total_wins}/{total_markets} ({v2_pct}%)\n"
        f"{leader}"
    )


async def send_comparison_alert(message: str, session: aiohttp.ClientSession) -> bool:
    """Send the comparison alert via Telegram Bot API."""
    if not config.TELEGRAM_BOT_TOKEN or not config.TELEGRAM_CHAT_ID:
        logger.warning("Telegram not configured, skipping alert")
        return False
    url = f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        async with session.post(url, json={
            "chat_id": config.TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": "HTML",
        }, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            resp.raise_for_status()
            return True
    except Exception as e:
        logger.error("Telegram send failed", error=str(e))
        return False
