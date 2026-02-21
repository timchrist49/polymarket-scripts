"""Enhanced AI analysis v2 — phase-weighted ensemble.

Architecture:
  1. Call OpenAI with regime-aware prompt
  2. Return action + confidence + reasoning

gap_usd convention: gap_usd = btc_price - ptb
  Positive = BTC ABOVE PTB = YES territory
  Negative = BTC BELOW PTB = NO territory
"""
import math
from dataclasses import dataclass
from typing import Optional

from openai import AsyncOpenAI
import structlog

from AI_analysis_upgrade import config
from AI_analysis_upgrade.trend_analyzer import TrendResult

logger = structlog.get_logger()

# Phase split: market is "late" when ≤33% of duration remains
LATE_PHASE_FRACTION = 0.33


@dataclass
class V2Prediction:
    action: str           # "YES" or "NO"
    confidence: float     # 0.0-1.0
    reasoning: str
    hard_rule: Optional[str] = None   # set if a hard rule fired
    gap_z: float = 0.0
    phase: str = "unknown"


def compute_gap_z(gap_usd: float, realized_vol_per_min: float, time_remaining_minutes: float) -> float:
    """Gap z-score: how many standard deviations is BTC from the target price?

    z = gap_usd / (realized_vol_per_min × √time_remaining_min)

    Positive z: BTC above PTB. High positive z with little time → YES certainty.
    Negative z: BTC below PTB. High negative z with little time → NO certainty.
    """
    if realized_vol_per_min <= 0 or time_remaining_minutes <= 0:
        return 0.0
    vol_over_window = realized_vol_per_min * math.sqrt(time_remaining_minutes)
    return gap_usd / vol_over_window


def get_phase(time_remaining_seconds: int, market_duration_seconds: int) -> str:
    """Return 'early' or 'late' based on how much of the market has elapsed."""
    if market_duration_seconds <= 0:
        return "late"
    fraction_remaining = time_remaining_seconds / market_duration_seconds
    return "late" if fraction_remaining <= LATE_PHASE_FRACTION else "early"



def build_prompt(
    market_slug: str,
    gap_usd: float,
    gap_z: float,
    clob_yes: float,
    time_remaining_seconds: int,
    btc_price: float,
    ptb: float,
    realized_vol_per_min: float,
    trend: TrendResult,
    phase: str,
    market_duration_seconds: int,
    cpi: Optional[float] = None,
    volume_spike: Optional[float] = None,
    funding_rate: Optional[float] = None,
) -> str:
    """Build the AI analysis prompt with phase-weighted ensemble guidance."""

    # Phase-dependent weight description for AI context
    if phase == "late":
        weight_desc = "LATE PHASE: gap signal = 60% weight, CLOB = 20%, trend prior = 10%, OFI = 10%"
    else:
        weight_desc = "EARLY PHASE: trend prior = 35%, CLOB = 30%, OFI = 20%, gap signal = 15%"

    trend_direction = "BULLISH" if trend.trend_score > 0.1 else ("BEARISH" if trend.trend_score < -0.1 else "NEUTRAL")
    fg_label = "Greed" if trend.fear_greed > 55 else ("Fear" if trend.fear_greed < 45 else "Neutral")

    tf_scores = trend.timeframe_scores
    tf_5m = tf_scores.get(5, 0.0)
    tf_15m = tf_scores.get(15, 0.0)
    tf_1h = tf_scores.get(60, 0.0)
    tf_4h = tf_scores.get(240, 0.0)
    tf_1d = tf_scores.get(1440, 0.0)

    # CoinGecko signals section (only shown if available)
    cg_lines = []
    if cpi is not None:
        cg_lines.append(f"  Coinbase Premium Index: {cpi:+.4f}%  [>0.07% = US institutional buying]")
    if volume_spike is not None:
        cg_lines.append(f"  Volume spike ratio: {volume_spike:.2f}x  [>1.8 = abnormal activity]")
    if funding_rate is not None:
        cg_lines.append(f"  Funding rate: {funding_rate:+.4f}%  [extreme — crowded positioning]")
    coingecko_section = "\nCOINGECKO SIGNALS:\n" + "\n".join(cg_lines) if cg_lines else ""

    return f"""You are analyzing a Polymarket BTC prediction market. Output ONLY: YES or NO, then a confidence (0.0-1.0), then one sentence of reasoning.

MARKET: {market_slug}
TIME REMAINING: {time_remaining_seconds}s
PHASE: {phase.upper()} — {weight_desc}

BTC CONTEXT:
  Current price: ${btc_price:,.2f}
  Price-to-beat: ${ptb:,.2f}
  Gap (btc - ptb): ${gap_usd:+.2f}  [positive = BTC ABOVE target (YES territory), negative = BTC BELOW target (NO territory)]
  Gap z-score: {gap_z:.2f}  [positive = above target, negative = below target]
  Realized vol/min: ${realized_vol_per_min:.2f}

MACRO TREND (multi-timeframe):
  Trend score: {trend.trend_score:+.2f}  [-1=strongly bearish, +1=strongly bullish]
  Direction: {trend_direction}
  5m score: {tf_5m:+.2f} | 15m: {tf_15m:+.2f} | 1H: {tf_1h:+.2f} | 4H: {tf_4h:+.2f} | 1D: {tf_1d:+.2f}
  Fear & Greed: {trend.fear_greed}/100 ({fg_label})
  Regime-adaptive prior P(YES): {trend.p_yes_prior:.2f}{coingecko_section}

MARKET SIGNAL:
  CLOB YES odds: {clob_yes:.2f}

TASK: Using the weighted ensemble for the {phase.upper()} phase, predict whether BTC will be
ABOVE the price-to-beat at market close.

YES = BTC will be ABOVE ${ptb:,.2f} at close
NO  = BTC will be AT OR BELOW ${ptb:,.2f} at close

Respond in this exact format:
ACTION: YES or NO
CONFIDENCE: 0.XX
REASONING: One sentence.
"""


async def run_analysis(
    market_slug: str,
    gap_usd: float,
    clob_yes: float,
    time_remaining_seconds: int,
    btc_price: float,
    ptb: float,
    realized_vol_per_min: float,
    trend: TrendResult,
    market_duration_seconds: int = 900,
    cpi: Optional[float] = None,
    volume_spike: Optional[float] = None,
    funding_rate: Optional[float] = None,
) -> V2Prediction:
    """Run v2 analysis for one market. Returns a prediction."""

    phase = get_phase(time_remaining_seconds, market_duration_seconds)
    gap_z = compute_gap_z(gap_usd, realized_vol_per_min, time_remaining_seconds / 60.0)

    prompt = build_prompt(
        market_slug=market_slug,
        gap_usd=gap_usd,
        gap_z=gap_z,
        clob_yes=clob_yes,
        time_remaining_seconds=time_remaining_seconds,
        btc_price=btc_price,
        ptb=ptb,
        realized_vol_per_min=realized_vol_per_min,
        trend=trend,
        phase=phase,
        market_duration_seconds=market_duration_seconds,
        cpi=cpi,
        volume_spike=volume_spike,
        funding_rate=funding_rate,
    )

    try:
        client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)
        response = await client.chat.completions.create(
            model=config.OPENAI_MODEL,
            max_completion_tokens=128,
            reasoning_effort=config.OPENAI_REASONING_EFFORT,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.choices[0].message.content.strip()
        return _parse_response(text, gap_z, phase)
    except Exception as e:
        logger.error("OpenAI v2 analysis failed", market=market_slug, error=str(e))
        # Fallback: use CLOB as tiebreaker
        action = "YES" if clob_yes > 0.55 else "NO"
        return V2Prediction(
            action=action,
            confidence=clob_yes if action == "YES" else (1.0 - clob_yes),
            reasoning=f"AI call failed; fallback to CLOB ({clob_yes:.2f})",
            gap_z=gap_z,
            phase=phase,
        )


def _parse_response(text: str, gap_z: float, phase: str) -> V2Prediction:
    """Parse Claude's response into V2Prediction."""
    action = "NO"
    confidence = 0.50
    reasoning = text

    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("ACTION:"):
            val = line.split(":", 1)[1].strip().upper()
            if val in ("YES", "NO"):
                action = val
        elif line.startswith("CONFIDENCE:"):
            try:
                confidence = float(line.split(":", 1)[1].strip())
            except ValueError:
                pass
        elif line.startswith("REASONING:"):
            reasoning = line.split(":", 1)[1].strip()

    return V2Prediction(
        action=action,
        confidence=min(max(confidence, 0.0), 1.0),
        reasoning=reasoning,
        gap_z=gap_z,
        phase=phase,
    )
