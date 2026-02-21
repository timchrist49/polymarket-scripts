"""Multi-timeframe BTC trend analyzer.

Fetches OHLCV at 5 timeframes from Kraken (free, no auth) + Fear & Greed from
Alternative.me. Outputs trend_score ∈ [-1, +1] and p_yes_prior ∈ [0.35, 0.65].

Why Kraken? Already in production codebase, supports all intervals (5, 15, 60,
240, 1440 minutes), true OHLCV, free, no auth required.

Timeframes: 5m + 15m + 1H + 4H + 1D (complete picture, not just macro).
Longer timeframes weighted more for stability; 5m adds immediate direction signal.
"""
import asyncio
from dataclasses import dataclass, field
from decimal import Decimal
from datetime import datetime, timedelta
from typing import Optional
import aiohttp
import structlog

from AI_analysis_upgrade import config

logger = structlog.get_logger()

# Timeframe weights (interval_minutes → weight): longer TFs weighted more for stability
# 1D=30%, 4H=25%, 1H=20%, 15m=15%, 5m=10%
_TF_WEIGHTS = {1440: 0.30, 240: 0.25, 60: 0.20, 15: 0.15, 5: 0.10}

# Fear & Greed blend weight (blended into final trend_score)
_FEAR_GREED_WEIGHT = 0.15
_TF_WEIGHT = 1.0 - _FEAR_GREED_WEIGHT


@dataclass
class OHLCCandle:
    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal


@dataclass
class TrendResult:
    trend_score: float          # [-1, +1] pre-blend (timeframes only)
    p_yes_prior: float = 0.50   # [0.35, 0.65] regime-adaptive prior
    fear_greed: int = 50        # 0-100
    timeframe_scores: dict = field(default_factory=dict)

    def __post_init__(self):
        # Blend Fear & Greed: normalize 0-100 → -1 to +1
        fg_normalized = (self.fear_greed - 50) / 50.0  # -1 to +1
        blended = self.trend_score * _TF_WEIGHT + fg_normalized * _FEAR_GREED_WEIGHT
        raw_prior = 0.50 + 0.15 * blended
        self.p_yes_prior = max(0.35, min(0.65, raw_prior))


def compute_ema(prices: list, period: int) -> float:
    if not prices:
        return 0.0
    k = 2.0 / (period + 1)
    ema = prices[0]
    for p in prices[1:]:
        ema = p * k + ema * (1 - k)
    return ema


def compute_rsi(prices: list, period: int = 14) -> float:
    if len(prices) < period + 1:
        return 50.0
    gains, losses = [], []
    for i in range(1, len(prices)):
        diff = prices[i] - prices[i - 1]
        gains.append(max(diff, 0))
        losses.append(max(-diff, 0))
    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period
    if avg_loss == 0:
        return 100.0
    return 100.0 - (100.0 / (1 + avg_gain / avg_loss))


def tf_score(candles: list) -> float:
    """Score one timeframe: -1 (fully bearish) to +1 (fully bullish).

    Three votes:
      1. Price vs EMA(20): +1 if above, -1 if below
      2. RSI(14) vs 50:    +1 if above, -1 if below
      3. EMA(20) vs EMA(50): +1 if above, -1 if below (trend structure)
    """
    if len(candles) < 55:
        return 0.0
    closes = [float(c.close) for c in candles]
    current = closes[-1]
    ema20 = compute_ema(closes, 20)
    ema50 = compute_ema(closes, 50)
    rsi14 = compute_rsi(closes, 14)
    return (
        (1.0 if current > ema20 else -1.0) +
        (1.0 if rsi14 > 50 else -1.0) +
        (1.0 if ema20 > ema50 else -1.0)
    ) / 3.0


class BTCTrendAnalyzer:
    def __init__(self):
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    async def fetch_ohlcv(self, interval_minutes: int, num_candles: int = 60) -> list:
        """Fetch OHLCV from Kraken. interval_minutes: 5, 15, 60, 240, 1440."""
        session = await self._get_session()
        since = int((datetime.now() - timedelta(minutes=interval_minutes * num_candles)).timestamp())
        params = {"pair": "XBTUSD", "interval": str(interval_minutes), "since": str(since)}
        try:
            async with session.get(
                config.KRAKEN_OHLC_URL,
                params=params,
                timeout=aiohttp.ClientTimeout(total=15)
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
                if data.get("error"):
                    logger.warning("Kraken OHLCV error", error=data["error"], interval=interval_minutes)
                    return []
                raw = data["result"].get("XXBTZUSD", [])
                return [
                    OHLCCandle(
                        timestamp=datetime.fromtimestamp(c[0]),
                        open=Decimal(str(c[1])), high=Decimal(str(c[2])),
                        low=Decimal(str(c[3])), close=Decimal(str(c[4])),
                        volume=Decimal(str(c[6])),
                    )
                    for c in raw
                ]
        except Exception as e:
            logger.error("fetch_ohlcv failed", interval=interval_minutes, error=str(e))
            return []

    async def fetch_fear_greed(self) -> int:
        """Fetch Alternative.me Fear & Greed Index (0=extreme fear, 100=extreme greed)."""
        session = await self._get_session()
        try:
            async with session.get(
                config.FEAR_GREED_URL,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                resp.raise_for_status()
                data = await resp.json(content_type=None)
                return int(data["data"][0]["value"])
        except Exception as e:
            logger.warning("Fear&Greed fetch failed, using neutral 50", error=str(e))
            return 50

    async def compute(self) -> TrendResult:
        """Compute trend_score and p_yes_prior from 5m/15m/1H/4H/1D data + Fear&Greed."""
        # Fetch all 5 timeframes + Fear&Greed concurrently
        results = await asyncio.gather(
            self.fetch_ohlcv(interval_minutes=5, num_candles=60),
            self.fetch_ohlcv(interval_minutes=15, num_candles=60),
            self.fetch_ohlcv(interval_minutes=60, num_candles=60),
            self.fetch_ohlcv(interval_minutes=240, num_candles=60),
            self.fetch_ohlcv(interval_minutes=1440, num_candles=60),
            self.fetch_fear_greed(),
        )
        candles_5m, candles_15m, candles_1h, candles_4h, candles_1d, fear_greed = results

        tf_candles = {
            5: candles_5m,
            15: candles_15m,
            60: candles_1h,
            240: candles_4h,
            1440: candles_1d,
        }
        scores = {interval: tf_score(candles) for interval, candles in tf_candles.items()}

        # If all fetches failed, return neutral
        if all(not candles for candles in tf_candles.values()):
            return TrendResult(trend_score=0.0, fear_greed=fear_greed, timeframe_scores=scores)

        # Weighted average (only count timeframes that returned data)
        total_weight = sum(_TF_WEIGHTS[tf] for tf, c in tf_candles.items() if c)
        if total_weight == 0:
            trend = 0.0
        else:
            trend = sum(_TF_WEIGHTS[tf] * scores[tf] for tf, c in tf_candles.items() if c) / total_weight

        return TrendResult(trend_score=trend, fear_greed=fear_greed, timeframe_scores=scores)
