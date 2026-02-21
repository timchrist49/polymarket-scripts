"""Tests for BTCTrendAnalyzer (multi-timeframe trend scoring)."""
import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from decimal import Decimal
from datetime import datetime


def make_candles(n: int, direction: str):
    """Helper: ascending (bull) or descending (bear) candles."""
    from AI_analysis_upgrade.trend_analyzer import OHLCCandle
    candles = []
    base = 95000.0
    step = 100.0 if direction == "bull" else -100.0
    for i in range(n):
        price = base + step * i
        candles.append(OHLCCandle(
            timestamp=datetime.now(),
            open=Decimal(str(price)),
            high=Decimal(str(price + 50)),
            low=Decimal(str(price - 50)),
            close=Decimal(str(price + 10)),
            volume=Decimal("10"),
        ))
    return candles


def test_compute_ema_ascending():
    """EMA on ascending prices should be below latest price."""
    from AI_analysis_upgrade.trend_analyzer import compute_ema
    prices = [float(90000 + i * 100) for i in range(25)]
    ema20 = compute_ema(prices, 20)
    assert ema20 < prices[-1]


def test_compute_rsi_all_gains():
    """RSI with all gains should be > 70."""
    from AI_analysis_upgrade.trend_analyzer import compute_rsi
    prices = [100.0 + i for i in range(20)]
    assert compute_rsi(prices, 14) > 70.0


def test_compute_rsi_all_losses():
    """RSI with all losses should be < 30."""
    from AI_analysis_upgrade.trend_analyzer import compute_rsi
    prices = [200.0 - i for i in range(20)]
    assert compute_rsi(prices, 14) < 30.0


def test_tf_score_bull():
    """Ascending candles → positive tf_score."""
    from AI_analysis_upgrade.trend_analyzer import tf_score
    candles = make_candles(55, "bull")
    assert tf_score(candles) > 0.0


def test_tf_score_bear():
    """Descending candles → negative tf_score."""
    from AI_analysis_upgrade.trend_analyzer import tf_score
    candles = make_candles(55, "bear")
    assert tf_score(candles) < 0.0


@pytest.mark.asyncio
async def test_trend_result_bull_market():
    """All-bull timeframes → trend_score > 0, p_yes_prior > 0.50."""
    from AI_analysis_upgrade.trend_analyzer import BTCTrendAnalyzer

    analyzer = BTCTrendAnalyzer()
    bull_candles = make_candles(60, "bull")

    # fetch_ohlcv is called 5 times (5m,15m,1H,4H,1D); all return bull candles
    with patch.object(analyzer, 'fetch_ohlcv', AsyncMock(return_value=bull_candles)), \
         patch.object(analyzer, 'fetch_fear_greed', AsyncMock(return_value=65)):
        result = await analyzer.compute()

    assert result.trend_score > 0.0
    assert result.p_yes_prior > 0.50
    assert 0.35 <= result.p_yes_prior <= 0.65  # always clamped


@pytest.mark.asyncio
async def test_trend_result_bear_market():
    """All-bear timeframes → trend_score < 0, p_yes_prior < 0.50."""
    from AI_analysis_upgrade.trend_analyzer import BTCTrendAnalyzer

    analyzer = BTCTrendAnalyzer()
    bear_candles = make_candles(60, "bear")

    with patch.object(analyzer, 'fetch_ohlcv', AsyncMock(return_value=bear_candles)), \
         patch.object(analyzer, 'fetch_fear_greed', AsyncMock(return_value=30)):
        result = await analyzer.compute()

    assert result.trend_score < 0.0
    assert result.p_yes_prior < 0.50
    assert result.p_yes_prior >= 0.35  # clamped floor


@pytest.mark.asyncio
async def test_trend_returns_neutral_on_fetch_failure():
    """If Kraken is unavailable for all TFs, return neutral (trend_score=0, prior=0.50)."""
    from AI_analysis_upgrade.trend_analyzer import BTCTrendAnalyzer

    analyzer = BTCTrendAnalyzer()
    # All 5 timeframes fail to fetch
    with patch.object(analyzer, 'fetch_ohlcv', AsyncMock(return_value=[])), \
         patch.object(analyzer, 'fetch_fear_greed', AsyncMock(return_value=50)):
        result = await analyzer.compute()

    assert result.trend_score == 0.0
    assert result.p_yes_prior == 0.50
