# tests/integration/test_contrarian_integration.py
"""Integration test for contrarian strategy end-to-end."""
import pytest
from polymarket.trading.contrarian import detect_contrarian_setup, get_movement_threshold
from polymarket.models import TechnicalIndicators


@pytest.mark.asyncio
async def test_contrarian_pipeline_oversold():
    """Test complete pipeline with OVERSOLD_REVERSAL signal."""
    # Step 1: Technical indicators show extreme oversold
    indicators = TechnicalIndicators(
        rsi=9.5,
        macd_value=-10.0,
        macd_signal=-8.0,
        macd_histogram=-2.0,
        ema_short=68300.0,
        ema_long=68400.0,
        sma_50=68500.0,
        volume_change=0.0,
        price_velocity=-50.0,
        trend="BEARISH"
    )

    # Step 2: Market shows strong DOWN consensus
    yes_odds = 0.28
    no_odds = 0.72

    # Step 3: Contrarian detection
    signal = detect_contrarian_setup(
        rsi=indicators.rsi,
        yes_odds=yes_odds,
        no_odds=no_odds
    )

    assert signal is not None
    assert signal.type == "OVERSOLD_REVERSAL"
    assert signal.suggested_direction == "UP"
    assert signal.rsi < 10
    assert signal.crowd_direction == "DOWN"
    assert signal.crowd_confidence > 0.70

    # Step 4: Movement threshold adjustment
    threshold = get_movement_threshold(signal)
    assert threshold == 50  # Reduced from 100


@pytest.mark.asyncio
async def test_contrarian_pipeline_overbought():
    """Test complete pipeline with OVERBOUGHT_REVERSAL signal."""
    # Step 1: Technical indicators show extreme overbought
    indicators = TechnicalIndicators(
        rsi=91.5,
        macd_value=10.0,
        macd_signal=8.0,
        macd_histogram=2.0,
        ema_short=68400.0,
        ema_long=68300.0,
        sma_50=68200.0,
        volume_change=0.0,
        price_velocity=50.0,
        trend="BULLISH"
    )

    # Step 2: Market shows strong UP consensus
    yes_odds = 0.75
    no_odds = 0.25

    # Step 3: Contrarian detection
    signal = detect_contrarian_setup(
        rsi=indicators.rsi,
        yes_odds=yes_odds,
        no_odds=no_odds
    )

    assert signal is not None
    assert signal.type == "OVERBOUGHT_REVERSAL"
    assert signal.suggested_direction == "DOWN"
    assert signal.rsi > 90
    assert signal.crowd_direction == "UP"
    assert signal.crowd_confidence > 0.70

    # Step 4: Movement threshold adjustment
    threshold = get_movement_threshold(signal)
    assert threshold == 50  # Reduced from 100


@pytest.mark.asyncio
async def test_contrarian_pipeline_no_signal():
    """Test pipeline with normal conditions (no contrarian signal)."""
    indicators = TechnicalIndicators(
        rsi=55.0,  # Not extreme
        macd_value=5.0,
        macd_signal=4.0,
        macd_histogram=1.0,
        ema_short=68400.0,
        ema_long=68300.0,
        sma_50=68200.0,
        volume_change=10.0,
        price_velocity=25.0,
        trend="BULLISH"
    )

    yes_odds = 0.60
    no_odds = 0.40

    signal = detect_contrarian_setup(
        rsi=indicators.rsi,
        yes_odds=yes_odds,
        no_odds=no_odds
    )

    assert signal is None

    threshold = get_movement_threshold(signal)
    assert threshold == 100  # Normal threshold


@pytest.mark.asyncio
async def test_contrarian_pipeline_extreme_rsi_weak_consensus():
    """Test that extreme RSI but weak consensus does NOT trigger contrarian."""
    # RSI is extreme but crowd is divided (no strong consensus)
    indicators = TechnicalIndicators(
        rsi=8.0,  # Extreme oversold
        macd_value=-10.0,
        macd_signal=-8.0,
        macd_histogram=-2.0,
        ema_short=68300.0,
        ema_long=68400.0,
        sma_50=68500.0,
        volume_change=0.0,
        price_velocity=-50.0,
        trend="BEARISH"
    )

    # Market is divided (no strong consensus)
    yes_odds = 0.55
    no_odds = 0.45

    signal = detect_contrarian_setup(
        rsi=indicators.rsi,
        yes_odds=yes_odds,
        no_odds=no_odds
    )

    # Should NOT trigger because crowd consensus is weak
    assert signal is None

    threshold = get_movement_threshold(signal)
    assert threshold == 100  # Normal threshold


@pytest.mark.asyncio
async def test_contrarian_pipeline_moderate_rsi_strong_consensus():
    """Test that moderate RSI but strong consensus does NOT trigger contrarian."""
    # RSI is moderate (not extreme)
    indicators = TechnicalIndicators(
        rsi=25.0,  # Moderately oversold but not extreme
        macd_value=-5.0,
        macd_signal=-4.0,
        macd_histogram=-1.0,
        ema_short=68300.0,
        ema_long=68350.0,
        sma_50=68400.0,
        volume_change=0.0,
        price_velocity=-20.0,
        trend="BEARISH"
    )

    # Strong DOWN consensus
    yes_odds = 0.25
    no_odds = 0.75

    signal = detect_contrarian_setup(
        rsi=indicators.rsi,
        yes_odds=yes_odds,
        no_odds=no_odds
    )

    # Should NOT trigger because RSI is not extreme enough
    assert signal is None

    threshold = get_movement_threshold(signal)
    assert threshold == 100  # Normal threshold


@pytest.mark.asyncio
async def test_contrarian_confidence_scaling():
    """Test that confidence scales with RSI extremity."""
    # Very extreme RSI
    signal_extreme = detect_contrarian_setup(
        rsi=5.0,  # Very extreme
        yes_odds=0.25,
        no_odds=0.75
    )

    # Moderately extreme RSI
    signal_moderate = detect_contrarian_setup(
        rsi=9.5,  # Moderately extreme
        yes_odds=0.25,
        no_odds=0.75
    )

    assert signal_extreme is not None
    assert signal_moderate is not None

    # More extreme RSI should have higher confidence
    assert signal_extreme.confidence > signal_moderate.confidence
