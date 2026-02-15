"""
Test Signal Weighting System

Tests for the tiered signal weighting that prioritizes:
Tier 1 (Price Reality): BTC price movements
Tier 2 (Market Structure): Volume, orderbook
Tier 3 (External Signals): CoinGecko Pro signals
Tier 4 (Opinion): Sentiment
"""

import pytest
from decimal import Decimal
from polymarket.trading.ai_decision import AIDecisionService


def test_price_movement_overrides_sentiment():
    """
    CRITICAL: When BTC is moving UP (+2%), sentiment saying BEARISH should NOT
    result in a NO trade. Price reality (Tier 1) must override sentiment (Tier 4).

    Expected: Weighted confidence should be LOW (< 0.50) due to conflict.
    """
    # BTC is UP +2.0% (BULLISH price reality)
    btc_price_data = {
        'current_price': Decimal('100000'),
        'price_to_beat': Decimal('98000'),
        'price_change_pct': 2.04,  # +2.04% UP
        'direction': 'UP'
    }

    # Sentiment is BEARISH (opinion)
    sentiment_score = -0.60  # Bearish
    sentiment_confidence = 0.80  # High confidence in bearish sentiment

    # Calculate weighted confidence
    from polymarket.trading.ai_decision import AIDecisionService
    weighted_conf = AIDecisionService._calculate_weighted_confidence(
        price_signal_strength=0.85,  # Strong bullish (price is UP)
        price_direction='UP',
        volume_confirmation=True,
        orderbook_bias='neutral',
        coingecko_signal_strength=0.60,  # Moderate
        coingecko_direction='neutral',
        sentiment_score=sentiment_score,
        sentiment_confidence=sentiment_confidence,
        base_confidence=0.75
    )

    # Expected: Conflict between Tier 1 (price UP) and Tier 4 (sentiment BEARISH)
    # Should result in LOW weighted confidence
    assert weighted_conf < 0.50, (
        f"Price UP +2% with BEARISH sentiment should result in LOW confidence, "
        f"got {weighted_conf:.2f}"
    )


def test_aligned_signals_boost_confidence():
    """
    When all tiers align (price UP, volume high, CoinGecko bullish, sentiment bullish),
    confidence should be boosted significantly.

    Expected: Weighted confidence > base confidence.
    """
    # All signals BULLISH
    weighted_conf = AIDecisionService._calculate_weighted_confidence(
        price_signal_strength=0.90,  # Strong bullish price
        price_direction='UP',
        volume_confirmation=True,  # High volume
        orderbook_bias='bullish',  # Buying pressure
        coingecko_signal_strength=0.85,  # Strong bullish
        coingecko_direction='bullish',
        sentiment_score=0.70,  # Bullish sentiment
        sentiment_confidence=0.80,
        base_confidence=0.75
    )

    # Expected: Strong alignment should boost confidence
    assert weighted_conf > 0.75, (
        f"All signals aligned BULLISH should boost confidence, got {weighted_conf:.2f}"
    )
    assert weighted_conf >= 0.85, (
        f"Strong alignment should result in high confidence (>= 0.85), got {weighted_conf:.2f}"
    )


def test_volume_requirement_reduces_confidence():
    """
    Large price move ($200+) WITHOUT volume confirmation should reduce confidence.

    Expected: No volume = reduced weighted confidence.
    """
    # Large price move UP without volume
    weighted_conf_no_volume = AIDecisionService._calculate_weighted_confidence(
        price_signal_strength=0.85,  # Strong signal
        price_direction='UP',
        volume_confirmation=False,  # ❌ No volume
        orderbook_bias='neutral',
        coingecko_signal_strength=0.70,
        coingecko_direction='bullish',
        sentiment_score=0.60,
        sentiment_confidence=0.75,
        base_confidence=0.75
    )

    # Same scenario WITH volume
    weighted_conf_with_volume = AIDecisionService._calculate_weighted_confidence(
        price_signal_strength=0.85,
        price_direction='UP',
        volume_confirmation=True,  # ✓ Volume confirmed
        orderbook_bias='neutral',
        coingecko_signal_strength=0.70,
        coingecko_direction='bullish',
        sentiment_score=0.60,
        sentiment_confidence=0.75,
        base_confidence=0.75
    )

    # Expected: Volume confirmation should increase confidence
    assert weighted_conf_with_volume > weighted_conf_no_volume, (
        f"Volume confirmation should increase confidence: "
        f"no_volume={weighted_conf_no_volume:.2f}, with_volume={weighted_conf_with_volume:.2f}"
    )


def test_coingecko_signals_moderate_weight():
    """
    CoinGecko Pro signals (Tier 3) should have moderate impact.
    Should influence confidence but NOT override price reality.

    Expected: CoinGecko impact < price impact.
    """
    # Price bullish, CoinGecko bearish
    weighted_conf_conflict = AIDecisionService._calculate_weighted_confidence(
        price_signal_strength=0.85,  # Strong bullish
        price_direction='UP',
        volume_confirmation=True,
        orderbook_bias='neutral',
        coingecko_signal_strength=0.75,  # Strong bearish
        coingecko_direction='bearish',  # Conflict
        sentiment_score=0.50,
        sentiment_confidence=0.70,
        base_confidence=0.75
    )

    # Price bullish, CoinGecko bullish
    weighted_conf_aligned = AIDecisionService._calculate_weighted_confidence(
        price_signal_strength=0.85,  # Strong bullish
        price_direction='UP',
        volume_confirmation=True,
        orderbook_bias='neutral',
        coingecko_signal_strength=0.75,  # Strong bullish
        coingecko_direction='bullish',  # Aligned
        sentiment_score=0.50,
        sentiment_confidence=0.70,
        base_confidence=0.75
    )

    # Expected: Alignment should result in higher confidence
    assert weighted_conf_aligned > weighted_conf_conflict, (
        f"Aligned CoinGecko signals should boost confidence more than conflicting signals: "
        f"conflict={weighted_conf_conflict:.2f}, aligned={weighted_conf_aligned:.2f}"
    )

    # But the difference should be moderate (not extreme like price conflicts)
    diff = weighted_conf_aligned - weighted_conf_conflict
    assert 0.05 <= diff <= 0.25, (
        f"CoinGecko impact should be moderate (0.05-0.25), got {diff:.2f}"
    )
