"""
Tests for signal conflict detector.
"""

import pytest
from polymarket.trading.conflict_detector import (
    SignalConflictDetector,
    ConflictSeverity,
    ConflictAnalysis
)


@pytest.fixture
def detector():
    """Create detector instance."""
    return SignalConflictDetector()


def test_no_conflicts(detector):
    """Test no conflicts when all signals align."""
    analysis = detector.analyze_conflicts(
        btc_direction="UP",
        technical_trend="BULLISH",
        sentiment_direction="BULLISH",
        regime_trend="TRENDING UP",
        timeframe_alignment="ALL_ALIGNED",
        market_signals_direction="bullish",
        market_signals_confidence=0.75
    )

    assert analysis.severity == ConflictSeverity.NONE
    assert analysis.confidence_penalty == 0.0
    assert analysis.should_hold is False
    assert len(analysis.conflicts_detected) == 0


def test_minor_conflict_one_signal(detector):
    """Test MINOR severity with 1 conflicting signal."""
    analysis = detector.analyze_conflicts(
        btc_direction="UP",
        technical_trend="BEARISH",  # Conflict
        sentiment_direction="BULLISH",
        regime_trend="TRENDING UP",
        timeframe_alignment="ALL_ALIGNED",
        market_signals_direction="bullish",
        market_signals_confidence=0.75
    )

    assert analysis.severity == ConflictSeverity.MINOR
    assert analysis.confidence_penalty == -0.10
    assert analysis.should_hold is False
    assert len(analysis.conflicts_detected) == 1
    assert "Technical (BEARISH) vs BTC actual (UP)" in analysis.conflicts_detected[0]


def test_moderate_conflict_two_signals(detector):
    """Test MODERATE severity with 2 conflicting signals."""
    analysis = detector.analyze_conflicts(
        btc_direction="UP",
        technical_trend="BEARISH",  # Conflict 1
        sentiment_direction="BEARISH",  # Conflict 2
        regime_trend="TRENDING UP",
        timeframe_alignment="ALL_ALIGNED",
        market_signals_direction="bullish",
        market_signals_confidence=0.75
    )

    assert analysis.severity == ConflictSeverity.MODERATE
    assert analysis.confidence_penalty == -0.20
    assert analysis.should_hold is False
    assert len(analysis.conflicts_detected) == 2


def test_severe_conflict_three_signals(detector):
    """Test SEVERE severity with 3+ conflicting signals."""
    analysis = detector.analyze_conflicts(
        btc_direction="UP",
        technical_trend="BEARISH",  # Conflict 1
        sentiment_direction="BEARISH",  # Conflict 2
        regime_trend="TRENDING DOWN",  # Conflict 3
        timeframe_alignment="ALL_ALIGNED",
        market_signals_direction="bullish",
        market_signals_confidence=0.75
    )

    assert analysis.severity == ConflictSeverity.SEVERE
    assert analysis.confidence_penalty == 0.0  # No penalty, just HOLD
    assert analysis.should_hold is True
    assert len(analysis.conflicts_detected) >= 3


def test_severe_conflict_timeframes_conflicting(detector):
    """Test SEVERE severity when timeframes CONFLICTING."""
    analysis = detector.analyze_conflicts(
        btc_direction="UP",
        technical_trend="BULLISH",
        sentiment_direction="BULLISH",
        regime_trend=None,
        timeframe_alignment="CONFLICTING",  # Trigger SEVERE
        market_signals_direction="bullish",
        market_signals_confidence=0.75
    )

    assert analysis.severity == ConflictSeverity.SEVERE
    assert analysis.should_hold is True
    assert any("Timeframes CONFLICTING" in conflict for conflict in analysis.conflicts_detected)


def test_market_signals_ignored_when_low_confidence(detector):
    """Test market signals ignored if confidence < 0.6."""
    analysis = detector.analyze_conflicts(
        btc_direction="UP",
        technical_trend="BULLISH",
        sentiment_direction="BULLISH",
        regime_trend=None,
        timeframe_alignment="ALL_ALIGNED",
        market_signals_direction="bearish",  # Conflicts
        market_signals_confidence=0.5  # Below 0.6 threshold
    )

    # Should have no conflicts because market signal ignored
    assert analysis.severity == ConflictSeverity.NONE
    assert len(analysis.conflicts_detected) == 0


def test_neutral_signals_dont_conflict(detector):
    """Test NEUTRAL signals don't create conflicts."""
    analysis = detector.analyze_conflicts(
        btc_direction="UP",
        technical_trend="NEUTRAL",
        sentiment_direction="NEUTRAL",
        regime_trend="RANGING",  # Maps to neither UP nor DOWN
        timeframe_alignment="MIXED",
        market_signals_direction="neutral",
        market_signals_confidence=0.75
    )

    # NEUTRAL/RANGING map to None, so no conflicts
    assert analysis.severity == ConflictSeverity.NONE
