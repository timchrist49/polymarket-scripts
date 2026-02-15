# tests/test_contrarian_sentiment.py
"""Tests for contrarian signal integration into sentiment aggregation."""

from polymarket.models import ContrarianSignal
from polymarket.trading.signal_aggregator import SignalAggregator


def test_contrarian_signal_added_to_aggregation():
    """Contrarian signal should be included in aggregated sentiment."""
    aggregator = SignalAggregator()

    contrarian_signal = ContrarianSignal(
        type="OVERSOLD_REVERSAL",
        suggested_direction="UP",
        rsi=9.5,
        crowd_direction="DOWN",
        crowd_confidence=0.72,
        confidence=0.95,
        reasoning="Test"
    )

    # Convert contrarian to regular signal format
    signal = aggregator.contrarian_to_signal(contrarian_signal)

    assert signal.name == "contrarian_rsi"
    assert signal.score == +1.0  # UP direction
    assert signal.confidence == 0.95
    assert signal.weight == 2.0  # High weight for contrarian


def test_contrarian_down_signal():
    """OVERBOUGHT_REVERSAL should create negative score (DOWN)."""
    aggregator = SignalAggregator()

    contrarian_signal = ContrarianSignal(
        type="OVERBOUGHT_REVERSAL",
        suggested_direction="DOWN",
        rsi=92.0,
        crowd_direction="UP",
        crowd_confidence=0.70,
        confidence=0.92,
        reasoning="Test"
    )

    signal = aggregator.contrarian_to_signal(contrarian_signal)

    assert signal.score == -1.0  # DOWN direction
    assert signal.confidence == 0.92
