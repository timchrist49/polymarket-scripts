# tests/test_contrarian_threshold.py
from polymarket.models import ContrarianSignal
from polymarket.trading.contrarian import get_movement_threshold

def test_movement_threshold_reduced_with_contrarian():
    """Movement threshold should be $50 when contrarian signal present."""
    contrarian_signal = ContrarianSignal(
        type="OVERSOLD_REVERSAL",
        suggested_direction="UP",
        rsi=9.5,
        crowd_direction="DOWN",
        crowd_confidence=0.72,
        confidence=0.95,
        reasoning="Test"
    )

    threshold = get_movement_threshold(contrarian_signal)
    assert threshold == 50

def test_movement_threshold_normal_without_contrarian():
    """Movement threshold should be $100 without contrarian signal."""
    threshold = get_movement_threshold(None)
    assert threshold == 100
