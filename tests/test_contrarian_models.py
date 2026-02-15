# tests/test_contrarian_models.py
from polymarket.models import ContrarianSignal

def test_contrarian_signal_oversold_creation():
    """Test OVERSOLD_REVERSAL signal creation."""
    signal = ContrarianSignal(
        type="OVERSOLD_REVERSAL",
        suggested_direction="UP",
        rsi=9.5,
        crowd_direction="DOWN",
        crowd_confidence=0.72,
        confidence=0.95,
        reasoning="Extreme oversold (RSI 9.5) + strong DOWN consensus (72%) = UP reversal likely"
    )

    assert signal.type == "OVERSOLD_REVERSAL"
    assert signal.suggested_direction == "UP"
    assert signal.rsi == 9.5
    assert signal.crowd_direction == "DOWN"
    assert signal.crowd_confidence == 0.72
    assert signal.confidence == 0.95
    assert "oversold" in signal.reasoning.lower()

def test_contrarian_signal_overbought_creation():
    """Test OVERBOUGHT_REVERSAL signal creation."""
    signal = ContrarianSignal(
        type="OVERBOUGHT_REVERSAL",
        suggested_direction="DOWN",
        rsi=92.0,
        crowd_direction="UP",
        crowd_confidence=0.70,
        confidence=0.92,
        reasoning="Extreme overbought (RSI 92.0) + strong UP consensus (70%) = DOWN reversal likely"
    )

    assert signal.type == "OVERBOUGHT_REVERSAL"
    assert signal.suggested_direction == "DOWN"
    assert signal.rsi == 92.0
    assert signal.crowd_direction == "UP"
