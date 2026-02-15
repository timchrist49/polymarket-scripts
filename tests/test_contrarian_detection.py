# tests/test_contrarian_detection.py
from polymarket.trading.contrarian import detect_contrarian_setup

def test_oversold_reversal_detected():
    """RSI < 10 + DOWN odds > 65% = OVERSOLD_REVERSAL."""
    signal = detect_contrarian_setup(
        rsi=9.5,
        yes_odds=0.28,  # UP odds
        no_odds=0.72    # DOWN odds
    )

    assert signal is not None
    assert signal.type == "OVERSOLD_REVERSAL"
    assert signal.suggested_direction == "UP"
    assert signal.rsi == 9.5
    assert signal.crowd_direction == "DOWN"
    assert signal.crowd_confidence == 0.72
    assert signal.confidence >= 0.90  # High confidence
    assert "oversold" in signal.reasoning.lower()

def test_overbought_reversal_detected():
    """RSI > 90 + UP odds > 65% = OVERBOUGHT_REVERSAL."""
    signal = detect_contrarian_setup(
        rsi=92.0,
        yes_odds=0.70,  # UP odds
        no_odds=0.30    # DOWN odds
    )

    assert signal is not None
    assert signal.type == "OVERBOUGHT_REVERSAL"
    assert signal.suggested_direction == "DOWN"
    assert signal.rsi == 92.0
    assert signal.crowd_direction == "UP"
    assert signal.crowd_confidence == 0.70
    assert signal.confidence >= 0.90

def test_no_contrarian_signal_rsi_not_extreme():
    """RSI 50 = no signal (not extreme)."""
    signal = detect_contrarian_setup(
        rsi=50.0,
        yes_odds=0.30,
        no_odds=0.70
    )

    assert signal is None

def test_no_contrarian_signal_odds_not_extreme():
    """RSI 5 but odds 50/50 = no signal (crowd not consensus)."""
    signal = detect_contrarian_setup(
        rsi=5.0,
        yes_odds=0.50,
        no_odds=0.50
    )

    assert signal is None

def test_oversold_edge_case_rsi_10():
    """RSI exactly 10 should NOT trigger (< 10 required)."""
    signal = detect_contrarian_setup(
        rsi=10.0,
        yes_odds=0.30,
        no_odds=0.70
    )

    assert signal is None

def test_oversold_edge_case_odds_65():
    """Odds exactly 65% should NOT trigger (> 65% required)."""
    signal = detect_contrarian_setup(
        rsi=9.0,
        yes_odds=0.35,
        no_odds=0.65
    )

    assert signal is None

def test_confidence_increases_with_extreme_rsi():
    """Lower RSI = higher confidence."""
    signal_9 = detect_contrarian_setup(rsi=9.0, yes_odds=0.25, no_odds=0.75)
    signal_5 = detect_contrarian_setup(rsi=5.0, yes_odds=0.25, no_odds=0.75)

    assert signal_5.confidence > signal_9.confidence
