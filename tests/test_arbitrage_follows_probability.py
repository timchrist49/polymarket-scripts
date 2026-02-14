"""Test arbitrage detector follows probability direction."""
import pytest
from polymarket.trading.arbitrage_detector import ArbitrageDetector


def test_follows_yes_probability_with_positive_edge():
    """When probability >50% and YES edge positive, should BUY YES."""
    detector = ArbitrageDetector()

    opp = detector.detect_arbitrage(
        actual_probability=0.65,  # Predicts YES
        market_yes_odds=0.55,     # YES edge = +10%
        market_no_odds=0.45,
        market_id="test-1"
    )

    assert opp.recommended_action == "BUY_YES"
    assert opp.edge_percentage == pytest.approx(0.10, rel=0.01)


def test_holds_when_yes_probability_but_negative_edge():
    """When probability >50% but YES edge negative, should HOLD (not bet NO)."""
    detector = ArbitrageDetector()

    opp = detector.detect_arbitrage(
        actual_probability=0.65,  # Predicts YES
        market_yes_odds=0.75,     # YES edge = -10% (market more bullish)
        market_no_odds=0.25,      # NO edge = +10% (but contradicts probability!)
        market_id="test-2"
    )

    # Should HOLD, not bet NO against probability
    assert opp.recommended_action == "HOLD"


def test_follows_no_probability_with_positive_edge():
    """When probability <50% and NO edge positive, should BUY NO."""
    detector = ArbitrageDetector()

    opp = detector.detect_arbitrage(
        actual_probability=0.35,  # Predicts NO
        market_yes_odds=0.50,
        market_no_odds=0.50,      # NO edge = +15%
        market_id="test-3"
    )

    assert opp.recommended_action == "BUY_NO"
    assert opp.edge_percentage == pytest.approx(0.15, rel=0.01)


def test_real_trade_247_scenario():
    """
    Reproduce Trade 247 bug: Bot calculated 56.8% YES but bet NO.
    Should now HOLD because YES edge is negative.
    """
    detector = ArbitrageDetector()

    opp = detector.detect_arbitrage(
        actual_probability=0.568,  # Bot predicts YES
        market_yes_odds=0.62,      # YES edge = -5.2%
        market_no_odds=0.39,       # NO edge = +4.2%
        market_id="trade-247"
    )

    # Should HOLD, not bet NO against probability
    assert opp.recommended_action == "HOLD"
