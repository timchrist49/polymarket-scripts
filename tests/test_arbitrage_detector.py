"""Tests for arbitrage detector."""
import pytest
from pytest import approx
from polymarket.trading.arbitrage_detector import ArbitrageDetector
from polymarket.models import ArbitrageOpportunity


def test_detect_yes_arbitrage():
    """Test detecting YES arbitrage opportunity (13% edge -> MEDIUM urgency)."""
    detector = ArbitrageDetector()

    # Actual probability is 68%, but market prices YES at 55%
    # Edge = 0.68 - 0.55 = 0.13 (13%)
    opp = detector.detect_arbitrage(
        actual_probability=0.68,
        market_yes_odds=0.55,
        market_no_odds=0.45,
        market_id="test-market-1"
    )

    assert isinstance(opp, ArbitrageOpportunity)
    assert opp.market_id == "test-market-1"
    assert opp.actual_probability == 0.68
    assert opp.polymarket_yes_odds == 0.55
    assert opp.polymarket_no_odds == 0.45
    assert opp.edge_percentage == approx(0.13, abs=1e-9)
    assert opp.recommended_action == "BUY_YES"
    assert opp.confidence_boost == approx(0.20, abs=1e-9)  # Min(0.13 * 2, 0.20) = 0.20 (capped)
    assert opp.urgency == "MEDIUM"  # 10% <= 13% < 15%
    assert opp.expected_profit_pct == approx(0.8182, abs=0.01)  # (1.0 - 0.55) / 0.55


def test_detect_no_arbitrage():
    """Test detecting NO arbitrage opportunity (5% edge -> LOW urgency)."""
    detector = ArbitrageDetector()

    # Actual probability is 35%, implied NO probability is 65%
    # Market prices NO at 60%, edge = 0.65 - 0.60 = 0.05 (5%)
    opp = detector.detect_arbitrage(
        actual_probability=0.35,
        market_yes_odds=0.40,
        market_no_odds=0.60,
        market_id="test-market-2"
    )

    assert opp.recommended_action == "BUY_NO"
    assert opp.edge_percentage == approx(0.05, abs=1e-9)
    assert opp.confidence_boost == approx(0.10, abs=1e-9)  # 0.05 * 2 = 0.10
    assert opp.urgency == "LOW"  # 5% < 10%
    assert opp.expected_profit_pct == approx(0.6667, abs=0.01)  # (1.0 - 0.60) / 0.60


def test_no_arbitrage_opportunity():
    """Test no edge -> HOLD action."""
    detector = ArbitrageDetector()

    # Actual probability is 52%, market prices YES at 50%
    # Edge = 0.52 - 0.50 = 0.02 (2%, below 5% MIN_EDGE)
    opp = detector.detect_arbitrage(
        actual_probability=0.52,
        market_yes_odds=0.50,
        market_no_odds=0.50,
        market_id="test-market-3"
    )

    assert opp.recommended_action == "HOLD"
    assert opp.edge_percentage < 0.05
    assert opp.confidence_boost == 0.0
    assert opp.urgency == "LOW"
    assert opp.expected_profit_pct == 0.0


def test_high_edge_urgency():
    """Test 15%+ edge -> HIGH urgency."""
    detector = ArbitrageDetector()

    # Actual probability is 75%, market prices YES at 55%
    # Edge = 0.75 - 0.55 = 0.20 (20%, >= 15%)
    opp = detector.detect_arbitrage(
        actual_probability=0.75,
        market_yes_odds=0.55,
        market_no_odds=0.45,
        market_id="test-market-4"
    )

    assert opp.edge_percentage == approx(0.20, abs=1e-9)
    assert opp.urgency == "HIGH"  # 20% >= 15%


def test_confidence_boost_scales_with_edge():
    """Test larger edge = larger boost (up to cap)."""
    detector = ArbitrageDetector()

    # Test 6% edge (avoiding floating point precision issues near MIN_EDGE)
    opp_6pct = detector.detect_arbitrage(
        actual_probability=0.62,
        market_yes_odds=0.56,
        market_no_odds=0.44,
        market_id="test-6pct"
    )

    # Test 10% edge
    opp_10pct = detector.detect_arbitrage(
        actual_probability=0.65,
        market_yes_odds=0.55,
        market_no_odds=0.45,
        market_id="test-10pct"
    )

    assert opp_6pct.confidence_boost == approx(0.12, abs=1e-9)  # 0.06 * 2 = 0.12
    assert opp_10pct.confidence_boost == approx(0.20, abs=1e-9)  # Min(0.10 * 2, 0.20) = 0.20
    assert opp_10pct.confidence_boost > opp_6pct.confidence_boost


def test_confidence_boost_capped_at_20pct():
    """Test max boost is 0.20."""
    detector = ArbitrageDetector()

    # Extreme edge: 30%
    opp = detector.detect_arbitrage(
        actual_probability=0.85,
        market_yes_odds=0.55,
        market_no_odds=0.45,
        market_id="test-extreme"
    )

    # Edge = 0.30, boost would be 0.60 without cap
    # But should be capped at 0.20
    assert opp.edge_percentage == approx(0.30, abs=1e-9)
    assert opp.confidence_boost == approx(0.20, abs=1e-9)  # Capped
