"""Tests for arbitrage-related data models."""
import pytest
from polymarket.models import ArbitrageOpportunity, LimitOrderStrategy

def test_arbitrage_opportunity_creation():
    """Test creating ArbitrageOpportunity with valid data."""
    opp = ArbitrageOpportunity(
        market_id="test-market",
        actual_probability=0.68,
        polymarket_yes_odds=0.55,
        polymarket_no_odds=0.45,
        edge_percentage=0.13,
        recommended_action="BUY_YES",
        confidence_boost=0.20,
        urgency="MEDIUM",
        expected_profit_pct=0.18
    )

    assert opp.market_id == "test-market"
    assert opp.actual_probability == 0.68
    assert opp.edge_percentage == 0.13
    assert opp.recommended_action == "BUY_YES"
    assert opp.urgency == "MEDIUM"

def test_arbitrage_opportunity_validation():
    """Test validation of ArbitrageOpportunity fields."""
    # Test invalid action
    with pytest.raises(ValueError, match="Invalid action"):
        ArbitrageOpportunity(
            market_id="test",
            actual_probability=0.68,
            polymarket_yes_odds=0.55,
            polymarket_no_odds=0.45,
            edge_percentage=0.13,
            recommended_action="INVALID",  # Should fail
            confidence_boost=0.20,
            urgency="MEDIUM",
            expected_profit_pct=0.18
        )

    # Test invalid urgency
    with pytest.raises(ValueError, match="Invalid urgency"):
        ArbitrageOpportunity(
            market_id="test",
            actual_probability=0.68,
            polymarket_yes_odds=0.55,
            polymarket_no_odds=0.45,
            edge_percentage=0.13,
            recommended_action="BUY_YES",
            confidence_boost=0.20,
            urgency="INVALID",  # Should fail
            expected_profit_pct=0.18
        )

def test_arbitrage_opportunity_probability_validation():
    """Test probability bounds validation."""
    # Test probability > 1.0
    with pytest.raises(ValueError, match="Invalid probability"):
        ArbitrageOpportunity(
            market_id="test",
            actual_probability=1.5,  # Invalid
            polymarket_yes_odds=0.55,
            polymarket_no_odds=0.45,
            edge_percentage=0.13,
            recommended_action="BUY_YES",
            confidence_boost=0.20,
            urgency="MEDIUM",
            expected_profit_pct=0.18
        )

    # Test probability < 0.0
    with pytest.raises(ValueError, match="Invalid probability"):
        ArbitrageOpportunity(
            market_id="test",
            actual_probability=-0.1,  # Invalid
            polymarket_yes_odds=0.55,
            polymarket_no_odds=0.45,
            edge_percentage=0.13,
            recommended_action="BUY_YES",
            confidence_boost=0.20,
            urgency="MEDIUM",
            expected_profit_pct=0.18
        )

def test_arbitrage_opportunity_odds_validation():
    """Test YES/NO odds validation."""
    # Test YES odds > 1.0
    with pytest.raises(ValueError, match="Invalid YES odds"):
        ArbitrageOpportunity(
            market_id="test",
            actual_probability=0.68,
            polymarket_yes_odds=1.5,  # Invalid
            polymarket_no_odds=0.45,
            edge_percentage=0.13,
            recommended_action="BUY_YES",
            confidence_boost=0.20,
            urgency="MEDIUM",
            expected_profit_pct=0.18
        )

    # Test NO odds < 0.0
    with pytest.raises(ValueError, match="Invalid NO odds"):
        ArbitrageOpportunity(
            market_id="test",
            actual_probability=0.68,
            polymarket_yes_odds=0.55,
            polymarket_no_odds=-0.1,  # Invalid
            edge_percentage=0.13,
            recommended_action="BUY_YES",
            confidence_boost=0.20,
            urgency="MEDIUM",
            expected_profit_pct=0.18
        )

def test_arbitrage_opportunity_all_actions():
    """Test all valid action types."""
    for action in ["BUY_YES", "BUY_NO", "HOLD"]:
        opp = ArbitrageOpportunity(
            market_id="test",
            actual_probability=0.68,
            polymarket_yes_odds=0.55,
            polymarket_no_odds=0.45,
            edge_percentage=0.13,
            recommended_action=action,
            confidence_boost=0.20,
            urgency="MEDIUM",
            expected_profit_pct=0.18
        )
        assert opp.recommended_action == action

def test_arbitrage_opportunity_all_urgencies():
    """Test all valid urgency levels."""
    for urgency in ["HIGH", "MEDIUM", "LOW"]:
        opp = ArbitrageOpportunity(
            market_id="test",
            actual_probability=0.68,
            polymarket_yes_odds=0.55,
            polymarket_no_odds=0.45,
            edge_percentage=0.13,
            recommended_action="BUY_YES",
            confidence_boost=0.20,
            urgency=urgency,
            expected_profit_pct=0.18
        )
        assert opp.urgency == urgency

def test_limit_order_strategy_creation():
    """Test creating LimitOrderStrategy with valid data."""
    strategy = LimitOrderStrategy(
        target_price=0.58,
        timeout_seconds=30,
        fallback_to_market=True,
        urgency="HIGH",
        price_improvement_pct=0.02
    )

    assert strategy.target_price == 0.58
    assert strategy.timeout_seconds == 30
    assert strategy.fallback_to_market is True
    assert strategy.urgency == "HIGH"
    assert strategy.price_improvement_pct == 0.02

def test_limit_order_strategy_urgency_validation():
    """Test urgency validation for LimitOrderStrategy."""
    with pytest.raises(ValueError, match="Invalid urgency"):
        LimitOrderStrategy(
            target_price=0.58,
            timeout_seconds=30,
            fallback_to_market=True,
            urgency="INVALID",  # Should fail
            price_improvement_pct=0.02
        )

def test_limit_order_strategy_target_price_validation():
    """Test target_price bounds validation."""
    # Test price > 1.0
    with pytest.raises(ValueError, match="Invalid target_price"):
        LimitOrderStrategy(
            target_price=1.5,  # Invalid
            timeout_seconds=30,
            fallback_to_market=True,
            urgency="HIGH",
            price_improvement_pct=0.02
        )

    # Test price < 0.0
    with pytest.raises(ValueError, match="Invalid target_price"):
        LimitOrderStrategy(
            target_price=-0.1,  # Invalid
            timeout_seconds=30,
            fallback_to_market=True,
            urgency="HIGH",
            price_improvement_pct=0.02
        )

def test_limit_order_strategy_timeout_validation():
    """Test timeout_seconds validation."""
    with pytest.raises(ValueError, match="timeout_seconds must be >= 0"):
        LimitOrderStrategy(
            target_price=0.58,
            timeout_seconds=-10,  # Invalid
            fallback_to_market=True,
            urgency="HIGH",
            price_improvement_pct=0.02
        )

def test_limit_order_strategy_all_urgencies():
    """Test all valid urgency levels for LimitOrderStrategy."""
    for urgency in ["HIGH", "MEDIUM", "LOW"]:
        strategy = LimitOrderStrategy(
            target_price=0.58,
            timeout_seconds=30,
            fallback_to_market=True,
            urgency=urgency,
            price_improvement_pct=0.02
        )
        assert strategy.urgency == urgency

def test_limit_order_strategy_boundary_values():
    """Test boundary values for LimitOrderStrategy."""
    # Test minimum valid values
    strategy_min = LimitOrderStrategy(
        target_price=0.0,
        timeout_seconds=0,
        fallback_to_market=False,
        urgency="LOW",
        price_improvement_pct=0.0
    )
    assert strategy_min.target_price == 0.0
    assert strategy_min.timeout_seconds == 0

    # Test maximum valid values
    strategy_max = LimitOrderStrategy(
        target_price=1.0,
        timeout_seconds=3600,
        fallback_to_market=True,
        urgency="HIGH",
        price_improvement_pct=0.5
    )
    assert strategy_max.target_price == 1.0
    assert strategy_max.timeout_seconds == 3600
