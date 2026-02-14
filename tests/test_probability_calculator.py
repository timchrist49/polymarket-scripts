"""Tests for probability calculator (gap-based calculation)."""
import pytest
from polymarket.trading.probability_calculator import ProbabilityCalculator


def test_above_target_high_probability():
    """Test: Current price ABOVE target = high probability of winning."""
    calc = ProbabilityCalculator()

    # BTC at $70,100, target is $70,000 (already $100 above!)
    # With 60 seconds left, very unlikely to drop $100
    prob = calc.calculate_directional_probability(
        current_price=70100.0,
        target_price=70000.0,  # Already above target
        price_5min_ago=70050.0,  # Slight upward momentum
        price_10min_ago=70000.0,
        volatility_15min=0.005,  # 0.5%
        time_remaining_seconds=60,  # 1 minute left
        orderbook_imbalance=0.0
    )

    # Should be very high (near max)
    assert prob >= 0.80, f"Expected high probability when above target, got {prob:.2%}"


def test_below_target_low_probability():
    """Test: Current price BELOW target = low probability of winning."""
    calc = ProbabilityCalculator()

    # BTC at $69,900, target is $70,000 (need to gain $100)
    # With only 60 seconds left, very unlikely to gain $100
    prob = calc.calculate_directional_probability(
        current_price=69900.0,
        target_price=70000.0,  # Below target
        price_5min_ago=69950.0,  # Slight downward momentum
        price_10min_ago=70000.0,
        volatility_15min=0.005,  # 0.5%
        time_remaining_seconds=60,  # 1 minute left
        orderbook_imbalance=0.0
    )

    # Should be low
    assert prob <= 0.30, f"Expected low probability when below target, got {prob:.2%}"


def test_at_target_neutral_probability():
    """Test: Current price AT target = ~50% probability."""
    calc = ProbabilityCalculator()

    # BTC at $70,000, target is $70,000 (exactly at target)
    prob = calc.calculate_directional_probability(
        current_price=70000.0,
        target_price=70000.0,  # At target
        price_5min_ago=70000.0,  # Flat momentum
        price_10min_ago=70000.0,
        volatility_15min=0.005,
        time_remaining_seconds=300,  # 5 minutes left
        orderbook_imbalance=0.0
    )

    # Should be near 50% (coin flip)
    assert 0.45 <= prob <= 0.55, f"Expected ~50% when at target, got {prob:.2%}"


def test_upward_momentum_helps_reach_target():
    """Test: Positive momentum increases probability of reaching target above."""
    calc = ProbabilityCalculator()

    # BTC at $69,900, target $70,000 (need $100 up)
    # Strong upward momentum: $69,700 -> $69,900 (+$200 in 5min)
    prob_with_momentum = calc.calculate_directional_probability(
        current_price=69900.0,
        target_price=70000.0,
        price_5min_ago=69700.0,  # Strong upward momentum
        price_10min_ago=69500.0,
        volatility_15min=0.005,
        time_remaining_seconds=300,  # 5 minutes left
        orderbook_imbalance=0.2  # Buy pressure
    )

    # Same position but no momentum
    prob_flat = calc.calculate_directional_probability(
        current_price=69900.0,
        target_price=70000.0,
        price_5min_ago=69900.0,  # Flat
        price_10min_ago=69900.0,
        volatility_15min=0.005,
        time_remaining_seconds=300,
        orderbook_imbalance=0.0
    )

    # Momentum should increase probability
    assert prob_with_momentum > prob_flat


def test_time_matters_when_far_from_target():
    """Test: More time = higher probability of reaching distant target."""
    calc = ProbabilityCalculator()

    # BTC at $69,800, target $70,000 (need $200 up)
    # Positive momentum
    prob_more_time = calc.calculate_directional_probability(
        current_price=69800.0,
        target_price=70000.0,
        price_5min_ago=69700.0,  # +$100 in 5min
        price_10min_ago=69600.0,
        volatility_15min=0.005,
        time_remaining_seconds=600,  # 10 minutes left
        orderbook_imbalance=0.0
    )

    # Same setup but less time
    prob_less_time = calc.calculate_directional_probability(
        current_price=69800.0,
        target_price=70000.0,
        price_5min_ago=69700.0,
        price_10min_ago=69600.0,
        volatility_15min=0.005,
        time_remaining_seconds=120,  # 2 minutes left
        orderbook_imbalance=0.0
    )

    # More time = higher probability of reaching target
    assert prob_more_time > prob_less_time


def test_volatility_affects_confidence():
    """Test: Higher volatility = probability closer to 50%."""
    calc = ProbabilityCalculator()

    # Low volatility = more confident in reaching target
    prob_low_vol = calc.calculate_directional_probability(
        current_price=69900.0,
        target_price=70000.0,
        price_5min_ago=69850.0,  # +$50 momentum
        price_10min_ago=69800.0,
        volatility_15min=0.001,  # Very low volatility
        time_remaining_seconds=300,
        orderbook_imbalance=0.0
    )

    # High volatility = less confident (anything can happen)
    prob_high_vol = calc.calculate_directional_probability(
        current_price=69900.0,
        target_price=70000.0,
        price_5min_ago=69850.0,
        price_10min_ago=69800.0,
        volatility_15min=0.020,  # High volatility
        time_remaining_seconds=300,
        orderbook_imbalance=0.0
    )

    # High vol probability should be closer to 0.5 (more uncertain)
    assert abs(prob_high_vol - 0.5) < abs(prob_low_vol - 0.5)


def test_end_phase_far_below_target():
    """Test: End phase (0s left) + far below target = very low probability."""
    calc = ProbabilityCalculator()

    # Real scenario from Trade ID 235:
    # Current: $69,669, Target: $69,726, Gap: +$57, Time: 0s
    prob = calc.calculate_directional_probability(
        current_price=69669.0,
        target_price=69726.0,  # Need $57 up
        price_5min_ago=69648.0,
        price_10min_ago=69640.0,
        volatility_15min=0.005,
        time_remaining_seconds=1,  # Effectively 0
        orderbook_imbalance=0.0
    )

    # Should be very low (impossible to gain $57 in 0 seconds)
    assert prob <= 0.10, f"Expected very low prob in end phase below target, got {prob:.2%}"


def test_probability_bounds():
    """Test that probability is always clipped to [0.05, 0.95]."""
    calc = ProbabilityCalculator()

    # Extreme case: way above target
    prob_max = calc.calculate_directional_probability(
        current_price=75000.0,
        target_price=70000.0,  # $5000 above!
        price_5min_ago=74900.0,
        price_10min_ago=74800.0,
        volatility_15min=0.001,
        time_remaining_seconds=60,
        orderbook_imbalance=0.5
    )

    assert prob_max <= 0.95, "Probability should be capped at 0.95"

    # Extreme case: way below target
    prob_min = calc.calculate_directional_probability(
        current_price=65000.0,
        target_price=70000.0,  # $5000 below!
        price_5min_ago=65100.0,
        price_10min_ago=65200.0,
        volatility_15min=0.001,
        time_remaining_seconds=60,
        orderbook_imbalance=-0.5
    )

    assert prob_min >= 0.05, "Probability should be floored at 0.05"


def test_invalid_prices():
    """Test that negative/zero prices raise ValueError."""
    calc = ProbabilityCalculator()
    with pytest.raises(ValueError, match="Prices must be positive"):
        calc.calculate_directional_probability(
            current_price=-70000.0,  # Invalid
            target_price=70000.0,
            price_5min_ago=69900.0,
            price_10min_ago=69800.0,
            volatility_15min=0.005,
            time_remaining_seconds=300,
            orderbook_imbalance=0.0
        )


def test_invalid_target_price():
    """Test that negative/zero target price raises ValueError."""
    calc = ProbabilityCalculator()
    with pytest.raises(ValueError, match="Prices must be positive"):
        calc.calculate_directional_probability(
            current_price=70000.0,
            target_price=0.0,  # Invalid
            price_5min_ago=69900.0,
            price_10min_ago=69800.0,
            volatility_15min=0.005,
            time_remaining_seconds=300,
            orderbook_imbalance=0.0
        )


def test_invalid_volatility():
    """Test that negative volatility raises ValueError."""
    calc = ProbabilityCalculator()
    with pytest.raises(ValueError, match="Volatility cannot be negative"):
        calc.calculate_directional_probability(
            current_price=70000.0,
            target_price=70000.0,
            price_5min_ago=69900.0,
            price_10min_ago=69800.0,
            volatility_15min=-0.005,  # Invalid
            time_remaining_seconds=300,
            orderbook_imbalance=0.0
        )


def test_invalid_orderbook_imbalance():
    """Test that invalid orderbook imbalance raises ValueError."""
    calc = ProbabilityCalculator()
    with pytest.raises(ValueError, match="Orderbook imbalance"):
        calc.calculate_directional_probability(
            current_price=70000.0,
            target_price=70000.0,
            price_5min_ago=69900.0,
            price_10min_ago=69800.0,
            volatility_15min=0.005,
            time_remaining_seconds=300,
            orderbook_imbalance=5.0  # Invalid (> 1.0)
        )
