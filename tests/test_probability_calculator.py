"""Tests for probability calculator."""
import pytest
from polymarket.trading.probability_calculator import ProbabilityCalculator

def test_calculate_upward_momentum():
    """Test probability calculation with upward momentum."""
    calc = ProbabilityCalculator()

    # BTC rising: $66,000 -> $66,200 (+0.30% in 5min)
    prob = calc.calculate_directional_probability(
        current_price=66200.0,
        price_5min_ago=66000.0,
        price_10min_ago=65900.0,
        volatility_15min=0.005,  # 0.5%
        time_remaining_seconds=600,  # 10 minutes left
        orderbook_imbalance=0.2  # Buy pressure
    )

    # Should be >0.5 due to upward momentum + buy pressure
    assert prob > 0.5
    assert prob < 0.95  # But not extreme
    assert 0.65 <= prob <= 0.85  # Reasonable range


def test_calculate_downward_momentum():
    """Test probability calculation with downward momentum."""
    calc = ProbabilityCalculator()

    # BTC falling: $66,000 -> $65,800 (-0.30% in 5min)
    prob = calc.calculate_directional_probability(
        current_price=65800.0,
        price_5min_ago=66000.0,
        price_10min_ago=66100.0,
        volatility_15min=0.005,
        time_remaining_seconds=600,
        orderbook_imbalance=-0.2  # Sell pressure
    )

    # Should be <0.5 due to downward momentum + sell pressure
    assert prob < 0.5
    assert prob > 0.05  # But not extreme
    assert 0.15 <= prob <= 0.35


def test_calculate_no_momentum():
    """Test probability calculation with sideways movement."""
    calc = ProbabilityCalculator()

    # BTC flat: $66,000 -> $66,000
    prob = calc.calculate_directional_probability(
        current_price=66000.0,
        price_5min_ago=66000.0,
        price_10min_ago=66000.0,
        volatility_15min=0.005,
        time_remaining_seconds=600,
        orderbook_imbalance=0.0
    )

    # Should be ~0.5 (neutral)
    assert 0.45 <= prob <= 0.55


def test_high_volatility_reduces_confidence():
    """Test that high volatility brings probability closer to 0.5."""
    calc = ProbabilityCalculator()

    # Same momentum, different volatility
    low_vol = calc.calculate_directional_probability(
        current_price=66200.0,
        price_5min_ago=66000.0,
        price_10min_ago=65900.0,
        volatility_15min=0.002,  # Low volatility
        time_remaining_seconds=600,
        orderbook_imbalance=0.0
    )

    high_vol = calc.calculate_directional_probability(
        current_price=66200.0,
        price_5min_ago=66000.0,
        price_10min_ago=65900.0,
        volatility_15min=0.020,  # High volatility
        time_remaining_seconds=600,
        orderbook_imbalance=0.0
    )

    # High volatility should reduce confidence (closer to 0.5)
    assert abs(low_vol - 0.5) > abs(high_vol - 0.5)


def test_time_decay():
    """Test that less time remaining amplifies momentum signal."""
    calc = ProbabilityCalculator()

    # Same momentum, different time remaining
    more_time = calc.calculate_directional_probability(
        current_price=66200.0,
        price_5min_ago=66000.0,
        price_10min_ago=65900.0,
        volatility_15min=0.005,
        time_remaining_seconds=800,  # 13+ minutes left
        orderbook_imbalance=0.0
    )

    less_time = calc.calculate_directional_probability(
        current_price=66200.0,
        price_5min_ago=66000.0,
        price_10min_ago=65900.0,
        volatility_15min=0.005,
        time_remaining_seconds=200,  # 3 minutes left
        orderbook_imbalance=0.0
    )

    # Less time = momentum matters more (probability further from 0.5)
    # With less time, there's less opportunity for mean reversion
    assert abs(less_time - 0.5) > abs(more_time - 0.5)


def test_probability_bounds():
    """Test that probability is always clipped to [0.05, 0.95]."""
    calc = ProbabilityCalculator()

    # Extreme upward momentum
    prob_up = calc.calculate_directional_probability(
        current_price=70000.0,
        price_5min_ago=66000.0,  # +6% spike
        price_10min_ago=65000.0,
        volatility_15min=0.001,  # Very low volatility
        time_remaining_seconds=900,
        orderbook_imbalance=0.5  # Max buy pressure
    )

    # Should be clipped to 0.95 max
    assert prob_up <= 0.95

    # Extreme downward momentum
    prob_down = calc.calculate_directional_probability(
        current_price=62000.0,
        price_5min_ago=66000.0,  # -6% drop
        price_10min_ago=67000.0,
        volatility_15min=0.001,
        time_remaining_seconds=900,
        orderbook_imbalance=-0.5  # Max sell pressure
    )

    # Should be clipped to 0.05 min
    assert prob_down >= 0.05
