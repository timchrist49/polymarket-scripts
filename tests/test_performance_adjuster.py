"""Tests for parameter adjustment and validation system."""

import pytest
from polymarket.performance.adjuster import ParameterAdjuster, ParameterBounds, AdjustmentTier
from polymarket.config import Settings
from unittest.mock import Mock

@pytest.fixture
def mock_settings():
    """Mock settings."""
    settings = Mock(spec=Settings)
    settings.bot_confidence_threshold = 0.75
    settings.bot_max_position_dollars = 10.0
    settings.bot_max_exposure_percent = 0.50
    return settings

def test_parameter_bounds():
    """Test parameter bounds definition."""
    bounds = ParameterBounds()

    # Confidence threshold bounds
    assert bounds.get_bounds("bot_confidence_threshold") == (0.50, 0.95)

    # Position size bounds
    assert bounds.get_bounds("bot_max_position_dollars") == (5.0, 50.0)

    # Exposure bounds
    assert bounds.get_bounds("bot_max_exposure_percent") == (0.10, 0.80)

def test_validate_within_bounds(mock_settings):
    """Test validation passes for values within bounds."""
    adjuster = ParameterAdjuster(mock_settings)

    # Valid adjustment
    is_valid = adjuster.validate_adjustment("bot_confidence_threshold", 0.70)
    assert is_valid is True

def test_validate_outside_bounds(mock_settings):
    """Test validation fails for values outside bounds."""
    adjuster = ParameterAdjuster(mock_settings)

    # Invalid - too low
    is_valid = adjuster.validate_adjustment("bot_confidence_threshold", 0.40)
    assert is_valid is False

    # Invalid - too high
    is_valid = adjuster.validate_adjustment("bot_confidence_threshold", 0.99)
    assert is_valid is False

def test_classify_tier_1_auto(mock_settings):
    """Test Tier 1 classification (±5% = auto-approve)."""
    adjuster = ParameterAdjuster(mock_settings)

    # 5% decrease from 0.75 = 0.7125 (within ±5%)
    tier = adjuster.classify_tier("bot_confidence_threshold", 0.75, 0.7125)
    assert tier == AdjustmentTier.TIER_1_AUTO

def test_classify_tier_2_approval(mock_settings):
    """Test Tier 2 classification (5-20% = requires approval)."""
    adjuster = ParameterAdjuster(mock_settings)

    # 10% decrease from 0.75 = 0.675 (requires approval)
    tier = adjuster.classify_tier("bot_confidence_threshold", 0.75, 0.675)
    assert tier == AdjustmentTier.TIER_2_APPROVAL

def test_classify_tier_3_pause(mock_settings):
    """Test Tier 3 classification (>20% = emergency pause)."""
    adjuster = ParameterAdjuster(mock_settings)

    # 30% decrease from 0.75 = 0.525 (emergency pause)
    tier = adjuster.classify_tier("bot_confidence_threshold", 0.75, 0.525)
    assert tier == AdjustmentTier.TIER_3_PAUSE

def test_calculate_change_percent(mock_settings):
    """Test change percentage calculation."""
    adjuster = ParameterAdjuster(mock_settings)

    # 10% decrease
    pct = adjuster.calculate_change_percent(10.0, 9.0)
    assert pct == -10.0

    # 20% increase
    pct = adjuster.calculate_change_percent(10.0, 12.0)
    assert pct == 20.0
