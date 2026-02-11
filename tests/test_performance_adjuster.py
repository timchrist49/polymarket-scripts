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

@pytest.mark.asyncio
async def test_apply_tier_1_adjustment(mock_settings):
    """Test applying Tier 1 auto-adjustment."""
    from polymarket.performance.database import PerformanceDatabase
    from polymarket.telegram.bot import TelegramBot
    from unittest.mock import AsyncMock, Mock

    db = PerformanceDatabase(":memory:")
    telegram = Mock(spec=TelegramBot)
    telegram.send_message = AsyncMock()
    telegram._send_message = AsyncMock()

    adjuster = ParameterAdjuster(mock_settings, db=db, telegram=telegram)

    # Apply Tier 1 adjustment (3% decrease from 0.75 to 0.7275)
    result = await adjuster.apply_adjustment(
        parameter_name="bot_confidence_threshold",
        old_value=0.75,
        new_value=0.7275,
        reason="Win rate is high, can be more aggressive",
        tier=AdjustmentTier.TIER_1_AUTO
    )

    assert result is True

    # Verify logged to database
    cursor = db.conn.cursor()
    cursor.execute("SELECT * FROM parameter_history ORDER BY timestamp DESC LIMIT 1")
    row = cursor.fetchone()

    assert row['parameter_name'] == 'bot_confidence_threshold'
    assert row['old_value'] == 0.75
    assert row['new_value'] == 0.7275
    assert row['approval_method'] == 'tier_1_auto'

    db.close()

@pytest.mark.asyncio
async def test_reject_tier_2_without_approval(mock_settings):
    """Test that Tier 2 adjustments are rejected without approval."""
    from polymarket.performance.database import PerformanceDatabase

    db = PerformanceDatabase(":memory:")
    adjuster = ParameterAdjuster(mock_settings, db=db, telegram=None)

    # Try to apply Tier 2 adjustment without approval
    result = await adjuster.apply_adjustment(
        parameter_name="bot_confidence_threshold",
        old_value=0.75,
        new_value=0.65,  # 13% decrease
        reason="Test",
        tier=AdjustmentTier.TIER_2_APPROVAL
    )

    assert result is False  # Should be rejected

    db.close()

@pytest.mark.asyncio
async def test_reject_out_of_bounds(mock_settings):
    """Test that out-of-bounds adjustments are rejected."""
    from polymarket.performance.database import PerformanceDatabase

    db = PerformanceDatabase(":memory:")
    adjuster = ParameterAdjuster(mock_settings, db=db, telegram=None)

    # Try to apply adjustment outside bounds
    result = await adjuster.apply_adjustment(
        parameter_name="bot_confidence_threshold",
        old_value=0.75,
        new_value=0.98,  # Above max of 0.95
        reason="Test",
        tier=AdjustmentTier.TIER_1_AUTO
    )

    assert result is False  # Should be rejected

    db.close()
