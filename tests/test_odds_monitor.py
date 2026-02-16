"""Tests for OddsMonitor."""

import pytest
from unittest.mock import Mock, AsyncMock
from polymarket.trading.odds_monitor import OddsMonitor
from polymarket.trading.realtime_odds_streamer import RealtimeOddsStreamer
from polymarket.trading.market_validator import MarketValidator


def test_odds_monitor_initialization():
    """Test OddsMonitor initializes with correct configuration."""
    # Create mock dependencies
    streamer = Mock(spec=RealtimeOddsStreamer)
    validator = Mock(spec=MarketValidator)
    on_opportunity = Mock()

    # Initialize monitor
    monitor = OddsMonitor(
        streamer=streamer,
        validator=validator,
        on_opportunity_detected=on_opportunity,
        threshold_percentage=70.0,
        sustained_duration_seconds=5.0,
        cooldown_seconds=30.0
    )

    # Verify initialization
    assert monitor._streamer is streamer
    assert monitor._validator is validator
    assert monitor._on_opportunity_detected is on_opportunity
    assert monitor._threshold == 0.70  # Converted to decimal
    assert monitor._sustained_duration == 5.0
    assert monitor._cooldown_duration == 30.0
    assert monitor._is_running is False
    assert monitor._monitor_task is None
    assert monitor._threshold_start_time == {}
    assert monitor._last_trigger_time == {}
