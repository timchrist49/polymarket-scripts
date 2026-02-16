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


@pytest.mark.asyncio
async def test_odds_monitor_start():
    """Test OddsMonitor can be started."""
    # Create mock dependencies
    streamer = Mock(spec=RealtimeOddsStreamer)
    validator = Mock(spec=MarketValidator)
    on_opportunity = Mock()

    # Initialize monitor
    monitor = OddsMonitor(
        streamer=streamer,
        validator=validator,
        on_opportunity_detected=on_opportunity
    )

    # Start monitoring
    await monitor.start()

    # Verify state
    assert monitor._is_running is True
    assert monitor._monitor_task is not None
    assert not monitor._monitor_task.done()

    # Cleanup
    await monitor.stop()


@pytest.mark.asyncio
async def test_odds_monitor_stop():
    """Test OddsMonitor can be stopped."""
    # Create mock dependencies
    streamer = Mock(spec=RealtimeOddsStreamer)
    validator = Mock(spec=MarketValidator)
    on_opportunity = Mock()

    # Initialize and start monitor
    monitor = OddsMonitor(
        streamer=streamer,
        validator=validator,
        on_opportunity_detected=on_opportunity
    )
    await monitor.start()

    # Stop monitoring
    await monitor.stop()

    # Verify state
    assert monitor._is_running is False
    assert monitor._monitor_task is not None  # Task object still exists
    assert monitor._monitor_task.done()       # But it's done/cancelled


@pytest.mark.asyncio
async def test_odds_monitor_double_start_prevented():
    """Test starting an already running monitor is prevented."""
    # Create mock dependencies
    streamer = Mock(spec=RealtimeOddsStreamer)
    validator = Mock(spec=MarketValidator)
    on_opportunity = Mock()

    # Initialize and start monitor
    monitor = OddsMonitor(
        streamer=streamer,
        validator=validator,
        on_opportunity_detected=on_opportunity
    )
    await monitor.start()
    first_task = monitor._monitor_task

    # Try to start again
    await monitor.start()

    # Verify same task (second start was no-op)
    assert monitor._monitor_task is first_task

    # Cleanup
    await monitor.stop()
