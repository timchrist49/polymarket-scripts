"""Tests for OddsMonitor."""

import pytest
from unittest.mock import Mock, AsyncMock
from polymarket.trading.odds_monitor import OddsMonitor
from polymarket.trading.realtime_odds_streamer import RealtimeOddsStreamer
from polymarket.trading.market_validator import MarketValidator
from polymarket.models import WebSocketOddsSnapshot
from datetime import datetime, timezone


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


@pytest.mark.asyncio
async def test_check_opportunities_above_threshold():
    """Test detecting opportunity when odds above threshold."""
    # Create mock dependencies
    streamer = Mock(spec=RealtimeOddsStreamer)
    validator = Mock(spec=MarketValidator)
    on_opportunity = Mock()

    # Mock current odds: YES=0.75 (above 70% threshold)
    mock_snapshot = WebSocketOddsSnapshot(
        market_id="test-market-123",
        yes_odds=0.75,
        no_odds=0.25,
        timestamp=datetime.now(timezone.utc),
        best_bid=0.75,
        best_ask=0.25
    )
    streamer.get_current_odds = Mock(return_value=mock_snapshot)

    # Mock market validation: market is active
    validator.is_market_active = Mock(return_value=True)

    # Initialize monitor with market tracking
    monitor = OddsMonitor(
        streamer=streamer,
        validator=validator,
        on_opportunity_detected=on_opportunity,
        threshold_percentage=70.0
    )
    # Set the market ID and slug mapping
    monitor._market_id = "test-market-123"
    monitor._market_slug = "btc-updown-15m-1234567890"

    # Check for opportunities
    result = await monitor._check_opportunities()

    # Verify opportunity detected
    assert result is not None
    assert result["market_slug"] == "btc-updown-15m-1234567890"
    assert result["direction"] == "YES"
    assert result["odds"] == 0.75


@pytest.mark.asyncio
async def test_check_opportunities_below_threshold():
    """Test no opportunity when odds below threshold."""
    # Create mock dependencies
    streamer = Mock(spec=RealtimeOddsStreamer)
    validator = Mock(spec=MarketValidator)
    on_opportunity = Mock()

    # Mock current odds: YES=0.60 (below 70% threshold)
    mock_snapshot = WebSocketOddsSnapshot(
        market_id="test-market-123",
        yes_odds=0.60,
        no_odds=0.40,
        timestamp=datetime.now(timezone.utc),
        best_bid=0.60,
        best_ask=0.40
    )
    streamer.get_current_odds = Mock(return_value=mock_snapshot)
    validator.is_market_active = Mock(return_value=True)

    # Initialize monitor
    monitor = OddsMonitor(
        streamer=streamer,
        validator=validator,
        on_opportunity_detected=on_opportunity,
        threshold_percentage=70.0
    )
    monitor._market_id = "test-market-123"
    monitor._market_slug = "btc-updown-15m-1234567890"

    # Check for opportunities
    result = await monitor._check_opportunities()

    # Verify no opportunity
    assert result is None


@pytest.mark.asyncio
async def test_check_opportunities_market_inactive():
    """Test no opportunity when market is not active."""
    # Create mock dependencies
    streamer = Mock(spec=RealtimeOddsStreamer)
    validator = Mock(spec=MarketValidator)
    on_opportunity = Mock()

    # Mock current odds: YES=0.75 (above threshold but market inactive)
    mock_snapshot = WebSocketOddsSnapshot(
        market_id="test-market-123",
        yes_odds=0.75,
        no_odds=0.25,
        timestamp=datetime.now(timezone.utc),
        best_bid=0.75,
        best_ask=0.25
    )
    streamer.get_current_odds = Mock(return_value=mock_snapshot)

    # Mock market validation: market is NOT active
    validator.is_market_active = Mock(return_value=False)

    # Initialize monitor
    monitor = OddsMonitor(
        streamer=streamer,
        validator=validator,
        on_opportunity_detected=on_opportunity,
        threshold_percentage=70.0
    )
    monitor._market_id = "test-market-123"
    monitor._market_slug = "btc-updown-15m-1234567890"

    # Check for opportunities
    result = await monitor._check_opportunities()

    # Verify no opportunity (market inactive)
    assert result is None


@pytest.mark.asyncio
async def test_check_opportunities_no_odds():
    """Test no opportunity when streamer returns None (no odds available)."""
    # Create mock dependencies
    streamer = Mock(spec=RealtimeOddsStreamer)
    validator = Mock(spec=MarketValidator)
    on_opportunity = Mock()

    # Mock streamer: returns None (no current odds)
    streamer.get_current_odds = Mock(return_value=None)

    # Initialize monitor
    monitor = OddsMonitor(
        streamer=streamer,
        validator=validator,
        on_opportunity_detected=on_opportunity,
        threshold_percentage=70.0
    )
    monitor._market_id = "test-market-123"
    monitor._market_slug = "btc-updown-15m-1234567890"

    # Check for opportunities
    result = await monitor._check_opportunities()

    # Verify no opportunity
    assert result is None


@pytest.mark.asyncio
async def test_check_opportunities_no_direction():
    """Test detecting NO opportunity when NO odds above threshold."""
    # Create mock dependencies
    streamer = Mock(spec=RealtimeOddsStreamer)
    validator = Mock(spec=MarketValidator)
    on_opportunity = Mock()

    # Mock current odds: NO=0.75 (above 70% threshold)
    mock_snapshot = WebSocketOddsSnapshot(
        market_id="test-market-123",
        yes_odds=0.25,  # Below threshold
        no_odds=0.75,   # Above threshold - this should trigger
        timestamp=datetime.now(timezone.utc),
        best_bid=0.25,
        best_ask=0.75
    )
    streamer.get_current_odds = Mock(return_value=mock_snapshot)

    # Mock market validation: market is active
    validator.is_market_active = Mock(return_value=True)

    # Initialize monitor with market configured
    monitor = OddsMonitor(
        streamer=streamer,
        validator=validator,
        on_opportunity_detected=on_opportunity,
        threshold_percentage=70.0
    )
    monitor._market_id = "test-market-123"
    monitor._market_slug = "btc-updown-15m-1234567890"

    # Check for opportunities
    result = await monitor._check_opportunities()

    # Verify NO opportunity detected
    assert result is not None
    assert result["direction"] == "NO"
    assert result["odds"] == 0.75


@pytest.mark.asyncio
async def test_check_opportunities_stale_odds():
    """Test rejecting stale odds (>120 seconds old)."""
    # Create mock dependencies
    streamer = Mock(spec=RealtimeOddsStreamer)
    validator = Mock(spec=MarketValidator)
    on_opportunity = Mock()

    # Mock current odds: YES=0.75 (above threshold) but stale (3 minutes old)
    from datetime import timedelta
    stale_time = datetime.now(timezone.utc) - timedelta(seconds=180)  # 3 minutes ago
    
    mock_snapshot = WebSocketOddsSnapshot(
        market_id="test-market-123",
        yes_odds=0.75,
        no_odds=0.25,
        timestamp=stale_time,  # Stale timestamp
        best_bid=0.75,
        best_ask=0.25
    )
    streamer.get_current_odds = Mock(return_value=mock_snapshot)

    # Mock market validation: market is active (but should never be checked due to staleness)
    validator.is_market_active = Mock(return_value=True)

    # Initialize monitor with market configured
    monitor = OddsMonitor(
        streamer=streamer,
        validator=validator,
        on_opportunity_detected=on_opportunity,
        threshold_percentage=70.0
    )
    monitor._market_id = "test-market-123"
    monitor._market_slug = "btc-updown-15m-1234567890"

    # Check for opportunities
    result = await monitor._check_opportunities()

    # Verify no opportunity (stale odds rejected)
    assert result is None
    # Verify market validation was never called (staleness check happens first)
    validator.is_market_active.assert_not_called()
