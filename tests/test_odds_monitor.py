"""Tests for OddsMonitor."""

import pytest
from unittest.mock import Mock, AsyncMock
from polymarket.trading.odds_monitor import OddsMonitor
from polymarket.trading.realtime_odds_streamer import RealtimeOddsStreamer
from polymarket.trading.market_validator import MarketValidator
from polymarket.models import WebSocketOddsSnapshot
from datetime import datetime, timezone, timedelta
from freezegun import freeze_time


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


@pytest.mark.asyncio
@freeze_time("2026-02-16 14:30:00")
async def test_sustained_threshold_triggers_after_5_seconds():
    """Test opportunity triggered only after sustained 5 seconds."""
    # Create mock dependencies
    streamer = Mock(spec=RealtimeOddsStreamer)
    validator = Mock(spec=MarketValidator)
    on_opportunity = Mock()

    # Mock current odds: YES=0.75 (above threshold)
    mock_snapshot = WebSocketOddsSnapshot(
        market_id="test-market-123",
        yes_odds=0.75,
        no_odds=0.25,
        timestamp=datetime.now(timezone.utc),
        best_bid=0.75,
        best_ask=0.25
    )
    streamer.get_current_odds = Mock(return_value=mock_snapshot)
    validator.is_market_active = Mock(return_value=True)

    # Initialize monitor
    monitor = OddsMonitor(
        streamer=streamer,
        validator=validator,
        on_opportunity_detected=on_opportunity,
        threshold_percentage=70.0,
        sustained_duration_seconds=5.0
    )
    monitor._market_id = "test-market-123"
    monitor._market_slug = "btc-updown-15m-1234567890"

    # First check - opportunity detected but not sustained
    await monitor._check_and_handle_opportunity()
    assert on_opportunity.call_count == 0  # Not called yet (< 5 seconds)

    # 3 seconds later - still not sustained
    with freeze_time("2026-02-16 14:30:03"):
        await monitor._check_and_handle_opportunity()
        assert on_opportunity.call_count == 0  # Not called yet (< 5 seconds)

    # 5 seconds later - sustained threshold met
    with freeze_time("2026-02-16 14:30:05"):
        await monitor._check_and_handle_opportunity()
        assert on_opportunity.call_count == 1  # NOW called (>= 5 seconds)
        on_opportunity.assert_called_with("btc-updown-15m-1234567890", "YES", 0.75)


@pytest.mark.asyncio
@freeze_time("2026-02-16 14:30:00")
async def test_cooldown_prevents_rapid_fire():
    """Test 30-second cooldown prevents multiple triggers."""
    # Create mock dependencies
    streamer = Mock(spec=RealtimeOddsStreamer)
    validator = Mock(spec=MarketValidator)
    on_opportunity = Mock()

    # Mock current odds: YES=0.75 (above threshold)
    mock_snapshot = WebSocketOddsSnapshot(
        market_id="test-market-123",
        yes_odds=0.75,
        no_odds=0.25,
        timestamp=datetime.now(timezone.utc),
        best_bid=0.75,
        best_ask=0.25
    )
    streamer.get_current_odds = Mock(return_value=mock_snapshot)
    validator.is_market_active = Mock(return_value=True)

    # Initialize monitor with cooldown
    monitor = OddsMonitor(
        streamer=streamer,
        validator=validator,
        on_opportunity_detected=on_opportunity,
        threshold_percentage=70.0,
        sustained_duration_seconds=5.0,
        cooldown_seconds=30.0
    )
    monitor._market_id = "test-market-123"
    monitor._market_slug = "btc-updown-15m-1234567890"

    # First call at 14:30:00 - start tracking
    await monitor._check_and_handle_opportunity()
    assert on_opportunity.call_count == 0  # Not yet sustained

    # Trigger first opportunity (5 seconds sustained)
    with freeze_time("2026-02-16 14:30:05"):
        await monitor._check_and_handle_opportunity()
        assert on_opportunity.call_count == 1

    # 10 seconds later - still in cooldown (< 30 seconds)
    with freeze_time("2026-02-16 14:30:15"):
        await monitor._check_and_handle_opportunity()
        assert on_opportunity.call_count == 1  # Not called again (cooldown)

    # 31 seconds later - cooldown expired, but need to start new sustained period
    with freeze_time("2026-02-16 14:30:36"):
        await monitor._check_and_handle_opportunity()
        assert on_opportunity.call_count == 1  # Not triggered yet (new sustained period started)

    # 5 seconds after cooldown expiry - sustained threshold met again
    with freeze_time("2026-02-16 14:30:41"):
        await monitor._check_and_handle_opportunity()
        assert on_opportunity.call_count == 2  # NOW triggered (5s sustained after cooldown)


@pytest.mark.asyncio
@freeze_time("2026-02-16 14:30:00")
async def test_threshold_broken_resets_timer():
    """Test dropping below threshold resets sustained timer."""
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
        sustained_duration_seconds=5.0
    )
    monitor._market_id = "test-market-123"
    monitor._market_slug = "btc-updown-15m-1234567890"
    validator.is_market_active = Mock(return_value=True)

    # First check - odds above threshold (0.75)
    high_odds = WebSocketOddsSnapshot(
        market_id="test-market-123",
        yes_odds=0.75,
        no_odds=0.25,
        timestamp=datetime.now(timezone.utc),
        best_bid=0.75,
        best_ask=0.25
    )
    streamer.get_current_odds = Mock(return_value=high_odds)
    await monitor._check_and_handle_opportunity()

    # 3 seconds later - odds drop below threshold (0.60)
    low_odds = WebSocketOddsSnapshot(
        market_id="test-market-123",
        yes_odds=0.60,
        no_odds=0.40,
        timestamp=datetime.now(timezone.utc),
        best_bid=0.60,
        best_ask=0.40
    )
    with freeze_time("2026-02-16 14:30:03"):
        streamer.get_current_odds = Mock(return_value=low_odds)
        await monitor._check_and_handle_opportunity()

    # 5 seconds from original - odds back above threshold (0.75)
    with freeze_time("2026-02-16 14:30:05"):
        streamer.get_current_odds = Mock(return_value=high_odds)
        await monitor._check_and_handle_opportunity()
        # Should NOT trigger (timer was reset at 3s mark)
        assert on_opportunity.call_count == 0

    # 3 seconds from reset (8 seconds from original) - still not sustained
    with freeze_time("2026-02-16 14:30:08"):
        await monitor._check_and_handle_opportunity()
        assert on_opportunity.call_count == 0  # Still not 5s from reset (only 3s)

    # 5 seconds from reset (10 seconds from original) - NOW sustained
    with freeze_time("2026-02-16 14:30:10"):
        await monitor._check_and_handle_opportunity()
        assert on_opportunity.call_count == 1  # Triggered (5s from reset at 14:30:05)
