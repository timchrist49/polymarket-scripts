"""Integration tests for AutoTrader with OddsMonitor."""

import pytest
from polymarket.config import Settings


def test_autotrader_initializes_odds_monitor():
    """Test AutoTrader initializes OddsMonitor correctly."""
    from scripts.auto_trade import AutoTrader

    settings = Settings()

    # Initialize AutoTrader (will create all dependencies including OddsMonitor)
    trader = AutoTrader(settings)

    # Verify OddsMonitor was initialized
    assert trader.odds_monitor is not None

    # Verify correct configuration
    assert trader.odds_monitor._threshold == 0.70
    assert trader.odds_monitor._sustained_duration == 5.0
    assert trader.odds_monitor._cooldown_duration == 30.0

    # Verify callback method exists
    assert hasattr(trader, '_handle_opportunity_detected')
    assert callable(trader._handle_opportunity_detected)


@pytest.mark.asyncio
async def test_autotrader_start_stop_with_odds_monitor():
    """Test AutoTrader initialize and cleanup work with OddsMonitor."""
    from scripts.auto_trade import AutoTrader

    settings = Settings()
    trader = AutoTrader(settings)

    # Initialize should start the monitor
    await trader.initialize()

    # Verify OddsMonitor is running
    assert trader.odds_monitor._is_running is True

    # Simulate cleanup (stop monitor)
    if trader.odds_monitor:
        await trader.odds_monitor.stop()

    # Verify OddsMonitor is stopped
    assert trader.odds_monitor._is_running is False

    # Cleanup other resources
    await trader.btc_service.close()
    await trader.social_service.close()
    await trader.realtime_streamer.stop()
