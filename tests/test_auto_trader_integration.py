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
