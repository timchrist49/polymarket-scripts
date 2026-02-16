"""Integration tests for AutoTrader with OddsMonitor.

NOTE: This is a minimal smoke test to verify OddsMonitor integration.
Full integration testing is done manually as per implementation plan.
"""

import pytest
from polymarket.config import Settings
from polymarket.trading.odds_monitor import OddsMonitor
from polymarket.trading.market_validator import MarketValidator


def test_odds_monitor_and_validator_imports():
    """Test that OddsMonitor and MarketValidator are importable."""
    # This test just verifies the classes exist and can be imported
    assert OddsMonitor is not None
    assert MarketValidator is not None


def test_market_validator_instantiation():
    """Test MarketValidator can be instantiated."""
    validator = MarketValidator()
    assert validator is not None


# NOTE: AutoTrader initialization test is manual as per implementation plan Task 8
# The AutoTrader class has many dependencies that are difficult to mock properly
# Manual verification will be done by:
# 1. Checking imports are added
# 2. Verifying OddsMonitor initialization code is present
# 3. Running scripts/auto_trade.py to ensure no runtime errors
