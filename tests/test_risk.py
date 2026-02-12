"""
Tests for risk management with odds-adjusted position sizing.
"""
import pytest
from decimal import Decimal
from polymarket.trading.risk import RiskManager
from polymarket.config import Settings


class TestOddsMultiplier:
    """Test odds multiplier calculation."""

    def test_high_odds_no_scaling(self):
        """Odds >= 0.50 should have 1.0x multiplier (no scaling)."""
        settings = Settings()
        risk_mgr = RiskManager(settings)

        assert risk_mgr._calculate_odds_multiplier(0.83) == 1.0
        assert risk_mgr._calculate_odds_multiplier(0.50) == 1.0
        assert risk_mgr._calculate_odds_multiplier(0.60) == 1.0

    def test_low_odds_scaled_down(self):
        """Odds < 0.50 should scale linearly from 1.0x down to 0.5x."""
        settings = Settings()
        risk_mgr = RiskManager(settings)

        # 0.40 odds → 0.80x multiplier
        assert abs(risk_mgr._calculate_odds_multiplier(0.40) - 0.80) < 0.01

        # 0.31 odds → 0.62x multiplier
        assert abs(risk_mgr._calculate_odds_multiplier(0.31) - 0.62) < 0.01

        # 0.25 odds → 0.50x multiplier (minimum)
        assert abs(risk_mgr._calculate_odds_multiplier(0.25) - 0.50) < 0.01

    def test_below_minimum_odds_rejected(self):
        """Odds < 0.25 should return 0.0 (bet rejected)."""
        settings = Settings()
        risk_mgr = RiskManager(settings)

        assert risk_mgr._calculate_odds_multiplier(0.20) == 0.0
        assert risk_mgr._calculate_odds_multiplier(0.15) == 0.0
        assert risk_mgr._calculate_odds_multiplier(0.10) == 0.0
