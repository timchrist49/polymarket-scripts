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

        assert risk_mgr._calculate_odds_multiplier(Decimal("0.83")) == Decimal("1.0")
        assert risk_mgr._calculate_odds_multiplier(Decimal("0.50")) == Decimal("1.0")
        assert risk_mgr._calculate_odds_multiplier(Decimal("0.60")) == Decimal("1.0")

    def test_low_odds_scaled_down(self):
        """Odds < 0.50 should scale linearly from 1.0x down to 0.5x."""
        settings = Settings()
        risk_mgr = RiskManager(settings)

        # 0.40 odds → 0.80x multiplier
        assert abs(risk_mgr._calculate_odds_multiplier(Decimal("0.40")) - Decimal("0.80")) < Decimal("0.01")

        # 0.31 odds → 0.62x multiplier
        assert abs(risk_mgr._calculate_odds_multiplier(Decimal("0.31")) - Decimal("0.62")) < Decimal("0.01")

        # 0.25 odds → 0.50x multiplier (minimum)
        assert abs(risk_mgr._calculate_odds_multiplier(Decimal("0.25")) - Decimal("0.50")) < Decimal("0.01")

    def test_below_minimum_odds_rejected(self):
        """Odds < 0.25 should return 0.0 (bet rejected)."""
        settings = Settings()
        risk_mgr = RiskManager(settings)

        assert risk_mgr._calculate_odds_multiplier(Decimal("0.20")) == Decimal("0")
        assert risk_mgr._calculate_odds_multiplier(Decimal("0.15")) == Decimal("0")
        assert risk_mgr._calculate_odds_multiplier(Decimal("0.10")) == Decimal("0")


class TestOddsExtraction:
    """Test odds extraction from market data."""

    def test_extract_yes_odds(self):
        """Extract odds for YES action."""
        settings = Settings()
        risk_mgr = RiskManager(settings)

        market = {"yes_price": 0.31, "no_price": 0.69}
        odds = risk_mgr._extract_odds_for_action("YES", market)

        assert odds == Decimal("0.31")

    def test_extract_no_odds(self):
        """Extract odds for NO action."""
        settings = Settings()
        risk_mgr = RiskManager(settings)

        market = {"yes_price": 0.17, "no_price": 0.83}
        odds = risk_mgr._extract_odds_for_action("NO", market)

        assert odds == Decimal("0.83")

    def test_extract_odds_default_fallback(self):
        """Use 0.50 default if odds missing."""
        settings = Settings()
        risk_mgr = RiskManager(settings)

        market = {}  # Empty market data
        odds = risk_mgr._extract_odds_for_action("YES", market)

        assert odds == Decimal("0.50")

    def test_extract_odds_invalid_action(self):
        """HOLD or invalid action returns 0.50 default."""
        settings = Settings()
        risk_mgr = RiskManager(settings)

        market = {"yes_price": 0.31, "no_price": 0.69}
        odds = risk_mgr._extract_odds_for_action("HOLD", market)

        assert odds == Decimal("0.50")
