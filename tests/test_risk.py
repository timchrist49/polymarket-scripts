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

    def test_extract_odds_none_value(self):
        """Handle explicit None values in market data."""
        settings = Settings()
        risk_mgr = RiskManager(settings)

        # API sometimes returns explicit None
        market = {"yes_price": None, "no_price": 0.69}
        odds = risk_mgr._extract_odds_for_action("YES", market)

        assert odds == Decimal("0.50")  # Should use default, not crash

        # Test NO action with None
        market = {"yes_price": 0.31, "no_price": None}
        odds = risk_mgr._extract_odds_for_action("NO", market)

        assert odds == Decimal("0.50")


class TestPositionSizingWithOdds:
    """Test position sizing with odds adjustment."""

    @pytest.mark.asyncio
    async def test_position_sizing_low_odds_scaled_down(self):
        """Low odds (0.31) should scale down position size."""
        settings = Settings()
        risk_mgr = RiskManager(settings)

        from polymarket.models import TradingDecision
        decision = TradingDecision(
            action="YES",
            confidence=0.85,
            reasoning="test",
            token_id="0x123",
            position_size=Decimal("9.56"),
            stop_loss_threshold=0.30
        )
        market = {"yes_price": 0.31, "no_price": 0.69}

        result = await risk_mgr.validate_decision(
            decision,
            portfolio_value=Decimal("100"),
            market=market
        )

        assert result.approved
        # With 0.31 odds, multiplier is ~0.62x
        # Position should be scaled down
        assert result.adjusted_position < Decimal("9.56")
        assert result.adjusted_position > Decimal("0")

    @pytest.mark.asyncio
    async def test_position_sizing_high_odds_unchanged(self):
        """High odds (0.83) should not scale position size."""
        settings = Settings()
        risk_mgr = RiskManager(settings)

        from polymarket.models import TradingDecision
        decision = TradingDecision(
            action="NO",
            confidence=0.85,
            reasoning="test",
            token_id="0x456",
            position_size=Decimal("5.0"),
            stop_loss_threshold=0.30
        )
        market = {"yes_price": 0.17, "no_price": 0.83}

        result = await risk_mgr.validate_decision(
            decision,
            portfolio_value=Decimal("100"),
            market=market
        )

        assert result.approved
        # With 0.83 odds, multiplier is 1.0x (no scaling)
        # Position should be unchanged
        assert result.adjusted_position == Decimal("5.00")

    @pytest.mark.asyncio
    async def test_position_sizing_below_minimum_odds_rejected(self):
        """Odds < 0.25 should reject the bet entirely."""
        settings = Settings()
        risk_mgr = RiskManager(settings)

        from polymarket.models import TradingDecision
        decision = TradingDecision(
            action="YES",
            confidence=Decimal("0.90"),  # High confidence
            reasoning="test",
            token_id="0x789",
            position_size=Decimal("10.0"),
            stop_loss_threshold=0.30
        )
        market = {"yes_price": 0.20, "no_price": 0.80}  # Very low odds

        result = await risk_mgr.validate_decision(
            decision,
            portfolio_value=Decimal("100"),
            market=market
        )

        assert not result.approved  # Should be rejected
        assert "odds" in result.reason.lower()  # Reason should mention odds
        assert "0.25" in result.reason  # Should mention minimum threshold
