"""Tests for trade settlement service."""

import pytest
from polymarket.performance.settler import TradeSettler
from polymarket.performance.database import PerformanceDatabase


class TestTimestampParsing:
    """Test parsing Unix timestamps from market slugs."""

    def test_parse_valid_market_slug(self):
        """Should extract timestamp from valid slug."""
        db = PerformanceDatabase(":memory:")
        settler = TradeSettler(db, btc_fetcher=None)

        timestamp = settler._parse_market_close_timestamp("btc-updown-15m-1770828300")

        assert timestamp == 1770828300

    def test_parse_different_format(self):
        """Should handle variations in slug format."""
        db = PerformanceDatabase(":memory:")
        settler = TradeSettler(db, btc_fetcher=None)

        # Different prefix but same pattern
        timestamp = settler._parse_market_close_timestamp("bitcoin-up-down-1770828900")

        assert timestamp == 1770828900

    def test_parse_invalid_slug_returns_none(self):
        """Should return None for invalid slug."""
        db = PerformanceDatabase(":memory:")
        settler = TradeSettler(db, btc_fetcher=None)

        timestamp = settler._parse_market_close_timestamp("no-timestamp-here")

        assert timestamp is None

    def test_parse_empty_slug_returns_none(self):
        """Should return None for empty slug."""
        db = PerformanceDatabase(":memory:")
        settler = TradeSettler(db, btc_fetcher=None)

        timestamp = settler._parse_market_close_timestamp("")

        assert timestamp is None


class TestOutcomeDetermination:
    """Test determining YES/NO outcome from price comparison."""

    def test_price_up_means_yes_wins(self):
        """When close > start, UP won (YES)."""
        db = PerformanceDatabase(":memory:")
        settler = TradeSettler(db, btc_fetcher=None)

        outcome = settler._determine_outcome(
            btc_close_price=72000.0,
            price_to_beat=70000.0
        )

        assert outcome == "YES"

    def test_price_down_means_no_wins(self):
        """When close < start, DOWN won (NO)."""
        db = PerformanceDatabase(":memory:")
        settler = TradeSettler(db, btc_fetcher=None)

        outcome = settler._determine_outcome(
            btc_close_price=69000.0,
            price_to_beat=70000.0
        )

        assert outcome == "NO"

    def test_price_tie_defaults_to_no(self):
        """When close == start, default to NO (rare case)."""
        db = PerformanceDatabase(":memory:")
        settler = TradeSettler(db, btc_fetcher=None)

        outcome = settler._determine_outcome(
            btc_close_price=70000.0,
            price_to_beat=70000.0
        )

        assert outcome == "NO"

    def test_small_price_increase_still_yes(self):
        """Even small increases count as YES."""
        db = PerformanceDatabase(":memory:")
        settler = TradeSettler(db, btc_fetcher=None)

        outcome = settler._determine_outcome(
            btc_close_price=70001.0,
            price_to_beat=70000.0
        )

        assert outcome == "YES"


class TestProfitLossCalculation:
    """Test profit/loss calculation for Polymarket binary markets."""

    def test_yes_wins_profit_calculation(self):
        """Calculate profit when YES bet wins."""
        db = PerformanceDatabase(":memory:")
        settler = TradeSettler(db, btc_fetcher=None)

        # Bet YES at 0.39 for $11.92
        # Shares: 11.92 / 0.39 = 30.56
        # Payout: 30.56 * $1 = $30.56
        # Profit: $30.56 - $11.92 = $18.64
        profit_loss, is_win = settler._calculate_profit_loss(
            action="YES",
            actual_outcome="YES",
            position_size=11.92,
            executed_price=0.39
        )

        assert is_win is True
        assert abs(profit_loss - 18.64) < 0.01  # Allow tiny float diff

    def test_yes_loses_loss_calculation(self):
        """Calculate loss when YES bet loses."""
        db = PerformanceDatabase(":memory:")
        settler = TradeSettler(db, btc_fetcher=None)

        # Bet YES but NO wins - lose entire position
        profit_loss, is_win = settler._calculate_profit_loss(
            action="YES",
            actual_outcome="NO",
            position_size=11.92,
            executed_price=0.39
        )

        assert is_win is False
        assert profit_loss == -11.92

    def test_no_wins_profit_calculation(self):
        """Calculate profit when NO bet wins."""
        db = PerformanceDatabase(":memory:")
        settler = TradeSettler(db, btc_fetcher=None)

        # Bet NO at 0.11 for $5.00
        # Shares: 5.00 / 0.11 = 45.45
        # Payout: 45.45 * $1 = $45.45
        # Profit: $45.45 - $5.00 = $40.45
        profit_loss, is_win = settler._calculate_profit_loss(
            action="NO",
            actual_outcome="NO",
            position_size=5.0,
            executed_price=0.11
        )

        assert is_win is True
        assert abs(profit_loss - 40.45) < 0.01

    def test_no_loses_loss_calculation(self):
        """Calculate loss when NO bet loses."""
        db = PerformanceDatabase(":memory:")
        settler = TradeSettler(db, btc_fetcher=None)

        # Bet NO but YES wins - lose entire position
        profit_loss, is_win = settler._calculate_profit_loss(
            action="NO",
            actual_outcome="YES",
            position_size=5.0,
            executed_price=0.89
        )

        assert is_win is False
        assert profit_loss == -5.0

    def test_high_confidence_yes_wins(self):
        """Test profit when buying expensive YES shares."""
        db = PerformanceDatabase(":memory:")
        settler = TradeSettler(db, btc_fetcher=None)

        # Bet YES at 0.89 for $10.00
        # Shares: 10.00 / 0.89 = 11.24
        # Payout: 11.24 * $1 = $11.24
        # Profit: $11.24 - $10.00 = $1.24
        profit_loss, is_win = settler._calculate_profit_loss(
            action="YES",
            actual_outcome="YES",
            position_size=10.0,
            executed_price=0.89
        )

        assert is_win is True
        assert abs(profit_loss - 1.24) < 0.01
