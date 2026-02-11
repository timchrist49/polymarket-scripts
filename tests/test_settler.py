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
