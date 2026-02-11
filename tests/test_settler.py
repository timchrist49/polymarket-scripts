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
