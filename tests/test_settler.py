"""Tests for trade settlement service."""

import pytest
from unittest.mock import AsyncMock, Mock
from datetime import datetime, timedelta
from decimal import Decimal
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


class TestDatabaseQuery:
    """Test querying unsettled trades from database."""

    def test_query_unsettled_trades(self):
        """Should return trades that need settlement."""
        db = PerformanceDatabase(":memory:")
        settler = TradeSettler(db, btc_fetcher=None)

        # Insert test trades
        cursor = db.conn.cursor()

        # Trade 1: Old YES trade, not settled
        cursor.execute("""
            INSERT INTO trades (
                timestamp, market_slug, action, confidence, position_size,
                btc_price, price_to_beat, executed_price, is_win
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now() - timedelta(minutes=20),
            "btc-updown-15m-1770828300",
            "YES",
            0.75,
            10.0,
            70000.0,
            70000.0,
            0.65,
            None  # Not settled
        ))

        # Trade 2: Recent trade, too new to settle
        cursor.execute("""
            INSERT INTO trades (
                timestamp, market_slug, action, confidence, position_size,
                btc_price, price_to_beat, executed_price, is_win
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now() - timedelta(minutes=5),
            "btc-updown-15m-1770829000",
            "NO",
            0.80,
            15.0,
            71000.0,
            71000.0,
            0.35,
            None
        ))

        # Trade 3: HOLD action, should be skipped
        cursor.execute("""
            INSERT INTO trades (
                timestamp, market_slug, action, confidence, position_size,
                btc_price, price_to_beat, executed_price, is_win
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now() - timedelta(minutes=20),
            "btc-updown-15m-1770828400",
            "HOLD",
            0.50,
            0.0,
            70500.0,
            70500.0,
            None,
            None
        ))

        # Trade 4: Already settled
        cursor.execute("""
            INSERT INTO trades (
                timestamp, market_slug, action, confidence, position_size,
                btc_price, price_to_beat, executed_price, is_win, profit_loss, actual_outcome
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now() - timedelta(minutes=30),
            "btc-updown-15m-1770828000",
            "YES",
            0.70,
            12.0,
            69000.0,
            69000.0,
            0.60,
            True,
            8.0,
            "YES"
        ))

        db.conn.commit()

        # Query unsettled trades
        trades = settler._get_unsettled_trades(batch_size=10)

        # Should only return Trade 1 (old enough, not settled, not HOLD)
        assert len(trades) == 1
        assert trades[0]['action'] == "YES"
        assert trades[0]['market_slug'] == "btc-updown-15m-1770828300"

    def test_batch_size_limit(self):
        """Should respect batch size limit."""
        db = PerformanceDatabase(":memory:")
        settler = TradeSettler(db, btc_fetcher=None)

        cursor = db.conn.cursor()

        # Insert 5 old unsettled trades
        for i in range(5):
            cursor.execute("""
                INSERT INTO trades (
                    timestamp, market_slug, action, confidence, position_size,
                    btc_price, price_to_beat, executed_price, is_win
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now() - timedelta(minutes=20 + i),
                f"btc-updown-15m-177082{i}000",
                "YES",
                0.75,
                10.0,
                70000.0,
                70000.0,
                0.65,
                None
            ))

        db.conn.commit()

        # Query with batch_size=3
        trades = settler._get_unsettled_trades(batch_size=3)

        # Should only return 3 trades
        assert len(trades) == 3


class TestSettlementOrchestration:
    """Test end-to-end settlement process."""

    @pytest.mark.asyncio
    async def test_settle_pending_trades_success(self):
        """Should settle trades successfully."""
        db = PerformanceDatabase(":memory:")

        # Mock BTC fetcher
        mock_btc_fetcher = Mock()
        mock_btc_fetcher.get_price_at_timestamp = AsyncMock(return_value=Decimal("72000.0"))

        settler = TradeSettler(db, mock_btc_fetcher)

        # Insert test trade
        cursor = db.conn.cursor()
        cursor.execute("""
            INSERT INTO trades (
                timestamp, market_slug, action, confidence, position_size,
                btc_price, price_to_beat, executed_price, is_win
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now() - timedelta(minutes=20),
            "btc-updown-15m-1770828300",
            "YES",
            0.75,
            10.0,
            70000.0,
            70000.0,
            0.65,
            None
        ))
        db.conn.commit()

        # Mock tracker to avoid circular dependency
        mock_tracker = Mock()
        mock_tracker.update_trade_outcome = Mock()
        settler._tracker = mock_tracker

        # Run settlement
        stats = await settler.settle_pending_trades(batch_size=10)

        # Verify stats
        assert stats['success'] is True
        assert stats['settled_count'] == 1
        assert stats['wins'] == 1
        assert stats['losses'] == 0
        assert stats['pending_count'] == 0

        # Verify BTC price was fetched
        mock_btc_fetcher.get_price_at_timestamp.assert_called_once_with(1770828300)

        # Verify database was updated
        mock_tracker.update_trade_outcome.assert_called_once()

    @pytest.mark.asyncio
    async def test_settle_skips_on_price_fetch_failure(self):
        """Should skip trade if BTC price unavailable."""
        db = PerformanceDatabase(":memory:")

        # Mock BTC fetcher that returns None
        mock_btc_fetcher = Mock()
        mock_btc_fetcher.get_price_at_timestamp = AsyncMock(return_value=None)

        settler = TradeSettler(db, mock_btc_fetcher)

        # Insert test trade
        cursor = db.conn.cursor()
        cursor.execute("""
            INSERT INTO trades (
                timestamp, market_slug, action, confidence, position_size,
                btc_price, price_to_beat, executed_price, is_win
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now() - timedelta(minutes=20),
            "btc-updown-15m-1770828300",
            "YES",
            0.75,
            10.0,
            70000.0,
            70000.0,
            0.65,
            None
        ))
        db.conn.commit()

        mock_tracker = Mock()
        mock_tracker.update_trade_outcome = Mock()
        settler._tracker = mock_tracker

        # Run settlement
        stats = await settler.settle_pending_trades(batch_size=10)

        # Should skip (not settle)
        assert stats['settled_count'] == 0
        assert stats['pending_count'] == 1

        # Should not update database
        mock_tracker.update_trade_outcome.assert_not_called()
