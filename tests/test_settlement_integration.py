"""Integration tests for settlement with order verification."""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, MagicMock
from decimal import Decimal
from polymarket.performance.settler import TradeSettler
from polymarket.performance.order_verifier import OrderVerifier
from polymarket.performance.database import PerformanceDatabase


class TestSettlementIntegration:
    """Integration tests for settlement with verification."""

    @pytest.fixture
    def mock_db(self):
        """Mock database with in-memory SQLite."""
        db = PerformanceDatabase(":memory:")

        # Insert test trade
        cursor = db.conn.cursor()
        cursor.execute("""
            INSERT INTO trades (
                timestamp, market_slug, action, confidence, position_size,
                btc_price, price_to_beat, executed_price,
                order_id, execution_status, is_test_mode
            ) VALUES (
                datetime('now', '-20 minutes'), 'btc-updown-15m-1707955200',
                'YES', 0.85, 10.0, 100000.0, 99500.0, 0.65,
                'order_123', 'executed', 0
            )
        """)
        db.conn.commit()

        return db

    @pytest.fixture
    def mock_btc_fetcher(self):
        """Mock BTC price fetcher."""
        fetcher = Mock()
        fetcher.get_price_at_timestamp = AsyncMock(return_value=Decimal("100500.0"))
        return fetcher

    @pytest.fixture
    def mock_client(self):
        """Mock Polymarket client."""
        client = Mock()
        client.check_order_status = AsyncMock(return_value={
            'status': 'MATCHED',
            'fillAmount': '10.0',
            'size': '10.0',
            'price': '0.66',  # Slightly different from estimated 0.65
            'timestamp': 1707955200
        })
        return client

    @pytest.fixture
    def verifier(self, mock_client, mock_db):
        """Create OrderVerifier."""
        return OrderVerifier(mock_client, mock_db)

    @pytest.fixture
    def settler(self, mock_db, mock_btc_fetcher, verifier):
        """Create TradeSettler with verification."""
        return TradeSettler(mock_db, mock_btc_fetcher, verifier)

    @pytest.mark.asyncio
    async def test_settlement_with_verification(self, settler, mock_db):
        """Test full settlement flow with order verification."""
        stats = await settler.settle_pending_trades(batch_size=10)

        # Check settlement stats
        assert stats['success'] == True
        assert stats['settled_count'] == 1
        assert stats['wins'] == 1  # BTC went up (100500 > 99500)
        assert stats['verification_failures'] == 0

        # Verify database was updated with verification data
        cursor = mock_db.conn.cursor()
        cursor.execute("SELECT * FROM trades WHERE order_id = 'order_123'")
        trade = cursor.fetchone()

        assert trade['verified_fill_price'] == 0.66
        assert trade['verified_fill_amount'] == 10.0
        assert trade['verification_status'] == 'verified'
        assert trade['is_win'] == 1
        assert trade['profit_loss'] > 0  # Winning trade

    @pytest.mark.asyncio
    async def test_settlement_with_failed_verification(self, mock_db, mock_btc_fetcher, mock_client):
        """Test settlement when order was not filled."""
        # Mock order as cancelled
        mock_client.check_order_status = AsyncMock(return_value={
            'status': 'CANCELLED',
            'fillAmount': '0',
            'size': '10.0'
        })

        verifier = OrderVerifier(mock_client, mock_db)
        settler = TradeSettler(mock_db, mock_btc_fetcher, verifier)

        stats = await settler.settle_pending_trades(batch_size=10)

        # Check stats
        assert stats['verification_failures'] == 1
        assert stats['settled_count'] == 0  # Should not settle failed orders

        # Verify database marked as failed
        cursor = mock_db.conn.cursor()
        cursor.execute("SELECT * FROM trades WHERE order_id = 'order_123'")
        trade = cursor.fetchone()

        assert trade['verification_status'] == 'failed'
        assert trade['is_win'] is None  # Not settled

    @pytest.mark.asyncio
    async def test_settlement_with_price_discrepancy(self, mock_db, mock_btc_fetcher, mock_client):
        """Test settlement with large price discrepancy."""
        # Mock order filled at much worse price
        mock_client.check_order_status = AsyncMock(return_value={
            'status': 'MATCHED',
            'fillAmount': '10.0',
            'size': '10.0',
            'price': '0.72',  # 10.8% worse than estimated 0.65
            'timestamp': 1707955200
        })

        verifier = OrderVerifier(mock_client, mock_db)
        settler = TradeSettler(mock_db, mock_btc_fetcher, verifier)

        stats = await settler.settle_pending_trades(batch_size=10)

        # Check stats
        assert stats['price_discrepancies'] == 1
        assert stats['settled_count'] == 1  # Still settle, just alert

        # Verify discrepancy was recorded
        cursor = mock_db.conn.cursor()
        cursor.execute("SELECT * FROM trades WHERE order_id = 'order_123'")
        trade = cursor.fetchone()

        assert trade['price_discrepancy_pct'] > 5.0  # Above alert threshold
        assert trade['verified_fill_price'] == 0.72

    @pytest.mark.asyncio
    async def test_settlement_with_partial_fill(self, mock_db, mock_btc_fetcher, mock_client):
        """Test settlement with partial fill."""
        # Mock partial fill (70% filled)
        mock_client.check_order_status = AsyncMock(return_value={
            'status': 'PARTIALLY_MATCHED',
            'fillAmount': '7.0',
            'size': '10.0',
            'price': '0.65',
            'timestamp': 1707955200
        })

        verifier = OrderVerifier(mock_client, mock_db)
        settler = TradeSettler(mock_db, mock_btc_fetcher, verifier)

        stats = await settler.settle_pending_trades(batch_size=10)

        # Check stats
        assert stats['partial_fills'] == 1
        assert stats['settled_count'] == 1

        # Verify partial fill recorded
        cursor = mock_db.conn.cursor()
        cursor.execute("SELECT * FROM trades WHERE order_id = 'order_123'")
        trade = cursor.fetchone()

        assert trade['partial_fill'] == 1
        assert trade['verified_fill_amount'] == 7.0  # Only 7 shares filled
        # P&L should be calculated on 7 shares, not 10
