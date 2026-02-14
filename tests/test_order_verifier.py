"""Unit tests for OrderVerifier."""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from polymarket.performance.order_verifier import OrderVerifier


class TestOrderVerifier:
    """Test order verification logic."""

    @pytest.fixture
    def mock_client(self):
        """Mock Polymarket client."""
        client = Mock()
        client.check_order_status = AsyncMock()
        return client

    @pytest.fixture
    def mock_db(self):
        """Mock database."""
        return Mock()

    @pytest.fixture
    def verifier(self, mock_client, mock_db):
        """Create OrderVerifier instance."""
        return OrderVerifier(mock_client, mock_db)

    @pytest.mark.asyncio
    async def test_quick_check_filled(self, verifier, mock_client):
        """Test quick check when order is filled."""
        mock_client.check_order_status.return_value = {
            'status': 'MATCHED',
            'fillAmount': '10.5',
            'price': '0.65'
        }

        result = await verifier.check_order_quick('order_123', trade_id=1)

        assert result['status'] == 'filled'
        assert result['fill_amount'] == 10.5
        assert result['needs_alert'] == False

    @pytest.mark.asyncio
    async def test_quick_check_partial_fill(self, verifier, mock_client):
        """Test quick check with partial fill."""
        mock_client.check_order_status.return_value = {
            'status': 'PARTIALLY_MATCHED',
            'fillAmount': '5.0',
            'size': '10.0'
        }

        result = await verifier.check_order_quick('order_123', trade_id=1)

        assert result['status'] == 'filled'
        assert result['needs_alert'] == True  # Alert for partial fill

    @pytest.mark.asyncio
    async def test_quick_check_failed(self, verifier, mock_client):
        """Test quick check when order fails."""
        mock_client.check_order_status.return_value = {
            'status': 'CANCELLED',
            'fillAmount': '0'
        }

        result = await verifier.check_order_quick('order_123', trade_id=1)

        assert result['status'] == 'failed'
        assert result['needs_alert'] == True

    @pytest.mark.asyncio
    async def test_quick_check_timeout(self, verifier, mock_client):
        """Test quick check with timeout."""
        # Simulate timeout
        async def slow_response(*args, **kwargs):
            await asyncio.sleep(5)  # Longer than 2s timeout
            return {}

        mock_client.check_order_status = slow_response

        result = await verifier.check_order_quick('order_123', trade_id=1, timeout=0.1)

        assert result['status'] == 'pending'
        assert result['raw_status'] == 'TIMEOUT'

    @pytest.mark.asyncio
    async def test_verify_order_full_success(self, verifier, mock_client):
        """Test full verification with successful order."""
        mock_client.check_order_status.return_value = {
            'status': 'MATCHED',
            'fillAmount': '10.0',
            'size': '10.0',
            'price': '0.65',
            'timestamp': 1707955200
        }

        result = await verifier.verify_order_full('order_123')

        assert result['verified'] == True
        assert result['fill_amount'] == 10.0
        assert result['fill_price'] == 0.65
        assert result['partial_fill'] == False

    @pytest.mark.asyncio
    async def test_verify_order_full_partial(self, verifier, mock_client):
        """Test full verification with partial fill."""
        mock_client.check_order_status.return_value = {
            'status': 'PARTIALLY_MATCHED',
            'fillAmount': '7.0',
            'size': '10.0',
            'price': '0.65',
            'timestamp': 1707955200
        }

        result = await verifier.verify_order_full('order_123')

        assert result['verified'] == True
        assert result['fill_amount'] == 7.0
        assert result['partial_fill'] == True
        assert result['original_size'] == 10.0

    @pytest.mark.asyncio
    async def test_verify_order_full_not_found(self, verifier, mock_client):
        """Test full verification when order not found."""
        mock_client.check_order_status.return_value = {
            'status': 'CANCELLED',
            'fillAmount': '0',
            'size': '10.0'
        }

        result = await verifier.verify_order_full('order_123')

        assert result['verified'] == False
        assert result['fill_amount'] == 0.0

    def test_calculate_price_discrepancy(self, verifier):
        """Test price discrepancy calculation."""
        # Paid more than expected
        discrepancy = verifier.calculate_price_discrepancy(
            estimated_price=0.60,
            actual_price=0.65
        )
        assert abs(discrepancy - 8.33) < 0.1  # ~8.33% higher

        # Paid less than expected (favorable)
        discrepancy = verifier.calculate_price_discrepancy(
            estimated_price=0.65,
            actual_price=0.60
        )
        assert abs(discrepancy - (-7.69)) < 0.1  # ~7.69% lower
