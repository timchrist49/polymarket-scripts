"""Tests for limit order functionality in PolymarketClient."""
import pytest
from unittest.mock import Mock, patch
from polymarket.client import PolymarketClient
from polymarket.exceptions import ValidationError, UpstreamAPIError

@pytest.mark.asyncio
async def test_place_limit_order():
    """Test placing a limit order."""
    client = PolymarketClient()

    with patch.object(client, '_get_clob_client') as mock_clob:
        mock_clob.return_value.post_order = Mock(return_value={
            "orderID": "test-order-123",
            "status": "LIVE"
        })

        result = await client.place_limit_order(
            token_id="test-token",
            side="BUY",
            price=0.55,
            size=10.0,
            tick_size=0.01
        )

        assert result["orderID"] == "test-order-123"
        assert result["status"] == "LIVE"
        mock_clob.return_value.post_order.assert_called_once()


@pytest.mark.asyncio
async def test_check_order_status():
    """Test checking order status."""
    client = PolymarketClient()

    with patch.object(client, '_get_clob_client') as mock_clob:
        mock_clob.return_value.get_order = Mock(return_value={
            "orderID": "test-order-123",
            "status": "MATCHED",
            "fillAmount": "10.0"
        })

        result = await client.check_order_status("test-order-123")

        assert result["status"] == "MATCHED"
        assert result["fillAmount"] == "10.0"


@pytest.mark.asyncio
async def test_cancel_order():
    """Test cancelling an order."""
    client = PolymarketClient()

    with patch.object(client, '_get_clob_client') as mock_clob:
        mock_clob.return_value.cancel = Mock(return_value={
            "orderID": "test-order-123",
            "status": "CANCELLED"
        })

        result = await client.cancel_order("test-order-123")

        assert result["status"] == "CANCELLED"
        mock_clob.return_value.cancel.assert_called_once_with("test-order-123")


# ============================================================================
# CRITICAL ERROR VALIDATION TESTS
# ============================================================================


@pytest.mark.asyncio
async def test_place_limit_order_invalid_price():
    """Price outside 0.0-1.0 should raise ValidationError."""
    client = PolymarketClient()
    with pytest.raises(ValidationError, match="Price must be 0.0-1.0"):
        await client.place_limit_order("test-token-1234567890", "BUY", price=1.5, size=10)


@pytest.mark.asyncio
async def test_place_limit_order_negative_size():
    """Negative size should raise ValidationError."""
    client = PolymarketClient()
    with pytest.raises(ValidationError, match="Size must be positive"):
        await client.place_limit_order("test-token-1234567890", "BUY", price=0.5, size=-10)


@pytest.mark.asyncio
async def test_place_limit_order_invalid_side():
    """Invalid side should raise ValidationError."""
    client = PolymarketClient()
    with pytest.raises(ValidationError, match="Side must be BUY or SELL"):
        await client.place_limit_order("test-token-1234567890", "INVALID", price=0.5, size=10)


@pytest.mark.asyncio
async def test_place_limit_order_invalid_tick_size_zero():
    """Zero tick_size should raise ValidationError."""
    client = PolymarketClient()
    with pytest.raises(ValidationError, match="tick_size must be"):
        await client.place_limit_order("test-token-1234567890", "BUY", price=0.5, size=10, tick_size=0)


@pytest.mark.asyncio
async def test_place_limit_order_invalid_tick_size_negative():
    """Negative tick_size should raise ValidationError."""
    client = PolymarketClient()
    with pytest.raises(ValidationError, match="tick_size must be"):
        await client.place_limit_order("test-token-1234567890", "BUY", price=0.5, size=10, tick_size=-0.01)


@pytest.mark.asyncio
async def test_place_limit_order_invalid_tick_size_too_large():
    """tick_size > 1.0 should raise ValidationError."""
    client = PolymarketClient()
    with pytest.raises(ValidationError, match="tick_size must be"):
        await client.place_limit_order("test-token-1234567890", "BUY", price=0.5, size=10, tick_size=1.5)


@pytest.mark.asyncio
async def test_place_limit_order_invalid_token_id_empty():
    """Empty token_id should raise ValidationError."""
    client = PolymarketClient()
    with pytest.raises(ValidationError, match="Invalid token_id"):
        await client.place_limit_order("", "BUY", price=0.5, size=10)


@pytest.mark.asyncio
async def test_place_limit_order_invalid_token_id_none():
    """None token_id should raise ValidationError."""
    client = PolymarketClient()
    with pytest.raises(ValidationError, match="Invalid token_id"):
        await client.place_limit_order(None, "BUY", price=0.5, size=10)


@pytest.mark.asyncio
async def test_place_limit_order_invalid_token_id_too_short():
    """Token_id shorter than 10 chars should raise ValidationError."""
    client = PolymarketClient()
    with pytest.raises(ValidationError, match="Invalid token_id"):
        await client.place_limit_order("short", "BUY", price=0.5, size=10)


@pytest.mark.asyncio
async def test_check_order_status_empty_id():
    """Empty order_id should raise ValidationError."""
    client = PolymarketClient()
    with pytest.raises(ValidationError, match="Invalid order_id"):
        await client.check_order_status("")


@pytest.mark.asyncio
async def test_check_order_status_none_id():
    """None order_id should raise ValidationError."""
    client = PolymarketClient()
    with pytest.raises(ValidationError, match="Invalid order_id"):
        await client.check_order_status(None)


@pytest.mark.asyncio
async def test_cancel_order_empty_id():
    """Empty order_id should raise ValidationError."""
    client = PolymarketClient()
    with pytest.raises(ValidationError, match="Invalid order_id"):
        await client.cancel_order("")


@pytest.mark.asyncio
async def test_cancel_order_none_id():
    """None order_id should raise ValidationError."""
    client = PolymarketClient()
    with pytest.raises(ValidationError, match="Invalid order_id"):
        await client.cancel_order(None)


@pytest.mark.asyncio
async def test_place_limit_order_api_error():
    """API errors should be wrapped in UpstreamAPIError with chain."""
    client = PolymarketClient()
    with patch.object(client, '_get_clob_client') as mock_clob:
        mock_clob.return_value.post_order = Mock(side_effect=Exception("API down"))

        with pytest.raises(UpstreamAPIError, match="Limit order failed") as exc_info:
            await client.place_limit_order("test-token-1234567890", "BUY", price=0.5, size=10)

        # Verify exception chain is preserved
        assert exc_info.value.__cause__ is not None
        assert str(exc_info.value.__cause__) == "API down"


@pytest.mark.asyncio
async def test_check_order_status_api_error():
    """API errors should be wrapped in UpstreamAPIError with chain."""
    client = PolymarketClient()
    with patch.object(client, '_get_clob_client') as mock_clob:
        mock_clob.return_value.get_order = Mock(side_effect=Exception("Network error"))

        with pytest.raises(UpstreamAPIError, match="Status check failed") as exc_info:
            await client.check_order_status("test-order-123")

        # Verify exception chain is preserved
        assert exc_info.value.__cause__ is not None
        assert str(exc_info.value.__cause__) == "Network error"


@pytest.mark.asyncio
async def test_cancel_order_api_error():
    """API errors should be wrapped in UpstreamAPIError with chain."""
    client = PolymarketClient()
    with patch.object(client, '_get_clob_client') as mock_clob:
        mock_clob.return_value.cancel = Mock(side_effect=Exception("Timeout"))

        with pytest.raises(UpstreamAPIError, match="Cancel failed") as exc_info:
            await client.cancel_order("test-order-123")

        # Verify exception chain is preserved
        assert exc_info.value.__cause__ is not None
        assert str(exc_info.value.__cause__) == "Timeout"


@pytest.mark.asyncio
async def test_place_limit_order_price_rounding():
    """Price should be rounded to tick_size and clamped to valid range."""
    client = PolymarketClient()
    with patch.object(client, '_get_clob_client') as mock_clob:
        mock_clob.return_value.post_order = Mock(return_value={"orderID": "123", "status": "LIVE"})

        await client.place_limit_order("test-token-1234567890", "BUY", price=0.556, size=10, tick_size=0.01)

        # Verify post_order was called with rounded price (0.56, not 0.556)
        call_kwargs = mock_clob.return_value.post_order.call_args.kwargs
        assert call_kwargs["price"] == 0.56


@pytest.mark.asyncio
async def test_place_limit_order_price_clamping_upper():
    """Price rounding that exceeds 1.0 should be clamped."""
    client = PolymarketClient()
    with patch.object(client, '_get_clob_client') as mock_clob:
        mock_clob.return_value.post_order = Mock(return_value={"orderID": "123", "status": "LIVE"})

        # Price very close to 1.0 that might round up due to floating point
        await client.place_limit_order("test-token-1234567890", "BUY", price=0.999999, size=10, tick_size=0.01)

        # Verify price was clamped to 1.0 maximum
        call_kwargs = mock_clob.return_value.post_order.call_args.kwargs
        assert call_kwargs["price"] <= 1.0


@pytest.mark.asyncio
async def test_place_limit_order_boundary_prices():
    """Prices at 0.0 and 1.0 boundaries should work."""
    client = PolymarketClient()
    with patch.object(client, '_get_clob_client') as mock_clob:
        mock_clob.return_value.post_order = Mock(return_value={"orderID": "123", "status": "LIVE"})

        # Should not raise for boundary values
        await client.place_limit_order("test-token-1234567890", "BUY", price=0.0, size=10)
        await client.place_limit_order("test-token-1234567890", "BUY", price=1.0, size=10)

        # Verify both calls succeeded
        assert mock_clob.return_value.post_order.call_count == 2
