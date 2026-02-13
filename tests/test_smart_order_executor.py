"""Tests for smart order executor."""
import pytest
from unittest.mock import AsyncMock, Mock, patch
from polymarket.trading.smart_order_executor import SmartOrderExecutor
from polymarket.models import LimitOrderStrategy

@pytest.mark.asyncio
async def test_execute_high_urgency_order():
    """Test execution of high urgency order (aggressive pricing)."""
    executor = SmartOrderExecutor()

    # Mock client
    mock_client = Mock()
    mock_client.place_limit_order = AsyncMock(return_value={"orderID": "test-123", "status": "LIVE"})
    mock_client.check_order_status = AsyncMock(return_value={"status": "MATCHED", "fillAmount": "10.0"})

    result = await executor.execute_smart_order(
        client=mock_client,
        token_id="test-token",
        side="BUY",
        amount=10.0,
        urgency="HIGH",
        current_best_ask=0.550,
        current_best_bid=0.540,
        tick_size=0.001
    )

    assert result["status"] == "FILLED"
    assert result["order_id"] == "test-123"

    # Verify limit order was placed (not market order)
    mock_client.place_limit_order.assert_called_once()

    # Verify price was improved (aggressive = +0.1%)
    call_args = mock_client.place_limit_order.call_args
    placed_price = call_args.kwargs["price"]
    assert placed_price > 0.540  # Better than best bid
    assert placed_price <= 0.551  # But close to ask (aggressive)


@pytest.mark.asyncio
async def test_execute_medium_urgency_order():
    """Test execution of medium urgency order (moderate pricing)."""
    executor = SmartOrderExecutor()

    mock_client = Mock()
    mock_client.place_limit_order = AsyncMock(return_value={"orderID": "test-456", "status": "LIVE"})
    mock_client.check_order_status = AsyncMock(return_value={"status": "MATCHED", "fillAmount": "10.0"})

    result = await executor.execute_smart_order(
        client=mock_client,
        token_id="test-token",
        side="BUY",
        amount=10.0,
        urgency="MEDIUM",
        current_best_ask=0.550,
        current_best_bid=0.540,
        tick_size=0.001
    )

    assert result["status"] == "FILLED"

    # Verify price improvement is moderate (+0.3%)
    call_args = mock_client.place_limit_order.call_args
    placed_price = call_args.kwargs["price"]
    assert 0.541 <= placed_price <= 0.543


@pytest.mark.asyncio
async def test_order_timeout_with_fallback():
    """Test that timeout triggers fallback to market order."""
    executor = SmartOrderExecutor()

    mock_client = Mock()
    mock_client.place_limit_order = AsyncMock(return_value={"orderID": "test-789", "status": "LIVE"})

    # Order never fills (timeout scenario)
    mock_client.check_order_status = AsyncMock(return_value={"status": "LIVE", "fillAmount": "0.0"})
    mock_client.cancel_order = AsyncMock(return_value={"status": "CANCELLED"})

    # Mock create_order for market order fallback (NOTE: create_order is sync, not async)
    from polymarket.models import OrderResponse
    mock_response = OrderResponse(
        order_id="market-123",
        status="posted",
        accepted=True,
        raw_response={}
    )
    mock_client.create_order = Mock(return_value=mock_response)

    # HIGH urgency with short timeout
    with patch('asyncio.sleep', new_callable=AsyncMock):
        result = await executor.execute_smart_order(
            client=mock_client,
            token_id="test-token",
            side="BUY",
            amount=10.0,
            urgency="HIGH",
            current_best_ask=0.550,
            current_best_bid=0.540,
            tick_size=0.001,
            timeout_override=1  # 1 second for testing
        )

    # Should fallback to market order
    assert result["status"] == "FILLED"
    assert result["filled_via"] == "market"  # Used fallback
    assert result["order_id"] == "market-123"
    mock_client.cancel_order.assert_called_once()
    mock_client.create_order.assert_called_once()


@pytest.mark.asyncio
async def test_low_urgency_no_fallback():
    """Test that low urgency orders don't fallback on timeout."""
    executor = SmartOrderExecutor()

    mock_client = Mock()
    mock_client.place_limit_order = AsyncMock(return_value={"orderID": "test-999", "status": "LIVE"})
    mock_client.check_order_status = AsyncMock(return_value={"status": "LIVE", "fillAmount": "0.0"})
    mock_client.cancel_order = AsyncMock(return_value={"status": "CANCELLED"})

    with patch('asyncio.sleep', new_callable=AsyncMock):
        result = await executor.execute_smart_order(
            client=mock_client,
            token_id="test-token",
            side="BUY",
            amount=10.0,
            urgency="LOW",
            current_best_ask=0.550,
            current_best_bid=0.540,
            tick_size=0.001,
            timeout_override=1
        )

    # Should NOT fallback (low urgency)
    assert result["status"] == "TIMEOUT"
    mock_client.cancel_order.assert_called_once()
    # Market order should NOT be placed
    assert not hasattr(mock_client, 'create_order') or \
           mock_client.create_order.call_count == 0


@pytest.mark.asyncio
async def test_sell_side_order():
    """Test execution of SELL side order."""
    executor = SmartOrderExecutor()

    mock_client = Mock()
    mock_client.place_limit_order = AsyncMock(return_value={"orderID": "sell-123", "status": "LIVE"})
    mock_client.check_order_status = AsyncMock(return_value={"status": "MATCHED", "fillAmount": "10.0"})

    result = await executor.execute_smart_order(
        client=mock_client,
        token_id="test-token",
        side="SELL",
        amount=10.0,
        urgency="MEDIUM",
        current_best_ask=0.550,
        current_best_bid=0.540,
        tick_size=0.001
    )

    assert result["status"] == "FILLED"

    # Verify SELL order placed
    call_args = mock_client.place_limit_order.call_args
    assert call_args.kwargs["side"] == "SELL"

    # SELL orders should be priced below best ask (to get filled)
    placed_price = call_args.kwargs["price"]
    assert placed_price < 0.550
