# tests/test_models.py
import pytest
from datetime import datetime
from polymarket.models import Market, OrderRequest, OrderResponse, TokenInfo

def test_market_model_minimal():
    """Test Market model with minimal required fields."""
    market = Market(
        id="0x123",
        condition_id="0x456",
    )
    assert market.id == "0x123"
    assert market.condition_id == "0x456"
    assert market.question is None

def test_market_model_with_outcomes():
    """Test Market model with outcomes parsing."""
    market = Market(
        id="0x123",
        condition_id="0x456",
        question="BTC will go up?",
        outcomes=["Yes", "No"],
        clob_token_ids='["0xaaa", "0xbbb"]',
    )
    assert market.question == "BTC will go up?"
    assert market.outcomes == ["Yes", "No"]

def test_order_request_validation():
    """Test OrderRequest validation."""
    # Valid buy order
    order = OrderRequest(
        token_id="0x123",
        side="BUY",
        price=0.55,
        size=10.0,
    )
    assert order.side == "BUY"

def test_order_request_invalid_side():
    """Test OrderRequest rejects invalid side."""
    with pytest.raises(ValueError):
        OrderRequest(
            token_id="0x123",
            side="INVALID",
            price=0.55,
            size=10.0,
        )

def test_order_request_invalid_price():
    """Test OrderRequest rejects invalid price."""
    with pytest.raises(ValueError):
        OrderRequest(
            token_id="0x123",
            side="BUY",
            price=1.5,  # Must be 0-1
            size=10.0,
        )
