import pytest
import asyncio
from datetime import datetime
from polymarket.trading.market_microstructure import MarketMicrostructureService
from polymarket.config import Settings
from polymarket.models import MarketSignals


@pytest.mark.asyncio
async def test_fetch_order_book():
    """Test Binance order book API."""
    service = MarketMicrostructureService(Settings())

    order_book = await service._fetch_order_book()

    assert "bids" in order_book
    assert "asks" in order_book
    assert len(order_book["bids"]) > 0
    assert len(order_book["asks"]) > 0


@pytest.mark.asyncio
async def test_fetch_recent_trades():
    """Test Binance recent trades API."""
    service = MarketMicrostructureService(Settings())

    trades = await service._fetch_recent_trades()

    assert isinstance(trades, list)
    assert len(trades) > 0
    assert "qty" in trades[0]
    assert "isBuyerMaker" in trades[0]


@pytest.mark.asyncio
async def test_fetch_24hr_ticker():
    """Test Binance 24hr ticker API."""
    service = MarketMicrostructureService(Settings())

    ticker = await service._fetch_24hr_ticker()

    assert "volume" in ticker
    assert "count" in ticker


@pytest.mark.asyncio
async def test_score_order_book():
    """Test order book scoring logic."""
    service = MarketMicrostructureService(Settings())

    # Mock order book with heavy bid walls
    order_book = {
        "bids": [["68000", "15.5"], ["67900", "12.0"]],  # Large bids
        "asks": [["68100", "2.1"], ["68200", "1.5"]]     # Small asks
    }

    score = service._score_order_book(order_book)

    assert score > 0.5  # Should be bullish


@pytest.mark.asyncio
async def test_score_whale_activity():
    """Test whale detection scoring."""
    service = MarketMicrostructureService(Settings())

    # Mock trades with whale buying
    trades = [
        {"qty": "8.5", "isBuyerMaker": True},   # Large buy
        {"qty": "7.0", "isBuyerMaker": True},   # Large buy
        {"qty": "1.0", "isBuyerMaker": False}   # Small sell
    ]

    score = service._score_whale_activity(trades)

    assert score > 0.6  # Should be bullish


@pytest.mark.asyncio
async def test_get_market_score():
    """Test full market microstructure scoring."""
    service = MarketMicrostructureService(Settings())

    signals = await service.get_market_score()

    assert isinstance(signals, MarketSignals)
    assert -1.0 <= signals.score <= 1.0
    assert 0.0 <= signals.confidence <= 1.0
    assert signals.whale_count >= 0

    await service.close()


@pytest.mark.asyncio
async def test_collect_market_data_structure():
    """Test that collect_market_data returns expected structure."""
    service = MarketMicrostructureService(
        Settings(),
        condition_id="test-condition-123"
    )

    # Mock WebSocket connection (will implement properly later)
    data = await service.collect_market_data(
        condition_id="test-condition-123",
        duration_seconds=1  # Short duration for test
    )

    # Verify structure
    assert isinstance(data, dict)
    assert 'trades' in data
    assert 'book_snapshots' in data
    assert 'price_changes' in data
    assert isinstance(data['trades'], list)
