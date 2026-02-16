# tests/test_realtime_odds_streamer.py
import pytest
import asyncio
from datetime import datetime
from polymarket.models import WebSocketOddsSnapshot


def test_websocket_odds_snapshot_creation():
    """Test WebSocketOddsSnapshot can be created with required fields."""
    snapshot = WebSocketOddsSnapshot(
        market_id="test-market-123",
        yes_odds=0.65,
        no_odds=0.35,
        timestamp=datetime.now(),
        best_bid=0.65,
        best_ask=0.35
    )

    assert snapshot.market_id == "test-market-123"
    assert snapshot.yes_odds == 0.65
    assert snapshot.no_odds == 0.35
    assert snapshot.best_bid == 0.65
    assert snapshot.best_ask == 0.35
    assert isinstance(snapshot.timestamp, datetime)


def test_websocket_odds_snapshot_validation():
    """Test WebSocketOddsSnapshot validates odds sum to 1.0."""
    with pytest.raises(ValueError, match="must sum to 1.0"):
        WebSocketOddsSnapshot(
            market_id="test-market-123",
            yes_odds=0.65,
            no_odds=0.40,  # Invalid: sums to 1.05
            timestamp=datetime.now(),
            best_bid=0.65,
            best_ask=0.40
        )


# === Task 2: RealtimeOddsStreamer Tests ===


@pytest.mark.asyncio
async def test_streamer_initialization():
    """Test RealtimeOddsStreamer can be instantiated."""
    from polymarket.trading.realtime_odds_streamer import RealtimeOddsStreamer
    from polymarket.client import PolymarketClient

    client = PolymarketClient()
    streamer = RealtimeOddsStreamer(client)

    assert streamer is not None
    assert streamer._current_odds == {}
    assert streamer._ws is None


@pytest.mark.asyncio
async def test_get_current_odds_returns_none_initially():
    """Test get_current_odds returns None when no data."""
    from polymarket.trading.realtime_odds_streamer import RealtimeOddsStreamer
    from polymarket.client import PolymarketClient

    client = PolymarketClient()
    streamer = RealtimeOddsStreamer(client)

    odds = streamer.get_current_odds("test-market-123")

    assert odds is None


# === Task 3: Odds Extraction Tests ===


@pytest.mark.asyncio
async def test_process_book_message_extracts_odds():
    """Test _process_book_message correctly extracts YES/NO odds."""
    from polymarket.trading.realtime_odds_streamer import RealtimeOddsStreamer
    from polymarket.client import PolymarketClient

    client = PolymarketClient()
    streamer = RealtimeOddsStreamer(client)

    # Mock book message payload
    payload = {
        'market': 'test-market-123',
        'asset_id': 'token-yes',
        'bids': [
            ['0.65', '100'],  # Best bid (YES odds)
            ['0.64', '200']
        ],
        'asks': [
            ['0.66', '150'],
            ['0.67', '100']
        ]
    }

    await streamer._process_book_message(payload)

    # Check odds were stored
    odds = streamer.get_current_odds('test-market-123')
    assert odds is not None
    assert odds.market_id == 'test-market-123'
    assert odds.yes_odds == 0.65
    assert odds.no_odds == 0.35
    assert odds.best_bid == 0.65
    assert odds.best_ask == 0.35
    assert isinstance(odds.timestamp, datetime)


@pytest.mark.asyncio
async def test_process_book_message_handles_empty_bids():
    """Test _process_book_message handles empty bids gracefully."""
    from polymarket.trading.realtime_odds_streamer import RealtimeOddsStreamer
    from polymarket.client import PolymarketClient

    client = PolymarketClient()
    streamer = RealtimeOddsStreamer(client)

    payload = {
        'market': 'test-market-123',
        'asset_id': 'token-yes',
        'bids': [],  # Empty
        'asks': [['0.55', '100']]
    }

    await streamer._process_book_message(payload)

    # Should default to 0.50
    odds = streamer.get_current_odds('test-market-123')
    assert odds is not None
    assert odds.yes_odds == 0.50
    assert odds.no_odds == 0.50
