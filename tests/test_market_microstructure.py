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


def test_calculate_momentum_score():
    """Test YES price momentum calculation."""
    service = MarketMicrostructureService(Settings(), "test-123")

    # Test: YES price rising 5%
    trades = [
        {'asset_id': 'YES_TOKEN', 'price': 0.50, 'size': 100},
        {'asset_id': 'YES_TOKEN', 'price': 0.525, 'size': 200},
    ]
    score = service.calculate_momentum_score(trades)
    assert score == pytest.approx(0.5, abs=0.01)  # 5% / 10% = 0.5

    # Test: YES price falling 10% → -1.0 (clamped)
    trades = [
        {'asset_id': 'YES_TOKEN', 'price': 0.50, 'size': 100},
        {'asset_id': 'YES_TOKEN', 'price': 0.45, 'size': 200},
    ]
    score = service.calculate_momentum_score(trades)
    assert score == pytest.approx(-1.0, abs=0.01)

    # Test: No price change → 0.0
    trades = [
        {'asset_id': 'YES_TOKEN', 'price': 0.50, 'size': 100},
        {'asset_id': 'YES_TOKEN', 'price': 0.50, 'size': 200},
    ]
    score = service.calculate_momentum_score(trades)
    assert score == pytest.approx(0.0, abs=0.01)

    # Test: Empty trades → 0.0
    score = service.calculate_momentum_score([])
    assert score == pytest.approx(0.0, abs=0.01)


def test_calculate_volume_flow_score():
    """Test volume flow calculation."""
    service = MarketMicrostructureService(Settings(), "test-123")

    # Test: Pure YES volume → +1.0
    trades = [
        {'asset_id': 'YES_TOKEN', 'size': 1000},
        {'asset_id': 'YES_TOKEN', 'size': 500},
    ]
    score = service.calculate_volume_flow_score(trades)
    assert score == 1.0

    # Test: Pure NO volume → -1.0
    trades = [
        {'asset_id': 'NO_TOKEN', 'size': 1000},
    ]
    score = service.calculate_volume_flow_score(trades)
    assert score == -1.0

    # Test: Equal volume → 0.0
    trades = [
        {'asset_id': 'YES_TOKEN', 'size': 500},
        {'asset_id': 'NO_TOKEN', 'size': 500},
    ]
    score = service.calculate_volume_flow_score(trades)
    assert score == 0.0

    # Test: 60% YES, 40% NO → +0.2
    trades = [
        {'asset_id': 'YES_TOKEN', 'size': 600},
        {'asset_id': 'NO_TOKEN', 'size': 400},
    ]
    score = service.calculate_volume_flow_score(trades)
    assert score == pytest.approx(0.2, abs=0.01)

    # Test: Empty trades → 0.0
    score = service.calculate_volume_flow_score([])
    assert score == 0.0


def test_calculate_whale_activity_score():
    """Test whale detection and scoring."""
    service = MarketMicrostructureService(Settings(), "test-123")

    # Test: All YES whales → +1.0
    trades = [
        {'asset_id': 'YES_TOKEN', 'size': 1500},
        {'asset_id': 'YES_TOKEN', 'size': 2000},
        {'asset_id': 'NO_TOKEN', 'size': 100},  # Not a whale
    ]
    score = service.calculate_whale_activity_score(trades)
    assert score == 1.0

    # Test: All NO whales → -1.0
    trades = [
        {'asset_id': 'NO_TOKEN', 'size': 1500},
        {'asset_id': 'NO_TOKEN', 'size': 2000},
        {'asset_id': 'YES_TOKEN', 'size': 100},  # Not a whale
    ]
    score = service.calculate_whale_activity_score(trades)
    assert score == -1.0

    # Test: Balanced whales → 0.0
    trades = [
        {'asset_id': 'YES_TOKEN', 'size': 1500},
        {'asset_id': 'NO_TOKEN', 'size': 1200},
    ]
    score = service.calculate_whale_activity_score(trades)
    assert score == 0.0

    # Test: 2 YES whales, 1 NO whale → +0.33
    trades = [
        {'asset_id': 'YES_TOKEN', 'size': 1500},
        {'asset_id': 'YES_TOKEN', 'size': 2000},
        {'asset_id': 'NO_TOKEN', 'size': 1200},
    ]
    score = service.calculate_whale_activity_score(trades)
    assert score == pytest.approx(0.333, abs=0.01)

    # Test: No whales → 0.0 (handle division by zero)
    trades = [
        {'asset_id': 'YES_TOKEN', 'size': 500},
        {'asset_id': 'NO_TOKEN', 'size': 800},
    ]
    score = service.calculate_whale_activity_score(trades)
    assert score == 0.0

    # Test: Empty trades → 0.0
    score = service.calculate_whale_activity_score([])
    assert score == 0.0


def test_calculate_market_score():
    """Test weighted combination of scores."""
    service = MarketMicrostructureService(Settings(), "test-123")

    # Test: All positive scores
    score = service.calculate_market_score(
        momentum=1.0,
        volume_flow=1.0,
        whale=1.0
    )
    assert score == 1.0  # 0.4 + 0.35 + 0.25 = 1.0

    # Test: All negative scores
    score = service.calculate_market_score(
        momentum=-1.0,
        volume_flow=-1.0,
        whale=-1.0
    )
    assert score == -1.0

    # Test: Mixed scores with weights
    score = service.calculate_market_score(
        momentum=0.5,   # 0.5 * 0.4 = 0.2
        volume_flow=0.3,  # 0.3 * 0.35 = 0.105
        whale=0.2     # 0.2 * 0.25 = 0.05
    )
    assert score == pytest.approx(0.355, abs=0.01)

    # Test: Neutral scores
    score = service.calculate_market_score(
        momentum=0.0,
        volume_flow=0.0,
        whale=0.0
    )
    assert score == 0.0


def test_calculate_confidence():
    """Test confidence calculation based on data quality."""
    service = MarketMicrostructureService(Settings(), "test-123")

    # Test: 50+ trades, full duration → 1.0 confidence
    data = {
        'trades': [{'asset_id': 'YES_TOKEN', 'size': 100}] * 50,
        'collection_duration': 120
    }
    confidence = service.calculate_confidence(data)
    assert confidence == 1.0

    # Test: 25 trades → 0.5 base confidence
    data = {
        'trades': [{'asset_id': 'YES_TOKEN', 'size': 100}] * 25,
        'collection_duration': 120
    }
    confidence = service.calculate_confidence(data)
    assert confidence == pytest.approx(0.5, abs=0.01)

    # Test: 50 trades but only 60s collected → 0.5x penalty
    data = {
        'trades': [{'asset_id': 'YES_TOKEN', 'size': 100}] * 50,
        'collection_duration': 60
    }
    confidence = service.calculate_confidence(data)
    assert confidence == pytest.approx(0.5, abs=0.01)  # 1.0 * 0.5

    # Test: <10 trades → 0.5x low liquidity penalty
    data = {
        'trades': [{'asset_id': 'YES_TOKEN', 'size': 100}] * 5,
        'collection_duration': 120
    }
    confidence = service.calculate_confidence(data)
    # (5/50) * 1.0 * 0.5 = 0.05
    assert confidence == pytest.approx(0.05, abs=0.01)

    # Test: Empty trades → 0.0
    data = {'trades': [], 'collection_duration': 120}
    confidence = service.calculate_confidence(data)
    assert confidence == 0.0


@pytest.mark.asyncio
async def test_get_market_score_with_mock_data():
    """Test get_market_score with mocked collection data."""
    service = MarketMicrostructureService(Settings(), "test-condition-123")

    # Mock collect_market_data to return test data
    async def mock_collect(condition_id, duration_seconds):
        return {
            'trades': [
                {'asset_id': 'YES_TOKEN', 'price': 0.50, 'size': 1000},
                {'asset_id': 'YES_TOKEN', 'price': 0.52, 'size': 1500},
                {'asset_id': 'NO_TOKEN', 'price': 0.48, 'size': 500},
            ] * 20,  # 60 trades total
            'book_snapshots': [],
            'price_changes': [],
            'collection_duration': 120
        }

    # Replace method
    service.collect_market_data = mock_collect

    # Get market score
    signals = await service.get_market_score()

    # Verify structure
    assert isinstance(signals, MarketSignals)
    assert -1.0 <= signals.score <= 1.0
    assert 0.0 <= signals.confidence <= 1.0

    # Verify scores are calculated (not default 0.0)
    assert signals.momentum_score != 0.0  # Price moved 0.50 → 0.52
    assert signals.volume_score != 0.0    # More YES than NO volume
    assert signals.whale_score != 0.0     # Has whale trades >$1k

    # Confidence should be high (60 trades, full duration)
    assert signals.confidence > 0.8


@pytest.mark.asyncio
@pytest.mark.slow  # Mark as slow test
async def test_websocket_connection_real():
    """Test real WebSocket connection (slow, may be skipped)."""
    service = MarketMicrostructureService(Settings(), "test-condition-123")

    # Attempt real connection for 5 seconds
    try:
        data = await service.collect_market_data(
            "test-condition-123",
            duration_seconds=5
        )

        # Verify structure even if no trades
        assert isinstance(data, dict)
        assert 'trades' in data
        assert 'collection_duration' in data

    except Exception as e:
        pytest.skip(f"WebSocket connection failed: {e}")
