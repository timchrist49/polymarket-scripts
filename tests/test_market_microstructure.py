import pytest
import asyncio
import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from polymarket.trading.market_microstructure import MarketMicrostructureService
from polymarket.config import Settings
from polymarket.models import MarketSignals


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
    assert score == 1.0  # 0.2 + 0.5 + 0.3 = 1.0

    # Test: All negative scores
    score = service.calculate_market_score(
        momentum=-1.0,
        volume_flow=-1.0,
        whale=-1.0
    )
    assert score == -1.0

    # Test: Mixed scores with weights
    score = service.calculate_market_score(
        momentum=0.5,   # 0.5 * 0.2 = 0.1
        volume_flow=0.3,  # 0.3 * 0.5 = 0.15
        whale=0.2     # 0.2 * 0.3 = 0.06
    )
    assert score == pytest.approx(0.31, abs=0.01)

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


@pytest.mark.asyncio
async def test_websocket_subscription_uses_correct_clob_format():
    """
    Test that WebSocket subscription uses correct CLOB format with token IDs.

    CLOB WebSocket expects:
    {
        "type": "MARKET",
        "assets_ids": ["token_id_1", "token_id_2"]
    }

    NOT the RTDS format:
    {
        "action": "subscribe",
        "subscriptions": [{
            "topic": "market",
            "condition_id": "..."
        }]
    }
    """
    service = MarketMicrostructureService(Settings(), "test-condition-123")

    # Token IDs to use for subscription
    token_ids = [
        "75436921419096805904008583680333623108653517040192373569717437397077840910753",
        "2328222210416595233000723821528707301486180950666654837266138688670056541763"
    ]

    # Mock websockets.connect to capture what message is sent
    sent_messages = []

    mock_ws = AsyncMock()
    mock_ws.send = AsyncMock(side_effect=lambda msg: sent_messages.append(msg))
    mock_ws.recv = AsyncMock(side_effect=asyncio.TimeoutError())  # Timeout immediately
    mock_ws.__aenter__ = AsyncMock(return_value=mock_ws)
    mock_ws.__aexit__ = AsyncMock(return_value=None)

    with patch('websockets.connect', return_value=mock_ws):
        try:
            # This will timeout, but we just want to check the subscription message
            await service.collect_market_data_with_token_ids(
                token_ids=token_ids,
                duration_seconds=1
            )
        except:
            pass  # Expected to fail, we're just checking the message

    # Verify a message was sent
    assert len(sent_messages) > 0, "No WebSocket message was sent"

    # Parse the message
    message = json.loads(sent_messages[0])

    # Verify CLOB format (not RTDS format)
    assert "type" in message, "Message should have 'type' field (CLOB format)"
    assert message["type"] == "market", "Message type should be 'market' (lowercase)"
    assert "assets_ids" in message, "Message should have 'assets_ids' field"
    assert message["assets_ids"] == token_ids, "Message should contain correct token IDs"

    # Verify it's NOT using RTDS format
    assert "action" not in message, "Message should NOT have 'action' field (RTDS format)"
    assert "subscriptions" not in message, "Message should NOT have 'subscriptions' field (RTDS format)"
    assert "topic" not in message, "Message should NOT have 'topic' field (RTDS format)"
    assert "condition_id" not in message, "Message should NOT have 'condition_id' field"
