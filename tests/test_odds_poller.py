"""
Tests for market odds poller.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime

from polymarket.trading.odds_poller import MarketOddsPoller
from polymarket.models import OddsSnapshot, Market


@pytest.fixture
def mock_client():
    """Create mock Polymarket client."""
    client = MagicMock()
    client.discover_btc_15min_market = MagicMock()
    client.get_market_by_slug = MagicMock()
    return client


@pytest.fixture
def poller(mock_client):
    """Create odds poller with mock client."""
    return MarketOddsPoller(mock_client)


@pytest.mark.asyncio
async def test_odds_polling_basic(poller, mock_client):
    """Test basic odds polling and storage."""
    # Setup mocks
    mock_market = Market(
        id="market-123",
        slug="btc-updown-15m-1771234500",
        question="BTC Up or Down?",
        active=True,
        end_date=datetime.now(),
        best_bid=0.82,  # 82% YES odds
        best_ask=0.83,
        outcomes=["Up", "Down"],
        conditionId="test-condition"
    )

    mock_client.discover_btc_15min_market.return_value = mock_market
    mock_client.get_market_by_slug.return_value = mock_market

    # Poll once
    await poller._poll_current_market()

    # Verify snapshot stored
    snapshot = await poller.get_odds("market-123")
    assert snapshot is not None
    assert snapshot.market_id == "market-123"
    assert snapshot.market_slug == "btc-updown-15m-1771234500"
    assert snapshot.yes_odds == pytest.approx(0.82)
    assert snapshot.no_odds == pytest.approx(0.18)  # 1 - 0.82
    assert snapshot.yes_qualifies is True  # > 0.75
    assert snapshot.no_qualifies is False  # < 0.75


@pytest.mark.asyncio
async def test_odds_polling_threshold_yes_qualifies(poller, mock_client):
    """Test YES qualifies when > 75%."""
    mock_market = Market(
        id="market-456",
        slug="btc-updown-15m-1771234600",
        question="BTC Up or Down?",
        active=True,
        end_date=datetime.now(),
        best_bid=0.80,  # 80% YES odds
        best_ask=0.81,
        outcomes=["Up", "Down"],
        conditionId="test-condition"
    )

    mock_client.discover_btc_15min_market.return_value = mock_market
    mock_client.get_market_by_slug.return_value = mock_market

    await poller._poll_current_market()

    snapshot = await poller.get_odds("market-456")
    assert snapshot.yes_odds == pytest.approx(0.80)
    assert snapshot.no_odds == pytest.approx(0.20)
    assert snapshot.yes_qualifies is True  # > 0.75
    assert snapshot.no_qualifies is False


@pytest.mark.asyncio
async def test_odds_polling_threshold_no_qualifies(poller, mock_client):
    """Test NO qualifies when > 75%."""
    mock_market = Market(
        id="market-789",
        slug="btc-updown-15m-1771234700",
        question="BTC Up or Down?",
        active=True,
        end_date=datetime.now(),
        best_bid=0.20,  # 20% YES odds -> 80% NO odds
        best_ask=0.21,
        outcomes=["Up", "Down"],
        conditionId="test-condition"
    )

    mock_client.discover_btc_15min_market.return_value = mock_market
    mock_client.get_market_by_slug.return_value = mock_market

    await poller._poll_current_market()

    snapshot = await poller.get_odds("market-789")
    assert snapshot.yes_odds == pytest.approx(0.20)
    assert snapshot.no_odds == pytest.approx(0.80)
    assert snapshot.yes_qualifies is False
    assert snapshot.no_qualifies is True  # > 0.75


@pytest.mark.asyncio
async def test_odds_polling_neither_qualifies(poller, mock_client):
    """Test neither side qualifies when close to 50/50."""
    mock_market = Market(
        id="market-999",
        slug="btc-updown-15m-1771234800",
        question="BTC Up or Down?",
        active=True,
        end_date=datetime.now(),
        best_bid=0.55,  # 55% YES, 45% NO (neither > 75%)
        best_ask=0.56,
        outcomes=["Up", "Down"],
        conditionId="test-condition"
    )

    mock_client.discover_btc_15min_market.return_value = mock_market
    mock_client.get_market_by_slug.return_value = mock_market

    await poller._poll_current_market()

    snapshot = await poller.get_odds("market-999")
    assert snapshot.yes_odds == pytest.approx(0.55)
    assert snapshot.no_odds == pytest.approx(0.45)
    assert snapshot.yes_qualifies is False
    assert snapshot.no_qualifies is False


@pytest.mark.asyncio
async def test_get_odds_returns_none_if_not_cached(poller):
    """Test get_odds returns None for non-existent market."""
    snapshot = await poller.get_odds("unknown-market")
    assert snapshot is None


@pytest.mark.asyncio
async def test_odds_polling_handles_exceptions(poller, mock_client):
    """Test polling gracefully handles exceptions."""
    # Simulate API failure
    mock_client.discover_btc_15min_market.side_effect = Exception("API timeout")

    # Should not raise, just log error
    await poller._poll_current_market()

    # Verify no snapshot stored
    snapshot = await poller.get_odds("any-market")
    assert snapshot is None
