"""Test end-phase market filtering."""
import pytest
from datetime import datetime, timedelta, timezone
from polymarket.models import Market


@pytest.mark.asyncio
async def test_filters_end_phase_markets():
    """Markets with <5min remaining should be filtered."""
    from scripts.auto_trade import AutoTrader
    from polymarket.config import Settings

    # Create trader instance
    settings = Settings()
    trader = AutoTrader(settings=settings, interval=60)

    # Mock the client's discover method to return test markets
    now = datetime.now(timezone.utc)

    # Create mock markets: one with <5 min, one with >5 min
    markets_data = [
        {
            "id": "market-1",
            "conditionId": "condition-1",
            "question": "Will BTC be above $50k at 10:15 AM?",
            "slug": "btc-15min-test-1",
            "active": True,
            "closed": False,
            "acceptingOrders": True,
            "endDate": (now + timedelta(minutes=3)).isoformat(),  # 3 min remaining - should filter
            "clobTokenIds": '["token-1", "token-2"]',
            "outcomes": ["Up", "Down"]
        },
        {
            "id": "market-2",
            "conditionId": "condition-2",
            "question": "Will BTC be above $51k at 10:30 AM?",
            "slug": "btc-15min-test-2",
            "active": True,
            "closed": False,
            "acceptingOrders": True,
            "endDate": (now + timedelta(minutes=10)).isoformat(),  # 10 min remaining - should pass
            "clobTokenIds": '["token-3", "token-4"]',
            "outcomes": ["Up", "Down"]
        }
    ]

    # Create Market objects
    mock_markets = [Market(**data) for data in markets_data]

    # Mock _discover_markets to return both test markets
    original_discover = trader._discover_markets

    async def mock_discover_markets():
        return mock_markets

    trader._discover_markets = mock_discover_markets

    try:
        # This will fail initially because get_tradeable_markets doesn't exist
        markets = await trader.get_tradeable_markets()

        # All returned markets should have >=5 minutes remaining
        for market in markets:
            now_check = datetime.now(timezone.utc)
            time_remaining = (market.end_date - now_check).total_seconds()
            assert time_remaining >= 300, f"Market {market.id} has only {time_remaining}s remaining"

        # Should have filtered out market-1 (3 min) and kept market-2 (10 min)
        assert len(markets) == 1
        assert markets[0].id == "market-2"

    finally:
        # Restore original method
        trader._discover_markets = original_discover


@pytest.mark.asyncio
async def test_logs_filtered_count(capsys):
    """Should log how many markets were filtered."""
    from scripts.auto_trade import AutoTrader
    from polymarket.config import Settings

    # Create trader instance
    settings = Settings()
    trader = AutoTrader(settings=settings, interval=60)

    # Mock the client's discover method
    now = datetime.now(timezone.utc)

    markets_data = [
        {
            "id": "market-end-phase",
            "conditionId": "condition-1",
            "question": "Test market",
            "slug": "test-1",
            "active": True,
            "closed": False,
            "acceptingOrders": True,
            "endDate": (now + timedelta(minutes=2)).isoformat(),  # <5 min - should filter
            "clobTokenIds": '["token-1", "token-2"]',
            "outcomes": ["Up", "Down"]
        }
    ]

    mock_market = Market(**markets_data[0])

    original_discover = trader._discover_markets

    async def mock_discover_markets():
        return [mock_market]

    trader._discover_markets = mock_discover_markets

    try:
        markets = await trader.get_tradeable_markets()

        # Capture stdout (where structlog logs to)
        captured = capsys.readouterr()
        log_output = captured.out.lower()

        # Should have logged filtered count
        assert "markets filtered" in log_output or "filtered_end_phase" in log_output

        # Verify the market was actually filtered
        assert len(markets) == 0, "Market with <5 min should be filtered out"

    finally:
        trader._discover_markets = original_discover
