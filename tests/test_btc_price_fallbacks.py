import pytest
from datetime import datetime
from decimal import Decimal
from unittest.mock import AsyncMock, patch
import aiohttp

from polymarket.trading.btc_price import BTCPriceService
from polymarket.config import Settings


@pytest.mark.asyncio
async def test_fetch_coingecko_history():
    """Fetch historical candles from CoinGecko."""
    settings = Settings()
    service = BTCPriceService(settings)

    # Mock the aiohttp response with proper context manager support
    class MockResponse:
        status = 200

        def raise_for_status(self):
            pass

        async def json(self):
            return {
                "prices": [
                    [1707696000000, 67123.45],  # [timestamp_ms, price]
                    [1707696060000, 67150.20],
                    [1707696120000, 67100.80]
                ],
                "total_volumes": [
                    [1707696000000, 1000000],
                    [1707696060000, 1100000],
                    [1707696120000, 1050000]
                ]
            }

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    # Create a mock session with proper get method
    class MockSession:
        closed = False

        def get(self, *args, **kwargs):
            return MockResponse()

    # Replace _get_session to return our mock
    async def mock_get_session():
        return MockSession()

    service._get_session = mock_get_session

    result = await service._fetch_coingecko_history(minutes=3)

    assert len(result) == 3
    assert result[0].price == Decimal("67123.45")
    assert result[1].price == Decimal("67150.20")
    assert result[2].price == Decimal("67100.80")

    await service.close()
