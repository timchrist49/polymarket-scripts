"""Tests for CoinGeckoSignals (Coinbase Premium Index, Volume Spike, Funding Rate)."""
import pytest
from unittest.mock import AsyncMock, patch


@pytest.mark.asyncio
async def test_coinbase_premium_positive():
    """When Coinbase BTC > Binance BTC, CPI is positive."""
    from AI_analysis_upgrade.coingecko import CoinGeckoSignals

    signals = CoinGeckoSignals(api_key="test_key")

    coinbase_resp = {"tickers": [{"base": "BTC", "target": "USD", "last": 95100.0, "coin_id": "bitcoin"}]}
    binance_resp  = {"tickers": [{"base": "BTC", "target": "USDT", "last": 95000.0, "coin_id": "bitcoin"}]}

    with patch.object(signals, '_get', side_effect=[coinbase_resp, binance_resp]):
        result = await signals.fetch_coinbase_premium()

    assert result > 0.0  # Coinbase higher than Binance


@pytest.mark.asyncio
async def test_coinbase_premium_negative():
    """When Binance BTC > Coinbase BTC, CPI is negative."""
    from AI_analysis_upgrade.coingecko import CoinGeckoSignals

    signals = CoinGeckoSignals(api_key="test_key")
    coinbase_resp = {"tickers": [{"base": "BTC", "target": "USD", "last": 94900.0, "coin_id": "bitcoin"}]}
    binance_resp  = {"tickers": [{"base": "BTC", "target": "USDT", "last": 95000.0, "coin_id": "bitcoin"}]}

    with patch.object(signals, '_get', side_effect=[coinbase_resp, binance_resp]):
        result = await signals.fetch_coinbase_premium()

    assert result < 0.0


@pytest.mark.asyncio
async def test_coinbase_premium_returns_zero_on_error():
    """API failure → return 0.0 (neutral signal)."""
    from AI_analysis_upgrade.coingecko import CoinGeckoSignals

    signals = CoinGeckoSignals(api_key="test_key")
    with patch.object(signals, '_get', side_effect=Exception("timeout")):
        result = await signals.fetch_coinbase_premium()

    assert result == 0.0


@pytest.mark.asyncio
async def test_volume_spike_ratio_above_threshold():
    """Volume 2x above rolling average → ratio > 1.0."""
    from AI_analysis_upgrade.coingecko import CoinGeckoSignals

    signals = CoinGeckoSignals(api_key="test_key")

    base_vol = 1000.0
    spike_vol = 2000.0
    volumes = [[i * 300_000, base_vol] for i in range(29)]
    volumes.append([29 * 300_000, spike_vol])
    mock_data = {"prices": [], "total_volumes": volumes}

    with patch.object(signals, '_get', return_value=mock_data):
        ratio = await signals.fetch_volume_spike_ratio()

    assert ratio > 1.5


@pytest.mark.asyncio
async def test_funding_rate_extreme_positive():
    """Funding rate > 0.05% → returns the rate (crowded longs)."""
    from AI_analysis_upgrade.coingecko import CoinGeckoSignals

    signals = CoinGeckoSignals(api_key="test_key")
    mock_data = [
        {"symbol": "BTCUSDT", "index_id": "BTC", "contract_type": "perpetual", "funding_rate": 0.07, "volume_24h": 1000},
        {"symbol": "BTCUSDT_OKX", "index_id": "BTC", "contract_type": "perpetual", "funding_rate": 0.06, "volume_24h": 500},
    ]
    with patch.object(signals, '_get', return_value=mock_data):
        rate = await signals.fetch_funding_rate()

    assert rate is not None
    assert rate > 0.05


@pytest.mark.asyncio
async def test_funding_rate_normal_returns_none():
    """Normal funding rate (0.01-0.03%) → returns None (no signal)."""
    from AI_analysis_upgrade.coingecko import CoinGeckoSignals

    signals = CoinGeckoSignals(api_key="test_key")
    mock_data = [
        {"symbol": "BTCUSDT", "index_id": "BTC", "contract_type": "perpetual", "funding_rate": 0.02, "volume_24h": 1000},
    ]
    with patch.object(signals, '_get', return_value=mock_data):
        rate = await signals.fetch_funding_rate()

    assert rate is None  # No signal in normal range
