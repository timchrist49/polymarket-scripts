"""
BTC Price Service

Fetches real-time BTC prices from Polymarket WebSocket (primary) with Binance fallback.
Provides current price, historical data, and price change calculations.
"""

import asyncio
import decimal
from datetime import datetime, timedelta
from typing import Optional
import structlog

import ccxt.async_support as ccxt
import aiohttp

from polymarket.models import BTCPriceData, PricePoint, PriceChange
from polymarket.config import Settings
from polymarket.trading.crypto_price_stream import CryptoPriceStream

logger = structlog.get_logger()


class BTCPriceService:
    """Real-time BTC price data from multiple sources."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._cache: Optional[BTCPriceData] = None
        self._cache_time: Optional[datetime] = None
        self._binance: ccxt.binance = ccxt.binance()
        self._session: Optional[aiohttp.ClientSession] = None

        # Polymarket WebSocket stream
        self._stream: Optional[CryptoPriceStream] = None
        self._stream_task: Optional[asyncio.Task] = None

    async def start(self):
        """Start Polymarket WebSocket stream."""
        if self._stream is None:
            self._stream = CryptoPriceStream(self.settings)
            self._stream_task = asyncio.create_task(self._stream.start())
            await asyncio.sleep(1)  # Initial connection time

            # Wait for connection and first price (up to 5 seconds)
            for _ in range(10):
                if self._stream.is_connected() and await self._stream.get_current_price():
                    break
                await asyncio.sleep(0.5)

            logger.info("BTCPriceService started with Polymarket WebSocket", connected=self._stream.is_connected() if self._stream else False)

    async def _get_session(self) -> aiohttp.ClientSession:
        """Lazy init of HTTP session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def get_current_price(self) -> BTCPriceData:
        """Get current BTC price with caching."""
        # Check cache
        if self._cache and self._cache_time:
            age = (datetime.now() - self._cache_time).total_seconds()
            if age < self.settings.btc_price_cache_seconds:
                logger.debug("Using cached BTC price", age_seconds=age)
                return self._cache

        # Try Polymarket WebSocket first
        if self._stream and self._stream.is_connected():
            data = await self._stream.get_current_price()
            if data:
                self._cache = data
                self._cache_time = datetime.now()
                return data
            else:
                logger.warning("No price from Polymarket WebSocket, falling back to Binance")

        # Fallback to Binance if WebSocket unavailable
        try:
            data = await self._fetch_binance()
        except Exception as e:
            logger.error("Failed to fetch price from Binance", error=str(e))
            # Return stale cache if available
            if self._cache:
                logger.warning("Returning stale cache", age_seconds=(datetime.now() - self._cache_time).total_seconds())
                return self._cache
            raise

        # Update cache
        self._cache = data
        self._cache_time = datetime.now()
        return data

    async def _fetch_binance(self) -> BTCPriceData:
        """Fetch from Binance API."""
        try:
            ticker = await self._binance.fetch_ticker("BTC/USDT")
            return BTCPriceData(
                price=decimal.Decimal(str(ticker["last"])),
                timestamp=datetime.fromtimestamp(ticker["timestamp"] / 1000),
                source="binance",
                volume_24h=decimal.Decimal(str(ticker["baseVolume"]))
            )
        except Exception as e:
            logger.error("Binance fetch failed", error=str(e))
            raise

    async def _fetch_coingecko(self) -> BTCPriceData:
        """Fetch from CoinGecko API (fallback)."""
        session = await self._get_session()
        url = "https://api.coingecko.com/api/v3/simple/price"
        params = {
            "ids": "bitcoin",
            "vs_currencies": "usd",
            "include_last_updated_at": "true",
            "include_24hr_vol": "true"
        }

        async with session.get(url, params=params) as resp:
            resp.raise_for_status()
            data = await resp.json()
            btc = data["bitcoin"]

            return BTCPriceData(
                price=decimal.Decimal(str(btc["usd"])),
                timestamp=datetime.fromtimestamp(btc["last_updated_at"]),
                source="coingecko",
                volume_24h=decimal.Decimal(str(btc.get("usd_24h_vol", 0)))
            )

    async def get_price_history(self, minutes: int = 60) -> list[PricePoint]:
        """Get historical price points for technical analysis."""
        # Use direct HTTP request to Binance API instead of ccxt
        # to avoid the derivatives API timeout issue
        try:
            session = await self._get_session()
            url = "https://api.binance.com/api/v3/klines"
            params = {
                "symbol": "BTCUSDT",
                "interval": "1m",
                "limit": str(minutes)
            }

            async with session.get(url, params=params, timeout=10) as resp:
                resp.raise_for_status()
                data = await resp.json()

                return [
                    PricePoint(
                        price=decimal.Decimal(str(candle[4])),  # Close price
                        volume=decimal.Decimal(str(candle[5])),  # Volume
                        timestamp=datetime.fromtimestamp(candle[0] / 1000)
                    )
                    for candle in data
                ]
        except asyncio.TimeoutError:
            logger.error("Failed to fetch price history", error="Binance API timeout after 10s")
            raise
        except Exception as e:
            logger.error("Failed to fetch price history", error=str(e) or type(e).__name__)
            raise

    async def get_price_at_timestamp(self, timestamp: int) -> Optional[decimal.Decimal]:
        """
        Get BTC price at a specific Unix timestamp.

        Args:
            timestamp: Unix timestamp (seconds since epoch)

        Returns:
            BTC price as Decimal, or None if unavailable
        """
        try:
            session = await self._get_session()
            url = "https://api.binance.com/api/v3/klines"

            # Convert to milliseconds and get single 1-minute candle
            timestamp_ms = timestamp * 1000
            params = {
                "symbol": "BTCUSDT",
                "interval": "1m",
                "startTime": str(timestamp_ms),
                "limit": "1"
            }

            async with session.get(url, params=params, timeout=10) as resp:
                resp.raise_for_status()
                data = await resp.json()

                if not data:
                    logger.warning("No price data at timestamp", timestamp=timestamp)
                    return None

                # Return close price of the candle
                price = decimal.Decimal(str(data[0][4]))
                logger.info(
                    "Fetched historical BTC price",
                    timestamp=timestamp,
                    price=f"${price:,.2f}"
                )
                return price

        except asyncio.TimeoutError:
            logger.error("Failed to fetch historical price", timestamp=timestamp, error="Binance API timeout after 10s")
            return None
        except Exception as e:
            logger.error("Failed to fetch historical price", timestamp=timestamp, error=str(e) or type(e).__name__)
            return None

    async def get_price_change(self, window_minutes: int = 5) -> PriceChange:
        """Calculate price change over a time window."""
        history = await self.get_price_history(minutes=window_minutes)
        if len(history) < 2:
            raise ValueError("Not enough price history")

        old = history[0]
        current = await self.get_current_price()

        change_amount = current.price - old.price
        change_percent = float(change_amount / old.price * 100)
        velocity = change_amount / decimal.Decimal(window_minutes)

        return PriceChange(
            current_price=current.price,
            change_percent=change_percent,
            change_amount=change_amount,
            velocity=velocity
        )

    async def close(self):
        """Clean up resources."""
        if self._stream:
            await self._stream.stop()
        if self._stream_task:
            self._stream_task.cancel()
            try:
                await self._stream_task
            except asyncio.CancelledError:
                pass
        await self._binance.close()
        if self._session and not self._session.closed:
            await self._session.close()
