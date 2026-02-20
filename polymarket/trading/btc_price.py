"""
BTC Price Service

Fetches real-time BTC prices from Polymarket WebSocket (primary) with Binance fallback.
Provides current price, historical data, and price change calculations.
"""

import asyncio
from decimal import Decimal
from datetime import datetime, timedelta, timezone
from typing import Optional
import statistics
import structlog

import ccxt.async_support as ccxt
import aiohttp

from polymarket.models import BTCPriceData, PricePoint, PriceChange, FundingRateSignal, BTCDominanceSignal, OrderFlowSignal
from polymarket.config import Settings
from polymarket.trading.crypto_price_stream import CryptoPriceStream
from polymarket.trading.price_cache import CandleCache
from polymarket.trading.retry_logic import fetch_with_retry, RetryConfig
from polymarket.trading.parallel_fetch import fetch_with_fallbacks
from polymarket.trading.stale_policy import StaleDataPolicy
from polymarket.performance.settlement_validator import SettlementPriceValidator

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

        # Candle cache
        self._candle_cache = CandleCache()

        # Stale data policy
        self._stale_policy = StaleDataPolicy(
            max_stale_age_seconds=settings.btc_cache_stale_max_age
        )

        # Settlement validator
        self._settlement_validator = SettlementPriceValidator(
            btc_service=self
        )

        # Retry config
        self._retry_config = RetryConfig.from_settings(settings)

    async def start(self):
        """Start Polymarket WebSocket stream."""
        if self._stream is None:
            # Enable price buffer for 24-hour price history
            # Use Chainlink data by default for higher quality prices
            self._stream = CryptoPriceStream(
                self.settings,
                buffer_enabled=True,
                buffer_file="data/price_history.json",
                use_chainlink=True
            )
            logger.info("Initializing BTC price service with Chainlink data source")
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
                logger.warning("No price from Polymarket WebSocket, falling back to CoinGecko")

        # Fallback to CoinGecko (Binance blocked in Indonesia)
        try:
            data = await self._fetch_coingecko()
            logger.info("Fetched BTC price from CoinGecko", price=float(data.price))
        except Exception as e:
            logger.error("Failed to fetch price from CoinGecko", error=str(e))
            # Try Binance as last resort (will likely fail in Indonesia)
            try:
                data = await self._fetch_binance()
                logger.info("Fetched BTC price from Binance", price=float(data.price))
            except Exception as binance_error:
                logger.error("Binance also failed", error=str(binance_error))
                # Return stale cache if available
                if self._cache:
                    age = (datetime.now() - self._cache_time).total_seconds()
                    logger.warning("Returning stale cache", age_seconds=age)
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
                price=Decimal(str(ticker["last"])),
                timestamp=datetime.fromtimestamp(ticker["timestamp"] / 1000),
                source="binance",
                volume_24h=Decimal(str(ticker["baseVolume"]))
            )
        except Exception as e:
            logger.error("Binance fetch failed", error=str(e))
            raise

    async def _fetch_coingecko(self) -> BTCPriceData:
        """Fetch from CoinGecko Pro API."""
        session = await self._get_session()

        # Use Pro API if API key available
        if self.settings.coingecko_api_key:
            url = "https://pro-api.coingecko.com/api/v3/simple/price"
            params = {
                "ids": "bitcoin",
                "vs_currencies": "usd",
                "include_last_updated_at": "true",
                "include_24hr_vol": "true",
                "include_24hr_change": "true",
                "include_market_cap": "true",
                "x_cg_pro_api_key": self.settings.coingecko_api_key
            }
        else:
            # Fallback to free tier
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
                price=Decimal(str(btc["usd"])),
                timestamp=datetime.fromtimestamp(btc["last_updated_at"]),
                source="coingecko_pro" if self.settings.coingecko_api_key else "coingecko",
                volume_24h=Decimal(str(btc.get("usd_24h_vol", 0)))
            )

    async def _fetch_coingecko_history(self, minutes: int = 60) -> list[PricePoint]:
        """
        Fetch historical price candles from CoinGecko Pro API.

        Args:
            minutes: Number of 1-minute candles to fetch

        Returns:
            List of price points
        """
        session = await self._get_session()

        # Use Pro API if API key available
        base_url = "https://pro-api.coingecko.com/api/v3" if self.settings.coingecko_api_key else "https://api.coingecko.com/api/v3"
        url = f"{base_url}/coins/bitcoin/market_chart/range"

        # Calculate time range (now - minutes)
        to_timestamp = int(datetime.now().timestamp())
        from_timestamp = to_timestamp - (minutes * 60)

        params = {
            "vs_currency": "usd",
            "from": str(from_timestamp),
            "to": str(to_timestamp)
        }

        # Add API key for Pro tier
        if self.settings.coingecko_api_key:
            params["x_cg_pro_api_key"] = self.settings.coingecko_api_key

        try:
            async with session.get(url, params=params, timeout=30) as resp:
                resp.raise_for_status()
                data = await resp.json()

                # CoinGecko returns arrays of [timestamp_ms, price]
                prices = data.get("prices", [])
                volumes = data.get("total_volumes", [])

                # Convert to PricePoint objects
                result = []
                for i, (timestamp_ms, price) in enumerate(prices):
                    volume = volumes[i][1] if i < len(volumes) else 0

                    result.append(PricePoint(
                        price=Decimal(str(price)),
                        volume=Decimal(str(volume)),
                        timestamp=datetime.fromtimestamp(timestamp_ms / 1000)
                    ))

                logger.debug(
                    "Fetched CoinGecko history",
                    candles=len(result),
                    minutes=minutes
                )

                return result

        except Exception as e:
            logger.error("Failed to fetch CoinGecko history", error=str(e))
            raise

    async def _fetch_kraken_history(self, minutes: int = 60) -> list[PricePoint]:
        """
        Fetch historical price candles from Kraken.

        Args:
            minutes: Number of 1-minute candles to fetch

        Returns:
            List of price points
        """
        session = await self._get_session()

        # Kraken OHLC endpoint
        url = "https://api.kraken.com/0/public/OHLC"

        # Kraken uses 'since' parameter (Unix timestamp) and interval
        since_timestamp = int((datetime.now() - timedelta(minutes=minutes)).timestamp())

        params = {
            "pair": "XBTUSD",  # BTC/USD pair
            "interval": "1",   # 1 minute candles
            "since": str(since_timestamp)
        }

        try:
            async with session.get(url, params=params, timeout=30) as resp:
                resp.raise_for_status()
                data = await resp.json()

                # Check for Kraken API errors
                if data.get("error"):
                    raise Exception(f"Kraken API error: {data['error']}")

                # Kraken OHLC format: [timestamp, open, high, low, close, vwap, volume, count]
                ohlc_data = data["result"].get("XXBTZUSD", [])

                result = []
                for candle in ohlc_data:
                    timestamp = candle[0]
                    close_price = candle[4]
                    volume = candle[6]

                    result.append(PricePoint(
                        price=Decimal(str(close_price)),
                        volume=Decimal(str(volume)),
                        timestamp=datetime.fromtimestamp(timestamp)
                    ))

                logger.debug(
                    "Fetched Kraken history",
                    candles=len(result),
                    minutes=minutes
                )

                return result

        except Exception as e:
            logger.error("Failed to fetch Kraken history", error=str(e))
            raise

    async def get_price_history(self, minutes: int = 60) -> list[PricePoint]:
        """Get historical price points for technical analysis with fallbacks."""
        # Check cache first
        cached_candles = []
        uncached_count = 0

        for i in range(minutes):
            timestamp = int((datetime.now() - timedelta(minutes=minutes-i)).timestamp())
            candle = self._candle_cache.get(timestamp)

            if candle:
                cached_candles.append(candle)
            else:
                uncached_count += 1

        # If we have enough cached candles, use them
        if uncached_count == 0:
            logger.debug("Using fully cached price history", minutes=minutes)
            self._stale_policy.record_success(cached_candles)
            return cached_candles

        # Try price buffer first (should have 24h of data)
        if self._stream and self._stream.price_buffer:
            try:
                now = datetime.now()
                end_time = int(now.timestamp())
                start_time = int((now - timedelta(minutes=minutes)).timestamp())

                buffer_entries = await self._stream.price_buffer.get_price_range(
                    start_time, end_time
                )

                # Convert buffer entries to PricePoint format
                if buffer_entries and len(buffer_entries) >= minutes * 0.8:  # Allow 20% missing
                    result = [
                        PricePoint(
                            price=entry.price,
                            volume=Decimal("0"),  # Volume not stored in buffer
                            timestamp=datetime.fromtimestamp(entry.timestamp)
                        )
                        for entry in buffer_entries
                    ]
                    logger.info(
                        "✓ Using price history from buffer (no Binance call!)",
                        minutes=minutes,
                        entries=len(result),
                        source="buffer"
                    )
                    self._stale_policy.record_success(result)

                    # Cache the buffer data for next time
                    for candle in result:
                        timestamp = int(candle.timestamp.timestamp())
                        self._candle_cache.put(timestamp, candle)

                    return result
                else:
                    logger.info(
                        "Buffer has insufficient data, falling back to CoinGecko",
                        requested=minutes,
                        available=len(buffer_entries) if buffer_entries else 0,
                        threshold=int(minutes * 0.8)
                    )
            except Exception as e:
                logger.warning("Buffer query failed, falling back to CoinGecko", error=str(e))

        # Fallback to external APIs if buffer doesn't have enough data
        # Use Kraken as primary: gives 1-min candles (CoinGecko only ~1 per 5 min)
        # 1-min granularity is needed for RSI-14 and regime detection
        result = await fetch_with_fallbacks(
            lambda: fetch_with_retry(lambda: self._fetch_kraken_history(minutes), "Kraken", self._retry_config),
            [
                ("CoinGecko", lambda: self._fetch_coingecko_history(minutes)),
                ("Binance", lambda: self._fetch_binance_history(minutes))
            ]
        )

        if result:
            # Cache all fetched candles
            for candle in result:
                timestamp = int(candle.timestamp.timestamp())
                self._candle_cache.put(timestamp, candle)

            self._stale_policy.record_success(result)
            logger.debug("Fetched and cached price history",
                       minutes=minutes, candles=len(result))
            return result

        # All sources failed - try stale cache
        self._stale_policy.record_failure()
        stale_data = self._stale_policy.get_stale_cache_with_warning()

        if stale_data:
            return stale_data

        # No options left
        raise Exception("Failed to fetch price history from all sources")

    async def _fetch_binance_history(self, minutes: int) -> list[PricePoint]:
        """Fetch from Binance (extracted for fallback logic)."""
        session = await self._get_session()
        url = "https://api.binance.com/api/v3/klines"
        params = {
            "symbol": "BTCUSDT",
            "interval": "1m",
            "limit": str(minutes)
        }

        async with session.get(url, params=params, timeout=30) as resp:
            resp.raise_for_status()
            data = await resp.json()

            return [
                PricePoint(
                    price=Decimal(str(candle[4])),  # Close price
                    volume=Decimal(str(candle[5])),  # Volume
                    timestamp=datetime.fromtimestamp(candle[0] / 1000)
                )
                for candle in data
            ]

    async def get_price_at_timestamp(self, timestamp: int) -> Optional[Decimal]:
        """
        Get BTC price at a specific Unix timestamp with validation.

        Uses multi-source validation for settlement accuracy.

        Args:
            timestamp: Unix timestamp (seconds since epoch)

        Returns:
            Validated BTC price as Decimal, or None if unavailable/invalid
        """
        # Use settlement validator for multi-source consensus
        validated_price = await self._settlement_validator.get_validated_price(timestamp)

        if validated_price:
            return validated_price

        # Validation failed - log and return None
        logger.warning(
            "Price validation failed for settlement",
            timestamp=timestamp
        )
        return None

    async def _fetch_chainlink_from_buffer(self, timestamp: int) -> Optional[Decimal]:
        """
        Fetch Chainlink historical price from buffer with ±30s tolerance.

        Args:
            timestamp: Unix timestamp (seconds)

        Returns:
            Chainlink price from buffer, or None if not available
        """
        if self._stream and self._stream.price_buffer:
            try:
                price_data = await self._stream.price_buffer.get_price_at(
                    timestamp,
                    tolerance=30  # ±30s window for market start times
                )

                if price_data and price_data.source == "chainlink":
                    logger.info(
                        "Historical price from Chainlink buffer",
                        timestamp=timestamp,
                        price=f"${price_data.price:,.2f}",
                        source="chainlink",
                        age_seconds=int((datetime.now().timestamp() - timestamp))
                    )
                    return price_data.price
            except Exception as e:
                logger.warning(
                    "Buffer lookup failed",
                    timestamp=timestamp,
                    error=str(e)
                )

        return None

    async def _fetch_binance_at_timestamp(self, timestamp: int) -> Optional[Decimal]:
        """
        Fetch BTC price at specific timestamp from Binance.

        Query order:
        1. Price history buffer (if available) - instant, no network latency
        2. Binance API (fallback) - slower, may timeout

        Args:
            timestamp: Unix timestamp (seconds)

        Returns:
            BTC price or None
        """
        # Try buffer first (instant lookup, no network)
        if self._stream and self._stream.price_buffer:
            try:
                price_data = await self._stream.price_buffer.get_price_at(
                    timestamp,
                    tolerance=30  # 30-second window
                )
                if price_data:
                    logger.debug(
                        "Price found in buffer",
                        timestamp=timestamp,
                        price=f"${price_data.price:,.2f}",
                        source=price_data.source
                    )
                    return price_data.price
                else:
                    logger.debug(
                        "Price not in buffer, falling back to CoinGecko",
                        timestamp=timestamp
                    )
            except Exception as e:
                logger.warning(
                    "Buffer query failed, falling back to CoinGecko",
                    error=str(e)
                )

        # Fallback to CoinGecko API first (Binance blocked in Indonesia)
        try:
            price = await self._fetch_coingecko_at_timestamp(timestamp)
            if price:
                return price
            logger.info("CoinGecko returned no data, trying Binance as last resort")
        except Exception as e:
            logger.warning("CoinGecko failed, trying Binance", error=str(e))

        # Last resort: Binance API (will likely fail in Indonesia)
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

            async with session.get(url, params=params, timeout=30) as resp:
                resp.raise_for_status()
                data = await resp.json()

                if not data:
                    logger.warning("No price data at timestamp", timestamp=timestamp)
                    return None

                # Return OPEN price of the candle (at exact timestamp, not 1 min later)
                # data[0] = [open_time, open, high, low, close, volume, close_time, ...]
                price = Decimal(str(data[0][1]))
                logger.debug(
                    "Fetched Binance historical price",
                    timestamp=timestamp,
                    price=f"${price:,.2f}",
                    source="binance"
                )
                return price

        except asyncio.TimeoutError:
            logger.error("Failed to fetch historical price", timestamp=timestamp,
                        error="Binance API timeout after 30s")
            return None
        except Exception as e:
            logger.error("Failed to fetch historical price", timestamp=timestamp,
                        error=str(e) or type(e).__name__)
            return None

    async def _fetch_coingecko_at_timestamp(self, timestamp: int) -> Optional[Decimal]:
        """
        Fetch BTC price at specific timestamp from CoinGecko Pro API.

        Uses /market_chart/range endpoint which provides 5-minute granularity
        for recent data (within 1 day), suitable for 15-minute markets.

        Args:
            timestamp: Unix timestamp (seconds)

        Returns:
            BTC price or None
        """
        session = await self._get_session()

        # Use Pro API if API key available
        base_url = "https://pro-api.coingecko.com/api/v3" if self.settings.coingecko_api_key else "https://api.coingecko.com/api/v3"

        # Use market_chart/range endpoint for minute-level granularity
        # This returns 5-minute data points for recent timestamps
        url = f"{base_url}/coins/bitcoin/market_chart/range"

        # Query a 5-minute window around the target timestamp
        params = {
            "vs_currency": "usd",
            "from": str(timestamp),
            "to": str(timestamp + 300),  # +5 minutes
        }

        # Add API key for Pro tier
        if self.settings.coingecko_api_key:
            params["x_cg_pro_api_key"] = self.settings.coingecko_api_key

        try:
            async with session.get(url, params=params, timeout=30) as resp:
                resp.raise_for_status()
                data = await resp.json()

                # Extract price data points [[timestamp_ms, price], ...]
                prices = data.get("prices", [])

                if not prices:
                    logger.warning("No price data returned from CoinGecko", timestamp=timestamp)
                    return None

                # Find the price closest to our target timestamp
                target_ms = timestamp * 1000  # Convert to milliseconds
                closest_price = min(prices, key=lambda p: abs(p[0] - target_ms))

                price = closest_price[1]
                time_diff = abs(closest_price[0] - target_ms) / 1000  # Convert back to seconds

                logger.debug(
                    "Fetched CoinGecko price at timestamp",
                    timestamp=timestamp,
                    price=f"${price:,.2f}",
                    time_diff_seconds=f"{time_diff:.0f}"
                )

                return Decimal(str(price))

        except Exception as e:
            logger.error("Failed to fetch CoinGecko price at timestamp",
                        timestamp=timestamp, error=str(e))
            return None

    async def _fetch_kraken_at_timestamp(self, timestamp: int) -> Optional[Decimal]:
        """
        Fetch BTC price at specific timestamp from Kraken.

        Args:
            timestamp: Unix timestamp (seconds)

        Returns:
            BTC price or None
        """
        session = await self._get_session()

        # Kraken OHLC endpoint with specific timestamp
        url = "https://api.kraken.com/0/public/OHLC"

        params = {
            "pair": "XBTUSD",
            "interval": "1",
            "since": str(timestamp)
        }

        try:
            async with session.get(url, params=params, timeout=30) as resp:
                resp.raise_for_status()
                data = await resp.json()

                if data.get("error"):
                    raise Exception(f"Kraken API error: {data['error']}")

                # Get first candle (closest to timestamp)
                ohlc_data = data["result"].get("XXBTZUSD", [])

                if ohlc_data:
                    close_price = ohlc_data[0][4]  # Close price
                    logger.debug("Fetched Kraken price at timestamp",
                                timestamp=timestamp, price=f"${close_price}")
                    return Decimal(str(close_price))

                return None

        except Exception as e:
            logger.error("Failed to fetch Kraken price at timestamp",
                        timestamp=timestamp, error=str(e))
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
        velocity = change_amount / Decimal(window_minutes)

        return PriceChange(
            current_price=current.price,
            change_percent=change_percent,
            change_amount=change_amount,
            velocity=velocity
        )

    async def get_funding_rate_raw(self, exchanges: list[str] | None = None) -> float | None:
        """
        Fetch raw BTC funding rate from derivatives exchanges.

        Tries multiple exchanges with reduced timeout to avoid hanging.
        Returns the first available funding rate.

        Args:
            exchanges: List of exchange names to try. Defaults to ['binance', 'bybit', 'okx'].

        Returns:
            Raw funding rate as decimal (e.g., 0.0001 = 0.01%), or None if unavailable.
        """
        if not self.settings.coingecko_api_key:
            logger.debug("CoinGecko Pro API key required for funding rates")
            return None

        if exchanges is None:
            exchanges = ['binance', 'bybit', 'okx']

        session = await self._get_session()
        url = "https://pro-api.coingecko.com/api/v3/derivatives"
        params = {"x_cg_pro_api_key": self.settings.coingecko_api_key}

        try:
            # Reduced timeout to prevent hanging (10s instead of 30s)
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                resp.raise_for_status()
                data = await resp.json()

                # Try each exchange in priority order
                for exchange_name in exchanges:
                    # Find BTC perpetual contract for this exchange
                    for ticker in data:
                        market = ticker.get("market", "").lower()
                        symbol = ticker.get("symbol", "")
                        contract_type = ticker.get("contract_type", "")

                        # Match exchange and BTC contract
                        if (
                            exchange_name.lower() in market
                            and "btc" in symbol.lower()
                            and "usdt" in symbol.lower()
                            and contract_type == "perpetual"
                        ):
                            funding_rate = ticker.get("funding_rate")
                            if funding_rate is not None:
                                logger.info(
                                    "Funding rate fetched",
                                    exchange=exchange_name,
                                    funding_rate=f"{float(funding_rate):.6f}",
                                    rate_pct=f"{float(funding_rate) * 100:.4f}%"
                                )
                                return float(funding_rate)

                logger.warning("No funding rate found for any exchange", tried=exchanges)
                return None

        except asyncio.TimeoutError:
            logger.warning("Funding rate fetch timeout (10s)", tried=exchanges)
            return None
        except Exception as e:
            logger.warning("Failed to fetch funding rates", error=str(e))
            return None

    async def get_funding_rates(self) -> FundingRateSignal | None:
        """
        Fetch BTC funding rates from Binance futures via CoinGecko Pro API.

        Positive funding = longs pay shorts (overheated/bearish signal)
        Negative funding = shorts pay longs (oversold/bullish signal)

        Returns:
            FundingRateSignal or None if unavailable
        """
        # Use new raw method with reduced timeout
        funding_rate_decimal = await self.get_funding_rate_raw()

        if funding_rate_decimal is None:
            return None

        # Convert to percentage
        funding_rate = funding_rate_decimal * 100  # e.g., 0.0001 -> 0.01%

        # Normalize to [-1, 1] range
        # Typical funding rates: -0.1% to +0.1% (extreme), -0.01% to +0.01% (normal)
        # Normalize: ±0.05% = ±0.5, ±0.1% = ±1.0
        funding_rate_normalized = max(min(funding_rate / 0.1, 1.0), -1.0)

        # Score: negative funding rate is bullish (shorts paying longs)
        # positive funding rate is bearish (longs paying shorts)
        score = -funding_rate_normalized  # Invert: negative funding = positive score

        # Confidence based on magnitude (higher = more extreme = more confident)
        confidence = min(abs(funding_rate_normalized), 1.0)

        # Signal classification
        if funding_rate > 0.05:
            signal_type = "OVERHEATED"  # Longs overheated, bearish
        elif funding_rate < -0.05:
            signal_type = "OVERSOLD"  # Shorts overheated, bullish
        else:
            signal_type = "NEUTRAL"

        logger.info(
            "Funding rate signal generated",
            funding_rate=f"{funding_rate:.4f}%",
            score=f"{score:+.2f}",
            confidence=f"{confidence:.2f}",
            signal=signal_type
        )

        return FundingRateSignal(
            score=score,
            confidence=confidence,
            funding_rate=funding_rate,
            funding_rate_normalized=funding_rate_normalized,
            signal_type=signal_type,
            source="multi_exchange",
            timestamp=datetime.now()
        )

    async def get_exchange_prices(self, exchanges: list[str] | None = None) -> dict[str, float] | None:
        """
        Fetch current BTC price from multiple exchanges for premium comparison.

        Args:
            exchanges: List of exchange IDs to query. Defaults to ['coinbase', 'binance', 'kraken'].

        Returns:
            Dictionary mapping exchange name to BTC/USD price, or None if unavailable.
        """
        if not self.settings.coingecko_api_key:
            logger.debug("CoinGecko Pro API key required for multi-exchange prices")
            return None

        if exchanges is None:
            exchanges = ['coinbase', 'binance', 'kraken']

        try:
            session = await self._get_session()
            url = "https://pro-api.coingecko.com/api/v3/coins/bitcoin/tickers"
            params = {
                "x_cg_pro_api_key": self.settings.coingecko_api_key,
                "depth": "true"  # Include bid/ask spread
            }

            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                resp.raise_for_status()
                data = await resp.json()

                tickers = data.get("tickers", [])
                prices = {}

                # Extract prices for requested exchanges
                for ticker in tickers:
                    exchange_id = ticker.get("market", {}).get("identifier", "").lower()
                    base = ticker.get("base", "")
                    target = ticker.get("target", "")

                    # Match BTC/USD or BTC/USDT pairs
                    if (
                        base == "BTC"
                        and target in ["USD", "USDT"]
                        and any(ex in exchange_id for ex in exchanges)
                    ):
                        last_price = ticker.get("last")
                        if last_price:
                            # Determine exchange name
                            exchange_name = None
                            for ex in exchanges:
                                if ex in exchange_id:
                                    exchange_name = ex
                                    break

                            if exchange_name and exchange_name not in prices:
                                prices[exchange_name] = float(last_price)

                if prices:
                    logger.info(
                        "Multi-exchange prices fetched",
                        exchanges=list(prices.keys()),
                        prices={k: f"${v:,.2f}" for k, v in prices.items()}
                    )
                    return prices
                else:
                    logger.warning("No exchange prices found", tried=exchanges)
                    return None

        except asyncio.TimeoutError:
            logger.warning("Exchange prices fetch timeout (10s)", tried=exchanges)
            return None
        except Exception as e:
            logger.warning("Failed to fetch exchange prices", error=str(e))
            return None

    async def get_recent_volumes(self, hours: int = 24) -> list[Decimal] | None:
        """
        Get recent volume data for volume confirmation analysis.

        Args:
            hours: Number of hours of volume data to retrieve (default 24).

        Returns:
            List of volume values in USD, or None if unavailable.
        """
        try:
            # Get historical price data which includes volumes
            minutes = hours * 60
            history = await self.get_price_history(minutes=minutes)

            if not history:
                logger.debug("No price history available for volume data")
                return None

            # Extract volumes from history
            volumes = [point.volume for point in history if point.volume > 0]

            if volumes:
                logger.debug(
                    "Volume history retrieved",
                    hours=hours,
                    data_points=len(volumes),
                    avg_volume=f"${float(sum(volumes) / len(volumes)):,.0f}"
                )
                return volumes
            else:
                logger.debug("No volume data in price history")
                return None

        except Exception as e:
            logger.warning("Failed to get volume history", error=str(e))
            return None

    async def get_btc_dominance(self) -> BTCDominanceSignal | None:
        """
        Fetch BTC dominance from CoinGecko Pro API global market data.

        Rising dominance = capital flowing into BTC (bullish for BTC)
        Falling dominance = capital flowing to alts (bearish for BTC)

        Returns:
            BTCDominanceSignal or None if unavailable
        """
        if not self.settings.coingecko_api_key:
            logger.warning("CoinGecko Pro API key required for BTC dominance")
            return None

        try:
            session = await self._get_session()
            url = "https://pro-api.coingecko.com/api/v3/global"
            params = {"x_cg_pro_api_key": self.settings.coingecko_api_key}

            async with session.get(url, params=params, timeout=10) as resp:
                resp.raise_for_status()
                data = await resp.json()

                global_data = data.get("data", {})

                # Extract BTC dominance percentage
                dominance_pct = global_data.get("market_cap_percentage", {}).get("btc")
                if dominance_pct is None:
                    logger.warning("BTC dominance not available in global data")
                    return None

                dominance_pct = float(dominance_pct)

                # Extract market caps
                market_caps = global_data.get("market_cap_percentage", {})
                total_market_cap = global_data.get("total_market_cap", {}).get("usd", 0)
                btc_market_cap = total_market_cap * (dominance_pct / 100.0) if total_market_cap else 0

                # Extract 24h change in dominance
                # Note: CoinGecko doesn't provide direct dominance change, so we estimate from price change
                dominance_change_24h = global_data.get("market_cap_change_percentage_24h_usd", 0.0)

                # Normalize dominance to score
                # Historical BTC dominance range: 40% (alt season) to 70% (BTC season)
                # Typical: 45-60%
                # Score: >55% = bullish BTC, <50% = bearish BTC (alt season)
                if dominance_pct > 55:
                    score = min((dominance_pct - 55) / 15, 1.0)  # 55-70% maps to 0.0-1.0
                    signal_type = "BTC_SEASON"
                elif dominance_pct < 50:
                    score = max((dominance_pct - 50) / 10, -1.0)  # 40-50% maps to -1.0-0.0
                    signal_type = "ALT_SEASON"
                else:
                    score = 0.0
                    signal_type = "NEUTRAL"

                # Confidence based on how far from neutral (50%)
                confidence = min(abs(dominance_pct - 50) / 20, 1.0)

                logger.info(
                    "BTC dominance fetched",
                    dominance=f"{dominance_pct:.2f}%",
                    change_24h=f"{dominance_change_24h:+.2f}%",
                    score=f"{score:+.2f}",
                    confidence=f"{confidence:.2f}",
                    signal=signal_type
                )

                return BTCDominanceSignal(
                    score=score,
                    confidence=confidence,
                    dominance_pct=dominance_pct,
                    dominance_change_24h=dominance_change_24h,
                    signal_type=signal_type,
                    market_cap_btc=btc_market_cap,
                    market_cap_total=total_market_cap,
                    timestamp=datetime.now()
                )

        except Exception as e:
            logger.error("Failed to fetch BTC dominance", error=str(e))
            return None

    async def get_volume_data(self) -> "VolumeData | None":
        """
        Fetch BTC volume data from CoinGecko.

        Returns:
            VolumeData with volume metrics for breakout detection
        """
        try:
            session = await self._get_session()

            # Get 24h volume data
            url = "https://api.coingecko.com/api/v3/coins/bitcoin"
            params = {"localization": "false", "tickers": "false", "community_data": "false"}

            if self.settings.coingecko_api_key:
                url = "https://pro-api.coingecko.com/api/v3/coins/bitcoin"
                params["x_cg_pro_api_key"] = self.settings.coingecko_api_key

            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                resp.raise_for_status()
                data = await resp.json()

                market_data = data.get("market_data", {})
                volume_24h = float(market_data.get("total_volume", {}).get("usd", 0))

                # Estimate current hour volume (24h / 24)
                volume_avg_hour = volume_24h / 24

                # Get current hour volume (approximate from recent trades)
                # For now, use average; TODO: Get real-time hourly volume
                volume_current_hour = volume_avg_hour
                volume_ratio = volume_current_hour / volume_avg_hour if volume_avg_hour > 0 else 1.0

                is_high_volume = volume_ratio > 1.5

                logger.info(
                    "Volume data fetched",
                    volume_24h=f"${volume_24h:,.0f}",
                    volume_ratio=f"{volume_ratio:.2f}x",
                    is_high_volume=is_high_volume
                )

                # Import here to avoid circular import
                from polymarket.models import VolumeData

                return VolumeData(
                    volume_24h=volume_24h,
                    volume_current_hour=volume_current_hour,
                    volume_avg_hour=volume_avg_hour,
                    volume_ratio=volume_ratio,
                    is_high_volume=is_high_volume,
                    timestamp=datetime.now()
                )

        except Exception as e:
            logger.error("Failed to fetch volume data", error=str(e))
            return None

    async def get_price_at_offset(self, hours: int) -> Decimal | None:
        """
        Get BTC price at a time offset (hours ago).

        Args:
            hours: Number of hours back to get price

        Returns:
            BTC price at that time, or None if unavailable
        """
        try:
            from datetime import timedelta

            # Calculate target timestamp
            target_time = datetime.now(timezone.utc) - timedelta(hours=hours)
            target_timestamp = int(target_time.timestamp())

            # Use existing method to get price at timestamp
            price = await self.get_price_at_timestamp(target_timestamp)

            if price:
                logger.debug(
                    f"Price {hours}h ago",
                    price=f"${price:,.2f}"
                )

            return price

        except Exception as e:
            logger.error(f"Failed to get price at {hours}h offset", error=str(e))
            return None

    async def calculate_15min_volatility(self) -> float:
        """
        Calculate 15-minute rolling volatility from price buffer.

        Uses standard deviation of returns over the last 15 minutes
        to measure market uncertainty for probability calculations.

        Returns:
            Volatility as decimal (e.g., 0.008 = 0.8%)
            Falls back to 0.005 if data unavailable
        """
        try:
            if not self._stream or not self._stream.price_buffer:
                logger.warning(
                    "Price buffer unavailable for volatility calculation",
                    has_stream=bool(self._stream),
                    has_buffer=bool(self._stream.price_buffer if self._stream else False)
                )
                return 0.005

            # Get prices from last 15 minutes (900 seconds)
            import time
            current_time = int(time.time())
            start_time = current_time - 900

            prices = await self._stream.price_buffer.get_price_range(
                start=start_time,
                end=current_time
            )

            if len(prices) < 2:
                logger.warning(
                    "Insufficient price data for volatility",
                    count=len(prices),
                    required=2
                )
                return 0.005

            # Calculate returns (percentage changes between consecutive prices)
            returns = []
            for i in range(1, len(prices)):
                prev_price = float(prices[i-1].price)
                curr_price = float(prices[i].price)

                if prev_price > 0:
                    ret = (curr_price - prev_price) / prev_price
                    returns.append(ret)

            if len(returns) < 2:
                logger.warning(
                    "Insufficient returns for volatility",
                    count=len(returns),
                    required=2
                )
                return 0.005

            # Calculate standard deviation (volatility)
            volatility = statistics.stdev(returns)

            # Sanity check (reasonable range for BTC)
            if volatility < 0.0001 or volatility > 0.05:
                logger.warning(
                    "Volatility outside expected range",
                    volatility=f"{volatility:.4f}",
                    expected_range="0.0001 to 0.05"
                )
                return 0.005

            logger.info(
                "Calculated 15min volatility",
                volatility=f"{volatility:.4f}",
                volatility_pct=f"{volatility*100:.2f}%",
                data_points=len(returns),
                price_points=len(prices)
            )

            return volatility

        except Exception as e:
            logger.error(
                "Volatility calculation failed",
                error=str(e),
                error_type=type(e).__name__
            )
            return 0.005

    async def get_order_flow_signal(self, minutes: int = 10) -> "OrderFlowSignal | None":
        """
        Compute buy/sell pressure signal from sequential Kraken 1-min candle CVD.

        Uses price direction between consecutive candles as a proxy for candle CVD:
        - Rising candle (close[N] > close[N-1]): volume counted as buy pressure
        - Falling candle: volume counted as sell pressure

        This detects whether recent BTC moves have been driven by buying or selling,
        which helps confirm if Polymarket CLOB is lagging a genuine momentum move.

        Returns:
            OrderFlowSignal or None if insufficient data
        """
        try:
            # Fetch slightly more than needed to have N+1 candles for N differences
            history = await self.get_price_history(minutes=max(minutes + 5, 20))
            if len(history) < 5:
                logger.warning("Insufficient price history for order flow signal", candles=len(history))
                return None

            # Use the last `minutes` candles (but keep one extra for first diff)
            window = history[-(minutes + 1):] if len(history) >= minutes + 1 else history

            # Compute candle CVD from sequential price direction
            buy_volume = Decimal("0")
            sell_volume = Decimal("0")
            for i in range(1, len(window)):
                vol = window[i].volume
                if vol == 0:
                    continue
                if window[i].price >= window[i - 1].price:
                    buy_volume += vol
                else:
                    sell_volume += vol

            total_volume = buy_volume + sell_volume
            if total_volume == 0:
                return None

            cvd_raw = float(buy_volume - sell_volume)
            cvd_normalized = float((buy_volume - sell_volume) / total_volume)

            # Volume acceleration: last 2 candles vs last 10 candles average
            recent_candles = window[-3:]   # last 2 diffs
            base_candles = window[-11:]    # last 10 diffs
            vol_recent = sum(float(p.volume) for p in recent_candles) / max(len(recent_candles), 1)
            vol_base = sum(float(p.volume) for p in base_candles) / max(len(base_candles), 1)
            volume_acceleration = vol_recent / vol_base if vol_base > 0 else 1.0

            # Price velocity: $ change per minute
            velocity_1min = float(window[-1].price - window[-2].price) if len(window) >= 2 else 0.0
            velocity_5min = (
                float(window[-1].price - window[-6].price) / 5.0
                if len(window) >= 6 else velocity_1min
            )

            # Direction classification based on CVD strength
            if cvd_normalized > 0.20:
                direction = "BUYING"
            elif cvd_normalized < -0.20:
                direction = "SELLING"
            else:
                direction = "NEUTRAL"

            confidence = min(abs(cvd_normalized) * 2.5, 1.0)

            signal = OrderFlowSignal(
                cvd_raw=cvd_raw,
                cvd_normalized=cvd_normalized,
                volume_acceleration=volume_acceleration,
                velocity_1min=velocity_1min,
                velocity_5min=velocity_5min,
                direction=direction,
                confidence=confidence,
            )
            logger.debug(
                "Order flow signal computed",
                cvd_normalized=f"{cvd_normalized:+.2f}",
                direction=direction,
                volume_acceleration=f"{volume_acceleration:.2f}x",
                velocity_1min=f"${velocity_1min:+.1f}",
                velocity_5min=f"${velocity_5min:+.1f}/min",
            )
            return signal

        except Exception as e:
            logger.warning("Order flow signal computation failed", error=str(e))
            return None

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
