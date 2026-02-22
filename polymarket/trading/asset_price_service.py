"""Generic price service for non-BTC assets (ETH, SOL, XRP, etc.).

Polls CoinGecko Pro every 30s and maintains a rolling price buffer.
Used by auto_trade_v2.py to get:
  - Current price (fresh live fetch)
  - Price at a past timestamp (PTB lookup from buffer)
  - Realized volatility per minute

BTC continues to use the existing BTCPriceService (Chainlink oracle + buffer).
This service is for all other assets.

CoinGecko coin IDs:
  ethereum → ETH
  solana   → SOL
  ripple   → XRP

Price accuracy:
  The MarketMonitor in auto_trade_v2.py captures a live price via
  get_current_price() at market open (T=0) to use as the PTB, and again
  at market close (T+5min/T+15min) for settlement.  This minimises the
  CoinGecko ↔ Chainlink price delta from ~$0.24 (stale buffer) to <$0.05
  (fresh fetch at the exact oracle-check moment).
"""
import asyncio
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

import aiohttp
import structlog

logger = structlog.get_logger()

_CG_BASE_URL = "https://pro-api.coingecko.com/api/v3"

# How often to poll CoinGecko for latest price (seconds)
_POLL_INTERVAL_SEC = 30

# How many price points to keep in buffer (~60 min at 30s interval = 120 points)
_BUFFER_MAX_LEN = 180


@dataclass
class PricePoint:
    timestamp: float   # Unix timestamp (seconds)
    price: float       # USD price


class AssetPriceService:
    """Price feed for a single CoinGecko asset.

    Continuously polls CoinGecko Pro and keeps a rolling buffer of price
    snapshots so that PTB lookup and realized volatility work the same way
    as the BTC Chainlink oracle buffer.

    Usage:
        service = AssetPriceService("ethereum", api_key="...")
        await service.start()           # starts background poll loop
        price = await service.get_current_price()
        ptb   = await service.get_price_at_timestamp(market_start_ts)
        vol   = service.get_realized_vol_per_min()
        await service.stop()
    """

    def __init__(self, coin_id: str, api_key: str):
        self.coin_id = coin_id
        self._api_key = api_key
        self._buffer: deque[PricePoint] = deque(maxlen=_BUFFER_MAX_LEN)
        self._session: Optional[aiohttp.ClientSession] = None
        self._task: Optional[asyncio.Task] = None
        self._current_price: Optional[float] = None

    async def start(self) -> None:
        """Start background polling loop. Call once at bot startup."""
        # Fetch immediately so PTB is available before any market fires
        await self._fetch_and_store()
        self._task = asyncio.create_task(self._poll_loop(), name=f"price_{self.coin_id}")
        logger.info("AssetPriceService started", coin=self.coin_id,
                    price=self._current_price)

    async def stop(self) -> None:
        """Stop polling and close HTTP session."""
        if self._task:
            self._task.cancel()
        if self._session and not self._session.closed:
            await self._session.close()

    async def get_current_price(self) -> Optional[float]:
        """Return a fresh live price (USD).

        Always performs a live CoinGecko fetch to minimise the delta vs the
        Chainlink oracle price used by Polymarket for market resolution.
        """
        await self._fetch_and_store()
        return self._current_price

    async def get_price_at_timestamp(self, ts: float) -> Optional[float]:
        """Return the price closest to the given Unix timestamp.

        Used as a fallback PTB lookup (when open-time capture is unavailable).
        Checks the CoinGecko rolling buffer, then falls back to historical API.
        """
        if not self._buffer:
            await self._fetch_and_store()

        # Find the CoinGecko buffer entry closest to ts
        best: Optional[PricePoint] = None
        best_diff = float("inf")
        for pt in self._buffer:
            diff = abs(pt.timestamp - ts)
            if diff < best_diff:
                best_diff = diff
                best = pt

        if best and best_diff <= 300:   # within 5 min is acceptable
            return best.price

        # Buffer doesn't cover it — fetch from CoinGecko history
        return await self._fetch_historical_price(ts)

    def get_realized_vol_per_min(self) -> float:
        """Compute realized volatility as std-dev of per-minute price changes.

        Returns 0.0 if insufficient data.
        """
        if len(self._buffer) < 4:
            return 0.0

        pts = list(self._buffer)
        # Group into ~60s buckets and compute changes
        changes = []
        for i in range(1, len(pts)):
            dt_sec = pts[i].timestamp - pts[i - 1].timestamp
            if dt_sec <= 0:
                continue
            # Scale change to per-minute rate
            change_per_min = abs(pts[i].price - pts[i - 1].price) * (60.0 / dt_sec)
            changes.append(change_per_min)

        if not changes:
            return 0.0

        mean = sum(changes) / len(changes)
        variance = sum((c - mean) ** 2 for c in changes) / len(changes)
        return variance ** 0.5

    # ------------------------------------------------------------------ #
    # Internal                                                             #
    # ------------------------------------------------------------------ #

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={"x-cg-pro-api-key": self._api_key}
            )
        return self._session

    async def _fetch_and_store(self) -> None:
        """Poll CoinGecko for current price and append to buffer."""
        session = await self._get_session()
        try:
            async with session.get(
                f"{_CG_BASE_URL}/simple/price",
                params={"ids": self.coin_id, "vs_currencies": "usd"},
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
                price = float(data[self.coin_id]["usd"])
                pt = PricePoint(timestamp=time.time(), price=price)
                self._buffer.append(pt)
                self._current_price = price
        except Exception as e:
            logger.warning("AssetPriceService poll failed", coin=self.coin_id, error=str(e))

    async def _fetch_historical_price(self, ts: float) -> Optional[float]:
        """Fetch a historical price from CoinGecko market_chart for PTB lookup."""
        session = await self._get_session()
        try:
            async with session.get(
                f"{_CG_BASE_URL}/coins/{self.coin_id}/market_chart",
                params={"vs_currency": "usd", "days": "1"},
                timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
                prices = data.get("prices", [])  # [[timestamp_ms, price], ...]
                if not prices:
                    return None
                # Find closest point to ts
                best_price = None
                best_diff = float("inf")
                for ts_ms, price in prices:
                    diff = abs(ts_ms / 1000.0 - ts)
                    if diff < best_diff:
                        best_diff = diff
                        best_price = price
                return best_price
        except Exception as e:
            logger.warning("AssetPriceService historical fetch failed",
                           coin=self.coin_id, error=str(e))
            return None

    async def _poll_loop(self) -> None:
        """Background loop: poll every _POLL_INTERVAL_SEC seconds."""
        while True:
            await asyncio.sleep(_POLL_INTERVAL_SEC)
            await self._fetch_and_store()
