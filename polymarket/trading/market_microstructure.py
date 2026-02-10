"""
Market Microstructure Service

Analyzes Binance order book, trades, volume for short-term signals.
"""

import asyncio
from datetime import datetime
from typing import Optional
import structlog
import aiohttp

from polymarket.models import MarketSignals
from polymarket.config import Settings

logger = structlog.get_logger()


class MarketMicrostructureService:
    """Market microstructure analysis using Binance public APIs."""

    # Binance API base URL
    BASE_URL = "https://api.binance.com/api/v3"

    # Weights for score calculation
    WEIGHTS = {
        "order_book": 0.20,
        "whales": 0.25,
        "volume": 0.25,
        "momentum": 0.30  # Highest - most predictive for 15-min
    }

    # Thresholds
    WHALE_SIZE_BTC = 5.0  # Orders > 5 BTC considered "whale"
    LARGE_WALL_BTC = 10.0  # Walls > 10 BTC considered significant

    def __init__(self, settings: Settings):
        self.settings = settings
        self._session: Optional[aiohttp.ClientSession] = None
        self._klines_cache: list = []  # Cache for momentum calculation

    async def _get_session(self) -> aiohttp.ClientSession:
        """Lazy init of aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5))
        return self._session

    async def close(self):
        """Close aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def _fetch_order_book(self) -> dict:
        """Fetch Binance order book (top 100 levels)."""
        try:
            session = await self._get_session()
            url = f"{self.BASE_URL}/depth?symbol=BTCUSDT&limit=100"
            async with session.get(url) as response:
                data = await response.json()
                logger.debug("Order book fetched", bids=len(data["bids"]), asks=len(data["asks"]))
                return data
        except Exception as e:
            logger.error("Order book fetch failed", error=str(e))
            return {"bids": [], "asks": []}

    async def _fetch_recent_trades(self) -> list:
        """Fetch last 100 trades from Binance."""
        try:
            session = await self._get_session()
            url = f"{self.BASE_URL}/trades?symbol=BTCUSDT&limit=100"
            async with session.get(url) as response:
                data = await response.json()
                logger.debug("Trades fetched", count=len(data))
                return data
        except Exception as e:
            logger.error("Trades fetch failed", error=str(e))
            return []

    async def _fetch_24hr_ticker(self) -> dict:
        """Fetch 24hr ticker data."""
        try:
            session = await self._get_session()
            url = f"{self.BASE_URL}/ticker/24hr?symbol=BTCUSDT"
            async with session.get(url) as response:
                data = await response.json()
                logger.debug("Ticker fetched", volume=data.get("volume"))
                return data
        except Exception as e:
            logger.error("Ticker fetch failed", error=str(e))
            return {"volume": "0", "count": 0}

    async def _fetch_klines(self, limit: int = 15) -> list:
        """Fetch 1-minute klines for momentum calculation."""
        try:
            session = await self._get_session()
            url = f"{self.BASE_URL}/klines?symbol=BTCUSDT&interval=1m&limit={limit}"
            async with session.get(url) as response:
                data = await response.json()
                self._klines_cache = data
                logger.debug("Klines fetched", count=len(data))
                return data
        except Exception as e:
            logger.error("Klines fetch failed", error=str(e))
            return self._klines_cache  # Use cached if available

    def _score_order_book(self, order_book: dict) -> float:
        """
        Score order book bid vs ask wall strength.

        Returns:
            -1.0 (heavy ask walls) to +1.0 (heavy bid walls)
        """
        bids = order_book.get("bids", [])
        asks = order_book.get("asks", [])

        # Sum BTC in large walls (>10 BTC)
        bid_walls = sum(float(qty) for price, qty in bids if float(qty) > self.LARGE_WALL_BTC)
        ask_walls = sum(float(qty) for price, qty in asks if float(qty) > self.LARGE_WALL_BTC)

        if bid_walls + ask_walls == 0:
            return 0.0

        score = (bid_walls - ask_walls) / (bid_walls + ask_walls)
        return score

    def _score_whale_activity(self, trades: list) -> float:
        """
        Score whale activity (large orders > 5 BTC).

        Returns:
            -1.0 (whales selling) to +1.0 (whales buying)
        """
        large_buys = sum(1 for t in trades if float(t["qty"]) > self.WHALE_SIZE_BTC and t["isBuyerMaker"])
        large_sells = sum(1 for t in trades if float(t["qty"]) > self.WHALE_SIZE_BTC and not t["isBuyerMaker"])

        if large_buys + large_sells == 0:
            return 0.0

        score = (large_buys - large_sells) / (large_buys + large_sells)
        return score

    def _score_volume_spike(self, ticker: dict, klines: list) -> float:
        """
        Score volume spike vs 24h average.

        Returns:
            -1.0 (very low volume) to +1.0 (high volume spike)
        """
        try:
            # Get recent 5-min volume from klines
            recent_volume = sum(float(k[5]) for k in klines[-5:])  # Last 5 minutes

            # Get 24h average volume per 5 minutes
            total_volume_24h = float(ticker.get("volume", 0))
            avg_volume_per_5min = total_volume_24h / (24 * 60 / 5)

            if avg_volume_per_5min == 0:
                return 0.0

            # Ratio: current / average
            ratio = recent_volume / avg_volume_per_5min

            # Normalize to -1 to +1 scale
            # ratio < 0.5 → -1 (very low)
            # ratio = 1.0 → 0 (normal)
            # ratio > 2.0 → +1 (spike)
            if ratio < 1.0:
                score = (ratio - 1.0) / 0.5  # -1 to 0
            else:
                score = min((ratio - 1.0) / 1.0, 1.0)  # 0 to +1

            return score
        except Exception as e:
            logger.warning("Volume scoring failed", error=str(e))
            return 0.0

    def _score_momentum(self, klines: list) -> float:
        """
        Score price momentum/velocity.

        Returns:
            -1.0 (strong downward) to +1.0 (strong upward)
        """
        if len(klines) < 5:
            return 0.0

        try:
            # Get prices from last 5 and 15 minutes
            price_5min_ago = float(klines[-5][4])  # Close price 5 min ago
            price_15min_ago = float(klines[0][4])  # Close price 15 min ago
            price_now = float(klines[-1][4])       # Current close

            # Calculate velocity ($/min)
            velocity_5min = (price_now - price_5min_ago) / 5
            velocity_15min = (price_now - price_15min_ago) / 15

            # Normalize to -1 to +1 (assume max $50/min velocity)
            score = (velocity_5min + velocity_15min) / 2 / 50
            score = max(min(score, 1.0), -1.0)  # Clamp

            return score
        except Exception as e:
            logger.warning("Momentum scoring failed", error=str(e))
            return 0.0

    def _calculate_metric_agreement(self, scores: list[float]) -> float:
        """
        Calculate agreement between metrics.

        Returns:
            0.0 (total conflict) to 1.0 (perfect agreement)
        """
        if not scores:
            return 0.0

        # Count how many are positive vs negative
        positive = sum(1 for s in scores if s > 0.1)
        negative = sum(1 for s in scores if s < -0.1)
        neutral = len(scores) - positive - negative

        # Perfect agreement: all same direction
        # Total conflict: equal split
        max_agreement = max(positive, negative, neutral)
        agreement = max_agreement / len(scores)

        return agreement

    def _classify_signal(self, score: float, confidence: float) -> str:
        """Classify signal strength."""
        direction = "BULLISH" if score > 0.1 else "BEARISH" if score < -0.1 else "NEUTRAL"
        strength = "STRONG" if confidence >= 0.7 else "WEAK" if confidence >= 0.5 else "CONFLICTED"
        return f"{strength}_{direction}"

    async def get_market_score(self) -> MarketSignals:
        """
        Get current market microstructure score.

        Returns:
            MarketSignals with score, confidence, and detailed metrics.
        """
        try:
            # Fetch all data in parallel
            results = await asyncio.gather(
                self._fetch_order_book(),
                self._fetch_recent_trades(),
                self._fetch_24hr_ticker(),
                self._fetch_klines(limit=15),
                return_exceptions=True
            )

            order_book, trades, ticker, klines = results

            # Handle failures
            if isinstance(order_book, Exception):
                logger.error("Order book failed", error=str(order_book))
                order_book = {"bids": [], "asks": []}

            if isinstance(trades, Exception):
                logger.error("Trades failed", error=str(trades))
                trades = []

            if isinstance(ticker, Exception):
                logger.error("Ticker failed", error=str(ticker))
                ticker = {"volume": "0", "count": 0}

            if isinstance(klines, Exception):
                logger.error("Klines failed", error=str(klines))
                klines = self._klines_cache or []

            # Score each metric
            ob_score = self._score_order_book(order_book)
            whale_score = self._score_whale_activity(trades)
            volume_score = self._score_volume_spike(ticker, klines)
            momentum_score = self._score_momentum(klines)

            # Calculate weighted average
            market_score = (
                ob_score * self.WEIGHTS["order_book"] +
                whale_score * self.WEIGHTS["whales"] +
                volume_score * self.WEIGHTS["volume"] +
                momentum_score * self.WEIGHTS["momentum"]
            )

            # Calculate confidence based on internal agreement
            confidence = self._calculate_metric_agreement([ob_score, whale_score, volume_score, momentum_score])

            # Extract metadata
            order_book_bias = "BID_HEAVY" if ob_score > 0.3 else "ASK_HEAVY" if ob_score < -0.3 else "BALANCED"
            whale_direction = "BUYING" if whale_score > 0.3 else "SELLING" if whale_score < -0.3 else "NEUTRAL"
            whale_count = sum(1 for t in trades if float(t["qty"]) > self.WHALE_SIZE_BTC)
            volume_ratio = 1.0 + volume_score  # Approximate
            momentum_direction = "UP" if momentum_score > 0.1 else "DOWN" if momentum_score < -0.1 else "FLAT"

            signal_type = self._classify_signal(market_score, confidence)

            logger.info(
                "Market microstructure calculated",
                score=f"{market_score:+.2f}",
                confidence=f"{confidence:.2f}",
                signal=signal_type,
                whales=whale_count
            )

            return MarketSignals(
                score=market_score,
                confidence=confidence,
                order_book_score=ob_score,
                whale_score=whale_score,
                volume_score=volume_score,
                momentum_score=momentum_score,
                order_book_bias=order_book_bias,
                whale_direction=whale_direction,
                whale_count=whale_count,
                volume_ratio=volume_ratio,
                momentum_direction=momentum_direction,
                signal_type=signal_type,
                timestamp=datetime.now()
            )

        except Exception as e:
            logger.error("Market microstructure failed", error=str(e))
            # Return neutral on complete failure
            return MarketSignals(
                score=0.0,
                confidence=0.0,
                order_book_score=0.0,
                whale_score=0.0,
                volume_score=0.0,
                momentum_score=0.0,
                order_book_bias="UNAVAILABLE",
                whale_direction="UNAVAILABLE",
                whale_count=0,
                volume_ratio=1.0,
                momentum_direction="UNAVAILABLE",
                signal_type="UNAVAILABLE",
                timestamp=datetime.now()
            )
