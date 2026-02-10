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
    WHALE_SIZE_USD = 1000  # Trades > $1,000 considered whale

    def __init__(self, settings: Settings, condition_id: Optional[str] = None):
        self.settings = settings
        self.condition_id = condition_id
        self._session: Optional[aiohttp.ClientSession] = None
        self._klines_cache: list = []  # Cache for momentum calculation

    async def _get_session(self) -> aiohttp.ClientSession:
        """Lazy init of aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15))  # Increased from 5 to 15s
        return self._session

    async def close(self):
        """Close aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def _fetch_order_book(self) -> dict:
        """Fetch Binance order book (top 100 levels)."""
        session = await self._get_session()
        url = f"{self.BASE_URL}/depth?symbol=BTCUSDT&limit=100"

        # Retry up to 2 times on failure
        for attempt in range(2):
            try:
                async with session.get(url) as response:
                    data = await response.json()
                    logger.debug("Order book fetched", bids=len(data["bids"]), asks=len(data["asks"]))
                    return data
            except Exception as e:
                if attempt == 1:  # Last attempt
                    logger.error("Order book fetch failed after retries", error=str(e))
                    return {"bids": [], "asks": []}
                logger.warning(f"Order book fetch attempt {attempt+1} failed, retrying...", error=str(e))
                await asyncio.sleep(1)  # Wait 1s before retry

        return {"bids": [], "asks": []}

    async def _fetch_recent_trades(self) -> list:
        """Fetch last 100 trades from Binance."""
        session = await self._get_session()
        url = f"{self.BASE_URL}/trades?symbol=BTCUSDT&limit=100"

        # Retry up to 2 times on failure
        for attempt in range(2):
            try:
                async with session.get(url) as response:
                    data = await response.json()
                    logger.debug("Trades fetched", count=len(data))
                    return data
            except Exception as e:
                if attempt == 1:  # Last attempt
                    logger.error("Trades fetch failed after retries", error=str(e))
                    return []
                logger.warning(f"Trades fetch attempt {attempt+1} failed, retrying...", error=str(e))
                await asyncio.sleep(1)  # Wait 1s before retry

        return []

    async def _fetch_24hr_ticker(self) -> dict:
        """Fetch 24hr ticker data."""
        session = await self._get_session()
        url = f"{self.BASE_URL}/ticker/24hr?symbol=BTCUSDT"

        # Retry up to 2 times on failure
        for attempt in range(2):
            try:
                async with session.get(url) as response:
                    data = await response.json()
                    logger.debug("Ticker fetched", volume=data.get("volume"))
                    return data
            except Exception as e:
                if attempt == 1:  # Last attempt
                    logger.error("Ticker fetch failed after retries", error=str(e))
                    return {"volume": "0", "count": 0}
                logger.warning(f"Ticker fetch attempt {attempt+1} failed, retrying...", error=str(e))
                await asyncio.sleep(1)  # Wait 1s before retry

        return {"volume": "0", "count": 0}

    async def _fetch_klines(self, limit: int = 15) -> list:
        """Fetch 1-minute klines for momentum calculation."""
        session = await self._get_session()
        url = f"{self.BASE_URL}/klines?symbol=BTCUSDT&interval=1m&limit={limit}"

        # Retry up to 2 times on failure
        for attempt in range(2):
            try:
                async with session.get(url) as response:
                    data = await response.json()
                    self._klines_cache = data
                    logger.debug("Klines fetched", count=len(data))
                    return data
            except Exception as e:
                if attempt == 1:  # Last attempt
                    logger.error("Klines fetch failed after retries", error=str(e))
                    return self._klines_cache or []  # Use cached if available
                logger.warning(f"Klines fetch attempt {attempt+1} failed, retrying...", error=str(e))
                await asyncio.sleep(1)  # Wait 1s before retry

        return self._klines_cache or []

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

    async def collect_market_data(
        self,
        condition_id: str,
        duration_seconds: int = 60
    ) -> dict:
        """
        Collect market data via WebSocket for the specified duration.

        SKELETON: Currently returns empty structure. Will implement
        real WebSocket connection to wss://ws-subscriptions-clob.polymarket.com/ws
        in next iteration.

        Args:
            condition_id: Polymarket condition ID to monitor
            duration_seconds: How long to collect data

        Returns:
            dict with keys:
                - trades: list of trade events
                - book_snapshots: list of order book snapshots
                - price_changes: list of price change events
        """
        logger.info(
            "collect_market_data called (skeleton)",
            condition_id=condition_id,
            duration=duration_seconds
        )

        # TODO: Implement real WebSocket connection
        # Will connect to wss://ws-subscriptions-clob.polymarket.com/ws
        # Subscribe to market channel for condition_id
        # Collect data for duration_seconds

        # For now, return empty structure
        return {
            'trades': [],
            'book_snapshots': [],
            'price_changes': []
        }

    def calculate_momentum_score(self, trades: list) -> float:
        """
        Calculate YES token price momentum over collection window.

        Args:
            trades: List of trade messages with asset_id, price, size

        Returns:
            -1.0 (strong bearish) to +1.0 (strong bullish)
        """
        if not trades:
            return 0.0

        # Filter YES token trades
        yes_trades = [t for t in trades if t.get('asset_id') == 'YES_TOKEN']
        if len(yes_trades) < 2:
            return 0.0

        # Get first and last YES price in window
        initial_yes_price = yes_trades[0]['price']
        final_yes_price = yes_trades[-1]['price']

        # Calculate percentage change
        price_change_pct = (final_yes_price - initial_yes_price) / initial_yes_price

        # Normalize: ±10% change maps to ±1.0 score
        # Clamp to [-1.0, 1.0] range
        momentum_score = max(min(price_change_pct * 10, 1.0), -1.0)

        logger.debug(
            "Momentum calculated",
            initial=initial_yes_price,
            final=final_yes_price,
            change_pct=f"{price_change_pct*100:+.2f}%",
            score=f"{momentum_score:+.2f}"
        )

        return momentum_score

    def calculate_volume_flow_score(self, trades: list) -> float:
        """
        Calculate net buying pressure (YES volume - NO volume).

        Args:
            trades: List of trade messages with asset_id and size

        Returns:
            -1.0 (all NO buying) to +1.0 (all YES buying)
        """
        if not trades:
            return 0.0

        yes_volume = sum(
            trade['size'] for trade in trades
            if trade.get('asset_id') == 'YES_TOKEN'
        )

        no_volume = sum(
            trade['size'] for trade in trades
            if trade.get('asset_id') == 'NO_TOKEN'
        )

        total_volume = yes_volume + no_volume
        if total_volume == 0:
            return 0.0

        # Already normalized to -1.0 to +1.0
        volume_flow_score = (yes_volume - no_volume) / total_volume

        logger.debug(
            "Volume flow calculated",
            yes_volume=yes_volume,
            no_volume=no_volume,
            score=f"{volume_flow_score:+.2f}"
        )

        return volume_flow_score

    def calculate_whale_activity_score(self, trades: list) -> float:
        """
        Calculate directional signal from whale trades (>$1,000).

        Args:
            trades: List of trade messages with asset_id and size

        Returns:
            -1.0 (all NO whales) to +1.0 (all YES whales)
        """
        if not trades:
            return 0.0

        # Identify whale trades (size > $1,000)
        yes_whales = sum(
            1 for trade in trades
            if trade['size'] > self.WHALE_SIZE_USD and trade.get('asset_id') == 'YES_TOKEN'
        )

        no_whales = sum(
            1 for trade in trades
            if trade['size'] > self.WHALE_SIZE_USD and trade.get('asset_id') == 'NO_TOKEN'
        )

        total_whales = yes_whales + no_whales
        if total_whales == 0:
            return 0.0

        whale_score = (yes_whales - no_whales) / total_whales

        logger.debug(
            "Whale activity calculated",
            yes_whales=yes_whales,
            no_whales=no_whales,
            score=f"{whale_score:+.2f}"
        )

        return whale_score

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
