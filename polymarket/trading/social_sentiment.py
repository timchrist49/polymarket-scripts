"""
Social Sentiment Service

Analyzes BTC market sentiment using crypto-specific APIs.
"""

import asyncio
from datetime import datetime
from typing import Optional
import structlog
import aiohttp

from polymarket.models import SocialSentiment
from polymarket.config import Settings

logger = structlog.get_logger()


class SocialSentimentService:
    """Social sentiment analysis using crypto-specific APIs."""

    # API endpoints (all public, no auth)
    FEAR_GREED_URL = "https://api.alternative.me/fng/?limit=1"
    COINGECKO_TRENDING_URL = "https://api.coingecko.com/api/v3/search/trending"
    COINGECKO_BTC_URL = "https://api.coingecko.com/api/v3/coins/bitcoin"

    # Weights for score calculation
    WEIGHTS = {
        "fear_greed": 0.4,
        "trending": 0.3,
        "votes": 0.3
    }

    def __init__(self, settings: Settings):
        self.settings = settings
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Lazy init of aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5))
        return self._session

    async def close(self):
        """Close aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def _fetch_fear_greed(self) -> int:
        """Fetch Fear & Greed Index (0-100)."""
        try:
            session = await self._get_session()
            async with session.get(self.FEAR_GREED_URL) as response:
                data = await response.json()
                value = int(data["data"][0]["value"])
                logger.debug("Fear & Greed fetched", value=value)
                return value
        except Exception as e:
            logger.warning("Fear & Greed fetch failed", error=str(e))
            return 50  # Neutral fallback

    async def _fetch_trending(self) -> bool:
        """Check if BTC is in top 3 trending on CoinGecko."""
        try:
            session = await self._get_session()
            async with session.get(self.COINGECKO_TRENDING_URL) as response:
                data = await response.json()
                coins = data.get("coins", [])

                # Check if BTC is in top 3
                for i, item in enumerate(coins[:3]):
                    coin_id = item.get("item", {}).get("id", "").lower()
                    if coin_id == "bitcoin":
                        logger.debug("BTC trending", rank=i+1)
                        return True

                logger.debug("BTC not trending")
                return False
        except Exception as e:
            logger.warning("Trending fetch failed", error=str(e))
            return False

    async def _fetch_sentiment_votes(self) -> tuple[float, float]:
        """Fetch CoinGecko community sentiment votes (up%, down%)."""
        try:
            session = await self._get_session()
            async with session.get(self.COINGECKO_BTC_URL) as response:
                data = await response.json()
                up_pct = float(data.get("sentiment_votes_up_percentage", 50))
                down_pct = float(data.get("sentiment_votes_down_percentage", 50))
                logger.debug("Sentiment votes fetched", up=up_pct, down=down_pct)
                return up_pct, down_pct
        except Exception as e:
            logger.warning("Sentiment votes fetch failed", error=str(e))
            return 50.0, 50.0  # Neutral fallback

    def _calculate_score(
        self,
        fear_greed: int,
        is_trending: bool,
        vote_up_pct: float,
        vote_down_pct: float
    ) -> float:
        """
        Calculate social sentiment score (-1 to +1).

        Components:
        - Fear/Greed: 0-100 → -1 to +1 (40% weight)
        - Trending: False=0, True=0.5 (30% weight)
        - Votes: up% vs down% → -1 to +1 (30% weight)
        """
        # Convert fear/greed to -1 to +1 scale
        fg_score = (fear_greed - 50) / 50  # 0→-1, 50→0, 100→+1

        # Trending score
        trend_score = 0.5 if is_trending else 0.0

        # Votes score
        vote_score = (vote_up_pct - vote_down_pct) / 100  # -1 to +1

        # Weighted average
        score = (
            fg_score * self.WEIGHTS["fear_greed"] +
            trend_score * self.WEIGHTS["trending"] +
            vote_score * self.WEIGHTS["votes"]
        )

        return score

    def _classify_signal(self, score: float, confidence: float) -> str:
        """Classify signal strength."""
        direction = "BULLISH" if score > 0.1 else "BEARISH" if score < -0.1 else "NEUTRAL"
        strength = "STRONG" if confidence >= 0.7 else "WEAK" if confidence >= 0.5 else "CONFLICTED"
        return f"{strength}_{direction}"

    async def get_social_score(self) -> SocialSentiment:
        """
        Get current social sentiment score.

        Returns:
            SocialSentiment with score (-0.7 to +0.85), confidence (0 to 1), and metadata.

        Note:
            Score range is asymmetric due to trending being one-sided (0 or +0.5).
            - Maximum (+0.85): Greed=100, Trending=True, Votes=100% up
            - Minimum (-0.7): Fear=0, Not trending, Votes=100% down
        """
        try:
            # Fetch all sources in parallel
            results = await asyncio.gather(
                self._fetch_fear_greed(),
                self._fetch_trending(),
                self._fetch_sentiment_votes(),
                return_exceptions=True
            )

            # Unpack results
            fear_greed, is_trending, votes = results

            # Track which sources succeeded
            sources_available = []

            # Handle Fear & Greed
            if isinstance(fear_greed, Exception):
                logger.warning("Fear/Greed failed", error=str(fear_greed))
                fear_greed = 50
            else:
                sources_available.append("fear_greed")

            # Handle Trending
            if isinstance(is_trending, Exception):
                logger.warning("Trending failed", error=str(is_trending))
                is_trending = False
            else:
                sources_available.append("trending")

            # Handle Votes
            if isinstance(votes, Exception):
                logger.warning("Votes failed", error=str(votes))
                vote_up_pct, vote_down_pct = 50.0, 50.0
            else:
                vote_up_pct, vote_down_pct = votes
                sources_available.append("votes")

            # Calculate score
            score = self._calculate_score(fear_greed, is_trending, vote_up_pct, vote_down_pct)

            # Calculate confidence based on sources available
            confidence = len(sources_available) / 3  # 3 total sources

            # Classify signal
            signal_type = self._classify_signal(score, confidence)

            logger.info(
                "Social sentiment calculated",
                score=f"{score:+.2f}",
                confidence=f"{confidence:.2f}",
                signal=signal_type,
                sources=sources_available
            )

            return SocialSentiment(
                score=score,
                confidence=confidence,
                fear_greed=fear_greed,
                is_trending=is_trending,
                vote_up_pct=vote_up_pct,
                vote_down_pct=vote_down_pct,
                signal_type=signal_type,
                sources_available=sources_available,
                timestamp=datetime.now()
            )

        except Exception as e:
            logger.error("Social sentiment failed", error=str(e))
            # Return neutral sentiment on complete failure
            return SocialSentiment(
                score=0.0,
                confidence=0.0,
                fear_greed=50,
                is_trending=False,
                vote_up_pct=50.0,
                vote_down_pct=50.0,
                signal_type="UNAVAILABLE",
                sources_available=[],
                timestamp=datetime.now()
            )
