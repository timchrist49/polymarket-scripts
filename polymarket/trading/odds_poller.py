"""
Market Odds Poller

Background service that polls Polymarket API for market odds every 60 seconds.
Stores odds in shared state for early market filtering.
"""

import asyncio
import structlog
from datetime import datetime

from polymarket.client import PolymarketClient
from polymarket.models import OddsSnapshot

logger = structlog.get_logger()


class MarketOddsPoller:
    """
    Background service that polls Polymarket API for current market odds.

    Runs every 60 seconds, stores odds in shared state accessible to main trading loop.
    Enables early filtering: skip markets where neither side > 75% odds.
    """

    def __init__(self, client: PolymarketClient):
        """
        Initialize odds poller.

        Args:
            client: Polymarket client for API calls
        """
        self.client = client
        self.current_odds: dict[str, OddsSnapshot] = {}  # market_id -> odds
        self._lock = asyncio.Lock()

    async def start_polling(self):
        """
        Run polling loop every 60 seconds.

        This should be run as a background task:
            asyncio.create_task(poller.start_polling())
        """
        logger.info("Odds poller started (interval: 60s)")

        while True:
            try:
                await self._poll_current_market()
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                logger.info("Odds poller stopped")
                raise  # Re-raise to properly cancel task
            except Exception as e:
                logger.error("Odds polling failed", error=str(e))
                # Continue running despite errors

    async def _poll_current_market(self):
        """
        Fetch odds for current active market.

        Discovers current BTC 15-min market, fetches fresh odds, stores snapshot.
        """
        try:
            # Discover current BTC 15-min market
            market = self.client.discover_btc_15min_market()

            # The discovered market already has fresh best_bid/ask odds
            # No need for a separate API call

            # Extract odds
            # best_bid = market maker's bid = price to buy YES token
            # NO odds = complement (1 - YES odds)
            yes_odds = market.best_bid if market.best_bid else 0.50
            no_odds = 1.0 - yes_odds

            # Create snapshot
            snapshot = OddsSnapshot(
                market_id=market.id,
                market_slug=market.slug,
                yes_odds=yes_odds,
                no_odds=no_odds,
                timestamp=datetime.now(),
                yes_qualifies=(yes_odds > 0.75),
                no_qualifies=(no_odds > 0.75)
            )

            # Store snapshot (thread-safe)
            async with self._lock:
                self.current_odds[fresh_market.id] = snapshot

            logger.debug(
                "Odds polled",
                market_id=fresh_market.id,
                yes_odds=f"{yes_odds:.2f}",
                no_odds=f"{no_odds:.2f}",
                yes_qualifies=snapshot.yes_qualifies,
                no_qualifies=snapshot.no_qualifies
            )

        except Exception as e:
            logger.error("Failed to poll current market", error=str(e))
            # Don't raise - let polling continue

    async def get_odds(self, market_id: str) -> OddsSnapshot | None:
        """
        Get cached odds for market.

        Args:
            market_id: Market ID to lookup

        Returns:
            OddsSnapshot if cached, None if not found
        """
        async with self._lock:
            return self.current_odds.get(market_id)
