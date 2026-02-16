"""Real-time odds streamer using Polymarket CLOB WebSocket."""

import asyncio
import structlog
from datetime import datetime
from typing import Optional
import websockets
from websockets.asyncio.client import ClientConnection

from polymarket.client import PolymarketClient
from polymarket.models import WebSocketOddsSnapshot

logger = structlog.get_logger()


class RealtimeOddsStreamer:
    """
    Persistent WebSocket streamer for real-time market odds.

    Maintains connection to Polymarket CLOB, processes book messages,
    provides zero-latency odds access.
    """

    # WebSocket URL
    WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"

    # Exponential backoff delays (seconds)
    BACKOFF_DELAYS = [1, 2, 4, 8, 16, 32, 60]

    def __init__(self, client: PolymarketClient):
        """
        Initialize streamer.

        Args:
            client: Polymarket client for market discovery
        """
        self.client = client
        self._current_odds: dict[str, WebSocketOddsSnapshot] = {}
        self._current_market_id: Optional[str] = None
        self._current_token_ids: Optional[list[str]] = None
        self._ws: Optional[ClientConnection] = None
        self._stream_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
        self._running = False

    def get_current_odds(self, market_id: str) -> Optional[WebSocketOddsSnapshot]:
        """
        Get current odds for market (thread-safe, non-blocking).

        Args:
            market_id: Market ID to lookup

        Returns:
            WebSocketOddsSnapshot if available, None if no data yet
        """
        return self._current_odds.get(market_id)

    async def _process_book_message(self, payload: dict):
        """
        Extract odds from book message and update state.

        Args:
            payload: Book message payload with bids/asks
        """
        try:
            market_id = payload.get('market')
            if not market_id:
                logger.warning("Book message missing market_id", payload=payload)
                return

            # Extract best bid (YES odds)
            bids = payload.get('bids', [])
            yes_odds = float(bids[0][0]) if bids else 0.50
            no_odds = 1.0 - yes_odds

            # Create snapshot
            snapshot = WebSocketOddsSnapshot(
                market_id=market_id,
                yes_odds=yes_odds,
                no_odds=no_odds,
                timestamp=datetime.now(),
                best_bid=yes_odds,
                best_ask=no_odds
            )

            # Store (thread-safe)
            async with self._lock:
                self._current_odds[market_id] = snapshot

            logger.debug(
                "Odds updated from book",
                market_id=market_id,
                yes_odds=f"{yes_odds:.2f}",
                no_odds=f"{no_odds:.2f}"
            )

        except Exception as e:
            logger.error("Failed to process book message", error=str(e), payload=payload)
