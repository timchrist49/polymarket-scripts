"""Real-time odds streamer using Polymarket CLOB WebSocket."""

import asyncio
import structlog
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
