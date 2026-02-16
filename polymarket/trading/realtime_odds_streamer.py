"""Real-time odds streamer using Polymarket CLOB WebSocket."""

import asyncio
import json
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
        snapshot = self._current_odds.get(market_id)

        if snapshot:
            # Check staleness
            from datetime import datetime, timedelta
            age = datetime.now() - snapshot.timestamp
            if age > timedelta(minutes=2):
                logger.warning(
                    "⚠️ Using stale odds data (WebSocket may be disconnected)",
                    market_id=market_id,
                    age_seconds=int(age.total_seconds())
                )

        return snapshot

    async def _process_book_message(self, payload: dict):
        """
        Extract odds from book message and update state.

        Args:
            payload: Book message with buys/sells arrays
        """
        try:
            market_id = payload.get('market')
            if not market_id:
                logger.warning("Book message missing market_id", payload=payload)
                return

            # Extract best buy price (YES odds)
            # Format: {"price": "0.45", "size": "100"}
            buys = payload.get('buys', [])
            if buys and len(buys) > 0:
                # Handle both dict format and array format
                if isinstance(buys[0], dict):
                    yes_odds = float(buys[0]['price'])
                else:
                    yes_odds = float(buys[0][0])
            else:
                yes_odds = 0.50

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

    async def start(self):
        """
        Start streaming (non-blocking).

        Launches background task to connect and stream messages.
        """
        if self._running:
            logger.warning("Streamer already running")
            return

        self._running = True
        self._stream_task = asyncio.create_task(self._stream_loop())
        logger.info("Real-time odds streamer started")

    async def stop(self):
        """
        Stop streaming gracefully.

        Closes WebSocket connection and cancels background task.
        """
        self._running = False

        if self._ws:
            try:
                await self._ws.close()
            except Exception as e:
                logger.debug("Error closing WebSocket", error=str(e))

        if self._stream_task:
            self._stream_task.cancel()
            try:
                await self._stream_task
            except asyncio.CancelledError:
                pass

        logger.info("Real-time odds streamer stopped")

    async def _stream_loop(self):
        """
        Main streaming loop with reconnection logic.

        Connects, subscribes, processes messages until stopped.
        Implements exponential backoff with alerts on extended failures.
        """
        backoff_index = 0
        consecutive_failures = 0

        while self._running:
            try:
                await self._connect_and_stream()

                # Success! Reset counters
                backoff_index = 0
                consecutive_failures = 0

            except asyncio.CancelledError:
                raise  # Re-raise to properly cancel
            except websockets.ConnectionClosed:
                consecutive_failures += 1
                logger.warning(
                    "WebSocket closed, reconnecting...",
                    consecutive_failures=consecutive_failures
                )
            except Exception as e:
                consecutive_failures += 1
                logger.error(
                    "Stream error",
                    error=str(e),
                    consecutive_failures=consecutive_failures
                )

            if not self._running:
                break

            # Alert on extended failures
            if consecutive_failures >= 5:
                logger.error(
                    "⚠️ Odds streamer disconnected for extended period",
                    consecutive_failures=consecutive_failures
                )
                # TODO: Add Telegram alert in Phase 3

            # Exponential backoff
            delay = self.BACKOFF_DELAYS[min(backoff_index, len(self.BACKOFF_DELAYS)-1)]
            backoff_index += 1
            logger.debug("Reconnecting after delay", delay=delay)
            await asyncio.sleep(delay)

    async def _check_market_transition(self) -> bool:
        """
        Check if current market has changed.

        Returns:
            True if market changed, False if same
        """
        try:
            market = self.client.discover_btc_15min_market()

            if market.id != self._current_market_id:
                logger.info(
                    "Market transition detected",
                    old_market=self._current_market_id,
                    new_market=market.id
                )
                return True

            return False

        except Exception as e:
            logger.error("Failed to check market transition", error=str(e))
            return False

    async def _connect_and_stream(self):
        """
        Connect to WebSocket and stream messages until error or disconnect.

        Checks for market transitions every 60 seconds and resubscribes if needed.
        """
        # Discover current market
        market = self.client.discover_btc_15min_market()
        token_ids = market.get_token_ids()

        if not token_ids:
            logger.error("No token IDs found for market", market_id=market.id)
            return

        self._current_market_id = market.id
        self._current_token_ids = token_ids

        logger.info(
            "Connecting to CLOB WebSocket",
            market_id=market.id,
            token_ids=token_ids
        )

        async with websockets.connect(
            self.WS_URL,
            ping_interval=20,
            ping_timeout=10
        ) as ws:
            self._ws = ws

            # Send subscription message (CLOB format)
            subscribe_msg = {
                "assets_ids": token_ids,
                "type": "market"  # lowercase per CLOB spec
            }
            await ws.send(json.dumps(subscribe_msg))
            logger.info("Subscribed to market", market_id=market.id, token_ids=token_ids)

            # Track last market check time
            last_market_check = asyncio.get_event_loop().time()
            MARKET_CHECK_INTERVAL = 60  # seconds

            # Process messages until disconnected
            async for message in ws:
                if not self._running:
                    break

                # Periodic market transition check
                now = asyncio.get_event_loop().time()
                if now - last_market_check > MARKET_CHECK_INTERVAL:
                    last_market_check = now

                    if await self._check_market_transition():
                        # Market changed! Close connection to trigger resubscription
                        logger.info("Closing connection to resubscribe to new market")
                        await ws.close()
                        break

                try:
                    data = json.loads(message)

                    # Handle both single message and array of messages
                    if isinstance(data, list):
                        # Array of messages - process each
                        for msg in data:
                            if isinstance(msg, dict):
                                await self._handle_single_message(msg)
                    elif isinstance(data, dict):
                        # Single message
                        await self._handle_single_message(data)

                except json.JSONDecodeError:
                    logger.warning("Invalid JSON message", message=message[:100])
                except Exception as e:
                    logger.error("Message processing error", error=str(e))

    async def _handle_single_message(self, data: dict):
        """Process a single WebSocket message."""
        event_type = data.get('event_type')

        logger.debug("WebSocket message received", event_type=event_type)

        if event_type == 'book':
            # For book messages, the data IS the payload
            await self._process_book_message(data)

        # Ignore other message types (last_trade_price, price_change)
