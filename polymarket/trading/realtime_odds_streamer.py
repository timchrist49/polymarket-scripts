"""Real-time odds streamer using Polymarket CLOB WebSocket."""

import asyncio
import json
import structlog
from datetime import datetime, timezone
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
        self._current_market_slug: Optional[str] = None
        self._current_token_ids: Optional[list[str]] = None
        self._ws: Optional[ClientConnection] = None
        self._stream_task: Optional[asyncio.Task] = None
        self._rest_task: Optional[asyncio.Task] = None
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
            from datetime import timedelta
            age = datetime.now(timezone.utc) - snapshot.timestamp
            if age > timedelta(minutes=2):
                logger.warning(
                    "⚠️ Using stale odds data (WebSocket may be disconnected)",
                    market_id=market_id,
                    age_seconds=int(age.total_seconds())
                )

        return snapshot

    async def _update_odds_from_orderbook(
        self,
        bids: list,
        asks: list,
        source: str
    ) -> None:
        """
        Update odds from orderbook data (WebSocket or REST).

        Args:
            bids: List of bids as [{"price": str, "size": str}] or [[price, size]]
            asks: List of asks as [{"price": str, "size": str}] or [[price, size]]
            source: 'WebSocket' or 'REST' for logging
        """
        if not self._current_market_id:
            logger.warning(
                "No current market ID set, skipping odds update",
                source=source
            )
            return

        # Extract best bid price (YES odds) with validation
        try:
            if bids and len(bids) > 0:
                if isinstance(bids[0], dict):
                    yes_odds = float(bids[0]['price'])
                else:
                    # Array format: [price, size]
                    yes_odds = float(bids[0][0])
            else:
                # Default if no bids
                yes_odds = 0.50

            # Validate range
            if not (0.0 <= yes_odds <= 1.0):
                logger.warning(
                    "Invalid yes_odds value, using default",
                    value=yes_odds,
                    source=source
                )
                yes_odds = 0.50

        except (ValueError, KeyError, IndexError, TypeError) as e:
            logger.warning(
                "Failed to parse best bid, using default",
                error=str(e),
                source=source
            )
            yes_odds = 0.50

        # Extract best ask price with validation
        try:
            if asks and len(asks) > 0:
                if isinstance(asks[0], dict):
                    best_ask = float(asks[0]['price'])
                else:
                    # Array format: [price, size]
                    best_ask = float(asks[0][0])
            else:
                # Default if no asks: use calculated value
                best_ask = 1.0 - yes_odds

            # Validate range
            if not (0.0 <= best_ask <= 1.0):
                logger.warning(
                    "Invalid best_ask value, using calculated",
                    value=best_ask,
                    source=source
                )
                best_ask = 1.0 - yes_odds

        except (ValueError, KeyError, IndexError, TypeError) as e:
            logger.warning(
                "Failed to parse best ask, using calculated",
                error=str(e),
                source=source
            )
            best_ask = 1.0 - yes_odds

        no_odds = 1.0 - yes_odds

        # Create snapshot
        snapshot = WebSocketOddsSnapshot(
            market_id=self._current_market_id,
            yes_odds=yes_odds,
            no_odds=no_odds,
            timestamp=datetime.now(timezone.utc),
            best_bid=yes_odds,
            best_ask=best_ask
        )

        # Store atomically
        async with self._lock:
            self._current_odds[self._current_market_id] = snapshot

        logger.debug(
            "📊 Odds updated",
            source=source,
            market_id=self._current_market_id,
            market_slug=self._current_market_slug,
            yes_odds=f"{yes_odds:.2f}",
            no_odds=f"{no_odds:.2f}",
            best_ask=f"{best_ask:.2f}"
        )

    async def _process_book_message(self, payload: dict):
        """
        Extract odds from book message and update state.

        Args:
            payload: Book message with bids/asks arrays
        """
        try:
            # Book messages contain both market (condition ID) and asset_id (token ID)
            # We use market ID for validation and asset_id identifies the specific token
            market_id = payload.get('market')
            asset_id = payload.get('asset_id')

            if not market_id or not asset_id:
                logger.warning("Book message missing IDs", market_id=market_id, asset_id=asset_id)
                return

            # Verify we have a current market being tracked
            if not self._current_market_id:
                logger.warning("No current market ID set, skipping book message")
                return

            # Extract orderbook data
            bids = payload.get('bids', [])
            asks = payload.get('asks', [])

            logger.info(
                "📥 Raw book message",
                market_id=market_id[:16] + "...",
                asset_id=asset_id[:16] + "...",
                bids_count=len(bids),
                asks_count=len(asks)
            )

            # Delegate to shared processing method
            await self._update_odds_from_orderbook(bids, asks, source='WebSocket')

        except Exception as e:
            logger.error(
                "Book message processing failed",
                error=str(e),
                payload=str(payload)[:200]
            )

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

    async def _monitor_market_transitions(self, ws):
        """
        Independently monitor for market transitions.

        Runs concurrently with message reception loop.
        Closes WebSocket when market changes to trigger resubscription.

        Args:
            ws: WebSocket connection to close if transition detected
        """
        while self._running:
            await asyncio.sleep(60)  # Check every minute

            try:
                if await self._check_market_transition():
                    logger.info("Market transition detected by monitor, closing WebSocket")
                    await ws.close()
                    break
            except Exception as e:
                logger.error("Market transition check failed in monitor", error=str(e))

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

        # Store market state atomically
        # Convert token IDs to hex format for comparison with book messages
        hex_token_ids = [hex(int(tid)) for tid in token_ids]

        async with self._lock:
            self._current_market_id = market.id
            self._current_market_slug = market.slug
            self._current_token_ids = hex_token_ids

        logger.info(
            "Connecting to CLOB WebSocket",
            market_id=market.id,
            market_slug=market.slug,
            token_ids=token_ids
        )

        async with websockets.connect(
            self.WS_URL,
            ping_interval=20,
            ping_timeout=10
        ) as ws:
            self._ws = ws

            # Step 1: Send initial handshake (empty subscription)
            # This is required per the working implementation
            handshake_msg = {
                "assets_ids": [],
                "type": "market"  # lowercase per working implementation
            }
            await ws.send(json.dumps(handshake_msg))
            logger.info("📤 Sent handshake", message=json.dumps(handshake_msg))

            # Step 2: Send actual subscription with operation field
            # Convert decimal token IDs to hex format (book messages use hex)
            hex_token_ids = [hex(int(tid)) for tid in token_ids]

            subscribe_msg = {
                "operation": "subscribe",
                "assets_ids": hex_token_ids
            }
            await ws.send(json.dumps(subscribe_msg))
            logger.info("📤 Sent subscription", message=json.dumps(subscribe_msg, indent=2))

            # Enhanced logging for debugging
            logger.info(
                "🔔 Subscription sent",
                market_id=market.id,
                market_slug=market.slug,
                token_count=len(token_ids),
                token_ids=[t[:16] + "..." for t in token_ids]
            )

            # Launch independent market transition monitor
            monitor_task = asyncio.create_task(
                self._monitor_market_transitions(ws)
            )
            logger.info("Market transition monitor started")

            try:
                # Process messages until disconnected
                async for message in ws:
                    # DEBUG: Log every raw message received
                    logger.info(
                        "📨 RAW WebSocket message received",
                        length=len(message),
                        content=message[:200] if len(message) <= 200 else message[:200] + "..."
                    )

                    if not self._running:
                        break

                    try:
                        data = json.loads(message)

                        # DEBUG: Log parsed data structure
                        logger.info(
                            "📦 Parsed WebSocket data",
                            data_type=type(data).__name__,
                            data_preview=str(data)[:200]
                        )

                        # Handle both single message and array of messages
                        if isinstance(data, list):
                            # Array of messages - process each
                            logger.info(f"Processing array of {len(data)} messages")
                            for msg in data:
                                if isinstance(msg, dict):
                                    await self._handle_single_message(msg)
                        elif isinstance(data, dict):
                            # Single message
                            logger.info("Processing single dict message")
                            await self._handle_single_message(data)

                    except json.JSONDecodeError:
                        logger.warning("Invalid JSON message", message=message[:100])
                    except Exception as e:
                        logger.error("Message processing error", error=str(e))

                # Loop exited - log why
                logger.warning(
                    "WebSocket message loop exited",
                    running=self._running,
                    market_id=self._current_market_id
                )

            finally:
                # Clean up monitor task
                logger.info("Cancelling market transition monitor")
                monitor_task.cancel()
                try:
                    await monitor_task
                except asyncio.CancelledError:
                    logger.info("Market transition monitor cancelled successfully")

    async def _handle_single_message(self, data: dict):
        """Process a single WebSocket message."""
        event_type = data.get('event_type')

        logger.info("🔍 WebSocket message received", event_type=event_type)

        if event_type == 'book':
            # CRITICAL FIX: Use 'asset_id' (token ID), not 'market' (condition ID)
            token_id = data.get('asset_id')  # hex string

            # Only process messages for subscribed tokens
            if token_id and self._current_token_ids:
                if token_id not in self._current_token_ids:
                    logger.debug(
                        "Ignoring book message for unsubscribed token",
                        token_id=token_id[:16] + "...",
                        subscribed_tokens=[t[:16] + "..." for t in self._current_token_ids]
                    )
                    return

            await self._process_book_message(data)

        # Ignore other message types (last_trade_price, price_change)
