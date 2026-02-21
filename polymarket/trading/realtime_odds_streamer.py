"""Real-time odds streamer using Polymarket CLOB WebSocket."""

import asyncio
import json
import os
import structlog
from datetime import datetime, timezone, timedelta
from typing import Optional
import websockets
from websockets.asyncio.client import ClientConnection

from polymarket.client import PolymarketClient
from polymarket.models import WebSocketOddsSnapshot
from polymarket.trading.market_validator import MarketValidator

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
        self._validator = MarketValidator()
        self._current_odds: dict[str, WebSocketOddsSnapshot] = {}
        self._current_market_id: Optional[str] = None
        self._current_market_slug: Optional[str] = None
        self._current_token_ids_decimal: Optional[list[str]] = None
        self._current_token_ids_hex: Optional[list[str]] = None
        self._current_market_obj = None  # cached Market object (fallback for snapshot)
        self._ws: Optional[ClientConnection] = None
        self._stream_task: Optional[asyncio.Task] = None
        self._rest_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
        self._running = False
        self._odds_callbacks: list = []  # Registered by AutoTrader for real-time entry checks

    def _discover_current_market(self):
        """Discover the active BTC market based on MARKET_TYPE env var."""
        if os.getenv("MARKET_TYPE", "15m").lower() == "5m":
            return self.client.discover_btc_5min_market()
        return self.client.discover_btc_15min_market()

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

    def register_odds_callback(self, callback) -> None:
        """Register a coroutine callback to be called on every odds update.

        The callback receives a single WebSocketOddsSnapshot argument.
        Used by AutoTrader to trigger real-time timed entry checks.
        """
        self._odds_callbacks.append(callback)

    def _fire_odds_callbacks(self, snapshot: "WebSocketOddsSnapshot") -> None:
        """Schedule all registered callbacks with the latest snapshot (non-blocking)."""
        for cb in self._odds_callbacks:
            asyncio.create_task(cb(snapshot))

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

        # Extract best bid price (for NO odds calculation) with validation
        try:
            if bids and len(bids) > 0:
                if isinstance(bids[0], dict):
                    best_bid = float(bids[0]['price'])
                else:
                    # Array format: [price, size]
                    best_bid = float(bids[0][0])
            else:
                # Default if no bids
                best_bid = 0.50

            # Validate range
            if not (0.0 <= best_bid <= 1.0):
                logger.warning(
                    "Invalid best_bid value, using default",
                    value=best_bid,
                    source=source
                )
                best_bid = 0.50

        except (ValueError, KeyError, IndexError, TypeError) as e:
            logger.warning(
                "Failed to parse best bid, using default",
                error=str(e),
                source=source
            )
            best_bid = 0.50

        # Extract best ask price (for YES odds calculation) with validation
        try:
            if asks and len(asks) > 0:
                if isinstance(asks[0], dict):
                    best_ask = float(asks[0]['price'])
                else:
                    # Array format: [price, size]
                    best_ask = float(asks[0][0])
            else:
                # Default if no asks: use calculated value
                best_ask = 1.0 - best_bid

            # Validate range
            if not (0.0 <= best_ask <= 1.0):
                logger.warning(
                    "Invalid best_ask value, using calculated",
                    value=best_ask,
                    source=source
                )
                best_ask = 1.0 - best_bid

        except (ValueError, KeyError, IndexError, TypeError) as e:
            logger.warning(
                "Failed to parse best ask, using calculated",
                error=str(e),
                source=source
            )
            best_ask = 1.0 - best_bid

        # Calculate market odds using midpoint for opportunity detection
        # Midpoint represents fair market value between bid and ask
        # This prevents both YES and NO from showing high "odds" due to wide spread
        midpoint = (best_bid + best_ask) / 2.0
        yes_odds = midpoint
        no_odds = 1.0 - midpoint

        # Create snapshot
        snapshot = WebSocketOddsSnapshot(
            market_id=self._current_market_id,
            yes_odds=yes_odds,
            no_odds=no_odds,
            timestamp=datetime.now(timezone.utc),
            best_bid=best_bid,
            best_ask=best_ask
        )

        # Store atomically
        async with self._lock:
            self._current_odds[self._current_market_id] = snapshot

        self._fire_odds_callbacks(snapshot)

        logger.info(
            "📊 Odds updated",
            source=source,
            market_id=self._current_market_id,
            market_slug=self._current_market_slug,
            yes_odds=f"{yes_odds:.2f}",
            no_odds=f"{no_odds:.2f}",
            best_bid=f"{best_bid:.2f}",
            best_ask=f"{best_ask:.2f}"
        )

    async def _update_odds_from_gamma(
        self,
        yes_odds: float,
        no_odds: float,
        best_bid: float,
        best_ask: float
    ):
        """
        Update odds from Gamma API market data.

        Args:
            yes_odds: Calculated YES odds (midpoint)
            no_odds: Calculated NO odds (1 - midpoint)
            best_bid: Best bid from Gamma API
            best_ask: Best ask from Gamma API
        """
        # Create snapshot
        snapshot = WebSocketOddsSnapshot(
            market_id=self._current_market_id,
            yes_odds=yes_odds,
            no_odds=no_odds,
            timestamp=datetime.now(timezone.utc),
            best_bid=best_bid,
            best_ask=best_ask
        )

        # Store atomically
        async with self._lock:
            self._current_odds[self._current_market_id] = snapshot

        self._fire_odds_callbacks(snapshot)

        logger.info(
            "📊 Odds updated",
            source="Gamma API",
            market_id=self._current_market_id,
            market_slug=self._current_market_slug,
            yes_odds=f"{yes_odds:.2f}",
            no_odds=f"{no_odds:.2f}",
            best_bid=f"{best_bid:.2f}",
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

    async def _rest_polling_loop(self):
        """
        Poll CLOB last-trade-price every 5 seconds for real-time odds.

        Uses CLOB last-trade-price endpoint which is accurate for CTF/AMM markets.
        Gamma API is severely lagged (30+ minutes behind) for BTC 15-min markets
        because it aggregates from on-chain data rather than CLOB trades.

        Token convention: token_ids[0] = YES, token_ids[1] = NO
        """
        logger.info("CLOB last-trade-price polling loop started (5s interval)")

        while self._running:
            try:
                # Wait for market and token IDs to be set
                if (not self._current_market_id or
                        not self._current_token_ids_decimal or
                        len(self._current_token_ids_decimal) < 2):
                    await asyncio.sleep(5)
                    continue

                token_ids = self._current_token_ids_decimal
                yes_token_id = token_ids[0]  # Polymarket convention: [YES, NO]
                no_token_id = token_ids[1]

                # Fetch last trade prices from CLOB (real-time, accurate)
                clob = self.client._get_clob_client()

                yes_data = await asyncio.to_thread(
                    clob.get_last_trade_price, yes_token_id
                )
                no_data = await asyncio.to_thread(
                    clob.get_last_trade_price, no_token_id
                )

                yes_price = float(yes_data.get("price", 0.5))
                no_price = float(no_data.get("price", 0.5))

                # Validate ranges
                if not (0.001 <= yes_price <= 0.999):
                    yes_price = 0.5
                if not (0.001 <= no_price <= 0.999):
                    no_price = 0.5

                # Normalize if sum < 0.99: last-trade-prices are independent and
                # may not sum exactly to 1.0 (spread between trades is normal).
                # WebSocketOddsSnapshot requires sum >= 0.99.
                total = yes_price + no_price
                if total < 0.99:
                    yes_price = yes_price / total
                    no_price = no_price / total

                # Create snapshot - last trade price IS the odds for CTF tokens
                snapshot = WebSocketOddsSnapshot(
                    market_id=self._current_market_id,
                    yes_odds=yes_price,
                    no_odds=no_price,
                    timestamp=datetime.now(timezone.utc),
                    best_bid=yes_price,
                    best_ask=yes_price
                )

                # Store atomically
                async with self._lock:
                    self._current_odds[self._current_market_id] = snapshot

                self._fire_odds_callbacks(snapshot)

                logger.info(
                    "📊 Odds updated from CLOB last-trade-price",
                    market_id=self._current_market_id,
                    market_slug=self._current_market_slug,
                    yes_odds=f"{yes_price:.3f}",
                    no_odds=f"{no_price:.3f}"
                )

            except Exception as e:
                logger.error(
                    "CLOB last-trade-price polling failed",
                    error=str(e),
                    market_id=self._current_market_id
                )

            await asyncio.sleep(5)

        logger.info("CLOB last-trade-price polling loop stopped")

    async def start(self):
        """
        Start streaming (non-blocking).

        Launches background tasks for WebSocket and REST polling.
        """
        if self._running:
            logger.warning("Streamer already running")
            return

        self._running = True
        self._stream_task = asyncio.create_task(self._stream_loop())
        self._rest_task = asyncio.create_task(self._rest_polling_loop())
        logger.info("Real-time odds streamer started (WebSocket + REST polling)")

    async def stop(self):
        """
        Stop streaming gracefully.

        Closes WebSocket connection and cancels background tasks.
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

        if self._rest_task:
            self._rest_task.cancel()
            try:
                await self._rest_task
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

        Uses multiple detection methods:
        1. Time-based: Current market past its active window (fallback for API lag)
        2. API-based: API returns different market ID (primary)

        Returns:
            True if market changed, False if same
        """
        try:
            # Fallback: Check if current market is no longer active (time-based)
            # Use 30-second tolerance for aggressive transition detection
            if self._current_market_slug and not self._validator.is_market_active(
                self._current_market_slug,
                tolerance_minutes=0.5  # 30 seconds
            ):
                logger.info(
                    "Market transition detected via time-based check (market inactive)",
                    current_market_slug=self._current_market_slug
                )
                return True

            # Primary: Check if API returns different market
            market = self._discover_current_market()

            if market.id != self._current_market_id:
                logger.info(
                    "Market transition detected via API response",
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

        Proactively checks at market end time for instant transitions.

        Args:
            ws: WebSocket connection to close if transition detected
        """
        while self._running:
            # Calculate time until current market ends
            if self._current_market_slug:
                market_start_timestamp = self._validator.parse_market_timestamp(self._current_market_slug)

                if market_start_timestamp:
                    current_time = datetime.now(timezone.utc)
                    market_start_time = datetime.fromtimestamp(market_start_timestamp, timezone.utc)
                    # Detect market duration from slug: 5m=300s, else 15m=900s
                    market_duration_s = 300 if self._current_market_slug and "-5m-" in self._current_market_slug else 900
                    market_end_time = market_start_time + timedelta(seconds=market_duration_s)
                    seconds_until_end = (market_end_time - current_time).total_seconds()

                    if seconds_until_end > 30:
                        # Market end is far away, sleep until 30 seconds before
                        sleep_duration = seconds_until_end - 30
                        logger.info(
                            "Market transition monitor: sleeping until near market end",
                            market_end_time=market_end_time.isoformat(),
                            sleep_seconds=sleep_duration
                        )
                        await asyncio.sleep(sleep_duration)
                    elif seconds_until_end > 0:
                        # Market ending in <30 seconds, check right at end time
                        logger.info(
                            "Market transition monitor: sleeping until market end time",
                            seconds_until_end=seconds_until_end
                        )
                        await asyncio.sleep(seconds_until_end + 2)  # +2 for API propagation
                    else:
                        # Market already ended, check immediately then wait 10s
                        logger.info(
                            "Market transition monitor: past end time, checking immediately"
                        )
                        await asyncio.sleep(10)
                else:
                    # Couldn't parse timestamp, fall back to 60s interval
                    await asyncio.sleep(60)
            else:
                # No current market slug, fall back to 60s interval
                await asyncio.sleep(60)

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
        # Discover current market (with retry if API is lagged after transition)
        old_market_id = self._current_market_id

        # If we transitioned due to time-based detection, the API might be lagged
        # Retry up to 6 times (30 seconds) until we get a NEW market
        max_retries = 6
        retry_delay = 5

        for attempt in range(max_retries):
            market = self._discover_current_market()

            # If this is a new market (or first connection), proceed
            if old_market_id is None or market.id != old_market_id:
                break

            # API still returning old market after time-based transition
            if attempt < max_retries - 1:
                logger.warning(
                    "API still returning old market, retrying...",
                    attempt=attempt + 1,
                    old_market_id=old_market_id,
                    returned_market_id=market.id,
                    retry_in_seconds=retry_delay
                )
                await asyncio.sleep(retry_delay)
            else:
                logger.error(
                    "API still lagged after retries, proceeding with old market",
                    old_market_id=old_market_id,
                    returned_market_id=market.id
                )

        token_ids = market.get_token_ids()

        if not token_ids:
            logger.error("No token IDs found for market", market_id=market.id)
            return

        # Store market state atomically
        # Store BOTH decimal and hex formats:
        # - decimal: for REST API queries (expects decimal token IDs)
        # - hex: for WebSocket message filtering (book messages use hex)
        decimal_token_ids = token_ids  # Already in decimal format
        hex_token_ids = [hex(int(tid)) for tid in token_ids]

        async with self._lock:
            self._current_market_id = market.id
            self._current_market_slug = market.slug
            self._current_token_ids_decimal = decimal_token_ids
            self._current_token_ids_hex = hex_token_ids
            self._current_market_obj = market  # cache for snapshot fallback

        logger.info(
            "Connecting to CLOB WebSocket",
            market_id=market.id,
            market_slug=market.slug,
            token_ids=token_ids
        )
        logger.info(
            "Stored token ID formats",
            decimal_first=decimal_token_ids[0] if decimal_token_ids else None,
            hex_first=hex_token_ids[0] if hex_token_ids else None
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
            if token_id and self._current_token_ids_hex:
                if token_id not in self._current_token_ids_hex:
                    logger.debug(
                        "Ignoring book message for unsubscribed token",
                        token_id=token_id[:16] + "...",
                        subscribed_tokens=[t[:16] + "..." for t in self._current_token_ids_hex]
                    )
                    return

            await self._process_book_message(data)

        elif event_type == 'last_trade_price':
            # Real-time trade price update - process immediately for zero-latency odds
            token_id = data.get('asset_id')  # hex format
            price_str = data.get('price')

            if (token_id and price_str and
                    self._current_token_ids_hex and
                    self._current_token_ids_decimal and
                    self._current_market_id):

                if token_id in self._current_token_ids_hex:
                    idx = self._current_token_ids_hex.index(token_id)
                    price = float(price_str)

                    # Get existing snapshot for the other token's price
                    current = self._current_odds.get(self._current_market_id)

                    if idx == 0:  # YES token traded
                        yes_odds = price
                        no_odds = current.no_odds if current else (1.0 - price)
                    else:  # NO token traded
                        no_odds = price
                        yes_odds = current.yes_odds if current else (1.0 - price)

                    snapshot = WebSocketOddsSnapshot(
                        market_id=self._current_market_id,
                        yes_odds=yes_odds,
                        no_odds=no_odds,
                        timestamp=datetime.now(timezone.utc),
                        best_bid=yes_odds,
                        best_ask=yes_odds
                    )

                    # Store atomically
                    async with self._lock:
                        self._current_odds[self._current_market_id] = snapshot

                    self._fire_odds_callbacks(snapshot)

                    logger.info(
                        "⚡ Odds updated from WS last_trade_price",
                        yes_odds=f"{yes_odds:.3f}",
                        no_odds=f"{no_odds:.3f}",
                        traded_token="YES" if idx == 0 else "NO",
                        price=price_str
                    )

        # Ignore price_change and other message types
