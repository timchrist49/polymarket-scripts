"""
Polymarket Crypto Price Stream

WebSocket client for real-time BTC price updates from Polymarket.
Uses crypto_prices topic for consistency with market resolution.
"""

import asyncio
import json
from decimal import Decimal
from datetime import datetime
from typing import Optional
import structlog
import websockets

from polymarket.models import BTCPriceData
from polymarket.config import Settings
from polymarket.trading.price_history_buffer import PriceHistoryBuffer

logger = structlog.get_logger()


class CryptoPriceStream:
    """Real-time BTC price stream from Polymarket WebSocket."""

    WS_URL = "wss://ws-live-data.polymarket.com"

    def __init__(
        self,
        settings: Settings,
        buffer_enabled: bool = False,
        buffer_file: str = "data/price_history.json",
        use_chainlink: bool = True  # ← NEW: default to Chainlink
    ):
        self.settings = settings
        self.use_chainlink = use_chainlink  # ← NEW
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._current_price: Optional[BTCPriceData] = None
        self._connected = False
        self._running = False

        # Initialize price history buffer if enabled
        self.price_buffer: Optional[PriceHistoryBuffer] = None
        if buffer_enabled:
            self.price_buffer = PriceHistoryBuffer(
                retention_hours=24,
                save_interval=300,  # 5 minutes
                persistence_file=buffer_file
            )

    async def start(self):
        """Start WebSocket connection and price stream."""
        self._running = True

        # Load historical price data if buffer is enabled
        if self.price_buffer:
            try:
                await self.price_buffer.load_from_disk()
                logger.info("Price history loaded from disk")
            except Exception as e:
                logger.warning(f"Could not load price history: {e}")

        try:
            async with websockets.connect(self.WS_URL, ping_interval=5) as ws:  # ← Add ping_interval
                self._ws = ws
                self._connected = True

                # Subscribe to appropriate feed
                await self._subscribe_to_feed(ws)  # ← NEW method

                # Listen for price updates
                while self._running:
                    try:
                        message = await asyncio.wait_for(ws.recv(), timeout=30.0)
                        await self._handle_message(message)
                    except asyncio.TimeoutError:
                        logger.debug("WebSocket recv timeout, continuing...")
                        continue
                    except websockets.exceptions.ConnectionClosed:
                        logger.warning("WebSocket connection closed")
                        self._connected = False
                        break

        except Exception as e:
            logger.error("WebSocket connection error", error=str(e))
            self._connected = False
        finally:
            self._ws = None
            self._connected = False

    async def _subscribe_to_feed(self, ws):  # ← NEW method
        """Subscribe to Chainlink or Binance feed based on configuration."""
        if self.use_chainlink:
            # Chainlink format (verified working)
            subscribe_msg = {
                "action": "subscribe",
                "subscriptions": [{
                    "topic": "crypto_prices_chainlink",
                    "type": "*",  # Must be "*" for Chainlink
                    "filters": '{"symbol":"btc/usd"}'  # JSON as string
                }]
            }
            logger.info(
                "Subscribed to Polymarket RTDS crypto_prices_chainlink",
                source="chainlink",
                symbol="btc/usd"
            )
        else:
            # Binance format (legacy fallback)
            subscribe_msg = {
                "action": "subscribe",
                "subscriptions": [{
                    "topic": "crypto_prices",
                    "type": "update",
                    "filters": "btcusdt"
                }]
            }
            logger.info(
                "Subscribed to Polymarket RTDS crypto_prices",
                source="binance",
                symbol="btcusdt"
            )

        await ws.send(json.dumps(subscribe_msg))

    async def _handle_message(self, message: str):
        """Parse and store price update."""
        try:
            # Skip empty messages (WebSocket pings/heartbeats)
            if not message or not message.strip():
                return

            data = json.loads(message)
            topic = data.get("topic")
            msg_type = data.get("type")
            payload = data.get("payload", {})

            if topic == "crypto_prices":
                # Handle initial subscription data dump
                if msg_type == "subscribe" and payload.get("symbol") == "btcusdt":
                    # Initial dump contains 'data' array with historical prices
                    # Use the most recent price
                    price_data = payload.get("data", [])
                    if price_data:
                        latest = price_data[-1]
                        self._current_price = BTCPriceData(
                            price=Decimal(str(latest["value"])),
                            timestamp=datetime.fromtimestamp(latest["timestamp"] / 1000),
                            source="polymarket",
                            volume_24h=Decimal("0")  # Not provided in crypto_prices
                        )
                        logger.debug(
                            "BTC price (initial)",
                            price=f"${self._current_price.price:,.2f}",
                            source="polymarket"
                        )

                        # Append to buffer if enabled
                        await self._append_to_buffer(
                            latest["timestamp"] // 1000,  # Convert ms to seconds
                            Decimal(str(latest["value"]))
                        )

                # Handle real-time price updates
                elif msg_type == "update" and payload.get("symbol") == "btcusdt":
                    self._current_price = BTCPriceData(
                        price=Decimal(str(payload["value"])),
                        timestamp=datetime.fromtimestamp(payload["timestamp"] / 1000),
                        source="polymarket",
                        volume_24h=Decimal("0")  # Not provided in crypto_prices
                    )
                    logger.debug(
                        "BTC price update",
                        price=f"${self._current_price.price:,.2f}",
                        source="polymarket"
                    )

                    # Append to buffer if enabled
                    await self._append_to_buffer(
                        payload["timestamp"] // 1000,  # Convert ms to seconds
                        Decimal(str(payload["value"]))
                    )

        except Exception as e:
            logger.error("Failed to parse price message", error=str(e))

    async def _append_to_buffer(self, timestamp: int, price: Decimal):
        """Append price to buffer if enabled. Errors don't crash WebSocket."""
        if self.price_buffer:
            try:
                await self.price_buffer.append(timestamp, price, source="polymarket")
            except Exception as e:
                logger.error(f"Failed to append to price buffer: {e}")
                # Don't crash WebSocket on buffer failure

    async def get_current_price(self) -> Optional[BTCPriceData]:
        """Get most recent BTC price."""
        return self._current_price

    def is_connected(self) -> bool:
        """Check if WebSocket is connected."""
        return self._connected

    async def stop(self):
        """Stop WebSocket connection."""
        self._running = False
        if self._ws:
            await self._ws.close()

        # Save buffer to disk before shutdown
        if self.price_buffer:
            try:
                await self.price_buffer.save_to_disk()
                logger.info("Price history saved to disk")
            except Exception as e:
                logger.error(f"Failed to save price history: {e}")
