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

logger = structlog.get_logger()


class CryptoPriceStream:
    """Real-time BTC price stream from Polymarket WebSocket."""

    WS_URL = "wss://ws-live-data.polymarket.com"

    def __init__(self, settings: Settings):
        self.settings = settings
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._current_price: Optional[BTCPriceData] = None
        self._connected = False
        self._running = False

    async def start(self):
        """Start WebSocket connection and price stream."""
        self._running = True

        try:
            async with websockets.connect(self.WS_URL) as ws:
                self._ws = ws
                self._connected = True

                # Subscribe to BTC prices from Binance source
                # RTDS subscription format per working implementations
                subscribe_msg = {
                    "action": "subscribe",
                    "subscriptions": [{
                        "topic": "crypto_prices",
                        "type": "update",
                        "filters": json.dumps({"symbol": "btcusdt"})
                    }]
                }
                await ws.send(json.dumps(subscribe_msg))
                logger.info("Subscribed to Polymarket RTDS crypto_prices", symbol="btcusdt")

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

    async def _handle_message(self, message: str):
        """Parse and store price update."""
        try:
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
        except Exception as e:
            logger.error("Failed to parse price message", error=str(e))

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
