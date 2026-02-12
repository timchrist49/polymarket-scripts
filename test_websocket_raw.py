#!/usr/bin/env python3
"""
Raw WebSocket test - prints ALL messages from Polymarket RTDS.
Run this to diagnose why crypto_prices updates aren't arriving.
"""

import asyncio
import json
import websockets
from datetime import datetime

WS_URL = "wss://ws-live-data.polymarket.com"

async def test_websocket():
    """Connect to Polymarket RTDS and print all messages."""
    print(f"[{datetime.now()}] Connecting to {WS_URL}...")

    try:
        # FIX: Add ping_interval=5 to match Polymarket RTDS requirements
        async with websockets.connect(
            WS_URL,
            ping_interval=5,   # Send PING every 5 seconds (RTDS requirement)
            ping_timeout=20    # Wait 20 seconds for PONG response
        ) as ws:
            print(f"[{datetime.now()}] ✓ Connected with ping_interval=5!")

            # Subscribe to crypto_prices_chainlink (oracle source instead of Binance)
            # Using type: "*" to receive all message types
            subscribe_msg = {
                "action": "subscribe",
                "subscriptions": [{
                    "topic": "crypto_prices_chainlink",
                    "type": "*",
                    "filters": json.dumps({"symbol": "btc/usd"})  # Chainlink format: slash-separated
                }]
            }

            await ws.send(json.dumps(subscribe_msg))
            print(f"[{datetime.now()}] Subscription sent:")
            print(json.dumps(subscribe_msg, indent=2))
            print("\n" + "="*80)
            print("Waiting for messages (will print ALL messages received)...")
            print("="*80 + "\n")

            message_count = 0

            # Listen for 60 seconds
            timeout = 60
            start_time = asyncio.get_event_loop().time()

            while asyncio.get_event_loop().time() - start_time < timeout:
                try:
                    message = await asyncio.wait_for(ws.recv(), timeout=10.0)
                    message_count += 1

                    print(f"\n[{datetime.now()}] MESSAGE #{message_count}:")
                    print("-" * 80)

                    try:
                        data = json.loads(message)
                        print(json.dumps(data, indent=2))
                    except json.JSONDecodeError:
                        print(f"RAW (not JSON): {message}")

                    print("-" * 80)

                except asyncio.TimeoutError:
                    elapsed = asyncio.get_event_loop().time() - start_time
                    print(f"[{datetime.now()}] No message for 10s (elapsed: {elapsed:.0f}s)")
                    continue

            print(f"\n{'='*80}")
            print(f"Test complete. Received {message_count} messages in {timeout}s.")
            print(f"{'='*80}")

    except Exception as e:
        print(f"[{datetime.now()}] ✗ Error: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(test_websocket())
