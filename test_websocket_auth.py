#!/usr/bin/env python3
"""
Test Polymarket RTDS with gamma_auth for both Binance and Chainlink sources.
"""

import asyncio
import json
import websockets
from datetime import datetime

WS_URL = "wss://ws-live-data.polymarket.com"
WALLET_ADDRESS = "0x15B6AF86B79278FE496585B802bD033aC2d2b525"

async def test_source(source_name, topic, filters):
    """Test a specific crypto price source with authentication."""
    print(f"\n{'='*80}")
    print(f"TESTING: {source_name}")
    print(f"{'='*80}\n")

    try:
        async with websockets.connect(
            WS_URL,
            ping_interval=5,
            ping_timeout=20
        ) as ws:
            print(f"[{datetime.now()}] ✓ Connected!")

            # Subscribe with gamma_auth
            subscribe_msg = {
                "action": "subscribe",
                "subscriptions": [{
                    "topic": topic,
                    "type": "*",
                    "filters": filters,
                    "gamma_auth": {
                        "address": WALLET_ADDRESS
                    }
                }]
            }

            await ws.send(json.dumps(subscribe_msg))
            print(f"[{datetime.now()}] Subscription sent with gamma_auth:")
            print(json.dumps(subscribe_msg, indent=2))
            print(f"\n{'='*80}")
            print("Waiting for messages (60 seconds)...")
            print(f"{'='*80}\n")

            message_count = 0
            update_count = 0
            start_time = asyncio.get_event_loop().time()

            while asyncio.get_event_loop().time() - start_time < 60:
                try:
                    message = await asyncio.wait_for(ws.recv(), timeout=10.0)
                    message_count += 1

                    # Parse message
                    try:
                        if not message or not message.strip():
                            print(f"[{datetime.now()}] Empty message (ping/pong)")
                            continue

                        data = json.loads(message)
                        msg_type = data.get("type", "unknown")
                        topic = data.get("topic", "unknown")

                        if msg_type == "subscribe":
                            print(f"[{datetime.now()}] SUBSCRIBE response received")
                            print(f"  - Topic: {topic}")
                            payload = data.get("payload", {})
                            if "data" in payload:
                                print(f"  - Historical data points: {len(payload['data'])}")
                        elif msg_type == "update":
                            update_count += 1
                            print(f"[{datetime.now()}] ✅ UPDATE #{update_count} received!")
                            payload = data.get("payload", {})
                            print(f"  - Symbol: {payload.get('symbol')}")
                            print(f"  - Price: ${payload.get('value', 0):,.2f}")
                            print(f"  - Timestamp: {payload.get('timestamp')}")
                        else:
                            print(f"[{datetime.now()}] Message type: {msg_type}")

                    except json.JSONDecodeError:
                        print(f"[{datetime.now()}] Non-JSON message: {message[:100]}")

                except asyncio.TimeoutError:
                    elapsed = asyncio.get_event_loop().time() - start_time
                    print(f"[{datetime.now()}] No message for 10s (elapsed: {elapsed:.0f}s)")
                    continue

            print(f"\n{'='*80}")
            print(f"RESULTS: {source_name}")
            print(f"{'='*80}")
            print(f"Total messages: {message_count}")
            print(f"UPDATE messages: {update_count}")
            if update_count > 0:
                print("✅ SUCCESS - Receiving continuous updates!")
            else:
                print("❌ FAILED - No update messages received")
            print(f"{'='*80}\n")

    except Exception as e:
        print(f"[{datetime.now()}] ✗ Error: {e}")
        raise

async def main():
    """Test both Binance and Chainlink sources."""

    # Test 1: Binance source
    await test_source(
        source_name="BINANCE (crypto_prices)",
        topic="crypto_prices",
        filters=json.dumps({"symbol": "BTCUSDT"})
    )

    # Wait a bit between tests
    await asyncio.sleep(2)

    # Test 2: Chainlink source
    await test_source(
        source_name="CHAINLINK (crypto_prices_chainlink)",
        topic="crypto_prices_chainlink",
        filters=json.dumps({"symbol": "btc/usd"})
    )

if __name__ == "__main__":
    asyncio.run(main())
