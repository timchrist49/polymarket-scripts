#!/usr/bin/env python3
"""
Debug test for Polymarket RTDS connection.
Shows ALL messages received from the WebSocket.
"""

import asyncio
import json
import websockets

WS_URL = "wss://ws-live-data.polymarket.com"


async def debug_connection(topic: str, filters: str, duration: int = 15):
    """Connect and print all messages for debugging."""
    print(f"\n{'='*60}")
    print(f"Testing: {topic} / {filters}")
    print(f"{'='*60}")

    try:
        async with websockets.connect(WS_URL) as ws:
            print(f"✓ WebSocket connected to {WS_URL}")

            # Subscribe
            subscribe_msg = {
                "action": "subscribe",
                "subscriptions": [{
                    "topic": topic,
                    "type": "update",
                    "filters": filters
                }]
            }

            print(f"\nSending subscription:")
            print(json.dumps(subscribe_msg, indent=2))
            await ws.send(json.dumps(subscribe_msg))
            print(f"✓ Subscription sent")

            # Listen for messages
            print(f"\nListening for {duration} seconds...")
            print("-" * 60)

            start_time = asyncio.get_event_loop().time()
            message_count = 0

            while (asyncio.get_event_loop().time() - start_time) < duration:
                try:
                    message = await asyncio.wait_for(ws.recv(), timeout=2.0)
                    message_count += 1

                    print(f"\n[Message #{message_count}]")

                    if not message or not message.strip():
                        print("  (empty message - heartbeat?)")
                        continue

                    try:
                        data = json.loads(message)
                        print(json.dumps(data, indent=2))
                    except json.JSONDecodeError:
                        print(f"  Raw: {message[:200]}...")

                except asyncio.TimeoutError:
                    print(".", end="", flush=True)
                    continue

            print(f"\n\n{'='*60}")
            print(f"Total messages received: {message_count}")
            print(f"{'='*60}")

    except Exception as e:
        print(f"\n✗ Connection error: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """Run debug tests."""
    print("\n" + "="*60)
    print("POLYMARKET RTDS DEBUG TEST")
    print("="*60)
    print("\nThis test shows ALL messages received from RTDS")
    print("to help diagnose connection issues.\n")

    # Test 1: Current Binance feed
    await debug_connection("crypto_prices", "btcusdt", duration=15)

    await asyncio.sleep(2)

    # Test 2: Chainlink feed
    await debug_connection("crypto_prices_chainlink", "btc/usd", duration=15)

    print("\n" + "="*60)
    print("DEBUG TEST COMPLETE")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())
