#!/usr/bin/env python3
"""
Correct test for Polymarket RTDS based on official documentation.
"""

import asyncio
import json
import sys
from decimal import Decimal
import websockets

WS_URL = "wss://ws-live-data.polymarket.com"


async def test_binance_feed():
    """Test Binance feed (current implementation)."""
    print(f"\n{'='*60}")
    print("TEST 1: Binance Feed (crypto_prices)")
    print(f"{'='*60}")

    try:
        async with websockets.connect(WS_URL, ping_interval=5) as ws:
            # Official format from documentation
            subscribe_msg = {
                "action": "subscribe",
                "subscriptions": [{
                    "topic": "crypto_prices",
                    "type": "update",
                    "filters": "btcusdt"
                }]
            }

            print(f"✓ Connected to RTDS")
            print(f"\nSubscription:")
            print(json.dumps(subscribe_msg, indent=2))

            await ws.send(json.dumps(subscribe_msg))
            print(f"\n✓ Subscription sent")

            # Collect prices
            prices = []
            message_count = 0

            print(f"\nListening for messages (30 seconds)...")
            print("-" * 60)

            timeout_count = 0
            while len(prices) < 3 and timeout_count < 10:
                try:
                    message = await asyncio.wait_for(ws.recv(), timeout=3.0)
                    message_count += 1

                    if not message or not message.strip():
                        continue

                    data = json.loads(message)

                    # Show first 3 messages
                    if message_count <= 3:
                        print(f"\n[Message #{message_count}]")
                        print(json.dumps(data, indent=2)[:500])

                    # Check for errors
                    if data.get("statusCode"):
                        print(f"\n✗ Error: {data.get('body', {}).get('message')}")
                        return None

                    # Parse price updates
                    topic = data.get("topic")
                    msg_type = data.get("type")
                    payload = data.get("payload", {})

                    if topic == "crypto_prices":
                        if msg_type == "subscribe" and payload.get("data"):
                            latest = payload["data"][-1]
                            price = Decimal(str(latest["value"]))
                            prices.append(price)
                            print(f"\n✓ Initial: ${price:,.2f}")

                        elif msg_type == "update":
                            price = Decimal(str(payload["value"]))
                            prices.append(price)
                            print(f"✓ Update: ${price:,.2f}")

                except asyncio.TimeoutError:
                    timeout_count += 1
                    print(".", end="", flush=True)

            print(f"\n\n{'='*60}")
            print(f"Result: {len(prices)} prices received")
            if prices:
                print(f"Average: ${sum(prices)/len(prices):,.2f}")
                return prices
            return None

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


async def test_chainlink_feed():
    """Test Chainlink feed (NEW implementation)."""
    print(f"\n{'='*60}")
    print("TEST 2: Chainlink Feed (crypto_prices_chainlink)")
    print(f"{'='*60}")

    try:
        async with websockets.connect(WS_URL, ping_interval=5) as ws:
            # Official format from documentation
            # CRITICAL: type must be "*" not "update"
            # CRITICAL: filters is JSON embedded as STRING
            subscribe_msg = {
                "action": "subscribe",
                "subscriptions": [{
                    "topic": "crypto_prices_chainlink",
                    "type": "*",  # ← Must be "*" for Chainlink!
                    "filters": "{\"symbol\":\"btc/usd\"}"  # ← JSON as STRING!
                }]
            }

            print(f"✓ Connected to RTDS")
            print(f"\nSubscription:")
            print(json.dumps(subscribe_msg, indent=2))

            await ws.send(json.dumps(subscribe_msg))
            print(f"\n✓ Subscription sent")

            # Collect prices
            prices = []
            message_count = 0

            print(f"\nListening for messages (30 seconds)...")
            print("-" * 60)

            timeout_count = 0
            while len(prices) < 3 and timeout_count < 10:
                try:
                    message = await asyncio.wait_for(ws.recv(), timeout=3.0)
                    message_count += 1

                    if not message or not message.strip():
                        continue

                    data = json.loads(message)

                    # Show first 3 messages
                    if message_count <= 3:
                        print(f"\n[Message #{message_count}]")
                        print(json.dumps(data, indent=2)[:500])

                    # Check for errors
                    if data.get("statusCode"):
                        print(f"\n✗ Error: {data.get('body', {}).get('message')}")
                        return None

                    # Parse price updates
                    topic = data.get("topic")
                    msg_type = data.get("type")
                    payload = data.get("payload", {})

                    if topic == "crypto_prices_chainlink":
                        if msg_type == "subscribe" and payload.get("data"):
                            latest = payload["data"][-1]
                            price = Decimal(str(latest["value"]))
                            prices.append(price)
                            print(f"\n✓ Initial: ${price:,.2f}")

                        elif msg_type == "update":
                            price = Decimal(str(payload["value"]))
                            prices.append(price)
                            print(f"✓ Update: ${price:,.2f}")

                except asyncio.TimeoutError:
                    timeout_count += 1
                    print(".", end="", flush=True)

            print(f"\n\n{'='*60}")
            print(f"Result: {len(prices)} prices received")
            if prices:
                print(f"Average: ${sum(prices)/len(prices):,.2f}")
                return prices
            return None

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


async def main():
    """Run both tests."""
    print("\n" + "="*60)
    print("POLYMARKET RTDS TEST (OFFICIAL FORMAT)")
    print("="*60)
    print("\nUsing exact format from Polymarket documentation:")
    print("- Binance: crypto_prices with comma-separated string filters")
    print("- Chainlink: crypto_prices_chainlink with JSON string filters")

    # Test Binance
    binance_prices = await test_binance_feed()
    await asyncio.sleep(2)

    # Test Chainlink
    chainlink_prices = await test_chainlink_feed()

    # Summary
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)

    binance_ok = binance_prices and len(binance_prices) > 0
    chainlink_ok = chainlink_prices and len(chainlink_prices) > 0

    print(f"\nBinance Feed:   {'✓ WORKS' if binance_ok else '✗ FAILED'}")
    print(f"Chainlink Feed: {'✓ WORKS' if chainlink_ok else '✗ FAILED'}")

    if binance_ok and chainlink_ok:
        b_avg = sum(binance_prices) / len(binance_prices)
        c_avg = sum(chainlink_prices) / len(chainlink_prices)
        diff = abs(b_avg - c_avg)
        pct = (diff / b_avg) * 100

        print(f"\nPrice Comparison:")
        print(f"  Binance:   ${b_avg:,.2f}")
        print(f"  Chainlink: ${c_avg:,.2f}")
        print(f"  Diff:      ${diff:,.2f} ({pct:.4f}%)")
        print(f"\n{'✓' if diff > 0.01 else '⚠'} Feeds are {'distinct' if diff > 0.01 else 'similar'}")

    print("\n" + "="*60)
    print(f"{'✓ CHAINLINK RTDS WORKS!' if chainlink_ok else '✗ TESTS FAILED'}")
    print("="*60)

    return 0 if chainlink_ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
