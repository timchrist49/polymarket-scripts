#!/usr/bin/env python3
"""
Fixed test for Polymarket RTDS with correct filter format.
"""

import asyncio
import json
import sys
from decimal import Decimal
import websockets

WS_URL = "wss://ws-live-data.polymarket.com"


async def test_feed(topic: str, symbol: str, label: str):
    """Test a specific feed with correct JSON filter format."""
    print(f"\n{'='*60}")
    print(f"{label}")
    print(f"{'='*60}")

    try:
        async with websockets.connect(WS_URL) as ws:
            # Subscribe with JSON filter format
            subscribe_msg = {
                "action": "subscribe",
                "subscriptions": [{
                    "topic": topic,
                    "type": "update",
                    "filters": {"symbol": symbol}  # JSON object format
                }]
            }

            print(f"✓ Connected to RTDS")
            print(f"  Topic: {topic}")
            print(f"  Symbol: {symbol}")
            print(f"\nSending subscription:")
            print(json.dumps(subscribe_msg, indent=2))

            await ws.send(json.dumps(subscribe_msg))
            print(f"\n✓ Subscription sent, waiting for messages...")

            # Collect price updates
            prices = []
            message_count = 0
            max_messages = 10
            max_timeouts = 5
            timeout_count = 0

            while len(prices) < 3 and timeout_count < max_timeouts and message_count < max_messages:
                try:
                    message = await asyncio.wait_for(ws.recv(), timeout=5.0)
                    message_count += 1

                    if not message or not message.strip():
                        continue

                    data = json.loads(message)

                    # Check for errors
                    if data.get("statusCode") == 400:
                        print(f"\n✗ Server error:")
                        print(json.dumps(data, indent=2))
                        return None

                    # Show first few messages for debugging
                    if message_count <= 3:
                        print(f"\n[Message #{message_count}]")
                        print(json.dumps(data, indent=2))

                    msg_topic = data.get("topic")
                    msg_type = data.get("type")
                    payload = data.get("payload", {})

                    if msg_topic == topic:
                        if msg_type == "subscribe":
                            # Initial data dump
                            price_data = payload.get("data", [])
                            if price_data:
                                latest = price_data[-1]
                                price = Decimal(str(latest["value"]))
                                prices.append(price)
                                print(f"\n✓ Initial price: ${price:,.2f}")

                        elif msg_type == "update":
                            # Real-time update
                            price = Decimal(str(payload["value"]))
                            prices.append(price)
                            print(f"✓ Update: ${price:,.2f}")

                except asyncio.TimeoutError:
                    timeout_count += 1
                    print(".", end="", flush=True)
                    continue

            print(f"\n\nReceived {len(prices)} price updates")

            if prices:
                avg = sum(prices) / len(prices)
                print(f"Average price: ${avg:,.2f}")
                return prices
            else:
                print("✗ No price updates received")
                return None

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


async def main():
    """Run tests for both feeds."""
    print("\n" + "="*60)
    print("POLYMARKET RTDS TEST (FIXED FORMAT)")
    print("="*60)

    # Test Binance feed
    binance_prices = await test_feed(
        "crypto_prices",
        "btcusdt",
        "TEST 1: Binance Feed (crypto_prices)"
    )

    await asyncio.sleep(2)

    # Test Chainlink feed
    chainlink_prices = await test_feed(
        "crypto_prices_chainlink",
        "btc/usd",
        "TEST 2: Chainlink Feed (crypto_prices_chainlink)"
    )

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    binance_ok = binance_prices is not None and len(binance_prices) > 0
    chainlink_ok = chainlink_prices is not None and len(chainlink_prices) > 0

    print(f"Binance Feed:   {'✓ PASS' if binance_ok else '✗ FAIL'}")
    print(f"Chainlink Feed: {'✓ PASS' if chainlink_ok else '✗ FAIL'}")

    if binance_ok and chainlink_ok:
        binance_avg = sum(binance_prices) / len(binance_prices)
        chainlink_avg = sum(chainlink_prices) / len(chainlink_prices)
        diff = abs(binance_avg - chainlink_avg)
        pct = (diff / binance_avg) * 100

        print(f"\nPrice Comparison:")
        print(f"  Binance:  ${binance_avg:,.2f}")
        print(f"  Chainlink: ${chainlink_avg:,.2f}")
        print(f"  Difference: ${diff:,.2f} ({pct:.4f}%)")

    print(f"\n{'✓ CHAINLINK RTDS WORKS!' if chainlink_ok else '✗ CHAINLINK RTDS FAILED'}")
    print("="*60)

    return 0 if chainlink_ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
