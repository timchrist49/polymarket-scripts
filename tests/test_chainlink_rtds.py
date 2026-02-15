#!/usr/bin/env python3
"""
Test script to verify Polymarket RTDS Chainlink subscription.

This script tests:
1. Connection to Polymarket RTDS WebSocket
2. Subscription to crypto_prices_chainlink topic
3. Receipt of BTC/USD price updates from Chainlink
4. Comparison with Binance prices to verify different sources
"""

import asyncio
import json
import sys
from decimal import Decimal
from datetime import datetime
import websockets

WS_URL = "wss://ws-live-data.polymarket.com"


async def test_binance_feed():
    """Test current Binance feed subscription."""
    print("\n" + "="*60)
    print("TEST 1: Binance Feed (Current Implementation)")
    print("="*60)

    try:
        async with websockets.connect(WS_URL) as ws:
            # Subscribe to Binance feed
            subscribe_msg = {
                "action": "subscribe",
                "subscriptions": [{
                    "topic": "crypto_prices",
                    "type": "update",
                    "filters": "btcusdt"
                }]
            }
            await ws.send(json.dumps(subscribe_msg))
            print(f"✓ Subscribed to crypto_prices (Binance)")
            print(f"  Topic: crypto_prices")
            print(f"  Filters: btcusdt")

            # Collect a few price updates
            prices = []
            timeout_count = 0
            max_timeouts = 3

            while len(prices) < 3 and timeout_count < max_timeouts:
                try:
                    message = await asyncio.wait_for(ws.recv(), timeout=10.0)

                    if not message or not message.strip():
                        continue

                    data = json.loads(message)
                    topic = data.get("topic")
                    msg_type = data.get("type")
                    payload = data.get("payload", {})

                    if topic == "crypto_prices":
                        if msg_type == "subscribe":
                            # Initial data dump
                            price_data = payload.get("data", [])
                            if price_data:
                                latest = price_data[-1]
                                prices.append({
                                    "price": Decimal(str(latest["value"])),
                                    "timestamp": latest["timestamp"],
                                    "type": "initial"
                                })
                                print(f"  Initial price: ${Decimal(str(latest['value'])):,.2f}")

                        elif msg_type == "update":
                            # Real-time update
                            prices.append({
                                "price": Decimal(str(payload["value"])),
                                "timestamp": payload["timestamp"],
                                "type": "update"
                            })
                            print(f"  Update: ${Decimal(str(payload['value'])):,.2f}")

                except asyncio.TimeoutError:
                    timeout_count += 1
                    print(f"  Waiting for updates... ({timeout_count}/{max_timeouts})")

            if prices:
                print(f"\n✓ Received {len(prices)} price updates from Binance feed")
                return prices
            else:
                print("\n✗ No price updates received from Binance feed")
                return None

    except Exception as e:
        print(f"\n✗ Binance feed test failed: {e}")
        return None


async def test_chainlink_feed():
    """Test Chainlink feed subscription."""
    print("\n" + "="*60)
    print("TEST 2: Chainlink Feed (New Implementation)")
    print("="*60)

    try:
        async with websockets.connect(WS_URL) as ws:
            # Subscribe to Chainlink feed
            subscribe_msg = {
                "action": "subscribe",
                "subscriptions": [{
                    "topic": "crypto_prices_chainlink",
                    "type": "update",
                    "filters": "btc/usd"
                }]
            }
            await ws.send(json.dumps(subscribe_msg))
            print(f"✓ Subscribed to crypto_prices_chainlink")
            print(f"  Topic: crypto_prices_chainlink")
            print(f"  Filters: btc/usd")

            # Collect a few price updates
            prices = []
            timeout_count = 0
            max_timeouts = 3

            while len(prices) < 3 and timeout_count < max_timeouts:
                try:
                    message = await asyncio.wait_for(ws.recv(), timeout=10.0)

                    if not message or not message.strip():
                        continue

                    data = json.loads(message)
                    topic = data.get("topic")
                    msg_type = data.get("type")
                    payload = data.get("payload", {})

                    if topic == "crypto_prices_chainlink":
                        if msg_type == "subscribe":
                            # Initial data dump
                            price_data = payload.get("data", [])
                            if price_data:
                                latest = price_data[-1]
                                prices.append({
                                    "price": Decimal(str(latest["value"])),
                                    "timestamp": latest["timestamp"],
                                    "type": "initial"
                                })
                                print(f"  Initial price: ${Decimal(str(latest['value'])):,.2f}")

                        elif msg_type == "update":
                            # Real-time update
                            prices.append({
                                "price": Decimal(str(payload["value"])),
                                "timestamp": payload["timestamp"],
                                "type": "update"
                            })
                            print(f"  Update: ${Decimal(str(payload['value'])):,.2f}")

                    # Check for error messages
                    elif data.get("error"):
                        print(f"\n✗ Received error from server: {data}")
                        return None

                except asyncio.TimeoutError:
                    timeout_count += 1
                    print(f"  Waiting for updates... ({timeout_count}/{max_timeouts})")

            if prices:
                print(f"\n✓ Received {len(prices)} price updates from Chainlink feed")
                return prices
            else:
                print("\n✗ No price updates received from Chainlink feed")
                return None

    except Exception as e:
        print(f"\n✗ Chainlink feed test failed: {e}")
        return None


async def compare_feeds(binance_prices, chainlink_prices):
    """Compare prices from both feeds."""
    print("\n" + "="*60)
    print("TEST 3: Feed Comparison")
    print("="*60)

    if not binance_prices or not chainlink_prices:
        print("✗ Cannot compare - one or both feeds failed")
        return False

    binance_avg = sum(p["price"] for p in binance_prices) / len(binance_prices)
    chainlink_avg = sum(p["price"] for p in chainlink_prices) / len(chainlink_prices)

    difference = abs(binance_avg - chainlink_avg)
    percent_diff = (difference / binance_avg) * 100

    print(f"Binance Average:  ${binance_avg:,.2f}")
    print(f"Chainlink Average: ${chainlink_avg:,.2f}")
    print(f"Difference:        ${difference:,.2f} ({percent_diff:.4f}%)")

    if difference > 0.01:
        print(f"\n✓ Feeds are distinct (expected behavior)")
        return True
    else:
        print(f"\n⚠ Feeds are identical (unexpected - may be same source)")
        return False


async def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("POLYMARKET RTDS CHAINLINK SUBSCRIPTION TEST")
    print("="*60)
    print("\nThis test verifies:")
    print("1. Current Binance feed works (crypto_prices)")
    print("2. New Chainlink feed works (crypto_prices_chainlink)")
    print("3. Both feeds provide distinct prices")
    print()

    # Test Binance feed (current implementation)
    binance_prices = await test_binance_feed()

    # Wait a bit between tests
    await asyncio.sleep(2)

    # Test Chainlink feed (new implementation)
    chainlink_prices = await test_chainlink_feed()

    # Compare results
    comparison_success = await compare_feeds(binance_prices, chainlink_prices)

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    binance_ok = binance_prices is not None
    chainlink_ok = chainlink_prices is not None

    print(f"Binance Feed:   {'✓ PASS' if binance_ok else '✗ FAIL'}")
    print(f"Chainlink Feed: {'✓ PASS' if chainlink_ok else '✗ FAIL'}")
    print(f"Comparison:     {'✓ PASS' if comparison_success else '✗ FAIL'}")

    all_pass = binance_ok and chainlink_ok and comparison_success

    print(f"\nOverall: {'✓ ALL TESTS PASSED' if all_pass else '✗ SOME TESTS FAILED'}")
    print("="*60)

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
