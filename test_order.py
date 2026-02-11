#!/usr/bin/env python3
"""
Test script to force an order placement (bypasses AI decision).
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from polymarket.client import PolymarketClient
from polymarket.config import Settings
from polymarket.models import OrderRequest

async def test_order_placement():
    """Test order placement with minimal position size."""

    settings = Settings()
    client = PolymarketClient()

    print("=" * 60)
    print("🧪 TESTING ORDER PLACEMENT")
    print("=" * 60)
    print()

    # Get current BTC market
    print("📊 Finding BTC market...")
    market = client.discover_btc_15min_market()
    token_ids = market.get_token_ids()

    print(f"Market: {market.question}")
    print(f"Market ID: {market.id}")
    print(f"YES Token: {token_ids[0]}")
    print(f"NO Token: {token_ids[1]}")
    print(f"Current Prices: YES=${market.best_bid:.2f} / NO=${market.best_ask:.2f}")
    print()

    # Create a test order ($2 to ensure > $1 minimum after price adjustment)
    print("💰 Creating TEST ORDER:")
    print(f"   Side: BUY NO (BTC DOWN)")
    print(f"   Size: $2.00")
    print(f"   Type: MARKET (immediate execution)")
    print(f"   Mode: {'🔴 LIVE TRADING' if settings.mode == 'trading' else '🟢 DRY RUN'}")
    print()

    if settings.mode != "trading":
        print("⚠️  DRY_RUN=true - No real order will be placed")
        print("   Set DRY_RUN=false in .env to test real orders")
        print()

    # Create order request
    order_request = OrderRequest(
        token_id=token_ids[1],  # NO token (BTC DOWN)
        side="BUY",
        price=0.99,  # Aggressive price for market order
        size=2.00,   # $2.00 to exceed $1 minimum
        order_type="market"
    )

    try:
        print("🚀 Attempting to place order...")
        result = client.create_order(order_request, dry_run=(settings.mode != "trading"))

        print()
        print("=" * 60)
        print("✅ ORDER PLACEMENT SUCCESSFUL!")
        print("=" * 60)
        print(f"Order ID: {result.order_id}")
        print(f"Status: {result.status}")
        print(f"Accepted: {result.accepted}")
        print()

        if settings.mode != "trading":
            print("ℹ️  This was a DRY RUN - no real money was spent")
        else:
            print("🎉 REAL ORDER PLACED - Check your Polymarket account!")

        return True

    except Exception as e:
        print()
        print("=" * 60)
        print("❌ ORDER PLACEMENT FAILED")
        print("=" * 60)
        print(f"Error: {e}")
        print()
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_order_placement())
    sys.exit(0 if success else 1)
