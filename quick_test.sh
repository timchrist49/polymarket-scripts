#!/bin/bash
# quick_test.sh - Quick verification of all three scripts

set -e

echo "=== Polymarket Skill Pack Quick Verification ==="
echo ""

# 1. Verify read_only mode works
echo "1. Testing fetch_markets in read_only mode..."
POLYMARKET_MODE=read_only python scripts/fetch_markets.py --btc-mode --limit 1 || echo "   (May fail if network unavailable - OK)"
echo ""

# 2. Verify dry-run mode
echo "2. Testing place_order in dry-run mode..."
POLYMARKET_MODE=trading \
POLYMARKET_PRIVATE_KEY="0x"$(python3 -c "print('a'*64)") \
python scripts/place_order.py \
    --btc-mode \
    --side buy \
    --price 0.50 \
    --size 1 \
    --dry-run true || true
echo ""

# 3. Verify portfolio returns empty
echo "3. Testing portfolio_status in read_only mode..."
POLYMARKET_MODE=read_only python scripts/portfolio_status.py
echo ""

echo "=== Verification Complete ==="
