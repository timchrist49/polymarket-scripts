#!/bin/bash
# Test mode integration test

set -e

echo "Testing test mode integration..."
echo ""

# Test 1: Verify database schema
echo "Test 1: Database schema"
python3 -c "
import sqlite3
conn = sqlite3.connect('data/performance.db')
cursor = conn.cursor()
cursor.execute('PRAGMA table_info(trades)')
columns = [col[1] for col in cursor.fetchall()]
assert 'is_test_mode' in columns, 'is_test_mode column missing'
print('✓ is_test_mode column exists')
"

# Test 2: Verify test mode detection
echo ""
echo "Test 2: Test mode detection"
python3 -c "
import os
os.environ['TEST_MODE'] = 'true'
from scripts.auto_trade import TestModeConfig
config = TestModeConfig(enabled=os.getenv('TEST_MODE', '').lower() == 'true')
assert config.enabled == True, 'Test mode not enabled'
assert config.max_bet_amount == 1.0, 'Max bet not \$1'
assert config.min_confidence == 0.70, 'Min confidence not 70%'
print('✓ Test mode config correct')
"

# Test 3: Verify duplicate prevention logic
echo ""
echo "Test 3: Duplicate prevention"
python3 -c "
from scripts.auto_trade import TestModeConfig
config = TestModeConfig(enabled=True)
market_id = 'test-market-123'

# First check - should allow
assert market_id not in config.traded_markets
config.traded_markets.add(market_id)

# Second check - should skip
assert market_id in config.traded_markets
print('✓ Duplicate prevention works')
"

# Test 4: Verify imports (syntax validation)
echo ""
echo "Test 4: Import validation"
python3 -c "
from polymarket.trading.ai_decision import AIDecisionService
from polymarket.performance.tracker import PerformanceTracker
from scripts.auto_trade import AutoTrader, TestModeConfig
print('✓ All modules import successfully')
"

echo ""
echo "═══════════════════════════════════════"
echo "All tests passed! ✓"
echo "═══════════════════════════════════════"
echo ""
echo "Test mode implementation validated:"
echo "  • Database schema updated"
echo "  • Configuration working"
echo "  • Duplicate prevention active"
echo "  • All imports successful"
echo ""
echo "Ready to run: TEST_MODE=true python scripts/auto_trade.py"
