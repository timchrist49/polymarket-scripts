# Manual Test Cases for Lagging Indicator Fixes

**Purpose:** Verify that the bot correctly handles contradictory signals between lagging indicators (momentum) and leading indicators (market volatility, external events).

**Background:** The bot previously over-relied on lagging momentum indicators, causing it to miss market reversals signaled by leading indicators. These test cases verify the fixes implemented in the February 2026 refactor.

---

## Test Case 1: Signal Contradiction Detection

**Setup:**
1. Start the bot in a test environment
2. Wait for a market where momentum and market signals contradict
3. Monitor logs for signal contradiction detection

**Expected Behavior:**
- Bot detects when momentum indicator conflicts with market volatility or external signals
- Logs show: `"Signal contradiction detected: momentum=X vs market=Y"`
- Bot applies reduced weight to momentum signal
- Decision is dominated by leading indicators (market volatility, external events)

**How to Verify:**
```bash
# Check for contradiction detection in logs
grep -i "signal contradiction" polymarket_bot.log

# Verify momentum weight reduction was applied
grep -i "momentum.*reduced" polymarket_bot.log

# Confirm leading indicators dominated decision
grep -A 5 "signal contradiction" polymarket_bot.log | grep -i "market\|external"
```

**Success Criteria:**
- ✅ Contradiction is detected and logged
- ✅ Momentum weight is reduced in decision
- ✅ Leading indicators (market/external) dominate final decision

---

## Test Case 2: BTC Momentum vs Market Signal

**Setup:**
1. Identify a BTC-related Polymarket event
2. Wait for scenario where:
   - BTC momentum shows strong upward trend (lagging)
   - Market volatility or external news suggests downward risk (leading)
3. Observe bot's trading decision

**Expected Behavior:**
- Bot recognizes BTC momentum as lagging indicator
- Bot prioritizes market volatility/external signals over momentum
- Bot either stays out of trade or trades based on leading indicators
- Logs show reduced momentum influence

**How to Verify:**
```bash
# Check BTC momentum detection
grep -i "btc.*momentum" polymarket_bot.log

# Verify market signal was prioritized
grep -i "market.*signal.*priority" polymarket_bot.log

# Confirm decision reasoning
grep -A 10 "Trading decision" polymarket_bot.log | grep -i "momentum\|market"
```

**Success Criteria:**
- ✅ BTC momentum detected but not blindly followed
- ✅ Market signals prioritized over momentum
- ✅ Decision rationale clearly logged

---

## Test Case 3: Reduced Momentum Weight Effect

**Setup:**
1. Run bot for 24 hours in production
2. Collect all trading decisions where momentum was present
3. Analyze weight applied to momentum vs other signals

**Expected Behavior:**
- Momentum weight reduced from previous 70-80% to ~30-40%
- Market volatility and external events receive higher weights
- Bot makes more balanced decisions across signal types

**How to Verify:**
```bash
# Extract all momentum weight values
grep -i "momentum.*weight" polymarket_bot.log | awk '{print $NF}'

# Calculate average momentum weight (should be ~30-40%)
grep -i "momentum.*weight" polymarket_bot.log | awk '{sum+=$NF; count++} END {print sum/count}'

# Compare with market signal weights
grep -i "market.*weight" polymarket_bot.log | awk '{sum+=$NF; count++} END {print sum/count}'
```

**Success Criteria:**
- ✅ Average momentum weight ≤ 40%
- ✅ Market/external signal weights ≥ 60% combined
- ✅ No single signal type dominates all decisions

---

## Test Case 4: Graceful Degradation

**Setup:**
1. Test bot with missing/incomplete signal data:
   - Scenario A: BTC API unavailable (no momentum data)
   - Scenario B: External news API down
   - Scenario C: Market volatility data stale
2. Observe bot's behavior in each scenario

**Expected Behavior:**
- Bot continues operating with available signals
- Logs show which signals are missing
- Bot adjusts confidence scores based on available data
- No crashes or unhandled exceptions

**How to Verify:**
```bash
# Check for missing signal handling
grep -i "missing.*signal\|signal.*unavailable" polymarket_bot.log

# Verify adjusted confidence scores
grep -i "confidence.*adjusted" polymarket_bot.log

# Confirm no exceptions were raised
grep -i "exception\|error\|traceback" polymarket_bot.log | wc -l  # Should be 0
```

**Success Criteria:**
- ✅ Bot handles missing signals gracefully
- ✅ Confidence scores adjusted appropriately
- ✅ No crashes or exceptions
- ✅ Clear logging of degraded state

---

## Test Case 5: End-to-End Contradiction Handling

**Setup:**
1. Run bot for full trading cycle (market discovery → analysis → decision → execution)
2. Target a market with known signal contradictions:
   - Strong historical momentum (UP)
   - Recent negative news event (DOWN)
   - High market volatility (UNCERTAIN)
3. Observe complete decision flow

**Expected Behavior:**
- Bot detects multiple contradictory signals
- Each signal is weighted appropriately:
  - Momentum: ~30-40% weight
  - External event: ~40-50% weight
  - Market volatility: ~20-30% weight
- Final decision reflects balanced analysis
- Confidence score reflects uncertainty from contradictions

**How to Verify:**
```bash
# Full decision chain for a specific market
grep -A 50 "Market: <market_id>" polymarket_bot.log | less

# Extract signal weights for the decision
grep -A 50 "Market: <market_id>" polymarket_bot.log | grep -i "weight"

# Verify confidence score reflects uncertainty
grep -A 50 "Market: <market_id>" polymarket_bot.log | grep -i "confidence"

# Check final decision reasoning
grep -A 50 "Market: <market_id>" polymarket_bot.log | grep -i "decision\|reasoning"
```

**Success Criteria:**
- ✅ All signals detected and analyzed
- ✅ Weights applied correctly to each signal type
- ✅ Confidence score reflects contradiction uncertainty
- ✅ Final decision is well-reasoned and logged
- ✅ Trade executed successfully (or skipped with reason)

---

## Testing Checklist

Before declaring fixes verified in production:

- [ ] Run all 5 test cases
- [ ] Document results for each test case
- [ ] Identify any unexpected behaviors
- [ ] Verify no regressions in existing functionality
- [ ] Confirm log output is clear and actionable
- [ ] Test graceful degradation scenarios
- [ ] Measure performance impact of changes
- [ ] Review confidence scores for reasonableness

---

## Notes for Testers

1. **Log Levels:** Ensure bot is running with `INFO` or `DEBUG` level for sufficient detail
2. **Market Selection:** Choose markets with active volatility for meaningful tests
3. **Timing:** Allow sufficient time for signal collection before trading decisions
4. **Baseline:** Compare results against pre-fix behavior (if available)
5. **Documentation:** Record any edge cases or unexpected behaviors for future improvements

---

**Last Updated:** 2026-02-11
**Related Documentation:**
- `/root/polymarket-scripts/docs/plans/2026-02-10-autonomous-trading-bot-design.md`
- `/root/polymarket-scripts/polymarket/signal_processor.py`
- `/root/polymarket-scripts/polymarket/decision_engine.py`
