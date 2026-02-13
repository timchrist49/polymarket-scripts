# Remove Conservative Trading Blocks Design

**Date**: 2026-02-13
**Status**: Approved
**Goal**: Remove outdated signal-based conservative rules to unlock arbitrage system's full trading potential

---

## Problem Statement

The trading bot has 5 conservative blockers that were designed for the old sentiment-based trading system:

1. **AVOID_HOURS** (0-6 AM UTC) - Blocks ALL trades during 25% of each day
2. **ENABLE_YES_TRADES = False** - Blocks ALL YES trades (cuts opportunities in half)
3. **OPTIMAL_HOURS penalty** - Reduces position size 30% outside 11AM-1PM UTC
4. **Confidence threshold (75%)** - Rejects decisions below 75% confidence
5. **Odds multiplier** - Penalizes/rejects low-odds bets (<0.25)

The new arbitrage system makes decisions based on **mathematical edges** (5-15% mispricing detected through price momentum analysis), so time-based and YES/NO directional restrictions are **outdated and counterproductive**.

---

## Solution Design

### Changes to Make

**REMOVE (outdated for arbitrage):**
1. ✅ **AVOID_HOURS block** - Allow 24/7 trading
2. ✅ **ENABLE_YES_TRADES flag** - Re-enable YES trades completely
3. ✅ **OPTIMAL_HOURS penalty** - Full position size at all times

**KEEP (still valuable):**
1. ✅ **Confidence threshold (75%)** - Still filters low-confidence decisions
2. ✅ **Odds multiplier** - Still penalizes/rejects low-odds bets (<0.25)
3. ✅ **All other risk checks** - Position limits, exposure limits, duplicate checks

### Rationale

**Why remove time blocks?**
- Arbitrage edges exist 24/7 (price feed lag doesn't follow market hours)
- Smart limit orders already handle low liquidity (timeout → market fallback)
- A 10% edge at 3 AM is just as valid as a 10% edge at noon

**Why re-enable YES trades?**
- Old system: YES trades had 10% win rate due to poor sentiment signals
- New system: Probability calculator uses actual BTC price momentum (direction-agnostic)
- Arbitrage detector treats YES/NO equally (both should achieve 70%+ win rate)

**Why keep confidence threshold?**
- Still valuable as quality filter (low-confidence = weak edge)
- Arbitrage system boosts confidence by up to 20% when edge > 10%
- Acts as safeguard against low-quality signals

**Why keep odds multiplier?**
- Low odds (<0.25) indicate asymmetric risk regardless of edge
- Position sizing should scale with odds to manage risk/reward
- Already tested and working well

---

## Implementation Details

### File Changes

**scripts/auto_trade.py:**

1. **Lines 728-745**: DELETE time-based logic entirely
   - Remove: `OPTIMAL_HOURS = range(11, 13)`
   - Remove: `AVOID_HOURS = range(0, 6)`
   - Remove: `if current_hour_utc in AVOID_HOURS: ... return`
   - Remove: `position_size_multiplier = 1.0 if in_optimal_window else 0.7`

2. **Lines 754-756**: DELETE YES trade blocker
   - Remove: `ENABLE_YES_TRADES = False  # TODO: Re-enable after strategy redesign`

3. **Line 1001**: DELETE YES trade check
   - Remove: `if decision.action == "YES" and not ENABLE_YES_TRADES: ... return`

**polymarket/trading/risk.py:**
- **NO CHANGES** - Confidence threshold and odds multiplier stay as-is

### Testing Strategy

**Automated tests:**
- Run full test suite (300 tests) - should all pass (no API changes)
- Verify no regressions in risk management logic

**Manual verification:**
1. Bot accepts trades during 0-6 AM UTC ✓
2. Bot accepts YES decisions from AI ✓
3. Confidence threshold still rejects <75% ✓
4. Odds multiplier still rejects <0.25 odds ✓

---

## Risks & Mitigations

### Risk 1: YES trades might still underperform
**Likelihood**: Low (arbitrage math is direction-agnostic)
**Impact**: Medium ($50-100 loss if win rate drops)
**Mitigation**:
- Monitor first 10 YES trades via Telegram alerts
- If YES win rate < 50% after 20 trades, investigate probability calculator
- Rollback plan: `git revert <commit>` restores YES trade block

### Risk 2: Night trading (0-6 AM UTC) has lower liquidity
**Likelihood**: Medium (known market characteristic)
**Impact**: Low (slippage, not losses)
**Mitigation**:
- Smart limit orders already handle low liquidity (timeout → fallback)
- Monitor fill rates and slippage during night hours
- Position sizing already scales with confidence

### Risk 3: Increased trade frequency hits API rate limits
**Likelihood**: Low (bot has 180s cycle interval)
**Impact**: Low (bot already handles API errors gracefully)
**Mitigation**:
- Monitor logs for API errors
- Rate limiting already implemented in client

---

## Rollback Plan

If critical issues arise (e.g., YES win rate < 40%, excessive losses):

```bash
# Restore conservative blocks
git revert <commit-hash>

# Or manually re-add blocks
# AVOID_HOURS = range(0, 6)
# ENABLE_YES_TRADES = False
```

---

## Success Metrics (24-hour monitoring)

**Trade frequency:**
- Before: ~0 trades/day (blocked by avoid hours + YES disable)
- Target: 15-25 trades/day (arbitrage system active 24/7)

**Win rate:**
- Target: 70%+ overall (arbitrage system design goal)
- YES trades: 60-70% (should match NO trades)
- Night trades (0-6 AM): 65%+ (similar to day)

**System health:**
- No crashes or errors
- API calls within rate limits
- Telegram alerts working
- Database logging accurate

---

## Expected Outcome

With conservative blocks removed:
- ✅ Trade frequency increases 10-20x (from ~0 to 15-25/day)
- ✅ Win rate maintained at 70%+ through mathematical edge detection
- ✅ YES and NO trades perform equally (both use same probability math)
- ✅ 24/7 trading captures arbitrage opportunities around the clock

The arbitrage system's 5-15% mathematical edges should produce consistent returns regardless of time-of-day or trade direction.

---

## Implementation Plan

After design approval:
1. Create git worktree for isolated development
2. Write detailed implementation plan (file-by-file changes)
3. Execute plan with test-driven approach
4. Run full test suite
5. Deploy and monitor for 24 hours

---

**Design Status**: ✅ Approved
**Next Step**: Create implementation plan
