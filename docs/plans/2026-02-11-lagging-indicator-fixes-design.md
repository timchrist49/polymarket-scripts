# Lagging Indicator Fixes - Design Document

**Date:** 2026-02-11
**Status:** Approved
**Goal:** Fix the bot's losing streak by addressing lagging indicator problems

---

## Problem Statement

**Root Cause Identified:** The bot follows prediction market sentiment (what traders THINK will happen) instead of actual BTC price movement (what IS happening).

**Evidence:**
- Bot analyzes 2 minutes of Polymarket trade data (momentum, volume flow, whale activity)
- All metrics measure **what already happened**, not what will happen next
- Bot consistently bets after the move already occurred and reversed

**Example from Logs:**
```
02:57 - BTC: $68,626 (down $17) | Market: BEARISH (-0.84) | Bot: NO
03:02 - BTC: $68,646 (up $20)   | Market: BEARISH (-0.71) | Bot: NO
→ Bot lost both trades - BTC rebounded while bot was following lagging signal
```

---

## Solution: Three Complementary Fixes

### Fix #1: Price-to-Beat Alignment Check
**What:** Add validation rules to AI prompt to prevent betting against actual BTC direction
**How:** Enhance AI prompt with explicit contradiction detection
**Impact:** Prevents betting NO when BTC is actually UP from market start

### Fix #2: BTC Movement Check
**What:** Compare market sentiment vs actual BTC movement over 5 minutes
**How:** Fetch price history, calculate momentum, add to AI prompt
**Impact:** Catches lagging signals in real-time

### Fix #3: Reduce Momentum Weight
**What:** Decrease influence of most lagging indicator (momentum)
**How:** Reweight: momentum 40%→20%, volume_flow 35%→50%, whale 25%→30%
**Impact:** Market score reacts faster to current conditions

---

## Architecture

### Implementation Strategy: AI Prompt Enhancement (Option A)
- **Chosen Approach:** Add validation checks to AI prompt (not hard-coded in risk manager)
- **Rationale:**
  - Faster to implement
  - More flexible (AI can reason about edge cases)
  - GPT-5-Nano reliable at following instructions
  - AI can explain reasoning for HOLD decisions
- **Fallback:** Can add hard validation in risk.py later if needed

### Data Flow
```
BTCPriceService
  ↓ [get_price_history(5min)]
  ↓ calculate momentum
  ↓
MarketMicrostructure
  ↓ [new weights: momentum=0.20, volume=0.50, whale=0.30]
  ↓ market score
  ↓
AI Decision
  ↓ [enhanced prompt with validation rules + momentum check]
  ↓ decision (with contradiction detection)
```

---

## Detailed Implementation

### Fix #1: Price-to-Beat Alignment Check

**Location:** `polymarket/trading/ai_decision.py` - `_build_prompt()` method

**Add after PRICE-TO-BEAT ANALYSIS section:**
```
⚠️ SIGNAL VALIDATION RULES:
1. If price-to-beat shows BTC is UP from start (+X%)
   AND market signals are BEARISH (score < -0.3)
   → This is a CONTRADICTION - market is lagging
   → Decision: HOLD (don't bet against actual movement)

2. If price-to-beat shows BTC is DOWN from start (-X%)
   AND market signals are BULLISH (score > +0.3)
   → This is a CONTRADICTION - market is lagging
   → Decision: HOLD

3. If signals ALIGN with actual direction → proceed normally
```

**Logic:**
- Only applies when `has_price_to_beat == True`
- Uses existing price_diff calculation (already in prompt)
- Threshold: |market_score| > 0.3 (STRONG signals only)

---

### Fix #2: BTC Movement Check

**Location:** `scripts/auto_trade.py` - new method + prompt enhancement

**New Method:**
```python
async def _get_btc_momentum(self, btc_service, current_price: Decimal) -> dict | None:
    """
    Calculate actual BTC momentum over last 5 minutes.

    Returns:
        {
            'price_5min_ago': Decimal,
            'momentum_pct': float,
            'direction': 'UP' | 'DOWN' | 'FLAT'
        }
        or None if history unavailable
    """
    try:
        # Use existing BTCPriceService.get_price_history()
        history = await btc_service.get_price_history(minutes=5)

        if not history or len(history) < 2:
            return None

        price_5min_ago = Decimal(str(history[0]['price']))
        momentum_pct = float((current_price - price_5min_ago) / price_5min_ago * 100)

        direction = 'UP' if momentum_pct > 0.1 else 'DOWN' if momentum_pct < -0.1 else 'FLAT'

        return {
            'price_5min_ago': price_5min_ago,
            'momentum_pct': momentum_pct,
            'direction': direction
        }
    except Exception as e:
        logger.warning("BTC momentum calculation failed", error=str(e))
        return None  # Graceful fallback
```

**Prompt Addition (in `ai_decision.py`):**
```
ACTUAL BTC MOMENTUM (last 5 minutes):
- 5 minutes ago: ${price_5min_ago:,.2f}
- Current: ${current_price:,.2f}
- Change: {momentum_pct:+.2f}% ({direction})

⚠️ COMPARE WITH MARKET SIGNALS:
- Market microstructure score: {market_score:+.2f}
- If market says DOWN but BTC is UP → market is LAGGING
- If market says UP but BTC is DOWN → market is LAGGING
- Lagging signals often lead to losing trades - consider HOLD
```

**Integration:**
- Call `_get_btc_momentum()` in `_process_market()` before AI decision
- Pass result to `ai_service.make_decision()` via market_dict
- Add to AI prompt only if momentum data available

---

### Fix #3: Reduce Momentum Weight

**Location:** `polymarket/trading/market_microstructure.py` - lines 32-36

**Change:**
```python
# OLD
WEIGHTS = {
    'momentum': 0.40,
    'volume_flow': 0.35,
    'whale': 0.25
}

# NEW
WEIGHTS = {
    'momentum': 0.20,      # Reduced from 0.40 (less lag)
    'volume_flow': 0.50,   # Increased from 0.35 (more current)
    'whale': 0.30          # Increased from 0.25 (behavioral)
}
```

**Rationale:**
- **Momentum** measures YES token price change over 2 minutes → MOST lagging
- **Volume flow** measures current YES vs NO buying → MORE current
- **Whale activity** measures large trader behavior → BEHAVIORAL (less time-dependent)

**Impact:**
- Market score formula: `score = momentum*0.20 + volume*0.50 + whale*0.30`
- Volume flow now dominates (50% vs old 35%)
- Momentum influence cut in half (20% vs old 40%)

---

## Testing Strategy

### Unit Tests

**Test 1: Momentum Calculation**
```python
def test_get_btc_momentum():
    # Mock price history: 67500 → 68000
    history = [{'price': 67500}]
    current = Decimal('68000')

    result = await _get_btc_momentum(mock_service, current)

    assert result['momentum_pct'] == 0.74  # (68000-67500)/67500
    assert result['direction'] == 'UP'
```

**Test 2: Graceful Fallback**
```python
def test_momentum_unavailable():
    # Mock service returns empty history
    result = await _get_btc_momentum(mock_service_empty, current)
    assert result is None  # Doesn't crash
```

**Test 3: Weight Changes**
```python
def test_new_weights():
    service = MarketMicrostructureService(...)
    assert service.WEIGHTS['momentum'] == 0.20
    assert service.WEIGHTS['volume_flow'] == 0.50
    assert service.WEIGHTS['whale'] == 0.30
```

---

### Integration Test

**Run bot once with new changes:**
```bash
python scripts/auto_trade.py --once
```

**Verify in logs:**
1. ✅ "Market weights applied: momentum=0.20, volume=0.50, whale=0.30"
2. ✅ "ACTUAL BTC MOMENTUM" section appears in prompt (if history available)
3. ✅ "SIGNAL VALIDATION RULES" section appears when price-to-beat exists
4. ✅ AI reasoning mentions contradiction checks when applicable
5. ✅ HOLD decisions when contradictions detected

---

### Live Testing (20 Trades)

**Metrics to Track:**

| Metric | Before | Target | Measurement |
|--------|--------|--------|-------------|
| Win Rate | ~30-40% | 55%+ | Wins / total trades |
| Contradictory Trades | High | <20% | Signal ≠ actual direction |
| HOLD Rate | Low | 30-40% | HOLDs / total cycles |
| Avg Confidence (when trading) | 0.85 | 0.80-0.85 | Mean confidence |

**What to Monitor:**
- HOLD decisions with reasoning: "Contradiction detected - market bearish but BTC up"
- Comparison logs: "BTC momentum +0.5% UP vs market signal -0.4 BEARISH"
- Win rate improvement over 20 trades
- Whether AI correctly identifies contradictions

**Success Criteria:**
- Win rate improves to 55%+ (from baseline ~30-40%)
- Contradictory trades reduced by 80%+
- Bot HOLDs 30-40% of time (filtering bad signals)

---

## Error Handling

### Price History Unavailable
```python
if btc_momentum is None:
    # Don't add momentum section to prompt
    # Fall back to price-to-beat check only
    logger.info("BTC momentum unavailable, using price-to-beat validation only")
```

### Price-to-Beat Unavailable
```python
# Already handled - shows "Not available (market just started)"
# Validation rules only apply when has_price_to_beat == True
```

### Graceful Degradation
- Each fix is independent
- If both momentum AND price-to-beat unavailable → bot works with market signals only (current behavior)
- Partial fixes still help (e.g., weights change works even if momentum calc fails)

---

## Expected Outcomes

### Immediate Effects

1. **Fewer Contradictory Trades**
   - Bot will HOLD when market signals contradict actual BTC movement
   - Prevents "following the wrong crowd" losses

2. **Better Signal Timing**
   - 5-minute momentum check catches reversals
   - Reduces betting on moves that already happened

3. **Faster Market Score Reaction**
   - Reduced momentum weight = less lag in overall score
   - Volume flow (50%) reacts to current trading activity

### Long-Term Benefits

1. **Improved Win Rate**
   - Target: 55%+ (from ~30-40% baseline)
   - More selective trading (higher HOLD rate)

2. **Better Risk Management**
   - AI provides reasoning for HOLD decisions
   - User can review contradiction logs

3. **Foundation for Future Improvements**
   - Can add contrarian strategy later (bet opposite when crowd very wrong)
   - Can add ML training on "when to trust vs ignore market signals"

---

## Rollback Plan

**If win rate drops below 40% after 20 trades:**

1. **Quick Revert:**
   ```bash
   # Revert weights
   WEIGHTS = {'momentum': 0.40, 'volume_flow': 0.35, 'whale': 0.25}

   # Remove prompt sections
   # (comment out validation rules and momentum sections)
   ```

2. **Partial Rollback:**
   - Can disable individual fixes to isolate which helped/hurt
   - Try combinations: weights only, validation only, etc.

3. **Debugging:**
   - Review logs for patterns in losing trades
   - Check if AI is ignoring validation rules
   - Analyze if contradictions were false positives

**All changes are in config/prompt (easy to modify) - no database schema changes or complex migrations needed.**

---

## Implementation Checklist

- [ ] Fix #3: Change weights in `market_microstructure.py`
- [ ] Fix #2: Add `_get_btc_momentum()` method to `auto_trade.py`
- [ ] Fix #2: Call momentum method in `_process_market()`
- [ ] Fix #2: Add momentum data to AI prompt
- [ ] Fix #1: Add validation rules section to AI prompt
- [ ] Unit tests for momentum calculation
- [ ] Integration test (`--once` run)
- [ ] Live testing (20 trades)
- [ ] Update documentation with new behavior
- [ ] Commit changes with detailed message

---

## Success Criteria

✅ **Design Approved**
✅ **Implementation Plan Ready**
⏳ **Code Implementation**
⏳ **Testing Complete**
⏳ **Win Rate Improved to 55%+**

---

*Design validated by user on 2026-02-11*
