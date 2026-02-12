# Odds-Adjusted Position Sizing Design

**Date:** 2026-02-12
**Status:** Approved
**Priority:** Critical - Bot losing money due to inverted position sizing

---

## Problem Statement

The bot is losing money because position sizing is inverted relative to odds:
- **Trade #273**: Bet $9.56 on 0.31 odds (low probability) → Lost entire $9.56 stake
- **Trade #269**: Bet $5.00 on 0.83 odds (high probability) → Won $1.02

**Root cause:** Position sizing scales with confidence but ignores odds asymmetry. On Polymarket:
- Low odds (0.31) = Risk entire stake to win 2.2x
- High odds (0.83) = Risk stake to win 0.2x

When accuracy is 50%, betting MORE on low odds creates catastrophic losses.

---

## Solution: Odds-Based Position Scaling

Add odds-aware scaling to `RiskManager._calculate_position_size()` that reduces position size for low-odds bets.

### Scaling Formula

```python
def _calculate_odds_multiplier(self, odds: float) -> float:
    """
    Scale down position size for low-odds bets.

    Logic:
    - odds >= 0.50: No scaling (100% of position)
    - odds < 0.50:  Linear scale from 100% down to 50%
    - odds < 0.25:  Reject bet entirely (too risky)

    Examples:
    - 0.83 odds → 1.00x (no reduction)
    - 0.50 odds → 1.00x (breakeven)
    - 0.40 odds → 0.80x (20% reduction)
    - 0.31 odds → 0.62x (38% reduction)
    - 0.25 odds → 0.50x (50% reduction, minimum)
    - 0.20 odds → REJECT (below threshold)
    """
    MINIMUM_ODDS = 0.25
    SCALE_THRESHOLD = 0.50

    if odds < MINIMUM_ODDS:
        return 0.0  # Reject bet

    if odds >= SCALE_THRESHOLD:
        return 1.0  # No scaling needed

    # Linear interpolation between 0.5x and 1.0x
    multiplier = 0.5 + (odds - MINIMUM_ODDS) / (SCALE_THRESHOLD - MINIMUM_ODDS) * 0.5
    return multiplier
```

---

## Architecture

### Integration Point

The change fits into the existing risk management flow:

```
AI Decision (position_size=9.56, confidence=0.85, action="YES")
    ↓
RiskManager.validate_decision(decision, portfolio, market)
    ├─ Extract odds for action from market data
    │  (YES → yes_price, NO → no_price)
    ↓
_calculate_position_size(decision, portfolio, max_position, odds)
    ├─ Calculate base size from confidence
    ├─ Apply dollar cap
    ├─ Consider AI-suggested size
    └─ **NEW: Apply odds-based scaling**
    ↓
Return adjusted_position (safe amount to bet)
```

### File Changes

**`polymarket/trading/risk.py`:**

1. **Add helper method** to extract odds:
```python
def _extract_odds_for_action(self, action: str, market: dict) -> float:
    """Get the odds for the side being bet."""
    if action == "YES":
        return float(market.get("yes_price", 0.50))
    elif action == "NO":
        return float(market.get("no_price", 0.50))
    return 0.50  # Default fallback
```

2. **Modify `validate_decision()`** to extract and pass odds:
```python
async def validate_decision(
    self,
    decision: TradingDecision,
    portfolio_value: Decimal,
    market: dict,
    open_positions: Optional[list[dict]] = None
) -> ValidationResult:
    # ... existing checks ...

    # Extract odds for the action
    odds = self._extract_odds_for_action(decision.action, market)

    # Calculate position size with odds awareness
    suggested_size = self._calculate_position_size(
        decision, portfolio_value, max_position, odds
    )
```

3. **Update `_calculate_position_size()` signature and logic**:
```python
def _calculate_position_size(
    self,
    decision: TradingDecision,
    portfolio_value: Decimal,
    max_position: Decimal,
    odds: float  # NEW parameter
) -> Decimal:
    """Calculate position size based on confidence and odds."""
    # ... existing confidence-based calculation ...

    # NEW: Apply odds-based scaling
    odds_multiplier = self._calculate_odds_multiplier(odds)

    if odds_multiplier == 0.0:
        logger.info(
            "Bet rejected - odds below minimum threshold",
            odds=odds,
            minimum=0.25
        )
        return Decimal("0")

    final_size = calculated * Decimal(str(odds_multiplier))

    logger.info(
        "Position sized with odds adjustment",
        original_size=calculated,
        odds=odds,
        multiplier=odds_multiplier,
        final_size=final_size
    )

    return final_size
```

4. **Add the scaling function** (formula shown above)

---

## Expected Impact

### On Historical Trades

With odds-adjusted sizing, the same 2 trades would have been:

**Trade #273 (LOSS):**
- Original: $9.56 stake at 0.31 odds → Lost $9.56
- **With fix**: $5.93 stake at 0.31 odds → Would lose $5.93
- **Improvement**: Saves $3.63 (38% reduction in loss)

**Trade #269 (WIN):**
- Original: $5.00 stake at 0.83 odds → Won $1.02
- **With fix**: $5.00 stake at 0.83 odds → Win $1.02
- **Impact**: Unchanged (high odds, no scaling)

**Net P&L:**
- Current: -$8.54
- **With fix**: -$4.91
- **Improvement**: 44% reduction in losses

### Long-Term Profitability

With 55-60% win rate (achievable with good signals):
- Reduced variance from low-odds bets
- Smaller losses on incorrect predictions
- Maintained upside on high-odds bets
- Expected profitability with >52% accuracy

---

## Testing Strategy

### 1. Unit Tests

```python
def test_odds_multiplier():
    """Test odds multiplier calculation."""
    risk_mgr = RiskManager(settings)

    # High odds - no scaling
    assert risk_mgr._calculate_odds_multiplier(0.83) == 1.0
    assert risk_mgr._calculate_odds_multiplier(0.50) == 1.0

    # Low odds - scaled down
    assert risk_mgr._calculate_odds_multiplier(0.40) == 0.80
    assert risk_mgr._calculate_odds_multiplier(0.31) == 0.62
    assert risk_mgr._calculate_odds_multiplier(0.25) == 0.50

    # Below threshold - rejected
    assert risk_mgr._calculate_odds_multiplier(0.20) == 0.0
    assert risk_mgr._calculate_odds_multiplier(0.15) == 0.0
```

### 2. Integration Tests

```python
async def test_position_sizing_with_odds():
    """Test position sizing with odds adjustment."""
    risk_mgr = RiskManager(settings)

    # Trade #273 scenario (should be reduced)
    decision = TradingDecision(
        action="YES",
        confidence=0.85,
        position_size=9.56
    )
    market = {"yes_price": 0.31, "no_price": 0.69}

    result = await risk_mgr.validate_decision(
        decision,
        portfolio_value=Decimal("100"),
        market=market
    )

    assert result.approved
    assert result.adjusted_position < Decimal("6.00")  # Scaled down

    # Trade #269 scenario (should be unchanged)
    decision = TradingDecision(
        action="NO",
        confidence=0.73,
        position_size=5.0
    )
    market = {"yes_price": 0.17, "no_price": 0.83}

    result = await risk_mgr.validate_decision(
        decision,
        portfolio_value=Decimal("100"),
        market=market
    )

    assert result.approved
    assert result.adjusted_position == Decimal("5.00")  # High odds, no scaling
```

### 3. Manual Validation

After deployment:
1. Monitor first 5-10 trades
2. Verify log output shows odds multiplier applied
3. Confirm:
   - Low-odds bets (<0.40) scaled down
   - High-odds bets (>0.60) remain full size
   - Sub-0.25 odds rejected with clear logs

---

## Risk Assessment

**Low risk:**
- Purely additive logic (doesn't break existing functionality)
- Only affects position sizing, not decision making
- Fail-safe: defaults to 0.50 odds if market data missing
- Extensive logging for debugging

**Backwards compatibility:**
- Existing position sizing logic preserved
- Only adds scaling multiplier on top
- High-odds bets unchanged (multiplier = 1.0)

---

## Success Criteria

1. ✅ Low-odds bets (<0.40) automatically scaled down
2. ✅ Sub-0.25 odds rejected with clear logging
3. ✅ High-odds bets (>0.60) unaffected
4. ✅ Net P&L improves over 10-trade sample
5. ✅ No unintended position rejections
6. ✅ Clear audit trail in logs

---

## Implementation Plan

1. Add `_calculate_odds_multiplier()` method
2. Add `_extract_odds_for_action()` helper
3. Update `validate_decision()` to extract odds
4. Update `_calculate_position_size()` signature and logic
5. Add unit tests
6. Add integration tests
7. Manual testing with bot restart
8. Monitor first 10 trades for validation

---

## References

- Trade analysis: Sequential Thinking session 2026-02-12
- Polymarket docs: Binary market payoff structure
- Kelly Criterion: Odds-adjusted position sizing
