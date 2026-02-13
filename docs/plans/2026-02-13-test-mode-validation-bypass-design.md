# Test Mode Validation Bypass Design

**Date:** 2026-02-13
**Status:** Approved
**Goal:** Simplify test mode to only enforce $1-2 position size and 70% AI confidence, bypassing production safety checks

---

## Overview

Test mode currently bypasses some safety checks but still enforces others that prevent trade execution. This design simplifies test mode to enforce only 2 rules:
1. Position size: $1-2 maximum
2. AI confidence: ≥70% on YES or NO (not HOLD)

All other production safety checks will be bypassed in test mode.

---

## Current Blockers (To Remove)

**Already Bypassed ✓:**
- Movement threshold ($100)
- Volume confirmation
- Timeframe conflicts
- Market regime checks
- Spread checks (>500 bps)

**Still Blocking (Need to Bypass):**
1. YES momentum check (requires $200 upward movement)
2. Unfavorable price movement during execution (>2%)
3. Risk validation: odds check, exposure limit, duplicate position

**To Keep Enforced:**
- Liquidity check (prevents unfillable orders)
- 70% confidence threshold
- Sufficient funds check
- Market active check

---

## Architecture

### Pattern
Pass `test_mode: bool` parameter to validation methods that need conditional bypass logic.

### Files Modified
1. `polymarket/trading/risk.py` - Add test_mode parameter to validate_decision()
2. `scripts/auto_trade.py` - Three bypass points:
   - Pass test_mode to risk manager
   - Bypass YES momentum check
   - Bypass unfavorable price movement check

---

## Implementation Details

### 1. Risk Manager Modifications

**File:** `polymarket/trading/risk.py`

**Change signature:**
```python
async def validate_decision(
    self,
    decision: TradingDecision,
    portfolio_value: Decimal,
    market: dict,
    open_positions: Optional[list[dict]] = None,
    test_mode: bool = False  # NEW parameter
) -> ValidationResult:
```

**Bypass logic:**
```python
# Check 3a: Odds minimum (BYPASS in test mode)
if not test_mode and suggested_size == Decimal("0"):
    return ValidationResult(
        approved=False,
        reason=f"Odds {float(odds):.2f} below minimum threshold 0.25",
        adjusted_position=None
    )

# Check 4: Total exposure (BYPASS in test mode)
if not test_mode:
    open_exposure = Decimal("0")
    if open_positions:
        open_exposure = sum(Decimal(str(p.get("amount", 0))) for p in open_positions)

    max_exposure = portfolio_value * Decimal(str(self.settings.bot_max_exposure_percent))
    if open_exposure + suggested_size > max_exposure:
        return ValidationResult(
            approved=False,
            reason=f"Total exposure {open_exposure + suggested_size} would exceed max {max_exposure}",
            adjusted_position=None
        )

# Check 6: Duplicate position (BYPASS in test mode)
if not test_mode and open_positions:
    for pos in open_positions:
        if pos.get("token_id") == decision.token_id:
            return ValidationResult(
                approved=False,
                reason=f"Already positioned in market {decision.token_id}",
                adjusted_position=None
            )
```

**Unchanged checks (always enforced):**
- Check 1: Confidence threshold
- Check 2: Not HOLD action
- Check 5: Sufficient funds
- Check 7: Market is active

---

### 2. Auto-Trader Modifications

**File:** `scripts/auto_trade.py`

**Change 1: Pass test_mode to risk manager (line ~1218)**
```python
validation = await self.risk_manager.validate_decision(
    decision=decision,
    portfolio_value=portfolio_value,
    market=market_dict,
    open_positions=self.open_positions,
    test_mode=self.test_mode.enabled  # NEW: pass test mode flag
)
```

**Change 2: Bypass YES momentum check (line ~1145)**
```python
# Additional validation: YES trades need stronger momentum to avoid mean reversion
# CHECK FIRST before logging to avoid phantom trades
if decision.action == "YES" and price_to_beat and not self.test_mode.enabled:
    diff, _ = self.market_tracker.calculate_price_difference(
        btc_data.price, price_to_beat
    )
    MIN_YES_MOVEMENT = 200  # $200 minimum for YES trades (higher threshold)

    if diff < MIN_YES_MOVEMENT:
        logger.info(
            "Skipping YES trade - insufficient upward momentum",
            market_id=market.id,
            movement=f"${diff:+,.2f}",
            threshold=f"${MIN_YES_MOVEMENT}",
            reason="Avoid buying exhausted momentum (mean reversion risk)"
        )
        return
```

**Change 3: Bypass unfavorable price movement (line ~1354)**

Wrap the unfavorable price movement check with test mode conditional:
```python
if not self.test_mode.enabled and price_movement_pct > unfavorable_threshold:
    logger.warning(
        f"Price moved {price_movement_pct:+.2f}% unfavorably "
        f"(threshold: {unfavorable_threshold}%)"
    )
    logger.warning(
        "Skipping trade due to unfavorable price movement",
        token=token_name,
        analysis_price=f"{analysis_price:.3f}",
        execution_price=f"{execution_price:.3f}",
        movement_pct=f"{price_movement_pct:+.2f}%"
    )
    # ... existing skip logic ...
    return
```

---

## Testing Strategy

1. **Verify bypass logic:**
   - Run bot in test mode with markets that would normally be blocked
   - Confirm YES trades execute even with <$200 movement
   - Confirm trades execute even with unfavorable price movement
   - Confirm duplicate positions are allowed
   - Confirm low-odds markets (>0.25) are accepted

2. **Verify enforcement:**
   - Confirm <70% confidence trades are rejected
   - Confirm HOLD decisions are forced to YES/NO
   - Confirm liquidity check still blocks unfillable orders
   - Confirm insufficient funds still blocks trades
   - Confirm inactive markets are skipped

3. **Monitor logs:**
   - Look for `[TEST] Bypassing...` messages
   - Verify no unexpected blocks
   - Track successful trade execution rate

---

## Success Criteria

- Test mode enforces exactly 2 rules: $1-2 position size + 70% confidence
- All other production safety checks are bypassed
- Bot successfully executes trades on markets that would normally be filtered
- No regression in production mode (all checks still enforced when test_mode=False)

---

## Rollout Plan

1. Implement changes in current test environment
2. Restart bot with `TEST_MODE=true`
3. Monitor for 1-2 trading cycles (~6-10 minutes)
4. Verify trades execute successfully
5. If successful: document behavior and keep test mode active for CoinGecko signals validation
