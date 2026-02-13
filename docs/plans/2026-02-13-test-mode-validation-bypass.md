# Test Mode Validation Bypass Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Simplify test mode to only enforce $1-2 position size and 70% AI confidence by bypassing production safety checks

**Architecture:** Pass `test_mode: bool` flag to risk manager for conditional bypass logic, wrap YES momentum and unfavorable price movement checks with test mode conditionals

**Tech Stack:** Python, asyncio, Decimal, structlog

---

## Task 1: Add test_mode Parameter to Risk Manager

**Files:**
- Modify: `polymarket/trading/risk.py:28-34`

**Step 1: Update validate_decision signature**

Add `test_mode: bool = False` parameter to method signature:

```python
async def validate_decision(
    self,
    decision: TradingDecision,
    portfolio_value: Decimal,
    market: dict,
    open_positions: Optional[list[dict]] = None,
    test_mode: bool = False  # NEW parameter
) -> ValidationResult:
    """Validate a trading decision against risk rules."""
```

**Step 2: Verify signature change**

Check that file was updated correctly:
```bash
grep -A6 "async def validate_decision" polymarket/trading/risk.py
```

Expected: See `test_mode: bool = False` parameter

**Step 3: Commit**

```bash
git add polymarket/trading/risk.py
git commit -m "feat: add test_mode parameter to risk manager validate_decision"
```

---

## Task 2: Bypass Odds Check in Test Mode

**Files:**
- Modify: `polymarket/trading/risk.py:64-70`

**Step 1: Wrap odds check with test_mode conditional**

Modify Check 3a (lines 64-70) to bypass in test mode:

```python
# Check 3a: Reject if odds below minimum threshold
if not test_mode and suggested_size == Decimal("0"):
    return ValidationResult(
        approved=False,
        reason=f"Odds {float(odds):.2f} below minimum threshold 0.25",
        adjusted_position=None
    )
```

**Step 2: Verify the change**

Check that the conditional was added:
```bash
grep -A6 "Check 3a: Reject if odds" polymarket/trading/risk.py
```

Expected: See `if not test_mode and suggested_size`

**Step 3: Commit**

```bash
git add polymarket/trading/risk.py
git commit -m "feat: bypass odds check in test mode"
```

---

## Task 3: Bypass Total Exposure Check in Test Mode

**Files:**
- Modify: `polymarket/trading/risk.py:72-86`

**Step 1: Wrap exposure check with test_mode conditional**

Modify Check 4 (lines 72-86) to bypass in test mode:

```python
# Check 4: Total exposure
if not test_mode:
    open_exposure = Decimal("0")
    if open_positions:
        open_exposure = sum(
            Decimal(str(p.get("amount", 0))) for p in open_positions
        )

    max_exposure = portfolio_value * Decimal(str(self.settings.bot_max_exposure_percent))

    if open_exposure + suggested_size > max_exposure:
        return ValidationResult(
            approved=False,
            reason=f"Total exposure {open_exposure + suggested_size} would exceed max {max_exposure}",
            adjusted_position=None
        )
```

**Step 2: Verify the change**

Check that the block is wrapped:
```bash
grep -B2 "Check 4: Total exposure" polymarket/trading/risk.py
```

Expected: See `if not test_mode:` before the check

**Step 3: Commit**

```bash
git add polymarket/trading/risk.py
git commit -m "feat: bypass total exposure check in test mode"
```

---

## Task 4: Bypass Duplicate Position Check in Test Mode

**Files:**
- Modify: `polymarket/trading/risk.py:96-104`

**Step 1: Wrap duplicate position check with test_mode conditional**

Modify Check 6 (lines 96-104) to bypass in test mode:

```python
# Check 6: Not already positioned in this market
if not test_mode and open_positions:
    for pos in open_positions:
        if pos.get("token_id") == decision.token_id:
            return ValidationResult(
                approved=False,
                reason=f"Already positioned in market {decision.token_id}",
                adjusted_position=None
            )
```

**Step 2: Verify the change**

Check that the conditional was added:
```bash
grep -A7 "Check 6: Not already positioned" polymarket/trading/risk.py
```

Expected: See `if not test_mode and open_positions:`

**Step 3: Commit**

```bash
git add polymarket/trading/risk.py
git commit -m "feat: bypass duplicate position check in test mode"
```

---

## Task 5: Pass test_mode to Risk Manager

**Files:**
- Modify: `scripts/auto_trade.py:1218-1223`

**Step 1: Add test_mode parameter to validate_decision call**

Update the risk manager call (lines 1218-1223):

```python
validation = await self.risk_manager.validate_decision(
    decision=decision,
    portfolio_value=portfolio_value,
    market=market_dict,
    open_positions=self.open_positions,
    test_mode=self.test_mode.enabled  # NEW: pass test mode flag
)
```

**Step 2: Verify the change**

Check that test_mode is passed:
```bash
grep -A5 "validation = await self.risk_manager.validate_decision" scripts/auto_trade.py
```

Expected: See `test_mode=self.test_mode.enabled`

**Step 3: Commit**

```bash
git add scripts/auto_trade.py
git commit -m "feat: pass test_mode flag to risk manager"
```

---

## Task 6: Bypass YES Momentum Check in Test Mode

**Files:**
- Modify: `scripts/auto_trade.py:1143-1158`

**Step 1: Wrap YES momentum check with test_mode conditional**

Modify the momentum check (lines 1143-1158) to add `not self.test_mode.enabled`:

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

**Step 2: Verify the change**

Check that the conditional includes test_mode:
```bash
grep -A2 "Additional validation: YES trades" scripts/auto_trade.py
```

Expected: See `and not self.test_mode.enabled` in the if condition

**Step 3: Commit**

```bash
git add scripts/auto_trade.py
git commit -m "feat: bypass YES momentum check in test mode"
```

---

## Task 7: Bypass Unfavorable Price Movement Check in Test Mode

**Files:**
- Modify: `scripts/auto_trade.py:1351-1357`

**Step 1: Wrap unfavorable price movement check with test_mode conditional**

Modify the price movement check (lines 1351-1357) to bypass in test mode:

```python
# Check unfavorable movement
if not self.test_mode.enabled and not is_favorable and abs(price_change_pct) > unfavorable_threshold:
    reason = (
        f"Price moved {price_change_pct:+.2f}% worse "
        f"(threshold: {unfavorable_threshold}%)"
    )
    logger.warning(
        "Skipping trade due to unfavorable price movement",
```

**Step 2: Verify the change**

Check that test_mode is in the condition:
```bash
grep -B3 "Skipping trade due to unfavorable price movement" scripts/auto_trade.py
```

Expected: See `not self.test_mode.enabled and` in the if condition

**Step 3: Commit**

```bash
git add scripts/auto_trade.py
git commit -m "feat: bypass unfavorable price movement check in test mode"
```

---

## Task 8: Integration Verification

**Files:**
- Test: Manual testing with bot in test mode

**Step 1: Restart bot with test mode**

Stop current bot and restart with test mode enabled:
```bash
pkill -f "python3 scripts/auto_trade.py"
cd /root/polymarket-scripts
TEST_MODE=true python3 scripts/auto_trade.py > /root/test-direct.log 2>&1 &
```

**Step 2: Monitor logs for test mode activation**

Wait 5 seconds then check logs:
```bash
sleep 5 && tail -20 /root/test-direct.log
```

Expected: See test mode banner and bot starting

**Step 3: Wait for first trading cycle**

Wait 3 minutes for a full cycle to complete:
```bash
sleep 180
```

**Step 4: Check for successful trade or appropriate rejection**

Check recent log entries:
```bash
tail -100 /root/test-direct.log | grep -E "TEST|AI Decision|Cycle completed|Skipping"
```

Expected outcomes:
- If AI confidence >= 70%: Trade executes or is logged with decision
- If AI confidence < 70%: `[TEST] Skipping trade - confidence below threshold`
- NO messages about: "insufficient upward momentum", "unfavorable price movement", "odds below minimum", "Total exposure", "Already positioned"

**Step 5: Verify bypasses are working**

Confirm test mode bypasses in logs:
```bash
tail -200 /root/test-direct.log | grep -E "Bypassing|TEST"
```

Expected: See `[TEST] Bypassing...` messages for various checks

---

## Task 9: Documentation Update

**Files:**
- Modify: `docs/plans/2026-02-13-test-mode-validation-bypass-design.md`

**Step 1: Update design document status**

Add implementation notes at the end:

```markdown
## Implementation Notes

**Completed:** 2026-02-13

**Changes made:**
1. Added `test_mode: bool = False` parameter to `RiskManager.validate_decision()`
2. Bypassed 3 risk checks in test mode: odds (check 3a), exposure (check 4), duplicates (check 6)
3. Modified `auto_trade.py` to pass `test_mode=self.test_mode.enabled` to risk manager
4. Bypassed YES momentum check with `not self.test_mode.enabled` conditional
5. Bypassed unfavorable price movement check with `not self.test_mode.enabled` conditional

**Commits:**
- feat: add test_mode parameter to risk manager validate_decision
- feat: bypass odds check in test mode
- feat: bypass total exposure check in test mode
- feat: bypass duplicate position check in test mode
- feat: pass test_mode flag to risk manager
- feat: bypass YES momentum check in test mode
- feat: bypass unfavorable price movement check in test mode
```

**Step 2: Commit documentation**

```bash
git add docs/plans/2026-02-13-test-mode-validation-bypass-design.md
git commit -m "docs: add implementation notes to test mode bypass design"
```

---

## Success Criteria

✅ Risk manager accepts `test_mode` parameter
✅ Odds check bypassed in test mode
✅ Exposure limit bypassed in test mode
✅ Duplicate position check bypassed in test mode
✅ YES momentum check bypassed in test mode
✅ Unfavorable price movement check bypassed in test mode
✅ Bot executes trades in test mode that would be blocked in production
✅ Bot still enforces 70% confidence threshold
✅ Bot still enforces $1-2 position size limit
✅ All changes committed with descriptive messages

---

## Rollback Plan

If issues arise, revert commits in reverse order:

```bash
git log --oneline -7  # Get last 7 commit hashes
git revert <commit-hash> --no-edit  # Revert each commit
```

Or reset to commit before changes:
```bash
git log --oneline  # Find commit hash before implementation
git reset --hard <commit-hash>
```
