# WebSocket Message Filtering Fix Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix bot's inability to detect correct market odds by implementing client-side WebSocket message filtering.

**Architecture:** Add defensive filtering to only process book messages for subscribed token IDs, enhance logging for subscription verification, and ensure atomic state updates during market transitions.

**Tech Stack:** Python 3.11, asyncio, websockets, structlog

---

## Task 1: Add Token Filtering to Message Handler

**Files:**
- Modify: `/root/polymarket-scripts/polymarket/trading/realtime_odds_streamer.py:333-343`

**Step 1: Read current implementation**

```bash
cd /root/polymarket-scripts
grep -A 15 "async def _handle_single_message" polymarket/trading/realtime_odds_streamer.py
```

Expected: See current implementation without token filtering

**Step 2: Add client-side token ID filtering**

Replace the `_handle_single_message()` method (lines 333-343):

```python
async def _handle_single_message(self, data: dict):
    """Process a single WebSocket message."""
    event_type = data.get('event_type')

    logger.debug("WebSocket message received", event_type=event_type)

    if event_type == 'book':
        token_id = data.get('market')  # hex string

        # CRITICAL FIX: Only process messages for subscribed tokens
        if token_id and self._current_token_ids:
            if token_id not in self._current_token_ids:
                logger.debug(
                    "Ignoring book message for unsubscribed token",
                    token_id=token_id[:16] + "...",
                    subscribed_tokens=[t[:16] + "..." for t in self._current_token_ids]
                )
                return

        await self._process_book_message(data)

    # Ignore other message types (last_trade_price, price_change)
```

**Step 3: Verify syntax**

```bash
python3 -m py_compile polymarket/trading/realtime_odds_streamer.py
```

Expected: No syntax errors

**Step 4: Commit**

```bash
git add polymarket/trading/realtime_odds_streamer.py
git commit -m "fix: add client-side token ID filtering to WebSocket handler

Only process book messages for subscribed token IDs to prevent
processing odds data from other markets.

Polymarket CLOB broadcasts all market messages regardless of
subscription. This defensive filtering ensures we only update
odds for our actual subscribed market.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 2: Add Enhanced Subscription Logging

**Files:**
- Modify: `/root/polymarket-scripts/polymarket/trading/realtime_odds_streamer.py:287-293`

**Step 1: Read current subscription code**

```bash
grep -A 10 "Send subscription message" polymarket/trading/realtime_odds_streamer.py
```

Expected: See current subscription without enhanced logging

**Step 2: Add enhanced logging after subscription**

Modify the subscription section (after line 292):

```python
# Send subscription message (CLOB format)
subscribe_msg = {
    "assets_ids": token_ids,
    "type": "market"  # lowercase per CLOB spec
}
await ws.send(json.dumps(subscribe_msg))

# Enhanced logging for debugging
logger.info(
    "🔔 Subscription sent",
    market_id=market.id,
    market_slug=market.slug,
    token_count=len(token_ids),
    token_ids=[t[:16] + "..." for t in token_ids]
)
```

**Step 3: Verify syntax**

```bash
python3 -m py_compile polymarket/trading/realtime_odds_streamer.py
```

Expected: No syntax errors

**Step 4: Commit**

```bash
git add polymarket/trading/realtime_odds_streamer.py
git commit -m "feat: add enhanced subscription logging

Add detailed logging when sending WebSocket subscriptions to help
debug subscription behavior and verify token IDs match what we
expect to receive.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 3: Ensure Atomic State Updates During Market Transitions

**Files:**
- Modify: `/root/polymarket-scripts/polymarket/trading/realtime_odds_streamer.py:269-271`

**Step 1: Read current state update code**

```bash
grep -B 5 -A 10 "self._current_market_id = market.id" polymarket/trading/realtime_odds_streamer.py | head -20
```

Expected: See state updates, verify if lock is used

**Step 2: Ensure all state updates happen under lock**

Find where market state is set (around line 269) and ensure it's wrapped in the lock:

```python
# Store market state atomically
async with self._lock:
    self._current_market_id = market.id
    self._current_market_slug = market.slug
    self._current_token_ids = token_ids
```

If already using lock, verify all three fields are updated together.
If not using lock, add the lock wrapper.

**Step 3: Verify syntax**

```bash
python3 -m py_compile polymarket/trading/realtime_odds_streamer.py
```

Expected: No syntax errors

**Step 4: Commit**

```bash
git add polymarket/trading/realtime_odds_streamer.py
git commit -m "fix: ensure atomic state updates during market transitions

Wrap all market state updates (ID, slug, token IDs) in async lock
to prevent race conditions where message handler checks token IDs
during transition.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 4: Restart Bot and Monitor Logs

**Files:**
- Monitor: `/root/polymarket-scripts/logs/bot.log`

**Step 1: Stop the bot**

```bash
cd /root/polymarket-scripts
systemctl stop polymarket-bot
```

Expected: Bot service stopped

**Step 2: Clear old logs for clean monitoring**

```bash
> logs/bot.log
```

Expected: Empty log file

**Step 3: Start the bot**

```bash
systemctl start polymarket-bot
```

Expected: Bot service started

**Step 4: Monitor logs for Phase 1 verification (5 minutes)**

```bash
tail -f logs/bot.log | grep -E "(Ignoring book|Odds updated from book|Subscription sent)"
```

**Expected output pattern:**
```
🔔 Subscription sent | market_id=1384xxx | token_ids=['0xd5a0140b...', '0x32a37e45...']
Ignoring book message for unsubscribed token | token_id=0x3f3eab02...
Odds updated from book | token_id=0xd5a0140b... | market_id=1384xxx
```

**Success Criteria:**
- ✅ See "Ignoring book" messages for unsubscribed tokens
- ✅ See "Odds updated" ONLY for subscribed token IDs
- ✅ Token IDs in "Odds updated" match token IDs in "Subscription sent"

**Step 5: Document verification results**

If successful, proceed to Task 5.
If issues found, investigate and adjust implementation.

---

## Task 5: Market Transition Verification (15-30 Minutes)

**Files:**
- Monitor: `/root/polymarket-scripts/logs/bot.log`

**Step 1: Wait for market transition**

Markets cycle every ~15 minutes. Watch logs for:

```bash
tail -f logs/bot.log | grep -E "(Market transition|Closing connection|Subscription sent)"
```

**Step 2: Verify transition handling**

**Expected log sequence:**
```
Market transition detected | old_market=1384xxx | new_market=1384yyy
Closing connection to resubscribe to new market
🔔 Subscription sent | market_id=1384yyy | token_ids=['0xabc...', '0xdef...']
Ignoring book for unsubscribed token | token_id=0xd5a0140b... (old market)
Odds updated from book | token_id=0xabc... | market_id=1384yyy (new market)
```

**Success Criteria:**
- ✅ Old market tokens are filtered after transition
- ✅ New market tokens are processed immediately
- ✅ No processing of wrong market during transition window

**Step 3: Document transition verification**

If successful, proceed to Task 6.
If issues found, investigate race conditions.

---

## Task 6: Alert Verification (When Odds Spike)

**Files:**
- Monitor: `/root/polymarket-scripts/logs/bot.log`
- Monitor: Telegram app

**Step 1: Check current market odds on Polymarket**

Visit: https://polymarket.com/

Find current BTC 15-minute market and note odds.

**Step 2: Compare with bot logs**

```bash
tail -50 logs/bot.log | grep "Odds updated from book"
```

Expected: Bot's odds should match Polymarket's displayed odds

**Step 3: Wait for 70%+ odds spike**

When odds reach 70%+ on Polymarket:

1. Verify bot detects same odds in logs
2. Verify sustained detection (5 consecutive checks)
3. Verify Telegram alert arrives

**Success Criteria:**
- ✅ Bot odds match Polymarket odds
- ✅ Alert arrives within 30 seconds of sustained 70%+

**Step 4: Document alert verification**

If successful, fixes are complete!
If no alert, investigate @superpowers:systematic-debugging

---

## Task 7: Update Design Document Status

**Files:**
- Modify: `/root/polymarket-scripts/docs/plans/2026-02-17-websocket-filtering-fix-design.md:4`

**Step 1: Update status**

Change line 4 from:
```markdown
**Status:** Approved
```

To:
```markdown
**Status:** Implemented ✅
```

**Step 2: Add implementation commit references**

At the end of the document, update the "Previous fixes" section with actual commit hashes:

```markdown
## References

- Polymarket CLOB WebSocket Docs: https://docs.polymarket.com/developers/CLOB/websocket/wss-overview
- Implementation commits:
  - Token ID filtering: [commit hash from Task 1]
  - Enhanced logging: [commit hash from Task 2]
  - Atomic state updates: [commit hash from Task 3]
```

**Step 3: Commit design document update**

```bash
git add docs/plans/2026-02-17-websocket-filtering-fix-design.md
git commit -m "docs: mark WebSocket filtering fix as implemented

Update design document status and add implementation commit
references for future reference.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 8: Push to Remote Repository

**Files:**
- Push: Git remote repository

**Step 1: Review all commits**

```bash
git log --oneline -8
```

Expected: See all commits from this implementation

**Step 2: Push to main**

```bash
git push origin main
```

Expected: Successfully pushed

**Step 3: Verify push**

```bash
git status
```

Expected: "Your branch is up to date with 'origin/main'"

---

## Rollback Plan

If issues occur after deployment:

```bash
cd /root/polymarket-scripts
git log --oneline -5  # Find commit hash before fixes
git revert <commit-hash>  # Revert the problematic commit
systemctl restart polymarket-bot
```

## Testing Summary

**Phase 1: Immediate (5 min)** - Verify filtering logs appear
**Phase 2: Transition (15-30 min)** - Verify market transition handling
**Phase 3: Alert (variable)** - Verify end-to-end alert delivery

## Success Criteria

- ✅ Bot only processes subscribed token IDs
- ✅ Market transitions handled cleanly
- ✅ Telegram alerts arrive when odds ≥70% sustained
- ✅ No stale odds warnings
- ✅ Bot odds match Polymarket odds

---

**Implementation Complete!** 🎉

The bot should now correctly detect and process only subscribed market odds, triggering alerts when sustained 70%+ odds are detected.
