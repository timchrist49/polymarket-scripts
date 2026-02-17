# Independent Market Transition Monitor Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix critical bug where market transition check only runs when WebSocket messages arrive, causing bot to get stuck on dead markets.

**Architecture:** Extract market transition check from message loop into independent `asyncio.create_task()` that runs every 60 seconds. Use try/finally for clean task cancellation.

**Tech Stack:** Python 3.11, asyncio, websockets, pytest-asyncio, structlog

---

## Task 1: Add Monitor Method Implementation

**Files:**
- Modify: `/root/polymarket-scripts/polymarket/trading/realtime_odds_streamer.py:231-254`

**Step 1: Read current _check_market_transition implementation**

```bash
cd /root/polymarket-scripts
grep -A 15 "async def _check_market_transition" polymarket/trading/realtime_odds_streamer.py
```

Expected: See the existing method that checks if market changed

**Step 2: Add new _monitor_market_transitions method**

Insert after `_check_market_transition` method (around line 254):

```python
    async def _monitor_market_transitions(self, ws):
        """
        Independently monitor for market transitions.

        Runs concurrently with message reception loop.
        Closes WebSocket when market changes to trigger resubscription.

        Args:
            ws: WebSocket connection to close if transition detected
        """
        while self._running:
            await asyncio.sleep(60)  # Check every minute

            try:
                if await self._check_market_transition():
                    logger.info("Market transition detected by monitor, closing WebSocket")
                    await ws.close()
                    break
            except Exception as e:
                logger.error("Market transition check failed in monitor", error=str(e))
```

**Step 3: Verify syntax**

```bash
python3 -m py_compile polymarket/trading/realtime_odds_streamer.py
```

Expected: No syntax errors

**Step 4: Commit**

```bash
git add polymarket/trading/realtime_odds_streamer.py
git commit -m "feat: add independent market transition monitor method

Add _monitor_market_transitions() that runs concurrently with
message reception loop, checking for market transitions every 60s.

This fixes critical bug where transition check only ran when
messages arrived, causing bot to get stuck on dead markets with
empty order books.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 2: Remove Transition Check from Message Loop

**Files:**
- Modify: `/root/polymarket-scripts/polymarket/trading/realtime_odds_streamer.py:337-346`

**Step 1: Find and remove old transition check**

Locate these lines in `_connect_and_stream` method (inside the `async for message in ws:` loop):

```python
                # Periodic market transition check
                now = asyncio.get_event_loop().time()
                if now - last_market_check > MARKET_CHECK_INTERVAL:
                    last_market_check = now

                    if await self._check_market_transition():
                        # Market changed! Close connection to trigger resubscription
                        logger.info("Closing connection to resubscribe to new market")
                        await ws.close()
                        break
```

**Delete these lines entirely** (they're now handled by monitor task).

Also remove the initialization of `last_market_check` (around line 322):

```python
            # Track last market check time
            last_market_check = asyncio.get_event_loop().time()
            MARKET_CHECK_INTERVAL = 60  # seconds
```

**Step 2: Verify syntax**

```bash
python3 -m py_compile polymarket/trading/realtime_odds_streamer.py
```

Expected: No syntax errors

**Step 3: Commit**

```bash
git add polymarket/trading/realtime_odds_streamer.py
git commit -m "refactor: remove market transition check from message loop

Remove transition check that only ran when messages arrived.
This logic is now handled by independent monitor task.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 3: Launch Monitor Task in _connect_and_stream

**Files:**
- Modify: `/root/polymarket-scripts/polymarket/trading/realtime_odds_streamer.py:~310-330`

**Step 1: Find subscription confirmation log**

Look for this line (around line 319):

```python
            logger.info(
                "🔔 Subscription sent",
                market_id=market.id,
                market_slug=market.slug,
                token_count=len(token_ids),
                token_ids=[t[:16] + "..." for t in token_ids]
            )
```

**Step 2: Add monitor task launch after subscription**

Insert immediately after the subscription log:

```python
            # Launch independent market transition monitor
            monitor_task = asyncio.create_task(
                self._monitor_market_transitions(ws)
            )
            logger.info("Market transition monitor started")
```

**Step 3: Wrap message loop in try/finally**

Find the message processing loop (starts around line 325):

```python
            # Process messages until disconnected
            async for message in ws:
                # ... message processing ...
```

Wrap the ENTIRE loop in try/finally:

```python
            try:
                # Process messages until disconnected
                async for message in ws:
                    # DEBUG: Log every raw message received
                    logger.info(
                        "📨 RAW WebSocket message received",
                        length=len(message),
                        content=message[:200] if len(message) <= 200 else message[:200] + "..."
                    )

                    if not self._running:
                        break

                    try:
                        data = json.loads(message)

                        # DEBUG: Log parsed data structure
                        logger.info(
                            "📦 Parsed WebSocket data",
                            data_type=type(data).__name__,
                            data_preview=str(data)[:200]
                        )

                        # Handle both single message and array of messages
                        if isinstance(data, list):
                            # Array of messages - process each
                            logger.info(f"Processing array of {len(data)} messages")
                            for msg in data:
                                if isinstance(msg, dict):
                                    await self._handle_single_message(msg)
                        elif isinstance(data, dict):
                            # Single message
                            logger.info("Processing single dict message")
                            await self._handle_single_message(data)

                    except json.JSONDecodeError:
                        logger.warning("Invalid JSON message", message=message[:100])
                    except Exception as e:
                        logger.error("Message processing error", error=str(e))

                # Loop exited - log why
                logger.warning(
                    "WebSocket message loop exited",
                    running=self._running,
                    market_id=self._current_market_id
                )

            finally:
                # Clean up monitor task
                logger.info("Cancelling market transition monitor")
                monitor_task.cancel()
                try:
                    await monitor_task
                except asyncio.CancelledError:
                    logger.info("Market transition monitor cancelled successfully")
```

**Step 4: Verify syntax**

```bash
python3 -m py_compile polymarket/trading/realtime_odds_streamer.py
```

Expected: No syntax errors

**Step 5: Commit**

```bash
git add polymarket/trading/realtime_odds_streamer.py
git commit -m "feat: launch independent market transition monitor

Launch monitor task concurrently with message loop. Use try/finally
to ensure clean cancellation when connection closes.

Monitor now runs every 60s regardless of message activity, fixing
bug where bot got stuck on dead markets.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 4: Manual Integration Test - Monitor Logs

**Files:**
- Monitor: `/root/polymarket-scripts/logs/bot.log`

**Step 1: Stop the bot**

```bash
cd /root/polymarket-scripts
./start_bot.sh stop
```

Expected: Bot stopped

**Step 2: Clear old logs**

```bash
> logs/bot.log
> logs/bot_daemon.log
```

Expected: Empty log files

**Step 3: Start the bot**

```bash
./start_bot.sh start
```

Expected: Bot started successfully

**Step 4: Monitor for monitor task startup (30 seconds)**

```bash
tail -f logs/bot_daemon.log | grep -E "(Market transition monitor|📤 Sent|🔔 Subscription)" | head -10
```

**Expected output:**
```
📤 Sent handshake | message='{"assets_ids": [], "type": "market"}'
📤 Sent subscription | message='{"operation": "subscribe", ...}'
🔔 Subscription sent | market_id=... | market_slug=...
Market transition monitor started
```

**Success Criteria:**
- ✅ See "Market transition monitor started" after subscription
- ✅ No errors about undefined methods or syntax issues

**Step 5: Document startup verification**

If successful, proceed to Task 5.
If issues found, debug before continuing.

---

## Task 5: Manual Integration Test - Market Transition

**Files:**
- Monitor: `/root/polymarket-scripts/logs/bot_daemon.log`

**Step 1: Wait for next market transition**

Markets transition every ~15 minutes at :00, :15, :30, :45 past the hour.

Check current time and wait for next transition:

```bash
date -u
```

If it's 07:32, wait ~13 minutes for 07:45 transition.

**Step 2: Monitor logs during transition window**

Start monitoring 2 minutes before expected transition:

```bash
tail -f logs/bot_daemon.log | grep -E "(Market transition|monitor|Closing|📤 Sent|🔔 Subscription)"
```

**Expected log sequence:**
```
[07:44:xx] Market transition detected by monitor, closing WebSocket
[07:44:xx] Cancelling market transition monitor
[07:44:xx] Market transition monitor cancelled successfully
[07:44:xx] WebSocket message loop exited
[07:44:xx] Connecting to CLOB WebSocket | market_slug=btc-updown-15m-1771314300
[07:44:xx] 📤 Sent handshake
[07:44:xx] 📤 Sent subscription
[07:44:xx] 🔔 Subscription sent | market_slug=btc-updown-15m-1771314300
[07:44:xx] Market transition monitor started
```

**Success Criteria:**
- ✅ Monitor detects transition within 60 seconds of market change
- ✅ WebSocket closes cleanly
- ✅ Monitor task cancels without errors
- ✅ Bot resubscribes to new market
- ✅ New monitor task starts

**Step 3: Document transition verification**

If successful, proceed to Task 6.
If transition not detected, investigate timing or logic issues.

---

## Task 6: Manual Integration Test - Book Messages on Fresh Market

**Files:**
- Monitor: `/root/polymarket-scripts/logs/bot_daemon.log`

**Step 1: Wait 2 minutes after transition for liquidity**

Fresh markets take 1-2 minutes to get initial orders placed.

```bash
sleep 120
```

**Step 2: Check for book messages**

```bash
tail -100 logs/bot_daemon.log | grep -E "(📨 RAW|📦 Parsed|📊 Odds updated|Ignoring book)"
```

**Expected output:**
```
📨 RAW WebSocket message received | length=... | content='{"event_type":"book",...'
📦 Parsed WebSocket data | data_type=dict | data_preview='{"event_type": "book",...'
Processing single dict message
🔍 WebSocket message received | event_type=book
📊 Odds updated from book | market_id=... | yes_odds=0.52 | no_odds=0.48
```

**Success Criteria:**
- ✅ Bot receives book messages (not just empty array)
- ✅ Odds are extracted and stored
- ✅ Market ID matches current market

**Step 3: Verify with REST API**

```bash
python3 -c "
from polymarket.client import PolymarketClient
client = PolymarketClient()
market = client.discover_btc_15min_market()
print(f'Current market: {market.slug}')
print(f'Market ends: {market.end_date}')
"
```

Compare market slug in logs with current market from API.

**Step 4: Document book message verification**

If successful, all tests pass! Proceed to Task 7.
If no book messages, check order book via REST API.

---

## Task 7: Update Design Document Status

**Files:**
- Modify: `/root/polymarket-scripts/docs/plans/2026-02-17-market-transition-independent-check.md:4`

**Step 1: Update status**

Change line 4 from:
```markdown
**Status:** Design
```

To:
```markdown
**Status:** Implemented ✅
```

**Step 2: Add implementation notes**

At the end of the document, add:

```markdown
## Implementation Notes

**Implemented:** 2026-02-17

**Key Changes:**
1. Added `_monitor_market_transitions(ws)` method that runs every 60s
2. Removed market transition check from message loop
3. Launch monitor as concurrent task with try/finally cleanup
4. Monitor closes WebSocket when transition detected

**Testing Results:**
- ✅ Monitor task starts successfully
- ✅ Transition detected within 60s of market change
- ✅ WebSocket closes and resubscribes cleanly
- ✅ Book messages received on fresh markets

**Commits:**
- feat: add independent market transition monitor method
- refactor: remove market transition check from message loop
- feat: launch independent market transition monitor
```

**Step 3: Commit design document update**

```bash
git add docs/plans/2026-02-17-market-transition-independent-check.md
git commit -m "docs: mark independent market monitor as implemented

Update design document with implementation status and notes.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 8: Push to Remote Repository

**Files:**
- Push: Git remote repository

**Step 1: Review all commits**

```bash
git log --oneline -5
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
git log --oneline -5  # Find commit hash before changes
git revert <commit-hash>  # Revert the problematic commit
./start_bot.sh restart
```

## Testing Summary

**Phase 1: Startup (1 min)** - Verify monitor task launches
**Phase 2: Transition (15-60 min)** - Verify transition detection
**Phase 3: Book Messages (2 min)** - Verify fresh market has data

## Success Criteria

- ✅ Monitor task runs independently every 60 seconds
- ✅ Transition detected even when no messages arrive
- ✅ WebSocket closes and resubscribes on transition
- ✅ Bot tracks markets across transitions successfully
- ✅ Book messages arrive on fresh markets with liquidity

---

**Implementation Complete!** 🎉

The bot should now successfully track market transitions even when order books are empty, fixing the critical bug that prevented it from ever seeing high-odds opportunities.
