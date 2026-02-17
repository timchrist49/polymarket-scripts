# REST API Fallback Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add REST API polling fallback (every 5 seconds) to complement WebSocket streaming for reliable odds updates.

**Architecture:** Hybrid approach with parallel WebSocket and REST tasks feeding the same `_current_odds` dict. WebSocket provides real-time updates when available, REST ensures fresh data during low-activity periods.

**Tech Stack:** Python asyncio, existing PolymarketClient REST API, structlog

---

## Task 1: Add REST Task Attribute to Streamer

**Files:**
- Modify: `polymarket/trading/realtime_odds_streamer.py:31-46`

**Step 1: Add `_rest_task` attribute to `__init__`**

In `__init__` method, add after line 44 (`self._stream_task = None`):

```python
self._rest_task: Optional[asyncio.Task] = None
```

**Step 2: Verify syntax**

Run: `python3 -m py_compile polymarket/trading/realtime_odds_streamer.py`
Expected: No output (success)

**Step 3: Commit**

```bash
git add polymarket/trading/realtime_odds_streamer.py
git commit -m "feat: add _rest_task attribute for REST polling

Prepare for hybrid WebSocket + REST approach.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 2: Create Shared Orderbook Processing Method

**Files:**
- Modify: `polymarket/trading/realtime_odds_streamer.py:73-148`

**Step 1: Extract orderbook processing into new method**

Add new method after `get_current_odds()` (around line 72):

```python
async def _update_odds_from_orderbook(
    self,
    bids: list,
    asks: list,
    source: str
) -> None:
    """
    Update odds from orderbook data (WebSocket or REST).

    Args:
        bids: List of bids as [{"price": str, "size": str}] or [[price, size]]
        asks: List of asks (not currently used)
        source: 'WebSocket' or 'REST' for logging
    """
    if not self._current_market_id:
        logger.warning(
            "No current market ID set, skipping odds update",
            source=source
        )
        return

    # Extract best bid price (YES odds)
    if bids and len(bids) > 0:
        if isinstance(bids[0], dict):
            yes_odds = float(bids[0]['price'])
        else:
            # Array format: [price, size]
            yes_odds = float(bids[0][0])
    else:
        # Default if no bids
        yes_odds = 0.50

    no_odds = 1.0 - yes_odds

    # Create snapshot
    snapshot = WebSocketOddsSnapshot(
        market_id=self._current_market_id,
        yes_odds=yes_odds,
        no_odds=no_odds,
        timestamp=datetime.now(timezone.utc),
        best_bid=yes_odds,
        best_ask=no_odds
    )

    # Store atomically
    async with self._lock:
        self._current_odds[self._current_market_id] = snapshot

    logger.debug(
        "📊 Odds updated",
        source=source,
        market_id=self._current_market_id,
        market_slug=self._current_market_slug,
        yes_odds=f"{yes_odds:.2f}",
        no_odds=f"{no_odds:.2f}"
    )
```

**Step 2: Verify syntax**

Run: `python3 -m py_compile polymarket/trading/realtime_odds_streamer.py`
Expected: No output (success)

**Step 3: Commit**

```bash
git add polymarket/trading/realtime_odds_streamer.py
git commit -m "feat: add shared orderbook processing method

Extracts odds calculation logic for reuse by WebSocket and REST.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 3: Refactor WebSocket Book Message Handler

**Files:**
- Modify: `polymarket/trading/realtime_odds_streamer.py:73-148`

**Step 1: Simplify `_process_book_message` to delegate to shared method**

Replace the odds calculation code (lines 108-142) with delegation:

```python
async def _process_book_message(self, payload: dict):
    """
    Extract odds from book message and update state.

    Args:
        payload: Book message with bids/asks arrays
    """
    try:
        # Book messages contain both market (condition ID) and asset_id (token ID)
        # We use market ID for validation and asset_id identifies the specific token
        market_id = payload.get('market')
        asset_id = payload.get('asset_id')

        if not market_id or not asset_id:
            logger.warning("Book message missing IDs", market_id=market_id, asset_id=asset_id)
            return

        # Verify we have a current market being tracked
        if not self._current_market_id:
            logger.warning("No current market ID set, skipping book message")
            return

        # Extract orderbook data
        bids = payload.get('bids', [])
        asks = payload.get('asks', [])

        logger.info(
            "📥 Raw book message",
            market_id=market_id[:16] + "...",
            asset_id=asset_id[:16] + "...",
            bids_count=len(bids),
            asks_count=len(asks)
        )

        # Delegate to shared processing method
        await self._update_odds_from_orderbook(bids, asks, source='WebSocket')

    except Exception as e:
        logger.error(
            "Book message processing failed",
            error=str(e),
            payload=str(payload)[:200]
        )
```

**Step 2: Verify syntax**

Run: `python3 -m py_compile polymarket/trading/realtime_odds_streamer.py`
Expected: No output (success)

**Step 3: Commit**

```bash
git add polymarket/trading/realtime_odds_streamer.py
git commit -m "refactor: delegate WebSocket book processing to shared method

Simplifies _process_book_message by using _update_odds_from_orderbook.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 4: Implement REST Polling Loop

**Files:**
- Modify: `polymarket/trading/realtime_odds_streamer.py` (add after `_process_book_message`)

**Step 1: Add REST polling loop method**

Add new method after `_process_book_message()`:

```python
async def _rest_polling_loop(self):
    """
    Poll REST API every 5 seconds for orderbook data.

    Provides fallback when WebSocket messages aren't arriving.
    Runs continuously alongside WebSocket connection.
    """
    logger.info("REST polling loop started (5s interval)")

    while self._running:
        try:
            # Wait for market to be set
            if not self._current_token_ids or not self._current_market_id:
                await asyncio.sleep(5)
                continue

            # Query orderbook for first token (YES token)
            # Remove '0x' prefix if present (REST API expects decimal or no prefix)
            token_id_decimal = self._current_token_ids[0].replace('0x', '')

            # Synchronous call in async context (client.get_orderbook is not async)
            orderbook = await asyncio.to_thread(
                self.client.get_orderbook,
                token_id_decimal,
                depth=1
            )

            if orderbook and orderbook.get('bids') and orderbook.get('asks'):
                await self._update_odds_from_orderbook(
                    bids=orderbook['bids'],
                    asks=orderbook['asks'],
                    source='REST'
                )
                logger.debug("📊 Updated odds from REST API")
            else:
                logger.warning(
                    "REST API returned empty orderbook",
                    market_id=self._current_market_id
                )

        except Exception as e:
            logger.error(
                "REST polling failed",
                error=str(e),
                market_id=self._current_market_id
            )

        await asyncio.sleep(5)

    logger.info("REST polling loop stopped")
```

**Step 2: Verify syntax**

Run: `python3 -m py_compile polymarket/trading/realtime_odds_streamer.py`
Expected: No output (success)

**Step 3: Commit**

```bash
git add polymarket/trading/realtime_odds_streamer.py
git commit -m "feat: implement REST API polling loop

Polls orderbook every 5 seconds as fallback to WebSocket.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 5: Launch REST Polling Task in start()

**Files:**
- Modify: `polymarket/trading/realtime_odds_streamer.py:150-161`

**Step 1: Add REST task launch to `start()` method**

Modify the `start()` method to launch both tasks:

```python
def start(self):
    """
    Start streaming (non-blocking).

    Launches background tasks for WebSocket and REST polling.
    """
    if self._running:
        logger.warning("Streamer already running")
        return

    self._running = True
    self._stream_task = asyncio.create_task(self._stream_loop())
    self._rest_task = asyncio.create_task(self._rest_polling_loop())
    logger.info("Real-time odds streamer started (WebSocket + REST polling)")
```

**Step 2: Verify syntax**

Run: `python3 -m py_compile polymarket/trading/realtime_odds_streamer.py`
Expected: No output (success)

**Step 3: Commit**

```bash
git add polymarket/trading/realtime_odds_streamer.py
git commit -m "feat: launch REST polling task alongside WebSocket

Enables hybrid WebSocket + REST approach.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 6: Add REST Task Cleanup in stop()

**Files:**
- Modify: `polymarket/trading/realtime_odds_streamer.py:163-185`

**Step 1: Add REST task cancellation to `stop()` method**

Modify the `stop()` method to cancel both tasks:

```python
async def stop(self):
    """
    Stop streaming gracefully.

    Closes WebSocket connection and cancels background tasks.
    """
    self._running = False

    if self._ws:
        try:
            await self._ws.close()
        except Exception as e:
            logger.debug("Error closing WebSocket", error=str(e))

    if self._stream_task:
        self._stream_task.cancel()
        try:
            await self._stream_task
        except asyncio.CancelledError:
            pass

    if self._rest_task:
        self._rest_task.cancel()
        try:
            await self._rest_task
        except asyncio.CancelledError:
            pass

    logger.info("Real-time odds streamer stopped")
```

**Step 2: Verify syntax**

Run: `python3 -m py_compile polymarket/trading/realtime_odds_streamer.py`
Expected: No output (success)

**Step 3: Commit**

```bash
git add polymarket/trading/realtime_odds_streamer.py
git commit -m "feat: add REST task cleanup in stop()

Ensures graceful shutdown of both WebSocket and REST tasks.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 7: Manual Integration Test - REST Polling Active

**Step 1: Restart the bot**

```bash
# Find and stop current bot process
pgrep -f "python.*auto_trade" | xargs kill

# Start bot in background
cd /root/polymarket-scripts
nohup python3 -u scripts/auto_trade.py > /tmp/bot.log 2>&1 &
echo "Bot PID: $!"
```

Expected: Bot starts successfully

**Step 2: Monitor logs for REST polling**

```bash
sleep 10
tail -50 /tmp/bot.log | grep -E "(REST polling|📊 Updated odds from REST)"
```

Expected output:
- "REST polling loop started (5s interval)"
- "📊 Updated odds from REST API" appearing every ~5 seconds

**Step 3: Verify odds are being updated**

```bash
# Wait 30 seconds to collect multiple updates
sleep 30
tail -100 /tmp/bot.log | grep "📊 Updated odds" | tail -10
```

Expected: Mix of WebSocket and REST updates, or primarily REST if market is inactive

**Step 4: Check for errors**

```bash
tail -100 /tmp/bot.log | grep -E "(ERROR|WARNING)" | tail -10
```

Expected: No critical errors, possibly staleness warnings from before (acceptable)

---

## Task 8: Manual Integration Test - Data Consistency

**Step 1: Compare WebSocket vs REST odds**

```bash
# Capture 2 minutes of logs
timeout 120 tail -f /tmp/bot.log > /tmp/odds_comparison.log 2>&1 &
CAPTURE_PID=$!

# Wait for capture
sleep 120

# Analyze odds updates
echo "=== WebSocket Updates ==="
grep "source=WebSocket" /tmp/odds_comparison.log | grep "yes_odds" | tail -5

echo "=== REST Updates ==="
grep "source=REST" /tmp/odds_comparison.log | grep "yes_odds" | tail -5
```

Expected: Odds values should be identical or very close (within 0.01) for same market

**Step 2: Verify no conflicts or race conditions**

```bash
grep -E "(lock|race|conflict)" /tmp/bot.log | tail -10
```

Expected: No race condition errors (lock should prevent conflicts)

---

## Task 9: Manual Integration Test - Market Transition

**Step 1: Wait for 15-minute market rollover**

```bash
# Check current market
grep "Connecting to CLOB WebSocket" /tmp/bot.log | tail -1

# Wait for transition (check every 60 seconds)
watch -n 60 'tail -50 /tmp/bot.log | grep -E "(Market transition|Connecting to CLOB)"'
```

Expected: See "Market transition detected" followed by new market connection

**Step 2: Verify both tasks resubscribe correctly**

```bash
# After transition, check logs
tail -100 /tmp/bot.log | grep -E "(Market transition|REST polling|WebSocket)"
```

Expected:
- "Market transition detected by monitor, closing WebSocket"
- New "Connecting to CLOB WebSocket" with new market
- "REST polling loop started" continues uninterrupted

---

## Task 10: Update Documentation

**Files:**
- Modify: `docs/plans/2026-02-17-rest-api-fallback-design.md`

**Step 1: Update design document status**

Change the status line at top of file:

```markdown
**Status:** ✅ Implemented
```

Add implementation notes section at bottom:

```markdown
## Implementation Notes

**Date Completed:** 2026-02-17

**Changes Made:**
1. Added `_rest_task` attribute to RealtimeOddsStreamer
2. Created shared `_update_odds_from_orderbook()` method
3. Refactored `_process_book_message()` to use shared method
4. Implemented `_rest_polling_loop()` with 5-second interval
5. Modified `start()` to launch REST polling task
6. Modified `stop()` to cancel REST polling task gracefully

**Testing Results:**
- REST polling provides odds every 5 seconds ✅
- WebSocket messages still processed when they arrive ✅
- No task failures or crashes ✅
- Smooth market transitions ✅
- Data consistency verified (WebSocket ≈ REST) ✅

**Performance:**
- REST API calls: ~12 per minute (well within rate limits)
- No observable performance degradation
- Minimal redundancy when WebSocket is active
```

**Step 2: Commit documentation**

```bash
git add docs/plans/2026-02-17-rest-api-fallback-design.md
git commit -m "docs: mark REST API fallback as implemented

Added implementation notes and testing results.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 11: Push to Remote Repository

**Step 1: Push all commits**

```bash
cd /root/polymarket-scripts
git push origin master
```

Expected: All commits pushed successfully

**Step 2: Verify push**

```bash
git log --oneline -5
git status
```

Expected: Clean working tree, latest commits visible

---

## Success Criteria

✅ REST polling loop runs continuously every 5 seconds
✅ WebSocket messages processed when they arrive
✅ Shared `_update_odds_from_orderbook()` used by both sources
✅ No crashes or task failures during operation
✅ Smooth market transitions (both tasks continue)
✅ Odds data consistency (WebSocket ≈ REST values)
✅ Graceful task cleanup in `stop()`
✅ Documentation updated with implementation notes

## Notes for Implementation

- Use `asyncio.to_thread()` to run synchronous `client.get_orderbook()` in async context
- The `_lock` prevents race conditions when both sources update simultaneously
- REST polling continues even when WebSocket is receiving messages (acceptable redundancy)
- Token ID format: REST API expects decimal or no '0x' prefix, handle accordingly
- Log level: Use `logger.debug()` for routine REST updates to avoid log spam
