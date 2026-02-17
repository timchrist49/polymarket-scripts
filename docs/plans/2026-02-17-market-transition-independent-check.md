# Independent Market Transition Check Fix

**Date:** 2026-02-17
**Status:** Implemented ✅
**Implementation Date:** 2026-02-17
**Test Results:** Verified in production - bot successfully auto-transitioned through 3 consecutive dead markets
**Priority:** CRITICAL (Bot cannot function without this)

## Problem Statement

WebSocket market transition check is inside the message reception loop, causing it to only run when messages arrive. When markets near expiration have empty order books (no trading activity), NO messages arrive, so transition check NEVER runs. Bot gets stuck subscribed to dead markets forever.

## Root Cause

**File:** `polymarket/trading/realtime_odds_streamer.py`
**Lines:** 337-346

```python
async for message in ws:
    # ... message processing ...

    # ❌ BROKEN: This only runs when messages arrive!
    if now - last_market_check > MARKET_CHECK_INTERVAL:
        if await self._check_market_transition():
            await ws.close()
            break
```

**Impact:**
- Bot subscribes to market at 07:15 (ends 07:30)
- Market goes quiet at 07:25 (order book empties)
- No more messages arrive
- Transition check never runs
- Bot stays on dead market forever
- New market at 07:30 has liquidity but bot misses it

## Solution: Concurrent Market Monitor Task

Use `asyncio.create_task()` to run market transition check independently of message reception.

### Architecture

```python
async def _connect_and_stream(self):
    async with websockets.connect(...) as ws:
        # Send subscription
        ...

        # Create independent market monitor task
        monitor_task = asyncio.create_task(
            self._monitor_market_transitions(ws)
        )

        try:
            # Process messages (may be infrequent or never)
            async for message in ws:
                ...
        finally:
            # Clean up monitor task
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass
```

### New Method

```python
async def _monitor_market_transitions(self, ws: ClientConnection):
    """
    Independently monitor for market transitions.

    Runs concurrently with message reception loop.
    Closes WebSocket when market changes to trigger resubscription.
    """
    while self._running:
        await asyncio.sleep(60)  # Check every minute

        try:
            if await self._check_market_transition():
                logger.info("Market transition detected, closing WebSocket")
                await ws.close()
                break
        except Exception as e:
            logger.error("Market transition check failed", error=str(e))
```

## Implementation Plan

### Step 1: Extract Monitor Method

Create new method `_monitor_market_transitions(ws)` that:
1. Loops every 60 seconds
2. Calls `_check_market_transition()`
3. Closes WebSocket if market changed
4. Handles exceptions gracefully

### Step 2: Update _connect_and_stream

1. Remove market transition check from message loop (lines 337-346)
2. Create monitor task before message loop
3. Add try/finally to clean up monitor task
4. Ensure task cancellation on exit

### Step 3: Test Transition

1. Start bot when market is 2 minutes from expiration
2. Verify WebSocket connects to old market
3. Wait for market to transition
4. Verify monitor detects transition and closes WebSocket
5. Verify bot resubscribes to new market
6. Verify book messages arrive on new market

## Success Criteria

- ✅ Monitor task runs independently every 60 seconds
- ✅ Transition detected even when no messages arrive
- ✅ WebSocket closes and resubscribes on transition
- ✅ Bot successfully tracks markets across transitions
- ✅ Order book messages arrive on fresh markets

## Alternative Considered: asyncio.wait_for()

Could use `asyncio.wait_for(ws.recv(), timeout=60)` to timeout message reception and check transitions. **Rejected because:**
- More complex error handling (timeouts are normal, not errors)
- Mixes timeout logic with message processing
- Harder to test and reason about
- Separate task is clearer intent

## Files to Modify

- `/root/polymarket-scripts/polymarket/trading/realtime_odds_streamer.py`

## Estimated Time

30 minutes (simple change, critical impact)
