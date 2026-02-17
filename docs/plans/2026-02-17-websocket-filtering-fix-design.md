# WebSocket Message Filtering Fix

**Date:** 2026-02-17
**Status:** Approved
**Priority:** Critical (Bot non-functional since deployment)

## Problem Statement

Polymarket trading bot not triggering alerts despite odds being above 70%. Investigation revealed bot processes ALL WebSocket book messages instead of only subscribed tokens, resulting in incorrect odds data.

## Root Cause Analysis

**Confirmed Issue**: Polymarket's CLOB WebSocket broadcasts ALL market messages regardless of subscription.

**Evidence:**
- Bot subscribes to token IDs: `0xd5a0140b83cb55c0afd77625aad28056288ab300a8c6f12b0e1ae27bf210466d` and `0x32a37e45d6b4cfba7260a50e2c53f2eb4506392588d5e0c3810a047a798cf494`
- Bot receives book messages for: `0x3f3eab02c4dc40...` (DIFFERENT MARKET!)
- Pattern confirmed across all market transitions

**Impact**: Bot has been non-functional since deployment because it's never seeing correct market odds. Dictionary lookups succeed (after Fix #2), but contain odds from wrong markets.

## Three-Pronged Fix Architecture

### Fix #1: Client-Side Token Filtering (Defense in Depth)

Add filtering in `_handle_single_message()` to only process messages for subscribed token IDs:

```python
async def _handle_single_message(self, data: dict):
    event_type = data.get('event_type')

    if event_type == 'book':
        token_id = data.get('market')  # hex string from message

        # FILTER: Only process if token matches our subscription
        if token_id and self._current_token_ids:
            if token_id not in self._current_token_ids:
                logger.debug("Ignoring book for unsubscribed token",
                           token_id=token_id[:16])
                return

        await self._process_book_message(data)
```

**Rationale**: Provides immediate fix regardless of server behavior. Defensive programming principle.

### Fix #2: Subscription Verification

Add enhanced logging to verify subscription confirmation and format:

```python
logger.info(
    "🔔 Subscription sent",
    market_id=market.id,
    token_ids=[t[:16] + "..." for t in token_ids]
)
```

This helps us understand WebSocket server behavior and confirm our subscription format is correct.

### Fix #3: Race Condition Protection

Ensure token ID list updates atomically during market transitions:

```python
async with self._lock:
    self._current_market_id = market.id
    self._current_market_slug = market.slug
    self._current_token_ids = token_ids
```

Already using a lock, but ensure it's applied consistently across all state updates.

## Implementation Details

### File: `polymarket/trading/realtime_odds_streamer.py`

**Change 1: Add Token Filtering (lines 333-343)**

Replace current `_handle_single_message()`:

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
```

**Change 2: Track Subscription State (lines 260-280)**

Add subscription confirmation tracking in `_connect_and_stream()`:

```python
async with websockets.connect(...) as ws:
    self._ws = ws

    # Send subscription
    subscribe_msg = {
        "assets_ids": token_ids,
        "type": "market"
    }
    await ws.send(json.dumps(subscribe_msg))

    logger.info(
        "🔔 Subscription sent",
        market_id=market.id,
        token_ids=[t[:16] + "..." for t in token_ids]
    )
```

**Change 3: Atomic State Updates (lines 269-271)**

Ensure all state updates happen under lock:

```python
async with self._lock:
    self._current_market_id = market.id
    self._current_market_slug = market.slug
    self._current_token_ids = token_ids
```

## Testing Strategy

### Phase 1: Immediate Verification (First 5 Minutes)

After implementing the fix, monitor logs for:

```bash
tail -f /root/polymarket-scripts/logs/bot.log | grep -E "(Ignoring book|Odds updated from book|Subscription sent)"
```

**Success Criteria:**
- See "Ignoring book message for unsubscribed token" for other markets
- See "Odds updated from book" ONLY for subscribed token IDs
- Token IDs in "Odds updated" match token IDs in "Subscription sent"

### Phase 2: Market Transition Test (15-30 Minutes)

Wait for one market transition cycle to verify:

1. **Old market messages filtered**: After resubscription, should ignore messages for previous market's tokens
2. **New market messages processed**: Should process messages for new market's tokens immediately
3. **No race condition**: No processing of wrong market during transition window

**Expected Log Sequence:**
```
21:45:00 - Market transition detected
21:45:01 - Closing connection to resubscribe
21:45:02 - Subscription sent (new tokens)
21:45:03 - Ignoring book for [old token]
21:45:04 - Odds updated from book [new token]
```

### Phase 3: Alert Verification (When Odds Spike)

When you see 70%+ odds on Polymarket:

1. **Verify bot sees same odds**: Check logs for matching YES/NO odds
2. **Verify sustained detection**: Should see 5 consecutive checks at 70%+
3. **Verify alert sent**: Should receive Telegram notification

**Success = Telegram alert arrives within 30 seconds of sustained 70%+ odds**

## Rollback Plan

If fix causes issues:

```bash
cd /root/polymarket-scripts
git revert HEAD
systemctl restart polymarket-bot
```

## References

- Polymarket CLOB WebSocket Docs: https://docs.polymarket.com/developers/CLOB/websocket/wss-overview
- Previous fixes:
  - Fix #1: Stale market ID cache (commit: TBD)
  - Fix #2: Token ID vs Market ID mismatch (commit: TBD)
