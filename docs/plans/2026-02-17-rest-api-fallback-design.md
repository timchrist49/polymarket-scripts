# REST API Fallback for Odds Streaming

**Date:** 2026-02-17
**Status:** Approved
**Approach:** Hybrid WebSocket + REST Polling

## Problem Statement

Polymarket CLOB WebSocket only sends book messages when orderbook changes occur (not snapshots). Low-activity markets (15-minute BTC) may have no WebSocket messages for minutes, causing the bot to operate with stale odds.

## Solution: Hybrid Data Sources

Maintain both WebSocket connection (for real-time updates) and REST API polling (for reliability).

## Architecture

### Core Components

1. **WebSocket Stream**: Existing connection to `wss://ws-subscriptions-clob.polymarket.com/ws/market`
2. **REST Poller**: New background task calling `client.get_orderbook(token_id)` every 5 seconds
3. **Unified Data Path**: Both sources update the same `_current_odds` dict

### Data Flow

```
WebSocket Messages → _process_book_message() → _update_odds_from_orderbook()
                                                          ↓
                                                   _current_odds
                                                          ↑
REST API (5s) → _rest_polling_loop() → _update_odds_from_orderbook()
```

### Background Tasks

- **WebSocket Task**: `_connect_and_stream()` (existing)
- **REST Polling Task**: `_rest_polling_loop()` (new)
- **Market Monitor**: `_monitor_market_transitions()` (existing)

## Implementation Details

### New Method: `_rest_polling_loop()`

```python
async def _rest_polling_loop(self):
    """Poll REST API every 5 seconds for orderbook data."""
    while self._running:
        try:
            if not self._current_token_ids or not self._current_market_id:
                await asyncio.sleep(5)
                continue

            # Query orderbook for first token (YES token)
            token_id = self._current_token_ids[0].replace('0x', '')
            orderbook = self.client.get_orderbook(token_id, depth=1)

            if orderbook and orderbook.get('bids') and orderbook.get('asks'):
                await self._update_odds_from_orderbook(
                    bids=orderbook['bids'],
                    asks=orderbook['asks'],
                    source='REST'
                )
                logger.debug("📊 Updated odds from REST API")

        except Exception as e:
            logger.error("REST polling failed", error=str(e))

        await asyncio.sleep(5)
```

### Refactored Method: `_update_odds_from_orderbook()`

Extract odds calculation logic from `_process_book_message()` into shared method:

```python
async def _update_odds_from_orderbook(self, bids: list, asks: list, source: str):
    """
    Update odds from orderbook data (WebSocket or REST).

    Args:
        bids: List of [price, size] or {"price": ..., "size": ...}
        asks: List of asks (not used for YES odds)
        source: 'WebSocket' or 'REST' for logging
    """
    # Parse best bid (handle both dict and array formats)
    # Create WebSocketOddsSnapshot
    # Update _current_odds atomically with lock
```

### Modified Method: `_process_book_message()`

Delegate to shared method:

```python
async def _process_book_message(self, payload: dict):
    # ... validation ...
    bids = payload.get('bids', [])
    asks = payload.get('asks', [])

    await self._update_odds_from_orderbook(bids, asks, source='WebSocket')
```

### Task Launch in `start()`

```python
self._stream_task = asyncio.create_task(self._connect_and_stream())
self._rest_task = asyncio.create_task(self._rest_polling_loop())  # NEW
```

### Task Cleanup in `stop()`

```python
if self._rest_task:
    self._rest_task.cancel()
    try:
        await self._rest_task
    except asyncio.CancelledError:
        pass
```

## Error Handling

### REST API Failures
- **Action**: Log error, continue polling
- **Impact**: WebSocket continues if available

### WebSocket Disconnection
- **Action**: REST poller unaffected
- **Impact**: Seamless failover to REST

### Both Sources Fail
- **Action**: Staleness warning after 2 minutes
- **Impact**: Bot uses last known odds

### Orderbook Format Differences
- **Action**: Handle both `{"price": ...}` and `["price", ...]` formats
- **Impact**: Works with both WebSocket and REST responses

### Market Transition
- **Action**: Check `self._current_token_ids` before each poll
- **Impact**: Graceful skip until new market ready

## Testing Strategy

### Manual Verification

1. **REST Poller Active**: Check logs for "📊 Updated odds from REST API" every 5s
2. **WebSocket Priority**: Verify WebSocket messages processed when they arrive
3. **Failover**: Confirm REST continues during WebSocket silence
4. **Market Transition**: Verify both sources resubscribe correctly
5. **Data Consistency**: Compare WebSocket vs REST odds (should match)

### Success Criteria

✅ REST poller provides odds every 5 seconds
✅ WebSocket messages still processed when available
✅ No crashes or task failures
✅ Smooth market transitions
✅ `get_current_odds()` returns fresh data (<5s old)

### Monitoring

Log messages indicate data source:
- "📥 Raw book message" → WebSocket
- "📊 Updated odds from REST API" → REST

## Benefits

1. **Reliability**: Always have fresh odds regardless of WebSocket activity
2. **Low Latency**: WebSocket messages processed immediately when available
3. **Simplicity**: Both sources coexist without complex state management
4. **Resilience**: Continues operating if either source fails

## Trade-offs

- **Additional API Calls**: 12 REST calls per minute (within rate limits)
- **Slight Redundancy**: May update odds twice when WebSocket is active
- **Acceptable**: REST calls are cheap, redundancy ensures reliability
