# Real-Time WebSocket Odds Streaming - Design Document

**Date:** 2026-02-16
**Status:** Approved
**Author:** Claude + User

## Problem Statement

The bot currently polls Polymarket's REST API for odds every 60 seconds, causing it to miss brief price spikes that appear on the Polymarket website (which uses WebSocket). Users report seeing 70%+ odds on the website that the bot never detects because they normalize before the next polling interval.

**Current State:**
- Polling interval: 60 seconds
- Latency: 0-60 seconds behind website
- Miss rate: ~79% of opportunities (155 out of 197 cycles skipped due to balanced odds)

**Target State:**
- Real-time WebSocket streaming
- Latency: < 1 second
- Catch rate: ≥90% of 70%+ spikes visible on website

## Design Overview

### Architecture

Create a new `RealtimeOddsStreamer` service that maintains a persistent WebSocket connection to Polymarket CLOB, providing zero-latency odds updates to the trading loop.

**Key Components:**

1. **RealtimeOddsStreamer** (`polymarket/trading/realtime_odds_streamer.py`)
   - Persistent WebSocket connection to `wss://ws-subscriptions-clob.polymarket.com/ws/market`
   - Subscribes to current BTC 15m market token IDs
   - Processes `book` messages in real-time
   - Exposes `get_current_odds(market_id)` method for instant access
   - Handles reconnection with exponential backoff

2. **Integration Points:**
   - `auto_trade.py`: Replace `odds_poller.get_odds()` with `realtime_streamer.get_current_odds()`
   - Background task: Launch streamer on bot startup alongside other services
   - Graceful shutdown: Stop streamer when bot stops

3. **Data Flow:**
   ```
   Bot discovers BTC 15m market
   └─> Streamer subscribes to token IDs via WebSocket
       └─> Book messages arrive (real-time)
           └─> Extract best_bid/ask → update current odds
               └─> Trading loop queries odds (instant, no API call)
   ```

4. **Migration Strategy:**
   - Keep `MarketOddsPoller` in codebase initially (not used)
   - Monitor real-time streamer for 24-48 hours
   - Remove old poller after validation
   - Fallback: Can revert to poller if needed

## Component Details

### RealtimeOddsStreamer Class

```python
class RealtimeOddsStreamer:
    """Persistent WebSocket streamer for real-time market odds."""

    # Constants
    WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
    BACKOFF_DELAYS = [1, 2, 4, 8, 16, 32, 60]  # Exponential backoff (seconds)

    # State
    _current_odds: dict[str, OddsSnapshot] = {}  # market_id -> odds
    _current_market_id: str | None = None
    _current_token_ids: list[str] | None = None
    _ws: WebSocketClientProtocol | None = None
    _reconnect_task: asyncio.Task | None = None
    _lock: asyncio.Lock  # Thread-safe state access
```

### Core Methods

1. **`start()`** - Launch background streaming task
   - Discovers current market
   - Connects WebSocket
   - Starts message processing loop
   - Returns immediately (non-blocking)

2. **`get_current_odds(market_id: str) -> OddsSnapshot | None`**
   - Thread-safe read from `_current_odds`
   - Returns latest odds snapshot with timestamp
   - Returns None if market not subscribed or no data yet

3. **`_stream_loop()`** - Main streaming coroutine
   - Connects WebSocket with exponential backoff
   - Subscribes to current market
   - Processes incoming messages
   - Handles disconnects and resubscribes

4. **`_process_book_message(payload)`** - Extract odds from book
   - Parses `book` message payload
   - Extracts best_bid (YES odds)
   - Calculates NO odds (1.0 - YES)
   - Updates `_current_odds` with timestamp

5. **`_resubscribe_if_market_changed()`** - Market transition handling
   - Checks if current market changed (every 60s)
   - Unsubscribes from old market
   - Subscribes to new market
   - Seamless transition without connection drop

6. **`stop()`** - Graceful shutdown
   - Close WebSocket connection
   - Cancel reconnection task
   - Clean up resources

## Data Flow & Message Processing

### WebSocket Connection & Subscription

```
Bot Startup
└─> RealtimeOddsStreamer.start()
    └─> Connect to wss://ws-subscriptions-clob.polymarket.com/ws/market
        └─> Send subscription message:
            {
              "action": "subscribe",
              "subscriptions": [{
                "topic": "market",
                "assets_ids": [token_id_yes, token_id_no]
              }]
            }
```

### Incoming Message Types

The WebSocket sends 3 types of messages:

1. **`book`** - Orderbook snapshot **(THIS IS WHAT WE NEED)**
   - Contains bids/asks arrays at different price levels
   - Extract best_bid (top of bids array) = YES odds
   - Calculate NO odds = 1.0 - YES odds

2. **`last_trade_price`** - Recent trade (can ignore)

3. **`price_change`** - Price delta (can ignore)

### Odds Extraction Logic

```python
# From 'book' message payload:
payload = {
  'market': market_id,
  'asset_id': token_id,
  'bids': [[price, size], ...],  # sorted high to low
  'asks': [[price, size], ...]   # sorted low to high
}

# Extract YES odds (best bid for YES token):
yes_odds = float(payload['bids'][0][0]) if payload['bids'] else 0.50
no_odds = 1.0 - yes_odds

# Store snapshot:
_current_odds[market_id] = OddsSnapshot(
  market_id=market_id,
  yes_odds=yes_odds,
  no_odds=no_odds,
  timestamp=datetime.now(),
  best_bid=yes_odds,
  best_ask=1.0 - yes_odds
)
```

### Trading Loop Integration

```python
# In auto_trade.py (replace old polling):
# OLD: odds = await self.odds_poller.get_odds(market.id)
# NEW:
odds = self.realtime_streamer.get_current_odds(market.id)
if odds:
    yes_odds = odds.yes_odds
    no_odds = odds.no_odds
    # Check if > 70%...
```

**Latency:** Zero! Odds updated within milliseconds of Polymarket book changes.

## Error Handling & Reconnection

### Exponential Backoff Strategy

```python
BACKOFF_DELAYS = [1, 2, 4, 8, 16, 32, 60]  # Max 60s

async def _stream_loop(self):
    backoff_index = 0
    consecutive_failures = 0

    while True:
        try:
            # Attempt connection
            await self._connect_and_stream()

            # Success! Reset backoff
            backoff_index = 0
            consecutive_failures = 0

        except websockets.ConnectionClosed:
            logger.warning("WebSocket closed, reconnecting...")
            consecutive_failures += 1

        except Exception as e:
            logger.error("Stream error", error=str(e))
            consecutive_failures += 1

        # Exponential backoff
        delay = BACKOFF_DELAYS[min(backoff_index, len(BACKOFF_DELAYS)-1)]
        backoff_index += 1

        # Alert on extended failure
        if consecutive_failures >= 5:
            await self._send_telegram_alert(
                f"⚠️ Odds streamer disconnected for {consecutive_failures} attempts"
            )

        await asyncio.sleep(delay)
```

### Failure Scenarios

1. **Network Blip (< 10s):**
   - Reconnects immediately (1s delay)
   - No impact on trading
   - Logged as warning

2. **Polymarket API Down (> 5min):**
   - Keeps trying with 60s max backoff
   - Telegram alert sent
   - Trading continues with last known odds (with staleness warning)

3. **Market Transition:**
   - Detect new market (every 60s check)
   - Unsubscribe old, subscribe new
   - Seamless, no reconnection needed

4. **Invalid Data:**
   - Skip malformed messages
   - Log error with payload
   - Continue processing other messages

### Graceful Shutdown

```python
async def stop(self):
    if self._ws and not self._ws.closed:
        await self._ws.close()
    if self._reconnect_task:
        self._reconnect_task.cancel()
```

## Testing Strategy

### Unit Tests (`tests/test_realtime_odds_streamer.py`)

1. **Test Odds Extraction:**
   ```python
   def test_extract_odds_from_book_message():
       # Given: Book message with bids/asks
       # When: Process message
       # Then: Correctly extracts YES/NO odds
   ```

2. **Test State Management:**
   ```python
   def test_thread_safe_odds_access():
       # Given: Multiple concurrent get_current_odds calls
       # When: Odds being updated simultaneously
       # Then: No race conditions, returns consistent snapshot
   ```

3. **Test Reconnection Logic:**
   ```python
   def test_exponential_backoff():
       # Given: Connection failures
       # When: Multiple reconnect attempts
       # Then: Backoff delays increase correctly
   ```

### Integration Tests

1. **Real WebSocket Test:**
   ```python
   @pytest.mark.integration
   async def test_real_polymarket_connection():
       # Connect to actual Polymarket CLOB
       # Subscribe to live market
       # Verify odds updates received
       # Duration: 30 seconds
   ```

2. **Market Transition Test:**
   ```python
   async def test_market_resubscription():
       # Mock market changing
       # Verify unsubscribe from old
       # Verify subscribe to new
   ```

### Validation Plan (Before Deploying)

1. **Side-by-Side Comparison (24 hours):**
   - Run both old poller + new streamer
   - Log odds from both sources
   - Compare: Are WebSocket odds matching REST API?
   - Validate: Do WebSocket odds catch 70%+ spikes that REST misses?

2. **Latency Verification:**
   - Measure: Time from Polymarket website showing 70% → Bot detecting it
   - Target: < 1 second (vs 0-60s with polling)

3. **Reliability Test:**
   - Monitor connection uptime over 48 hours
   - Verify: Automatic reconnection works
   - Check: No memory leaks from persistent connection

### Success Criteria

- ✓ WebSocket odds match REST API (when REST updates)
- ✓ Catch ≥90% of 70%+ spikes visible on website
- ✓ Connection uptime ≥99.9%
- ✓ Reconnection within 10s of disconnect

## Implementation Plan

### Phase 1: Core Streamer (Tasks 1-4)
1. Create RealtimeOddsStreamer class with WebSocket connection
2. Implement odds extraction from book messages
3. Add thread-safe state management
4. Write unit tests for odds extraction

### Phase 2: Reconnection & Reliability (Tasks 5-7)
5. Implement exponential backoff reconnection
6. Add market transition detection and resubscription
7. Add Telegram alerts for extended failures

### Phase 3: Integration (Tasks 8-10)
8. Integrate streamer into auto_trade.py startup
9. Replace odds_poller.get_odds() calls with realtime_streamer.get_current_odds()
10. Add graceful shutdown handling

### Phase 4: Testing & Validation (Tasks 11-13)
11. Write integration tests with real WebSocket
12. Run 24-hour side-by-side comparison with old poller
13. Validate latency and spike detection rate

### Phase 5: Deployment (Tasks 14-15)
14. Deploy to production with monitoring
15. Remove old MarketOddsPoller after 48-hour validation

## Rollout Plan

1. **Development:** Implement and test in feature branch
2. **Staging:** Deploy to test bot, monitor for 24 hours
3. **Production:** Merge to main, deploy with old poller still available
4. **Validation:** Monitor for 48 hours, compare with website spikes
5. **Cleanup:** Remove old poller after successful validation

## Success Metrics

- **Latency:** < 1 second from website spike to bot detection
- **Spike Detection:** ≥90% of 70%+ opportunities visible on website
- **Uptime:** ≥99.9% WebSocket connection uptime
- **Reconnection:** ≤10 seconds to recover from disconnects
- **No Regressions:** All existing tests pass, no trading disruptions

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| WebSocket API changes | High | Keep old poller as fallback, monitor API docs |
| Connection instability | Medium | Exponential backoff, Telegram alerts |
| Memory leaks from persistent connection | Low | Monitor memory usage, add connection cycling if needed |
| Market transition gaps | Low | 60s resubscription check, log transitions |

## Dependencies

- `websockets` library (already installed)
- Polymarket CLOB WebSocket API (stable, public)
- No breaking changes to existing code

## Notes

- The WebSocket connection is similar to existing BTC price streaming (crypto_price_stream.py)
- Market microstructure service already connects to same WebSocket temporarily
- This design reuses proven patterns from existing codebase
- Zero latency advantage makes this a high-impact improvement for catching opportunities
