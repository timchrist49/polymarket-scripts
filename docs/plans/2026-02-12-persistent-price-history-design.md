# Persistent 24-Hour Price History Design

**Date:** 2026-02-12
**Status:** Approved
**Goal:** Eliminate Binance API dependency by building internal price history from Polymarket WebSocket

## Problem Statement

Current bot depends on Binance API for historical BTC prices, causing:
- Frequent timeouts (7 occurrences in 15 minutes)
- Failed price_to_beat lookups (falls back to current price)
- Unable to determine UP/DOWN direction when fallback occurs
- Bot unable to trade effectively

**Critical Issue:** When Binance times out, bot uses current price as price_to_beat, making comparison meaningless:
```
Current BTC Price:  $67,018.35
Price to Beat:      $67,018.35  ← Should be price from market start!
Difference:         $0.00       ← Can't determine direction
```

## Solution Architecture

### High-Level Design

```
┌──────────────────────────────────────────────────┐
│  Polymarket WebSocket (RTDS)                     │
│  └─► Real-time price updates (every change)     │
└──────────────────────┬───────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────┐
│  PriceHistoryBuffer (NEW)                        │
│  ├─► In-memory: collections.deque(maxlen=2880)  │
│  ├─► On update: append to deque (all updates)   │
│  ├─► Every 5 min: save to data/price_history.json│
│  └─► Every 1 hour: cleanup old entries (>24h)   │
└──────────────────────┬───────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────┐
│  BTCPriceService                                 │
│  ├─► Query buffer for historical prices          │
│  ├─► Fallback to Binance if insufficient         │
│  └─► Used by: price_to_beat, technical analysis  │
└──────────────────────────────────────────────────┘
```

### Memory Impact

- **Current Usage:** 260MB (stable)
- **New Data:** ~50-100KB (24 hours of real-time updates)
- **Total:** ~260MB (negligible increase)
- **No OOM Risk:** Previous 7.7GB leak was from infinite recursion bug (fixed)

## Data Storage

### File Format: `data/price_history.json`

```json
{
  "version": "1.0",
  "last_updated": "2026-02-12T06:00:00Z",
  "retention_hours": 24,
  "prices": [
    {
      "timestamp": 1770875100,
      "price": "67018.35",
      "source": "polymarket",
      "received_at": "2026-02-12T05:45:00Z"
    }
  ]
}
```

### Persistence Strategy

**On Startup:**
- Load `data/price_history.json` into memory deque
- If file doesn't exist, start with empty buffer
- WebSocket will populate on connection
- **Bot operational immediately** (no waiting period)

**During Operation:**
- Every **5 minutes**: Save buffer to disk (background task)
- Only writes if buffer has new data
- **Atomic write:** save to `.tmp`, then rename (prevents corruption)
- Captures **all real-time updates** (not just snapshots)

**On Shutdown:**
- Final save to disk (graceful shutdown handler)

**Why 5-minute intervals?**
- Balances data safety vs disk I/O overhead
- Worst case: lose 5 minutes of data on crash (acceptable)
- WebSocket repopulates quickly on reconnection

## Data Flow

### Price Update Flow

```
1. WebSocket Update Arrives
   └─► crypto_price_stream.py receives price
       └─► Stores in _current_price (existing)
       └─► NEW: Appends to PriceHistoryBuffer
           └─► In-memory deque gets new entry
           └─► If buffer full (>24h), oldest entry auto-removed

2. Bot Needs Historical Price (e.g., price_to_beat)
   └─► btc_price.py calls get_price_at_timestamp(1770875100)
       └─► NEW: Query PriceHistoryBuffer first
           ├─► If found: return immediately (fast!)
           └─► If not found: fallback to Binance API

3. Technical Analysis Needs 60-Minute History
   └─► btc_price.py calls get_historical_prices(start, end)
       └─► NEW: Query buffer for all prices in range
       └─► Aggregate into 1-minute candles (OHLCV)
       └─► If insufficient data: fallback to Binance
```

### Integration Points

**Modified Files:**
1. `crypto_price_stream.py` - Add buffer integration
2. `btc_price.py` - Query buffer before Binance
3. **NEW:** `price_history_buffer.py` - Core buffer implementation

**Fallback Strategy:**
Buffer unavailable → Falls back to Binance API (existing behavior)

**Fallback Triggers:**
- Buffer empty (bot just started, WebSocket disconnected)
- Timestamp too old (>24 hours)
- WebSocket connection lost
- Data corrupted or missing

## Error Handling

### Critical Scenarios

**1. WebSocket Disconnection**
```
Problem: WebSocket drops, no new prices arriving
Solution:
  ├─► Buffer retains existing 24h history (in-memory + disk)
  ├─► Bot continues using cached prices for recent queries
  ├─► Falls back to Binance for real-time prices
  └─► On reconnection: Resume appending to buffer
```

**2. Bot Restart/Update**
```
Problem: Bot stops, in-memory buffer lost
Solution:
  ├─► On shutdown: Save buffer to disk (final write)
  ├─► On startup: Load from data/price_history.json
  ├─► Validate loaded data (remove corrupted entries)
  └─► Bot operational immediately (no waiting period)
```

**3. Corrupted Disk File**
```
Problem: price_history.json corrupted/invalid JSON
Solution:
  ├─► Try to load file
  ├─► If JSON parse fails: log error, start with empty buffer
  ├─► WebSocket will repopulate on connection
  └─► Fallback to Binance for immediate needs
```

**4. Clock Skew / Timestamp Issues**
```
Problem: System clock adjusted, timestamps out of order
Solution:
  ├─► Always use WebSocket timestamps (not system time)
  ├─► Validate new entries are >= last entry
  └─► Reject out-of-order updates (log warning)
```

**5. Memory Pressure**
```
Problem: Buffer growing beyond 24 hours
Solution:
  ├─► Deque has maxlen=2880 (24h × 2 for safety)
  ├─► Automatic overflow (oldest removed)
  ├─► Hourly cleanup task removes >24h entries
  └─► Monitor: Log buffer size every 10 minutes
```

**Graceful Degradation:**
If buffer fails completely → Bot continues using Binance API (current behavior). No worse than before, just not better.

## Implementation

### New Component: `price_history_buffer.py`

**Core Class Structure:**
```python
class PriceHistoryBuffer:
    def __init__(self, retention_hours: int = 24, save_interval: int = 300):
        self._buffer = deque(maxlen=2880)  # 24h × 2 for safety
        self._lock = asyncio.Lock()  # Thread-safe operations
        self._persistence_file = "data/price_history.json"
        self._save_interval = save_interval  # 300 seconds = 5 minutes

    async def append(self, timestamp: int, price: Decimal):
        """Add new price to buffer."""

    async def get_price_at(self, timestamp: int) -> Optional[Decimal]:
        """Get price at specific timestamp (with tolerance)."""

    async def get_price_range(self, start: int, end: int) -> List[PriceEntry]:
        """Get all prices in time range."""

    async def save_to_disk(self):
        """Persist buffer to JSON file."""

    async def load_from_disk(self):
        """Load buffer from JSON file on startup."""

    async def cleanup_old_entries(self):
        """Remove entries older than 24 hours."""
```

### Background Tasks

```python
# In auto_trade.py
async def price_history_saver(buffer):
    """Save buffer every 5 minutes."""
    while True:
        await asyncio.sleep(300)
        await buffer.save_to_disk()

async def price_history_cleaner(buffer):
    """Cleanup old entries every hour."""
    while True:
        await asyncio.sleep(3600)
        await buffer.cleanup_old_entries()
```

## Testing Strategy

### Unit Tests (`test_price_history_buffer.py`)
- Append prices, verify retrieval
- Test 24h overflow behavior
- Test persistence (save/load)
- Test timestamp lookup with tolerance
- Test corrupted file handling

### Integration Tests
- WebSocket → Buffer → BTCPriceService flow
- Fallback to Binance when buffer empty
- Bot restart with disk load
- Concurrent access (lock behavior)

### Manual Testing
- Run bot for 30 minutes, verify buffer grows
- Restart bot, verify history preserved
- Disconnect WebSocket, verify fallback works
- Delete JSON file, verify graceful recovery

## Rollout Plan

1. **Phase 1: Implement buffer** (isolated from existing code)
   - Create `price_history_buffer.py`
   - Write unit tests
   - Test independently

2. **Phase 2: Integrate with WebSocket**
   - Modify `crypto_price_stream.py`
   - Add buffer append on price updates
   - Test WebSocket → Buffer flow

3. **Phase 3: Update BTCPriceService**
   - Modify `btc_price.py`
   - Query buffer before Binance
   - Test fallback behavior

4. **Phase 4: Monitor**
   - Deploy to production
   - Monitor for 24 hours
   - Verify buffer grows correctly
   - Check Binance fallback rate

5. **Phase 5: (Optional) Remove Binance dependency**
   - After 1 week of stable operation
   - Only if buffer proves 100% reliable

## Success Criteria

- ✅ Bot trades consistently (no price_to_beat fallback failures)
- ✅ Memory usage remains stable (~260MB)
- ✅ Bot restarts don't lose price history
- ✅ Binance timeout errors don't prevent trading
- ✅ Technical analysis has sufficient data (60+ minutes)
- ✅ No performance degradation (latency < 100ms for queries)

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Buffer doesn't accumulate fast enough | WebSocket updates arrive every price change (plenty of data) |
| Disk saves fail | Keep operating from memory, retry on next cycle |
| JSON file grows too large | Limited to 24h × real-time updates (~100KB max) |
| WebSocket disconnects frequently | Buffer retains 24h, fallback to Binance available |
| Bot updates lose data | 5-min save interval, max 5 min data loss (acceptable) |

## Future Enhancements

- **Compression:** Compress JSON file for older entries (if needed)
- **Candle Aggregation:** Pre-compute 1-min candles for faster queries
- **Multiple Timeframes:** Store 1m, 5m, 15m candles for different strategies
- **Database Storage:** Move to SQLite for better query performance (if needed)
- **Multi-Asset Support:** Extend to ETH, SOL, etc. (if trading other markets)

---

**Approved By:** User
**Next Steps:** Create implementation plan, execute in phases
