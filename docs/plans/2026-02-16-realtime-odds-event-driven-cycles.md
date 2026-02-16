# Real-Time Odds Event-Driven Cycle Triggering

**Date:** 2026-02-16
**Status:** Design Approved
**Goal:** Replace timer-based cycle triggering with event-driven architecture that monitors real-time odds and triggers trading cycles only when sustained opportunities exist (>70% odds for UP or DOWN).

---

## Problem Statement

**Current Architecture Issues:**
- Timer-based: Runs cycles every 60 seconds regardless of market conditions
- Wastes resources checking markets with low odds (<70%)
- May miss opportunities between 60-second intervals
- 3 of 5 recent cycles were skipped due to odds filter

**AI Confidence Issue:**
- Cycle 1 confidence cascade: AI 66% → tiered weighting 62.3% → conflict penalty 52%
- Flat -10% conflict penalty too harsh when sentiment disagrees with BTC direction
- Drops below 70% threshold causing trades to be skipped

---

## Architecture Overview

**Event-Driven Components:**

```
┌─────────────────────────────────────────────────────────────┐
│                    Event-Driven Architecture                 │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  RealtimeOddsStreamer (existing)                             │
│  └─► WebSocket connection to Polymarket CLOB                 │
│      └─► Continuously streams odds updates                   │
│          └─► Stores in memory: {market_id → odds snapshot}   │
│                                                              │
│  OddsMonitor (new)                                           │
│  └─► Polls streamer every 1 second                           │
│      └─► Checks if odds >70% for UP or DOWN                  │
│          └─► Tracks sustained threshold (5 seconds)          │
│              └─► Triggers AutoTrader.run_cycle()             │
│                                                              │
│  MarketValidator (new)                                       │
│  └─► Parses market slug timestamp                            │
│      └─► Verifies market is active (±2 min tolerance)        │
│          └─► Prevents trading on expired markets             │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Key Design Decisions:**
1. **Sustained Threshold:** Odds must stay >70% for 5 seconds (prevents false triggers on spikes)
2. **Smart Cooldown:** 30-second cooldown after cycle completion, but immediate trigger on market transition
3. **Strict Market Validation:** Parse timestamp from slug, verify within current 15-min window
4. **Graceful Market Transitions:** Finish current cycle before switching to new market

---

## Component 1: OddsMonitor

**Purpose:** Monitor real-time odds and trigger cycles when sustained opportunities detected.

**Implementation:**

```python
class OddsMonitor:
    """Monitors real-time odds and triggers cycles on sustained thresholds."""

    def __init__(
        self,
        streamer: RealtimeOddsStreamer,
        bot: AutoTrader,
        threshold: float = 0.70,
        sustained_seconds: int = 5,
        cooldown_seconds: int = 30
    ):
        self.streamer = streamer
        self.bot = bot
        self.threshold = threshold
        self.sustained_seconds = sustained_seconds
        self.cooldown_seconds = cooldown_seconds

        # State tracking
        self.threshold_start_time: Dict[str, datetime] = {}
        self.last_trigger_time: Dict[str, datetime] = {}
        self.cycle_in_progress = False
        self._running = False
        self._monitor_task: Optional[asyncio.Task] = None

    async def start(self):
        """Start monitoring loop (non-blocking)."""
        if self._running:
            return
        self._running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("OddsMonitor started", threshold=self.threshold)

    async def stop(self):
        """Stop monitoring loop."""
        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("OddsMonitor stopped")

    async def _monitor_loop(self):
        """Main monitoring loop - checks odds every 1 second."""
        while self._running:
            try:
                await self._check_opportunities()
            except Exception as e:
                logger.error("Monitor loop error", error=str(e))

            await asyncio.sleep(1)  # Check every second

    async def _check_opportunities(self):
        """Check current market for trading opportunities."""
        # Get currently active market
        try:
            market = self.bot.client.discover_btc_15min_market()
        except Exception as e:
            logger.error("Failed to discover market", error=str(e))
            return

        # Validate market is active
        if not MarketValidator.is_market_active(market.slug):
            logger.warning("Market not active, skipping", slug=market.slug)
            return

        # Get current odds from streamer
        odds = self.streamer.get_current_odds(market.id)
        if not odds:
            logger.debug("No odds data available", market_id=market.id)
            return

        # Check staleness
        age = (datetime.now() - odds.timestamp).total_seconds()
        if age > 120:  # 2 minutes
            logger.warning("Odds data too stale", age_seconds=age, market_id=market.id)
            return

        # Check if threshold met
        if odds.yes_odds > self.threshold:
            await self._handle_threshold_met(market, "UP", odds.yes_odds)
        elif odds.no_odds > self.threshold:
            await self._handle_threshold_met(market, "DOWN", odds.no_odds)
        else:
            # Threshold not met - reset tracking
            self.threshold_start_time.pop(market.id, None)

    async def _handle_threshold_met(self, market, direction: str, odds_value: float):
        """Handle case where odds exceed threshold."""
        market_id = market.id
        now = datetime.now()

        # Check cooldown period
        if market_id in self.last_trigger_time:
            time_since_last = (now - self.last_trigger_time[market_id]).total_seconds()
            if time_since_last < self.cooldown_seconds:
                logger.debug(
                    "In cooldown period",
                    market_id=market_id,
                    remaining=self.cooldown_seconds - time_since_last
                )
                return

        # Track sustained threshold
        if market_id not in self.threshold_start_time:
            self.threshold_start_time[market_id] = now
            logger.info(
                "Threshold met - tracking sustained period",
                market_id=market_id,
                direction=direction,
                odds=f"{odds_value:.2f}"
            )
            return

        # Check if sustained long enough
        sustained_duration = (now - self.threshold_start_time[market_id]).total_seconds()
        if sustained_duration >= self.sustained_seconds:
            # Trigger cycle!
            logger.info(
                "✅ Sustained threshold detected - triggering cycle",
                market_id=market_id,
                direction=direction,
                odds=f"{odds_value:.2f}",
                sustained_seconds=sustained_duration
            )
            await self._trigger_cycle(market, direction, odds_value)

            # Update state
            self.last_trigger_time[market_id] = now
            self.threshold_start_time.pop(market_id, None)

    async def _trigger_cycle(self, market, direction: str, odds_value: float):
        """Trigger a trading cycle."""
        if self.cycle_in_progress:
            logger.info("Cycle already running, ignoring trigger")
            return

        self.cycle_in_progress = True
        try:
            await self.bot.run_cycle()
        finally:
            self.cycle_in_progress = False
```

**Key Features:**
- Polls streamer every 1 second for current odds
- Tracks sustained threshold: starts timer when odds >70%, triggers after 5 seconds
- Enforces 30-second cooldown between triggers
- Prevents concurrent cycles with `cycle_in_progress` flag
- Validates market is active before triggering

---

## Component 2: MarketValidator

**Purpose:** Ensure only trading on markets active for current 15-min window.

**Implementation:**

```python
class MarketValidator:
    """Validates market slugs are active for current time window."""

    @staticmethod
    def parse_market_timestamp(slug: str) -> Optional[int]:
        """
        Parse timestamp from market slug.

        Example: 'btc-updown-15m-1771270200' -> 1771270200

        Args:
            slug: Market slug string

        Returns:
            Unix timestamp if valid format, None otherwise
        """
        parts = slug.split('-')
        if len(parts) >= 4 and parts[0] == 'btc' and parts[1] == 'updown':
            try:
                return int(parts[3])
            except ValueError:
                logger.error("Invalid timestamp in market slug", slug=slug, timestamp_part=parts[3])
                return None

        logger.error("Unexpected market slug format", slug=slug)
        return None

    @staticmethod
    def is_market_active(slug: str, tolerance_minutes: int = 2) -> bool:
        """
        Verify market is active for current time.

        A market is considered active if its timestamp is within:
        - Now - 2 minutes (tolerance for early markets)
        - Now + 17 minutes (15-min window + 2 min tolerance)

        Args:
            slug: Market slug to validate
            tolerance_minutes: Minutes of tolerance before/after window

        Returns:
            True if market is active, False otherwise
        """
        market_timestamp = MarketValidator.parse_market_timestamp(slug)
        if not market_timestamp:
            return False

        now = datetime.now().timestamp()
        min_valid = now - (tolerance_minutes * 60)
        max_valid = now + (15 * 60) + (tolerance_minutes * 60)

        is_active = min_valid <= market_timestamp <= max_valid

        if not is_active:
            market_time = datetime.fromtimestamp(market_timestamp).strftime("%Y-%m-%d %H:%M:%S")
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            logger.warning(
                "Market not active for current time",
                slug=slug,
                market_time=market_time,
                current_time=current_time
            )

        return is_active
```

**Key Features:**
- Extracts timestamp from slug format: `btc-updown-15m-{timestamp}`
- Validates timestamp is within current 15-min window ± 2 minutes
- Returns False for malformed slugs
- Logs detailed warnings when markets are expired/future

---

## Component 3: Error Handling

**Critical error scenarios and handling:**

### 1. Stale or Missing Odds Data
```python
# In OddsMonitor._check_opportunities()
odds = self.streamer.get_current_odds(market_id)
if not odds:
    logger.debug("No odds data available, skipping check")
    return

age = (datetime.now() - odds.timestamp).total_seconds()
if age > 120:  # 2 minutes
    logger.warning("Odds data too stale", age_seconds=age, market_id=market_id)
    return  # Don't trigger on stale data
```

### 2. Cycle Already Running
```python
# In OddsMonitor._trigger_cycle()
if self.cycle_in_progress:
    logger.info("Cycle already running, ignoring trigger")
    return

self.cycle_in_progress = True
try:
    await self.bot.run_cycle()
finally:
    self.cycle_in_progress = False  # Always reset flag
```

### 3. Market Validation Failures
```python
# In OddsMonitor._check_opportunities()
if not MarketValidator.is_market_active(market.slug):
    logger.warning("Market not active, skipping", slug=market.slug)
    return
```

### 4. WebSocket Disconnections
- `RealtimeOddsStreamer` already handles reconnection with exponential backoff
- `OddsMonitor` relies on streamer's staleness warnings
- If odds become stale (>2 min), OddsMonitor won't trigger cycles until fresh data arrives

### 5. Rapid Odds Fluctuations
- Sustained threshold (5 seconds) prevents triggers on brief spikes
- Cooldown period (30 seconds) prevents multiple triggers on same opportunity
- Market transition resets cooldown for new opportunities

---

## Integration with AutoTrader

**Modified AutoTrader architecture:**

```python
class AutoTrader:
    def __init__(self, ...):
        # Existing components
        self.client = PolymarketClient(...)
        self.streamer = RealtimeOddsStreamer(self.client)

        # New components
        self.market_validator = MarketValidator()
        self.odds_monitor = OddsMonitor(
            streamer=self.streamer,
            bot=self,  # Pass self so monitor can trigger cycles
            threshold=0.70,
            sustained_seconds=5,
            cooldown_seconds=30
        )

    async def start(self):
        """Start event-driven monitoring."""
        logger.info("Starting AutoTrader with event-driven monitoring...")

        # 1. Start WebSocket streamer
        await self.streamer.start()
        await asyncio.sleep(2)  # Let it connect and get initial data

        # 2. Start odds monitor (triggers cycles on threshold)
        await self.odds_monitor.start()

        logger.info("AutoTrader running - monitoring for opportunities...")

    async def stop(self):
        """Stop all components gracefully."""
        logger.info("Stopping AutoTrader...")
        await self.odds_monitor.stop()
        await self.streamer.stop()
```

**Code to Remove:**
```python
# DELETE: Old timer-based loop in auto_trade.py lines 540-579
# while self.running:
#     await self.run_cycle()
#     if self.running:
#         logger.info(f"Waiting {self.interval} seconds until next cycle...")
#         await asyncio.sleep(self.interval)
```

**Key Changes:**
- `run_cycle()` method stays unchanged - still contains all trading logic
- Instead of being called on timer, it's now called by `OddsMonitor` when opportunities detected
- Logs change from "Waiting 60 seconds..." to "Monitoring for opportunities..."
- Startup sequence ensures streamer connects before monitor starts

---

## Testing Strategy

### Unit Tests for MarketValidator

```python
def test_parse_market_timestamp():
    """Test timestamp extraction from slug."""
    assert MarketValidator.parse_market_timestamp(
        'btc-updown-15m-1771270200'
    ) == 1771270200

    assert MarketValidator.parse_market_timestamp(
        'invalid-slug'
    ) is None

def test_is_market_active():
    """Test market active validation."""
    # Mock current time to test validation window
    with freeze_time("2026-02-16 10:30:00"):
        # Market for 10:30 should be active
        assert MarketValidator.is_market_active('btc-updown-15m-1771270200')

        # Market for 10:00 (30 minutes ago) should not be active
        assert not MarketValidator.is_market_active('btc-updown-15m-1771268400')

        # Market for 11:00 (30 minutes future) should not be active
        assert not MarketValidator.is_market_active('btc-updown-15m-1771272000')
```

### Integration Tests for OddsMonitor

```python
async def test_sustained_threshold_triggers_cycle():
    """Test that cycles only trigger after sustained threshold."""
    # Mock streamer returning >70% odds
    mock_streamer = MockStreamer(yes_odds=0.75)
    mock_bot = MockBot()
    monitor = OddsMonitor(
        streamer=mock_streamer,
        bot=mock_bot,
        threshold=0.70,
        sustained_seconds=5
    )

    # Should NOT trigger immediately
    await monitor._check_opportunities()
    assert not mock_bot.cycle_triggered

    # After 5 seconds, SHOULD trigger
    await asyncio.sleep(5)
    await monitor._check_opportunities()
    assert mock_bot.cycle_triggered

async def test_cooldown_prevents_rapid_triggers():
    """Test that cooldown period is enforced."""
    mock_streamer = MockStreamer(yes_odds=0.75)
    mock_bot = MockBot()
    monitor = OddsMonitor(
        streamer=mock_streamer,
        bot=mock_bot,
        cooldown_seconds=30
    )

    # First trigger
    await monitor._trigger_cycle(market, "UP", 0.75)
    assert mock_bot.cycle_count == 1

    # Immediate retry should be blocked
    await monitor._trigger_cycle(market, "UP", 0.75)
    assert mock_bot.cycle_count == 1  # Still 1, not 2

    # After 30 seconds, should allow
    await asyncio.sleep(30)
    await monitor._trigger_cycle(market, "UP", 0.75)
    assert mock_bot.cycle_count == 2
```

### Manual Testing Process

**1. Dry-Run Mode:**
Add `--dry-run` flag that logs "Would trigger cycle" instead of executing:
```python
if self.dry_run:
    logger.info("🧪 [DRY RUN] Would trigger cycle", market_id=market.id, direction=direction)
    return
```

**2. Monitor Logs:**
```bash
# Watch for sustained threshold detection
tail -f logs/bot.log | grep "Sustained threshold"

# Verify cooldown enforcement
tail -f logs/bot.log | grep "cooldown"

# Check market validation
tail -f logs/bot.log | grep "Market not active"
```

**3. Success Criteria:**
- ✅ Only triggers when odds >70% sustained for 5 seconds
- ✅ Respects 30-second cooldown period
- ✅ Only monitors currently active markets (timestamp validation)
- ✅ Handles WebSocket disconnections gracefully (reconnects automatically)
- ✅ Logs clearly show: threshold detection → sustained period → cycle trigger

---

## Implementation Phases

### Phase 1: Core Components (MarketValidator + OddsMonitor)
1. Create `MarketValidator` class with unit tests
2. Create `OddsMonitor` class with integration tests
3. Verify tests pass in isolation

### Phase 2: AutoTrader Integration
1. Modify `AutoTrader.__init__()` to instantiate new components
2. Modify `AutoTrader.start()` to use event-driven startup
3. Remove old timer-based loop code
4. Add dry-run mode for safe testing

### Phase 3: Production Validation
1. Run in dry-run mode for 1 hour, verify logs
2. Run in live mode with small position sizes
3. Monitor for 24 hours, verify behavior
4. Scale up to normal position sizes

---

## Rollback Plan

If event-driven architecture causes issues:

1. **Immediate Rollback (5 minutes):**
```bash
git revert <commit-hash>  # Revert to timer-based architecture
systemctl restart polymarket-bot
```

2. **Monitoring Points:**
- Missed opportunities (should decrease, not increase)
- False triggers (should be zero with 5-second sustained threshold)
- Resource usage (should be similar or lower)

3. **Success Metrics:**
- Number of cycles per hour (should match opportunity frequency)
- Trigger accuracy (should be >95% for valid opportunities)
- System uptime (should maintain 99%+ availability)

---

## Future Enhancements (Out of Scope)

- **Multi-market monitoring:** Monitor multiple markets simultaneously
- **Dynamic threshold adjustment:** Adjust threshold based on market volatility
- **Telegram alerts:** Send notifications on cycle triggers
- **Performance metrics dashboard:** Real-time monitoring of trigger accuracy

---

**Design Status:** ✅ Approved for implementation
**Next Steps:** Create detailed implementation plan with file paths and code snippets
