# Trade Settlement System Design

**Date**: 2026-02-11
**Status**: Approved
**Priority**: Critical (blocks self-reflection system)

## Problem Statement

The bot logs trades to the database but never determines if they won or lost. All 36 trades have `is_win = NULL` and `profit_loss = NULL`, which prevents the self-reflection system from analyzing performance.

**Current State:**
- Trades are logged with `action`, `position_size`, `executed_price`, `price_to_beat`
- Markets close after 15 minutes
- No settlement mechanism exists to determine outcomes
- Self-reflection triggers (3 consecutive losses, 10 trades) cannot function

## Solution Overview

Implement a background settlement service that:
1. Periodically checks for unsettled trades (every 5-10 minutes)
2. Fetches BTC price at market close timestamp
3. Compares close price vs price_to_beat to determine outcome
4. Calculates profit/loss based on Polymarket payout mechanics
5. Updates database with settlement data

## Architecture

### Components

**1. TradeSettler (`polymarket/performance/settler.py`)**
- Core settlement logic
- Queries unsettled trades from database
- Fetches BTC prices at specific timestamps
- Determines win/loss outcomes
- Calculates profit/loss
- Updates trade records

**2. Settlement Scheduler**
- Runs every 5-10 minutes (configurable)
- Processes up to 50 unsettled trades per cycle (batched)
- Integrated into `auto_trade.py` main loop
- Continues running even if bot restarts

**3. BTC Price Fetcher (reused)**
- Uses existing BTC price service from bot
- Fetches historical BTC price at specific Unix timestamps
- Consistent with trading price source

**4. Database Updater**
- Updates `trades` table with outcome data
- Sets `actual_outcome`, `profit_loss`, `is_win`
- Transactional updates (rollback on failure)

## Settlement Process Flow

### 1. Query Unsettled Trades

```sql
SELECT * FROM trades
WHERE action IN ('YES', 'NO')           -- Exclude HOLD actions
  AND is_win IS NULL                    -- Not yet settled
  AND timestamp < NOW() - INTERVAL '15 minutes'  -- Market has closed
ORDER BY timestamp ASC
LIMIT 50  -- Process in batches
```

### 2. For Each Trade

**Step 1: Parse Market Close Timestamp**
- Extract timestamp from `market_slug`
- Format: `"btc-updown-15m-1770828300"` → timestamp = `1770828300`
- This is the Unix timestamp when the market closes

**Step 2: Fetch BTC Price at Close**
- Call BTC price fetcher with the close timestamp
- Get the exact BTC price when the market resolved
- Example: Timestamp 1770828300 → BTC = $72,000

**Step 3: Determine Outcome**
```python
if btc_price_at_close > price_to_beat:
    actual_outcome = "YES"  # UP won
elif btc_price_at_close < price_to_beat:
    actual_outcome = "NO"   # DOWN won
else:
    actual_outcome = "NO"   # Tie defaults to NO (rare)
```

**Step 4: Calculate Profit/Loss**

Polymarket binary market mechanics:
- Shares bought: `position_size / executed_price`
- If win: Payout = `shares × $1.00`, Profit = `payout - position_size`
- If loss: Profit = `-position_size`

```python
shares = position_size / executed_price

if (action == "YES" and actual_outcome == "YES") or \
   (action == "NO" and actual_outcome == "NO"):
    # Win
    payout = shares * 1.00
    profit_loss = payout - position_size
    is_win = True
else:
    # Loss
    profit_loss = -position_size
    is_win = False
```

**Example:**
- Trade: YES at $0.39, position $11.92
- Shares: $11.92 / $0.39 = 30.56 shares
- If YES wins: Payout $30.56, Profit = +$18.64
- If NO wins: Payout $0, Loss = -$11.92

### 3. Update Database

```python
UPDATE trades
SET actual_outcome = ?,
    profit_loss = ?,
    is_win = ?,
    settled_at = NOW()
WHERE id = ?
```

### 4. Return Statistics

```python
{
    "success": True,
    "settled_count": 15,
    "wins": 8,
    "losses": 7,
    "total_profit": 45.23,
    "pending_count": 5,
    "errors": []
}
```

## Error Handling & Retry Logic

### Scenario 1: Can't Parse Market Timestamp
**Cause**: Unexpected `market_slug` format or missing timestamp

**Action**:
- Log error with trade details
- Mark as `actual_outcome = "UNKNOWN"`
- Set `is_win = NULL` (exclude from win rate calculations)
- Continue processing other trades

**Alert**: Send warning notification

### Scenario 2: Can't Fetch BTC Price
**Cause**: BTC price service down, timestamp data unavailable, API error

**Action**:
- **Skip this trade** (don't mark as settled)
- Keep `is_win = NULL` so it's retried next cycle
- Log attempt count
- Retry indefinitely until successful

**Alert**: If trade pending >1 hour, send warning

### Scenario 3: Database Update Fails
**Cause**: Transaction error, connection lost, constraint violation

**Action**:
- Rollback transaction
- Log error with trade ID
- Keep `is_win = NULL` for retry
- Continue processing other trades

**Alert**: Log database errors for monitoring

### Retry Strategy

**Timing:**
- Settlement runs every **5-10 minutes** (configurable: `SETTLEMENT_INTERVAL_MINUTES`)
- No maximum retry limit - keep trying until all trades settled
- Batch size: 50 trades per cycle (configurable: `SETTLEMENT_BATCH_SIZE`)

**Behavior:**
- Trades remain in "pending" state (`is_win = NULL`) until successfully processed
- Each settlement cycle picks up all pending trades older than 15 minutes
- Successful settlements are removed from queue
- Failed settlements stay in queue for next cycle

**Monitoring:**
- Track settlement lag: `settlement_time - (trade_time + 15 minutes)`
- Alert if lag exceeds 1 hour (indicates persistent failure)
- Log settlement attempts per trade (detect stuck trades)

## Implementation Details

### New File: `polymarket/performance/settler.py`

```python
"""Trade settlement service for determining win/loss outcomes."""

import structlog
from datetime import datetime, timedelta
from typing import Dict, Optional
import re

from polymarket.performance.database import PerformanceDatabase

logger = structlog.get_logger()


class TradeSettler:
    """Settles trades by comparing BTC prices at market close."""

    def __init__(self, db: PerformanceDatabase, btc_fetcher):
        """
        Initialize trade settler.

        Args:
            db: Performance database
            btc_fetcher: BTC price service (reused from bot)
        """
        self.db = db
        self.btc_fetcher = btc_fetcher

    async def settle_pending_trades(self, batch_size: int = 50) -> Dict:
        """
        Settle all pending trades that have closed.

        Args:
            batch_size: Max trades to process per cycle

        Returns:
            Settlement statistics
        """
        # Implementation details in plan

    def _parse_market_close_timestamp(self, market_slug: str) -> Optional[int]:
        """
        Extract Unix timestamp from market slug.

        Args:
            market_slug: Format "btc-updown-15m-1770828300"

        Returns:
            Unix timestamp or None if parsing fails
        """
        # Implementation details in plan

    async def _get_btc_price_at_timestamp(self, timestamp: int) -> Optional[float]:
        """
        Fetch BTC price at specific timestamp.

        Args:
            timestamp: Unix timestamp

        Returns:
            BTC price or None if unavailable
        """
        # Implementation details in plan

    def _determine_outcome(
        self,
        btc_close_price: float,
        price_to_beat: float
    ) -> str:
        """
        Determine which outcome won (YES or NO).

        Args:
            btc_close_price: BTC price at market close
            price_to_beat: Baseline BTC price from cycle start

        Returns:
            "YES" if UP won, "NO" if DOWN won
        """
        if btc_close_price > price_to_beat:
            return "YES"  # UP won
        else:
            return "NO"   # DOWN won (includes tie)

    def _calculate_profit_loss(
        self,
        action: str,
        actual_outcome: str,
        position_size: float,
        executed_price: float
    ) -> tuple[float, bool]:
        """
        Calculate profit/loss for a settled trade.

        Args:
            action: "YES" or "NO"
            actual_outcome: "YES" or "NO"
            position_size: Dollar amount invested
            executed_price: Price paid per share

        Returns:
            (profit_loss, is_win)
        """
        shares = position_size / executed_price

        if (action == "YES" and actual_outcome == "YES") or \
           (action == "NO" and actual_outcome == "NO"):
            # Win
            payout = shares * 1.00
            profit_loss = payout - position_size
            is_win = True
        else:
            # Loss
            profit_loss = -position_size
            is_win = False

        return profit_loss, is_win
```

### Modified: `polymarket/performance/tracker.py`

Add method for updating trade outcomes:

```python
def update_trade_outcome(
    self,
    trade_id: int,
    actual_outcome: str,
    profit_loss: float,
    is_win: bool
) -> None:
    """
    Update trade record with settlement outcome.

    Args:
        trade_id: Trade ID to update
        actual_outcome: "YES" or "NO"
        profit_loss: Dollar profit/loss
        is_win: Whether trade won
    """
    cursor = self.db.conn.cursor()

    cursor.execute("""
        UPDATE trades
        SET actual_outcome = ?,
            profit_loss = ?,
            is_win = ?
        WHERE id = ?
    """, (actual_outcome, profit_loss, is_win, trade_id))

    self.db.conn.commit()

    logger.info(
        "Trade outcome updated",
        trade_id=trade_id,
        outcome=actual_outcome,
        is_win=is_win,
        profit_loss=f"${profit_loss:.2f}"
    )
```

### Modified: `scripts/auto_trade.py`

**Import and Initialize:**

```python
from polymarket.performance.settler import TradeSettler

# In __init__:
self.trade_settler = TradeSettler(
    db=self.performance_tracker.db,
    btc_fetcher=self.btc_service  # Reuse existing BTC price service
)
```

**Start Settlement Loop:**

```python
async def _run_settlement_loop(self):
    """Background loop for settling trades."""
    interval_seconds = 600  # 10 minutes (configurable)

    logger.info(
        "Settlement loop started",
        interval_minutes=interval_seconds // 60
    )

    while True:
        try:
            stats = await self.trade_settler.settle_pending_trades()

            if stats["settled_count"] > 0:
                logger.info(
                    "Settlement cycle complete",
                    settled=stats["settled_count"],
                    wins=stats["wins"],
                    losses=stats["losses"],
                    pending=stats["pending_count"]
                )

            # Check for stuck trades
            if stats["pending_count"] > 0 and stats["settled_count"] == 0:
                logger.warning(
                    "No trades settled but pending exist",
                    pending_count=stats["pending_count"]
                )

        except Exception as e:
            logger.error("Settlement loop error", error=str(e))

        await asyncio.sleep(interval_seconds)

# In main():
asyncio.create_task(self._run_settlement_loop())
```

## Configuration

**New Environment Variables:**

```bash
# Settlement Configuration
SETTLEMENT_INTERVAL_MINUTES=10  # How often to run settlement
SETTLEMENT_BATCH_SIZE=50        # Max trades per cycle
SETTLEMENT_ALERT_LAG_HOURS=1    # Alert if trade pending this long
```

**Settings Class:**

```python
settlement_interval_minutes: int = 10
settlement_batch_size: int = 50
settlement_alert_lag_hours: int = 1
```

## Testing Strategy

**1. Unit Tests (`tests/test_settler.py`)**
- Test timestamp parsing from market slugs
- Test outcome determination (UP vs DOWN)
- Test profit/loss calculations (win/loss scenarios)
- Test error handling (missing data, invalid formats)

**2. Integration Tests**
- Test with real database
- Test BTC price fetching
- Test settlement of multiple trades
- Test retry behavior on failures

**3. End-to-End Test**
- Execute a real trade
- Wait 15+ minutes
- Verify settlement occurs automatically
- Check database updates correctly

**4. Dry Run**
- Run settlement on existing 36 unsettled trades
- Verify all can be settled
- Check profit/loss accuracy

## Success Metrics

- **Settlement rate**: 100% of trades settled within 30 minutes of market close
- **Settlement lag**: Average <15 minutes from market close to settlement
- **Accuracy**: Profit/loss calculations match expected Polymarket payouts
- **Reliability**: Zero trades stuck in pending state for >1 hour
- **Self-reflection enabled**: System can now analyze wins/losses

## Rollout Plan

**Phase 1: Implementation (Day 1)**
- Create `settler.py` with core logic
- Add database update method
- Write unit tests

**Phase 2: Integration (Day 1)**
- Integrate into `auto_trade.py`
- Add configuration settings
- Test with dry run on existing trades

**Phase 3: Monitoring (Day 2)**
- Deploy with 10-minute interval
- Monitor settlement lag and success rate
- Verify self-reflection system activates

**Phase 4: Optimization (Day 3+)**
- Tune interval based on performance
- Add Telegram notifications for settlement stats
- Consider reducing interval to 5 minutes

## Dependencies

- Existing BTC price fetcher (reused from bot)
- Performance database with `trades` table
- Market slugs with embedded timestamps

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| BTC price service unavailable | Settlements blocked | Retry indefinitely, alert if >1hr pending |
| Timestamp parsing fails | Can't determine close time | Log error, mark as UNKNOWN, alert |
| Database transaction fails | Data inconsistency | Rollback, retry next cycle |
| Settlement loop crashes | No new settlements | Integrate into main bot loop, restart-safe |

## Future Enhancements

1. **Telegram notifications**: Send daily settlement summary
2. **Faster settlement**: Reduce interval to 5 minutes after validation
3. **Settlement dashboard**: Show pending vs settled trades
4. **Historical backfill**: Settle any trades missed during downtime
5. **Verification mode**: Optionally cross-check against Polymarket API

---

**Next Steps**: Create implementation plan with detailed code and tests.
