# Order Verification and P&L Tracking System

**Status:** Design Phase
**Created:** 2026-02-14
**Author:** Claude Code

## Executive Summary

This document presents a comprehensive design for adding order verification and accurate P&L tracking to the Polymarket trading bot. The current system calculates P&L mathematically without verifying against the Polymarket API, which can lead to discrepancies between calculated performance and actual payouts.

## Problem Statement

### Current System Limitations

1. **No Order Verification**: System assumes orders fill at estimated prices without API confirmation
2. **Unverified Fill Prices**: Uses estimated prices instead of actual fill prices from Polymarket
3. **No Audit Trail**: Missing transaction hashes and order confirmation data
4. **Blind Settlement**: Doesn't verify market resolution status before calculating outcomes
5. **Phantom Trades**: Could show trades that were never actually executed
6. **Price Discrepancies**: Slippage and partial fills not reflected in P&L calculations
7. **Reconciliation Gaps**: No way to reconcile against actual USDC balance changes

### Recent Context

- Settlement bug recently fixed (now only settles trades with `execution_status = 'executed'`)
- Database already has `order_id` field for linking to Polymarket orders
- Settlement runs every cycle for trades >15 minutes old
- System tracks execution metadata (filled_via, limit_order_timeout, etc.)

## Design Options

### Option 1: Immediate Verification (Aggressive)

**When:** Verify immediately after order execution (within same cycle)

**Flow:**
```
1. Place order → Get order_id
2. Wait 2-5 seconds for fill
3. Call check_order_status(order_id)
4. Update database with actual fill data
5. Continue to next cycle
```

**Pros:**
- Fastest feedback loop
- Catches execution failures immediately
- Can retry failed orders in same session

**Cons:**
- Adds 2-5 seconds latency per trade
- May check too early for limit orders
- Increases API calls per cycle
- Could delay trading decision for next market

**Best For:** Market orders (FOK) that fill immediately

---

### Option 2: Deferred Verification (Conservative)

**When:** Verify at settlement time (when trade is >15 minutes old)

**Flow:**
```
1. Place order → Get order_id → Store in DB
2. [15+ minutes pass]
3. Settlement cycle runs
4. For each unsettled trade:
   a. Verify order actually filled
   b. Get actual fill price and quantity
   c. Get transaction hash
   d. Verify market resolution
   e. Calculate P&L with actual data
5. Update database
```

**Pros:**
- No impact on trading cycle latency
- Consolidates API calls in settlement batch
- Works well for both market and limit orders
- Natural batching reduces API load

**Cons:**
- Delayed feedback (won't know about failures for 15+ min)
- Can't retry failed orders
- May process stale data

**Best For:** Production systems prioritizing speed

---

### Option 3: Hybrid Two-Phase (Recommended)

**When:** Quick check immediately + full verification at settlement

**Phase 1 - Immediate Check (2 seconds):**
```
1. Place order → Get order_id
2. Wait 2 seconds
3. Quick status check:
   - If MATCHED/PARTIALLY_MATCHED: Update status to 'filled'
   - If LIVE/PENDING: Update status to 'pending'
   - If CANCELLED/FAILED: Update status to 'failed', alert
4. Store order_id and initial status
```

**Phase 2 - Settlement Verification:**
```
1. For trades >15 minutes old with order_id:
   a. Call check_order_status(order_id) to verify final fill
   b. Call get_trades() to get fill history with exact prices
   c. Extract transaction hashes from trade records
   d. Verify market resolution timestamp
   e. Calculate P&L using actual fill prices
   f. Compare to USDC balance delta (optional reconciliation)
2. Update database with verified data
3. Alert on discrepancies
```

**Pros:**
- Fast detection of critical failures
- Full verification without impacting latency
- Best of both worlds
- Can handle partial fills properly

**Cons:**
- More complex implementation
- Two API calls per order (but at different times)

**Best For:** Production systems requiring both speed and accuracy

---

## Recommended Architecture: Option 3 (Hybrid)

### Component Design

#### 1. Order Verification Service

**Location:** `/root/polymarket-scripts/polymarket/performance/order_verifier.py`

```python
class OrderVerifier:
    """Verifies order execution and extracts actual fill data from Polymarket API."""

    def __init__(self, client: PolymarketClient, db: PerformanceDatabase):
        self.client = client
        self.db = db

    async def quick_status_check(self, order_id: str, trade_id: int) -> dict:
        """Phase 1: Quick check immediately after order placement.

        Returns:
            {
                'status': 'filled'|'pending'|'failed',
                'fill_amount': float,
                'needs_alert': bool
            }
        """
        pass

    async def verify_order_execution(self, order_id: str) -> dict:
        """Phase 2: Full verification at settlement time.

        Returns:
            {
                'verified': bool,
                'status': str,
                'fill_amount': float,
                'fill_price': float,
                'transaction_hash': str,
                'fill_timestamp': int,
                'partial_fill': bool
            }
        """
        pass

    async def get_trade_fills(self, order_id: str) -> list[dict]:
        """Get detailed fill history for an order from get_trades() API.

        Returns list of fills with exact prices and timestamps.
        """
        pass
```

#### 2. Enhanced Settlement Service

**Location:** Update `/root/polymarket-scripts/polymarket/performance/settler.py`

```python
class TradeSettler:
    """Enhanced settler with order verification."""

    def __init__(self, db: PerformanceDatabase, btc_fetcher, order_verifier: OrderVerifier):
        self.db = db
        self.btc_fetcher = btc_fetcher
        self.order_verifier = order_verifier  # NEW

    async def settle_pending_trades(self, batch_size: int = 50) -> dict:
        """Settle trades with order verification."""

        for trade in unsettled_trades:
            # NEW: Verify order execution first
            if trade['order_id']:
                verification = await self.order_verifier.verify_order_execution(
                    trade['order_id']
                )

                if not verification['verified']:
                    # Order never filled - mark as failed
                    self._mark_trade_failed(trade['id'], verification)
                    self._alert_verification_failure(trade, verification)
                    continue

                # Update with actual fill data
                actual_price = verification['fill_price']
                actual_size = verification['fill_amount']
                tx_hash = verification['transaction_hash']
            else:
                # Fallback to existing behavior for old trades
                actual_price = trade['executed_price']
                actual_size = trade['position_size']
                tx_hash = None

            # Calculate P&L using ACTUAL fill data
            profit_loss, is_win = self._calculate_profit_loss(
                action=trade['action'],
                actual_outcome=actual_outcome,
                position_size=actual_size,  # Use verified size
                executed_price=actual_price  # Use verified price
            )

            # Store verification data
            self._update_trade_with_verification(
                trade_id=trade['id'],
                actual_outcome=actual_outcome,
                profit_loss=profit_loss,
                is_win=is_win,
                verified_price=actual_price,
                verified_size=actual_size,
                transaction_hash=tx_hash
            )
```

#### 3. Database Schema Updates

**Location:** Update `/root/polymarket-scripts/polymarket/performance/database.py`

Add new columns to `trades` table:

```python
new_columns = [
    # Verification data
    ("verified_fill_price", "REAL"),  # Actual fill price from API
    ("verified_fill_amount", "REAL"),  # Actual fill amount from API
    ("transaction_hash", "TEXT"),  # Blockchain transaction hash
    ("fill_timestamp", "INTEGER"),  # Unix timestamp of fill
    ("partial_fill", "BOOLEAN"),  # Whether order was partially filled
    ("verification_status", "TEXT"),  # 'verified', 'unverified', 'failed'
    ("verification_timestamp", "INTEGER"),  # When verification was performed

    # Discrepancy tracking
    ("price_discrepancy_pct", "REAL"),  # Diff between estimated and actual price
    ("amount_discrepancy_pct", "REAL"),  # Diff between expected and actual fill amount
]
```

#### 4. Alert System

**Location:** `/root/polymarket-scripts/polymarket/performance/alerts.py`

```python
class VerificationAlerts:
    """Alert system for order verification discrepancies."""

    async def alert_order_not_filled(self, trade_id: int, order_id: str):
        """Alert when order shows as unfilled in API."""
        pass

    async def alert_price_mismatch(self, trade_id: int, estimated: float, actual: float):
        """Alert when fill price differs significantly from estimate."""
        pass

    async def alert_partial_fill(self, trade_id: int, expected: float, filled: float):
        """Alert when order only partially fills."""
        pass

    async def alert_verification_failed(self, trade_id: int, error: str):
        """Alert when verification API call fails."""
        pass
```

### Integration Points

#### 1. Auto-Trader Integration

**Location:** Update `/root/polymarket-scripts/scripts/auto_trade.py`

```python
# After order execution
if order_response.order_id:
    # Phase 1: Quick status check (2 seconds)
    await asyncio.sleep(2)
    status = await order_verifier.quick_status_check(
        order_id=order_response.order_id,
        trade_id=trade_id
    )

    if status['status'] == 'failed':
        # Alert immediately
        await alerts.alert_order_not_filled(trade_id, order_response.order_id)
        # Mark as failed
        await tracker.update_trade_status(
            trade_id=trade_id,
            execution_status='failed'
        )
```

#### 2. Settlement Integration

**Location:** Update settlement cycle in auto_trade.py

```python
# During settlement (every cycle)
settlement_stats = await settler.settle_pending_trades(batch_size=50)

# Log verification results
if settlement_stats.get('verification_failures', 0) > 0:
    logger.warning(
        "Verification failures detected",
        failures=settlement_stats['verification_failures']
    )
```

### API Methods Needed

From `py_clob_client`:

1. **`client.get_order(order_id)`** - Already available via `check_order_status()`
   - Returns: status, fillAmount, price, timestamp

2. **`client.get_trades()`** - Already available via `get_portfolio_summary()`
   - Returns: List of trade history with asset_id, side, size, price, timestamp
   - Need to filter by order_id or match by timestamp/amount

3. **Market resolution API** (if available)
   - Verify market has actually resolved before settling
   - Get official resolution outcome

### Handling Edge Cases

#### Partial Fills

```python
if verification['partial_fill']:
    # Recalculate position_size based on actual fill
    actual_position_size = verification['fill_amount'] * verification['fill_price']

    # Alert user
    await alerts.alert_partial_fill(
        trade_id=trade['id'],
        expected=trade['position_size'],
        filled=actual_position_size
    )

    # Calculate P&L on actual filled amount only
    profit_loss, is_win = self._calculate_profit_loss(
        action=trade['action'],
        actual_outcome=actual_outcome,
        position_size=actual_position_size,  # Use partial fill amount
        executed_price=verification['fill_price']
    )
```

#### Order Not Found

```python
if verification['status'] == 'not_found':
    # Order may have expired or been cancelled
    logger.warning(f"Order {order_id} not found in API")

    # Mark as failed in database
    self.db.update_verification_status(
        trade_id=trade['id'],
        verification_status='order_not_found'
    )

    # Alert
    await alerts.alert_verification_failed(trade['id'], "Order not found")

    # Don't calculate P&L - no trade occurred
    continue
```

#### Price Discrepancy

```python
estimated_price = trade['executed_price']
actual_price = verification['fill_price']
discrepancy_pct = abs((actual_price - estimated_price) / estimated_price) * 100

if discrepancy_pct > 5.0:  # Alert threshold: 5%
    await alerts.alert_price_mismatch(
        trade_id=trade['id'],
        estimated=estimated_price,
        actual=actual_price
    )

    # Log discrepancy
    self.db.update_price_discrepancy(
        trade_id=trade['id'],
        price_discrepancy_pct=discrepancy_pct
    )
```

#### Verification API Failure

```python
try:
    verification = await self.order_verifier.verify_order_execution(order_id)
except Exception as e:
    logger.error(f"Verification failed for order {order_id}: {e}")

    # Fall back to estimated data
    verification = {
        'verified': False,
        'fallback_mode': True,
        'fill_price': trade['executed_price'],
        'fill_amount': trade['position_size']
    }

    # Alert about fallback
    await alerts.alert_verification_failed(trade['id'], str(e))
```

### USDC Balance Reconciliation (Optional Enhancement)

```python
class BalanceReconciler:
    """Reconcile calculated P&L against actual USDC balance changes."""

    async def reconcile_session(self, start_balance: float, end_balance: float) -> dict:
        """Compare session P&L to actual balance change.

        Returns:
            {
                'calculated_pnl': float,
                'actual_balance_change': float,
                'discrepancy': float,
                'discrepancy_pct': float,
                'within_tolerance': bool
            }
        """
        # Get all trades in session
        trades = self.db.get_trades_in_timerange(start_time, end_time)

        # Sum up calculated P&L
        calculated_pnl = sum(t['profit_loss'] for t in trades if t['profit_loss'])

        # Compare to actual balance change
        actual_change = end_balance - start_balance
        discrepancy = abs(calculated_pnl - actual_change)

        tolerance = 0.50  # $0.50 tolerance for fees/rounding

        return {
            'calculated_pnl': calculated_pnl,
            'actual_balance_change': actual_change,
            'discrepancy': discrepancy,
            'discrepancy_pct': (discrepancy / max(abs(actual_change), 1)) * 100,
            'within_tolerance': discrepancy < tolerance
        }
```

## Implementation Plan

### Phase 1: Core Verification (Week 1)

1. **Create OrderVerifier service**
   - Implement `quick_status_check()`
   - Implement `verify_order_execution()`
   - Implement `get_trade_fills()`
   - Add retry logic and error handling

2. **Update database schema**
   - Add verification columns
   - Create migration script
   - Add indexes for order_id lookups

3. **Add alert system**
   - Create VerificationAlerts class
   - Integrate with Telegram bot
   - Add Slack integration (if available)

### Phase 2: Settlement Integration (Week 1)

1. **Update TradeSettler**
   - Add OrderVerifier integration
   - Implement verification before P&L calculation
   - Use actual fill prices in calculations
   - Store transaction hashes

2. **Update auto_trade.py**
   - Add Phase 1 quick checks
   - Store order_id from responses
   - Handle immediate failures

### Phase 3: Testing & Validation (Week 2)

1. **Unit tests**
   - Test verification logic
   - Test partial fill handling
   - Test discrepancy detection

2. **Integration tests**
   - Test full verification flow
   - Test with mock Polymarket responses
   - Test error scenarios

3. **Manual validation**
   - Run in test mode for 48 hours
   - Compare verified vs estimated prices
   - Validate transaction hashes on blockchain

### Phase 4: Monitoring & Refinement (Ongoing)

1. **Add metrics dashboard**
   - Verification success rate
   - Average price discrepancy
   - Partial fill rate
   - Alert frequency

2. **Tune thresholds**
   - Adjust price discrepancy alert threshold
   - Optimize quick check timeout
   - Refine retry strategies

## Risk Assessment

### High Risk

1. **API Rate Limits**: Verification adds 2x API calls per order
   - **Mitigation**: Batch verification in settlement, add rate limiting

2. **False Positives**: Over-alerting on normal variation
   - **Mitigation**: Set reasonable thresholds (5% for price, $0.50 for balance)

### Medium Risk

1. **Latency Impact**: Quick checks add 2 seconds per trade
   - **Mitigation**: Make Phase 1 checks optional, configurable

2. **Partial Fill Complexity**: Edge cases with partial fills
   - **Mitigation**: Comprehensive testing, clear documentation

### Low Risk

1. **Database Migration**: Adding new columns
   - **Mitigation**: Standard ALTER TABLE, non-breaking change

## Configuration

Add to `.env`:

```bash
# Order Verification
ENABLE_ORDER_VERIFICATION=true
ENABLE_QUICK_STATUS_CHECK=true
QUICK_CHECK_TIMEOUT_SECONDS=2
PRICE_DISCREPANCY_ALERT_PCT=5.0
ENABLE_BALANCE_RECONCILIATION=false  # Optional feature
```

## Success Metrics

1. **Verification Coverage**: >95% of trades have verified fill data
2. **Price Accuracy**: <2% average discrepancy between estimated and actual
3. **Alert Precision**: <5% false positive rate
4. **Zero Phantom Trades**: All P&L matches actual Polymarket records
5. **Settlement Accuracy**: 100% of settlements use verified data

## Alternatives Considered

### Alternative A: Poll-Based Verification
- Continuously poll order status until filled
- **Rejected**: Wastes API calls, adds complex polling logic

### Alternative B: Settlement-Only Verification
- Only verify at settlement time (no quick checks)
- **Rejected**: Miss critical failures until much later

### Alternative C: Blockchain Event Monitoring
- Monitor blockchain for transaction confirmations
- **Rejected**: Too complex, requires blockchain node access

## Open Questions

1. **Q: Does py_clob_client expose transaction hashes?**
   - A: Need to investigate client.get_trades() response structure

2. **Q: How to link get_trades() records back to order_id?**
   - A: May need to match by timestamp + amount if no direct link

3. **Q: What's the rate limit on get_order() calls?**
   - A: Need to check Polymarket API documentation

4. **Q: Should we verify market resolution status before settling?**
   - A: Yes, add market resolution verification to prevent premature settlement

## Appendix A: API Response Examples

### check_order_status() Response
```json
{
  "orderID": "0x1234...",
  "status": "MATCHED",
  "fillAmount": "10.5",
  "price": "0.65",
  "size": "15.0",
  "timestamp": 1707955200
}
```

### get_trades() Response
```json
[
  {
    "id": "trade_123",
    "asset_id": "token_456",
    "side": "BUY",
    "size": "10.5",
    "price": "0.65",
    "timestamp": 1707955200,
    "order_id": "0x1234...",
    "transaction_hash": "0xabcd..."
  }
]
```

## Appendix B: Database Schema Changes

```sql
-- Migration script
ALTER TABLE trades ADD COLUMN verified_fill_price REAL;
ALTER TABLE trades ADD COLUMN verified_fill_amount REAL;
ALTER TABLE trades ADD COLUMN transaction_hash TEXT;
ALTER TABLE trades ADD COLUMN fill_timestamp INTEGER;
ALTER TABLE trades ADD COLUMN partial_fill BOOLEAN;
ALTER TABLE trades ADD COLUMN verification_status TEXT DEFAULT 'unverified';
ALTER TABLE trades ADD COLUMN verification_timestamp INTEGER;
ALTER TABLE trades ADD COLUMN price_discrepancy_pct REAL;
ALTER TABLE trades ADD COLUMN amount_discrepancy_pct REAL;

-- Index for order_id lookups
CREATE INDEX IF NOT EXISTS idx_trades_order_id ON trades(order_id);
CREATE INDEX IF NOT EXISTS idx_trades_verification_status ON trades(verification_status);
```

## Conclusion

The **Hybrid Two-Phase approach (Option 3)** provides the best balance of speed, accuracy, and reliability. It enables:

1. Fast detection of critical failures (Phase 1)
2. Comprehensive verification without latency impact (Phase 2)
3. Accurate P&L matching actual Polymarket payouts
4. Complete audit trail with transaction hashes
5. Graceful handling of partial fills and edge cases

**Recommendation: Proceed with implementation starting Phase 1 (Core Verification)**

---

**Next Steps:**
1. Review and approve this design
2. Begin Phase 1 implementation (OrderVerifier service)
3. Set up test environment with mock Polymarket responses
4. Plan integration testing strategy
