# Just-In-Time Price Fetching with FOK Orders

**Date**: 2026-02-11
**Status**: Approved
**Priority**: High (fixes order fill failures)

## Problem Statement

Orders fail to fill because prices are fetched at cycle start and used 2-3 minutes later for execution. By the time the bot places orders, prices have moved and GTC limit orders sit on the order book unfilled.

### Current Workflow Timeline
```
00:00 - Fetch market.best_ask=0.89, best_bid=0.87
00:00 - Start 120s market microstructure collection
02:00 - Market microstructure complete
02:10 - Fetch social sentiment, technical analysis
02:30 - AI decision (30-60s with reasoning)
03:20 - Place order with STALE price from 00:00
```

### Root Cause
- Prices fetched at cycle start (T+0)
- Order placed at T+3min with stale prices
- Using GTC limit orders (not true market orders)
- No safety checks for price movement

## Solution Design

### Architecture Overview

```
1. AI Decision Made (with snapshot prices from cycle start)
   ↓
2. Fetch Fresh Market Data (RIGHT before execution)
   ↓
3. Adaptive Safety Check
   - Calculate price movement since analysis
   - If moved >10% WORSE → Skip trade
   - If moved >5% BETTER → Log warning, proceed
   - Otherwise → Proceed normally
   ↓
4. Execute FOK Market Order
   - Use fresh prices
   - Fill-or-Kill for guaranteed execution
   ↓
5. Log Execution Metrics
   - Track slippage
   - Feed into self-reflection system
```

### Key Components

#### 1. Fresh Price Fetching

**New Method**: `_get_fresh_market_data(market_id: str) -> Market`

Fetches current market data immediately before order execution:
- Called after AI decision, before order creation
- Returns Market object with fresh best_bid/best_ask
- Logs elapsed time since analysis
- Raises exception if market not found

**Integration Point**: `auto_trade.py:_execute_trade()`

#### 2. Adaptive Safety Checks

**New Method**: `_analyze_price_movement(...) -> tuple[bool, str]`

Determines if price movement is favorable or unfavorable:

- **Favorable** (price decreased = better for buyer):
  - If improvement >5%: Log warning, proceed
  - Otherwise: Proceed normally

- **Unfavorable** (price increased = worse for buyer):
  - If deterioration >10%: Skip trade
  - Otherwise: Proceed

Returns: `(should_execute: bool, reason: str)`

#### 3. FOK Market Orders

**Changes to**: `client.py:create_order()`

Replace GTC limit orders with true FOK market orders:

```python
if request.order_type == "market":
    market_order_args = MarketOrderArgs(
        token_id=request.token_id,
        amount=float(request.size),
        side=request.side
    )
    signed_order = client.create_market_order(market_order_args)
    result = client.post_order(signed_order, OrderType.FOK)
```

**Order Type**: FOK (Fill-or-Kill)
- Executes immediately at best available price
- If can't fill completely, cancels order
- Guaranteed immediate execution or cancellation

#### 4. Configuration

**New Settings** (`.env`):
```bash
TRADE_MAX_UNFAVORABLE_MOVE_PCT=10.0  # Skip if price 10% worse
TRADE_MAX_FAVORABLE_WARN_PCT=5.0     # Warn if price 5% better
```

**Settings Class** (`config.py`):
```python
trade_max_unfavorable_move_pct: float = 10.0
trade_max_favorable_warn_pct: float = 5.0
```

#### 5. Performance Tracking

**Enhanced Logging** for self-reflection:
```python
trade_data = {
    # ... existing fields ...
    "analysis_price": float,
    "execution_price": float,
    "price_staleness_seconds": int,
    "price_slippage_pct": float,
    "price_movement_favorable": bool,
    "skipped_unfavorable_move": bool,
}
```

### Self-Reflection Integration

The self-reflection system can analyze:

1. **Skipped trades** - Were we right to skip?
2. **Slippage patterns** - Are thresholds optimal?
3. **Fill success rate** - FOK vs GTC comparison

**Recommendation triggers**:
- If 30%+ trades skipped but market moved favorably → Loosen thresholds
- If high slippage but good win rate → Current thresholds good
- If low slippage but missing profits → Tighten thresholds

## Implementation Plan

### Phase 1: Fresh Price Fetching
1. Add `_get_fresh_market_data()` method
2. Call it before order execution
3. Add logging for price staleness

### Phase 2: Safety Checks
1. Add `_analyze_price_movement()` method
2. Integrate into `_execute_trade()`
3. Add configuration settings

### Phase 3: FOK Orders
1. Enable FOK market orders in `client.py`
2. Remove GTC limit order workaround
3. Add error handling for FOK failures

### Phase 4: Performance Tracking
1. Add new fields to performance database
2. Log execution metrics
3. Update self-reflection analysis

## Success Metrics

- **Fill rate**: Increase from ~60% to >95%
- **Slippage**: Track avg slippage per trade
- **Skipped trades**: Monitor false positive rate
- **Win rate**: No degradation from better fills

## Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| FOK slippage in thin markets | 10% threshold prevents worst cases |
| Missing profitable trades | Self-reflection tunes thresholds |
| API latency for fresh prices | <1s fetch, negligible vs 3min analysis |
| Market moves during fetch | Race condition window ~500ms, acceptable |

## Testing Strategy

1. **Unit tests**: Price movement logic
2. **Integration tests**: Fresh price fetching
3. **Dry run**: Monitor skipped trades
4. **Live test**: Small position sizes
5. **Monitor**: Fill rates, slippage, win rate

## Rollout Plan

1. Deploy to production with `DRY_RUN=true`
2. Monitor for 24 hours, check logs
3. Enable live trading with small positions
4. Gradually increase to normal position sizes
5. Monitor self-reflection recommendations

## Related Documents

- Self-reflection system: `docs/plans/2026-02-11-self-reflection-implementation.md`
- Performance tracking: `polymarket/performance/tracker.py`
- CLOB client docs: https://github.com/polymarket/py-clob-client
