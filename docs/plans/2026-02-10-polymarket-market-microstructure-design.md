# Polymarket Market Microstructure Redesign

**Date:** 2026-02-10
**Status:** Design Approved
**Context:** Replace Binance API with Polymarket's own CLOB Market Channel for more relevant 15-minute trading signals

## Problem Statement

Current `MarketMicrostructureService` analyzes Binance BTC/USDT order flow, which has two issues:

1. **Binance API is unreachable** from our server (regional blocking, timeouts)
2. **Wrong signal source** - We trade Polymarket predictions, not spot BTC. Analyzing the actual prediction market we're betting on is more relevant.

## Solution Overview

Redesign `MarketMicrostructureService` to analyze **Polymarket's own order flow** for the specific BTC 15-minute prediction market we're actively trading.

**Key Insight:** If the market "Will BTC be above $95,500 at 3:45 PM?" sees YES token price rising with large buyers, that's a direct bullish signal for our decision - much more relevant than Binance spot market activity.

---

## Architecture

### High-Level Flow

1. At the start of each trading cycle (every 3 minutes), connect to Polymarket's CLOB Market Channel WebSocket
2. Subscribe to the specific market (condition_id from the discovered market)
3. Collect market data for **2 minutes**: order book snapshots, trade executions, price updates
4. Disconnect and analyze the accumulated data
5. Calculate three weighted scores:
   - **YES price momentum** (40%): How much did YES token price move?
   - **Volume flow** (35%): Net buying pressure (YES volume - NO volume)
   - **Whale activity** (25%): Large trades (>$1,000) directional signal
6. Return `MarketSignals` with combined score (-1.0 to +1.0) and confidence

### Component Diagram

```
┌─────────────────────────────────────────────────┐
│          Auto Trade Main Loop                    │
│                                                  │
│  1. Discover BTC 15-min market                   │
│  2. Extract condition_id                         │
│  3. Pass to MarketMicrostructureService          │
└─────────────────┬───────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────┐
│    MarketMicrostructureService                   │
│                                                  │
│  ┌───────────────────────────────────┐          │
│  │  WebSocket Data Collector          │          │
│  │  - Connect to CLOB WSS             │          │
│  │  - Subscribe to condition_id       │          │
│  │  - Collect for 2 minutes           │          │
│  │  - Accumulate trades & prices      │          │
│  └───────────────┬───────────────────┘          │
│                  │                               │
│                  ▼                               │
│  ┌───────────────────────────────────┐          │
│  │  Signal Calculator                 │          │
│  │  - Momentum score (40%)            │          │
│  │  - Volume flow score (35%)         │          │
│  │  - Whale activity score (25%)      │          │
│  └───────────────┬───────────────────┘          │
│                  │                               │
│                  ▼                               │
│  ┌───────────────────────────────────┐          │
│  │  Return MarketSignals              │          │
│  │  - score: -1.0 to +1.0             │          │
│  │  - confidence: 0.0 to 1.0          │          │
│  │  - metadata                        │          │
│  └───────────────────────────────────┘          │
└─────────────────────────────────────────────────┘
```

---

## Data Collection Layer

### WebSocket Connection

- **Endpoint:** Polymarket CLOB WebSocket (exact URL TBD from docs, likely `wss://clob.polymarket.com/ws`)
- **Authentication:** Not required for public market data
- **Subscribe to:** `market` channel with the specific `condition_id`

### 2-Minute Collection Window

```python
async def collect_market_data(condition_id: str, duration_seconds: int = 120):
    """
    Connect to WebSocket, collect market data for specified duration.

    Returns:
        {
            'trades': [...],           # last_trade_price messages
            'book_snapshots': [...],   # book messages
            'price_changes': [...]     # price_change messages
        }
    """
    accumulated_data = {
        'trades': [],
        'book_snapshots': [],
        'price_changes': []
    }

    async with websocket.connect(url) as ws:
        # Send subscription message
        await ws.send(json.dumps({
            "action": "subscribe",
            "subscriptions": [
                {"topic": "market", "condition_id": condition_id}
            ]
        }))

        start_time = time.time()
        while time.time() - start_time < duration_seconds:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
                data = json.loads(msg)

                # Accumulate based on message type
                if data['type'] == 'last_trade_price':
                    accumulated_data['trades'].append(data)
                elif data['type'] == 'book':
                    accumulated_data['book_snapshots'].append(data)
                elif data['type'] == 'price_change':
                    accumulated_data['price_changes'].append(data)

            except asyncio.TimeoutError:
                continue  # No message in 5s, keep waiting

    return accumulated_data
```

### Message Types We Track

1. **`last_trade_price`**:
   - Actual executions with `price`, `size`, `side` (BUY/SELL)
   - `asset_id` identifies YES or NO token
   - Used for: Volume flow, whale detection, momentum

2. **`book`**:
   - Order book snapshots with bid/ask levels
   - Optional for depth analysis
   - Used for: Order book imbalance (if needed)

3. **`price_change`**:
   - Individual price level updates
   - Includes `best_bid` and `best_ask` context
   - Used for: Momentum tracking

---

## Signal Calculation

After collecting 2 minutes of data, we calculate three independent scores and combine them with weights.

### 1. YES Price Momentum Score (40% weight)

**Logic:** Track YES token price movement over the collection window.

```python
def calculate_momentum_score(trades: list) -> float:
    """
    Calculate YES token price momentum.

    Returns: -1.0 (strong bearish) to +1.0 (strong bullish)
    """
    if not trades:
        return 0.0

    # Get YES token trades
    yes_trades = [t for t in trades if t['asset_id'] == YES_TOKEN_ID]
    if len(yes_trades) < 2:
        return 0.0

    # First and last YES price in window
    initial_yes_price = yes_trades[0]['price']
    final_yes_price = yes_trades[-1]['price']

    # Calculate percentage change
    price_change_pct = (final_yes_price - initial_yes_price) / initial_yes_price

    # Normalize: ±10% change maps to ±1.0 score
    # Clamp to [-1.0, 1.0] range
    momentum_score = max(min(price_change_pct * 10, 1.0), -1.0)

    return momentum_score
```

**Interpretation:**
- YES price rising → positive score (bullish)
- YES price falling → negative score (bearish)
- 10% price increase → +1.0 score
- 5% price increase → +0.5 score

### 2. Volume Flow Score (35% weight)

**Logic:** Sum YES purchases vs NO purchases, weighted by trade size.

```python
def calculate_volume_flow_score(trades: list) -> float:
    """
    Calculate net buying pressure (YES volume - NO volume).

    Returns: -1.0 (all NO buying) to +1.0 (all YES buying)
    """
    if not trades:
        return 0.0

    yes_volume = sum(
        trade['size'] for trade in trades
        if trade['asset_id'] == YES_TOKEN_ID
    )

    no_volume = sum(
        trade['size'] for trade in trades
        if trade['asset_id'] == NO_TOKEN_ID
    )

    total_volume = yes_volume + no_volume
    if total_volume == 0:
        return 0.0

    # Already normalized to -1.0 to +1.0
    volume_flow_score = (yes_volume - no_volume) / total_volume

    return volume_flow_score
```

**Interpretation:**
- Net YES buying → positive score (bullish)
- Net NO buying → negative score (bearish)
- All YES volume → +1.0 score
- 60% YES, 40% NO → +0.2 score

### 3. Whale Activity Score (25% weight)

**Logic:** Count large trades (>$1,000) by direction.

```python
def calculate_whale_activity_score(trades: list, whale_threshold: float = 1000.0) -> float:
    """
    Calculate directional signal from whale trades (>$1,000).

    Returns: -1.0 (all NO whales) to +1.0 (all YES whales)
    """
    if not trades:
        return 0.0

    # Identify whale trades (size > $1,000)
    yes_whales = sum(
        1 for trade in trades
        if trade['size'] > whale_threshold and trade['asset_id'] == YES_TOKEN_ID
    )

    no_whales = sum(
        1 for trade in trades
        if trade['size'] > whale_threshold and trade['asset_id'] == NO_TOKEN_ID
    )

    total_whales = yes_whales + no_whales
    if total_whales == 0:
        return 0.0

    whale_score = (yes_whales - no_whales) / total_whales

    return whale_score
```

**Interpretation:**
- YES whales dominating → positive score (bullish conviction)
- NO whales dominating → negative score (bearish conviction)
- $1,000+ threshold = meaningful position size at 50¢ odds (2,000 shares)

### 4. Combined Score

```python
def calculate_market_score(momentum: float, volume_flow: float, whale: float) -> float:
    """
    Combine three scores with weights.

    Returns: -1.0 (strong bearish) to +1.0 (strong bullish)
    """
    WEIGHTS = {
        'momentum': 0.40,
        'volume_flow': 0.35,
        'whale': 0.25
    }

    market_score = (
        momentum * WEIGHTS['momentum'] +
        volume_flow * WEIGHTS['volume_flow'] +
        whale * WEIGHTS['whale']
    )

    return market_score
```

**Weight Rationale:**
- **Momentum (40%)**: Market's aggregate signal, highest weight
- **Volume Flow (35%)**: Actual money deployment, strong confirmation
- **Whale Activity (25%)**: Conviction signal, meaningful but lower weight

---

## Integration & Error Handling

### Drop-in Replacement

The new service maintains the same interface as the Binance version:

```python
class MarketMicrostructureService:
    """Polymarket market microstructure analysis using CLOB WebSocket."""

    def __init__(self, settings: Settings, condition_id: str):
        """
        Args:
            settings: Bot configuration
            condition_id: The Polymarket condition_id to analyze
        """
        self.settings = settings
        self.condition_id = condition_id
        self._ws_url = "wss://clob.polymarket.com/ws"  # TBD from docs

    async def get_market_score(self) -> MarketSignals:
        """
        Get current market microstructure score.

        Returns:
            MarketSignals with score, confidence, and detailed metrics.
        """
        # Collect data for 2 minutes
        data = await self.collect_market_data(self.condition_id, duration_seconds=120)

        # Calculate scores
        momentum = self.calculate_momentum_score(data['trades'])
        volume_flow = self.calculate_volume_flow_score(data['trades'])
        whale = self.calculate_whale_activity_score(data['trades'])

        # Combine
        market_score = self.calculate_market_score(momentum, volume_flow, whale)

        # Calculate confidence
        confidence = self.calculate_confidence(data)

        return MarketSignals(
            score=market_score,
            confidence=confidence,
            # ... metadata fields
        )

    async def close(self):
        """Cleanup WebSocket connections."""
        # No persistent connections, nothing to close
        pass
```

### Changes Needed in `auto_trade.py`

**Before:**
```python
self.market_service = MarketMicrostructureService(settings)
```

**After:**
```python
# In run_cycle(), after market discovery:
market = self.client.discover_btc_15min_market()
condition_id = market.condition_id  # Extract from discovered market

# Initialize service with condition_id
self.market_service = MarketMicrostructureService(settings, condition_id)
```

**Alternative (if reusing service):**
Add `set_market()` method to update condition_id between cycles without recreating service.

### Error Handling & Graceful Degradation

#### 1. WebSocket Connection Fails

```python
try:
    data = await self.collect_market_data(condition_id, 120)
except (ConnectionError, TimeoutError, websockets.exceptions.WebSocketException) as e:
    logger.error("WebSocket connection failed", error=str(e))
    return MarketSignals(
        score=0.0,
        confidence=0.0,
        signal_type="MARKET_UNAVAILABLE",
        # ... neutral metadata
    )
```

**System behavior:** Falls back to social sentiment only (with 0.7x confidence penalty in aggregator).

#### 2. Insufficient Data Collected

```python
def calculate_confidence(self, data: dict) -> float:
    """
    Calculate confidence based on data quality.

    Returns: 0.0 to 1.0
    """
    trade_count = len(data['trades'])

    # Base confidence from trade volume
    # 50+ trades = full confidence, scales linearly
    base_confidence = min(trade_count / 50, 1.0)

    # Penalty if didn't collect full 2 minutes
    collection_time = data.get('collection_duration', 120)
    if collection_time < 120:
        base_confidence *= (collection_time / 120)

    # Penalty for low liquidity
    if trade_count < 10:
        logger.warning("Low liquidity", trades=trade_count)
        base_confidence *= 0.5

    return base_confidence
```

**Signal type includes warning:** `"LOW_LIQUIDITY_WEAK_BULLISH"`

#### 3. Market Not Found

```python
# Before collecting data
if not self.is_market_active(condition_id):
    logger.warning("Market closed or not found", condition_id=condition_id)
    return MarketSignals(
        score=0.0,
        confidence=0.0,
        signal_type="MARKET_CLOSED",
        # ...
    )
```

### Confidence Calculation Summary

```python
confidence = (
    base_confidence          # 0.0 to 1.0 from trade count
    * collection_time_ratio  # Penalty if <120 seconds
    * liquidity_penalty      # 0.5x if <10 trades
)
```

**Quality tiers:**
- **High confidence (>0.7):** 50+ trades, full 2-minute window
- **Medium confidence (0.5-0.7):** 25-50 trades or partial window
- **Low confidence (<0.5):** <25 trades or significant data loss

---

## Testing Strategy

### Unit Tests

```python
# tests/test_market_microstructure.py

def test_momentum_score():
    """Test YES price momentum calculation."""
    # Test: YES price rising 5% → positive score
    trades = [
        {'asset_id': YES_TOKEN, 'price': 0.50, 'size': 100},
        {'asset_id': YES_TOKEN, 'price': 0.525, 'size': 200},
    ]
    score = calculate_momentum_score(trades)
    assert score == pytest.approx(0.5, abs=0.01)  # 5% / 10% = 0.5

    # Test: YES price falling 10% → -1.0 (clamped)
    trades = [
        {'asset_id': YES_TOKEN, 'price': 0.50, 'size': 100},
        {'asset_id': YES_TOKEN, 'price': 0.45, 'size': 200},
    ]
    score = calculate_momentum_score(trades)
    assert score == -1.0

    # Test: No price change → 0.0
    trades = [
        {'asset_id': YES_TOKEN, 'price': 0.50, 'size': 100},
        {'asset_id': YES_TOKEN, 'price': 0.50, 'size': 200},
    ]
    score = calculate_momentum_score(trades)
    assert score == 0.0


def test_volume_flow_score():
    """Test volume flow calculation."""
    # Test: Pure YES volume → +1.0
    trades = [
        {'asset_id': YES_TOKEN, 'size': 1000},
        {'asset_id': YES_TOKEN, 'size': 500},
    ]
    score = calculate_volume_flow_score(trades)
    assert score == 1.0

    # Test: Pure NO volume → -1.0
    trades = [
        {'asset_id': NO_TOKEN, 'size': 1000},
    ]
    score = calculate_volume_flow_score(trades)
    assert score == -1.0

    # Test: Equal volume → 0.0
    trades = [
        {'asset_id': YES_TOKEN, 'size': 500},
        {'asset_id': NO_TOKEN, 'size': 500},
    ]
    score = calculate_volume_flow_score(trades)
    assert score == 0.0


def test_whale_activity_score():
    """Test whale detection and scoring."""
    # Test: All YES whales → +1.0
    trades = [
        {'asset_id': YES_TOKEN, 'size': 1500},
        {'asset_id': YES_TOKEN, 'size': 2000},
        {'asset_id': NO_TOKEN, 'size': 100},  # Not a whale
    ]
    score = calculate_whale_activity_score(trades, whale_threshold=1000)
    assert score == 1.0

    # Test: Balanced whales → 0.0
    trades = [
        {'asset_id': YES_TOKEN, 'size': 1500},
        {'asset_id': NO_TOKEN, 'size': 1200},
    ]
    score = calculate_whale_activity_score(trades, whale_threshold=1000)
    assert score == 0.0

    # Test: No whales → 0.0 (handle division by zero)
    trades = [
        {'asset_id': YES_TOKEN, 'size': 500},
        {'asset_id': NO_TOKEN, 'size': 800},
    ]
    score = calculate_whale_activity_score(trades, whale_threshold=1000)
    assert score == 0.0
```

### Integration Tests

```python
@pytest.mark.asyncio
async def test_market_microstructure_collection():
    """Test WebSocket data collection with mocked responses."""
    # Mock WebSocket server
    async with create_mock_ws_server() as mock_server:
        # Send realistic trade messages
        await mock_server.send(json.dumps({
            'type': 'last_trade_price',
            'asset_id': YES_TOKEN,
            'price': 0.52,
            'size': 1500,
            'timestamp': 1234567890
        }))

        service = MarketMicrostructureService(settings, condition_id='test')
        signals = await service.get_market_score()

        # Verify score calculated correctly
        assert -1.0 <= signals.score <= 1.0
        assert 0.0 <= signals.confidence <= 1.0
        assert signals.signal_type != "MARKET_UNAVAILABLE"


@pytest.mark.asyncio
async def test_error_handling():
    """Test graceful degradation on WebSocket failure."""
    # Mock connection failure
    with patch('websocket.connect', side_effect=ConnectionError):
        service = MarketMicrostructureService(settings, condition_id='test')
        signals = await service.get_market_score()

        # Should return neutral with zero confidence
        assert signals.score == 0.0
        assert signals.confidence == 0.0
        assert signals.signal_type == "MARKET_UNAVAILABLE"
```

### Live Testing Approach

1. **Run in `read_only` mode for 1 day**
   - Log all scores alongside current system
   - No actual trades executed

2. **Compare signal quality:**
   - Correlation with actual market outcomes
   - Confidence distribution (should have meaningful variance)
   - Error rate (WebSocket failures)

3. **Monitoring metrics:**
   - Trade count per 2-minute window (should average 30-50)
   - WebSocket connection success rate (target: >95%)
   - Score vs actual outcome correlation (target: >55%)

### Success Criteria

- ✅ **Scores correlate with market direction** (>55% accuracy on YES/NO outcomes)
- ✅ **Confidence varies meaningfully** (not always 0.0 or 1.0, should follow trade volume)
- ✅ **<5% WebSocket error rate** in production
- ✅ **Integration tests pass** with mocked WebSocket data
- ✅ **System handles errors gracefully** (falls back to social sentiment only)

---

## Implementation Checklist

### Phase 1: Core Service (Tasks 1-5)
- [ ] Find CLOB WebSocket endpoint URL from Polymarket docs
- [ ] Implement WebSocket connection and subscription logic
- [ ] Implement 2-minute data collection window
- [ ] Add trade message parsing (last_trade_price)
- [ ] Implement momentum, volume flow, whale scoring functions

### Phase 2: Integration (Tasks 6-7)
- [ ] Update `auto_trade.py` to pass condition_id
- [ ] Update `MarketSignals` model if needed (add new metadata fields)
- [ ] Implement error handling and graceful degradation
- [ ] Add confidence calculation based on data quality

### Phase 3: Testing (Tasks 8-10)
- [ ] Write unit tests for all scoring functions
- [ ] Write integration tests with mocked WebSocket
- [ ] Run live testing in read_only mode for 1 day
- [ ] Validate success criteria

### Phase 4: Documentation & Cleanup
- [ ] Update `docs/SENTIMENT-ANALYSIS.md` with new approach
- [ ] Remove old Binance code and tests
- [ ] Final commit and merge

---

## Open Questions

1. **What is the exact CLOB WebSocket URL?**
   - Need to find from Polymarket documentation
   - Likely `wss://clob.polymarket.com/ws` or similar

2. **How to extract condition_id from discovered market?**
   - Check Market model for `condition_id` field
   - May need to add to discovery method

3. **Should we cache condition_id between cycles?**
   - If same market for entire session, initialize once
   - If market changes (new 15-min window), update dynamically

4. **Fallback if specific market has no activity?**
   - Return neutral with low confidence
   - System will rely on social sentiment + technical indicators

---

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Low liquidity on 15-min markets | Confidence penalty (<10 trades), fall back to social sentiment |
| WebSocket connection failures | Error handling, return neutral, log for investigation |
| Market closes mid-collection | Check market status before collecting, return MARKET_CLOSED |
| Wrong condition_id subscribed | Validate condition_id exists before connecting |
| 2-minute collection too slow | Make duration configurable (default 120s, can reduce to 60s) |

---

## Future Enhancements

1. **Multi-market aggregation**: If multiple active BTC markets, aggregate signals
2. **Order book depth analysis**: Add bid/ask imbalance as 4th signal (10% weight)
3. **Historical comparison**: Compare current activity to past hour/day baselines
4. **Adaptive weights**: Learn optimal weights based on historical accuracy

---

## References

- Polymarket CLOB WebSocket docs: https://docs.polymarket.com/developers/CLOB/websocket/market-channel
- Current Binance implementation: `polymarket/trading/market_microstructure.py`
- Sentiment analysis architecture: `docs/SENTIMENT-ANALYSIS.md`
