# Chainlink Migration Guide

## Problem Statement

### Price Source Discrepancy
Our trading bot was using Binance prices from Polymarket's RTDS API, but Polymarket settles markets using Chainlink oracle prices. This caused significant price discrepancies.

**Historical Example:**
- Market: `btc-updown-15m-1771096500` (15-min BTC direction prediction)
- Polymarket settlement price (Chainlink): $69,726.92
- Our bot's price (Binance): $67,257.39
- **Discrepancy: $2,469.53 (3.6%)**

This led to incorrect directional analysis - we calculated price_to_beat using Binance data while the market settled using Chainlink data.

### Signal Weighting Issue
The AI decision system was treating all signals equally, allowing lagging sentiment data to override actual BTC price movement. This caused the bot to bet against clear price direction.

## Solution

### 1. Chainlink Integration
- Modified `CryptoPriceStream` to subscribe to `crypto_prices_chainlink` topic
- Implemented Chainlink message parsing with proper subscription format:
  - Topic: `"crypto_prices_chainlink"`
  - Type: `"*"` (required for Chainlink)
  - Filters: `'{"symbol":"btc/usd"}'` (JSON as string)
- Added price source attribution to all price data
- Enabled Chainlink by default in auto-trader

### 2. Tiered Signal Weighting
Implemented a 4-tier weighted confidence system:

| Tier | Signals | Weight | Priority |
|------|---------|--------|----------|
| 1 | Price Reality (actual BTC movement) | 50% | HIGHEST |
| 2 | Market Structure (volume, orderbook) | 25% | SECONDARY |
| 3 | External Signals (CoinGecko Pro) | 15% | SUPPORTING |
| 4 | Sentiment (social, community) | 10% | LOWEST |

**Conflict Resolution:**
- If sentiment conflicts with price direction → 50% confidence penalty
- Large price moves without volume → false breakout → HOLD
- Price reality ALWAYS overrides opinion-based signals

## Changes Made

### Modified Files
1. **`polymarket/trading/crypto_price_stream.py`**
   - Added `use_chainlink` parameter (default: True)
   - Implemented Chainlink subscription and message parsing
   - Added price source attribution

2. **`polymarket/trading/btc_price.py`**
   - Enabled Chainlink by default

3. **`polymarket/trading/ai_decision.py`**
   - Implemented `_calculate_weighted_confidence()` method
   - Updated system prompt with signal priority hierarchy
   - Added conflict detection and penalty logic

4. **`scripts/auto_trade.py`**
   - Updated to pass full `BTCPriceData` object through execution pipeline

5. **`polymarket/performance/tracker.py`**
   - Updated trade logging to include `price_source` column

### New Files
1. **`scripts/migrations/add_price_source_column.py`**
   - Database migration adding `price_source` column to trades tables

2. **`tests/test_crypto_price_stream_chainlink.py`**
   - Tests for Chainlink subscription and message parsing

3. **`tests/test_signal_weighting.py`**
   - Tests for tiered signal weighting system

4. **`tests/test_chainlink_integration.py`**
   - Integration test documenting the fix

## Migration Steps

### For Existing Users

1. **Pull the latest code:**
   ```bash
   git pull origin main
   ```

2. **Run the database migration:**
   ```bash
   python scripts/migrations/add_price_source_column.py
   ```
   This adds the `price_source` column to track which price source was used for each trade.

3. **Restart the bot:**
   ```bash
   python scripts/auto_trade.py
   ```
   The bot will now use Chainlink prices by default.

### Verification

1. **Check logs for Chainlink connection:**
   ```
   Initializing BTC price service with Chainlink data source
   Connected to Polymarket RTDS (Chainlink)
   ```

2. **Verify database logging:**
   ```bash
   sqlite3 data/performance.db "SELECT market_slug, price_source FROM paper_trades ORDER BY id DESC LIMIT 5;"
   ```
   You should see `price_source='chainlink'` for new trades.

3. **Compare prices:**
   - Check bot's `price_to_beat` in logs
   - Compare with Polymarket UI market page
   - Difference should be <$1 (vs. previous $2,469 discrepancy)

## Rollback Instructions

If you need to revert to Binance prices:

1. **Modify `polymarket/trading/btc_price.py`:**
   ```python
   # Line 69
   self.price_stream = CryptoPriceStream(
       buffer_size=buffer_size,
       use_chainlink=False  # Change to False
   )
   ```

2. **Restart the bot**

Note: The database `price_source` column will remain but will show `'binance'` for subsequent trades.

## Expected Impact

### Accuracy Improvements
- **Price accuracy:** <$1 discrepancy (vs. previous $2,469)
- **Directional accuracy:** Correct reference price for all predictions
- **Settlement alignment:** Bot now uses same price source as Polymarket

### Decision Quality
- **Price priority:** AI now correctly prioritizes actual BTC movement (50% weight)
- **Sentiment handling:** Lagging sentiment cannot override clear price signals
- **Conflict detection:** System reduces confidence when signals conflict

### Audit Trail
- Every trade now logs which price source was used
- Enables post-trade analysis of price source impact
- Supports debugging price-related issues

## Technical Details

### Chainlink Message Format

**Initial data dump (on subscribe):**
```json
{
  "type": "subscribe",
  "payload": {
    "data": [
      {
        "symbol": "btc/usd",
        "value": "69726.92",
        "timestamp": 1738771500000
      }
    ]
  }
}
```

**Real-time updates:**
```json
{
  "type": "update",
  "payload": {
    "symbol": "btc/usd",
    "value": "69850.00",
    "timestamp": 1738771515000
  }
}
```

### Signal Weighting Formula

```python
weighted_confidence = (
    price_confidence * 0.50 +        # Tier 1: Price Reality
    market_confidence * 0.25 +        # Tier 2: Market Structure
    external_confidence * 0.15 +      # Tier 3: External Signals
    sentiment_confidence * 0.10       # Tier 4: Sentiment
)

# Apply conflict penalty if price-sentiment mismatch
if price_direction != sentiment_direction:
    weighted_confidence *= 0.50  # 50% penalty
```

## Testing

Run the test suite to verify the changes:

```bash
# Chainlink integration tests
pytest tests/test_crypto_price_stream_chainlink.py -v

# Signal weighting tests
pytest tests/test_signal_weighting.py -v

# Integration test
pytest tests/test_chainlink_integration.py -v
```

All tests should pass.

## Support

If you encounter issues:
1. Check that you're connected to Polymarket RTDS WebSocket
2. Verify the database migration ran successfully
3. Review logs for Chainlink connection messages
4. Compare bot prices with Polymarket UI to verify accuracy

For bugs or questions, open an issue on the repository.
