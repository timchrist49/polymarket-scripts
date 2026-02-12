# Funding Rates + BTC Dominance Signals - IMPLEMENTATION COMPLETE ✅

**Date Completed:** 2026-02-12
**Status:** Production Ready
**Implementation Time:** ~20 minutes (as promised!)

---

## Summary

Successfully integrated two new CoinGecko Pro API endpoints to enhance trading bot accuracy:
1. **Funding Rates** (Binance perpetual futures sentiment)
2. **BTC Dominance** (capital flow indicator)

Bot now uses **4 signal sources** instead of 2, with rebalanced weights for better decision-making.

---

## Implementation Details

### New API Endpoints

#### 1. Funding Rates (`/derivatives`)
- **Source**: Binance BTCUSDT perpetual futures via CoinGecko Pro
- **Signal Interpretation**:
  - **Negative funding** (-%) = Bullish (shorts paying longs, oversold)
  - **Positive funding** (+%) = Bearish (longs paying shorts, overheated)
- **Normalization**: ±0.1% maps to ±1.0 score
- **Signal Types**: OVERHEATED, NEUTRAL, OVERSOLD
- **Current Reading**: -0.0482% → +0.48 score (slightly bullish)

#### 2. BTC Dominance (`/global`)
- **Source**: CoinGecko global market data
- **Signal Interpretation**:
  - **>55%** = BTC Season (capital flowing to BTC, bullish)
  - **<50%** = Alt Season (capital flowing to alts, bearish)
- **Normalization**: 40-70% range mapped to -1.0 to +1.0
- **Signal Types**: BTC_SEASON, NEUTRAL, ALT_SEASON
- **Current Reading**: 56.76% → +0.12 score (BTC season)

### Updated Signal Weights

**Before (2 signals):**
- Market Microstructure: 60%
- Social Sentiment: 40%

**After (4 signals):**
- Market Microstructure: 40% ⬇️ (-20%)
- Social Sentiment: 20% ⬇️ (-20%)
- **Funding Rates: 20%** 🆕
- **BTC Dominance: 15%** 🆕
- (Order Book: 5% - included in Market Microstructure)

### Files Modified

1. **polymarket/models.py**
   - Added `FundingRateSignal` dataclass (7 fields)
   - Added `BTCDominanceSignal` dataclass (7 fields)
   - Updated `AggregatedSentiment` to include optional funding + dominance

2. **polymarket/trading/btc_price.py**
   - Added `get_funding_rates()` method (~90 lines)
   - Added `get_btc_dominance()` method (~95 lines)
   - Both use CoinGecko Pro API with authentication

3. **polymarket/trading/signal_aggregator.py**
   - Updated weights (4 new constants)
   - Rewrote `aggregate()` method to handle 2-4 signals
   - Multi-signal agreement calculation (all pairs)
   - Dynamic weight normalization

4. **scripts/auto_trade.py**
   - Updated data collection to fetch 5 values (was 3)
   - Added funding + dominance to logging
   - Pass new signals to aggregator

### Commit Hash
```
300df44 - feat: add funding rates and BTC dominance signals
```

---

## Test Results

✅ **Endpoint Test**: Both endpoints successfully fetching data
```
Funding Rates:  -0.0482% (score +0.48, NEUTRAL)
BTC Dominance:  56.76% (score +0.12, BTC_SEASON)
```

✅ **Signal Aggregator Tests**: All 5 tests passing
✅ **Bot Startup**: Successful restart with new signals

---

## Expected Impact

### Signal Diversity
- **Before**: 2 data sources (social + market microstructure)
- **After**: 4 data sources (social + market + funding + dominance)
- **Benefit**: More robust signals, less vulnerable to single-source errors

### Signal Agreement
- Aggregator now calculates agreement across **all signal pairs**
- Conflicting signals reduce confidence (0.5x - 1.0x)
- Aligned signals boost confidence (1.0x - 1.5x)
- More sophisticated than 2-signal agreement

### Capital Flow Detection
- **BTC dominance** reveals where money is flowing (BTC vs alts)
- Rising dominance → bullish for BTC
- Falling dominance → bearish for BTC (alt season)

### Market Sentiment
- **Funding rates** reveal futures trader sentiment
- Positive funding → overheated longs (bearish contrarian signal)
- Negative funding → oversold shorts (bullish contrarian signal)

---

## Production Monitoring

### What to Monitor (First 20 Trades)

1. **Signal Availability**
   - Check logs for funding/dominance fetch failures
   - Verify graceful degradation if APIs unavailable

2. **Signal Weights in Action**
   - Observe how 4-signal aggregation affects final scores
   - Compare decisions to previous 2-signal system

3. **API Rate Limits**
   - CoinGecko Pro: 500 calls/min limit
   - Current usage: 2 calls per 3-minute cycle = ~40 calls/hour
   - Well within limits ✓

4. **Decision Quality**
   - Does BTC dominance improve timing? (capital flow indicator)
   - Do funding rates catch overheated markets? (contrarian signal)

### Log Patterns to Look For

**Success:**
```
[info] Funding rate fetched    funding_rate=-0.0482% score=+0.48 signal=NEUTRAL
[info] BTC dominance fetched   dominance=56.76% score=+0.12 signal=BTC_SEASON
[info] Data collected          funding_score=+0.48 dominance_score=+0.12
[info] Signals aggregated      num_signals=4 agreement=1.23x
```

**Graceful Degradation (OK):**
```
[warning] CoinGecko Pro API key required for funding rates
[warning] Using 3 signals (funding unavailable)
```

**Error (Needs Investigation):**
```
[error] Failed to fetch funding rates    error=...
[error] Failed to fetch BTC dominance    error=...
```

---

## API Details

### CoinGecko Pro Endpoints Used

**1. Derivatives Endpoint**
```bash
GET https://pro-api.coingecko.com/api/v3/derivatives
    ?x_cg_pro_api_key=CG-RQHWR...

Response: Array of derivative contracts
Find: market="Binance (Futures)" + symbol="BTCUSDT" + contract_type="perpetual"
Extract: funding_rate (as decimal, e.g., -0.000482 = -0.0482%)
```

**2. Global Market Data Endpoint**
```bash
GET https://pro-api.coingecko.com/api/v3/global
    ?x_cg_pro_api_key=CG-RQHWR...

Response: Global crypto market data
Extract:
  - market_cap_percentage.btc (56.76%)
  - total_market_cap.usd ($2.4T)
  - market_cap_change_percentage_24h_usd (+1.04%)
```

---

## Risk Assessment

**Risk Level:** LOW

**Why Low Risk:**
- Both signals are **optional** (None if unavailable)
- Bot degrades gracefully to 2-3 signal operation
- Signal aggregator handles 1-4 signals dynamically
- No breaking changes to existing logic
- All existing tests still passing
- Clear logging for debugging

**Rollback Plan:**
```bash
git revert 300df44
# Reverts to 2-signal system (market 60% + social 40%)
```

---

## Future Enhancements

### Potential Improvements

1. **Order Book Depth Analysis**
   - Already referenced in weights (5%)
   - Could implement dedicated order book signals
   - Requires deeper Binance API integration

2. **Funding Rate Trends**
   - Track funding rate changes over time
   - Detect trend reversals (e.g., funding going from +0.1% to -0.05%)
   - More sophisticated than single snapshot

3. **Multi-Exchange Funding Rates**
   - Average funding across Binance, Bybit, OKX
   - Reduces single-exchange bias
   - More robust signal

4. **BTC Dominance Trends**
   - Track dominance change velocity
   - Detect capital flow acceleration/deceleration
   - Complement current snapshot approach

5. **Weighted Signal Confidence**
   - Weight signals by their individual confidences
   - Currently uses equal weight for all available signals
   - Could give more weight to high-confidence signals

---

## References

- **CoinGecko Pro API**: https://docs.coingecko.com/
- **Funding Rates Explained**: Perpetual futures mechanism
- **BTC Dominance**: Market cap percentage as capital flow indicator
- **Previous Work**: Odds-adjusted position sizing (2026-02-12)
- **Commit**: 300df44

---

## Sign-Off

**Developer:** Claude Sonnet 4.5
**Testing:** Endpoint verification passed
**Status:** ✅ **DEPLOYED TO PRODUCTION**
**Bot Status:** Running with new signals (PID 351505, 351566)

---

**Next Steps:**
1. ✅ Funding + dominance endpoints implemented
2. ✅ Signal aggregator updated
3. ✅ Bot restarted with new signals
4. ⏳ Monitor first 20 trades for signal quality
5. ⏳ Evaluate performance after 50 trades
6. ⏳ Consider adding order book depth analysis (5% weight)

**Timeline:** Implemented in ~20 minutes as promised! 🚀
