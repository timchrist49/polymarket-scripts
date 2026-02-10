# Short-Term Sentiment Analysis Redesign

**Date:** 2026-02-10
**Status:** Design Complete - Ready for Implementation
**Author:** Claude + User

## Problem Statement

The current sentiment analysis system uses **news articles** (24-hour time range) that focus on long-term investment narratives. This is fundamentally mismatched with our **15-minute trading timeframe**.

**Current Issue:**
- Technical indicators: RSI, MACD, EMA → Short-term (15-min candles)
- Sentiment analysis: News articles about "bottom signals", "investment thesis" → Long-term (days/weeks)
- Result: Sentiment confidence always low (~50%), signals don't align with price movements

## Solution Overview

Replace keyword-based news sentiment with a **multi-signal scoring system** that combines:
1. **Social Sentiment** (crypto-specific APIs) - Real-time crowd psychology
2. **Market Microstructure** (Binance public APIs) - Actual money flow in last 5-15 minutes
3. **Dynamic Confidence** - Agreement-based scoring (high when signals align, low when they conflict)

## Architecture

### High-Level Design

```
┌─────────────────┐  ┌─────────────────┐
│  Social APIs    │  │  Binance APIs   │
│  (Fear/Greed,   │  │  (Order Book,   │
│   Trending)     │  │   Trades, Vol)  │
└────────┬────────┘  └────────┬────────┘
         │                     │
         │ (parallel fetch)    │
         ▼                     ▼
   ┌──────────┐          ┌──────────┐
   │  Social  │          │  Market  │
   │  Scorer  │          │  Scorer  │
   └────┬─────┘          └────┬─────┘
        │                     │
        │ score: -1 to +1     │
        │ confidence: 0 to 1  │
        └──────────┬──────────┘
                   ▼
         ┌──────────────────┐
         │  Agreement       │
         │  Calculator      │
         │  (dynamic conf)  │
         └────────┬─────────┘
                  │
                  ▼
         ┌──────────────────┐
         │  OpenAI          │ ← Validates/adjusts
         │  (AI Decision)   │
         └──────────────────┘
```

### Component Structure

```
polymarket/trading/
├── sentiment.py              → DELETE (old Tavily-based)
├── social_sentiment.py       → NEW (crypto APIs)
├── market_microstructure.py  → NEW (Binance microstructure)
├── signal_aggregator.py      → NEW (agreement calculator)
└── ai_decision.py            → MODIFY (receives scored signals)
```

### Data Flow (3-minute cycle)

```
1. FETCH PHASE (parallel, ~2-3 seconds):
   ├─► Social APIs → social_score, social_confidence
   └─► Binance APIs → market_score, market_confidence

2. SCORING PHASE (<1 second):
   ├─► Social Scorer: Aggregates → score: -1 to +1
   ├─► Market Scorer: Combines metrics → score: -1 to +1
   └─► Agreement Calculator → final_confidence (0 to 1)

3. AI DECISION PHASE (~3-5 seconds):
   └─► OpenAI receives pre-scored data, can adjust ±0.15
```

---

## Component 1: Social Sentiment Scorer

### APIs Used (All Free, No Auth)

**1. Alternative.me Fear & Greed Index**
- Endpoint: `https://api.alternative.me/fng/?limit=1`
- Returns: 0-100 score (0=Extreme Fear, 100=Extreme Greed)
- Update: Daily
- Signal: Overall market sentiment baseline

**2. CoinGecko Trending**
- Endpoint: `https://api.coingecko.com/api/v3/search/trending`
- Returns: Top 7 trending coins (24h)
- Signal: Is BTC in top 3? → High social attention (bullish)

**3. CoinGecko Sentiment Votes**
- Endpoint: `https://api.coingecko.com/api/v3/coins/bitcoin`
- Returns: `sentiment_votes_up_percentage`, `sentiment_votes_down_percentage`
- Signal: Community voting on price direction

### Scoring Logic

```python
def calculate_social_score() -> tuple[float, float]:
    """Returns (score: -1 to +1, confidence: 0 to 1)"""

    # Fetch all sources in parallel
    fear_greed = fetch_fear_greed()      # 0-100
    is_trending = fetch_trending()       # bool
    votes = fetch_sentiment_votes()      # {up%, down%}

    # Convert to -1 to +1 scale
    fg_score = (fear_greed - 50) / 50             # 0→-1, 100→+1
    trend_score = 0.5 if is_trending else 0.0
    vote_score = (votes.up - votes.down) / 100

    # Weighted average
    social_score = (fg_score * 0.4) + (trend_score * 0.3) + (vote_score * 0.3)

    # Confidence = sources available / total sources
    confidence = count_available_sources() / 3

    return social_score, confidence
```

### Error Handling

- If API fails: Skip that source, reduce confidence
- If ALL fail: Return (0.0, 0.0) - neutral with no confidence
- Cache: 5-10 minutes per source (avoid rate limits)

---

## Component 2: Market Microstructure Scorer

### Binance Public APIs (No Auth Required)

**1. Order Book Depth**
- Endpoint: `GET /api/v3/depth?symbol=BTCUSDT&limit=100`
- Returns: Top 100 bid/ask levels
- Signal: Large walls (>10 BTC) = support/resistance
- Timeframe: Real-time snapshot

**2. Recent Trades**
- Endpoint: `GET /api/v3/trades?symbol=BTCUSDT&limit=100`
- Returns: Last 100 trades with size and direction
- Signal: Large orders (>5 BTC) = whale activity
- Timeframe: Last 5 minutes

**3. 24hr Ticker**
- Endpoint: `GET /api/v3/ticker/24hr?symbol=BTCUSDT`
- Returns: Volume, price change%, trade count
- Signal: Volume spike vs 24h average
- Timeframe: Current 5-min vs 24h average

**4. Kline Data** (already using)
- Endpoint: `GET /api/v3/klines?symbol=BTCUSDT&interval=1m`
- Returns: 1-minute candles
- Signal: Price velocity, acceleration
- Timeframe: Last 15 minutes

### Scoring Logic

```python
def calculate_market_score() -> tuple[float, float]:
    """Returns (score: -1 to +1, confidence: 0 to 1)"""

    # Fetch all data in parallel
    order_book = fetch_order_book()
    trades = fetch_recent_trades()
    ticker = fetch_24hr_ticker()
    klines = fetch_klines(interval="1m", limit=15)

    # Score each metric (-1 to +1)
    ob_score = score_order_book(order_book)      # Bid vs ask walls
    whale_score = score_whale_activity(trades)   # Large buys vs sells
    volume_score = score_volume_spike(ticker)    # Volume vs average
    momentum_score = score_momentum(klines)      # Price velocity

    # Weighted average (momentum most important for 15-min)
    weights = {
        "order_book": 0.20,
        "whales": 0.25,
        "volume": 0.25,
        "momentum": 0.30  # Highest weight
    }

    market_score = sum(score * weights[name] for name, score in [
        ("order_book", ob_score),
        ("whales", whale_score),
        ("volume", volume_score),
        ("momentum", momentum_score)
    ])

    # Confidence based on internal agreement
    agreement = calculate_metric_agreement([ob_score, whale_score, volume_score, momentum_score])
    confidence = agreement  # 0.0 (conflict) to 1.0 (perfect alignment)

    return market_score, confidence
```

### Specific Metric Calculations

**Order Book Score:**
```python
def score_order_book(order_book: dict) -> float:
    """Bid wall strength vs ask wall strength."""
    bid_walls = sum(qty for price, qty in bids if qty > 10)  # Large bids
    ask_walls = sum(qty for price, qty in asks if qty > 10)  # Large asks

    if bid_walls + ask_walls == 0:
        return 0.0
    return (bid_walls - ask_walls) / (bid_walls + ask_walls)
```

**Whale Activity Score:**
```python
def score_whale_activity(trades: list) -> float:
    """Large buy orders vs large sell orders."""
    large_buys = sum(1 for t in trades if t.qty > 5 and t.is_buy)
    large_sells = sum(1 for t in trades if t.qty > 5 and not t.is_buy)

    if large_buys + large_sells == 0:
        return 0.0
    return (large_buys - large_sells) / (large_buys + large_sells)
```

### Error Handling

- If Binance fails: Fall back to technical indicators only, confidence = 0.0
- Timeout: 5 seconds per API call
- Cache: 30-60 seconds (frequently updated data)

---

## Component 3: Agreement Calculator (Dynamic Confidence)

### Purpose

Calculate final confidence based on how much social and market signals **agree**.

### Agreement Formula

```python
def calculate_agreement_score(score1: float, score2: float) -> float:
    """
    Returns agreement multiplier (0.5 to 1.5):
    - Perfect agreement: 1.5x confidence boost
    - Same direction: 1.0-1.3x
    - Neutral: 1.0x
    - Opposite directions: 0.5-0.8x confidence penalty
    """

    # Both in same direction (both positive or both negative)
    if score1 * score2 > 0:
        alignment = 1 - abs(score1 - score2) / 2
        return 1.0 + (alignment * 0.5)  # Boost: 1.0 to 1.5x

    # Opposite directions (conflict)
    elif score1 * score2 < 0:
        conflict = abs(score1 - score2) / 2
        return 1.0 - (conflict * 0.5)  # Penalty: 0.5 to 1.0x

    # Neutral
    else:
        return 1.0
```

### Final Confidence Calculation

```python
def calculate_final_confidence(
    social_score: float,      # -1 to +1
    social_conf: float,       # 0 to 1
    market_score: float,      # -1 to +1
    market_conf: float        # 0 to 1
) -> tuple[float, float, str]:
    """Returns (final_score, final_confidence, signal_type)."""

    # 1. Weighted average (market slightly higher weight)
    final_score = (market_score * 0.6) + (social_score * 0.4)

    # 2. Base confidence from individual confidences
    base_confidence = (social_conf + market_conf) / 2

    # 3. Agreement multiplier
    agreement = calculate_agreement_score(social_score, market_score)

    # 4. Final confidence (boosted or penalized)
    final_confidence = min(base_confidence * agreement, 1.0)

    # 5. Signal classification
    signal_type = classify_signal(final_score, final_confidence)

    return final_score, final_confidence, signal_type
```

### Examples

**Example 1: Strong Agreement (Bullish)**
```
Social: score=0.8, confidence=0.7
Market: score=0.9, confidence=0.8

Result:
- final_score = (0.9 * 0.6) + (0.8 * 0.4) = 0.86
- base_conf = (0.7 + 0.8) / 2 = 0.75
- agreement = 1.45 (high alignment boost)
- final_conf = min(0.75 * 1.45, 1.0) = 1.0

→ STRONG_BULLISH, 100% confidence ✓ TRADE
```

**Example 2: Weak Signals**
```
Social: score=0.2, confidence=0.5
Market: score=0.1, confidence=0.6

Result:
- final_score = 0.14
- base_conf = 0.55
- agreement = 1.05
- final_conf = 0.58

→ WEAK_BULLISH, 58% confidence ✗ HOLD (below 70%)
```

**Example 3: Conflict**
```
Social: score=-0.6, confidence=0.7 (bearish)
Market: score=0.8, confidence=0.9 (bullish - whales buying!)

Result:
- final_score = 0.24 (market wins)
- base_conf = 0.8
- agreement = 0.6 (conflict penalty!)
- final_conf = 0.48

→ CONFLICTED, 48% confidence ✗ HOLD (signals disagree)
```

---

## Component 4: AI Decision Engine Integration

### Modified Prompt Structure

OpenAI receives **pre-scored signals** instead of raw news:

```python
prompt = f"""
SOCIAL SENTIMENT (Real-time):
- Score: {social.score:+.2f} (-1 bearish to +1 bullish)
- Confidence: {social.confidence:.2f}
- Sources: Fear/Greed={fear_greed}, Trending={is_trending}
- Signal: {social.signal_type}

MARKET MICROSTRUCTURE (Binance, last 5-15 min):
- Score: {market.score:+.2f}
- Confidence: {market.confidence:.2f}
- Order Book: {order_book_bias} (bid vs ask walls)
- Whale Activity: {whale_direction} ({whale_count} large orders)
- Volume: {volume_ratio:.1f}x normal
- Momentum: {momentum_direction}

AGGREGATED SIGNAL:
- Final Score: {final_score:+.2f} (market 60% + social 40%)
- Final Confidence: {final_confidence:.2f}
- Signal Type: {signal_type}
- Agreement: {"HIGH" if agreement > 1.2 else "LOW" if agreement < 0.8 else "MODERATE"}

INSTRUCTIONS:
1. Pre-calculated confidence based on signal agreement
2. You may ADJUST confidence by max ±0.15 if you spot patterns
3. Only trade if final confidence >= 0.70
4. Consider: Do all signals point the same direction?

Return JSON with confidence_adjustment field.
"""
```

### AI's Role: Validator

- **Sees**: Pre-scored signals + raw metrics + suggested confidence
- **Can do**:
  - Accept confidence as-is (most common)
  - Boost by +0.15 max (spots strong pattern)
  - Reduce by -0.15 max (spots red flag)
- **Cannot do**: Override fundamental scoring logic

---

## Error Handling & Fallback Strategy

### Graceful Degradation

```
Priority Levels:
1. Technical Indicators (existing) - CRITICAL
2. Market Microstructure (Binance) - HIGH
3. Social Sentiment (APIs) - MEDIUM

Fallback Chain:
BEST:       Social + Market + Technical → High confidence
DEGRADED 1: Market + Technical → Moderate confidence (0.7x penalty)
DEGRADED 2: Social + Technical → Moderate confidence (0.7x penalty)
DEGRADED 3: Technical only → Low confidence
WORST:      Nothing → HOLD (0% confidence)
```

### Error Handling by Component

**Social Sentiment:**
- If one API fails: Use remaining sources, reduce confidence
- If all fail: Return (0.0, 0.0)
- Cache: 5-10 minutes (avoid rate limits)

**Market Microstructure:**
- If Binance fails: Return (0.0, 0.0)
- Timeout: 5 seconds per API call
- Cache: 30-60 seconds

**Agreement Calculator:**
- Handles missing signals gracefully
- If only market available: Use market score * 0.7 confidence
- If only social available: Use social score * 0.7 confidence
- If both missing: Return (0.0, 0.0, "TECHNICAL_ONLY")

### Rate Limiting

- Implement simple rate limiter (calls per minute)
- Use cache to avoid unnecessary API calls
- Exponential backoff on repeated failures

---

## Testing Strategy

### Unit Tests

```python
# test_social_sentiment.py
- test_social_scorer_all_sources_available()
- test_social_scorer_partial_failure()
- test_social_scorer_all_fail()

# test_market_microstructure.py
- test_order_book_scoring()
- test_whale_detection()
- test_volume_spike_detection()
- test_momentum_calculation()

# test_agreement.py
- test_strong_agreement_bullish()
- test_strong_agreement_bearish()
- test_conflict_signals()
- test_neutral_signals()
```

### Integration Tests

```python
# test_integration.py
- test_full_sentiment_pipeline()
- test_api_failure_handling()
- test_confidence_threshold_enforcement()
```

### Manual Testing Scenarios

**Scenario 1: Bull Market Confirmation**
```
Setup: F&G=80, BTC Trending, Heavy bids, Whale buys, 2x volume, RSI 65
Expected: Social ~0.8, Market ~0.8, Final ~0.95+ → YES trade
```

**Scenario 2: Mixed Signals (HOLD)**
```
Setup: F&G=30, Not trending, Balanced book, Normal volume, RSI 50
Expected: Social ~-0.3, Market ~-0.2, Final ~0.4-0.5 → HOLD
```

**Scenario 3: Smart Money vs Crowd**
```
Setup: F&G=20 (fear), Heavy bids (whales buying), 1.5x volume
Expected: Social ~-0.6, Market ~0.6, Final ~0.4-0.5 (conflict) → HOLD
```

### Testing Timeline

```
Week 1: Unit Tests (all components in isolation)
Week 2: Integration Tests (full pipeline)
Week 3: Paper Trading (read_only mode, 3-7 days)
Week 4: Live Trading (small scale, 1% position size)
```

### Success Metrics

**Good Indicators:**
- 20-30% trades above 70% threshold (selective)
- 70-80% HOLD rate (conservative)
- Social/market agree 60%+ of time
- False positive rate <30%

**Red Flags:**
- Never above 70% → Too conservative
- Always above 70% → Too aggressive
- Always 100% agreement → Not detecting conflicts
- Constant API failures → Need better fallbacks

---

## Implementation Plan

### Phase 1: Core Infrastructure (Week 1)
1. Create `social_sentiment.py` with API integrations
2. Create `market_microstructure.py` with Binance APIs
3. Create `signal_aggregator.py` with agreement calculator
4. Write unit tests for each component

### Phase 2: Integration (Week 2)
1. Modify `ai_decision.py` to accept scored signals
2. Update prompts with new data structure
3. Update `auto_trade.py` orchestration
4. Write integration tests

### Phase 3: Testing (Week 3)
1. Paper trading in read_only mode
2. Collect metrics (confidence distribution, agreement rate)
3. Tune weights if needed
4. Validate error handling

### Phase 4: Deployment (Week 4)
1. Enable live trading with small positions
2. Monitor for 48 hours
3. Gradually increase position size if successful
4. Document learnings

---

## Configuration

### New Environment Variables

```bash
# Social Sentiment (all optional - have fallbacks)
COINGECKO_API_KEY=  # Optional - free tier works
ALTERNATIVE_ME_API_KEY=  # Not needed - public API

# Market Microstructure (uses existing Binance)
# No new credentials needed - using public APIs

# Scoring Configuration
SOCIAL_WEIGHT=0.4           # Social contribution to final score
MARKET_WEIGHT=0.6           # Market contribution to final score
AGREEMENT_BOOST_MAX=0.5     # Max confidence boost from agreement
AGREEMENT_PENALTY_MAX=0.5   # Max confidence penalty from conflict

# Caching
FEAR_GREED_CACHE_SECONDS=3600   # 1 hour
TRENDING_CACHE_SECONDS=300      # 5 minutes
ORDER_BOOK_CACHE_SECONDS=30     # 30 seconds
```

---

## Migration Strategy

### From Current System

1. **Keep technical indicators unchanged** (working well)
2. **Replace** `sentiment.py` (Tavily-based) with new multi-signal system
3. **Maintain** same interfaces (returns score + confidence)
4. **Add** new fields to AI prompt (but keep structure similar)

### Rollback Plan

If new system fails:
1. Revert to old `sentiment.py`
2. Use technical indicators only temporarily
3. Fix issues and retry

---

## Success Criteria

### After 1 Week
- ✅ No critical bugs or crashes
- ✅ Confidence distribution looks reasonable (not all 0% or all 100%)
- ✅ Agreement calculator detects conflicts correctly

### After 1 Month
- ✅ Confidence >70% rate: 20-40% (selective but not too rare)
- ✅ Profitable trades ratio >60%
- ✅ No false positives due to API failures

### After 3 Months
- ✅ Consistent profitability
- ✅ System adapts to different market conditions
- ✅ Error handling proven robust

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| APIs go down | No sentiment data | Fallback to technical indicators only |
| Rate limits hit | Incomplete data | Caching + rate limiter |
| Signals always conflict | Never trades | Tune weights, investigate signal quality |
| Whales manipulate order book | False bullish signals | Cross-validate with volume + trades |
| Social APIs lag market | Stale sentiment | Prioritize market microstructure (60% weight) |

---

## Future Enhancements

**Phase 2 (Optional):**
- Add LunarCrush social metrics (requires paid API)
- Add funding rates from derivatives
- Add liquidation data
- Machine learning for weight optimization

**Phase 3 (Advanced):**
- Multi-timeframe analysis (1-min, 5-min, 15-min)
- Cross-exchange arbitrage detection
- On-chain metrics (large transfers)
- Twitter/X sentiment with AI analysis

---

## Appendix: API Documentation

### Alternative.me Fear & Greed

```bash
GET https://api.alternative.me/fng/?limit=1

Response:
{
  "data": [{
    "value": "75",           # 0-100
    "value_classification": "Greed",
    "timestamp": "1707566400"
  }]
}
```

### CoinGecko Trending

```bash
GET https://api.coingecko.com/api/v3/search/trending

Response:
{
  "coins": [
    {"item": {"id": "bitcoin", "name": "Bitcoin", ...}},
    ...
  ]
}
```

### Binance Order Book

```bash
GET https://api.binance.com/api/v3/depth?symbol=BTCUSDT&limit=100

Response:
{
  "bids": [["68000.00", "15.5"], ...],  # [price, quantity]
  "asks": [["68100.00", "2.1"], ...]
}
```

### Binance Recent Trades

```bash
GET https://api.binance.com/api/v3/trades?symbol=BTCUSDT&limit=100

Response:
[
  {
    "id": 123456,
    "price": "68050.00",
    "qty": "8.5",
    "time": 1707566400000,
    "isBuyerMaker": true
  },
  ...
]
```

---

**Status:** Design Complete - Ready for Implementation Planning

**Next Steps:**
1. Use `/superpowers:write-plan` to create detailed implementation plan
2. Break down into executable tasks with file paths and code snippets
3. Execute with `/superpowers:execute-plan`
