# Sentiment Analysis System

## Overview

The sentiment analysis system combines multiple real-time signals to predict BTC price movement in the next 15 minutes.

## Architecture

```
Social APIs          Binance APIs
     ↓                    ↓
Social Scorer      Market Scorer
     ↓                    ↓
     └─────── Aggregator ─────┘
                  ↓
           Final Sentiment
           (score + confidence)
                  ↓
            AI Decision
```

## Components

### 1. Social Sentiment Scorer (`polymarket/trading/social_sentiment.py`)

**Data Sources:**
- Alternative.me Fear & Greed Index (0-100)
- CoinGecko Trending (Is BTC in top 3?)
- CoinGecko Community Votes (up% vs down%)

**Output:**
- Score: -0.7 to +0.85 (asymmetric - trending is one-sided)
- Confidence: Based on sources available (0.0 to 1.0)

**Weights:**
- Fear/Greed: 40%
- Trending: 30%
- Votes: 30%

### 2. Market Microstructure Scorer (`polymarket/trading/market_microstructure.py`)

**Data Sources (Binance Public APIs):**
- Order book depth (bid/ask walls)
- Recent trades (whale detection >5 BTC)
- 24hr ticker (volume spike detection)
- Klines (price momentum/velocity)

**Weights:**
- Order book: 20%
- Whales: 25%
- Volume: 25%
- Momentum: 30% (highest - most predictive for 15-min)

**Output:**
- Score: -1.0 to +1.0
- Confidence: Based on metric agreement (0.0 to 1.0)

### 3. Signal Aggregator (`polymarket/trading/signal_aggregator.py`)

**Combines social + market with dynamic confidence:**

**Agreement Boost/Penalty:**
- Perfect agreement (both bullish or both bearish): 1.5x confidence
- Moderate agreement: 1.0-1.3x confidence
- Conflict (one bullish, one bearish): 0.5-0.8x confidence

**Final Score:**
```python
final_score = (market_score * 0.6) + (social_score * 0.4)
```

**Final Confidence:**
```python
base_confidence = (social_conf + market_conf) / 2
agreement = calculate_agreement(social_score, market_score)
final_confidence = base_confidence * agreement
```

## Signal Classification

| Final Confidence | Strength |
|-----------------|----------|
| >= 0.7 | STRONG |
| 0.5 - 0.7 | WEAK |
| < 0.5 | CONFLICTED |

| Final Score | Direction |
|------------|-----------|
| > 0.1 | BULLISH |
| -0.1 to 0.1 | NEUTRAL |
| < -0.1 | BEARISH |

**Signal Types:**
- `STRONG_BULLISH` - High confidence, bullish
- `WEAK_BEARISH` - Low confidence, bearish
- `CONFLICTED_NEUTRAL` - Very low confidence, mixed signals
- `MARKET_ONLY_STRONG_BULLISH` - Social unavailable, using market only
- `TECHNICAL_ONLY` - Both sentiment sources failed

## Error Handling

**Graceful Degradation:**
1. If social APIs fail → Use market microstructure only (0.7x confidence penalty)
2. If Binance APIs fail → Use social only (0.7x confidence penalty)
3. If both fail → Fall back to technical indicators (confidence = 0.0)

## Usage Example

```python
from polymarket.trading.social_sentiment import SocialSentimentService
from polymarket.trading.market_microstructure import MarketMicrostructureService
from polymarket.trading.signal_aggregator import SignalAggregator

# Initialize
social_service = SocialSentimentService(settings)
market_service = MarketMicrostructureService(settings)
aggregator = SignalAggregator()

# Fetch data (parallel for speed)
social, market = await asyncio.gather(
    social_service.get_social_score(),
    market_service.get_market_score()
)

# Aggregate
aggregated = aggregator.aggregate(social, market)

print(f"Score: {aggregated.final_score:+.2f}")
print(f"Confidence: {aggregated.final_confidence:.2f}")
print(f"Signal: {aggregated.signal_type}")

# Use in AI decision
decision = await ai_service.make_decision(
    btc_price=btc_data,
    technical_indicators=indicators,
    aggregated_sentiment=aggregated,
    market_data=market_dict,
    portfolio_value=portfolio_value
)
```

## Monitoring

**Key Metrics:**
- Confidence distribution (should have 20-40% above 70% threshold)
- HOLD rate (should be 60-80% - selective trading)
- Agreement rate (social + market agree 60%+ of time)

**Red Flags:**
- Always <70% confidence → Signals too conservative
- Always >70% confidence → Signals too aggressive
- Always 100% agreement → Not detecting real conflicts

## References

- Design: `docs/plans/2026-02-10-sentiment-redesign-design.md`
- Implementation: `docs/plans/2026-02-10-sentiment-redesign-implementation.md`
