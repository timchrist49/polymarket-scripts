# Polymarket BTC Trading Bot - Complete Technical Reference

**Version:** 2.1 (Odds-Adjusted Position Sizing)
**Last Updated:** 2026-02-12
**Model:** GPT-5-Nano (gpt-o1-mini)

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [System Architecture](#system-architecture)
3. [Data Sources Deep Dive](#data-sources-deep-dive)
4. [Signal Processing Pipeline](#signal-processing-pipeline)
5. [Orderbook Tracking & History](#orderbook-tracking--history)
6. [Whale Detection & Tracking](#whale-detection--tracking)
7. [Technical Indicators Methodology](#technical-indicators-methodology)
8. [Historical Data Fetching](#historical-data-fetching)
9. [AI Decision Engine](#ai-decision-engine)
10. [Risk Management & Validation](#risk-management--validation)
11. [Complete Workflow Timeline](#complete-workflow-timeline)
12. [Example Trading Scenarios](#example-trading-scenarios)
13. [Configuration Reference](#configuration-reference)
14. [Monitoring & Troubleshooting](#monitoring--troubleshooting)

---

## Executive Summary

### Purpose
Autonomous trading bot for Polymarket BTC 15-minute up/down markets using multi-source signal aggregation, AI-powered decision making, and lagging indicator protection.

### Core Approach
1. **Multi-Source Data Collection** - Real-time WebSocket streams + public APIs
2. **Weighted Signal Aggregation** - Market microstructure (60%) + Social sentiment (40%)
3. **AI-Powered Analysis** - GPT-5-Nano with reasoning tokens for decision making
4. **Validation Rules** - Contradiction detection to prevent lagging indicator losses
5. **Risk Management** - Portfolio limits, position sizing, stop-loss monitoring

### Key Innovations (Recent Fixes)
- **Lagging Indicator Protection**: Compares market sentiment vs actual BTC movement
- **Reduced Momentum Weight**: 40% → 20% to minimize lag dependency
- **Increased Volume/Whale Weight**: More emphasis on current market behavior
- **5-Minute BTC Momentum**: Independent verification of actual price direction
- **Price-to-Beat Tracking**: Baseline comparison for detecting true market direction
- **Odds-Adjusted Position Sizing** (2026-02-12): Scales down position sizes on low-odds bets (reduces losses by 38-44% on risky trades)

### Performance Goals
- **Target Win Rate**: 55%+ (baseline was ~30-40% before fixes)
- **Cycle Frequency**: Every 3 minutes (180 seconds)
- **Data Collection**: 2-minute market observation window
- **Decision Time**: 5-20 seconds (AI reasoning + validation)

---

## System Architecture

### High-Level Component Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         AutoTrader                                  │
│                    (Main Orchestrator)                              │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ├─────────────────┬─────────────────┬───────────────
                              │                 │                 │
                    ┌─────────▼────────┐ ┌─────▼──────┐ ┌───────▼────────┐
                    │ BTCPriceService  │ │  Social    │ │    Market      │
                    │                  │ │ Sentiment  │ │ Microstructure │
                    │ • WebSocket      │ │ Service    │ │    Service     │
                    │ • Binance API    │ │            │ │                │
                    │ • Price History  │ │ • Fear/    │ │ • Polymarket   │
                    │ • Momentum Calc  │ │   Greed    │ │   CLOB WS      │
                    │                  │ │ • Trending │ │ • Trade Data   │
                    └──────────────────┘ │ • Votes    │ │ • Whale Track  │
                                         └────────────┘ └────────────────┘
                              │
                    ┌─────────▼────────┐
                    │ SignalAggregator │
                    │                  │
                    │ Market 60% +     │
                    │ Social 40%       │
                    │                  │
                    │ Agreement        │
                    │ Multiplier       │
                    └──────────────────┘
                              │
                    ┌─────────▼────────┐
                    │ AIDecisionService│
                    │                  │
                    │ GPT-5-Nano       │
                    │ (gpt-o1-mini)    │
                    │                  │
                    │ • Reasoning      │
                    │ • Validation     │
                    │ • Contradiction  │
                    │   Detection      │
                    └──────────────────┘
                              │
                    ┌─────────▼────────┐
                    │  RiskManager     │
                    │                  │
                    │ • Portfolio Limit│
                    │ • Position Size  │
                    │ • Stop Loss      │
                    └──────────────────┘
                              │
                    ┌─────────▼────────┐
                    │ Trade Execution  │
                    │                  │
                    │ Polymarket CLOB  │
                    └──────────────────┘
```

### Data Flow Summary

```
Every 3 minutes:
    Collect Data (parallel, ~130s)
         ↓
    Calculate Scores (~3s)
         ↓
    Aggregate Signals (~1s)
         ↓
    AI Decision (5-20s)
         ↓
    Risk Validation (<1ms)
         ↓
    Execute Trade (if approved)
         ↓
    Wait 180s → Repeat
```

---

## Data Sources Deep Dive

### 1. BTCPriceService

**Purpose:** Real-time BTC price tracking with historical data for analysis

**Primary Source:** Polymarket Real-Time Data Service (RTDS)
- **Protocol:** WebSocket (`wss://ws-subscriptions-clob.polymarket.com/ws/market`)
- **Channel:** `crypto_prices` with symbol `btcusdt`
- **Update Frequency:** Real-time (sub-second)
- **Cache Duration:** 5 seconds
- **Connection Management:** Auto-reconnect on disconnect

**Fallback Source:** Binance Public API
- **Endpoint:** `https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT`
- **Triggered When:** WebSocket unavailable or no recent price
- **Timeout:** 10 seconds

**Implementation Details:**

```python
class BTCPriceService:
    def __init__(self, settings: Settings):
        self._stream = CryptoPriceStream(settings)  # Polymarket WebSocket
        self._binance = ccxt.binance()  # Fallback exchange
        self._cache = None  # Cached price data
        self._cache_time = None  # Cache timestamp

    async def get_current_price(self) -> BTCPriceData:
        # Check cache (5 second TTL)
        if self._cache and (datetime.now() - self._cache_time).total_seconds() < 5:
            return self._cache

        # Try Polymarket WebSocket first
        if self._stream.is_connected():
            data = await self._stream.get_current_price()
            if data:
                self._cache = data
                return data

        # Fallback to Binance
        return await self._fetch_binance()
```

**Data Model:**

```python
@dataclass
class BTCPriceData:
    price: Decimal           # Current BTC/USDT price
    timestamp: datetime      # Price timestamp
    source: str             # "polymarket" or "binance"
    volume_24h: Decimal     # 24-hour trading volume
```

**Historical Data Fetching:**

```python
async def get_price_history(self, minutes: int = 60) -> list[PricePoint]:
    """
    Fetch historical 1-minute candles from Binance.

    Uses direct HTTP request (not ccxt) to avoid timeout issues.
    """
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": "BTCUSDT",
        "interval": "1m",
        "limit": str(minutes)
    }

    async with session.get(url, params=params) as resp:
        data = await resp.json()
        return [
            PricePoint(
                price=Decimal(candle[4]),      # Close price
                volume=Decimal(candle[5]),     # Volume
                timestamp=datetime.fromtimestamp(candle[0] / 1000)
            )
            for candle in data
        ]
```

**Used For:**
- Current price display
- Technical indicator calculation (60-minute history)
- BTC momentum calculation (5-minute history)
- Price-to-beat comparison

---

### 2. SocialSentimentService

**Purpose:** Crypto market sentiment from public community data

**Three Data Sources (Parallel Collection):**

#### A. Fear & Greed Index (40% weight)
- **API:** `https://api.alternative.me/fng/?limit=1`
- **Provider:** Alternative.me
- **Update Frequency:** Daily (updated ~8 AM UTC)
- **Range:** 0-100
  - 0-24: Extreme Fear
  - 25-49: Fear
  - 50: Neutral
  - 51-75: Greed
  - 76-100: Extreme Greed
- **Transformation:** `(value - 50) / 50` → -1.0 to +1.0 scale

**Example Response:**
```json
{
  "data": [{
    "value": "65",
    "value_classification": "Greed",
    "timestamp": "1707648000"
  }]
}
```

#### B. CoinGecko Trending (30% weight)
- **API:** `https://api.coingecko.com/api/v3/search/trending`
- **Check:** Is BTC in top 3 trending coins?
- **Scoring:**
  - BTC in top 3 → +0.5 bonus
  - BTC not in top 3 → 0.0
- **Note:** One-sided (only positive bonus possible)

**Example Response:**
```json
{
  "coins": [
    {"item": {"id": "bitcoin", "name": "Bitcoin", "coin_id": 1}},
    {"item": {"id": "ethereum", "name": "Ethereum", "coin_id": 1027}}
  ]
}
```

#### C. CoinGecko Community Votes (30% weight)
- **API:** `https://api.coingecko.com/api/v3/coins/bitcoin`
- **Metrics:** `sentiment_votes_up_percentage`, `sentiment_votes_down_percentage`
- **Transformation:** `(up% - down%) / 100` → -1.0 to +1.0 scale

**Example Response:**
```json
{
  "sentiment_votes_up_percentage": 62.5,
  "sentiment_votes_down_percentage": 37.5
}
```

**Scoring Formula:**

```python
def _calculate_score(self, fear_greed: int, is_trending: bool,
                     vote_up_pct: float, vote_down_pct: float) -> float:
    # Convert Fear/Greed to -1 to +1
    fg_score = (fear_greed - 50) / 50

    # Trending bonus
    trend_score = 0.5 if is_trending else 0.0

    # Community votes
    vote_score = (vote_up_pct - vote_down_pct) / 100

    # Weighted average
    score = (
        fg_score * 0.4 +      # Fear/Greed: 40%
        trend_score * 0.3 +   # Trending: 30%
        vote_score * 0.3      # Votes: 30%
    )

    return score
```

**Score Range:** -0.7 to +0.85 (asymmetric due to one-sided trending bonus)

**Confidence Calculation:**
```python
confidence = successful_sources / 3  # Maximum 3 sources

# If all 3 sources succeed: confidence = 1.0
# If 2 sources succeed: confidence = 0.67
# If 1 source succeeds: confidence = 0.33
# If all fail: confidence = 0.0 (fallback to neutral)
```

**Data Model:**

```python
@dataclass
class SocialSentiment:
    score: float                    # -0.7 to +0.85
    confidence: float               # 0.0 to 1.0
    fear_greed: int                # 0-100
    is_trending: bool              # BTC in top 3?
    vote_up_pct: float             # Community up votes %
    vote_down_pct: float           # Community down votes %
    signal_type: str               # "STRONG_BULLISH", "WEAK_BEARISH", etc.
    sources_available: list[str]   # Which sources succeeded
    timestamp: datetime
```

**Graceful Degradation:**
- If all APIs fail: Returns neutral score (0.0) with 0 confidence
- If Fear/Greed fails: Uses 50 (neutral)
- If Trending fails: Assumes False (not trending)
- If Votes fail: Uses 50/50 split (neutral)

---

### 3. MarketMicrostructureService

**Purpose:** Analyze Polymarket market behavior through real-time trade data

**Data Source:** Polymarket CLOB (Central Limit Order Book) WebSocket
- **URL:** `wss://ws-subscriptions-clob.polymarket.com/ws/market`
- **Subscription Format:**
```json
{
  "assets_ids": ["token_id_1", "token_id_2"],
  "type": "market"
}
```
- **Collection Duration:** 120 seconds (2 minutes)
- **Message Types:**
  - `last_trade_price`: Individual trade execution
  - `book`: Order book snapshot
  - `price_change`: Price change events

**Trade Data Collection:**

```python
async def collect_market_data_with_token_ids(
    self,
    token_ids: list[str],
    duration_seconds: int = 120
) -> dict:
    accumulated_data = {
        'trades': [],
        'book_snapshots': [],
        'price_changes': [],
        'collection_duration': 0
    }

    async with websockets.connect(self.WS_URL) as ws:
        # Subscribe
        subscribe_msg = {
            "assets_ids": token_ids,
            "type": "market"
        }
        await ws.send(json.dumps(subscribe_msg))

        # Collect for 2 minutes
        start_time = time.time()
        while time.time() - start_time < duration_seconds:
            msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
            data = json.loads(msg)

            for message in (data if isinstance(data, list) else [data]):
                if message.get('event_type') == 'last_trade_price':
                    accumulated_data['trades'].append(message)

        return accumulated_data
```

**Three Calculated Scores:**

#### A. Momentum Score (20% weight) - *REDUCED FROM 40%*

**Purpose:** YES token price change over collection window

**Calculation:**
```python
def calculate_momentum_score(self, trades: list) -> float:
    # Filter YES token trades
    yes_trades = [
        t for t in trades
        if self._is_yes_token(t.get('asset_id'))
    ]

    if len(yes_trades) < 2:
        return 0.0

    # First and last YES price in 2-minute window
    initial_yes_price = float(yes_trades[0]['price'])
    final_yes_price = float(yes_trades[-1]['price'])

    # Percentage change
    price_change_pct = (final_yes_price - initial_yes_price) / initial_yes_price

    # Normalize: ±10% change maps to ±1.0 score (clamped)
    momentum_score = max(min(price_change_pct * 10, 1.0), -1.0)

    return momentum_score
```

**Interpretation:**
- +1.0: YES price rose 10%+ (strong bullish)
- +0.5: YES price rose 5% (moderate bullish)
- 0.0: YES price unchanged
- -0.5: YES price dropped 5% (moderate bearish)
- -1.0: YES price dropped 10%+ (strong bearish)

#### B. Volume Flow Score (50% weight) - *INCREASED FROM 35%*

**Purpose:** Net buying pressure (YES volume vs NO volume)

**Calculation:**
```python
def calculate_volume_flow_score(self, trades: list) -> float:
    yes_volume = sum(
        float(trade['size']) for trade in trades
        if self._is_yes_token(trade.get('asset_id'))
    )

    no_volume = sum(
        float(trade['size']) for trade in trades
        if self._is_no_token(trade.get('asset_id'))
    )

    total_volume = yes_volume + no_volume
    if total_volume == 0:
        return 0.0

    # Already normalized to -1.0 to +1.0
    return (yes_volume - no_volume) / total_volume
```

**Interpretation:**
- +1.0: 100% YES buying (all volume in YES tokens)
- +0.5: 75% YES, 25% NO (strong YES buying)
- 0.0: 50/50 split (balanced)
- -0.5: 25% YES, 75% NO (strong NO buying)
- -1.0: 100% NO buying (all volume in NO tokens)

#### C. Whale Activity Score (30% weight) - *INCREASED FROM 25%*

**Purpose:** Directional signal from large trades

**Whale Definition:** Trades with size > $1,000 USD

**Calculation:**
```python
def calculate_whale_activity_score(self, trades: list) -> float:
    WHALE_SIZE_USD = 1000

    yes_whales = sum(
        1 for trade in trades
        if float(trade['size']) > WHALE_SIZE_USD
        and self._is_yes_token(trade.get('asset_id'))
    )

    no_whales = sum(
        1 for trade in trades
        if float(trade['size']) > WHALE_SIZE_USD
        and self._is_no_token(trade.get('asset_id'))
    )

    total_whales = yes_whales + no_whales
    if total_whales == 0:
        return 0.0

    return (yes_whales - no_whales) / total_whales
```

**Interpretation:**
- +1.0: All whales buying YES (strong conviction bullish)
- +0.5: 75% whale YES, 25% whale NO
- 0.0: 50/50 whale split or no whales
- -0.5: 25% whale YES, 75% whale NO
- -1.0: All whales buying NO (strong conviction bearish)

**Combined Market Score:**

```python
WEIGHTS = {
    'momentum': 0.20,      # Reduced from 0.40 to minimize lag
    'volume_flow': 0.50,   # Increased from 0.35 for current data
    'whale': 0.30          # Increased from 0.25 for behavioral signals
}

market_score = (
    momentum * 0.20 +
    volume_flow * 0.50 +
    whale * 0.30
)
```

**Confidence Calculation:**

```python
def calculate_confidence(self, data: dict) -> float:
    trade_count = len(data.get('trades', []))
    collection_duration = data.get('collection_duration', 120)

    # Base: 50+ trades = full confidence
    base_confidence = min(trade_count / 50, 1.0)

    # Penalty if didn't collect full 2 minutes
    if collection_duration < 120:
        base_confidence *= (collection_duration / 120)

    # Low liquidity penalty
    if trade_count < 10:
        base_confidence *= 0.5

    return base_confidence
```

**Data Model:**

```python
@dataclass
class MarketSignals:
    score: float                    # -1.0 to +1.0
    confidence: float               # 0.0 to 1.0
    momentum_score: float           # Individual component
    volume_score: float             # Individual component
    whale_score: float              # Individual component
    momentum_direction: str         # "UP", "DOWN", "FLAT"
    whale_direction: str            # "BUYING", "SELLING", "NEUTRAL"
    whale_count: int               # Number of whale trades
    volume_ratio: float            # Relative volume level
    signal_type: str               # "STRONG_BULLISH", etc.
    timestamp: datetime
```

---

### 4. BTC Momentum Service (NEW - Lagging Fix)

**Purpose:** Independent verification of actual BTC price movement

**Why Added:**
Market microstructure collects data over a 2-minute window, which may reflect past price action. This creates lag where sentiment shows bearish (based on past drop) but BTC has since rebounded. BTC Momentum provides current directional confirmation.

**Implementation:**

```python
async def _get_btc_momentum(
    self,
    btc_service: BTCPriceService,
    current_price: Decimal
) -> dict | None:
    """
    Calculate actual BTC momentum over last 5 minutes.

    Compares current price to 5 minutes ago to detect actual BTC direction,
    independent of Polymarket sentiment.
    """
    # Fetch 5-minute price history
    history = await btc_service.get_price_history(minutes=5)

    if not history or len(history) < 2:
        return None  # Graceful fallback

    # Get oldest price in 5-minute window
    price_5min_ago = history[0].price

    # Calculate percentage change
    momentum_pct = float((current_price - price_5min_ago) / price_5min_ago * 100)

    # Classify direction (>0.1% threshold to filter noise)
    if momentum_pct > 0.1:
        direction = 'UP'
    elif momentum_pct < -0.1:
        direction = 'DOWN'
    else:
        direction = 'FLAT'

    return {
        'price_5min_ago': price_5min_ago,
        'momentum_pct': momentum_pct,
        'direction': direction
    }
```

**Threshold Logic:**
- **>+0.1%**: Classified as UP (filters noise)
- **<-0.1%**: Classified as DOWN (filters noise)
- **-0.1% to +0.1%**: Classified as FLAT (sideways movement)

**Usage in AI Prompt:**
Included as separate context section to compare against market signals:
```
ACTUAL BTC MOMENTUM (last 5 minutes):
- 5 minutes ago: $67,500.00
- Current: $67,650.00
- Change: +0.22% (UP)

⚠️ COMPARE WITH MARKET SIGNALS:
- If market sentiment is BEARISH but BTC is UP → market is LAGGING
- If market sentiment is BULLISH but BTC is DOWN → market is LAGGING
```

**Graceful Degradation:**
- If insufficient price history (< 2 data points): Returns None
- AI receives message: "ACTUAL BTC MOMENTUM: Not available"
- Decision proceeds without momentum data (no failure)

---

### 5. Technical Analysis

**Purpose:** 60-minute technical indicators for trend confirmation

**Data Source:** BTCPriceService.get_price_history(minutes=60)

**Indicators Calculated:**

#### A. RSI (Relative Strength Index)
- **Period:** 14
- **Range:** 0-100
- **Interpretation:**
  - 70-100: Overbought (potential reversal down)
  - 30-70: Normal range
  - 0-30: Oversold (potential reversal up)

**Calculation:**
```python
def _calculate_rsi(prices: pd.Series, period: int) -> float:
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1]
```

#### B. MACD (Moving Average Convergence Divergence)
- **Fast EMA:** 12-period
- **Slow EMA:** 26-period
- **Signal Line:** 9-period EMA of MACD
- **Histogram:** MACD - Signal

**Interpretation:**
- MACD > Signal: Bullish momentum
- MACD < Signal: Bearish momentum
- Histogram growing: Strengthening trend
- Histogram shrinking: Weakening trend

**Calculation:**
```python
macd_12 = prices.ewm(span=12, adjust=False).mean()
macd_26 = prices.ewm(span=26, adjust=False).mean()
macd_line = macd_12 - macd_26
macd_signal = macd_line.ewm(span=9, adjust=False).mean()
macd_histogram = macd_line - macd_signal
```

#### C. EMA (Exponential Moving Average)
- **Short:** 9-period
- **Long:** 21-period
- **Crossover Strategy:**
  - Short > Long: Bullish trend
  - Short < Long: Bearish trend

**Calculation:**
```python
ema_short = prices.ewm(span=9, adjust=False).mean().iloc[-1]
ema_long = prices.ewm(span=21, adjust=False).mean().iloc[-1]
```

#### D. SMA (Simple Moving Average)
- **Period:** 50
- **Purpose:** Long-term support/resistance level

**Calculation:**
```python
sma_50 = prices.rolling(window=50).mean().iloc[-1]
```

#### E. Volume Change
**Purpose:** Detect unusual trading activity

**Calculation:**
```python
recent_vol = volumes.tail(5).mean()      # Last 5 minutes
avg_vol = volumes.tail(30).mean()        # Last 30 minutes
volume_change = ((recent_vol - avg_vol) / avg_vol) * 100
```

**Interpretation:**
- +50%+: High volume spike (significant interest)
- +0% to +50%: Above average volume
- -50% to 0%: Below average volume
- -50%-: Very low volume (low conviction)

#### F. Price Velocity
**Purpose:** Rate of price change

**Calculation:**
```python
price_change = prices.iloc[-1] - prices.iloc[-5]  # Last 5 minutes
velocity = price_change / 5  # $/minute
```

**Interpretation:**
- +$10/min: Rapid upward movement
- +$1/min: Moderate upward movement
- $0/min: Sideways
- -$1/min: Moderate downward movement
- -$10/min: Rapid downward movement

#### G. Trend Classification

**Logic:**
```python
if ema_short > ema_long and macd_histogram > 0:
    trend = "BULLISH"
elif ema_short < ema_long and macd_histogram < 0:
    trend = "BEARISH"
else:
    trend = "NEUTRAL"
```

**Data Model:**

```python
@dataclass
class TechnicalIndicators:
    rsi: float                  # 0-100
    macd_value: float          # MACD line
    macd_signal: float         # Signal line
    macd_histogram: float      # MACD - Signal
    ema_short: float           # 9-period EMA
    ema_long: float            # 21-period EMA
    sma_50: float              # 50-period SMA
    volume_change: float       # % change
    price_velocity: float      # $/minute
    trend: Literal["BULLISH", "BEARISH", "NEUTRAL"]
```

**Graceful Degradation:**
```python
# If price history unavailable or insufficient:
indicators = TechnicalIndicators(
    rsi=50.0,                      # Neutral
    macd_value=0.0,                # No momentum
    macd_signal=0.0,
    macd_histogram=0.0,
    ema_short=float(current_price),
    ema_long=float(current_price),
    sma_50=float(current_price),
    volume_change=0.0,
    price_velocity=0.0,
    trend="NEUTRAL"
)
```

---

## Signal Processing Pipeline

### Overview

Raw signals from 5 sources → Scoring → Aggregation → Confidence Adjustment

```
BTCPriceService         → Current Price + History
SocialSentimentService  → Social Score (-0.7 to +0.85)
MarketMicrostructure    → Market Score (-1.0 to +1.0)
BTC Momentum            → Direction (UP/DOWN/FLAT)
Technical Analysis      → Indicators + Trend

         ↓

SignalAggregator:
  final_score = market(60%) + social(40%)
  agreement = calculate_alignment(market, social)
  final_confidence = base_confidence * agreement

         ↓

AggregatedSentiment → To AI Decision Engine
```

---

### SignalAggregator Logic

**Purpose:** Combine market microstructure and social sentiment with dynamic confidence adjustment based on signal agreement.

#### Weighting Strategy

```python
SOCIAL_WEIGHT = 0.4   # 40% - Community sentiment
MARKET_WEIGHT = 0.6   # 60% - Actual market behavior

final_score = (market.score * 0.6) + (social.score * 0.4)
```

**Rationale:**
- Market microstructure shows *actual* trading behavior (real money)
- Social sentiment shows *perceived* market mood (community views)
- Market behavior weighted higher as it represents concrete actions

#### Agreement Multiplier

**Purpose:** Boost confidence when signals align, penalize when they conflict

**Formula:**

```python
def _calculate_agreement_score(score1: float, score2: float) -> float:
    """
    Calculate agreement multiplier.

    Returns:
        0.5 (total conflict) to 1.5 (perfect agreement)
    """
    # Both same direction (both positive OR both negative)
    if score1 * score2 > 0:
        # How aligned? (0 to 1)
        alignment = 1 - abs(score1 - score2) / 2
        # Boost confidence (1.0 to 1.5x)
        return 1.0 + (alignment * 0.5)

    # Opposite directions (conflict)
    elif score1 * score2 < 0:
        # How conflicted? (0 to 1)
        conflict = abs(score1 - score2) / 2
        # Penalize confidence (0.5 to 1.0x)
        return 1.0 - (conflict * 0.5)

    # One or both neutral
    else:
        return 1.0  # No change
```

**Examples:**

| Market Score | Social Score | Agreement | Multiplier | Interpretation |
|--------------|--------------|-----------|------------|----------------|
| +0.8 | +0.7 | High | 1.45x | Both strongly bullish → boost |
| +0.5 | +0.3 | Moderate | 1.2x | Both moderately bullish → small boost |
| +0.6 | -0.5 | Conflict | 0.72x | Market bullish, social bearish → penalty |
| -0.7 | -0.8 | High | 1.48x | Both strongly bearish → boost |
| +0.2 | -0.1 | Low | 0.92x | Slight conflict → small penalty |
| 0.0 | +0.5 | Neutral | 1.0x | One neutral → no change |

#### Confidence Calculation

```python
# Base confidence from individual source confidences
base_confidence = (social.confidence + market.confidence) / 2

# Apply agreement multiplier
final_confidence = min(base_confidence * agreement_multiplier, 1.0)
```

**Example:**
```
market.confidence = 1.0
social.confidence = 1.0
base_confidence = 1.0

market.score = +0.6 (bullish)
social.score = -0.4 (bearish)
agreement = 0.5x (conflict)

final_confidence = 1.0 * 0.5 = 0.50
```

#### Fallback Modes

**Scenario 1: Only Market Available**
```python
final_score = market.score
final_confidence = market.confidence * 0.7  # 70% penalty
signal_type = f"MARKET_ONLY_{market.signal_type}"
```

**Scenario 2: Only Social Available**
```python
final_score = social.score
final_confidence = social.confidence * 0.7  # 70% penalty
signal_type = f"SOCIAL_ONLY_{social.signal_type}"
```

**Scenario 3: Both Unavailable**
```python
final_score = 0.0
final_confidence = 0.0
signal_type = "TECHNICAL_ONLY"
# AI will rely solely on technical indicators
```

#### Signal Classification

```python
def _classify_signal(score: float, confidence: float) -> str:
    """Classify combined signal strength and direction."""

    # Determine direction
    if score > 0.1:
        direction = "BULLISH"
    elif score < -0.1:
        direction = "BEARISH"
    else:
        direction = "NEUTRAL"

    # Determine strength
    if confidence >= 0.7:
        strength = "STRONG"
    elif confidence >= 0.5:
        strength = "WEAK"
    else:
        strength = "CONFLICTED"

    return f"{strength}_{direction}"
```

**Signal Types:**
- `STRONG_BULLISH`: Score > 0.1, Confidence ≥ 0.7
- `WEAK_BULLISH`: Score > 0.1, Confidence 0.5-0.7
- `CONFLICTED_BULLISH`: Score > 0.1, Confidence < 0.5
- `STRONG_BEARISH`: Score < -0.1, Confidence ≥ 0.7
- `WEAK_BEARISH`: Score < -0.1, Confidence 0.5-0.7
- `CONFLICTED_BEARISH`: Score < -0.1, Confidence < 0.5
- `STRONG_NEUTRAL`: Score -0.1 to 0.1, Confidence ≥ 0.7
- `WEAK_NEUTRAL`: Score -0.1 to 0.1, Confidence 0.5-0.7
- `CONFLICTED_NEUTRAL`: Score -0.1 to 0.1, Confidence < 0.5

#### Data Model

```python
@dataclass
class AggregatedSentiment:
    social: SocialSentiment         # Original social data
    market: MarketSignals           # Original market data
    final_score: float              # Weighted combination
    final_confidence: float         # Agreement-adjusted
    agreement_multiplier: float     # 0.5x to 1.5x
    signal_type: str               # Classification
    timestamp: datetime
```

---

## Orderbook Tracking & History

### Historical Implementation (Deprecated)

**Original Approach:** Binance BTC/USDT order book analysis

#### Data Collection

```python
async def _fetch_order_book(self) -> dict:
    """Fetch Binance order book (top 100 levels)."""
    url = f"{BASE_URL}/depth?symbol=BTCUSDT&limit=100"

    async with session.get(url) as response:
        data = await response.json()
        # Returns:
        # {
        #   "bids": [["67500.00", "1.5"], ["67499.50", "2.3"], ...],
        #   "asks": [["67500.50", "1.8"], ["67501.00", "2.1"], ...]
        # }
        return data
```

#### Wall Detection

**Definition:** Large orders that create "walls" of liquidity

**Threshold:** Orders > 10 BTC considered significant walls

```python
def _score_order_book(self, order_book: dict) -> float:
    """
    Score order book bid vs ask wall strength.

    Returns:
        -1.0 (heavy ask walls) to +1.0 (heavy bid walls)
    """
    bids = order_book.get("bids", [])
    asks = order_book.get("asks", [])

    LARGE_WALL_BTC = 10.0

    # Sum BTC in large bid walls
    bid_walls = sum(
        float(qty) for price, qty in bids
        if float(qty) > LARGE_WALL_BTC
    )

    # Sum BTC in large ask walls
    ask_walls = sum(
        float(qty) for price, qty in asks
        if float(qty) > LARGE_WALL_BTC
    )

    if bid_walls + ask_walls == 0:
        return 0.0

    # Normalize to -1 to +1
    score = (bid_walls - ask_walls) / (bid_walls + ask_walls)
    return score
```

**Interpretation:**
- +1.0: Heavy bid walls, light ask walls → Strong buying support
- +0.5: More bid walls than ask walls → Moderate buying support
- 0.0: Balanced walls
- -0.5: More ask walls than bid walls → Moderate selling pressure
- -1.0: Heavy ask walls, light bid walls → Strong selling pressure

#### Why Deprecated

**Problem:** Binance BTC/USDT order book doesn't reflect Polymarket market dynamics

**Reason for Change:**
1. **Different Market:** Binance is spot trading, Polymarket is prediction market
2. **Different Assets:** BTC/USDT orderbook vs YES/NO tokens
3. **Lag Issue:** Binance orderbook state doesn't predict Polymarket outcome
4. **Better Alternative:** Polymarket CLOB shows *actual* market trades with YES/NO direction

**Transition:** Replaced with real-time trade analysis (momentum, volume flow, whale activity)

---

### Current Implementation

**Orderbook analysis is NO LONGER USED in market scoring.**

**Current Status in Code:**
```python
# In MarketSignals dataclass:
order_book_score: float = 0.0        # Not calculated
order_book_bias: str = "N/A"         # Not used
```

**Replacement Strategy:**

Instead of orderbook walls, we now track:
1. **Momentum Score** - YES token price movement (actual executed trades)
2. **Volume Flow Score** - YES vs NO volume (actual trading direction)
3. **Whale Activity Score** - Large trades direction (actual conviction)

**Advantages:**
- Real-time trade execution data (not just placed orders)
- Direct YES/NO token relevance
- Can't be spoofed with fake walls
- Shows actual money flow, not just intent

---

## Whale Detection & Tracking

### Definition

**Whale Trade:** Any single trade with size > $1,000 USD

**Rationale:**
- Average trade size on Polymarket BTC markets: $50-200
- Trades > $1,000 represent 5-20x average size
- Indicates high conviction from sophisticated traders
- Often precedes larger market movements

### Detection Logic

```python
WHALE_SIZE_USD = 1000  # Threshold in dollars

def calculate_whale_activity_score(self, trades: list) -> float:
    """
    Calculate directional signal from whale trades.

    Args:
        trades: List of trade messages with asset_id and size

    Returns:
        -1.0 (all NO whales) to +1.0 (all YES whales)
    """
    # Count YES whales
    yes_whales = sum(
        1 for trade in trades
        if float(trade['size']) > WHALE_SIZE_USD
        and self._is_yes_token(trade.get('asset_id'))
    )

    # Count NO whales
    no_whales = sum(
        1 for trade in trades
        if float(trade['size']) > WHALE_SIZE_USD
        and self._is_no_token(trade.get('asset_id'))
    )

    total_whales = yes_whales + no_whales

    if total_whales == 0:
        return 0.0  # No whales detected

    # Directional score
    whale_score = (yes_whales - no_whales) / total_whales

    return whale_score
```

### Token Classification

**Challenge:** Identify which token is YES vs NO from token IDs

**Solution:**
```python
def _is_yes_token(self, asset_id: str) -> bool:
    """
    Check if asset_id is the YES token.

    YES token = first token_id in market's token pair
    """
    if not asset_id:
        return False

    # Test data compatibility
    if asset_id == 'YES_TOKEN':
        return True

    # Real data: compare to first token_id
    if self.token_ids:
        return asset_id == str(self.token_ids[0])

    return False

def _is_no_token(self, asset_id: str) -> bool:
    """
    Check if asset_id is the NO token.

    NO token = second token_id in market's token pair
    """
    if not asset_id:
        return False

    # Test data compatibility
    if asset_id == 'NO_TOKEN':
        return True

    # Real data: compare to second token_id
    if self.token_ids and len(self.token_ids) >= 2:
        return asset_id == str(self.token_ids[1])

    return False
```

### Whale Metrics

**Metadata Tracked:**

```python
# In MarketSignals:
whale_score: float              # -1.0 to +1.0 directional score
whale_count: int               # Total number of whale trades
whale_direction: str           # "BUYING", "SELLING", "NEUTRAL"
```

**Direction Classification:**
```python
if whale_score > 0.3:
    whale_direction = "BUYING"      # Net YES buying
elif whale_score < -0.3:
    whale_direction = "SELLING"     # Net NO buying
else:
    whale_direction = "NEUTRAL"     # Balanced or no whales
```

### Weight in Final Score

**Current Weight:** 30% of market microstructure score (increased from 25%)

**Rationale for Increase:**
- Whale trades show conviction
- Often lead price movements (not follow)
- Less susceptible to lag than momentum
- Behavioral indicator (real money decisions)

**Combined Calculation:**
```python
market_score = (
    momentum * 0.20 +
    volume_flow * 0.50 +
    whale_activity * 0.30  # ← Increased weight
)
```

### Example Scenarios

#### Scenario 1: Strong Whale Buying
```
Collection window: 2 minutes
Total trades: 45

Whale trades:
- 5 YES trades > $1,000 (sizes: $1,200, $1,500, $2,000, $1,100, $3,500)
- 1 NO trade > $1,000 (size: $1,300)

Calculation:
yes_whales = 5
no_whales = 1
whale_score = (5 - 1) / (5 + 1) = 4/6 = +0.67

Classification: BUYING (>0.3)
Contribution to market_score: +0.67 * 0.30 = +0.20
```

#### Scenario 2: No Whale Activity
```
Collection window: 2 minutes
Total trades: 32

Whale trades: 0 (all trades < $1,000)

Calculation:
whale_score = 0.0 (no whales)

Classification: NEUTRAL
Contribution to market_score: 0.0 * 0.30 = 0.00
```

#### Scenario 3: Conflicting Whales
```
Collection window: 2 minutes
Total trades: 28

Whale trades:
- 3 YES trades > $1,000
- 3 NO trades > $1,000

Calculation:
yes_whales = 3
no_whales = 3
whale_score = (3 - 3) / (3 + 3) = 0/6 = 0.0

Classification: NEUTRAL (balanced)
Contribution to market_score: 0.0 * 0.30 = 0.00
```

---

## Technical Indicators Methodology

### Calculation Pipeline

```
1. Fetch 60-minute price history
      ↓
2. Convert to DataFrame (if pandas available)
      ↓
3. Calculate all indicators:
   - RSI (14-period)
   - MACD (12, 26, 9)
   - EMA (9 vs 21)
   - SMA (50)
   - Volume Change
   - Price Velocity
      ↓
4. Classify Trend (BULLISH/BEARISH/NEUTRAL)
      ↓
5. Return TechnicalIndicators dataclass
```

### Implementation Details

**With Pandas (Fast Path):**
```python
def _calculate_with_pandas(price_history: list[PricePoint]) -> TechnicalIndicators:
    # Create DataFrame
    df = pd.DataFrame([
        {
            "price": float(p.price),
            "volume": float(p.volume),
            "timestamp": p.timestamp
        }
        for p in price_history
    ])

    # RSI
    rsi = _calculate_rsi(df["price"], 14)

    # EMAs
    ema_short = df["price"].ewm(span=9, adjust=False).mean().iloc[-1]
    ema_long = df["price"].ewm(span=21, adjust=False).mean().iloc[-1]
    sma_50 = df["price"].rolling(window=50).mean().iloc[-1] if len(df) >= 50 else ema_long

    # MACD
    macd_12 = df["price"].ewm(span=12, adjust=False).mean()
    macd_26 = df["price"].ewm(span=26, adjust=False).mean()
    macd_line = macd_12 - macd_26
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    macd_histogram = macd_line - macd_signal

    # Volume
    recent_vol = df["volume"].tail(5).mean()
    avg_vol = df["volume"].tail(30).mean()
    volume_change = ((recent_vol - avg_vol) / avg_vol) * 100

    # Velocity
    price_change = df["price"].iloc[-1] - df["price"].iloc[-5]
    velocity = price_change / 5  # $/minute

    # Trend
    if ema_short > ema_long and macd_histogram.iloc[-1] > 0:
        trend = "BULLISH"
    elif ema_short < ema_long and macd_histogram.iloc[-1] < 0:
        trend = "BEARISH"
    else:
        trend = "NEUTRAL"

    return TechnicalIndicators(...)
```

**Without Pandas (Fallback):**
```python
def _calculate_manual(price_history: list[PricePoint]) -> TechnicalIndicators:
    prices = [float(p.price) for p in price_history]

    # Simple EMA
    def ema(data: list[float], span: int) -> float:
        multiplier = 2 / (span + 1)
        ema_val = data[0]
        for price in data[1:]:
            ema_val = (price * multiplier) + (ema_val * (1 - multiplier))
        return ema_val

    # Simple RSI
    def rsi(prices: list[float], period: int) -> float:
        gains, losses = [], []
        for i in range(1, len(prices)):
            change = prices[i] - prices[i-1]
            gains.append(max(change, 0))
            losses.append(abs(min(change, 0)))

        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    ema_short = ema(prices, 9)
    ema_long = ema(prices, 21)
    rsi_val = rsi(prices, 14)

    # Simple trend
    trend = "BULLISH" if ema_short > ema_long else "BEARISH"

    return TechnicalIndicators(
        rsi=rsi_val,
        ema_short=ema_short,
        ema_long=ema_long,
        macd_value=0.0,  # Not calculated in fallback
        ...
    )
```

### Usage in AI Prompt

Technical indicators are formatted as context for AI analysis:

```
TECHNICAL INDICATORS (60-min analysis):
- RSI(14): 58.3 (Overbought >70, Oversold <30)
- MACD: 45.20 (Signal: 42.10)
- MACD Histogram: 3.10 (Positive momentum)
- EMA Trend: 67,520.00 vs 67,450.00 (Short > Long = Bullish)
- Trend: BULLISH
- Volume Change: +23.5% (Above average activity)
- Price Velocity: +$2.50/min (Moderate upward movement)
```

**AI Interpretation Guidance:**
- RSI extreme levels suggest potential reversal
- MACD crossovers indicate momentum shifts
- EMA alignment confirms trend direction
- High volume validates moves
- Velocity shows rate of change

### Graceful Degradation

**If insufficient data (<26 candles):**
```python
logger.warning("Insufficient data for all indicators", points=len(price_history))
```

**If complete failure:**
```python
# Return neutral indicators
TechnicalIndicators(
    rsi=50.0,                          # Neutral
    macd_value=0.0,                    # No momentum
    macd_signal=0.0,
    macd_histogram=0.0,
    ema_short=float(current_price),    # Current price
    ema_long=float(current_price),
    sma_50=float(current_price),
    volume_change=0.0,
    price_velocity=0.0,
    trend="NEUTRAL"
)
```

AI receives fallback data and continues decision process without technical indicators.

---

## Historical Data Fetching

### BTC Price History

**Source:** Binance Public API
**Endpoint:** `https://api.binance.com/api/v3/klines`

#### API Parameters

```python
params = {
    "symbol": "BTCUSDT",        # BTC/USDT pair
    "interval": "1m",           # 1-minute candles
    "limit": str(minutes)       # Number of candles
}
```

**Supported Intervals:**
- `1m`: 1-minute candles (used for all analysis)
- `5m`: 5-minute candles (not used)
- `15m`: 15-minute candles (not used)
- `1h`: 1-hour candles (not used)

#### Response Format

```json
[
  [
    1707648000000,      // Open time (ms timestamp)
    "67500.00",         // Open price
    "67550.00",         // High price
    "67480.00",         // Low price
    "67520.00",         // Close price (index 4)
    "125.50",           // Volume (index 5)
    1707648059999,      // Close time
    "8472500.00",       // Quote asset volume
    1250,               // Number of trades
    "62.75",            // Taker buy base asset volume
    "4236250.00",       // Taker buy quote asset volume
    "0"                 // Ignore
  ],
  ...
]
```

#### Data Transformation

```python
async def get_price_history(self, minutes: int = 60) -> list[PricePoint]:
    """
    Get historical price points for technical analysis.

    Args:
        minutes: Number of 1-minute candles to fetch (max 1000)

    Returns:
        List of PricePoint objects sorted by timestamp (oldest first)
    """
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": "BTCUSDT", "interval": "1m", "limit": str(minutes)}

    async with session.get(url, params=params, timeout=10) as resp:
        data = await resp.json()

        return [
            PricePoint(
                price=Decimal(str(candle[4])),     # Close price
                volume=Decimal(str(candle[5])),     # Volume
                timestamp=datetime.fromtimestamp(candle[0] / 1000)
            )
            for candle in data
        ]
```

#### Usage Patterns

**Technical Analysis (60 minutes):**
```python
history = await btc_service.get_price_history(minutes=60)
indicators = TechnicalAnalysis.calculate_indicators(history)
```

**BTC Momentum (5 minutes):**
```python
history = await btc_service.get_price_history(minutes=5)
price_5min_ago = history[0].price
current_price = btc_data.price
momentum_pct = (current_price - price_5min_ago) / price_5min_ago * 100
```

#### Error Handling

**Timeout Protection:**
```python
timeout=10  # 10-second timeout per request
```

**Retry Logic:** Not implemented (relies on graceful degradation)

**Fallback Behavior:**
- If fetch fails → Technical indicators return neutral defaults
- If fetch fails → BTC momentum returns None
- System continues without historical data

---

### Polymarket CLOB Trade History

**Source:** Polymarket CLOB WebSocket
**URL:** `wss://ws-subscriptions-clob.polymarket.com/ws/market`

#### Collection Method

```python
async def collect_market_data_with_token_ids(
    self,
    token_ids: list[str],
    duration_seconds: int = 120
) -> dict:
    """
    Connect to CLOB WebSocket and collect market data for 2 minutes.

    Args:
        token_ids: List of token IDs (assets_ids) for YES/NO tokens
        duration_seconds: Collection duration (default 120 seconds)

    Returns:
        {
            'trades': list[dict],          # Trade execution messages
            'book_snapshots': list[dict],  # Orderbook snapshots (not used)
            'price_changes': list[dict],   # Price change events (not used)
            'collection_duration': int     # Actual seconds collected
        }
    """
    accumulated_data = {
        'trades': [],
        'book_snapshots': [],
        'price_changes': [],
        'collection_duration': 0
    }

    async with websockets.connect(WS_URL) as ws:
        # Subscribe to market
        subscribe_msg = {
            "assets_ids": token_ids,
            "type": "market"
        }
        await ws.send(json.dumps(subscribe_msg))

        # Collect for 2 minutes
        start_time = time.time()
        while time.time() - start_time < duration_seconds:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
                data = json.loads(msg)

                # Handle message arrays or single messages
                messages = data if isinstance(data, list) else [data]

                for message in messages:
                    msg_type = message.get('event_type')
                    if msg_type == 'last_trade_price':
                        accumulated_data['trades'].append(message)
            except asyncio.TimeoutError:
                continue  # No message in 5s, keep waiting

        accumulated_data['collection_duration'] = int(time.time() - start_time)

        return accumulated_data
```

#### Trade Message Format

```json
{
  "event_type": "last_trade_price",
  "asset_id": "95248311319330480098743674418977156070443495373353768046472795885598409789134",
  "price": "0.52",
  "size": "250.00",
  "timestamp": 1707648120
}
```

**Key Fields:**
- `asset_id`: Token ID (YES or NO)
- `price`: Execution price (0.0-1.0 for prediction markets)
- `size`: Trade size in USD
- `timestamp`: Unix timestamp

#### Usage

**Momentum Calculation:**
```python
trades = data['trades']
yes_trades = [t for t in trades if _is_yes_token(t['asset_id'])]

initial_yes_price = yes_trades[0]['price']
final_yes_price = yes_trades[-1]['price']
momentum = (final_yes_price - initial_yes_price) / initial_yes_price
```

**Volume Flow:**
```python
yes_volume = sum(float(t['size']) for t in trades if _is_yes_token(t['asset_id']))
no_volume = sum(float(t['size']) for t in trades if _is_no_token(t['asset_id']))
flow = (yes_volume - no_volume) / (yes_volume + no_volume)
```

**Whale Detection:**
```python
yes_whales = sum(
    1 for t in trades
    if float(t['size']) > 1000 and _is_yes_token(t['asset_id'])
)
```

---

## AI Decision Engine

### Model Configuration

**Model:** GPT-5-Nano (`gpt-o1-mini`)
**Provider:** OpenAI API
**Key Features:**
- Built-in reasoning tokens (think before answering)
- Optimized for logical analysis
- Faster than GPT-4 with comparable reasoning

#### API Configuration

```python
response = await client.chat.completions.create(
    model="gpt-o1-mini",
    messages=[
        {
            "role": "system",
            "content": "You are an autonomous trading bot for Polymarket BTC 15-minute up/down markets. Use reasoning tokens to analyze all signals carefully. Always return valid JSON."
        },
        {"role": "user", "content": prompt}
    ],
    temperature=1.0,  # Only supported temperature for GPT-5-Nano
    reasoning_effort="medium",  # minimal/low/medium/high
    max_completion_tokens=8000,  # Includes reasoning (~2k-4k) + output (~1k)
    response_format={"type": "json_object"}
)
```

**Reasoning Effort Levels:**
- `minimal`: Fast decisions (~5s), basic analysis
- `low`: Quick reasoning (~8s), simple patterns
- `medium`: Balanced (~15s), thorough analysis **← DEFAULT**
- `high`: Deep reasoning (~20s+), complex scenarios

### Prompt Engineering

#### Prompt Structure (13 Sections)

1. **Role & Instructions** - Bot identity and objective
2. **Price-to-Beat Context** - Starting price vs current
3. **Signal Validation Rules** - Contradiction detection logic
4. **Market Timing** - Time remaining and phase
5. **BTC Actual Momentum** - 5-minute movement
6. **Current Market Data** - Question, outcomes, odds
7. **Technical Indicators** - 60-minute analysis
8. **Social Sentiment** - Fear/Greed, trending, votes
9. **Market Microstructure** - CLOB analysis
10. **Aggregated Signal** - Final score and confidence
11. **Risk Parameters** - Thresholds and limits
12. **Decision Instructions** - Analysis framework
13. **Output Format** - JSON schema

#### Critical Section: Validation Rules

**Purpose:** Prevent betting against actual BTC direction (lagging indicator fix)

```
⚠️ SIGNAL VALIDATION RULES:

You MUST check for contradictions between market signals and actual BTC movement:

1. **BEARISH Signal + BTC Actually UP:**
   - If aggregated market score < -0.3 (BEARISH)
   - AND BTC is UP from price-to-beat (+0.30%)
   - → This is a CONTRADICTION - market is lagging behind reality
   - → Decision: HOLD (do NOT bet NO when BTC is going UP)

2. **BULLISH Signal + BTC Actually DOWN:**
   - If aggregated market score > +0.3 (BULLISH)
   - AND BTC is DOWN from price-to-beat (-0.30%)
   - → This is a CONTRADICTION - market is lagging behind reality
   - → Decision: HOLD (do NOT bet YES when BTC is going DOWN)

3. **Signals ALIGN:**
   - If market sentiment matches actual BTC direction
   - → Proceed with normal confidence-based decision

**Why This Matters:**
- Polymarket sentiment shows what traders THINK, not what IS happening
- The 2-minute collection window often lags actual BTC movement
- Following contradictory signals leads to consistent losses
- Example: Market says "bearish" based on old data, but BTC already bounced

**When to Override:**
- Only if you have VERY STRONG conviction (>0.95 confidence)
- AND can explain in reasoning why the contradiction is temporary
- Otherwise: HOLD and wait for signals to align
```

#### BTC Momentum Context

```
ACTUAL BTC MOMENTUM (last 5 minutes):
- 5 minutes ago: $67,500.00
- Current: $67,650.00
- Change: +0.22% (UP)

⚠️ COMPARE WITH MARKET SIGNALS:
- If market sentiment is BEARISH but BTC is UP → market is LAGGING
- If market sentiment is BULLISH but BTC is DOWN → market is LAGGING
- Lagging signals often lead to losing trades - consider HOLD
```

#### Market Timing Context

```
MARKET TIMING:
- Time Remaining: 7m 30s
- Market Phase: 🟢 EARLY/MID PHASE

⚠️ END-OF-MARKET STRATEGY (< 3 min):
- Trend is likely established (less time for reversal)
- Price movements now have higher predictive value
- If signals strongly align, confidence can be boosted
- Still require full analysis - no rushed decisions
```

### Decision Output Format

**Required JSON Schema:**

```json
{
  "action": "YES" | "NO" | "HOLD",
  "confidence": 0.85,
  "reasoning": "Market signals show strong bearish with -0.47 score and full confidence. However, BTC price-to-beat shows UP +0.30%, and 5-min momentum confirms UP +0.22%. This is a clear contradiction - market sentiment lags actual BTC movement. HOLD to avoid betting against reality.",
  "confidence_adjustment": "-0.15",
  "position_size": 50.0,
  "stop_loss": 0.40
}
```

**Field Definitions:**

- **action**: Trading action
  - `"YES"`: Buy first outcome token (e.g., "Up")
  - `"NO"`: Buy second outcome token (e.g., "Down")
  - `"HOLD"`: Don't trade

- **confidence**: Final confidence level (0.0-1.0)
  - Must be ≥ 0.70 (default threshold) to trade
  - Includes AI's adjustment (±0.15 max)

- **reasoning**: 2-3 sentence explanation
  - Must reference key signals
  - Must explain contradiction if HOLD due to validation rules
  - Must explain confidence adjustment if applied

- **confidence_adjustment**: AI's adjustment to aggregated confidence
  - Range: -0.15 to +0.15
  - Applied when AI spots patterns not captured by aggregator

- **position_size**: Trade size in USDC
  - Will be validated by RiskManager
  - Max 5% of portfolio (default)

- **stop_loss**: Odds threshold to exit position
  - Range: 0.0-1.0
  - Default: 0.40 (exit if odds drop 40%+)

### Decision Logic Flow

```
1. Parse all input data (price, signals, timing, momentum)
      ↓
2. CHECK VALIDATION RULES FIRST
   - Price-to-beat direction (UP/DOWN/FLAT)
   - Market signal direction (BULLISH/BEARISH)
   - BTC momentum direction (UP/DOWN/FLAT)
   - Any contradictions? YES → HOLD
      ↓
3. Analyze signal strength
   - Aggregated confidence level
   - Agreement multiplier
   - Individual signal strengths
      ↓
4. Consider market timing
   - End-of-market? (< 3 min)
   - If yes + signals align → boost confidence
      ↓
5. Review technical indicators
   - Do they confirm or contradict?
   - Extreme levels (RSI overbought/oversold)?
      ↓
6. Adjust confidence (±0.15 max)
   - Spotted pattern aggregator missed?
   - Suspicious data quality?
      ↓
7. Compare to threshold (0.70 default)
   - Below threshold → HOLD
   - Above threshold → Trade (YES or NO)
      ↓
8. Generate reasoning
   - Explain key factors
   - Justify confidence adjustment
   - Note any contradictions
      ↓
9. Return JSON decision
```

### Example AI Reasoning

**Scenario: Contradiction Detected (HOLD)**

```json
{
  "action": "HOLD",
  "confidence": 0.00,
  "reasoning": "Aggregated signals show STRONG_BEARISH (-0.32 score, 1.00 confidence) based on 2-minute market collection. However, price-to-beat shows BTC UP +0.30% from start, and 5-minute momentum confirms UP +0.25%. This is a clear contradiction - market sentiment reflects past drop but BTC has since rebounded. Signals are lagging actual movement. HOLD until alignment.",
  "confidence_adjustment": "0.0",
  "position_size": 0,
  "stop_loss": 0.40
}
```

**Scenario: Strong Alignment (TRADE)**

```json
{
  "action": "YES",
  "confidence": 0.95,
  "reasoning": "All signals align bullish. Market microstructure +0.47 (strong) with whale support. Social +0.30 (moderate). Price-to-beat UP +0.44%. BTC momentum UP +0.15%. Technical indicators confirm bullish trend (RSI 58, MACD positive, EMA bullish). No contradictions. High conviction trade.",
  "confidence_adjustment": "+0.10",
  "position_size": 50.0,
  "stop_loss": 0.40
}
```

**Scenario: End-of-Market Boost (TRADE)**

```json
{
  "action": "NO",
  "confidence": 0.90,
  "reasoning": "Bearish signals align with actual BTC movement. Market -0.55 (strong) with selling pressure. Price-to-beat DOWN -0.15%. BTC momentum DOWN -0.18%. Only 2m 30s remaining - trend is established with limited reversal time. End-of-market timing justifies confidence boost. Strong conviction bearish.",
  "confidence_adjustment": "+0.15",
  "position_size": 50.0,
  "stop_loss": 0.40
}
```

---

## Risk Management & Validation

### RiskManager Role

**Purpose:** Final validation layer before trade execution

**When:** After AI makes YES/NO decision (before execution)

**What:** Validates decision against portfolio limits and risk parameters

### Validation Checks

#### 1. Confidence Threshold

```python
if decision.confidence < settings.bot_confidence_threshold:
    return ValidationResult(
        approved=False,
        reason=f"Confidence {decision.confidence:.2f} < threshold {threshold:.2f}"
    )
```

**Default:** 0.70 (70%)
**Rationale:** Only trade when reasonably confident

#### 2. Position Size Limits

```python
max_position = portfolio_value * settings.bot_max_position_percent

if decision.position_size > max_position:
    # Reduce to max allowed
    adjusted_position = max_position
    return ValidationResult(
        approved=True,
        adjusted_position=adjusted_position,
        reason=f"Position reduced from {decision.position_size} to {adjusted_position}"
    )
```

**Default:** 5% of portfolio
**Rationale:** Limit exposure per trade

#### 3. Portfolio Concentration

```python
# Check total exposure to this market
existing_exposure = sum(
    p['amount'] for p in open_positions
    if p['market_id'] == market['id']
)

if existing_exposure + decision.position_size > max_market_exposure:
    return ValidationResult(
        approved=False,
        reason=f"Market exposure would exceed {max_market_exposure}"
    )
```

**Default:** 10% max per market
**Rationale:** Avoid over-concentration

#### 4. Daily Trade Limits

```python
if trades_today >= settings.bot_max_daily_trades:
    return ValidationResult(
        approved=False,
        reason=f"Daily trade limit reached ({trades_today}/{max_daily_trades})"
    )
```

**Default:** 20 trades per day
**Rationale:** Prevent overtrading

#### 5. Stop-Loss Validation

```python
if decision.stop_loss_threshold < 0.2 or decision.stop_loss_threshold > 0.8:
    # Use default if invalid
    adjusted_stop_loss = 0.40
```

**Valid Range:** 0.2 to 0.8
**Default:** 0.40 (40% loss triggers exit)
**Rationale:** Reasonable risk/reward

### Validation Output

```python
@dataclass
class ValidationResult:
    approved: bool                     # Trade approved?
    adjusted_position: Decimal        # Modified position size (if adjusted)
    reason: str                       # Why approved/rejected/adjusted
```

**Possible Outcomes:**

1. **Approved (No Changes):**
```python
ValidationResult(
    approved=True,
    adjusted_position=decision.position_size,
    reason="All checks passed"
)
```

2. **Approved (Position Reduced):**
```python
ValidationResult(
    approved=True,
    adjusted_position=Decimal("45.0"),  # Reduced from 50
    reason="Position reduced to stay within 5% portfolio limit"
)
```

3. **Rejected:**
```python
ValidationResult(
    approved=False,
    adjusted_position=Decimal("0"),
    reason="Confidence 0.65 < threshold 0.70"
)
```

### Odds-Adjusted Position Sizing

**Added:** 2026-02-12
**Purpose:** Prevent large losses on low-probability bets by scaling position size based on odds

#### Problem Addressed

Prior to this fix, the bot was betting MORE on low-odds (risky) trades:
- **Trade #273**: Bet $9.56 on 0.31 odds (31% probability) → Lost entire $9.56 stake
- **Trade #269**: Bet $5.00 on 0.83 odds (83% probability) → Won $1.02

This is backwards! Low odds mean you risk your entire stake to win small, while high odds mean you risk stake to win large.

#### Solution: Odds-Based Scaling

Position sizes are now scaled down for low-odds bets:

```python
def _calculate_odds_multiplier(self, odds: Decimal) -> Decimal:
    """
    Scale down position size for low-odds bets.

    Logic:
    - odds >= 0.50: No scaling (100% of position)
    - odds < 0.50:  Linear scale from 100% down to 50%
    - odds < 0.25:  Reject bet entirely (too risky)
    """
    MINIMUM_ODDS = Decimal("0.25")
    SCALE_THRESHOLD = Decimal("0.50")

    if odds < MINIMUM_ODDS:
        return Decimal("0")  # Reject bet

    if odds >= SCALE_THRESHOLD:
        return Decimal("1.0")  # No scaling needed

    # Linear interpolation between 0.5x and 1.0x
    multiplier = Decimal("0.5") + (odds - MINIMUM_ODDS) / (SCALE_THRESHOLD - MINIMUM_ODDS) * Decimal("0.5")
    return multiplier
```

**Examples:**
- **0.83 odds** → 1.00x multiplier (no reduction)
- **0.50 odds** → 1.00x multiplier (breakeven)
- **0.40 odds** → 0.80x multiplier (20% reduction)
- **0.31 odds** → 0.62x multiplier (38% reduction)
- **0.25 odds** → 0.50x multiplier (50% reduction, minimum)
- **0.20 odds** → REJECTED (below threshold)

#### Validation Check 3a: Odds Rejection

Added to `validate_decision()`:

```python
# Extract odds for the action
odds = self._extract_odds_for_action(decision.action, market)

# Calculate position size with odds awareness
suggested_size = self._calculate_position_size(
    decision, portfolio_value, max_position, odds
)

# Check 3a: Reject if odds below minimum threshold
if suggested_size == Decimal("0"):
    return ValidationResult(
        approved=False,
        reason=f"Odds {float(odds):.2f} below minimum threshold 0.25",
        adjusted_position=None
    )
```

#### Expected Impact

With odds-adjusted sizing, the historical losing trades would have been:

**Trade #273 (LOSS):**
- **Before**: $9.56 stake at 0.31 odds → Lost $9.56
- **After**: $5.93 stake at 0.31 odds → Would lose $5.93
- **Improvement**: Saves $3.63 (38% reduction in loss)

**Trade #269 (WIN):**
- **Before**: $5.00 stake at 0.83 odds → Won $1.02
- **After**: $5.00 stake at 0.83 odds → Win $1.02
- **Impact**: Unchanged (high odds, no scaling)

**Net P&L:**
- **Before**: -$8.54 (over 2 trades)
- **After**: -$4.91
- **Improvement**: 44% reduction in losses

With 55-60% win rate (achievable with good signals), this adjustment should lead to profitability.

### Configuration Parameters

**All risk parameters in Settings:**

```python
@dataclass
class Settings:
    # Trading mode
    mode: str = "read_only"  # "read_only" or "trading"

    # Confidence
    bot_confidence_threshold: float = 0.70  # Minimum confidence to trade

    # Position sizing
    bot_max_position_percent: float = 0.05  # 5% of portfolio max
    bot_max_market_exposure: float = 0.10   # 10% max per market

    # Trade limits
    bot_max_daily_trades: int = 20          # Max trades per day

    # Stop-loss
    bot_stop_loss_percent: float = 0.15     # 15% loss triggers exit

    # Logging
    bot_log_decisions: bool = True          # Log all AI decisions
```

---

## Complete Workflow Timeline

### Full 3-Minute Cycle Breakdown

```
┌─────────────────────────────────────────────────────────────────────┐
│                    T+0s: CYCLE START                                │
└─────────────────────────────────────────────────────────────────────┘
  Log cycle number, timestamp
  Discover active BTC 15-minute markets
  Extract condition_id and token_ids

┌─────────────────────────────────────────────────────────────────────┐
│              T+1s: DATA COLLECTION (PARALLEL)                       │
└─────────────────────────────────────────────────────────────────────┘

  Thread 1: BTCPriceService
    ├─ Check WebSocket connection (~1ms)
    ├─ Get cached price if < 5s old (~1ms)
    └─ Return: BTCPriceData (price, source, volume)
    Total: ~2ms (cached) or ~500ms (fresh)

  Thread 2: SocialSentimentService
    ├─ Fear/Greed API (~200ms)
    ├─ CoinGecko Trending (~300ms)
    └─ CoinGecko Votes (~300ms)
    Total: ~500ms (parallel)

  Thread 3: MarketMicrostructureService ← BOTTLENECK
    ├─ Connect to Polymarket CLOB WebSocket (~500ms)
    ├─ Send subscription message (~100ms)
    ├─ Collect trades for 120 seconds (2 minutes)
    │   └─ Receive trade messages continuously
    ├─ Disconnect (~100ms)
    └─ Calculate scores (~10s)
        ├─ Momentum score (YES price change)
        ├─ Volume flow score (YES vs NO volume)
        └─ Whale activity score (large trades)
    Total: ~130 seconds

┌─────────────────────────────────────────────────────────────────────┐
│           T+130s: POST-COLLECTION PROCESSING                        │
└─────────────────────────────────────────────────────────────────────┘

  BTC Momentum Calculation
    ├─ Fetch 5-minute price history (~1s)
    ├─ Compare current to 5min ago (~1ms)
    └─ Classify direction (UP/DOWN/FLAT)
    Total: ~1s

  Technical Analysis
    ├─ Fetch 60-minute price history (~2s)
    ├─ Calculate RSI, MACD, EMA (~100ms with pandas)
    ├─ Calculate volume and velocity (~10ms)
    └─ Classify trend
    Total: ~2s

  Signal Aggregation
    ├─ Combine market (60%) + social (40%) (~1ms)
    ├─ Calculate agreement multiplier (~1ms)
    └─ Adjust confidence (~1ms)
    Total: ~3ms

┌─────────────────────────────────────────────────────────────────────┐
│               T+133s: AI DECISION MAKING                            │
└─────────────────────────────────────────────────────────────────────┘

  AIDecisionService
    ├─ Build prompt with all data (~1ms)
    │   ├─ Price-to-beat context
    │   ├─ Validation rules
    │   ├─ Market timing
    │   ├─ BTC momentum
    │   ├─ Technical indicators
    │   ├─ Social sentiment
    │   ├─ Market microstructure
    │   ├─ Aggregated signal
    │   └─ Risk parameters
    │
    ├─ Call OpenAI GPT-5-Nano API
    │   └─ Reasoning effort: medium (~15s)
    │       ├─ Internal reasoning tokens (~2-4k tokens)
    │       └─ JSON output (~200 tokens)
    │
    └─ Parse JSON response (~1ms)
        └─ Return: TradingDecision
    Total: ~15-20s

┌─────────────────────────────────────────────────────────────────────┐
│             T+153s: RISK VALIDATION                                 │
└─────────────────────────────────────────────────────────────────────┘

  RiskManager
    ├─ Check confidence threshold (<1ms)
    ├─ Validate position size (<1ms)
    ├─ Check portfolio limits (<1ms)
    ├─ Verify daily trade count (<1ms)
    └─ Validate stop-loss (<1ms)
    Total: <1ms

  Result: ValidationResult (approved/rejected/adjusted)

┌─────────────────────────────────────────────────────────────────────┐
│             T+153s: TRADE EXECUTION (IF APPROVED)                   │
└─────────────────────────────────────────────────────────────────────┘

  Mode: "trading"
    ├─ Create OrderRequest
    ├─ Submit to Polymarket CLOB API (~500ms)
    ├─ Log trade execution
    └─ Track position for stop-loss
    Total: ~500ms

  Mode: "read_only" (DRY RUN)
    └─ Log "would execute trade"
    Total: ~1ms

┌─────────────────────────────────────────────────────────────────────┐
│               T+154s: STOP-LOSS CHECK                               │
└─────────────────────────────────────────────────────────────────────┘

  Review open positions
  Check current market prices
  Close positions if loss > threshold
  (Currently stub implementation)
  Total: <100ms

┌─────────────────────────────────────────────────────────────────────┐
│                T+155s: CYCLE COMPLETE                               │
└─────────────────────────────────────────────────────────────────────┘

  Log cycle completion
  Update trade statistics
  Wait 25 seconds for next cycle

┌─────────────────────────────────────────────────────────────────────┐
│               T+180s: NEXT CYCLE BEGINS                             │
└─────────────────────────────────────────────────────────────────────┘
```

### Timing Summary

| Phase | Duration | % of Cycle |
|-------|----------|-----------|
| Market Data Collection | 130s | 72% |
| Signal Processing | 3s | 2% |
| AI Decision | 15-20s | 11% |
| Risk Validation | <1ms | 0% |
| Trade Execution | <1s | <1% |
| Stop-Loss Check | <1s | <1% |
| **Total Active** | **~155s** | **86%** |
| Idle Time | 25s | 14% |
| **Total Cycle** | **180s** | **100%** |

### Bottleneck Analysis

**Primary Bottleneck:** Market microstructure collection (130s / 72%)

**Why 2 Minutes?**
- Need sufficient trade data for reliable scores
- Too short → Low sample size, high variance
- Too long → Increased lag risk
- 2 minutes = sweet spot for BTC 15-min markets

**Potential Optimizations:**
1. Reduce collection to 90s (would decrease confidence)
2. Use 30s snapshots instead of full collection (less accurate)
3. Cache recent collection and update incrementally (complex)

**Current Decision:** Keep 2-minute collection for data quality

---

## Example Trading Scenarios

### Scenario 1: Perfect Alignment - TRADE YES

**Market Setup:**
- Market: "BTC price in 15 minutes"
- Outcomes: ["Up", "Down"]
- Current odds: Up 0.48, Down 0.52
- Time remaining: 7m 15s (mid-market)

**Data Collected:**

**BTC Price:**
- Price-to-beat (start): $67,500.00
- Current price: $67,800.00
- Difference: +$300 (+0.44%)
- Direction: **UP** ✓

**Market Microstructure (2-min CLOB):**
- Score: **+0.47** (STRONG_BULLISH)
- Confidence: 1.00
- Momentum: +0.80 (YES price rose 8%)
- Volume Flow: +0.60 (more YES buying)
- Whale Activity: +0.40 (4 YES whales, 1 NO whale)
- Whale Count: 5 total
- Trades Collected: 142

**Social Sentiment:**
- Score: **+0.30** (WEAK_BULLISH)
- Confidence: 1.00
- Fear/Greed: 65 (Greed)
- Trending: Yes (+0.5 bonus)
- Community Votes: 60% up, 40% down
- Sources: All 3 available

**BTC 5-Min Momentum:**
- 5 minutes ago: $67,725.00
- Current: $67,800.00
- Change: **+0.11%** (UP)
- Direction: **UP** ✓

**Technical Indicators (60-min):**
- RSI: 58.2 (neutral-bullish)
- MACD: +45.3 (Signal: +42.1, Histogram: +3.2)
- EMA: Short $67,520 > Long $67,450 (bullish)
- Trend: **BULLISH**
- Volume Change: +23.5% (above average)
- Price Velocity: +$2.50/min (upward)

**Signal Aggregation:**
```
final_score = (0.47 * 0.6) + (0.30 * 0.4) = 0.282 + 0.120 = +0.40

base_confidence = (1.00 + 1.00) / 2 = 1.00

agreement_multiplier = calculate_agreement(+0.47, +0.30)
  → both positive, alignment = 1 - abs(0.47-0.30)/2 = 1 - 0.085 = 0.915
  → multiplier = 1.0 + (0.915 * 0.5) = 1.458x

final_confidence = min(1.00 * 1.458, 1.0) = 1.00 (capped)

signal_type = "STRONG_BULLISH"
```

**AI Analysis:**
```
Validation Check:
✓ Price-to-beat: UP +0.44%
✓ Market signals: BULLISH (+0.47)
✓ BTC momentum: UP +0.11%
✓ No contradictions detected

Signal Strength:
✓ Market microstructure: Strong (+0.47, conf 1.00)
✓ Social sentiment: Moderate (+0.30, conf 1.00)
✓ Agreement: High (1.46x multiplier)
✓ Final score: +0.40 (moderate-strong bullish)

Technical Confirmation:
✓ All indicators bullish
✓ RSI neutral (not overbought)
✓ MACD positive and growing
✓ EMA alignment confirms trend

Market Timing:
✓ Mid-market phase (7m 15s remaining)
✓ Sufficient time for move to develop

Confidence Adjustment:
+0.05 - All signals align, strong whale support, no red flags

Final Decision: BUY "Up" token (YES)
```

**AI Output:**
```json
{
  "action": "YES",
  "confidence": 0.95,
  "reasoning": "All signals align bullish. Market microstructure +0.47 (strong) with heavy YES buying and whale support. Social sentiment +0.30 confirms community bullish mood. Price-to-beat shows BTC UP +0.44% with momentum continuing UP +0.11%. Technical indicators confirm bullish trend. No contradictions. High conviction trade.",
  "confidence_adjustment": "+0.05",
  "position_size": 50.0,
  "stop_loss": 0.40
}
```

**Risk Validation:**
- Confidence 0.95 > threshold 0.70 ✓
- Position $50 < max $50 (5% of $1,000) ✓
- No existing exposure to this market ✓
- Trades today: 3 < max 20 ✓
- **APPROVED**

**Execution:**
```
Mode: read_only
Log: "Dry run - would execute trade"
  Market: btc-updown-15m-1770784200
  Action: YES (buy "Up" token)
  Amount: $50.00
  Expected odds: ~0.48
```

---

### Scenario 2: Lagging Indicator - HOLD (Contradiction)

**Market Setup:**
- Market: "BTC price in 15 minutes"
- Outcomes: ["Up", "Down"]
- Current odds: Up 0.35, Down 0.65
- Time remaining: 9m 45s (early-mid market)

**Data Collected:**

**BTC Price:**
- Price-to-beat (start): $67,500.00
- Current price: $67,700.00
- Difference: +$200 (+0.30%)
- Direction: **UP** ✓

**Market Microstructure (2-min CLOB):**
- Score: **-0.47** (STRONG_BEARISH)
- Confidence: 1.00
- Momentum: -0.70 (YES price dropped 7% during window)
- Volume Flow: -0.50 (heavy NO buying during window)
- Whale Activity: -0.20 (2 YES whales, 3 NO whales)
- Note: *Collection window captured past drop, not current rebound*

**Social Sentiment:**
- Score: **-0.10** (WEAK_NEUTRAL)
- Confidence: 1.00
- Fear/Greed: 45 (Fear)
- Trending: No
- Community Votes: 48% up, 52% down

**BTC 5-Min Momentum:**
- 5 minutes ago: $67,550.00
- Current: $67,700.00
- Change: **+0.22%** (UP) ← **CONTRADICTS MARKET SIGNALS**
- Direction: **UP** ✓

**Technical Indicators (60-min):**
- RSI: 52.1 (neutral)
- MACD: -5.2 (Signal: -3.1, Histogram: -2.1)
- EMA: Short $67,480 < Long $67,520 (slightly bearish)
- Trend: **BEARISH** (but weakening)
- Volume Change: +12.3%
- Price Velocity: +$1.20/min (currently upward)

**Signal Aggregation:**
```
final_score = (-0.47 * 0.6) + (-0.10 * 0.4) = -0.282 - 0.040 = -0.32 (BEARISH)

base_confidence = (1.00 + 1.00) / 2 = 1.00

agreement_multiplier = calculate_agreement(-0.47, -0.10)
  → both negative, alignment = 1 - abs(-0.47-(-0.10))/2 = 1 - 0.185 = 0.815
  → multiplier = 1.0 + (0.815 * 0.5) = 1.408x

final_confidence = 1.00 * 1.408 = 1.00 (capped)

signal_type = "STRONG_BEARISH"
```

**AI Analysis:**
```
⚠️ VALIDATION CHECK - CONTRADICTION DETECTED:

Price-to-beat: UP +0.30%
Market signals: BEARISH (-0.47)
BTC momentum: UP +0.22%

→ Market sentiment shows BEARISH based on 2-minute collection window
→ But BTC has moved UP +0.30% from market start
→ 5-minute momentum confirms UP +0.22%

Analysis:
- Market collected data during PAST BTC drop (~8-10 minutes ago)
- BTC has since rebounded (+0.30% from start)
- Market sentiment LAGS actual price movement
- Following this signal would mean betting DOWN when BTC is going UP

Market Timing Context:
- 9m 45s remaining (still early, could reverse again)
- But current signals are 2+ minutes old

Technical Indicators:
- Mixed: Trend says bearish, but weakening
- Price velocity currently positive (+$1.20/min)
- Recent momentum contradicts longer trend

Confidence:
- Aggregated confidence: 1.00 (very high)
- But signals are CONTRADICTORY to actual price movement
- Cannot trust lagging signals

Decision: HOLD (override high confidence)
Reason: Signals lag reality - do NOT bet against actual BTC direction
```

**AI Output:**
```json
{
  "action": "HOLD",
  "confidence": 0.00,
  "reasoning": "Aggregated signals show STRONG_BEARISH (-0.32 score, 1.00 confidence) based on 2-minute market collection. However, price-to-beat shows BTC UP +0.30% from start, and 5-minute momentum confirms UP +0.22%. This is a clear contradiction - market sentiment reflects past drop but BTC has since rebounded. Signals are lagging actual movement. HOLD until alignment.",
  "confidence_adjustment": "0.0",
  "position_size": 0,
  "stop_loss": 0.40
}
```

**Risk Validation:**
- Action: HOLD
- No validation needed (not trading)

**Execution:**
```
Mode: read_only
Decision: HOLD
Reason: Contradiction between signals and actual BTC movement
```

**This is the key lagging indicator protection in action!**

---

### Scenario 3: End-of-Market Boost - TRADE NO

**Market Setup:**
- Market: "BTC price in 15 minutes"
- Outcomes: ["Up", "Down"]
- Current odds: Up 0.45, Down 0.55
- Time remaining: **2m 30s** (**END PHASE**)

**Data Collected:**

**BTC Price:**
- Price-to-beat (start): $67,500.00
- Current price: $67,400.00
- Difference: -$100 (-0.15%)
- Direction: **DOWN** ✓

**Market Microstructure (2-min CLOB):**
- Score: **-0.55** (STRONG_BEARISH)
- Confidence: 1.00
- Momentum: -0.85 (YES price dropped 8.5%)
- Volume Flow: -0.60 (heavy NO buying)
- Whale Activity: -0.20 (1 YES whale, 4 NO whales)
- Trades: 156

**Social Sentiment:**
- Score: **-0.20** (WEAK_BEARISH)
- Confidence: 1.00
- Fear/Greed: 42 (Fear)
- Trending: No
- Community Votes: 45% up, 55% down

**BTC 5-Min Momentum:**
- 5 minutes ago: $67,480.00
- Current: $67,400.00
- Change: **-0.17%** (DOWN)
- Direction: **DOWN** ✓

**Technical Indicators (60-min):**
- RSI: 43.8 (slightly oversold)
- MACD: -15.3 (Signal: -12.1, Histogram: -3.2)
- EMA: Short $67,420 < Long $67,510 (bearish)
- Trend: **BEARISH**
- Volume Change: +35.2% (high activity)
- Price Velocity: -$3.10/min (downward)

**Signal Aggregation:**
```
final_score = (-0.55 * 0.6) + (-0.20 * 0.4) = -0.330 - 0.080 = -0.41

agreement_multiplier = calculate_agreement(-0.55, -0.20)
  → both negative, alignment = 1 - abs(-0.55-(-0.20))/2 = 1 - 0.175 = 0.825
  → multiplier = 1.0 + (0.825 * 0.5) = 1.413x

final_confidence = 1.00 * 1.413 = 1.00 (capped)

signal_type = "STRONG_BEARISH"
```

**AI Analysis:**
```
Validation Check:
✓ Price-to-beat: DOWN -0.15%
✓ Market signals: BEARISH (-0.55)
✓ BTC momentum: DOWN -0.17%
✓ All signals ALIGN - no contradictions

Signal Strength:
✓ Market microstructure: Very strong (-0.55, conf 1.00)
✓ Social sentiment: Weak bearish (-0.20, conf 1.00)
✓ Agreement: High (1.41x multiplier)
✓ Final score: -0.41 (strong bearish)

Technical Confirmation:
✓ All indicators bearish
✓ RSI slightly oversold (potential support but not extreme)
✓ MACD strongly negative
✓ EMA alignment confirms downtrend
✓ High volume validates move

Market Timing - CRITICAL:
⚠️ Only 2m 30s remaining (END PHASE)
✓ Trend is ESTABLISHED with limited time for reversal
✓ BTC would need to gain +$100 (+0.15%) in 2.5 minutes to reverse
✓ Recent velocity is -$3.10/min (would take ~5+ minutes to reverse)
✓ End-of-market timing makes current trend HIGHLY PREDICTIVE

Confidence Adjustment:
+0.15 (max boost):
- All signals strongly align
- End-of-market phase
- Insufficient time for reversal given current velocity
- Whale conviction (4 NO whales vs 1 YES)

Final Decision: BUY "Down" token (NO)
```

**AI Output:**
```json
{
  "action": "NO",
  "confidence": 0.95,
  "reasoning": "Strong bearish signals align with actual BTC movement. Market microstructure -0.55 (very strong) with heavy selling pressure and 4 NO whales. Price-to-beat DOWN -0.15%, 5-min momentum DOWN -0.17%. Technical indicators confirm bearish trend. CRITICAL: Only 2m 30s remaining - trend is established with insufficient time for reversal at current velocity (-$3.10/min). End-of-market timing justifies maximum confidence boost.",
  "confidence_adjustment": "+0.15",
  "position_size": 50.0,
  "stop_loss": 0.40
}
```

**Risk Validation:**
- Confidence 0.95 > threshold 0.70 ✓
- Position $50 < max $50 ✓
- Trades today: 5 < max 20 ✓
- **APPROVED**

**Execution:**
```
Mode: read_only
Log: "Dry run - would execute trade"
  Market: btc-updown-15m-1770784200
  Action: NO (buy "Down" token)
  Amount: $50.00
  Expected odds: ~0.55

Reasoning: End-of-market conviction trade
```

---

### Scenario 4: Conflicting Signals - HOLD (Low Confidence)

**Market Setup:**
- Market: "BTC price in 15 minutes"
- Outcomes: ["Up", "Down"]
- Current odds: Up 0.50, Down 0.50
- Time remaining: 10m 30s (mid-market)

**Data Collected:**

**BTC Price:**
- Price-to-beat (start): $67,500.00
- Current price: $67,510.00
- Difference: +$10 (+0.01%)
- Direction: **FLAT**

**Market Microstructure (2-min CLOB):**
- Score: **+0.25** (WEAK_BULLISH)
- Confidence: 0.80 (lower than usual)
- Momentum: +0.35 (YES price rose 3.5%)
- Volume Flow: +0.30 (slightly more YES buying)
- Whale Activity: +0.10 (3 YES whales, 2 NO whales)
- Trades: 38 (low liquidity warning)

**Social Sentiment:**
- Score: **-0.30** (WEAK_BEARISH)
- Confidence: 1.00
- Fear/Greed: 38 (Fear)
- Trending: No
- Community Votes: 42% up, 58% down

**BTC 5-Min Momentum:**
- 5 minutes ago: $67,505.00
- Current: $67,510.00
- Change: **+0.01%** (FLAT)
- Direction: **FLAT**

**Technical Indicators (60-min):**
- RSI: 50.2 (neutral)
- MACD: +1.2 (Signal: +0.8, Histogram: +0.4)
- EMA: Short $67,502 ≈ Long $67,498 (neutral)
- Trend: **NEUTRAL**
- Volume Change: -8.3% (below average)
- Price Velocity: +$0.10/min (minimal movement)

**Signal Aggregation:**
```
final_score = (+0.25 * 0.6) + (-0.30 * 0.4) = +0.150 - 0.120 = +0.03 (NEUTRAL)

base_confidence = (0.80 + 1.00) / 2 = 0.90

agreement_multiplier = calculate_agreement(+0.25, -0.30)
  → opposite directions (conflict)
  → conflict = abs(+0.25 - (-0.30)) / 2 = 0.55 / 2 = 0.275
  → multiplier = 1.0 - (0.275 * 0.5) = 0.863x (penalty)

final_confidence = 0.90 * 0.863 = 0.78

signal_type = "CONFLICTED_NEUTRAL"
```

**AI Analysis:**
```
Validation Check:
? Price-to-beat: FLAT +0.01% (no clear direction)
? Market signals: Weak bullish (+0.25)
? Social signals: Weak bearish (-0.30)
? BTC momentum: FLAT +0.01%
✓ No strong contradiction, but signals CONFLICT

Signal Analysis:
⚠️ Market says BULLISH (+0.25, moderate strength)
⚠️ Social says BEARISH (-0.30, similar strength)
⚠️ Signals oppose each other
⚠️ Agreement penalty: 0.86x (reduced confidence)

Price Movement:
⚠️ Price-to-beat: Nearly unchanged (+0.01%)
⚠️ 5-min momentum: Nearly unchanged (+0.01%)
⚠️ No clear directional move

Technical Indicators:
? All neutral (RSI 50, MACD barely positive, EMA flat)
? Low volume (-8.3% suggests low conviction)
? Price velocity minimal (+$0.10/min)

Market Timing:
✓ Mid-market (10m 30s remaining)
? Enough time for move to develop, but which direction?

Data Quality:
⚠️ Market confidence only 0.80 (low liquidity - 38 trades)
⚠️ Low trade count reduces reliability

Confidence Calculation:
Base: 0.78 (after agreement penalty)
Adjustment: -0.10 (low liquidity concern)
Final: 0.68

Threshold: 0.70
Result: 0.68 < 0.70 → BELOW THRESHOLD

Decision: HOLD (insufficient confidence)
```

**AI Output:**
```json
{
  "action": "HOLD",
  "confidence": 0.68,
  "reasoning": "Conflicting signals with no clear direction. Market microstructure shows weak bullish (+0.25) but social sentiment shows weak bearish (-0.30). Price-to-beat and momentum both flat (+0.01%). Technical indicators neutral across the board. Low trading volume (38 trades) reduces data quality. Agreement penalty brings confidence to 0.68, below 0.70 threshold. Wait for clearer signals.",
  "confidence_adjustment": "-0.10",
  "position_size": 0,
  "stop_loss": 0.40
}
```

**Risk Validation:**
- Action: HOLD
- No validation needed (not trading)

**Execution:**
```
Mode: read_only
Decision: HOLD
Reason: Confidence 0.68 < threshold 0.70
```

---

## Configuration Reference

### Environment Variables

```bash
# Trading Mode
POLYMARKET_MODE=read_only        # "read_only" or "trading"

# API Keys
POLYMARKET_API_KEY=your_key      # Polymarket CLOB API key
POLYMARKET_PRIVATE_KEY=0x...     # Ethereum private key for signing
OPENAI_API_KEY=sk-...            # OpenAI API key

# OpenAI Model Settings
OPENAI_MODEL=gpt-o1-mini         # GPT-5-Nano model
OPENAI_REASONING_EFFORT=medium   # minimal/low/medium/high

# Risk Parameters
BOT_CONFIDENCE_THRESHOLD=0.70    # Minimum confidence (0.0-1.0)
BOT_MAX_POSITION_PERCENT=0.05    # Max % of portfolio per trade
BOT_MAX_DAILY_TRADES=20          # Daily trade limit

# Data Collection
BTC_PRICE_CACHE_SECONDS=5        # Price cache TTL
MARKET_COLLECTION_SECONDS=120    # CLOB collection duration

# Logging
LOG_LEVEL=INFO                   # DEBUG/INFO/WARNING/ERROR
LOG_JSON=false                   # JSON log format
BOT_LOG_DECISIONS=true           # Log all AI decisions
```

### Settings Dataclass

```python
@dataclass
class Settings:
    """Bot configuration settings."""

    # Trading mode
    mode: str = "read_only"  # "read_only" or "trading"

    # API credentials
    polymarket_api_key: str = ""
    polymarket_private_key: str = ""
    openai_api_key: str = ""

    # OpenAI configuration
    openai_model: str = "gpt-o1-mini"
    openai_reasoning_effort: str = "medium"  # minimal/low/medium/high

    # Risk management
    bot_confidence_threshold: float = 0.70    # Min confidence to trade
    bot_max_position_percent: float = 0.05    # 5% max per trade
    bot_max_market_exposure: float = 0.10     # 10% max per market
    bot_max_daily_trades: int = 20            # Daily trade limit
    bot_stop_loss_percent: float = 0.15       # 15% loss triggers exit

    # Data collection
    btc_price_cache_seconds: int = 5          # Price cache TTL
    market_collection_seconds: int = 120      # CLOB collection duration

    # Logging
    log_level: str = "INFO"
    log_json: bool = False
    bot_log_decisions: bool = True

    @classmethod
    def from_env(cls) -> "Settings":
        """Load settings from environment variables."""
        import os
        return cls(
            mode=os.getenv("POLYMARKET_MODE", "read_only"),
            polymarket_api_key=os.getenv("POLYMARKET_API_KEY", ""),
            polymarket_private_key=os.getenv("POLYMARKET_PRIVATE_KEY", ""),
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-o1-mini"),
            openai_reasoning_effort=os.getenv("OPENAI_REASONING_EFFORT", "medium"),
            bot_confidence_threshold=float(os.getenv("BOT_CONFIDENCE_THRESHOLD", "0.70")),
            bot_max_position_percent=float(os.getenv("BOT_MAX_POSITION_PERCENT", "0.05")),
            bot_max_market_exposure=float(os.getenv("BOT_MAX_MARKET_EXPOSURE", "0.10")),
            bot_max_daily_trades=int(os.getenv("BOT_MAX_DAILY_TRADES", "20")),
            bot_stop_loss_percent=float(os.getenv("BOT_STOP_LOSS_PERCENT", "0.15")),
            btc_price_cache_seconds=int(os.getenv("BTC_PRICE_CACHE_SECONDS", "5")),
            market_collection_seconds=int(os.getenv("MARKET_COLLECTION_SECONDS", "120")),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            log_json=os.getenv("LOG_JSON", "false").lower() == "true",
            bot_log_decisions=os.getenv("BOT_LOG_DECISIONS", "true").lower() == "true"
        )
```

### Tuning Recommendations

#### Conservative Trading
```bash
BOT_CONFIDENCE_THRESHOLD=0.80    # Higher threshold
BOT_MAX_POSITION_PERCENT=0.03    # Smaller positions (3%)
BOT_MAX_DAILY_TRADES=10          # Fewer trades
OPENAI_REASONING_EFFORT=high     # More thorough analysis
```

#### Aggressive Trading
```bash
BOT_CONFIDENCE_THRESHOLD=0.60    # Lower threshold
BOT_MAX_POSITION_PERCENT=0.08    # Larger positions (8%)
BOT_MAX_DAILY_TRADES=30          # More trades
OPENAI_REASONING_EFFORT=low      # Faster decisions
```

#### Balanced (Current Defaults)
```bash
BOT_CONFIDENCE_THRESHOLD=0.70    # Moderate threshold
BOT_MAX_POSITION_PERCENT=0.05    # Standard 5%
BOT_MAX_DAILY_TRADES=20          # Reasonable limit
OPENAI_REASONING_EFFORT=medium   # Balanced speed/quality
```

---

## Monitoring & Troubleshooting

### Key Log Patterns

**Successful Cycle:**
```
[info] Starting trading cycle          cycle=1
[info] Found markets                   count=1
[info] Data collected                  btc_price=$67,536.91 market_score=+0.47 social_score=-0.10
[info] Signals aggregated              final_score=-0.32 final_conf=1.00 signal=STRONG_BEARISH
[info] AI Decision                     action=HOLD confidence=0.00 reasoning="..."
[info] Cycle completed                 cycle=1
```

**Market Discovery Issues:**
```
[warning] Market discovery failed      error="No active markets found"
[info] No BTC markets found, skipping cycle
```

**Data Collection Failures:**
```
[error] WebSocket collection failed    error="Connection refused"
[warning] Using market signals only (social unavailable)
```

**Lagging Indicator Detection:**
```
[info] Price comparison               current=$67,700 price_to_beat=$67,500 difference=$+200
[info] BTC actual movement            direction=UP change_pct=+0.22%
[info] AI Decision                    action=HOLD reasoning="Contradiction detected..."
```

### Common Issues

#### 1. WebSocket Connection Failures

**Symptom:**
```
[error] WebSocket collection failed
[warning] Using market signals only
```

**Causes:**
- Network connectivity issues
- Polymarket CLOB API rate limits
- Invalid token IDs

**Solution:**
- Check internet connection
- Verify token_ids are correct
- Reduce concurrent requests
- System continues with degraded data (graceful)

#### 2. OpenAI API Timeouts

**Symptom:**
```
[error] OpenAI timeout
[info] AI Decision    action=HOLD reason="OpenAI timeout"
```

**Causes:**
- API latency
- Reasoning effort too high
- Network issues

**Solution:**
```bash
# Reduce reasoning effort
OPENAI_REASONING_EFFORT=low

# Or increase timeout (in code):
timeout=45.0  # Was 30.0
```

#### 3. Low Confidence Decisions

**Symptom:**
```
[info] AI Decision    action=HOLD confidence=0.65
```

**Causes:**
- Conflicting signals
- Low data quality
- Unclear market direction

**Solution:**
- Wait for next cycle (signals may align)
- Check if market is choppy/sideways
- Consider lowering threshold (risky)

#### 4. Insufficient Price History

**Symptom:**
```
[warning] Technical analysis unavailable, using neutral defaults
[error] Failed to fetch price history
```

**Causes:**
- Binance API rate limit
- Network issues
- API maintenance

**Solution:**
- System uses neutral defaults (graceful)
- Check Binance status
- Add retry logic if frequent

### Health Check Metrics

**Monitor These Values:**

1. **Cycle Completion Rate**
   - Target: >95% cycles complete successfully
   - Alert: <90% completion rate

2. **Data Source Availability**
   - Target: All 3 social sources available >90%
   - Target: Market microstructure available >95%
   - Alert: Any source <80% available

3. **Confidence Distribution**
   - Healthy: Mix of 0.70-1.00 confidence levels
   - Unhealthy: All HOLD (0.0 confidence)
   - Unhealthy: All max confidence (1.0) without variance

4. **Contradiction Detection Rate**
   - Expected: 10-30% of cycles detect contradictions
   - Too High: >50% (market constantly lagging)
   - Too Low: <5% (validation rules may not be working)

5. **Trade Execution Rate** (if mode=trading)
   - Expected: 30-50% of cycles result in trades
   - Too High: >70% (possibly overtrading)
   - Too Low: <10% (threshold too high or constant HOLD)

### Performance Optimization

**If cycles are too slow:**

1. Reduce market collection duration:
```bash
MARKET_COLLECTION_SECONDS=90  # From 120
```

2. Lower reasoning effort:
```bash
OPENAI_REASONING_EFFORT=low  # From medium
```

3. Disable unused features:
```bash
BOT_LOG_DECISIONS=false
```

**If too many false signals:**

1. Increase confidence threshold:
```bash
BOT_CONFIDENCE_THRESHOLD=0.80  # From 0.70
```

2. Extend market collection for better data:
```bash
MARKET_COLLECTION_SECONDS=150  # From 120
```

---

## Version History

**Version 2.0** (2026-02-11) - Lagging Indicator Fixes
- Added BTC momentum calculation (5-min lookback)
- Added price-to-beat tracking and comparison
- Added AI validation rules for contradiction detection
- Reduced momentum weight: 40% → 20%
- Increased volume flow weight: 35% → 50%
- Increased whale weight: 25% → 30%
- Deprecated orderbook analysis (replaced with CLOB trades)

**Version 1.0** (2026-02-09) - Initial Release
- Multi-source signal aggregation
- GPT-5-Nano AI decision engine
- Market microstructure analysis
- Social sentiment tracking
- Risk management validation

---

## Glossary

**Aggregated Sentiment**: Combined signal from market microstructure (60%) and social sentiment (40%)

**Agreement Multiplier**: Confidence adjustment (0.5x to 1.5x) based on signal alignment

**BTC Momentum**: 5-minute BTC price change used to detect lagging indicators

**Confidence Threshold**: Minimum confidence (default 0.70) required to execute trade

**Lagging Indicator**: Signal that reflects past price action instead of current/future movement

**Market Microstructure**: Analysis of Polymarket CLOB trades (momentum, volume flow, whales)

**Price-to-Beat**: BTC price at market start, used as baseline for direction detection

**Signal Contradiction**: When market sentiment opposes actual BTC price direction

**Whale Trade**: Trade > $1,000 indicating high-conviction position

**YES/NO Tokens**: Prediction market outcomes - YES = Up, NO = Down (for BTC markets)

---

## Contact & Support

**Repository**: [Your repo URL]

**Documentation**: This file

**Issues**: [Your issues URL]

**License**: [Your license]

---

*End of Trading Logic Reference*
