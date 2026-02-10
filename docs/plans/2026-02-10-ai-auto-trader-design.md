# AI-Powered BTC Auto-Trading Agent Design

**Date:** 2026-02-10
**Author:** Claude Code
**Status:** Approved

## Overview

Build an autonomous trading agent for Polymarket's BTC 15-minute binary options markets that uses real-time BTC price data, technical indicators, multi-source sentiment analysis, and OpenAI GPT-4o-mini to make intelligent UP/DOWN trading decisions.

## Problem Statement

The current `fetch_markets.py` script does not include BTC prices. To build an effective auto-trading agent, we need:
1. Real-time BTC price streaming
2. BTC price at market start (the "price to beat")
3. Technical analysis optimized for 15-minute timeframes
4. Sentiment analysis from multiple sources
5. AI-powered decision making with detailed logging

## Architecture

### Components

```
polymarket/
├── client.py              # Existing Polymarket API client
├── auth.py                # Existing authentication
├── config.py              # Existing configuration
├── models.py              # Existing data models (extend with new models)
├── exceptions.py          # Existing exceptions (extend with new exceptions)
│
├── btc_stream.py          # NEW: Binance WebSocket client for BTC prices
├── market_tracker.py      # NEW: Tracks price_to_beat vs current price
│
├── analysis/              # NEW: Analysis package
│   ├── __init__.py
│   ├── indicators.py      # Technical indicators (RSI, MACD, MA, BB, Volume)
│   ├── sentiment.py       # Sentiment analysis (Tavily, Twitter, Reddit)
│   └── ai_analyst.py      # OpenAI GPT-4o-mini integration
│
└── utils/
    ├── logging.py         # Existing
    └── retry.py           # Existing

scripts/
├── fetch_markets.py       # Existing
├── place_order.py         # Existing
├── portfolio_status.py    # Existing
├── stream_prices.py       # NEW: Real-time BTC price streaming
└── auto_trade_btc.py      # NEW: Main auto-trading agent
```

### Data Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      AI-POWERED BTC AUTO TRADER                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  DATA COLLECTION:                                                        │
│  ├─ Binance WebSocket (wss://stream.binance.com:9443/ws/btcusdt@trade) │
│  │  └─→ Real-time BTC price every ~100ms                               │
│  ├─ Price History                                                        │
│  │  └─→ 15-min OHLCV candles for indicator calculation                 │
│  ├─ Technical Indicators                                                 │
│  │  └─→ RSI(14), MACD, SMA/EMA, Bollinger Bands, Volume Profile        │
│  └─ Sentiment Sources                                                    │
│     ├─ Tavily news search (crypto/BTC news)                             │
│     ├─ Twitter/X mentions (via Twitter MCP)                             │
│     ├─ Reddit crypto subs (via Reddit MCP)                              │
│     └─ Tavily deep research (comprehensive analysis)                    │
│                                                                          │
│  AI ANALYSIS (GPT-4o-mini):                                             │
│  ├─ Input:                                                               │
│  │  ├─ Current BTC price vs price_to_beat                               │
│  │  ├─ Technical indicator signals                                       │
│  │  ├─ Aggregated sentiment score                                       │
│  │  └─ Time until market close                                          │
│  ├─ Processing:                                                          │
│  │  └─ OpenAI API call with structured prompt                           │
│  └─ Output:                                                              │
│     ├─ Decision: UP / DOWN / HOLD                                       │
│     ├─ Confidence: 0-100%                                                │
│     ├─ Reasoning: Detailed explanation                                  │
│     └─ Risk assessment: Low/Medium/High                                 │
│                                                                          │
│  EXECUTION:                                                              │
│  ├─ If confidence > threshold → Place market order (FOK)                │
│  ├─ Log full AI reasoning to file + structured log                      │
│  └─ Track outcome for future learning                                   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Module Specifications

### 1. Binance WebSocket Client (`polymarket/btc_stream.py`)

```python
class BinanceWebSocketClient:
    """Streams real-time BTC/USDT prices from Binance"""

    def __init__(self, url: str = "wss://stream.binance.com:9443/ws/btcusdt@trade"):
        self.url = url
        self.ws: Optional[websocket.WebSocketApp] = None
        self.current_price: Optional[Decimal] = None
        self.callbacks: List[Callable[[Decimal, datetime], None]] = []
        self._running = False

    def connect(self) -> None:
        """Connect to Binance WebSocket"""

    def disconnect(self) -> None:
        """Disconnect from WebSocket"""

    def subscribe(self, callback: Callable[[Decimal, datetime], None]) -> None:
        """Subscribe to price updates"""

    def get_current_price(self) -> Optional[Decimal]:
        """Get the latest BTC price"""
```

### 2. Market Price Tracker (`polymarket/market_tracker.py`)

```python
class MarketPriceTracker:
    """Tracks BTC price at market start vs current price"""

    def __init__(self, btc_stream: BinanceWebSocketClient):
        self.btc_stream = btc_stream
        self.start_prices: Dict[int, Decimal] = {}  # epoch -> price

    def record_market_start_price(self, market_epoch: int) -> None:
        """Record BTC price at market open"""

    def get_start_price(self, market_epoch: int) -> Optional[Decimal]:
        """Get recorded start price for a market"""

    def get_current_price(self) -> Optional[Decimal]:
        """Get current BTC price from stream"""

    def should_bet_up(self, market_epoch: int) -> Optional[bool]:
        """Compare current vs start price. None if insufficient data."""
```

### 3. Technical Indicators (`polymarket/analysis/indicators.py`)

```python
@dataclass
class IndicatorSignal:
    rsi: Optional[Decimal] = None
    macd: Optional[Decimal] = None
    macd_signal: Optional[Decimal] = None
    macd_histogram: Optional[Decimal] = None
    bb_upper: Optional[Decimal] = None
    bb_middle: Optional[Decimal] = None
    bb_lower: Optional[Decimal] = None
    ma_trend: Optional[str] = None  # "bullish" | "bearish" | "neutral"
    volume_ratio: Optional[Decimal] = None

    preliminary_signal: Optional[str] = None  # "up" | "down" | "neutral"


class TechnicalIndicators:
    """Calculate technical indicators optimized for 15-minute BTC trading"""

    def calculate_rsi(self, prices: List[Decimal], period: int = 14) -> Decimal:
        """Relative Strength Index"""

    def calculate_macd(self, prices: List[Decimal],
                      fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple:
        """MACD indicator"""

    def calculate_bollinger_bands(self, prices: List[Decimal],
                                   period: int = 20, std_dev: int = 2) -> Tuple:
        """Bollinger Bands"""

    def calculate_ma_crossover(self, prices: List[Decimal],
                               fast: int = 9, slow: int = 21) -> Dict:
        """Moving average crossover"""

    def analyze_volume(self, trades: List) -> VolumeProfile:
        """Volume analysis"""

    def generate_signal(self, ohlcv_data: List[OHLCV]) -> IndicatorSignal:
        """Combine all indicators into preliminary signal"""
```

### 4. Sentiment Analyzer (`polymarket/analysis/sentiment.py`)

```python
@dataclass
class SentimentScore:
    source: str
    score: Decimal  # -100 to +100
    confidence: Decimal  # 0 to 1
    key_insights: List[str]
    timestamp: datetime


@dataclass
class AggregatedSentiment:
    overall_score: Decimal  # -100 to +100
    overall_confidence: Decimal
    by_source: Dict[str, SentimentScore]
    recommendation: str  # "bullish" | "bearish" | "neutral"


class SentimentAnalyzer:
    """Multi-source sentiment analysis"""

    def __init__(self, tavily_client, twitter_mcp=None, reddit_mcp=None):
        self.tavily = tavily_client
        self.twitter = twitter_mcp
        self.reddit = reddit_mcp

    async def analyze_tavily_news(self, query: str = "BTC bitcoin price news today") -> SentimentScore:
        """Analyze sentiment from recent news via Tavily"""

    async def analyze_twitter(self) -> SentimentScore:
        """Analyze sentiment from Twitter/X mentions"""

    async def analyze_reddit(self, subreddits: List[str] = None) -> SentimentScore:
        """Analyze sentiment from Reddit crypto discussions"""

    async def deep_research(self, query: str) -> SentimentScore:
        """Deep research via Tavily"""

    async def aggregate_sentiment(self) -> AggregatedSentiment:
        """Combine all sources with weighted scoring"""
```

### 5. AI Trading Analyst (`polymarket/analysis/ai_analyst.py`)

```python
@dataclass
class AIAnalysis:
    decision: str  # "up" | "down" | "hold"
    confidence: Decimal  # 0-100
    reasoning: str
    risk_level: str  # "low" | "medium" | "high"
    timestamp: datetime


class AITradingAnalyst:
    """OpenAI GPT-4o-mini for final trading decision"""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    async def analyze(self, context: TradingContext) -> AIAnalysis:
        """
        Analyze market data and generate trading decision.

        TradingContext includes:
        - current_btc_price: Decimal
        - price_to_beat: Decimal
        - price_diff_percent: Decimal
        - indicators: IndicatorSignal
        - sentiment: AggregatedSentiment
        - time_to_close: int (seconds)
        """

    def _build_prompt(self, context: TradingContext) -> str:
        """Build structured prompt for OpenAI"""

    def log_decision(self, analysis: AIAnalysis, context: TradingContext) -> None:
        """Store detailed logs for review and learning"""
```

### 6. Auto-Trading Script (`scripts/auto_trade_btc.py`)

```python
class BTCAutoTrader:
    """Autonomous trading agent for BTC 15-minute markets"""

    def __init__(self):
        self.polymarket = PolymarketClient()
        self.btc_stream = BinanceWebSocketClient()
        self.price_tracker = MarketPriceTracker(self.btc_stream)
        self.indicators = TechnicalIndicators()
        self.sentiment = SentimentAnalyzer(...)
        self.ai_analyst = AITradingAnalyst(...)

    async def run(self):
        """Main trading loop"""
        while True:
            # 1. Discover active BTC 15-minute markets
            markets = await self.polymarket.get_btc_markets()

            for market in markets:
                # 2. Record price at market start
                if market.is_new():
                    self.price_tracker.record_market_start_price(market.epoch)

                # 3. Calculate technical indicators
                indicators = self.indicators.generate_signal(market.ohlcv_data)

                # 4. Analyze sentiment
                sentiment = await self.sentiment.aggregate_sentiment()

                # 5. AI analysis
                context = TradingContext(
                    current_btc_price=self.price_tracker.get_current_price(),
                    price_to_beat=self.price_tracker.get_start_price(market.epoch),
                    indicators=indicators,
                    sentiment=sentiment,
                    time_to_close=market.seconds_remaining
                )

                analysis = await self.ai_analyst.analyze(context)

                # 6. Execute trade if confident
                if analysis.decision != "hold" and analysis.confidence >= MIN_CONFIDENCE:
                    await self.execute_trade(market, analysis)

                # 7. Log everything
                self.ai_analyst.log_decision(analysis, context)

            await asyncio.sleep(10)
```

## Configuration

### Environment Variables (.env)

```bash
# ============================================
# Polymarket Configuration (Existing)
# ============================================
POLYMARKET_MODE=trading
POLYMARKET_PRIVATE_KEY=0x...
POLYMARKET_API_KEY=...
POLYMARKET_API_SECRET=...
POLYMARKET_API_PASSPHRASE=...
POLYMARKET_SIGNATURE_TYPE=2

# ============================================
# OpenAI Configuration
# ============================================
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini

# ============================================
# Binance WebSocket
# ============================================
BINANCE_WS_URL=wss://stream.binance.com:9443/ws/btcusdt@trade

# ============================================
# Sentiment Analysis
# ============================================
SENTIMENT_ENABLED=true
SENTIMENT_SOURCES=tavily,twitter,reddit
TWITTER_BEARER_TOKEN=...
REDDIT_CLIENT_ID=...
REDDIT_CLIENT_SECRET=...
REDDIT_USER_AGENT=...

# ============================================
# Trading Strategy
# ============================================
MIN_CONFIDENCE_THRESHOLD=70
MAX_POSITION_SIZE=100
RISK_LEVEL=medium

# ============================================
# Logging
# ============================================
LOG_LEVEL=INFO
LOG_JSON=false
AI_DECISION_LOG_PATH=/root/polymarket-scripts/logs/ai_decisions.json
```

## Dependencies

Add to `requirements.txt`:

```
# Existing
py-clob-client>=0.34.0
pydantic>=2.0.0
requests>=2.31.0
typer>=0.9.0
rich>=13.0.0
structlog>=23.1.0
pytest>=7.4.0

# New
websocket-client>=1.0.0
openai>=1.0.0
ta-lib>=0.4.0  # Technical analysis library
pandas>=2.0.0   # Data manipulation
numpy>=1.24.0   # Numerical computing
aiohttp>=3.9.0  # Async HTTP for sentiment analysis
```

## Testing Strategy

### Unit Tests

- `tests/test_btc_stream.py` - Mock WebSocket server
- `tests/test_market_tracker.py` - Price tracking logic
- `tests/test_indicators.py` - Indicator calculations
- `tests/test_sentiment.py` - Sentiment aggregation (with mocks)
- `tests/test_ai_analyst.py` - AI prompt building and response parsing

### Integration Tests

- `tests/test_auto_trade_integration.py` - End-to-end flow with mocked APIs

## Success Criteria

1. **Real-time BTC Price**: Successfully stream and display BTC prices
2. **Price Tracking**: Accurately track price_to_beat vs current price
3. **Technical Analysis**: Correctly calculate all 5 indicators
4. **Sentiment Analysis**: Successfully aggregate sentiment from 3+ sources
5. **AI Decisions**: OpenAI returns valid trading decisions with reasoning
6. **Auto-Execution**: Correctly places orders based on AI decisions
7. **Logging**: All decisions logged with full context
8. **Safety**: Dry-run mode works, no accidental live trades

## Implementation Phases

### Phase 1: Foundation
1. Set up Binance WebSocket client
2. Create market price tracker
3. Add new models and exceptions
4. Update .env configuration

### Phase 2: Analysis
5. Implement technical indicators
6. Build sentiment analyzer
7. Create AI analyst with OpenAI

### Phase 3: Integration
8. Build auto-trading script
9. Add comprehensive logging
10. Write tests

### Phase 4: Testing & Deployment
11. Integration testing
12. Dry-run validation
13. Documentation
14. Production deployment

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| API rate limits | Implement backoff, caching |
| WebSocket disconnection | Auto-reconnect with exponential backoff |
| Hallucination in AI | Confidence threshold, human review logs |
| Bad trading decisions | Start with small amounts, dry-run mode |
| Latency in 15-min markets | Optimize for speed, async operations |

## Future Enhancements

- Machine learning model for predictions
- Backtesting framework
- Portfolio optimization
- Multi-market support beyond BTC
- Stop-loss and take-profit orders
