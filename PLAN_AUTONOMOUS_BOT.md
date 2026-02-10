# Implementation Plan: Autonomous Polymarket Trading Bot

**Project:** Autonomous trading bot for Polymarket BTC 15-minute markets
**Location:** `/root/polymarket-scripts/`
**Created:** 2026-02-10
**Status:** Ready for Implementation

---

## Executive Summary

This plan implements an autonomous trading bot that:
- Runs continuously with 3-minute trading cycles
- Fetches BTC price data from Binance (fallback: CoinGecko)
- Analyzes market sentiment using Tavily search
- Calculates technical indicators (RSI, MACD, EMA)
- Uses OpenAI to make trading decisions
- Applies risk management rules before executing trades
- Supports dry-run mode for testing

**Key Design Principles:**
- Graceful degradation (returns HOLD on failures)
- Minimal testing (focus on dry-run with real APIs)
- Subpackage approach (`polymarket/trading/`)
- Async throughout for performance
- Comprehensive logging for all decisions

---

## Phase 1: Dependencies & Configuration

### 1.1 Update requirements.txt

**File:** `/root/polymarket-scripts/requirements.txt`

**Action:** Add bot dependencies

```diff
 # Core dependencies
 py-clob-client>=0.34.0
 pydantic>=2.0.0
 requests>=2.31.0
 python-dotenv>=1.0.0

 # Logging and utilities
 structlog>=23.1.0
 colorama>=0.4.6
 colorlog>=6.7.0

 # CLI and output
 typer>=0.9.0
 rich>=13.0.0
 tabulate>=0.9.0

+# Bot dependencies
+ccxt>=4.0.0              # Crypto exchange APIs (Binance)
+openai>=1.0.0            # OpenAI API
+tavily-python>=0.3.0     # Tavily search
+pandas>=2.0.0            # Data analysis
+numpy>=1.24.0            # Numerical computing
+aiohttp>=3.9.0           # Async HTTP
+
 # Development
 pytest>=7.4.0
 pytest-asyncio>=0.21.0
```

**Verification:**
```bash
cd /root/polymarket-scripts
pip install -r requirements.txt
python -c "import ccxt, openai, tavily; print('Dependencies OK')"
```

---

### 1.2 Update polymarket/config.py

**File:** `/root/polymarket-scripts/polymarket/config.py`

**Action:** Add bot configuration fields to the `Settings` dataclass

**Location:** After line 104 (after `log_json` field)

```python
    # === OpenAI Configuration ===
    openai_api_key: str | None = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY")
    )
    openai_model: str = field(
        default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    )

    # === Tavily Configuration ===
    tavily_api_key: str | None = field(
        default_factory=lambda: os.getenv("TAVILY_API_KEY")
    )

    # === BTC Price Service ===
    btc_price_source: str = field(
        default_factory=lambda: os.getenv("BTC_PRICE_SOURCE", "binance")
    )
    btc_price_cache_seconds: int = field(
        default_factory=lambda: int(os.getenv("BTC_PRICE_CACHE_SECONDS", "30"))
    )

    # === Trading Bot Configuration ===
    bot_interval_seconds: int = field(
        default_factory=lambda: int(os.getenv("BOT_INTERVAL_SECONDS", "180"))
    )
    bot_confidence_threshold: float = field(
        default_factory=lambda: float(os.getenv("BOT_CONFIDENCE_THRESHOLD", "0.75"))
    )
    bot_max_position_percent: float = field(
        default_factory=lambda: float(os.getenv("BOT_MAX_POSITION_PERCENT", "0.10"))
    )
    bot_max_exposure_percent: float = field(
        default_factory=lambda: float(os.getenv("BOT_MAX_EXPOSURE_PERCENT", "0.50"))
    )

    # === Stop-Loss Configuration ===
    stop_loss_odds_threshold: float = field(
        default_factory=lambda: float(os.getenv("STOP_LOSS_ODDS_THRESHOLD", "0.40"))
    )
    stop_loss_force_exit_minutes: int = field(
        default_factory=lambda: int(os.getenv("STOP_LOSS_FORCE_EXIT_MINUTES", "5"))
    )

    # === Bot Logging ===
    bot_log_decisions: bool = field(
        default_factory=lambda: os.getenv("BOT_LOG_DECISIONS", "true").lower() == "true"
    )
    bot_log_file: str = field(
        default_factory=lambda: os.getenv("BOT_LOG_FILE", "logs/auto_trade.log")
    )
```

**Verification:**
```bash
python -c "from polymarket.config import Settings; s = Settings(); print(f'Bot interval: {s.bot_interval_seconds}s')"
```

---

### 1.3 Update polymarket/models.py

**File:** `/root/polymarket-scripts/polymarket/models.py`

**Action:** Add trading bot data models

**Location:** After the `BalanceInfo` class (after line 161)

```python
# === BTC Price Models ===

from decimal import Decimal

@dataclass
class BTCPriceData:
    """Current BTC price data."""
    price: Decimal
    timestamp: datetime
    source: str
    volume_24h: Decimal


@dataclass
class PricePoint:
    """Historical price point for technical analysis."""
    price: Decimal
    volume: Decimal
    timestamp: datetime


@dataclass
class PriceChange:
    """Price change over a time window."""
    current_price: Decimal
    change_percent: float
    change_amount: Decimal
    velocity: Decimal  # $/minute


# === Sentiment Models ===

@dataclass
class SentimentAnalysis:
    """Market sentiment analysis from Tavily."""
    score: float           # -1.0 (bearish) to +1.0 (bullish)
    confidence: float      # 0.0 to 1.0
    key_factors: list[str]
    sources_analyzed: int
    timestamp: datetime


# === Technical Analysis Models ===

@dataclass
class TechnicalIndicators:
    """Technical analysis indicators."""
    rsi: float
    macd_value: float
    macd_signal: float
    macd_histogram: float
    ema_short: float
    ema_long: float
    sma_50: float
    volume_change: float
    price_velocity: float
    trend: Literal["BULLISH", "BEARISH", "NEUTRAL"]


# === Trading Decision Models ===

@dataclass
class TradingDecision:
    """AI-generated trading decision."""
    action: Literal["YES", "NO", "HOLD"]
    confidence: float
    reasoning: str
    token_id: str
    position_size: Decimal
    stop_loss_threshold: float


@dataclass
class ValidationResult:
    """Risk validation result."""
    approved: bool
    reason: str
    adjusted_position: Decimal | None
```

**Required imports to add at top:**
```python
from dataclasses import dataclass
from decimal import Decimal
```

**Verification:**
```bash
python -c "from polymarket.models import BTCPriceData, TradingDecision; print('Models OK')"
```

---

## Phase 2: Trading Subpackage Setup

### 2.1 Create directory structure

**Actions:**
```bash
cd /root/polymarket-scripts
mkdir -p polymarket/trading
mkdir -p logs
touch polymarket/trading/__init__.py
```

**File:** `/root/polymarket-scripts/polymarket/trading/__init__.py`

**Content:**
```python
"""
Trading bot subpackage for Polymarket.

This package contains all trading bot components:
- btc_price: BTC price data service
- sentiment: Market sentiment analysis
- technical: Technical indicators
- ai_decision: OpenAI decision engine
- risk: Risk management
"""

from polymarket.trading.btc_price import BTCPriceService
from polymarket.trading.sentiment import SentimentService
from polymarket.trading.technical import TechnicalAnalysis
from polymarket.trading.ai_decision import AIDecisionService
from polymarket.trading.risk import RiskManager

__all__ = [
    "BTCPriceService",
    "SentimentService",
    "TechnicalAnalysis",
    "AIDecisionService",
    "RiskManager",
]
```

**Verification:**
```bash
python -c "import polymarket.trading; print('Package created')"
```

---

## Phase 3: BTC Price Service

### 3.1 Create btc_price.py

**File:** `/root/polymarket-scripts/polymarket/trading/btc_price.py`

**Full implementation:** See brainstorming Section 3

**Key features:**
- Async fetching from Binance (primary) and CoinGecko (fallback)
- 30-second cache to avoid rate limits
- Historical data for technical analysis (60-minute window)
- Price change calculations

**Code structure:**
```python
class BTCPriceService:
    def __init__(self, settings: Settings)
    async def get_current_price(self) -> BTCPriceData
    async def _fetch_binance(self) -> BTCPriceData
    async def _fetch_coingecko(self) -> BTCPriceData
    async def get_price_history(self, minutes: int = 60) -> list[PricePoint]
    async def get_price_change(self, window_minutes: int = 5) -> PriceChange
    async def close(self)
```

**Testing approach:**
```bash
# Test price fetching
python -c "
import asyncio
from polymarket.config import Settings
from polymarket.trading.btc_price import BTCPriceService

async def test():
    settings = Settings()
    service = BTCPriceService(settings)

    # Test current price
    price = await service.get_current_price()
    print(f'BTC: \${price.price:,.2f} from {price.source}')

    # Test history
    history = await service.get_price_history(minutes=10)
    print(f'History: {len(history)} points')

    # Test price change
    change = await service.get_price_change(window_minutes=5)
    print(f'5-min change: {change.change_percent:+.2f}%')

    await service.close()

asyncio.run(test())
"
```

---

## Phase 4: Sentiment Service

### 4.1 Create sentiment.py

**File:** `/root/polymarket-scripts/polymarket/trading/sentiment.py`

**Full implementation:** See brainstorming Section 4

**Key features:**
- Tavily search for recent BTC news (24-hour window)
- Keyword-based sentiment scoring (bullish/bearish)
- Domain filtering (reputable crypto news sources)
- Confidence based on source count

**Code structure:**
```python
class SentimentService:
    BULLISH_KEYWORDS = [...]
    BEARISH_KEYWORDS = [...]

    def __init__(self, settings: Settings)
    async def get_btc_sentiment(self) -> SentimentAnalysis
    def _analyze_results(self, results: dict) -> SentimentAnalysis
```

**Testing approach:**
```bash
# Test sentiment analysis
python -c "
import asyncio
from polymarket.config import Settings
from polymarket.trading.sentiment import SentimentService

async def test():
    settings = Settings()
    service = SentimentService(settings)

    sentiment = await service.get_btc_sentiment()
    print(f'Sentiment: {sentiment.score:+.2f}')
    print(f'Confidence: {sentiment.confidence:.2f}')
    print(f'Sources: {sentiment.sources_analyzed}')
    print(f'Factors: {sentiment.key_factors[:3]}')

asyncio.run(test())
"
```

**Environment requirement:**
```bash
export TAVILY_API_KEY="tvly-..."
```

---

## Phase 5: Technical Analysis

### 5.1 Create technical.py

**File:** `/root/polymarket-scripts/polymarket/trading/technical.py`

**Full implementation:** See brainstorming Section 5

**Key features:**
- Dual mode: pandas/numpy (fast) or manual (fallback)
- Standard indicators: RSI(14), MACD(12,26,9), EMA(9,21)
- Graceful degradation for insufficient data
- Trend determination (BULLISH/BEARISH/NEUTRAL)

**Code structure:**
```python
class TechnicalAnalysis:
    @staticmethod
    def calculate_indicators(price_history: list[PricePoint]) -> TechnicalIndicators

    @staticmethod
    def _calculate_with_pandas(price_history: list[PricePoint]) -> TechnicalIndicators

    @staticmethod
    def _calculate_manual(price_history: list[PricePoint]) -> TechnicalIndicators

    @staticmethod
    def _calculate_rsi(prices: pd.Series, period: int) -> float
```

**Testing approach:**
```bash
# Test technical analysis with real data
python -c "
import asyncio
from polymarket.config import Settings
from polymarket.trading.btc_price import BTCPriceService
from polymarket.trading.technical import TechnicalAnalysis

async def test():
    settings = Settings()
    price_service = BTCPriceService(settings)

    # Get price history
    history = await price_service.get_price_history(minutes=60)
    print(f'Analyzing {len(history)} price points...')

    # Calculate indicators
    indicators = TechnicalAnalysis.calculate_indicators(history)

    print(f'RSI: {indicators.rsi:.1f}')
    print(f'MACD: {indicators.macd_value:.2f}')
    print(f'Trend: {indicators.trend}')
    print(f'EMA Short: {indicators.ema_short:,.2f}')
    print(f'EMA Long: {indicators.ema_long:,.2f}')

    await price_service.close()

asyncio.run(test())
"
```

---

## Phase 6: AI Decision Engine

### 6.1 Create ai_decision.py

**File:** `/root/polymarket-scripts/polymarket/trading/ai_decision.py`

**Full implementation:** See brainstorming Section 6

**Key features:**
- OpenAI API with JSON mode for structured output
- 10-second timeout to avoid blocking cycles
- Context-aware prompts (includes market type: UP/DOWN)
- Low temperature (0.3) for consistent decisions
- Graceful fallback to HOLD on errors

**Code structure:**
```python
class AIDecisionService:
    def __init__(self, settings: Settings)
    async def make_decision(
        btc_price: BTCPriceData,
        technical_indicators: TechnicalIndicators,
        sentiment: SentimentAnalysis,
        market_data: dict,
        portfolio_value: Decimal
    ) -> TradingDecision

    def _build_prompt(...) -> str
    def _parse_decision(data: dict, token_id: str) -> TradingDecision
    def _hold_decision(token_id: str, reason: str) -> TradingDecision
```

**Testing approach:**
```bash
# Test AI decision with real market data
python -c "
import asyncio
from decimal import Decimal
from polymarket.config import Settings
from polymarket.trading.btc_price import BTCPriceService
from polymarket.trading.sentiment import SentimentService
from polymarket.trading.technical import TechnicalAnalysis
from polymarket.trading.ai_decision import AIDecisionService

async def test():
    settings = Settings()

    # Initialize services
    btc_service = BTCPriceService(settings)
    sentiment_service = SentimentService(settings)
    ai_service = AIDecisionService(settings)

    # Gather data
    btc_price = await btc_service.get_current_price()
    sentiment = await sentiment_service.get_btc_sentiment()
    history = await btc_service.get_price_history(60)
    indicators = TechnicalAnalysis.calculate_indicators(history)

    # Mock market data
    market_data = {
        'token_id': 'test-token',
        'question': 'Will BTC go UP in 15 minutes?',
        'yes_price': 0.52,
        'no_price': 0.48
    }

    # Make decision
    decision = await ai_service.make_decision(
        btc_price, indicators, sentiment, market_data, Decimal('1000')
    )

    print(f'Action: {decision.action}')
    print(f'Confidence: {decision.confidence:.2f}')
    print(f'Reasoning: {decision.reasoning}')
    print(f'Position: \${decision.position_size}')

    await btc_service.close()

asyncio.run(test())
"
```

**Environment requirement:**
```bash
export OPENAI_API_KEY="sk-..."
```

---

## Phase 7: Risk Management

### 7.1 Create risk.py

**File:** `/root/polymarket-scripts/polymarket/trading/risk.py`

**Full implementation:** See brainstorming Section 7

**Key features:**
- Position sizing based on confidence (75-80%→50%, 80-90%→75%, 90%+→100%)
- Exposure tracking (blocks if total exposure > 50% portfolio)
- Duplicate market prevention
- Three stop-loss triggers:
  1. Odds for position < 0.40
  2. Counter-odds > 0.70
  3. Time-based exit (5 min before expiry)

**Code structure:**
```python
class RiskManager:
    def __init__(self, settings: Settings)

    async def validate_decision(
        decision: TradingDecision,
        portfolio_value: Decimal,
        market: dict,
        open_positions: list[dict] | None
    ) -> ValidationResult

    def _calculate_position_size(...) -> Decimal

    async def evaluate_stop_loss(
        open_positions: list[dict],
        current_markets: dict[str, dict]
    ) -> list[dict]
```

**Testing approach:**
```bash
# Test risk validation
python -c "
import asyncio
from decimal import Decimal
from polymarket.config import Settings
from polymarket.trading.risk import RiskManager
from polymarket.models import TradingDecision

async def test():
    settings = Settings()
    risk = RiskManager(settings)

    # Mock decision
    decision = TradingDecision(
        action='YES',
        confidence=0.85,
        reasoning='Test decision',
        token_id='test-token',
        position_size=Decimal('100'),
        stop_loss_threshold=0.40
    )

    # Mock market
    market = {'active': True, 'token_id': 'test-token'}

    # Validate
    result = await risk.validate_decision(
        decision,
        portfolio_value=Decimal('1000'),
        market=market,
        open_positions=[]
    )

    print(f'Approved: {result.approved}')
    print(f'Reason: {result.reason}')
    print(f'Adjusted position: \${result.adjusted_position}')

asyncio.run(test())
"
```

---

## Phase 8: Main Orchestration Script

### 8.1 Create auto_trade.py

**File:** `/root/polymarket-scripts/scripts/auto_trade.py`

**Full implementation:** See brainstorming Section 8

**Key features:**
- Continuous loop with configurable interval (default: 180s)
- Signal handling for graceful shutdown (SIGINT/SIGTERM)
- Parallel data fetching (BTC price + sentiment)
- Market filtering (only BTC 15-minute markets)
- Dry-run support via `POLYMARKET_MODE=read_only`
- Comprehensive structured logging

**Code structure:**
```python
class AutoTrader:
    def __init__(self, settings: Settings, interval: int)
    async def run_cycle(self) -> None
    async def _discover_markets(self) -> list[Market]
    async def _process_market(...) -> None
    async def _execute_trade(...) -> None
    async def _check_stop_loss(self) -> None
    async def _close_position(close: dict) -> None
    async def run(self) -> None
    async def run_once(self) -> None

@app.command()
def main(interval: int, once: bool) -> None
```

**Make executable:**
```bash
chmod +x /root/polymarket-scripts/scripts/auto_trade.py
```

---

## Phase 9: Testing & Validation

### 9.1 Dry-run testing

**Test command:**
```bash
cd /root/polymarket-scripts

# Single cycle test
POLYMARKET_MODE=read_only \
OPENAI_API_KEY="sk-..." \
TAVILY_API_KEY="tvly-..." \
python scripts/auto_trade.py --once
```

**Expected output:**
```
INFO     Starting trading cycle cycle=1
INFO     Found markets count=2
INFO     Data collected btc_price=$95,234.56 sentiment_score=+0.34
INFO     Technical indicators rsi=58.3 macd=123.45 trend=BULLISH
INFO     AI Decision action=YES confidence=0.82 reasoning="..."
INFO     Decision rejected by risk manager reason="Confidence 0.82..."
INFO     Cycle completed cycle=1
```

---

### 9.2 Integration testing

**Test full cycle with mocked APIs:**

Create `/root/polymarket-scripts/tests/test_auto_trade.py`:

```python
"""Integration test for auto_trade.py"""
import pytest
import asyncio
from decimal import Decimal
from polymarket.config import Settings
from polymarket.trading.btc_price import BTCPriceService
from polymarket.trading.sentiment import SentimentService
from polymarket.trading.technical import TechnicalAnalysis
from polymarket.trading.ai_decision import AIDecisionService
from polymarket.trading.risk import RiskManager


@pytest.mark.asyncio
async def test_full_trading_cycle():
    """Test complete trading cycle with real APIs."""
    settings = Settings()

    # Skip if missing API keys
    if not settings.openai_api_key or not settings.tavily_api_key:
        pytest.skip("Missing API keys")

    # Initialize services
    btc_service = BTCPriceService(settings)
    sentiment_service = SentimentService(settings)
    ai_service = AIDecisionService(settings)
    risk_manager = RiskManager(settings)

    try:
        # Step 1: Fetch data
        btc_price = await btc_service.get_current_price()
        assert btc_price.price > 0

        sentiment = await sentiment_service.get_btc_sentiment()
        assert -1.0 <= sentiment.score <= 1.0

        # Step 2: Technical analysis
        history = await btc_service.get_price_history(60)
        indicators = TechnicalAnalysis.calculate_indicators(history)
        assert 0 <= indicators.rsi <= 100

        # Step 3: AI decision
        market_data = {
            'token_id': 'test',
            'question': 'Will BTC go UP in 15 minutes?',
            'yes_price': 0.50,
            'no_price': 0.50,
            'active': True
        }

        decision = await ai_service.make_decision(
            btc_price, indicators, sentiment,
            market_data, Decimal('1000')
        )

        assert decision.action in ('YES', 'NO', 'HOLD')
        assert 0.0 <= decision.confidence <= 1.0

        # Step 4: Risk validation
        validation = await risk_manager.validate_decision(
            decision, Decimal('1000'), market_data, []
        )

        assert isinstance(validation.approved, bool)

        print(f"✓ Full cycle test passed")
        print(f"  BTC: ${btc_price.price:,.2f}")
        print(f"  Sentiment: {sentiment.score:+.2f}")
        print(f"  RSI: {indicators.rsi:.1f}")
        print(f"  Decision: {decision.action} ({decision.confidence:.2f})")

    finally:
        await btc_service.close()


if __name__ == "__main__":
    asyncio.run(test_full_trading_cycle())
```

**Run test:**
```bash
cd /root/polymarket-scripts
pytest tests/test_auto_trade.py -v -s
```

---

## Phase 10: Environment Setup & Documentation

### 10.1 Update .env.example

**File:** `/root/polymarket-scripts/.env.example`

**Add:**
```bash
# === Bot Configuration ===
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
TAVILY_API_KEY=tvly-...

# BTC Price Service
BTC_PRICE_SOURCE=binance
BTC_PRICE_CACHE_SECONDS=30

# Trading Parameters
BOT_INTERVAL_SECONDS=180
BOT_CONFIDENCE_THRESHOLD=0.75
BOT_MAX_POSITION_PERCENT=0.10
BOT_MAX_EXPOSURE_PERCENT=0.50

# Stop-Loss
STOP_LOSS_ODDS_THRESHOLD=0.40
STOP_LOSS_FORCE_EXIT_MINUTES=5

# Logging
BOT_LOG_DECISIONS=true
BOT_LOG_FILE=logs/auto_trade.log
```

---

### 10.2 Create README_BOT.md

**File:** `/root/polymarket-scripts/README_BOT.md`

```markdown
# Autonomous Polymarket Trading Bot

Autonomous trading bot for Polymarket BTC 15-minute up/down markets.

## Features

- **Data Sources:**
  - BTC price: Binance (primary) + CoinGecko (fallback)
  - Sentiment: Tavily search API
  - Technical analysis: RSI, MACD, EMA

- **Decision Engine:**
  - OpenAI GPT-4o-mini for trading decisions
  - Risk management with confidence-based position sizing
  - Stop-loss with three triggers

- **Trading Cycle:**
  - Runs every 3 minutes (configurable)
  - Discovers active BTC 15-min markets
  - Analyzes all data sources
  - Makes AI-powered decision
  - Validates against risk rules
  - Executes trade (if approved)

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

Create `.env`:

```bash
# Required
POLYMARKET_MODE=read_only  # Start in dry-run mode
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...

# Optional (defaults shown)
BOT_INTERVAL_SECONDS=180
BOT_CONFIDENCE_THRESHOLD=0.75
BOT_MAX_POSITION_PERCENT=0.10
```

### 3. Test in dry-run mode

```bash
python scripts/auto_trade.py --once
```

### 4. Run continuously (dry-run)

```bash
python scripts/auto_trade.py
```

### 5. Enable live trading

**⚠️ CAUTION: This will execute real trades with real money**

```bash
# Set trading mode + credentials
export POLYMARKET_MODE=trading
export POLYMARKET_PRIVATE_KEY=0x...
export POLYMARKET_API_KEY=...
export POLYMARKET_API_SECRET=...
export POLYMARKET_API_PASSPHRASE=...

# Run bot
python scripts/auto_trade.py
```

## Architecture

```
polymarket/trading/
├── btc_price.py      # BTC price from Binance/CoinGecko
├── sentiment.py      # Market sentiment via Tavily
├── technical.py      # RSI, MACD, EMA calculations
├── ai_decision.py    # OpenAI decision engine
└── risk.py           # Position sizing + stop-loss

scripts/
└── auto_trade.py     # Main orchestration loop
```

## Risk Management

### Position Sizing
- Confidence 75-80%: 50% of max position
- Confidence 80-90%: 75% of max position
- Confidence 90%+: 100% of max position
- Max single position: 10% of portfolio
- Max total exposure: 50% of portfolio

### Stop-Loss Triggers
1. **Odds threshold**: Exit if position odds < 0.40
2. **Counter-odds surge**: Exit if opposing odds > 0.70
3. **Time-based**: Force exit 5 minutes before market expiry

## Monitoring

Logs are written to:
- Console: Structured logs (colorized)
- File: `logs/auto_trade.log` (JSON format)

Key metrics logged:
- BTC price and source
- Sentiment score and confidence
- Technical indicators (RSI, MACD, trend)
- AI decision and reasoning
- Risk validation results
- Trade execution status

## Troubleshooting

### "No BTC markets found"
- Markets may not be active during off-hours
- Try wider search: `POLYMARKET_MODE=read_only python scripts/fetch_markets.py --search "bitcoin"`

### "OpenAI timeout"
- Check API key is valid
- Network issues may cause timeouts
- Bot will return HOLD on timeout (graceful degradation)

### "Tavily search failed"
- Check API key is valid
- Sentiment will default to neutral (score=0.0)

### Dependency errors
```bash
# Install all dependencies
pip install -r requirements.txt

# Test imports
python -c "import ccxt, openai, tavily, pandas, numpy; print('OK')"
```

## Safety Features

1. **Dry-run by default**: Must explicitly enable trading mode
2. **Graceful degradation**: Returns HOLD on any service failure
3. **Timeout protection**: 10-second OpenAI timeout
4. **Duplicate prevention**: Won't trade same market twice
5. **Exposure limits**: Hard caps on position sizes
6. **Comprehensive logging**: All decisions logged for audit

## Development

### Run single cycle test
```bash
python scripts/auto_trade.py --once
```

### Run integration test
```bash
pytest tests/test_auto_trade.py -v -s
```

### Adjust interval (for testing)
```bash
# Run every 60 seconds instead of 180
python scripts/auto_trade.py --interval 60
```

## License

Same as parent project.
```

---

## Success Criteria

Implementation is complete when:

1. ✅ All dependencies install without errors
2. ✅ `pytest tests/test_auto_trade.py` passes
3. ✅ Dry-run executes complete cycle without errors
4. ✅ All services return valid data:
   - BTC price fetched from Binance
   - Sentiment analysis returns score
   - Technical indicators calculated
   - AI decision generated
   - Risk validation completes
5. ✅ Logs show full decision trail
6. ✅ Bot runs continuously without crashes (10+ cycles)

---

## Rollback Plan

If implementation fails:

1. **Dependencies issue:**
   ```bash
   git checkout requirements.txt
   pip install -r requirements.txt
   ```

2. **Config/models issue:**
   ```bash
   git checkout polymarket/config.py polymarket/models.py
   ```

3. **Complete rollback:**
   ```bash
   rm -rf polymarket/trading
   rm scripts/auto_trade.py
   git checkout .
   ```

---

## Execution Order

1. Phase 1: Dependencies & Configuration (30 min)
2. Phase 2: Trading Subpackage Setup (10 min)
3. Phase 3: BTC Price Service (20 min)
4. Phase 4: Sentiment Service (20 min)
5. Phase 5: Technical Analysis (30 min)
6. Phase 6: AI Decision Engine (30 min)
7. Phase 7: Risk Management (30 min)
8. Phase 8: Main Orchestration Script (40 min)
9. Phase 9: Testing & Validation (30 min)
10. Phase 10: Environment Setup & Documentation (20 min)

**Total estimated time:** 4 hours

---

## Notes

- All times are estimates for focused implementation
- Testing is integrated throughout (not a separate phase at end)
- Dry-run mode is enforced by default for safety
- Each phase can be tested independently before moving to next
- Stop-loss functionality requires active monitoring of open positions

---

**Plan Status:** ✅ Ready for Implementation
**Next Step:** Begin Phase 1 (Dependencies & Configuration)
