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

## Configuration Reference

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | - | OpenAI API key (required) |
| `OPENAI_MODEL` | `gpt-4o-mini` | Model for decisions |
| `TAVILY_API_KEY` | - | Tavily API key (required) |
| `BTC_PRICE_SOURCE` | `binance` | Price source (binance/coingecko) |
| `BTC_PRICE_CACHE_SECONDS` | `30` | Cache TTL for BTC price |
| `BOT_INTERVAL_SECONDS` | `180` | Cycle interval |
| `BOT_CONFIDENCE_THRESHOLD` | `0.75` | Min confidence to trade |
| `BOT_MAX_POSITION_PERCENT` | `0.10` | Max position (% of portfolio) |
| `BOT_MAX_EXPOSURE_PERCENT` | `0.50` | Max total exposure |
| `STOP_LOSS_ODDS_THRESHOLD` | `0.40` | Exit if odds drop below |
| `STOP_LOSS_FORCE_EXIT_MINUTES` | `5` | Force exit before expiry |
| `BOT_LOG_DECISIONS` | `true` | Log all decisions |
| `BOT_LOG_FILE` | `logs/auto_trade.log` | Log file path |

## License

Same as parent project.
