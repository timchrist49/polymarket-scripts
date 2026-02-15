# Polymarket BTC 15-Minute Trading Skill Pack

Production-grade Python scripts for trading the "BTC Up or Down 15 Minutes" market on Polymarket.

## 🤖 Autonomous Trading Bot

**NEW**: Fully autonomous trading bot that runs 24/7. See [README_BOT.md](./README_BOT.md) for complete documentation.

**Quick Start:**
```bash
# Configure .env with trading credentials
cp .env.example .env

# Start bot as daemon (survives terminal disconnect)
./start_bot.sh start

# Monitor live
./start_bot.sh logs
```

The bot:
- Analyzes BTC price, social sentiment, and market microstructure every 3 minutes
- Makes AI-powered trading decisions using GPT-4o-mini
- Executes trades automatically with risk management
- Runs as a background daemon independent of terminal sessions

### Price Data Source

The bot uses **Chainlink oracle prices** from Polymarket's RTDS API, ensuring alignment with how Polymarket settles markets. This provides:

- **Accurate price references:** Same price source used for settlement
- **Real-time updates:** WebSocket connection to Chainlink price feed
- **Source attribution:** Every trade logs which price source was used

Previous versions used Binance prices, which could differ by thousands of dollars from settlement prices. See `docs/CHAINLINK_MIGRATION.md` for migration details.

## Manual Trading Scripts

For manual/scripted trading, use these individual scripts:

- **fetch_markets.py**: Fetch market data, discover active BTC 15-min markets
- **place_order.py**: Place buy/sell orders with dry-run mode
- **portfolio_status.py**: Check open orders and positions

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your credentials
```

### 3. Authentication Modes

**READ_ONLY Mode** (Market data only):
```bash
POLYMARKET_MODE=read_only
# No private key needed
```

**TRADING Mode** (Full trading):
```bash
POLYMARKET_MODE=trading
POLYMARKET_PRIVATE_KEY=0x...
POLYMARKET_API_KEY=...
POLYMARKET_API_SECRET=...
POLYMARKET_API_PASSPHRASE=...
```

To get API credentials, run:
```bash
python -c "
from py_clob_client.client import ClobClient
import os

client = ClobClient(
    host='https://clob.polymarket.com',
    chain_id=137,
    key=os.getenv('POLYMARKET_PRIVATE_KEY')
)
creds = client.create_or_derive_api_key()
print(f'API_KEY={creds[\"apiKey\"]}')
print(f'API_SECRET={creds[\"secret\"]}')
print(f'API_PASSPHRASE={creds[\"passphrase\"]}')
"
```

## Usage

### Fetch Markets

```bash
# Fetch active BTC 15-min market
python scripts/fetch_markets.py --btc-mode

# Search any market
python scripts/fetch_markets.py --search "bitcoin" --limit 50

# JSON output
python scripts/fetch_markets.py --btc-mode --json
```

### Place Orders

```bash
# Dry run (default)
python scripts/place_order.py \
  --btc-mode \
  --side buy \
  --price 0.55 \
  --size 10 \
  --dry-run true

# Live order
python scripts/place_order.py \
  --btc-mode \
  --side buy \
  --price 0.55 \
  --size 10 \
  --dry-run false

# Manual market/token IDs
python scripts/place_order.py \
  --market-id 0x... \
  --token-id 0x... \
  --side sell \
  --price 0.60 \
  --size 5
```

### Portfolio Status

```bash
# Check all open orders
python scripts/portfolio_status.py

# Filter by market
python scripts/portfolio_status.py --market-id 0x...

# JSON output
python scripts/portfolio_status.py --json
```

## BTC Market Discovery

The "BTC Up or Down 15 Minutes" market uses slugs with Unix timestamps:

```
btc-updown-15m-{epoch_timestamp}
```

The timestamp represents the **start** of the 15-minute interval:
- Current time: 10:09 AM UTC
- Interval floor: 10:00 AM UTC
- Slug: `btc-updown-15m-1770608700`

## Quick Verification Checklist

```bash
# 1. Verify read_only mode works (no credentials needed)
python scripts/fetch_markets.py --btc-mode

# 2. Verify dry-run mode (no order sent)
python scripts/place_order.py --btc-mode --side buy --price 0.50 --size 1 --dry-run true

# 3. Verify trading mode fails without credentials
POLYMARKET_MODE=trading python scripts/place_order.py --btc-mode --side buy --price 0.50 --size 1 --dry-run true
# Should error: "Missing required credentials for TRADING mode"

# 4. Verify portfolio returns empty (not crash)
python scripts/portfolio_status.py
```

## Troubleshooting

### "Market not found"
- Check market is active and accepting orders
- Try manual `--market-id` from Polymarket dashboard URL

### "Authentication failed"
- Verify private key format (0x prefix)
- Ensure API credentials are current (derive again if needed)

### "Order rejected"
- Check price is in valid range (0-1 for binary)
- Verify size meets minimum order size
- Ensure market is `acceptingOrders=true`

## Security

- Never commit `.env` file
- Never log private keys or signatures
- Use read-only mode when possible
- Keep dry-run enabled until tested

## OpenClaw Agent Integration

This repository includes agent-specific resources in the `openclaw/` directory for use with OpenClaw agents.

### Available Skills

- **polymarket_fetch**: Fetch market data
- **polymarket_trade**: Place orders
- **polymarket_status**: Check portfolio

### Quick Start for Agents

1. Copy `.env.example` to `.env` and add credentials
2. Install: `pip install -r requirements.txt`
3. Run: `python scripts/fetch_markets.py --btc-mode`

See [`openclaw/README.md`](openclaw/README.md) for detailed agent documentation.

### Example Agent Workflow

```bash
# 1. Fetch market data
python scripts/fetch_markets.py --btc-mode

# 2. Check portfolio
python scripts/portfolio_status.py

# 3. Place order (dry run first)
python scripts/place_order.py --btc-mode --side buy --price 0.55 --size 10

# 4. Place live order
python scripts/place_order.py --btc-mode --side buy --price 0.55 --size 10 --live
```

See `openclaw/examples/` for complete workflow examples.

## Known Limitations

- MARKET orders are emulated via aggressive LIMIT orders
- BTC market discovery may need manual verification
- Rate limits: implement backoff for 429 responses
- WebSocket streaming: TODO for future
