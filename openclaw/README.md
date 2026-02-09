# Polymarket Scripts for OpenClaw Agents

Quick-start guide for OpenClaw agents to use Polymarket trading scripts.

## Quick Start

1. Copy `.env.example` to `.env` and add credentials
2. Install dependencies: `pip install -r requirements.txt`
3. Run a command: `python scripts/fetch_markets.py --btc-mode`

## Available Skills

### polymarket_fetch
Fetch Polymarket market data, including BTC 15-minute markets.

### polymarket_trade
Place buy/sell orders on Polymarket markets.

### polymarket_status
Check portfolio status, open orders, and positions.

## Agent Prompts

See `openclaw/prompts/` for reusable prompt templates:
- `fetch_markets.txt` - Fetch current market data
- `place_order.txt` - Place trading orders
- `portfolio_check.txt` - Check portfolio status

## Example Workflows

See `openclaw/examples/` for complete agent workflows:
- `daily_trading_flow.md` - Daily trading workflow
- `monitoring_workflow.md` - Monitoring workflow

## Environment Setup

Required environment variables in `.env`:

```bash
POLYMARKET_MODE=trading
POLYMARKET_PRIVATE_KEY=0x...
POLYMARKET_API_KEY=...
POLYMARKET_API_SECRET=...
POLYMARKET_API_PASSPHRASE=...
POLYMARKET_FUNDER=0x...
POLYMARKET_SIGNATURE_TYPE=1
```

## Common Commands

```bash
# Fetch BTC 15-min market
python scripts/fetch_markets.py --btc-mode

# Check portfolio
python scripts/portfolio_status.py

# Place order (dry run)
python scripts/place_order.py --btc-mode --side buy --price 0.55 --size 10 --dry-run true
```

## Troubleshooting

- Authentication failed: Verify API credentials are current
- Market not found: Check market is active and accepting orders
- Order rejected: Verify price range (0-1) and minimum order size
