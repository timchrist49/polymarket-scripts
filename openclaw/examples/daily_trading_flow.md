# Daily Trading Flow

This workflow demonstrates how an agent can perform daily trading on the BTC 15-minute market.

## Workflow Steps

### 1. Fetch Current Market Data

Get the active BTC 15-minute market:

```bash
python scripts/fetch_markets.py --btc-mode --json
```

Extract:
- `token_id`: For placing orders
- `price`: Current market price
- `volume`: Liquidity check
- `expiry_time`: Market closing time

### 2. Check Portfolio Status

Review current positions and balance:

```bash
python scripts/portfolio_status.py --json
```

Extract:
- `usdc_balance`: Available trading capital
- `open_orders`: Any pending orders
- `total_value`: Overall portfolio value

### 3. Analyze and Decide

Based on market data and portfolio:
- Assess market direction
- Determine position size
- Decide buy/sell/no action

### 4. Place Test Order (Dry Run)

Always test first (dry-run is default):

```bash
python scripts/place_order.py \
  --btc-mode \
  --side buy \
  --price 0.55 \
  --size 10
```

Verify:
- Order parameters are correct
- No errors returned
- Sufficient balance

### 5. Place Live Order

If dry run succeeds, add --live flag:

```bash
python scripts/place_order.py \
  --btc-mode \
  --side buy \
  --price 0.55 \
  --size 10 \
  --live
```

### 6. Verify Order

Check that order was placed:

```bash
python scripts/portfolio_status.py
```

Confirm order appears in open orders.

## Example Agent Prompt

```
1. Fetch the current BTC 15-minute market from Polymarket
2. Check my portfolio status
3. If I have more than $100 available and the market price is below 0.50, place a buy order for $50 worth of shares at the current price
4. Use dry-run mode first, then execute if the test succeeds
5. Report the final portfolio status
```

## Risk Management

- Never trade more than 10% of portfolio per trade
- Always test without --live flag before live orders
- Check market expiry before trading
- Verify order was placed successfully
