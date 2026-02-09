# Monitoring Workflow

This workflow demonstrates periodic monitoring of Polymarket positions and orders.

## Workflow Steps

### 1. Check Portfolio Status

```bash
python scripts/portfolio_status.py --json
```

### 2. Analyze Open Orders

Check for:
- Stale orders (older than 1 hour)
- Orders at unfavorable prices
- Orders that should be cancelled

### 3. Check Market Prices

```bash
python scripts/fetch_markets.py --btc-mode --json
```

### 4. Decision Points

**If open orders exist:**
- Are they still relevant?
- Should any be cancelled?

**If positions exist:**
- Should profit be taken?
- Should stop-loss be triggered?

**If cash available:**
- Are there good entry opportunities?

## Scheduling

Run this workflow every 5 minutes for active trading:

```bash
# Every 5 minutes via cron
*/5 * * * * cd /path/to/polymarket-scripts && python scripts/portfolio_status.py >> monitoring.log
```

## Alerting

Set up alerts for:
- Total portfolio value changes > 10%
- New orders filled
- Orders rejected
- Authentication failures
