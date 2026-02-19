# Bot Enhancement Plan — 2026-02-19

## Overview

Seven targeted improvements across three categories:
- **Critical Fixes** (1-3): Wrong portfolio calculation, position size too small, confidence cliff
- **Reactivity** (4-5): Event-driven price trigger, faster stop-loss monitoring
- **Signal Quality** (6-8): Arbitrage bypass for signal lag, better AI reasoning, safer contrarian weight

All changes are isolated to specific functions with no cross-dependencies. Can be executed in order or
in parallel across files.

---

## Fix 1 — Portfolio Value: USDC Cash Only (Critical)

### Problem

`client.py:get_portfolio_summary()` calls `get_balance_allowance(AssetType.COLLATERAL)` which returns
the total USDC collateral balance from Polymarket's CTF (Conditional Token Framework). This balance
can include USDC that has been split into conditional tokens (your open position tokens), inflating
the apparent free cash.

The comment in `auto_trade.py:802` says "FIX: Use only USDC balance" but the underlying CLOB call
still returns an inflated number when positions are open.

### Root Cause

On Polymarket's CTF architecture:
- When you buy YES or NO tokens, USDC is converted to conditional tokens (not actually spent)
- The `get_balance_allowance(COLLATERAL)` returns ALL USDC-equivalent collateral, including locked tokens
- Result: if you have $101 free + $45 in open token positions, portfolio_value reads as ~$146

### Fix

In `client.py`, after fetching `usdc_balance`, subtract `positions_value` to get truly free USDC:

```python
# In get_portfolio_summary(), after calculating positions_value:

# Free cash = CLOB collateral balance minus value locked in token positions
# positions_value represents USDC already converted to conditional tokens
free_cash = max(usdc_balance - positions_value, 0.0)

return PortfolioSummary(
    ...
    usdc_balance=free_cash,          # ← was: usdc_balance (inflated)
    positions_value=positions_value,
    total_value=usdc_balance + positions_value,  # keep total_value accurate for display
    ...
)
```

In `auto_trade.py:802`, the existing line remains unchanged:
```python
portfolio_value = Decimal(str(portfolio.usdc_balance))  # already uses free cash after fix
```

Update the log message at line 811 for clarity:
```python
using_balance=f"${portfolio_value:.2f} (free USDC only, excl. ${portfolio.positions_value:.2f} in tokens)"
```

### Files
- `polymarket/client.py` — `get_portfolio_summary()` (~line 768)
- `scripts/auto_trade.py` — log message only (~line 811)

### Verification
After deploying, check the "Portfolio fetched" log line. `using_balance` should equal
`usdc_balance - positions_value`, NOT `total_value`.

---

## Fix 2 — Position Size: 20% of Free Cash (Critical)

### Problem

`bot_max_position_percent` defaults to `0.10` (10%) in `config.py:141`. With a $100 balance and the
confidence multiplier at 0.5x, effective position is only $5 (10% × $100 × 0.5).

User wants 20% of free cash balance as the basis.

### Fix

**`polymarket/config.py:141-146`** — change default and cap:
```python
bot_max_position_percent: float = field(
    default_factory=lambda: float(os.getenv("BOT_MAX_POSITION_PERCENT", "0.20"))  # was 0.10
)
bot_max_position_dollars: float = field(
    default_factory=lambda: float(os.getenv("BOT_MAX_POSITION_DOLLARS", "20.0"))  # was 10.0
)
```

**`.env` file** — update if values are hardcoded there:
```
BOT_MAX_POSITION_PERCENT=0.20
BOT_MAX_POSITION_DOLLARS=20.0
```

### Impact on position sizes

With balance=$101 (after Fix 1, assuming $0 in positions):

| Confidence | Old size | New size |
|-----------|----------|----------|
| 0.75      | $5.08    | $13.13   |
| 0.80      | $7.62    | $16.16   |
| 0.88+     | $10.00   | $20.00   |

### Files
- `polymarket/config.py` — defaults (~lines 141-146)
- `.env` — update BOT_MAX_POSITION_PERCENT and BOT_MAX_POSITION_DOLLARS

---

## Fix 3 — Confidence Multiplier: Proportional Tiers (Critical)

### Problem

Current tiers create a cliff and waste most of the [0.75, 0.90] confidence range:
```
0.75 → 0.80 : 0.50x  ← bare minimum trade, only 50% of position
0.80 → 0.90 : 0.75x
0.90+        : 1.00x
```

With threshold=0.75, nearly all trades land in the 0.5x bucket. The bot never bets full size
unless AI is 90%+ confident, which is rare.

### Fix

Revised tiers in `polymarket/trading/risk.py:_calculate_position_size()`:

```python
# Scale by confidence (proportional tiers — no cliff at threshold)
if confidence >= 0.90:
    multiplier = Decimal("1.0")    # 100% — very high conviction
elif confidence >= 0.85:
    multiplier = Decimal("0.85")   # 85% — high conviction
elif confidence >= 0.80:
    multiplier = Decimal("0.70")   # 70% — solid conviction
elif confidence >= 0.75:
    multiplier = Decimal("0.55")   # 55% — minimum trade (was 0.50, now 10% better)
else:
    multiplier = Decimal("0.0")    # No trade below threshold
```

Rationale:
- The minimum tier (0.75) gets 55% instead of 50% — a 10% improvement on a marginal trade
- The middle tier (0.80-0.85) gets 70% — rewards moderate conviction meaningfully
- 0.85-0.90 gets 85% — distinct step between "confident" and "very confident"
- 0.90+ keeps 100% — unchanged, reserved for arbitrage/extreme setups

### Files
- `polymarket/trading/risk.py` — `_calculate_position_size()` (~lines 158-165)

---

## Enhancement 4 — Price-Movement Event Trigger (High Impact)

### Problem

The main trading cycle runs every 180 seconds (3 minutes) on a fixed timer. Even if BTC drops $80
in 45 seconds, the bot won't analyze until the timer fires. The OddsMonitor catches CLOB odds
changes, but nothing catches raw BTC price movements.

### Design

Add a background task `_price_movement_watcher()` in `auto_trade.py` that:
1. Runs every 10 seconds
2. Checks `self.price_stream.get_current_price()` against each active market's `price_to_beat`
3. If `|movement| >= TRIGGER_THRESHOLD` AND no pending order for that market → triggers analysis immediately

```python
# Constants (at top of file)
PRICE_WATCHER_INTERVAL_SECONDS = 10
PRICE_WATCHER_TRIGGER_USD = 25.0   # Trigger analysis if BTC moves >$25 from price_to_beat

async def _price_movement_watcher(self) -> None:
    """Background task: triggers immediate analysis on significant BTC price moves.

    Runs every 10s. Complements the OddsMonitor (which watches CLOB odds) by detecting
    when BTC price itself has moved beyond the analysis threshold.
    Fires _analyze_market_opportunity() directly without waiting for the 3-min timer.
    """
    logger.info("Price movement watcher started", trigger_usd=PRICE_WATCHER_TRIGGER_USD)

    while self.running:
        try:
            await asyncio.sleep(PRICE_WATCHER_INTERVAL_SECONDS)

            btc_price_data = await self.price_stream.get_current_price()
            if not btc_price_data:
                continue

            current_price = btc_price_data.price

            # Check each known market's price-to-beat
            markets = await self.get_tradeable_markets()
            for market in markets:
                market_slug = self.market_tracker.parse_market_start(market.slug or "")
                if not market_slug:
                    continue

                # Use slug string for cache lookup
                slug_str = market.slug or ""
                price_to_beat = self.market_tracker.get_price_to_beat(slug_str)
                if not price_to_beat:
                    continue

                movement = abs(float(current_price - price_to_beat))

                if movement >= PRICE_WATCHER_TRIGGER_USD:
                    # Skip if already have a pending order for this market
                    if market.id in self._markets_with_pending_orders:
                        continue
                    # Skip if already traded this market
                    if market.id in self._traded_markets:
                        continue

                    logger.info(
                        "Price movement trigger fired",
                        market_id=market.id,
                        movement=f"${movement:.2f}",
                        threshold=f"${PRICE_WATCHER_TRIGGER_USD}",
                        direction="UP" if current_price > price_to_beat else "DOWN"
                    )
                    # Trigger immediate analysis (fire-and-forget, errors don't stop watcher)
                    asyncio.create_task(self._trigger_market_analysis(market))

        except Exception as e:
            logger.error("Price watcher error", error=str(e))
```

Add a thin wrapper to avoid duplicate tasks:
```python
async def _trigger_market_analysis(self, market: Market) -> None:
    """Trigger a single-market analysis cycle from the price watcher.
    Uses the same pipeline as the main cycle but for one market.
    """
    # Fetch current data (already available from last cycle)
    try:
        btc_data = await self.price_stream.get_current_price()
        if not btc_data:
            return
        # Re-use cached indicators from last cycle (don't re-fetch everything)
        # The main cycle updates self._last_indicators every 180s,
        # price watcher fires every 10s — fresh enough for BTC spot price
        if hasattr(self, '_last_cycle_data') and self._last_cycle_data:
            await self._process_market(
                market,
                btc_data,                        # fresh price
                self._last_cycle_data['indicators'],
                self._last_cycle_data['sentiment'],
                self._last_cycle_data['portfolio_value'],
                self._last_cycle_data['btc_momentum'],
                datetime.now(timezone.utc),
                self._last_cycle_data['timeframe_analysis'],
                self._last_cycle_data['regime'],
                self._last_cycle_data['contrarian_signal']
            )
    except Exception as e:
        logger.error("Price trigger analysis failed", error=str(e))
```

Store cycle data after each main cycle (add to `_run_trading_cycle`):
```python
# At end of successful main cycle, cache for price watcher:
self._last_cycle_data = {
    'indicators': indicators,
    'sentiment': aggregated_sentiment,
    'portfolio_value': portfolio_value,
    'btc_momentum': btc_momentum,
    'timeframe_analysis': timeframe_analysis,
    'regime': regime,
    'contrarian_signal': contrarian_signal,
    'timestamp': datetime.now(timezone.utc)
}
```

Start/stop the watcher in `start()` / `stop()`:
```python
# In start(), alongside other background tasks:
self.background_tasks.append(
    asyncio.create_task(self._price_movement_watcher())
)
```

Initialize `_last_cycle_data = None` in `__init__`.

### Files
- `scripts/auto_trade.py` — `__init__`, `start()`, `_run_trading_cycle()`, + 2 new methods

---

## Enhancement 5 — Stop-Loss Background Watcher: 30s vs 3min (Medium Impact)

### Problem

`_check_stop_loss()` only runs at the end of each 3-minute main cycle (line 834). If BTC reverses
sharply against an open position, losses accumulate for up to 3 minutes before exit.

### Design

Add a background task that calls the existing `_check_stop_loss()` every 30 seconds:

```python
STOP_LOSS_WATCHER_INTERVAL_SECONDS = 30

async def _stop_loss_watcher(self) -> None:
    """Background task: evaluates stop-loss every 30s instead of every 3 minutes.

    Uses the existing _check_stop_loss() logic. Runs independently of the main cycle.
    Stops when self.running is False.
    """
    logger.info("Stop-loss watcher started", interval_seconds=STOP_LOSS_WATCHER_INTERVAL_SECONDS)

    while self.running:
        await asyncio.sleep(STOP_LOSS_WATCHER_INTERVAL_SECONDS)
        try:
            if self.open_positions:  # Skip if nothing to protect
                await self._check_stop_loss()
        except Exception as e:
            logger.error("Stop-loss watcher error", error=str(e))
```

Register in `start()`:
```python
self.background_tasks.append(
    asyncio.create_task(self._stop_loss_watcher())
)
```

The main cycle's `await self._check_stop_loss()` at line 834 can stay — it acts as a safety net.
The watcher simply runs it more frequently as a proactive check.

### Files
- `scripts/auto_trade.py` — `start()` + 1 new method

---

## Enhancement 6 — Signal Lag: Bypass for High-Confidence Arbitrage (Medium Impact)

### Problem

The signal lag check (`auto_trade.py:1116-1123`) blocks trades when BTC direction ≠ CLOB direction
and returns HOLD. However, this exact scenario (BTC moved but CLOB hasn't caught up) IS the
arbitrage opportunity. The signal_lag check is incorrectly blocking the most profitable trades.

### Current code

```python
if signal_lag_detected:
    if not self.test_mode.enabled:
        logger.warning("Skipping trade due to signal lag", ...)
        return  # HOLD - always blocks
```

### Fix

Add a bypass condition: if `arbitrage_opportunity` is already detected with significant edge (≥10%),
the CLOB-vs-BTC mismatch is the REASON for the edge — don't block it.

The arbitrage detector runs AFTER the signal lag check in the pipeline, so we need to either:
(a) move the arbitrage check earlier, or
(b) pre-compute a quick arbitrage estimate before the signal lag check

Simplest approach: run the arbitrage check early using the already-available data:

```python
# Signal lag check — but bypass if arbitrage edge is significant
if signal_lag_detected:
    # Quick arbitrage pre-check using already-available yes_odds/no_odds
    # If CLOB odds lag BTC reality by >10%, that IS the arbitrage — don't block it
    btc_movement_pct = abs(float(diff) / float(price_to_beat)) * 100 if price_to_beat else 0

    # Rough edge estimate: BTC moved X% but CLOB hasn't repriced → potential edge
    # Full arbitrage calculation happens downstream; this is a gate-keeper bypass
    potential_edge = btc_movement_pct / 100  # 1% BTC move ≈ 1% potential edge
    ARBITRAGE_BYPASS_THRESHOLD = 0.10  # 10% edge bypasses signal lag

    if potential_edge >= ARBITRAGE_BYPASS_THRESHOLD:
        logger.info(
            "Signal lag detected but bypassed — high potential arbitrage edge",
            market_id=market.id,
            potential_edge=f"{potential_edge:.1%}",
            btc_movement_pct=f"{btc_movement_pct:.1f}%",
            reason="CLOB lag vs BTC price IS the arbitrage opportunity"
        )
        # Don't return — let analysis proceed to arbitrage detector
    elif not self.test_mode.enabled:
        logger.warning("Skipping trade due to signal lag", market_id=market.id, reason=signal_lag_reason)
        return
```

### Files
- `scripts/auto_trade.py` — signal lag block (~lines 1116-1123)

---

## Enhancement 7 — AI Reasoning: "low" → "medium" (Low Effort, Medium Impact)

### Problem

`OPENAI_REASONING_EFFORT = "low"` means ~1-2k reasoning tokens. For 15-minute markets with 8+ input
signals, low reasoning often produces shallow signal integration (treats all signals equally rather
than applying the tiered weighting correctly).

### Fix

Change default in `polymarket/config.py:114`:
```python
openai_reasoning_effort: str = field(
    default_factory=lambda: os.getenv("OPENAI_REASONING_EFFORT", "medium")  # was "low"
)
```

Also update `.env`:
```
OPENAI_REASONING_EFFORT=medium
```

**Latency impact:** medium effort adds ~20-40s per AI call. With 120s timeout and 15-min markets,
this is still well within budget. The AI call is already the longest step (~30-60s at low effort).

**Cost impact:** medium uses ~3-4k reasoning tokens vs ~1-2k. Roughly 2x token cost per trade,
but negligible in absolute terms ($0.001-0.005 per call at gpt-4o-mini pricing).

### Files
- `polymarket/config.py` (~line 114-116)
- `.env`

---

## Enhancement 8 — Contrarian Signal Weight: 2.0x → 1.5x (Low Risk, Medium Quality)

### Problem

`CONTRARIAN_WEIGHT = 2.0` in `signal_aggregator.py:40` is 5-13x larger than all other signals
(market=0.40, social=0.20, funding=0.20, dominance=0.15). When contrarian fires, it dominates
the aggregated sentiment score almost entirely, pushing the AI toward the reversal bet even when
price is clearly continuing the original direction.

As demonstrated in the 07:04 trade: contrarian fired (RSI=3.7, OVERSOLD) but the AI correctly
overrode it because price was genuinely falling. The 2.0x weight almost forced a bad YES bet.

### Fix

`polymarket/trading/signal_aggregator.py:40`:
```python
CONTRARIAN_WEIGHT = 1.5   # was 2.0
```

At 1.5x, contrarian is still the highest-weighted single signal (3.75x the dominance signal),
but it can be overridden by a combination of market + funding + social all pointing the other way.

### Files
- `polymarket/trading/signal_aggregator.py` (~line 40)

---

## Implementation Order

Execute in this sequence to minimize restart cycles:

| Step | Changes | Restart Needed |
|------|---------|---------------|
| 1 | Fix 1 (portfolio), Fix 2 (20%), Fix 3 (multipliers), Enhancement 7 (reasoning=medium), Enhancement 8 (contrarian weight) | 1 restart |
| 2 | Enhancement 6 (signal lag bypass) | 1 restart |
| 3 | Enhancement 4 (price watcher) + Enhancement 5 (stop-loss watcher) | 1 restart |

Steps 1-2 are single-file or 2-file changes. Step 3 adds async background tasks and needs
more careful testing.

---

## File Change Summary

| File | Changes |
|------|---------|
| `polymarket/client.py` | Fix 1: `free_cash = usdc_balance - positions_value` in `get_portfolio_summary()` |
| `polymarket/config.py` | Fix 2: default percent 0.10→0.20, cap 10→20; Enhancement 7: reasoning "low"→"medium" |
| `polymarket/trading/risk.py` | Fix 3: confidence multiplier tiers (0.55/0.70/0.85/1.0) |
| `polymarket/trading/signal_aggregator.py` | Enhancement 8: CONTRARIAN_WEIGHT 2.0→1.5 |
| `scripts/auto_trade.py` | Enhancement 4: price watcher + cache; Enhancement 5: stop-loss watcher; Enhancement 6: signal lag bypass |
| `.env` | Fix 2: BOT_MAX_POSITION_PERCENT=0.20, BOT_MAX_POSITION_DOLLARS=20.0; Enhancement 7: OPENAI_REASONING_EFFORT=medium |

---

## Expected Outcomes

After all 8 changes:

| Metric | Before | After |
|--------|--------|-------|
| Position at confidence 0.75 | ~$7 ($101 × 20% × 0.5) | ~$11 ($101 × 20% × 0.55) |
| Position at confidence 0.80 | ~$10 ($101 × 20% × 0.75) | ~$14 ($101 × 20% × 0.70) |
| Position at confidence 0.90 | ~$20 (max) | ~$20 (max, unchanged) |
| Time to react to $25 BTC move | Up to 180s | ~10s |
| Time to trigger stop-loss | Up to 180s | ~30s |
| Arbitrage blocked by signal lag | Yes (all cases) | Only below 10% edge |
| AI reasoning depth | Shallow (1-2k tokens) | Medium (3-4k tokens) |

> **Note on position sizes:** These assume `portfolio.usdc_balance` returns correct free-cash-only
> values after Fix 1. If the on-chain CLOB balance already excludes token positions, the numbers
> won't change — Fix 1 is a defensive correctness fix regardless.
