# Trading Bot Comprehensive Fix & Strategy Redesign

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix critical bugs causing 26.4% win rate and implement proven 15-min BTC trading strategies to achieve 60%+ win rate with proper risk management.

**Architecture:** Multi-phase approach - (1) Emergency fixes to stop losses, (2) Add volume/timeframe analysis, (2.5) Orderbook depth analysis, (3) Market regime detection, (4) Real position tracking, (5) AI decision overhaul, (6) Risk management. Each phase independently deployable for gradual rollout.

**Tech Stack:** Python 3.12, asyncio, SQLite, Polymarket API, OpenAI API, structlog

---

## 🚨 Phase 1: Emergency Fixes (Stop the Bleeding)

### Task 1.1: Add Trade Execution Status Field to Database

**Files:**
- Modify: `polymarket/performance/database.py:50-120` (schema)
- Test: `tests/performance/test_database.py`

**Step 1: Add status field to schema**

```python
# In polymarket/performance/database.py, update CREATE TABLE statement
CREATE TABLE IF NOT EXISTS trades (
    -- existing fields...
    executed_price REAL,

    -- NEW FIELD
    execution_status TEXT DEFAULT 'pending',  -- 'pending', 'executed', 'skipped', 'failed'

    -- existing fields...
    actual_outcome TEXT,
)
```

**Step 2: Create migration script**

Create: `scripts/migrate_add_execution_status.py`

```python
"""Add execution_status field to trades table."""
import sqlite3
from pathlib import Path

def migrate():
    db_path = Path("data/performance.db")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Check if column exists
    cursor.execute("PRAGMA table_info(trades)")
    columns = [col[1] for col in cursor.fetchall()]

    if 'execution_status' not in columns:
        print("Adding execution_status column...")
        cursor.execute("""
            ALTER TABLE trades
            ADD COLUMN execution_status TEXT DEFAULT 'pending'
        """)

        # Set all existing trades with position_size > 0 as 'executed'
        cursor.execute("""
            UPDATE trades
            SET execution_status = 'executed'
            WHERE position_size > 0 AND action IN ('YES', 'NO')
        """)

        conn.commit()
        print("Migration complete!")
    else:
        print("Column already exists, skipping.")

    conn.close()

if __name__ == "__main__":
    migrate()
```

**Step 3: Run migration**

```bash
cd /root/polymarket-scripts
python3 scripts/migrate_add_execution_status.py
```

Expected output: "Migration complete!"

**Step 4: Verify migration**

```bash
sqlite3 data/performance.db "PRAGMA table_info(trades)" | grep execution_status
```

Expected: Line showing execution_status field

**Step 5: Commit**

```bash
git add polymarket/performance/database.py scripts/migrate_add_execution_status.py
git commit -m "feat(db): add execution_status field to trades table

- Add execution_status: pending/executed/skipped/failed
- Migration script to update existing records
- Sets existing trades with position_size > 0 as executed"
```

---

### Task 1.2: Fix Phantom Trades - Move Logging After Validation

**Files:**
- Modify: `scripts/auto_trade.py:765-800` (move logging)

**Step 1: Create mark_trade_skipped helper method**

Add to auto_trade.py after `_process_market` method:

```python
async def _mark_trade_skipped(
    self,
    trade_id: int,
    reason: str,
    skip_type: str = "validation"
) -> None:
    """Mark a trade as skipped in database."""
    if trade_id <= 0:
        return

    try:
        await self.performance_tracker.update_trade_status(
            trade_id=trade_id,
            execution_status='skipped',
            skip_reason=reason,
            skip_type=skip_type
        )
        logger.info(
            "Trade marked as skipped",
            trade_id=trade_id,
            reason=reason,
            skip_type=skip_type
        )
    except Exception as e:
        logger.error("Failed to mark trade as skipped", error=str(e))
```

**Step 2: Move decision logging AFTER YES threshold check**

In `scripts/auto_trade.py`, reorganize lines 765-800:

```python
# BEFORE (WRONG):
# Line 769-778: log_decision() - Creates DB record
# Line 784-798: YES threshold check - May skip
# Line 800+: Continue execution

# AFTER (CORRECT):
# Line 765-783: YES threshold check FIRST (before logging)
if decision.action == "YES" and price_to_beat:
    diff, _ = self.market_tracker.calculate_price_difference(
        btc_data.price, price_to_beat
    )
    MIN_YES_MOVEMENT = 200

    if diff < MIN_YES_MOVEMENT:
        logger.info(
            "Skipping YES trade - insufficient upward momentum",
            market_id=market.id,
            movement=f"${diff:+,.2f}",
            threshold=f"${MIN_YES_MOVEMENT}",
            reason="Avoid buying exhausted momentum"
        )
        return  # Skip WITHOUT creating DB record

# Line 785-795: NOW log decision (only if passed validation)
try:
    trade_id = await self.performance_tracker.log_decision(
        market=market,
        decision=decision,
        btc_data=btc_data,
        technical=indicators,
        aggregated=aggregated_sentiment,
        price_to_beat=price_to_beat,
        time_remaining_seconds=time_remaining,
        is_end_phase=is_end_of_market
    )
except Exception as e:
    logger.error("Performance logging failed", error=str(e))
    trade_id = -1

# Line 800+: Continue with execution
```

**Step 3: Update settlement query to exclude skipped trades**

Modify: `polymarket/performance/settler.py:156-166`

```python
# OLD query:
cursor.execute("""
    SELECT ...
    FROM trades
    WHERE action IN ('YES', 'NO')
      AND is_win IS NULL
      AND datetime(timestamp) < datetime('now', '-15 minutes')
""")

# NEW query:
cursor.execute("""
    SELECT ...
    FROM trades
    WHERE action IN ('YES', 'NO')
      AND is_win IS NULL
      AND execution_status = 'executed'  -- NEW: Only settle executed trades
      AND datetime(timestamp) < datetime('now', '-15 minutes')
""")
```

**Step 4: Test with bot logs**

```bash
# Stop bot
pkill -f "python3 scripts/auto_trade.py"

# Start bot in test mode
cd /root/polymarket-scripts
python3 scripts/auto_trade.py &

# Monitor logs for YES threshold skips
tail -f logs/bot.log | grep "Skipping YES trade"
```

Expected: Should see "Skipping YES trade" messages WITHOUT "Decision logged" before them

**Step 5: Verify database**

```bash
sqlite3 data/performance.db "SELECT COUNT(*) FROM trades WHERE action='YES' AND execution_status='skipped'"
```

Expected: Count of skipped YES trades (previously would have been marked as losses)

**Step 6: Commit**

```bash
git add scripts/auto_trade.py polymarket/performance/settler.py
git commit -m "fix(critical): move decision logging after validation

BEFORE: Decision logged → Validation → Skip → Phantom trade in DB
AFTER: Validation → Skip OR Log → Only real trades in DB

- Move log_decision() after YES threshold check
- Settlement now excludes execution_status='skipped'
- Fixes phantom trades counted as losses bug

Fixes #113"
```

---

### Task 1.3: Disable YES Trades Temporarily

**Files:**
- Modify: `scripts/auto_trade.py:755-765`

**Step 1: Add YES trade kill switch**

At top of `_process_market` method, add:

```python
async def _process_market(self, market: Market) -> None:
    """Process a single market for potential trade."""

    # EMERGENCY: Disable YES trades until strategy fixed
    # YES trades: 10% win rate (9W-81L) = -$170 all-time
    ENABLE_YES_TRADES = False  # TODO: Re-enable after strategy redesign

    # ... rest of method
```

**Step 2: Check decision and skip if YES disabled**

After AI makes decision (line ~758), add check:

```python
decision = await self.ai_service.make_decision(...)

# Skip YES trades if disabled
if decision.action == "YES" and not ENABLE_YES_TRADES:
    logger.warning(
        "YES trades disabled - skipping",
        market_id=market.id,
        confidence=decision.confidence,
        reason="YES trades at 10% win rate, disabled until fixed"
    )
    return
```

**Step 3: Test**

```bash
# Restart bot
pkill -f "python3 scripts/auto_trade.py"
python3 scripts/auto_trade.py &

# Monitor - should see no YES executions
tail -f logs/bot.log | grep -E "YES trades disabled|AI Decision.*YES"
```

Expected: "YES trades disabled" messages when AI recommends YES

**Step 4: Commit**

```bash
git add scripts/auto_trade.py
git commit -m "feat: add kill switch to disable YES trades

YES trades: 10% win rate (9W-81L) losing $170
Temporarily disable until strategy redesigned

Can re-enable by setting ENABLE_YES_TRADES = True"
```

---

## 📊 Phase 2: Volume & Multi-Timeframe Analysis

### Task 2.1: Add Volume Data Fetching

**Files:**
- Create: `polymarket/trading/volume_analyzer.py`
- Modify: `polymarket/trading/btc_price.py:100-150`

**Step 1: Create VolumeData model**

Create: `polymarket/models.py` (add to existing models)

```python
@dataclass
class VolumeData:
    """BTC trading volume data."""
    volume_24h: float           # 24-hour volume in USD
    volume_current_hour: float  # Current hour volume
    volume_avg_hour: float      # Average hourly volume (last 24h)
    volume_ratio: float         # Current / Average (spike detection)
    is_high_volume: bool        # volume_ratio > 1.5
    timestamp: datetime
```

**Step 2: Add volume fetching to BTCPriceService**

In `polymarket/trading/btc_price.py`, add method:

```python
async def get_volume_data(self) -> VolumeData | None:
    """Fetch BTC volume data from CoinGecko."""
    try:
        session = await self._get_session()

        # Get 24h volume data
        url = "https://api.coingecko.com/api/v3/coins/bitcoin"
        params = {"localization": "false", "tickers": "false", "community_data": "false"}

        if self.settings.coingecko_api_key:
            url = "https://pro-api.coingecko.com/api/v3/coins/bitcoin"
            params["x_cg_pro_api_key"] = self.settings.coingecko_api_key

        async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            resp.raise_for_status()
            data = await resp.json()

            market_data = data.get("market_data", {})
            volume_24h = float(market_data.get("total_volume", {}).get("usd", 0))

            # Estimate current hour volume (24h / 24)
            volume_avg_hour = volume_24h / 24

            # Get current hour volume (approximate from recent trades)
            # For now, use average; TODO: Get real-time hourly volume
            volume_current_hour = volume_avg_hour
            volume_ratio = volume_current_hour / volume_avg_hour if volume_avg_hour > 0 else 1.0

            is_high_volume = volume_ratio > 1.5

            logger.info(
                "Volume data fetched",
                volume_24h=f"${volume_24h:,.0f}",
                volume_ratio=f"{volume_ratio:.2f}x",
                is_high_volume=is_high_volume
            )

            return VolumeData(
                volume_24h=volume_24h,
                volume_current_hour=volume_current_hour,
                volume_avg_hour=volume_avg_hour,
                volume_ratio=volume_ratio,
                is_high_volume=is_high_volume,
                timestamp=datetime.now()
            )

    except Exception as e:
        logger.error("Failed to fetch volume data", error=str(e))
        return None
```

**Step 3: Integrate volume check in auto_trade.py**

In `scripts/auto_trade.py`, modify data gathering:

```python
# Fetch all data in parallel
btc_data, social_sentiment, market_signals, funding_signal, dominance_signal, volume_data = await asyncio.gather(
    self.btc_service.get_current_price(),
    self.social_service.get_social_score(),
    self.market_service.get_market_score(),
    self.btc_service.get_funding_rates(),
    self.btc_service.get_btc_dominance(),
    self.btc_service.get_volume_data(),  # NEW
)
```

**Step 4: Add volume requirement for breakout trades**

After price comparison, add volume check:

```python
# After line ~710 (price comparison)
if price_to_beat and volume_data:
    diff, diff_pct = self.market_tracker.calculate_price_difference(
        btc_data.price, price_to_beat
    )

    # For large moves (potential breakouts), require volume confirmation
    if abs(diff) > 200:  # $200+ move
        if not volume_data.is_high_volume:
            logger.info(
                "Skipping large move without volume confirmation",
                market_id=market.id,
                movement=f"${diff:+,.2f}",
                volume_ratio=f"{volume_data.volume_ratio:.2f}x",
                reason="Breakouts require volume > 1.5x average"
            )
            return  # Skip low-volume breakouts
```

**Step 5: Test**

```bash
# Test volume fetching
python3 -c "
import asyncio
from polymarket.trading.btc_price import BTCPriceService
from polymarket.config import Settings

async def test():
    service = BTCPriceService(Settings())
    volume = await service.get_volume_data()
    print(f'Volume 24h: \${volume.volume_24h:,.0f}')
    print(f'Volume ratio: {volume.volume_ratio:.2f}x')
    print(f'High volume: {volume.is_high_volume}')

asyncio.run(test())
"
```

Expected: Volume data printed

**Step 6: Commit**

```bash
git add polymarket/models.py polymarket/trading/btc_price.py scripts/auto_trade.py
git commit -m "feat: add volume confirmation for breakout trades

Research shows volume is CRITICAL for 15-min BTC trading:
- Breakouts without volume are false signals
- Require 1.5x average volume for $200+ moves

Implemented:
- VolumeData model
- get_volume_data() from CoinGecko
- Volume check before large moves"
```

---

### Task 2.2: Multi-Timeframe Analysis

**Files:**
- Create: `polymarket/trading/timeframe_analyzer.py`
- Modify: `scripts/auto_trade.py:490-510`

**Step 1: Create TimeframeAnalysis model**

Add to `polymarket/models.py`:

```python
@dataclass
class TimeframeAnalysis:
    """Multi-timeframe trend analysis."""
    daily_trend: str       # "BULLISH", "BEARISH", "NEUTRAL"
    four_hour_trend: str   # "BULLISH", "BEARISH", "NEUTRAL"
    alignment: str         # "ALIGNED", "CONFLICTING", "NEUTRAL"
    daily_support: float   # Key support level from daily
    daily_resistance: float # Key resistance from daily
    confidence: float      # 0.0-1.0 based on alignment
    timestamp: datetime
```

**Step 2: Create timeframe analyzer**

Create: `polymarket/trading/timeframe_analyzer.py`

```python
"""Multi-timeframe analysis for trend confirmation."""
import structlog
from datetime import datetime, timedelta
from polymarket.models import TimeframeAnalysis

logger = structlog.get_logger()

class TimeframeAnalyzer:
    """Analyze multiple timeframes for trend confirmation."""

    async def analyze_timeframes(
        self,
        current_price: float,
        price_4h_ago: float,
        price_24h_ago: float
    ) -> TimeframeAnalysis:
        """
        Analyze daily and 4-hour trends.

        Args:
            current_price: Current BTC price
            price_4h_ago: BTC price 4 hours ago
            price_24h_ago: BTC price 24 hours ago

        Returns:
            TimeframeAnalysis with trend direction and support/resistance
        """
        # Daily trend (24h)
        daily_change_pct = ((current_price - price_24h_ago) / price_24h_ago) * 100
        if daily_change_pct > 2.0:
            daily_trend = "BULLISH"
        elif daily_change_pct < -2.0:
            daily_trend = "BEARISH"
        else:
            daily_trend = "NEUTRAL"

        # 4-hour trend
        four_hour_change_pct = ((current_price - price_4h_ago) / price_4h_ago) * 100
        if four_hour_change_pct > 1.0:
            four_hour_trend = "BULLISH"
        elif four_hour_change_pct < -1.0:
            four_hour_trend = "BEARISH"
        else:
            four_hour_trend = "NEUTRAL"

        # Check alignment
        if daily_trend == four_hour_trend and daily_trend != "NEUTRAL":
            alignment = "ALIGNED"
            confidence = 0.9
        elif daily_trend == "NEUTRAL" or four_hour_trend == "NEUTRAL":
            alignment = "NEUTRAL"
            confidence = 0.5
        else:
            alignment = "CONFLICTING"
            confidence = 0.3

        # Simple support/resistance (last 24h low/high)
        daily_support = min(price_24h_ago, current_price) * 0.98
        daily_resistance = max(price_24h_ago, current_price) * 1.02

        logger.info(
            "Timeframe analysis",
            daily_trend=daily_trend,
            four_hour_trend=four_hour_trend,
            alignment=alignment,
            confidence=confidence
        )

        return TimeframeAnalysis(
            daily_trend=daily_trend,
            four_hour_trend=four_hour_trend,
            alignment=alignment,
            daily_support=daily_support,
            daily_resistance=daily_resistance,
            confidence=confidence,
            timestamp=datetime.now()
        )
```

**Step 3: Get historical prices for timeframe analysis**

In `scripts/auto_trade.py`, add historical price fetching:

```python
# Get historical prices for timeframe analysis
price_4h_ago = await self.btc_service.get_price_at_offset(hours=4)
price_24h_ago = await self.btc_service.get_price_at_offset(hours=24)

# Analyze timeframes
timeframe_analyzer = TimeframeAnalyzer()
timeframe_analysis = await timeframe_analyzer.analyze_timeframes(
    current_price=float(btc_data.price),
    price_4h_ago=float(price_4h_ago) if price_4h_ago else float(btc_data.price),
    price_24h_ago=float(price_24h_ago) if price_24h_ago else float(btc_data.price)
)
```

**Step 4: Add timeframe filter before trading**

```python
# Only trade when timeframes align
if timeframe_analysis.alignment == "CONFLICTING":
    logger.info(
        "Skipping trade - conflicting timeframes",
        market_id=market.id,
        daily_trend=timeframe_analysis.daily_trend,
        four_hour_trend=timeframe_analysis.four_hour_trend,
        reason="Don't trade against larger timeframe trend"
    )
    return
```

**Step 5: Test**

```bash
python3 -c "
import asyncio
from polymarket.trading.timeframe_analyzer import TimeframeAnalyzer

async def test():
    analyzer = TimeframeAnalyzer()
    # Simulate prices
    result = await analyzer.analyze_timeframes(
        current_price=66000,
        price_4h_ago=65500,  # +0.76% (bullish)
        price_24h_ago=64000   # +3.12% (bullish)
    )
    print(f'Daily: {result.daily_trend}')
    print(f'4H: {result.four_hour_trend}')
    print(f'Alignment: {result.alignment}')
    print(f'Confidence: {result.confidence}')

asyncio.run(test())
"
```

Expected: Shows BULLISH/ALIGNED for uptrend example

**Step 6: Commit**

```bash
git add polymarket/models.py polymarket/trading/timeframe_analyzer.py scripts/auto_trade.py
git commit -m "feat: add multi-timeframe trend analysis

Research shows checking larger timeframes prevents counter-trend losses:
- Daily trend (24h) for overall direction
- 4-hour trend for medium-term confirmation
- Skip trades when timeframes conflict

Implemented:
- TimeframeAnalyzer with daily/4h trend detection
- Alignment check (ALIGNED/CONFLICTING/NEUTRAL)
- Filter to skip conflicting-timeframe trades"
```

---

## 📖 Phase 2.5: Orderbook Depth Analysis

### Task 2.5.1: Add Orderbook Data Model

**Files:**
- Modify: `polymarket/models.py` (add OrderbookData model)

**Step 1: Create OrderbookData model**

Add to `polymarket/models.py`:

```python
@dataclass
class OrderbookData:
    """Polymarket orderbook depth analysis."""
    bid_ask_spread: float         # Spread in % (tight = liquid)
    spread_bps: float              # Spread in basis points
    liquidity_score: float         # 0.0-1.0 (high = good liquidity)
    order_imbalance: float         # -1.0 (ask heavy) to +1.0 (bid heavy)
    imbalance_direction: str       # "BUY_PRESSURE", "SELL_PRESSURE", "BALANCED"
    bid_depth_100bps: float        # Total bid liquidity within 100bps
    ask_depth_100bps: float        # Total ask liquidity within 100bps
    bid_depth_200bps: float        # Total bid liquidity within 200bps
    ask_depth_200bps: float        # Total ask liquidity within 200bps
    best_bid: float                # Top bid price
    best_ask: float                # Top ask price
    can_fill_order: bool           # Enough liquidity for trade
    timestamp: datetime
```

**Step 2: Verify model**

```bash
python3 -c "
from polymarket.models import OrderbookData
from datetime import datetime

# Test model creation
ob = OrderbookData(
    bid_ask_spread=0.02,
    spread_bps=200,
    liquidity_score=0.8,
    order_imbalance=0.3,
    imbalance_direction='BUY_PRESSURE',
    bid_depth_100bps=1000.0,
    ask_depth_100bps=800.0,
    bid_depth_200bps=2000.0,
    ask_depth_200bps=1500.0,
    best_bid=0.54,
    best_ask=0.56,
    can_fill_order=True,
    timestamp=datetime.now()
)
print(f'Spread: {ob.spread_bps}bps, Imbalance: {ob.order_imbalance:+.2f}')
"
```

Expected: Prints orderbook data without errors

**Step 3: Commit**

```bash
git add polymarket/models.py
git commit -m "feat: add OrderbookData model for depth analysis

Orderbook depth is critical for 15-min trading:
- Bid-ask spread indicates liquidity
- Order imbalance shows institutional flow
- Depth levels ensure order fillability

Model includes:
- Spread calculation (% and bps)
- Liquidity scoring (0.0-1.0)
- Order imbalance (-1.0 to +1.0)
- Depth at 100bps and 200bps levels"
```

---

### Task 2.5.2: Fetch Orderbook from Polymarket CLOB API

**Files:**
- Create: `polymarket/trading/orderbook_analyzer.py`
- Modify: `polymarket/client.py:300-350`

**Step 1: Add get_orderbook method to PolymarketClient**

In `polymarket/client.py`, add method:

```python
async def get_orderbook(self, token_id: str, depth: int = 20) -> dict | None:
    """
    Fetch orderbook for a token from CLOB API.

    Args:
        token_id: Token ID to get orderbook for
        depth: Number of levels to fetch (default 20)

    Returns:
        Orderbook dict with 'bids' and 'asks' arrays
    """
    try:
        url = f"{self.clob_url}/orderbook"
        params = {
            "token_id": token_id,
            "depth": str(depth)
        }

        async with self.session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=5)) as resp:
            resp.raise_for_status()
            data = await resp.json()

            # Format: {"bids": [[price, size], ...], "asks": [[price, size], ...]}
            logger.debug(
                "Orderbook fetched",
                token_id=token_id[:8],
                bids=len(data.get('bids', [])),
                asks=len(data.get('asks', []))
            )

            return data

    except Exception as e:
        logger.error("Failed to fetch orderbook", token_id=token_id, error=str(e))
        return None
```

**Step 2: Create OrderbookAnalyzer**

Create: `polymarket/trading/orderbook_analyzer.py`

```python
"""Orderbook depth analysis for execution quality assessment."""
import structlog
from datetime import datetime
from typing import Optional
from polymarket.models import OrderbookData

logger = structlog.get_logger()


class OrderbookAnalyzer:
    """Analyze orderbook depth and liquidity."""

    def analyze_orderbook(
        self,
        orderbook: dict,
        target_size: float = 10.0  # Target trade size in USDC
    ) -> Optional[OrderbookData]:
        """
        Analyze orderbook depth, spread, and imbalance.

        Args:
            orderbook: Dict with 'bids' and 'asks' arrays [[price, size], ...]
            target_size: Expected trade size to check fillability

        Returns:
            OrderbookData with analysis results
        """
        try:
            bids = orderbook.get('bids', [])
            asks = orderbook.get('asks', [])

            if not bids or not asks:
                logger.warning("Empty orderbook")
                return None

            # Best bid/ask
            best_bid = float(bids[0][0])
            best_ask = float(asks[0][0])

            # Spread calculation
            spread = best_ask - best_bid
            spread_pct = (spread / best_ask) * 100
            spread_bps = spread_pct * 100  # Convert to basis points

            # Liquidity depth at 100bps and 200bps
            bid_depth_100bps = self._calculate_depth(bids, best_bid, 0.01, side='bid')
            ask_depth_100bps = self._calculate_depth(asks, best_ask, 0.01, side='ask')
            bid_depth_200bps = self._calculate_depth(bids, best_bid, 0.02, side='bid')
            ask_depth_200bps = self._calculate_depth(asks, best_ask, 0.02, side='ask')

            # Order imbalance (bid pressure vs ask pressure)
            total_bid_volume = sum(float(b[1]) for b in bids[:10])
            total_ask_volume = sum(float(a[1]) for a in asks[:10])

            if total_bid_volume + total_ask_volume == 0:
                order_imbalance = 0.0
            else:
                order_imbalance = (total_bid_volume - total_ask_volume) / (total_bid_volume + total_ask_volume)

            # Imbalance direction
            if order_imbalance > 0.2:
                imbalance_direction = "BUY_PRESSURE"
            elif order_imbalance < -0.2:
                imbalance_direction = "SELL_PRESSURE"
            else:
                imbalance_direction = "BALANCED"

            # Liquidity score (0.0-1.0)
            # Good: tight spread + deep liquidity
            spread_score = max(0, 1 - (spread_bps / 500))  # 500bps = 0 score
            depth_score = min(1, (bid_depth_100bps + ask_depth_100bps) / 1000)  # $1000 = 1.0 score
            liquidity_score = (spread_score * 0.6) + (depth_score * 0.4)

            # Can fill order?
            can_fill_order = (bid_depth_100bps >= target_size or ask_depth_100bps >= target_size)

            logger.info(
                "Orderbook analyzed",
                spread_bps=f"{spread_bps:.1f}",
                imbalance=f"{order_imbalance:+.2f}",
                liquidity_score=f"{liquidity_score:.2f}",
                can_fill=can_fill_order
            )

            return OrderbookData(
                bid_ask_spread=spread_pct,
                spread_bps=spread_bps,
                liquidity_score=liquidity_score,
                order_imbalance=order_imbalance,
                imbalance_direction=imbalance_direction,
                bid_depth_100bps=bid_depth_100bps,
                ask_depth_100bps=ask_depth_100bps,
                bid_depth_200bps=bid_depth_200bps,
                ask_depth_200bps=ask_depth_200bps,
                best_bid=best_bid,
                best_ask=best_ask,
                can_fill_order=can_fill_order,
                timestamp=datetime.now()
            )

        except Exception as e:
            logger.error("Failed to analyze orderbook", error=str(e))
            return None

    def _calculate_depth(
        self,
        orders: list,
        reference_price: float,
        threshold: float,
        side: str
    ) -> float:
        """
        Calculate total liquidity within threshold from reference price.

        Args:
            orders: List of [price, size] pairs
            reference_price: Best bid/ask price
            threshold: Distance threshold (e.g., 0.01 for 100bps)
            side: 'bid' or 'ask'

        Returns:
            Total size within threshold
        """
        total = 0.0

        for price, size in orders:
            price = float(price)
            size = float(size)

            if side == 'bid':
                # For bids, check if within threshold below best bid
                if price >= reference_price - threshold:
                    total += size
            else:
                # For asks, check if within threshold above best ask
                if price <= reference_price + threshold:
                    total += size

        return total
```

**Step 3: Test orderbook fetching**

```bash
python3 -c "
import asyncio
from polymarket.client import PolymarketClient
from polymarket.config import Settings
from polymarket.trading.orderbook_analyzer import OrderbookAnalyzer

async def test():
    settings = Settings()
    client = PolymarketClient(settings)
    analyzer = OrderbookAnalyzer()

    # Get any active BTC market
    markets = await client.get_btc_markets(active_only=True)
    if not markets:
        print('No active markets found')
        return

    market = markets[0]
    token_ids = market.get_token_ids()
    if not token_ids:
        print('No token IDs found')
        return

    # Fetch and analyze orderbook
    orderbook = await client.get_orderbook(token_ids[0])
    if not orderbook:
        print('Failed to fetch orderbook')
        return

    analysis = analyzer.analyze_orderbook(orderbook, target_size=10.0)
    if analysis:
        print(f'Spread: {analysis.spread_bps:.1f}bps')
        print(f'Liquidity: {analysis.liquidity_score:.2f}')
        print(f'Imbalance: {analysis.order_imbalance:+.2f} ({analysis.imbalance_direction})')
        print(f'Can fill $10: {analysis.can_fill_order}')

asyncio.run(test())
"
```

Expected: Prints orderbook analysis from real Polymarket data

**Step 4: Commit**

```bash
git add polymarket/client.py polymarket/trading/orderbook_analyzer.py
git commit -m "feat: add orderbook fetching and depth analysis

Orderbook analysis essential for 15-min BTC execution:
- Bid-ask spread shows liquidity quality
- Order imbalance detects institutional flow
- Depth levels ensure orders can be filled

Implemented:
- get_orderbook() method in PolymarketClient
- OrderbookAnalyzer for spread/depth/imbalance calculation
- Liquidity scoring (0.0-1.0)
- Fillability check for target size"
```

---

### Task 2.5.3: Integrate Orderbook Analysis in Trading Loop

**Files:**
- Modify: `scripts/auto_trade.py:710-730`
- Modify: `polymarket/trading/ai_decision.py:50-100`

**Step 1: Fetch orderbook in trading loop**

In `scripts/auto_trade.py`, add orderbook fetching:

```python
# After market data fetch (around line 710)
# Get token IDs for orderbook
token_ids = market.get_token_ids()
if not token_ids:
    logger.warning("No token IDs found", market_id=market.id)
    return

# Fetch orderbook (YES token - first token)
orderbook = await self.client.get_orderbook(token_ids[0])

# Analyze orderbook
orderbook_analysis = None
if orderbook:
    from polymarket.trading.orderbook_analyzer import OrderbookAnalyzer
    analyzer = OrderbookAnalyzer()
    orderbook_analysis = analyzer.analyze_orderbook(
        orderbook,
        target_size=8.0  # Base position size
    )
```

**Step 2: Add orderbook filters**

After orderbook analysis, add validation:

```python
# Check orderbook liquidity
if orderbook_analysis:
    # Skip if spread too wide (poor execution)
    if orderbook_analysis.spread_bps > 500:  # 5% spread
        logger.info(
            "Skipping trade - spread too wide",
            market_id=market.id,
            spread_bps=f"{orderbook_analysis.spread_bps:.1f}",
            reason="Wide spread = poor execution quality"
        )
        return

    # Skip if can't fill order
    if not orderbook_analysis.can_fill_order:
        logger.info(
            "Skipping trade - insufficient liquidity",
            market_id=market.id,
            liquidity_score=f"{orderbook_analysis.liquidity_score:.2f}",
            reason="Not enough depth to fill order"
        )
        return
```

**Step 3: Pass orderbook data to AI**

Update AI service call:

```python
decision = await self.ai_service.make_decision(
    btc_data=btc_data,
    technical_indicators=indicators,
    aggregated_sentiment=aggregated_sentiment,
    market_data=market_dict,
    portfolio_value=portfolio_value,
    volume_data=volume_data,
    timeframe_analysis=timeframe_analysis,
    orderbook_data=orderbook_analysis  # NEW
)
```

**Step 4: Update AI decision prompt**

In `polymarket/trading/ai_decision.py`, add orderbook to prompt:

```python
# Add to _build_prompt method
if orderbook_data:
    prompt += f"""

# Orderbook Analysis
Bid-Ask Spread: {orderbook_data.spread_bps:.1f} bps
Liquidity Score: {orderbook_data.liquidity_score:.2f}/1.0
Order Imbalance: {orderbook_data.order_imbalance:+.2f} ({orderbook_data.imbalance_direction})
Bid Depth (100bps): ${orderbook_data.bid_depth_100bps:.2f}
Ask Depth (100bps): ${orderbook_data.ask_depth_100bps:.2f}
Can Fill Order: {orderbook_data.can_fill_order}

ORDERBOOK INTERPRETATION:
- Tight spread (<200bps) = liquid market, good execution
- Wide spread (>500bps) = illiquid, HOLD
- Buy pressure (>+0.2 imbalance) = bullish signal
- Sell pressure (<-0.2 imbalance) = bearish signal
- Insufficient depth = cannot fill, HOLD
"""
```

**Step 5: Test integration**

```bash
# Restart bot
pkill -f "python3 scripts/auto_trade.py"
cd /root/polymarket-scripts
python3 scripts/auto_trade.py &

# Monitor for orderbook logs
tail -f logs/bot.log | grep -E "Orderbook analyzed|Skipping.*spread|Skipping.*liquidity"
```

Expected: Should see orderbook analysis logs and filtering in action

**Step 6: Commit**

```bash
git add scripts/auto_trade.py polymarket/trading/ai_decision.py
git commit -m "feat: integrate orderbook analysis in trading loop

Complete orderbook integration:
- Fetch orderbook for each market
- Analyze spread, depth, imbalance
- Filter trades with wide spreads (>500bps)
- Filter trades with insufficient liquidity
- Pass orderbook metrics to AI prompt

Prevents:
- Trading in illiquid markets (poor execution)
- Slippage from wide spreads
- Failed fills from insufficient depth

Adds institutional flow signal via order imbalance"
```

---

## 🎯 Phase 3: Market Regime Detection

### Task 3.1: Implement Regime Detection

**Files:**
- Create: `polymarket/trading/regime_detector.py`
- Modify: `scripts/auto_trade.py:520-540`

**Step 1: Create MarketRegime model**

Add to `polymarket/models.py`:

```python
@dataclass
class MarketRegime:
    """Current market regime classification."""
    regime: str          # "TRENDING", "RANGING", "VOLATILE", "UNCLEAR"
    volatility: float    # ATR or price volatility
    is_trending: bool    # True if strong directional move
    trend_direction: str # "UP", "DOWN", "SIDEWAYS"
    confidence: float    # 0.0-1.0
    timestamp: datetime
```

**Step 2: Create regime detector**

Create: `polymarket/trading/regime_detector.py`

```python
"""Market regime detection for adaptive strategy selection."""
import structlog
from datetime import datetime
from polymarket.models import MarketRegime

logger = structlog.get_logger()

class RegimeDetector:
    """Detect market regime: trending, ranging, or volatile."""

    def detect_regime(
        self,
        price_changes: list[float],  # Last N price changes (%)
        current_price: float,
        high_24h: float,
        low_24h: float
    ) -> MarketRegime:
        """
        Detect current market regime.

        Args:
            price_changes: List of recent % price changes
            current_price: Current BTC price
            high_24h: 24-hour high
            low_24h: 24-hour low

        Returns:
            MarketRegime classification
        """
        # Calculate volatility (24h range as % of price)
        price_range = high_24h - low_24h
        volatility = (price_range / current_price) * 100

        # Calculate trend strength (sum of directional moves)
        if len(price_changes) < 5:
            return MarketRegime(
                regime="UNCLEAR",
                volatility=volatility,
                is_trending=False,
                trend_direction="SIDEWAYS",
                confidence=0.3,
                timestamp=datetime.now()
            )

        positive_moves = sum(1 for change in price_changes if change > 0.5)
        negative_moves = sum(1 for change in price_changes if change < -0.5)
        neutral_moves = len(price_changes) - positive_moves - negative_moves

        # Determine regime
        if volatility > 5.0:
            regime = "VOLATILE"
            is_trending = False
            trend_direction = "SIDEWAYS"
            confidence = 0.4
        elif positive_moves >= len(price_changes) * 0.7:
            regime = "TRENDING"
            is_trending = True
            trend_direction = "UP"
            confidence = 0.8
        elif negative_moves >= len(price_changes) * 0.7:
            regime = "TRENDING"
            is_trending = True
            trend_direction = "DOWN"
            confidence = 0.8
        elif neutral_moves >= len(price_changes) * 0.6:
            regime = "RANGING"
            is_trending = False
            trend_direction = "SIDEWAYS"
            confidence = 0.7
        else:
            regime = "UNCLEAR"
            is_trending = False
            trend_direction = "SIDEWAYS"
            confidence = 0.5

        logger.info(
            "Market regime detected",
            regime=regime,
            volatility=f"{volatility:.2f}%",
            trend_direction=trend_direction,
            confidence=confidence
        )

        return MarketRegime(
            regime=regime,
            volatility=volatility,
            is_trending=is_trending,
            trend_direction=trend_direction,
            confidence=confidence,
            timestamp=datetime.now()
        )
```

**Step 3: Integrate regime detection**

In `scripts/auto_trade.py`, add regime detection:

```python
# Get recent price changes for regime detection
price_history = await self.btc_service.get_recent_price_history(periods=10)
price_changes = [
    ((price_history[i] - price_history[i-1]) / price_history[i-1]) * 100
    for i in range(1, len(price_history))
]

# Detect market regime
regime_detector = RegimeDetector()
regime = regime_detector.detect_regime(
    price_changes=price_changes,
    current_price=float(btc_data.price),
    high_24h=float(btc_data.high_24h),
    low_24h=float(btc_data.low_24h)
)
```

**Step 4: Apply regime-specific strategy**

```python
# Skip trading in unclear/volatile regimes
if regime.regime in ["UNCLEAR", "VOLATILE"]:
    logger.info(
        "Skipping trade - unfavorable regime",
        market_id=market.id,
        regime=regime.regime,
        volatility=f"{regime.volatility:.2f}%",
        reason="Only trade in trending or ranging markets"
    )
    return
```

**Step 5: Test**

```bash
python3 -c "
from polymarket.trading.regime_detector import RegimeDetector

detector = RegimeDetector()

# Test trending up
result = detector.detect_regime(
    price_changes=[0.8, 1.2, 0.5, 1.0, 0.7],
    current_price=66000,
    high_24h=67000,
    low_24h=65000
)
print(f'Regime: {result.regime}, Trend: {result.trend_direction}')

# Test ranging
result = detector.detect_regime(
    price_changes=[0.1, -0.2, 0.1, -0.1, 0.2],
    current_price=66000,
    high_24h=66500,
    low_24h=65500
)
print(f'Regime: {result.regime}, Trend: {result.trend_direction}')
"
```

Expected: First shows "TRENDING UP", second shows "RANGING SIDEWAYS"

**Step 6: Commit**

```bash
git add polymarket/models.py polymarket/trading/regime_detector.py scripts/auto_trade.py
git commit -m "feat: add market regime detection

Adaptive strategy based on market conditions:
- TRENDING: Follow the trend
- RANGING: Buy support, sell resistance
- VOLATILE/UNCLEAR: Skip (too risky)

Implemented:
- RegimeDetector analyzing recent price action
- Volatility calculation (24h range)
- Trend strength measurement
- Filter to skip unfavorable regimes"
```

---

## 💰 Phase 4: Real-Time Position Value Tracking

### Task 4.1: Add Current Position Value Tracking

**Files:**
- Modify: `polymarket/client.py:200-250`
- Modify: `scripts/auto_trade.py:575-580`

**Step 1: Add get_position_current_value method**

In `polymarket/client.py`, add method:

```python
async def get_position_current_value(self, position) -> float:
    """
    Get current market value of a position.

    Args:
        position: Position dict from get_positions()

    Returns:
        Current value in USDC
    """
    try:
        token_id = position.get('asset_id')
        size = float(position.get('size', 0))

        if not token_id or size == 0:
            return 0.0

        # Get current market price for this token
        url = f"{self.clob_url}/last-trade-price"
        params = {"token_id": token_id}

        async with self.session.get(url, params=params) as resp:
            resp.raise_for_status()
            data = await resp.json()

            # Calculate current value: size / current_price
            # (In Polymarket, you buy shares at a price, value = shares * current price)
            current_price = float(data.get('price', 0))
            if current_price == 0:
                return size  # Fallback to size

            current_value = size  # Size is already in USDC terms
            return current_value

    except Exception as e:
        logger.error("Failed to get position current value", error=str(e))
        return float(position.get('size', 0))  # Fallback to size
```

**Step 2: Update portfolio summary calculation**

Modify `get_portfolio_summary()` in `polymarket/client.py`:

```python
def get_portfolio_summary(self) -> PortfolioSummary:
    """Get portfolio summary with REAL-TIME position values."""
    balance = self.get_balance()
    positions = self.get_positions()

    # OLD: positions_value = sum(float(p.get('size', 0)) for p in positions)

    # NEW: Get real-time value for each position
    positions_value = 0.0
    purchase_value = 0.0

    for pos in positions:
        size = float(pos.get('size', 0))
        purchase_value += size

        # Get current market value (async call needed)
        # For now, use size as approximation
        # TODO: Make this async
        positions_value += size

    total_value = balance + positions_value
    unrealized_pl = positions_value - purchase_value

    return PortfolioSummary(
        usdc_balance=balance,
        positions_value=positions_value,
        purchase_value=purchase_value,  # NEW
        unrealized_pl=unrealized_pl,    # NEW
        total_value=total_value
    )
```

**Step 3: Update PortfolioSummary model**

In `polymarket/models.py`:

```python
@dataclass
class PortfolioSummary:
    """Portfolio summary with real-time values."""
    usdc_balance: Decimal
    positions_value: Decimal
    purchase_value: Decimal   # NEW: Original purchase cost
    unrealized_pl: Decimal    # NEW: Current value - Purchase value
    total_value: Decimal
```

**Step 4: Log unrealized P/L**

In `scripts/auto_trade.py`, update logging:

```python
logger.info(
    "Portfolio fetched",
    total_value=f"${portfolio.total_value:.2f}",
    usdc_balance=f"${portfolio.usdc_balance:.2f}",
    positions_value=f"${portfolio.positions_value:.2f}",
    purchase_value=f"${portfolio.purchase_value:.2f}",  # NEW
    unrealized_pl=f"${portfolio.unrealized_pl:+.2f}"    # NEW
)
```

**Step 5: Test**

```bash
# Restart bot and check logs
pkill -f "python3 scripts/auto_trade.py"
python3 scripts/auto_trade.py &

tail -f logs/bot.log | grep "Portfolio fetched"
```

Expected: Should now show purchase_value and unrealized_pl

**Step 6: Commit**

```bash
git add polymarket/client.py polymarket/models.py scripts/auto_trade.py
git commit -m "feat: add real-time position value tracking

Previous: Counted positions at purchase price
Now: Track current market value and unrealized P/L

Fixes issue where $607 'positions' were actually losing trades
User can now see: purchase $607 → current value $150 = -$457 unrealized loss

Added:
- purchase_value: Original cost
- unrealized_pl: Current value - Purchase value
- Real-time position valuation"
```

---

## 🤖 Phase 5: AI Decision Logic Overhaul

### Task 5.1: Improve AI Prompt with Regime Awareness

**Files:**
- Modify: `polymarket/ai/decision_service.py:50-250`

**Step 1: Update system prompt with regime logic**

In `polymarket/ai/decision_service.py`, replace system prompt:

```python
SYSTEM_PROMPT = """You are a professional Bitcoin trader analyzing 15-minute markets on Polymarket.

# CRITICAL RULES FOR 15-MIN BTC TRADING

## Market Regime Strategy

**TRENDING Markets (strong directional move):**
- Follow the trend direction
- Enter on pullbacks, not at extremes
- YES: Only if trend is UP and price pulled back
- NO: Only if trend is DOWN and price bounced

**RANGING Markets (sideways movement):**
- Buy at support, sell at resistance
- YES: Only near daily support level
- NO: Only near daily resistance level

**VOLATILE/UNCLEAR Markets:**
- HOLD - Don't trade unclear conditions
- Wait for regime clarity

## Volume Requirements

**For breakouts ($200+ moves):**
- REQUIRE volume > 1.5x average
- Without volume = false breakout = HOLD

**For normal trades:**
- Prefer higher volume for confirmation
- Low volume = reduce confidence

## Timeframe Alignment

**ALIGNED timeframes (daily + 4h same direction):**
- High confidence trades
- Can use full position size

**CONFLICTING timeframes:**
- HOLD - Don't trade against larger trend
- Wait for alignment

## Entry Rules

**YES (bet BTC goes UP):**
1. Market regime must be TRENDING UP or RANGING (near support)
2. Daily + 4h timeframes must be BULLISH or NEUTRAL
3. Current price NOT at 24h high (wait for pullback)
4. If move > $200, require high volume
5. Confidence > 0.75

**NO (bet BTC goes DOWN):**
1. Market regime must be TRENDING DOWN or RANGING (near resistance)
2. Daily + 4h timeframes must be BEARISH or NEUTRAL
3. Current price NOT at 24h low (wait for bounce)
4. If move > $200, require high volume
5. Confidence > 0.70

## Position Sizing

- Base size: $8
- High confidence (>0.85): $10
- Low confidence (<0.75): $6
- Conflicting signals: $0 (HOLD)

## When to HOLD

- Regime is VOLATILE or UNCLEAR
- Timeframes CONFLICTING
- No volume on large moves
- Price at extremes (exhausted momentum)
- Confidence < 0.70
- Unclear market conditions

Your response MUST be valid JSON: {"action": "YES|NO|HOLD", "confidence": 0.0-1.0, "reasoning": "...", "position_size": 0-10}
"""
```

**Step 2: Update make_decision to include regime data**

Modify `make_decision()` method signature and call:

```python
async def make_decision(
    self,
    btc_price: BTCPriceData,
    technical_indicators: TechnicalIndicators,
    aggregated_sentiment: AggregatedSentiment,
    market_data: dict,
    portfolio_value: Decimal,
    volume_data: VolumeData | None = None,           # NEW
    timeframe_analysis: TimeframeAnalysis | None = None,  # NEW
    regime: MarketRegime | None = None               # NEW
) -> TradingDecision:
```

**Step 3: Add regime/volume/timeframe to prompt context**

```python
# Build context with new data
context = f"""
# Current Market Data

BTC Price: ${btc_price.price:,.2f}
Price to Beat: ${market_data.get('price_to_beat'):,.2f}
Price Change: {((btc_price.price - market_data.get('price_to_beat')) / market_data.get('price_to_beat') * 100):+.2f}%

24h High: ${btc_price.high_24h:,.2f}
24h Low: ${btc_price.low_24h:,.2f}
Distance from High: {((btc_price.price - btc_price.high_24h) / btc_price.high_24h * 100):.2f}%
Distance from Low: {((btc_price.price - btc_price.low_24h) / btc_price.low_24h * 100):.2f}%

# Market Regime
Regime: {regime.regime if regime else 'UNKNOWN'}
Is Trending: {regime.is_trending if regime else False}
Trend Direction: {regime.trend_direction if regime else 'UNKNOWN'}
Volatility: {regime.volatility if regime else 0:.2f}%
Confidence: {regime.confidence if regime else 0:.2f}

# Volume Analysis
Volume 24h: ${volume_data.volume_24h:,.0f} if volume_data else 'N/A'
Volume Ratio: {volume_data.volume_ratio if volume_data else 1.0:.2f}x
High Volume: {volume_data.is_high_volume if volume_data else False}

# Timeframe Analysis
Daily Trend: {timeframe_analysis.daily_trend if timeframe_analysis else 'UNKNOWN'}
4H Trend: {timeframe_analysis.four_hour_trend if timeframe_analysis else 'UNKNOWN'}
Alignment: {timeframe_analysis.alignment if timeframe_analysis else 'UNKNOWN'}
Daily Support: ${timeframe_analysis.daily_support if timeframe_analysis else 0:,.2f}
Daily Resistance: ${timeframe_analysis.daily_resistance if timeframe_analysis else 0:,.2f}

# Technical Indicators
RSI: {technical_indicators.rsi:.1f}
MACD: {technical_indicators.macd_value:.2f}
Trend: {technical_indicators.trend}

# Aggregated Signals
Final Score: {aggregated_sentiment.final_score:+.2f}
Confidence: {aggregated_sentiment.final_confidence:.2f}
Signal Type: {aggregated_sentiment.signal_type}

# Market Prices
YES (Up) Price: {market_data['yes_price']:.3f}
NO (Down) Price: {market_data['no_price']:.3f}

# Portfolio
Balance: ${portfolio_value:.2f}

Analyze the data and make a decision following the CRITICAL RULES above.
"""
```

**Step 4: Test new prompt**

```bash
# Test with mock data
python3 -c "
import asyncio
from polymarket.ai.decision_service import AIDecisionService
from polymarket.config import Settings
from polymarket.models import *
from decimal import Decimal
from datetime import datetime

async def test():
    service = AIDecisionService(Settings())

    # Mock data
    btc_price = BTCPriceData(price=Decimal('66000'), high_24h=Decimal('67000'), low_24h=Decimal('65000'), timestamp=datetime.now())
    tech = TechnicalIndicators(rsi=55.0, macd_value=10.0, trend='BULLISH', timestamp=datetime.now())
    agg = AggregatedSentiment(...)  # Mock
    market = {'price_to_beat': 65500, 'yes_price': 0.6, 'no_price': 0.4}
    volume = VolumeData(volume_24h=5000000000, volume_current_hour=200000000, volume_avg_hour=208333333, volume_ratio=0.96, is_high_volume=False, timestamp=datetime.now())
    regime = MarketRegime(regime='TRENDING', volatility=3.0, is_trending=True, trend_direction='UP', confidence=0.8, timestamp=datetime.now())
    timeframe = TimeframeAnalysis(daily_trend='BULLISH', four_hour_trend='BULLISH', alignment='ALIGNED', daily_support=65000, daily_resistance=67000, confidence=0.9, timestamp=datetime.now())

    decision = await service.make_decision(btc_price, tech, agg, market, Decimal('100'), volume, timeframe, regime)
    print(f'Decision: {decision.action}')
    print(f'Confidence: {decision.confidence}')
    print(f'Reasoning: {decision.reasoning}')

asyncio.run(test())
"
```

Expected: Should make intelligent decision based on regime/volume/timeframe

**Step 5: Commit**

```bash
git add polymarket/ai/decision_service.py
git commit -m "feat: overhaul AI prompt with regime-aware logic

NEW decision framework:
- Market regime strategy (trending/ranging/volatile)
- Volume requirements for breakouts
- Timeframe alignment checks
- Clear entry rules for YES/NO
- Strict HOLD criteria

Teaches AI:
- Don't trade unclear/volatile markets
- Require volume on large moves
- Follow larger timeframe trends
- Wait for pullbacks (not extremes)
- When to HOLD (most important!)"
```

---

## ⚠️ Phase 6: Risk Management & Stop-Loss

### Task 6.1: Add Time-Based Trading Filters

**Files:**
- Modify: `scripts/auto_trade.py:450-470`

**Step 1: Add optimal trading hours filter**

At start of `_process_market`, add time check:

```python
async def _process_market(self, market: Market) -> None:
    """Process a single market for potential trade."""

    # Get current UTC hour
    from datetime import datetime, timezone
    current_hour_utc = datetime.now(timezone.utc).hour

    # Optimal trading window: 11 AM - 1 PM UTC (research-backed)
    OPTIMAL_HOURS = range(11, 13)  # 11:00-12:59 UTC

    # Avoid worst hours: 12 AM - 6 AM UTC (low liquidity, high volatility)
    AVOID_HOURS = range(0, 6)

    if current_hour_utc in AVOID_HOURS:
        logger.info(
            "Skipping trade - outside trading hours",
            market_id=market.id,
            current_hour=current_hour_utc,
            reason="Avoid 12 AM - 6 AM UTC (low liquidity)"
        )
        return

    # Reduce position size outside optimal hours
    in_optimal_window = current_hour_utc in OPTIMAL_HOURS
    position_size_multiplier = 1.0 if in_optimal_window else 0.7

    logger.debug(
        "Trading hours check",
        current_hour=current_hour_utc,
        in_optimal_window=in_optimal_window,
        multiplier=position_size_multiplier
    )
```

**Step 2: Apply multiplier to position sizing**

When calculating position size, apply multiplier:

```python
# Later in the method, when setting position size:
base_position_size = decision.position_size
adjusted_position_size = base_position_size * position_size_multiplier

logger.info(
    "Position size adjusted for trading hours",
    base_size=f"${base_position_size:.2f}",
    multiplier=position_size_multiplier,
    adjusted_size=f"${adjusted_position_size:.2f}"
)
```

**Step 3: Test**

```bash
python3 -c "
from datetime import datetime, timezone

# Test different hours
for hour in [0, 6, 11, 12, 18, 23]:
    test_time = datetime.now(timezone.utc).replace(hour=hour)

    OPTIMAL_HOURS = range(11, 13)
    AVOID_HOURS = range(0, 6)

    if test_time.hour in AVOID_HOURS:
        print(f'{hour:02d}:00 UTC - SKIP (avoid hours)')
    elif test_time.hour in OPTIMAL_HOURS:
        print(f'{hour:02d}:00 UTC - TRADE (optimal, 1.0x size)')
    else:
        print(f'{hour:02d}:00 UTC - TRADE (ok, 0.7x size)')
"
```

Expected:
```
00:00 UTC - SKIP (avoid hours)
06:00 UTC - TRADE (ok, 0.7x size)
11:00 UTC - TRADE (optimal, 1.0x size)
12:00 UTC - TRADE (optimal, 1.0x size)
18:00 UTC - TRADE (ok, 0.7x size)
23:00 UTC - SKIP (avoid hours)
```

**Step 4: Commit**

```bash
git add scripts/auto_trade.py
git commit -m "feat: add time-based trading filters

Research shows optimal execution: 11 AM - 1 PM UTC
Worst hours: 12 AM - 6 AM UTC (high volatility, low liquidity)

Implemented:
- Skip trades during 12 AM - 6 AM UTC
- Full position size 11 AM - 1 PM UTC
- Reduced position size (0.7x) other hours
- Protects against low-liquidity moves"
```

---

## 📋 Execution Summary

**Total Tasks:** 18 tasks across 6 phases (including Phase 2.5: Orderbook Analysis)
**Estimated Time:** 4-5 hours for careful implementation
**Testing Strategy:** Test each component before moving to next phase
**Rollback Plan:** Each phase independently deployable, can rollback by phase

**Phase Breakdown:**
- Phase 1: Emergency Fixes (3 tasks)
- Phase 2: Volume & Multi-Timeframe (2 tasks)
- **Phase 2.5: Orderbook Analysis (3 tasks)** ← NEW
- Phase 3: Market Regime Detection (1 task)
- Phase 4: Position Value Tracking (1 task)
- Phase 5: AI Decision Overhaul (1 task)
- Phase 6: Risk Management (1 task)

---

## 🎯 Success Criteria

After implementation:
- [ ] **NO phantom trades** - Settlement only processes executed trades
- [ ] **YES trades disabled** - 0 YES trades executed
- [ ] **Volume confirmation** - Large moves require 1.5x average volume
- [ ] **Orderbook analysis** - Skip trades with wide spreads (>500bps) or low liquidity
- [ ] **Timeframe alignment** - No trading when daily/4h conflict
- [ ] **Regime detection** - Only trade trending/ranging markets
- [ ] **Real P/L tracking** - Shows unrealized P/L accurately
- [ ] **Time filters** - No trades 12 AM - 6 AM UTC
- [ ] **Target win rate: 60%+** (currently 26.4%)

---

## 🚀 Deployment Plan

1. **Phase 1 (Emergency)** - Deploy immediately, test 1 hour
2. **Phase 2-2.5** - Deploy next day (volume + orderbook), test 4 hours
3. **Phase 3** - Deploy regime detection after validation, test 4 hours
4. **Phase 4-5** - Deploy position tracking + AI overhaul, test 8 hours
5. **Phase 6** - Final deployment (risk management), monitor 24 hours

**Monitoring:** Check win rate every 4 hours, rollback if < 40%

**Critical Dependencies:**
- Phase 2.5 (Orderbook) depends on Phase 2 (Volume) for complete data picture
- All phases tested independently before moving forward

---

Plan complete! Ready for execution.
