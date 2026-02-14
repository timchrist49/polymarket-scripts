# Test Mode Comprehensive Overhaul - Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix test mode execution blocker (0% fill rate), add multi-timeframe analysis, implement metrics tracking

**Architecture:** Four-phase implementation: (1) Fix execution with min bet enforcement and edge filtering, (2) Add multi-timeframe analyzer for 15m/1H/4H trends, (3) Add metrics tracking with Telegram reports every 20 trades, (4) Enhance AI prompt with timeframe context

**Tech Stack:** Python 3.11, asyncio, Decimal, dataclasses, SQLite, Telegram Bot API

---

## Phase 1: Core Execution Fixes (Priority 0)

### Task 1: Update TestModeConfig with New Parameters

**Files:**
- Modify: `scripts/auto_trade.py:105-110`

**Step 1: Update TestModeConfig dataclass**

Replace lines 105-110:

```python
@dataclass
class TestModeConfig:
    """Configuration for test mode trading."""
    enabled: bool = False
    max_bet_amount: Decimal = Decimal("10.0")  # Allow Kelly sizing room
    min_bet_amount: Decimal = Decimal("5.0")   # Enforce Polymarket minimum
    min_arbitrage_edge: float = 0.02           # Require 2% edge minimum
    min_confidence: float = 0.70
    traded_markets: set[str] = field(default_factory=set)
```

**Step 2: Update TestModeConfig initialization (lines 173-178)**

Replace:

```python
# Initialize test mode
self.test_mode = TestModeConfig(
    enabled=os.getenv("TEST_MODE", "").lower() == "true",
    max_bet_amount=Decimal("10.0"),
    min_bet_amount=Decimal("5.0"),
    min_arbitrage_edge=0.02,
    min_confidence=0.70,
    traded_markets=set()
)
```

**Step 3: Update test mode banner log (lines 187-189)**

Replace:

```python
logger.warning(
    "[TEST] Trading with min $5, max $10 bets, 70% min confidence, 2% min edge",
    max_bet=str(self.test_mode.max_bet_amount),
    min_bet=str(self.test_mode.min_bet_amount),
    min_confidence=self.test_mode.min_confidence,
    min_edge=f"{self.test_mode.min_arbitrage_edge:.1%}"
)
```

**Step 4: Verify changes**

```bash
grep -A8 "class TestModeConfig" scripts/auto_trade.py
```

Expected: See new fields `min_bet_amount` and `min_arbitrage_edge`

**Step 5: Commit**

```bash
git add scripts/auto_trade.py
git commit -m "feat: add min_bet_amount and min_arbitrage_edge to test mode config

- Set min_bet_amount = $5 (Polymarket minimum)
- Set max_bet_amount = $10 (allow Kelly sizing room)
- Set min_arbitrage_edge = 2% (filter noise trades)
- Update initialization and logging

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 2: Add Edge Validation Before Order Execution

**Files:**
- Modify: `scripts/auto_trade.py` (find edge validation location)

**Step 1: Find order execution section**

```bash
grep -n "Executing smart order" scripts/auto_trade.py
```

Expected: Shows line number where orders are executed

**Step 2: Add edge validation before execution**

Add BEFORE order execution (around line ~1350):

```python
# Test mode: Validate minimum arbitrage edge
if self.test_mode.enabled:
    arb_edge = getattr(decision, 'arbitrage_edge', 0.0)
    if arb_edge < self.test_mode.min_arbitrage_edge:
        logger.info(
            "[TEST] Skipping trade - arbitrage edge below minimum",
            market_id=market.id,
            edge=f"{arb_edge:.2%}",
            minimum=f"{self.test_mode.min_arbitrage_edge:.2%}",
            reason="Edge too small - likely noise trade"
        )
        return
```

**Step 3: Verify placement**

```bash
grep -B3 -A7 "Skipping trade - arbitrage edge" scripts/auto_trade.py
```

Expected: See validation before order execution

**Step 4: Commit**

```bash
git add scripts/auto_trade.py
git commit -m "feat: add minimum arbitrage edge validation in test mode

- Skip trades with edge < 2% in test mode
- Filters noise trades (0.1-0.3% edges)
- Logs skip reason for visibility

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 3: Enforce Minimum Bet Size in Position Sizing

**Files:**
- Modify: `scripts/auto_trade.py` (position sizing section around line 1140)

**Step 1: Find position sizing override section**

```bash
grep -n "Overriding position size" scripts/auto_trade.py
```

Expected: Shows line number of test mode position override

**Step 2: Add minimum bet enforcement**

Replace the position sizing logic:

```python
if self.test_mode.enabled:
    # Calculate Kelly-suggested size first
    kelly_size = decision.position_size

    # Enforce minimum bet (Polymarket requires $5 minimum)
    final_size = max(kelly_size, self.test_mode.min_bet_amount)

    # Enforce maximum bet
    final_size = min(final_size, self.test_mode.max_bet_amount)

    logger.info(
        "[TEST] Position sizing",
        market_id=market.id,
        kelly_suggested=f"${kelly_size:.2f}",
        final_amount=f"${final_size:.2f}",
        min_enforced=kelly_size < self.test_mode.min_bet_amount,
        max_enforced=kelly_size > self.test_mode.max_bet_amount
    )

    decision.position_size = final_size
```

**Step 3: Verify changes**

```bash
grep -A15 "Position sizing" scripts/auto_trade.py | grep -A10 "TEST"
```

Expected: See min/max enforcement logic

**Step 4: Commit**

```bash
git add scripts/auto_trade.py
git commit -m "feat: enforce min/max bet size in test mode position sizing

- Enforce min $5 (Polymarket minimum order size)
- Enforce max $10 (risk management)
- Apply Kelly sizing within constraints
- Log when min/max enforcement triggers

Fixes: 0% execution rate (orders were below $5 minimum)

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 4: Test Phase 1 Changes

**Step 1: Stop current bot**

```bash
pkill -f "python3 scripts/auto_trade.py"
```

**Step 2: Start bot with test mode**

```bash
cd /root/polymarket-scripts
TEST_MODE=true python3 scripts/auto_trade.py > /root/test-phase1.log 2>&1 &
```

**Step 3: Wait for first trade attempt (3-5 minutes)**

```bash
sleep 300
```

**Step 4: Check for successful order execution**

```bash
tail -100 /root/test-phase1.log | grep -E "Position sizing|arbitrage edge|Order|filled"
```

Expected:
- See `[TEST] Position sizing` with final_amount >= $5.00
- See `min_enforced=True` if Kelly size was < $5
- NO "Size (X) lower than the minimum: 5" errors
- Possible: Successful order fills

**Step 5: Document Phase 1 completion**

If orders execute successfully, Phase 1 is complete!

---

## Phase 2: Multi-Timeframe Analysis

### Task 5: Create TimeframeTrend and TimeframeAnalysis Data Structures

**Files:**
- Create: `polymarket/trading/timeframe_analyzer.py`

**Step 1: Create new file with data structures**

```python
"""Multi-timeframe trend analysis for improved trading decisions."""

from dataclasses import dataclass
from typing import Optional
from decimal import Decimal
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class TimeframeTrend:
    """Represents trend direction for a single timeframe."""

    timeframe: str  # "15m", "1h", "4h"
    direction: str  # "UP", "DOWN", "NEUTRAL"
    strength: float  # 0.0 to 1.0 (how strong the trend)
    price_change_pct: float  # Actual percentage change
    price_start: Decimal  # Starting price
    price_end: Decimal  # Ending price


@dataclass
class TimeframeAnalysis:
    """Complete multi-timeframe analysis result."""

    tf_15m: TimeframeTrend
    tf_1h: TimeframeTrend
    tf_4h: TimeframeTrend
    alignment_score: str  # "ALIGNED_BULLISH", "ALIGNED_BEARISH", "MIXED", "CONFLICTING"
    confidence_modifier: float  # +0.15, 0.0, or -0.15

    def __str__(self) -> str:
        return (
            f"15m: {self.tf_15m.direction} ({self.tf_15m.price_change_pct:+.2f}%), "
            f"1H: {self.tf_1h.direction} ({self.tf_1h.price_change_pct:+.2f}%), "
            f"4H: {self.tf_4h.direction} ({self.tf_4h.price_change_pct:+.2f}%) "
            f"| Alignment: {self.alignment_score} | Modifier: {self.confidence_modifier:+.2%}"
        )
```

**Step 2: Verify file created**

```bash
ls -lh polymarket/trading/timeframe_analyzer.py
```

Expected: File exists

**Step 3: Commit**

```bash
git add polymarket/trading/timeframe_analyzer.py
git commit -m "feat: add timeframe analysis data structures

- TimeframeTrend: Stores single timeframe trend data
- TimeframeAnalysis: Complete multi-timeframe result
- Supports 15m, 1H, 4H timeframes
- Includes alignment scoring and confidence modifier

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 6: Implement TimeframeAnalyzer Service

**Files:**
- Modify: `polymarket/trading/timeframe_analyzer.py`

**Step 1: Add TimeframeAnalyzer class**

Append to file:

```python
class TimeframeAnalyzer:
    """Analyzes BTC price trends across multiple timeframes."""

    def __init__(self, price_buffer):
        """Initialize analyzer with price history buffer.

        Args:
            price_buffer: PriceHistoryBuffer instance for historical lookback
        """
        self.price_buffer = price_buffer
        self.direction_threshold_pct = 0.5  # 0.5% move to be directional

    def _calculate_trend(
        self,
        timeframe: str,
        lookback_seconds: int
    ) -> Optional[TimeframeTrend]:
        """Calculate trend for a single timeframe.

        Args:
            timeframe: Human-readable name ("15m", "1h", "4h")
            lookback_seconds: How far back to look

        Returns:
            TimeframeTrend if data available, None otherwise
        """
        try:
            # Get price from lookback_seconds ago
            price_start = self.price_buffer.get_price_at(lookback_seconds)

            # Get current price (0 seconds ago)
            price_end = self.price_buffer.get_price_at(0)

            if not price_start or not price_end:
                logger.warning(
                    "Insufficient price data for timeframe",
                    timeframe=timeframe,
                    lookback_seconds=lookback_seconds
                )
                return None

            # Calculate percentage change
            price_change_pct = float(
                ((price_end - price_start) / price_start) * 100
            )

            # Determine direction
            if price_change_pct > self.direction_threshold_pct:
                direction = "UP"
                strength = min(abs(price_change_pct) / 2.0, 1.0)  # Cap at 1.0
            elif price_change_pct < -self.direction_threshold_pct:
                direction = "DOWN"
                strength = min(abs(price_change_pct) / 2.0, 1.0)
            else:
                direction = "NEUTRAL"
                strength = 0.0

            return TimeframeTrend(
                timeframe=timeframe,
                direction=direction,
                strength=strength,
                price_change_pct=price_change_pct,
                price_start=price_start,
                price_end=price_end
            )

        except Exception as e:
            logger.error(
                "Error calculating trend",
                timeframe=timeframe,
                error=str(e)
            )
            return None

    def _calculate_alignment(
        self,
        tf_15m: TimeframeTrend,
        tf_1h: TimeframeTrend,
        tf_4h: TimeframeTrend
    ) -> tuple[str, float]:
        """Calculate alignment score and confidence modifier.

        Returns:
            (alignment_score, confidence_modifier)
        """
        directions = [tf_15m.direction, tf_1h.direction, tf_4h.direction]

        # Count directional votes (ignore NEUTRAL)
        up_count = directions.count("UP")
        down_count = directions.count("DOWN")

        # All aligned in same direction
        if up_count == 3:
            return ("ALIGNED_BULLISH", 0.15)
        elif down_count == 3:
            return ("ALIGNED_BEARISH", 0.15)

        # 2 of 3 agree (mixed signals)
        elif up_count == 2 or down_count == 2:
            return ("MIXED", 0.0)

        # 15m contradicts both longer timeframes (conflicting)
        elif (tf_15m.direction == "UP" and tf_1h.direction == "DOWN" and tf_4h.direction == "DOWN") or \
             (tf_15m.direction == "DOWN" and tf_1h.direction == "UP" and tf_4h.direction == "UP"):
            return ("CONFLICTING", -0.15)

        # Default: Mixed signals
        return ("MIXED", 0.0)

    async def analyze(self) -> Optional[TimeframeAnalysis]:
        """Analyze trends across 15m, 1H, 4H timeframes.

        Returns:
            TimeframeAnalysis if sufficient data, None otherwise
        """
        # Calculate trends for each timeframe
        tf_15m = self._calculate_trend("15m", 15 * 60)  # 15 minutes
        tf_1h = self._calculate_trend("1h", 60 * 60)    # 1 hour
        tf_4h = self._calculate_trend("4h", 4 * 60 * 60)  # 4 hours

        # Require all timeframes to have data
        if not all([tf_15m, tf_1h, tf_4h]):
            logger.warning(
                "Skipping timeframe analysis - insufficient historical data",
                tf_15m=bool(tf_15m),
                tf_1h=bool(tf_1h),
                tf_4h=bool(tf_4h)
            )
            return None

        # Calculate alignment and confidence modifier
        alignment_score, confidence_modifier = self._calculate_alignment(
            tf_15m, tf_1h, tf_4h
        )

        analysis = TimeframeAnalysis(
            tf_15m=tf_15m,
            tf_1h=tf_1h,
            tf_4h=tf_4h,
            alignment_score=alignment_score,
            confidence_modifier=confidence_modifier
        )

        logger.info(
            "Timeframe analysis completed",
            analysis=str(analysis)
        )

        return analysis
```

**Step 2: Verify implementation**

```bash
grep -c "def " polymarket/trading/timeframe_analyzer.py
```

Expected: 4 functions (_calculate_trend, _calculate_alignment, analyze, __str__)

**Step 3: Commit**

```bash
git add polymarket/trading/timeframe_analyzer.py
git commit -m "feat: implement TimeframeAnalyzer service

- Calculate trends for 15m, 1H, 4H timeframes
- Determine trend direction (UP/DOWN/NEUTRAL)
- Calculate alignment score (ALIGNED/MIXED/CONFLICTING)
- Apply confidence modifiers (+15%, 0%, -15%)
- Uses existing PriceHistoryBuffer for data

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 7: Write Unit Tests for TimeframeAnalyzer

**Files:**
- Create: `tests/test_timeframe_analyzer.py`

**Step 1: Create test file**

```python
"""Unit tests for TimeframeAnalyzer."""

import pytest
from decimal import Decimal
from polymarket.trading.timeframe_analyzer import (
    TimeframeAnalyzer,
    TimeframeTrend,
    TimeframeAnalysis
)


class MockPriceBuffer:
    """Mock price buffer for testing."""

    def __init__(self, prices: dict[int, Decimal]):
        """Initialize with prices keyed by seconds ago.

        Args:
            prices: {seconds_ago: price}
        """
        self.prices = prices

    def get_price_at(self, seconds_ago: int) -> Decimal:
        """Return price from X seconds ago."""
        return self.prices.get(seconds_ago)


@pytest.mark.asyncio
async def test_aligned_bullish_trend():
    """Test all timeframes aligned bullish."""
    # Setup: All timeframes showing upward movement
    buffer = MockPriceBuffer({
        0: Decimal("98000"),      # Current price
        15 * 60: Decimal("97000"),    # 15m ago (-1%)
        60 * 60: Decimal("96000"),    # 1H ago (-2%)
        4 * 60 * 60: Decimal("94000")  # 4H ago (-4%)
    })

    analyzer = TimeframeAnalyzer(buffer)
    result = await analyzer.analyze()

    assert result is not None
    assert result.tf_15m.direction == "UP"
    assert result.tf_1h.direction == "UP"
    assert result.tf_4h.direction == "UP"
    assert result.alignment_score == "ALIGNED_BULLISH"
    assert result.confidence_modifier == 0.15


@pytest.mark.asyncio
async def test_aligned_bearish_trend():
    """Test all timeframes aligned bearish."""
    buffer = MockPriceBuffer({
        0: Decimal("94000"),      # Current price
        15 * 60: Decimal("95000"),    # 15m ago (+1%)
        60 * 60: Decimal("96000"),    # 1H ago (+2%)
        4 * 60 * 60: Decimal("98000")  # 4H ago (+4%)
    })

    analyzer = TimeframeAnalyzer(buffer)
    result = await analyzer.analyze()

    assert result is not None
    assert result.tf_15m.direction == "DOWN"
    assert result.tf_1h.direction == "DOWN"
    assert result.tf_4h.direction == "DOWN"
    assert result.alignment_score == "ALIGNED_BEARISH"
    assert result.confidence_modifier == 0.15


@pytest.mark.asyncio
async def test_mixed_signals():
    """Test mixed timeframe signals."""
    buffer = MockPriceBuffer({
        0: Decimal("97000"),      # Current
        15 * 60: Decimal("96000"),    # 15m ago: UP
        60 * 60: Decimal("98000"),    # 1H ago: DOWN
        4 * 60 * 60: Decimal("96500")  # 4H ago: UP
    })

    analyzer = TimeframeAnalyzer(buffer)
    result = await analyzer.analyze()

    assert result is not None
    assert result.alignment_score == "MIXED"
    assert result.confidence_modifier == 0.0


@pytest.mark.asyncio
async def test_conflicting_signals():
    """Test 15m contradicting longer timeframes."""
    buffer = MockPriceBuffer({
        0: Decimal("99000"),      # Current
        15 * 60: Decimal("97000"),    # 15m ago: UP (+2%)
        60 * 60: Decimal("98000"),    # 1H ago: DOWN (-1%)
        4 * 60 * 60: Decimal("100000")  # 4H ago: DOWN (-1%)
    })

    analyzer = TimeframeAnalyzer(buffer)
    result = await analyzer.analyze()

    assert result is not None
    assert result.tf_15m.direction == "UP"
    assert result.tf_1h.direction == "DOWN"
    assert result.tf_4h.direction == "DOWN"
    assert result.alignment_score == "CONFLICTING"
    assert result.confidence_modifier == -0.15


@pytest.mark.asyncio
async def test_insufficient_data():
    """Test graceful degradation with missing data."""
    buffer = MockPriceBuffer({
        0: Decimal("97000"),
        15 * 60: Decimal("96000"),
        # Missing 1H and 4H data
    })

    analyzer = TimeframeAnalyzer(buffer)
    result = await analyzer.analyze()

    assert result is None  # Should return None when data missing
```

**Step 2: Run tests**

```bash
cd /root/polymarket-scripts
pytest tests/test_timeframe_analyzer.py -v
```

Expected: All 5 tests PASS

**Step 3: Commit**

```bash
git add tests/test_timeframe_analyzer.py
git commit -m "test: add comprehensive unit tests for TimeframeAnalyzer

- Test aligned bullish scenario
- Test aligned bearish scenario
- Test mixed signals
- Test conflicting signals (15m vs longer)
- Test graceful degradation with missing data

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 8: Integrate TimeframeAnalyzer into AutoTrader

**Files:**
- Modify: `scripts/auto_trade.py`

**Step 1: Add import**

Add to imports section (around line 20-30):

```python
from polymarket.trading.timeframe_analyzer import TimeframeAnalyzer, TimeframeAnalysis
```

**Step 2: Initialize analyzer in __init__**

Add after line ~155 (after trade_settler initialization):

```python
# Multi-timeframe analysis
self.timeframe_analyzer = TimeframeAnalyzer(
    price_buffer=self.btc_service.price_stream.buffer
)
```

**Step 3: Run timeframe analysis in trading cycle**

Find the trading cycle (around line 950-1000) and add before AI decision:

```python
# Multi-timeframe trend analysis
timeframe_analysis = None
if self.test_mode.enabled:
    timeframe_analysis = await self.timeframe_analyzer.analyze()
    if timeframe_analysis:
        logger.info(
            "Multi-timeframe analysis",
            tf_15m=timeframe_analysis.tf_15m.direction,
            tf_1h=timeframe_analysis.tf_1h.direction,
            tf_4h=timeframe_analysis.tf_4h.direction,
            alignment=timeframe_analysis.alignment_score,
            modifier=f"{timeframe_analysis.confidence_modifier:+.1%}"
        )
```

**Step 4: Verify integration**

```bash
grep -A5 "Multi-timeframe analysis" scripts/auto_trade.py
```

Expected: See analyzer integration in trading cycle

**Step 5: Commit**

```bash
git add scripts/auto_trade.py
git commit -m "feat: integrate TimeframeAnalyzer into trading cycle

- Initialize analyzer with price buffer
- Run analysis before AI decisions in test mode
- Log 15m/1H/4H trends and alignment
- Prepare for AI prompt enhancement

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Phase 3: Metrics & Reporting

### Task 9: Add Timeframe Columns to Database

**Files:**
- Modify: `polymarket/performance/database.py`

**Step 1: Create schema migration**

Add new method to PerformanceDatabase class:

```python
def _migrate_add_timeframe_columns(self):
    """Add timeframe analysis columns to trades table."""
    try:
        cursor = self.conn.cursor()

        # Check if columns already exist
        cursor.execute("PRAGMA table_info(trades)")
        columns = {row[1] for row in cursor.fetchall()}

        if 'timeframe_15m_direction' not in columns:
            cursor.execute("""
                ALTER TABLE trades ADD COLUMN timeframe_15m_direction TEXT
            """)
            cursor.execute("""
                ALTER TABLE trades ADD COLUMN timeframe_1h_direction TEXT
            """)
            cursor.execute("""
                ALTER TABLE trades ADD COLUMN timeframe_4h_direction TEXT
            """)
            cursor.execute("""
                ALTER TABLE trades ADD COLUMN timeframe_alignment TEXT
            """)
            cursor.execute("""
                ALTER TABLE trades ADD COLUMN confidence_modifier REAL
            """)
            self.conn.commit()
            logger.info("Database migration: Added timeframe columns")

    except Exception as e:
        logger.error("Failed to migrate database", error=str(e))
        raise
```

**Step 2: Call migration in __init__**

Find the `__init__` method and add after table creation:

```python
# Run migrations
self._migrate_add_timeframe_columns()
```

**Step 3: Update log_decision method**

Find `log_decision` method and add parameters:

```python
def log_decision(
    self,
    # ... existing parameters ...
    timeframe_analysis: Optional[TimeframeAnalysis] = None  # NEW
):
    """Log a trading decision to the database."""

    # Extract timeframe data if available
    tf_15m_dir = None
    tf_1h_dir = None
    tf_4h_dir = None
    tf_alignment = None
    tf_modifier = None

    if timeframe_analysis:
        tf_15m_dir = timeframe_analysis.tf_15m.direction
        tf_1h_dir = timeframe_analysis.tf_1h.direction
        tf_4h_dir = timeframe_analysis.tf_4h.direction
        tf_alignment = timeframe_analysis.alignment_score
        tf_modifier = timeframe_analysis.confidence_modifier

    # ... existing INSERT statement ...
    # Add new columns to INSERT
```

**Step 4: Verify migration**

```bash
cd /root/polymarket-scripts
python3 -c "
from polymarket.performance.database import PerformanceDatabase
db = PerformanceDatabase('data/performance.db')
cursor = db.conn.cursor()
cursor.execute('PRAGMA table_info(trades)')
columns = [row[1] for row in cursor.fetchall()]
print('Timeframe columns added:')
for col in columns:
    if 'timeframe' in col or 'confidence_modifier' in col:
        print(f'  - {col}')
"
```

Expected: See 5 new columns listed

**Step 5: Commit**

```bash
git add polymarket/performance/database.py
git commit -m "feat: add timeframe columns to database schema

- Add timeframe_15m_direction, timeframe_1h_direction, timeframe_4h_direction
- Add timeframe_alignment and confidence_modifier
- Create migration to add columns safely
- Update log_decision to accept TimeframeAnalysis

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 10: Implement Test Mode Metrics Aggregation

**Files:**
- Modify: `polymarket/performance/tracker.py`

**Step 1: Add TestModeMetrics dataclass**

Add at top of file:

```python
@dataclass
class TestModeMetrics:
    """Aggregated metrics for test mode performance."""
    total_trades: int
    executed_trades: int
    execution_rate: float
    wins: int
    losses: int
    win_rate: float
    total_pnl: Decimal
    avg_arbitrage_edge: float
    avg_confidence: float
    timeframe_alignment_stats: dict  # {alignment_type: count}
```

**Step 2: Add calculate_test_mode_metrics method**

Add to PerformanceTracker class:

```python
def calculate_test_mode_metrics(self, last_n_trades: int = 20) -> TestModeMetrics:
    """Calculate aggregated metrics for last N trades.

    Args:
        last_n_trades: Number of recent trades to analyze

    Returns:
        TestModeMetrics with aggregated statistics
    """
    cursor = self.db.conn.cursor()

    # Get last N trades
    cursor.execute("""
        SELECT
            execution_status,
            is_win,
            profit_loss,
            arbitrage_edge,
            confidence,
            timeframe_alignment
        FROM trades
        WHERE is_test_mode = 1
        ORDER BY timestamp DESC
        LIMIT ?
    """, (last_n_trades,))

    trades = cursor.fetchall()

    if not trades:
        return None

    total_trades = len(trades)
    executed = sum(1 for t in trades if t[0] in ['filled', 'FILLED'])
    execution_rate = executed / total_trades if total_trades > 0 else 0.0

    # Only count settled trades for win rate
    settled = [t for t in trades if t[1] is not None]
    wins = sum(1 for t in settled if t[1] == 1)
    losses = len(settled) - wins
    win_rate = wins / len(settled) if settled else 0.0

    # Total P&L from settled trades
    total_pnl = sum(Decimal(str(t[2])) for t in settled if t[2] is not None)

    # Average metrics
    avg_edge = sum(t[3] or 0 for t in trades) / total_trades
    avg_confidence = sum(t[4] or 0 for t in trades) / total_trades

    # Timeframe alignment breakdown
    alignment_stats = {}
    for t in trades:
        alignment = t[5] or "UNKNOWN"
        alignment_stats[alignment] = alignment_stats.get(alignment, 0) + 1

    return TestModeMetrics(
        total_trades=total_trades,
        executed_trades=executed,
        execution_rate=execution_rate,
        wins=wins,
        losses=losses,
        win_rate=win_rate,
        total_pnl=total_pnl,
        avg_arbitrage_edge=avg_edge,
        avg_confidence=avg_confidence,
        timeframe_alignment_stats=alignment_stats
    )
```

**Step 3: Verify implementation**

```bash
grep -c "def calculate_test_mode_metrics" polymarket/performance/tracker.py
```

Expected: 1 occurrence

**Step 4: Commit**

```bash
git add polymarket/performance/tracker.py
git commit -m "feat: implement test mode metrics aggregation

- Add TestModeMetrics dataclass
- Calculate execution rate, win rate, P&L
- Aggregate arbitrage edge and confidence
- Track timeframe alignment distribution
- Analyze last 20 trades

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 11: Add Telegram Test Mode Report

**Files:**
- Modify: `polymarket/telegram/bot.py`

**Step 1: Add send_test_mode_report method**

Add to TelegramBot class:

```python
async def send_test_mode_report(self, metrics: 'TestModeMetrics', trade_range: str) -> None:
    """Send test mode performance report.

    Args:
        metrics: Aggregated test mode metrics
        trade_range: Description like "Trades 21-40"
    """
    if not self.enabled:
        return

    # Format alignment stats
    alignment_lines = []
    for alignment, count in metrics.timeframe_alignment_stats.items():
        pct = (count / metrics.total_trades) * 100
        alignment_lines.append(f"• {alignment}: {count} trades ({pct:.0f}%)")

    alignment_text = "\n".join(alignment_lines)

    message = f"""🎯 TEST MODE REPORT ({trade_range})

📊 Performance:
• Win Rate: {metrics.wins}/{metrics.wins + metrics.losses} ({metrics.win_rate:.1%})
• Total P&L: ${metrics.total_pnl:+.2f}
• Execution Rate: {metrics.executed_trades}/{metrics.total_trades} ({metrics.execution_rate:.1%})

📈 Trade Quality:
• Avg Arbitrage Edge: {metrics.avg_arbitrage_edge:.2%}
• Avg Confidence: {metrics.avg_confidence:.1%}

🕐 Timeframe Analysis:
{alignment_text}

Next report after 20 more trades."""

    await self.send_message(message)
    logger.info("Test mode report sent", trade_range=trade_range)
```

**Step 2: Verify implementation**

```bash
grep -A20 "send_test_mode_report" polymarket/telegram/bot.py
```

Expected: See method implementation

**Step 3: Commit**

```bash
git add polymarket/telegram/bot.py
git commit -m "feat: add Telegram test mode reporting

- Send formatted report every 20 trades
- Include win rate, P&L, execution rate
- Show average edge and confidence
- Display timeframe alignment distribution
- Clean, actionable format

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 12: Integrate Metrics Reporting into Trading Cycle

**Files:**
- Modify: `scripts/auto_trade.py`

**Step 1: Add report trigger after trade logging**

Find where trades are logged and add:

```python
# After logging decision to database
if self.test_mode.enabled:
    # Check if we've hit report milestone (every 20 trades)
    if self.total_trades > 0 and self.total_trades % 20 == 0:
        try:
            metrics = self.performance_tracker.calculate_test_mode_metrics(last_n_trades=20)
            if metrics:
                trade_range = f"Trades {self.total_trades - 19}-{self.total_trades}"
                await self.telegram_bot.send_test_mode_report(metrics, trade_range)
        except Exception as e:
            logger.error("Failed to send test mode report", error=str(e))
```

**Step 2: Pass timeframe_analysis to log_decision**

Find the log_decision call and add parameter:

```python
self.performance_tracker.log_decision(
    # ... existing parameters ...
    timeframe_analysis=timeframe_analysis  # NEW
)
```

**Step 3: Verify integration**

```bash
grep -B2 -A8 "test_mode_report" scripts/auto_trade.py
```

Expected: See report trigger every 20 trades

**Step 4: Commit**

```bash
git add scripts/auto_trade.py
git commit -m "feat: integrate test mode metrics reporting

- Trigger Telegram report every 20 trades
- Pass timeframe analysis to database logger
- Graceful error handling for report failures
- Track total_trades for milestone detection

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Phase 4: AI Prompt Enhancement

### Task 13: Update AI Prompt with Timeframe Context

**Files:**
- Modify: `polymarket/trading/ai_decision.py`

**Step 1: Add timeframe_analysis parameter to make_decision**

Find the `make_decision` method signature and add parameter:

```python
async def make_decision(
    self,
    # ... existing parameters ...
    timeframe_analysis: Optional[TimeframeAnalysis] = None  # NEW
) -> TradingDecision:
```

**Step 2: Add timeframe context to prompt**

In the prompt construction, add new section:

```python
# Build timeframe context if available
timeframe_context = ""
if timeframe_analysis:
    tf = timeframe_analysis
    timeframe_context = f"""

TIMEFRAME CONTEXT:
- 15-min trend: {tf.tf_15m.direction} ({tf.tf_15m.price_change_pct:+.2f}%)
- 1-hour trend: {tf.tf_1h.direction} ({tf.tf_1h.price_change_pct:+.2f}%)
- 4-hour trend: {tf.tf_4h.direction} ({tf.tf_4h.price_change_pct:+.2f}%)
- Alignment: {tf.alignment_score}

Consider multi-timeframe alignment when forming conviction:
- If all timeframes aligned in same direction: Strong directional signal
- If timeframes mixed: Exercise caution, look for strong arbitrage edge
- If 15m contradicts longer timeframes: Short-term move against trend (mean reversion risk)

Your confidence will be automatically adjusted based on alignment:
- Aligned timeframes: +15% confidence boost
- Mixed signals: No adjustment
- Conflicting signals: -15% confidence reduction
"""

# Add to main prompt
prompt = f"""
{existing_prompt_sections}
{timeframe_context}
{remaining_prompt_sections}
"""
```

**Step 3: Apply confidence modifier after AI decision**

After getting AI response, apply modifier:

```python
# Parse AI decision
decision = self._parse_decision(response)

# Apply timeframe confidence modifier if available
if timeframe_analysis:
    base_confidence = decision.confidence
    modifier = timeframe_analysis.confidence_modifier
    decision.confidence = min(base_confidence + modifier, 1.0)

    logger.info(
        "Applied timeframe confidence modifier",
        base_confidence=f"{base_confidence:.1%}",
        modifier=f"{modifier:+.1%}",
        final_confidence=f"{decision.confidence:.1%}"
    )

return decision
```

**Step 4: Verify changes**

```bash
grep -A5 "TIMEFRAME CONTEXT" polymarket/trading/ai_decision.py
grep -A8 "timeframe confidence modifier" polymarket/trading/ai_decision.py
```

Expected: See timeframe context in prompt and modifier application

**Step 5: Commit**

```bash
git add polymarket/trading/ai_decision.py
git commit -m "feat: enhance AI prompt with multi-timeframe context

- Add timeframe analysis to decision-making prompt
- Explain how to interpret alignment signals
- Apply confidence modifiers after AI decision
- Log base and final confidence levels
- No artificial constraints on direction

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 14: Update AutoTrader to Pass Timeframe Analysis to AI

**Files:**
- Modify: `scripts/auto_trade.py`

**Step 1: Find AI decision call**

```bash
grep -n "ai_service.make_decision" scripts/auto_trade.py
```

Expected: Shows line number of AI decision call

**Step 2: Pass timeframe_analysis parameter**

Update the call:

```python
decision = await self.ai_service.make_decision(
    # ... existing parameters ...
    timeframe_analysis=timeframe_analysis  # NEW
)
```

**Step 3: Verify integration**

```bash
grep -B3 -A10 "ai_service.make_decision" scripts/auto_trade.py
```

Expected: See timeframe_analysis passed as parameter

**Step 4: Commit**

```bash
git add scripts/auto_trade.py
git commit -m "feat: pass timeframe analysis to AI decision service

- Connect timeframe analyzer output to AI
- Enable multi-timeframe informed decisions
- Complete Phase 4 integration

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Integration Testing

### Task 15: End-to-End Test

**Step 1: Stop bot**

```bash
pkill -f "python3 scripts/auto_trade.py"
```

**Step 2: Clear test database (optional - fresh start)**

```bash
cd /root/polymarket-scripts
cp data/performance.db data/performance.db.backup
rm data/performance.db
```

**Step 3: Start bot with all phases integrated**

```bash
TEST_MODE=true python3 scripts/auto_trade.py > /root/test-complete.log 2>&1 &
```

**Step 4: Monitor for first 3 cycles (10 minutes)**

```bash
sleep 600
tail -100 /root/test-complete.log | grep -E "TEST|Timeframe|Position sizing|Edge|Order"
```

Expected to see:
- ✅ `[TEST] Trading with min $5, max $10 bets, 70% min confidence, 2% min edge`
- ✅ `Multi-timeframe analysis` logs
- ✅ `[TEST] Position sizing` with amounts >= $5
- ✅ Possible: Trades skipped due to edge < 2%
- ✅ NO "Size lower than minimum" errors

**Step 5: Wait for 20 trades to verify Telegram report**

Monitor total_trades counter and wait for Telegram report.

**Step 6: Verify database has timeframe data**

```bash
cd /root/polymarket-scripts
python3 -c "
import sqlite3
conn = sqlite3.connect('data/performance.db')
cursor = conn.cursor()
cursor.execute('SELECT timeframe_15m_direction, timeframe_1h_direction, timeframe_4h_direction, timeframe_alignment FROM trades WHERE timeframe_alignment IS NOT NULL LIMIT 5')
for row in cursor.fetchall():
    print(f'15m: {row[0]}, 1H: {row[1]}, 4H: {row[2]}, Alignment: {row[3]}')
"
```

Expected: See timeframe data populated

**Step 7: Document success criteria**

Verify all success criteria met:
- ✅ Execution rate > 80%
- ✅ No minimum size errors
- ✅ Trades filtered by 2% edge
- ✅ Timeframe data in logs and database
- ✅ Telegram report received after 20 trades

---

## Documentation Update

### Task 16: Add Implementation Notes to Design Document

**Files:**
- Modify: `docs/plans/2026-02-14-test-mode-comprehensive-overhaul-design.md`

**Step 1: Add Implementation Notes section**

Append to design document:

```markdown
---

## Implementation Notes

**Completed:** 2026-02-14

**Phases Implemented:**

1. **Phase 1: Core Execution Fixes** ✅
   - Added min_bet_amount ($5) and max_bet_amount ($10) to TestModeConfig
   - Added min_arbitrage_edge (2%) filtering
   - Implemented position sizing with min/max enforcement
   - Result: Execution rate improved from 0% to >80%

2. **Phase 2: Multi-Timeframe Analysis** ✅
   - Created TimeframeAnalyzer service with 15m/1H/4H trend calculation
   - Implemented alignment scoring (ALIGNED/MIXED/CONFLICTING)
   - Integrated confidence modifiers (+15%, 0%, -15%)
   - Uses existing PriceHistoryBuffer for data

3. **Phase 3: Metrics & Reporting** ✅
   - Added 5 timeframe columns to database
   - Implemented metrics aggregation for last 20 trades
   - Created Telegram reporting with win rate, P&L, edge, alignment stats
   - Reports trigger every 20 trades

4. **Phase 4: AI Prompt Enhancement** ✅
   - Enhanced AI prompt with multi-timeframe context
   - Applied confidence modifiers after AI decision
   - No artificial constraints on direction or confidence

**Commits:**
- (List commit hashes here after completion)

**Test Results:**
- Execution rate: (Record actual %)
- First 20 trades win rate: (Record actual %)
- Average arbitrage edge: (Record actual %)
- Timeframe alignment distribution: (Record actual stats)

**Lessons Learned:**
- (Add any insights from implementation)
```

**Step 2: Commit documentation**

```bash
git add docs/plans/2026-02-14-test-mode-comprehensive-overhaul-design.md
git commit -m "docs: add implementation notes to design document

- Document completion of all 4 phases
- Record test results and metrics
- Add lessons learned

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Rollout

### Task 17: Final Validation and Production Deployment

**Step 1: Run for 50 trades in test mode**

Monitor bot until 50 trades executed and 2 Telegram reports received.

**Step 2: Analyze results**

```bash
cd /root/polymarket-scripts
python3 << 'EOF'
import sqlite3
from decimal import Decimal

conn = sqlite3.connect('data/performance.db')
cursor = conn.cursor()

# Get test mode performance
cursor.execute("""
    SELECT
        COUNT(*) as total,
        SUM(CASE WHEN execution_status = 'filled' THEN 1 ELSE 0 END) as executed,
        SUM(CASE WHEN is_win = 1 THEN 1 ELSE 0 END) as wins,
        SUM(CASE WHEN is_win = 0 THEN 1 ELSE 0 END) as losses,
        SUM(profit_loss) as total_pnl,
        AVG(arbitrage_edge) as avg_edge,
        AVG(confidence) as avg_confidence
    FROM trades
    WHERE is_test_mode = 1
""")

stats = cursor.fetchone()
print("TEST MODE FINAL RESULTS:")
print(f"  Total trades: {stats[0]}")
print(f"  Executed: {stats[1]} ({stats[1]/stats[0]*100:.1f}%)")
print(f"  Wins: {stats[2]}")
print(f"  Losses: {stats[3]}")
print(f"  Win Rate: {stats[2]/(stats[2]+stats[3])*100:.1f}%" if stats[2] or stats[3] else "  Win Rate: N/A")
print(f"  Total P&L: ${stats[4]:.2f}" if stats[4] else "  Total P&L: N/A")
print(f"  Avg Edge: {stats[5]:.2%}")
print(f"  Avg Confidence: {stats[6]:.1%}")

conn.close()
EOF
```

**Step 3: Decision point**

If results are satisfactory (execution rate >80%, reasonable win rate, positive P&L trend):
- ✅ Mark as production-ready
- ✅ Merge to main branch
- ✅ Deploy to production

If results need iteration:
- Analyze failure modes
- Adjust parameters (min_edge, timeframe thresholds)
- Re-test

**Step 4: Merge and deploy**

```bash
git checkout main
git merge test-mode-overhaul
git push origin main
```

---

## Success Criteria Summary

**After implementation, test mode should achieve:**

### Execution Metrics:
- ✅ Execution rate > 80% (orders actually fill)
- ✅ Zero "Size lower than minimum" errors
- ✅ All trades have position_size >= $5.00

### Decision Quality:
- ✅ Average arbitrage edge > 3% (filter working)
- ✅ No trades taken with edge < 2%
- ✅ Trades show multi-timeframe context in logs

### Timeframe Integration:
- ✅ All trades logged with 15m/1H/4H trend data
- ✅ Confidence modifiers applied based on alignment
- ✅ Database contains complete timeframe columns

### Reporting:
- ✅ Telegram reports sent every 20 trades
- ✅ Reports include: win rate, P&L, execution rate, edge, alignment stats
- ✅ Reports are actionable and easy to understand

### Philosophy:
- ✅ Bot can be 90% NO if trend is bearish (no forced balance)
- ✅ Bot can be 95% confident if signals align (no calibration caps)
- ✅ Focus on outcomes (win rate, P&L), not process (bias alerts)

---

**Implementation Complete!**
