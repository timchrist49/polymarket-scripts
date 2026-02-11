# Self-Reflection System Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a self-healing and self-reflection system that tracks performance, generates AI insights, adapts parameters autonomously, and provides Telegram control.

**Architecture:** 4-component system (Performance Tracker → Reflection Engine → Parameter Adjuster → Telegram Bot) with SQLite+JSON hybrid storage, adaptive triggers, and tiered autonomy.

**Tech Stack:** SQLite, OpenAI API, python-telegram-bot, asyncio, structlog, pytest

---

## Phase 1: Performance Tracker (Foundation)

### Task 1: Create Database Schema

**Files:**
- Create: `polymarket/performance/__init__.py`
- Create: `polymarket/performance/database.py`
- Create: `tests/test_performance_database.py`

**Step 1: Write the failing test**

```python
# tests/test_performance_database.py
import pytest
from datetime import datetime
from decimal import Decimal
from polymarket.performance.database import PerformanceDatabase

@pytest.fixture
def db():
    """Create in-memory test database."""
    db = PerformanceDatabase(":memory:")
    yield db
    db.close()

def test_create_tables(db):
    """Test tables are created with correct schema."""
    cursor = db.conn.cursor()

    # Check trades table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='trades'")
    assert cursor.fetchone() is not None

    # Check columns exist
    cursor.execute("PRAGMA table_info(trades)")
    columns = {row[1] for row in cursor.fetchall()}

    expected_columns = {
        'id', 'timestamp', 'market_slug', 'market_id',
        'action', 'confidence', 'position_size', 'reasoning',
        'btc_price', 'price_to_beat', 'time_remaining_seconds', 'is_end_phase',
        'social_score', 'market_score', 'final_score', 'final_confidence', 'signal_type',
        'rsi', 'macd', 'trend',
        'yes_price', 'no_price', 'executed_price',
        'actual_outcome', 'profit_loss', 'is_win', 'is_missed_opportunity'
    }

    assert expected_columns.issubset(columns)

def test_create_indexes(db):
    """Test indexes are created for fast queries."""
    cursor = db.conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
    indexes = {row[0] for row in cursor.fetchall()}

    expected = {'idx_trades_timestamp', 'idx_trades_signal_type', 'idx_trades_is_win'}
    assert expected.issubset(indexes)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_performance_database.py::test_create_tables -v`
Expected: `ModuleNotFoundError: No module named 'polymarket.performance'`

**Step 3: Write minimal implementation**

```python
# polymarket/performance/__init__.py
"""Performance tracking, reflection, and self-healing components."""

from polymarket.performance.database import PerformanceDatabase

__all__ = ["PerformanceDatabase"]
```

```python
# polymarket/performance/database.py
"""SQLite database for performance tracking."""

import sqlite3
from pathlib import Path
import structlog

logger = structlog.get_logger()


class PerformanceDatabase:
    """SQLite database for storing trade performance data."""

    def __init__(self, db_path: str = "data/performance.db"):
        """
        Initialize database connection and create schema.

        Args:
            db_path: Path to SQLite database file (':memory:' for in-memory)
        """
        self.db_path = db_path

        # Create data directory if needed
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        # Connect to database
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row

        # Create schema
        self._create_tables()
        self._create_indexes()

        logger.info("Performance database initialized", db_path=db_path)

    def _create_tables(self):
        """Create database tables."""
        cursor = self.conn.cursor()

        # Trades table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                market_slug TEXT NOT NULL,
                market_id INTEGER,

                -- Decision
                action TEXT NOT NULL,
                confidence REAL NOT NULL,
                position_size REAL NOT NULL,
                reasoning TEXT,

                -- Market Context
                btc_price REAL NOT NULL,
                price_to_beat REAL,
                time_remaining_seconds INTEGER,
                is_end_phase BOOLEAN,

                -- Signals
                social_score REAL,
                market_score REAL,
                final_score REAL,
                final_confidence REAL,
                signal_type TEXT,

                -- Technical Indicators
                rsi REAL,
                macd REAL,
                trend TEXT,

                -- Pricing
                yes_price REAL,
                no_price REAL,
                executed_price REAL,

                -- Outcome (filled after market closes)
                actual_outcome TEXT,
                profit_loss REAL,
                is_win BOOLEAN,
                is_missed_opportunity BOOLEAN
            )
        """)

        # Reflections table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS reflections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                trigger_type TEXT NOT NULL,
                trades_analyzed INTEGER NOT NULL,
                insights TEXT NOT NULL,
                adjustments_made TEXT
            )
        """)

        # Parameter history table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS parameter_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                parameter_name TEXT NOT NULL,
                old_value REAL NOT NULL,
                new_value REAL NOT NULL,
                reason TEXT NOT NULL,
                approval_method TEXT NOT NULL
            )
        """)

        self.conn.commit()

    def _create_indexes(self):
        """Create indexes for fast queries."""
        cursor = self.conn.cursor()

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_trades_timestamp
            ON trades(timestamp)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_trades_signal_type
            ON trades(signal_type)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_trades_is_win
            ON trades(is_win)
        """)

        self.conn.commit()

    def close(self):
        """Close database connection."""
        self.conn.close()
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_performance_database.py -v`
Expected: `2 passed`

**Step 5: Commit**

```bash
git add polymarket/performance/ tests/test_performance_database.py
git commit -m "feat(performance): add database schema with trades, reflections, and parameter_history tables

- SQLite schema with comprehensive trade tracking
- Indexes for fast queries (timestamp, signal_type, is_win)
- In-memory testing support
- Auto-creates data directory

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 2: Implement Trade Logging

**Files:**
- Modify: `polymarket/performance/database.py`
- Modify: `tests/test_performance_database.py`

**Step 1: Write the failing test**

```python
# tests/test_performance_database.py (add to existing file)
def test_log_trade(db):
    """Test logging a trade decision."""
    trade_data = {
        "timestamp": datetime(2026, 2, 11, 10, 30, 0),
        "market_slug": "btc-updown-15m-1234567890",
        "market_id": 1362391,
        "action": "NO",
        "confidence": 1.0,
        "position_size": 5.0,
        "reasoning": "Bearish signals aligned",
        "btc_price": 66940.0,
        "price_to_beat": 66826.14,
        "time_remaining_seconds": 480,
        "is_end_phase": False,
        "social_score": -0.10,
        "market_score": -0.21,
        "final_score": -0.17,
        "final_confidence": 1.0,
        "signal_type": "STRONG_BEARISH",
        "rsi": 60.1,
        "macd": 1.74,
        "trend": "BULLISH",
        "yes_price": 0.51,
        "no_price": 0.50,
        "executed_price": 0.52
    }

    trade_id = db.log_trade(trade_data)
    assert trade_id > 0

    # Verify data was stored
    cursor = db.conn.cursor()
    cursor.execute("SELECT * FROM trades WHERE id = ?", (trade_id,))
    row = cursor.fetchone()

    assert row is not None
    assert row['action'] == 'NO'
    assert row['confidence'] == 1.0
    assert row['market_slug'] == 'btc-updown-15m-1234567890'
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_performance_database.py::test_log_trade -v`
Expected: `AttributeError: 'PerformanceDatabase' object has no attribute 'log_trade'`

**Step 3: Write minimal implementation**

```python
# polymarket/performance/database.py (add method to PerformanceDatabase class)

def log_trade(self, trade_data: dict) -> int:
    """
    Log a trade decision to the database.

    Args:
        trade_data: Dictionary with trade information

    Returns:
        Trade ID of inserted record
    """
    cursor = self.conn.cursor()

    cursor.execute("""
        INSERT INTO trades (
            timestamp, market_slug, market_id,
            action, confidence, position_size, reasoning,
            btc_price, price_to_beat, time_remaining_seconds, is_end_phase,
            social_score, market_score, final_score, final_confidence, signal_type,
            rsi, macd, trend,
            yes_price, no_price, executed_price
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        trade_data["timestamp"],
        trade_data["market_slug"],
        trade_data.get("market_id"),
        trade_data["action"],
        trade_data["confidence"],
        trade_data["position_size"],
        trade_data.get("reasoning"),
        trade_data["btc_price"],
        trade_data.get("price_to_beat"),
        trade_data.get("time_remaining_seconds"),
        trade_data.get("is_end_phase", False),
        trade_data.get("social_score"),
        trade_data.get("market_score"),
        trade_data.get("final_score"),
        trade_data.get("final_confidence"),
        trade_data.get("signal_type"),
        trade_data.get("rsi"),
        trade_data.get("macd"),
        trade_data.get("trend"),
        trade_data.get("yes_price"),
        trade_data.get("no_price"),
        trade_data.get("executed_price")
    ))

    self.conn.commit()
    trade_id = cursor.lastrowid

    logger.debug("Trade logged", trade_id=trade_id, action=trade_data["action"])
    return trade_id
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_performance_database.py::test_log_trade -v`
Expected: `1 passed`

**Step 5: Commit**

```bash
git add polymarket/performance/database.py tests/test_performance_database.py
git commit -m "feat(performance): add trade logging to database

- log_trade method stores all trade context
- Handles optional fields gracefully
- Returns trade ID for reference

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 3: Implement Outcome Updates

**Files:**
- Modify: `polymarket/performance/database.py`
- Modify: `tests/test_performance_database.py`

**Step 1: Write the failing test**

```python
# tests/test_performance_database.py (add to existing file)
def test_update_outcome(db):
    """Test updating trade outcome after market closes."""
    # First log a trade
    trade_data = {
        "timestamp": datetime(2026, 2, 11, 10, 30, 0),
        "market_slug": "btc-updown-15m-1234567890",
        "market_id": 1362391,
        "action": "NO",
        "confidence": 1.0,
        "position_size": 5.0,
        "btc_price": 66940.0,
    }
    trade_id = db.log_trade(trade_data)

    # Update with outcome
    db.update_outcome(
        market_slug="btc-updown-15m-1234567890",
        actual_outcome="DOWN",
        profit_loss=4.50
    )

    # Verify outcome was stored
    cursor = db.conn.cursor()
    cursor.execute("SELECT * FROM trades WHERE id = ?", (trade_id,))
    row = cursor.fetchone()

    assert row['actual_outcome'] == 'DOWN'
    assert row['profit_loss'] == 4.50
    assert row['is_win'] == True  # NO bet + DOWN outcome = win

def test_update_outcome_missed_opportunity(db):
    """Test HOLD decision marked as missed opportunity."""
    trade_data = {
        "timestamp": datetime(2026, 2, 11, 10, 30, 0),
        "market_slug": "btc-updown-15m-1234567890",
        "action": "HOLD",
        "confidence": 0.85,
        "position_size": 0.0,
        "btc_price": 66940.0,
        "price_to_beat": 66826.14,
    }
    trade_id = db.log_trade(trade_data)

    # Update - price went up, would have won YES
    db.update_outcome(
        market_slug="btc-updown-15m-1234567890",
        actual_outcome="UP",
        profit_loss=0.0  # Didn't trade
    )

    cursor = db.conn.cursor()
    cursor.execute("SELECT * FROM trades WHERE id = ?", (trade_id,))
    row = cursor.fetchone()

    assert row['actual_outcome'] == 'UP'
    assert row['is_missed_opportunity'] == True
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_performance_database.py::test_update_outcome -v`
Expected: `AttributeError: 'PerformanceDatabase' object has no attribute 'update_outcome'`

**Step 3: Write minimal implementation**

```python
# polymarket/performance/database.py (add method to PerformanceDatabase class)

def update_outcome(self, market_slug: str, actual_outcome: str, profit_loss: float):
    """
    Update trade outcome after market closes.

    Args:
        market_slug: Market identifier
        actual_outcome: 'UP' or 'DOWN'
        profit_loss: Profit/loss amount (0 for HOLD)
    """
    cursor = self.conn.cursor()

    # Get the trade
    cursor.execute("""
        SELECT id, action, position_size
        FROM trades
        WHERE market_slug = ?
        ORDER BY timestamp DESC
        LIMIT 1
    """, (market_slug,))

    row = cursor.fetchone()
    if not row:
        logger.warning("No trade found for outcome update", market_slug=market_slug)
        return

    trade_id = row['id']
    action = row['action']
    position_size = row['position_size']

    # Determine if win
    is_win = None
    is_missed_opportunity = False

    if action == "HOLD":
        is_win = None  # Didn't trade
        is_missed_opportunity = (position_size == 0)  # Would have won
    elif action == "YES":
        is_win = (actual_outcome == "UP")
    elif action == "NO":
        is_win = (actual_outcome == "DOWN")

    # Update database
    cursor.execute("""
        UPDATE trades
        SET actual_outcome = ?,
            profit_loss = ?,
            is_win = ?,
            is_missed_opportunity = ?
        WHERE id = ?
    """, (actual_outcome, profit_loss, is_win, is_missed_opportunity, trade_id))

    self.conn.commit()

    logger.info(
        "Outcome updated",
        trade_id=trade_id,
        action=action,
        actual_outcome=actual_outcome,
        is_win=is_win,
        profit_loss=profit_loss
    )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_performance_database.py::test_update_outcome -v`
Expected: `2 passed`

**Step 5: Commit**

```bash
git add polymarket/performance/database.py tests/test_performance_database.py
git commit -m "feat(performance): add outcome tracking with win/loss and missed opportunities

- update_outcome method marks wins/losses
- Detects missed opportunities from HOLD decisions
- Calculates is_win based on action vs actual_outcome

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 4: Implement Performance Tracker Service

**Files:**
- Create: `polymarket/performance/tracker.py`
- Create: `tests/test_performance_tracker.py`

**Step 1: Write the failing test**

```python
# tests/test_performance_tracker.py
import pytest
from datetime import datetime
from decimal import Decimal
from polymarket.performance.tracker import PerformanceTracker
from polymarket.models import TradingDecision, BTCPriceData, TechnicalIndicators, AggregatedSentiment, SocialSentiment, MarketSignals

@pytest.fixture
def tracker():
    """Create performance tracker with in-memory DB."""
    tracker = PerformanceTracker(db_path=":memory:")
    yield tracker
    tracker.close()

@pytest.fixture
def sample_market():
    """Sample market data."""
    return {
        "id": 1362391,
        "question": "Will BTC go up?",
        "condition_id": "test",
        "outcomes": ["Up", "Down"],
        "best_bid": 0.50,
        "best_ask": 0.51,
        "active": True
    }

@pytest.fixture
def sample_decision():
    """Sample trading decision."""
    return TradingDecision(
        action="NO",
        confidence=1.0,
        reasoning="Bearish signals aligned",
        token_id="test",
        position_size=Decimal("5.0"),
        stop_loss_threshold=0.40
    )

@pytest.fixture
def sample_btc_data():
    """Sample BTC price data."""
    return BTCPriceData(
        price=Decimal("66940.0"),
        timestamp=datetime(2026, 2, 11, 10, 30, 0),
        source="binance",
        volume_24h=Decimal("1000.0")
    )

@pytest.fixture
def sample_technical():
    """Sample technical indicators."""
    return TechnicalIndicators(
        rsi=60.1,
        macd_value=1.74,
        macd_signal=1.50,
        macd_histogram=0.24,
        ema_short=66950.0,
        ema_long=66900.0,
        sma_50=66800.0,
        volume_change=5.0,
        price_velocity=2.0,
        trend="BULLISH"
    )

@pytest.fixture
def sample_aggregated():
    """Sample aggregated sentiment."""
    social = SocialSentiment(
        score=-0.10,
        confidence=1.0,
        fear_greed=45,
        is_trending=False,
        vote_up_pct=48.0,
        vote_down_pct=52.0,
        signal_type="STRONG_BEARISH",
        sources_available=["fear_greed", "votes"],
        timestamp=datetime(2026, 2, 11, 10, 30, 0)
    )

    market_signals = MarketSignals(
        score=-0.21,
        confidence=1.0,
        order_book_score=0.0,
        whale_score=-0.15,
        volume_score=-0.10,
        momentum_score=-0.20,
        order_book_bias="N/A",
        whale_direction="SELLING",
        whale_count=2,
        volume_ratio=0.9,
        momentum_direction="DOWN",
        signal_type="STRONG_BEARISH",
        timestamp=datetime(2026, 2, 11, 10, 30, 0)
    )

    return AggregatedSentiment(
        social=social,
        market=market_signals,
        final_score=-0.17,
        final_confidence=1.0,
        agreement_multiplier=1.47,
        signal_type="STRONG_BEARISH",
        timestamp=datetime(2026, 2, 11, 10, 30, 0)
    )

@pytest.mark.asyncio
async def test_log_decision(
    tracker,
    sample_market,
    sample_decision,
    sample_btc_data,
    sample_technical,
    sample_aggregated
):
    """Test logging a trading decision."""
    trade_id = await tracker.log_decision(
        market=sample_market,
        decision=sample_decision,
        btc_data=sample_btc_data,
        technical=sample_technical,
        aggregated=sample_aggregated,
        price_to_beat=Decimal("66826.14"),
        time_remaining_seconds=480,
        is_end_phase=False
    )

    assert trade_id > 0

    # Verify data in database
    cursor = tracker.db.conn.cursor()
    cursor.execute("SELECT * FROM trades WHERE id = ?", (trade_id,))
    row = cursor.fetchone()

    assert row is not None
    assert row['action'] == 'NO'
    assert row['confidence'] == 1.0
    assert row['market_id'] == 1362391
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_performance_tracker.py::test_log_decision -v`
Expected: `ModuleNotFoundError: No module named 'polymarket.performance.tracker'`

**Step 3: Write minimal implementation**

```python
# polymarket/performance/tracker.py
"""Performance tracking service."""

from datetime import datetime
from decimal import Decimal
from typing import Optional
import structlog

from polymarket.performance.database import PerformanceDatabase
from polymarket.models import TradingDecision, BTCPriceData, TechnicalIndicators, AggregatedSentiment

logger = structlog.get_logger()


class PerformanceTracker:
    """Tracks trading performance and stores to database."""

    def __init__(self, db_path: str = "data/performance.db"):
        """
        Initialize performance tracker.

        Args:
            db_path: Path to SQLite database (':memory:' for testing)
        """
        self.db = PerformanceDatabase(db_path)
        logger.info("Performance tracker initialized")

    async def log_decision(
        self,
        market: dict,
        decision: TradingDecision,
        btc_data: BTCPriceData,
        technical: TechnicalIndicators,
        aggregated: AggregatedSentiment,
        price_to_beat: Optional[Decimal] = None,
        time_remaining_seconds: Optional[int] = None,
        is_end_phase: bool = False
    ) -> int:
        """
        Log a trading decision to the database.

        Args:
            market: Market data dict
            decision: Trading decision
            btc_data: BTC price data
            technical: Technical indicators
            aggregated: Aggregated sentiment
            price_to_beat: Baseline price for comparison
            time_remaining_seconds: Time until market closes
            is_end_phase: Whether in end-of-market phase

        Returns:
            Trade ID
        """
        try:
            # Extract market slug from question or ID
            market_slug = self._extract_market_slug(market)

            # Build trade data dict
            trade_data = {
                "timestamp": datetime.now(),
                "market_slug": market_slug,
                "market_id": market.get("id"),

                # Decision
                "action": decision.action,
                "confidence": decision.confidence,
                "position_size": float(decision.position_size),
                "reasoning": decision.reasoning,

                # Market Context
                "btc_price": float(btc_data.price),
                "price_to_beat": float(price_to_beat) if price_to_beat else None,
                "time_remaining_seconds": time_remaining_seconds,
                "is_end_phase": is_end_phase,

                # Signals
                "social_score": aggregated.social.score,
                "market_score": aggregated.market.score,
                "final_score": aggregated.final_score,
                "final_confidence": aggregated.final_confidence,
                "signal_type": aggregated.signal_type,

                # Technical
                "rsi": technical.rsi,
                "macd": technical.macd_value,
                "trend": technical.trend,

                # Pricing
                "yes_price": market.get("best_ask"),
                "no_price": 1 - market.get("best_bid", 0.5),
                "executed_price": market.get("best_ask") if decision.action == "YES"
                                else 1 - market.get("best_bid", 0.5) if decision.action == "NO"
                                else None
            }

            trade_id = self.db.log_trade(trade_data)

            logger.info(
                "Decision logged to database",
                trade_id=trade_id,
                action=decision.action,
                market_slug=market_slug
            )

            return trade_id

        except Exception as e:
            logger.error("Failed to log decision", error=str(e))
            # Don't block trading on logging failure
            return -1

    def _extract_market_slug(self, market: dict) -> str:
        """Extract market slug from market data."""
        # Try explicit slug field first
        if "slug" in market:
            return market["slug"]

        # Fall back to question-based slug
        question = market.get("question", "unknown")
        # Simplified slug generation
        slug = question.lower().replace(" ", "-").replace("?", "")[:50]
        return slug

    def close(self):
        """Close database connection."""
        self.db.close()
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_performance_tracker.py::test_log_decision -v`
Expected: `1 passed`

**Step 5: Commit**

```bash
git add polymarket/performance/tracker.py tests/test_performance_tracker.py
git commit -m "feat(performance): add PerformanceTracker service for logging decisions

- Wraps database with convenient async API
- Extracts and formats all trade context
- Handles errors gracefully (don't block trading)

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 5: Integrate Tracker into Trading Bot

**Files:**
- Modify: `scripts/auto_trade.py`
- Create: `polymarket/performance/__init__.py` (if not exists)

**Step 1: Write integration test**

```python
# tests/test_integration_performance.py
import pytest
from unittest.mock import Mock, AsyncMock
from polymarket.performance.tracker import PerformanceTracker

@pytest.mark.asyncio
async def test_trading_bot_logs_decisions():
    """Test that trading bot logs all decisions to performance tracker."""
    # This is an integration test - will verify after implementation
    pass  # Placeholder for now
```

**Step 2: Add tracker initialization to auto_trade.py**

Find the section in `scripts/auto_trade.py` where services are initialized (around line 50-100) and add:

```python
# After btc_service initialization
from polymarket.performance.tracker import PerformanceTracker

# Initialize performance tracker
performance_tracker = PerformanceTracker()
logger.info("Performance tracking enabled")
```

**Step 3: Add logging hook after decision**

Find the section where trading decision is made (search for `ai_decision.make_decision`) and add after:

```python
# After AI decision is made (around line 400-450)
try:
    await performance_tracker.log_decision(
        market=market,
        decision=decision,
        btc_data=btc_data,
        technical=technical_indicators,
        aggregated=aggregated_sentiment,
        price_to_beat=price_to_beat,
        time_remaining_seconds=time_remaining,
        is_end_phase=is_end_of_market
    )
except Exception as e:
    logger.error("Performance logging failed", error=str(e))
    # Continue trading
```

**Step 4: Add cleanup on bot shutdown**

Find the cleanup section (usually in `finally` block or signal handler) and add:

```python
# In cleanup section
performance_tracker.close()
logger.info("Performance tracker closed")
```

**Step 5: Test integration manually**

Run: `python scripts/auto_trade.py` (with appropriate env vars)
Expected: Bot runs, logs appear showing "Decision logged to database"
Check: `ls data/performance.db` exists

**Step 6: Commit**

```bash
git add scripts/auto_trade.py polymarket/performance/__init__.py
git commit -m "feat(bot): integrate performance tracking into trading loop

- Initialize PerformanceTracker on bot startup
- Log all decisions after AI analysis
- Graceful error handling (don't block trading)
- Cleanup on shutdown

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Phase 2: Reflection Engine (AI Analysis)

### Task 6: Create Metrics Calculator

**Files:**
- Create: `polymarket/performance/metrics.py`
- Create: `tests/test_performance_metrics.py`

**Step 1: Write the failing test**

```python
# tests/test_performance_metrics.py
import pytest
from datetime import datetime, timedelta
from polymarket.performance.metrics import MetricsCalculator
from polymarket.performance.database import PerformanceDatabase

@pytest.fixture
def db_with_trades():
    """Create database with sample trades."""
    db = PerformanceDatabase(":memory:")

    # Add 10 sample trades
    base_time = datetime(2026, 2, 11, 10, 0, 0)

    for i in range(10):
        trade_data = {
            "timestamp": base_time + timedelta(minutes=i*15),
            "market_slug": f"btc-updown-15m-{i}",
            "action": "NO" if i % 2 == 0 else "YES",
            "confidence": 0.7 + (i * 0.02),
            "position_size": 5.0,
            "btc_price": 66000.0 + (i * 100),
            "signal_type": "STRONG_BEARISH" if i < 5 else "STRONG_BULLISH",
            "is_end_phase": i > 7,
            "executed_price": 0.50
        }
        trade_id = db.log_trade(trade_data)

        # Set outcomes for first 8 trades
        if i < 8:
            is_win = (i % 3 != 0)  # 6 wins, 2 losses
            db.update_outcome(
                market_slug=f"btc-updown-15m-{i}",
                actual_outcome="DOWN" if is_win == (trade_data["action"] == "NO") else "UP",
                profit_loss=4.0 if is_win else -5.0
            )

    yield db
    db.close()

def test_calculate_win_rate(db_with_trades):
    """Test win rate calculation."""
    calc = MetricsCalculator(db_with_trades)

    win_rate = calc.calculate_win_rate()
    assert win_rate == 0.75  # 6 wins / 8 trades

def test_calculate_profit_loss(db_with_trades):
    """Test total profit/loss calculation."""
    calc = MetricsCalculator(db_with_trades)

    total_profit = calc.calculate_total_profit()
    assert total_profit == 14.0  # (6 * 4.0) - (2 * 5.0)

def test_signal_performance(db_with_trades):
    """Test win rate by signal type."""
    calc = MetricsCalculator(db_with_trades)

    signal_perf = calc.calculate_signal_performance()

    assert "STRONG_BEARISH" in signal_perf
    assert "STRONG_BULLISH" in signal_perf
    assert all(0 <= perf["win_rate"] <= 1 for perf in signal_perf.values())
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_performance_metrics.py::test_calculate_win_rate -v`
Expected: `ModuleNotFoundError: No module named 'polymarket.performance.metrics'`

**Step 3: Write minimal implementation**

```python
# polymarket/performance/metrics.py
"""Performance metrics calculation."""

from typing import Dict, Optional
import structlog

from polymarket.performance.database import PerformanceDatabase

logger = structlog.get_logger()


class MetricsCalculator:
    """Calculates performance metrics from trade database."""

    def __init__(self, db: PerformanceDatabase):
        """
        Initialize metrics calculator.

        Args:
            db: PerformanceDatabase instance
        """
        self.db = db

    def calculate_win_rate(self, days: Optional[int] = None) -> float:
        """
        Calculate win rate percentage.

        Args:
            days: Number of days to look back (None = all time)

        Returns:
            Win rate as decimal (0.0 to 1.0)
        """
        cursor = self.db.conn.cursor()

        query = """
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN is_win = 1 THEN 1 ELSE 0 END) as wins
            FROM trades
            WHERE is_win IS NOT NULL
        """

        if days:
            query += f" AND timestamp >= datetime('now', '-{days} days')"

        cursor.execute(query)
        row = cursor.fetchone()

        if not row or row['total'] == 0:
            return 0.0

        win_rate = row['wins'] / row['total']
        return win_rate

    def calculate_total_profit(self, days: Optional[int] = None) -> float:
        """
        Calculate total profit/loss.

        Args:
            days: Number of days to look back (None = all time)

        Returns:
            Total profit/loss
        """
        cursor = self.db.conn.cursor()

        query = """
            SELECT SUM(profit_loss) as total
            FROM trades
            WHERE profit_loss IS NOT NULL
        """

        if days:
            query += f" AND timestamp >= datetime('now', '-{days} days')"

        cursor.execute(query)
        row = cursor.fetchone()

        return row['total'] or 0.0

    def calculate_signal_performance(self, days: Optional[int] = None) -> Dict[str, Dict]:
        """
        Calculate win rate by signal type.

        Args:
            days: Number of days to look back (None = all time)

        Returns:
            Dict mapping signal_type to performance stats
        """
        cursor = self.db.conn.cursor()

        query = """
            SELECT
                signal_type,
                COUNT(*) as total,
                SUM(CASE WHEN is_win = 1 THEN 1 ELSE 0 END) as wins,
                AVG(profit_loss) as avg_profit
            FROM trades
            WHERE is_win IS NOT NULL AND signal_type IS NOT NULL
        """

        if days:
            query += f" AND timestamp >= datetime('now', '-{days} days')"

        query += " GROUP BY signal_type"

        cursor.execute(query)

        results = {}
        for row in cursor.fetchall():
            signal_type = row['signal_type']
            total = row['total']
            wins = row['wins'] or 0

            results[signal_type] = {
                "total": total,
                "wins": wins,
                "losses": total - wins,
                "win_rate": wins / total if total > 0 else 0.0,
                "avg_profit": row['avg_profit'] or 0.0
            }

        return results
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_performance_metrics.py -v`
Expected: `3 passed`

**Step 5: Commit**

```bash
git add polymarket/performance/metrics.py tests/test_performance_metrics.py
git commit -m "feat(performance): add metrics calculator for win rate, profit, and signal performance

- Win rate calculation with time filtering
- Total profit/loss aggregation
- Signal type performance breakdown
- Basis for reflection engine analysis

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 7: Create Reflection Engine (OpenAI Integration)

**Files:**
- Create: `polymarket/performance/reflection.py`
- Create: `tests/test_performance_reflection.py`

**Step 1: Write the failing test**

```python
# tests/test_performance_reflection.py
import pytest
from unittest.mock import Mock, AsyncMock, patch
from polymarket.performance.reflection import ReflectionEngine
from polymarket.performance.database import PerformanceDatabase
from polymarket.config import Settings

@pytest.fixture
def mock_settings():
    """Mock settings with OpenAI key."""
    settings = Mock(spec=Settings)
    settings.openai_api_key = "test-key"
    settings.openai_model = "gpt-4o-mini"
    settings.bot_confidence_threshold = 0.75
    settings.bot_max_position_dollars = 10.0
    settings.bot_max_exposure_percent = 0.50
    return settings

@pytest.fixture
def db_with_trades():
    """Database with sample trades for analysis."""
    db = PerformanceDatabase(":memory:")
    # Add trades...
    return db

@pytest.mark.asyncio
async def test_generate_reflection_prompt(mock_settings, db_with_trades):
    """Test reflection prompt generation."""
    engine = ReflectionEngine(db_with_trades, mock_settings)

    prompt = await engine._generate_prompt(trade_count=10)

    assert "trading performance" in prompt.lower()
    assert "win rate" in prompt.lower()
    assert "recommendations" in prompt.lower()
    assert "0.75" in prompt  # confidence threshold

@pytest.mark.asyncio
async def test_analyze_performance(mock_settings, db_with_trades):
    """Test full reflection analysis."""
    engine = ReflectionEngine(db_with_trades, mock_settings)

    # Mock OpenAI response
    mock_response = {
        "insights": ["Test insight 1", "Test insight 2"],
        "patterns": {
            "winning": ["Pattern 1"],
            "losing": ["Pattern 2"]
        },
        "recommendations": [
            {
                "parameter": "bot_confidence_threshold",
                "current": 0.75,
                "recommended": 0.70,
                "reason": "Test reason",
                "tier": 2,
                "expected_impact": "Test impact"
            }
        ]
    }

    with patch.object(engine, '_call_openai', new_callable=AsyncMock) as mock_openai:
        mock_openai.return_value = mock_response

        insights = await engine.analyze_performance(
            trigger_type="10_trades",
            trades_analyzed=10
        )

        assert insights is not None
        assert "insights" in insights
        assert len(insights["insights"]) == 2
        assert len(insights["recommendations"]) == 1
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_performance_reflection.py::test_generate_reflection_prompt -v`
Expected: `ModuleNotFoundError: No module named 'polymarket.performance.reflection'`

**Step 3: Write minimal implementation**

```python
# polymarket/performance/reflection.py
"""Reflection engine for AI-powered performance analysis."""

import json
from typing import Dict, Optional
from datetime import datetime
import structlog
from openai import AsyncOpenAI

from polymarket.performance.database import PerformanceDatabase
from polymarket.performance.metrics import MetricsCalculator
from polymarket.config import Settings

logger = structlog.get_logger()


class ReflectionEngine:
    """AI-powered trading performance analysis."""

    def __init__(self, db: PerformanceDatabase, settings: Settings):
        """
        Initialize reflection engine.

        Args:
            db: PerformanceDatabase instance
            settings: Bot settings
        """
        self.db = db
        self.settings = settings
        self.metrics = MetricsCalculator(db)
        self._client: Optional[AsyncOpenAI] = None

    def _get_client(self) -> AsyncOpenAI:
        """Lazy init OpenAI client."""
        if self._client is None:
            if not self.settings.openai_api_key:
                raise ValueError("OPENAI_API_KEY not configured")
            self._client = AsyncOpenAI(api_key=self.settings.openai_api_key)
        return self._client

    async def analyze_performance(
        self,
        trigger_type: str,
        trades_analyzed: int
    ) -> Dict:
        """
        Analyze trading performance and generate insights.

        Args:
            trigger_type: '3_losses', '10_trades', 'end_of_day'
            trades_analyzed: Number of trades to analyze

        Returns:
            Dict with insights and recommendations
        """
        try:
            # Generate prompt
            prompt = await self._generate_prompt(trade_count=trades_analyzed)

            # Call OpenAI
            insights = await self._call_openai(prompt)

            # Store in database
            self._store_reflection(trigger_type, trades_analyzed, insights)

            logger.info(
                "Reflection complete",
                trigger_type=trigger_type,
                trades_analyzed=trades_analyzed,
                insights_count=len(insights.get("insights", []))
            )

            return insights

        except Exception as e:
            logger.error("Reflection failed", error=str(e), trigger_type=trigger_type)
            return {"insights": [], "patterns": {}, "recommendations": []}

    async def _generate_prompt(self, trade_count: int) -> str:
        """Generate reflection prompt with current metrics."""
        # Calculate metrics
        win_rate = self.metrics.calculate_win_rate(days=7)
        total_profit = self.metrics.calculate_total_profit(days=7)
        signal_perf = self.metrics.calculate_signal_performance(days=7)

        # Format signal performance
        signal_breakdown = "\n".join([
            f"- {sig}: {perf['win_rate']*100:.1f}% win rate ({perf['wins']}W-{perf['losses']}L), avg profit ${perf['avg_profit']:.2f}"
            for sig, perf in signal_perf.items()
        ])

        prompt = f"""You are analyzing your own Polymarket trading performance as a self-improving AI.

**Recent Performance (Last 7 Days):**
- Trades: {trade_count}
- Win Rate: {win_rate*100:.1f}%
- Total Profit: ${total_profit:.2f}

**Signal Performance:**
{signal_breakdown}

**Current Parameters:**
- Confidence Threshold: {self.settings.bot_confidence_threshold}
- Max Position: ${self.settings.bot_max_position_dollars}
- Max Exposure: {self.settings.bot_max_exposure_percent*100:.0f}%

**Analysis Tasks:**
1. What patterns led to winning trades?
2. What mistakes are being repeated?
3. Should confidence threshold be adjusted? Why?
4. Which signal types should be trusted more/less?
5. Recommend 2-3 specific parameter adjustments with detailed reasoning.

**Output Format (JSON):**
{{
  "insights": [
    "Specific actionable insight 1",
    "Specific actionable insight 2"
  ],
  "patterns": {{
    "winning": ["Pattern that leads to wins"],
    "losing": ["Pattern that leads to losses"]
  }},
  "recommendations": [
    {{
      "parameter": "bot_confidence_threshold",
      "current": {self.settings.bot_confidence_threshold},
      "recommended": 0.70,
      "reason": "Detailed reasoning",
      "tier": 2,
      "expected_impact": "Impact description"
    }}
  ]
}}
"""
        return prompt

    async def _call_openai(self, prompt: str) -> Dict:
        """Call OpenAI API for reflection."""
        client = self._get_client()

        response = await client.chat.completions.create(
            model=self.settings.openai_model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a self-improving trading AI analyzing your own performance. Always return valid JSON."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=2000,
            response_format={"type": "json_object"}
        )

        content = response.choices[0].message.content
        insights = json.loads(content)

        return insights

    def _store_reflection(self, trigger_type: str, trades_analyzed: int, insights: Dict):
        """Store reflection results to database."""
        cursor = self.db.conn.cursor()

        cursor.execute("""
            INSERT INTO reflections (timestamp, trigger_type, trades_analyzed, insights, adjustments_made)
            VALUES (?, ?, ?, ?, ?)
        """, (
            datetime.now(),
            trigger_type,
            trades_analyzed,
            json.dumps(insights),
            None  # Adjustments filled in later by Parameter Adjuster
        ))

        self.db.conn.commit()
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_performance_reflection.py -v`
Expected: `2 passed`

**Step 5: Commit**

```bash
git add polymarket/performance/reflection.py tests/test_performance_reflection.py
git commit -m "feat(performance): add reflection engine with OpenAI analysis

- Generates performance analysis prompts
- Calls OpenAI for insights and recommendations
- Stores reflections in database
- Returns structured insights for action

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Phase 3: Telegram Bot (Notifications & Control)

### Task 8: Set up Telegram Bot Infrastructure

**Files:**
- Create: `polymarket/telegram/__init__.py`
- Create: `polymarket/telegram/bot.py`
- Create: `tests/test_telegram_bot.py`
- Modify: `.env.example`

**Step 1: Write the failing test**

```python
# tests/test_telegram_bot.py
import pytest
from unittest.mock import Mock, AsyncMock, patch
from polymarket.telegram.bot import TelegramBot
from polymarket.config import Settings

@pytest.fixture
def mock_settings():
    """Mock settings with Telegram config."""
    settings = Mock(spec=Settings)
    settings.telegram_bot_token = "test-token"
    settings.telegram_chat_id = "test-chat-id"
    settings.telegram_enabled = True
    return settings

@pytest.mark.asyncio
async def test_send_notification(mock_settings):
    """Test sending a notification."""
    bot = TelegramBot(mock_settings)

    with patch.object(bot, '_send_message', new_callable=AsyncMock) as mock_send:
        await bot.send_trade_alert(
            market_slug="btc-updown-15m-123",
            action="NO",
            confidence=1.0,
            position_size=5.0,
            price=0.52,
            reasoning="Test reasoning"
        )

        mock_send.assert_called_once()
        args = mock_send.call_args[0]
        message = args[0]

        assert "Trade Executed" in message
        assert "NO" in message
        assert "0.52" in message

@pytest.mark.asyncio
async def test_disabled_telegram(mock_settings):
    """Test that disabled telegram doesn't send."""
    mock_settings.telegram_enabled = False
    bot = TelegramBot(mock_settings)

    # Should not raise, just return silently
    await bot.send_trade_alert(
        market_slug="test",
        action="YES",
        confidence=0.8,
        position_size=5.0,
        price=0.50,
        reasoning="Test"
    )
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_telegram_bot.py::test_send_notification -v`
Expected: `ModuleNotFoundError: No module named 'polymarket.telegram'`

**Step 3: Install telegram dependency**

```bash
pip install python-telegram-bot
echo "python-telegram-bot>=20.0" >> requirements.txt
```

**Step 4: Write minimal implementation**

```python
# polymarket/telegram/__init__.py
"""Telegram bot for notifications and control."""

from polymarket.telegram.bot import TelegramBot

__all__ = ["TelegramBot"]
```

```python
# polymarket/telegram/bot.py
"""Telegram bot implementation."""

from typing import Optional
import structlog
from telegram import Bot
from telegram.constants import ParseMode

from polymarket.config import Settings

logger = structlog.get_logger()


class TelegramBot:
    """Telegram bot for notifications and interactive control."""

    def __init__(self, settings: Settings):
        """
        Initialize Telegram bot.

        Args:
            settings: Bot settings with Telegram config
        """
        self.settings = settings
        self._bot: Optional[Bot] = None

        if settings.telegram_enabled:
            if not settings.telegram_bot_token:
                raise ValueError("TELEGRAM_BOT_TOKEN not configured")
            if not settings.telegram_chat_id:
                raise ValueError("TELEGRAM_CHAT_ID not configured")

            self._bot = Bot(token=settings.telegram_bot_token)
            logger.info("Telegram bot initialized")
        else:
            logger.info("Telegram bot disabled")

    async def send_trade_alert(
        self,
        market_slug: str,
        action: str,
        confidence: float,
        position_size: float,
        price: float,
        reasoning: str
    ):
        """Send trade execution alert."""
        if not self._bot:
            return

        message = f"""🎯 **Trade Executed**

Market: `{market_slug}`
Action: **{action}** ({"UP" if action == "YES" else "DOWN"})
Confidence: {confidence*100:.0f}%
Position: ${position_size:.2f} @ {price:.2f}

Reasoning: {reasoning}

Expected profit: ~${position_size * (1/price - 1):.2f} if correct
"""

        await self._send_message(message)

    async def _send_message(self, text: str):
        """Send message to configured chat."""
        try:
            await self._bot.send_message(
                chat_id=self.settings.telegram_chat_id,
                text=text,
                parse_mode=ParseMode.MARKDOWN
            )
            logger.debug("Telegram message sent")
        except Exception as e:
            logger.error("Failed to send Telegram message", error=str(e))
```

**Step 5: Update .env.example**

```bash
# .env.example (add at end)
# Telegram Bot Configuration
TELEGRAM_ENABLED=false
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here
```

**Step 6: Update config.py to include Telegram settings**

```python
# polymarket/config.py (add to Settings class)

# === Telegram Configuration ===
telegram_enabled: bool = field(
    default_factory=lambda: os.getenv("TELEGRAM_ENABLED", "false").lower() == "true"
)
telegram_bot_token: str | None = field(
    default_factory=lambda: os.getenv("TELEGRAM_BOT_TOKEN")
)
telegram_chat_id: str | None = field(
    default_factory=lambda: os.getenv("TELEGRAM_CHAT_ID")
)
```

**Step 7: Run test to verify it passes**

Run: `pytest tests/test_telegram_bot.py -v`
Expected: `2 passed`

**Step 8: Commit**

```bash
git add polymarket/telegram/ tests/test_telegram_bot.py requirements.txt .env.example polymarket/config.py
git commit -m "feat(telegram): add Telegram bot infrastructure with trade alerts

- TelegramBot class with notification methods
- Trade execution alerts with formatting
- Graceful handling when disabled
- Config integration

Depends: python-telegram-bot>=20.0

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Remaining Tasks (Summary)

Due to length constraints, here's the high-level outline for remaining phases:

### Phase 4: Parameter Adjuster (Tasks 9-12)
- **Task 9:** Create parameter bounds and validation
- **Task 10:** Implement Tier 1 (auto-adjust) logic
- **Task 11:** Implement Tier 2 (approval workflow) with Telegram buttons
- **Task 12:** Implement Tier 3 (emergency pause) triggers

### Phase 5: Cleanup System (Tasks 13-15)
- **Task 13:** Implement archival logic (SQLite → compressed JSON)
- **Task 14:** Implement weekly cleanup job with cron scheduling
- **Task 15:** Add emergency cleanup triggers (disk space monitoring)

### Phase 6: Integration & Testing (Tasks 16-18)
- **Task 16:** End-to-end integration test (trade → reflect → adjust)
- **Task 17:** Load testing and error resilience
- **Task 18:** Production deployment with phased rollout

---

## Testing Checklist

**Unit Tests:**
- [ ] Database operations (CRUD, queries, indexes)
- [ ] Metrics calculations (win rate, profit, signals)
- [ ] Reflection prompt generation
- [ ] Parameter validation and bounds checking
- [ ] Telegram message formatting

**Integration Tests:**
- [ ] Trade logging → database → metrics
- [ ] Reflection trigger → OpenAI → database
- [ ] Parameter adjustment → config update → database
- [ ] Telegram approval workflow (mock)

**Manual Testing:**
- [ ] Run bot with performance tracking enabled
- [ ] Trigger reflection manually
- [ ] Test Telegram notifications
- [ ] Test parameter adjustments
- [ ] Verify cleanup job execution

---

## Deployment Plan

**Phase 1: Database & Logging** (Week 1)
- Deploy performance tracker
- Monitor for errors
- Verify data quality

**Phase 2: Reflection Engine** (Week 2)
- Enable insights generation
- Review AI output quality
- Tune prompts if needed

**Phase 3: Telegram Notifications** (Week 3)
- Enable notifications only
- Test interactive commands
- No auto-adjustments yet

**Phase 4: Auto-Adjustments** (Week 4)
- Enable Tier 1 only
- Monitor adjustments closely
- Rollback if issues

**Phase 5: Full System** (Week 5+)
- Enable all tiers
- Enable cleanup
- Full production

---

## Success Criteria

**Short-term (Week 1-4):**
- [ ] 100% of trades logged successfully
- [ ] Reflection runs on schedule (0 missed)
- [ ] Telegram messages delivered (<5% fail rate)
- [ ] No trading interruptions

**Medium-term (Month 2-3):**
- [ ] Win rate improvement: +5% vs baseline
- [ ] Profit improvement: +10% vs baseline
- [ ] Missed opportunities: -30% reduction

**Long-term (Month 4+):**
- [ ] Self-optimizing without human input
- [ ] Adaptive to market conditions
- [ ] Consistent profitability

---

## Notes

- **TDD Approach:** Every feature starts with a failing test
- **Incremental:** Small commits, frequent deploys
- **DRY:** Reuse database/metrics logic across components
- **YAGNI:** Only build what's in the design, no extras
- **Error Handling:** Never block trading on reflection/logging failures
- **Safety:** All parameter adjustments have bounds and rollback

---

**Plan complete and ready for execution!**
