# Contrarian RSI Strategy Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement contrarian RSI strategy that detects extreme technical divergences (RSI < 10 or > 90) from crowd consensus (odds > 65%) and alerts AI to potential mean-reversion opportunities.

**Architecture:** Create ContrarianSignal dataclass and detection function in new contrarian.py module. Integrate into auto_trade.py trading pipeline after technical analysis, adjusting movement threshold to $50 when detected. Add dual AI integration: explicit prompt flag + sentiment scoring. Track in database.

**Tech Stack:** Python 3.12, pytest, dataclasses, structlog, SQLite

---

## Task 1: ContrarianSignal Data Model

**Files:**
- Modify: `polymarket/models.py` (add ContrarianSignal dataclass)
- Test: `tests/test_contrarian_models.py` (new file)

**Step 1: Write the failing test**

```python
# tests/test_contrarian_models.py
from polymarket.models import ContrarianSignal

def test_contrarian_signal_oversold_creation():
    """Test OVERSOLD_REVERSAL signal creation."""
    signal = ContrarianSignal(
        type="OVERSOLD_REVERSAL",
        suggested_direction="UP",
        rsi=9.5,
        crowd_direction="DOWN",
        crowd_confidence=0.72,
        confidence=0.95,
        reasoning="Extreme oversold (RSI 9.5) + strong DOWN consensus (72%) = UP reversal likely"
    )

    assert signal.type == "OVERSOLD_REVERSAL"
    assert signal.suggested_direction == "UP"
    assert signal.rsi == 9.5
    assert signal.crowd_direction == "DOWN"
    assert signal.crowd_confidence == 0.72
    assert signal.confidence == 0.95
    assert "oversold" in signal.reasoning.lower()

def test_contrarian_signal_overbought_creation():
    """Test OVERBOUGHT_REVERSAL signal creation."""
    signal = ContrarianSignal(
        type="OVERBOUGHT_REVERSAL",
        suggested_direction="DOWN",
        rsi=92.0,
        crowd_direction="UP",
        crowd_confidence=0.70,
        confidence=0.92,
        reasoning="Extreme overbought (RSI 92.0) + strong UP consensus (70%) = DOWN reversal likely"
    )

    assert signal.type == "OVERBOUGHT_REVERSAL"
    assert signal.suggested_direction == "DOWN"
    assert signal.rsi == 92.0
    assert signal.crowd_direction == "UP"
```

**Step 2: Run test to verify it fails**

Run: `cd /root/polymarket-scripts/.worktrees/contrarian-rsi-strategy && python3 -m pytest tests/test_contrarian_models.py::test_contrarian_signal_oversold_creation -v`
Expected: FAIL with "cannot import name 'ContrarianSignal'"

**Step 3: Write minimal implementation**

```python
# polymarket/models.py (add to end of file)

@dataclass
class ContrarianSignal:
    """Signal indicating extreme RSI divergence from crowd consensus."""
    type: Literal["OVERSOLD_REVERSAL", "OVERBOUGHT_REVERSAL"]
    suggested_direction: Literal["UP", "DOWN"]
    rsi: float
    crowd_direction: Literal["UP", "DOWN"]
    crowd_confidence: float  # Crowd's odds (0-1)
    confidence: float  # Our confidence in reversal (0-1)
    reasoning: str
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_contrarian_models.py -v`
Expected: PASS (2 tests)

**Step 5: Commit**

```bash
cd /root/polymarket-scripts/.worktrees/contrarian-rsi-strategy
git add polymarket/models.py tests/test_contrarian_models.py
git commit -m "feat: add ContrarianSignal dataclass

- Supports OVERSOLD_REVERSAL and OVERBOUGHT_REVERSAL types
- Tracks RSI, crowd consensus, and reversal confidence
- Includes human-readable reasoning"
```

---

## Task 2: Contrarian Detection Logic

**Files:**
- Create: `polymarket/trading/contrarian.py`
- Test: `tests/test_contrarian_detection.py` (new file)

**Step 1: Write the failing tests**

```python
# tests/test_contrarian_detection.py
from polymarket.trading.contrarian import detect_contrarian_setup

def test_oversold_reversal_detected():
    """RSI < 10 + DOWN odds > 65% = OVERSOLD_REVERSAL."""
    signal = detect_contrarian_setup(
        rsi=9.5,
        yes_odds=0.28,  # UP odds
        no_odds=0.72    # DOWN odds
    )

    assert signal is not None
    assert signal.type == "OVERSOLD_REVERSAL"
    assert signal.suggested_direction == "UP"
    assert signal.rsi == 9.5
    assert signal.crowd_direction == "DOWN"
    assert signal.crowd_confidence == 0.72
    assert signal.confidence >= 0.90  # High confidence
    assert "oversold" in signal.reasoning.lower()

def test_overbought_reversal_detected():
    """RSI > 90 + UP odds > 65% = OVERBOUGHT_REVERSAL."""
    signal = detect_contrarian_setup(
        rsi=92.0,
        yes_odds=0.70,  # UP odds
        no_odds=0.30    # DOWN odds
    )

    assert signal is not None
    assert signal.type == "OVERBOUGHT_REVERSAL"
    assert signal.suggested_direction == "DOWN"
    assert signal.rsi == 92.0
    assert signal.crowd_direction == "UP"
    assert signal.crowd_confidence == 0.70
    assert signal.confidence >= 0.90

def test_no_contrarian_signal_rsi_not_extreme():
    """RSI 50 = no signal (not extreme)."""
    signal = detect_contrarian_setup(
        rsi=50.0,
        yes_odds=0.30,
        no_odds=0.70
    )

    assert signal is None

def test_no_contrarian_signal_odds_not_extreme():
    """RSI 5 but odds 50/50 = no signal (crowd not consensus)."""
    signal = detect_contrarian_setup(
        rsi=5.0,
        yes_odds=0.50,
        no_odds=0.50
    )

    assert signal is None

def test_oversold_edge_case_rsi_10():
    """RSI exactly 10 should NOT trigger (< 10 required)."""
    signal = detect_contrarian_setup(
        rsi=10.0,
        yes_odds=0.30,
        no_odds=0.70
    )

    assert signal is None

def test_oversold_edge_case_odds_65():
    """Odds exactly 65% should NOT trigger (> 65% required)."""
    signal = detect_contrarian_setup(
        rsi=9.0,
        yes_odds=0.35,
        no_odds=0.65
    )

    assert signal is None

def test_confidence_increases_with_extreme_rsi():
    """Lower RSI = higher confidence."""
    signal_9 = detect_contrarian_setup(rsi=9.0, yes_odds=0.25, no_odds=0.75)
    signal_5 = detect_contrarian_setup(rsi=5.0, yes_odds=0.25, no_odds=0.75)

    assert signal_5.confidence > signal_9.confidence
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_contrarian_detection.py -v`
Expected: FAIL with "cannot import name 'detect_contrarian_setup'"

**Step 3: Write minimal implementation**

```python
# polymarket/trading/contrarian.py
"""
Contrarian RSI Strategy
Detects extreme technical divergences from crowd consensus.
"""
from typing import Optional
from polymarket.models import ContrarianSignal

def detect_contrarian_setup(
    rsi: float,
    yes_odds: float,  # UP odds (best_bid)
    no_odds: float    # DOWN odds (1 - best_bid)
) -> Optional[ContrarianSignal]:
    """
    Detect extreme RSI divergence from crowd consensus.

    Args:
        rsi: RSI indicator value (0-100)
        yes_odds: UP token odds (0-1)
        no_odds: DOWN token odds (0-1)

    Returns:
        ContrarianSignal if conditions met, None otherwise

    Detection Rules:
        - OVERSOLD: RSI < 10 AND DOWN odds > 65%
        - OVERBOUGHT: RSI > 90 AND UP odds > 65%
    """
    # OVERSOLD: RSI extremely low, crowd betting DOWN
    if rsi < 10 and no_odds > 0.65:
        # Higher confidence for more extreme RSI
        confidence = 0.90 + (10 - rsi) * 0.01
        confidence = min(confidence, 1.0)  # Cap at 1.0

        return ContrarianSignal(
            type="OVERSOLD_REVERSAL",
            suggested_direction="UP",
            rsi=rsi,
            crowd_direction="DOWN",
            crowd_confidence=no_odds,
            confidence=confidence,
            reasoning=f"Extreme oversold (RSI {rsi:.1f}) + strong DOWN consensus ({no_odds:.0%}) = UP reversal likely"
        )

    # OVERBOUGHT: RSI extremely high, crowd betting UP
    if rsi > 90 and yes_odds > 0.65:
        # Higher confidence for more extreme RSI
        confidence = 0.90 + (rsi - 90) * 0.01
        confidence = min(confidence, 1.0)  # Cap at 1.0

        return ContrarianSignal(
            type="OVERBOUGHT_REVERSAL",
            suggested_direction="DOWN",
            rsi=rsi,
            crowd_direction="UP",
            crowd_confidence=yes_odds,
            confidence=confidence,
            reasoning=f"Extreme overbought (RSI {rsi:.1f}) + strong UP consensus ({yes_odds:.0%}) = DOWN reversal likely"
        )

    return None
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_contrarian_detection.py -v`
Expected: PASS (7 tests)

**Step 5: Commit**

```bash
git add polymarket/trading/contrarian.py tests/test_contrarian_detection.py
git commit -m "feat: add contrarian setup detection logic

- Detects OVERSOLD_REVERSAL (RSI < 10, DOWN odds > 65%)
- Detects OVERBOUGHT_REVERSAL (RSI > 90, UP odds > 65%)
- Confidence scales with RSI extremeness
- Returns None if conditions not met
- Comprehensive edge case testing"
```

---

## Task 3: Movement Threshold Adjustment

**Files:**
- Modify: `scripts/auto_trade.py:979-1000` (movement threshold check)
- Test: `tests/test_contrarian_threshold.py` (new file)

**Step 1: Write the failing test**

```python
# tests/test_contrarian_threshold.py
from polymarket.models import ContrarianSignal

def test_movement_threshold_reduced_with_contrarian():
    """Movement threshold should be $50 when contrarian signal present."""
    contrarian_signal = ContrarianSignal(
        type="OVERSOLD_REVERSAL",
        suggested_direction="UP",
        rsi=9.5,
        crowd_direction="DOWN",
        crowd_confidence=0.72,
        confidence=0.95,
        reasoning="Test"
    )

    # Function to test (will be extracted from auto_trade.py)
    threshold = get_movement_threshold(contrarian_signal)
    assert threshold == 50

def test_movement_threshold_normal_without_contrarian():
    """Movement threshold should be $100 without contrarian signal."""
    threshold = get_movement_threshold(None)
    assert threshold == 100
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_contrarian_threshold.py -v`
Expected: FAIL with "cannot import name 'get_movement_threshold'"

**Step 3: Write minimal implementation**

First, extract threshold logic into testable function:

```python
# polymarket/trading/contrarian.py (add to end)

def get_movement_threshold(contrarian_signal: Optional[ContrarianSignal]) -> int:
    """
    Get BTC movement threshold based on contrarian signal presence.

    Args:
        contrarian_signal: Contrarian signal if detected, None otherwise

    Returns:
        Movement threshold in USD ($50 if contrarian, $100 otherwise)
    """
    if contrarian_signal:
        return 50  # Reduced threshold for reversals
    return 100  # Normal threshold
```

Then modify auto_trade.py to use this function:

```python
# scripts/auto_trade.py (around line 979)
# OLD CODE:
# MIN_MOVEMENT_THRESHOLD = 100  # $100 minimum BTC movement

# NEW CODE:
from polymarket.trading.contrarian import get_movement_threshold

# Calculate threshold based on contrarian signal
MIN_MOVEMENT_THRESHOLD = get_movement_threshold(contrarian_signal)
if contrarian_signal:
    logger.info(
        "Contrarian setup - reducing movement threshold",
        default_threshold="$100",
        contrarian_threshold="$50",
        reasoning="Reversals start with small movements"
    )
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_contrarian_threshold.py -v`
Expected: PASS (2 tests)

**Step 5: Commit**

```bash
git add polymarket/trading/contrarian.py scripts/auto_trade.py tests/test_contrarian_threshold.py
git commit -m "feat: add dynamic movement threshold for contrarian signals

- Reduce threshold to $50 when contrarian detected
- Keep $100 for normal trades
- Extract testable get_movement_threshold() function
- Log threshold adjustment with reasoning"
```

---

## Task 4: Integration into Auto-Trade Pipeline

**Files:**
- Modify: `scripts/auto_trade.py:630-650` (after technical analysis)
- Test: Integration tested manually (existing test suite)

**Step 1: Add contrarian detection call**

```python
# scripts/auto_trade.py (after line 630 - after technical indicators calculated)

# Step 3.5: Contrarian Signal Detection
contrarian_signal = None
try:
    from polymarket.trading.contrarian import detect_contrarian_setup

    # Calculate odds from market
    yes_odds = market.best_bid if market.best_bid else 0.50
    no_odds = 1.0 - yes_odds

    contrarian_signal = detect_contrarian_setup(
        rsi=indicators.rsi,
        yes_odds=yes_odds,
        no_odds=no_odds
    )

    if contrarian_signal:
        logger.info(
            "Contrarian signal detected",
            type=contrarian_signal.type,
            rsi=f"{contrarian_signal.rsi:.1f}",
            suggested_direction=contrarian_signal.suggested_direction,
            crowd_direction=contrarian_signal.crowd_direction,
            crowd_confidence=f"{contrarian_signal.crowd_confidence:.0%}",
            confidence=f"{contrarian_signal.confidence:.0%}"
        )
except Exception as e:
    logger.warning("Contrarian detection failed, continuing without", error=str(e))
    contrarian_signal = None
```

**Step 2: Verify integration**

Run: `cd /root/polymarket-scripts/.worktrees/contrarian-rsi-strategy && python3 -c "from scripts.auto_trade import AutoTrader; print('Import successful')"`
Expected: "Import successful"

**Step 3: Commit**

```bash
git add scripts/auto_trade.py
git commit -m "feat: integrate contrarian detection into trading pipeline

- Call detect_contrarian_setup() after technical analysis
- Calculate odds from market.best_bid
- Log contrarian signal details
- Graceful fallback if detection fails"
```

---

## Task 5: AI Prompt Enhancement (Explicit Flag)

**Files:**
- Modify: `polymarket/trading/ai_decision.py` (add contrarian context to prompt)
- Test: Manual verification (check AI prompt includes contrarian flag)

**Step 1: Find AI prompt construction**

Search for where AI prompt is built:
```bash
cd /root/polymarket-scripts/.worktrees/contrarian-rsi-strategy
grep -n "def make_decision\|prompt.*=" polymarket/trading/ai_decision.py | head -20
```

**Step 2: Add contrarian flag to prompt**

```python
# polymarket/trading/ai_decision.py (in make_decision method, after building base prompt)

# Add contrarian signal context if present
if contrarian_signal:
    prompt += f"""

🔥 CONTRARIAN SETUP DETECTED 🔥

Type: {contrarian_signal.type}
RSI: {contrarian_signal.rsi:.1f} ({"EXTREMELY OVERSOLD" if contrarian_signal.rsi < 10 else "EXTREMELY OVERBOUGHT"})
Crowd Betting: {contrarian_signal.crowd_direction} at {contrarian_signal.crowd_confidence:.0%} odds
Contrarian Suggestion: BET {contrarian_signal.suggested_direction}

Reasoning: {contrarian_signal.reasoning}

⚠️ This is a mean-reversion signal. The crowd is heavily positioned for {contrarian_signal.crowd_direction},
but extreme technical indicators suggest imminent reversal to {contrarian_signal.suggested_direction}.

Confidence: {contrarian_signal.confidence:.0%}

Consider:
- RSI extremes (< 10 or > 90) often precede sharp reversals
- Crowd consensus can indicate exhaustion of move
- Mean reversion trades have favorable risk/reward at extremes
"""
```

**Step 3: Update method signature**

```python
# polymarket/trading/ai_decision.py (method signature)

async def make_decision(
    self,
    market_data: dict,
    btc_data: BTCData,
    technical: TechnicalIndicators,
    sentiment_data: AggregatedSentiment,
    price_to_beat: Optional[float] = None,
    contrarian_signal: Optional[ContrarianSignal] = None,  # NEW
    # ... other parameters
) -> Decision:
```

**Step 4: Update call site in auto_trade.py**

```python
# scripts/auto_trade.py (where make_decision is called, around line 1231)

decision = await self.ai_service.make_decision(
    market_data=market_dict,
    btc_data=btc_data,
    technical=indicators,
    sentiment_data=aggregated_sentiment,
    price_to_beat=price_to_beat,
    contrarian_signal=contrarian_signal,  # NEW
    # ... other parameters
)
```

**Step 5: Commit**

```bash
git add polymarket/trading/ai_decision.py scripts/auto_trade.py
git commit -m "feat: add contrarian flag to AI prompt

- Add explicit 🔥 CONTRARIAN SETUP 🔥 section to prompt
- Include RSI, crowd consensus, suggested direction
- Provide mean-reversion context and reasoning
- Pass contrarian_signal to make_decision() method"
```

---

## Task 6: Sentiment Integration

**Files:**
- Modify: `polymarket/trading/signal_aggregator.py` (add contrarian to signals list)
- Test: `tests/test_contrarian_sentiment.py` (new file)

**Step 1: Write the failing test**

```python
# tests/test_contrarian_sentiment.py
from polymarket.models import ContrarianSignal, Signal
from polymarket.trading.signal_aggregator import SignalAggregator

def test_contrarian_signal_added_to_aggregation():
    """Contrarian signal should be included in aggregated sentiment."""
    aggregator = SignalAggregator()

    contrarian_signal = ContrarianSignal(
        type="OVERSOLD_REVERSAL",
        suggested_direction="UP",
        rsi=9.5,
        crowd_direction="DOWN",
        crowd_confidence=0.72,
        confidence=0.95,
        reasoning="Test"
    )

    # Convert contrarian to regular signal format
    signal = aggregator.contrarian_to_signal(contrarian_signal)

    assert signal.name == "contrarian_rsi"
    assert signal.score == +1.0  # UP direction
    assert signal.confidence == 0.95
    assert signal.weight == 2.0  # High weight for contrarian

def test_contrarian_down_signal():
    """OVERBOUGHT_REVERSAL should create negative score (DOWN)."""
    aggregator = SignalAggregator()

    contrarian_signal = ContrarianSignal(
        type="OVERBOUGHT_REVERSAL",
        suggested_direction="DOWN",
        rsi=92.0,
        crowd_direction="UP",
        crowd_confidence=0.70,
        confidence=0.92,
        reasoning="Test"
    )

    signal = aggregator.contrarian_to_signal(contrarian_signal)

    assert signal.score == -1.0  # DOWN direction
    assert signal.confidence == 0.92
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_contrarian_sentiment.py -v`
Expected: FAIL with "no attribute 'contrarian_to_signal'"

**Step 3: Write minimal implementation**

```python
# polymarket/trading/signal_aggregator.py (add method)

def contrarian_to_signal(self, contrarian: ContrarianSignal) -> Signal:
    """
    Convert ContrarianSignal to regular Signal for aggregation.

    Args:
        contrarian: Contrarian signal to convert

    Returns:
        Signal with contrarian scoring
    """
    # Score: +1.0 for UP, -1.0 for DOWN
    score = +1.0 if contrarian.suggested_direction == "UP" else -1.0

    return Signal(
        name="contrarian_rsi",
        score=score,
        confidence=contrarian.confidence,
        weight=2.0  # High weight for extreme signals
    )
```

**Step 4: Integrate into auto_trade.py aggregation**

```python
# scripts/auto_trade.py (where signals are aggregated, around line 640-660)

# Add contrarian signal to aggregation if detected
if contrarian_signal:
    from polymarket.trading.signal_aggregator import SignalAggregator
    aggregator = SignalAggregator()
    contrarian_as_signal = aggregator.contrarian_to_signal(contrarian_signal)

    # Add to signals list before aggregation
    signals.append(contrarian_as_signal)

    logger.info(
        "Added contrarian signal to sentiment aggregation",
        score=f"{contrarian_as_signal.score:+.2f}",
        weight=contrarian_as_signal.weight
    )
```

**Step 5: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_contrarian_sentiment.py -v`
Expected: PASS (2 tests)

**Step 6: Commit**

```bash
git add polymarket/trading/signal_aggregator.py scripts/auto_trade.py tests/test_contrarian_sentiment.py
git commit -m "feat: integrate contrarian signal into sentiment aggregation

- Add contrarian_to_signal() converter method
- Contrarian gets high weight (2.0) in aggregation
- UP direction = +1.0 score, DOWN = -1.0
- Include in final_score calculation"
```

---

## Task 7: Database Schema Update

**Files:**
- Modify: `polymarket/performance/database.py` (add contrarian fields to trades table)
- Test: `tests/test_contrarian_database.py` (new file)

**Step 1: Write the failing test**

```python
# tests/test_contrarian_database.py
import sqlite3
from polymarket.performance.database import PerformanceDatabase

def test_contrarian_fields_in_schema():
    """Trades table should have contrarian_detected and contrarian_type fields."""
    db = PerformanceDatabase(":memory:")

    # Query schema
    cursor = db.conn.cursor()
    cursor.execute("PRAGMA table_info(trades)")
    columns = {row[1]: row[2] for row in cursor.fetchall()}

    assert "contrarian_detected" in columns
    assert columns["contrarian_detected"] == "BOOLEAN"

    assert "contrarian_type" in columns
    assert columns["contrarian_type"] == "VARCHAR(50)"

def test_store_trade_with_contrarian():
    """Should store contrarian trade data."""
    db = PerformanceDatabase(":memory:")

    trade_data = {
        "market_id": "test_market",
        "action": "YES",
        "contrarian_detected": True,
        "contrarian_type": "OVERSOLD_REVERSAL",
        # ... other required fields
    }

    # Store trade (method signature may vary)
    # db.log_trade(**trade_data)

    # Verify stored
    cursor = db.conn.cursor()
    cursor.execute("SELECT contrarian_detected, contrarian_type FROM trades WHERE market_id = ?", ("test_market",))
    row = cursor.fetchone()

    assert row[0] == 1  # True as integer
    assert row[1] == "OVERSOLD_REVERSAL"
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_contrarian_database.py -v`
Expected: FAIL (no contrarian fields in schema)

**Step 3: Add migration**

```python
# polymarket/performance/database.py (in _create_tables or _migrate method)

# Add to trades table schema
cursor.execute("""
    ALTER TABLE trades ADD COLUMN contrarian_detected BOOLEAN DEFAULT 0
""")

cursor.execute("""
    ALTER TABLE trades ADD COLUMN contrarian_type VARCHAR(50) DEFAULT NULL
""")

# Or if creating fresh table, add to CREATE TABLE:
"""
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ... (existing fields),
    contrarian_detected BOOLEAN DEFAULT 0,
    contrarian_type VARCHAR(50) DEFAULT NULL,
    ... (more fields)
)
"""
```

**Step 4: Update log_trade method signature**

```python
# polymarket/performance/database.py (log_trade method)

def log_trade(
    self,
    # ... existing parameters
    contrarian_detected: bool = False,
    contrarian_type: Optional[str] = None,
    # ... more parameters
) -> int:
    """Log trade with contrarian data."""
    # ... existing code

    cursor.execute("""
        INSERT INTO trades (..., contrarian_detected, contrarian_type, ...)
        VALUES (..., ?, ?, ...)
    """, (..., contrarian_detected, contrarian_type, ...))
```

**Step 5: Update call site in auto_trade.py**

```python
# scripts/auto_trade.py (where trade is logged, around line 1350)

trade_id = self.performance_tracker.log_trade(
    # ... existing parameters
    contrarian_detected=bool(contrarian_signal),
    contrarian_type=contrarian_signal.type if contrarian_signal else None,
    # ... more parameters
)
```

**Step 6: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_contrarian_database.py -v`
Expected: PASS (2 tests)

**Step 7: Commit**

```bash
git add polymarket/performance/database.py scripts/auto_trade.py tests/test_contrarian_database.py
git commit -m "feat: add contrarian fields to trades database

- Add contrarian_detected BOOLEAN field
- Add contrarian_type VARCHAR(50) field
- Update log_trade() signature
- Enable tracking contrarian trade performance separately"
```

---

## Task 8: AI Decision Logging

**Files:**
- Modify: `scripts/auto_trade.py:1380-1395` (after AI decision)

**Step 1: Add AI acceptance/rejection logging**

```python
# scripts/auto_trade.py (after AI decision is made, around line 1390)

# Log AI's response to contrarian suggestion
if contrarian_signal:
    if decision.action == contrarian_signal.suggested_direction:
        logger.info(
            "AI accepted contrarian suggestion",
            market_id=market.id,
            contrarian_type=contrarian_signal.type,
            suggested=contrarian_signal.suggested_direction,
            ai_action=decision.action,
            ai_confidence=f"{decision.confidence:.2f}"
        )
    else:
        logger.info(
            "AI rejected contrarian suggestion",
            market_id=market.id,
            contrarian_type=contrarian_signal.type,
            suggested=contrarian_signal.suggested_direction,
            ai_action=decision.action,
            ai_reasoning=decision.reasoning[:100]  # First 100 chars
        )
```

**Step 2: Commit**

```bash
git add scripts/auto_trade.py
git commit -m "feat: log AI acceptance/rejection of contrarian signals

- Log when AI follows contrarian suggestion
- Log when AI rejects contrarian suggestion (with reasoning)
- Track contrarian_type for analysis"
```

---

## Task 9: Comprehensive Testing

**Files:**
- Test: `tests/integration/test_contrarian_integration.py` (new file)

**Step 1: Write integration test**

```python
# tests/integration/test_contrarian_integration.py
"""Integration test for contrarian strategy end-to-end."""
import pytest
from polymarket.trading.contrarian import detect_contrarian_setup
from polymarket.models import TechnicalIndicators

@pytest.mark.asyncio
async def test_contrarian_pipeline_oversold():
    """Test complete pipeline with OVERSOLD_REVERSAL signal."""
    # Step 1: Technical indicators show extreme oversold
    indicators = TechnicalIndicators(
        rsi=9.5,
        macd_value=-10.0,
        macd_signal=-8.0,
        macd_histogram=-2.0,
        ema_short=68300.0,
        ema_long=68400.0,
        sma_50=68500.0,
        volume_change=0.0,
        price_velocity=-50.0,
        trend="BEARISH"
    )

    # Step 2: Market shows strong DOWN consensus
    yes_odds = 0.28
    no_odds = 0.72

    # Step 3: Contrarian detection
    signal = detect_contrarian_setup(
        rsi=indicators.rsi,
        yes_odds=yes_odds,
        no_odds=no_odds
    )

    assert signal is not None
    assert signal.type == "OVERSOLD_REVERSAL"
    assert signal.suggested_direction == "UP"

    # Step 4: Movement threshold adjustment
    from polymarket.trading.contrarian import get_movement_threshold
    threshold = get_movement_threshold(signal)
    assert threshold == 50  # Reduced from 100

@pytest.mark.asyncio
async def test_contrarian_pipeline_no_signal():
    """Test pipeline with normal conditions (no contrarian signal)."""
    indicators = TechnicalIndicators(
        rsi=55.0,  # Not extreme
        macd_value=5.0,
        macd_signal=4.0,
        macd_histogram=1.0,
        ema_short=68400.0,
        ema_long=68300.0,
        sma_50=68200.0,
        volume_change=10.0,
        price_velocity=25.0,
        trend="BULLISH"
    )

    yes_odds = 0.60
    no_odds = 0.40

    signal = detect_contrarian_setup(
        rsi=indicators.rsi,
        yes_odds=yes_odds,
        no_odds=no_odds
    )

    assert signal is None

    from polymarket.trading.contrarian import get_movement_threshold
    threshold = get_movement_threshold(signal)
    assert threshold == 100  # Normal threshold
```

**Step 2: Run integration tests**

Run: `python3 -m pytest tests/integration/test_contrarian_integration.py -v`
Expected: PASS (2 tests)

**Step 3: Commit**

```bash
git add tests/integration/test_contrarian_integration.py
git commit -m "test: add end-to-end integration tests for contrarian strategy

- Test OVERSOLD_REVERSAL pipeline
- Test normal conditions (no signal)
- Verify threshold adjustment
- Validate detection logic"
```

---

## Task 10: Run Full Test Suite

**Files:**
- None (verification step)

**Step 1: Run all tests**

Run: `cd /root/polymarket-scripts/.worktrees/contrarian-rsi-strategy && python3 -m pytest tests/ -v --tb=short 2>&1 | tee test_results.txt`

**Step 2: Verify no regressions**

Expected: All existing tests still pass, new tests pass

**Step 3: Fix any failures**

If tests fail, investigate and fix before proceeding.

**Step 4: Commit test results**

```bash
git add test_results.txt
git commit -m "test: verify full test suite passes with contrarian strategy"
```

---

## Task 11: Documentation Update

**Files:**
- Modify: `README_BOT.md` (add contrarian strategy section)

**Step 1: Add section to README_BOT.md**

```markdown
## Contrarian RSI Strategy

The bot includes a mean-reversion strategy that detects extreme RSI divergences from crowd consensus:

### Detection Criteria

**OVERSOLD_REVERSAL (Bet UP):**
- RSI < 10 (extremely oversold)
- DOWN odds > 65% (strong crowd consensus for DOWN)
- Suggests betting UP (contrarian to crowd)

**OVERBOUGHT_REVERSAL (Bet DOWN):**
- RSI > 90 (extremely overbought)
- UP odds > 65% (strong crowd consensus for UP)
- Suggests betting DOWN (contrarian to crowd)

### Strategy Benefits

1. **Mean Reversion Edge:** Extreme RSI levels often precede reversals
2. **Crowd Exhaustion:** Heavy consensus suggests move may be exhausted
3. **Favorable Risk/Reward:** Entry at extremes provides good R:R

### Implementation Details

- **Movement Threshold:** Reduced to $50 (from $100) when contrarian detected
- **AI Integration:** Both explicit flag and sentiment scoring
- **Confidence Scaling:** Higher confidence for more extreme RSI (e.g., RSI 5 > RSI 9)
- **All Filters Active:** Signal lag, volume, regime checks still enforced

### Performance Tracking

Contrarian trades are tracked separately in the database:
- `contrarian_detected`: Boolean flag
- `contrarian_type`: OVERSOLD_REVERSAL or OVERBOUGHT_REVERSAL

Query contrarian performance:
```sql
SELECT
    contrarian_type,
    COUNT(*) as trades,
    SUM(CASE WHEN outcome = action THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as win_rate
FROM trades
WHERE contrarian_detected = 1
GROUP BY contrarian_type;
```

### Example Market

**Market:** btc-updown-15m-1771186500
- RSI: 9.5 (extremely oversold)
- DOWN odds: 72% (strong consensus)
- Result: BTC went UP (contrarian signal was correct)
- This strategy would have detected this opportunity
```

**Step 2: Commit**

```bash
git add README_BOT.md
git commit -m "docs: add contrarian RSI strategy documentation

- Explain detection criteria
- Document benefits and implementation
- Show performance tracking queries
- Include real-world example"
```

---

## Task 12: Final Integration Verification

**Files:**
- None (verification step)

**Step 1: Start bot in test mode**

```bash
cd /root/polymarket-scripts/.worktrees/contrarian-rsi-strategy
# Set test mode in config if needed
python3 scripts/auto_trade.py
```

**Step 2: Monitor logs for contrarian detection**

Watch for:
- "Contrarian signal detected" messages
- "Contrarian setup - reducing movement threshold" messages
- "AI accepted/rejected contrarian suggestion" messages

**Step 3: Verify database schema**

```bash
sqlite3 data/performance.db ".schema trades" | grep contrarian
```

Expected: contrarian_detected and contrarian_type fields present

**Step 4: Create verification report**

```bash
cat > CONTRARIAN_VERIFICATION.md << 'EOF'
# Contrarian Strategy Verification Report

## Tests Passed
- [x] ContrarianSignal dataclass creation
- [x] detect_contrarian_setup() logic
- [x] OVERSOLD_REVERSAL detection
- [x] OVERBOUGHT_REVERSAL detection
- [x] Edge cases (RSI 10, odds 65%)
- [x] Confidence scaling
- [x] Movement threshold adjustment
- [x] AI prompt integration
- [x] Sentiment aggregation
- [x] Database schema update
- [x] Integration tests

## Manual Verification
- [ ] Bot starts successfully
- [ ] Contrarian detection logs appear
- [ ] Movement threshold reduces to $50
- [ ] AI receives contrarian flag
- [ ] Database stores contrarian data

## Performance Tracking
- [ ] Monitor contrarian trades for 24 hours
- [ ] Measure AI acceptance rate
- [ ] Calculate contrarian win rate
- [ ] Compare vs baseline performance

## Next Steps
1. Merge to main branch
2. Deploy to production
3. Monitor contrarian trade performance
4. Adjust thresholds if needed (RSI < 10, odds > 65%)
EOF

git add CONTRARIAN_VERIFICATION.md
git commit -m "docs: add verification checklist"
```

---

## Summary

**Implementation Complete:**

1. ✅ ContrarianSignal dataclass
2. ✅ detect_contrarian_setup() detection logic
3. ✅ Dynamic movement threshold ($50 vs $100)
4. ✅ Auto-trade pipeline integration
5. ✅ AI prompt enhancement (explicit flag)
6. ✅ Sentiment aggregation (scoring)
7. ✅ Database schema update
8. ✅ Comprehensive logging
9. ✅ Test coverage (unit + integration)
10. ✅ Documentation

**Files Modified:**
- `polymarket/models.py` - ContrarianSignal dataclass
- `polymarket/trading/contrarian.py` - Detection logic
- `polymarket/trading/signal_aggregator.py` - Sentiment integration
- `polymarket/trading/ai_decision.py` - AI prompt enhancement
- `polymarket/performance/database.py` - Database schema
- `scripts/auto_trade.py` - Pipeline integration
- `README_BOT.md` - Documentation

**New Test Files:**
- `tests/test_contrarian_models.py`
- `tests/test_contrarian_detection.py`
- `tests/test_contrarian_threshold.py`
- `tests/test_contrarian_sentiment.py`
- `tests/test_contrarian_database.py`
- `tests/integration/test_contrarian_integration.py`

**Total Commits:** 12 (one per task)

---

## Execution Next Steps

Ready to execute this plan! Choose your approach:

**1. Subagent-Driven (this session)** - Fast iteration with review between tasks
**2. Parallel Session (separate)** - Batch execution in dedicated session

Which would you prefer?
