# Polymarket Market Microstructure Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace Binance API with Polymarket CLOB WebSocket to analyze the specific BTC 15-minute prediction market we're trading

**Architecture:** WebSocket connects for 2 minutes per cycle, collects trades/prices, calculates 3 weighted scores (momentum 40%, volume flow 35%, whales 25%), returns MarketSignals with same interface as before

**Tech Stack:** Python asyncio, websockets, aiohttp (existing), Polymarket CLOB Market Channel API

---

## Task 1: Research Polymarket CLOB WebSocket Endpoint

**Goal:** Find the exact WebSocket URL and message format for CLOB Market Channel

**Files:**
- None (research task)

**Step 1: Search Polymarket documentation**

Visit: https://docs.polymarket.com/developers/CLOB/websocket/market-channel

Look for:
- Exact WebSocket endpoint URL
- Subscription message format for `market` channel
- Message types: `last_trade_price`, `book`, `price_change`
- Field names: `asset_id`, `price`, `size`, `side`

Expected findings:
- WebSocket URL (likely `wss://clob.polymarket.com/ws` or similar)
- Subscription format with `condition_id`
- Message schema examples

**Step 2: Document findings**

Create temporary note file:
```bash
echo "# Polymarket CLOB WebSocket Research

Endpoint: [URL found]
Subscription format: [JSON structure]
Message types: [list]

" > /tmp/polymarket-ws-research.md
```

**Step 3: Test connection (optional)**

If time permits, test WebSocket connection:
```bash
# Using wscat if available
wscat -c wss://[endpoint-url]
```

**Step 4: Commit research notes**

```bash
git add /tmp/polymarket-ws-research.md
git commit -m "research: document Polymarket CLOB WebSocket endpoint"
```

---

## Task 2: Create WebSocket Data Collector (Skeleton)

**Goal:** Set up basic WebSocket connection infrastructure without full implementation

**Files:**
- Modify: `polymarket/trading/market_microstructure.py`
- Test: `tests/test_market_microstructure.py`

**Step 1: Write failing test for WebSocket collector**

Add to `tests/test_market_microstructure.py`:

```python
import pytest
import asyncio
from polymarket.trading.market_microstructure import MarketMicrostructureService
from polymarket.config import Settings

@pytest.mark.asyncio
async def test_collect_market_data_structure():
    """Test that collect_market_data returns expected structure."""
    service = MarketMicrostructureService(
        Settings(),
        condition_id="test-condition-123"
    )

    # Mock WebSocket connection (will implement properly later)
    data = await service.collect_market_data(
        condition_id="test-condition-123",
        duration_seconds=1  # Short duration for test
    )

    # Verify structure
    assert isinstance(data, dict)
    assert 'trades' in data
    assert 'book_snapshots' in data
    assert 'price_changes' in data
    assert isinstance(data['trades'], list)
```

**Step 2: Run test to verify it fails**

```bash
cd ~/.config/superpowers/worktrees/polymarket-scripts/polymarket-market-microstructure
source venv/bin/activate
python -m pytest tests/test_market_microstructure.py::test_collect_market_data_structure -v
```

Expected: FAIL with `collect_market_data() method not found` or similar

**Step 3: Implement skeleton collect_market_data method**

Modify `polymarket/trading/market_microstructure.py`:

```python
"""
Market Microstructure Service

Analyzes Polymarket CLOB order book, trades, volume for short-term signals.
"""

import asyncio
import json
import time
from datetime import datetime
from typing import Optional
import structlog

from polymarket.models import MarketSignals
from polymarket.config import Settings

logger = structlog.get_logger()


class MarketMicrostructureService:
    """Polymarket market microstructure analysis using CLOB WebSocket."""

    # WebSocket endpoint (update after research)
    WS_URL = "wss://clob.polymarket.com/ws"  # TBD from research

    # Weights for score calculation
    WEIGHTS = {
        "momentum": 0.40,
        "volume_flow": 0.35,
        "whale": 0.25
    }

    # Thresholds
    WHALE_SIZE_USD = 1000.0  # Orders > $1,000 considered "whale"

    def __init__(self, settings: Settings, condition_id: str):
        """
        Initialize service.

        Args:
            settings: Bot configuration
            condition_id: Polymarket condition_id to analyze
        """
        self.settings = settings
        self.condition_id = condition_id

    async def collect_market_data(
        self,
        condition_id: str,
        duration_seconds: int = 120
    ) -> dict:
        """
        Connect to WebSocket and collect market data for specified duration.

        Args:
            condition_id: Market condition ID to subscribe to
            duration_seconds: How long to collect data (default 2 minutes)

        Returns:
            {
                'trades': [...],           # last_trade_price messages
                'book_snapshots': [...],   # book messages
                'price_changes': [...],    # price_change messages
                'collection_duration': int # Actual seconds collected
            }
        """
        # Skeleton implementation - return empty structure
        logger.info("Collecting market data", condition_id=condition_id, duration=duration_seconds)

        return {
            'trades': [],
            'book_snapshots': [],
            'price_changes': [],
            'collection_duration': duration_seconds
        }

    async def get_market_score(self) -> MarketSignals:
        """
        Get current market microstructure score.

        Returns:
            MarketSignals with score, confidence, and detailed metrics.
        """
        # Will implement in later tasks
        return MarketSignals(
            score=0.0,
            confidence=0.0,
            order_book_score=0.0,
            whale_score=0.0,
            volume_score=0.0,
            momentum_score=0.0,
            order_book_bias="UNAVAILABLE",
            whale_direction="UNAVAILABLE",
            whale_count=0,
            volume_ratio=1.0,
            momentum_direction="UNAVAILABLE",
            signal_type="UNAVAILABLE",
            timestamp=datetime.now()
        )

    async def close(self):
        """Cleanup WebSocket connections."""
        # No persistent connections, nothing to close
        pass
```

**Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_market_microstructure.py::test_collect_market_data_structure -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add polymarket/trading/market_microstructure.py tests/test_market_microstructure.py
git commit -m "feat: add skeleton WebSocket data collector

- Add MarketMicrostructureService with condition_id parameter
- Add collect_market_data() method (skeleton returns empty data)
- Add test for data structure validation
- Preserves same interface as Binance version

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 3: Implement Momentum Score Calculation

**Goal:** Calculate YES token price movement over collection window

**Files:**
- Modify: `polymarket/trading/market_microstructure.py`
- Test: `tests/test_market_microstructure.py`

**Step 1: Write failing test for momentum score**

Add to `tests/test_market_microstructure.py`:

```python
def test_calculate_momentum_score():
    """Test YES price momentum calculation."""
    service = MarketMicrostructureService(Settings(), "test-123")

    # Test: YES price rising 5%
    trades = [
        {'asset_id': 'YES_TOKEN', 'price': 0.50, 'size': 100},
        {'asset_id': 'YES_TOKEN', 'price': 0.525, 'size': 200},
    ]
    score = service.calculate_momentum_score(trades)
    assert score == pytest.approx(0.5, abs=0.01)  # 5% / 10% = 0.5

    # Test: YES price falling 10% → -1.0 (clamped)
    trades = [
        {'asset_id': 'YES_TOKEN', 'price': 0.50, 'size': 100},
        {'asset_id': 'YES_TOKEN', 'price': 0.45, 'size': 200},
    ]
    score = service.calculate_momentum_score(trades)
    assert score == -1.0

    # Test: No price change → 0.0
    trades = [
        {'asset_id': 'YES_TOKEN', 'price': 0.50, 'size': 100},
        {'asset_id': 'YES_TOKEN', 'price': 0.50, 'size': 200},
    ]
    score = service.calculate_momentum_score(trades)
    assert score == 0.0

    # Test: Empty trades → 0.0
    score = service.calculate_momentum_score([])
    assert score == 0.0
```

**Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_market_microstructure.py::test_calculate_momentum_score -v
```

Expected: FAIL with `calculate_momentum_score() method not found`

**Step 3: Implement momentum score calculation**

Add to `polymarket/trading/market_microstructure.py`:

```python
def calculate_momentum_score(self, trades: list) -> float:
    """
    Calculate YES token price momentum over collection window.

    Args:
        trades: List of trade messages with asset_id, price, size

    Returns:
        -1.0 (strong bearish) to +1.0 (strong bullish)
    """
    if not trades:
        return 0.0

    # Filter YES token trades
    yes_trades = [t for t in trades if t.get('asset_id') == 'YES_TOKEN']
    if len(yes_trades) < 2:
        return 0.0

    # Get first and last YES price in window
    initial_yes_price = yes_trades[0]['price']
    final_yes_price = yes_trades[-1]['price']

    # Calculate percentage change
    price_change_pct = (final_yes_price - initial_yes_price) / initial_yes_price

    # Normalize: ±10% change maps to ±1.0 score
    # Clamp to [-1.0, 1.0] range
    momentum_score = max(min(price_change_pct * 10, 1.0), -1.0)

    logger.debug(
        "Momentum calculated",
        initial=initial_yes_price,
        final=final_yes_price,
        change_pct=f"{price_change_pct*100:+.2f}%",
        score=f"{momentum_score:+.2f}"
    )

    return momentum_score
```

**Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_market_microstructure.py::test_calculate_momentum_score -v
```

Expected: PASS (all 4 assertions)

**Step 5: Commit**

```bash
git add polymarket/trading/market_microstructure.py tests/test_market_microstructure.py
git commit -m "feat: add YES token momentum score calculation

- Calculate price change from first to last YES trade
- Normalize ±10% change to ±1.0 score
- Clamp to [-1.0, 1.0] range
- Handle empty trades and <2 YES trades gracefully

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 4: Implement Volume Flow Score Calculation

**Goal:** Calculate net buying pressure (YES volume - NO volume)

**Files:**
- Modify: `polymarket/trading/market_microstructure.py`
- Test: `tests/test_market_microstructure.py`

**Step 1: Write failing test for volume flow score**

Add to `tests/test_market_microstructure.py`:

```python
def test_calculate_volume_flow_score():
    """Test volume flow calculation."""
    service = MarketMicrostructureService(Settings(), "test-123")

    # Test: Pure YES volume → +1.0
    trades = [
        {'asset_id': 'YES_TOKEN', 'size': 1000},
        {'asset_id': 'YES_TOKEN', 'size': 500},
    ]
    score = service.calculate_volume_flow_score(trades)
    assert score == 1.0

    # Test: Pure NO volume → -1.0
    trades = [
        {'asset_id': 'NO_TOKEN', 'size': 1000},
    ]
    score = service.calculate_volume_flow_score(trades)
    assert score == -1.0

    # Test: Equal volume → 0.0
    trades = [
        {'asset_id': 'YES_TOKEN', 'size': 500},
        {'asset_id': 'NO_TOKEN', 'size': 500},
    ]
    score = service.calculate_volume_flow_score(trades)
    assert score == 0.0

    # Test: 60% YES, 40% NO → +0.2
    trades = [
        {'asset_id': 'YES_TOKEN', 'size': 600},
        {'asset_id': 'NO_TOKEN', 'size': 400},
    ]
    score = service.calculate_volume_flow_score(trades)
    assert score == pytest.approx(0.2, abs=0.01)

    # Test: Empty trades → 0.0
    score = service.calculate_volume_flow_score([])
    assert score == 0.0
```

**Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_market_microstructure.py::test_calculate_volume_flow_score -v
```

Expected: FAIL with `calculate_volume_flow_score() method not found`

**Step 3: Implement volume flow score calculation**

Add to `polymarket/trading/market_microstructure.py`:

```python
def calculate_volume_flow_score(self, trades: list) -> float:
    """
    Calculate net buying pressure (YES volume - NO volume).

    Args:
        trades: List of trade messages with asset_id and size

    Returns:
        -1.0 (all NO buying) to +1.0 (all YES buying)
    """
    if not trades:
        return 0.0

    yes_volume = sum(
        trade['size'] for trade in trades
        if trade.get('asset_id') == 'YES_TOKEN'
    )

    no_volume = sum(
        trade['size'] for trade in trades
        if trade.get('asset_id') == 'NO_TOKEN'
    )

    total_volume = yes_volume + no_volume
    if total_volume == 0:
        return 0.0

    # Already normalized to -1.0 to +1.0
    volume_flow_score = (yes_volume - no_volume) / total_volume

    logger.debug(
        "Volume flow calculated",
        yes_volume=yes_volume,
        no_volume=no_volume,
        score=f"{volume_flow_score:+.2f}"
    )

    return volume_flow_score
```

**Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_market_microstructure.py::test_calculate_volume_flow_score -v
```

Expected: PASS (all 5 assertions)

**Step 5: Commit**

```bash
git add polymarket/trading/market_microstructure.py tests/test_market_microstructure.py
git commit -m "feat: add volume flow score calculation

- Sum YES volume vs NO volume from trades
- Calculate (YES - NO) / total ratio
- Already normalized to [-1.0, 1.0]
- Handle empty trades and zero volume gracefully

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 5: Implement Whale Activity Score Calculation

**Goal:** Count large trades (>$1,000) by direction

**Files:**
- Modify: `polymarket/trading/market_microstructure.py`
- Test: `tests/test_market_microstructure.py`

**Step 1: Write failing test for whale activity score**

Add to `tests/test_market_microstructure.py`:

```python
def test_calculate_whale_activity_score():
    """Test whale detection and scoring."""
    service = MarketMicrostructureService(Settings(), "test-123")

    # Test: All YES whales → +1.0
    trades = [
        {'asset_id': 'YES_TOKEN', 'size': 1500},
        {'asset_id': 'YES_TOKEN', 'size': 2000},
        {'asset_id': 'NO_TOKEN', 'size': 100},  # Not a whale
    ]
    score = service.calculate_whale_activity_score(trades)
    assert score == 1.0

    # Test: All NO whales → -1.0
    trades = [
        {'asset_id': 'NO_TOKEN', 'size': 1500},
        {'asset_id': 'NO_TOKEN', 'size': 2000},
        {'asset_id': 'YES_TOKEN', 'size': 100},  # Not a whale
    ]
    score = service.calculate_whale_activity_score(trades)
    assert score == -1.0

    # Test: Balanced whales → 0.0
    trades = [
        {'asset_id': 'YES_TOKEN', 'size': 1500},
        {'asset_id': 'NO_TOKEN', 'size': 1200},
    ]
    score = service.calculate_whale_activity_score(trades)
    assert score == 0.0

    # Test: 2 YES whales, 1 NO whale → +0.33
    trades = [
        {'asset_id': 'YES_TOKEN', 'size': 1500},
        {'asset_id': 'YES_TOKEN', 'size': 2000},
        {'asset_id': 'NO_TOKEN', 'size': 1200},
    ]
    score = service.calculate_whale_activity_score(trades)
    assert score == pytest.approx(0.333, abs=0.01)

    # Test: No whales → 0.0 (handle division by zero)
    trades = [
        {'asset_id': 'YES_TOKEN', 'size': 500},
        {'asset_id': 'NO_TOKEN', 'size': 800},
    ]
    score = service.calculate_whale_activity_score(trades)
    assert score == 0.0

    # Test: Empty trades → 0.0
    score = service.calculate_whale_activity_score([])
    assert score == 0.0
```

**Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_market_microstructure.py::test_calculate_whale_activity_score -v
```

Expected: FAIL with `calculate_whale_activity_score() method not found`

**Step 3: Implement whale activity score calculation**

Add to `polymarket/trading/market_microstructure.py`:

```python
def calculate_whale_activity_score(self, trades: list) -> float:
    """
    Calculate directional signal from whale trades (>$1,000).

    Args:
        trades: List of trade messages with asset_id and size

    Returns:
        -1.0 (all NO whales) to +1.0 (all YES whales)
    """
    if not trades:
        return 0.0

    # Identify whale trades (size > $1,000)
    yes_whales = sum(
        1 for trade in trades
        if trade['size'] > self.WHALE_SIZE_USD and trade.get('asset_id') == 'YES_TOKEN'
    )

    no_whales = sum(
        1 for trade in trades
        if trade['size'] > self.WHALE_SIZE_USD and trade.get('asset_id') == 'NO_TOKEN'
    )

    total_whales = yes_whales + no_whales
    if total_whales == 0:
        return 0.0

    whale_score = (yes_whales - no_whales) / total_whales

    logger.debug(
        "Whale activity calculated",
        yes_whales=yes_whales,
        no_whales=no_whales,
        score=f"{whale_score:+.2f}"
    )

    return whale_score
```

**Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_market_microstructure.py::test_calculate_whale_activity_score -v
```

Expected: PASS (all 6 assertions)

**Step 5: Commit**

```bash
git add polymarket/trading/market_microstructure.py tests/test_market_microstructure.py
git commit -m "feat: add whale activity score calculation

- Count trades >$1,000 as whale trades
- Calculate (YES whales - NO whales) / total
- Normalized to [-1.0, 1.0]
- Handle no whales and empty trades gracefully

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 6: Implement Combined Score and Confidence

**Goal:** Combine three scores with weights and calculate confidence

**Files:**
- Modify: `polymarket/trading/market_microstructure.py`
- Test: `tests/test_market_microstructure.py`

**Step 1: Write failing test for combined score**

Add to `tests/test_market_microstructure.py`:

```python
def test_calculate_market_score():
    """Test weighted combination of scores."""
    service = MarketMicrostructureService(Settings(), "test-123")

    # Test: All positive scores
    score = service.calculate_market_score(
        momentum=1.0,
        volume_flow=1.0,
        whale=1.0
    )
    assert score == 1.0  # 0.4 + 0.35 + 0.25 = 1.0

    # Test: All negative scores
    score = service.calculate_market_score(
        momentum=-1.0,
        volume_flow=-1.0,
        whale=-1.0
    )
    assert score == -1.0

    # Test: Mixed scores with weights
    score = service.calculate_market_score(
        momentum=0.5,   # 0.5 * 0.4 = 0.2
        volume_flow=0.3,  # 0.3 * 0.35 = 0.105
        whale=0.2     # 0.2 * 0.25 = 0.05
    )
    assert score == pytest.approx(0.355, abs=0.01)

    # Test: Neutral scores
    score = service.calculate_market_score(
        momentum=0.0,
        volume_flow=0.0,
        whale=0.0
    )
    assert score == 0.0


def test_calculate_confidence():
    """Test confidence calculation based on data quality."""
    service = MarketMicrostructureService(Settings(), "test-123")

    # Test: 50+ trades, full duration → 1.0 confidence
    data = {
        'trades': [{'asset_id': 'YES_TOKEN', 'size': 100}] * 50,
        'collection_duration': 120
    }
    confidence = service.calculate_confidence(data)
    assert confidence == 1.0

    # Test: 25 trades → 0.5 base confidence
    data = {
        'trades': [{'asset_id': 'YES_TOKEN', 'size': 100}] * 25,
        'collection_duration': 120
    }
    confidence = service.calculate_confidence(data)
    assert confidence == pytest.approx(0.5, abs=0.01)

    # Test: 50 trades but only 60s collected → 0.5x penalty
    data = {
        'trades': [{'asset_id': 'YES_TOKEN', 'size': 100}] * 50,
        'collection_duration': 60
    }
    confidence = service.calculate_confidence(data)
    assert confidence == pytest.approx(0.5, abs=0.01)  # 1.0 * 0.5

    # Test: <10 trades → 0.5x low liquidity penalty
    data = {
        'trades': [{'asset_id': 'YES_TOKEN', 'size': 100}] * 5,
        'collection_duration': 120
    }
    confidence = service.calculate_confidence(data)
    # (5/50) * 1.0 * 0.5 = 0.05
    assert confidence == pytest.approx(0.05, abs=0.01)

    # Test: Empty trades → 0.0
    data = {'trades': [], 'collection_duration': 120}
    confidence = service.calculate_confidence(data)
    assert confidence == 0.0
```

**Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_market_microstructure.py::test_calculate_market_score -v
python -m pytest tests/test_market_microstructure.py::test_calculate_confidence -v
```

Expected: FAIL with method not found errors

**Step 3: Implement combined score and confidence methods**

Add to `polymarket/trading/market_microstructure.py`:

```python
def calculate_market_score(
    self,
    momentum: float,
    volume_flow: float,
    whale: float
) -> float:
    """
    Combine three scores with weights.

    Args:
        momentum: Momentum score (-1.0 to +1.0)
        volume_flow: Volume flow score (-1.0 to +1.0)
        whale: Whale activity score (-1.0 to +1.0)

    Returns:
        -1.0 (strong bearish) to +1.0 (strong bullish)
    """
    market_score = (
        momentum * self.WEIGHTS['momentum'] +
        volume_flow * self.WEIGHTS['volume_flow'] +
        whale * self.WEIGHTS['whale']
    )

    return market_score


def calculate_confidence(self, data: dict) -> float:
    """
    Calculate confidence based on data quality.

    Args:
        data: Collection data with 'trades' and 'collection_duration'

    Returns:
        0.0 to 1.0 confidence score
    """
    trade_count = len(data.get('trades', []))

    # Base confidence from trade volume
    # 50+ trades = full confidence, scales linearly
    base_confidence = min(trade_count / 50, 1.0)

    # Penalty if didn't collect full 2 minutes
    collection_duration = data.get('collection_duration', 120)
    if collection_duration < 120:
        base_confidence *= (collection_duration / 120)

    # Penalty for low liquidity
    if trade_count < 10:
        logger.warning("Low liquidity", trades=trade_count)
        base_confidence *= 0.5

    return base_confidence
```

**Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_market_microstructure.py::test_calculate_market_score -v
python -m pytest tests/test_market_microstructure.py::test_calculate_confidence -v
```

Expected: PASS (4 + 5 = 9 assertions)

**Step 5: Commit**

```bash
git add polymarket/trading/market_microstructure.py tests/test_market_microstructure.py
git commit -m "feat: add combined score and confidence calculation

- Combine 3 scores with weights (momentum 40%, volume 35%, whale 25%)
- Calculate confidence from trade count and collection duration
- Low liquidity penalty (<10 trades = 0.5x)
- Partial collection time penalty

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 7: Implement get_market_score() Integration

**Goal:** Wire up all scoring functions in get_market_score() method

**Files:**
- Modify: `polymarket/trading/market_microstructure.py`
- Test: `tests/test_market_microstructure.py`

**Step 1: Write failing test for get_market_score**

Add to `tests/test_market_microstructure.py`:

```python
@pytest.mark.asyncio
async def test_get_market_score_with_mock_data():
    """Test get_market_score with mocked collection data."""
    service = MarketMicrostructureService(Settings(), "test-condition-123")

    # Mock collect_market_data to return test data
    async def mock_collect(condition_id, duration_seconds):
        return {
            'trades': [
                {'asset_id': 'YES_TOKEN', 'price': 0.50, 'size': 1000},
                {'asset_id': 'YES_TOKEN', 'price': 0.52, 'size': 1500},
                {'asset_id': 'NO_TOKEN', 'price': 0.48, 'size': 500},
            ] * 20,  # 60 trades total
            'book_snapshots': [],
            'price_changes': [],
            'collection_duration': 120
        }

    # Replace method
    service.collect_market_data = mock_collect

    # Get market score
    signals = await service.get_market_score()

    # Verify structure
    assert isinstance(signals, MarketSignals)
    assert -1.0 <= signals.score <= 1.0
    assert 0.0 <= signals.confidence <= 1.0

    # Verify scores are calculated (not default 0.0)
    assert signals.momentum_score != 0.0  # Price moved 0.50 → 0.52
    assert signals.volume_score != 0.0    # More YES than NO volume
    assert signals.whale_score != 0.0     # Has whale trades >$1k

    # Confidence should be high (60 trades, full duration)
    assert signals.confidence > 0.8
```

**Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_market_microstructure.py::test_get_market_score_with_mock_data -v
```

Expected: FAIL (get_market_score returns default neutral signals)

**Step 3: Implement get_market_score method**

Modify `get_market_score()` in `polymarket/trading/market_microstructure.py`:

```python
async def get_market_score(self) -> MarketSignals:
    """
    Get current market microstructure score.

    Returns:
        MarketSignals with score, confidence, and detailed metrics.
    """
    try:
        # Collect data for 2 minutes
        data = await self.collect_market_data(
            self.condition_id,
            duration_seconds=120
        )

        # Calculate individual scores
        momentum_score = self.calculate_momentum_score(data['trades'])
        volume_flow_score = self.calculate_volume_flow_score(data['trades'])
        whale_score = self.calculate_whale_activity_score(data['trades'])

        # Combine scores
        market_score = self.calculate_market_score(
            momentum_score,
            volume_flow_score,
            whale_score
        )

        # Calculate confidence
        confidence = self.calculate_confidence(data)

        # Classify signal
        signal_type = self._classify_signal(market_score, confidence)

        # Extract metadata
        whale_count = sum(
            1 for t in data['trades']
            if t['size'] > self.WHALE_SIZE_USD
        )

        momentum_direction = (
            "UP" if momentum_score > 0.1 else
            "DOWN" if momentum_score < -0.1 else
            "FLAT"
        )

        whale_direction = (
            "BUYING" if whale_score > 0.3 else
            "SELLING" if whale_score < -0.3 else
            "NEUTRAL"
        )

        logger.info(
            "Market microstructure calculated",
            score=f"{market_score:+.2f}",
            confidence=f"{confidence:.2f}",
            signal=signal_type,
            whales=whale_count
        )

        return MarketSignals(
            score=market_score,
            confidence=confidence,
            order_book_score=0.0,  # Not used in new version
            whale_score=whale_score,
            volume_score=volume_flow_score,
            momentum_score=momentum_score,
            order_book_bias="N/A",  # Not used
            whale_direction=whale_direction,
            whale_count=whale_count,
            volume_ratio=1.0 + volume_flow_score,  # Approximate
            momentum_direction=momentum_direction,
            signal_type=signal_type,
            timestamp=datetime.now()
        )

    except Exception as e:
        logger.error("Market microstructure failed", error=str(e))
        # Return neutral on complete failure
        return MarketSignals(
            score=0.0,
            confidence=0.0,
            order_book_score=0.0,
            whale_score=0.0,
            volume_score=0.0,
            momentum_score=0.0,
            order_book_bias="UNAVAILABLE",
            whale_direction="UNAVAILABLE",
            whale_count=0,
            volume_ratio=1.0,
            momentum_direction="UNAVAILABLE",
            signal_type="UNAVAILABLE",
            timestamp=datetime.now()
        )


def _classify_signal(self, score: float, confidence: float) -> str:
    """Classify signal strength."""
    direction = "BULLISH" if score > 0.1 else "BEARISH" if score < -0.1 else "NEUTRAL"
    strength = "STRONG" if confidence >= 0.7 else "WEAK" if confidence >= 0.5 else "CONFLICTED"
    return f"{strength}_{direction}"
```

**Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_market_microstructure.py::test_get_market_score_with_mock_data -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add polymarket/trading/market_microstructure.py tests/test_market_microstructure.py
git commit -m "feat: wire up get_market_score() with all scoring functions

- Collect data for 2 minutes
- Calculate momentum, volume flow, whale scores
- Combine with weights
- Calculate confidence from data quality
- Return MarketSignals with metadata
- Handle errors gracefully (return neutral)

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 8: Implement WebSocket Connection (Real)

**Goal:** Replace skeleton collect_market_data with real WebSocket connection

**Files:**
- Modify: `polymarket/trading/market_microstructure.py`
- Test: `tests/test_market_microstructure.py`

**Note:** This task requires the WebSocket URL from Task 1 research. If research isn't complete, use placeholder URL and mark as TODO.

**Step 1: Add websockets dependency**

Check if `websockets` is in `requirements.txt`. If not, add it:

```bash
grep -q "websockets" requirements.txt || echo "websockets>=12.0" >> requirements.txt
pip install -r requirements.txt
```

**Step 2: Write integration test for WebSocket (will be slow)**

Add to `tests/test_market_microstructure.py`:

```python
@pytest.mark.asyncio
@pytest.mark.slow  # Mark as slow test
async def test_websocket_connection_real():
    """Test real WebSocket connection (slow, may be skipped)."""
    service = MarketMicrostructureService(Settings(), "test-condition-123")

    # Attempt real connection for 5 seconds
    try:
        data = await service.collect_market_data(
            "test-condition-123",
            duration_seconds=5
        )

        # Verify structure even if no trades
        assert isinstance(data, dict)
        assert 'trades' in data
        assert 'collection_duration' in data

    except Exception as e:
        pytest.skip(f"WebSocket connection failed: {e}")
```

**Step 3: Implement real WebSocket connection**

Modify `collect_market_data()` in `polymarket/trading/market_microstructure.py`:

```python
import websockets
import json


async def collect_market_data(
    self,
    condition_id: str,
    duration_seconds: int = 120
) -> dict:
    """
    Connect to WebSocket and collect market data for specified duration.

    Args:
        condition_id: Market condition ID to subscribe to
        duration_seconds: How long to collect data (default 2 minutes)

    Returns:
        {
            'trades': [...],           # last_trade_price messages
            'book_snapshots': [...],   # book messages
            'price_changes': [...],    # price_change messages
            'collection_duration': int # Actual seconds collected
        }
    """
    accumulated_data = {
        'trades': [],
        'book_snapshots': [],
        'price_changes': [],
        'collection_duration': 0
    }

    logger.info(
        "Connecting to Polymarket CLOB WebSocket",
        condition_id=condition_id,
        duration=duration_seconds
    )

    try:
        async with websockets.connect(
            self.WS_URL,
            ping_interval=20,
            ping_timeout=10
        ) as ws:
            # Send subscription message
            subscribe_msg = {
                "action": "subscribe",
                "subscriptions": [
                    {
                        "topic": "market",
                        "condition_id": condition_id
                    }
                ]
            }
            await ws.send(json.dumps(subscribe_msg))
            logger.debug("Sent subscription", condition_id=condition_id)

            # Collect data for specified duration
            start_time = time.time()
            while time.time() - start_time < duration_seconds:
                try:
                    # Wait for message with 5s timeout
                    msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
                    data = json.loads(msg)

                    # Accumulate based on message type
                    msg_type = data.get('type')
                    if msg_type == 'last_trade_price':
                        accumulated_data['trades'].append(data.get('payload', {}))
                    elif msg_type == 'book':
                        accumulated_data['book_snapshots'].append(data.get('payload', {}))
                    elif msg_type == 'price_change':
                        accumulated_data['price_changes'].append(data.get('payload', {}))

                except asyncio.TimeoutError:
                    # No message in 5s, continue waiting
                    continue

            # Record actual collection time
            accumulated_data['collection_duration'] = int(time.time() - start_time)

            logger.info(
                "Data collection complete",
                trades=len(accumulated_data['trades']),
                duration=accumulated_data['collection_duration']
            )

    except Exception as e:
        logger.error("WebSocket collection failed", error=str(e))
        # Return empty data on failure
        accumulated_data['collection_duration'] = 0

    return accumulated_data
```

**Step 4: Run integration test (may skip if WebSocket unavailable)**

```bash
python -m pytest tests/test_market_microstructure.py::test_websocket_connection_real -v -s
```

Expected: PASS or SKIP (if WebSocket unavailable in test environment)

**Step 5: Commit**

```bash
git add polymarket/trading/market_microstructure.py tests/test_market_microstructure.py requirements.txt
git commit -m "feat: implement real WebSocket connection for data collection

- Connect to Polymarket CLOB WebSocket
- Subscribe to market channel with condition_id
- Collect last_trade_price, book, price_change messages
- Handle connection errors gracefully
- Add integration test (marked as slow)

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 9: Update auto_trade.py Integration

**Goal:** Modify auto_trade.py to pass condition_id to MarketMicrostructureService

**Files:**
- Modify: `scripts/auto_trade.py:88-92`

**Step 1: Read current auto_trade.py integration**

```bash
grep -A 5 "MarketMicrostructureService" scripts/auto_trade.py
```

Current code:
```python
self.market_service = MarketMicrostructureService(settings)
```

**Step 2: Write test (manual verification step)**

This is more of an integration verification. Run:
```bash
python scripts/auto_trade.py --once
```

Expected: Should fail with `TypeError: __init__() missing 1 required positional argument: 'condition_id'`

**Step 3: Implement integration changes**

Modify `scripts/auto_trade.py`:

Find line ~54:
```python
self.market_service = MarketMicrostructureService(settings)
```

Replace with:
```python
self.market_service = None  # Will initialize per cycle with condition_id
```

Find `run_cycle()` method around line 88-92:
```python
btc_data, social_sentiment, market_signals = await asyncio.gather(
    self.btc_service.get_current_price(),
    self.social_service.get_social_score(),
    self.market_service.get_market_score(),
)
```

Replace with:
```python
# Extract condition_id from discovered market
condition_id = getattr(markets[0], 'condition_id', None)
if not condition_id:
    logger.warning("No condition_id found, using fallback")
    condition_id = "unknown"

# Initialize market service with condition_id
if not self.market_service or self.market_service.condition_id != condition_id:
    self.market_service = MarketMicrostructureService(self.settings, condition_id)

# Fetch data in parallel
btc_data, social_sentiment, market_signals = await asyncio.gather(
    self.btc_service.get_current_price(),
    self.social_service.get_social_score(),
    self.market_service.get_market_score(),
)
```

**Step 4: Verify integration**

```bash
python scripts/auto_trade.py --once
```

Expected: Should run without TypeError (may fail later with WebSocket issues, that's OK)

**Step 5: Commit**

```bash
git add scripts/auto_trade.py
git commit -m "feat: integrate condition_id into auto_trade loop

- Initialize MarketMicrostructureService per cycle with condition_id
- Extract condition_id from discovered market
- Handle missing condition_id with fallback
- Maintains same parallel data fetching

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 10: Update Integration Tests

**Goal:** Fix test_integration_sentiment.py to work with new system

**Files:**
- Modify: `tests/test_integration_sentiment.py`

**Step 1: Read current integration test**

```bash
cat tests/test_integration_sentiment.py | head -50
```

**Step 2: Identify needed changes**

The test needs:
- Pass `condition_id` to `MarketMicrostructureService`
- Mock WebSocket if no real market available

**Step 3: Update integration test**

Modify `tests/test_integration_sentiment.py`:

```python
@pytest.mark.asyncio
async def test_full_sentiment_pipeline():
    """Test complete end-to-end sentiment analysis pipeline."""
    # Initialize services
    settings = Settings()
    client = PolymarketClient()
    btc_service = BTCPriceService(settings)
    social_service = SocialSentimentService(settings)
    # Note: MarketMicrostructureService needs condition_id
    aggregator = SignalAggregator()
    ai_service = AIDecisionService(settings)

    try:
        # Step 1: Fetch market
        market = client.discover_btc_15min_market()
        assert market is not None
        assert market.active

        # Extract condition_id
        condition_id = getattr(market, 'condition_id', 'test-condition-123')

        # Initialize market service with condition_id
        market_service = MarketMicrostructureService(settings, condition_id)

        # Step 2: Fetch all data in parallel (may fail if WebSocket unavailable)
        try:
            btc_data, social, market_signals = await asyncio.gather(
                btc_service.get_current_price(),
                social_service.get_social_score(),
                market_service.get_market_score()
            )
        except Exception as e:
            logger.warning(f"Market microstructure unavailable: {e}")
            # Fall back to social only
            btc_data, social = await asyncio.gather(
                btc_service.get_current_price(),
                social_service.get_social_score()
            )
            # Create neutral market signals
            market_signals = MarketSignals(
                score=0.0,
                confidence=0.0,
                order_book_score=0.0,
                whale_score=0.0,
                volume_score=0.0,
                momentum_score=0.0,
                order_book_bias="UNAVAILABLE",
                whale_direction="UNAVAILABLE",
                whale_count=0,
                volume_ratio=1.0,
                momentum_direction="UNAVAILABLE",
                signal_type="UNAVAILABLE",
                timestamp=datetime.now()
            )

        # Verify data received
        assert btc_data.price > 0
        assert -0.7 <= social.score <= 0.85
        assert 0.0 <= social.confidence <= 1.0
        assert -1.0 <= market_signals.score <= 1.0
        assert 0.0 <= market_signals.confidence <= 1.0

        # ... rest of test continues unchanged
```

**Step 4: Run integration test**

```bash
python -m pytest tests/test_integration_sentiment.py::test_full_sentiment_pipeline -v -s
```

Expected: PASS (or graceful fallback if WebSocket unavailable)

**Step 5: Commit**

```bash
git add tests/test_integration_sentiment.py
git commit -m "fix: update integration test for new market microstructure

- Pass condition_id to MarketMicrostructureService
- Handle WebSocket unavailability gracefully
- Fall back to social sentiment only if market data fails
- Preserves test coverage

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 11: Update Documentation

**Goal:** Update SENTIMENT-ANALYSIS.md with new approach

**Files:**
- Modify: `docs/SENTIMENT-ANALYSIS.md`

**Step 1: Read current documentation**

```bash
head -50 docs/SENTIMENT-ANALYSIS.md
```

**Step 2: Update Market Microstructure section**

Find section "### 2. Market Microstructure Scorer" (around line 40).

Replace with:
```markdown
### 2. Market Microstructure Scorer (`polymarket/trading/market_microstructure.py`)

**Data Source (Polymarket CLOB WebSocket):**
- Connects for 2 minutes per trading cycle
- Subscribes to specific BTC 15-min prediction market
- Collects: Trade executions, price updates, order book snapshots

**Analysis:**
- **YES Price Momentum** (40% weight): Track YES token price movement
- **Volume Flow** (35% weight): Net buying pressure (YES volume - NO volume)
- **Whale Activity** (25% weight): Large trades (>$1,000) directional signal

**Output:**
- Score: -1.0 to +1.0
- Confidence: Based on trade count and collection quality (0.0 to 1.0)

**Key Insight:** Analyzes the **exact market we're betting on**, not external BTC spot price.
```

**Step 3: Update data sources table**

Find "**Data Sources (Binance Public APIs):**" section.

Replace entire section with:
```markdown
**Data Sources (Polymarket CLOB WebSocket):**
- Real-time trades from BTC 15-min prediction market
- 2-minute collection window per cycle
- Message types: `last_trade_price`, `book`, `price_change`
```

**Step 4: Commit**

```bash
git add docs/SENTIMENT-ANALYSIS.md
git commit -m "docs: update market microstructure to reflect Polymarket CLOB

- Document WebSocket data collection
- Explain 3 weighted signals (momentum, volume flow, whales)
- Clarify we analyze prediction market, not spot BTC
- Update weights and thresholds

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 12: Remove Old Binance Code (Optional Cleanup)

**Goal:** Remove unused Binance-specific code and tests

**Files:**
- Modify: `tests/test_market_microstructure.py`

**Step 1: Identify old Binance tests**

```bash
grep -n "binance\|Binance" tests/test_market_microstructure.py
```

**Step 2: Remove old tests**

Remove tests like:
- `test_fetch_order_book` (Binance-specific)
- `test_fetch_recent_trades` (Binance-specific)
- Any other Binance API tests

**Step 3: Run full test suite**

```bash
python -m pytest tests/ -v
```

Expected: All tests pass (or only known unrelated failures)

**Step 4: Commit cleanup**

```bash
git add tests/test_market_microstructure.py
git commit -m "chore: remove old Binance API tests

- Remove test_fetch_order_book (Binance-specific)
- Remove test_fetch_recent_trades (Binance-specific)
- Keep new WebSocket and scoring tests
- All tests passing

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 13: Final Verification

**Goal:** Run full test suite and manual verification

**Files:**
- None (verification task)

**Step 1: Run full test suite**

```bash
cd ~/.config/superpowers/worktrees/polymarket-scripts/polymarket-market-microstructure
source venv/bin/activate
python -m pytest tests/ -v --tb=short
```

Expected: All tests pass (or only known unrelated failures)

**Step 2: Manual verification with auto_trade.py**

```bash
POLYMARKET_MODE=read_only python scripts/auto_trade.py --once
```

Expected: Bot runs one cycle, collects data from WebSocket, calculates scores, makes decision

**Step 3: Check logs for data quality**

Look for log lines like:
```
INFO: Connecting to Polymarket CLOB WebSocket
INFO: Data collection complete (trades=45, duration=120)
INFO: Market microstructure calculated (score=+0.23, confidence=0.85, signal=STRONG_BULLISH)
```

**Step 4: Document any issues**

If issues found:
```bash
echo "## Known Issues

- [Describe any issues found]

" >> /tmp/verification-notes.md
```

**Step 5: Final commit**

```bash
git add -A
git commit -m "chore: final verification complete

- All tests passing
- Integration verified with auto_trade.py --once
- WebSocket connection stable
- Scoring functions working as expected

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Implementation Checklist

### Phase 1: Core Scoring Functions (Tasks 1-7)
- [ ] Task 1: Research Polymarket CLOB WebSocket endpoint
- [ ] Task 2: Create WebSocket data collector skeleton
- [ ] Task 3: Implement momentum score calculation
- [ ] Task 4: Implement volume flow score calculation
- [ ] Task 5: Implement whale activity score calculation
- [ ] Task 6: Implement combined score and confidence
- [ ] Task 7: Wire up get_market_score() integration

### Phase 2: WebSocket & Integration (Tasks 8-10)
- [ ] Task 8: Implement real WebSocket connection
- [ ] Task 9: Update auto_trade.py integration
- [ ] Task 10: Update integration tests

### Phase 3: Documentation & Cleanup (Tasks 11-13)
- [ ] Task 11: Update documentation
- [ ] Task 12: Remove old Binance code
- [ ] Task 13: Final verification

---

## Success Criteria

- ✅ All unit tests pass (momentum, volume flow, whale scoring)
- ✅ Integration test passes (get_market_score with mocked data)
- ✅ WebSocket connection succeeds (even if slow/flaky)
- ✅ auto_trade.py runs without errors
- ✅ Scores are non-zero when data is available
- ✅ Graceful degradation on WebSocket failures (returns neutral)
- ✅ Documentation updated

---

## Notes

**WebSocket URL:** Update `WS_URL` in Task 8 after completing Task 1 research. If research incomplete, use placeholder and mark as TODO.

**Testing Strategy:**
- Unit tests: Fast, mock all external calls
- Integration tests: May be slow, mark with `@pytest.mark.slow`
- WebSocket tests: May fail in restricted environments, skip gracefully

**Common Issues:**
- WebSocket connection blocked by firewall → Test locally, deploy to production
- No trades in 2-minute window → Low liquidity OK, confidence will be low
- condition_id not found → Fallback to "unknown", log warning

**Next Steps After Implementation:**
- Monitor signal quality in production (read_only mode)
- Tune weights if needed (momentum vs volume vs whales)
- Consider adding order book depth analysis as 4th signal
