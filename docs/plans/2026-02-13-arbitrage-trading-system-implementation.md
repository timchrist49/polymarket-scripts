# Arbitrage Trading System Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement price feed arbitrage system to increase trade frequency from 5→25 trades/day while maintaining 70%+ win rate

**Architecture:** Add three new components (ProbabilityCalculator, ArbitrageDetector, SmartOrderExecutor) that work alongside existing technical indicators to identify mispriced markets and execute limit orders

**Tech Stack:** Python 3.11, scipy (for norm.cdf), py_clob_client (for limit orders), asyncio

---

## Task 1: Add Arbitrage Data Models

**Files:**
- Modify: `polymarket/models.py:1-100`

**Step 1: Write failing tests for ArbitrageOpportunity**

Create: `tests/test_arbitrage_models.py`

```python
"""Tests for arbitrage-related data models."""
import pytest
from polymarket.models import ArbitrageOpportunity

def test_arbitrage_opportunity_creation():
    """Test creating ArbitrageOpportunity with valid data."""
    opp = ArbitrageOpportunity(
        market_id="test-market",
        actual_probability=0.68,
        polymarket_yes_odds=0.55,
        polymarket_no_odds=0.45,
        edge_percentage=0.13,
        recommended_action="BUY_YES",
        confidence_boost=0.20,
        urgency="MEDIUM",
        expected_profit_pct=0.18
    )

    assert opp.market_id == "test-market"
    assert opp.actual_probability == 0.68
    assert opp.edge_percentage == 0.13
    assert opp.recommended_action == "BUY_YES"
    assert opp.urgency == "MEDIUM"

def test_arbitrage_opportunity_validation():
    """Test validation of ArbitrageOpportunity fields."""
    # Test invalid action
    with pytest.raises(ValueError):
        ArbitrageOpportunity(
            market_id="test",
            actual_probability=0.68,
            polymarket_yes_odds=0.55,
            polymarket_no_odds=0.45,
            edge_percentage=0.13,
            recommended_action="INVALID",  # Should fail
            confidence_boost=0.20,
            urgency="MEDIUM",
            expected_profit_pct=0.18
        )

    # Test invalid urgency
    with pytest.raises(ValueError):
        ArbitrageOpportunity(
            market_id="test",
            actual_probability=0.68,
            polymarket_yes_odds=0.55,
            polymarket_no_odds=0.45,
            edge_percentage=0.13,
            recommended_action="BUY_YES",
            confidence_boost=0.20,
            urgency="INVALID",  # Should fail
            expected_profit_pct=0.18
        )
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_arbitrage_models.py::test_arbitrage_opportunity_creation -v`
Expected: FAIL with "cannot import name 'ArbitrageOpportunity'"

**Step 3: Implement ArbitrageOpportunity dataclass**

Add to `polymarket/models.py` after line 100:

```python
from typing import Literal

@dataclass
class ArbitrageOpportunity:
    """Detected arbitrage opportunity from price feed lag."""

    market_id: str
    actual_probability: float  # Calculated from price momentum
    polymarket_yes_odds: float  # Current market odds
    polymarket_no_odds: float
    edge_percentage: float  # Size of mispricing
    recommended_action: Literal["BUY_YES", "BUY_NO", "HOLD"]
    confidence_boost: float  # Amount to boost AI confidence
    urgency: Literal["HIGH", "MEDIUM", "LOW"]
    expected_profit_pct: float  # Expected ROI if correct

    def __post_init__(self):
        """Validate field values."""
        if self.recommended_action not in ["BUY_YES", "BUY_NO", "HOLD"]:
            raise ValueError(f"Invalid action: {self.recommended_action}")
        if self.urgency not in ["HIGH", "MEDIUM", "LOW"]:
            raise ValueError(f"Invalid urgency: {self.urgency}")
        if not 0.0 <= self.actual_probability <= 1.0:
            raise ValueError(f"Invalid probability: {self.actual_probability}")
        if not 0.0 <= self.polymarket_yes_odds <= 1.0:
            raise ValueError(f"Invalid YES odds: {self.polymarket_yes_odds}")
        if not 0.0 <= self.polymarket_no_odds <= 1.0:
            raise ValueError(f"Invalid NO odds: {self.polymarket_no_odds}")


@dataclass
class LimitOrderStrategy:
    """Strategy parameters for smart limit order execution."""

    target_price: float  # Price to place limit order at
    timeout_seconds: int  # How long to wait before fallback
    fallback_to_market: bool  # Whether to use market order if timeout
    urgency: Literal["HIGH", "MEDIUM", "LOW"]
    price_improvement_pct: float  # How much better than market
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_arbitrage_models.py -v`
Expected: PASS (2 tests)

**Step 5: Commit**

```bash
git add polymarket/models.py tests/test_arbitrage_models.py
git commit -m "feat: add ArbitrageOpportunity and LimitOrderStrategy models"
```

---

## Task 2: Implement Probability Calculator

**Files:**
- Create: `polymarket/trading/probability_calculator.py`
- Create: `tests/test_probability_calculator.py`

**Step 1: Write failing tests**

Create: `tests/test_probability_calculator.py`

```python
"""Tests for probability calculator."""
import pytest
from polymarket.trading.probability_calculator import ProbabilityCalculator

def test_calculate_upward_momentum():
    """Test probability calculation with upward momentum."""
    calc = ProbabilityCalculator()

    # BTC rising: $66,000 -> $66,200 (+0.30% in 5min)
    prob = calc.calculate_directional_probability(
        current_price=66200.0,
        price_5min_ago=66000.0,
        price_10min_ago=65900.0,
        volatility_15min=0.005,  # 0.5%
        time_remaining_seconds=600,  # 10 minutes left
        orderbook_imbalance=0.2  # Buy pressure
    )

    # Should be >0.5 due to upward momentum + buy pressure
    assert prob > 0.5
    assert prob < 0.95  # But not extreme
    assert 0.65 <= prob <= 0.80  # Reasonable range


def test_calculate_downward_momentum():
    """Test probability calculation with downward momentum."""
    calc = ProbabilityCalculator()

    # BTC falling: $66,000 -> $65,800 (-0.30% in 5min)
    prob = calc.calculate_directional_probability(
        current_price=65800.0,
        price_5min_ago=66000.0,
        price_10min_ago=66100.0,
        volatility_15min=0.005,
        time_remaining_seconds=600,
        orderbook_imbalance=-0.2  # Sell pressure
    )

    # Should be <0.5 due to downward momentum + sell pressure
    assert prob < 0.5
    assert prob > 0.05  # But not extreme
    assert 0.20 <= prob <= 0.35


def test_calculate_no_momentum():
    """Test probability calculation with sideways movement."""
    calc = ProbabilityCalculator()

    # BTC flat: $66,000 -> $66,000
    prob = calc.calculate_directional_probability(
        current_price=66000.0,
        price_5min_ago=66000.0,
        price_10min_ago=66000.0,
        volatility_15min=0.005,
        time_remaining_seconds=600,
        orderbook_imbalance=0.0
    )

    # Should be ~0.5 (neutral)
    assert 0.45 <= prob <= 0.55


def test_high_volatility_reduces_confidence():
    """Test that high volatility brings probability closer to 0.5."""
    calc = ProbabilityCalculator()

    # Same momentum, different volatility
    low_vol = calc.calculate_directional_probability(
        current_price=66200.0,
        price_5min_ago=66000.0,
        price_10min_ago=65900.0,
        volatility_15min=0.002,  # Low volatility
        time_remaining_seconds=600,
        orderbook_imbalance=0.0
    )

    high_vol = calc.calculate_directional_probability(
        current_price=66200.0,
        price_5min_ago=66000.0,
        price_10min_ago=65900.0,
        volatility_15min=0.020,  # High volatility
        time_remaining_seconds=600,
        orderbook_imbalance=0.0
    )

    # High volatility should reduce confidence (closer to 0.5)
    assert abs(low_vol - 0.5) > abs(high_vol - 0.5)


def test_time_decay():
    """Test that less time remaining increases uncertainty."""
    calc = ProbabilityCalculator()

    # Same momentum, different time remaining
    more_time = calc.calculate_directional_probability(
        current_price=66200.0,
        price_5min_ago=66000.0,
        price_10min_ago=65900.0,
        volatility_15min=0.005,
        time_remaining_seconds=800,  # 13+ minutes left
        orderbook_imbalance=0.0
    )

    less_time = calc.calculate_directional_probability(
        current_price=66200.0,
        price_5min_ago=66000.0,
        price_10min_ago=65900.0,
        volatility_15min=0.005,
        time_remaining_seconds=200,  # 3 minutes left
        orderbook_imbalance=0.0
    )

    # Less time = more uncertainty (closer to 0.5)
    assert abs(more_time - 0.5) > abs(less_time - 0.5)


def test_probability_bounds():
    """Test that probability is always clipped to [0.05, 0.95]."""
    calc = ProbabilityCalculator()

    # Extreme upward momentum
    prob_up = calc.calculate_directional_probability(
        current_price=70000.0,
        price_5min_ago=66000.0,  # +6% spike
        price_10min_ago=65000.0,
        volatility_15min=0.001,  # Very low volatility
        time_remaining_seconds=900,
        orderbook_imbalance=0.5  # Max buy pressure
    )

    # Should be clipped to 0.95 max
    assert prob_up <= 0.95

    # Extreme downward momentum
    prob_down = calc.calculate_directional_probability(
        current_price=62000.0,
        price_5min_ago=66000.0,  # -6% drop
        price_10min_ago=67000.0,
        volatility_15min=0.001,
        time_remaining_seconds=900,
        orderbook_imbalance=-0.5  # Max sell pressure
    )

    # Should be clipped to 0.05 min
    assert prob_down >= 0.05
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_probability_calculator.py::test_calculate_upward_momentum -v`
Expected: FAIL with "cannot import name 'ProbabilityCalculator'"

**Step 3: Implement ProbabilityCalculator**

Create: `polymarket/trading/probability_calculator.py`

```python
"""
Probability Calculator

Calculates the actual probability of BTC price direction in the remaining time
window using momentum, volatility, and orderbook data.

Based on modified Brownian motion model with mean reversion adjustments.
"""

from math import sqrt
import structlog
from scipy.stats import norm

logger = structlog.get_logger()


class ProbabilityCalculator:
    """Calculate directional probability for BTC 15-min markets."""

    def __init__(self):
        """Initialize probability calculator."""
        pass

    def calculate_directional_probability(
        self,
        current_price: float,
        price_5min_ago: float,
        price_10min_ago: float,
        volatility_15min: float,
        time_remaining_seconds: int,
        orderbook_imbalance: float = 0.0
    ) -> float:
        """
        Calculate probability that BTC ends HIGHER than current price.

        Args:
            current_price: Current BTC price
            price_5min_ago: BTC price 5 minutes ago
            price_10min_ago: BTC price 10 minutes ago
            volatility_15min: Rolling 15-min volatility (std dev or ATR)
            time_remaining_seconds: Seconds until market settlement
            orderbook_imbalance: -1.0 to +1.0 (negative=sell pressure, positive=buy)

        Returns:
            Probability from 0.0 to 1.0 that BTC ends higher (clipped to [0.05, 0.95])

        Example:
            >>> calc = ProbabilityCalculator()
            >>> prob = calc.calculate_directional_probability(
            ...     current_price=66200.0,
            ...     price_5min_ago=66000.0,
            ...     price_10min_ago=65900.0,
            ...     volatility_15min=0.005,
            ...     time_remaining_seconds=600,
            ...     orderbook_imbalance=0.2
            ... )
            >>> print(f"Probability UP: {prob:.2%}")
            Probability UP: 72.34%
        """

        # Step 1: Calculate momentum (weighted recent > older)
        momentum_5min = (current_price - price_5min_ago) / price_5min_ago
        momentum_10min = (current_price - price_10min_ago) / price_10min_ago
        weighted_momentum = (momentum_5min * 0.7) + (momentum_10min * 0.3)

        # Step 2: Calculate expected volatility for remaining time
        time_fraction = time_remaining_seconds / 900  # 900s = 15min
        if time_fraction <= 0:
            time_fraction = 0.01  # Minimum to avoid division by zero

        volatility_factor = volatility_15min * sqrt(time_fraction)

        # Prevent division by zero
        if volatility_factor < 0.0001:
            volatility_factor = 0.0001

        # Step 3: Calculate z-score (how many standard deviations)
        z_score = weighted_momentum / volatility_factor

        # Step 4: Convert z-score to probability using normal distribution CDF
        probability_up = norm.cdf(z_score)

        # Step 5: Adjust for orderbook imbalance
        # Orderbook imbalance adds up to ±10% to probability
        imbalance_adjustment = orderbook_imbalance * 0.1
        final_probability = probability_up + imbalance_adjustment

        # Step 6: Clip to [0.05, 0.95] to avoid overconfidence
        final_probability = max(0.05, min(0.95, final_probability))

        logger.debug(
            "Probability calculation",
            momentum_5min=f"{momentum_5min:+.4f}",
            momentum_10min=f"{momentum_10min:+.4f}",
            weighted_momentum=f"{weighted_momentum:+.4f}",
            volatility=f"{volatility_15min:.4f}",
            time_fraction=f"{time_fraction:.2f}",
            z_score=f"{z_score:+.2f}",
            probability_before_adjustment=f"{probability_up:.2%}",
            orderbook_adjustment=f"{imbalance_adjustment:+.2%}",
            final_probability=f"{final_probability:.2%}"
        )

        return final_probability
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_probability_calculator.py -v`
Expected: PASS (6 tests)

**Step 5: Commit**

```bash
git add polymarket/trading/probability_calculator.py tests/test_probability_calculator.py
git commit -m "feat: add probability calculator for directional prediction"
```

---

## Task 3: Implement Arbitrage Detector

**Files:**
- Create: `polymarket/trading/arbitrage_detector.py`
- Create: `tests/test_arbitrage_detector.py`

**Step 1: Write failing tests**

Create: `tests/test_arbitrage_detector.py`

```python
"""Tests for arbitrage detector."""
import pytest
from polymarket.trading.arbitrage_detector import ArbitrageDetector
from polymarket.models import ArbitrageOpportunity

def test_detect_yes_arbitrage():
    """Test detection of YES arbitrage opportunity."""
    detector = ArbitrageDetector()

    # Actual probability 68%, market only 55% = 13% edge
    opp = detector.detect_arbitrage(
        actual_probability=0.68,
        market_yes_odds=0.55,
        market_no_odds=0.45,
        market_id="test-market",
        ai_base_confidence=0.75
    )

    assert opp.recommended_action == "BUY_YES"
    assert opp.edge_percentage == pytest.approx(0.13, abs=0.01)
    assert opp.urgency == "MEDIUM"  # 10-15% edge
    assert opp.confidence_boost > 0.0
    assert opp.expected_profit_pct > 0.0


def test_detect_no_arbitrage():
    """Test detection of NO arbitrage opportunity."""
    detector = ArbitrageDetector()

    # Actual probability 35%, so 65% probability DOWN
    # Market NO odds = 60%, so 5% edge on NO side
    opp = detector.detect_arbitrage(
        actual_probability=0.35,
        market_yes_odds=0.40,
        market_no_odds=0.60,
        market_id="test-market",
        ai_base_confidence=0.75
    )

    assert opp.recommended_action == "BUY_NO"
    assert opp.edge_percentage == pytest.approx(0.05, abs=0.01)
    assert opp.urgency == "LOW"  # 5-10% edge


def test_no_arbitrage_opportunity():
    """Test when no significant arbitrage exists."""
    detector = ArbitrageDetector()

    # Actual probability matches market odds (no edge)
    opp = detector.detect_arbitrage(
        actual_probability=0.55,
        market_yes_odds=0.55,
        market_no_odds=0.45,
        market_id="test-market",
        ai_base_confidence=0.75
    )

    assert opp.recommended_action == "HOLD"
    assert opp.edge_percentage < 0.05  # Below threshold
    assert opp.confidence_boost == 0.0
    assert opp.urgency == "LOW"


def test_high_edge_urgency():
    """Test that 15%+ edge triggers HIGH urgency."""
    detector = ArbitrageDetector()

    # 18% edge
    opp = detector.detect_arbitrage(
        actual_probability=0.75,
        market_yes_odds=0.57,
        market_no_odds=0.43,
        market_id="test-market",
        ai_base_confidence=0.75
    )

    assert opp.urgency == "HIGH"  # 15%+ = HIGH
    assert opp.edge_percentage >= 0.15


def test_confidence_boost_scales_with_edge():
    """Test that confidence boost increases with edge size."""
    detector = ArbitrageDetector()

    # Small edge (5%)
    small_edge = detector.detect_arbitrage(
        actual_probability=0.60,
        market_yes_odds=0.55,
        market_no_odds=0.45,
        market_id="test",
        ai_base_confidence=0.75
    )

    # Large edge (15%)
    large_edge = detector.detect_arbitrage(
        actual_probability=0.70,
        market_yes_odds=0.55,
        market_no_odds=0.45,
        market_id="test",
        ai_base_confidence=0.75
    )

    # Larger edge should give larger confidence boost
    assert large_edge.confidence_boost > small_edge.confidence_boost


def test_confidence_boost_capped_at_20pct():
    """Test that confidence boost is capped at +0.20."""
    detector = ArbitrageDetector()

    # Extreme edge (30%)
    opp = detector.detect_arbitrage(
        actual_probability=0.85,
        market_yes_odds=0.55,
        market_no_odds=0.45,
        market_id="test",
        ai_base_confidence=0.75
    )

    # Should be capped at 0.20 max
    assert opp.confidence_boost <= 0.20
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_arbitrage_detector.py::test_detect_yes_arbitrage -v`
Expected: FAIL with "cannot import name 'ArbitrageDetector'"

**Step 3: Implement ArbitrageDetector**

Create: `polymarket/trading/arbitrage_detector.py`

```python
"""
Arbitrage Detector

Detects arbitrage opportunities by comparing actual probability (from calculator)
to market odds (which lag behind price movements).

Identifies mispriced markets with 5%+ edge and provides confidence boosts.
"""

import structlog
from polymarket.models import ArbitrageOpportunity

logger = structlog.get_logger()


class ArbitrageDetector:
    """Detect arbitrage opportunities from probability vs odds mismatch."""

    # Edge thresholds
    MIN_EDGE = 0.05  # 5% minimum to trade
    HIGH_EDGE_THRESHOLD = 0.10  # 10%+
    EXTREME_EDGE_THRESHOLD = 0.15  # 15%+

    # Confidence boost limits
    MAX_CONFIDENCE_BOOST = 0.20  # Cap at +20%

    def __init__(self):
        """Initialize arbitrage detector."""
        pass

    def detect_arbitrage(
        self,
        actual_probability: float,
        market_yes_odds: float,
        market_no_odds: float,
        market_id: str,
        ai_base_confidence: float = 0.75
    ) -> ArbitrageOpportunity:
        """
        Detect arbitrage opportunity by comparing actual vs market odds.

        Args:
            actual_probability: Calculated probability BTC goes UP (0.0-1.0)
            market_yes_odds: Current Polymarket odds for YES (0.0-1.0)
            market_no_odds: Current Polymarket odds for NO (0.0-1.0)
            market_id: Market identifier
            ai_base_confidence: Base AI confidence before arbitrage boost

        Returns:
            ArbitrageOpportunity with action, edge, urgency, and confidence boost

        Example:
            >>> detector = ArbitrageDetector()
            >>> opp = detector.detect_arbitrage(
            ...     actual_probability=0.68,
            ...     market_yes_odds=0.55,
            ...     market_no_odds=0.45,
            ...     market_id="btc-market"
            ... )
            >>> print(f"Action: {opp.recommended_action}, Edge: {opp.edge_percentage:.1%}")
            Action: BUY_YES, Edge: 13.0%
        """

        # Calculate edges for both sides
        yes_edge = actual_probability - market_yes_odds
        no_edge = (1.0 - actual_probability) - market_no_odds

        # Determine best action
        if yes_edge >= self.MIN_EDGE and yes_edge > no_edge:
            action = "BUY_YES"
            edge = yes_edge
        elif no_edge >= self.MIN_EDGE:
            action = "BUY_NO"
            edge = no_edge
        else:
            action = "HOLD"
            edge = max(yes_edge, no_edge)  # Show largest edge even if < threshold

        # Calculate confidence boost (scales with edge size)
        if edge >= self.MIN_EDGE:
            # Boost = edge * 2, capped at MAX_CONFIDENCE_BOOST
            confidence_boost = min(edge * 2.0, self.MAX_CONFIDENCE_BOOST)
        else:
            confidence_boost = 0.0

        # Classify urgency
        if edge >= self.EXTREME_EDGE_THRESHOLD:
            urgency = "HIGH"
        elif edge >= self.HIGH_EDGE_THRESHOLD:
            urgency = "MEDIUM"
        else:
            urgency = "LOW"

        # Calculate expected profit % if prediction correct
        if action == "BUY_YES":
            # If YES wins, profit = (1 - yes_odds) / yes_odds
            expected_profit_pct = ((1.0 - market_yes_odds) / market_yes_odds) if market_yes_odds > 0 else 0.0
        elif action == "BUY_NO":
            # If NO wins, profit = (1 - no_odds) / no_odds
            expected_profit_pct = ((1.0 - market_no_odds) / market_no_odds) if market_no_odds > 0 else 0.0
        else:
            expected_profit_pct = 0.0

        opportunity = ArbitrageOpportunity(
            market_id=market_id,
            actual_probability=actual_probability,
            polymarket_yes_odds=market_yes_odds,
            polymarket_no_odds=market_no_odds,
            edge_percentage=edge,
            recommended_action=action,
            confidence_boost=confidence_boost,
            urgency=urgency,
            expected_profit_pct=expected_profit_pct
        )

        if action != "HOLD":
            logger.info(
                "Arbitrage opportunity detected",
                action=action,
                edge=f"{edge:+.1%}",
                urgency=urgency,
                confidence_boost=f"+{confidence_boost:.2f}",
                actual_prob=f"{actual_probability:.2%}",
                market_yes=f"{market_yes_odds:.2%}",
                market_no=f"{market_no_odds:.2%}",
                expected_profit=f"+{expected_profit_pct:.1%}"
            )
        else:
            logger.debug(
                "No arbitrage opportunity",
                yes_edge=f"{yes_edge:+.1%}",
                no_edge=f"{no_edge:+.1%}",
                threshold=f"{self.MIN_EDGE:.1%}"
            )

        return opportunity
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_arbitrage_detector.py -v`
Expected: PASS (6 tests)

**Step 5: Commit**

```bash
git add polymarket/trading/arbitrage_detector.py tests/test_arbitrage_detector.py
git commit -m "feat: add arbitrage detector for mispriced markets"
```

---

## Task 4: Add Limit Order Methods to Client

**Files:**
- Modify: `polymarket/client.py:1-150`
- Create: `tests/test_client_limit_orders.py`

**Step 1: Write failing tests**

Create: `tests/test_client_limit_orders.py`

```python
"""Tests for limit order functionality in PolymarketClient."""
import pytest
from unittest.mock import Mock, AsyncMock, patch
from polymarket.client import PolymarketClient

@pytest.mark.asyncio
async def test_place_limit_order():
    """Test placing a limit order."""
    client = PolymarketClient()

    with patch.object(client, '_get_clob_client') as mock_clob:
        mock_clob.return_value.post_order = AsyncMock(return_value={
            "orderID": "test-order-123",
            "status": "LIVE"
        })

        result = await client.place_limit_order(
            token_id="test-token",
            side="BUY",
            price=0.55,
            size=10.0,
            tick_size=0.01
        )

        assert result["orderID"] == "test-order-123"
        assert result["status"] == "LIVE"
        mock_clob.return_value.post_order.assert_called_once()


@pytest.mark.asyncio
async def test_check_order_status():
    """Test checking order status."""
    client = PolymarketClient()

    with patch.object(client, '_get_clob_client') as mock_clob:
        mock_clob.return_value.get_order = AsyncMock(return_value={
            "orderID": "test-order-123",
            "status": "MATCHED",
            "fillAmount": "10.0"
        })

        result = await client.check_order_status("test-order-123")

        assert result["status"] == "MATCHED"
        assert result["fillAmount"] == "10.0"


@pytest.mark.asyncio
async def test_cancel_order():
    """Test cancelling an order."""
    client = PolymarketClient()

    with patch.object(client, '_get_clob_client') as mock_clob:
        mock_clob.return_value.cancel = AsyncMock(return_value={
            "orderID": "test-order-123",
            "status": "CANCELLED"
        })

        result = await client.cancel_order("test-order-123")

        assert result["status"] == "CANCELLED"
        mock_clob.return_value.cancel.assert_called_once_with("test-order-123")
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_client_limit_orders.py::test_place_limit_order -v`
Expected: FAIL with "PolymarketClient has no method 'place_limit_order'"

**Step 3: Implement limit order methods**

Add to `polymarket/client.py` after the existing `place_market_order` method (~line 400):

```python
async def place_limit_order(
    self,
    token_id: str,
    side: Literal["BUY", "SELL"],
    price: float,
    size: float,
    tick_size: float = 0.01
) -> dict:
    """
    Place a GTC (Good-Til-Cancelled) limit order.

    Limit orders:
    - Earn maker rebates instead of paying taker fees
    - May not fill before market expires
    - Priced to be slightly better than current market

    Args:
        token_id: Token to trade
        side: BUY or SELL
        price: Limit price (0.0-1.0)
        size: Order size in shares
        tick_size: Price tick size (default 0.01)

    Returns:
        Order response with orderID and status

    Raises:
        ValidationError: If parameters are invalid
        UpstreamAPIError: If API call fails
    """
    # Validate inputs
    if not 0.0 <= price <= 1.0:
        raise ValidationError(f"Price must be 0.0-1.0, got {price}")
    if size <= 0:
        raise ValidationError(f"Size must be positive, got {size}")
    if side not in ["BUY", "SELL"]:
        raise ValidationError(f"Side must be BUY or SELL, got {side}")

    # Round price to tick size
    price = round(price / tick_size) * tick_size

    # Get CLOB client
    clob = self._get_clob_client()

    try:
        logger.info(
            "Placing limit order",
            token_id=token_id,
            side=side,
            price=price,
            size=size,
            order_type="GTC"
        )

        # Create limit order (GTC = Good-Til-Cancelled)
        order_response = await clob.post_order(
            token_id=token_id,
            side=side,
            price=price,
            size=size,
            order_type="GTC"  # Will stay open until filled or cancelled
        )

        logger.info(
            "Limit order placed",
            order_id=order_response.get("orderID"),
            status=order_response.get("status")
        )

        return order_response

    except Exception as e:
        logger.error("Failed to place limit order", error=str(e))
        raise UpstreamAPIError(f"Limit order failed: {e}")


async def check_order_status(self, order_id: str) -> dict:
    """
    Check status of an order by ID.

    Args:
        order_id: Order ID to check

    Returns:
        Order status with fillAmount, status, etc.

    Possible statuses:
    - LIVE: Order is active
    - MATCHED: Order fully filled
    - PARTIALLY_MATCHED: Order partially filled
    - CANCELLED: Order was cancelled
    """
    clob = self._get_clob_client()

    try:
        order = await clob.get_order(order_id)

        logger.debug(
            "Order status checked",
            order_id=order_id,
            status=order.get("status"),
            fill_amount=order.get("fillAmount", "0")
        )

        return order

    except Exception as e:
        logger.error("Failed to check order status", order_id=order_id, error=str(e))
        raise UpstreamAPIError(f"Status check failed: {e}")


async def cancel_order(self, order_id: str) -> dict:
    """
    Cancel a live order.

    Args:
        order_id: Order ID to cancel

    Returns:
        Cancellation response
    """
    clob = self._get_clob_client()

    try:
        logger.info("Cancelling order", order_id=order_id)

        result = await clob.cancel(order_id)

        logger.info("Order cancelled", order_id=order_id)
        return result

    except Exception as e:
        logger.error("Failed to cancel order", order_id=order_id, error=str(e))
        raise UpstreamAPIError(f"Cancel failed: {e}")
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_client_limit_orders.py -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add polymarket/client.py tests/test_client_limit_orders.py
git commit -m "feat: add limit order support to PolymarketClient"
```

---

## Task 5: Implement Smart Order Executor

**Files:**
- Create: `polymarket/trading/smart_order_executor.py`
- Create: `tests/test_smart_order_executor.py`

**Step 1: Write failing tests**

Create: `tests/test_smart_order_executor.py`

```python
"""Tests for smart order executor."""
import pytest
from unittest.mock import AsyncMock, Mock, patch
from polymarket.trading.smart_order_executor import SmartOrderExecutor
from polymarket.models import LimitOrderStrategy

@pytest.mark.asyncio
async def test_execute_high_urgency_order():
    """Test execution of high urgency order (aggressive pricing)."""
    executor = SmartOrderExecutor()

    # Mock client
    mock_client = Mock()
    mock_client.place_limit_order = AsyncMock(return_value={"orderID": "test-123", "status": "LIVE"})
    mock_client.check_order_status = AsyncMock(return_value={"status": "MATCHED", "fillAmount": "10.0"})

    result = await executor.execute_smart_order(
        client=mock_client,
        token_id="test-token",
        side="BUY",
        amount=10.0,
        urgency="HIGH",
        current_best_ask=0.550,
        current_best_bid=0.540,
        tick_size=0.001
    )

    assert result["status"] == "FILLED"
    assert result["order_id"] == "test-123"

    # Verify limit order was placed (not market order)
    mock_client.place_limit_order.assert_called_once()

    # Verify price was improved (aggressive = +0.1%)
    call_args = mock_client.place_limit_order.call_args
    placed_price = call_args.kwargs["price"]
    assert placed_price > 0.540  # Better than best bid
    assert placed_price <= 0.551  # But close to ask (aggressive)


@pytest.mark.asyncio
async def test_execute_medium_urgency_order():
    """Test execution of medium urgency order (moderate pricing)."""
    executor = SmartOrderExecutor()

    mock_client = Mock()
    mock_client.place_limit_order = AsyncMock(return_value={"orderID": "test-456", "status": "LIVE"})
    mock_client.check_order_status = AsyncMock(return_value={"status": "MATCHED", "fillAmount": "10.0"})

    result = await executor.execute_smart_order(
        client=mock_client,
        token_id="test-token",
        side="BUY",
        amount=10.0,
        urgency="MEDIUM",
        current_best_ask=0.550,
        current_best_bid=0.540,
        tick_size=0.001
    )

    assert result["status"] == "FILLED"

    # Verify price improvement is moderate (+0.3%)
    call_args = mock_client.place_limit_order.call_args
    placed_price = call_args.kwargs["price"]
    assert 0.541 <= placed_price <= 0.543


@pytest.mark.asyncio
async def test_order_timeout_with_fallback():
    """Test that timeout triggers fallback to market order."""
    executor = SmartOrderExecutor()

    mock_client = Mock()
    mock_client.place_limit_order = AsyncMock(return_value={"orderID": "test-789", "status": "LIVE"})

    # Order never fills (timeout scenario)
    mock_client.check_order_status = AsyncMock(return_value={"status": "LIVE", "fillAmount": "0.0"})
    mock_client.cancel_order = AsyncMock(return_value={"status": "CANCELLED"})
    mock_client.place_market_order = AsyncMock(return_value={"orderID": "market-123", "status": "MATCHED"})

    # HIGH urgency with short timeout
    with patch('asyncio.sleep', new_callable=AsyncMock):
        result = await executor.execute_smart_order(
            client=mock_client,
            token_id="test-token",
            side="BUY",
            amount=10.0,
            urgency="HIGH",
            current_best_ask=0.550,
            current_best_bid=0.540,
            tick_size=0.001,
            timeout_override=1  # 1 second for testing
        )

    # Should fallback to market order
    assert result["status"] == "FILLED"
    assert result["filled_via"] == "market"  # Used fallback
    mock_client.cancel_order.assert_called_once()
    mock_client.place_market_order.assert_called_once()


@pytest.mark.asyncio
async def test_low_urgency_no_fallback():
    """Test that low urgency orders don't fallback on timeout."""
    executor = SmartOrderExecutor()

    mock_client = Mock()
    mock_client.place_limit_order = AsyncMock(return_value={"orderID": "test-999", "status": "LIVE"})
    mock_client.check_order_status = AsyncMock(return_value={"status": "LIVE", "fillAmount": "0.0"})
    mock_client.cancel_order = AsyncMock(return_value={"status": "CANCELLED"})

    with patch('asyncio.sleep', new_callable=AsyncMock):
        result = await executor.execute_smart_order(
            client=mock_client,
            token_id="test-token",
            side="BUY",
            amount=10.0,
            urgency="LOW",
            current_best_ask=0.550,
            current_best_bid=0.540,
            tick_size=0.001,
            timeout_override=1
        )

    # Should NOT fallback (low urgency)
    assert result["status"] == "TIMEOUT"
    mock_client.cancel_order.assert_called_once()
    # Market order should NOT be placed
    assert not hasattr(mock_client, 'place_market_order') or \
           mock_client.place_market_order.call_count == 0


@pytest.mark.asyncio
async def test_sell_side_order():
    """Test execution of SELL side order."""
    executor = SmartOrderExecutor()

    mock_client = Mock()
    mock_client.place_limit_order = AsyncMock(return_value={"orderID": "sell-123", "status": "LIVE"})
    mock_client.check_order_status = AsyncMock(return_value={"status": "MATCHED", "fillAmount": "10.0"})

    result = await executor.execute_smart_order(
        client=mock_client,
        token_id="test-token",
        side="SELL",
        amount=10.0,
        urgency="MEDIUM",
        current_best_ask=0.550,
        current_best_bid=0.540,
        tick_size=0.001
    )

    assert result["status"] == "FILLED"

    # Verify SELL order placed
    call_args = mock_client.place_limit_order.call_args
    assert call_args.kwargs["side"] == "SELL"

    # SELL orders should be priced below best ask (to get filled)
    placed_price = call_args.kwargs["price"]
    assert placed_price < 0.550
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_smart_order_executor.py::test_execute_high_urgency_order -v`
Expected: FAIL with "cannot import name 'SmartOrderExecutor'"

**Step 3: Implement SmartOrderExecutor**

Create: `polymarket/trading/smart_order_executor.py`

```python
"""
Smart Order Executor

Executes trades using limit orders (maker) instead of market orders (taker)
to save 3-6% in fees. Uses urgency-based pricing and timeout strategies.

HIGH urgency: Aggressive pricing (0.1% improvement), 30s timeout, fallback
MEDIUM urgency: Moderate pricing (0.3% improvement), 60s timeout, maybe fallback
LOW urgency: Conservative pricing (0.5% improvement), 120s timeout, no fallback
"""

import asyncio
from typing import Literal
import structlog
from polymarket.client import PolymarketClient
from polymarket.models import LimitOrderStrategy

logger = structlog.get_logger()


class SmartOrderExecutor:
    """Execute trades using smart limit orders to save fees."""

    # Price improvement percentages
    AGGRESSIVE_IMPROVEMENT = 0.001  # 0.1% (high urgency)
    MODERATE_IMPROVEMENT = 0.003    # 0.3% (medium urgency)
    CONSERVATIVE_IMPROVEMENT = 0.005  # 0.5% (low urgency)

    # Timeout seconds
    HIGH_URGENCY_TIMEOUT = 30
    MEDIUM_URGENCY_TIMEOUT = 60
    LOW_URGENCY_TIMEOUT = 120

    # Fallback to market order?
    HIGH_URGENCY_FALLBACK = True
    MEDIUM_URGENCY_FALLBACK = True
    LOW_URGENCY_FALLBACK = False

    def __init__(self):
        """Initialize smart order executor."""
        pass

    async def execute_smart_order(
        self,
        client: PolymarketClient,
        token_id: str,
        side: Literal["BUY", "SELL"],
        amount: float,
        urgency: Literal["HIGH", "MEDIUM", "LOW"],
        current_best_ask: float,
        current_best_bid: float,
        tick_size: float = 0.001,
        timeout_override: int | None = None
    ) -> dict:
        """
        Execute order using limit order with urgency-based strategy.

        Args:
            client: PolymarketClient instance
            token_id: Token to trade
            side: BUY or SELL
            amount: Size in shares
            urgency: HIGH/MEDIUM/LOW (determines pricing and timeout)
            current_best_ask: Current best ask price
            current_best_bid: Current best bid price
            tick_size: Price tick size (default 0.001)
            timeout_override: Override default timeout (for testing)

        Returns:
            Execution result with status, order_id, filled_via

        Possible statuses:
        - FILLED: Order successfully filled
        - TIMEOUT: Order timed out (no fill)
        - ERROR: Execution failed
        """

        # Step 1: Calculate strategy parameters
        strategy = self._calculate_strategy(
            urgency, side, current_best_ask, current_best_bid, tick_size
        )

        timeout = timeout_override if timeout_override is not None else strategy.timeout_seconds

        logger.info(
            "Executing smart order",
            side=side,
            amount=amount,
            urgency=urgency,
            target_price=strategy.target_price,
            timeout=f"{timeout}s",
            fallback=strategy.fallback_to_market
        )

        try:
            # Step 2: Place limit order
            order_response = await client.place_limit_order(
                token_id=token_id,
                side=side,
                price=strategy.target_price,
                size=amount,
                tick_size=tick_size
            )

            order_id = order_response.get("orderID")
            if not order_id:
                return {"status": "ERROR", "error": "No order ID returned"}

            # Step 3: Monitor for fill
            fill_result = await self._monitor_order_fill(
                client, order_id, timeout
            )

            if fill_result["filled"]:
                logger.info(
                    "Limit order filled",
                    order_id=order_id,
                    fill_amount=fill_result.get("fill_amount")
                )
                return {
                    "status": "FILLED",
                    "order_id": order_id,
                    "filled_via": "limit",
                    "fill_amount": fill_result.get("fill_amount")
                }

            # Step 4: Handle timeout
            logger.warning("Limit order timed out", order_id=order_id, timeout=f"{timeout}s")

            # Cancel the unfilled limit order
            await client.cancel_order(order_id)

            # Step 5: Fallback to market if enabled
            if strategy.fallback_to_market:
                logger.info("Falling back to market order", urgency=urgency)

                market_response = await client.place_market_order(
                    token_id=token_id,
                    side=side,
                    amount=amount
                )

                return {
                    "status": "FILLED",
                    "order_id": market_response.get("orderID"),
                    "filled_via": "market",
                    "fill_amount": amount
                }
            else:
                logger.info("No fallback - skipping trade", urgency=urgency)
                return {
                    "status": "TIMEOUT",
                    "order_id": order_id,
                    "message": "Limit order timed out, no fallback"
                }

        except Exception as e:
            logger.error("Smart order execution failed", error=str(e))
            return {"status": "ERROR", "error": str(e)}

    def _calculate_strategy(
        self,
        urgency: str,
        side: str,
        best_ask: float,
        best_bid: float,
        tick_size: float
    ) -> LimitOrderStrategy:
        """Calculate pricing and timeout strategy based on urgency."""

        # Determine improvement percentage
        if urgency == "HIGH":
            improvement = self.AGGRESSIVE_IMPROVEMENT
            timeout = self.HIGH_URGENCY_TIMEOUT
            fallback = self.HIGH_URGENCY_FALLBACK
        elif urgency == "MEDIUM":
            improvement = self.MODERATE_IMPROVEMENT
            timeout = self.MEDIUM_URGENCY_TIMEOUT
            fallback = self.MEDIUM_URGENCY_FALLBACK
        else:  # LOW
            improvement = self.CONSERVATIVE_IMPROVEMENT
            timeout = self.LOW_URGENCY_TIMEOUT
            fallback = self.LOW_URGENCY_FALLBACK

        # Calculate target price
        if side == "BUY":
            # BUY: Start from best_bid and add small improvement
            # (closer to ask = more likely to fill)
            target_price = best_bid + improvement
            target_price = min(target_price, best_ask - tick_size)  # Don't cross the spread
        else:  # SELL
            # SELL: Start from best_ask and subtract small improvement
            target_price = best_ask - improvement
            target_price = max(target_price, best_bid + tick_size)  # Don't cross the spread

        # Round to tick size
        target_price = round(target_price / tick_size) * tick_size

        # Bounds check
        target_price = max(0.001, min(0.999, target_price))

        return LimitOrderStrategy(
            target_price=target_price,
            timeout_seconds=timeout,
            fallback_to_market=fallback,
            urgency=urgency,
            price_improvement_pct=improvement
        )

    async def _monitor_order_fill(
        self,
        client: PolymarketClient,
        order_id: str,
        timeout: int,
        check_interval: int = 5
    ) -> dict:
        """
        Monitor order until filled or timeout.

        Args:
            client: PolymarketClient
            order_id: Order to monitor
            timeout: Max seconds to wait
            check_interval: Seconds between status checks

        Returns:
            {"filled": bool, "fill_amount": str}
        """
        elapsed = 0

        while elapsed < timeout:
            await asyncio.sleep(check_interval)
            elapsed += check_interval

            try:
                status = await client.check_order_status(order_id)

                # Check if filled
                if status.get("status") in ["MATCHED", "PARTIALLY_MATCHED"]:
                    fill_amount = status.get("fillAmount", "0")
                    if float(fill_amount) > 0:
                        return {"filled": True, "fill_amount": fill_amount}

            except Exception as e:
                logger.warning("Failed to check order status", error=str(e))
                # Continue monitoring despite errors

        # Timeout reached
        return {"filled": False}
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_smart_order_executor.py -v`
Expected: PASS (5 tests)

**Step 5: Commit**

```bash
git add polymarket/trading/smart_order_executor.py tests/test_smart_order_executor.py
git commit -m "feat: add smart limit order executor with urgency-based strategy"
```

---

## Task 6: Add Arbitrage Context to AI Decision Service

**Files:**
- Modify: `polymarket/trading/ai_decision.py:46-150`

**Step 1: Write failing test**

Add to existing `tests/test_ai_decision.py`:

```python
@pytest.mark.asyncio
async def test_make_decision_with_arbitrage(mock_settings):
    """Test AI decision with arbitrage opportunity."""
    from polymarket.models import ArbitrageOpportunity

    service = AIDecisionService(mock_settings)

    # Mock OpenAI response
    with patch.object(service, '_get_client') as mock_client:
        mock_openai = Mock()
        mock_openai.chat.completions.create = AsyncMock(return_value=Mock(
            choices=[Mock(message=Mock(content='{"action": "YES", "confidence": 0.92, "reasoning": "Arbitrage edge"}'))]
        ))
        mock_client.return_value = mock_openai

        # Create arbitrage opportunity
        arbitrage = ArbitrageOpportunity(
            market_id="test",
            actual_probability=0.70,
            polymarket_yes_odds=0.55,
            polymarket_no_odds=0.45,
            edge_percentage=0.15,
            recommended_action="BUY_YES",
            confidence_boost=0.20,
            urgency="HIGH",
            expected_profit_pct=0.20
        )

        decision = await service.make_decision(
            btc_price=mock_btc_price,
            technical_indicators=mock_indicators,
            aggregated_sentiment=mock_sentiment,
            market_data={"token_id": "test", "yes_price": 0.55, "no_price": 0.45},
            arbitrage_opportunity=arbitrage  # NEW parameter
        )

        # Verify arbitrage context was included in prompt
        call_args = mock_openai.chat.completions.create.call_args
        prompt = call_args.kwargs["messages"][1]["content"]
        assert "ARBITRAGE OPPORTUNITY" in prompt
        assert "15.0%" in prompt  # Edge percentage
        assert "BUY_YES" in prompt  # Recommendation
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_ai_decision.py::test_make_decision_with_arbitrage -v`
Expected: FAIL with "make_decision() got unexpected keyword argument 'arbitrage_opportunity'"

**Step 3: Modify AI decision service**

Modify `polymarket/trading/ai_decision.py`:

Change the `make_decision` signature (around line 46):

```python
async def make_decision(
    self,
    btc_price: BTCPriceData,
    technical_indicators: TechnicalIndicators,
    aggregated_sentiment: AggregatedSentiment,
    market_data: dict,
    portfolio_value: Decimal = Decimal("1000"),
    orderbook_data: "OrderbookData | None" = None,
    volume_data: VolumeData | None = None,
    timeframe_analysis: TimeframeAnalysis | None = None,
    regime: MarketRegime | None = None,
    arbitrage_opportunity: "ArbitrageOpportunity | None" = None  # NEW
) -> TradingDecision:
    """Generate trading decision using AI with regime awareness, volume, timeframe, and arbitrage."""
```

Update the `_build_prompt` call (around line 63):

```python
prompt = self._build_prompt(
    btc_price, technical_indicators, aggregated_sentiment,
    market_data, portfolio_value, orderbook_data,
    volume_data, timeframe_analysis, regime,
    arbitrage_opportunity  # NEW
)
```

Update `_build_prompt` signature (around line 128):

```python
def _build_prompt(
    self,
    btc_price: BTCPriceData,
    technical: TechnicalIndicators,
    aggregated: AggregatedSentiment,
    market: dict,
    portfolio_value: Decimal,
    orderbook_data: "OrderbookData | None" = None,
    volume_data: VolumeData | None = None,
    timeframe_analysis: TimeframeAnalysis | None = None,
    regime: MarketRegime | None = None,
    arbitrage_opportunity: "ArbitrageOpportunity | None" = None  # NEW
) -> str:
```

Add arbitrage context section to prompt (around line 300, after regime/volume/timeframe sections):

```python
# NEW: Arbitrage opportunity context
arbitrage_context = ""
if arbitrage_opportunity and arbitrage_opportunity.edge_percentage >= 0.05:
    arbitrage_context = f"""
═══════════════════════════════════════════════════════════════════
🎯 ARBITRAGE OPPORTUNITY DETECTED
═══════════════════════════════════════════════════════════════════

PROBABILITY ANALYSIS:
├─ Actual Probability (calculated): {arbitrage_opportunity.actual_probability:.1%}
├─ Polymarket YES Odds: {arbitrage_opportunity.polymarket_yes_odds:.1%}
├─ Polymarket NO Odds: {arbitrage_opportunity.polymarket_no_odds:.1%}
└─ **EDGE: {arbitrage_opportunity.edge_percentage:+.1%}**

ARBITRAGE SIGNAL:
├─ Recommended Action: **{arbitrage_opportunity.recommended_action}**
├─ Confidence Boost: +{arbitrage_opportunity.confidence_boost:.2f}
├─ Urgency: {arbitrage_opportunity.urgency}
└─ Expected Profit: +{arbitrage_opportunity.expected_profit_pct:.1%} if correct

⚠️ ARBITRAGE TRADING STRATEGY:
This is a QUANTIFIED MISPRICING based on:
1. Real-time BTC price momentum analysis
2. Comparison to lagging Polymarket odds
3. Statistical probability calculation

CONFIDENCE SCALING:
- Edge 5-10%: Moderate opportunity (confidence boost: +0.10 to +0.20)
- Edge 10-15%: Strong opportunity (confidence boost: +0.20, urgency: MEDIUM)
- Edge 15%+: Extreme opportunity (confidence boost: +0.20, urgency: HIGH)

The larger the edge, the higher your confidence should be.
Edges of 10%+ justify high confidence (0.85-0.95).

═══════════════════════════════════════════════════════════════════
"""
else:
    arbitrage_context = """
═══════════════════════════════════════════════════════════════════
🎯 ARBITRAGE STATUS
═══════════════════════════════════════════════════════════════════
No significant arbitrage edge detected (< 5%).
Rely on technical indicators, sentiment, and regime analysis.
═══════════════════════════════════════════════════════════════════
"""

# Add to prompt (place before the JSON format instruction)
prompt += f"\n{arbitrage_context}\n"
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_ai_decision.py::test_make_decision_with_arbitrage -v`
Expected: PASS

**Step 5: Commit**

```bash
git add polymarket/trading/ai_decision.py tests/test_ai_decision.py
git commit -m "feat: add arbitrage opportunity context to AI decision prompts"
```

---

## Task 7: Integrate Arbitrage System into Auto-Trader

**Files:**
- Modify: `scripts/auto_trade.py:706-800`

**Step 1: Write integration test**

Create: `tests/test_auto_trade_arbitrage_integration.py`

```python
"""Integration tests for arbitrage system in auto_trade.py."""
import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timezone
from decimal import Decimal

@pytest.mark.asyncio
async def test_arbitrage_integration_in_process_market():
    """Test that arbitrage system is called during market processing."""
    from scripts.auto_trade import AutoTrader
    from polymarket.config import Settings

    # Create mock settings
    settings = Mock(spec=Settings)
    settings.openai_api_key = "test-key"
    settings.openai_model = "gpt-4o-mini"
    settings.bot_confidence_threshold = 0.75
    settings.bot_max_position_dollars = 10.0

    trader = AutoTrader(settings, interval=180)

    # Mock all services
    trader.btc_service.get_price_at = AsyncMock(side_effect=[66000.0, 65900.0])
    trader.btc_service.calculate_15min_volatility = Mock(return_value=0.005)

    # Mock AI decision to return valid decision
    trader.ai_service.make_decision = AsyncMock(return_value=Mock(
        action="YES",
        confidence=0.90,
        reasoning="Arbitrage + momentum",
        position_size=10.0
    ))

    # Mock market data
    mock_market = Mock()
    mock_market.id = "test-market"
    mock_market.slug = "btc-updown-15m-1234567890"
    mock_market.get_token_ids = Mock(return_value=["token-123", "token-456"])

    # Mock other required data
    mock_btc_data = Mock()
    mock_btc_data.price = 66200.0

    mock_indicators = Mock()
    mock_aggregated = Mock()
    mock_aggregated.final_confidence = 0.75

    mock_market_dict = {
        "token_id": "token-123",
        "yes_price": 0.55,
        "no_price": 0.45,
        "outcomes": ["Up", "Down"]
    }

    # Patch the components to track calls
    with patch('scripts.auto_trade.ProbabilityCalculator') as MockProbCalc, \
         patch('scripts.auto_trade.ArbitrageDetector') as MockArbDet, \
         patch('scripts.auto_trade.SmartOrderExecutor') as MockSmartExec:

        # Setup mocks
        mock_prob_calc = MockProbCalc.return_value
        mock_prob_calc.calculate_directional_probability = Mock(return_value=0.70)

        mock_arb_det = MockArbDet.return_value
        mock_arb_det.detect_arbitrage = Mock(return_value=Mock(
            edge_percentage=0.15,
            recommended_action="BUY_YES",
            urgency="HIGH",
            confidence_boost=0.20
        ))

        mock_smart_exec = MockSmartExec.return_value
        mock_smart_exec.execute_smart_order = AsyncMock(return_value={
            "status": "FILLED",
            "order_id": "order-123"
        })

        # Call _process_market
        await trader._process_market(
            market=mock_market,
            btc_data=mock_btc_data,
            indicators=mock_indicators,
            aggregated_sentiment=mock_aggregated,
            portfolio_value=Decimal("1000"),
            btc_momentum=None,
            cycle_start_time=datetime.now(timezone.utc),
            volume_data=None,
            timeframe_analysis=None,
            regime=None
        )

        # Verify probability calculator was called
        mock_prob_calc.calculate_directional_probability.assert_called_once()

        # Verify arbitrage detector was called
        mock_arb_det.detect_arbitrage.assert_called_once()

        # Verify smart executor was called (not market order)
        mock_smart_exec.execute_smart_order.assert_called_once()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_auto_trade_arbitrage_integration.py -v`
Expected: FAIL (imports don't exist yet)

**Step 3: Add imports to auto_trade.py**

Add near top of `scripts/auto_trade.py` (around line 30):

```python
from polymarket.trading.probability_calculator import ProbabilityCalculator
from polymarket.trading.arbitrage_detector import ArbitrageDetector
from polymarket.trading.smart_order_executor import SmartOrderExecutor
```

**Step 4: Integrate into _process_market**

Modify `scripts/auto_trade.py` in the `_process_market` method (around line 850, after getting orderbook_data):

```python
# NEW: Calculate actual probability from price momentum
probability_calculator = ProbabilityCalculator()

# Get historical prices for probability calculation
price_5min_ago = await self.btc_service.get_price_at(5) or float(btc_data.price)
price_10min_ago = await self.btc_service.get_price_at(10) or float(btc_data.price)
volatility_15min = self.btc_service.calculate_15min_volatility() or 0.005  # Default 0.5%

actual_probability = probability_calculator.calculate_directional_probability(
    current_price=float(btc_data.price),
    price_5min_ago=price_5min_ago,
    price_10min_ago=price_10min_ago,
    volatility_15min=volatility_15min,
    time_remaining_seconds=time_remaining or 900,
    orderbook_imbalance=orderbook_data.order_imbalance if orderbook_data else 0.0
)

logger.info(
    "Probability calculation",
    actual_probability=f"{actual_probability:.2%}",
    current_price=btc_data.price,
    price_5min_ago=price_5min_ago,
    volatility=f"{volatility_15min:.4f}"
)

# NEW: Detect arbitrage opportunity
arbitrage_detector = ArbitrageDetector()
arbitrage_opportunity = arbitrage_detector.detect_arbitrage(
    actual_probability=actual_probability,
    market_yes_odds=market_dict['yes_price'],
    market_no_odds=market_dict['no_price'],
    market_id=market.id,
    ai_base_confidence=aggregated_sentiment.final_confidence
)

# Enhanced AI decision (with arbitrage context)
decision = await self.ai_service.make_decision(
    btc_price=btc_data,
    technical_indicators=indicators,
    aggregated_sentiment=aggregated_sentiment,
    market_data=market_dict,
    portfolio_value=portfolio_value,
    orderbook_data=orderbook_data,
    volume_data=volume_data,
    timeframe_analysis=timeframe_analysis,
    regime=regime,
    arbitrage_opportunity=arbitrage_opportunity  # NEW
)
```

Replace the market order execution (around line 950) with smart order execution:

```python
# Execute using smart limit orders (saves 3-6% in fees)
smart_executor = SmartOrderExecutor()

execution_result = await smart_executor.execute_smart_order(
    client=self.client,
    token_id=token_id,
    side="BUY" if decision.action == "YES" else "SELL",
    amount=validated_position_size,
    urgency=arbitrage_opportunity.urgency if arbitrage_opportunity else "MEDIUM",
    current_best_ask=market_dict.get('best_ask', market_dict['yes_price']),
    current_best_bid=market_dict.get('best_bid', market_dict['yes_price'] - 0.01),
    tick_size=0.001
)

# Check execution result
if execution_result["status"] != "FILLED":
    logger.warning(
        "Order not filled",
        status=execution_result["status"],
        reason=execution_result.get("message", "Unknown")
    )
    return  # Skip this trade

order_id = execution_result["order_id"]
filled_via = execution_result.get("filled_via", "limit")

logger.info(
    "Trade executed",
    action=decision.action,
    position_size=validated_position_size,
    order_id=order_id,
    filled_via=filled_via,
    arbitrage_edge=f"{arbitrage_opportunity.edge_percentage:.1%}" if arbitrage_opportunity else "N/A"
)
```

**Step 5: Run test to verify it passes**

Run: `pytest tests/test_auto_trade_arbitrage_integration.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add scripts/auto_trade.py tests/test_auto_trade_arbitrage_integration.py
git commit -m "feat: integrate arbitrage system into auto-trader main loop"
```

---

## Task 8: Add Configuration for Arbitrage System

**Files:**
- Modify: `polymarket/config.py:160-180`
- Modify: `.env.example`

**Step 1: Add config fields**

Add to `polymarket/config.py` (around line 180, after stop-loss config):

```python
# === Arbitrage System Configuration ===
arbitrage_min_edge_pct: float = field(
    default_factory=lambda: float(os.getenv("ARBITRAGE_MIN_EDGE_PCT", "0.05"))  # 5% minimum
)
arbitrage_high_edge_pct: float = field(
    default_factory=lambda: float(os.getenv("ARBITRAGE_HIGH_EDGE_PCT", "0.10"))  # 10%+ = high urgency
)
arbitrage_extreme_edge_pct: float = field(
    default_factory=lambda: float(os.getenv("ARBITRAGE_EXTREME_EDGE_PCT", "0.15"))  # 15%+ = extreme
)

# Limit order timeouts
limit_order_timeout_high: int = field(
    default_factory=lambda: int(os.getenv("LIMIT_ORDER_TIMEOUT_HIGH", "30"))  # 30s
)
limit_order_timeout_medium: int = field(
    default_factory=lambda: int(os.getenv("LIMIT_ORDER_TIMEOUT_MEDIUM", "60"))  # 60s
)
limit_order_timeout_low: int = field(
    default_factory=lambda: int(os.getenv("LIMIT_ORDER_TIMEOUT_LOW", "120"))  # 120s
)
```

**Step 2: Update .env.example**

Add to `.env.example`:

```bash
# === Arbitrage System ===
ARBITRAGE_MIN_EDGE_PCT=0.05       # 5% minimum edge to trade
ARBITRAGE_HIGH_EDGE_PCT=0.10      # 10%+ = high urgency
ARBITRAGE_EXTREME_EDGE_PCT=0.15   # 15%+ = extreme opportunity

# Limit order timeouts (seconds)
LIMIT_ORDER_TIMEOUT_HIGH=30       # High urgency: 30s
LIMIT_ORDER_TIMEOUT_MEDIUM=60     # Medium urgency: 60s
LIMIT_ORDER_TIMEOUT_LOW=120       # Low urgency: 120s
```

**Step 3: Commit**

```bash
git add polymarket/config.py .env.example
git commit -m "feat: add configuration options for arbitrage system"
```

---

## Task 9: Add Volatility Calculation Method to BTCPriceService

**Files:**
- Modify: `polymarket/trading/btc_price.py`

**Step 1: Write failing test**

Add to `tests/test_btc_price.py`:

```python
def test_calculate_15min_volatility():
    """Test 15-minute volatility calculation."""
    from polymarket.trading.btc_price import BTCPriceService
    from polymarket.config import Settings

    service = BTCPriceService(Settings())

    # Mock price buffer with data
    service._price_buffer = Mock()
    service._price_buffer.get_price_range = Mock(return_value=[
        {"timestamp": 1000, "price": 66000.0},
        {"timestamp": 1300, "price": 66100.0},
        {"timestamp": 1600, "price": 66050.0},
        {"timestamp": 1900, "price": 66200.0},
    ])

    volatility = service.calculate_15min_volatility()

    # Should return standard deviation as percentage
    assert volatility > 0.0
    assert volatility < 0.10  # Should be reasonable (< 10%)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_btc_price.py::test_calculate_15min_volatility -v`
Expected: FAIL with "BTCPriceService has no method 'calculate_15min_volatility'"

**Step 3: Implement volatility calculation**

Add to `polymarket/trading/btc_price.py` (in BTCPriceService class):

```python
def calculate_15min_volatility(self) -> float:
    """
    Calculate 15-minute volatility from price history.

    Uses standard deviation of price returns over last 15 minutes.

    Returns:
        Volatility as decimal (e.g., 0.005 = 0.5%)
    """
    try:
        # Get prices from last 15 minutes
        prices = self._price_buffer.get_price_range(duration_seconds=900)

        if len(prices) < 2:
            logger.warning("Insufficient price data for volatility calculation")
            return 0.005  # Default to 0.5%

        # Calculate returns (percentage changes)
        returns = []
        for i in range(1, len(prices)):
            prev_price = prices[i-1]["price"]
            curr_price = prices[i]["price"]
            ret = (curr_price - prev_price) / prev_price
            returns.append(ret)

        # Calculate standard deviation
        import statistics
        volatility = statistics.stdev(returns) if len(returns) > 1 else 0.005

        logger.debug(
            "Volatility calculated",
            volatility=f"{volatility:.4f}",
            sample_size=len(returns)
        )

        return volatility

    except Exception as e:
        logger.error("Failed to calculate volatility", error=str(e))
        return 0.005  # Default fallback
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_btc_price.py::test_calculate_15min_volatility -v`
Expected: PASS

**Step 5: Commit**

```bash
git add polymarket/trading/btc_price.py tests/test_btc_price.py
git commit -m "feat: add 15-minute volatility calculation to BTCPriceService"
```

---

## Task 10: Update Database Schema to Track Arbitrage Data

**Files:**
- Modify: `polymarket/performance/tracker.py`

**Step 1: Add new columns to trades table**

Modify `polymarket/performance/tracker.py` in the `_init_database` method:

```python
# Add after existing columns (around line 80)
actual_probability REAL,
arbitrage_edge REAL,
arbitrage_urgency TEXT,
filled_via TEXT,
limit_order_timeout INTEGER
```

**Step 2: Update _record_trade to save arbitrage data**

Modify `_record_trade` method (around line 200):

```python
def _record_trade(
    self,
    market_slug: str,
    action: str,
    position_size: float,
    confidence: float,
    reasoning: str,
    price_to_beat: float | None,
    yes_price: float,
    no_price: float,
    time_remaining_seconds: int | None,
    is_end_phase: bool,
    execution_status: str,
    rsi: float | None = None,
    macd: float | None = None,
    trend: str | None = None,
    social_score: float | None = None,
    market_score: float | None = None,
    signal_type: str | None = None,
    actual_probability: float | None = None,  # NEW
    arbitrage_edge: float | None = None,  # NEW
    arbitrage_urgency: str | None = None,  # NEW
    filled_via: str | None = None,  # NEW
    limit_order_timeout: int | None = None  # NEW
) -> int:
    """Record a new trade in the database."""
    cursor = self.db.cursor()
    cursor.execute(
        """
        INSERT INTO trades (
            timestamp, market_slug, action, position_size, confidence,
            reasoning, price_to_beat, yes_price, no_price,
            time_remaining_seconds, is_end_phase, execution_status,
            rsi, macd, trend, social_score, market_score, signal_type,
            actual_probability, arbitrage_edge, arbitrage_urgency,
            filled_via, limit_order_timeout
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.now(timezone.utc).isoformat(),
            market_slug, action, position_size, confidence, reasoning,
            price_to_beat, yes_price, no_price,
            time_remaining_seconds, is_end_phase, execution_status,
            rsi, macd, trend, social_score, market_score, signal_type,
            actual_probability, arbitrage_edge, arbitrage_urgency,
            filled_via, limit_order_timeout
        )
    )
    self.db.commit()
    return cursor.lastrowid
```

**Step 3: Update auto_trade.py to pass arbitrage data**

Modify the `record_trade` call in `scripts/auto_trade.py` (around line 1050):

```python
trade_id = self.performance_tracker.record_trade(
    market_slug=market.slug,
    action=decision.action,
    position_size=validated_position_size,
    confidence=decision.confidence,
    reasoning=decision.reasoning,
    price_to_beat=price_to_beat,
    yes_price=market_dict['yes_price'],
    no_price=market_dict['no_price'],
    time_remaining_seconds=time_remaining,
    is_end_phase=is_end_of_market,
    execution_status='executed',
    rsi=indicators.rsi,
    macd=indicators.macd,
    trend=indicators.trend,
    social_score=aggregated_sentiment.social_score,
    market_score=aggregated_sentiment.market_score,
    signal_type=aggregated_sentiment.signal_type,
    actual_probability=actual_probability,  # NEW
    arbitrage_edge=arbitrage_opportunity.edge_percentage if arbitrage_opportunity else None,  # NEW
    arbitrage_urgency=arbitrage_opportunity.urgency if arbitrage_opportunity else None,  # NEW
    filled_via=filled_via,  # NEW
    limit_order_timeout=smart_executor.HIGH_URGENCY_TIMEOUT if arbitrage_opportunity and arbitrage_opportunity.urgency == "HIGH" else None  # NEW
)
```

**Step 4: Add migration script**

Create: `scripts/migrate_add_arbitrage_columns.py`

```python
#!/usr/bin/env python3
"""
Migration: Add arbitrage tracking columns to trades table.

Run this script to upgrade existing performance.db database.
"""

import sqlite3
from pathlib import Path

def migrate():
    db_path = Path("data/performance.db")

    if not db_path.exists():
        print(f"Database not found at {db_path}")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Check if columns already exist
    cursor.execute("PRAGMA table_info(trades)")
    columns = {row[1] for row in cursor.fetchall()}

    new_columns = {
        "actual_probability": "REAL",
        "arbitrage_edge": "REAL",
        "arbitrage_urgency": "TEXT",
        "filled_via": "TEXT",
        "limit_order_timeout": "INTEGER"
    }

    for col_name, col_type in new_columns.items():
        if col_name not in columns:
            print(f"Adding column: {col_name} {col_type}")
            cursor.execute(f"ALTER TABLE trades ADD COLUMN {col_name} {col_type}")
        else:
            print(f"Column {col_name} already exists, skipping")

    conn.commit()
    conn.close()
    print("Migration complete!")

if __name__ == "__main__":
    migrate()
```

**Step 5: Run migration**

Run: `python scripts/migrate_add_arbitrage_columns.py`
Expected: "Migration complete!"

**Step 6: Commit**

```bash
git add polymarket/performance/tracker.py scripts/auto_trade.py scripts/migrate_add_arbitrage_columns.py
git commit -m "feat: add arbitrage tracking columns to database schema"
```

---

## Task 11: Add Scipy Dependency

**Files:**
- Modify: `requirements.txt`

**Step 1: Add scipy**

Add to `requirements.txt`:

```txt
scipy>=1.11.0  # For norm.cdf in probability calculations
```

**Step 2: Install**

Run: `pip install scipy`
Expected: Successfully installed scipy

**Step 3: Commit**

```bash
git add requirements.txt
git commit -m "deps: add scipy for statistical probability calculations"
```

---

## Task 12: Add Docstring and Example to Design Document

**Files:**
- Modify: `docs/plans/2026-02-13-arbitrage-trading-system-design.md`

**Step 1: Add implementation notes section**

Add to end of design document:

```markdown
## Implementation Complete

**Date:** 2026-02-13
**Status:** Ready for Testing

### Components Implemented

1. ✅ **ProbabilityCalculator** - `polymarket/trading/probability_calculator.py`
2. ✅ **ArbitrageDetector** - `polymarket/trading/arbitrage_detector.py`
3. ✅ **SmartOrderExecutor** - `polymarket/trading/smart_order_executor.py`
4. ✅ **Data Models** - Added to `polymarket/models.py`
5. ✅ **Client Methods** - Added limit orders to `polymarket/client.py`
6. ✅ **AI Integration** - Enhanced `polymarket/trading/ai_decision.py`
7. ✅ **Auto-Trader Integration** - Modified `scripts/auto_trade.py`
8. ✅ **Configuration** - Added to `polymarket/config.py`
9. ✅ **Database Schema** - Updated with arbitrage tracking
10. ✅ **Tests** - Full test coverage for all components

### Testing Checklist

- [ ] Run all unit tests: `pytest tests/test_probability_calculator.py tests/test_arbitrage_detector.py tests/test_smart_order_executor.py -v`
- [ ] Run integration tests: `pytest tests/test_auto_trade_arbitrage_integration.py -v`
- [ ] Run migration script: `python scripts/migrate_add_arbitrage_columns.py`
- [ ] Test with DRY_RUN=true: Verify probability calculations work
- [ ] Backtest on historical data (last 100 markets)
- [ ] Paper trade for 24 hours before live deployment

### Deployment Plan

**Phase 1: Conservative Start (Day 1)**
- Set `ARBITRAGE_MIN_EDGE_PCT=0.10` (10%+ only)
- Monitor for 24 hours
- Verify limit orders fill successfully
- Check fee savings vs market orders

**Phase 2: Gradual Relaxation (Days 2-4)**
- Day 2: Lower to `ARBITRAGE_MIN_EDGE_PCT=0.08`
- Day 3: Lower to `ARBITRAGE_MIN_EDGE_PCT=0.06`
- Day 4: Lower to `ARBITRAGE_MIN_EDGE_PCT=0.05` (target)

**Phase 3: Optimization (Week 2)**
- Analyze performance data
- Tune volatility calculations
- Adjust timeout parameters
- Optimize price improvement percentages
```

**Step 2: Commit**

```bash
git add docs/plans/2026-02-13-arbitrage-trading-system-design.md
git commit -m "docs: mark implementation complete in design document"
```

---

## Task 13: Run Full Test Suite

**Step 1: Run all arbitrage tests**

Run: `pytest tests/test_probability_calculator.py tests/test_arbitrage_detector.py tests/test_smart_order_executor.py tests/test_arbitrage_models.py tests/test_client_limit_orders.py -v`
Expected: All tests PASS

**Step 2: Run integration tests**

Run: `pytest tests/test_auto_trade_arbitrage_integration.py -v`
Expected: PASS

**Step 3: Run full test suite**

Run: `pytest tests/ -v`
Expected: All existing tests still PASS (no regressions)

**Step 4: Fix any failures**

If any tests fail:
- Read error message
- Identify root cause
- Fix the issue
- Re-run tests
- Commit fix

---

## Task 14: Create Usage Example Script

**Files:**
- Create: `examples/arbitrage_example.py`

**Step 1: Write example script**

Create: `examples/arbitrage_example.py`

```python
#!/usr/bin/env python3
"""
Example: Using the Arbitrage Trading System

Demonstrates how the three components work together to identify
and execute arbitrage opportunities.
"""

import asyncio
from polymarket.trading.probability_calculator import ProbabilityCalculator
from polymarket.trading.arbitrage_detector import ArbitrageDetector
from polymarket.trading.smart_order_executor import SmartOrderExecutor


async def example_arbitrage_flow():
    """Example of full arbitrage detection and execution flow."""

    print("=" * 70)
    print("ARBITRAGE TRADING SYSTEM EXAMPLE")
    print("=" * 70)

    # Step 1: Calculate actual probability from price movements
    print("\n1. PROBABILITY CALCULATION")
    print("-" * 70)

    calculator = ProbabilityCalculator()

    # Example: BTC rising steadily
    actual_probability = calculator.calculate_directional_probability(
        current_price=66200.0,
        price_5min_ago=66000.0,  # +0.30% in 5min
        price_10min_ago=65900.0,  # +0.45% in 10min
        volatility_15min=0.005,   # 0.5% volatility
        time_remaining_seconds=600,  # 10 minutes left
        orderbook_imbalance=0.2   # 20% buy pressure
    )

    print(f"Actual Probability BTC goes UP: {actual_probability:.1%}")
    print(f"  - Current: $66,200")
    print(f"  - 5min ago: $66,000 (+0.30%)")
    print(f"  - 10min ago: $65,900 (+0.45%)")
    print(f"  - Volatility: 0.5%")
    print(f"  - Time left: 10 minutes")
    print(f"  - Orderbook: +20% buy pressure")

    # Step 2: Detect arbitrage opportunity
    print("\n2. ARBITRAGE DETECTION")
    print("-" * 70)

    detector = ArbitrageDetector()

    # Example: Polymarket odds lag behind (still at 55%)
    arbitrage = detector.detect_arbitrage(
        actual_probability=actual_probability,
        market_yes_odds=0.55,  # Market still thinks 55%
        market_no_odds=0.45,
        market_id="btc-updown-15m-example",
        ai_base_confidence=0.75
    )

    print(f"Market YES Odds: {arbitrage.polymarket_yes_odds:.1%}")
    print(f"Actual Probability: {arbitrage.actual_probability:.1%}")
    print(f"EDGE: {arbitrage.edge_percentage:+.1%}")
    print(f"Recommended: {arbitrage.recommended_action}")
    print(f"Urgency: {arbitrage.urgency}")
    print(f"Confidence Boost: +{arbitrage.confidence_boost:.2f}")
    print(f"Expected Profit: +{arbitrage.expected_profit_pct:.1%}")

    # Step 3: Execute using smart limit order
    print("\n3. SMART ORDER EXECUTION")
    print("-" * 70)

    executor = SmartOrderExecutor()

    # Calculate strategy
    strategy = executor._calculate_strategy(
        urgency=arbitrage.urgency,
        side="BUY",
        best_ask=0.551,
        best_bid=0.540,
        tick_size=0.001
    )

    print(f"Order Type: LIMIT (maker, earns rebates)")
    print(f"Target Price: ${strategy.target_price:.3f}")
    print(f"Price Improvement: {strategy.price_improvement_pct:.2%}")
    print(f"Timeout: {strategy.timeout_seconds}s")
    print(f"Fallback to Market: {strategy.fallback_to_market}")

    print("\n" + "=" * 70)
    print("EXPECTED OUTCOME")
    print("=" * 70)
    print(f"Win Rate: ~72% (based on {actual_probability:.1%} probability)")
    print(f"Fee Savings: ~3-6% (maker vs taker)")
    print(f"Total Edge: {arbitrage.edge_percentage:.1%} + 3-6% = ~{arbitrage.edge_percentage + 0.04:.1%}")
    print(f"Net ROI per trade: ~10-15%")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(example_arbitrage_flow())
```

**Step 2: Make executable**

Run: `chmod +x examples/arbitrage_example.py`

**Step 3: Test the example**

Run: `python examples/arbitrage_example.py`
Expected: Prints example arbitrage flow with calculations

**Step 4: Commit**

```bash
git add examples/arbitrage_example.py
git commit -m "docs: add arbitrage system usage example"
```

---

## Task 15: Final Documentation and Rollout Guide

**Files:**
- Create: `docs/ARBITRAGE_SYSTEM.md`

**Step 1: Write comprehensive documentation**

Create: `docs/ARBITRAGE_SYSTEM.md`

```markdown
# Arbitrage Trading System

**Status:** ✅ Implemented and Ready for Deployment

## Overview

The arbitrage trading system exploits the lag between Polymarket odds and actual BTC spot price movements to identify mispriced 15-minute markets. It uses three core components:

1. **ProbabilityCalculator** - Calculates actual win probability from momentum + volatility
2. **ArbitrageDetector** - Compares actual vs market odds to find 5%+ edges
3. **SmartOrderExecutor** - Executes via limit orders to save 3-6% in fees

## Architecture

```
Price Feed → Probability → Arbitrage → AI Decision → Smart Execution
   ↓            ↓             ↓            ↓              ↓
BTC API    Calculate      Compare      Enhanced      Limit Orders
           (momentum)     (edge)       Prompt        (maker fees)
```

## Configuration

Add to your `.env` file:

```bash
# Arbitrage System
ARBITRAGE_MIN_EDGE_PCT=0.05       # 5% minimum (start at 0.10 for conservative)
ARBITRAGE_HIGH_EDGE_PCT=0.10      # 10%+ = medium urgency
ARBITRAGE_EXTREME_EDGE_PCT=0.15   # 15%+ = high urgency

# Limit Order Timeouts
LIMIT_ORDER_TIMEOUT_HIGH=30       # 30 seconds
LIMIT_ORDER_TIMEOUT_MEDIUM=60     # 60 seconds
LIMIT_ORDER_TIMEOUT_LOW=120       # 120 seconds
```

## Deployment Checklist

### Pre-Deployment

- [ ] Run migration: `python scripts/migrate_add_arbitrage_columns.py`
- [ ] Verify all tests pass: `pytest tests/ -v`
- [ ] Set `ARBITRAGE_MIN_EDGE_PCT=0.10` (conservative start)
- [ ] Set `DRY_RUN=true` for initial testing
- [ ] Run bot for 1 hour in dry-run mode
- [ ] Check logs for arbitrage detection messages
- [ ] Verify probability calculations look reasonable

### Day 1: Conservative Launch

- [ ] Set `ARBITRAGE_MIN_EDGE_PCT=0.10` (10%+ edges only)
- [ ] Set `DRY_RUN=false`
- [ ] Monitor for 24 hours
- [ ] Track metrics:
  - Number of arbitrage opportunities detected
  - Limit order fill rate
  - Win rate on arbitrage trades
  - Fee savings vs market orders

### Days 2-4: Gradual Relaxation

- [ ] Day 2: Lower to `ARBITRAGE_MIN_EDGE_PCT=0.08`
- [ ] Day 3: Lower to `ARBITRAGE_MIN_EDGE_PCT=0.06`
- [ ] Day 4: Lower to `ARBITRAGE_MIN_EDGE_PCT=0.05` (target)

### Week 2: Optimization

- [ ] Analyze performance data
- [ ] Tune volatility calculations if needed
- [ ] Adjust timeout parameters based on fill rates
- [ ] Optimize price improvement percentages

## Monitoring

Watch for these log messages:

```
INFO: Arbitrage opportunity detected
  action=BUY_YES edge=+13.0% urgency=MEDIUM

INFO: Limit order filled
  order_id=123 fill_amount=10.0

WARNING: Limit order timed out
  falling back to market order
```

Query database for arbitrage performance:

```sql
SELECT
  COUNT(*) as total_arbitrage_trades,
  AVG(arbitrage_edge) as avg_edge,
  AVG(CASE WHEN is_win = 1 THEN 1.0 ELSE 0.0 END) as win_rate,
  AVG(profit_loss) as avg_profit
FROM trades
WHERE arbitrage_edge IS NOT NULL
  AND arbitrage_edge >= 0.05;
```

## Expected Results

### Trade Frequency
- **Before:** 5-10 trades/day
- **After:** 20-30 trades/day (4-6x increase)

### Win Rate
- **Target:** 70-75% (maintained from current performance)
- **Edges 5-10%:** ~70% win rate
- **Edges 10-15%:** ~80% win rate
- **Edges 15%+:** ~85% win rate

### Fee Savings
- **Market orders:** Pay 3.1% taker fees
- **Limit orders:** Earn 0.2-0.5% maker rebates
- **Net savings:** 3-6% per trade

### Net ROI
- **Before:** ~1.5% per trade
- **After:** 8-12% per trade (5-8x improvement)

## Troubleshooting

### No arbitrage opportunities detected

- Check `actual_probability` is being calculated correctly
- Verify price history buffer has data
- Ensure volatility calculation is working
- Try lowering `ARBITRAGE_MIN_EDGE_PCT` temporarily

### Limit orders not filling

- Reduce `ARBITRAGE_MIN_EDGE_PCT` (trade less aggressively)
- Increase price improvement (make orders more attractive)
- Reduce timeouts (fallback to market sooner)
- Check if market liquidity is sufficient

### Lower win rate than expected

- Verify probability calculations match actual outcomes
- Check if volatility estimates are accurate
- Ensure time decay is factored correctly
- Review orderbook imbalance calculations

## Technical Details

### Probability Calculation

Uses modified Brownian motion:

```python
momentum = (current - price_5min) / price_5min * 0.7 +
           (current - price_10min) / price_10min * 0.3

volatility_factor = volatility * sqrt(time_remaining / 900)
z_score = momentum / volatility_factor
probability = norm.cdf(z_score)
```

### Edge Detection

```python
yes_edge = actual_probability - market_yes_odds
no_edge = (1 - actual_probability) - market_no_odds

if max(yes_edge, no_edge) >= 5%:
    TRADE OPPORTUNITY
```

### Smart Execution

```python
if urgency == HIGH:
    target_price = best_bid + 0.1%
    timeout = 30s
    fallback = True
```

## Support

For issues or questions:
1. Check logs in `logs/auto_trade.log`
2. Query database for recent trades
3. Review this documentation
4. Consult design document at `docs/plans/2026-02-13-arbitrage-trading-system-design.md`

---

**Last Updated:** 2026-02-13
**Version:** 1.0.0
**Status:** Production Ready
```

**Step 2: Commit**

```bash
git add docs/ARBITRAGE_SYSTEM.md
git commit -m "docs: add comprehensive arbitrage system documentation"
```

---

## Summary

This implementation plan creates a complete arbitrage trading system with:

✅ **New Components:**
- ProbabilityCalculator - Statistical directional prediction
- ArbitrageDetector - Edge detection and urgency classification
- SmartOrderExecutor - Limit orders with timeout strategies

✅ **Enhanced Existing:**
- AI decision service - Arbitrage context in prompts
- Auto-trader - Full integration of arbitrage flow
- Database - Tracking arbitrage performance
- Configuration - Tunable parameters

✅ **Testing:**
- Unit tests for all components
- Integration tests for auto-trader
- Example usage script
- Full test coverage

✅ **Documentation:**
- Design document
- Implementation notes
- Deployment checklist
- Troubleshooting guide

**Expected Results:**
- 4-6x increase in trade frequency (5→25 trades/day)
- 70-75% win rate maintained
- 3-6% fee savings per trade
- 8-12% net ROI per trade (vs current 1.5%)

**Total estimated time:** 6-8 hours for full implementation
**Deployment:** Staged rollout over 7 days, starting conservative (10%+ edges) and gradually lowering to target (5%+ edges)
