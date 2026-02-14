"""
Integration tests for paper trading components.

Tests the integration of signal lag detection, conflict analysis, and odds polling.
Focuses on component integration rather than full AutoTrader initialization.
"""

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime

from polymarket.trading.signal_lag_detector import detect_signal_lag
from polymarket.trading.conflict_detector import SignalConflictDetector, ConflictSeverity
from polymarket.trading.odds_poller import MarketOddsPoller
from polymarket.models import Market, OddsSnapshot


def test_signal_lag_and_conflict_integration():
    """
    Test 1: Signal lag detection integrates with conflict detection.

    Scenario: BTC moving UP, market sentiment DOWN with high confidence.
    Expected: Signal lag detected AND conflict analysis shows severity.
    """
    # Detect signal lag
    btc_direction = "UP"
    sentiment_direction = "BEARISH"
    sentiment_confidence = 0.75  # High confidence

    lag_detected, lag_reason = detect_signal_lag(
        btc_actual_direction=btc_direction,
        market_sentiment_direction=sentiment_direction,
        sentiment_confidence=sentiment_confidence
    )

    # Verify signal lag detected
    assert lag_detected is True
    assert "SIGNAL LAG DETECTED" in lag_reason
    assert "BTC moving UP" in lag_reason

    # Now run conflict detection with same signals
    detector = SignalConflictDetector()
    analysis = detector.analyze_conflicts(
        btc_direction="UP",
        technical_trend="DOWN",  # Conflict
        sentiment_direction="BEARISH",  # Conflict
        regime_trend="UP",  # Aligned
        timeframe_alignment="CONFLICTING",  # Conflict
        market_signals_direction="DOWN",  # Conflict
        market_signals_confidence=0.75
    )

    # Verify conflict detector also identifies issues
    assert analysis.severity == ConflictSeverity.SEVERE
    assert analysis.should_hold is True
    assert len(analysis.conflicts_detected) >= 3

    # Integration point: Both signal lag AND conflict detection would block trade
    assert lag_detected and analysis.should_hold


@pytest.mark.asyncio
async def test_odds_poller_integration():
    """
    Test 2: Odds poller integration with market filtering.

    Verifies that odds poller correctly identifies qualified markets
    and provides filtering capability.
    """
    # Create mock client
    mock_client = MagicMock()

    # Setup mock market with 82% YES odds
    mock_market = Market(
        id="market-123",
        slug="btc-updown-15m-1771234500",
        question="BTC Up or Down?",
        active=True,
        end_date=datetime.now(),
        best_bid=0.82,  # 82% YES
        best_ask=0.83,
        outcomes=["Up", "Down"],
        conditionId="test-condition"
    )

    mock_client.discover_btc_15min_market.return_value = mock_market
    mock_client.get_market_by_slug.return_value = mock_market

    # Create poller
    poller = MarketOddsPoller(mock_client)

    # Poll once
    await poller._poll_current_market()

    # Verify odds cached
    snapshot = await poller.get_odds("market-123")
    assert snapshot is not None
    assert snapshot.yes_odds == pytest.approx(0.82)
    assert snapshot.no_odds == pytest.approx(0.18)
    assert snapshot.yes_qualifies is True  # > 75%
    assert snapshot.no_qualifies is False

    # Integration point: Would pass early odds filter
    assert snapshot.yes_qualifies or snapshot.no_qualifies


@pytest.mark.asyncio
async def test_odds_poller_filtering_integration():
    """
    Test 3: Odds poller filtering - neither side qualifies.

    Verifies early market filtering when odds are 60/40 (neither > 75%).
    """
    # Create mock client
    mock_client = MagicMock()

    # Setup mock market with 60/40 odds
    mock_market = Market(
        id="market-999",
        slug="btc-updown-15m-1771234800",
        question="BTC Up or Down?",
        active=True,
        end_date=datetime.now(),
        best_bid=0.60,  # 60% YES, 40% NO
        best_ask=0.61,
        outcomes=["Up", "Down"],
        conditionId="test-condition"
    )

    mock_client.discover_btc_15min_market.return_value = mock_market
    mock_client.get_market_by_slug.return_value = mock_market

    # Create poller
    poller = MarketOddsPoller(mock_client)

    # Poll once
    await poller._poll_current_market()

    # Verify odds cached
    snapshot = await poller.get_odds("market-999")
    assert snapshot is not None
    assert snapshot.yes_odds == 0.60
    assert snapshot.no_odds == 0.40
    assert snapshot.yes_qualifies is False
    assert snapshot.no_qualifies is False

    # Integration point: Would skip this market (early filter)
    should_skip = not (snapshot.yes_qualifies or snapshot.no_qualifies)
    assert should_skip is True


def test_conflict_severity_classification_integration():
    """
    Test 4: Conflict severity classification integration.

    Tests all severity levels and confidence penalties.
    """
    detector = SignalConflictDetector()

    # Test 1: NONE - all aligned
    analysis_none = detector.analyze_conflicts(
        btc_direction="UP",
        technical_trend="UP",
        sentiment_direction="BULLISH",
        regime_trend="UP",
        timeframe_alignment="ALIGNED",
        market_signals_direction="UP",
        market_signals_confidence=0.80
    )
    assert analysis_none.severity == ConflictSeverity.NONE
    assert analysis_none.confidence_penalty == 0.0
    assert analysis_none.should_hold is False

    # Test 2: MINOR - 1 conflict
    analysis_minor = detector.analyze_conflicts(
        btc_direction="UP",
        technical_trend="DOWN",  # Conflict
        sentiment_direction="BULLISH",
        regime_trend="UP",
        timeframe_alignment="ALIGNED",
        market_signals_direction="UP",
        market_signals_confidence=0.80
    )
    assert analysis_minor.severity == ConflictSeverity.MINOR
    assert analysis_minor.confidence_penalty == -0.10
    assert analysis_minor.should_hold is False

    # Test 3: MODERATE - 2 conflicts
    analysis_moderate = detector.analyze_conflicts(
        btc_direction="UP",
        technical_trend="DOWN",  # Conflict 1
        sentiment_direction="BEARISH",  # Conflict 2
        regime_trend="UP",
        timeframe_alignment="ALIGNED",
        market_signals_direction="UP",
        market_signals_confidence=0.80
    )
    assert analysis_moderate.severity == ConflictSeverity.MODERATE
    assert analysis_moderate.confidence_penalty == -0.20
    assert analysis_moderate.should_hold is False

    # Test 4: SEVERE - 3+ conflicts
    analysis_severe = detector.analyze_conflicts(
        btc_direction="UP",
        technical_trend="DOWN",  # Conflict 1
        sentiment_direction="BEARISH",  # Conflict 2
        regime_trend="DOWN",  # Conflict 3
        timeframe_alignment="ALIGNED",
        market_signals_direction="UP",
        market_signals_confidence=0.80
    )
    assert analysis_severe.severity == ConflictSeverity.SEVERE
    assert analysis_severe.should_hold is True
    assert len(analysis_severe.conflicts_detected) >= 3


def test_signal_lag_threshold_integration():
    """
    Test 5: Signal lag detection threshold integration.

    Verifies that only high-confidence contradictions (>0.6) trigger lag detection.
    """
    # Test 1: High confidence (0.75) contradiction - LAG DETECTED
    lag_detected_high, reason_high = detect_signal_lag(
        btc_actual_direction="UP",
        market_sentiment_direction="BEARISH",
        sentiment_confidence=0.75
    )
    assert lag_detected_high is True
    assert "SIGNAL LAG DETECTED" in reason_high

    # Test 2: Medium confidence (0.55) contradiction - NO LAG
    lag_detected_med, reason_med = detect_signal_lag(
        btc_actual_direction="UP",
        market_sentiment_direction="BEARISH",
        sentiment_confidence=0.55
    )
    assert lag_detected_med is False
    assert "No lag detected" in reason_med

    # Test 3: Low confidence (0.40) contradiction - NO LAG
    lag_detected_low, reason_low = detect_signal_lag(
        btc_actual_direction="UP",
        market_sentiment_direction="BEARISH",
        sentiment_confidence=0.40
    )
    assert lag_detected_low is False

    # Integration point: Confidence threshold prevents false positives
    assert lag_detected_high and not lag_detected_med and not lag_detected_low
