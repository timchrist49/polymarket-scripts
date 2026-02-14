"""
Tests for signal lag detector.
"""

import pytest
from polymarket.trading.signal_lag_detector import detect_signal_lag


def test_lag_detection_btc_up_sentiment_bearish():
    """Test lag detector catches BTC UP + BEARISH sentiment."""
    is_lagging, reason = detect_signal_lag("UP", "BEARISH", 0.75)

    assert is_lagging is True
    assert "SIGNAL LAG DETECTED" in reason
    assert "BTC moving UP" in reason
    assert "market sentiment is BEARISH" in reason


def test_lag_detection_btc_down_sentiment_bullish():
    """Test lag detector catches BTC DOWN + BULLISH sentiment."""
    is_lagging, reason = detect_signal_lag("DOWN", "BULLISH", 0.80)

    assert is_lagging is True
    assert "SIGNAL LAG DETECTED" in reason
    assert "BTC moving DOWN" in reason
    assert "market sentiment is BULLISH" in reason


def test_no_lag_when_aligned_bullish():
    """Test no lag when BTC UP and sentiment BULLISH."""
    is_lagging, reason = detect_signal_lag("UP", "BULLISH", 0.75)

    assert is_lagging is False
    assert "No lag detected" in reason


def test_no_lag_when_aligned_bearish():
    """Test no lag when BTC DOWN and sentiment BEARISH."""
    is_lagging, reason = detect_signal_lag("DOWN", "BEARISH", 0.75)

    assert is_lagging is False
    assert "No lag detected" in reason


def test_no_lag_when_low_confidence_contradiction():
    """Test no lag flag when contradiction but low confidence."""
    # BTC UP but sentiment BEARISH, but confidence only 0.5 (< 0.6 threshold)
    is_lagging, reason = detect_signal_lag("UP", "BEARISH", 0.5)

    assert is_lagging is False
    assert "No lag detected" in reason


def test_neutral_sentiment_no_lag():
    """Test neutral sentiment doesn't trigger lag."""
    is_lagging, reason = detect_signal_lag("UP", "NEUTRAL", 0.75)

    # NEUTRAL maps to UP, so aligned
    assert is_lagging is False
