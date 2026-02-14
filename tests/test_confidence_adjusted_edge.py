"""Test confidence-adjusted edge thresholds."""
import pytest
from polymarket.trading.arbitrage_detector import ArbitrageDetector


def test_high_confidence_low_threshold():
    """High confidence (70%+) should accept 5% edge."""
    detector = ArbitrageDetector()

    # This will fail initially - method doesn't exist
    threshold = detector._get_minimum_edge(0.70)

    assert threshold == 0.05  # 5% for high confidence


def test_medium_confidence_medium_threshold():
    """Medium confidence (60-70%) should require 8% edge."""
    detector = ArbitrageDetector()

    threshold = detector._get_minimum_edge(0.65)

    assert threshold == 0.08  # 8% for medium confidence


def test_low_confidence_high_threshold():
    """Low confidence (50-60%) should require 12% edge."""
    detector = ArbitrageDetector()

    threshold = detector._get_minimum_edge(0.55)

    assert threshold == 0.12  # 12% for low confidence


def test_symmetric_for_probability_below_50():
    """Edge threshold should be symmetric around 50%."""
    detector = ArbitrageDetector()

    # 70% and 30% should have same threshold (both high confidence)
    threshold_70 = detector._get_minimum_edge(0.70)
    threshold_30 = detector._get_minimum_edge(0.30)

    assert threshold_70 == threshold_30 == 0.05
