import pytest
from datetime import datetime
from polymarket.trading.signal_aggregator import SignalAggregator
from polymarket.models import SocialSentiment, MarketSignals, AggregatedSentiment


def test_agreement_score_perfect_alignment():
    """Test agreement when both signals strongly agree."""
    aggregator = SignalAggregator()

    # Both strongly bullish
    agreement = aggregator._calculate_agreement_score(0.8, 0.9)

    assert agreement > 1.3  # Should get boost


def test_agreement_score_conflict():
    """Test agreement when signals conflict."""
    aggregator = SignalAggregator()

    # One bullish, one bearish
    agreement = aggregator._calculate_agreement_score(0.8, -0.6)

    assert agreement < 0.8  # Should get penalty


def test_aggregate_strong_bullish():
    """Test aggregation with strong bullish signals."""
    aggregator = SignalAggregator()

    social = SocialSentiment(
        score=0.8, confidence=0.7,
        fear_greed=80, is_trending=True,
        vote_up_pct=70, vote_down_pct=30,
        signal_type="STRONG_BULLISH",
        sources_available=["fear_greed", "trending", "votes"],
        timestamp=datetime.now()
    )

    market = MarketSignals(
        score=0.9, confidence=0.8,
        order_book_score=0.8, whale_score=0.9, volume_score=0.7, momentum_score=0.95,
        order_book_bias="BID_HEAVY", whale_direction="BUYING", whale_count=10,
        volume_ratio=1.5, momentum_direction="UP",
        signal_type="STRONG_BULLISH",
        timestamp=datetime.now()
    )

    aggregated = aggregator.aggregate(social, market)

    assert isinstance(aggregated, AggregatedSentiment)
    assert aggregated.final_score > 0.8  # Strong bullish
    assert aggregated.final_confidence > 0.9  # High confidence (boosted by agreement)
    assert aggregated.agreement_multiplier > 1.2  # Agreement boost
    assert "STRONG" in aggregated.signal_type


def test_aggregate_conflicting_signals():
    """Test aggregation when signals conflict."""
    aggregator = SignalAggregator()

    social = SocialSentiment(
        score=-0.6, confidence=0.7,  # Bearish
        fear_greed=20, is_trending=False,
        vote_up_pct=30, vote_down_pct=70,
        signal_type="STRONG_BEARISH",
        sources_available=["fear_greed", "votes"],
        timestamp=datetime.now()
    )

    market = MarketSignals(
        score=0.8, confidence=0.9,  # Bullish (whales buying)
        order_book_score=0.7, whale_score=0.9, volume_score=0.6, momentum_score=0.8,
        order_book_bias="BID_HEAVY", whale_direction="BUYING", whale_count=8,
        volume_ratio=1.4, momentum_direction="UP",
        signal_type="STRONG_BULLISH",
        timestamp=datetime.now()
    )

    aggregated = aggregator.aggregate(social, market)

    assert aggregated.final_confidence < 0.6  # Low confidence due to conflict
    assert aggregated.agreement_multiplier < 0.8  # Conflict penalty
    assert "CONFLICTED" in aggregated.signal_type or "WEAK" in aggregated.signal_type


def test_aggregate_missing_social():
    """Test when social sentiment unavailable."""
    aggregator = SignalAggregator()

    social = SocialSentiment(
        score=0.0, confidence=0.0,  # Unavailable
        fear_greed=50, is_trending=False,
        vote_up_pct=50, vote_down_pct=50,
        signal_type="UNAVAILABLE",
        sources_available=[],
        timestamp=datetime.now()
    )

    market = MarketSignals(
        score=0.7, confidence=0.8,
        order_book_score=0.6, whale_score=0.7, volume_score=0.5, momentum_score=0.8,
        order_book_bias="BID_HEAVY", whale_direction="BUYING", whale_count=5,
        volume_ratio=1.3, momentum_direction="UP",
        signal_type="STRONG_BULLISH",
        timestamp=datetime.now()
    )

    aggregated = aggregator.aggregate(social, market)

    # Should use market only with penalty
    assert aggregated.final_score == pytest.approx(market.score, abs=0.01)
    assert aggregated.final_confidence < market.confidence  # Penalty for missing social
    assert "MARKET_ONLY" in aggregated.signal_type
