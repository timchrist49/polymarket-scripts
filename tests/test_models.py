# tests/test_models.py
import pytest
from datetime import datetime
from polymarket.models import Market, OrderRequest, OrderResponse, TokenInfo, SocialSentiment, MarketSignals, AggregatedSentiment

def test_market_model_minimal():
    """Test Market model with minimal required fields."""
    market = Market(
        id="0x123",
        condition_id="0x456",
    )
    assert market.id == "0x123"
    assert market.condition_id == "0x456"
    assert market.question is None

def test_market_model_with_outcomes():
    """Test Market model with outcomes parsing."""
    market = Market(
        id="0x123",
        condition_id="0x456",
        question="BTC will go up?",
        outcomes=["Yes", "No"],
        clob_token_ids='["0xaaa", "0xbbb"]',
    )
    assert market.question == "BTC will go up?"
    assert market.outcomes == ["Yes", "No"]

def test_order_request_validation():
    """Test OrderRequest validation."""
    # Valid buy order
    order = OrderRequest(
        token_id="0x123",
        side="BUY",
        price=0.55,
        size=10.0,
    )
    assert order.side == "BUY"

def test_order_request_invalid_side():
    """Test OrderRequest rejects invalid side."""
    with pytest.raises(ValueError):
        OrderRequest(
            token_id="0x123",
            side="INVALID",
            price=0.55,
            size=10.0,
        )

def test_order_request_invalid_price():
    """Test OrderRequest rejects invalid price."""
    with pytest.raises(ValueError):
        OrderRequest(
            token_id="0x123",
            side="BUY",
            price=1.5,  # Must be 0-1
            size=10.0,
        )


def test_social_sentiment_model():
    """Test SocialSentiment dataclass."""
    sentiment = SocialSentiment(
        score=0.75,
        confidence=0.8,
        fear_greed=80,
        is_trending=True,
        vote_up_pct=65,
        vote_down_pct=35,
        signal_type="STRONG_BULLISH",
        sources_available=["fear_greed", "trending", "votes"],
        timestamp=datetime.now()
    )

    assert sentiment.score == 0.75
    assert sentiment.confidence == 0.8
    assert sentiment.signal_type == "STRONG_BULLISH"


def test_market_signals_model():
    """Test MarketSignals dataclass."""
    signals = MarketSignals(
        score=0.6, confidence=0.9,
        order_book_score=0.5, whale_score=0.7, volume_score=0.6, momentum_score=0.8,
        order_book_bias="BID_HEAVY", whale_direction="BUYING", whale_count=8,
        volume_ratio=1.5, momentum_direction="UP",
        signal_type="STRONG_BULLISH",
        timestamp=datetime.now()
    )

    assert signals.score == 0.6
    assert signals.whale_count == 8
    assert signals.signal_type == "STRONG_BULLISH"


def test_aggregated_sentiment_model():
    """Test AggregatedSentiment dataclass."""
    social = SocialSentiment(
        score=0.7, confidence=0.8, fear_greed=75, is_trending=True,
        vote_up_pct=65, vote_down_pct=35, signal_type="STRONG_BULLISH",
        sources_available=["fear_greed"], timestamp=datetime.now()
    )
    market = MarketSignals(
        score=0.6, confidence=0.9,
        order_book_score=0.5, whale_score=0.6, volume_score=0.6, momentum_score=0.7,
        order_book_bias="BALANCED", whale_direction="NEUTRAL", whale_count=5,
        volume_ratio=1.2, momentum_direction="UP",
        signal_type="WEAK_BULLISH", timestamp=datetime.now()
    )

    aggregated = AggregatedSentiment(
        social=social, market=market,
        final_score=0.64, final_confidence=0.95,
        agreement_multiplier=1.3, signal_type="STRONG_BULLISH",
        timestamp=datetime.now()
    )

    assert aggregated.final_score == 0.64
    assert aggregated.agreement_multiplier == 1.3


def test_social_sentiment_validation():
    """Test SocialSentiment validation."""
    # Valid data should not raise
    valid = SocialSentiment(
        score=0.5, confidence=0.8, fear_greed=60,
        is_trending=True, vote_up_pct=55.0, vote_down_pct=45.0,
        signal_type="NEUTRAL", sources_available=["fear_greed"],
        timestamp=datetime.now()
    )
    valid.validate()  # Should not raise

    # Invalid score (out of range)
    invalid_score = SocialSentiment(
        score=1.5, confidence=0.8, fear_greed=60,
        is_trending=True, vote_up_pct=55.0, vote_down_pct=45.0,
        signal_type="NEUTRAL", sources_available=[],
        timestamp=datetime.now()
    )
    with pytest.raises(ValueError, match="score must be"):
        invalid_score.validate()

    # Invalid confidence (out of range)
    invalid_confidence = SocialSentiment(
        score=0.5, confidence=1.5, fear_greed=60,
        is_trending=True, vote_up_pct=55.0, vote_down_pct=45.0,
        signal_type="NEUTRAL", sources_available=[],
        timestamp=datetime.now()
    )
    with pytest.raises(ValueError, match="confidence must be"):
        invalid_confidence.validate()

    # Invalid fear_greed (out of range)
    invalid_fear_greed = SocialSentiment(
        score=0.5, confidence=0.8, fear_greed=150,
        is_trending=True, vote_up_pct=55.0, vote_down_pct=45.0,
        signal_type="NEUTRAL", sources_available=[],
        timestamp=datetime.now()
    )
    with pytest.raises(ValueError, match="fear_greed must be"):
        invalid_fear_greed.validate()

    # Invalid vote_up_pct (out of range)
    invalid_vote_up = SocialSentiment(
        score=0.5, confidence=0.8, fear_greed=60,
        is_trending=True, vote_up_pct=150.0, vote_down_pct=45.0,
        signal_type="NEUTRAL", sources_available=[],
        timestamp=datetime.now()
    )
    with pytest.raises(ValueError, match="vote_up_pct must be"):
        invalid_vote_up.validate()

    # Invalid vote_down_pct (out of range)
    invalid_vote_down = SocialSentiment(
        score=0.5, confidence=0.8, fear_greed=60,
        is_trending=True, vote_up_pct=55.0, vote_down_pct=150.0,
        signal_type="NEUTRAL", sources_available=[],
        timestamp=datetime.now()
    )
    with pytest.raises(ValueError, match="vote_down_pct must be"):
        invalid_vote_down.validate()


def test_market_signals_validation():
    """Test MarketSignals validation."""
    # Valid data should not raise
    valid = MarketSignals(
        score=0.6, confidence=0.9,
        order_book_score=0.5, whale_score=0.7, volume_score=0.6, momentum_score=0.8,
        order_book_bias="BID_HEAVY", whale_direction="BUYING", whale_count=8,
        volume_ratio=1.5, momentum_direction="UP",
        signal_type="STRONG_BULLISH",
        timestamp=datetime.now()
    )
    valid.validate()  # Should not raise

    # Invalid score (out of range)
    invalid_score = MarketSignals(
        score=2.0, confidence=0.9,
        order_book_score=0.5, whale_score=0.7, volume_score=0.6, momentum_score=0.8,
        order_book_bias="BID_HEAVY", whale_direction="BUYING", whale_count=8,
        volume_ratio=1.5, momentum_direction="UP",
        signal_type="STRONG_BULLISH",
        timestamp=datetime.now()
    )
    with pytest.raises(ValueError, match="score must be"):
        invalid_score.validate()

    # Invalid confidence (out of range)
    invalid_confidence = MarketSignals(
        score=0.6, confidence=1.5,
        order_book_score=0.5, whale_score=0.7, volume_score=0.6, momentum_score=0.8,
        order_book_bias="BID_HEAVY", whale_direction="BUYING", whale_count=8,
        volume_ratio=1.5, momentum_direction="UP",
        signal_type="STRONG_BULLISH",
        timestamp=datetime.now()
    )
    with pytest.raises(ValueError, match="confidence must be"):
        invalid_confidence.validate()

    # Invalid order_book_score (out of range)
    invalid_order_book = MarketSignals(
        score=0.6, confidence=0.9,
        order_book_score=1.5, whale_score=0.7, volume_score=0.6, momentum_score=0.8,
        order_book_bias="BID_HEAVY", whale_direction="BUYING", whale_count=8,
        volume_ratio=1.5, momentum_direction="UP",
        signal_type="STRONG_BULLISH",
        timestamp=datetime.now()
    )
    with pytest.raises(ValueError, match="order_book_score must be"):
        invalid_order_book.validate()

    # Invalid whale_count (negative)
    invalid_whale_count = MarketSignals(
        score=0.6, confidence=0.9,
        order_book_score=0.5, whale_score=0.7, volume_score=0.6, momentum_score=0.8,
        order_book_bias="BID_HEAVY", whale_direction="BUYING", whale_count=-5,
        volume_ratio=1.5, momentum_direction="UP",
        signal_type="STRONG_BULLISH",
        timestamp=datetime.now()
    )
    with pytest.raises(ValueError, match="whale_count must be"):
        invalid_whale_count.validate()

    # Invalid volume_ratio (negative)
    invalid_volume_ratio = MarketSignals(
        score=0.6, confidence=0.9,
        order_book_score=0.5, whale_score=0.7, volume_score=0.6, momentum_score=0.8,
        order_book_bias="BID_HEAVY", whale_direction="BUYING", whale_count=8,
        volume_ratio=-1.0, momentum_direction="UP",
        signal_type="STRONG_BULLISH",
        timestamp=datetime.now()
    )
    with pytest.raises(ValueError, match="volume_ratio must be"):
        invalid_volume_ratio.validate()


def test_aggregated_sentiment_validation():
    """Test AggregatedSentiment validation."""
    social = SocialSentiment(
        score=0.7, confidence=0.8, fear_greed=75, is_trending=True,
        vote_up_pct=65.0, vote_down_pct=35.0, signal_type="STRONG_BULLISH",
        sources_available=["fear_greed"], timestamp=datetime.now()
    )
    market = MarketSignals(
        score=0.6, confidence=0.9,
        order_book_score=0.5, whale_score=0.6, volume_score=0.6, momentum_score=0.7,
        order_book_bias="BALANCED", whale_direction="NEUTRAL", whale_count=5,
        volume_ratio=1.2, momentum_direction="UP",
        signal_type="WEAK_BULLISH", timestamp=datetime.now()
    )

    # Valid data should not raise
    valid = AggregatedSentiment(
        social=social, market=market,
        final_score=0.64, final_confidence=0.95,
        agreement_multiplier=1.3, signal_type="STRONG_BULLISH",
        timestamp=datetime.now()
    )
    valid.validate()  # Should not raise

    # Invalid final_score (out of range)
    invalid_final_score = AggregatedSentiment(
        social=social, market=market,
        final_score=1.5, final_confidence=0.95,
        agreement_multiplier=1.3, signal_type="STRONG_BULLISH",
        timestamp=datetime.now()
    )
    with pytest.raises(ValueError, match="final_score must be"):
        invalid_final_score.validate()

    # Invalid final_confidence (out of range)
    invalid_final_confidence = AggregatedSentiment(
        social=social, market=market,
        final_score=0.64, final_confidence=1.5,
        agreement_multiplier=1.3, signal_type="STRONG_BULLISH",
        timestamp=datetime.now()
    )
    with pytest.raises(ValueError, match="final_confidence must be"):
        invalid_final_confidence.validate()

    # Invalid agreement_multiplier (out of range)
    invalid_agreement = AggregatedSentiment(
        social=social, market=market,
        final_score=0.64, final_confidence=0.95,
        agreement_multiplier=2.0, signal_type="STRONG_BULLISH",
        timestamp=datetime.now()
    )
    with pytest.raises(ValueError, match="agreement_multiplier must be"):
        invalid_agreement.validate()

    # Invalid nested social (should propagate)
    invalid_social = SocialSentiment(
        score=1.5, confidence=0.8, fear_greed=75, is_trending=True,
        vote_up_pct=65.0, vote_down_pct=35.0, signal_type="STRONG_BULLISH",
        sources_available=["fear_greed"], timestamp=datetime.now()
    )
    invalid_nested = AggregatedSentiment(
        social=invalid_social, market=market,
        final_score=0.64, final_confidence=0.95,
        agreement_multiplier=1.3, signal_type="STRONG_BULLISH",
        timestamp=datetime.now()
    )
    with pytest.raises(ValueError, match="score must be"):
        invalid_nested.validate()
