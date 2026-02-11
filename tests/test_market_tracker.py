"""Tests for market timing and price-to-beat tracker."""

import pytest
from decimal import Decimal
from datetime import datetime, timezone
from polymarket.trading.market_tracker import MarketTracker
from polymarket.config import Settings


def test_parse_market_slug():
    """Test parsing epoch timestamp from market slug."""
    settings = Settings()
    tracker = MarketTracker(settings)

    slug = "btc-updown-15m-1739203200"  # Example: 2026-02-11 00:00:00 UTC
    start_time = tracker.parse_market_start(slug)

    assert start_time == datetime.fromtimestamp(1739203200, tz=timezone.utc)


def test_calculate_time_remaining():
    """Test calculating time remaining in market."""
    settings = Settings()
    tracker = MarketTracker(settings)

    slug = "btc-updown-15m-1739203200"
    start_time = tracker.parse_market_start(slug)

    # Mock current time as 5 minutes after start
    current_time = datetime.fromtimestamp(1739203200 + 300, tz=timezone.utc)  # +5 min
    remaining = tracker.calculate_time_remaining(start_time, current_time)

    assert remaining == 600  # 10 minutes remaining (15 - 5)


def test_track_price_to_beat():
    """Test tracking starting price for market."""
    settings = Settings()
    tracker = MarketTracker(settings)

    slug = "btc-updown-15m-1739203200"
    starting_price = Decimal("95000.50")

    # Set starting price
    tracker.set_price_to_beat(slug, starting_price)

    # Retrieve it
    price = tracker.get_price_to_beat(slug)
    assert price == starting_price


@pytest.mark.asyncio
async def test_is_end_of_market():
    """Test detecting last 3 minutes of market."""
    settings = Settings()
    tracker = MarketTracker(settings)

    slug = "btc-updown-15m-1739203200"
    start_time = tracker.parse_market_start(slug)

    # 13 minutes elapsed (2 remaining) = END OF MARKET
    current_time = datetime.fromtimestamp(1739203200 + 780, tz=timezone.utc)  # +13 min
    is_end = tracker.is_end_of_market(start_time, current_time)
    assert is_end is True

    # 10 minutes elapsed (5 remaining) = NOT end
    current_time = datetime.fromtimestamp(1739203200 + 600, tz=timezone.utc)  # +10 min
    is_end = tracker.is_end_of_market(start_time, current_time)
    assert is_end is False


@pytest.mark.asyncio
async def test_ai_decision_with_gpt5_nano():
    """Test AI decision service uses GPT-5-Nano correctly."""
    import os
    from polymarket.trading.ai_decision import AIDecisionService
    from polymarket.models import (
        BTCPriceData, TechnicalIndicators,
        AggregatedSentiment, SocialSentiment, MarketSignals
    )
    from decimal import Decimal

    # Set environment variables for GPT-5-Nano config
    os.environ["OPENAI_MODEL"] = "gpt-5-nano"
    os.environ["OPENAI_REASONING_EFFORT"] = "medium"

    settings = Settings()
    ai_service = AIDecisionService(settings)

    # Create mock data
    btc_price = BTCPriceData(
        price=Decimal("95000"),
        timestamp=datetime.now(),
        source="polymarket",
        volume_24h=Decimal("1000000")
    )

    technical = TechnicalIndicators(
        rsi=50.0,
        macd_value=0.0,
        macd_signal=0.0,
        macd_histogram=0.0,
        ema_short=95000.0,
        ema_long=95000.0,
        sma_50=95000.0,
        volume_change=0.0,
        price_velocity=0.0,
        trend="NEUTRAL"
    )

    social = SocialSentiment(
        score=0.0,
        confidence=0.5,
        fear_greed=50,
        is_trending=False,
        vote_up_pct=50.0,
        vote_down_pct=50.0,
        signal_type="NEUTRAL",
        sources_available=["test"],
        timestamp=datetime.now()
    )

    market_signals = MarketSignals(
        score=0.0,
        confidence=0.5,
        order_book_bias="NEUTRAL",
        order_book_score=0.0,
        whale_direction="NEUTRAL",
        whale_count=0,
        whale_score=0.0,
        volume_ratio=1.0,
        volume_score=0.0,
        momentum_direction="NEUTRAL",
        momentum_score=0.0,
        signal_type="NEUTRAL",
        timestamp=datetime.now()
    )

    aggregated = AggregatedSentiment(
        final_score=0.0,
        final_confidence=0.5,
        signal_type="NEUTRAL",
        agreement_multiplier=1.0,
        social=social,
        market=market_signals,
        timestamp=datetime.now()
    )

    market_data = {
        "token_id": "test",
        "question": "BTC Up or Down - Test",
        "yes_price": 0.50,
        "no_price": 0.50,
        "active": True,
        "outcomes": ["Up", "Down"],
        "price_to_beat": Decimal("94000"),  # NEW
        "time_remaining_seconds": 180,  # NEW: 3 minutes
        "is_end_of_market": True  # NEW
    }

    # Make decision (will call OpenAI API)
    decision = await ai_service.make_decision(
        btc_price=btc_price,
        technical_indicators=technical,
        aggregated_sentiment=aggregated,
        market_data=market_data,
        portfolio_value=Decimal("1000")
    )

    # Verify decision structure
    assert decision.action in ("YES", "NO", "HOLD")
    assert 0.0 <= decision.confidence <= 1.0
    assert decision.reasoning is not None
