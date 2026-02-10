import pytest
import asyncio
from datetime import datetime
from polymarket.trading.social_sentiment import SocialSentimentService
from polymarket.config import Settings
from polymarket.models import SocialSentiment


@pytest.mark.asyncio
async def test_fetch_fear_greed():
    """Test Fear & Greed API integration."""
    service = SocialSentimentService(Settings())

    result = await service._fetch_fear_greed()

    assert isinstance(result, int)
    assert 0 <= result <= 100


@pytest.mark.asyncio
async def test_fetch_trending():
    """Test CoinGecko trending API."""
    service = SocialSentimentService(Settings())

    is_trending = await service._fetch_trending()

    assert isinstance(is_trending, bool)


@pytest.mark.asyncio
async def test_fetch_sentiment_votes():
    """Test CoinGecko sentiment votes API."""
    service = SocialSentimentService(Settings())

    up_pct, down_pct = await service._fetch_sentiment_votes()

    assert isinstance(up_pct, float)
    assert isinstance(down_pct, float)
    assert up_pct + down_pct == pytest.approx(100.0, abs=0.1)


@pytest.mark.asyncio
async def test_get_social_score_all_sources():
    """Test full social scoring with all sources available."""
    service = SocialSentimentService(Settings())

    sentiment = await service.get_social_score()

    assert isinstance(sentiment, SocialSentiment)
    assert -1.0 <= sentiment.score <= 1.0
    assert 0.0 <= sentiment.confidence <= 1.0
    assert 0 <= sentiment.fear_greed <= 100
    assert isinstance(sentiment.is_trending, bool)
    assert len(sentiment.sources_available) > 0


@pytest.mark.asyncio
async def test_score_calculation_bullish():
    """Test scoring logic for bullish scenario."""
    service = SocialSentimentService(Settings())

    # Mock bullish data
    score = service._calculate_score(
        fear_greed=80,  # Greed
        is_trending=True,
        vote_up_pct=70,
        vote_down_pct=30
    )

    assert score > 0.5  # Should be bullish


@pytest.mark.asyncio
async def test_score_calculation_bearish():
    """Test scoring logic for bearish scenario."""
    service = SocialSentimentService(Settings())

    # Mock bearish data
    score = service._calculate_score(
        fear_greed=20,  # Fear
        is_trending=False,
        vote_up_pct=30,
        vote_down_pct=70
    )

    assert score < -0.3  # Should be bearish
