"""Tests for AI Decision Service."""
import pytest
from decimal import Decimal
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock

from polymarket.trading.ai_decision import AIDecisionService
from polymarket.models import (
    BTCPriceData,
    TechnicalIndicators,
    AggregatedSentiment,
    SocialSentiment,
    MarketSignals,
    ArbitrageOpportunity
)


@pytest.fixture
def mock_settings():
    """Mock settings for testing."""
    settings = Mock()
    settings.openai_api_key = "test-key"
    settings.openai_model = "gpt-5-nano"
    settings.openai_reasoning_effort = "medium"
    settings.bot_confidence_threshold = 0.75
    settings.bot_max_position_percent = 0.05
    return settings


@pytest.fixture
def mock_btc_price():
    """Mock BTC price data."""
    return BTCPriceData(
        price=Decimal("50000.00"),
        timestamp=datetime.now(),
        source="test",
        volume_24h=Decimal("1000000000")
    )


@pytest.fixture
def mock_indicators():
    """Mock technical indicators."""
    return TechnicalIndicators(
        rsi=55.0,
        macd_value=50.0,
        macd_signal=45.0,
        macd_histogram=5.0,
        ema_short=50000.0,
        ema_long=49900.0,
        sma_50=50000.0,
        trend="BULLISH",
        volume_change=10.0,
        price_velocity=5.0
    )


@pytest.fixture
def mock_sentiment():
    """Mock aggregated sentiment."""
    social = SocialSentiment(
        score=0.5,
        confidence=0.8,
        fear_greed=60,
        is_trending=False,
        vote_up_pct=60.0,
        vote_down_pct=40.0,
        signal_type="BULLISH",
        sources_available=["test"],
        timestamp=datetime.now()
    )
    market = MarketSignals(
        score=0.4,
        confidence=0.7,
        order_book_bias="BULLISH",
        order_book_score=0.3,
        whale_direction="UP",
        whale_count=5,
        whale_score=0.4,
        volume_ratio=1.2,
        volume_score=0.2,
        momentum_direction="UP",
        momentum_score=0.5,
        signal_type="BULLISH",
        timestamp=datetime.now()
    )
    return AggregatedSentiment(
        social=social,
        market=market,
        final_score=0.45,
        final_confidence=0.75,
        signal_type="BULLISH",
        agreement_multiplier=1.1,
        timestamp=datetime.now()
    )


@pytest.mark.asyncio
async def test_make_decision_with_arbitrage(mock_settings, mock_btc_price, mock_indicators, mock_sentiment):
    """Test AI decision with arbitrage opportunity."""
    service = AIDecisionService(mock_settings)

    # Mock OpenAI response
    with patch.object(service, '_get_client') as mock_client:
        mock_openai = Mock()
        mock_openai.chat.completions.create = AsyncMock(return_value=Mock(
            choices=[Mock(message=Mock(content='{"action": "YES", "confidence": 0.92, "reasoning": "Arbitrage edge", "position_size": 50}'))]
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

        # Verify decision was made
        assert decision.action == "YES"
        assert decision.confidence == 0.92

        # Verify arbitrage context was included in prompt
        call_args = mock_openai.chat.completions.create.call_args
        prompt = call_args.kwargs["messages"][1]["content"]
        assert "ARBITRAGE OPPORTUNITY" in prompt
        assert "15.0%" in prompt  # Edge percentage
        assert "BUY_YES" in prompt  # Recommendation
