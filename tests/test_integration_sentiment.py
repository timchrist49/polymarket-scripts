"""
Integration tests for complete sentiment analysis pipeline.
"""

import pytest
import asyncio
from decimal import Decimal
from datetime import datetime

from polymarket.config import Settings
from polymarket.client import PolymarketClient
from polymarket.trading.btc_price import BTCPriceService
from polymarket.trading.social_sentiment import SocialSentimentService
from polymarket.trading.market_microstructure import MarketMicrostructureService
from polymarket.trading.signal_aggregator import SignalAggregator
from polymarket.trading.technical import TechnicalAnalysis
from polymarket.trading.ai_decision import AIDecisionService


@pytest.mark.asyncio
async def test_full_sentiment_pipeline():
    """Test complete end-to-end sentiment analysis pipeline."""
    # Initialize services
    settings = Settings()
    client = PolymarketClient()
    btc_service = BTCPriceService(settings)
    social_service = SocialSentimentService(settings)
    market_service = MarketMicrostructureService(settings)
    aggregator = SignalAggregator()
    ai_service = AIDecisionService(settings)

    try:
        # Step 1: Fetch market
        market = client.discover_btc_15min_market()
        assert market is not None
        assert market.active

        # Step 2: Fetch all data in parallel
        btc_data, social, market_signals = await asyncio.gather(
            btc_service.get_current_price(),
            social_service.get_social_score(),
            market_service.get_market_score()
        )

        # Verify data received
        assert btc_data.price > 0
        assert -0.7 <= social.score <= 0.85
        assert 0.0 <= social.confidence <= 1.0
        assert -1.0 <= market_signals.score <= 1.0
        assert 0.0 <= market_signals.confidence <= 1.0

        # Step 3: Get technical indicators
        price_history = await btc_service.get_price_history(minutes=60)
        indicators = TechnicalAnalysis.calculate_indicators(price_history)

        # Step 4: Aggregate signals
        aggregated = aggregator.aggregate(social, market_signals)

        # Verify aggregation
        assert isinstance(aggregated.final_score, float)
        assert isinstance(aggregated.final_confidence, float)
        assert 0.0 <= aggregated.final_confidence <= 1.0
        assert 0.5 <= aggregated.agreement_multiplier <= 1.5

        # Step 5: Make AI decision
        token_ids = market.get_token_ids()
        market_dict = {
            "token_id": token_ids[0],
            "question": market.question,
            "yes_price": market.best_bid or 0.50,
            "no_price": market.best_ask or 0.50,
            "active": market.active
        }

        decision = await ai_service.make_decision(
            btc_price=btc_data,
            technical_indicators=indicators,
            aggregated_sentiment=aggregated,
            market_data=market_dict,
            portfolio_value=Decimal("1000")
        )

        # Verify decision
        assert decision.action in ("YES", "NO", "HOLD")
        assert 0.0 <= decision.confidence <= 1.0

        # AI confidence should be close to aggregated confidence (within ±0.15)
        assert abs(decision.confidence - aggregated.final_confidence) <= 0.15

        # If confidence below threshold, should be HOLD
        if decision.confidence < settings.bot_confidence_threshold:
            assert decision.action == "HOLD"

        print("\n=== Integration Test Results ===")
        print(f"BTC Price: ${btc_data.price:,.2f}")
        print(f"Social Score: {social.score:+.2f} (conf: {social.confidence:.2f})")
        print(f"Market Score: {market_signals.score:+.2f} (conf: {market_signals.confidence:.2f})")
        print(f"Final Score: {aggregated.final_score:+.2f} (conf: {aggregated.final_confidence:.2f})")
        print(f"Agreement: {aggregated.agreement_multiplier:.2f}x")
        print(f"AI Decision: {decision.action} (conf: {decision.confidence:.2f})")
        print(f"Signal Type: {aggregated.signal_type}")
        print("=== Test Passed ===\n")

    finally:
        # Cleanup
        await btc_service.close()
        await social_service.close()
        await market_service.close()


@pytest.mark.asyncio
async def test_graceful_degradation():
    """Test that system handles API failures gracefully."""
    settings = Settings()
    social_service = SocialSentimentService(settings)
    market_service = MarketMicrostructureService(settings)
    aggregator = SignalAggregator()

    # Even if APIs fail, should return valid (neutral) data
    social = await social_service.get_social_score()
    market_signals = await market_service.get_market_score()
    aggregated = aggregator.aggregate(social, market_signals)

    # Should have valid structure even on failure
    assert isinstance(aggregated.final_score, float)
    assert isinstance(aggregated.final_confidence, float)
    assert -1.0 <= aggregated.final_score <= 1.0
    assert 0.0 <= aggregated.final_confidence <= 1.0

    await social_service.close()
    await market_service.close()
