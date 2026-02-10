"""Integration test for auto_trade.py"""
import pytest
import asyncio
from decimal import Decimal
from polymarket.config import Settings
from polymarket.trading.btc_price import BTCPriceService
from polymarket.trading.social_sentiment import SocialSentimentService
from polymarket.trading.market_microstructure import MarketMicrostructureService
from polymarket.trading.signal_aggregator import SignalAggregator
from polymarket.trading.technical import TechnicalAnalysis
from polymarket.trading.ai_decision import AIDecisionService
from polymarket.trading.risk import RiskManager


@pytest.mark.asyncio
async def test_full_trading_cycle():
    """Test complete trading cycle with real APIs."""
    settings = Settings()

    # Skip if missing API keys
    if not settings.openai_api_key:
        pytest.skip("Missing OpenAI API key")

    # Initialize services
    btc_service = BTCPriceService(settings)
    social_service = SocialSentimentService(settings)
    market_service = MarketMicrostructureService(settings)
    aggregator = SignalAggregator()
    ai_service = AIDecisionService(settings)
    risk_manager = RiskManager(settings)

    try:
        # Step 1: Fetch data in parallel
        btc_price, social_sentiment, market_signals = await asyncio.gather(
            btc_service.get_current_price(),
            social_service.get_social_score(),
            market_service.get_market_score()
        )

        assert btc_price.price > 0
        assert -1.0 <= social_sentiment.score <= 1.0
        assert -1.0 <= market_signals.score <= 1.0

        # Step 2: Aggregate sentiment signals
        aggregated_sentiment = aggregator.aggregate(social_sentiment, market_signals)
        assert -1.0 <= aggregated_sentiment.final_score <= 1.0
        assert 0.0 <= aggregated_sentiment.final_confidence <= 1.0

        # Step 3: Technical analysis
        history = await btc_service.get_price_history(60)
        indicators = TechnicalAnalysis.calculate_indicators(history)
        assert 0 <= indicators.rsi <= 100

        # Step 4: AI decision
        market_data = {
            'token_id': 'test',
            'question': 'Will BTC go UP in 15 minutes?',
            'yes_price': 0.50,
            'no_price': 0.50,
            'active': True
        }

        decision = await ai_service.make_decision(
            btc_price, indicators, aggregated_sentiment,
            market_data, Decimal('1000')
        )

        assert decision.action in ('YES', 'NO', 'HOLD')
        assert 0.0 <= decision.confidence <= 1.0

        # Step 5: Risk validation
        validation = await risk_manager.validate_decision(
            decision, Decimal('1000'), market_data, []
        )

        assert isinstance(validation.approved, bool)

        print(f"✓ Full cycle test passed")
        print(f"  BTC: ${btc_price.price:,.2f}")
        print(f"  Social: {social_sentiment.score:+.2f} (conf: {social_sentiment.confidence:.2f})")
        print(f"  Market: {market_signals.score:+.2f} (conf: {market_signals.confidence:.2f})")
        print(f"  Aggregated: {aggregated_sentiment.final_score:+.2f} (conf: {aggregated_sentiment.final_confidence:.2f})")
        print(f"  RSI: {indicators.rsi:.1f}")
        print(f"  Decision: {decision.action} ({decision.confidence:.2f})")

    finally:
        await btc_service.close()
        await social_service.close()
        await market_service.close()


if __name__ == "__main__":
    asyncio.run(test_full_trading_cycle())
