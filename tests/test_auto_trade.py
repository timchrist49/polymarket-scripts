"""Integration test for auto_trade.py"""
import pytest
import asyncio
from decimal import Decimal
from polymarket.config import Settings
from polymarket.trading.btc_price import BTCPriceService
from polymarket.trading.sentiment import SentimentService
from polymarket.trading.technical import TechnicalAnalysis
from polymarket.trading.ai_decision import AIDecisionService
from polymarket.trading.risk import RiskManager


@pytest.mark.asyncio
async def test_full_trading_cycle():
    """Test complete trading cycle with real APIs."""
    settings = Settings()

    # Skip if missing API keys
    if not settings.openai_api_key or not settings.tavily_api_key:
        pytest.skip("Missing API keys")

    # Initialize services
    btc_service = BTCPriceService(settings)
    sentiment_service = SentimentService(settings)
    ai_service = AIDecisionService(settings)
    risk_manager = RiskManager(settings)

    try:
        # Step 1: Fetch data
        btc_price = await btc_service.get_current_price()
        assert btc_price.price > 0

        sentiment = await sentiment_service.get_btc_sentiment()
        assert -1.0 <= sentiment.score <= 1.0

        # Step 2: Technical analysis
        history = await btc_service.get_price_history(60)
        indicators = TechnicalAnalysis.calculate_indicators(history)
        assert 0 <= indicators.rsi <= 100

        # Step 3: AI decision
        market_data = {
            'token_id': 'test',
            'question': 'Will BTC go UP in 15 minutes?',
            'yes_price': 0.50,
            'no_price': 0.50,
            'active': True
        }

        decision = await ai_service.make_decision(
            btc_price, indicators, sentiment,
            market_data, Decimal('1000')
        )

        assert decision.action in ('YES', 'NO', 'HOLD')
        assert 0.0 <= decision.confidence <= 1.0

        # Step 4: Risk validation
        validation = await risk_manager.validate_decision(
            decision, Decimal('1000'), market_data, []
        )

        assert isinstance(validation.approved, bool)

        print(f"✓ Full cycle test passed")
        print(f"  BTC: ${btc_price.price:,.2f}")
        print(f"  Sentiment: {sentiment.score:+.2f}")
        print(f"  RSI: {indicators.rsi:.1f}")
        print(f"  Decision: {decision.action} ({decision.confidence:.2f})")

    finally:
        await btc_service.close()


if __name__ == "__main__":
    asyncio.run(test_full_trading_cycle())
