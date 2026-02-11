#!/usr/bin/env python3
"""
Test the AI decision to token mapping logic.
Verify that bearish signals result in buying the DOWN token.
"""

import asyncio
from decimal import Decimal
from polymarket.config import Settings
from polymarket.trading.ai_decision import AIDecisionService
from polymarket.models import BTCPriceData, TechnicalIndicators, AggregatedSentiment, SentimentAnalysis

async def test_decision_mapping():
    settings = Settings()
    ai_service = AIDecisionService(settings)

    # Create mock data with STRONG BEARISH signals
    btc_price = BTCPriceData(
        price=Decimal("69000"),
        source="test",
        timestamp=0,
        volume_24h=Decimal("1000000")
    )

    technical = TechnicalIndicators(
        rsi=35.0,  # Oversold (bearish)
        macd_value=-50.0,  # Negative (bearish)
        macd_signal=-40.0,
        macd_histogram=-10.0,
        ema_short=68000.0,
        ema_long=70000.0,  # Short below long (bearish)
        sma_50=70000.0,
        volume_change=-15.0,  # Volume declining (bearish)
        price_velocity=-100.0,  # Negative velocity (bearish)
        trend="BEARISH"
    )

    # Create bearish sentiment
    social = SentimentAnalysis(
        score=-0.50,  # Strong bearish
        confidence=1.0,
        fear_greed=20,  # Extreme fear
        is_trending=False,
        vote_up_pct=20.0,
        vote_down_pct=80.0,
        signal_type="STRONG_BEARISH",
        sources_available=["fear_greed", "trending", "votes"]
    )

    market_sentiment = SentimentAnalysis(
        score=-0.60,  # Strong bearish
        confidence=1.0,
        fear_greed=0,
        is_trending=False,
        vote_up_pct=0.0,
        vote_down_pct=0.0,
        signal_type="STRONG_BEARISH",
        sources_available=["microstructure"],
        # Market microstructure fields
        momentum_direction="DOWN",
        momentum_score=-0.70,
        whale_direction="SELLING",
        whale_count=5,
        whale_score=-0.50,
        order_book_bias="SELL_HEAVY",
        order_book_score=-0.60,
        volume_ratio=1.5,
        volume_score=-0.40
    )

    aggregated = AggregatedSentiment(
        final_score=-0.56,  # Strong bearish
        final_confidence=1.0,
        signal_type="STRONG_BEARISH",
        agreement_multiplier=1.5,  # High agreement
        social=social,
        market=market_sentiment
    )

    # Market data for "Bitcoin Up or Down" market
    market_data = {
        "token_id": "test_token_up",
        "question": "Bitcoin Up or Down - Test Market",
        "yes_price": 0.30,  # Low odds for UP
        "no_price": 0.70,  # High odds for DOWN
        "active": True,
        "outcomes": ["Up", "Down"]
    }

    print("=" * 60)
    print("TEST: AI Decision Mapping with STRONG BEARISH Signals")
    print("=" * 60)
    print(f"\nSignals:")
    print(f"  Social Score: {social.score:+.2f} ({social.signal_type})")
    print(f"  Market Score: {market_sentiment.score:+.2f} ({market_sentiment.signal_type})")
    print(f"  Aggregated: {aggregated.final_score:+.2f} ({aggregated.signal_type})")
    print(f"  Agreement: {aggregated.agreement_multiplier:.2f}x")
    print(f"\nMarket:")
    print(f"  Question: {market_data['question']}")
    print(f"  Outcomes: {market_data['outcomes']}")
    print(f"  YES (Up) odds: {market_data['yes_price']:.2f}")
    print(f"  NO (Down) odds: {market_data['no_price']:.2f}")

    # Make decision
    decision = await ai_service.make_decision(
        btc_price=btc_price,
        technical_indicators=technical,
        aggregated_sentiment=aggregated,
        market_data=market_data,
        portfolio_value=Decimal("1000")
    )

    print(f"\n" + "=" * 60)
    print(f"AI DECISION:")
    print(f"=" * 60)
    print(f"  Action: {decision.action}")
    print(f"  Confidence: {decision.confidence:.2f}")
    print(f"  Reasoning: {decision.reasoning}")
    print(f"  Position Size: ${decision.position_size:.2f}")

    # Expected outcome
    print(f"\n" + "=" * 60)
    print(f"EXPECTED vs ACTUAL:")
    print(f"=" * 60)

    if decision.action == "NO":
        print(f"  ✅ CORRECT! AI chose 'NO' to buy the DOWN token")
        print(f"  ✅ This aligns with STRONG_BEARISH signals")
    elif decision.action == "YES":
        print(f"  ❌ WRONG! AI chose 'YES' (would buy UP token)")
        print(f"  ❌ This contradicts STRONG_BEARISH signals")
    elif decision.action == "HOLD":
        print(f"  ⚠️  AI chose HOLD")
        print(f"  Reason: {decision.reasoning}")
        if decision.confidence < 0.70:
            print(f"  ℹ️  Confidence {decision.confidence:.2f} below threshold 0.70")
        else:
            print(f"  ⚠️  Confidence {decision.confidence:.2f} is sufficient but AI is cautious")

    print("\n" + "=" * 60)

if __name__ == "__main__":
    asyncio.run(test_decision_mapping())
