"""
AI Decision Service

Core decision-making engine using OpenAI's API.
Takes market data, technical indicators, and sentiment to generate trading decisions.
"""

import asyncio
import json
from decimal import Decimal
from typing import Optional
import structlog

from openai import AsyncOpenAI

from polymarket.models import (
    BTCPriceData,
    TechnicalIndicators,
    SentimentAnalysis,  # Keep for backwards compatibility
    AggregatedSentiment,
    TradingDecision
)
from polymarket.config import Settings

logger = structlog.get_logger()


class AIDecisionService:
    """AI-powered trading decision engine using OpenAI."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._client: Optional[AsyncOpenAI] = None

    def _get_client(self) -> AsyncOpenAI:
        """Lazy init of OpenAI client."""
        if self._client is None:
            if not self.settings.openai_api_key:
                raise ValueError("OPENAI_API_KEY not configured")
            self._client = AsyncOpenAI(api_key=self.settings.openai_api_key)
        return self._client

    async def make_decision(
        self,
        btc_price: BTCPriceData,
        technical_indicators: TechnicalIndicators,
        aggregated_sentiment: AggregatedSentiment,
        market_data: dict,
        portfolio_value: Decimal = Decimal("1000")
    ) -> TradingDecision:
        """Generate trading decision using AI with aggregated sentiment."""
        try:
            client = self._get_client()

            # Build the prompt
            prompt = self._build_prompt(
                btc_price, technical_indicators, aggregated_sentiment,
                market_data, portfolio_value
            )

            # Call OpenAI with timeout
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=self.settings.openai_model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an autonomous trading bot for Polymarket BTC 15-minute up/down markets. Always return valid JSON."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,  # Lower temp for more consistent decisions
                    max_tokens=500,
                    response_format={"type": "json_object"}
                ),
                timeout=10.0
            )

            # Parse response
            content = response.choices[0].message.content
            decision_data = json.loads(content)

            # Validate and create decision
            return self._parse_decision(decision_data, market_data.get("token_id", ""))

        except asyncio.TimeoutError:
            logger.error("OpenAI timeout")
            return self._hold_decision(market_data.get("token_id", ""), "OpenAI timeout")
        except Exception as e:
            logger.error("AI decision failed", error=str(e))
            return self._hold_decision(market_data.get("token_id", ""), f"Error: {str(e)}")

    def _build_prompt(
        self,
        btc_price: BTCPriceData,
        technical: TechnicalIndicators,
        aggregated: AggregatedSentiment,
        market: dict,
        portfolio_value: Decimal
    ) -> str:
        """Build the AI prompt with aggregated sentiment data."""

        # Get market outcomes (e.g., ["Up", "Down"])
        outcomes = market.get("outcomes", ["Yes", "No"])
        yes_outcome = outcomes[0] if len(outcomes) > 0 else "Yes"
        no_outcome = outcomes[1] if len(outcomes) > 1 else "No"

        yes_price = float(market.get("yes_price", 0.5))
        no_price = float(market.get("no_price", 0.5))

        # Extract social and market details
        social = aggregated.social
        mkt = aggregated.market

        return f"""You are an autonomous trading bot for Polymarket BTC 15-minute up/down markets.

CURRENT MARKET DATA:
- BTC Price: ${btc_price.price:,.2f}
- Market Question: {market.get("question", "Unknown")}
- Token Outcomes:
  * YES token = "{yes_outcome}" (current odds: {yes_price:.2f})
  * NO token = "{no_outcome}" (current odds: {no_price:.2f})

TECHNICAL INDICATORS (60-min analysis):
- RSI(14): {technical.rsi:.1f} (Overbought >70, Oversold <30)
- MACD: {technical.macd_value:.2f} (Signal: {technical.macd_signal:.2f})
- MACD Histogram: {technical.macd_histogram:.2f}
- EMA Trend: {technical.ema_short:,.2f} vs {technical.ema_long:,.2f}
- Trend: {technical.trend}
- Volume Change: {technical.volume_change:+.1f}%
- Price Velocity: ${technical.price_velocity:+.2f}/min

SOCIAL SENTIMENT (Real-time crypto APIs):
- Score: {social.score:+.2f} (-0.7 to +0.85)
- Confidence: {social.confidence:.2f}
- Fear/Greed Index: {social.fear_greed} (0=Fear, 100=Greed)
- BTC Trending: {"Yes" if social.is_trending else "No"}
- Community Votes: {social.vote_up_pct:.0f}% up, {social.vote_down_pct:.0f}% down
- Signal: {social.signal_type}
- Sources: {", ".join(social.sources_available)}

MARKET MICROSTRUCTURE (Binance, last 5-15 min):
- Score: {mkt.score:+.2f} (-1.0 to +1.0)
- Confidence: {mkt.confidence:.2f}
- Order Book: {mkt.order_book_bias} (bid walls vs ask walls, score: {mkt.order_book_score:+.2f})
- Whale Activity: {mkt.whale_direction} ({mkt.whale_count} large orders >5 BTC, score: {mkt.whale_score:+.2f})
- Volume: {mkt.volume_ratio:.1f}x normal (score: {mkt.volume_score:+.2f})
- Momentum: {mkt.momentum_direction} (score: {mkt.momentum_score:+.2f})
- Signal: {mkt.signal_type}

AGGREGATED SIGNAL:
- Final Score: {aggregated.final_score:+.2f} (market 60% + social 40%)
- Final Confidence: {aggregated.final_confidence:.2f} ({aggregated.final_confidence*100:.0f}%)
- Signal Type: {aggregated.signal_type}
- Agreement: {aggregated.agreement_multiplier:.2f}x {"(signals align - boosted confidence)" if aggregated.agreement_multiplier > 1.1 else "(signals conflict - reduced confidence)" if aggregated.agreement_multiplier < 0.9 else "(moderate agreement)"}

RISK PARAMETERS:
- Confidence threshold: {self.settings.bot_confidence_threshold * 100:.0f}%
- Max position: {self.settings.bot_max_position_percent * 100:.0f}% of portfolio
- Current portfolio value: ${portfolio_value:,.2f}

DECISION INSTRUCTIONS:
1. The aggregated confidence ({aggregated.final_confidence:.2f}) is pre-calculated based on:
   - Individual signal confidences
   - Agreement between social and market signals

2. You may ADJUST this confidence by max ±0.15 if you spot patterns we missed:
   - Boost if: All signals (social + market + technical) strongly align
   - Reduce if: You spot a red flag (e.g., whale activity contradicts price action)

3. Only trade if final confidence >= {self.settings.bot_confidence_threshold}

4. Consider the 15-minute timeframe:
   - Technical indicators show momentum
   - Market microstructure shows current order flow
   - Social sentiment shows crowd psychology

DECISION FORMAT:
Return JSON with:
{{
  "action": "YES" | "NO" | "HOLD",
  "confidence": 0.0-1.0,  // Can adjust ±0.15 from {aggregated.final_confidence:.2f}
  "reasoning": "Brief explanation (1-2 sentences)",
  "confidence_adjustment": "+0.1" or "-0.05" or "0.0",  // Explain why you adjusted
  "position_size": "amount in USDC as number",
  "stop_loss": "odds threshold to cancel bet (0.0-1.0)"
}}

Only trade if confidence >= {self.settings.bot_confidence_threshold}. Otherwise return HOLD.

ACTION MAPPING:
- Return "YES" to buy the "{yes_outcome}" token (currently {yes_price:.2f} odds)
- Return "NO" to buy the "{no_outcome}" token (currently {no_price:.2f} odds)
- Return "HOLD" if signals are unclear or confidence is too low

IMPORTANT: Choose the action that aligns with your signal direction:
- BULLISH signals (BTC going up) → Buy token that profits from BTC going up
- BEARISH signals (BTC going down) → Buy token that profits from BTC going down"""

    def _parse_decision(self, data: dict, token_id: str) -> TradingDecision:
        """Parse AI response into TradingDecision."""
        action = data.get("action", "HOLD").upper()

        # Validate action
        if action not in ("YES", "NO", "HOLD"):
            action = "HOLD"

        confidence = float(data.get("confidence", 0.0))
        reasoning = data.get("reasoning", "No reasoning provided")

        # Parse position size
        try:
            position_size = Decimal(str(data.get("position_size", 0)))
        except:
            position_size = Decimal("0")

        stop_loss = float(data.get("stop_loss", 0.40))

        return TradingDecision(
            action=action,
            confidence=confidence,
            reasoning=reasoning,
            token_id=token_id,
            position_size=position_size,
            stop_loss_threshold=stop_loss
        )

    def _hold_decision(self, token_id: str, reason: str) -> TradingDecision:
        """Return a HOLD decision."""
        return TradingDecision(
            action="HOLD",
            confidence=0.0,
            reasoning=f"Auto-HOLD: {reason}",
            token_id=token_id,
            position_size=Decimal("0"),
            stop_loss_threshold=0.40
        )
