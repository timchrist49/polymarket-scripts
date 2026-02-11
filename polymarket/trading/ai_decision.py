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

            # Call OpenAI with GPT-5-Nano parameters
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=self.settings.openai_model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an autonomous trading bot for Polymarket BTC 15-minute up/down markets. Use reasoning tokens to analyze all signals carefully. Always return valid JSON."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    temperature=1.0,  # GPT-5-Nano only supports default temperature=1.0
                    reasoning_effort=self.settings.openai_reasoning_effort,  # minimal/low/medium/high
                    max_completion_tokens=8000,  # Increased for reasoning tokens (medium effort ~2k-4k reasoning + 1k output)
                    response_format={"type": "json_object"}
                ),
                timeout=60.0  # Increased timeout for reasoning (gpt-5-nano with medium effort needs ~40s)
            )

            # Parse response (handle GPT-5-Nano reasoning format)
            content = response.choices[0].message.content

            # Log response for debugging
            if not content or content.strip() == "":
                # GPT-5-Nano may return empty content with reasoning tokens
                logger.warning("Empty content from GPT-5-Nano, using refusal if present")
                if hasattr(response.choices[0].message, 'refusal') and response.choices[0].message.refusal:
                    raise ValueError(f"Model refused: {response.choices[0].message.refusal}")
                raise ValueError("Empty response from AI model")

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
        """Build the AI prompt with all context including price-to-beat and timing."""

        # Get market outcomes (e.g., ["Up", "Down"])
        outcomes = market.get("outcomes", ["Yes", "No"])
        yes_outcome = outcomes[0] if len(outcomes) > 0 else "Yes"
        no_outcome = outcomes[1] if len(outcomes) > 1 else "No"

        yes_price = float(market.get("yes_price", 0.5))
        no_price = float(market.get("no_price", 0.5))

        # NEW: Price-to-beat context
        price_to_beat = market.get("price_to_beat")
        has_price_to_beat = price_to_beat is not None

        if has_price_to_beat:
            price_diff = float(btc_price.price - price_to_beat)
            price_diff_pct = (price_diff / float(price_to_beat)) * 100
            price_context = f"""
PRICE-TO-BEAT ANALYSIS:
- Starting Price (Market Open): ${price_to_beat:,.2f}
- Current Price: ${btc_price.price:,.2f}
- Difference: ${price_diff:+,.2f} ({price_diff_pct:+.2f}%)
- Direction: {"UP ✓" if price_diff > 0 else "DOWN ✓" if price_diff < 0 else "UNCHANGED"}
"""
        else:
            price_context = "PRICE-TO-BEAT: Not available (market just started)"

        # NEW: Signal Validation Rules (only when price-to-beat available)
        if has_price_to_beat:
            validation_rules = f"""
⚠️ SIGNAL VALIDATION RULES:

You MUST check for contradictions between market signals and actual BTC movement:

1. **BEARISH Signal + BTC Actually UP:**
   - If aggregated market score < -0.3 (BEARISH)
   - AND BTC is UP from price-to-beat (+{price_diff_pct:+.2f}%)
   - → This is a CONTRADICTION - market is lagging behind reality
   - → Decision: HOLD (do NOT bet NO when BTC is going UP)

2. **BULLISH Signal + BTC Actually DOWN:**
   - If aggregated market score > +0.3 (BULLISH)
   - AND BTC is DOWN from price-to-beat ({price_diff_pct:+.2f}%)
   - → This is a CONTRADICTION - market is lagging behind reality
   - → Decision: HOLD (do NOT bet YES when BTC is going DOWN)

3. **Signals ALIGN:**
   - If market sentiment matches actual BTC direction
   - → Proceed with normal confidence-based decision

**Why This Matters:**
- Polymarket sentiment shows what traders THINK, not what IS happening
- The 2-minute collection window often lags actual BTC movement
- Following contradictory signals leads to consistent losses
- Example: Market says "bearish" based on old data, but BTC already bounced

**When to Override:**
- Only if you have VERY STRONG conviction (>0.95 confidence)
- AND can explain in reasoning why the contradiction is temporary
- Otherwise: HOLD and wait for signals to align
"""
        else:
            validation_rules = ""

        # NEW: Timing context
        time_remaining = market.get("time_remaining_seconds", 900)
        is_end_of_market = market.get("is_end_of_market", False)

        minutes_remaining = time_remaining // 60
        seconds_remaining = time_remaining % 60

        timing_context = f"""
MARKET TIMING:
- Time Remaining: {minutes_remaining}m {seconds_remaining}s
- Market Phase: {"🔴 END PHASE (< 3 min)" if is_end_of_market else "🟢 EARLY/MID PHASE"}
"""

        if is_end_of_market:
            timing_context += """
⚠️ END-OF-MARKET STRATEGY:
- Trend is likely established (less time for reversal)
- Price movements now have higher predictive value
- If signals strongly align, confidence can be boosted
- Still require full analysis - no rushed decisions
"""

        # NEW: BTC Actual Momentum context
        btc_momentum = market.get("btc_momentum")
        has_momentum = btc_momentum is not None

        if has_momentum:
            momentum_pct = btc_momentum['momentum_pct']
            momentum_dir = btc_momentum['direction']
            price_5min = btc_momentum['price_5min_ago']

            momentum_context = f"""
ACTUAL BTC MOMENTUM (last 5 minutes):
- 5 minutes ago: ${price_5min:,.2f}
- Current: ${btc_price.price:,.2f}
- Change: {momentum_pct:+.2f}% ({momentum_dir})

⚠️ COMPARE WITH MARKET SIGNALS:
- If market sentiment is BEARISH but BTC is UP → market is LAGGING
- If market sentiment is BULLISH but BTC is DOWN → market is LAGGING
- Lagging signals often lead to losing trades - consider HOLD
"""
        else:
            momentum_context = "ACTUAL BTC MOMENTUM: Not available (insufficient price history)"

        # Extract social and market details
        social = aggregated.social
        mkt = aggregated.market

        return f"""You are an autonomous trading bot for Polymarket BTC 15-minute up/down markets.
Use your reasoning tokens to carefully analyze all signals before making a decision.

{price_context}

{validation_rules}

{timing_context}

{momentum_context}

CURRENT MARKET DATA:
- BTC Current Price: ${btc_price.price:,.2f} (source: {btc_price.source})
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

MARKET MICROSTRUCTURE (Polymarket CLOB, last 5-15 min):
- Score: {mkt.score:+.2f} (-1.0 to +1.0)
- Confidence: {mkt.confidence:.2f}
- Order Book: {mkt.order_book_bias} (bid walls vs ask walls, score: {mkt.order_book_score:+.2f})
- Whale Activity: {mkt.whale_direction} ({mkt.whale_count} large orders >$1000, score: {mkt.whale_score:+.2f})
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
1. USE YOUR REASONING TOKENS to analyze:
   - ⚠️ CHECK VALIDATION RULES FIRST - any contradictions?
   - Price-to-beat direction (is current price up or down from start?)
   - Actual BTC momentum (is BTC moving up or down right now?)
   - Market signals (what does Polymarket sentiment say?)
   - Technical indicators alignment
   - Time remaining (end-of-market = established trend)

2. CONSIDER END-OF-MARKET STRATEGY:
   - If < 3 minutes remaining AND all signals align → higher confidence justified
   - Trend is less likely to reverse with limited time
   - Price-to-beat difference becomes more predictive

3. The aggregated confidence ({aggregated.final_confidence:.2f}) is pre-calculated.
   - You may ADJUST by max ±0.15 if you spot patterns we missed
   - Boost if: All signals strongly align + end-of-market + clear price direction
   - Reduce if: Conflicting signals or suspicious patterns

4. Only trade if final confidence >= {self.settings.bot_confidence_threshold}

DECISION FORMAT:
Return JSON with:
{{
  "action": "YES" | "NO" | "HOLD",
  "confidence": 0.0-1.0,
  "reasoning": "Brief explanation with reasoning chain (2-3 sentences)",
  "confidence_adjustment": "+0.1" or "-0.05" or "0.0",
  "position_size": "amount in USDC as number",
  "stop_loss": "odds threshold to cancel bet (0.0-1.0)"
}}

ACTION MAPPING:
- Return "YES" to buy the "{yes_outcome}" token (currently {yes_price:.2f} odds)
- Return "NO" to buy the "{no_outcome}" token (currently {no_price:.2f} odds)
- Return "HOLD" if signals are unclear or confidence is too low

CRITICAL ALIGNMENT CHECK:
- BULLISH signals (BTC going UP from price-to-beat) → Buy "{yes_outcome}" token
- BEARISH signals (BTC going DOWN from price-to-beat) → Buy "{no_outcome}" token
- If price-to-beat shows +2% but you're bearish → HOLD (conflicting signals)
"""

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
