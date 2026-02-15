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
    TradingDecision,
    VolumeData,
    MarketRegime,
    ArbitrageOpportunity
)
from polymarket.trading.timeframe_analyzer import TimeframeAnalysis
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
        portfolio_value: Decimal = Decimal("1000"),
        orderbook_data: "OrderbookData | None" = None,  # orderbook analysis
        volume_data: VolumeData | None = None,  # NEW: volume confirmation
        timeframe_analysis: TimeframeAnalysis | None = None,  # NEW: multi-timeframe
        regime: MarketRegime | None = None,  # NEW: market regime
        arbitrage_opportunity: "ArbitrageOpportunity | None" = None,  # NEW: arbitrage detection
        market_signals: "Any | None" = None,  # NEW: CoinGecko Pro market signals
        force_trade: bool = False  # NEW: TEST MODE - force YES/NO decision
    ) -> TradingDecision:
        """Generate trading decision using AI with regime awareness, volume, timeframe, arbitrage, and market signals."""
        try:
            client = self._get_client()

            # Build the prompt
            prompt = self._build_prompt(
                btc_price, technical_indicators, aggregated_sentiment,
                market_data, portfolio_value, orderbook_data,
                volume_data, timeframe_analysis, regime,
                arbitrage_opportunity, market_signals, force_trade
            )

            # Call OpenAI with GPT-5-Nano parameters
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=self.settings.openai_model,
                    messages=[
                        {
                            "role": "system",
                            "content": """You are a professional Bitcoin trader analyzing 15-minute markets on Polymarket with regime-aware strategy.

CRITICAL RULES FOR 15-MIN BTC TRADING:

## Market Regime Strategy
- TRENDING Markets: Follow trend, enter on pullbacks (not extremes)
- RANGING Markets: Buy support, sell resistance
- VOLATILE/UNCLEAR Markets: HOLD - Don't trade unclear conditions

## Volume Requirements
- Breakouts ($200+ moves): REQUIRE volume > 1.5x average
- Without volume = false breakout = HOLD

## Timeframe Alignment
- ALIGNED timeframes (daily + 4h same): High confidence trades
- CONFLICTING timeframes: HOLD - Don't trade against larger trend

Use reasoning tokens to analyze all signals carefully. Always return valid JSON."""
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
            decision = self._parse_decision(decision_data, market_data.get("token_id", ""))

            # Apply weighted confidence calculation (tiered signal prioritization)
            # This prevents sentiment from overriding price reality
            if decision.action in ("YES", "NO"):
                # Extract signal data
                price_direction = "UP" if decision.action == "YES" else "DOWN"
                price_signal_strength = decision.confidence  # AI's base confidence

                # Volume confirmation
                volume_confirmation = volume_data.is_high_volume if volume_data else False

                # Orderbook bias
                orderbook_bias = "neutral"
                if orderbook_data:
                    if orderbook_data.order_imbalance > 0.2:
                        orderbook_bias = "bullish"
                    elif orderbook_data.order_imbalance < -0.2:
                        orderbook_bias = "bearish"

                # CoinGecko signals
                coingecko_signal_strength = 0.5
                coingecko_direction = "neutral"
                if market_signals and hasattr(market_signals, 'confidence'):
                    coingecko_signal_strength = market_signals.confidence
                    coingecko_direction = market_signals.direction

                # Sentiment
                sentiment_score = aggregated_sentiment.final_score
                sentiment_confidence = aggregated_sentiment.final_confidence

                # Calculate weighted confidence
                weighted_confidence = self._calculate_weighted_confidence(
                    price_signal_strength=price_signal_strength,
                    price_direction=price_direction,
                    volume_confirmation=volume_confirmation,
                    orderbook_bias=orderbook_bias,
                    coingecko_signal_strength=coingecko_signal_strength,
                    coingecko_direction=coingecko_direction,
                    sentiment_score=sentiment_score,
                    sentiment_confidence=sentiment_confidence,
                    base_confidence=decision.confidence
                )

                logger.info(
                    "Applied tiered signal weighting",
                    ai_confidence=f"{decision.confidence:.1%}",
                    weighted_confidence=f"{weighted_confidence:.1%}",
                    price_direction=price_direction,
                    volume_confirmed=volume_confirmation,
                    sentiment_score=f"{sentiment_score:+.2f}"
                )

                decision.confidence = weighted_confidence

            # Apply timeframe confidence modifier if available
            if timeframe_analysis:
                base_confidence = decision.confidence
                modifier = timeframe_analysis.confidence_modifier
                decision.confidence = min(base_confidence + modifier, 1.0)

                logger.info(
                    "Applied timeframe confidence modifier",
                    base_confidence=f"{base_confidence:.1%}",
                    modifier=f"{modifier:+.1%}",
                    final_confidence=f"{decision.confidence:.1%}"
                )

            return decision

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
        portfolio_value: Decimal,
        orderbook_data: "OrderbookData | None" = None,
        volume_data: VolumeData | None = None,
        timeframe_analysis: TimeframeAnalysis | None = None,
        regime: MarketRegime | None = None,
        arbitrage_opportunity: "ArbitrageOpportunity | None" = None,
        market_signals: "Any | None" = None,
        force_trade: bool = False
    ) -> str:
        """Build the AI prompt with all context including regime, volume, timeframe, orderbook, and market signals."""

        # NEW: Build force_trade instruction if enabled
        if force_trade:
            force_trade_instruction = """
⚠️ TEST MODE ACTIVE - FORCED TRADING
═══════════════════════════════════════════════════════════════════
CRITICAL: You MUST return either "YES" or "NO" - HOLD is NOT allowed.

Use ALL available data to make the most informed decision:
- Technical indicators (RSI, MACD, trend)
- CoinGecko Pro signals (funding rates, exchange premium, volume)
- Sentiment analysis
- Market regime and volatility
- Timeframe alignment

If signals are mixed or unclear, favor the direction suggested by
the strongest signal with highest confidence. Make a decision.
═══════════════════════════════════════════════════════════════════
"""
        else:
            force_trade_instruction = ""

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

        # NEW: Market Regime context
        regime_context = ""
        if regime:
            regime_context = f"""
MARKET REGIME ANALYSIS:
- Regime: {regime.regime}
- Is Trending: {regime.is_trending}
- Trend Direction: {regime.trend_direction}
- Volatility: {regime.volatility:.2f}%
- Confidence: {regime.confidence:.2f}

⚠️ REGIME-BASED STRATEGY:
- TRENDING UP: Only YES trades on pullbacks (not at 24h high)
- TRENDING DOWN: Only NO trades on bounces (not at 24h low)
- RANGING: YES near support, NO near resistance
- VOLATILE/UNCLEAR: HOLD - Don't trade unclear conditions
"""
        else:
            regime_context = "MARKET REGIME: Not available (insufficient data)"

        # NEW: Volume context
        volume_context = ""
        if volume_data:
            volume_context = f"""
VOLUME CONFIRMATION:
- 24h Volume: ${volume_data.volume_24h:,.0f}
- Current Hour Volume: ${volume_data.volume_current_hour:,.0f}
- Average Hour Volume: ${volume_data.volume_avg_hour:,.0f}
- Volume Ratio: {volume_data.volume_ratio:.2f}x (current / average)
- High Volume: {volume_data.is_high_volume}

⚠️ VOLUME RULES:
- For breakouts ($200+ moves): REQUIRE volume_ratio > 1.5x
- Without volume = false breakout = HOLD
- Low volume = reduce confidence by 0.1-0.2
"""
        else:
            volume_context = "VOLUME CONFIRMATION: Not available"

        # NEW: Timeframe alignment context
        timeframe_context = ""
        if timeframe_analysis:
            tf = timeframe_analysis
            timeframe_context = f"""
TIMEFRAME ANALYSIS:
- 1-minute trend: {tf.tf_1m.direction} ({tf.tf_1m.price_change_pct:+.2f}%)
- 5-minute trend: {tf.tf_5m.direction} ({tf.tf_5m.price_change_pct:+.2f}%)
- 15-minute trend: {tf.tf_15m.direction} ({tf.tf_15m.price_change_pct:+.2f}%)
- 30-minute trend: {tf.tf_30m.direction} ({tf.tf_30m.price_change_pct:+.2f}%)
- Alignment: {tf.alignment_score}
- Confidence Modifier: {tf.confidence_modifier:+.2%}

INTERPRETATION:
- All 4 aligned = Strongest signal (use for directional trades)
- 3 of 4 aligned = Strong trend emerging
- Mixed = Consolidation or reversal in progress
- Conflicting = Avoid trading or wait for clarity

Your confidence will be automatically adjusted based on alignment:
- All 4 aligned: +20% confidence boost
- 3 of 4 aligned: +15% confidence boost
- Mixed signals: No adjustment
- Conflicting signals: -15% confidence reduction
"""
        else:
            timeframe_context = "TIMEFRAME ANALYSIS: Not available (insufficient price history)"

        # NEW: Arbitrage opportunity context
        arbitrage_context = ""
        if arbitrage_opportunity and arbitrage_opportunity.edge_percentage >= 0.05:
            arbitrage_context = f"""
═══════════════════════════════════════════════════════════════════
🎯 ARBITRAGE OPPORTUNITY DETECTED
═══════════════════════════════════════════════════════════════════

PROBABILITY ANALYSIS:
├─ Actual Probability (calculated): {arbitrage_opportunity.actual_probability:.1%}
├─ Polymarket YES Odds: {arbitrage_opportunity.polymarket_yes_odds:.1%}
├─ Polymarket NO Odds: {arbitrage_opportunity.polymarket_no_odds:.1%}
└─ **EDGE: {arbitrage_opportunity.edge_percentage:+.1%}**

ARBITRAGE SIGNAL:
├─ Recommended Action: **{arbitrage_opportunity.recommended_action}**
├─ Confidence Boost: +{arbitrage_opportunity.confidence_boost:.2f}
├─ Urgency: {arbitrage_opportunity.urgency}
└─ Expected Profit: +{arbitrage_opportunity.expected_profit_pct:.1%} if correct

⚠️ ARBITRAGE TRADING STRATEGY:
This is a QUANTIFIED MISPRICING based on:
1. Real-time BTC price momentum analysis
2. Comparison to lagging Polymarket odds
3. Statistical probability calculation

CONFIDENCE SCALING:
- Edge 5-10%: Moderate opportunity (confidence boost: +0.10 to +0.20)
- Edge 10-15%: Strong opportunity (confidence boost: +0.20, urgency: MEDIUM)
- Edge 15%+: Extreme opportunity (confidence boost: +0.20, urgency: HIGH)

The larger the edge, the higher your confidence should be.
Edges of 10%+ justify high confidence (0.85-0.95).

═══════════════════════════════════════════════════════════════════
"""
        else:
            arbitrage_context = """
═══════════════════════════════════════════════════════════════════
🎯 ARBITRAGE STATUS
═══════════════════════════════════════════════════════════════════
No significant arbitrage edge detected (< 5%).
Rely on technical indicators, sentiment, and regime analysis.
═══════════════════════════════════════════════════════════════════
"""

        # NEW: Market signals from CoinGecko Pro
        market_signals_context = ""
        if market_signals:
            # Build context from composite signal
            direction_icon = {"bullish": "🟢", "bearish": "🔴", "neutral": "⚪"}.get(market_signals.direction, "⚪")

            signal_details = []
            for signal in market_signals.contributing_signals:
                signal_icon = {"bullish": "🟢", "bearish": "🔴", "neutral": "⚪"}.get(signal.direction, "⚪")
                signal_details.append(
                    f"  {signal_icon} {signal.signal_type.upper()}: {signal.direction.upper()} "
                    f"(confidence: {signal.confidence:.2f})"
                )

                # Add metadata details
                if signal.metadata:
                    if signal.signal_type == "funding_rate" and "rate_pct" in signal.metadata:
                        signal_details.append(f"     ├─ Rate: {signal.metadata['rate_pct']:.4f}%")
                    elif signal.signal_type == "exchange_premium" and "premiums" in signal.metadata:
                        premiums_str = ", ".join(f"{k}: {v:.2f}%" for k, v in signal.metadata["premiums"].items())
                        signal_details.append(f"     ├─ Premiums: {premiums_str}")
                    elif signal.signal_type == "volume" and "percentile" in signal.metadata:
                        signal_details.append(f"     ├─ Volume Percentile: {signal.metadata['percentile']:.1f}%")

            market_signals_context = f"""
═══════════════════════════════════════════════════════════════════
📊 COINGECKO PRO MARKET SIGNALS
═══════════════════════════════════════════════════════════════════

COMPOSITE SIGNAL:
{direction_icon} Direction: {market_signals.direction.upper()}
└─ Confidence: {market_signals.confidence:.2f}

CONTRIBUTING SIGNALS:
{chr(10).join(signal_details)}

SIGNAL WEIGHTS:
""" + "\n".join(f"- {k}: {v:.0%}" for k, v in market_signals.weights.items()) + """

⚠️ MANDATORY SIGNAL INTEGRATION RULES:

You MUST apply these quantified rules when making your decision:

1️⃣ DIRECTION DETERMINATION (Up or Down):
   ├─ If market_signals.confidence > 0.6 AND direction = "BULLISH" → FAVOR YES
   ├─ If market_signals.confidence > 0.6 AND direction = "BEARISH" → FAVOR NO
   ├─ If market_signals.confidence < 0.4 OR direction = "NEUTRAL" → Ignore signals
   └─ Strong signals (>0.7 confidence) can override weak technical indicators

2️⃣ CONFIDENCE ADJUSTMENT (Bet or Not):

   A) ALIGNMENT (signals match technical/sentiment direction):
      ├─ High signal confidence (>0.7) → BOOST final confidence by +0.10 to +0.15
      ├─ Medium signal confidence (0.5-0.7) → BOOST final confidence by +0.05 to +0.10
      └─ Low signal confidence (0.4-0.5) → Minimal boost +0.02 to +0.05

   B) CONFLICT (signals contradict technical/sentiment):
      ├─ High signal confidence (>0.7) → REDUCE confidence by -0.15 OR recommend HOLD
      ├─ Medium signal confidence (0.5-0.7) → REDUCE confidence by -0.10
      └─ Low signal confidence (0.4-0.5) → REDUCE confidence by -0.05

3️⃣ INTEGRATION EXAMPLES:

   Example A - Strong Alignment:
   - Technical: BULLISH (RSI oversold, MACD positive)
   - Signals: BULLISH (confidence 0.75)
   - Assessment: ALIGNED → Apply +0.12 boost
   - Calculation: Base 0.70 + 0.12 = 0.82 final confidence
   - Decision: Strong YES with high confidence

   Example B - Strong Conflict:
   - Technical: BEARISH (RSI overbought, MACD negative)
   - Signals: BULLISH (confidence 0.72)
   - Assessment: CONFLICT → Reduce -0.13 OR HOLD
   - Calculation: Base 0.65 - 0.13 = 0.52 (borderline) OR HOLD
   - Decision: Either weak YES or HOLD (prefer HOLD with strong conflicts)

   Example C - Weak Signals:
   - Technical: BULLISH
   - Signals: NEUTRAL (confidence 0.35)
   - Assessment: IGNORE → No adjustment
   - Calculation: Base 0.68 + 0.00 = 0.68 final confidence
   - Decision: Proceed with technical analysis only

4️⃣ REASONING OUTPUT REQUIREMENT:

   Your reasoning MUST explicitly show:
   - Base confidence before signal adjustment
   - Market signals direction and confidence level
   - Whether signals ALIGN or CONFLICT with primary analysis
   - Exact confidence adjustment applied (+/- specific amount)
   - Final confidence after adjustment

   Example format:
   "Technical indicators BULLISH (base confidence: 0.70). Market signals: BULLISH
   (0.75 confidence). Signals ALIGN strongly. Applying +0.12 confidence boost.
   Final confidence: 0.82. Strong YES signal."

═══════════════════════════════════════════════════════════════════
"""
        else:
            market_signals_context = """
═══════════════════════════════════════════════════════════════════
📊 COINGECKO PRO MARKET SIGNALS
═══════════════════════════════════════════════════════════════════
Market signals not available (API timeout or rate limit).
Proceed with standard technical and sentiment analysis.
═══════════════════════════════════════════════════════════════════
"""

        # Extract social and market details
        social = aggregated.social
        mkt = aggregated.market

        return f"""You are an autonomous trading bot for Polymarket BTC 15-minute up/down markets.
Use your reasoning tokens to carefully analyze all signals before making a decision.

{price_context}

{validation_rules}

{timing_context}

{momentum_context}

{regime_context}

{volume_context}

{timeframe_context}

{arbitrage_context}

{market_signals_context}

{force_trade_instruction}

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
{self._format_orderbook_section(orderbook_data)}

AGGREGATED SIGNAL:
- Final Score: {aggregated.final_score:+.2f} (market 60% + social 40%)
- Final Confidence: {aggregated.final_confidence:.2f} ({aggregated.final_confidence*100:.0f}%)
- Signal Type: {aggregated.signal_type}
- Agreement: {aggregated.agreement_multiplier:.2f}x {"(signals align - boosted confidence)" if aggregated.agreement_multiplier > 1.1 else "(signals conflict - reduced confidence)" if aggregated.agreement_multiplier < 0.9 else "(moderate agreement)"}

🎯 ODDS AND EXPECTED VALUE STRATEGY:

Understanding Binary Market Payouts:
- Buying at LOW odds (< 0.40) = CONTRARIAN bet = Much higher profit when you win
  Example: Buy $5 @ $0.25 odds → Get 20 shares → Win pays $20 → Profit = $15
- Buying at HIGH odds (> 0.60) = FOLLOWING CROWD = Much lower profit when you win
  Example: Buy $5 @ $0.75 odds → Get 6.67 shares → Win pays $6.67 → Profit = $1.67

Current Market Odds Analysis:
- YES token odds: {yes_price:.2f} {"(CONTRARIAN opportunity - high profit potential!)" if yes_price < 0.40 else "(CROWDED - low profit potential)" if yes_price > 0.60 else "(MODERATE odds)"}
- NO token odds: {no_price:.2f} {"(CONTRARIAN opportunity - high profit potential!)" if no_price < 0.40 else "(CROWDED - low profit potential)" if no_price > 0.60 else "(MODERATE odds)"}

STRATEGY PRIORITY (CRITICAL):
1. **PREFER CONTRARIAN BETS** (odds < 0.40) with reasonable confidence (> 0.75)
   - These trades have 3-5x higher profit potential
   - Historical data: Contrarian trades average $5.67 profit vs $0.83 for crowd-following

2. **AVOID HIGH-ODDS BETS** (odds > 0.70) unless EXTREMELY high confidence (> 0.95)
   - Even when you win, profit is minimal
   - Only justified if signals are overwhelmingly strong

3. **CALCULATE EXPECTED VALUE:**
   - Low odds bet: 60% win rate × $15 profit = $9.00 EV (GOOD!)
   - High odds bet: 60% win rate × $1.67 profit = $1.00 EV (POOR!)

4. When both signals and odds align for contrarian bet → STRONG BUY signal
   - Example: Bearish signals + YES at $0.30 (crowd thinks UP) → Buy NO token
   - You're betting against the crowd with data supporting your position

RISK PARAMETERS:
- Confidence threshold: {self.settings.bot_confidence_threshold * 100:.0f}%
- Max position: {self.settings.bot_max_position_percent * 100:.0f}% of portfolio
- Current portfolio value: ${portfolio_value:,.2f}

DECISION INSTRUCTIONS:
1. USE YOUR REASONING TOKENS to analyze:
   - ⚠️ CHECK VALIDATION RULES FIRST - any contradictions?
   - ⚠️ CHECK MARKET REGIME - is it TRENDING/RANGING/VOLATILE/UNCLEAR?
   - ⚠️ CHECK TIMEFRAME ALIGNMENT - are daily + 4h aligned or conflicting?
   - ⚠️ CHECK VOLUME - is there sufficient volume for the move?
   - Price-to-beat direction (is current price up or down from start?)
   - Actual BTC momentum (is BTC moving up or down right now?)
   - Market signals (what does Polymarket sentiment say?)
   - Technical indicators alignment
   - Time remaining (end-of-market = established trend)

2. TIMING STRATEGY (CRITICAL):
   ⚠️ PREFER EARLY TRADES - Historical data shows trades with > 10 min remaining have 5x better returns!

   - **> 10 minutes remaining:** OPTIMAL timing (avg $8.31 profit per trade)
     → Full price movement capture, less rushed analysis
     → PREFERRED: Place trades early in the 15-min window

   - **5-10 minutes remaining:** ACCEPTABLE if strong signals
     → Moderate profit potential (avg $0.51 profit per trade)

   - **< 3 minutes remaining:** AVOID unless EXCEPTIONAL circumstances
     → Very low profit potential (avg $1.66 profit per trade)
     → Rushed decisions, limited time for position to develop
     → Only justified if: (a) Extremely strong signals (> 0.95 confidence)
                         (b) Clear contrarian opportunity (odds < 0.30)
                         (c) No prior opportunity to enter

3. REGIME-BASED ENTRY RULES (CRITICAL):

   **YES (bet BTC goes UP):**
   - Market regime must be TRENDING UP or RANGING (near support)
   - Daily + 4h timeframes must be BULLISH or NEUTRAL (not CONFLICTING)
   - Current price NOT at 24h high (wait for pullback)
   - If move > $200, require high volume (volume_ratio > 1.5x)
   - Confidence > 0.75

   **NO (bet BTC goes DOWN):**
   - Market regime must be TRENDING DOWN or RANGING (near resistance)
   - Daily + 4h timeframes must be BEARISH or NEUTRAL (not CONFLICTING)
   - Current price NOT at 24h low (wait for bounce)
   - If move > $200, require high volume (volume_ratio > 1.5x)
   - Confidence > 0.70

   **HOLD (do not trade):**
   - Regime is VOLATILE or UNCLEAR
   - Timeframes CONFLICTING
   - No volume on large moves
   - Price at extremes (exhausted momentum)
   - Confidence < 0.70
   - Unclear market conditions

4. The aggregated confidence ({aggregated.final_confidence:.2f}) is pre-calculated.
   - You may ADJUST by max ±0.15 if you spot patterns we missed
   - Boost if: All signals strongly align + end-of-market + clear price direction + regime clear
   - Reduce if: Conflicting signals, unclear regime, or suspicious patterns

5. Only trade if final confidence >= {self.settings.bot_confidence_threshold}

DECISION FORMAT:
Return JSON with:
{{
  "action": "YES" | "NO" | "HOLD",
  "confidence": 0.0-1.0,
  "reasoning": "MUST include: (1) Base confidence before signals, (2) Market signals direction & confidence, (3) Alignment or conflict assessment, (4) Signal adjustment applied (+/- amount), (5) Final confidence. Example: 'Technical BULLISH (base: 0.70). Signals: BULLISH (0.75). ALIGNED. Applied +0.12 boost. Final: 0.82.'",
  "confidence_adjustment": "+0.1" or "-0.05" or "0.0",
  "position_size": "amount in USDC as number",
  "stop_loss": "odds threshold to cancel bet (0.0-1.0)"
}}

ACTION MAPPING:
- Return "YES" to buy the "{yes_outcome}" token (currently {yes_price:.2f} odds)
- Return "NO" to buy the "{no_outcome}" token (currently {no_price:.2f} odds)
- Return "HOLD" if signals are unclear or confidence is too low

CRITICAL ALIGNMENT CHECK:
- BULLISH signals + TRENDING UP/RANGING regime + ALIGNED timeframes → Buy "{yes_outcome}" token
- BEARISH signals + TRENDING DOWN/RANGING regime + ALIGNED timeframes → Buy "{no_outcome}" token
- If regime is VOLATILE/UNCLEAR → HOLD (wait for clarity)
- If timeframes CONFLICTING → HOLD (don't trade against larger trend)
- If large move without volume → HOLD (false breakout)
- If price-to-beat shows +2% but you're bearish → HOLD (conflicting signals)
"""

    def _format_orderbook_section(self, orderbook_data: "OrderbookData | None") -> str:
        """Format orderbook data section for AI prompt."""
        if not orderbook_data:
            return ""

        return f"""

ORDERBOOK DEPTH ANALYSIS (Execution Quality):
- Bid-Ask Spread: {orderbook_data.spread_bps:.1f} bps
- Liquidity Score: {orderbook_data.liquidity_score:.2f}/1.0 {"(EXCELLENT)" if orderbook_data.liquidity_score > 0.7 else "(POOR - risk of slippage)" if orderbook_data.liquidity_score < 0.4 else "(MODERATE)"}
- Order Imbalance: {orderbook_data.order_imbalance:+.2f} ({orderbook_data.imbalance_direction})
- Bid Depth (100bps): ${orderbook_data.bid_depth_100bps:.2f}
- Ask Depth (100bps): ${orderbook_data.ask_depth_100bps:.2f}
- Can Fill $8 Order: {orderbook_data.can_fill_order}

ORDERBOOK INTERPRETATION:
- Tight spread (<200bps) = liquid market, good execution quality
- Wide spread (>500bps) = illiquid, should HOLD (already filtered)
- Buy pressure (imbalance >+0.2) = bullish institutional flow signal
- Sell pressure (imbalance <-0.2) = bearish institutional flow signal
- Insufficient depth = cannot fill order (already filtered)
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

    @staticmethod
    def _calculate_weighted_confidence(
        price_signal_strength: float,
        price_direction: str,
        volume_confirmation: bool,
        orderbook_bias: str,
        coingecko_signal_strength: float,
        coingecko_direction: str,
        sentiment_score: float,
        sentiment_confidence: float,
        base_confidence: float
    ) -> float:
        """
        Calculate weighted confidence using tiered signal prioritization.

        Tier 1 (Price Reality): BTC price movements - 50% weight
        Tier 2 (Market Structure): Volume + orderbook - 25% weight
        Tier 3 (External Signals): CoinGecko Pro - 15% weight
        Tier 4 (Opinion): Sentiment - 10% weight

        This prevents sentiment from overriding price reality.

        Args:
            price_signal_strength: 0.0-1.0, strength of price signal
            price_direction: "UP", "DOWN", or "neutral"
            volume_confirmation: True if volume supports the move
            orderbook_bias: "bullish", "bearish", or "neutral"
            coingecko_signal_strength: 0.0-1.0, strength of CoinGecko signal
            coingecko_direction: "bullish", "bearish", or "neutral"
            sentiment_score: -1.0 to +1.0, sentiment score
            sentiment_confidence: 0.0-1.0, confidence in sentiment
            base_confidence: 0.0-1.0, starting confidence

        Returns:
            Weighted confidence (0.0-1.0)
        """
        # Tier 1: Price Reality (50% weight)
        # Price movements are ground truth
        price_contribution = price_signal_strength * 0.50

        # Tier 2: Market Structure (25% weight)
        # Volume and orderbook provide confirmation
        volume_score = 1.0 if volume_confirmation else 0.3
        orderbook_score = {
            "bullish": 0.8 if price_direction == "UP" else 0.2,
            "bearish": 0.8 if price_direction == "DOWN" else 0.2,
            "neutral": 0.5
        }.get(orderbook_bias, 0.5)
        market_structure = (volume_score + orderbook_score) / 2
        structure_contribution = market_structure * 0.25

        # Tier 3: External Signals (15% weight)
        # CoinGecko Pro signals have moderate influence
        coingecko_aligned = (
            (price_direction == "UP" and coingecko_direction == "bullish") or
            (price_direction == "DOWN" and coingecko_direction == "bearish") or
            coingecko_direction == "neutral"
        )
        coingecko_score = coingecko_signal_strength if coingecko_aligned else (1.0 - coingecko_signal_strength)
        coingecko_contribution = coingecko_score * 0.15

        # Tier 4: Opinion (10% weight)
        # Sentiment has lowest priority
        sentiment_direction = "UP" if sentiment_score > 0.2 else "DOWN" if sentiment_score < -0.2 else "neutral"
        sentiment_aligned = (
            (price_direction == "UP" and sentiment_direction == "UP") or
            (price_direction == "DOWN" and sentiment_direction == "DOWN") or
            sentiment_direction == "neutral"
        )
        sentiment_strength = abs(sentiment_score)  # 0.0-1.0
        sentiment_effective = sentiment_strength * sentiment_confidence
        sentiment_final_score = sentiment_effective if sentiment_aligned else (1.0 - sentiment_effective)
        sentiment_contribution = sentiment_final_score * 0.10

        # Calculate weighted confidence
        weighted_conf = (
            price_contribution +
            structure_contribution +
            coingecko_contribution +
            sentiment_contribution
        )

        # Detect major conflicts (price vs sentiment)
        # If price and sentiment strongly disagree, reduce confidence significantly
        price_is_bullish = price_direction == "UP" and price_signal_strength > 0.7
        price_is_bearish = price_direction == "DOWN" and price_signal_strength > 0.7
        sentiment_is_bullish = sentiment_score > 0.5 and sentiment_confidence > 0.7
        sentiment_is_bearish = sentiment_score < -0.5 and sentiment_confidence > 0.7

        major_conflict = (
            (price_is_bullish and sentiment_is_bearish) or
            (price_is_bearish and sentiment_is_bullish)
        )

        if major_conflict:
            # Severe penalty for price-sentiment conflicts
            weighted_conf *= 0.5

        # Ensure result is in valid range
        return max(0.0, min(1.0, weighted_conf))
