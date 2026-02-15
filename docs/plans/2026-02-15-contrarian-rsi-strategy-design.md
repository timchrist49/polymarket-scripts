# Contrarian RSI Strategy Design

> **Date:** 2026-02-15
> **Status:** Approved for Implementation

## Goal

Add contrarian RSI strategy that detects extreme technical divergences from crowd sentiment and alerts the AI to potential reversals.

**Inspiration:** Market btc-updown-15m-1771186500 showed RSI 9.5 with DOWN odds 72%, but price reversed UP. This strategy aims to catch such setups.

## Architecture Overview

### Core Logic

```python
if RSI < 10 and DOWN_odds > 65%:
    contrarian_signal = "OVERSOLD_REVERSAL" (suggest UP)
    movement_threshold = $50  # Reduced from $100

if RSI > 90 and UP_odds > 65%:
    contrarian_signal = "OVERBOUGHT_REVERSAL" (suggest DOWN)
    movement_threshold = $50  # Reduced from $100
```

### Integration Points

1. **Detection:** After technical analysis (Step 3), check RSI + odds
2. **Threshold Adjustment:** If contrarian detected, reduce movement requirement to $50
3. **Signal Creation:** Build explicit contrarian signal object with metadata
4. **Sentiment Integration:** Add contrarian score to aggregated sentiment calculation
5. **AI Presentation:** Pass both explicit flag and updated sentiment to AI
6. **Logging:** Track contrarian setups for analysis

### Key Principles

- **Non-invasive:** Respects all existing filters (signal lag, volume, regime)
- **AI-guided:** AI makes final decision considering contrarian context
- **Conservative thresholds:** RSI < 10 or > 90 (truly extreme)
- **Evidence-based:** Inspired by real market behavior

## Detection Logic

### Function

```python
def detect_contrarian_setup(
    rsi: float,
    yes_odds: float,  # UP odds (best_bid)
    no_odds: float    # DOWN odds (1 - best_bid)
) -> Optional[ContrarianSignal]:
    """
    Detect extreme RSI divergence from crowd consensus.

    Returns ContrarianSignal if conditions met, None otherwise.
    """
    # OVERSOLD: RSI extremely low, crowd betting DOWN
    if rsi < 10 and no_odds > 0.65:
        return ContrarianSignal(
            type="OVERSOLD_REVERSAL",
            suggested_direction="UP",
            rsi=rsi,
            crowd_direction="DOWN",
            crowd_confidence=no_odds,
            confidence=0.90 + (10 - rsi) * 0.01,  # Higher confidence for lower RSI
            reasoning=f"Extreme oversold (RSI {rsi:.1f}) + strong DOWN consensus ({no_odds:.0%}) = UP reversal likely"
        )

    # OVERBOUGHT: RSI extremely high, crowd betting UP
    if rsi > 90 and yes_odds > 0.65:
        return ContrarianSignal(
            type="OVERBOUGHT_REVERSAL",
            suggested_direction="DOWN",
            rsi=rsi,
            crowd_direction="UP",
            crowd_confidence=yes_odds,
            confidence=0.90 + (rsi - 90) * 0.01,  # Higher confidence for higher RSI
            reasoning=f"Extreme overbought (RSI {rsi:.1f}) + strong UP consensus ({yes_odds:.0%}) = DOWN reversal likely"
        )

    return None
```

### Data Structure

```python
@dataclass
class ContrarianSignal:
    type: Literal["OVERSOLD_REVERSAL", "OVERBOUGHT_REVERSAL"]
    suggested_direction: Literal["UP", "DOWN"]
    rsi: float
    crowd_direction: Literal["UP", "DOWN"]
    crowd_confidence: float  # Crowd's odds
    confidence: float  # Our confidence in reversal (0.90-1.00)
    reasoning: str
```

## AI Integration

### Dual Approach

**1. Explicit Contrarian Flag (High Visibility):**

```python
if contrarian_signal:
    prompt += f"""

🔥 CONTRARIAN SETUP DETECTED 🔥

Type: {contrarian_signal.type}
RSI: {contrarian_signal.rsi:.1f} ({"EXTREMELY OVERSOLD" if contrarian_signal.rsi < 10 else "EXTREMELY OVERBOUGHT"})
Crowd Betting: {contrarian_signal.crowd_direction} at {contrarian_signal.crowd_confidence:.0%} odds
Contrarian Suggestion: BET {contrarian_signal.suggested_direction}

Reasoning: {contrarian_signal.reasoning}

⚠️ This is a mean-reversion signal. The crowd is heavily positioned for {contrarian_signal.crowd_direction},
but extreme technical indicators suggest imminent reversal to {contrarian_signal.suggested_direction}.

Confidence: {contrarian_signal.confidence:.0%}
"""
```

**2. Sentiment Integration (Unified Scoring):**

```python
# Add contrarian score to signal aggregation
if contrarian_signal:
    contrarian_score = +1.0 if contrarian_signal.suggested_direction == "UP" else -1.0
    contrarian_weight = 2.0  # High weight for extreme signals

    signals.append({
        "name": "contrarian_rsi",
        "score": contrarian_score,
        "confidence": contrarian_signal.confidence,
        "weight": contrarian_weight
    })
```

**Result:** AI sees both explicit warning AND numerical signal influence.

## Movement Threshold Adjustment

### Dynamic Threshold

```python
# Current: auto_trade.py around line 979
MIN_MOVEMENT_THRESHOLD = 100  # Default $100

# NEW: Adjust based on contrarian signal
if contrarian_signal:
    MIN_MOVEMENT_THRESHOLD = 50  # Reduced to $50 for reversals
    logger.info(
        "Contrarian setup - reducing movement threshold",
        default_threshold="$100",
        contrarian_threshold="$50",
        reasoning="Reversals start with small movements"
    )

abs_diff = abs(diff)
if abs_diff < MIN_MOVEMENT_THRESHOLD:
    # ... skip market logic
```

### Filter Interaction

All other filters remain **ACTIVE**:
- ✅ Signal lag check: Still enforced
- ✅ Volume confirmation: Still enforced for large moves
- ✅ Orderbook spread: Still enforced
- ✅ Regime filter: Still enforced
- ⚙️ Movement threshold: **Reduced** to $50 (not bypassed)

**Rationale:** Contrarian setups indicate *potential* reversals, but we still want quality execution conditions. The $50 threshold acknowledges reversals start small while maintaining directional confirmation.

## Implementation Plan

### Files to Modify

1. **polymarket/models.py**
   - Add `ContrarianSignal` dataclass

2. **polymarket/trading/contrarian.py** (NEW)
   - Create `detect_contrarian_setup()` function
   - Unit tests for edge cases

3. **scripts/auto_trade.py**
   - Import contrarian detector
   - Call detection after technical analysis
   - Adjust movement threshold conditionally
   - Add to AI prompt context
   - Integrate into sentiment aggregation
   - Add logging

4. **polymarket/performance/database.py**
   - Add `contrarian_detected` boolean field
   - Add `contrarian_type` varchar field

### Testing Strategy

1. **Unit Tests:** Test detection logic with edge cases
   - RSI 9.9, 10.0, 10.1
   - Odds 64.9%, 65.0%, 65.1%
   - Both oversold and overbought scenarios

2. **Integration Tests:** Full pipeline with contrarian signal
   - Verify movement threshold adjusts to $50
   - Verify AI receives contrarian context
   - Verify sentiment includes contrarian score

3. **Backtest:** Run against historical data
   - Find markets with RSI < 10 or > 90
   - Measure reversal rate
   - Compare with/without contrarian strategy

### Success Metrics

- **Detection Rate:** How often do contrarian setups appear?
- **AI Acceptance Rate:** How often does AI follow contrarian signal?
- **Win Rate:** Target 60-70% (mean reversion edge)
- **False Positive Rate:** RSI extremes that didn't reverse

## Logging Strategy

```python
# Track contrarian detection
logger.info(
    "Contrarian signal detected",
    type=contrarian_signal.type,
    rsi=contrarian_signal.rsi,
    suggested_direction=contrarian_signal.suggested_direction,
    crowd_direction=contrarian_signal.crowd_direction,
    crowd_confidence=f"{contrarian_signal.crowd_confidence:.0%}",
    confidence=f"{contrarian_signal.confidence:.0%}"
)

# Track AI decision
if ai_decision.action == contrarian_signal.suggested_direction:
    logger.info("AI accepted contrarian suggestion")
else:
    logger.info(
        "AI rejected contrarian suggestion",
        ai_chose=ai_decision.action,
        reasoning=ai_decision.reasoning
    )
```

## Risk Considerations

### Potential Issues

1. **False Signals:** RSI extremes don't always reverse immediately
   - **Mitigation:** AI makes final decision, can reject contrarian signal

2. **Timing:** Reversal might happen after market closes
   - **Mitigation:** 15-min markets give reasonable window for mean reversion

3. **Whipsaw:** Price could briefly reverse then continue original direction
   - **Mitigation:** All filters still active (signal lag, volume, regime)

4. **Overtrading:** Too many contrarian signals
   - **Mitigation:** Extremely conservative thresholds (RSI < 10 or > 90)

### Monitoring

- Track contrarian trade performance separately in database
- Weekly review of contrarian win rate vs baseline
- Adjust thresholds if false positive rate > 40%

## Example Scenario

**Market:** btc-updown-15m-1771186500
- **Price to beat:** $68,380.69
- **Current BTC:** $68,370.49 (DOWN $10)
- **RSI:** 9.5 (extremely oversold)
- **DOWN odds:** 72%

**Contrarian Detection:**
```
✅ RSI < 10: 9.5 qualifies
✅ DOWN odds > 65%: 72% qualifies
✅ Contrarian signal: OVERSOLD_REVERSAL (suggest UP)
✅ Movement threshold: Reduced to $50
✅ Movement: $10 < $50 ❌ Still fails threshold
```

**Note:** Even with contrarian signal, this market would still fail the $50 threshold. To catch it, we'd need to further reduce threshold to $10-$20 for extreme RSI (< 10), which could be a future enhancement.

**Alternative:** Could make threshold RSI-dependent:
- RSI 10: $50 threshold
- RSI 5: $25 threshold
- RSI 1: $10 threshold

## Future Enhancements

1. **RSI-Dependent Thresholds:** Lower movement requirement for more extreme RSI
2. **Time-Decay Confidence:** Reduce confidence as time passes without reversal
3. **Multi-Timeframe RSI:** Check 5-min and 15-min RSI alignment
4. **Volume Divergence:** Detect volume spikes during RSI extremes (exhaustion signals)
5. **Historical Win Rate:** Use past contrarian performance to adjust confidence

## Conclusion

This contrarian RSI strategy provides the AI with critical mean-reversion signals when technical extremes diverge from crowd sentiment. By maintaining all filters and giving AI final decision authority, we balance opportunity capture with risk management.

**Next Steps:**
1. Create git worktree for isolated development
2. Write detailed implementation plan
3. Implement with TDD approach
4. Deploy and monitor contrarian trade performance
