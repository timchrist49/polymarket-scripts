# polymarket/trading/contrarian.py
"""
Contrarian RSI Strategy
Detects extreme technical divergences from crowd consensus.
"""
from typing import Optional
from polymarket.models import ContrarianSignal

def detect_contrarian_setup(
    rsi: float,
    yes_odds: float,  # UP odds (best_bid)
    no_odds: float    # DOWN odds (1 - best_bid)
) -> Optional[ContrarianSignal]:
    """
    Detect extreme RSI divergence from crowd consensus.

    Args:
        rsi: RSI indicator value (0-100)
        yes_odds: UP token odds (0-1)
        no_odds: DOWN token odds (0-1)

    Returns:
        ContrarianSignal if conditions met, None otherwise

    Detection Rules:
        - OVERSOLD: RSI < 10 AND DOWN odds > 65%
        - OVERBOUGHT: RSI > 90 AND UP odds > 65%
    """
    # OVERSOLD: RSI extremely low, crowd betting DOWN
    if rsi < 10 and no_odds > 0.65:
        # Higher confidence for more extreme RSI
        confidence = 0.90 + (10 - rsi) * 0.01
        confidence = min(confidence, 1.0)  # Cap at 1.0

        return ContrarianSignal(
            type="OVERSOLD_REVERSAL",
            suggested_direction="UP",
            rsi=rsi,
            crowd_direction="DOWN",
            crowd_confidence=no_odds,
            confidence=confidence,
            reasoning=f"Extreme oversold (RSI {rsi:.1f}) + strong DOWN consensus ({no_odds:.0%}) = UP reversal likely"
        )

    # OVERBOUGHT: RSI extremely high, crowd betting UP
    if rsi > 90 and yes_odds > 0.65:
        # Higher confidence for more extreme RSI
        confidence = 0.90 + (rsi - 90) * 0.01
        confidence = min(confidence, 1.0)  # Cap at 1.0

        return ContrarianSignal(
            type="OVERBOUGHT_REVERSAL",
            suggested_direction="DOWN",
            rsi=rsi,
            crowd_direction="UP",
            crowd_confidence=yes_odds,
            confidence=confidence,
            reasoning=f"Extreme overbought (RSI {rsi:.1f}) + strong UP consensus ({yes_odds:.0%}) = DOWN reversal likely"
        )

    return None
