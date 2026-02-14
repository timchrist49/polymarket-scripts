"""
Signal Lag Detector

Detects when market sentiment lags behind actual BTC price movement.
This prevents trading on stale data when BTC moves faster than Polymarket odds update.
"""


def detect_signal_lag(
    btc_actual_direction: str,
    market_sentiment_direction: str,
    sentiment_confidence: float
) -> tuple[bool, str]:
    """
    Detect when market sentiment lags behind actual BTC movement.

    Args:
        btc_actual_direction: "UP" or "DOWN" (from price-to-beat comparison)
        market_sentiment_direction: "BULLISH", "BEARISH", or "NEUTRAL"
        sentiment_confidence: 0.0-1.0 confidence level

    Returns:
        Tuple of (is_lagging: bool, reason: str)

    Example:
        >>> detect_signal_lag("UP", "BEARISH", 0.75)
        (True, "SIGNAL LAG DETECTED: BTC moving UP but market sentiment is BEARISH...")
    """
    # Map sentiment to direction
    if market_sentiment_direction == "BULLISH":
        sentiment_dir = "UP"
    elif market_sentiment_direction == "BEARISH":
        sentiment_dir = "DOWN"
    else:
        # NEUTRAL maps to same direction (no contradiction)
        sentiment_dir = btc_actual_direction

    # Check for contradiction
    if btc_actual_direction != sentiment_dir:
        # Only flag if sentiment is confident (> 0.6)
        # Low confidence contradictions are just uncertain, not lag
        if sentiment_confidence > 0.6:
            reason = (
                f"SIGNAL LAG DETECTED: BTC moving {btc_actual_direction} "
                f"but market sentiment is {market_sentiment_direction} "
                f"(confidence: {sentiment_confidence:.2f}). "
                f"Market odds lagging behind reality."
            )
            return True, reason

    return False, "No lag detected"
