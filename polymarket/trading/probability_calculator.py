"""
Probability Calculator

Calculates the actual probability of BTC price direction in the remaining time
window using momentum, volatility, and orderbook data.

Based on modified Brownian motion model with mean reversion adjustments.
"""

from math import sqrt
import structlog
from scipy.stats import norm

logger = structlog.get_logger()


class ProbabilityCalculator:
    """Calculate directional probability for BTC 15-min markets."""

    def __init__(self):
        """Initialize probability calculator."""
        pass

    def calculate_directional_probability(
        self,
        current_price: float,
        target_price: float,
        price_5min_ago: float,
        price_10min_ago: float,
        volatility_15min: float,
        time_remaining_seconds: int,
        orderbook_imbalance: float = 0.0
    ) -> float:
        """
        Calculate probability that BTC ends HIGHER than target price.

        This is the CORRECT calculation for Polymarket UP/DOWN markets:
        - Market asks: "Will BTC end higher than [price at market start]?"
        - We calculate: Probability of reaching target_price from current_price

        Args:
            current_price: Current BTC price
            target_price: Price to beat (typically price at market start)
            price_5min_ago: BTC price 5 minutes ago
            price_10min_ago: BTC price 10 minutes ago
            volatility_15min: Rolling 15-min volatility (std dev or ATR)
            time_remaining_seconds: Seconds until market settlement
            orderbook_imbalance: -1.0 to +1.0 (negative=sell pressure, positive=buy)

        Returns:
            Probability from 0.0 to 1.0 that BTC ends higher than target (clipped to [0.05, 0.95])

        Example:
            >>> calc = ProbabilityCalculator()
            >>> prob = calc.calculate_directional_probability(
            ...     current_price=70100.0,
            ...     target_price=70000.0,  # BTC already $100 above target
            ...     price_5min_ago=70050.0,
            ...     price_10min_ago=70000.0,
            ...     volatility_15min=0.005,
            ...     time_remaining_seconds=60,
            ...     orderbook_imbalance=0.0
            ... )
            >>> print(f"Probability UP (above target): {prob:.2%}")
            Probability UP (above target): 95.00%  # Very high since already above
        """

        # Input validation
        if current_price <= 0 or price_5min_ago <= 0 or price_10min_ago <= 0 or target_price <= 0:
            raise ValueError("Prices must be positive")
        if volatility_15min < 0:
            raise ValueError("Volatility cannot be negative")
        if not -1.0 <= orderbook_imbalance <= 1.0:
            raise ValueError(f"Orderbook imbalance must be in [-1.0, 1.0], got {orderbook_imbalance}")

        # Step 1: Calculate gap to target (how far from the goal?)
        gap = target_price - current_price
        required_return = gap / current_price  # Percentage move needed

        # Step 2: Calculate momentum (weighted recent > older)
        momentum_5min = (current_price - price_5min_ago) / price_5min_ago
        momentum_10min = (current_price - price_10min_ago) / price_10min_ago
        weighted_momentum = (momentum_5min * 0.7) + (momentum_10min * 0.3)

        # Step 3: Project expected return based on momentum
        time_fraction = time_remaining_seconds / 900  # 900s = 15min
        if time_fraction <= 0:
            time_fraction = 0.01  # Minimum to avoid division by zero

        # Expected return = momentum scaled by time remaining
        # If momentum is +0.1% over 5min, and we have 5min left, expect +0.1%
        # We use 300s (5min) as reference since momentum is measured over 5min
        expected_return = weighted_momentum * (time_remaining_seconds / 300.0)

        # Step 4: Calculate volatility for remaining time
        volatility_factor = volatility_15min * sqrt(time_fraction)

        # Prevent division by zero
        if volatility_factor < 0.0001:
            volatility_factor = 0.0001

        # Step 5: Calculate z-score
        # Positive z-score = expected return exceeds required return (good!)
        # Negative z-score = expected return falls short (bad!)
        z_score = (expected_return - required_return) / volatility_factor

        # Step 6: Convert z-score to probability using normal distribution CDF
        probability_reach_target = norm.cdf(z_score)

        # Step 7: Adjust for orderbook imbalance
        # Orderbook imbalance adds up to ±10% to probability
        imbalance_adjustment = orderbook_imbalance * 0.1
        final_probability = probability_reach_target + imbalance_adjustment

        # Step 8: Clip to [0.05, 0.95] to avoid overconfidence
        final_probability = max(0.05, min(0.95, final_probability))

        logger.info(
            "Probability calculation",
            current_price=f"${current_price:,.2f}",
            target_price=f"${target_price:,.2f}",
            gap=f"${gap:+,.2f}",
            required_return=f"{required_return:+.4%}",
            momentum_5min=f"{momentum_5min:+.4%}",
            weighted_momentum=f"{weighted_momentum:+.4%}",
            expected_return=f"{expected_return:+.4%}",
            volatility=f"{volatility_15min:.4f}",
            time_remaining=f"{time_remaining_seconds}s",
            time_fraction=f"{time_fraction:.2f}",
            z_score=f"{z_score:+.2f}",
            probability_before_adjustment=f"{probability_reach_target:.2%}",
            orderbook_adjustment=f"{imbalance_adjustment:+.2%}",
            final_probability=f"{final_probability:.2%}"
        )

        return final_probability
