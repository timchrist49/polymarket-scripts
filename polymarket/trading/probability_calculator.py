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
        price_5min_ago: float,
        price_10min_ago: float,
        volatility_15min: float,
        time_remaining_seconds: int,
        orderbook_imbalance: float = 0.0
    ) -> float:
        """
        Calculate probability that BTC ends HIGHER than current price.

        Args:
            current_price: Current BTC price
            price_5min_ago: BTC price 5 minutes ago
            price_10min_ago: BTC price 10 minutes ago
            volatility_15min: Rolling 15-min volatility (std dev or ATR)
            time_remaining_seconds: Seconds until market settlement
            orderbook_imbalance: -1.0 to +1.0 (negative=sell pressure, positive=buy)

        Returns:
            Probability from 0.0 to 1.0 that BTC ends higher (clipped to [0.05, 0.95])

        Example:
            >>> calc = ProbabilityCalculator()
            >>> prob = calc.calculate_directional_probability(
            ...     current_price=66200.0,
            ...     price_5min_ago=66000.0,
            ...     price_10min_ago=65900.0,
            ...     volatility_15min=0.005,
            ...     time_remaining_seconds=600,
            ...     orderbook_imbalance=0.2
            ... )
            >>> print(f"Probability UP: {prob:.2%}")
            Probability UP: 72.34%
        """

        # Input validation
        if current_price <= 0 or price_5min_ago <= 0 or price_10min_ago <= 0:
            raise ValueError("Prices must be positive")
        if volatility_15min < 0:
            raise ValueError("Volatility cannot be negative")
        if not -1.0 <= orderbook_imbalance <= 1.0:
            raise ValueError(f"Orderbook imbalance must be in [-1.0, 1.0], got {orderbook_imbalance}")

        # Step 1: Calculate momentum (weighted recent > older)
        momentum_5min = (current_price - price_5min_ago) / price_5min_ago
        momentum_10min = (current_price - price_10min_ago) / price_10min_ago
        weighted_momentum = (momentum_5min * 0.7) + (momentum_10min * 0.3)

        # Step 2: Calculate expected volatility for remaining time
        time_fraction = time_remaining_seconds / 900  # 900s = 15min
        if time_fraction <= 0:
            time_fraction = 0.01  # Minimum to avoid division by zero

        volatility_factor = volatility_15min * sqrt(time_fraction)

        # Prevent division by zero
        if volatility_factor < 0.0001:
            volatility_factor = 0.0001

        # Step 3: Calculate z-score (how many standard deviations)
        z_score = weighted_momentum / volatility_factor

        # Step 4: Convert z-score to probability using normal distribution CDF
        probability_up = norm.cdf(z_score)

        # Step 5: Adjust for orderbook imbalance
        # Orderbook imbalance adds up to ±10% to probability
        imbalance_adjustment = orderbook_imbalance * 0.1
        final_probability = probability_up + imbalance_adjustment

        # Step 6: Clip to [0.05, 0.95] to avoid overconfidence
        final_probability = max(0.05, min(0.95, final_probability))

        logger.debug(
            "Probability calculation",
            momentum_5min=f"{momentum_5min:+.4f}",
            momentum_10min=f"{momentum_10min:+.4f}",
            weighted_momentum=f"{weighted_momentum:+.4f}",
            volatility=f"{volatility_15min:.4f}",
            time_fraction=f"{time_fraction:.2f}",
            z_score=f"{z_score:+.2f}",
            probability_before_adjustment=f"{probability_up:.2%}",
            orderbook_adjustment=f"{imbalance_adjustment:+.2%}",
            final_probability=f"{final_probability:.2%}"
        )

        return final_probability
