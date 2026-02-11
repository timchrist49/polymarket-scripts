"""Parameter adjustment and validation system."""

from enum import Enum
from typing import Tuple, Dict
import structlog

from polymarket.config import Settings

logger = structlog.get_logger()


class AdjustmentTier(Enum):
    """Tier classification for parameter adjustments."""
    TIER_1_AUTO = "tier_1_auto"  # ±5% - auto-approve
    TIER_2_APPROVAL = "tier_2_approval"  # 5-20% - requires approval
    TIER_3_PAUSE = "tier_3_pause"  # >20% - emergency pause


class ParameterBounds:
    """Defines safe bounds for each parameter."""

    BOUNDS: Dict[str, Tuple[float, float]] = {
        "bot_confidence_threshold": (0.50, 0.95),  # 50-95%
        "bot_max_position_dollars": (5.0, 50.0),   # $5-$50
        "bot_max_exposure_percent": (0.10, 0.80),  # 10-80%
    }

    def get_bounds(self, parameter_name: str) -> Tuple[float, float]:
        """Get min/max bounds for a parameter."""
        if parameter_name not in self.BOUNDS:
            raise ValueError(f"Unknown parameter: {parameter_name}")
        return self.BOUNDS[parameter_name]


class ParameterAdjuster:
    """Validates and classifies parameter adjustments."""

    def __init__(self, settings: Settings):
        """
        Initialize parameter adjuster.

        Args:
            settings: Current bot settings
        """
        self.settings = settings
        self.bounds = ParameterBounds()

    def validate_adjustment(self, parameter_name: str, new_value: float) -> bool:
        """
        Validate that new value is within safe bounds.

        Args:
            parameter_name: Name of parameter to adjust
            new_value: Proposed new value

        Returns:
            True if valid, False otherwise
        """
        try:
            min_val, max_val = self.bounds.get_bounds(parameter_name)

            if new_value < min_val or new_value > max_val:
                logger.warning(
                    "Parameter adjustment outside bounds",
                    parameter=parameter_name,
                    new_value=new_value,
                    min=min_val,
                    max=max_val
                )
                return False

            return True

        except ValueError as e:
            logger.error("Invalid parameter name", parameter=parameter_name, error=str(e))
            return False

    def classify_tier(
        self,
        parameter_name: str,
        current_value: float,
        new_value: float
    ) -> AdjustmentTier:
        """
        Classify adjustment into tier based on magnitude of change.

        Args:
            parameter_name: Name of parameter
            current_value: Current value
            new_value: Proposed new value

        Returns:
            AdjustmentTier classification
        """
        change_percent = abs(self.calculate_change_percent(current_value, new_value))

        if change_percent <= 5.0:
            return AdjustmentTier.TIER_1_AUTO
        elif change_percent <= 20.0:
            return AdjustmentTier.TIER_2_APPROVAL
        else:
            return AdjustmentTier.TIER_3_PAUSE

    def calculate_change_percent(self, current: float, new: float) -> float:
        """
        Calculate percentage change.

        Args:
            current: Current value
            new: New value

        Returns:
            Percentage change (positive = increase, negative = decrease)
        """
        if current == 0:
            return 0.0

        return ((new - current) / current) * 100.0
