"""Parameter adjustment and validation system."""

from enum import Enum
from typing import Tuple, Dict, Optional, TYPE_CHECKING
import subprocess
import structlog
from pathlib import Path

from polymarket.config import Settings

if TYPE_CHECKING:
    from polymarket.performance.database import PerformanceDatabase
    from polymarket.telegram.bot import TelegramBot

logger = structlog.get_logger()


def get_repo_root() -> Path:
    """
    Get git repository root (works in worktrees, submodules, etc).

    Uses --git-common-dir to find the shared .git directory, then navigates
    to parent to get the main repository root (not worktree root).
    """
    try:
        # Get common git directory (shared across worktrees)
        git_common_dir = subprocess.check_output(
            ['git', 'rev-parse', '--git-common-dir'],
            text=True,
            stderr=subprocess.DEVNULL
        ).strip()
        # Navigate up from .git to repository root
        return Path(git_common_dir).parent
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Fallback to relative path if git command fails
        return Path(__file__).parent.parent.parent


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

    def __init__(
        self,
        settings: Settings,
        db: Optional['PerformanceDatabase'] = None,
        telegram: Optional['TelegramBot'] = None
    ):
        """
        Initialize parameter adjuster.

        Args:
            settings: Current bot settings
            db: Performance database for logging
            telegram: Telegram bot for notifications
        """
        self.settings = settings
        self.bounds = ParameterBounds()
        self.db = db
        self.telegram = telegram

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

    async def apply_adjustment(
        self,
        parameter_name: str,
        old_value: float,
        new_value: float,
        reason: str,
        tier: AdjustmentTier
    ) -> bool:
        """
        Apply a parameter adjustment.

        Args:
            parameter_name: Name of parameter to adjust
            old_value: Current value
            new_value: Proposed new value
            reason: Reasoning for adjustment
            tier: Tier classification

        Returns:
            True if applied, False if rejected
        """
        # Validate bounds
        if not self.validate_adjustment(parameter_name, new_value):
            logger.warning(
                "Adjustment rejected - out of bounds",
                parameter=parameter_name,
                new_value=new_value
            )
            return False

        # Handle by tier
        if tier == AdjustmentTier.TIER_1_AUTO:
            approval_method = "tier_1_auto"
        elif tier == AdjustmentTier.TIER_2_APPROVAL:
            # Request approval via Telegram
            if not self.telegram:
                logger.warning("Telegram not configured, rejecting Tier 2 adjustment")
                return False

            change_pct = self.calculate_change_percent(old_value, new_value)
            approved = await self.telegram.request_approval(
                parameter_name=parameter_name,
                old_value=old_value,
                new_value=new_value,
                reason=reason,
                change_pct=change_pct,
                timeout_hours=4
            )

            if not approved:
                logger.info("Tier 2 adjustment rejected", parameter=parameter_name)
                return False

            approval_method = "tier_2_approved"
        else:
            # Tier 3 - Emergency pause
            change_pct = self.calculate_change_percent(old_value, new_value)

            logger.critical(
                "Emergency pause triggered - dangerous adjustment",
                parameter=parameter_name,
                old_value=old_value,
                new_value=new_value,
                change_percent=change_pct
            )

            # Write emergency pause flag to file
            self._set_emergency_pause()

            # Send urgent alert
            if self.telegram:
                await self.telegram.send_emergency_alert(
                    parameter_name=parameter_name,
                    old_value=old_value,
                    new_value=new_value,
                    reason=reason,
                    change_pct=change_pct
                )

            # Log to database with emergency flag
            if self.db:
                self._log_adjustment(
                    parameter_name=parameter_name,
                    old_value=old_value,
                    new_value=new_value,
                    reason=f"EMERGENCY PAUSE: {reason}",
                    approval_method="tier_3_emergency_pause"
                )

            logger.critical("Emergency pause activated - bot should stop trading")
            return False

        # Apply adjustment
        setattr(self.settings, parameter_name, new_value)

        # Log to database
        if self.db:
            self._log_adjustment(
                parameter_name=parameter_name,
                old_value=old_value,
                new_value=new_value,
                reason=reason,
                approval_method=approval_method
            )

        # Notify via Telegram
        if self.telegram:
            await self._notify_adjustment(
                parameter_name=parameter_name,
                old_value=old_value,
                new_value=new_value,
                reason=reason
            )

        logger.info(
            "Parameter adjusted automatically",
            parameter=parameter_name,
            old_value=old_value,
            new_value=new_value,
            reason=reason
        )

        return True

    def _log_adjustment(
        self,
        parameter_name: str,
        old_value: float,
        new_value: float,
        reason: str,
        approval_method: str
    ):
        """Log adjustment to database."""
        from datetime import datetime

        cursor = self.db.conn.cursor()
        cursor.execute("""
            INSERT INTO parameter_history
            (timestamp, parameter_name, old_value, new_value, reason, approval_method)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (datetime.now(), parameter_name, old_value, new_value, reason, approval_method))

        self.db.conn.commit()

    async def _notify_adjustment(
        self,
        parameter_name: str,
        old_value: float,
        new_value: float,
        reason: str
    ):
        """Send Telegram notification about adjustment."""
        change_pct = self.calculate_change_percent(old_value, new_value)

        message = f"""⚙️ **Parameter Auto-Adjusted** (Tier 1)

Parameter: `{parameter_name}`
Old Value: {old_value:.4f}
New Value: {new_value:.4f}
Change: {change_pct:+.1f}%

Reason: {reason}

✅ Applied automatically (within ±5% threshold)
"""

        try:
            await self.telegram._send_message(message)
        except Exception as e:
            logger.error("Failed to send adjustment notification", error=str(e))

    def _set_emergency_pause(self):
        """Write emergency pause flag to file for trading bot to check."""
        # Write to .emergency_pause file in repository root
        pause_file = get_repo_root() / ".emergency_pause"

        try:
            pause_file.write_text("EMERGENCY_PAUSE_ACTIVE\n")
            logger.critical("Emergency pause flag written", file=str(pause_file))
        except Exception as e:
            logger.error("Failed to write emergency pause flag", error=str(e))
