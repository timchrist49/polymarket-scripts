"""
Signal Conflict Detector

Detects and classifies conflicts between trading signals.
Applies confidence penalties or forces HOLD based on severity.
"""

from enum import Enum
from dataclasses import dataclass


class ConflictSeverity(Enum):
    """Conflict severity levels."""
    NONE = "NONE"
    MINOR = "MINOR"      # 1 conflict: -0.10 penalty
    MODERATE = "MODERATE"  # 2 conflicts: -0.20 penalty
    SEVERE = "SEVERE"    # 3+ conflicts OR timeframes CONFLICTING: AUTO-HOLD


@dataclass
class ConflictAnalysis:
    """Result of conflict analysis."""
    severity: ConflictSeverity
    confidence_penalty: float
    should_hold: bool
    conflicts_detected: list[str]


class SignalConflictDetector:
    """Detects and classifies conflicts between trading signals."""

    def analyze_conflicts(
        self,
        btc_direction: str,
        technical_trend: str,
        sentiment_direction: str,
        regime_trend: str | None,
        timeframe_alignment: str | None,
        market_signals_direction: str | None,
        market_signals_confidence: float | None
    ) -> ConflictAnalysis:
        """
        Analyze all signals for conflicts and classify severity.

        Args:
            btc_direction: "UP" or "DOWN" (actual BTC movement)
            technical_trend: "BULLISH", "BEARISH", "NEUTRAL"
            sentiment_direction: "BULLISH", "BEARISH", "NEUTRAL"
            regime_trend: "TRENDING UP", "TRENDING DOWN", "RANGING", etc.
            timeframe_alignment: "ALL_ALIGNED", "MOSTLY_ALIGNED", "MIXED", "CONFLICTING"
            market_signals_direction: "bullish", "bearish", "neutral"
            market_signals_confidence: 0.0-1.0

        Returns:
            ConflictAnalysis with severity, penalty, and conflict descriptions
        """
        conflicts = []

        # Map directions to UP/DOWN for comparison
        technical_dir = self._map_to_direction(technical_trend)
        sentiment_dir = self._map_to_direction(sentiment_direction)
        regime_dir = self._map_to_direction(regime_trend) if regime_trend else None
        market_dir = self._map_to_direction(market_signals_direction) if market_signals_direction else None

        # Check conflicts (only flag if direction disagrees with actual BTC movement)
        if technical_dir and technical_dir != btc_direction:
            conflicts.append(f"Technical ({technical_trend}) vs BTC actual ({btc_direction})")

        if sentiment_dir and sentiment_dir != btc_direction:
            conflicts.append(f"Sentiment ({sentiment_direction}) vs BTC actual ({btc_direction})")

        if regime_dir and regime_dir != btc_direction:
            conflicts.append(f"Regime ({regime_trend}) vs BTC actual ({btc_direction})")

        # Only flag market signals if confident (> 0.6)
        if market_dir and market_signals_confidence and market_signals_confidence > 0.6:
            if market_dir != btc_direction:
                conflicts.append(
                    f"Market Signals ({market_signals_direction}, {market_signals_confidence:.2f}) "
                    f"vs BTC actual ({btc_direction})"
                )

        # Timeframe conflict is special (always triggers SEVERE)
        if timeframe_alignment == "CONFLICTING":
            conflicts.append("Timeframes CONFLICTING (don't trade against larger trend)")

        # Classify severity
        severity = self._classify_severity(len(conflicts), timeframe_alignment)

        # Determine action
        if severity == ConflictSeverity.SEVERE:
            return ConflictAnalysis(
                severity=severity,
                confidence_penalty=0.0,  # No penalty, just HOLD
                should_hold=True,
                conflicts_detected=conflicts
            )
        elif severity == ConflictSeverity.MODERATE:
            return ConflictAnalysis(
                severity=severity,
                confidence_penalty=-0.20,
                should_hold=False,
                conflicts_detected=conflicts
            )
        elif severity == ConflictSeverity.MINOR:
            return ConflictAnalysis(
                severity=severity,
                confidence_penalty=-0.10,
                should_hold=False,
                conflicts_detected=conflicts
            )
        else:
            return ConflictAnalysis(
                severity=ConflictSeverity.NONE,
                confidence_penalty=0.0,
                should_hold=False,
                conflicts_detected=[]
            )

    def _map_to_direction(self, signal: str | None) -> str | None:
        """
        Map signal to UP/DOWN direction.

        Args:
            signal: Signal string (case-insensitive)

        Returns:
            "UP", "DOWN", or None (for NEUTRAL/RANGING/unclear)
        """
        if not signal:
            return None

        signal = signal.upper()

        if "BULL" in signal or signal == "UP" or "TRENDING UP" in signal:
            return "UP"
        elif "BEAR" in signal or signal == "DOWN" or "TRENDING DOWN" in signal:
            return "DOWN"
        else:
            # NEUTRAL, RANGING, MIXED, etc. -> no clear direction
            return None

    def _classify_severity(
        self,
        num_conflicts: int,
        timeframe_alignment: str | None
    ) -> ConflictSeverity:
        """
        Classify conflict severity based on number and type.

        Rules:
        - 3+ conflicts OR timeframes CONFLICTING -> SEVERE
        - 2 conflicts -> MODERATE
        - 1 conflict -> MINOR
        - 0 conflicts -> NONE
        """
        if num_conflicts >= 3 or timeframe_alignment == "CONFLICTING":
            return ConflictSeverity.SEVERE
        elif num_conflicts == 2:
            return ConflictSeverity.MODERATE
        elif num_conflicts == 1:
            return ConflictSeverity.MINOR
        else:
            return ConflictSeverity.NONE
