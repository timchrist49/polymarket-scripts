"""
Technical Analysis Module

Calculates technical indicators from price history including:
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- EMA (Exponential Moving Average)
- Volume analysis
- Price velocity
"""

from datetime import datetime
from decimal import Decimal
from typing import Literal
import structlog

try:
    import pandas as pd
    import numpy as np
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

from polymarket.models import (
    PricePoint,
    TechnicalIndicators
)

logger = structlog.get_logger()


class TechnicalAnalysis:
    """Technical indicator calculations."""

    @staticmethod
    def calculate_indicators(price_history: list[PricePoint]) -> TechnicalIndicators:
        """Calculate all technical indicators from price history."""
        if not price_history:
            raise ValueError("Price history is empty")

        if len(price_history) < 26:
            logger.warning("Insufficient data for all indicators", points=len(price_history))

        # Convert to DataFrame if available, otherwise manual calculation
        if HAS_PANDAS:
            return TechnicalAnalysis._calculate_with_pandas(price_history)
        else:
            return TechnicalAnalysis._calculate_manual(price_history)

    @staticmethod
    def _calculate_with_pandas(price_history: list[PricePoint]) -> TechnicalIndicators:
        """Calculate indicators using pandas/numpy (faster)."""
        # Create DataFrame
        df = pd.DataFrame([
            {
                "price": float(p.price),
                "volume": float(p.volume),
                "timestamp": p.timestamp
            }
            for p in price_history
        ])

        # RSI (14-period)
        rsi = TechnicalAnalysis._calculate_rsi(df["price"], 14)

        # EMAs
        ema_short = df["price"].ewm(span=9, adjust=False).mean().iloc[-1]
        ema_long = df["price"].ewm(span=21, adjust=False).mean().iloc[-1]
        sma_50 = df["price"].rolling(window=50).mean().iloc[-1] if len(df) >= 50 else ema_long

        # MACD (12, 26, 9)
        macd_12 = df["price"].ewm(span=12, adjust=False).mean()
        macd_26 = df["price"].ewm(span=26, adjust=False).mean()
        macd_line = macd_12 - macd_26
        macd_signal = macd_line.ewm(span=9, adjust=False).mean()
        macd_histogram = macd_line - macd_signal

        # Volume change
        if len(df) >= 30:
            recent_vol = df["volume"].tail(5).mean()
            avg_vol = df["volume"].tail(30).mean()
            volume_change = ((recent_vol - avg_vol) / avg_vol) * 100
        else:
            volume_change = 0.0

        # Price velocity ($/min)
        if len(df) >= 5:
            price_change = df["price"].iloc[-1] - df["price"].iloc[-5]
            velocity = price_change / 5
        else:
            velocity = 0.0

        # Determine trend
        trend: Literal["BULLISH", "BEARISH", "NEUTRAL"]
        if ema_short > ema_long and macd_histogram.iloc[-1] > 0:
            trend = "BULLISH"
        elif ema_short < ema_long and macd_histogram.iloc[-1] < 0:
            trend = "BEARISH"
        else:
            trend = "NEUTRAL"

        return TechnicalIndicators(
            rsi=float(rsi),
            macd_value=float(macd_line.iloc[-1]),
            macd_signal=float(macd_signal.iloc[-1]),
            macd_histogram=float(macd_histogram.iloc[-1]),
            ema_short=float(ema_short),
            ema_long=float(ema_long),
            sma_50=float(sma_50) if not pd.isna(sma_50) else float(ema_long),
            volume_change=volume_change,
            price_velocity=velocity,
            trend=trend
        )

    @staticmethod
    def _calculate_manual(price_history: list[PricePoint]) -> TechnicalIndicators:
        """Fallback calculation without pandas (slower but works)."""
        prices = [float(p.price) for p in price_history]
        volumes = [float(p.volume) for p in price_history]

        # Simple EMA calculation
        def ema(data: list[float], span: int) -> float:
            multiplier = 2 / (span + 1)
            ema_val = data[0]
            for price in data[1:]:
                ema_val = (price * multiplier) + (ema_val * (1 - multiplier))
            return ema_val

        # Simple RSI
        def rsi(prices: list[float], period: int) -> float:
            if len(prices) < period + 1:
                return 50.0

            gains = []
            losses = []
            for i in range(1, len(prices)):
                change = prices[i] - prices[i-1]
                if change > 0:
                    gains.append(change)
                    losses.append(0)
                else:
                    gains.append(0)
                    losses.append(abs(change))

            avg_gain = sum(gains[-period:]) / period
            avg_loss = sum(losses[-period:]) / period

            if avg_loss == 0:
                return 100.0

            rs = avg_gain / avg_loss
            return 100 - (100 / (1 + rs))

        ema_short = ema(prices, 9) if len(prices) >= 9 else prices[-1]
        ema_long = ema(prices, 21) if len(prices) >= 21 else prices[-1]
        rsi_val = rsi(prices, 14) if len(prices) >= 15 else 50.0

        # Simple trend determination
        if ema_short > ema_long:
            trend = "BULLISH"
        elif ema_short < ema_long:
            trend = "BEARISH"
        else:
            trend = "NEUTRAL"

        return TechnicalIndicators(
            rsi=rsi_val,
            macd_value=0.0,
            macd_signal=0.0,
            macd_histogram=0.0,
            ema_short=ema_short,
            ema_long=ema_long,
            sma_50=ema_long,
            volume_change=0.0,
            price_velocity=0.0,
            trend=trend
        )

    @staticmethod
    def _calculate_rsi(prices: pd.Series, period: int) -> float:
        """Calculate RSI using pandas."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1]
