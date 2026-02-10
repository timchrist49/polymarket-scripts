# polymarket/models.py
"""
Data models for Polymarket API requests and responses.

This module defines Pydantic models for type-safe API interactions including
order requests, market data, portfolio summaries, and error responses.

Classes:
    OrderRequest: Request model for placing orders
    PortfolioSummary: Portfolio status summary
    MarketInfo: Market information model

Example:
    >>> from polymarket.models import OrderRequest
    >>> request = OrderRequest(
    ...     token_id="0x...",
    ...     side="BUY",
    ...     price=0.55,
    ...     size=10
    ... )
"""

from datetime import datetime
from typing import Literal
from dataclasses import dataclass
from decimal import Decimal
from pydantic import BaseModel, Field, field_validator, ConfigDict
import json


class Market(BaseModel):
    """Polymarket market data from Gamma API."""

    # Required fields
    id: str
    condition_id: str = Field(alias="conditionId")

    # Market info
    question: str | None = None
    slug: str | None = None
    description: str | None = None

    # Status flags
    active: bool | None = None
    closed: bool | None = None
    accepting_orders: bool | None = Field(alias="acceptingOrders", default=None)

    # Timing
    end_date: datetime | None = Field(None, alias="endDate")
    start_date: datetime | None = Field(None, alias="startDate")

    # Outcomes
    outcomes: str | list[str] | None = None
    outcome_prices: str | None = Field(None, alias="outcomePrices")

    # CLOB token IDs (JSON string in API)
    clob_token_ids: str | None = Field(None, alias="clobTokenIds")

    # Trading constraints
    order_price_min_tick_size: float | None = Field(None, alias="orderPriceMinTickSize")
    order_min_size: float | None = Field(None, alias="orderMinSize")

    # Market data
    best_bid: float | None = Field(None, alias="bestBid")
    best_ask: float | None = Field(None, alias="bestAsk")
    last_trade_price: float | None = Field(None, alias="lastTradePrice")

    # Volume and liquidity
    volume_num: float | None = Field(None, alias="volumeNum")
    volume24hr: float | None = Field(None, alias="volume24hr")
    liquidity_num: float | None = Field(None, alias="liquidityNum")

    # Category info
    category: str | None = None

    model_config = ConfigDict(populate_by_name=True)  # Allow both alias and original field names

    def get_token_ids(self) -> list[str]:
        """Parse clobTokenIds JSON string into list."""
        if not self.clob_token_ids:
            return []
        try:
            return json.loads(self.clob_token_ids)
        except (json.JSONDecodeError, TypeError):
            return []

    def is_tradeable(self) -> bool:
        """Check if market is accepting orders."""
        return bool(
            self.active is True
            and self.closed is False
            and self.accepting_orders is True
        )


class TokenInfo(BaseModel):
    """Information about a tradeable token."""

    token_id: str
    outcome: str  # e.g., "Yes" or "No"
    index: int  # 0 for Yes, 1 for No in binary markets


class OrderRequest(BaseModel):
    """Request to place an order."""

    token_id: str
    side: Literal["BUY", "SELL"]
    price: float = Field(ge=0.0, le=1.0, description="Price from 0 to 1")
    size: float = Field(gt=0, description="Order size in shares")
    order_type: Literal["limit", "market"] = "market"  # Default to market for immediate execution

    @field_validator("side", mode="before")
    @classmethod
    def normalize_side(cls, v: str) -> str:
        """Normalize side to uppercase."""
        if isinstance(v, str):
            v = v.upper()
        if v not in ("BUY", "SELL"):
            raise ValueError(f"Invalid side: {v}. Must be BUY or SELL")
        return v

    @field_validator("order_type", mode="before")
    @classmethod
    def normalize_order_type(cls, v: str) -> str:
        """Normalize order_type to lowercase."""
        if isinstance(v, str):
            v = v.lower()
        if v not in ("limit", "market"):
            raise ValueError(f"Invalid order_type: {v}. Must be limit or market")
        return v


class OrderResponse(BaseModel):
    """Response from placing an order."""

    order_id: str
    status: str
    accepted: bool
    raw_response: dict
    error_message: str | None = None


class PortfolioSummary(BaseModel):
    """Summary of portfolio status."""

    open_orders: list[dict]
    total_notional: float
    positions: dict[str, float]  # token_id -> quantity
    total_exposure: float
    trades: list[dict] = []  # Trade history for position tracking
    usdc_balance: float = 0.0  # Available USDC for trading (CLOB balance)
    positions_value: float = 0.0  # Total value of all positions at current prices
    total_value: float = 0.0  # Total portfolio value (cash + positions)


class BalanceInfo(BaseModel):
    """Token balance information."""

    token_id: str
    balance: float
    allowance: float | None = None


# === BTC Price Models ===

@dataclass
class BTCPriceData:
    """Current BTC price data."""
    price: Decimal
    timestamp: datetime
    source: str
    volume_24h: Decimal


@dataclass
class PricePoint:
    """Historical price point for technical analysis."""
    price: Decimal
    volume: Decimal
    timestamp: datetime


@dataclass
class PriceChange:
    """Price change over a time window."""
    current_price: Decimal
    change_percent: float
    change_amount: Decimal
    velocity: Decimal  # $/minute


# === Sentiment Models ===

@dataclass
class SentimentAnalysis:
    """Market sentiment analysis from Tavily."""
    score: float           # -1.0 (bearish) to +1.0 (bullish)
    confidence: float      # 0.0 to 1.0
    key_factors: list[str]
    sources_analyzed: int
    timestamp: datetime


# === Technical Analysis Models ===

@dataclass
class TechnicalIndicators:
    """Technical analysis indicators."""
    rsi: float
    macd_value: float
    macd_signal: float
    macd_histogram: float
    ema_short: float
    ema_long: float
    sma_50: float
    volume_change: float
    price_velocity: float
    trend: Literal["BULLISH", "BEARISH", "NEUTRAL"]


# === Trading Decision Models ===

@dataclass
class TradingDecision:
    """AI-generated trading decision."""
    action: Literal["YES", "NO", "HOLD"]
    confidence: float
    reasoning: str
    token_id: str
    position_size: Decimal
    stop_loss_threshold: float


@dataclass
class ValidationResult:
    """Risk validation result."""
    approved: bool
    reason: str
    adjusted_position: Decimal | None


# === New Sentiment Models ===

@dataclass
class SocialSentiment:
    """Social sentiment from crypto-specific APIs."""
    score: float                      # -1.0 (bearish) to +1.0 (bullish)
    confidence: float                 # 0.0 to 1.0
    fear_greed: int                   # 0-100 from alternative.me
    is_trending: bool                 # BTC in top 3 trending
    vote_up_pct: float                # CoinGecko sentiment votes up %
    vote_down_pct: float              # CoinGecko sentiment votes down %
    signal_type: str                  # "STRONG_BULLISH", "WEAK_BEARISH", etc.
    sources_available: list[str]      # Which APIs succeeded
    timestamp: datetime


@dataclass
class MarketSignals:
    """Market microstructure signals from Binance."""
    score: float                      # -1.0 (bearish) to +1.0 (bullish)
    confidence: float                 # 0.0 to 1.0
    order_book_score: float           # Bid vs ask wall strength
    whale_score: float                # Large buy vs sell orders
    volume_score: float               # Volume spike vs average
    momentum_score: float             # Price velocity
    order_book_bias: str              # "BID_HEAVY", "ASK_HEAVY", "BALANCED"
    whale_direction: str              # "BUYING", "SELLING", "NEUTRAL"
    whale_count: int                  # Number of large orders
    volume_ratio: float               # Current volume / 24h average
    momentum_direction: str           # "UP", "DOWN", "FLAT"
    signal_type: str                  # "STRONG_BULLISH", etc.
    timestamp: datetime


@dataclass
class AggregatedSentiment:
    """Final aggregated sentiment with agreement-based confidence."""
    social: SocialSentiment
    market: MarketSignals
    final_score: float                # Weighted: market 60% + social 40%
    final_confidence: float           # Base confidence * agreement multiplier
    agreement_multiplier: float       # 0.5 (conflict) to 1.5 (perfect agreement)
    signal_type: str                  # "STRONG_BULLISH", "CONFLICTED", etc.
    timestamp: datetime
