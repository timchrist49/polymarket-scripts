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
    purchase_value: float = 0.0  # Original purchase cost of positions
    unrealized_pl: float = 0.0  # Current value - Purchase value


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


@dataclass
class VolumeData:
    """BTC trading volume data."""
    volume_24h: float           # 24-hour volume in USD
    volume_current_hour: float  # Current hour volume
    volume_avg_hour: float      # Average hourly volume (last 24h)
    volume_ratio: float         # Current / Average (spike detection)
    is_high_volume: bool        # volume_ratio > 1.5
    timestamp: datetime


@dataclass
class OrderbookData:
    """Polymarket orderbook depth analysis."""
    bid_ask_spread: float         # Spread in % (tight = liquid)
    spread_bps: float              # Spread in basis points
    liquidity_score: float         # 0.0-1.0 (high = good liquidity)
    order_imbalance: float         # -1.0 (ask heavy) to +1.0 (bid heavy)
    imbalance_direction: str       # "BUY_PRESSURE", "SELL_PRESSURE", "BALANCED"
    bid_depth_100bps: float        # Total bid liquidity within 100bps
    ask_depth_100bps: float        # Total ask liquidity within 100bps
    bid_depth_200bps: float        # Total bid liquidity within 200bps
    ask_depth_200bps: float        # Total ask liquidity within 200bps
    best_bid: float                # Top bid price
    best_ask: float                # Top ask price
    can_fill_order: bool           # Enough liquidity for trade
    timestamp: datetime


@dataclass
class MarketRegime:
    """Current market regime classification."""
    regime: str          # "TRENDING", "RANGING", "VOLATILE", "UNCLEAR"
    volatility: float    # ATR or price volatility %
    is_trending: bool    # True if strong directional move
    trend_direction: str # "UP", "DOWN", "SIDEWAYS"
    confidence: float    # 0.0-1.0
    timestamp: datetime


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
    score: float                      # -0.7 to +0.85 (asymmetric: trending is one-sided)
    confidence: float                 # 0.0 to 1.0
    fear_greed: int                   # 0-100 from alternative.me
    is_trending: bool                 # BTC in top 3 trending
    vote_up_pct: float                # CoinGecko sentiment votes up %
    vote_down_pct: float              # CoinGecko sentiment votes down %
    signal_type: str                  # "STRONG_BULLISH", "WEAK_BEARISH", etc.
    sources_available: list[str]      # Which APIs succeeded
    timestamp: datetime

    def validate(self) -> None:
        """Validate field constraints. Raises ValueError if invalid."""
        if not -1.0 <= self.score <= 1.0:
            raise ValueError(f"score must be in [-1.0, 1.0], got {self.score}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0.0, 1.0], got {self.confidence}")
        if not 0 <= self.fear_greed <= 100:
            raise ValueError(f"fear_greed must be in [0, 100], got {self.fear_greed}")
        if not 0.0 <= self.vote_up_pct <= 100.0:
            raise ValueError(f"vote_up_pct must be in [0.0, 100.0], got {self.vote_up_pct}")
        if not 0.0 <= self.vote_down_pct <= 100.0:
            raise ValueError(f"vote_down_pct must be in [0.0, 100.0], got {self.vote_down_pct}")


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

    def validate(self) -> None:
        """Validate field constraints. Raises ValueError if invalid."""
        if not -1.0 <= self.score <= 1.0:
            raise ValueError(f"score must be in [-1.0, 1.0], got {self.score}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0.0, 1.0], got {self.confidence}")
        if not -1.0 <= self.order_book_score <= 1.0:
            raise ValueError(f"order_book_score must be in [-1.0, 1.0], got {self.order_book_score}")
        if not -1.0 <= self.whale_score <= 1.0:
            raise ValueError(f"whale_score must be in [-1.0, 1.0], got {self.whale_score}")
        if not -1.0 <= self.volume_score <= 1.0:
            raise ValueError(f"volume_score must be in [-1.0, 1.0], got {self.volume_score}")
        if not -1.0 <= self.momentum_score <= 1.0:
            raise ValueError(f"momentum_score must be in [-1.0, 1.0], got {self.momentum_score}")
        if self.whale_count < 0:
            raise ValueError(f"whale_count must be >= 0, got {self.whale_count}")
        if self.volume_ratio < 0:
            raise ValueError(f"volume_ratio must be >= 0, got {self.volume_ratio}")


@dataclass
class FundingRateSignal:
    """Funding rate signals from perpetual futures markets."""
    score: float                      # -1.0 (oversold) to +1.0 (overheated)
    confidence: float                 # 0.0 to 1.0
    funding_rate: float              # Raw funding rate (positive = longs pay shorts)
    funding_rate_normalized: float   # Normalized to [-1, 1] range
    signal_type: str                 # "OVERHEATED", "NEUTRAL", "OVERSOLD"
    source: str                      # Exchange name (e.g., "binance")
    timestamp: datetime

    def validate(self) -> None:
        """Validate field constraints."""
        if not -1.0 <= self.score <= 1.0:
            raise ValueError(f"score must be in [-1.0, 1.0], got {self.score}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0.0, 1.0], got {self.confidence}")


@dataclass
class BTCDominanceSignal:
    """BTC dominance signals indicating capital flow."""
    score: float                      # -1.0 (alt season) to +1.0 (BTC season)
    confidence: float                 # 0.0 to 1.0
    dominance_pct: float             # Current BTC dominance %
    dominance_change_24h: float      # 24h change in dominance
    signal_type: str                 # "BTC_SEASON", "NEUTRAL", "ALT_SEASON"
    market_cap_btc: float           # BTC market cap in USD
    market_cap_total: float         # Total crypto market cap in USD
    timestamp: datetime

    def validate(self) -> None:
        """Validate field constraints."""
        if not -1.0 <= self.score <= 1.0:
            raise ValueError(f"score must be in [-1.0, 1.0], got {self.score}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0.0, 1.0], got {self.confidence}")
        if not 0.0 <= self.dominance_pct <= 100.0:
            raise ValueError(f"dominance_pct must be in [0.0, 100.0], got {self.dominance_pct}")


@dataclass
class AggregatedSentiment:
    """Final aggregated sentiment with agreement-based confidence."""
    social: SocialSentiment
    market: MarketSignals
    final_score: float                # Weighted: market 40% + social 20% + funding 20% + dominance 15% + orderbook 5%
    final_confidence: float           # Base confidence * agreement multiplier
    agreement_multiplier: float       # 0.5 (conflict) to 1.5 (perfect agreement)
    signal_type: str                  # "STRONG_BULLISH", "CONFLICTED", etc.
    timestamp: datetime
    funding: FundingRateSignal | None = None  # New: funding rate signals (optional)
    dominance: BTCDominanceSignal | None = None  # New: BTC dominance signals (optional)

    def validate(self) -> None:
        """Validate field constraints. Raises ValueError if invalid."""
        if not -1.0 <= self.final_score <= 1.0:
            raise ValueError(f"final_score must be in [-1.0, 1.0], got {self.final_score}")
        if not 0.0 <= self.final_confidence <= 1.0:
            raise ValueError(f"final_confidence must be in [0.0, 1.0], got {self.final_confidence}")
        if not 0.5 <= self.agreement_multiplier <= 1.5:
            raise ValueError(f"agreement_multiplier must be in [0.5, 1.5], got {self.agreement_multiplier}")
        # Validate nested objects
        self.social.validate()
        self.market.validate()
        if self.funding:
            self.funding.validate()
        if self.dominance:
            self.dominance.validate()


# === Arbitrage Models ===

@dataclass
class ArbitrageOpportunity:
    """Detected arbitrage opportunity from price feed lag."""

    market_id: str
    actual_probability: float  # Calculated from price momentum
    polymarket_yes_odds: float  # Current market odds
    polymarket_no_odds: float
    edge_percentage: float  # Size of mispricing
    recommended_action: Literal["BUY_YES", "BUY_NO", "HOLD"]
    confidence_boost: float  # Amount to boost AI confidence
    urgency: Literal["HIGH", "MEDIUM", "LOW"]
    expected_profit_pct: float  # Expected ROI if correct

    def __post_init__(self):
        """Validate field values."""
        if self.recommended_action not in ["BUY_YES", "BUY_NO", "HOLD"]:
            raise ValueError(f"Invalid action: {self.recommended_action}")
        if self.urgency not in ["HIGH", "MEDIUM", "LOW"]:
            raise ValueError(f"Invalid urgency: {self.urgency}")
        if not 0.0 <= self.actual_probability <= 1.0:
            raise ValueError(f"Invalid probability: {self.actual_probability}")
        if not 0.0 <= self.polymarket_yes_odds <= 1.0:
            raise ValueError(f"Invalid YES odds: {self.polymarket_yes_odds}")
        if not 0.0 <= self.polymarket_no_odds <= 1.0:
            raise ValueError(f"Invalid NO odds: {self.polymarket_no_odds}")


@dataclass
class LimitOrderStrategy:
    """Strategy parameters for smart limit order execution."""

    target_price: float  # Price to place limit order at
    timeout_seconds: int  # How long to wait before fallback
    fallback_to_market: bool  # Whether to use market order if timeout
    urgency: Literal["HIGH", "MEDIUM", "LOW"]
    price_improvement_pct: float  # How much better than market

    def __post_init__(self):
        """Validate field values."""
        if self.urgency not in ["HIGH", "MEDIUM", "LOW"]:
            raise ValueError(f"Invalid urgency: {self.urgency}")
        if not 0.0 <= self.target_price <= 1.0:
            raise ValueError(f"Invalid target_price: {self.target_price}")
        if self.timeout_seconds < 0:
            raise ValueError(f"timeout_seconds must be >= 0, got {self.timeout_seconds}")
