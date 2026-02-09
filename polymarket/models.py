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
