"""Polymarket BTC Trading Skill Pack.

This package provides a high-level Python client for interacting with the
Polymarket Central Limit Order Book (CLOB) API.

Public API:
    - PolymarketClient: Main client for Polymarket CLOB operations
    - get_auth_manager: Get global authentication manager
    - Market: Market data model
    - OrderRequest: Order request model
    - OrderResponse: Order response model
    - PortfolioSummary: Portfolio summary model

Example:
    >>> from polymarket import PolymarketClient
    >>> client = PolymarketClient()
    >>> markets = client.get_markets(search="btc")
    >>> portfolio = client.get_portfolio_summary()
"""

__version__ = "0.1.0"

from polymarket.client import PolymarketClient
from polymarket.auth import get_auth_manager
from polymarket.models import Market, OrderRequest, OrderResponse, PortfolioSummary

__all__ = [
    "PolymarketClient",
    "get_auth_manager",
    "Market",
    "OrderRequest",
    "OrderResponse",
    "PortfolioSummary",
]

