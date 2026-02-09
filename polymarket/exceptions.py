# polymarket/exceptions.py
"""
Polymarket-specific exceptions.

This module defines custom exceptions for Polymarket API errors including
authentication failures, order rejections, and network issues.

Classes:
    PolymarketError: Base exception for all Polymarket errors
    AuthenticationError: Raised when authentication fails
    OrderError: Raised when order placement fails
    MarketNotFoundError: Raised when market is not found

Example:
    >>> try:
    ...     client.place_order(...)
    ... except AuthenticationError:
    ...     print("Check your credentials")
"""


class PolymarketError(Exception):
    """Base exception for all Polymarket errors."""

    def __init__(self, message: str, details: dict | None = None):
        self.message = message
        self.details = details or {}
        super().__init__(message)

    def __str__(self) -> str:
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} ({details_str})"
        return self.message


class ConfigError(PolymarketError):
    """Missing or invalid environment configuration."""


class AuthError(PolymarketError):
    """Authentication or authorization failure."""


class ValidationError(PolymarketError):
    """Input validation failed."""


class RateLimitError(PolymarketError):
    """HTTP 429 - rate limit exceeded."""


class UpstreamAPIError(PolymarketError):
    """5xx errors from Polymarket."""


class NetworkError(PolymarketError):
    """Network connectivity issues."""


class MarketDiscoveryError(PolymarketError):
    """Failed to discover active BTC 15-min market."""
