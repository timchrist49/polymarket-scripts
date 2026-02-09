# tests/test_exceptions.py
import pytest
from polymarket.exceptions import (
    PolymarketError,
    ConfigError,
    AuthError,
    ValidationError,
    RateLimitError,
    UpstreamAPIError,
    NetworkError,
    MarketDiscoveryError,
)

def test_exception_hierarchy():
    """Test all exceptions inherit from PolymarketError."""
    assert issubclass(ConfigError, PolymarketError)
    assert issubclass(AuthError, PolymarketError)
    assert issubclass(ValidationError, PolymarketError)
    assert issubclass(RateLimitError, PolymarketError)
    assert issubclass(UpstreamAPIError, PolymarketError)
    assert issubclass(NetworkError, PolymarketError)
    assert issubclass(MarketDiscoveryError, PolymarketError)

def test_exception_messages():
    """Test exceptions preserve messages."""
    assert str(ConfigError("test")) == "test"
    assert str(AuthError("unauthorized")) == "unauthorized"
