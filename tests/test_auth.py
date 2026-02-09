"""Tests for auth module."""

import pytest
from polymarket.config import Settings, reset_settings
from polymarket.auth import AuthManager, get_auth_manager, reset_auth_manager
from polymarket.exceptions import ConfigError


def test_auth_manager_read_only_mode():
    """Test read_only mode doesn't require credentials."""
    reset_settings()
    reset_auth_manager()
    import os
    os.environ["POLYMARKET_MODE"] = "read_only"
    settings = Settings()
    auth = AuthManager(settings)
    assert auth.mode == "read_only"
    assert not auth.requires_private_key()


def test_auth_manager_trading_mode_requires_key():
    """Test trading mode requires private key."""
    reset_settings()
    reset_auth_manager()
    import os
    os.environ["POLYMARKET_MODE"] = "trading"
    os.environ["POLYMARKET_PRIVATE_KEY"] = "0x" + "a" * 64
    settings = Settings()
    auth = AuthManager(settings)
    assert auth.mode == "trading"
    assert auth.requires_private_key()
    assert auth.private_key is not None


def test_auth_manager_trading_mode_missing_key():
    """Test trading mode fails without private key."""
    reset_settings()
    reset_auth_manager()
    import os
    os.environ["POLYMARKET_MODE"] = "trading"
    os.environ.pop("POLYMARKET_PRIVATE_KEY", None)
    # Settings validation happens in __post_init__
    with pytest.raises(ValueError, match="PRIVATE_KEY.*required.*TRADING"):
        Settings()


def test_auth_manager_masked_key():
    """Test private key masking for logging."""
    reset_settings()
    reset_auth_manager()
    import os
    os.environ["POLYMARKET_MODE"] = "trading"
    os.environ["POLYMARKET_PRIVATE_KEY"] = "0x" + "a" * 64
    settings = Settings()
    auth = AuthManager(settings)
    masked = auth.get_masked_key()
    assert masked.startswith("0x")
    assert "..." in masked
    assert len(masked) < 20  # Should be truncated


def test_auth_manager_singleton():
    """Test get_auth_manager returns singleton."""
    reset_settings()
    reset_auth_manager()
    import os
    os.environ["POLYMARKET_MODE"] = "read_only"

    auth1 = get_auth_manager()
    auth2 = get_auth_manager()
    assert auth1 is auth2


def test_auth_manager_clob_kwargs_read_only():
    """Test CLOB client kwargs for read_only mode."""
    reset_settings()
    reset_auth_manager()
    import os
    os.environ["POLYMARKET_MODE"] = "read_only"
    settings = Settings()
    auth = AuthManager(settings)

    kwargs = auth.get_clob_client_kwargs()
    assert "host" in kwargs
    assert "chain_id" in kwargs
    assert "key" not in kwargs


def test_auth_manager_clob_kwargs_trading():
    """Test CLOB client kwargs for trading mode."""
    reset_settings()
    reset_auth_manager()
    import os
    os.environ["POLYMARKET_MODE"] = "trading"
    os.environ["POLYMARKET_PRIVATE_KEY"] = "0x" + "a" * 64
    settings = Settings()
    auth = AuthManager(settings)

    kwargs = auth.get_clob_client_kwargs()
    assert "host" in kwargs
    assert "chain_id" in kwargs
    assert "key" in kwargs


def test_auth_manager_api_credentials():
    """Test API credentials handling."""
    reset_settings()
    reset_auth_manager()
    import os
    os.environ["POLYMARKET_MODE"] = "trading"
    os.environ["POLYMARKET_PRIVATE_KEY"] = "0x" + "a" * 64
    os.environ["POLYMARKET_API_KEY"] = "test-key"
    os.environ["POLYMARKET_API_SECRET"] = "test-secret"
    os.environ["POLYMARKET_API_PASSPHRASE"] = "test-pass"
    settings = Settings()
    auth = AuthManager(settings)

    assert auth.has_api_credentials()
    assert auth.api_key == "test-key"
