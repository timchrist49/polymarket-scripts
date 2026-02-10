# tests/test_config.py
import os
import pytest
from polymarket.config import Settings, get_settings, reset_settings

@pytest.fixture(autouse=True)
def clean_env():
    """Clean environment before each test."""
    # Store original values
    original_env = dict(os.environ)
    yield
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)
    # Reset settings singleton
    reset_settings()

def test_settings_default_mode():
    """Test default mode is read_only."""
    os.environ.pop("POLYMARKET_MODE", None)
    settings = Settings()
    assert settings.mode == "read_only"

def test_settings_trading_mode():
    """Test trading mode requires credentials."""
    os.environ["POLYMARKET_MODE"] = "trading"
    os.environ["POLYMARKET_PRIVATE_KEY"] = "0x" + "a" * 64
    settings = Settings()
    assert settings.mode == "trading"

def test_settings_trading_mode_missing_credentials():
    """Test trading mode fails without private key."""
    os.environ["POLYMARKET_MODE"] = "trading"
    os.environ.pop("POLYMARKET_PRIVATE_KEY", None)
    with pytest.raises(ValueError, match="PRIVATE_KEY.*required.*TRADING"):
        Settings()

def test_settings_clob_url_default():
    """Test default CLOB URL."""
    settings = Settings()
    assert settings.clob_url == "https://clob.polymarket.com"

def test_settings_gamma_url_default():
    """Test default Gamma URL."""
    settings = Settings()
    assert settings.gamma_url == "https://gamma-api.polymarket.com"

def test_settings_repr_masks_credentials():
    """Test that __repr__ masks sensitive credentials."""
    os.environ["POLYMARKET_MODE"] = "trading"
    os.environ["POLYMARKET_PRIVATE_KEY"] = "0x" + "a" * 64
    os.environ["POLYMARKET_API_KEY"] = "api_key_1234567890abcdef"
    os.environ["POLYMARKET_API_SECRET"] = "secret_1234567890abcdef"
    os.environ["POLYMARKET_API_PASSPHRASE"] = "passphrase_1234567890abcdef"

    settings = Settings()
    repr_str = repr(settings)

    # Verify full credentials are NOT in repr
    assert "0x" + "a" * 64 not in repr_str
    assert "api_key_1234567890abcdef" not in repr_str
    assert "secret_1234567890abcdef" not in repr_str
    assert "passphrase_1234567890abcdef" not in repr_str

    # Verify masked format is present (first 6 chars + ... + last 4 chars)
    assert "0xaaaa...aaaa" in repr_str  # private_key masked
    assert "api_ke...cdef" in repr_str  # api_key masked
    assert "secret...cdef" in repr_str  # api_secret masked
    assert "passph...cdef" in repr_str  # api_passphrase masked

def test_settings_repr_with_none_credentials():
    """Test that __repr__ handles None credentials correctly."""
    os.environ.pop("POLYMARKET_MODE", None)
    os.environ.pop("POLYMARKET_PRIVATE_KEY", None)
    os.environ.pop("POLYMARKET_API_KEY", None)

    settings = Settings()
    repr_str = repr(settings)

    # Verify None values are shown as "None"
    assert "private_key=None" in repr_str
    assert "api_key=None" in repr_str

