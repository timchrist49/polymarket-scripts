# tests/conftest.py
"""Test fixtures and configuration."""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

# Import reset functions if modules exist
try:
    from polymarket.config import reset_settings
except ImportError:
    def reset_settings():
        pass

try:
    from polymarket.auth import reset_auth_manager
except ImportError:
    def reset_auth_manager():
        pass


@pytest.fixture(autouse=True)
def reset_state():
    """Reset global state before each test."""
    reset_settings()
    reset_auth_manager()
    yield
    reset_settings()
    reset_auth_manager()


@pytest.fixture
def read_only_env(monkeypatch):
    """Set up read_only environment."""
    monkeypatch.setenv("POLYMARKET_MODE", "read_only")


@pytest.fixture
def trading_env(monkeypatch):
    """Set up trading environment."""
    monkeypatch.setenv("POLYMARKET_MODE", "trading")
    monkeypatch.setenv("POLYMARKET_PRIVATE_KEY", "0x" + "a" * 64)
    monkeypatch.setenv("POLYMARKET_API_KEY", "test-key")
    monkeypatch.setenv("POLYMARKET_API_SECRET", "test-secret")
    monkeypatch.setenv("POLYMARKET_API_PASSPHRASE", "test-pass")
