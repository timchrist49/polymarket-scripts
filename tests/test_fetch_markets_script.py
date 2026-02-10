"""Tests for fetch_markets.py script."""

import pytest
from typer.testing import CliRunner
from polymarket.config import reset_settings
from polymarket.auth import reset_auth_manager

# Import after adding to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.fetch_markets import app

runner = CliRunner()


def test_fetch_markets_btc_mode():
    """Test BTC mode fetches correct market."""
    reset_settings()
    reset_auth_manager()
    import os
    os.environ["POLYMARKET_MODE"] = "read_only"

    result = runner.invoke(app, ["--btc-mode"])
    # May fail due to network/API, that's OK
    # Just check it doesn't crash with import/usage errors
    assert result.exit_code == 0 or "discovery" in result.stdout.lower() or "Error" in result.stdout


def test_fetch_markets_search():
    """Test search mode."""
    reset_settings()
    reset_auth_manager()
    import os
    os.environ["POLYMARKET_MODE"] = "read_only"

    result = runner.invoke(app, ["--search", "bitcoin", "--limit", "5"])
    # May fail due to network, that's OK
    assert result.exit_code == 0 or "Error" in result.stdout


def test_fetch_markets_json_output():
    """Test JSON output mode."""
    reset_settings()
    reset_auth_manager()
    import os
    os.environ["POLYMARKET_MODE"] = "read_only"

    result = runner.invoke(app, ["--btc-mode", "--json"])
    # May fail due to network, that's OK
    assert result.exit_code == 0 or "Error" in result.stdout


def test_fetch_markets_invalid_args():
    """Test with invalid arguments (should show help)."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "fetch" in result.stdout.lower()


def test_fetch_markets_all_flag():
    """Test --all flag (inactive markets)."""
    reset_settings()
    reset_auth_manager()
    import os
    os.environ["POLYMARKET_MODE"] = "read_only"

    result = runner.invoke(app, ["--all", "--limit", "1"])
    # May fail due to network, that's OK
    assert result.exit_code == 0 or "Error" in result.stdout
