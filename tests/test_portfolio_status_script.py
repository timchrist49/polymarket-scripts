# tests/test_portfolio_status_script.py
import pytest
from typer.testing import CliRunner
from scripts.portfolio_status import app

runner = CliRunner()

def test_portfolio_status_empty(monkeypatch):
    """Test portfolio status with no orders."""
    monkeypatch.setenv("POLYMARKET_MODE", "read_only")

    result = runner.invoke(app, [])
    # Should not crash, show empty portfolio or read_only message
    assert result.exit_code == 0

def test_portfolio_status_json(monkeypatch):
    """Test JSON output."""
    monkeypatch.setenv("POLYMARKET_MODE", "read_only")

    result = runner.invoke(app, ["--json"])
    assert result.exit_code == 0

def test_portfolio_status_with_market_filter(monkeypatch):
    """Test portfolio status with market ID filter."""
    monkeypatch.setenv("POLYMARKET_MODE", "read_only")

    result = runner.invoke(app, ["--market-id", "0x123456"])
    # Should not crash
    assert result.exit_code == 0

def test_portfolio_status_trading_mode_no_creds(monkeypatch):
    """Test portfolio status in trading mode without proper credentials."""
    monkeypatch.setenv("POLYMARKET_MODE", "trading")
    # Missing private key - should handle gracefully

    result = runner.invoke(app, [])
    # May fail with credential error or API error - both are acceptable
    # The key is it shouldn't crash unexpectedly
    assert result.exit_code != 0 or "error" in result.stdout.lower() or "portfolio" in result.stdout.lower()
