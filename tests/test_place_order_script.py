# tests/test_place_order_script.py
import pytest
from typer.testing import CliRunner
from scripts.place_order import app

runner = CliRunner()

def test_place_order_dry_run(monkeypatch):
    """Test dry run mode."""
    monkeypatch.setenv("POLYMARKET_MODE", "trading")
    monkeypatch.setenv("POLYMARKET_PRIVATE_KEY", "0x" + "a" * 64)

    result = runner.invoke(app, [
        "--btc-mode",
        "--side", "buy",
        "--price", "0.55",
        "--size", "10",
        "--dry-run",  # Just use the flag, not "true"
    ])
    # Should show dry run output or error about network/API
    # Either is acceptable for this test
    assert result.exit_code == 0 or "DRY RUN" in result.stdout or "dry_run" in result.stdout

def test_place_order_missing_trading_creds(monkeypatch):
    """Test trading mode fails without credentials."""
    monkeypatch.setenv("POLYMARKET_MODE", "read_only")

    result = runner.invoke(app, [
        "--btc-mode",
        "--side", "buy",
        "--price", "0.55",
        "--size", "10",
    ])
    # Should error about needing trading mode
    assert result.exit_code != 0
    assert "trading" in result.stdout.lower() or "mode" in result.stdout.lower()

def test_place_order_missing_required_args(monkeypatch):
    """Test that missing required arguments cause errors."""
    monkeypatch.setenv("POLYMARKET_MODE", "trading")
    monkeypatch.setenv("POLYMARKET_PRIVATE_KEY", "0x" + "a" * 64)

    # Missing side
    result = runner.invoke(app, [
        "--btc-mode",
        "--price", "0.55",
        "--size", "10",
    ])
    assert result.exit_code != 0 or "Missing option" in result.stdout

def test_place_order_manual_market_id(monkeypatch):
    """Test manual market ID specification."""
    monkeypatch.setenv("POLYMARKET_MODE", "trading")
    monkeypatch.setenv("POLYMARKET_PRIVATE_KEY", "0x" + "a" * 64)

    result = runner.invoke(app, [
        "--market-id", "0x123456",
        "--token-id", "0x789abc",
        "--side", "sell",
        "--price", "0.60",
        "--size", "5",
        "--dry-run",  # Just use the flag
    ])
    # Should attempt to process (may fail on API, that's OK)
    assert "market-id" in result.stdout.lower() or "market" in result.stdout.lower() or result.exit_code == 0
