# tests/test_contrarian_database.py
"""Test contrarian tracking fields in database."""

import sqlite3
from polymarket.performance.database import PerformanceDatabase


def test_contrarian_fields_in_schema():
    """Trades table should have contrarian_detected and contrarian_type fields."""
    db = PerformanceDatabase(":memory:")

    # Query schema
    cursor = db.conn.cursor()
    cursor.execute("PRAGMA table_info(trades)")
    columns = {row[1]: row[2] for row in cursor.fetchall()}

    assert "contrarian_detected" in columns
    assert "contrarian_type" in columns


def test_store_trade_with_contrarian():
    """Should store contrarian trade data."""
    db = PerformanceDatabase(":memory:")

    # Create minimal trade data
    cursor = db.conn.cursor()
    cursor.execute("""
        INSERT INTO trades (
            market_id, market_slug, action, confidence, position_size, reasoning,
            btc_price, contrarian_detected, contrarian_type,
            timestamp
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
    """, ("test_market", "test-market", "YES", 0.85, 100.0, "Test trade", 45000.0, 1, "OVERSOLD_REVERSAL"))

    db.conn.commit()

    # Verify stored
    cursor.execute("SELECT contrarian_detected, contrarian_type FROM trades WHERE market_id = ?", ("test_market",))
    row = cursor.fetchone()

    assert row[0] == 1  # True as integer
    assert row[1] == "OVERSOLD_REVERSAL"


def test_store_trade_without_contrarian():
    """Should store non-contrarian trade with defaults."""
    db = PerformanceDatabase(":memory:")

    # Create trade without contrarian data
    cursor = db.conn.cursor()
    cursor.execute("""
        INSERT INTO trades (
            market_id, market_slug, action, confidence, position_size, reasoning,
            btc_price, timestamp
        ) VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))
    """, ("test_market_2", "test-market-2", "NO", 0.75, 50.0, "Regular trade", 46000.0))

    db.conn.commit()

    # Verify defaults
    cursor.execute("SELECT contrarian_detected, contrarian_type FROM trades WHERE market_id = ?", ("test_market_2",))
    row = cursor.fetchone()

    assert row[0] == 0  # False as integer (default)
    assert row[1] is None  # NULL (default)


def test_contrarian_index_exists():
    """Should have index for contrarian queries."""
    db = PerformanceDatabase(":memory:")

    # Query indexes
    cursor = db.conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='trades'")
    indexes = [row[0] for row in cursor.fetchall()]

    assert "idx_trades_contrarian_detected" in indexes


def test_query_contrarian_trades_only():
    """Should efficiently query contrarian trades."""
    db = PerformanceDatabase(":memory:")

    cursor = db.conn.cursor()

    # Insert mixed trades
    cursor.execute("""
        INSERT INTO trades (
            market_id, market_slug, action, confidence, position_size, reasoning,
            btc_price, contrarian_detected, contrarian_type, timestamp
        ) VALUES
        (?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now')),
        (?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now')),
        (?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
    """, (
        "m1", "market-1", "YES", 0.85, 100.0, "Contrarian", 45000.0, 1, "OVERSOLD_REVERSAL",
        "m2", "market-2", "NO", 0.75, 50.0, "Regular", 46000.0, 0, None,
        "m3", "market-3", "YES", 0.90, 75.0, "Contrarian", 47000.0, 1, "OVERBOUGHT_REVERSAL"
    ))
    db.conn.commit()

    # Query only contrarian trades
    cursor.execute("SELECT market_id, contrarian_type FROM trades WHERE contrarian_detected = 1 ORDER BY market_id")
    rows = cursor.fetchall()

    assert len(rows) == 2
    assert rows[0][0] == "m1"
    assert rows[0][1] == "OVERSOLD_REVERSAL"
    assert rows[1][0] == "m3"
    assert rows[1][1] == "OVERBOUGHT_REVERSAL"
