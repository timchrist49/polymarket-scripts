# tests/test_contrarian_integration.py
"""Integration test for contrarian tracking through log_trade."""

from polymarket.performance.database import PerformanceDatabase


def test_log_trade_with_contrarian_data():
    """Test that contrarian data flows through log_trade to database."""
    from datetime import datetime

    db = PerformanceDatabase(":memory:")

    # Create trade data with contrarian fields
    trade_data = {
        "timestamp": datetime.now(),
        "market_slug": "test-market",
        "market_id": 123,
        "action": "YES",
        "confidence": 0.85,
        "position_size": 100.0,
        "reasoning": "Contrarian oversold reversal signal detected",
        "btc_price": 48500.00,
        "price_to_beat": 47000.00,
        "time_remaining_seconds": 3600,
        "is_end_phase": False,
        "contrarian_detected": True,
        "contrarian_type": "OVERSOLD_REVERSAL",
        "is_test_mode": 0
    }

    # Log trade
    trade_id = db.log_trade(trade_data)

    # Verify trade was logged
    assert trade_id > 0

    # Query database directly
    cursor = db.conn.cursor()
    cursor.execute("""
        SELECT contrarian_detected, contrarian_type, action, reasoning
        FROM trades WHERE id = ?
    """, (trade_id,))
    row = cursor.fetchone()

    assert row is not None
    assert row[0] == 1  # contrarian_detected = True
    assert row[1] == "OVERSOLD_REVERSAL"  # contrarian_type
    assert row[2] == "YES"  # action
    assert "Contrarian" in row[3]  # reasoning contains "Contrarian"


def test_log_trade_without_contrarian_data():
    """Test that non-contrarian trades use defaults."""
    from datetime import datetime

    db = PerformanceDatabase(":memory:")

    # Create trade data WITHOUT contrarian fields (using defaults)
    trade_data = {
        "timestamp": datetime.now(),
        "market_slug": "regular-market",
        "market_id": 456,
        "action": "YES",
        "confidence": 0.90,
        "position_size": 150.0,
        "reasoning": "Strong momentum continuation",
        "btc_price": 50000.00,
        "contrarian_detected": False,
        "contrarian_type": None,
        "is_test_mode": 0
    }

    # Log trade
    trade_id = db.log_trade(trade_data)

    assert trade_id > 0

    # Verify defaults in database
    cursor = db.conn.cursor()
    cursor.execute("""
        SELECT contrarian_detected, contrarian_type
        FROM trades WHERE id = ?
    """, (trade_id,))
    row = cursor.fetchone()

    assert row[0] == 0  # False
    assert row[1] is None  # NULL
