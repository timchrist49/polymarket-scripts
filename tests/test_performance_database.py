import pytest
from datetime import datetime
from decimal import Decimal
from polymarket.performance.database import PerformanceDatabase

@pytest.fixture
def db():
    """Create in-memory test database."""
    db = PerformanceDatabase(":memory:")
    yield db
    db.close()

def test_create_tables(db):
    """Test tables are created with correct schema."""
    cursor = db.conn.cursor()

    # Check trades table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='trades'")
    assert cursor.fetchone() is not None

    # Check columns exist
    cursor.execute("PRAGMA table_info(trades)")
    columns = {row[1] for row in cursor.fetchall()}

    expected_columns = {
        'id', 'timestamp', 'market_slug', 'market_id',
        'action', 'confidence', 'position_size', 'reasoning',
        'btc_price', 'price_to_beat', 'time_remaining_seconds', 'is_end_phase',
        'social_score', 'market_score', 'final_score', 'final_confidence', 'signal_type',
        'rsi', 'macd', 'trend',
        'yes_price', 'no_price', 'executed_price',
        'actual_outcome', 'profit_loss', 'is_win', 'is_missed_opportunity'
    }

    assert expected_columns.issubset(columns)

def test_create_indexes(db):
    """Test indexes are created for fast queries."""
    cursor = db.conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
    indexes = {row[0] for row in cursor.fetchall()}

    expected = {'idx_trades_timestamp', 'idx_trades_signal_type', 'idx_trades_is_win'}
    assert expected.issubset(indexes)

def test_log_trade(db):
    """Test logging a trade decision."""
    trade_data = {
        "timestamp": datetime(2026, 2, 11, 10, 30, 0),
        "market_slug": "btc-updown-15m-1234567890",
        "market_id": 1362391,
        "action": "NO",
        "confidence": 1.0,
        "position_size": 5.0,
        "reasoning": "Bearish signals aligned",
        "btc_price": 66940.0,
        "price_to_beat": 66826.14,
        "time_remaining_seconds": 480,
        "is_end_phase": False,
        "social_score": -0.10,
        "market_score": -0.21,
        "final_score": -0.17,
        "final_confidence": 1.0,
        "signal_type": "STRONG_BEARISH",
        "rsi": 60.1,
        "macd": 1.74,
        "trend": "BULLISH",
        "yes_price": 0.51,
        "no_price": 0.50,
        "executed_price": 0.52
    }

    trade_id = db.log_trade(trade_data)
    assert trade_id > 0

    # Verify data was stored
    cursor = db.conn.cursor()
    cursor.execute("SELECT * FROM trades WHERE id = ?", (trade_id,))
    row = cursor.fetchone()

    assert row is not None
    assert row['action'] == 'NO'
    assert row['confidence'] == 1.0
    assert row['market_slug'] == 'btc-updown-15m-1234567890'

def test_update_outcome(db):
    """Test updating trade outcome after market closes."""
    # First log a trade
    trade_data = {
        "timestamp": datetime(2026, 2, 11, 10, 30, 0),
        "market_slug": "btc-updown-15m-1234567890",
        "market_id": 1362391,
        "action": "NO",
        "confidence": 1.0,
        "position_size": 5.0,
        "btc_price": 66940.0,
    }
    trade_id = db.log_trade(trade_data)

    # Update with outcome
    db.update_outcome(
        market_slug="btc-updown-15m-1234567890",
        actual_outcome="DOWN",
        profit_loss=4.50
    )

    # Verify outcome was stored
    cursor = db.conn.cursor()
    cursor.execute("SELECT * FROM trades WHERE id = ?", (trade_id,))
    row = cursor.fetchone()

    assert row['actual_outcome'] == 'DOWN'
    assert row['profit_loss'] == 4.50
    assert row['is_win'] == True  # NO bet + DOWN outcome = win

def test_update_outcome_missed_opportunity(db):
    """Test HOLD decision marked as missed opportunity."""
    trade_data = {
        "timestamp": datetime(2026, 2, 11, 10, 30, 0),
        "market_slug": "btc-updown-15m-1234567890",
        "action": "HOLD",
        "confidence": 0.85,
        "position_size": 0.0,
        "btc_price": 66940.0,
        "price_to_beat": 66826.14,
    }
    trade_id = db.log_trade(trade_data)

    # Update - price went up, would have won YES
    db.update_outcome(
        market_slug="btc-updown-15m-1234567890",
        actual_outcome="UP",
        profit_loss=0.0  # Didn't trade
    )

    cursor = db.conn.cursor()
    cursor.execute("SELECT * FROM trades WHERE id = ?", (trade_id,))
    row = cursor.fetchone()

    assert row['actual_outcome'] == 'UP'
    assert row['is_missed_opportunity'] == True
