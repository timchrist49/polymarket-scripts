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
