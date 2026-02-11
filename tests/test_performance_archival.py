# tests/test_performance_archival.py
import pytest
from datetime import datetime, timedelta
import json
import gzip
from pathlib import Path
from polymarket.performance.archival import ArchivalManager
from polymarket.performance.database import PerformanceDatabase

@pytest.fixture
def db_with_old_trades():
    """Database with trades of different ages."""
    db = PerformanceDatabase(":memory:")

    # Add trades from different time periods
    now = datetime.now()

    # Recent trades (< 30 days)
    for i in range(5):
        trade_data = {
            "timestamp": now - timedelta(days=i),
            "market_slug": f"recent-{i}",
            "action": "YES",
            "confidence": 0.8,
            "position_size": 5.0,
            "btc_price": 66000.0,
        }
        db.log_trade(trade_data)

    # Old trades (30-180 days)
    for i in range(5):
        trade_data = {
            "timestamp": now - timedelta(days=60 + i),
            "market_slug": f"old-{i}",
            "action": "NO",
            "confidence": 0.7,
            "position_size": 5.0,
            "btc_price": 65000.0,
        }
        db.log_trade(trade_data)

    # Very old trades (>180 days)
    for i in range(5):
        trade_data = {
            "timestamp": now - timedelta(days=200 + i),
            "market_slug": f"ancient-{i}",
            "action": "YES",
            "confidence": 0.75,
            "position_size": 5.0,
            "btc_price": 64000.0,
        }
        db.log_trade(trade_data)

    return db

def test_identify_archivable_trades(db_with_old_trades):
    """Test identifying trades that need archival."""
    manager = ArchivalManager(db_with_old_trades)

    # Get trades older than 30 days
    old_trades = manager.get_archivable_trades(days_threshold=30)

    # Should include 30-180 day trades + 180+ day trades
    assert len(old_trades) == 10  # 5 old + 5 ancient

def test_archive_trades_to_json(db_with_old_trades, tmp_path):
    """Test archiving trades to JSON files."""
    manager = ArchivalManager(db_with_old_trades, archive_dir=str(tmp_path))

    # Archive trades older than 30 days
    archived_count = manager.archive_old_trades(days_threshold=30)

    assert archived_count == 10

    # Check archive file created
    archive_files = list(tmp_path.glob("**/*.json.gz"))
    assert len(archive_files) > 0

    # Verify trade removed from database
    cursor = db_with_old_trades.conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM trades WHERE timestamp < datetime('now', '-30 days')")
    count = cursor.fetchone()[0]
    assert count == 0  # Old trades removed from DB

def test_archive_file_format(db_with_old_trades, tmp_path):
    """Test archive file format is valid JSON."""
    manager = ArchivalManager(db_with_old_trades, archive_dir=str(tmp_path))

    manager.archive_old_trades(days_threshold=30)

    # Read and verify archive file
    archive_files = list(tmp_path.glob("**/*.json.gz"))
    assert len(archive_files) > 0

    with gzip.open(archive_files[0], 'rt') as f:
        data = json.load(f)
        assert "trades" in data
        assert len(data["trades"]) > 0
        assert "timestamp" in data["trades"][0]
