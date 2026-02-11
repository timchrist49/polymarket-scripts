# tests/test_performance_cleanup.py
import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from polymarket.performance.cleanup import CleanupScheduler
from polymarket.performance.database import PerformanceDatabase
import shutil

@pytest.fixture
def db_with_old_data():
    """Database with old trades."""
    db = PerformanceDatabase(":memory:")

    # Add old trades
    now = datetime.now()
    for i in range(10):
        trade_data = {
            "timestamp": now - timedelta(days=35 + i),
            "market_slug": f"old-{i}",
            "action": "YES",
            "confidence": 0.8,
            "position_size": 5.0,
            "btc_price": 66000.0,
        }
        db.log_trade(trade_data)

    return db

@pytest.mark.asyncio
async def test_run_cleanup(db_with_old_data):
    """Test running cleanup archives old data."""
    scheduler = CleanupScheduler(db_with_old_data, telegram=None)

    # Run cleanup
    result = await scheduler.run_cleanup()

    assert result["archived_count"] > 0
    assert result["success"] is True

    # Verify trades were archived
    cursor = db_with_old_data.conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM trades WHERE timestamp < datetime('now', '-30 days')")
    count = cursor.fetchone()[0]
    assert count == 0

@pytest.mark.asyncio
async def test_cleanup_with_notification(db_with_old_data):
    """Test cleanup sends Telegram notification."""
    from polymarket.telegram.bot import TelegramBot

    telegram = Mock(spec=TelegramBot)
    telegram._send_message = AsyncMock()

    scheduler = CleanupScheduler(db_with_old_data, telegram=telegram)

    result = await scheduler.run_cleanup()

    # Verify notification sent
    telegram._send_message.assert_called_once()
    call_args = telegram._send_message.call_args[0][0]
    assert "Cleanup Complete" in call_args
    assert str(result["archived_count"]) in call_args

@pytest.mark.asyncio
async def test_scheduler_interval():
    """Test scheduler runs at correct interval."""
    db = PerformanceDatabase(":memory:")
    scheduler = CleanupScheduler(db, telegram=None, interval_hours=168)  # Weekly

    assert scheduler.interval_seconds == 168 * 3600  # 7 days in seconds

@pytest.mark.asyncio
async def test_check_emergency_triggers_database_size():
    """Test emergency trigger for large database."""
    db = PerformanceDatabase(":memory:")
    scheduler = CleanupScheduler(db, telegram=None)

    # Mock database size check
    with patch.object(scheduler, '_get_database_size_mb', return_value=600):
        needs_emergency = await scheduler.check_emergency_triggers()
        assert needs_emergency is not False
        assert "database_size" in needs_emergency

@pytest.mark.asyncio
async def test_check_emergency_triggers_disk_space():
    """Test emergency trigger for low disk space."""
    db = PerformanceDatabase(":memory:")
    scheduler = CleanupScheduler(db, telegram=None)

    # Mock disk usage check to return 95% used
    with patch.object(scheduler, '_get_disk_usage_percent', return_value=95.0):
        needs_emergency = await scheduler.check_emergency_triggers()
        assert needs_emergency is not False
        assert "disk_space" in needs_emergency

@pytest.mark.asyncio
async def test_emergency_cleanup_aggressive():
    """Test emergency cleanup archives more aggressively."""
    db = PerformanceDatabase(":memory:")

    # Add recent trades that wouldn't normally be archived
    now = datetime.now()
    for i in range(10):
        trade_data = {
            "timestamp": now - timedelta(days=15 + i),  # 15-25 days old
            "market_slug": f"recent-{i}",
            "action": "YES",
            "confidence": 0.8,
            "position_size": 5.0,
            "btc_price": 66000.0,
        }
        db.log_trade(trade_data)

    scheduler = CleanupScheduler(db, telegram=None)

    # Run emergency cleanup (threshold 7 days instead of 30)
    result = await scheduler.run_emergency_cleanup()

    assert result["archived_count"] > 0
    assert result["emergency"] is True
