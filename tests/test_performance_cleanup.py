# tests/test_performance_cleanup.py
import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from polymarket.performance.cleanup import CleanupScheduler
from polymarket.performance.database import PerformanceDatabase

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
