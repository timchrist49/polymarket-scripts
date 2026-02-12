"""
Tests for background tasks (auto-save and cleanup).

These tests verify that periodic background tasks run correctly:
- price_history_saver() saves buffer to disk periodically
- price_history_cleaner() removes old entries periodically
"""

import pytest
import asyncio
import os
import tempfile
from decimal import Decimal
from datetime import datetime
from polymarket.trading.price_history_buffer import PriceHistoryBuffer


# Import the functions we'll be testing (they don't exist yet)
# These imports will fail until we implement them in Step 4
try:
    from scripts.auto_trade import price_history_saver, price_history_cleaner
    FUNCTIONS_EXIST = True
except ImportError:
    FUNCTIONS_EXIST = False
    price_history_saver = None
    price_history_cleaner = None


@pytest.mark.skipif(not FUNCTIONS_EXIST, reason="Background task functions not yet implemented")
@pytest.mark.asyncio
async def test_saver_runs_periodically():
    """Test saver task runs every 5 minutes (using 1 sec interval for testing)."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        test_file = f.name

    try:
        buffer = PriceHistoryBuffer(retention_hours=24, save_interval=1, persistence_file=test_file)

        # Add price to make buffer dirty
        timestamp = int(datetime.now().timestamp())
        await buffer.append(timestamp, Decimal("67000.00"))

        # Verify buffer is dirty before save
        assert buffer._dirty, "Buffer should be dirty after append"

        # Run saver in background for 2.5 seconds (should trigger 2 saves at interval=1)
        task = asyncio.create_task(price_history_saver(buffer, interval=1))
        await asyncio.sleep(2.5)
        task.cancel()

        try:
            await task
        except asyncio.CancelledError:
            pass

        # Verify buffer was saved (dirty flag cleared)
        assert not buffer._dirty, "Buffer should be clean after auto-save"

        # Verify file exists and contains data
        assert os.path.exists(test_file), "Persistence file should exist"

        # Verify we can load from the file
        buffer2 = PriceHistoryBuffer(retention_hours=24, persistence_file=test_file)
        await buffer2.load_from_disk()
        assert buffer2.size() == 1, "Should load 1 entry from disk"

    finally:
        if os.path.exists(test_file):
            os.unlink(test_file)


@pytest.mark.skipif(not FUNCTIONS_EXIST, reason="Background task functions not yet implemented")
@pytest.mark.asyncio
async def test_cleaner_runs_periodically():
    """Test cleaner task runs every hour (using 1 sec interval for testing)."""
    buffer = PriceHistoryBuffer(retention_hours=24)

    # Add old entry (25 hours ago - should be removed)
    old_timestamp = int(datetime.now().timestamp()) - (25 * 3600)
    await buffer.append(old_timestamp, Decimal("60000.00"))

    # Add recent entry (should be kept)
    recent_timestamp = int(datetime.now().timestamp())
    await buffer.append(recent_timestamp, Decimal("67000.00"))

    assert buffer.size() == 2, "Should start with 2 entries"

    # Run cleaner in background for 2.5 seconds (should trigger 2 cleanups at interval=1)
    task = asyncio.create_task(price_history_cleaner(buffer, interval=1))
    await asyncio.sleep(2.5)
    task.cancel()

    try:
        await task
    except asyncio.CancelledError:
        pass

    # Verify old entry was removed
    assert buffer.size() == 1, "Old entry should be removed, leaving 1 entry"

    # Verify the remaining entry is the recent one
    entries = await buffer.get_price_range(recent_timestamp - 10, recent_timestamp + 10)
    assert len(entries) == 1, "Should have 1 recent entry"
    assert entries[0].price == Decimal("67000.00"), "Should be the recent price"


@pytest.mark.skipif(not FUNCTIONS_EXIST, reason="Background task functions not yet implemented")
@pytest.mark.asyncio
async def test_saver_handles_errors_gracefully():
    """Test saver doesn't crash on save errors."""
    # Create buffer with invalid persistence file path (will fail on save)
    buffer = PriceHistoryBuffer(
        retention_hours=24,
        persistence_file="/invalid/path/that/does/not/exist/file.json"
    )

    # Add data to make buffer dirty
    timestamp = int(datetime.now().timestamp())
    await buffer.append(timestamp, Decimal("67000.00"))

    # Run saver (should handle permission error gracefully without crashing)
    task = asyncio.create_task(price_history_saver(buffer, interval=0.1))
    await asyncio.sleep(0.3)
    task.cancel()

    try:
        await task
    except asyncio.CancelledError:
        pass  # Expected - task was cancelled
    except Exception as e:
        pytest.fail(f"Saver task should handle errors gracefully, but raised: {e}")

    # If we reach here, the task handled errors correctly
    # (It logged the error but didn't crash)


@pytest.mark.skipif(not FUNCTIONS_EXIST, reason="Background task functions not yet implemented")
@pytest.mark.asyncio
async def test_cleaner_handles_errors_gracefully():
    """Test cleaner doesn't crash on cleanup errors."""
    buffer = PriceHistoryBuffer(retention_hours=24)

    # Run cleaner (should handle any errors gracefully)
    task = asyncio.create_task(price_history_cleaner(buffer, interval=0.1))
    await asyncio.sleep(0.3)
    task.cancel()

    try:
        await task
    except asyncio.CancelledError:
        pass  # Expected - task was cancelled
    except Exception as e:
        pytest.fail(f"Cleaner task should handle errors gracefully, but raised: {e}")


@pytest.mark.skipif(not FUNCTIONS_EXIST, reason="Background task functions not yet implemented")
@pytest.mark.asyncio
async def test_tasks_respond_to_cancellation():
    """Test that tasks properly handle cancellation."""
    buffer = PriceHistoryBuffer(retention_hours=24)

    # Start both tasks
    saver_task = asyncio.create_task(price_history_saver(buffer, interval=10))
    cleaner_task = asyncio.create_task(price_history_cleaner(buffer, interval=10))

    # Let them run briefly
    await asyncio.sleep(0.1)

    # Cancel both
    saver_task.cancel()
    cleaner_task.cancel()

    # Verify they handle cancellation properly
    with pytest.raises(asyncio.CancelledError):
        await saver_task

    with pytest.raises(asyncio.CancelledError):
        await cleaner_task
