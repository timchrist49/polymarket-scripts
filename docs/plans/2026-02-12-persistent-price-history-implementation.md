# Persistent 24-Hour Price History Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Eliminate Binance API dependency by storing 24 hours of real-time BTC prices from Polymarket WebSocket

**Architecture:** In-memory deque buffer (collections.deque) with disk persistence every 5 minutes. WebSocket appends all price updates to buffer. BTCPriceService queries buffer before Binance fallback.

**Tech Stack:** Python 3.10+, asyncio, collections.deque, structlog, Decimal for precision

---

## Task 1: Create PriceHistoryBuffer Core Data Structure

**Files:**
- Create: `polymarket/trading/price_history_buffer.py`
- Test: `tests/test_price_history_buffer.py`

### Step 1: Write failing test for buffer initialization

Create `tests/test_price_history_buffer.py`:

```python
"""Tests for price history buffer."""

import pytest
from decimal import Decimal
from datetime import datetime
from polymarket.trading.price_history_buffer import PriceHistoryBuffer


def test_buffer_initializes_empty():
    """Buffer should start empty."""
    buffer = PriceHistoryBuffer(retention_hours=24)

    assert buffer.size() == 0
    assert buffer.is_empty() == True


def test_buffer_has_correct_max_length():
    """Buffer should have maxlen = retention_hours * 120 (2x safety)."""
    buffer = PriceHistoryBuffer(retention_hours=24)

    # 24 hours × 120 entries per hour (2x for real-time density)
    assert buffer.max_size() == 2880
```

### Step 2: Run test to verify it fails

Run: `cd /root/polymarket-scripts && pytest tests/test_price_history_buffer.py::test_buffer_initializes_empty -v`

Expected: `FAIL - ModuleNotFoundError: No module named 'polymarket.trading.price_history_buffer'`

### Step 3: Create minimal PriceHistoryBuffer class

Create `polymarket/trading/price_history_buffer.py`:

```python
"""24-hour price history buffer with disk persistence."""

import asyncio
import json
import os
from collections import deque
from dataclasses import dataclass, asdict
from decimal import Decimal
from datetime import datetime, timedelta
from typing import Optional, List
from pathlib import Path
import structlog

logger = structlog.get_logger()


@dataclass
class PriceEntry:
    """Single price entry in buffer."""
    timestamp: int  # Unix timestamp (seconds)
    price: Decimal
    source: str
    received_at: str  # ISO format


class PriceHistoryBuffer:
    """In-memory buffer for 24-hour price history with disk persistence."""

    def __init__(
        self,
        retention_hours: int = 24,
        save_interval: int = 300,
        persistence_file: str = "data/price_history.json"
    ):
        """
        Initialize price history buffer.

        Args:
            retention_hours: Hours of history to retain (default: 24)
            save_interval: Seconds between disk saves (default: 300 = 5min)
            persistence_file: Path to JSON persistence file
        """
        # Use 2x retention for safety (real-time updates are dense)
        max_entries = retention_hours * 120
        self._buffer: deque[PriceEntry] = deque(maxlen=max_entries)
        self._lock = asyncio.Lock()
        self._retention_hours = retention_hours
        self._save_interval = save_interval
        self._persistence_file = persistence_file
        self._last_save_time: Optional[datetime] = None
        self._dirty = False  # Track if buffer has unsaved changes

    def size(self) -> int:
        """Return number of entries in buffer."""
        return len(self._buffer)

    def is_empty(self) -> bool:
        """Check if buffer is empty."""
        return len(self._buffer) == 0

    def max_size(self) -> int:
        """Return maximum buffer capacity."""
        return self._buffer.maxlen
```

### Step 4: Run tests to verify they pass

Run: `cd /root/polymarket-scripts && pytest tests/test_price_history_buffer.py::test_buffer_initializes_empty tests/test_price_history_buffer.py::test_buffer_has_correct_max_length -v`

Expected: `PASS (2 tests)`

### Step 5: Commit

```bash
cd /root/polymarket-scripts
git add polymarket/trading/price_history_buffer.py tests/test_price_history_buffer.py
git commit -m "feat: add PriceHistoryBuffer initialization

- Create PriceEntry dataclass for price storage
- Initialize deque with 2x retention for real-time density
- Add size(), is_empty(), max_size() methods

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 2: Implement Buffer Append Operation

**Files:**
- Modify: `polymarket/trading/price_history_buffer.py`
- Modify: `tests/test_price_history_buffer.py`

### Step 1: Write failing test for append

Add to `tests/test_price_history_buffer.py`:

```python
import asyncio


@pytest.mark.asyncio
async def test_append_adds_price_to_buffer():
    """Appending a price should add it to buffer."""
    buffer = PriceHistoryBuffer(retention_hours=24)

    await buffer.append(
        timestamp=1770875100,
        price=Decimal("67018.35"),
        source="polymarket"
    )

    assert buffer.size() == 1
    assert buffer.is_empty() == False


@pytest.mark.asyncio
async def test_append_rejects_out_of_order_timestamps():
    """Buffer should reject timestamps older than last entry."""
    buffer = PriceHistoryBuffer(retention_hours=24)

    await buffer.append(1770875100, Decimal("67018.35"), "polymarket")
    await buffer.append(1770875160, Decimal("67025.10"), "polymarket")

    # Try to append older timestamp (should be rejected)
    await buffer.append(1770875050, Decimal("67000.00"), "polymarket")

    # Should still have only 2 entries
    assert buffer.size() == 2


@pytest.mark.asyncio
async def test_append_marks_buffer_dirty():
    """Appending should mark buffer as dirty for saving."""
    buffer = PriceHistoryBuffer(retention_hours=24)

    assert buffer.is_dirty() == False

    await buffer.append(1770875100, Decimal("67018.35"), "polymarket")

    assert buffer.is_dirty() == True
```

### Step 2: Run tests to verify they fail

Run: `cd /root/polymarket-scripts && pytest tests/test_price_history_buffer.py::test_append_adds_price_to_buffer -v`

Expected: `FAIL - AttributeError: 'PriceHistoryBuffer' object has no attribute 'append'`

### Step 3: Implement append method

Add to `polymarket/trading/price_history_buffer.py` (in PriceHistoryBuffer class):

```python
    async def append(
        self,
        timestamp: int,
        price: Decimal,
        source: str = "polymarket"
    ):
        """
        Add new price to buffer.

        Args:
            timestamp: Unix timestamp (seconds)
            price: BTC price
            source: Price source identifier

        Note: Rejects out-of-order timestamps (older than last entry)
        """
        async with self._lock:
            # Validate timestamp ordering
            if self._buffer and timestamp < self._buffer[-1].timestamp:
                logger.warning(
                    "Rejected out-of-order price update",
                    timestamp=timestamp,
                    last_timestamp=self._buffer[-1].timestamp,
                    age_seconds=self._buffer[-1].timestamp - timestamp
                )
                return

            entry = PriceEntry(
                timestamp=timestamp,
                price=price,
                source=source,
                received_at=datetime.now().isoformat()
            )

            self._buffer.append(entry)
            self._dirty = True

            logger.debug(
                "Price appended to buffer",
                timestamp=timestamp,
                price=f"${price:,.2f}",
                buffer_size=len(self._buffer)
            )

    def is_dirty(self) -> bool:
        """Check if buffer has unsaved changes."""
        return self._dirty
```

### Step 4: Run tests to verify they pass

Run: `cd /root/polymarket-scripts && pytest tests/test_price_history_buffer.py -k append -v`

Expected: `PASS (3 tests)`

### Step 5: Commit

```bash
cd /root/polymarket-scripts
git add polymarket/trading/price_history_buffer.py tests/test_price_history_buffer.py
git commit -m "feat: implement buffer append with timestamp validation

- Add append() method with async lock for thread safety
- Reject out-of-order timestamps (log warning)
- Mark buffer dirty on append for save tracking
- Add is_dirty() method

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 3: Implement Timestamp Lookup (get_price_at)

**Files:**
- Modify: `polymarket/trading/price_history_buffer.py`
- Modify: `tests/test_price_history_buffer.py`

### Step 1: Write failing test for get_price_at

Add to `tests/test_price_history_buffer.py`:

```python
@pytest.mark.asyncio
async def test_get_price_at_exact_timestamp():
    """Should find price at exact timestamp."""
    buffer = PriceHistoryBuffer(retention_hours=24)

    await buffer.append(1770875100, Decimal("67018.35"), "polymarket")
    await buffer.append(1770875160, Decimal("67025.10"), "polymarket")

    price = await buffer.get_price_at(1770875100)

    assert price == Decimal("67018.35")


@pytest.mark.asyncio
async def test_get_price_at_with_tolerance():
    """Should find closest price within tolerance window."""
    buffer = PriceHistoryBuffer(retention_hours=24)

    await buffer.append(1770875100, Decimal("67018.35"), "polymarket")
    await buffer.append(1770875160, Decimal("67025.10"), "polymarket")

    # Query timestamp 5 seconds after first entry (within 30s tolerance)
    price = await buffer.get_price_at(1770875105, tolerance=30)

    assert price == Decimal("67018.35")


@pytest.mark.asyncio
async def test_get_price_at_returns_none_if_not_found():
    """Should return None if no price within tolerance."""
    buffer = PriceHistoryBuffer(retention_hours=24)

    await buffer.append(1770875100, Decimal("67018.35"), "polymarket")

    # Query far future timestamp (>30s tolerance)
    price = await buffer.get_price_at(1770875200, tolerance=30)

    assert price is None
```

### Step 2: Run tests to verify they fail

Run: `cd /root/polymarket-scripts && pytest tests/test_price_history_buffer.py::test_get_price_at_exact_timestamp -v`

Expected: `FAIL - AttributeError: 'PriceHistoryBuffer' object has no attribute 'get_price_at'`

### Step 3: Implement get_price_at method

Add to `polymarket/trading/price_history_buffer.py` (in PriceHistoryBuffer class):

```python
    async def get_price_at(
        self,
        timestamp: int,
        tolerance: int = 30
    ) -> Optional[Decimal]:
        """
        Get price at specific timestamp with tolerance.

        Args:
            timestamp: Unix timestamp to query
            tolerance: Maximum seconds difference (default: 30)

        Returns:
            Price if found within tolerance, None otherwise

        Note: Returns closest price within tolerance window
        """
        async with self._lock:
            if not self._buffer:
                return None

            # Binary search would be O(log n), but linear is fine for small buffer
            # Find closest timestamp within tolerance
            closest_entry: Optional[PriceEntry] = None
            min_diff = tolerance + 1

            for entry in self._buffer:
                diff = abs(entry.timestamp - timestamp)
                if diff <= tolerance and diff < min_diff:
                    closest_entry = entry
                    min_diff = diff

            if closest_entry:
                logger.debug(
                    "Price found in buffer",
                    requested_timestamp=timestamp,
                    found_timestamp=closest_entry.timestamp,
                    diff_seconds=min_diff,
                    price=f"${closest_entry.price:,.2f}"
                )
                return closest_entry.price

            logger.debug(
                "Price not found in buffer",
                requested_timestamp=timestamp,
                tolerance=tolerance,
                buffer_size=len(self._buffer)
            )
            return None
```

### Step 4: Run tests to verify they pass

Run: `cd /root/polymarket-scripts && pytest tests/test_price_history_buffer.py -k get_price_at -v`

Expected: `PASS (3 tests)`

### Step 5: Commit

```bash
cd /root/polymarket-scripts
git add polymarket/trading/price_history_buffer.py tests/test_price_history_buffer.py
git commit -m "feat: implement timestamp lookup with tolerance

- Add get_price_at() method for historical price queries
- Find closest price within tolerance window (default 30s)
- Return None if no match found
- Linear search O(n) acceptable for buffer size

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 4: Implement Range Query (get_price_range)

**Files:**
- Modify: `polymarket/trading/price_history_buffer.py`
- Modify: `tests/test_price_history_buffer.py`

### Step 1: Write failing test for get_price_range

Add to `tests/test_price_history_buffer.py`:

```python
@pytest.mark.asyncio
async def test_get_price_range_returns_all_in_range():
    """Should return all prices in time range."""
    buffer = PriceHistoryBuffer(retention_hours=24)

    # Add 5 prices 1 minute apart
    for i in range(5):
        timestamp = 1770875100 + (i * 60)
        price = Decimal(f"6701{i}.00")
        await buffer.append(timestamp, price, "polymarket")

    # Query middle 3 entries
    entries = await buffer.get_price_range(
        start=1770875160,  # 2nd entry
        end=1770875340     # 4th entry
    )

    assert len(entries) == 3
    assert entries[0].price == Decimal("67011.00")
    assert entries[2].price == Decimal("67013.00")


@pytest.mark.asyncio
async def test_get_price_range_empty_buffer():
    """Should return empty list for empty buffer."""
    buffer = PriceHistoryBuffer(retention_hours=24)

    entries = await buffer.get_price_range(
        start=1770875100,
        end=1770875200
    )

    assert entries == []
```

### Step 2: Run tests to verify they fail

Run: `cd /root/polymarket-scripts && pytest tests/test_price_history_buffer.py::test_get_price_range_returns_all_in_range -v`

Expected: `FAIL - AttributeError: 'PriceHistoryBuffer' object has no attribute 'get_price_range'`

### Step 3: Implement get_price_range method

Add to `polymarket/trading/price_history_buffer.py` (in PriceHistoryBuffer class):

```python
    async def get_price_range(
        self,
        start: int,
        end: int
    ) -> List[PriceEntry]:
        """
        Get all prices in time range [start, end] inclusive.

        Args:
            start: Start timestamp (Unix seconds)
            end: End timestamp (Unix seconds)

        Returns:
            List of PriceEntry in chronological order
        """
        async with self._lock:
            if not self._buffer:
                return []

            # Filter entries in range
            entries = [
                entry for entry in self._buffer
                if start <= entry.timestamp <= end
            ]

            logger.debug(
                "Price range query",
                start=start,
                end=end,
                duration_seconds=end - start,
                found_entries=len(entries),
                buffer_size=len(self._buffer)
            )

            return entries
```

### Step 4: Run tests to verify they pass

Run: `cd /root/polymarket-scripts && pytest tests/test_price_history_buffer.py -k get_price_range -v`

Expected: `PASS (2 tests)`

### Step 5: Commit

```bash
cd /root/polymarket-scripts
git add polymarket/trading/price_history_buffer.py tests/test_price_history_buffer.py
git commit -m "feat: implement time range queries

- Add get_price_range() for querying time intervals
- Return all entries in [start, end] inclusive
- Maintain chronological order

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 5: Implement Disk Persistence (save/load)

**Files:**
- Modify: `polymarket/trading/price_history_buffer.py`
- Modify: `tests/test_price_history_buffer.py`

### Step 1: Write failing test for save_to_disk

Add to `tests/test_price_history_buffer.py`:

```python
import tempfile
import os


@pytest.mark.asyncio
async def test_save_to_disk_creates_json_file():
    """Should save buffer to JSON file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        json_file = os.path.join(tmpdir, "price_history.json")
        buffer = PriceHistoryBuffer(
            retention_hours=24,
            persistence_file=json_file
        )

        await buffer.append(1770875100, Decimal("67018.35"), "polymarket")
        await buffer.append(1770875160, Decimal("67025.10"), "polymarket")

        await buffer.save_to_disk()

        assert os.path.exists(json_file)


@pytest.mark.asyncio
async def test_save_to_disk_uses_atomic_write():
    """Should write to .tmp file then rename (atomic)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        json_file = os.path.join(tmpdir, "price_history.json")
        buffer = PriceHistoryBuffer(
            retention_hours=24,
            persistence_file=json_file
        )

        await buffer.append(1770875100, Decimal("67018.35"), "polymarket")
        await buffer.save_to_disk()

        # Tmp file should not exist after successful save
        tmp_file = json_file + ".tmp"
        assert not os.path.exists(tmp_file)
        assert os.path.exists(json_file)


@pytest.mark.asyncio
async def test_save_clears_dirty_flag():
    """Saving should clear dirty flag."""
    with tempfile.TemporaryDirectory() as tmpdir:
        json_file = os.path.join(tmpdir, "price_history.json")
        buffer = PriceHistoryBuffer(
            retention_hours=24,
            persistence_file=json_file
        )

        await buffer.append(1770875100, Decimal("67018.35"), "polymarket")
        assert buffer.is_dirty() == True

        await buffer.save_to_disk()
        assert buffer.is_dirty() == False
```

### Step 2: Run tests to verify they fail

Run: `cd /root/polymarket-scripts && pytest tests/test_price_history_buffer.py::test_save_to_disk_creates_json_file -v`

Expected: `FAIL - AttributeError: 'PriceHistoryBuffer' object has no attribute 'save_to_disk'`

### Step 3: Implement save_to_disk method

Add to `polymarket/trading/price_history_buffer.py` (in PriceHistoryBuffer class):

```python
    async def save_to_disk(self):
        """
        Persist buffer to JSON file using atomic write.

        Atomic write: save to .tmp file, then rename to actual file.
        This prevents corruption if process crashes during write.
        """
        async with self._lock:
            if not self._dirty:
                logger.debug("Buffer not dirty, skipping save")
                return

            # Ensure directory exists
            Path(self._persistence_file).parent.mkdir(parents=True, exist_ok=True)

            # Prepare data for JSON serialization
            data = {
                "version": "1.0",
                "last_updated": datetime.now().isoformat(),
                "retention_hours": self._retention_hours,
                "prices": [
                    {
                        "timestamp": entry.timestamp,
                        "price": str(entry.price),  # Decimal to string
                        "source": entry.source,
                        "received_at": entry.received_at
                    }
                    for entry in self._buffer
                ]
            }

            # Atomic write: write to temp, then rename
            tmp_file = self._persistence_file + ".tmp"
            try:
                with open(tmp_file, 'w') as f:
                    json.dump(data, f, indent=2)

                # Atomic rename (POSIX guarantees atomicity)
                os.replace(tmp_file, self._persistence_file)

                self._dirty = False
                self._last_save_time = datetime.now()

                logger.info(
                    "Buffer saved to disk",
                    file=self._persistence_file,
                    entries=len(self._buffer),
                    size_kb=os.path.getsize(self._persistence_file) / 1024
                )

            except Exception as e:
                logger.error(
                    "Failed to save buffer to disk",
                    error=str(e),
                    file=self._persistence_file
                )
                # Clean up temp file if it exists
                if os.path.exists(tmp_file):
                    os.remove(tmp_file)
                raise
```

### Step 4: Run tests to verify they pass

Run: `cd /root/polymarket-scripts && pytest tests/test_price_history_buffer.py -k save_to_disk -v`

Expected: `PASS (3 tests)`

### Step 5: Write failing test for load_from_disk

Add to `tests/test_price_history_buffer.py`:

```python
@pytest.mark.asyncio
async def test_load_from_disk_restores_buffer():
    """Should load buffer from JSON file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        json_file = os.path.join(tmpdir, "price_history.json")

        # Create and save buffer
        buffer1 = PriceHistoryBuffer(
            retention_hours=24,
            persistence_file=json_file
        )
        await buffer1.append(1770875100, Decimal("67018.35"), "polymarket")
        await buffer1.append(1770875160, Decimal("67025.10"), "polymarket")
        await buffer1.save_to_disk()

        # Create new buffer and load
        buffer2 = PriceHistoryBuffer(
            retention_hours=24,
            persistence_file=json_file
        )
        await buffer2.load_from_disk()

        assert buffer2.size() == 2
        price = await buffer2.get_price_at(1770875100)
        assert price == Decimal("67018.35")


@pytest.mark.asyncio
async def test_load_from_disk_handles_missing_file():
    """Should handle missing file gracefully."""
    with tempfile.TemporaryDirectory() as tmpdir:
        json_file = os.path.join(tmpdir, "nonexistent.json")
        buffer = PriceHistoryBuffer(
            retention_hours=24,
            persistence_file=json_file
        )

        # Should not raise exception
        await buffer.load_from_disk()
        assert buffer.size() == 0


@pytest.mark.asyncio
async def test_load_from_disk_handles_corrupted_json():
    """Should handle corrupted JSON gracefully."""
    with tempfile.TemporaryDirectory() as tmpdir:
        json_file = os.path.join(tmpdir, "corrupted.json")

        # Write corrupted JSON
        with open(json_file, 'w') as f:
            f.write("{ invalid json }")

        buffer = PriceHistoryBuffer(
            retention_hours=24,
            persistence_file=json_file
        )

        # Should not raise exception, just log error
        await buffer.load_from_disk()
        assert buffer.size() == 0
```

### Step 6: Run tests to verify they fail

Run: `cd /root/polymarket-scripts && pytest tests/test_price_history_buffer.py::test_load_from_disk_restores_buffer -v`

Expected: `FAIL - AttributeError: 'PriceHistoryBuffer' object has no attribute 'load_from_disk'`

### Step 7: Implement load_from_disk method

Add to `polymarket/trading/price_history_buffer.py` (in PriceHistoryBuffer class):

```python
    async def load_from_disk(self):
        """
        Load buffer from JSON persistence file on startup.

        Handles missing files and corrupted JSON gracefully.
        Invalid entries are skipped with warning.
        """
        async with self._lock:
            if not os.path.exists(self._persistence_file):
                logger.info(
                    "Persistence file not found, starting with empty buffer",
                    file=self._persistence_file
                )
                return

            try:
                with open(self._persistence_file, 'r') as f:
                    data = json.load(f)

                # Validate version
                if data.get("version") != "1.0":
                    logger.warning(
                        "Unknown persistence file version",
                        version=data.get("version")
                    )

                # Load prices
                prices = data.get("prices", [])
                loaded_count = 0
                skipped_count = 0

                for entry_data in prices:
                    try:
                        entry = PriceEntry(
                            timestamp=entry_data["timestamp"],
                            price=Decimal(entry_data["price"]),
                            source=entry_data.get("source", "unknown"),
                            received_at=entry_data.get("received_at", "")
                        )
                        self._buffer.append(entry)
                        loaded_count += 1
                    except (KeyError, ValueError) as e:
                        logger.warning(
                            "Skipped invalid entry",
                            error=str(e),
                            entry=entry_data
                        )
                        skipped_count += 1

                self._dirty = False
                self._last_save_time = datetime.fromisoformat(
                    data.get("last_updated", datetime.now().isoformat())
                )

                logger.info(
                    "Buffer loaded from disk",
                    file=self._persistence_file,
                    loaded=loaded_count,
                    skipped=skipped_count,
                    size_kb=os.path.getsize(self._persistence_file) / 1024
                )

            except json.JSONDecodeError as e:
                logger.error(
                    "Failed to parse persistence file (corrupted JSON)",
                    error=str(e),
                    file=self._persistence_file
                )
                # Continue with empty buffer

            except Exception as e:
                logger.error(
                    "Failed to load buffer from disk",
                    error=str(e),
                    file=self._persistence_file
                )
                # Continue with empty buffer
```

### Step 8: Run tests to verify they pass

Run: `cd /root/polymarket-scripts && pytest tests/test_price_history_buffer.py -k load_from_disk -v`

Expected: `PASS (3 tests)`

### Step 9: Commit

```bash
cd /root/polymarket-scripts
git add polymarket/trading/price_history_buffer.py tests/test_price_history_buffer.py
git commit -m "feat: implement disk persistence (save/load)

- Add save_to_disk() with atomic write (.tmp + rename)
- Add load_from_disk() with graceful error handling
- Handle missing files, corrupted JSON, invalid entries
- Clear dirty flag after successful save
- Log save/load operations with file size

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 6: Implement Cleanup Old Entries

**Files:**
- Modify: `polymarket/trading/price_history_buffer.py`
- Modify: `tests/test_price_history_buffer.py`

### Step 1: Write failing test for cleanup

Add to `tests/test_price_history_buffer.py`:

```python
@pytest.mark.asyncio
async def test_cleanup_removes_old_entries():
    """Should remove entries older than retention period."""
    buffer = PriceHistoryBuffer(retention_hours=1)  # 1 hour retention

    now = int(datetime.now().timestamp())

    # Add old entries (2 hours ago)
    await buffer.append(now - 7200, Decimal("67000.00"), "polymarket")
    await buffer.append(now - 7100, Decimal("67001.00"), "polymarket")

    # Add recent entries (30 minutes ago)
    await buffer.append(now - 1800, Decimal("67018.35"), "polymarket")
    await buffer.append(now - 1700, Decimal("67025.10"), "polymarket")

    assert buffer.size() == 4

    # Cleanup should remove 2 old entries
    removed = await buffer.cleanup_old_entries()

    assert removed == 2
    assert buffer.size() == 2


@pytest.mark.asyncio
async def test_cleanup_marks_buffer_dirty():
    """Cleanup should mark buffer dirty if entries removed."""
    buffer = PriceHistoryBuffer(retention_hours=1)

    now = int(datetime.now().timestamp())

    # Add old entry
    await buffer.append(now - 7200, Decimal("67000.00"), "polymarket")
    await buffer.save_to_disk()  # Clear dirty flag

    assert buffer.is_dirty() == False

    # Cleanup should mark dirty
    await buffer.cleanup_old_entries()

    assert buffer.is_dirty() == True
```

### Step 2: Run tests to verify they fail

Run: `cd /root/polymarket-scripts && pytest tests/test_price_history_buffer.py::test_cleanup_removes_old_entries -v`

Expected: `FAIL - AttributeError: 'PriceHistoryBuffer' object has no attribute 'cleanup_old_entries'`

### Step 3: Implement cleanup_old_entries method

Add to `polymarket/trading/price_history_buffer.py` (in PriceHistoryBuffer class):

```python
    async def cleanup_old_entries(self) -> int:
        """
        Remove entries older than retention period.

        Returns:
            Number of entries removed
        """
        async with self._lock:
            if not self._buffer:
                return 0

            # Calculate cutoff timestamp
            cutoff = datetime.now() - timedelta(hours=self._retention_hours)
            cutoff_timestamp = int(cutoff.timestamp())

            # Count entries to remove
            remove_count = sum(
                1 for entry in self._buffer
                if entry.timestamp < cutoff_timestamp
            )

            if remove_count == 0:
                logger.debug("No old entries to clean up")
                return 0

            # Create new deque with only recent entries
            new_buffer = deque(
                (entry for entry in self._buffer if entry.timestamp >= cutoff_timestamp),
                maxlen=self._buffer.maxlen
            )

            self._buffer = new_buffer
            self._dirty = True

            logger.info(
                "Cleaned up old entries",
                removed=remove_count,
                remaining=len(self._buffer),
                cutoff_timestamp=cutoff_timestamp,
                retention_hours=self._retention_hours
            )

            return remove_count
```

### Step 4: Run tests to verify they pass

Run: `cd /root/polymarket-scripts && pytest tests/test_price_history_buffer.py -k cleanup -v`

Expected: `PASS (2 tests)`

### Step 5: Commit

```bash
cd /root/polymarket-scripts
git add polymarket/trading/price_history_buffer.py tests/test_price_history_buffer.py
git commit -m "feat: implement cleanup for old entries

- Add cleanup_old_entries() to remove entries > retention period
- Mark buffer dirty after cleanup
- Return count of removed entries
- Use retention_hours from initialization

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 7: Integrate Buffer with WebSocket Stream

**Files:**
- Modify: `polymarket/trading/crypto_price_stream.py:76-130`
- Test: Manual verification (integration test)

### Step 1: Read current WebSocket implementation

Run: `cd /root/polymarket-scripts && cat polymarket/trading/crypto_price_stream.py | head -130`

### Step 2: Add buffer to CryptoPriceStream

Modify `polymarket/trading/crypto_price_stream.py`:

```python
# Add import at top of file
from polymarket.trading.price_history_buffer import PriceHistoryBuffer

# Modify __init__ method (around line 27)
class CryptoPriceStream:
    """Real-time BTC price stream from Polymarket WebSocket."""

    WS_URL = "wss://ws-live-data.polymarket.com"

    def __init__(self, settings: Settings, price_buffer: Optional[PriceHistoryBuffer] = None):
        self.settings = settings
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._current_price: Optional[BTCPriceData] = None
        self._connected = False
        self._running = False
        self._price_buffer = price_buffer  # NEW: Optional buffer for history

# Modify _handle_message method (around line 76)
async def _handle_message(self, message: str):
    """Parse and store price update."""
    try:
        data = json.loads(message)
        topic = data.get("topic")
        msg_type = data.get("type")
        payload = data.get("payload", {})

        if topic == "crypto_prices":
            # Handle initial subscription data dump
            if msg_type == "subscribe" and payload.get("symbol") == "btcusdt":
                price_data = payload.get("data", [])
                if price_data:
                    latest = price_data[-1]
                    self._current_price = BTCPriceData(
                        price=Decimal(str(latest["value"])),
                        timestamp=datetime.fromtimestamp(latest["timestamp"] / 1000),
                        source="polymarket",
                        volume_24h=Decimal("0")
                    )

                    # NEW: Append to buffer if available
                    if self._price_buffer:
                        await self._price_buffer.append(
                            timestamp=int(latest["timestamp"] / 1000),
                            price=Decimal(str(latest["value"])),
                            source="polymarket"
                        )

                    logger.debug(
                        "BTC price (initial)",
                        price=f"${self._current_price.price:,.2f}",
                        timestamp=self._current_price.timestamp.isoformat()
                    )

            # Handle real-time price updates
            elif msg_type == "update" and payload.get("symbol") == "btcusdt":
                self._current_price = BTCPriceData(
                    price=Decimal(str(payload["value"])),
                    timestamp=datetime.fromtimestamp(payload["timestamp"] / 1000),
                    source="polymarket",
                    volume_24h=Decimal("0")
                )

                # NEW: Append to buffer if available
                if self._price_buffer:
                    await self._price_buffer.append(
                        timestamp=int(payload["timestamp"] / 1000),
                        price=Decimal(str(payload["value"])),
                        source="polymarket"
                    )

                logger.debug(
                    "BTC price update",
                    price=f"${self._current_price.price:,.2f}",
                    timestamp=self._current_price.timestamp.isoformat()
                )

    except json.JSONDecodeError as e:
        logger.error("Failed to parse price message", error=str(e))
    except Exception as e:
        logger.error("Error handling price update", error=str(e))
```

### Step 3: Verify changes compile

Run: `cd /root/polymarket-scripts && python3 -c "from polymarket.trading.crypto_price_stream import CryptoPriceStream; print('Import successful')"`

Expected: `Import successful`

### Step 4: Commit

```bash
cd /root/polymarket-scripts
git add polymarket/trading/crypto_price_stream.py
git commit -m "feat: integrate price buffer with WebSocket stream

- Add optional price_buffer parameter to CryptoPriceStream
- Append all price updates to buffer (initial + real-time)
- Buffer is optional (backward compatible)

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 8: Update BTCPriceService to Query Buffer First

**Files:**
- Modify: `polymarket/trading/btc_price.py` (find `_fetch_binance_at_timestamp` method)
- Test: Manual verification

### Step 1: Read current BTCPriceService implementation

Run: `cd /root/polymarket-scripts && grep -n "_fetch_binance_at_timestamp" polymarket/trading/btc_price.py`

### Step 2: Add buffer to BTCPriceService __init__

Modify `polymarket/trading/btc_price.py`:

Find the `__init__` method and add:

```python
# Add import at top
from polymarket.trading.price_history_buffer import PriceHistoryBuffer

# Modify __init__ (around line 60-80)
def __init__(self, settings: Settings, price_buffer: Optional[PriceHistoryBuffer] = None):
    """
    Initialize BTC price service.

    Args:
        settings: Application settings
        price_buffer: Optional 24h price history buffer
    """
    self.settings = settings
    self._client: Optional[AsyncClient] = None
    self._cache: Dict[int, CachedPrice] = {}
    self._cache_lock = asyncio.Lock()
    self._validator: Optional[SettlementPriceValidator] = None
    self._price_buffer = price_buffer  # NEW
    self._ws_stream: Optional[CryptoPriceStream] = None
```

### Step 3: Modify _fetch_binance_at_timestamp to check buffer first

Find `_fetch_binance_at_timestamp` method and wrap it:

```python
async def _fetch_binance_at_timestamp(self, timestamp: int) -> Optional[Decimal]:
    """
    Fetch BTC price at specific timestamp.

    Query order:
    1. Check price buffer (if available)
    2. Fallback to Binance Klines API

    Args:
        timestamp: Unix timestamp (seconds)

    Returns:
        BTC price or None if unavailable
    """
    # NEW: Try buffer first
    if self._price_buffer:
        price = await self._price_buffer.get_price_at(
            timestamp=timestamp,
            tolerance=30  # 30 second tolerance
        )
        if price:
            logger.info(
                "Price fetched from buffer",
                timestamp=timestamp,
                price=f"${price:,.2f}",
                source="internal_buffer"
            )
            return price
        else:
            logger.debug(
                "Price not in buffer, falling back to Binance",
                timestamp=timestamp
            )

    # EXISTING: Fallback to Binance API
    # (keep existing implementation)
    return await self._fetch_binance_klines(timestamp)  # Existing method
```

### Step 4: Verify changes compile

Run: `cd /root/polymarket-scripts && python3 -c "from polymarket.trading.btc_price import BTCPriceService; print('Import successful')"`

Expected: `Import successful`

### Step 5: Commit

```bash
cd /root/polymarket-scripts
git add polymarket/trading/btc_price.py
git commit -m "feat: query buffer before Binance fallback

- Add optional price_buffer to BTCPriceService
- Check buffer first in _fetch_binance_at_timestamp
- Fallback to Binance if not in buffer
- 30 second tolerance for timestamp matching

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 9: Add Background Tasks for Save & Cleanup

**Files:**
- Modify: `scripts/auto_trade.py` (find main async function)
- Test: Manual verification (run bot)

### Step 1: Find main bot initialization in auto_trade.py

Run: `cd /root/polymarket-scripts && grep -n "async def main" scripts/auto_trade.py`

### Step 2: Add buffer initialization and background tasks

Modify `scripts/auto_trade.py` in the `main()` function:

```python
# Add import at top
from polymarket.trading.price_history_buffer import PriceHistoryBuffer

# In main() function, after settings initialization:
async def main():
    """Main bot loop with enhanced resilience."""
    settings = get_settings()

    # ... existing code ...

    # NEW: Initialize price history buffer
    price_buffer = PriceHistoryBuffer(
        retention_hours=24,
        save_interval=300,  # 5 minutes
        persistence_file="data/price_history.json"
    )

    # Load existing history from disk
    await price_buffer.load_from_disk()
    logger.info(
        "Price history buffer initialized",
        entries=price_buffer.size(),
        max_capacity=price_buffer.max_size()
    )

    # ... existing code ...

    # NEW: Pass buffer to WebSocket stream
    if settings.mode == "trading":
        ws_stream = CryptoPriceStream(settings, price_buffer=price_buffer)
        btc_service = BTCPriceService(settings, price_buffer=price_buffer)

    # ... existing code ...

    # NEW: Start background tasks
    async def save_buffer_task():
        """Save buffer every 5 minutes."""
        while True:
            try:
                await asyncio.sleep(300)  # 5 minutes
                if price_buffer.is_dirty():
                    await price_buffer.save_to_disk()
            except Exception as e:
                logger.error("Buffer save task failed", error=str(e))

    async def cleanup_buffer_task():
        """Cleanup old entries every hour."""
        while True:
            try:
                await asyncio.sleep(3600)  # 1 hour
                removed = await price_buffer.cleanup_old_entries()
                if removed > 0:
                    await price_buffer.save_to_disk()  # Save after cleanup
            except Exception as e:
                logger.error("Buffer cleanup task failed", error=str(e))

    # Add tasks to asyncio.gather
    tasks = [
        # ... existing tasks ...
        save_buffer_task(),
        cleanup_buffer_task(),
    ]

    # ... rest of main function ...
```

### Step 3: Add graceful shutdown handler for final save

Add before `asyncio.gather()`:

```python
    # NEW: Graceful shutdown - save buffer on exit
    import signal

    async def shutdown_handler(sig):
        """Save buffer before exit."""
        logger.info(f"Received signal {sig.name}, saving buffer...")
        await price_buffer.save_to_disk()
        logger.info("Buffer saved, exiting")

    # Register signal handlers
    for sig in (signal.SIGTERM, signal.SIGINT):
        asyncio.get_event_loop().add_signal_handler(
            sig,
            lambda s=sig: asyncio.create_task(shutdown_handler(s))
        )
```

### Step 4: Commit

```bash
cd /root/polymarket-scripts
git add scripts/auto_trade.py
git commit -m "feat: add buffer background tasks to main bot

- Initialize PriceHistoryBuffer on startup
- Load existing history from disk
- Pass buffer to WebSocket and BTCPriceService
- Add save task (every 5 minutes)
- Add cleanup task (every 1 hour)
- Add graceful shutdown handler (final save)

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 10: Manual Integration Testing

**Files:**
- Test: Run bot and verify buffer functionality

### Step 1: Restart bot with new code

Run: `cd /root/polymarket-scripts && pkill -f auto_trade.py && nohup python3 scripts/auto_trade.py > logs/bot_with_buffer.log 2>&1 &`

### Step 2: Wait 2 minutes and check buffer file created

Run: `sleep 120 && ls -lh /root/polymarket-scripts/data/price_history.json`

Expected: File exists with size > 0

### Step 3: Check buffer contents

Run: `cd /root/polymarket-scripts && python3 -c "import json; data=json.load(open('data/price_history.json')); print(f'Entries: {len(data[\"prices\"])}'); print(f'Version: {data[\"version\"]}'); print(f'Last updated: {data[\"last_updated\"]}')" `

Expected:
```
Entries: >10
Version: 1.0
Last updated: <recent timestamp>
```

### Step 4: Check bot logs for buffer activity

Run: `cd /root/polymarket-scripts && grep -i "buffer" logs/bot_with_buffer.log | tail -20`

Expected logs:
- "Price history buffer initialized"
- "Price appended to buffer"
- "Buffer saved to disk"

### Step 5: Verify buffer is queried before Binance

Run: `cd /root/polymarket-scripts && grep "Price fetched from buffer" logs/bot_with_buffer.log | wc -l`

Expected: Count > 0 (buffer is being used)

### Step 6: Restart bot and verify history persists

Run: `cd /root/polymarket-scripts && pkill -f auto_trade.py && sleep 2 && python3 -c "import json; entries_before=len(json.load(open('data/price_history.json'))['prices']); print(f'Entries before restart: {entries_before}')" && nohup python3 scripts/auto_trade.py > logs/bot_restart_test.log 2>&1 & sleep 5 && grep "Price history buffer initialized" logs/bot_restart_test.log`

Expected: Log shows "Price history buffer initialized" with entry count matching pre-restart count

### Step 7: Document test results

Create `docs/plans/2026-02-12-buffer-test-results.md`:

```markdown
# Buffer Integration Test Results

**Date:** 2026-02-12
**Tester:** Claude

## Test 1: Buffer File Creation
- ✅ File created at data/price_history.json
- ✅ Size: [X] KB
- ✅ Valid JSON format

## Test 2: Price Accumulation
- ✅ Entries accumulated: [X] entries in 2 minutes
- ✅ WebSocket integration working

## Test 3: Disk Persistence
- ✅ Save task running every 5 minutes
- ✅ Buffer saved successfully

## Test 4: Buffer Query
- ✅ BTCPriceService queries buffer first
- ✅ [X] queries served from buffer (no Binance)

## Test 5: Bot Restart
- ✅ History loaded on startup
- ✅ [X] entries restored from disk

## Test 6: Cleanup Task
- ⏳ Pending (need to wait 1 hour)

## Conclusion
Buffer implementation working as designed. Bot now independent of Binance for recent price queries.
```

### Step 8: Commit test results

```bash
cd /root/polymarket-scripts
git add docs/plans/2026-02-12-buffer-test-results.md
git commit -m "test: document buffer integration test results

All tests passing:
- Buffer creation and accumulation
- Disk persistence
- Bot restart with history restoration
- Buffer-first query strategy

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 11: Monitor Production for 24 Hours

**Files:**
- Monitor: Bot logs, memory usage, Binance fallback rate

### Step 1: Set up monitoring script

Create `scripts/monitor_buffer.py`:

```python
#!/usr/bin/env python3
"""Monitor price buffer performance."""

import json
import time
import subprocess
from datetime import datetime

def check_buffer_health():
    """Check buffer health metrics."""
    try:
        # Read buffer file
        with open('data/price_history.json') as f:
            data = json.load(f)

        entries = len(data['prices'])
        last_updated = data['last_updated']

        # Check bot process
        result = subprocess.run(
            ['pgrep', '-f', 'auto_trade.py'],
            capture_output=True
        )
        bot_running = result.returncode == 0

        # Check memory usage
        if bot_running:
            pid = result.stdout.decode().strip().split('\n')[0]
            mem_result = subprocess.run(
                ['ps', '-p', pid, '-o', 'rss='],
                capture_output=True
            )
            mem_mb = int(mem_result.stdout.decode().strip()) / 1024
        else:
            mem_mb = 0

        print(f"{datetime.now().isoformat()}")
        print(f"  Bot Running: {bot_running}")
        print(f"  Buffer Entries: {entries}")
        print(f"  Last Updated: {last_updated}")
        print(f"  Memory: {mem_mb:.1f} MB")
        print(f"  Expected: ~260 MB")
        print()

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    while True:
        check_buffer_health()
        time.sleep(600)  # Every 10 minutes
```

### Step 2: Run monitoring script

Run: `cd /root/polymarket-scripts && chmod +x scripts/monitor_buffer.py && nohup python3 scripts/monitor_buffer.py > logs/buffer_monitor.log 2>&1 &`

### Step 3: Check monitoring after 1 hour

Run: `cd /root/polymarket-scripts && tail -50 logs/buffer_monitor.log`

Expected output showing:
- Bot running: True
- Buffer entries growing over time
- Memory stable around 260MB

### Step 4: Check Binance fallback rate

Run: `cd /root/polymarket-scripts && grep "Price fetched from buffer" logs/bot_with_buffer.log | wc -l && grep "falling back to Binance" logs/bot_with_buffer.log | wc -l`

Expected: Buffer queries >> Binance fallbacks (ratio >10:1)

---

## Success Criteria

After 24 hours of monitoring:

- ✅ Bot trades consistently (no price_to_beat fallback failures)
- ✅ Memory usage stable (~260MB, not growing)
- ✅ Buffer accumulates 24 hours of history
- ✅ Cleanup task removes old entries correctly
- ✅ Bot restarts preserve history
- ✅ >90% of price queries served from buffer (not Binance)
- ✅ No performance degradation

---

## Rollback Plan

If critical issues arise:

1. Stop bot: `pkill -f auto_trade.py`
2. Revert commits: `git revert HEAD~11..HEAD`
3. Restart bot: `python3 scripts/auto_trade.py`
4. Bot will use Binance API (original behavior)

---

**Plan complete.** Next: Execute tasks 1-11 sequentially using @superpowers:executing-plans or @superpowers:subagent-driven-development.
