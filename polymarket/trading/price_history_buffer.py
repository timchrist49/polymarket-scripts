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
