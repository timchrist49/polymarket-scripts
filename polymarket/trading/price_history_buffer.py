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
