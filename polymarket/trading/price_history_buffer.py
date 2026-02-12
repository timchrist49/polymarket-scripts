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
