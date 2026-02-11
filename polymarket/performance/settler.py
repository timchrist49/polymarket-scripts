"""Trade settlement service for determining win/loss outcomes."""

import re
import structlog
from datetime import datetime, timedelta
from typing import Dict, Optional
from decimal import Decimal

from polymarket.performance.database import PerformanceDatabase

logger = structlog.get_logger()


class TradeSettler:
    """Settles trades by comparing BTC prices at market close."""

    def __init__(self, db: PerformanceDatabase, btc_fetcher):
        """
        Initialize trade settler.

        Args:
            db: Performance database
            btc_fetcher: BTC price service (from auto_trade.py)
        """
        self.db = db
        self.btc_fetcher = btc_fetcher

    def _parse_market_close_timestamp(self, market_slug: str) -> Optional[int]:
        """
        Extract Unix timestamp from market slug.

        Args:
            market_slug: Format "btc-updown-15m-1770828300" or variations

        Returns:
            Unix timestamp or None if parsing fails
        """
        if not market_slug:
            return None

        # Pattern: any text ending with a 10-digit Unix timestamp
        # Unix timestamps are 10 digits for dates between 2001-2286
        match = re.search(r'(\d{10})$', market_slug)

        if match:
            return int(match.group(1))

        logger.warning(
            "Failed to parse timestamp from market slug",
            market_slug=market_slug
        )
        return None
