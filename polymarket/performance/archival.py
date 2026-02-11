# polymarket/performance/archival.py
"""Data archival system for performance database."""

import json
import gzip
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict
import structlog

from polymarket.performance.database import PerformanceDatabase

logger = structlog.get_logger()


class ArchivalManager:
    """Manages archival of old performance data."""

    def __init__(
        self,
        db: PerformanceDatabase,
        archive_dir: str = "data/archives"
    ):
        """
        Initialize archival manager.

        Args:
            db: Performance database
            archive_dir: Directory for archive files
        """
        self.db = db
        self.archive_dir = Path(archive_dir)
        self.archive_dir.mkdir(parents=True, exist_ok=True)

    def get_archivable_trades(self, days_threshold: int = 30) -> List[Dict]:
        """
        Get trades older than threshold.

        Args:
            days_threshold: Days to consider old (default 30)

        Returns:
            List of trade dictionaries
        """
        cursor = self.db.conn.cursor()

        cursor.execute("""
            SELECT * FROM trades
            WHERE timestamp < datetime('now', '-{} days')
            ORDER BY timestamp
        """.format(days_threshold))

        trades = []
        for row in cursor.fetchall():
            trade = dict(row)
            # Convert datetime to string for JSON serialization
            if trade['timestamp']:
                trade['timestamp'] = str(trade['timestamp'])
            trades.append(trade)

        return trades

    def archive_old_trades(self, days_threshold: int = 30, delete_after_archive: bool = True) -> int:
        """
        Archive old trades to JSON files.

        Args:
            days_threshold: Days to consider old (default 30)
            delete_after_archive: Remove from DB after archiving (default True)

        Returns:
            Number of trades archived
        """
        # Get trades to archive
        trades = self.get_archivable_trades(days_threshold)

        if not trades:
            logger.info("No trades to archive")
            return 0

        # Group by month
        trades_by_month = {}
        for trade in trades:
            timestamp = datetime.fromisoformat(trade['timestamp'])
            month_key = timestamp.strftime("%Y-%m")

            if month_key not in trades_by_month:
                trades_by_month[month_key] = []

            trades_by_month[month_key].append(trade)

        # Archive each month
        total_archived = 0
        for month_key, month_trades in trades_by_month.items():
            archive_file = self.archive_dir / month_key / "trades.json.gz"
            archive_file.parent.mkdir(parents=True, exist_ok=True)

            # Load existing archive if it exists
            existing_trades = []
            if archive_file.exists():
                with gzip.open(archive_file, 'rt') as f:
                    existing_data = json.load(f)
                    existing_trades = existing_data.get("trades", [])

            # Merge with new trades
            all_trades = existing_trades + month_trades

            # Write archive
            archive_data = {
                "archived_at": datetime.now().isoformat(),
                "month": month_key,
                "trade_count": len(all_trades),
                "trades": all_trades
            }

            with gzip.open(archive_file, 'wt') as f:
                json.dump(archive_data, f, indent=2)

            logger.info(
                "Archived trades",
                month=month_key,
                count=len(month_trades),
                archive_file=str(archive_file)
            )

            total_archived += len(month_trades)

        # Delete from database if requested
        if delete_after_archive:
            cursor = self.db.conn.cursor()
            cursor.execute("""
                DELETE FROM trades
                WHERE timestamp < datetime('now', '-{} days')
            """.format(days_threshold))
            self.db.conn.commit()

            logger.info(
                "Deleted archived trades from database",
                count=total_archived
            )

        return total_archived

    def get_archive_stats(self) -> Dict:
        """Get statistics about archives."""
        archive_files = list(self.archive_dir.glob("**/*.json.gz"))

        total_trades = 0
        months = []

        for archive_file in archive_files:
            with gzip.open(archive_file, 'rt') as f:
                data = json.load(f)
                total_trades += data.get("trade_count", 0)
                months.append(data.get("month"))

        return {
            "archive_count": len(archive_files),
            "total_archived_trades": total_trades,
            "months": sorted(months)
        }
