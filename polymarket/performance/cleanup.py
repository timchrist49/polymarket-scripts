# polymarket/performance/cleanup.py
"""Scheduled cleanup job for performance data."""

import asyncio
from datetime import datetime
from typing import Optional, Dict
import structlog

from polymarket.performance.database import PerformanceDatabase
from polymarket.performance.archival import ArchivalManager

logger = structlog.get_logger()


class CleanupScheduler:
    """Schedules and runs periodic cleanup jobs."""

    def __init__(
        self,
        db: PerformanceDatabase,
        telegram: Optional['TelegramBot'] = None,
        interval_hours: int = 168,  # 7 days
        days_threshold: int = 30
    ):
        """
        Initialize cleanup scheduler.

        Args:
            db: Performance database
            telegram: Telegram bot for notifications
            interval_hours: Hours between cleanup runs (default 168 = weekly)
            days_threshold: Days before data is archived (default 30)
        """
        self.db = db
        self.telegram = telegram
        self.interval_seconds = interval_hours * 3600
        self.days_threshold = days_threshold
        self.archival_manager = ArchivalManager(db)
        self._running = False

    async def run_cleanup(self) -> Dict:
        """
        Run cleanup job once.

        Returns:
            Dict with cleanup results
        """
        try:
            logger.info("Starting scheduled cleanup")

            # Archive old trades
            archived_count = self.archival_manager.archive_old_trades(
                days_threshold=self.days_threshold
            )

            # Get archive stats
            stats = self.archival_manager.get_archive_stats()

            # Get current database stats
            cursor = self.db.conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM trades")
            active_trades = cursor.fetchone()[0]

            result = {
                "success": True,
                "archived_count": archived_count,
                "active_trades": active_trades,
                "total_archives": stats["total_archived_trades"],
                "archive_months": len(stats["months"]),
                "timestamp": datetime.now().isoformat()
            }

            # Send notification
            if self.telegram:
                await self._send_cleanup_notification(result)

            logger.info(
                "Cleanup complete",
                archived_count=archived_count,
                active_trades=active_trades
            )

            return result

        except Exception as e:
            logger.error("Cleanup failed", error=str(e))
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    async def _send_cleanup_notification(self, result: Dict):
        """Send Telegram notification about cleanup."""
        message = f"""🧹 **Cleanup Complete**

Archived: {result['archived_count']} trades
Active trades: {result['active_trades']}
Total archived: {result['total_archives']}
Archive coverage: {result['archive_months']} months

Next cleanup: {self.interval_seconds // 3600} hours
"""

        try:
            await self.telegram._send_message(message)
        except Exception as e:
            logger.error("Failed to send cleanup notification", error=str(e))

    async def start(self):
        """Start the cleanup scheduler loop."""
        if self._running:
            logger.warning("Cleanup scheduler already running")
            return

        self._running = True
        logger.info(
            "Cleanup scheduler started",
            interval_hours=self.interval_seconds // 3600,
            days_threshold=self.days_threshold
        )

        while self._running:
            try:
                # Run cleanup
                await self.run_cleanup()

                # Wait until next run
                await asyncio.sleep(self.interval_seconds)

            except Exception as e:
                logger.error("Cleanup scheduler error", error=str(e))
                # Wait a bit before retrying
                await asyncio.sleep(3600)  # Retry after 1 hour

    def stop(self):
        """Stop the cleanup scheduler."""
        self._running = False
        logger.info("Cleanup scheduler stopped")
