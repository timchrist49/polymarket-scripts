# polymarket/performance/cleanup.py
"""Scheduled cleanup job for performance data."""

import asyncio
from datetime import datetime
from typing import Optional, Dict, Union, Literal
import structlog
import shutil
from pathlib import Path

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

    def _get_database_size_mb(self) -> float:
        """Get database file size in MB."""
        if self.db.db_path == ":memory:":
            return 0.0

        db_file = Path(self.db.db_path)
        if not db_file.exists():
            return 0.0

        size_bytes = db_file.stat().st_size
        return size_bytes / (1024 * 1024)

    def _get_disk_usage_percent(self) -> float:
        """Get disk usage percentage."""
        if self.db.db_path == ":memory:":
            return 0.0

        db_file = Path(self.db.db_path)
        if not db_file.exists():
            return 0.0

        usage = shutil.disk_usage(db_file.parent)
        used_percent = (usage.used / usage.total) * 100
        return used_percent

    async def check_emergency_triggers(self) -> Union[Dict, Literal[False]]:
        """
        Check if emergency cleanup is needed.

        Returns:
            Dict with trigger info if emergency needed, False otherwise
        """
        triggers = {}

        # Check database size
        db_size_mb = self._get_database_size_mb()
        if db_size_mb > 500:
            triggers["database_size"] = {
                "current_mb": db_size_mb,
                "threshold_mb": 500,
                "reason": "Database exceeds 500MB"
            }

        # Check disk space
        disk_used_percent = self._get_disk_usage_percent()
        if disk_used_percent > 90:  # >90% used = <10% free
            triggers["disk_space"] = {
                "used_percent": disk_used_percent,
                "threshold_percent": 90,
                "reason": "Disk usage above 90%"
            }

        if triggers:
            logger.warning(
                "Emergency cleanup triggers detected",
                triggers=list(triggers.keys()),
                db_size_mb=db_size_mb,
                disk_used_percent=disk_used_percent
            )
            return triggers

        return False

    async def run_emergency_cleanup(self) -> Dict:
        """
        Run aggressive emergency cleanup.

        Returns:
            Dict with cleanup results
        """
        try:
            logger.critical("Running EMERGENCY cleanup")

            # More aggressive threshold (7 days instead of 30)
            emergency_threshold = 7

            # Archive with aggressive threshold
            archived_count = self.archival_manager.archive_old_trades(
                days_threshold=emergency_threshold
            )

            # Get stats
            stats = self.archival_manager.get_archive_stats()

            cursor = self.db.conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM trades")
            active_trades = cursor.fetchone()[0]

            result = {
                "success": True,
                "emergency": True,
                "archived_count": archived_count,
                "active_trades": active_trades,
                "threshold_days": emergency_threshold,
                "db_size_mb": self._get_database_size_mb(),
                "disk_used_percent": self._get_disk_usage_percent(),
                "timestamp": datetime.now().isoformat()
            }

            # Send urgent notification
            if self.telegram:
                await self._send_emergency_cleanup_notification(result)

            logger.critical(
                "Emergency cleanup complete",
                archived_count=archived_count,
                active_trades=active_trades,
                db_size_mb=result["db_size_mb"]
            )

            return result

        except Exception as e:
            logger.error("Emergency cleanup failed", error=str(e))
            return {
                "success": False,
                "emergency": True,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    async def _send_emergency_cleanup_notification(self, result: Dict):
        """Send urgent Telegram notification about emergency cleanup."""
        message = f"""🚨 **EMERGENCY CLEANUP EXECUTED** 🚨

⚠️ Storage limits approaching

Archived: {result['archived_count']} trades (7-day threshold)
Active trades: {result['active_trades']}
Database size: {result['db_size_mb']:.1f} MB
Disk usage: {result['disk_used_percent']:.1f}%

Cleanup was triggered automatically to prevent storage issues.
Monitor database growth and consider adjusting retention policies.
"""

        try:
            await self.telegram._send_message(message)
        except Exception as e:
            logger.error("Failed to send emergency cleanup notification", error=str(e))

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
                # Check for emergency triggers
                emergency_triggers = await self.check_emergency_triggers()

                if emergency_triggers:
                    # Run emergency cleanup
                    await self.run_emergency_cleanup()
                else:
                    # Run normal cleanup
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
