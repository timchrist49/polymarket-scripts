# polymarket/performance/metrics.py
"""Performance metrics calculation."""

from typing import Dict, Optional
import structlog

from polymarket.performance.database import PerformanceDatabase

logger = structlog.get_logger()


class MetricsCalculator:
    """Calculates performance metrics from trade database."""

    def __init__(self, db: PerformanceDatabase):
        """
        Initialize metrics calculator.

        Args:
            db: PerformanceDatabase instance
        """
        self.db = db

    def calculate_win_rate(self, days: Optional[int] = None) -> float:
        """
        Calculate win rate percentage.

        Args:
            days: Number of days to look back (None = all time)

        Returns:
            Win rate as decimal (0.0 to 1.0)
        """
        cursor = self.db.conn.cursor()

        query = """
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN is_win = 1 THEN 1 ELSE 0 END) as wins
            FROM trades
            WHERE is_win IS NOT NULL
        """

        if days:
            query += f" AND timestamp >= datetime('now', '-{days} days')"

        cursor.execute(query)
        row = cursor.fetchone()

        if not row or row['total'] == 0:
            return 0.0

        win_rate = row['wins'] / row['total']
        return win_rate

    def calculate_total_profit(self, days: Optional[int] = None) -> float:
        """
        Calculate total profit/loss.

        Args:
            days: Number of days to look back (None = all time)

        Returns:
            Total profit/loss
        """
        cursor = self.db.conn.cursor()

        query = """
            SELECT SUM(profit_loss) as total
            FROM trades
            WHERE profit_loss IS NOT NULL
        """

        if days:
            query += f" AND timestamp >= datetime('now', '-{days} days')"

        cursor.execute(query)
        row = cursor.fetchone()

        return row['total'] or 0.0

    def calculate_signal_performance(self, days: Optional[int] = None) -> Dict[str, Dict]:
        """
        Calculate win rate by signal type.

        Args:
            days: Number of days to look back (None = all time)

        Returns:
            Dict mapping signal_type to performance stats
        """
        cursor = self.db.conn.cursor()

        query = """
            SELECT
                signal_type,
                COUNT(*) as total,
                SUM(CASE WHEN is_win = 1 THEN 1 ELSE 0 END) as wins,
                AVG(profit_loss) as avg_profit
            FROM trades
            WHERE is_win IS NOT NULL AND signal_type IS NOT NULL
        """

        if days:
            query += f" AND timestamp >= datetime('now', '-{days} days')"

        query += " GROUP BY signal_type"

        cursor.execute(query)

        results = {}
        for row in cursor.fetchall():
            signal_type = row['signal_type']
            total = row['total']
            wins = row['wins'] or 0

            results[signal_type] = {
                "total": total,
                "wins": wins,
                "losses": total - wins,
                "win_rate": wins / total if total > 0 else 0.0,
                "avg_profit": row['avg_profit'] or 0.0
            }

        return results
