"""Tests for tracker settlement updates."""

import pytest
from datetime import datetime
from polymarket.performance.tracker import PerformanceTracker
from polymarket.performance.database import PerformanceDatabase


class TestTrackerSettlementUpdate:
    """Test updating trade outcomes in tracker."""

    def test_update_trade_outcome(self):
        """Should update trade with settlement data."""
        tracker = PerformanceTracker(db_path=":memory:")

        # Insert test trade
        cursor = tracker.db.conn.cursor()
        cursor.execute("""
            INSERT INTO trades (
                timestamp, market_slug, action, confidence, position_size,
                btc_price, price_to_beat, executed_price, is_win
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now(),
            "btc-updown-15m-1770828300",
            "YES",
            0.75,
            10.0,
            70000.0,
            70000.0,
            0.65,
            None
        ))
        tracker.db.conn.commit()

        trade_id = cursor.lastrowid

        # Update with settlement
        tracker.update_trade_outcome(
            trade_id=trade_id,
            actual_outcome="YES",
            profit_loss=5.38,
            is_win=True
        )

        # Verify update
        cursor.execute("SELECT actual_outcome, profit_loss, is_win FROM trades WHERE id = ?", (trade_id,))
        row = cursor.fetchone()

        assert row[0] == "YES"
        assert abs(row[1] - 5.38) < 0.01
        assert row[2] == 1  # SQLite stores True as 1

    def test_update_nonexistent_trade(self):
        """Should handle updating nonexistent trade gracefully."""
        tracker = PerformanceTracker(db_path=":memory:")

        # Should not raise exception
        tracker.update_trade_outcome(
            trade_id=99999,
            actual_outcome="NO",
            profit_loss=-10.0,
            is_win=False
        )

        # Verify no trades exist
        cursor = tracker.db.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM trades WHERE id = 99999")
        count = cursor.fetchone()[0]

        assert count == 0
