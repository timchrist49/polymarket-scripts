#!/usr/bin/env python3
"""Add fee_paid column to paper_trades table.

This migration adds the fee_paid column to track fees on settled trades.
"""

import sqlite3
import sys
from pathlib import Path


def migrate(db_path: str = "data/performance.db"):
    """Add fee_paid column to paper_trades table."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # Check if column already exists
        cursor.execute("PRAGMA table_info(paper_trades)")
        columns = [row[1] for row in cursor.fetchall()]

        if "fee_paid" in columns:
            print("✅ Column 'fee_paid' already exists, skipping migration")
            return

        # Add fee_paid column (default 0.0 for existing rows)
        cursor.execute("""
            ALTER TABLE paper_trades
            ADD COLUMN fee_paid REAL DEFAULT 0.0
        """)

        conn.commit()
        print("✅ Successfully added 'fee_paid' column to paper_trades table")

    except sqlite3.OperationalError as e:
        if "duplicate column name" in str(e).lower():
            print("⚠️  Column already exists, skipping migration")
        else:
            print(f"❌ Migration failed: {e}")
            sys.exit(1)
    finally:
        conn.close()


if __name__ == "__main__":
    migrate()
