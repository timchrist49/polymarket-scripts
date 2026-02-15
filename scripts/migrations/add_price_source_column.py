#!/usr/bin/env python3
"""
Migration: Add price_source column to trades and paper_trades tables.

This column tracks which price source was used (chainlink, binance, coingecko)
for audit trail and debugging price discrepancies.
"""

import sqlite3
import os
from pathlib import Path


def migrate(db_path: str = "data/performance.db"):
    """Add price_source column to tables."""
    print(f"Running migration on: {db_path}")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # Check which tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]

        # Add to trades table if it exists
        if "trades" in tables:
            try:
                cursor.execute("""
                    ALTER TABLE trades
                    ADD COLUMN price_source TEXT DEFAULT 'unknown'
                """)
                print("✓ Added price_source to trades table")
            except sqlite3.OperationalError as e:
                if "duplicate column name" in str(e).lower():
                    print("⚠ price_source column already exists in trades table")
                else:
                    raise
        else:
            print("⚠ trades table not found, skipping")

        # Add to paper_trades table if it exists
        if "paper_trades" in tables:
            try:
                cursor.execute("""
                    ALTER TABLE paper_trades
                    ADD COLUMN price_source TEXT DEFAULT 'unknown'
                """)
                print("✓ Added price_source to paper_trades table")
            except sqlite3.OperationalError as e:
                if "duplicate column name" in str(e).lower():
                    print("⚠ price_source column already exists in paper_trades table")
                else:
                    raise
        else:
            print("⚠ paper_trades table not found, skipping")

        conn.commit()
        print("\n✓ Migration complete!")

    finally:
        conn.close()


if __name__ == "__main__":
    # Run on both main and worktree databases if they exist
    databases = [
        "data/performance.db",
        "/root/polymarket-scripts/data/performance.db"
    ]

    for db_path in databases:
        if Path(db_path).exists():
            migrate(db_path)
        else:
            print(f"⚠ Database not found: {db_path}")
