#!/usr/bin/env python3
"""
Migration: Add fee_paid column to trades table

Polymarket charges a 2% fee on winning trades. This migration adds tracking
for these fees so P/L calculations match actual wallet balance.

Usage:
    python scripts/migrate_add_fee_column.py
    python scripts/migrate_add_fee_column.py --rollback  # Undo migration
"""

import sqlite3
import sys
import shutil
from pathlib import Path
from datetime import datetime

DB_PATH = "data/performance.db"
BACKUP_SUFFIX = ".backup_before_fee_migration"


def backup_database():
    """Create backup before migration."""
    if not Path(DB_PATH).exists():
        print(f"❌ Database not found: {DB_PATH}")
        sys.exit(1)

    backup_path = f"{DB_PATH}{BACKUP_SUFFIX}"
    shutil.copy2(DB_PATH, backup_path)
    print(f"✅ Backup created: {backup_path}")
    return backup_path


def migrate_up():
    """Add fee_paid column and set historical fees."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        # Check if column already exists
        cursor.execute("PRAGMA table_info(trades)")
        columns = {row[1] for row in cursor.fetchall()}

        if 'fee_paid' in columns:
            print("⚠️  fee_paid column already exists.")

            # Check if migration was fully completed (profit_loss already adjusted)
            cursor.execute("""
                SELECT COUNT(*)
                FROM trades
                WHERE is_win = 1
                  AND profit_loss > 0
                  AND fee_paid > 0
            """)
            trades_with_fees = cursor.fetchone()[0]

            if trades_with_fees > 0:
                print("⚠️  Migration already completed (fees found). Skipping to prevent double-adjustment.")
                return
            else:
                print("⚠️  Column exists but no fees found. This shouldn't happen - migration may have failed previously.")
                print("⚠️  Skipping to avoid potential data corruption. Please investigate manually.")
                return

        # Add fee_paid column
        print("Adding fee_paid column...")
        cursor.execute("""
            ALTER TABLE trades
            ADD COLUMN fee_paid REAL DEFAULT 0.0
        """)

        # Calculate and backfill fees for historical winning trades
        # Fee = 2% of GROSS profit (current profit_loss is gross)
        print("Backfilling historical fees for winning trades...")
        cursor.execute("""
            UPDATE trades
            SET fee_paid = ROUND(profit_loss * 0.02, 2)
            WHERE is_win = 1
              AND profit_loss > 0
        """)

        fee_rows = cursor.rowcount

        # CRITICAL: Adjust profit_loss to be net of fees
        # Current profit_loss values are GROSS, we need to make them NET
        print("Adjusting profit_loss to be net of fees...")
        cursor.execute("""
            UPDATE trades
            SET profit_loss = ROUND(profit_loss - fee_paid, 2)
            WHERE is_win = 1
              AND fee_paid > 0
        """)

        adjusted_rows = cursor.rowcount
        conn.commit()

        print(f"✅ Migration complete!")
        print(f"   - Added fee_paid column")
        print(f"   - Backfilled {fee_rows} winning trades with 2% fees")
        print(f"   - Adjusted {adjusted_rows} profit_loss values to net of fees")

    except Exception as e:
        conn.rollback()
        print(f"❌ Migration failed: {e}")
        sys.exit(1)
    finally:
        conn.close()


def migrate_down():
    """Remove fee_paid column (rollback)."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        # Check if column exists
        cursor.execute("PRAGMA table_info(trades)")
        columns = {row[1] for row in cursor.fetchall()}

        if 'fee_paid' not in columns:
            print("⚠️  fee_paid column doesn't exist. Nothing to rollback.")
            return

        # CRITICAL: Restore gross profit_loss before removing fee_paid
        print("Restoring gross profit_loss values (adding fees back)...")
        cursor.execute("""
            UPDATE trades
            SET profit_loss = ROUND(profit_loss + fee_paid, 2)
            WHERE is_win = 1
              AND fee_paid > 0
        """)
        restored_rows = cursor.rowcount
        print(f"   - Restored {restored_rows} profit_loss values to gross")

        # SQLite doesn't support DROP COLUMN easily, so we recreate the table
        print("Rolling back fee_paid column (recreating table)...")

        # Get current schema without fee_paid
        cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='trades'")
        original_schema = cursor.fetchone()[0]

        # Create temp table without fee_paid
        cursor.execute("ALTER TABLE trades RENAME TO trades_old")

        # Recreate trades table (get all columns except fee_paid)
        cursor.execute("PRAGMA table_info(trades_old)")
        columns = [row[1] for row in cursor.fetchall() if row[1] != 'fee_paid']
        columns_str = ', '.join(columns)

        # Copy schema from old table but exclude fee_paid
        cursor.execute(f"""
            CREATE TABLE trades AS
            SELECT {columns_str}
            FROM trades_old
        """)

        # Drop old table
        cursor.execute("DROP TABLE trades_old")

        conn.commit()
        print("✅ Rollback complete! fee_paid column removed.")

    except Exception as e:
        conn.rollback()
        print(f"❌ Rollback failed: {e}")
        sys.exit(1)
    finally:
        conn.close()


def main():
    """Run migration."""
    if "--rollback" in sys.argv:
        print("🔄 Rolling back migration...")
        backup_path = backup_database()
        migrate_down()
        print(f"\n💡 Backup available at: {backup_path}")
    else:
        print("🚀 Running migration: Add fee_paid column")
        backup_path = backup_database()
        migrate_up()
        print(f"\n💡 Backup available at: {backup_path}")


if __name__ == "__main__":
    main()
