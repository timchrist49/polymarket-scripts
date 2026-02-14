#!/usr/bin/env python3
"""
Migration: Fix execution_status for existing trades.

Problem: All 192 existing trades have execution_status='pending' because
update_execution_metrics() never set it to 'executed'.

Solution: Mark trades as 'executed' if they have filled_via set (indicating
they were actually filled). Mark trades as 'skipped' if they have
skipped_unfavorable_move=True.
"""

import sqlite3
from pathlib import Path

def migrate_execution_status(db_path: str = "data/performance.db"):
    """Fix execution_status for existing trades."""

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    print("=" * 60)
    print("Migration: Fix execution_status for existing trades")
    print("=" * 60)

    # Check current state
    cursor.execute("""
        SELECT
            COUNT(*) as total,
            execution_status,
            COUNT(CASE WHEN filled_via IS NOT NULL THEN 1 END) as has_filled_via
        FROM trades
        WHERE action IN ('YES', 'NO')
        GROUP BY execution_status
    """)

    print("\nBEFORE migration:")
    print("-" * 60)
    for row in cursor.fetchall():
        print(f"  Status: {row[1] or 'NULL':12} | Total: {row[0]:3} | Has filled_via: {row[2]:3}")

    # Migration 1: Mark as 'executed' if filled_via is set
    cursor.execute("""
        UPDATE trades
        SET execution_status = 'executed'
        WHERE action IN ('YES', 'NO')
          AND execution_status = 'pending'
          AND filled_via IS NOT NULL
    """)

    executed_count = cursor.rowcount
    print(f"\n✅ Marked {executed_count} trades as 'executed' (had filled_via)")

    # Migration 2: Mark as 'skipped' if skipped_unfavorable_move=True
    cursor.execute("""
        UPDATE trades
        SET execution_status = 'skipped'
        WHERE action IN ('YES', 'NO')
          AND execution_status = 'pending'
          AND skipped_unfavorable_move = 1
    """)

    skipped_count = cursor.rowcount
    print(f"✅ Marked {skipped_count} trades as 'skipped' (had skipped_unfavorable_move)")

    conn.commit()

    # Check final state
    cursor.execute("""
        SELECT
            COUNT(*) as total,
            execution_status
        FROM trades
        WHERE action IN ('YES', 'NO')
        GROUP BY execution_status
        ORDER BY execution_status
    """)

    print("\nAFTER migration:")
    print("-" * 60)
    for row in cursor.fetchall():
        print(f"  Status: {row[1] or 'NULL':12} | Total: {row[0]:3}")

    # Show trades ready for settlement
    cursor.execute("""
        SELECT COUNT(*)
        FROM trades
        WHERE action IN ('YES', 'NO')
          AND execution_status = 'executed'
          AND is_win IS NULL
          AND datetime(timestamp) < datetime('now', '-15 minutes')
    """)

    settleable = cursor.fetchone()[0]
    print(f"\n📊 Trades ready for settlement: {settleable}")

    conn.close()

    print("\n" + "=" * 60)
    print("✅ Migration complete!")
    print("=" * 60)


if __name__ == "__main__":
    # Change to project root
    import os
    os.chdir(Path(__file__).parent.parent)

    migrate_execution_status()
