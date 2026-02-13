"""Add execution_status field to trades table."""
import sqlite3
from pathlib import Path

def migrate():
    db_path = Path("data/performance.db")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Check if column exists
    cursor.execute("PRAGMA table_info(trades)")
    columns = [col[1] for col in cursor.fetchall()]

    if 'execution_status' not in columns:
        print("Adding execution_status column...")
        cursor.execute("""
            ALTER TABLE trades
            ADD COLUMN execution_status TEXT DEFAULT 'pending'
        """)

        # Set all existing trades with position_size > 0 as 'executed'
        cursor.execute("""
            UPDATE trades
            SET execution_status = 'executed'
            WHERE position_size > 0 AND action IN ('YES', 'NO')
        """)

        conn.commit()
        print("Migration complete!")
    else:
        print("Column already exists, skipping.")

    conn.close()

if __name__ == "__main__":
    migrate()
