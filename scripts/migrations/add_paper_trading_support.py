"""
Database migration: Add paper trading support and signal tracking.

Adds:
1. paper_trades table (mirrors trades schema + signal analysis)
2. Signal tracking columns to trades table
"""

import sqlite3
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from polymarket.config import Settings


def migrate_database(db_path: str):
    """Run migration to add paper trading support."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    print("Starting migration: add_paper_trading_support")

    # Check if paper_trades already exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='paper_trades'")
    if cursor.fetchone():
        print("  paper_trades table already exists, skipping creation")
    else:
        print("  Creating paper_trades table...")
        cursor.execute("""
            CREATE TABLE paper_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                market_id TEXT NOT NULL,
                market_slug TEXT NOT NULL,
                question TEXT,
                action TEXT NOT NULL,
                confidence REAL NOT NULL,
                reasoning TEXT,

                -- Execution details
                executed_price REAL NOT NULL,
                position_size REAL NOT NULL,
                simulated_shares REAL NOT NULL,

                -- Market context
                btc_price_current REAL,
                btc_price_to_beat REAL,
                time_remaining_seconds INTEGER,

                -- Signal analysis
                signal_lag_detected BOOLEAN DEFAULT 0,
                signal_lag_reason TEXT,
                conflict_severity TEXT,
                conflicts_list TEXT,
                odds_yes REAL,
                odds_no REAL,
                odds_qualified BOOLEAN,

                -- Outcome (filled during settlement)
                actual_outcome TEXT,
                is_win BOOLEAN,
                profit_loss REAL,
                settled_at TEXT
            )
        """)
        cursor.execute("CREATE INDEX idx_paper_trades_timestamp ON paper_trades(timestamp)")
        cursor.execute("CREATE INDEX idx_paper_trades_market ON paper_trades(market_slug)")
        print("  ✓ paper_trades table created")

    # Add columns to trades table (use ALTER TABLE)
    new_columns = [
        ("signal_lag_detected", "BOOLEAN DEFAULT 0"),
        ("signal_lag_reason", "TEXT"),
        ("conflict_severity", "TEXT"),
        ("conflicts_list", "TEXT"),
        ("odds_yes", "REAL"),
        ("odds_no", "REAL")
    ]

    for col_name, col_type in new_columns:
        # Check if column exists
        cursor.execute(f"PRAGMA table_info(trades)")
        columns = [row[1] for row in cursor.fetchall()]

        if col_name not in columns:
            print(f"  Adding column trades.{col_name}...")
            cursor.execute(f"ALTER TABLE trades ADD COLUMN {col_name} {col_type}")
            print(f"  ✓ Added trades.{col_name}")
        else:
            print(f"  Column trades.{col_name} already exists, skipping")

    conn.commit()
    conn.close()

    print("Migration complete!")


if __name__ == "__main__":
    settings = Settings()
    db_path = "data/performance.db"

    # Check for worktree database
    worktree_db = Path(".worktrees/bot-loss-fixes-comprehensive/data/performance.db")
    if worktree_db.exists():
        print(f"Migrating worktree database: {worktree_db}")
        migrate_database(str(worktree_db))
        print()

    # Also migrate main database
    main_db = Path(db_path)
    if main_db.exists():
        print(f"Migrating main database: {db_path}")
        migrate_database(str(main_db))
    else:
        print(f"Main database not found: {db_path}")
