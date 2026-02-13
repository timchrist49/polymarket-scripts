# polymarket/performance/database.py
"""SQLite database for performance tracking."""

import sqlite3
from pathlib import Path
import structlog

logger = structlog.get_logger()


class PerformanceDatabase:
    """SQLite database for storing trade performance data."""

    def __init__(self, db_path: str = "data/performance.db"):
        """
        Initialize database connection and create schema.

        Args:
            db_path: Path to SQLite database file (':memory:' for in-memory)
        """
        self.db_path = db_path

        # Create data directory if needed
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        # Connect to database
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row

        # Create schema
        self._create_tables()
        self._create_indexes()
        self._migrate_schema()

        logger.info("Performance database initialized", db_path=db_path)

    def _create_tables(self):
        """Create database tables."""
        cursor = self.conn.cursor()

        # Trades table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                market_slug TEXT NOT NULL,
                market_id INTEGER,

                -- Decision
                action TEXT NOT NULL,
                confidence REAL NOT NULL,
                position_size REAL NOT NULL,
                reasoning TEXT,

                -- Market Context
                btc_price REAL NOT NULL,
                price_to_beat REAL,
                time_remaining_seconds INTEGER,
                is_end_phase BOOLEAN,

                -- Signals
                social_score REAL,
                market_score REAL,
                final_score REAL,
                final_confidence REAL,
                signal_type TEXT,

                -- Technical Indicators
                rsi REAL,
                macd REAL,
                trend TEXT,

                -- Pricing
                yes_price REAL,
                no_price REAL,
                executed_price REAL,

                -- Outcome (filled after market closes)
                actual_outcome TEXT,
                profit_loss REAL,
                is_win BOOLEAN,
                is_missed_opportunity BOOLEAN
            )
        """)

        # Reflections table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS reflections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                trigger_type TEXT NOT NULL,
                trades_analyzed INTEGER NOT NULL,
                insights TEXT NOT NULL,
                adjustments_made TEXT
            )
        """)

        # Parameter history table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS parameter_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                parameter_name TEXT NOT NULL,
                old_value REAL NOT NULL,
                new_value REAL NOT NULL,
                reason TEXT NOT NULL,
                approval_method TEXT NOT NULL
            )
        """)

        self.conn.commit()

    def _create_indexes(self):
        """Create indexes for fast queries."""
        cursor = self.conn.cursor()

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_trades_timestamp
            ON trades(timestamp)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_trades_signal_type
            ON trades(signal_type)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_trades_is_win
            ON trades(is_win)
        """)

        self.conn.commit()

    def _migrate_schema(self):
        """Add new columns for JIT price fetching metrics and arbitrage tracking."""
        cursor = self.conn.cursor()

        # Get existing columns
        cursor.execute("PRAGMA table_info(trades)")
        existing_columns = {row[1] for row in cursor.fetchall()}

        # Add new columns if they don't exist
        new_columns = [
            ("analysis_price", "REAL"),  # Price from cycle start
            ("price_staleness_seconds", "INTEGER"),  # Time between analysis and execution
            ("price_slippage_pct", "REAL"),  # Percentage change
            ("price_movement_favorable", "BOOLEAN"),  # Was movement favorable?
            ("skipped_unfavorable_move", "BOOLEAN"),  # Was trade skipped due to safety check?
            ("execution_status", "TEXT DEFAULT 'pending'"),  # 'pending', 'executed', 'skipped', 'failed'
            # Arbitrage tracking columns
            ("actual_probability", "REAL"),  # Calculated probability from price momentum
            ("arbitrage_edge", "REAL"),  # Edge percentage over market odds
            ("arbitrage_urgency", "TEXT"),  # 'LOW', 'MEDIUM', 'HIGH'
            ("filled_via", "TEXT"),  # 'market', 'limit', 'limit_partial'
            ("limit_order_timeout", "INTEGER"),  # Timeout used for limit orders (seconds)
        ]

        for column_name, column_type in new_columns:
            if column_name not in existing_columns:
                try:
                    cursor.execute(f"ALTER TABLE trades ADD COLUMN {column_name} {column_type}")
                    logger.info(f"Added column: {column_name}")
                except sqlite3.OperationalError as e:
                    logger.warning(f"Could not add column {column_name}: {e}")

        self.conn.commit()

    def log_trade(self, trade_data: dict) -> int:
        """
        Log a trade decision to the database.

        Args:
            trade_data: Dictionary with trade information

        Returns:
            Trade ID of inserted record
        """
        cursor = self.conn.cursor()

        cursor.execute("""
            INSERT INTO trades (
                timestamp, market_slug, market_id,
                action, confidence, position_size, reasoning,
                btc_price, price_to_beat, time_remaining_seconds, is_end_phase,
                social_score, market_score, final_score, final_confidence, signal_type,
                rsi, macd, trend,
                yes_price, no_price, executed_price,
                analysis_price, price_staleness_seconds, price_slippage_pct,
                price_movement_favorable, skipped_unfavorable_move,
                actual_probability, arbitrage_edge, arbitrage_urgency,
                filled_via, limit_order_timeout
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            trade_data["timestamp"],
            trade_data["market_slug"],
            trade_data.get("market_id"),
            trade_data["action"],
            trade_data["confidence"],
            trade_data["position_size"],
            trade_data.get("reasoning"),
            trade_data["btc_price"],
            trade_data.get("price_to_beat"),
            trade_data.get("time_remaining_seconds"),
            trade_data.get("is_end_phase", False),
            trade_data.get("social_score"),
            trade_data.get("market_score"),
            trade_data.get("final_score"),
            trade_data.get("final_confidence"),
            trade_data.get("signal_type"),
            trade_data.get("rsi"),
            trade_data.get("macd"),
            trade_data.get("trend"),
            trade_data.get("yes_price"),
            trade_data.get("no_price"),
            trade_data.get("executed_price"),
            trade_data.get("analysis_price"),
            trade_data.get("price_staleness_seconds"),
            trade_data.get("price_slippage_pct"),
            trade_data.get("price_movement_favorable"),
            trade_data.get("skipped_unfavorable_move", False),
            trade_data.get("actual_probability"),
            trade_data.get("arbitrage_edge"),
            trade_data.get("arbitrage_urgency"),
            trade_data.get("filled_via"),
            trade_data.get("limit_order_timeout")
        ))

        self.conn.commit()
        trade_id = cursor.lastrowid

        logger.debug("Trade logged", trade_id=trade_id, action=trade_data["action"])
        return trade_id

    def update_outcome(self, market_slug: str, actual_outcome: str, profit_loss: float):
        """
        Update trade outcome after market closes.

        Args:
            market_slug: Market identifier
            actual_outcome: 'UP' or 'DOWN'
            profit_loss: Profit/loss amount (0 for HOLD)
        """
        cursor = self.conn.cursor()

        # Get the trade
        cursor.execute("""
            SELECT id, action, position_size
            FROM trades
            WHERE market_slug = ?
            ORDER BY timestamp DESC
            LIMIT 1
        """, (market_slug,))

        row = cursor.fetchone()
        if not row:
            logger.warning("No trade found for outcome update", market_slug=market_slug)
            return

        trade_id = row['id']
        action = row['action']
        position_size = row['position_size']

        # Determine if win
        is_win = None
        is_missed_opportunity = False

        if action == "HOLD":
            is_win = None  # Didn't trade
            is_missed_opportunity = (position_size == 0)  # Would have won
        elif action == "YES":
            is_win = (actual_outcome == "UP")
        elif action == "NO":
            is_win = (actual_outcome == "DOWN")

        # Update database
        cursor.execute("""
            UPDATE trades
            SET actual_outcome = ?,
                profit_loss = ?,
                is_win = ?,
                is_missed_opportunity = ?
            WHERE id = ?
        """, (actual_outcome, profit_loss, is_win, is_missed_opportunity, trade_id))

        self.conn.commit()

        logger.info(
            "Outcome updated",
            trade_id=trade_id,
            action=action,
            actual_outcome=actual_outcome,
            is_win=is_win,
            profit_loss=profit_loss
        )

    def close(self):
        """Close database connection."""
        self.conn.close()
