"""Tests for UpgradeDB — DB layer for AI Analysis Upgrade service."""
import pytest
import sqlite3
import tempfile
import os
from AI_analysis_upgrade.database import UpgradeDB


def test_create_table_on_init():
    """ai_analysis_v2 table must exist after UpgradeDB init."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        db = UpgradeDB(db_path)
        conn = sqlite3.connect(db_path)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='ai_analysis_v2'"
        )
        assert cursor.fetchone() is not None, "Table ai_analysis_v2 was not created"
        conn.close()
    finally:
        os.unlink(db_path)


def test_write_and_read_v2_prediction():
    """Can write a v2 prediction and read it back."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        db = UpgradeDB(db_path)
        db.save_v2_prediction(
            v1_id=42,
            market_slug="btc-up-test",
            bot_type="5m",
            v1_action="YES",
            v1_confidence=0.72,
            v2_action="NO",
            v2_confidence=0.65,
            v2_reasoning="Hard rule: negative gap",
            trend_score=-0.4,
            p_yes_prior=0.44,
            fear_greed=30,
            gap_usd=-35.0,
            gap_z=-1.8,
            phase="late",
            clob_yes=0.70,
        )
        rows = db.get_unsent_settled_rows()
        assert len(rows) == 0  # not settled yet

        db.mark_outcome(v1_id=42, actual_outcome="NO")
        rows = db.get_unsent_settled_rows()
        assert len(rows) == 1
        row = rows[0]
        assert row["v2_action"] == "NO"
        assert row["v2_correct"] == 1   # v2 predicted NO, outcome is NO
        assert row["v1_correct"] == 0   # v1 predicted YES, outcome is NO
    finally:
        os.unlink(db_path)


def test_get_pending_v1_rows():
    """Can read v1 predictions that haven't been analysed by v2 yet."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        # Seed the ai_analysis_log table (like production would have)
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE ai_analysis_log (
                id INTEGER PRIMARY KEY,
                market_slug TEXT,
                bot_type TEXT,
                action TEXT,
                confidence REAL,
                reasoning TEXT,
                btc_price REAL,
                ptb REAL,
                rsi REAL,
                fired_at TEXT,
                actual_outcome TEXT,
                telegram_sent INTEGER DEFAULT 0
            )
        """)
        conn.execute("""
            INSERT INTO ai_analysis_log
            (id, market_slug, bot_type, action, confidence, btc_price, ptb, fired_at)
            VALUES (1, 'btc-usd-test', '15m', 'YES', 0.74, 95000.0, 94800.0, datetime('now'))
        """)
        conn.commit()
        conn.close()

        db = UpgradeDB(db_path)
        rows = db.get_unanalyzed_v1_rows()
        assert len(rows) == 1
        assert rows[0]["market_slug"] == "btc-usd-test"
    finally:
        os.unlink(db_path)
