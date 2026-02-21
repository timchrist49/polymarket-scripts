"""DB layer for AI Analysis Upgrade service.

Reads from: ai_analysis_log  (production writes this)
Writes to:  ai_analysis_v2   (this service writes this)
"""
import sqlite3
from typing import Optional

from AI_analysis_upgrade import config


_CREATE_V2_TABLE = """
CREATE TABLE IF NOT EXISTS ai_analysis_v2 (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    v1_id           INTEGER NOT NULL,
    market_slug     TEXT NOT NULL,
    bot_type        TEXT NOT NULL,
    v1_action       TEXT,
    v1_confidence   REAL,
    v2_action       TEXT,
    v2_confidence   REAL,
    v2_reasoning    TEXT,
    trend_score     REAL,
    p_yes_prior     REAL,
    fear_greed      INTEGER,
    cpi             REAL,
    volume_spike    REAL,
    funding_rate    REAL,
    gap_usd         REAL,
    gap_z           REAL,
    phase           TEXT,
    clob_yes        REAL,
    actual_outcome  TEXT,
    v1_correct      INTEGER,
    v2_correct      INTEGER,
    comparison_sent INTEGER DEFAULT 0,
    created_at      TEXT DEFAULT (datetime('now'))
)
"""


class UpgradeDB:
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or config.PRODUCTION_DB_PATH
        self._ensure_table()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_table(self) -> None:
        with self._connect() as conn:
            conn.execute(_CREATE_V2_TABLE)

    def get_unanalyzed_v1_rows(self) -> list:
        """Return ai_analysis_log rows that have no ai_analysis_v2 entry yet.

        Deduplicates by market_slug: takes only the latest row per market so
        the production bot's duplicate logging doesn't cause double analysis.
        """
        with self._connect() as conn:
            cursor = conn.execute("""
                SELECT l.*
                FROM ai_analysis_log l
                WHERE l.action IN ('YES', 'NO')
                  AND l.actual_outcome IS NULL
                  AND l.id = (
                      SELECT MAX(l2.id) FROM ai_analysis_log l2
                      WHERE l2.market_slug = l.market_slug
                  )
                  AND NOT EXISTS (
                      SELECT 1 FROM ai_analysis_v2 v
                      WHERE v.market_slug = l.market_slug
                  )
                ORDER BY l.fired_at ASC
                LIMIT 20
            """)
            return [dict(r) for r in cursor.fetchall()]

    def save_v2_prediction(
        self,
        v1_id: int,
        market_slug: str,
        bot_type: str,
        v1_action: str,
        v1_confidence: float,
        v2_action: str,
        v2_confidence: float,
        v2_reasoning: str,
        trend_score: float,
        p_yes_prior: float,
        fear_greed: int,
        gap_usd: float,
        gap_z: float,
        phase: str,
        clob_yes: float,
        cpi: Optional[float] = None,
        volume_spike: Optional[float] = None,
        funding_rate: Optional[float] = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO ai_analysis_v2
                (v1_id, market_slug, bot_type, v1_action, v1_confidence,
                 v2_action, v2_confidence, v2_reasoning, trend_score,
                 p_yes_prior, fear_greed, gap_usd, gap_z, phase, clob_yes,
                 cpi, volume_spike, funding_rate)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                v1_id, market_slug, bot_type, v1_action, v1_confidence,
                v2_action, v2_confidence, v2_reasoning, trend_score,
                p_yes_prior, fear_greed, gap_usd, gap_z, phase, clob_yes,
                cpi, volume_spike, funding_rate,
            ))

    def mark_outcome(self, v1_id: int, actual_outcome: str) -> None:
        """Called when settlement is known; computes v1_correct and v2_correct."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT v1_action, v2_action FROM ai_analysis_v2 WHERE v1_id = ?",
                (v1_id,)
            ).fetchone()
            if not row:
                return
            v1_correct = 1 if row["v1_action"] == actual_outcome else 0
            v2_correct = 1 if row["v2_action"] == actual_outcome else 0
            conn.execute("""
                UPDATE ai_analysis_v2
                SET actual_outcome = ?, v1_correct = ?, v2_correct = ?
                WHERE v1_id = ?
            """, (actual_outcome, v1_correct, v2_correct, v1_id))

    def get_unsent_settled_rows(self) -> list:
        """Return rows that are settled but haven't had a comparison alert sent.

        Deduplicates by market_slug: returns only the earliest row per market
        so duplicate log entries never produce duplicate alerts.
        """
        with self._connect() as conn:
            cursor = conn.execute("""
                SELECT * FROM ai_analysis_v2
                WHERE actual_outcome IS NOT NULL
                  AND comparison_sent = 0
                  AND id = (
                      SELECT MIN(v2.id) FROM ai_analysis_v2 v2
                      WHERE v2.market_slug = ai_analysis_v2.market_slug
                        AND v2.actual_outcome IS NOT NULL
                  )
            """)
            return [dict(r) for r in cursor.fetchall()]

    def mark_comparison_sent(self, v1_id: int) -> None:
        """Mark all v2 rows for this market as sent (handles duplicates)."""
        with self._connect() as conn:
            # Look up market_slug for this v1_id, then mark all rows for that market
            row = conn.execute(
                "SELECT market_slug FROM ai_analysis_v2 WHERE v1_id = ?", (v1_id,)
            ).fetchone()
            if row:
                conn.execute(
                    "UPDATE ai_analysis_v2 SET comparison_sent = 1 WHERE market_slug = ?",
                    (row["market_slug"],)
                )
            else:
                conn.execute(
                    "UPDATE ai_analysis_v2 SET comparison_sent = 1 WHERE v1_id = ?",
                    (v1_id,)
                )

    def get_running_score(self) -> dict:
        """Return running win counts for v1 and v2."""
        with self._connect() as conn:
            cursor = conn.execute("""
                SELECT
                    COUNT(*) as total,
                    SUM(v1_correct) as v1_wins,
                    SUM(v2_correct) as v2_wins
                FROM ai_analysis_v2
                WHERE actual_outcome IS NOT NULL
            """)
            row = cursor.fetchone()
            return dict(row) if row else {"total": 0, "v1_wins": 0, "v2_wins": 0}
