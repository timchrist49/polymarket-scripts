"""Logistic regression model for quant shadow service.

Trains on ai_analysis_v2 table (134+ settled rows).
Features: gap_z, clob_yes, trend_score, phase (binary: late=1, early=0).
Output: P(YES) → edge vs CLOB midpoint → action (YES / NO / HOLD).

Edge formula:
  edge_yes = P(YES) - (clob_yes + SPREAD_HALF)   # cost to buy YES token
  edge_no  = (clob_yes - SPREAD_HALF) - P(YES)   # cost to buy NO token

Execute signal if edge > TAU (3%).
"""

import pickle
import sqlite3
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import structlog

logger = structlog.get_logger()

MODEL_PATH = Path("data/quant_model.pkl")
TAU = 0.03          # minimum edge to generate YES/NO signal (3%)
SPREAD_HALF = 0.01  # assumed half-spread cost (1 cent)


def _load_training_data(db_path: str = "data/performance.db") -> tuple[np.ndarray, np.ndarray]:
    """Load and prepare training data from ai_analysis_v2 table."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT gap_z, clob_yes, trend_score, phase, actual_outcome
        FROM ai_analysis_v2
        WHERE actual_outcome IS NOT NULL
          AND gap_z IS NOT NULL
          AND clob_yes IS NOT NULL
          AND trend_score IS NOT NULL
          AND phase IS NOT NULL
    """)
    rows = cursor.fetchall()
    conn.close()

    if len(rows) < 50:
        raise ValueError(f"Insufficient training data: {len(rows)} rows (need ≥50)")

    X, y = [], []
    for gap_z, clob_yes, trend_score, phase, outcome in rows:
        phase_bin = 1.0 if phase == "late" else 0.0
        X.append([float(gap_z), float(clob_yes), float(trend_score), phase_bin])
        y.append(1 if outcome == "YES" else 0)

    logger.info("Training data loaded", n=len(rows), pos_rate=f"{sum(y)/len(y):.2%}")
    return np.array(X), np.array(y)


def train_and_save(db_path: str = "data/performance.db") -> dict:
    """Train logistic regression on ai_analysis_v2 data and save to disk.

    Returns metadata dict: {n_rows, pos_rate, coef}.
    """
    X, y = _load_training_data(db_path)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(C=1.0, solver="liblinear", max_iter=200, random_state=42)
    model.fit(X_scaled, y)

    MODEL_PATH.parent.mkdir(exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": model, "scaler": scaler, "n_rows": len(y)}, f)

    feature_names = ["gap_z", "clob_yes", "trend_score", "phase_bin"]
    coef = dict(zip(feature_names, model.coef_[0].tolist()))
    logger.info("Model trained and saved", n_rows=len(y), pos_rate=f"{y.mean():.2%}", coef=coef)
    return {"n_rows": len(y), "pos_rate": float(y.mean()), "coef": coef}


def load_model(db_path: str = "data/performance.db") -> tuple:
    """Load model from disk, training first if not found.

    Returns (LogisticRegression, StandardScaler).
    """
    if not MODEL_PATH.exists():
        logger.info("Model file not found — training now")
        train_and_save(db_path)

    with open(MODEL_PATH, "rb") as f:
        data = pickle.load(f)

    logger.info("Model loaded", n_rows=data["n_rows"])
    return data["model"], data["scaler"]


def predict(
    model: LogisticRegression,
    scaler: StandardScaler,
    gap_z: float,
    clob_yes: float,
    trend_score: float,
    phase: str,
) -> tuple[float, str, float]:
    """Predict P(YES), action, and edge.

    Returns:
        p_yes   — model's estimated probability of YES outcome
        action  — "YES", "NO", or "HOLD"
        edge    — positive = YES edge, negative = NO edge magnitude, 0 = HOLD
    """
    phase_bin = 1.0 if phase == "late" else 0.0
    X = np.array([[float(gap_z), float(clob_yes), float(trend_score), phase_bin]])
    X_scaled = scaler.transform(X)
    p_yes = float(model.predict_proba(X_scaled)[0][1])

    # edge_yes: what we'd gain buying YES vs cost of YES token
    # edge_no:  what we'd gain buying NO vs cost of NO token
    edge_yes = p_yes - (clob_yes + SPREAD_HALF)
    edge_no = (clob_yes - SPREAD_HALF) - p_yes   # = (1-p_yes) - ((1-clob_yes) + SPREAD_HALF)

    if edge_yes > TAU:
        return p_yes, "YES", edge_yes
    elif edge_no > TAU:
        return p_yes, "NO", -edge_no   # negative to indicate NO direction
    else:
        return p_yes, "HOLD", max(edge_yes, edge_no)
