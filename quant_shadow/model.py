"""Logistic regression model for quant shadow service.

Trains on quant_shadow_log table (288+ settled rows).
Features: gap_z, clob_yes, trend_score, phase_bin, gap_usd_abs, asset_code,
          realized_vol_per_min, time_remaining_seconds.
Output: P(YES) → edge vs CLOB midpoint → action (YES / NO / HOLD).

Edge formula:
  edge_yes = P(YES) - (clob_yes + SPREAD_HALF)   # cost to buy YES token
  edge_no  = (clob_yes - SPREAD_HALF) - P(YES)   # cost to buy NO token

Execute signal if edge > TAU (3%).

Features:
  gap_usd_abs           — absolute USD gap; prevents inflated gap_z from near-zero vol
  asset_code            — 0=BTC, 1=ETH, 2=SOL, 3=XRP; asset-specific calibration
  realized_vol_per_min  — calibrated vol; directly captures noise floor
  time_remaining_seconds — exact TTE; better than binary phase_bin alone
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

# Asset code mapping: used in both training and inference
ASSET_CODES = {"btc": 0, "eth": 1, "sol": 2, "xrp": 3}
FEATURE_NAMES = [
    "gap_z", "clob_yes", "trend_score", "phase_bin",
    "gap_usd_abs", "asset_code", "realized_vol_per_min", "time_remaining_seconds",
]


def _slug_to_asset_code(slug: str) -> float:
    """Derive asset code from market slug (e.g. 'btc-updown-5m-...' → 0.0)."""
    for asset, code in ASSET_CODES.items():
        if slug.startswith(asset):
            return float(code)
    return 0.0


def _load_training_data(db_path: str = "data/performance.db") -> tuple[np.ndarray, np.ndarray]:
    """Load and prepare training data from quant_shadow_log table.

    Features (8):
      gap_z                 — normalized gap between current price and PTB
      clob_yes              — CLOB midpoint probability of YES
      trend_score           — macro trend signal
      phase_bin             — 1=late, 0=early
      gap_usd_abs           — absolute USD gap (captures noise-floor reliability)
      asset_code            — 0=BTC, 1=ETH, 2=SOL, 3=XRP (asset-specific calibration)
      realized_vol_per_min  — calibrated vol per minute; directly measures noise floor
      time_remaining_seconds — exact time-to-expiry (more precise than binary phase)
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT gap_z, clob_yes, trend_score, phase, actual_outcome,
               gap_usd, realized_vol_per_min, time_remaining_seconds, asset
        FROM quant_shadow_log
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
    for gap_z, clob_yes, trend_score, phase, outcome, gap_usd, rvpm, tte_secs, asset in rows:
        phase_bin = 1.0 if phase == "late" else 0.0
        gap_usd_abs = abs(float(gap_usd)) if gap_usd is not None else 0.0
        asset_code = float(ASSET_CODES.get(asset or "", 0))
        rvpm_val = float(rvpm) if rvpm is not None else 0.0
        tte_val = float(tte_secs) if tte_secs is not None else 120.0
        X.append([
            float(gap_z), float(clob_yes), float(trend_score), phase_bin,
            gap_usd_abs, asset_code, rvpm_val, tte_val,
        ])
        y.append(1 if outcome == "YES" else 0)

    logger.info("Training data loaded", n=len(rows), pos_rate=f"{sum(y)/len(y):.2%}")
    return np.array(X), np.array(y)


def train_and_save(db_path: str = "data/performance.db") -> dict:
    """Train logistic regression on quant_shadow_log data and save to disk.

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

    coef = dict(zip(FEATURE_NAMES, model.coef_[0].tolist()))
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

    # If the saved model has old feature count (4 features), retrain with new 6-feature set
    if data.get("scaler") is not None and hasattr(data["scaler"], "n_features_in_"):
        if data["scaler"].n_features_in_ != len(FEATURE_NAMES):
            logger.info(
                "Saved model has wrong feature count — retraining",
                saved_features=data["scaler"].n_features_in_,
                expected=len(FEATURE_NAMES),
            )
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
    gap_usd_abs: float = 0.0,
    asset_code: int = 0,
    realized_vol_per_min: float = 0.0,
    time_remaining_seconds: int = 120,
) -> tuple[float, str, float]:
    """Predict P(YES), action, and edge.

    Args:
        gap_usd_abs           — absolute USD gap (default 0.0 for backward compat)
        asset_code            — 0=BTC, 1=ETH, 2=SOL, 3=XRP (default 0 for BTC)
        realized_vol_per_min  — calibrated vol; captures noise floor (default 0.0)
        time_remaining_seconds — exact TTE in seconds (default 120)

    Returns:
        p_yes   — model's estimated probability of YES outcome
        action  — "YES", "NO", or "HOLD"
        edge    — positive = YES edge, negative = NO edge magnitude, 0 = HOLD
    """
    phase_bin = 1.0 if phase == "late" else 0.0
    X = np.array([[
        float(gap_z), float(clob_yes), float(trend_score), phase_bin,
        float(gap_usd_abs), float(asset_code),
        float(realized_vol_per_min), float(time_remaining_seconds),
    ]])
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
