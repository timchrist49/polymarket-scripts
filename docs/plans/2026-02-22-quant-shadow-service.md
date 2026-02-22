# Quant Shadow Service — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a shadow (non-trading) service that monitors the same 8 markets as V2, computes quantitative ML features at analysis time, makes logistic-regression-based predictions with explicit edge calculation, and sends Telegram alerts comparing quant vs V2 after settlement.

**Architecture:** A `quant_shadow/` module provides model training and prediction. `scripts/quant_shadow.py` runs 8 async `QuantShadowMonitor` coroutines (one per market), fires at the same trigger times as V2 (T+3min for 5m, T+12min for 15m), logs to a new `quant_shadow_log` table, and sends comparison alerts. No orders are placed.

**Tech Stack:** Python 3.12, asyncio, scikit-learn 1.8 (LogisticRegression + StandardScaler), sqlite3 (existing `data/performance.db`), structlog, existing `PolymarketClient`, `AssetPriceService`, `BTCPriceService`, `AssetTrendAnalyzer`, `TelegramBot`.

---

## Task 1: Install sklearn and add `quant_shadow_log` table to database

**Files:**
- Modify: `polymarket/performance/database.py` (add table creation + 2 methods)

### Step 1: Verify sklearn is installed

```bash
cd /root/polymarket-scripts
python3 -c "import sklearn; print(sklearn.__version__)"
```

Expected: `1.8.0` (already installed in prior step — if not, run `pip install scikit-learn --break-system-packages`)

### Step 2: Add table creation and migration method to `_create_tables()`

In `polymarket/performance/database.py`, find `_create_tables()`. After the `ai_analysis_log` table creation (search for `CREATE TABLE IF NOT EXISTS ai_analysis_log`), add:

```python
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS quant_shadow_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                market_slug TEXT NOT NULL,
                market_id TEXT,
                asset TEXT NOT NULL,
                timeframe TEXT NOT NULL,

                -- Features at analysis time
                gap_usd REAL,
                gap_z REAL,
                clob_yes REAL,
                realized_vol_per_min REAL,
                trend_score REAL,
                time_remaining_seconds INTEGER,
                phase TEXT,

                -- Quant model output
                quant_p_yes REAL,
                quant_action TEXT,
                quant_edge REAL,

                -- V2 comparison (filled at settlement time)
                v2_action TEXT,
                v2_confidence REAL,

                -- Outcome (filled at settlement)
                actual_outcome TEXT,
                quant_correct INTEGER,
                v2_correct INTEGER,

                -- Metadata
                fired_at TEXT NOT NULL,
                telegram_sent INTEGER DEFAULT 0
            )
        """)
```

Also add an index after the ai_analysis_log index block:

```python
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_quant_shadow_slug
            ON quant_shadow_log(market_slug)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_quant_shadow_unsettled
            ON quant_shadow_log(actual_outcome)
            WHERE actual_outcome IS NULL
        """)
```

### Step 3: Add `log_quant_shadow()` method to `PerformanceDatabase`

After the `mark_ai_alerts_sent()` method (around line 549), add:

```python
    def log_quant_shadow(
        self,
        market_slug: str,
        market_id: str | None,
        asset: str,
        timeframe: str,
        gap_usd: float,
        gap_z: float,
        clob_yes: float,
        realized_vol_per_min: float,
        trend_score: float,
        time_remaining_seconds: int,
        phase: str,
        quant_p_yes: float,
        quant_action: str,
        quant_edge: float,
    ) -> int:
        """Insert one quant shadow prediction row. Returns inserted id."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO quant_shadow_log
                (market_slug, market_id, asset, timeframe,
                 gap_usd, gap_z, clob_yes, realized_vol_per_min,
                 trend_score, time_remaining_seconds, phase,
                 quant_p_yes, quant_action, quant_edge, fired_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                market_slug, market_id, asset, timeframe,
                gap_usd, gap_z, clob_yes, realized_vol_per_min,
                trend_score, time_remaining_seconds, phase,
                quant_p_yes, quant_action, quant_edge,
                datetime.utcnow().isoformat(),
            ),
        )
        self.conn.commit()
        return cursor.lastrowid

    def settle_quant_shadow(
        self,
        row_id: int,
        actual_outcome: str,
        quant_correct: int | None,
        v2_action: str | None,
        v2_confidence: float | None,
        v2_correct: int | None,
    ) -> None:
        """Update quant shadow row with settlement data."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            UPDATE quant_shadow_log
            SET actual_outcome=?, quant_correct=?, v2_action=?,
                v2_confidence=?, v2_correct=?, telegram_sent=1
            WHERE id=?
            """,
            (actual_outcome, quant_correct, v2_action, v2_confidence, v2_correct, row_id),
        )
        self.conn.commit()
```

### Step 4: Verify the table creates correctly

```bash
cd /root/polymarket-scripts
python3 -c "
from polymarket.performance.database import PerformanceDatabase
db = PerformanceDatabase('data/performance.db')
cur = db.conn.cursor()
cur.execute(\"SELECT name FROM sqlite_master WHERE name='quant_shadow_log'\")
print('Table created:', cur.fetchone())
"
```

Expected: `Table created: ('quant_shadow_log',)`

### Step 5: Commit

```bash
cd /root/polymarket-scripts
git add polymarket/performance/database.py
git commit -m "feat(quant-shadow): add quant_shadow_log table + log/settle methods"
```

---

## Task 2: Build `quant_shadow/model.py` — logistic regression trainer + predictor

**Files:**
- Create: `quant_shadow/__init__.py`
- Create: `quant_shadow/model.py`

### Step 1: Create `quant_shadow/__init__.py`

```python
"""Quant Shadow — logistic regression model for prediction market edge estimation."""
```

### Step 2: Create `quant_shadow/model.py`

```python
"""Logistic regression model for quant shadow service.

Trains on ai_analysis_v2 table (134+ settled rows).
Features: gap_z, clob_yes, trend_score, phase (binary).
Output: P(YES) → edge vs CLOB midpoint → action (YES / NO / HOLD).

Edge formula:
  edge_yes = P(YES) - (clob_yes + SPREAD_HALF)   # cost to buy YES token
  edge_no  = (clob_yes - SPREAD_HALF) - P(YES)   # cost to buy NO token (= 1 - clob)
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
        edge    — positive = YES edge, negative = NO edge, 0 = HOLD
    """
    phase_bin = 1.0 if phase == "late" else 0.0
    X = np.array([[float(gap_z), float(clob_yes), float(trend_score), phase_bin]])
    X_scaled = scaler.transform(X)
    p_yes = float(model.predict_proba(X_scaled)[0][1])

    # Cost of buying YES = clob_yes + spread; cost of buying NO = (1-clob_yes) + spread
    edge_yes = p_yes - (clob_yes + SPREAD_HALF)
    edge_no = (1.0 - p_yes) - ((1.0 - clob_yes) + SPREAD_HALF)
    # edge_no can be rewritten as: (clob_yes - SPREAD_HALF) - p_yes

    if edge_yes > TAU:
        return p_yes, "YES", edge_yes
    elif edge_no > TAU:
        return p_yes, "NO", -edge_no   # negative to indicate NO direction
    else:
        return p_yes, "HOLD", max(edge_yes, edge_no)
```

### Step 3: Verify model trains successfully

```bash
cd /root/polymarket-scripts
python3 -c "
from quant_shadow.model import train_and_save
result = train_and_save()
print('Training result:', result)
"
```

Expected output (approximate):
```
Training result: {'n_rows': 134, 'pos_rate': 0.37..., 'coef': {'gap_z': ..., 'clob_yes': ..., ...}}
```

### Step 4: Verify prediction works

```bash
python3 -c "
from quant_shadow.model import load_model, predict
model, scaler = load_model()
# Test: clear YES signal (gap_z=3, clob says 55%)
p, action, edge = predict(model, scaler, gap_z=3.0, clob_yes=0.55, trend_score=0.3, phase='late')
print(f'p_yes={p:.2f}, action={action}, edge={edge:+.3f}')
# Test: clear NO signal (gap_z=-3, clob says 55% YES)
p, action, edge = predict(model, scaler, gap_z=-3.0, clob_yes=0.55, trend_score=-0.3, phase='late')
print(f'p_yes={p:.2f}, action={action}, edge={edge:+.3f}')
# Test: ambiguous (gap_z=0.2)
p, action, edge = predict(model, scaler, gap_z=0.2, clob_yes=0.5, trend_score=0.0, phase='early')
print(f'p_yes={p:.2f}, action={action}, edge={edge:+.3f}')
"
```

Expected: first call → action=YES with positive edge, second → action=NO with negative edge, third → action=HOLD.

### Step 5: Commit

```bash
cd /root/polymarket-scripts
git add quant_shadow/__init__.py quant_shadow/model.py
git commit -m "feat(quant-shadow): logistic regression model — train/predict with edge calculation"
```

---

## Task 3: Build `scripts/quant_shadow.py` — the shadow service

**Files:**
- Create: `scripts/quant_shadow.py`

### Step 1: Write the full service file

```python
#!/usr/bin/env python3
"""
Quant Shadow Service

Monitors 8 markets (BTC/ETH/SOL/XRP × 5m/15m) in pure shadow mode.
No orders placed. At analysis time fires logistic-regression prediction,
logs features + edge to quant_shadow_log. After settlement sends Telegram
comparing quant prediction vs V2 decision vs actual outcome.

Usage:
    python3 -u scripts/quant_shadow.py
"""

import asyncio
import os
import re
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import structlog

sys.path.insert(0, str(Path(__file__).parent.parent))

from polymarket.client import PolymarketClient, ASSET_COINGECKO_IDS
from polymarket.config import Settings
from polymarket.models import Market
from polymarket.performance.database import PerformanceDatabase
from polymarket.telegram.bot import TelegramBot
from polymarket.trading.btc_price import BTCPriceService
from polymarket.trading.asset_price_service import AssetPriceService
from AI_analysis_upgrade.trend_analyzer import AssetTrendAnalyzer
from AI_analysis_upgrade.ai_analysis_v2 import compute_gap_z, get_phase
from quant_shadow.model import load_model, predict as quant_predict

logger = structlog.get_logger()

# ── Config ──────────────────────────────────────────────────────────────────

MARKETS = [
    ("btc", "5m"), ("btc", "15m"),
    ("eth", "5m"), ("eth", "15m"),
    ("sol", "5m"), ("sol", "15m"),
    ("xrp", "5m"), ("xrp", "15m"),
]
DURATION = {"5m": 300, "15m": 900}
AI_TRIGGER_ELAPSED = {"5m": 180, "15m": 720}
SETTLEMENT_CHECK_INTERVAL = 10   # seconds between close-price retry attempts
SETTLEMENT_MAX_ATTEMPTS = 12     # up to 2 minutes of retries
POST_MARKET_BUFFER = 30
DISCOVERY_RETRY_INTERVAL = 15


# ── Per-market monitor ───────────────────────────────────────────────────────

class QuantShadowMonitor:
    """Monitors one (asset, timeframe) pair and fires quant prediction at trigger time."""

    def __init__(
        self,
        asset: str,
        timeframe: str,
        client: PolymarketClient,
        btc_service: BTCPriceService,
        asset_services: dict,
        trend_analyzers: dict,
        db: PerformanceDatabase,
        telegram: Optional[TelegramBot],
        model,
        scaler,
    ):
        self.asset = asset
        self.timeframe = timeframe
        self.duration = DURATION[timeframe]
        self.ai_trigger_elapsed = AI_TRIGGER_ELAPSED[timeframe]
        self.label = f"{asset.upper()}-{timeframe}"

        self.client = client
        self.btc_service = btc_service
        self.asset_services = asset_services
        self.trend_analyzers = trend_analyzers
        self.db = db
        self.telegram = telegram
        self.model = model
        self.scaler = scaler

        self._analyzed_markets: set[str] = set()

    # ── Price helpers ────────────────────────────────────────────────────────

    async def _get_current_price(self) -> Optional[float]:
        try:
            if self.asset == "btc":
                data = await self.btc_service.get_current_price()
                return float(data.price) if data else None
            svc = self.asset_services.get(self.asset)
            return await svc.get_current_price() if svc else None
        except Exception as e:
            logger.warning(f"{self.label} current price failed", error=str(e))
            return None

    async def _get_price_at(self, ts: float) -> Optional[float]:
        try:
            if self.asset == "btc":
                r = await self.btc_service.get_price_at_timestamp(int(ts))
                return float(r) if r else None
            svc = self.asset_services.get(self.asset)
            return await svc.get_price_at_timestamp(ts) if svc else None
        except Exception as e:
            logger.warning(f"{self.label} price_at failed", error=str(e))
            return None

    def _get_realized_vol(self) -> float:
        svc = self.asset_services.get(self.asset)
        if svc:
            v = svc.get_realized_vol_per_min()
            if v > 0:
                return v
        return 15.0 if self.asset == "btc" else 5.0

    def _get_clob_yes(self, market: Market) -> float:
        try:
            token_ids = market.get_token_ids()
            if token_ids:
                book = self.client.get_orderbook(token_ids[0], depth=5)
                if book:
                    bids = book.get("bids", [])
                    asks = book.get("asks", [])
                    if bids and asks:
                        return (float(bids[0][0]) + float(asks[0][0])) / 2.0
        except Exception:
            pass
        return float(market.best_bid) if market.best_bid else 0.5

    async def _get_trend_score(self) -> float:
        try:
            analyzer = self.trend_analyzers.get(self.asset)
            if analyzer:
                result = await asyncio.to_thread(analyzer.analyze)
                return float(result.trend_score) if result else 0.0
        except Exception:
            pass
        return 0.0

    # ── Main lifecycle ───────────────────────────────────────────────────────

    async def run(self) -> None:
        """Main loop: discover market → predict → settle → repeat."""
        while True:
            try:
                await self._run_one_market()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"{self.label} run loop error", error=str(e))
                await asyncio.sleep(DISCOVERY_RETRY_INTERVAL)

    async def _discover_market(self) -> Optional[Market]:
        while True:
            try:
                market = await asyncio.to_thread(
                    self.client.discover_market, self.asset, self.timeframe
                )
                return market
            except Exception as e:
                logger.warning(f"{self.label} discovery failed", error=str(e))
                await asyncio.sleep(DISCOVERY_RETRY_INTERVAL)

    async def _run_one_market(self) -> None:
        market = await self._discover_market()
        if market is None:
            return

        # Derive timestamps (end_date is reliable; start_date may be stale)
        if market.end_date:
            market_end_ts = market.end_date.timestamp()
            market_start_ts = market_end_ts - self.duration
        elif market.start_date:
            market_start_ts = market.start_date.timestamp()
            market_end_ts = market_start_ts + self.duration
        else:
            market_start_ts = time.time()
            market_end_ts = market_start_ts + self.duration

        ai_fire_ts = market_start_ts + self.ai_trigger_elapsed
        now = time.time()

        # Sleep until trigger time
        if now < ai_fire_ts:
            wait = ai_fire_ts - now
            logger.info(
                f"{self.label} waiting for trigger",
                fires_in=f"{wait:.0f}s",
                ends_in=f"{market_end_ts - now:.0f}s",
                slug=market.slug,
            )
            await asyncio.sleep(wait)

        # Skip if already analyzed this market
        if market.id in self._analyzed_markets:
            await asyncio.sleep(max(0, market_end_ts - time.time() + POST_MARKET_BUFFER))
            return

        # Gather features
        now = time.time()
        time_remaining = int(market_end_ts - now)
        if time_remaining <= 0:
            self._analyzed_markets.add(market.id)
            return

        current_price = await self._get_current_price()
        ptb = await self._get_price_at(market_start_ts)
        if current_price is None or ptb is None:
            logger.warning(f"{self.label} missing price data, skipping prediction")
            return

        vol = self._get_realized_vol()
        gap_usd = current_price - ptb
        gap_z = compute_gap_z(gap_usd, vol, time_remaining / 60.0)
        phase = get_phase(time_remaining, self.duration)
        clob_yes = await asyncio.to_thread(self._get_clob_yes, market)
        trend_score = await self._get_trend_score()

        # Quant prediction
        p_yes, action, edge = quant_predict(self.model, self.scaler, gap_z, clob_yes, trend_score, phase)

        logger.info(
            f"{self.label} QUANT prediction",
            slug=market.slug,
            gap_z=f"{gap_z:+.2f}",
            clob=f"{clob_yes:.2f}",
            trend=f"{trend_score:+.2f}",
            phase=phase,
            p_yes=f"{p_yes:.2f}",
            action=action,
            edge=f"{edge:+.3f}",
        )

        # Persist to DB
        row_id = self.db.log_quant_shadow(
            market_slug=market.slug or market.id,
            market_id=market.id,
            asset=self.asset,
            timeframe=self.timeframe,
            gap_usd=gap_usd,
            gap_z=gap_z,
            clob_yes=clob_yes,
            realized_vol_per_min=vol,
            trend_score=trend_score,
            time_remaining_seconds=time_remaining,
            phase=phase,
            quant_p_yes=p_yes,
            quant_action=action,
            quant_edge=edge,
        )

        self._analyzed_markets.add(market.id)

        # Wait for market to close (+ grace period)
        sleep_secs = max(5, market_end_ts - time.time() + 15)
        await asyncio.sleep(sleep_secs)

        # Settle and alert
        await self._settle_and_alert(
            row_id, market, ptb, p_yes, action, edge,
            gap_z, clob_yes, vol, trend_score, time_remaining, phase, market_end_ts,
        )

        await asyncio.sleep(POST_MARKET_BUFFER)

    async def _settle_and_alert(
        self, row_id, market, ptb, p_yes, action, edge,
        gap_z, clob_yes, vol, trend_score, tte, phase, market_end_ts,
    ) -> None:
        """Fetch close price, compute outcome, look up V2 decision, send Telegram."""
        close_price = None
        for _ in range(SETTLEMENT_MAX_ATTEMPTS):
            close_price = await self._get_price_at(market_end_ts)
            if close_price is not None:
                break
            await asyncio.sleep(SETTLEMENT_CHECK_INTERVAL)

        if close_price is None:
            logger.warning(f"{self.label} settlement failed: close price unavailable", slug=market.slug)
            return

        actual_outcome = "YES" if close_price > ptb else "NO"
        quant_correct = None
        if action != "HOLD":
            quant_correct = 1 if action == actual_outcome else 0

        # Look up V2's decision for this market
        v2_action = v2_confidence = v2_correct = None
        try:
            cursor = self.db.conn.cursor()
            cursor.execute(
                """
                SELECT action, confidence FROM ai_analysis_log
                WHERE market_slug = ? AND bot_type LIKE 'v2_%'
                ORDER BY id DESC LIMIT 1
                """,
                (market.slug or market.id,),
            )
            row = cursor.fetchone()
            if row:
                v2_action = row[0]
                v2_confidence = row[1]
                v2_correct = 1 if v2_action == actual_outcome else 0
        except Exception as e:
            logger.warning(f"{self.label} V2 lookup failed", error=str(e))

        # Persist settlement
        self.db.settle_quant_shadow(row_id, actual_outcome, quant_correct, v2_action, v2_confidence, v2_correct)

        logger.info(
            f"{self.label} settled",
            actual=actual_outcome,
            quant_action=action,
            quant_correct=quant_correct,
            v2_action=v2_action,
            v2_correct=v2_correct,
            close_price=f"{close_price:,.0f}",
            ptb=f"{ptb:,.0f}",
        )

        # Telegram alert
        await self._send_telegram_alert(
            market.slug or market.id, actual_outcome, close_price, ptb,
            p_yes, action, edge, quant_correct,
            v2_action, v2_confidence, v2_correct,
            gap_z, clob_yes, vol, trend_score, tte, phase,
        )

    async def _send_telegram_alert(
        self,
        slug, actual_outcome, close_price, ptb,
        p_yes, action, edge, quant_correct,
        v2_action, v2_conf, v2_correct,
        gap_z, clob_yes, vol, trend_score, tte, phase,
    ) -> None:
        if not self.telegram:
            return

        def result_icon(pred_action, correct):
            if pred_action is None or pred_action == "HOLD":
                return "⏭️"
            return "✅" if correct else "❌"

        actual_icon = "✅" if actual_outcome == "YES" else "❌"
        quant_icon = result_icon(action, quant_correct)
        v2_icon = result_icon(v2_action, v2_correct)

        edge_str = f"{edge:+.1%}" if action != "HOLD" else "N/A"
        v2_conf_str = f"{v2_conf:.0%}" if v2_conf else "N/A"

        asset_sym = self.asset.upper()
        msg = (
            f"🔬 Quant Shadow: {self.label} settled\n"
            f"{'─' * 32}\n"
            f"Outcome: {actual_icon} {actual_outcome} "
            f"({asset_sym} ${close_price:,.0f} vs PTB ${ptb:,.0f})\n\n"
            f"Quant: {action} (p={p_yes:.2f}, edge={edge_str}) {quant_icon}\n"
            f"V2:    {v2_action or 'N/A'} ({v2_conf_str}) {v2_icon}\n\n"
            f"Features:\n"
            f"  gap_z={gap_z:+.2f} | clob={clob_yes:.2f} | vol={vol:.1f}/min\n"
            f"  trend={trend_score:+.2f} | phase={phase} | tte={tte}s"
        )

        try:
            await self.telegram.send_message(msg)
        except Exception as e:
            logger.warning(f"{self.label} Telegram alert failed", error=str(e))


# ── Settlement loop ──────────────────────────────────────────────────────────

async def settlement_loop(db: PerformanceDatabase) -> None:
    """Periodically log quant shadow accuracy stats."""
    while True:
        await asyncio.sleep(600)  # every 10 minutes
        try:
            cursor = db.conn.cursor()
            cursor.execute("""
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN quant_action != 'HOLD' THEN 1 ELSE 0 END) as acted,
                    SUM(CASE WHEN quant_correct = 1 THEN 1 ELSE 0 END) as quant_wins,
                    SUM(CASE WHEN v2_correct = 1 THEN 1 ELSE 0 END) as v2_wins
                FROM quant_shadow_log
                WHERE actual_outcome IS NOT NULL
            """)
            r = cursor.fetchone()
            if r and r[0] > 0:
                logger.info(
                    "Quant shadow stats",
                    total_settled=r[0],
                    acted=r[1],
                    quant_wins=r[2],
                    v2_wins=r[3],
                    quant_win_rate=f"{(r[2] or 0) / max(r[1], 1):.1%}",
                    v2_win_rate=f"{(r[3] or 0) / r[0]:.1%}",
                )
        except Exception as e:
            logger.warning("Stats query failed", error=str(e))


# ── Entry point ──────────────────────────────────────────────────────────────

async def main() -> None:
    logger.info("Quant Shadow Service starting", markets=len(MARKETS))

    # Shared resources
    settings = Settings()
    client = PolymarketClient()
    db = PerformanceDatabase()

    telegram = None
    try:
        telegram = TelegramBot(settings)
        await telegram.send_message("🔬 Quant Shadow Service started — monitoring 8 markets")
    except Exception:
        logger.warning("Telegram not available")

    # Load logistic regression model (trains if not found)
    model, scaler = load_model()

    # Price services
    btc_service = BTCPriceService(settings=settings)
    asset_services: dict[str, AssetPriceService] = {}
    for asset, _ in MARKETS:
        if asset not in asset_services and asset != "btc":
            cg_id = ASSET_COINGECKO_IDS.get(asset, asset)
            asset_services[asset] = AssetPriceService(asset_key=asset, coingecko_id=cg_id, settings=settings)
    # BTC also needs an AssetPriceService for vol calculation
    asset_services["btc"] = AssetPriceService(asset_key="btc", coingecko_id="bitcoin", settings=settings)

    # Start price services
    for svc in asset_services.values():
        asyncio.create_task(svc.start())
    asyncio.create_task(btc_service.start())

    # Trend analyzers
    trend_analyzers: dict[str, AssetTrendAnalyzer] = {}
    for asset, _ in MARKETS:
        if asset not in trend_analyzers:
            cg_id = ASSET_COINGECKO_IDS.get(asset, asset)
            trend_analyzers[asset] = AssetTrendAnalyzer(
                coingecko_id=cg_id,
                coingecko_api_key=getattr(settings, "COINGECKO_API_KEY", None),
            )

    # Give price services 5s to warm up
    await asyncio.sleep(5)

    # Launch 8 market monitors + stats loop
    tasks = [
        asyncio.create_task(
            QuantShadowMonitor(
                asset=asset,
                timeframe=tf,
                client=client,
                btc_service=btc_service,
                asset_services=asset_services,
                trend_analyzers=trend_analyzers,
                db=db,
                telegram=telegram,
                model=model,
                scaler=scaler,
            ).run(),
            name=f"quant-{asset}-{tf}",
        )
        for asset, tf in MARKETS
    ]
    tasks.append(asyncio.create_task(settlement_loop(db), name="stats"))

    # Graceful shutdown
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, lambda: [t.cancel() for t in tasks])

    logger.info("Quant Shadow running", n_monitors=len(MARKETS))
    try:
        await asyncio.gather(*tasks, return_exceptions=True)
    finally:
        logger.info("Quant Shadow shutting down")
        if telegram:
            try:
                await telegram.send_message("🔬 Quant Shadow Service stopped")
            except Exception:
                pass


if __name__ == "__main__":
    asyncio.run(main())
```

### Step 2: Syntax check

```bash
cd /root/polymarket-scripts
python3 -c "import ast; ast.parse(open('scripts/quant_shadow.py').read()); print('Syntax OK')"
```

Expected: `Syntax OK`

### Step 3: Dry import check

```bash
python3 -c "
import sys
sys.path.insert(0, '.')
# Just import the modules used (not run main)
from quant_shadow.model import load_model, predict
from polymarket.performance.database import PerformanceDatabase
print('All imports OK')
"
```

Expected: `All imports OK`

### Step 4: Commit

```bash
cd /root/polymarket-scripts
git add scripts/quant_shadow.py
git commit -m "feat(quant-shadow): main shadow service — 8-market monitor + Telegram alerts"
```

---

## Task 4: Start the service and verify it runs

### Step 1: Start the service in background

```bash
cd /root/polymarket-scripts
source .env 2>/dev/null || true
python3 -u scripts/quant_shadow.py > /tmp/quant_shadow.log 2>&1 &
QUANT_PID=$!
echo "Quant Shadow PID: $QUANT_PID"
sleep 5
tail -20 /tmp/quant_shadow.log
```

### Step 2: Verify expected startup lines appear

Expected log lines (may vary by order):
```
Model loaded n_rows=134
Quant Shadow Service starting markets=8
BTC-5m waiting for trigger fires_in=...s ends_in=...s
BTC-15m waiting for trigger fires_in=...s ends_in=...s
...
```

**If you see `fires_in=0s, ends_in=0s`**: the `.pyc` cache may be stale. Run:
```bash
find /root/polymarket-scripts/scripts/__pycache__ -name "*.pyc" -newer /root/polymarket-scripts/scripts/quant_shadow.py -delete
```

### Step 3: Verify no crash within 30 seconds

```bash
sleep 30 && ps aux | grep "quant_shadow" | grep -v grep && echo "Still running ✓"
```

### Step 4: Check that first prediction fires (wait for T+3min of a 5m market)

```bash
tail -f /tmp/quant_shadow.log | grep -E "QUANT prediction|prediction"
```

Expected when trigger fires:
```
BTC-5m QUANT prediction gap_z=+1.23 clob=0.62 trend=+0.10 phase=early p_yes=0.71 action=YES edge=+0.090
```

### Step 5: Verify DB row was written

```bash
python3 -c "
import sqlite3
conn = sqlite3.connect('data/performance.db')
cur = conn.cursor()
cur.execute('SELECT asset, timeframe, gap_z, clob_yes, quant_action, quant_edge, fired_at FROM quant_shadow_log ORDER BY id DESC LIMIT 5')
for r in cur.fetchall():
    print(r)
"
```

### Step 6: Final commit

```bash
cd /root/polymarket-scripts
git add -A
git commit -m "feat(quant-shadow): service verified and running"
```

---

## Summary

After completion, the system has:

| Service | PID | Purpose |
|---|---|---|
| V2 live bot | 1071231 | Actual trading |
| Quant Shadow | new | Shadow predictions only |

**Log:** `/tmp/quant_shadow.log`

**DB table:** `quant_shadow_log` — features + prediction + outcome per market

**Telegram alerts:** After each market settles:
```
🔬 Quant Shadow: BTC-5m settled
────────────────────────────────
Outcome: ✅ YES (BTC $68,100 vs PTB $68,050)

Quant: YES (p=0.72, edge=+11%) ✅
V2:    YES (93%) ✅

Features:
  gap_z=+2.10 | clob=0.61 | vol=8.2/min
  trend=+0.30 | phase=late | tte=120s
```

**After ~2 weeks:** Retrain model with `python3 -c "from quant_shadow.model import train_and_save; train_and_save()"` — the growing `quant_shadow_log` data will improve calibration.
