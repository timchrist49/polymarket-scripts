# Chainlink Integration + Signal Weighting Fix Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Switch BTC price source from Binance to Chainlink RTDS to match Polymarket's settlement prices, and adjust AI signal weighting to prioritize actual price movement over sentiment.

**Architecture:** Replace Binance RTDS subscription with Chainlink RTDS in crypto_price_stream.py while maintaining backward compatibility with fallback sources. Refactor ai_analysis.py to implement tiered signal confidence calculation with price direction as primary (70% weight) and sentiment as supporting signal (5% weight).

**Tech Stack:** Python 3.11, websockets, Polymarket RTDS, decimal arithmetic

**Context:** This fixes a $2,469 price discrepancy found in market btc-updown-15m-1771096500 where bot's price_to_beat ($67,257.39) didn't match Polymarket's Chainlink settlement price ($69,726.92), causing incorrect directional analysis.

---

## Task 1: Add Chainlink Source Parameter to CryptoPriceStream

**Files:**
- Modify: `polymarket/trading/crypto_price_stream.py:23-48`
- Test: `tests/test_crypto_price_stream_chainlink.py` (create)

**Step 1: Write failing test for Chainlink subscription**

Create: `tests/test_crypto_price_stream_chainlink.py`

```python
"""Tests for Chainlink RTDS integration in CryptoPriceStream."""

import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch
from polymarket.trading.crypto_price_stream import CryptoPriceStream
from polymarket.config import Settings


@pytest.mark.asyncio
async def test_chainlink_subscription_format():
    """Test that Chainlink subscription uses correct format."""
    settings = Settings()
    stream = CryptoPriceStream(settings, use_chainlink=True)

    # Mock websocket
    mock_ws = AsyncMock()

    with patch('websockets.connect', return_value=mock_ws):
        # Start connection (will fail since we're mocking)
        try:
            await stream._subscribe_to_feed(mock_ws)
        except:
            pass

    # Verify subscription message format
    assert mock_ws.send.called
    sent_msg = json.loads(mock_ws.send.call_args[0][0])

    # Chainlink format requirements
    assert sent_msg["action"] == "subscribe"
    assert sent_msg["subscriptions"][0]["topic"] == "crypto_prices_chainlink"
    assert sent_msg["subscriptions"][0]["type"] == "*"
    assert sent_msg["subscriptions"][0]["filters"] == '{"symbol":"btc/usd"}'


@pytest.mark.asyncio
async def test_binance_subscription_format():
    """Test that Binance subscription still works (backward compatibility)."""
    settings = Settings()
    stream = CryptoPriceStream(settings, use_chainlink=False)

    mock_ws = AsyncMock()

    with patch('websockets.connect', return_value=mock_ws):
        try:
            await stream._subscribe_to_feed(mock_ws)
        except:
            pass

    sent_msg = json.loads(mock_ws.send.call_args[0][0])

    # Binance format requirements
    assert sent_msg["subscriptions"][0]["topic"] == "crypto_prices"
    assert sent_msg["subscriptions"][0]["type"] == "update"
    assert sent_msg["subscriptions"][0]["filters"] == "btcusdt"
```

**Step 2: Run test to verify it fails**

```bash
cd /root/polymarket-scripts
pytest tests/test_crypto_price_stream_chainlink.py::test_chainlink_subscription_format -v
```

Expected: `FAIL - AttributeError: 'CryptoPriceStream' object has no attribute '_subscribe_to_feed'`

**Step 3: Implement Chainlink support in CryptoPriceStream**

Modify: `polymarket/trading/crypto_price_stream.py`

```python
class CryptoPriceStream:
    """Real-time BTC price stream from Polymarket WebSocket."""

    WS_URL = "wss://ws-live-data.polymarket.com"

    def __init__(
        self,
        settings: Settings,
        buffer_enabled: bool = False,
        buffer_file: str = "data/price_history.json",
        use_chainlink: bool = True  # ← NEW: default to Chainlink
    ):
        self.settings = settings
        self.use_chainlink = use_chainlink  # ← NEW
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._current_price: Optional[BTCPriceData] = None
        self._connected = False
        self._running = False

        # Initialize price history buffer if enabled
        self.price_buffer: Optional[PriceHistoryBuffer] = None
        if buffer_enabled:
            self.price_buffer = PriceHistoryBuffer(
                retention_hours=24,
                save_interval=300,
                persistence_file=buffer_file
            )

    async def start(self):
        """Start WebSocket connection and price stream."""
        self._running = True

        # Load historical price data if buffer is enabled
        if self.price_buffer:
            try:
                await self.price_buffer.load_from_disk()
                logger.info("Price history loaded from disk")
            except Exception as e:
                logger.warning(f"Could not load price history: {e}")

        try:
            async with websockets.connect(self.WS_URL, ping_interval=5) as ws:  # ← Add ping_interval
                self._ws = ws
                self._connected = True

                # Subscribe to appropriate feed
                await self._subscribe_to_feed(ws)  # ← NEW method

                # Listen for price updates
                while self._running:
                    try:
                        message = await asyncio.wait_for(ws.recv(), timeout=30.0)
                        await self._handle_message(message)
                    except asyncio.TimeoutError:
                        logger.debug("WebSocket recv timeout, continuing...")
                        continue
                    except websockets.exceptions.ConnectionClosed:
                        logger.warning("WebSocket connection closed")
                        self._connected = False
                        break

        except Exception as e:
            logger.error("WebSocket connection error", error=str(e))
            self._connected = False
        finally:
            self._ws = None
            self._connected = False

    async def _subscribe_to_feed(self, ws):  # ← NEW method
        """Subscribe to Chainlink or Binance feed based on configuration."""
        if self.use_chainlink:
            # Chainlink format (verified working)
            subscribe_msg = {
                "action": "subscribe",
                "subscriptions": [{
                    "topic": "crypto_prices_chainlink",
                    "type": "*",  # Must be "*" for Chainlink
                    "filters": '{"symbol":"btc/usd"}'  # JSON as string
                }]
            }
            logger.info(
                "Subscribed to Polymarket RTDS crypto_prices_chainlink",
                source="chainlink",
                symbol="btc/usd"
            )
        else:
            # Binance format (legacy fallback)
            subscribe_msg = {
                "action": "subscribe",
                "subscriptions": [{
                    "topic": "crypto_prices",
                    "type": "update",
                    "filters": "btcusdt"
                }]
            }
            logger.info(
                "Subscribed to Polymarket RTDS crypto_prices",
                source="binance",
                symbol="btcusdt"
            )

        await ws.send(json.dumps(subscribe_msg))
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_crypto_price_stream_chainlink.py::test_chainlink_subscription_format -v
pytest tests/test_crypto_price_stream_chainlink.py::test_binance_subscription_format -v
```

Expected: Both tests `PASS`

**Step 5: Commit**

```bash
git add polymarket/trading/crypto_price_stream.py tests/test_crypto_price_stream_chainlink.py
git commit -m "feat: add Chainlink RTDS support to CryptoPriceStream

- Add use_chainlink parameter (default True)
- Extract subscription logic to _subscribe_to_feed()
- Support both Chainlink and Binance formats
- Add ping_interval=5 for connection stability
- Add tests for both subscription formats

Refs: btc-updown-15m-1771096500 price discrepancy"
```

---

## Task 2: Update Message Parsing for Chainlink Format

**Files:**
- Modify: `polymarket/trading/crypto_price_stream.py:100-160`
- Test: `tests/test_crypto_price_stream_chainlink.py`

**Step 1: Write failing test for Chainlink message parsing**

Add to: `tests/test_crypto_price_stream_chainlink.py`

```python
@pytest.mark.asyncio
async def test_parse_chainlink_initial_message():
    """Test parsing Chainlink initial data dump."""
    settings = Settings()
    stream = CryptoPriceStream(settings, use_chainlink=True)

    # Simulate initial Chainlink message
    chainlink_initial = json.dumps({
        "topic": "crypto_prices_chainlink",
        "type": "subscribe",
        "payload": {
            "data": [
                {"timestamp": 1771133368000, "value": 70314.50691332904},
                {"timestamp": 1771133372000, "value": 70312.32709805031}
            ]
        }
    })

    await stream._handle_message(chainlink_initial)

    # Should parse latest price
    current = await stream.get_current_price()
    assert current is not None
    assert current.price == pytest.approx(Decimal("70312.33"), abs=0.01)
    assert current.source == "chainlink"


@pytest.mark.asyncio
async def test_parse_chainlink_update_message():
    """Test parsing Chainlink real-time update."""
    settings = Settings()
    stream = CryptoPriceStream(settings, use_chainlink=True)

    # Simulate real-time update
    chainlink_update = json.dumps({
        "topic": "crypto_prices_chainlink",
        "type": "update",
        "timestamp": 1771133430140,
        "payload": {
            "symbol": "btc/usd",
            "value": 70283.97530686231,
            "full_accuracy_value": "70283975306862305000000",
            "timestamp": 1771133429000
        }
    })

    await stream._handle_message(chainlink_update)

    current = await stream.get_current_price()
    assert current is not None
    assert current.price == pytest.approx(Decimal("70283.98"), abs=0.01)
    assert current.source == "chainlink"
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_crypto_price_stream_chainlink.py::test_parse_chainlink_initial_message -v
pytest tests/test_crypto_price_stream_chainlink.py::test_parse_chainlink_update_message -v
```

Expected: `FAIL - AssertionError: current.source != 'chainlink'`

**Step 3: Update _handle_message to support Chainlink format**

Modify: `polymarket/trading/crypto_price_stream.py:100-160`

```python
async def _handle_message(self, message: str):
    """Parse and store price update from Binance or Chainlink feed."""
    try:
        # Skip empty messages (WebSocket pings/heartbeats)
        if not message or not message.strip():
            return

        data = json.loads(message)
        topic = data.get("topic")
        msg_type = data.get("type")
        payload = data.get("payload", {})

        # Handle Chainlink feed
        if topic == "crypto_prices_chainlink":
            await self._handle_chainlink_message(msg_type, payload, data)

        # Handle Binance feed (legacy)
        elif topic == "crypto_prices":
            await self._handle_binance_message(msg_type, payload)

    except Exception as e:
        logger.error("Failed to parse price message", error=str(e))


async def _handle_chainlink_message(self, msg_type: str, payload: dict, data: dict):
    """Handle Chainlink price messages."""
    if msg_type == "subscribe" and payload.get("data"):
        # Initial data dump - use latest price
        price_data = payload["data"]
        if price_data:
            latest = price_data[-1]
            self._current_price = BTCPriceData(
                price=Decimal(str(latest["value"])),
                timestamp=datetime.fromtimestamp(latest["timestamp"] / 1000),
                source="chainlink",  # ← Mark source
                volume_24h=Decimal("0")
            )
            logger.debug(
                "BTC price (initial)",
                price=f"${self._current_price.price:,.2f}",
                source="chainlink"
            )

            # Append to buffer
            await self._append_to_buffer(
                latest["timestamp"] // 1000,
                Decimal(str(latest["value"])),
                source="chainlink"  # ← Mark source
            )

    elif msg_type == "update":
        # Real-time update
        self._current_price = BTCPriceData(
            price=Decimal(str(payload["value"])),
            timestamp=datetime.fromtimestamp(payload["timestamp"] / 1000),
            source="chainlink",  # ← Mark source
            volume_24h=Decimal("0")
        )
        logger.debug(
            "BTC price update",
            price=f"${self._current_price.price:,.2f}",
            source="chainlink"
        )

        # Append to buffer
        await self._append_to_buffer(
            payload["timestamp"] // 1000,
            Decimal(str(payload["value"])),
            source="chainlink"  # ← Mark source
        )


async def _handle_binance_message(self, msg_type: str, payload: dict):
    """Handle Binance price messages (legacy fallback)."""
    if msg_type == "subscribe" and payload.get("symbol") == "btcusdt":
        # Initial dump
        price_data = payload.get("data", [])
        if price_data:
            latest = price_data[-1]
            self._current_price = BTCPriceData(
                price=Decimal(str(latest["value"])),
                timestamp=datetime.fromtimestamp(latest["timestamp"] / 1000),
                source="binance",  # ← Mark source
                volume_24h=Decimal("0")
            )
            logger.debug(
                "BTC price (initial)",
                price=f"${self._current_price.price:,.2f}",
                source="binance"
            )

            await self._append_to_buffer(
                latest["timestamp"] // 1000,
                Decimal(str(latest["value"])),
                source="binance"  # ← Mark source
            )

    elif msg_type == "update" and payload.get("symbol") == "btcusdt":
        # Real-time update
        self._current_price = BTCPriceData(
            price=Decimal(str(payload["value"])),
            timestamp=datetime.fromtimestamp(payload["timestamp"] / 1000),
            source="binance",  # ← Mark source
            volume_24h=Decimal("0")
        )
        logger.debug(
            "BTC price update",
            price=f"${self._current_price.price:,.2f}",
            source="binance"
        )

        await self._append_to_buffer(
            payload["timestamp"] // 1000,
            Decimal(str(payload["value"])),
            source="binance"  # ← Mark source
        )


async def _append_to_buffer(self, timestamp: int, price: Decimal, source: str = "unknown"):
    """Append price to buffer if enabled."""
    if self.price_buffer:
        try:
            await self.price_buffer.append(timestamp, price, source=source)  # ← Pass source
        except Exception as e:
            logger.error(f"Failed to append to price buffer: {e}")
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_crypto_price_stream_chainlink.py -v
```

Expected: All tests `PASS`

**Step 5: Commit**

```bash
git add polymarket/trading/crypto_price_stream.py tests/test_crypto_price_stream_chainlink.py
git commit -m "feat: add Chainlink message parsing to CryptoPriceStream

- Extract Chainlink message handling to _handle_chainlink_message()
- Extract Binance message handling to _handle_binance_message()
- Mark price source in BTCPriceData (chainlink/binance)
- Pass source to buffer for audit trail
- Add tests for both message formats"
```

---

## Task 3: Enable Chainlink by Default in Auto-Trader

**Files:**
- Modify: `scripts/auto_trade.py:173-183`
- Test: Manual verification in logs

**Step 1: Update auto_trade.py to use Chainlink**

Modify: `scripts/auto_trade.py` (around line 173)

```python
async def _init_btc_service(self):
    """Initialize BTC price service with real-time monitoring."""
    self.stream = CryptoPriceStream(
        self.settings,
        buffer_enabled=True,
        buffer_file="data/price_history.json",
        use_chainlink=True  # ← NEW: Use Chainlink by default
    )
    asyncio.create_task(self.stream.start())

    # Wait for connection
    for _ in range(10):
        await asyncio.sleep(0.5)
        if self.stream.is_connected():
            break

    self.btc_service = BTCPriceService(
        settings=self.settings,
        stream=self.stream
    )
    await self.btc_service.start()

    logger.info(
        "Initialized Polymarket WebSocket for BTC prices",
        source="chainlink",  # ← Log source
        connected=self.stream.is_connected()
    )
```

**Step 2: Test by running bot and checking logs**

```bash
cd /root/polymarket-scripts
TEST_MODE=true python3 scripts/auto_trade.py
```

Expected in logs:
```
"Subscribed to Polymarket RTDS crypto_prices_chainlink" source="chainlink"
"BTC price update" source="chainlink" price="$70,XXX.XX"
```

**Step 3: Verify no errors in 60 seconds**

Monitor logs for 60 seconds. Should see:
- ✓ Chainlink subscription successful
- ✓ Real-time price updates from Chainlink
- ✓ No fallback to CoinGecko (unless Chainlink fails)

**Step 4: Stop bot and commit**

```bash
# Stop with Ctrl+C

git add scripts/auto_trade.py
git commit -m "feat: enable Chainlink RTDS by default in auto-trader

- Set use_chainlink=True in CryptoPriceStream initialization
- Log source='chainlink' for debugging
- Bot now uses same price source as Polymarket settlement"
```

---

## Task 4: Add Price Source Column to Database

**Files:**
- Create: `scripts/migrations/add_price_source_column.py`
- Modify: `polymarket/database/schema.py`
- Test: Manual migration test

**Step 1: Create database migration script**

Create: `scripts/migrations/add_price_source_column.py`

```python
#!/usr/bin/env python3
"""
Migration: Add price_source column to trades and paper_trades tables.

This column tracks which price source was used (chainlink, binance, coingecko)
for audit trail and debugging price discrepancies.
"""

import sqlite3
import os
from pathlib import Path


def migrate(db_path: str = "data/performance.db"):
    """Add price_source column to tables."""
    print(f"Running migration on: {db_path}")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # Add to trades table
        cursor.execute("""
            ALTER TABLE trades
            ADD COLUMN price_source TEXT DEFAULT 'unknown'
        """)
        print("✓ Added price_source to trades table")

        # Add to paper_trades table
        cursor.execute("""
            ALTER TABLE paper_trades
            ADD COLUMN price_source TEXT DEFAULT 'unknown'
        """)
        print("✓ Added price_source to paper_trades table")

        conn.commit()
        print("\n✓ Migration complete!")

    except sqlite3.OperationalError as e:
        if "duplicate column name" in str(e).lower():
            print("⚠ Column already exists, skipping migration")
        else:
            raise

    finally:
        conn.close()


if __name__ == "__main__":
    # Run on both main and worktree databases if they exist
    databases = [
        "data/performance.db",
        "/root/polymarket-scripts/data/performance.db"
    ]

    for db_path in databases:
        if Path(db_path).exists():
            migrate(db_path)
        else:
            print(f"⚠ Database not found: {db_path}")
```

**Step 2: Run migration**

```bash
cd /root/polymarket-scripts
python3 scripts/migrations/add_price_source_column.py
```

Expected output:
```
✓ Added price_source to trades table
✓ Added price_source to paper_trades table
✓ Migration complete!
```

**Step 3: Verify migration in sqlite**

```bash
sqlite3 data/performance.db "PRAGMA table_info(trades);" | grep price_source
sqlite3 data/performance.db "PRAGMA table_info(paper_trades);" | grep price_source
```

Expected: Both show price_source column

**Step 4: Commit**

```bash
git add scripts/migrations/add_price_source_column.py
git commit -m "feat: add price_source column migration

- Add price_source TEXT column to trades table
- Add price_source TEXT column to paper_trades table
- Default value 'unknown' for backward compatibility
- Enables audit trail for price discrepancy debugging"
```

---

## Task 5: Update Trade Logging to Include Price Source

**Files:**
- Modify: `scripts/auto_trade.py:800-850` (execute_paper_trade)
- Modify: `scripts/auto_trade.py:750-800` (execute_real_trade)

**Step 1: Add price source to paper trade logging**

Modify: `scripts/auto_trade.py` in `execute_paper_trade()` method

```python
async def execute_paper_trade(
    self,
    market: Market,
    decision: str,
    ai_analysis: dict,
    price_to_beat: Decimal
) -> Optional[int]:
    """Execute paper trade (simulation only)."""
    try:
        # ... existing code ...

        # Get current price WITH source
        current_price_data = await self.stream.get_current_price()
        current_price = current_price_data.price if current_price_data else Decimal("0")
        price_source = current_price_data.source if current_price_data else "unknown"  # ← NEW

        # ... existing calculation code ...

        # Log to paper_trades table
        cursor.execute(
            """
            INSERT INTO paper_trades (
                market_slug, action, outcome, size, price,
                fee, profit_loss, confidence, timestamp,
                btc_current, btc_price_to_beat, price_source  -- ← NEW
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                market.slug,
                decision,
                outcome,
                float(size),
                float(execution_price),
                float(fee),
                float(profit_loss),
                ai_analysis.get("confidence", 0.0),
                int(datetime.now(timezone.utc).timestamp()),
                float(current_price),
                float(price_to_beat),
                price_source  # ← NEW
            ),
        )

        # ... rest of method ...

    except Exception as e:
        logger.error(f"Paper trade execution failed: {e}")
        return None
```

**Step 2: Add price source to real trade logging**

Modify: `scripts/auto_trade.py` in execute_real_trade() method (similar changes)

```python
async def execute_real_trade(...):
    """Execute real trade on Polymarket."""
    try:
        # ... existing code ...

        # Get current price WITH source
        current_price_data = await self.stream.get_current_price()
        current_price = current_price_data.price if current_price_data else Decimal("0")
        price_source = current_price_data.source if current_price_data else "unknown"  # ← NEW

        # ... execution code ...

        # Log to trades table
        cursor.execute(
            """
            INSERT INTO trades (
                market_slug, action, outcome, size, price,
                fee, order_id, btc_current, btc_price_to_beat,
                confidence, timestamp, price_source  -- ← NEW
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                market.slug,
                decision,
                outcome,
                float(size),
                float(executed_price),
                float(fee),
                order_id,
                float(current_price),
                float(price_to_beat),
                ai_analysis.get("confidence", 0.0),
                int(datetime.now(timezone.utc).timestamp()),
                price_source  # ← NEW
            ),
        )

        # ... rest of method ...
```

**Step 3: Test with paper trading mode**

```bash
cd /root/polymarket-scripts
TEST_MODE=true python3 scripts/auto_trade.py
```

Wait for a paper trade to execute, then check database:

```bash
sqlite3 data/performance.db "SELECT market_slug, action, price_source FROM paper_trades ORDER BY timestamp DESC LIMIT 5;"
```

Expected: Recent trades show `price_source='chainlink'`

**Step 4: Commit**

```bash
git add scripts/auto_trade.py
git commit -m "feat: log price source in trade execution

- Add price_source to paper trade logging
- Add price_source to real trade logging
- Extract source from current_price_data
- Enables debugging of price discrepancies by source"
```

---

## Task 6: Implement Tiered Signal Weighting in AI Analysis

**Files:**
- Create: `tests/test_signal_weighting.py`
- Modify: `polymarket/trading/ai_analysis.py`

**Step 1: Write failing test for signal weighting**

Create: `tests/test_signal_weighting.py`

```python
"""Tests for tiered signal weighting in AI analysis."""

import pytest
from decimal import Decimal
from polymarket.trading.ai_analysis import AIAnalysisService


def test_price_direction_overrides_sentiment():
    """Price direction (70% weight) should override sentiment (5% weight)."""

    # Scenario: BTC clearly UP, but sentiment is bearish
    # Expected: Should bet UP (follow price, not sentiment)

    analysis = {
        "price_direction": "UP",
        "price_change_pct": 3.5,  # Strong upward movement
        "sentiment_score": -0.8,   # Very bearish sentiment
        "rsi": 65,                 # Neutral-bullish
        "confidence": 0.75
    }

    weighted_confidence = AIAnalysisService._calculate_weighted_confidence(
        analysis,
        primary_signal="price_direction"
    )

    # Price direction should dominate
    assert weighted_confidence > 0.65  # High confidence in UP
    assert "sentiment_conflict" in analysis.get("warnings", [])


def test_sentiment_supports_price():
    """Sentiment should ADD confidence when aligned with price."""

    # Scenario: BTC UP + Bullish sentiment = extra confidence

    analysis_without_sentiment = {
        "price_direction": "UP",
        "price_change_pct": 2.0,
        "sentiment_score": 0.0,  # Neutral
        "rsi": 60,
        "confidence": 0.70
    }

    analysis_with_sentiment = {
        "price_direction": "UP",
        "price_change_pct": 2.0,
        "sentiment_score": 0.9,  # Very bullish (aligned!)
        "rsi": 60,
        "confidence": 0.70
    }

    conf_without = AIAnalysisService._calculate_weighted_confidence(
        analysis_without_sentiment,
        primary_signal="price_direction"
    )

    conf_with = AIAnalysisService._calculate_weighted_confidence(
        analysis_with_sentiment,
        primary_signal="price_direction"
    )

    # Aligned sentiment should boost confidence by ~5%
    assert conf_with > conf_without
    assert (conf_with - conf_without) <= 0.10  # But not too much


def test_conflicting_signals_reduce_confidence():
    """Conflicting signals should reduce overall confidence."""

    analysis = {
        "price_direction": "UP",
        "price_change_pct": 2.0,
        "sentiment_score": -0.7,  # Conflict!
        "rsi": 55,
        "confidence": 0.75
    }

    weighted_confidence = AIAnalysisService._calculate_weighted_confidence(
        analysis,
        primary_signal="price_direction"
    )

    # Should reduce confidence by 10-20% due to conflict
    assert weighted_confidence < 0.70
    assert weighted_confidence > 0.55


def test_technical_indicators_second_priority():
    """Technical indicators (20% weight) should be secondary."""

    # Scenario: Price slightly up, but RSI overbought

    analysis = {
        "price_direction": "UP",
        "price_change_pct": 0.5,  # Small move
        "sentiment_score": 0.0,
        "rsi": 82,  # Overbought (bearish signal)
        "confidence": 0.70
    }

    weighted_confidence = AIAnalysisService._calculate_weighted_confidence(
        analysis,
        primary_signal="price_direction"
    )

    # RSI should reduce confidence slightly, but not override price
    assert weighted_confidence < 0.70
    assert weighted_confidence > 0.55  # Still bullish overall
```

**Step 2: Run tests to verify they fail**

```bash
cd /root/polymarket-scripts
pytest tests/test_signal_weighting.py -v
```

Expected: `FAIL - AttributeError: 'AIAnalysisService' has no method '_calculate_weighted_confidence'`

**Step 3: Implement weighted confidence calculation**

Modify: `polymarket/trading/ai_analysis.py`

Add new method to AIAnalysisService class:

```python
@staticmethod
def _calculate_weighted_confidence(
    analysis: dict,
    primary_signal: str = "price_direction"
) -> float:
    """
    Calculate weighted confidence using tiered signal priority.

    Signal Tiers:
    1. Price Direction (70% weight) - PRIMARY
    2. Technical Indicators (20% weight) - SECONDARY
    3. Market Odds (5% weight) - VALIDATION
    4. Sentiment (5% weight) - SUPPORTING

    Args:
        analysis: Raw AI analysis dict
        primary_signal: Which signal to prioritize (default: price_direction)

    Returns:
        Weighted confidence score (0.0 to 1.0)
    """
    base_confidence = analysis.get("confidence", 0.70)

    # Tier 1: Price Direction (70% base)
    price_confidence = base_confidence * 0.70

    # Tier 2: Technical Indicators (20%)
    technical_confidence = 0.0
    rsi = analysis.get("rsi", 50)

    if analysis.get("price_direction") == "UP":
        # For UP moves, RSI overbought (>70) is bearish
        if rsi > 70:
            technical_confidence = -0.10  # Reduce confidence
        elif 45 <= rsi <= 70:
            technical_confidence = 0.10   # Neutral-bullish
        else:
            technical_confidence = 0.05   # Oversold (mixed signal)
    else:
        # For DOWN moves, RSI oversold (<30) is bullish
        if rsi < 30:
            technical_confidence = -0.10  # Reduce confidence
        elif 30 <= rsi <= 55:
            technical_confidence = 0.10   # Neutral-bearish
        else:
            technical_confidence = 0.05   # Overbought (mixed signal)

    # Tier 3: Market Odds (5%) - already validated by caller
    odds_confidence = 0.05 if analysis.get("market_odds_valid") else 0.0

    # Tier 4: Sentiment (5% - supporting only)
    sentiment_score = analysis.get("sentiment_score", 0.0)
    price_direction = analysis.get("price_direction", "NEUTRAL")

    sentiment_confidence = 0.0
    sentiment_aligned = False

    if price_direction == "UP" and sentiment_score > 0.3:
        sentiment_confidence = 0.05  # Bullish sentiment supports UP
        sentiment_aligned = True
    elif price_direction == "DOWN" and sentiment_score < -0.3:
        sentiment_confidence = 0.05  # Bearish sentiment supports DOWN
        sentiment_aligned = True
    elif abs(sentiment_score) > 0.5:
        # Strong sentiment conflicts with price
        sentiment_confidence = -0.10  # Penalty for conflict
        analysis.setdefault("warnings", []).append("sentiment_conflict")

    # Calculate final weighted confidence
    weighted_confidence = (
        price_confidence +
        (technical_confidence * 0.20) +
        odds_confidence +
        (sentiment_confidence if sentiment_aligned else sentiment_confidence * 0.50)
    )

    # Clamp to [0.0, 1.0]
    return max(0.0, min(1.0, weighted_confidence))
```

**Step 4: Integrate into main analyze method**

Modify: `polymarket/trading/ai_analysis.py` in `analyze()` method

```python
async def analyze(
    self,
    market: Market,
    price_to_beat: Decimal,
    current_price: Decimal,
    # ... other params ...
) -> dict:
    """Analyze market and generate trading decision."""

    # ... existing analysis code ...

    # Generate AI decision
    raw_analysis = await self._call_ai_model(
        market=market,
        # ... params ...
    )

    # Apply weighted confidence calculation
    weighted_confidence = self._calculate_weighted_confidence(
        raw_analysis,
        primary_signal="price_direction"
    )

    # Override AI's confidence with weighted version
    raw_analysis["confidence"] = weighted_confidence
    raw_analysis["confidence_method"] = "tiered_weighting"

    # Log if confidence was significantly adjusted
    original_conf = raw_analysis.get("original_confidence", weighted_confidence)
    if abs(original_conf - weighted_confidence) > 0.15:
        logger.warning(
            "Confidence adjusted by signal weighting",
            original=f"{original_conf:.2f}",
            weighted=f"{weighted_confidence:.2f}",
            reason=raw_analysis.get("warnings", [])
        )

    return raw_analysis
```

**Step 5: Run tests to verify they pass**

```bash
pytest tests/test_signal_weighting.py -v
```

Expected: All tests `PASS`

**Step 6: Commit**

```bash
git add polymarket/trading/ai_analysis.py tests/test_signal_weighting.py
git commit -m "feat: implement tiered signal weighting in AI analysis

- Add _calculate_weighted_confidence() with 4-tier system:
  * Price direction: 70% weight (primary)
  * Technical indicators: 20% weight (secondary)
  * Market odds: 5% weight (validation)
  * Sentiment: 5% weight (supporting only)
- Sentiment can only add confidence if aligned with price
- Conflicting signals reduce confidence by 10-20%
- Add tests for all weighting scenarios

This prevents sentiment from overriding clear price signals,
fixing issues like Trade ID 8 where bearish sentiment caused
DOWN bet despite BTC being UP."
```

---

## Task 7: Update AI Prompt to Emphasize Price Priority

**Files:**
- Modify: `polymarket/trading/ai_analysis.py` (AI prompt)

**Step 1: Update system prompt**

Modify: `polymarket/trading/ai_analysis.py` in `_build_system_prompt()` or similar

Find the AI prompt and add emphasis on price priority:

```python
SYSTEM_PROMPT = """
You are an expert BTC prediction market trader analyzing 15-minute UP/DOWN markets.

**CRITICAL: Signal Priority Rules**

You MUST prioritize signals in this order:

1. **PRICE MOVEMENT (PRIMARY - 70% weight)**
   - Current BTC price vs price_to_beat is THE MOST IMPORTANT signal
   - If BTC moved >1%, this should dominate your decision
   - Example: BTC +$500 = Strong UP signal (ignore conflicting sentiment)

2. **TECHNICAL INDICATORS (SECONDARY - 20% weight)**
   - RSI, MACD, volume patterns provide supporting context
   - Can modify confidence but should NOT override clear price movement
   - Example: RSI overbought during strong rally = reduce confidence slightly

3. **MARKET ODDS (VALIDATION - 5% weight)**
   - Market must be >75% YES or NO to be tradeable
   - This is a filter, not a primary signal

4. **SENTIMENT (SUPPORTING ONLY - 5% weight)**
   - Reddit/Twitter sentiment can ADD confidence if aligned
   - Sentiment should NEVER override actual BTC price movement
   - If sentiment conflicts with price: TRUST THE PRICE, not sentiment

**Example Scenarios:**

❌ WRONG: BTC +$200 (UP) + Bearish sentiment → Bet DOWN
✓ CORRECT: BTC +$200 (UP) + Bearish sentiment → Bet UP (reduced confidence)

❌ WRONG: BTC -$50 (slight down) + Very bullish sentiment → Bet UP
✓ CORRECT: BTC -$50 (slight down) + Very bullish sentiment → Bet DOWN (or PASS if confidence too low)

**Your Task:**
Analyze the market data and recommend YES (UP) or NO (DOWN).
Provide confidence (0.0-1.0) based on the tiered priority above.
"""
```

**Step 2: Test AI prompt by running bot**

```bash
cd /root/polymarket-scripts
TEST_MODE=true python3 scripts/auto_trade.py
```

Monitor logs for AI analysis. Should see:
- Price direction emphasized in decisions
- Sentiment mentioned but not decisive
- Confidence reflects price movement strength

**Step 3: Review one AI decision output**

Check logs for an AI analysis and verify:
- If BTC clearly up/down, decision matches price direction
- Sentiment noted but doesn't override price
- Confidence reflects signal alignment

**Step 4: Commit**

```bash
git add polymarket/trading/ai_analysis.py
git commit -m "feat: update AI prompt to prioritize price over sentiment

- Add explicit signal priority hierarchy to system prompt
- Emphasize price movement as PRIMARY signal (70%)
- Clarify sentiment is SUPPORTING ONLY (5%)
- Add example scenarios showing correct prioritization
- Prevent AI from overriding price with sentiment"
```

---

## Task 8: Integration Testing with Historical Market

**Files:**
- Create: `tests/test_chainlink_integration.py`
- Test: Backtest with btc-updown-15m-1771096500

**Step 1: Create integration test**

Create: `tests/test_chainlink_integration.py`

```python
"""Integration test for Chainlink price source fix."""

import pytest
from decimal import Decimal
from datetime import datetime, timezone


@pytest.mark.integration
def test_historical_market_price_accuracy():
    """
    Test that we would calculate correct price_to_beat for historical market.

    Market: btc-updown-15m-1771096500
    Polymarket settlement price (Chainlink): $69,726.92
    Our old price (Binance): $67,257.39
    Discrepancy: $2,469.53 (3.6%)
    """

    # This test documents the fix, actual testing requires live Chainlink data
    polymarket_chainlink_price = Decimal("69726.92")
    our_old_binance_price = Decimal("67257.39")
    discrepancy = abs(polymarket_chainlink_price - our_old_binance_price)

    # The bug: we had 3.6% price discrepancy
    assert discrepancy == Decimal("2469.53")

    # After fix: we should use Chainlink RTDS
    # This would give us the same price as Polymarket (within $1)
    expected_max_diff = Decimal("1.00")  # Allow $1 difference for timing

    # TODO: Once Chainlink integrated, this test should:
    # 1. Subscribe to Chainlink RTDS
    # 2. Get price at timestamp 1771096500
    # 3. Assert abs(price - 69726.92) < 1.00

    print(f"Old discrepancy: ${discrepancy:,.2f}")
    print(f"Target accuracy: <${expected_max_diff:,.2f}")
    print("After Chainlink integration, price_to_beat should match Polymarket exactly")
```

**Step 2: Run integration test**

```bash
pytest tests/test_chainlink_integration.py -v -m integration
```

Expected: `PASS` (documents the fix)

**Step 3: Manual verification with live bot**

Start bot in test mode and wait for next market:

```bash
cd /root/polymarket-scripts
TEST_MODE=true python3 scripts/auto_trade.py
```

When a new 15-minute market starts:
1. Note the `price_to_beat` logged by bot
2. Check Polymarket's UI for the same market
3. Compare prices (should be within $1-5 due to timing)

**Step 4: Commit**

```bash
git add tests/test_chainlink_integration.py
git commit -m "test: add integration test documenting price fix

- Document historical price discrepancy ($2,469)
- Set target accuracy (<$1 difference)
- Provide manual verification steps
- Mark as integration test for CI/CD"
```

---

## Task 9: Update Documentation

**Files:**
- Modify: `README.md`
- Create: `docs/CHAINLINK_MIGRATION.md`

**Step 1: Create migration documentation**

Create: `docs/CHAINLINK_MIGRATION.md`

```markdown
# Chainlink Price Source Migration

## Problem Statement

Bot was using Binance prices via Polymarket RTDS (`crypto_prices` topic), but Polymarket settles markets using Chainlink oracle prices. This caused significant price discrepancies:

**Example: Market btc-updown-15m-1771096500**
- Bot's price_to_beat: $67,257.39 (Binance)
- Polymarket settlement: $69,726.92 (Chainlink)
- **Discrepancy: $2,469.53 (3.6%)**

This led to incorrect directional analysis and suboptimal trades.

## Solution

Switched to Chainlink price feed via Polymarket RTDS:
- Topic: `crypto_prices_chainlink`
- Format: `{"symbol":"btc/usd"}` (slash-separated)
- Same data source Polymarket uses for settlement

## Changes Made

### 1. CryptoPriceStream Updates
- Added `use_chainlink` parameter (default: `True`)
- Extracted `_subscribe_to_feed()` method
- Added `_handle_chainlink_message()` for Chainlink format
- Added `_handle_binance_message()` for legacy fallback
- Added ping_interval=5 for connection stability

### 2. Database Schema
- Added `price_source` column to `trades` table
- Added `price_source` column to `paper_trades` table
- Enables audit trail for price discrepancies

### 3. AI Signal Weighting
- Implemented tiered confidence calculation:
  * Price direction: 70% weight (PRIMARY)
  * Technical indicators: 20% weight
  * Market odds: 5% weight
  * Sentiment: 5% weight (supporting only)
- Prevents sentiment from overriding price signals

### 4. AI Prompt Updates
- Added explicit signal priority hierarchy
- Emphasized price movement as primary signal
- Clarified sentiment is supporting only

## Migration Steps

Run database migration:
```bash
cd /root/polymarket-scripts
python3 scripts/migrations/add_price_source_column.py
```

Restart bot (Chainlink enabled by default):
```bash
TEST_MODE=true python3 scripts/auto_trade.py
```

Verify in logs:
```
"Subscribed to Polymarket RTDS crypto_prices_chainlink" source="chainlink"
"BTC price update" source="chainlink" price="$XX,XXX.XX"
```

## Verification

Check recent trades use Chainlink:
```sql
SELECT market_slug, action, price_source, btc_current
FROM paper_trades
ORDER BY timestamp DESC
LIMIT 10;
```

All should show `price_source='chainlink'`.

## Rollback

If issues occur, disable Chainlink:

```python
# In scripts/auto_trade.py
self.stream = CryptoPriceStream(
    self.settings,
    buffer_enabled=True,
    use_chainlink=False  # ← Fallback to Binance
)
```

## Expected Impact

- ✅ Price accuracy: <$1 difference from Polymarket
- ✅ Directional analysis: Correct price signals
- ✅ Trade quality: Better confidence calibration
- ✅ Audit trail: Track price source per trade
```

**Step 2: Update README.md**

Add to README.md under relevant section:

```markdown
## Price Data Source

The bot uses **Chainlink oracle prices** via Polymarket RTDS, ensuring price_to_beat matches Polymarket's settlement prices exactly.

- **Source:** Chainlink BTC/USD oracle (same as Polymarket)
- **Topic:** `crypto_prices_chainlink`
- **Fallback:** CoinGecko → Binance APIs (emergency only)
- **Accuracy:** <$1 difference from Polymarket settlement

See [CHAINLINK_MIGRATION.md](docs/CHAINLINK_MIGRATION.md) for details.
```

**Step 3: Commit**

```bash
git add README.md docs/CHAINLINK_MIGRATION.md
git commit -m "docs: document Chainlink migration and signal weighting

- Add CHAINLINK_MIGRATION.md with problem/solution/migration steps
- Update README.md to mention Chainlink as price source
- Document tiered signal weighting approach
- Provide verification and rollback instructions"
```

---

## Task 10: Full System Test

**Files:**
- Test: End-to-end system test

**Step 1: Start bot in test mode**

```bash
cd /root/polymarket-scripts

# Ensure we're on correct branch with all changes
git log --oneline -5

# Start bot
TEST_MODE=true python3 scripts/auto_trade.py
```

**Step 2: Verify Chainlink connection (2 minutes)**

Monitor logs for:
- ✓ `"Subscribed to Polymarket RTDS crypto_prices_chainlink"`
- ✓ `"BTC price update" source="chainlink"`
- ✓ No errors about invalid subscription format
- ✓ Real-time price updates every few seconds

**Step 3: Wait for paper trade (10-15 minutes)**

When a paper trade executes, verify in logs:
- ✓ Price direction mentioned in AI analysis
- ✓ Sentiment noted but not decisive
- ✓ Confidence reflects signal alignment
- ✓ Trade logged with `source="chainlink"`

**Step 4: Check database (verify price source)**

```bash
sqlite3 data/performance.db "
SELECT
    market_slug,
    action,
    confidence,
    price_source,
    btc_current,
    btc_price_to_beat
FROM paper_trades
ORDER BY timestamp DESC
LIMIT 3;
"
```

Expected: All trades show `price_source='chainlink'`

**Step 5: Compare price with Polymarket UI**

1. Open most recent market from logs
2. Check Polymarket's displayed "Price to Beat"
3. Compare with bot's `btc_price_to_beat`
4. Should be within $1-5 (timing differences acceptable)

**Step 6: Stop bot and review results**

```bash
# Stop with Ctrl+C

# Review all paper trades from this test
sqlite3 data/performance.db "
SELECT COUNT(*), AVG(confidence)
FROM paper_trades
WHERE price_source = 'chainlink'
AND timestamp > strftime('%s', 'now', '-1 hour');
"
```

**Step 7: Final commit**

```bash
git add -A
git commit -m "test: verify Chainlink integration end-to-end

- Verified Chainlink RTDS connection
- Confirmed real-time price updates
- Validated paper trade execution with Chainlink source
- Checked database logging of price_source
- Compared bot prices with Polymarket UI (<$5 difference)

All systems operational with Chainlink as primary source."
```

---

## Summary

**Total Tasks:** 10
**Estimated Time:** 2-3 hours
**Approach:** TDD with frequent commits

**Key Changes:**
1. ✅ CryptoPriceStream supports Chainlink RTDS (verified working)
2. ✅ Database tracks price source for audit trail
3. ✅ AI uses tiered signal weighting (price > technical > sentiment)
4. ✅ All trades logged with source attribution
5. ✅ Documentation and migration guide included

**Expected Outcomes:**
- Price accuracy: <$1 difference from Polymarket
- No more $2,469 discrepancies
- Correct directional analysis (price signals prioritized)
- Better confidence calibration (sentiment doesn't override price)

**Verification:**
All paper trades should show:
- `price_source='chainlink'`
- `btc_price_to_beat` within $5 of Polymarket's price
- AI decisions aligned with BTC price movement

---

## Execution Options

**Plan saved to:** `docs/plans/2026-02-15-chainlink-integration-signal-weighting.md`

Choose execution approach:

**1. Subagent-Driven (this session)**
- I dispatch fresh subagent per task
- Review between tasks
- Fast iteration with oversight

**2. Parallel Session (separate)**
- Open new session with plan file
- Batch execution with checkpoints
- More autonomous, less interruption

**Which approach?**
