# Order Verification and P&L Tracking - Implementation Plan

**Status:** Ready for Execution
**Created:** 2026-02-14
**Approach:** Hybrid Two-Phase (Design approved in 2026-02-14-order-verification-pnl-tracking-design.md)

---

## Executive Summary

Detailed implementation plan for adding order verification and accurate P&L tracking to the Polymarket trading bot using the Hybrid Two-Phase approach:
- **Phase 1:** Quick status check (2s timeout) immediately after order placement
- **Phase 2:** Full verification with fill details before settlement (at 15+ minute mark)

## Implementation Sequence

### Step 1: Create OrderVerifier Service
**File:** `/root/polymarket-scripts/polymarket/performance/order_verifier.py` (NEW)

**Code:**
```python
"""Order verification service for Polymarket trades."""

import asyncio
import structlog
from typing import Dict, Optional, List
from datetime import datetime

logger = structlog.get_logger()


class OrderVerifier:
    """Verifies order execution and extracts actual fill data from Polymarket API."""

    def __init__(self, client, db):
        """
        Initialize order verifier.

        Args:
            client: PolymarketClient instance
            db: PerformanceDatabase instance
        """
        self.client = client
        self.db = db

    async def check_order_quick(self, order_id: str, trade_id: int, timeout: float = 2.0) -> Dict:
        """
        Phase 1: Quick status check immediately after order placement.

        Uses 2-second timeout for fast feedback. Returns basic status only.

        Args:
            order_id: Polymarket order ID from execution response
            trade_id: Database trade ID for logging
            timeout: Maximum time to wait (default 2.0 seconds)

        Returns:
            {
                'status': 'filled'|'pending'|'failed',
                'fill_amount': float | None,
                'needs_alert': bool,
                'raw_status': str  # Original API status
            }
        """
        try:
            # Call check_order_status with timeout
            order_status = await asyncio.wait_for(
                self.client.check_order_status(order_id),
                timeout=timeout
            )

            # Map Polymarket status to our simplified status
            # Possible statuses: MATCHED, PARTIALLY_MATCHED, LIVE, PENDING, CANCELLED
            raw_status = order_status.get('status', 'UNKNOWN')

            if raw_status in ['MATCHED', 'FILLED']:
                status = 'filled'
                needs_alert = False
            elif raw_status in ['PARTIALLY_MATCHED']:
                status = 'filled'  # Partially filled but OK
                needs_alert = True  # Alert for partial fill
            elif raw_status in ['LIVE', 'PENDING']:
                status = 'pending'
                needs_alert = False  # Normal for limit orders
            elif raw_status in ['CANCELLED', 'FAILED', 'REJECTED']:
                status = 'failed'
                needs_alert = True  # Critical failure
            else:
                status = 'unknown'
                needs_alert = True

            fill_amount = order_status.get('fillAmount')

            logger.info(
                "Quick order check complete",
                order_id=order_id,
                trade_id=trade_id,
                status=status,
                raw_status=raw_status,
                fill_amount=fill_amount
            )

            return {
                'status': status,
                'fill_amount': float(fill_amount) if fill_amount else None,
                'needs_alert': needs_alert,
                'raw_status': raw_status
            }

        except asyncio.TimeoutError:
            logger.warning(
                "Quick order check timed out",
                order_id=order_id,
                trade_id=trade_id,
                timeout=timeout
            )
            return {
                'status': 'pending',
                'fill_amount': None,
                'needs_alert': False,
                'raw_status': 'TIMEOUT'
            }

        except Exception as e:
            logger.error(
                "Quick order check failed",
                order_id=order_id,
                trade_id=trade_id,
                error=str(e)
            )
            return {
                'status': 'unknown',
                'fill_amount': None,
                'needs_alert': True,
                'raw_status': 'ERROR'
            }

    async def verify_order_full(self, order_id: str) -> Dict:
        """
        Phase 2: Full verification at settlement time (15+ minutes after execution).

        Gets complete fill details including actual fill price, amount, and transaction hash.

        Args:
            order_id: Polymarket order ID

        Returns:
            {
                'verified': bool,  # True if order was filled
                'status': str,  # 'MATCHED', 'PARTIALLY_MATCHED', 'CANCELLED', etc.
                'fill_amount': float,  # Actual amount filled (shares)
                'fill_price': float,  # Actual fill price per share
                'transaction_hash': str | None,  # Blockchain tx hash
                'fill_timestamp': int | None,  # Unix timestamp of fill
                'partial_fill': bool,  # True if not fully filled
                'original_size': float,  # Original order size for comparison
            }
        """
        try:
            # Get order status (includes fill amount and price)
            order_status = await self.client.check_order_status(order_id)

            status = order_status.get('status', 'UNKNOWN')
            fill_amount = float(order_status.get('fillAmount', 0))
            order_size = float(order_status.get('size', 0))
            price = float(order_status.get('price', 0))
            timestamp = order_status.get('timestamp')

            # Determine if order was filled
            verified = status in ['MATCHED', 'FILLED', 'PARTIALLY_MATCHED']
            partial_fill = (fill_amount < order_size) if order_size > 0 else False

            # Try to get transaction hash from trade history
            transaction_hash = await self._get_transaction_hash(order_id, timestamp)

            logger.info(
                "Full order verification complete",
                order_id=order_id,
                verified=verified,
                status=status,
                fill_amount=fill_amount,
                fill_price=price,
                partial_fill=partial_fill
            )

            return {
                'verified': verified,
                'status': status,
                'fill_amount': fill_amount,
                'fill_price': price,
                'transaction_hash': transaction_hash,
                'fill_timestamp': timestamp,
                'partial_fill': partial_fill,
                'original_size': order_size
            }

        except Exception as e:
            logger.error(
                "Full order verification failed",
                order_id=order_id,
                error=str(e)
            )
            return {
                'verified': False,
                'status': 'ERROR',
                'fill_amount': 0.0,
                'fill_price': 0.0,
                'transaction_hash': None,
                'fill_timestamp': None,
                'partial_fill': False,
                'original_size': 0.0
            }

    async def _get_transaction_hash(self, order_id: str, timestamp: Optional[int]) -> Optional[str]:
        """
        Extract transaction hash from trade history.

        Strategy:
        1. Call client.get_trades() to get trade history
        2. Filter by timestamp (±60 seconds) to find matching trade
        3. Extract transaction_hash from trade record

        Args:
            order_id: Order ID to find
            timestamp: Order timestamp for filtering (Unix timestamp)

        Returns:
            Transaction hash string or None if not found
        """
        try:
            # Get recent trade history
            # Note: py_clob_client may not directly support order_id filter
            # So we'll need to filter by timestamp
            portfolio = self.client.get_portfolio_summary()

            # portfolio should have trade history
            # This is a placeholder - actual implementation depends on API structure
            # May need to parse from portfolio.trades or similar

            logger.debug(
                "Transaction hash lookup",
                order_id=order_id,
                timestamp=timestamp
            )

            # TODO: Implement actual trade history lookup
            # For now, return None (non-blocking)
            return None

        except Exception as e:
            logger.warning(
                "Failed to get transaction hash",
                order_id=order_id,
                error=str(e)
            )
            return None

    def calculate_price_discrepancy(
        self,
        estimated_price: float,
        actual_price: float
    ) -> float:
        """
        Calculate percentage discrepancy between estimated and actual fill price.

        Args:
            estimated_price: Price expected at decision time
            actual_price: Actual fill price from API

        Returns:
            Percentage discrepancy (positive = paid more than expected)
        """
        if estimated_price == 0:
            return 0.0

        discrepancy_pct = ((actual_price - estimated_price) / estimated_price) * 100
        return discrepancy_pct
```

**Success Criteria:**
- ✅ check_order_quick() returns status within 2 seconds
- ✅ verify_order_full() returns complete fill data
- ✅ Handles timeouts gracefully
- ✅ Handles API errors without crashing

---

### Step 2: Database Migration for Verification Columns
**File:** `/root/polymarket-scripts/migrations/add_order_verification.sql` (NEW)

**Migration SQL:**
```sql
-- Migration: Add order verification columns to trades table
-- Run date: 2026-02-14
-- Author: Claude Code

-- Add verification columns
ALTER TABLE trades ADD COLUMN verified_fill_price REAL;
ALTER TABLE trades ADD COLUMN verified_fill_amount REAL;
ALTER TABLE trades ADD COLUMN transaction_hash TEXT;
ALTER TABLE trades ADD COLUMN fill_timestamp INTEGER;
ALTER TABLE trades ADD COLUMN partial_fill BOOLEAN DEFAULT 0;
ALTER TABLE trades ADD COLUMN verification_status TEXT DEFAULT 'unverified';
ALTER TABLE trades ADD COLUMN verification_timestamp INTEGER;
ALTER TABLE trades ADD COLUMN price_discrepancy_pct REAL;
ALTER TABLE trades ADD COLUMN amount_discrepancy_pct REAL;
ALTER TABLE trades ADD COLUMN skip_reason TEXT;
ALTER TABLE trades ADD COLUMN skip_type TEXT;

-- Create indexes for fast lookups
CREATE INDEX IF NOT EXISTS idx_trades_order_id ON trades(order_id) WHERE order_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_trades_verification_status ON trades(verification_status);
CREATE INDEX IF NOT EXISTS idx_trades_execution_status ON trades(execution_status);

-- Verify migration
SELECT COUNT(*) as trades_count FROM trades;
PRAGMA table_info(trades);
```

**Python Migration Function:**
Add to `/root/polymarket-scripts/polymarket/performance/database.py`:

```python
def _migrate_add_verification_columns(self):
    """Add order verification columns to trades table."""
    cursor = self.conn.cursor()

    # Get existing columns
    cursor.execute("PRAGMA table_info(trades)")
    existing_columns = {row[1] for row in cursor.fetchall()}

    # Add verification columns if they don't exist
    verification_columns = [
        ("verified_fill_price", "REAL"),
        ("verified_fill_amount", "REAL"),
        ("transaction_hash", "TEXT"),
        ("fill_timestamp", "INTEGER"),
        ("partial_fill", "BOOLEAN DEFAULT 0"),
        ("verification_status", "TEXT DEFAULT 'unverified'"),
        ("verification_timestamp", "INTEGER"),
        ("price_discrepancy_pct", "REAL"),
        ("amount_discrepancy_pct", "REAL"),
        ("skip_reason", "TEXT"),
        ("skip_type", "TEXT"),
    ]

    for column_name, column_type in verification_columns:
        if column_name not in existing_columns:
            try:
                cursor.execute(f"ALTER TABLE trades ADD COLUMN {column_name} {column_type}")
                logger.info(f"Added verification column: {column_name}")
            except sqlite3.OperationalError as e:
                logger.warning(f"Could not add column {column_name}: {e}")

    # Create indexes
    try:
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_trades_order_id
            ON trades(order_id) WHERE order_id IS NOT NULL
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_trades_verification_status
            ON trades(verification_status)
        """)
    except sqlite3.OperationalError as e:
        logger.warning(f"Index creation failed: {e}")

    self.conn.commit()
    logger.info("Order verification migration complete")
```

**Integration:**
Update `__init__` method in PerformanceDatabase class:
```python
def __init__(self, db_path: str = "data/performance.db"):
    # ... existing code ...

    # Run migrations
    self._migrate_add_timeframe_columns()
    self._migrate_add_verification_columns()  # NEW

    logger.info("Performance database initialized", db_path=db_path)
```

**Success Criteria:**
- ✅ Migration runs without errors on existing database
- ✅ New columns appear in trades table
- ✅ Indexes created successfully
- ✅ Existing data preserved

---

### Step 3: Enhanced TradeSettler with Verification
**File:** `/root/polymarket-scripts/polymarket/performance/settler.py` (UPDATE)

**Changes:**

1. **Add OrderVerifier to __init__:**
```python
def __init__(self, db: PerformanceDatabase, btc_fetcher, order_verifier = None):
    """
    Initialize trade settler.

    Args:
        db: Performance database
        btc_fetcher: BTC price service (from auto_trade.py)
        order_verifier: OrderVerifier instance for order verification (optional)
    """
    self.db = db
    self.btc_fetcher = btc_fetcher
    self.order_verifier = order_verifier  # NEW
```

2. **Update _get_unsettled_trades to include order_id:**
```python
def _get_unsettled_trades(self, batch_size: int = 50) -> list[dict]:
    """Query unsettled trades from database."""
    cursor = self.db.conn.cursor()

    cursor.execute("""
        SELECT
            id, timestamp, market_slug, action,
            position_size, executed_price, price_to_beat,
            order_id, verification_status  -- NEW: Include verification fields
        FROM trades
        WHERE action IN ('YES', 'NO')
          AND is_win IS NULL
          AND execution_status = 'executed'
          AND datetime(timestamp) < datetime('now', '-15 minutes')
        ORDER BY timestamp ASC
        LIMIT ?
    """, (batch_size,))

    trades = []
    for row in cursor.fetchall():
        trades.append({
            'id': row[0],
            'timestamp': row[1],
            'market_slug': row[2],
            'action': row[3],
            'position_size': row[4],
            'executed_price': row[5],
            'price_to_beat': row[6],
            'order_id': row[7],  # NEW
            'verification_status': row[8]  # NEW
        })

    return trades
```

3. **Add verification logic to settle_pending_trades:**
```python
async def settle_pending_trades(self, batch_size: int = 50) -> Dict:
    """Settle all pending trades with order verification."""
    stats = {
        "success": True,
        "settled_count": 0,
        "wins": 0,
        "losses": 0,
        "total_profit": 0.0,
        "pending_count": 0,
        "verification_failures": 0,  # NEW
        "price_discrepancies": 0,  # NEW
        "partial_fills": 0,  # NEW
        "errors": []
    }

    try:
        trades = self._get_unsettled_trades(batch_size)

        logger.info(
            "Starting settlement cycle with verification",
            pending_trades=len(trades),
            verifier_enabled=self.order_verifier is not None
        )

        for trade in trades:
            try:
                # NEW: Verify order execution BEFORE calculating P&L
                actual_price = trade['executed_price']
                actual_size = trade['position_size']
                tx_hash = None

                if self.order_verifier and trade.get('order_id'):
                    verification = await self.order_verifier.verify_order_full(
                        trade['order_id']
                    )

                    if not verification['verified']:
                        # Order never filled - mark as failed
                        logger.warning(
                            "Order verification failed - trade not filled",
                            trade_id=trade['id'],
                            order_id=trade['order_id'],
                            status=verification['status']
                        )

                        self._mark_trade_failed(trade['id'], verification)
                        stats['verification_failures'] += 1
                        continue  # Skip P&L calculation

                    # Use verified data
                    actual_price = verification['fill_price']
                    actual_size = verification['fill_amount']
                    tx_hash = verification['transaction_hash']

                    # Calculate discrepancy
                    estimated_price = trade['executed_price']
                    price_discrepancy_pct = self.order_verifier.calculate_price_discrepancy(
                        estimated_price, actual_price
                    )

                    # Alert on large discrepancies
                    if abs(price_discrepancy_pct) > 5.0:
                        logger.warning(
                            "Large price discrepancy detected",
                            trade_id=trade['id'],
                            estimated=f"${estimated_price:.3f}",
                            actual=f"${actual_price:.3f}",
                            discrepancy=f"{price_discrepancy_pct:+.2f}%"
                        )
                        stats['price_discrepancies'] += 1

                    # Track partial fills
                    if verification['partial_fill']:
                        logger.info(
                            "Partial fill detected",
                            trade_id=trade['id'],
                            filled=verification['fill_amount'],
                            expected=verification['original_size'],
                            fill_pct=f"{(verification['fill_amount'] / verification['original_size'] * 100):.1f}%"
                        )
                        stats['partial_fills'] += 1

                    # Store verification data
                    self._update_verification_data(
                        trade_id=trade['id'],
                        verification=verification,
                        price_discrepancy_pct=price_discrepancy_pct
                    )

                # Parse close timestamp (existing code)
                close_timestamp = self._parse_market_close_timestamp(trade['market_slug'])
                if close_timestamp is None:
                    error_msg = f"Failed to parse timestamp from {trade['market_slug']}"
                    logger.error(error_msg, trade_id=trade['id'])
                    stats['errors'].append(error_msg)
                    continue

                # Fetch BTC price at close (existing code)
                btc_close_price = await self.btc_fetcher.get_price_at_timestamp(close_timestamp)
                if btc_close_price is None:
                    logger.warning(
                        "BTC price unavailable, will retry",
                        trade_id=trade['id'],
                        timestamp=close_timestamp
                    )
                    stats['pending_count'] += 1
                    continue

                btc_close_price = float(btc_close_price)

                # Determine outcome (existing code)
                actual_outcome = self._determine_outcome(
                    btc_close_price=btc_close_price,
                    price_to_beat=trade['price_to_beat']
                )

                # Calculate profit/loss using VERIFIED data
                profit_loss, is_win = self._calculate_profit_loss(
                    action=trade['action'],
                    actual_outcome=actual_outcome,
                    position_size=actual_size,  # Use verified size
                    executed_price=actual_price  # Use verified price
                )

                # Update database (existing code)
                if hasattr(self, '_tracker'):
                    self._tracker.update_trade_outcome(
                        trade_id=trade['id'],
                        actual_outcome=actual_outcome,
                        profit_loss=profit_loss,
                        is_win=is_win
                    )
                else:
                    cursor = self.db.conn.cursor()
                    cursor.execute("""
                        UPDATE trades
                        SET actual_outcome = ?,
                            profit_loss = ?,
                            is_win = ?
                        WHERE id = ?
                    """, (actual_outcome, profit_loss, is_win, trade['id']))
                    self.db.conn.commit()

                # Update stats (existing code)
                stats['settled_count'] += 1
                if is_win:
                    stats['wins'] += 1
                else:
                    stats['losses'] += 1
                stats['total_profit'] += profit_loss

                logger.info(
                    "Trade settled with verification",
                    trade_id=trade['id'],
                    action=trade['action'],
                    outcome=actual_outcome,
                    is_win=is_win,
                    profit_loss=f"${profit_loss:.2f}",
                    verified=self.order_verifier is not None and trade.get('order_id') is not None
                )

            except Exception as e:
                error_msg = f"Failed to settle trade {trade.get('id', '?')}: {str(e)}"
                logger.error(error_msg)
                stats['errors'].append(error_msg)
                continue
            finally:
                await asyncio.sleep(2)  # Rate limit

    except Exception as e:
        logger.error("Settlement cycle failed", error=str(e))
        stats['success'] = False
        stats['errors'].append(str(e))

    return stats
```

4. **Add helper methods:**
```python
def _mark_trade_failed(self, trade_id: int, verification: dict) -> None:
    """Mark a trade as failed due to verification failure."""
    cursor = self.db.conn.cursor()
    cursor.execute("""
        UPDATE trades
        SET verification_status = 'failed',
            verification_timestamp = ?,
            skip_reason = ?
        WHERE id = ?
    """, (
        int(datetime.now().timestamp()),
        f"Order not filled: {verification['status']}",
        trade_id
    ))
    self.db.conn.commit()

def _update_verification_data(
    self,
    trade_id: int,
    verification: dict,
    price_discrepancy_pct: float
) -> None:
    """Update trade with verification data."""
    cursor = self.db.conn.cursor()
    cursor.execute("""
        UPDATE trades
        SET verified_fill_price = ?,
            verified_fill_amount = ?,
            transaction_hash = ?,
            fill_timestamp = ?,
            partial_fill = ?,
            verification_status = 'verified',
            verification_timestamp = ?,
            price_discrepancy_pct = ?
        WHERE id = ?
    """, (
        verification['fill_price'],
        verification['fill_amount'],
        verification['transaction_hash'],
        verification['fill_timestamp'],
        verification['partial_fill'],
        int(datetime.now().timestamp()),
        price_discrepancy_pct,
        trade_id
    ))
    self.db.conn.commit()
```

**Success Criteria:**
- ✅ Settlement verifies orders before calculating P&L
- ✅ Uses verified fill prices and amounts
- ✅ Handles verification failures gracefully
- ✅ Logs price discrepancies and partial fills
- ✅ Stores transaction hashes

---

### Step 4: Integration with auto_trade.py
**File:** `/root/polymarket-scripts/scripts/auto_trade.py` (UPDATE)

**Changes:**

1. **Initialize OrderVerifier in __init__:**
```python
def __init__(self, settings: Settings, interval: int = 180):
    # ... existing code ...

    # NEW: Order verification
    from polymarket.performance.order_verifier import OrderVerifier
    self.order_verifier = OrderVerifier(
        client=self.client,
        db=self.performance_tracker.db
    )

    # Trade settlement (update to pass verifier)
    self.trade_settler = TradeSettler(
        db=self.performance_tracker.db,
        btc_fetcher=self.btc_service,
        order_verifier=self.order_verifier  # NEW
    )
    # ... rest of init ...
```

2. **Add quick check after order execution in _execute_trade:**
```python
async def _execute_trade(
    self,
    market,
    decision,
    amount: Decimal,
    token_id: str,
    token_name: str,
    market_price: float,
    trade_id: int,
    cycle_start_time: datetime,
    btc_current: Optional[float] = None,
    btc_price_to_beat: Optional[float] = None,
    arbitrage_opportunity = None
) -> None:
    """Execute a trade order with JIT price fetching and safety checks."""
    try:
        # ... existing code up to order execution ...

        order_id = execution_result["order_id"]
        filled_via = execution_result.get("filled_via", "limit")

        # NEW: Phase 1 Quick Status Check (2 seconds)
        logger.info(
            "Running quick order verification",
            order_id=order_id,
            trade_id=trade_id
        )

        await asyncio.sleep(2)  # Wait for order to process

        quick_status = await self.order_verifier.check_order_quick(
            order_id=order_id,
            trade_id=trade_id,
            timeout=2.0
        )

        # Handle quick check results
        if quick_status['status'] == 'failed':
            logger.error(
                "Order failed immediately",
                order_id=order_id,
                trade_id=trade_id,
                raw_status=quick_status['raw_status']
            )
            # Update trade status
            if trade_id > 0:
                await self.performance_tracker.update_trade_status(
                    trade_id=trade_id,
                    execution_status='failed',
                    skip_reason=f"Order failed: {quick_status['raw_status']}"
                )
            return  # Don't count this as a successful trade

        elif quick_status['needs_alert']:
            logger.warning(
                "Order requires attention",
                order_id=order_id,
                trade_id=trade_id,
                status=quick_status['status'],
                raw_status=quick_status['raw_status']
            )
            # Send Telegram alert for partial fills or issues
            try:
                await self.telegram_bot.send_message(
                    f"⚠️ Order Alert\n"
                    f"Order ID: {order_id[:8]}...\n"
                    f"Status: {quick_status['raw_status']}\n"
                    f"Trade ID: {trade_id}"
                )
            except Exception as e:
                logger.warning("Failed to send alert", error=str(e))

        # Log success (existing code)
        logger.info(
            "Trade executed and verified",
            market_id=market.id,
            action=decision.action,
            token=token_name,
            amount=str(amount),
            order_id=order_id,
            filled_via=filled_via,
            quick_status=quick_status['status'],
            arbitrage_edge=f"{arbitrage_opportunity.edge_percentage:.1%}" if arbitrage_opportunity else "N/A"
        )

        # ... rest of existing code ...

    except Exception as e:
        logger.error(
            "Trade execution failed",
            market_id=market.id,
            error=str(e)
        )
```

**Success Criteria:**
- ✅ Quick check runs after every order execution
- ✅ Failed orders detected immediately
- ✅ Alerts sent for partial fills or failures
- ✅ Trade status updated correctly

---

### Step 5: Alert System
**File:** `/root/polymarket-scripts/polymarket/performance/alerts.py` (NEW)

**Code:**
```python
"""Alert system for order verification anomalies."""

import structlog
from typing import Optional

logger = structlog.get_logger()


class VerificationAlerts:
    """Send alerts for order verification issues."""

    def __init__(self, telegram_bot):
        """
        Initialize alert system.

        Args:
            telegram_bot: TelegramBot instance for sending alerts
        """
        self.telegram = telegram_bot

    async def alert_order_not_filled(self, trade_id: int, order_id: str, status: str):
        """
        Alert when order shows as unfilled in API.

        Args:
            trade_id: Database trade ID
            order_id: Polymarket order ID
            status: Order status from API
        """
        message = (
            f"🚨 Order Not Filled\n"
            f"Trade ID: {trade_id}\n"
            f"Order ID: {order_id[:8]}...\n"
            f"Status: {status}\n"
            f"Action: Check Polymarket UI"
        )

        try:
            await self.telegram.send_message(message)
            logger.info("Sent unfilled order alert", trade_id=trade_id)
        except Exception as e:
            logger.error("Failed to send alert", error=str(e))

    async def alert_price_mismatch(
        self,
        trade_id: int,
        estimated: float,
        actual: float,
        discrepancy_pct: float
    ):
        """
        Alert when fill price differs significantly from estimate.

        Args:
            trade_id: Database trade ID
            estimated: Estimated price at decision time
            actual: Actual fill price from API
            discrepancy_pct: Percentage difference
        """
        message = (
            f"⚠️ Price Mismatch\n"
            f"Trade ID: {trade_id}\n"
            f"Expected: ${estimated:.3f}\n"
            f"Actual: ${actual:.3f}\n"
            f"Discrepancy: {discrepancy_pct:+.2f}%\n"
            f"Impact: {'Favorable' if discrepancy_pct < 0 else 'Unfavorable'}"
        )

        try:
            await self.telegram.send_message(message)
            logger.info(
                "Sent price mismatch alert",
                trade_id=trade_id,
                discrepancy_pct=discrepancy_pct
            )
        except Exception as e:
            logger.error("Failed to send alert", error=str(e))

    async def alert_partial_fill(
        self,
        trade_id: int,
        expected: float,
        filled: float,
        fill_pct: float
    ):
        """
        Alert when order only partially fills.

        Args:
            trade_id: Database trade ID
            expected: Expected fill amount (shares)
            filled: Actual filled amount (shares)
            fill_pct: Percentage filled
        """
        message = (
            f"📊 Partial Fill\n"
            f"Trade ID: {trade_id}\n"
            f"Expected: {expected:.2f} shares\n"
            f"Filled: {filled:.2f} shares\n"
            f"Fill Rate: {fill_pct:.1f}%\n"
            f"Note: P&L calculated on filled amount only"
        )

        try:
            await self.telegram.send_message(message)
            logger.info(
                "Sent partial fill alert",
                trade_id=trade_id,
                fill_pct=fill_pct
            )
        except Exception as e:
            logger.error("Failed to send alert", error=str(e))

    async def alert_verification_failed(self, trade_id: int, error: str):
        """
        Alert when verification API call fails.

        Args:
            trade_id: Database trade ID
            error: Error message
        """
        message = (
            f"❌ Verification Failed\n"
            f"Trade ID: {trade_id}\n"
            f"Error: {error}\n"
            f"Fallback: Using estimated data"
        )

        try:
            await self.telegram.send_message(message)
            logger.info("Sent verification failure alert", trade_id=trade_id)
        except Exception as e:
            logger.error("Failed to send alert", error=str(e))
```

**Success Criteria:**
- ✅ Alerts sent to Telegram for all anomalies
- ✅ Clear, actionable messages
- ✅ Non-blocking (errors logged, not raised)

---

### Step 6: Unit Tests
**File:** `/root/polymarket-scripts/tests/test_order_verifier.py` (NEW)

**Code:**
```python
"""Unit tests for OrderVerifier."""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from polymarket.performance.order_verifier import OrderVerifier


class TestOrderVerifier:
    """Test order verification logic."""

    @pytest.fixture
    def mock_client(self):
        """Mock Polymarket client."""
        client = Mock()
        client.check_order_status = AsyncMock()
        return client

    @pytest.fixture
    def mock_db(self):
        """Mock database."""
        return Mock()

    @pytest.fixture
    def verifier(self, mock_client, mock_db):
        """Create OrderVerifier instance."""
        return OrderVerifier(mock_client, mock_db)

    @pytest.mark.asyncio
    async def test_quick_check_filled(self, verifier, mock_client):
        """Test quick check when order is filled."""
        mock_client.check_order_status.return_value = {
            'status': 'MATCHED',
            'fillAmount': '10.5',
            'price': '0.65'
        }

        result = await verifier.check_order_quick('order_123', trade_id=1)

        assert result['status'] == 'filled'
        assert result['fill_amount'] == 10.5
        assert result['needs_alert'] == False

    @pytest.mark.asyncio
    async def test_quick_check_partial_fill(self, verifier, mock_client):
        """Test quick check with partial fill."""
        mock_client.check_order_status.return_value = {
            'status': 'PARTIALLY_MATCHED',
            'fillAmount': '5.0',
            'size': '10.0'
        }

        result = await verifier.check_order_quick('order_123', trade_id=1)

        assert result['status'] == 'filled'
        assert result['needs_alert'] == True  # Alert for partial fill

    @pytest.mark.asyncio
    async def test_quick_check_failed(self, verifier, mock_client):
        """Test quick check when order fails."""
        mock_client.check_order_status.return_value = {
            'status': 'CANCELLED',
            'fillAmount': '0'
        }

        result = await verifier.check_order_quick('order_123', trade_id=1)

        assert result['status'] == 'failed'
        assert result['needs_alert'] == True

    @pytest.mark.asyncio
    async def test_quick_check_timeout(self, verifier, mock_client):
        """Test quick check with timeout."""
        # Simulate timeout
        async def slow_response(*args, **kwargs):
            await asyncio.sleep(5)  # Longer than 2s timeout
            return {}

        mock_client.check_order_status = slow_response

        result = await verifier.check_order_quick('order_123', trade_id=1, timeout=0.1)

        assert result['status'] == 'pending'
        assert result['raw_status'] == 'TIMEOUT'

    @pytest.mark.asyncio
    async def test_verify_order_full_success(self, verifier, mock_client):
        """Test full verification with successful order."""
        mock_client.check_order_status.return_value = {
            'status': 'MATCHED',
            'fillAmount': '10.0',
            'size': '10.0',
            'price': '0.65',
            'timestamp': 1707955200
        }

        result = await verifier.verify_order_full('order_123')

        assert result['verified'] == True
        assert result['fill_amount'] == 10.0
        assert result['fill_price'] == 0.65
        assert result['partial_fill'] == False

    @pytest.mark.asyncio
    async def test_verify_order_full_partial(self, verifier, mock_client):
        """Test full verification with partial fill."""
        mock_client.check_order_status.return_value = {
            'status': 'PARTIALLY_MATCHED',
            'fillAmount': '7.0',
            'size': '10.0',
            'price': '0.65',
            'timestamp': 1707955200
        }

        result = await verifier.verify_order_full('order_123')

        assert result['verified'] == True
        assert result['fill_amount'] == 7.0
        assert result['partial_fill'] == True
        assert result['original_size'] == 10.0

    @pytest.mark.asyncio
    async def test_verify_order_full_not_found(self, verifier, mock_client):
        """Test full verification when order not found."""
        mock_client.check_order_status.return_value = {
            'status': 'CANCELLED',
            'fillAmount': '0',
            'size': '10.0'
        }

        result = await verifier.verify_order_full('order_123')

        assert result['verified'] == False
        assert result['fill_amount'] == 0.0

    def test_calculate_price_discrepancy(self, verifier):
        """Test price discrepancy calculation."""
        # Paid more than expected
        discrepancy = verifier.calculate_price_discrepancy(
            estimated_price=0.60,
            actual_price=0.65
        )
        assert abs(discrepancy - 8.33) < 0.1  # ~8.33% higher

        # Paid less than expected (favorable)
        discrepancy = verifier.calculate_price_discrepancy(
            estimated_price=0.65,
            actual_price=0.60
        )
        assert abs(discrepancy - (-7.69)) < 0.1  # ~7.69% lower
```

**Run Tests:**
```bash
cd /root/polymarket-scripts
pytest tests/test_order_verifier.py -v
```

**Success Criteria:**
- ✅ All tests pass
- ✅ Edge cases covered (timeout, partial fill, failure)
- ✅ Mocks isolate unit under test

---

### Step 7: Integration Tests
**File:** `/root/polymarket-scripts/tests/test_settlement_integration.py` (NEW)

**Code:**
```python
"""Integration tests for settlement with order verification."""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, MagicMock
from decimal import Decimal
from polymarket.performance.settler import TradeSettler
from polymarket.performance.order_verifier import OrderVerifier
from polymarket.performance.database import PerformanceDatabase


class TestSettlementIntegration:
    """Integration tests for settlement with verification."""

    @pytest.fixture
    def mock_db(self):
        """Mock database with in-memory SQLite."""
        db = PerformanceDatabase(":memory:")

        # Insert test trade
        cursor = db.conn.cursor()
        cursor.execute("""
            INSERT INTO trades (
                timestamp, market_slug, action, confidence, position_size,
                btc_price, price_to_beat, executed_price,
                order_id, execution_status, is_test_mode
            ) VALUES (
                datetime('now', '-20 minutes'), 'btc-updown-15m-1707955200',
                'YES', 0.85, 10.0, 100000.0, 99500.0, 0.65,
                'order_123', 'executed', 0
            )
        """)
        db.conn.commit()

        return db

    @pytest.fixture
    def mock_btc_fetcher(self):
        """Mock BTC price fetcher."""
        fetcher = Mock()
        fetcher.get_price_at_timestamp = AsyncMock(return_value=Decimal("100500.0"))
        return fetcher

    @pytest.fixture
    def mock_client(self):
        """Mock Polymarket client."""
        client = Mock()
        client.check_order_status = AsyncMock(return_value={
            'status': 'MATCHED',
            'fillAmount': '10.0',
            'size': '10.0',
            'price': '0.66',  # Slightly different from estimated 0.65
            'timestamp': 1707955200
        })
        return client

    @pytest.fixture
    def verifier(self, mock_client, mock_db):
        """Create OrderVerifier."""
        return OrderVerifier(mock_client, mock_db)

    @pytest.fixture
    def settler(self, mock_db, mock_btc_fetcher, verifier):
        """Create TradeSettler with verification."""
        return TradeSettler(mock_db, mock_btc_fetcher, verifier)

    @pytest.mark.asyncio
    async def test_settlement_with_verification(self, settler, mock_db):
        """Test full settlement flow with order verification."""
        stats = await settler.settle_pending_trades(batch_size=10)

        # Check settlement stats
        assert stats['success'] == True
        assert stats['settled_count'] == 1
        assert stats['wins'] == 1  # BTC went up (100500 > 99500)
        assert stats['verification_failures'] == 0

        # Verify database was updated with verification data
        cursor = mock_db.conn.cursor()
        cursor.execute("SELECT * FROM trades WHERE order_id = 'order_123'")
        trade = cursor.fetchone()

        assert trade['verified_fill_price'] == 0.66
        assert trade['verified_fill_amount'] == 10.0
        assert trade['verification_status'] == 'verified'
        assert trade['is_win'] == 1
        assert trade['profit_loss'] > 0  # Winning trade

    @pytest.mark.asyncio
    async def test_settlement_with_failed_verification(self, mock_db, mock_btc_fetcher, mock_client):
        """Test settlement when order was not filled."""
        # Mock order as cancelled
        mock_client.check_order_status = AsyncMock(return_value={
            'status': 'CANCELLED',
            'fillAmount': '0',
            'size': '10.0'
        })

        verifier = OrderVerifier(mock_client, mock_db)
        settler = TradeSettler(mock_db, mock_btc_fetcher, verifier)

        stats = await settler.settle_pending_trades(batch_size=10)

        # Check stats
        assert stats['verification_failures'] == 1
        assert stats['settled_count'] == 0  # Should not settle failed orders

        # Verify database marked as failed
        cursor = mock_db.conn.cursor()
        cursor.execute("SELECT * FROM trades WHERE order_id = 'order_123'")
        trade = cursor.fetchone()

        assert trade['verification_status'] == 'failed'
        assert trade['is_win'] is None  # Not settled

    @pytest.mark.asyncio
    async def test_settlement_with_price_discrepancy(self, mock_db, mock_btc_fetcher, mock_client):
        """Test settlement with large price discrepancy."""
        # Mock order filled at much worse price
        mock_client.check_order_status = AsyncMock(return_value={
            'status': 'MATCHED',
            'fillAmount': '10.0',
            'size': '10.0',
            'price': '0.72',  # 10.8% worse than estimated 0.65
            'timestamp': 1707955200
        })

        verifier = OrderVerifier(mock_client, mock_db)
        settler = TradeSettler(mock_db, mock_btc_fetcher, verifier)

        stats = await settler.settle_pending_trades(batch_size=10)

        # Check stats
        assert stats['price_discrepancies'] == 1
        assert stats['settled_count'] == 1  # Still settle, just alert

        # Verify discrepancy was recorded
        cursor = mock_db.conn.cursor()
        cursor.execute("SELECT * FROM trades WHERE order_id = 'order_123'")
        trade = cursor.fetchone()

        assert trade['price_discrepancy_pct'] > 5.0  # Above alert threshold
        assert trade['verified_fill_price'] == 0.72

    @pytest.mark.asyncio
    async def test_settlement_with_partial_fill(self, mock_db, mock_btc_fetcher, mock_client):
        """Test settlement with partial fill."""
        # Mock partial fill (70% filled)
        mock_client.check_order_status = AsyncMock(return_value={
            'status': 'PARTIALLY_MATCHED',
            'fillAmount': '7.0',
            'size': '10.0',
            'price': '0.65',
            'timestamp': 1707955200
        })

        verifier = OrderVerifier(mock_client, mock_db)
        settler = TradeSettler(mock_db, mock_btc_fetcher, verifier)

        stats = await settler.settle_pending_trades(batch_size=10)

        # Check stats
        assert stats['partial_fills'] == 1
        assert stats['settled_count'] == 1

        # Verify partial fill recorded
        cursor = mock_db.conn.cursor()
        cursor.execute("SELECT * FROM trades WHERE order_id = 'order_123'")
        trade = cursor.fetchone()

        assert trade['partial_fill'] == 1
        assert trade['verified_fill_amount'] == 7.0  # Only 7 shares filled
        # P&L should be calculated on 7 shares, not 10
```

**Run Integration Tests:**
```bash
cd /root/polymarket-scripts
pytest tests/test_settlement_integration.py -v
```

**Success Criteria:**
- ✅ Settlement flow works end-to-end with verification
- ✅ Failed orders skipped from settlement
- ✅ Price discrepancies detected and logged
- ✅ Partial fills handled correctly

---

## Rollback Procedures

### If Migration Fails:
```sql
-- Rollback verification columns
ALTER TABLE trades DROP COLUMN verified_fill_price;
ALTER TABLE trades DROP COLUMN verified_fill_amount;
ALTER TABLE trades DROP COLUMN transaction_hash;
ALTER TABLE trades DROP COLUMN fill_timestamp;
ALTER TABLE trades DROP COLUMN partial_fill;
ALTER TABLE trades DROP COLUMN verification_status;
ALTER TABLE trades DROP COLUMN verification_timestamp;
ALTER TABLE trades DROP COLUMN price_discrepancy_pct;
ALTER TABLE trades DROP COLUMN amount_discrepancy_pct;
ALTER TABLE trades DROP COLUMN skip_reason;
ALTER TABLE trades DROP COLUMN skip_type;

DROP INDEX IF EXISTS idx_trades_order_id;
DROP INDEX IF EXISTS idx_trades_verification_status;
```

### If OrderVerifier Has Bugs:
- Set `order_verifier=None` in TradeSettler initialization
- System falls back to existing behavior (uses estimated prices)

### If Settlement Breaks:
```python
# Disable verification temporarily
self.trade_settler = TradeSettler(
    db=self.performance_tracker.db,
    btc_fetcher=self.btc_service,
    order_verifier=None  # Disable verification
)
```

---

## Testing Checklist

### Phase 1: Unit Tests
- [ ] Run `pytest tests/test_order_verifier.py -v`
- [ ] All tests pass
- [ ] Code coverage >80%

### Phase 2: Integration Tests
- [ ] Run `pytest tests/test_settlement_integration.py -v`
- [ ] All tests pass
- [ ] Verify database updates correctly

### Phase 3: Manual Testing
- [ ] Run bot in read-only mode for 1 hour
- [ ] Verify quick checks execute after orders
- [ ] Verify settlement uses verified data
- [ ] Check Telegram alerts for anomalies

### Phase 4: Production Deployment
- [ ] Backup production database
- [ ] Run migration on production DB
- [ ] Deploy code changes
- [ ] Monitor for 24 hours
- [ ] Verify P&L matches Polymarket UI

---

## Configuration

Add to `.env`:
```bash
# Order Verification Settings
ENABLE_ORDER_VERIFICATION=true
ENABLE_QUICK_STATUS_CHECK=true
QUICK_CHECK_TIMEOUT_SECONDS=2
PRICE_DISCREPANCY_ALERT_PCT=5.0
PARTIAL_FILL_ALERT_THRESHOLD_PCT=80.0
```

---

## Success Metrics

### Coverage Metrics:
- ✅ **Verification Coverage**: >95% of trades have verified fill data
- ✅ **Price Accuracy**: <2% average discrepancy between estimated and actual
- ✅ **Alert Precision**: <5% false positive rate on alerts
- ✅ **Zero Phantom Trades**: All P&L matches actual Polymarket records

### Performance Metrics:
- ✅ **Quick Check Latency**: <2s average response time
- ✅ **Settlement Success**: 100% of settlements use verified data
- ✅ **API Reliability**: <1% verification API failure rate

---

## Implementation Timeline

| Step | Time Estimate | Blocker Dependencies |
|------|---------------|---------------------|
| 1. Create OrderVerifier | 2 hours | None |
| 2. Database Migration | 1 hour | Step 1 complete |
| 3. Update TradeSettler | 2 hours | Steps 1-2 complete |
| 4. Update auto_trade.py | 1 hour | Steps 1-3 complete |
| 5. Create Alert System | 1 hour | Step 4 complete |
| 6. Unit Tests | 2 hours | Steps 1-5 complete |
| 7. Integration Tests | 2 hours | Steps 1-6 complete |
| **Total** | **11 hours** | Sequential |

---

## Next Steps

1. **Review this plan** and approve for execution
2. **Execute Step 1** (Create OrderVerifier)
3. **Run unit tests** after each step
4. **Deploy to test environment** after Step 7
5. **Monitor for 24 hours** before production

---

*Plan created by: Claude Code*
*Design reference: /root/polymarket-scripts/docs/plans/2026-02-14-order-verification-pnl-tracking-design.md*
*Ready for execution with: `/superpowers:execute-plan`*
