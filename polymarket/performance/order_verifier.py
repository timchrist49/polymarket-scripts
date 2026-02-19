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

            # Polymarket API returns 'size_matched' not 'fillAmount'
            fill_amount = order_status.get('size_matched', order_status.get('fillAmount'))

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
            # Polymarket API returns 'size_matched' not 'fillAmount'
            fill_amount = float(order_status.get('size_matched', order_status.get('fillAmount', 0)))
            order_size = float(order_status.get('original_size', order_status.get('size', 0)))
            price = float(order_status.get('price', 0))
            timestamp = order_status.get('created_at', order_status.get('timestamp'))
            associate_trades = order_status.get('associate_trades', [])

            # Determine if order was filled
            verified = status in ['MATCHED', 'FILLED', 'PARTIALLY_MATCHED']
            partial_fill = (fill_amount < order_size) if order_size > 0 else False

            # Try to get transaction hash from trade history
            transaction_hash = await self._get_transaction_hash(order_id, timestamp, associate_trades)

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

    async def _get_transaction_hash(
        self,
        order_id: str,
        timestamp: Optional[int],
        associate_trades: Optional[List[str]] = None
    ) -> Optional[str]:
        """
        Extract transaction hash via associate_trades on the order.

        The Polymarket CLOB order response includes 'associate_trades' (list of trade UUIDs).
        We match these against the trade history to find the blockchain transaction hash.

        Args:
            order_id: Order ID for logging
            timestamp: Order timestamp (unused, kept for API compatibility)
            associate_trades: List of trade UUIDs from the order response

        Returns:
            Transaction hash string or None if not found
        """
        if not associate_trades:
            return None

        try:
            # Get recent trade history synchronously via the CLOB client
            # This runs in the settlement loop (every 5 min) so blocking briefly is acceptable
            loop = asyncio.get_event_loop()
            all_trades = await loop.run_in_executor(
                None, self.client._get_clob_client().get_trades
            )

            for trade in all_trades:
                if trade.get('id') in associate_trades:
                    tx_hash = trade.get('transaction_hash')
                    if tx_hash:
                        logger.debug(
                            "Found transaction hash",
                            order_id=order_id,
                            tx_hash=tx_hash[:16]
                        )
                        return tx_hash

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
