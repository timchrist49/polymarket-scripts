"""
Smart Order Executor

Executes trades using limit orders (maker) instead of market orders (taker)
to save 3-6% in fees. Uses urgency-based pricing and timeout strategies.

HIGH urgency: Aggressive pricing (0.1% improvement), 30s timeout, fallback
MEDIUM urgency: Moderate pricing (0.3% improvement), 60s timeout, maybe fallback
LOW urgency: Conservative pricing (0.5% improvement), 120s timeout, no fallback
"""

import asyncio
from typing import Literal
import structlog
from polymarket.client import PolymarketClient
from polymarket.models import LimitOrderStrategy

logger = structlog.get_logger()


class SmartOrderExecutor:
    """Execute trades using smart limit orders to save fees."""

    # Price improvement percentages
    AGGRESSIVE_IMPROVEMENT = 0.001  # 0.1% (high urgency)
    MODERATE_IMPROVEMENT = 0.003    # 0.3% (medium urgency)
    CONSERVATIVE_IMPROVEMENT = 0.005  # 0.5% (low urgency)

    # Timeout seconds
    HIGH_URGENCY_TIMEOUT = 30
    MEDIUM_URGENCY_TIMEOUT = 60
    LOW_URGENCY_TIMEOUT = 120

    # Fallback to market order?
    HIGH_URGENCY_FALLBACK = True
    MEDIUM_URGENCY_FALLBACK = True
    LOW_URGENCY_FALLBACK = False

    def __init__(self):
        """Initialize smart order executor."""
        pass

    async def execute_smart_order(
        self,
        client: PolymarketClient,
        token_id: str,
        side: Literal["BUY", "SELL"],
        amount: float,
        urgency: Literal["HIGH", "MEDIUM", "LOW"],
        current_best_ask: float,
        current_best_bid: float,
        tick_size: float = 0.001,
        timeout_override: int | None = None
    ) -> dict:
        """
        Execute order using limit order with urgency-based strategy.

        Args:
            client: PolymarketClient instance
            token_id: Token to trade
            side: BUY or SELL
            amount: Size in shares
            urgency: HIGH/MEDIUM/LOW (determines pricing and timeout)
            current_best_ask: Current best ask price
            current_best_bid: Current best bid price
            tick_size: Price tick size (default 0.001)
            timeout_override: Override default timeout (for testing)

        Returns:
            Execution result with status, order_id, filled_via

        Possible statuses:
        - FILLED: Order successfully filled
        - TIMEOUT: Order timed out (no fill)
        - ERROR: Execution failed
        """

        # Step 1: Calculate strategy parameters
        strategy = self._calculate_strategy(
            urgency, side, current_best_ask, current_best_bid, tick_size
        )

        timeout = timeout_override if timeout_override is not None else strategy.timeout_seconds

        logger.info(
            "Executing smart order",
            side=side,
            amount=amount,
            urgency=urgency,
            target_price=strategy.target_price,
            timeout=f"{timeout}s",
            fallback=strategy.fallback_to_market
        )

        # Validate price inputs
        if not (0.0 <= current_best_bid <= 1.0):
            logger.error("Invalid best_bid", best_bid=current_best_bid)
            return {"status": "ERROR", "error": f"Invalid best_bid: {current_best_bid}"}
        if not (0.0 <= current_best_ask <= 1.0):
            logger.error("Invalid best_ask", best_ask=current_best_ask)
            return {"status": "ERROR", "error": f"Invalid best_ask: {current_best_ask}"}
        if current_best_bid >= current_best_ask:
            logger.error("Crossed market", bid=current_best_bid, ask=current_best_ask)
            return {"status": "ERROR", "error": "Bid >= Ask (crossed market)"}

        try:
            # Step 2: Place limit order
            order_response = await client.place_limit_order(
                token_id=token_id,
                side=side,
                price=strategy.target_price,
                size=amount,
                tick_size=tick_size
            )

            order_id = order_response.get("orderID")
            if not order_id:
                return {"status": "ERROR", "error": "No order ID returned"}

            # Step 3: Monitor for fill
            fill_result = await self._monitor_order_fill(
                client, order_id, timeout
            )

            if fill_result["filled"]:
                logger.info(
                    "Limit order filled",
                    order_id=order_id,
                    fill_amount=fill_result.get("fill_amount")
                )
                return {
                    "status": "FILLED",
                    "order_id": order_id,
                    "filled_via": "limit",
                    "fill_amount": fill_result.get("fill_amount")
                }

            # Step 4: Handle timeout
            logger.warning("Limit order timed out", order_id=order_id, timeout=f"{timeout}s")

            # Final check before cancelling - order may have filled during timeout
            try:
                final_status = await client.check_order_status(order_id)
                if final_status.get("status") in ["MATCHED", "PARTIALLY_MATCHED"]:
                    fill_amount = final_status.get("fillAmount", "0")
                    if float(fill_amount) > 0:
                        logger.info("Order filled just before cancel", order_id=order_id)
                        return {
                            "status": "FILLED",
                            "order_id": order_id,
                            "filled_via": "limit",
                            "fill_amount": fill_amount
                        }
            except Exception as e:
                logger.warning("Failed final status check before cancel", error=str(e))

            # Now safe to cancel
            await client.cancel_order(order_id)

            # Step 5: Fallback to market if enabled
            if strategy.fallback_to_market:
                logger.info("Falling back to market order", urgency=urgency)

                # Use create_order() with market order type
                from polymarket.models import OrderRequest

                market_request = OrderRequest(
                    token_id=token_id,
                    side=side,
                    price=current_best_ask if side == "BUY" else current_best_bid,
                    size=amount,
                    order_type="market"
                )

                try:
                    market_response = client.create_order(market_request, dry_run=False)

                    logger.info("Market order fallback executed", order_id=market_response.order_id)

                    return {
                        "status": "FILLED",
                        "order_id": market_response.order_id,
                        "filled_via": "market",
                        "fill_amount": amount
                    }
                except Exception as e:
                    logger.error("Market order fallback failed", error=str(e))
                    return {"status": "ERROR", "error": f"Fallback failed: {e}"}
            else:
                logger.info("No fallback - skipping trade", urgency=urgency)
                return {
                    "status": "TIMEOUT",
                    "order_id": order_id,
                    "message": "Limit order timed out, no fallback"
                }

        except Exception as e:
            logger.error("Smart order execution failed", error=str(e))
            return {"status": "ERROR", "error": str(e)}

    def _calculate_strategy(
        self,
        urgency: str,
        side: str,
        best_ask: float,
        best_bid: float,
        tick_size: float
    ) -> LimitOrderStrategy:
        """Calculate pricing and timeout strategy based on urgency."""

        # Determine improvement percentage
        if urgency == "HIGH":
            improvement = self.AGGRESSIVE_IMPROVEMENT
            timeout = self.HIGH_URGENCY_TIMEOUT
            fallback = self.HIGH_URGENCY_FALLBACK
        elif urgency == "MEDIUM":
            improvement = self.MODERATE_IMPROVEMENT
            timeout = self.MEDIUM_URGENCY_TIMEOUT
            fallback = self.MEDIUM_URGENCY_FALLBACK
        else:  # LOW
            improvement = self.CONSERVATIVE_IMPROVEMENT
            timeout = self.LOW_URGENCY_TIMEOUT
            fallback = self.LOW_URGENCY_FALLBACK

        # Calculate target price
        if side == "BUY":
            # BUY: Start from best_bid and add small improvement
            # (closer to ask = more likely to fill)
            target_price = best_bid + improvement
            target_price = min(target_price, best_ask - tick_size)  # Don't cross the spread
        else:  # SELL
            # SELL: Start from best_ask and subtract small improvement
            target_price = best_ask - improvement
            target_price = max(target_price, best_bid + tick_size)  # Don't cross the spread

        # Round to tick size
        target_price = round(target_price / tick_size) * tick_size

        # Bounds check
        target_price = max(0.001, min(0.999, target_price))

        return LimitOrderStrategy(
            target_price=target_price,
            timeout_seconds=timeout,
            fallback_to_market=fallback,
            urgency=urgency,
            price_improvement_pct=improvement
        )

    async def _monitor_order_fill(
        self,
        client: PolymarketClient,
        order_id: str,
        timeout: int,
        check_interval: int = 5
    ) -> dict:
        """
        Monitor order until filled or timeout.

        Args:
            client: PolymarketClient
            order_id: Order to monitor
            timeout: Max seconds to wait
            check_interval: Seconds between status checks

        Returns:
            {"filled": bool, "fill_amount": str}
        """
        elapsed = 0

        while elapsed < timeout:
            await asyncio.sleep(check_interval)
            elapsed += check_interval

            try:
                status = await client.check_order_status(order_id)

                # Check if filled
                if status.get("status") in ["MATCHED", "PARTIALLY_MATCHED"]:
                    fill_amount = status.get("fillAmount", "0")
                    if float(fill_amount) > 0:
                        return {"filled": True, "fill_amount": fill_amount}

            except Exception as e:
                logger.warning("Failed to check order status", error=str(e))
                # Continue monitoring despite errors

        # Timeout reached
        return {"filled": False}
