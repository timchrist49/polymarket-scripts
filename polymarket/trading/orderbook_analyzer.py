"""Orderbook depth analysis for execution quality assessment."""
import structlog
from datetime import datetime
from typing import Optional
from polymarket.models import OrderbookData

logger = structlog.get_logger()


class OrderbookAnalyzer:
    """Analyze orderbook depth and liquidity."""

    def analyze_orderbook(
        self,
        orderbook: dict,
        target_size: float = 10.0  # Target trade size in USDC
    ) -> Optional[OrderbookData]:
        """
        Analyze orderbook depth, spread, and imbalance.

        Args:
            orderbook: Dict with 'bids' and 'asks' arrays [{"price": "...", "size": "..."}, ...]
            target_size: Expected trade size to check fillability

        Returns:
            OrderbookData with analysis results
        """
        try:
            bids = orderbook.get('bids', [])
            asks = orderbook.get('asks', [])

            if not bids or not asks:
                logger.warning("Empty orderbook")
                return None

            # Best bid/ask
            best_bid = float(bids[0]['price'])
            best_ask = float(asks[0]['price'])

            # Spread calculation
            spread = best_ask - best_bid
            spread_pct = (spread / best_ask) * 100
            spread_bps = spread_pct * 100  # Convert to basis points

            # Liquidity depth at 100bps and 200bps
            bid_depth_100bps = self._calculate_depth(bids, best_bid, 0.01, side='bid')
            ask_depth_100bps = self._calculate_depth(asks, best_ask, 0.01, side='ask')
            bid_depth_200bps = self._calculate_depth(bids, best_bid, 0.02, side='bid')
            ask_depth_200bps = self._calculate_depth(asks, best_ask, 0.02, side='ask')

            # Order imbalance (bid pressure vs ask pressure)
            total_bid_volume = sum(float(b['size']) for b in bids[:10])
            total_ask_volume = sum(float(a['size']) for a in asks[:10])

            if total_bid_volume + total_ask_volume == 0:
                order_imbalance = 0.0
            else:
                order_imbalance = (total_bid_volume - total_ask_volume) / (total_bid_volume + total_ask_volume)

            # Imbalance direction
            if order_imbalance > 0.2:
                imbalance_direction = "BUY_PRESSURE"
            elif order_imbalance < -0.2:
                imbalance_direction = "SELL_PRESSURE"
            else:
                imbalance_direction = "BALANCED"

            # Liquidity score (0.0-1.0)
            # Good: tight spread + deep liquidity
            spread_score = max(0, 1 - (spread_bps / 500))  # 500bps = 0 score
            depth_score = min(1, (bid_depth_100bps + ask_depth_100bps) / 1000)  # $1000 = 1.0 score
            liquidity_score = (spread_score * 0.6) + (depth_score * 0.4)

            # Can fill order?
            can_fill_order = (bid_depth_100bps >= target_size or ask_depth_100bps >= target_size)

            logger.info(
                "Orderbook analyzed",
                spread_bps=f"{spread_bps:.1f}",
                imbalance=f"{order_imbalance:+.2f}",
                liquidity_score=f"{liquidity_score:.2f}",
                can_fill=can_fill_order
            )

            return OrderbookData(
                bid_ask_spread=spread_pct,
                spread_bps=spread_bps,
                liquidity_score=liquidity_score,
                order_imbalance=order_imbalance,
                imbalance_direction=imbalance_direction,
                bid_depth_100bps=bid_depth_100bps,
                ask_depth_100bps=ask_depth_100bps,
                bid_depth_200bps=bid_depth_200bps,
                ask_depth_200bps=ask_depth_200bps,
                best_bid=best_bid,
                best_ask=best_ask,
                can_fill_order=can_fill_order,
                timestamp=datetime.now()
            )

        except Exception as e:
            logger.error("Failed to analyze orderbook", error=str(e))
            return None

    def _calculate_depth(
        self,
        orders: list,
        reference_price: float,
        threshold: float,
        side: str
    ) -> float:
        """
        Calculate total liquidity within threshold from reference price.

        Args:
            orders: List of {"price": "...", "size": "..."} dicts
            reference_price: Best bid/ask price
            threshold: Distance threshold (e.g., 0.01 for 100bps)
            side: 'bid' or 'ask'

        Returns:
            Total size within threshold
        """
        total = 0.0

        for order in orders:
            price = float(order['price'])
            size = float(order['size'])

            if side == 'bid':
                # For bids, check if within threshold below best bid
                if price >= reference_price - threshold:
                    total += size
            else:
                # For asks, check if within threshold above best ask
                if price <= reference_price + threshold:
                    total += size

        return total
