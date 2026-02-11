"""
Polymarket CLOB API Client.

This module provides a high-level Python client for interacting with the
Polymarket Central Limit Order Book (CLOB) API. It handles authentication,
order placement, market data fetching, and portfolio management.

Classes:
    PolymarketClient: Main client for Polymarket CLOB operations

Example:
    >>> from polymarket import PolymarketClient
    >>> client = PolymarketClient()
    >>> markets = client.get_markets(search="btc")
    >>> portfolio = client.get_portfolio_summary()

Authentication:
    - L1 (Private key): Required for signing requests
    - L2 (API credentials): Required for trading operations
    - Supports both Web3 wallets and Gmail/Magic Link accounts

Note:
    Market orders use FOK (Fill-Or-Kill) for immediate execution.
    Limit orders use GTC (Good-Til-Cancelled) and may not fill
    before 15-min markets expire.
"""

from datetime import datetime, timezone, timedelta
from typing import Literal, Any
import json
import requests

from polymarket.config import get_settings
from polymarket.auth import get_auth_manager
from polymarket.models import Market, OrderRequest, OrderResponse, PortfolioSummary
from polymarket.exceptions import (
    MarketDiscoveryError,
    ValidationError,
    UpstreamAPIError,
    NetworkError,
)
from polymarket.utils.retry import retry
from polymarket.utils.logging import get_logger

logger = get_logger(__name__)


def floor_to_15min_interval(utc_dt: datetime) -> datetime:
    """
    Round down datetime to the start of its 15-minute interval.

    Examples:
        10:09 AM -> 10:00 AM
        10:15 AM -> 10:15 AM (exact boundary)
        10:23 AM -> 10:15 AM

    Args:
        utc_dt: UTC datetime to floor

    Returns:
        Datetime floored to 15-minute interval boundary
    """
    minute = (utc_dt.minute // 15) * 15
    return utc_dt.replace(minute=minute, second=0, microsecond=0)


def generate_btc_15min_slug(utc_dt: datetime | None = None) -> str:
    """
    Generate the BTC 15-min market slug for a given time.

    The slug format is: btc-updown-15m-{timestamp}
    Where timestamp is the Unix epoch of the interval START time.

    Args:
        utc_dt: UTC datetime (defaults to current time)

    Returns:
        Market slug like "btc-updown-15m-1770608700"
    """
    if utc_dt is None:
        utc_dt = datetime.now(timezone.utc)

    interval_start = floor_to_15min_interval(utc_dt)
    timestamp = int(interval_start.timestamp())
    return f"btc-updown-15m-{timestamp}"


class PolymarketClient:
    """
    High-level client for Polymarket CLOB API operations.

    This client wraps py_clob_client to provide a simpler interface for
    common Polymarket operations including market data, order placement,
    and portfolio management.

    Attributes:
        _settings: Configuration settings from environment
        _private_key: Wallet private key for L1 authentication
        _funder: Proxy wallet address (for Gmail/Magic accounts)

    Example:
        >>> client = PolymarketClient()
        >>> # Fetch BTC market
        >>> market = client.get_btc_15min_market()
        >>> # Place order
        >>> result = client.place_order(
        ...     token_id=market["token_id"],
        ...     side="BUY",
        ...     price=0.55,
        ...     size=10
        ... )

    Note:
        For Gmail/Magic Link accounts, ensure POLYMARKET_SIGNATURE_TYPE=1
        and POLYMARKET_FUNDER is set to your proxy wallet address.
    """

    def __init__(self):
        """Initialize client with current settings."""
        self._settings = get_settings()
        self._auth = get_auth_manager()
        self._mode = self._auth.mode
        self._gamma_url = self._settings.gamma_url

        # Lazy initialization of CLOB client (only for trading mode)
        self._clob_client = None

        logger.info(f"Initialized PolymarketClient in {self._mode} mode")

    @property
    def mode(self) -> str:
        """Get current mode (read_only or trading)."""
        return self._mode

    @retry(max_attempts=3, initial_delay=1.0)
    def _fetch_gamma_markets(
        self,
        search: str | None = None,
        slug: str | None = None,
        limit: int = 100,
        active: bool | None = None,
        accepting_orders: bool | None = None,
    ) -> list[dict]:
        """
        Fetch markets from Gamma API.

        Args:
            search: Search query string (partial match)
            slug: Exact slug match (overrides search)
            limit: Max results to return
            active: Filter by active status
            accepting_orders: Filter by acceptingOrders status

        Returns:
            List of market dictionaries from API
        """
        url = f"{self._gamma_url}/markets"
        params: dict[str, Any] = {"limit": limit}

        if slug:
            # Slug takes priority for exact matching
            params["slug"] = slug
        elif search:
            params["search"] = search
        if active is not None:
            params["closed"] = not active  # API uses 'closed' not 'active'
        if accepting_orders is not None:
            params["accepting_orders"] = "true" if accepting_orders else "false"

        logger.debug(f"Fetching markets from Gamma API: {url} params={params}")

        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            raise NetworkError("Request timeout fetching markets")
        except requests.exceptions.ConnectionError as e:
            raise NetworkError(f"Connection error: {e}")
        except requests.exceptions.HTTPError as e:
            if e.response.status_code >= 500:
                raise UpstreamAPIError(f"Server error: {e}")
            raise

    def discover_btc_15min_market(self) -> Market:
        """
        Discover the currently active BTC 15-minute market.

        Strategy:
        1. Search Gamma API for "BTC Up or Down 15 Minutes"
        2. Filter for active and accepting orders
        3. Fallback: Generate slug from current time

        Returns:
            Market object for the active BTC 15-min market

        Raises:
            MarketDiscoveryError: If no active market found
        """
        logger.info("Discovering BTC 15-min market...")

        # Primary: Slug-based discovery (most reliable for 15-min markets)
        # Try current and adjacent intervals
        current_slug = generate_btc_15min_slug()
        offsets = [0, -15, 15, -30, 30]
        now = datetime.now(timezone.utc)

        for offset_minutes in offsets:
            test_time = now + timedelta(minutes=offset_minutes)
            test_slug = generate_btc_15min_slug(test_time)

            logger.debug(f"Trying slug: {test_slug}")
            markets = self._fetch_gamma_markets(slug=test_slug, limit=1)
            if markets:
                market = Market(**markets[0])
                if market.is_tradeable():
                    logger.info(f"Found BTC 15-min market via slug: {market.slug}")
                    return market

        # Secondary: Search by query (may not include 15-min markets)
        logger.info("Slug discovery failed, trying search query...")
        markets = self._fetch_gamma_markets(
            search="Bitcoin Up or Down",
            limit=50,
            active=True,
            accepting_orders=True,
        )

        # Parse and filter - prioritize 15-minute BTC markets, then other BTC markets
        btc_15min_market = None
        other_btc_market = None

        for market_data in markets:
            market = Market(**market_data)
            if not market.is_tradeable():
                continue

            question_lower = (market.question or "").lower()
            slug_lower = (market.slug or "").lower()

            # Check if this is a 15-minute BTC market (highest priority)
            is_btc_15min = (
                ("15m" in slug_lower or "15-minute" in question_lower or "15 min" in question_lower)
                and ("btc" in slug_lower or "bitcoin" in question_lower)
            )

            # Check if this is any Bitcoin market
            is_btc_market = (
                "bitcoin" in question_lower
                or "btc" in question_lower
                or "bitcoin" in slug_lower
                or "btc" in slug_lower
            )

            if is_btc_15min:
                logger.info(f"Found BTC 15-min market: {market.slug} (ID: {market.id})")
                return market
            elif is_btc_market and other_btc_market is None:
                other_btc_market = market

        # If no 15-minute market found but we have another BTC market, warn and use it
        if other_btc_market:
            logger.warning(
                f"No BTC 15-min market found. "
                f"Using other BTC market: {other_btc_market.slug} (ID: {other_btc_market.id})"
            )
            return other_btc_market

        raise MarketDiscoveryError(
            "Could not discover active BTC 15-min market. "
            "Try manual --market-id from Polymarket dashboard."
        )

    def get_markets(
        self,
        search: str | None = None,
        limit: int = 100,
        active_only: bool = True,
    ) -> list[Market]:
        """
        Fetch markets with optional filtering.

        Args:
            search: Search query
            limit: Max results
            active_only: Only return active markets

        Returns:
            List of Market objects
        """
        markets_data = self._fetch_gamma_markets(
            search=search,
            limit=limit,
            active=True if active_only else None,
            accepting_orders=True if active_only else None,
        )

        return [Market(**m) for m in markets_data]

    def get_market_by_id(self, market_id: str) -> Market | None:
        """
        Fetch a specific market by ID.

        Args:
            market_id: The market ID

        Returns:
            Market object or None if not found
        """
        # Try slug first (Gamma API supports this)
        try:
            markets_data = self._fetch_gamma_markets(search=market_id, limit=1)
            if markets_data:
                return Market(**markets_data[0])
        except Exception as e:
            logger.debug(f"Failed to fetch market {market_id}: {e}")

        return None

    def _get_clob_client(self):
        """Lazy initialize CLOB client (only in trading mode)."""
        if self._mode != "trading":
            raise ValidationError("CLOB operations require TRADING mode")

        if self._clob_client is None:
            # Import here to avoid dependency in read_only mode
            try:
                from py_clob_client.client import ClobClient
            except ImportError:
                raise ValidationError(
                    "py-clob-client not installed. "
                    "Install with: pip install py-clob-client"
                )

            kwargs = self._auth.get_clob_client_kwargs()
            self._clob_client = ClobClient(**kwargs)
            logger.debug("Initialized CLOB client")

        return self._clob_client

    def create_order(
        self,
        request: OrderRequest,
        dry_run: bool = True,
    ) -> OrderResponse:
        """
        Create and optionally submit an order.

        Args:
            request: Order request details
            dry_run: If True, validate but don't submit

        Returns:
            Order response with status

        Raises:
            ValidationError: Invalid order parameters
            AuthError: Authentication failed
        """
        logger.info(f"Creating order: {request.side} {request.size} @ {request.price}")

        if self._mode != "trading":
            raise ValidationError("Trading requires TRADING mode")

        # Validate market exists (preflight)
        # TODO: Add market validation

        # For market orders, we use true FOK (Fill-or-Kill) orders
        # Price is used for dry-run display only
        price = request.price

        if dry_run:
            logger.info("[DRY RUN] Would submit order:")
            logger.info(f"  Token ID: {request.token_id}")
            logger.info(f"  Side: {request.side}")
            logger.info(f"  Price: {price}")
            logger.info(f"  Size: {request.size}")
            logger.info(f"  Type: {request.order_type}")

            return OrderResponse(
                order_id="dry-run-" + str(hash(str(request))),
                status="dry_run",
                accepted=True,
                raw_response={"dry_run": True},
            )

        # Submit live order
        client = self._get_clob_client()

        try:
            # For market orders, use create_market_order() with MarketOrderArgs
            # For limit orders, use create_and_post_order() with OrderArgs
            from py_clob_client.clob_types import OrderArgs, MarketOrderArgs, OrderType

            if request.order_type == "market":
                # True FOK (Fill-or-Kill) market order for guaranteed immediate execution
                size = round(request.size, 2)  # Use 2 decimals for size

                logger.info(f"Placing FOK market order: side={request.side}, size={size}")

                # Create market order arguments
                market_order_args = MarketOrderArgs(
                    token_id=request.token_id,
                    amount=float(size),  # Amount in USDC
                    side=request.side,
                )

                # Sign the market order
                signed_order = client.create_market_order(market_order_args)

                # Post with FOK (Fill-or-Kill) - fills immediately or cancels
                result = client.post_order(signed_order, OrderType.FOK)
                order_id = result.get("orderID", "") if isinstance(result, dict) else ""

                logger.info(f"FOK market order placed successfully: {order_id}")

                return OrderResponse(
                    order_id=order_id,
                    status="posted",
                    accepted=True,
                    raw_response=result if isinstance(result, dict) else {},
                )

            elif False:  # Skip the old market order code
                # Market order - API validates decimal precision strictly
                # For BUY: makerAmount = tokens (≤4 decimals), takerAmount = USDC (≤2 decimals)
                # API calculates: takerAmount = makerAmount / price

                price = round(price, 4)  # Max 4 decimals for price

                if request.side == "BUY":
                    # Start with token amount (this becomes makerAmount)
                    token_amount = round(request.size, 4)  # Tokens: max 4 decimals
                    # Calculate USDC amount ensuring it has ≤2 decimals
                    usdc_amount = round(price * token_amount, 2)  # USDC: max 2 decimals
                    # Recalculate token amount to ensure consistency
                    token_amount = round(usdc_amount / price, 4)  # Re-derive to match
                    amount = usdc_amount  # Pass USDC amount to API
                else:
                    # For SELL: reverse (taker gets USDC, maker gives tokens)
                    token_amount = round(request.size, 4)
                    amount = token_amount

                print(f"DEBUG: side={request.side}, amount={amount}, price={price}, calculated_token_amount={token_amount}")
                logger.info(f"Order details: side={request.side}, amount={amount}, price={price}")

                market_order_args = MarketOrderArgs(
                    token_id=request.token_id,
                    amount=amount,
                    side=request.side,
                    price=price,  # Optional price limit
                    order_type=OrderType.GTC,  # Good-Til-Cancel allows order to sit on book
                )

                # Create and post market order
                signed_order = client.create_market_order(market_order_args)
                print(f"DEBUG: signed_order = {signed_order.dict() if hasattr(signed_order, 'dict') else signed_order}")
                result = client.post_order(signed_order, orderType=OrderType.GTC)
            else:
                # Limit order (GTC) - use size
                order_args = OrderArgs(
                    token_id=request.token_id,
                    side=request.side,
                    price=price,
                    size=request.size,
                )

                result = client.create_and_post_order(order_args)

            return OrderResponse(
                order_id=result.get("orderID", "") if isinstance(result, dict) else "",
                status="posted",
                accepted=True,
                raw_response=result if isinstance(result, dict) else signed_order.dict(),
            )

        except Exception as e:
            logger.error(f"Order failed: {e}")
            return OrderResponse(
                order_id="",
                status="failed",
                accepted=False,
                raw_response={},
                error_message=str(e),
            )

    def get_portfolio_summary(self) -> PortfolioSummary:
        """
        Get portfolio summary including open orders and positions.

        Returns:
            Portfolio summary with open orders and positions
        """
        if self._mode != "trading":
            return PortfolioSummary(
                open_orders=[],
                total_notional=0.0,
                positions={},
                total_exposure=0.0,
            )

        try:
            client = self._get_clob_client()

            # Get open orders
            open_orders = client.get_orders()

            # Get trade history to calculate current positions
            trades = client.get_trades()

            # Get USDC balance (CLOB collateral)
            usdc_balance = 0.0
            try:
                from py_clob_client.clob_types import AssetType, BalanceAllowanceParams
                params = BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
                client.update_balance_allowance(params=params)
                balance_result = client.get_balance_allowance(params=params)
                raw_balance = int(balance_result.get("balance", 0))
                usdc_balance = raw_balance / 10**6  # USDC has 6 decimals
            except Exception as e:
                logger.debug(f"Could not fetch USDC balance: {e}")

            # Calculate summary
            total_notional = 0.0
            positions: dict[str, float] = {}

            for order in open_orders:
                total_notional += float(order.get("size", 0)) * float(order.get("price", 0))

            # Calculate positions from trades (buy = +position, sell = -position)
            for trade in trades:
                asset_id = trade.get("asset_id", "")
                side = trade.get("side", "").upper()
                size = float(trade.get("size", 0))

                if side == "BUY":
                    positions[asset_id] = positions.get(asset_id, 0) + size
                elif side == "SELL":
                    positions[asset_id] = positions.get(asset_id, 0) - size

            # Remove zero positions
            positions = {k: v for k, v in positions.items() if abs(v) > 0.0001}

            # Calculate position values at current market prices
            positions_value = 0.0
            for token_id, quantity in positions.items():
                try:
                    price_data = client.get_last_trade_price(token_id)
                    price = float(price_data.get("price", 0))
                    positions_value += abs(quantity) * price
                except Exception as e:
                    logger.debug(f"Could not get price for {token_id}: {e}")

            total_value = usdc_balance + positions_value

            return PortfolioSummary(
                open_orders=open_orders,
                total_notional=total_notional,
                positions=positions,
                total_exposure=total_notional,
                trades=trades,  # Include raw trades for detailed display
                usdc_balance=usdc_balance,
                positions_value=positions_value,
                total_value=total_value,
            )

        except Exception as e:
            logger.error(f"Failed to fetch portfolio: {e}")
            raise UpstreamAPIError(f"Portfolio fetch failed: {e}")
