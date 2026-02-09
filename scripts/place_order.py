#!/usr/bin/env python3
"""place_order.py - Place orders on Polymarket.

Usage:
    python scripts/place_order.py --btc-mode --side buy --price 0.55 --size 10 --dry-run true
    python scripts/place_order.py --market-id 0x... --token-id 0x... --side sell --price 0.60 --size 5
"""

import sys
from pathlib import Path
from typing import Literal

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import typer
from rich.console import Console

from polymarket.config import get_settings
from polymarket.auth import reset_auth_manager
from polymarket.client import PolymarketClient
from polymarket.models import OrderRequest
from polymarket.exceptions import PolymarketError, ValidationError, ConfigError
from polymarket.utils.logging import setup_logging, get_logger

app = typer.Typer(
    name="place-order",
    help="Place orders on Polymarket",
    add_completion=False,
)

console = Console()
logger = get_logger(__name__)


@app.command()
def main(
    btc_mode: bool = typer.Option(
        False,
        "--btc-mode",
        help="Auto-discover BTC 15-min market and use YES token",
    ),
    market_id: str | None = typer.Option(
        None,
        "--market-id",
        "-m",
        help="Specific market ID (overrides --btc-mode)",
    ),
    token_id: str | None = typer.Option(
        None,
        "--token-id",
        "-t",
        help="Specific token ID (overrides --btc-mode)",
    ),
    side: Literal["buy", "sell"] = typer.Option(
        ...,
        "--side",
        "-s",
        help="Order side: buy or sell",
    ),
    price: float = typer.Option(
        ...,
        "--price",
        "-p",
        help="Limit price (0.0 to 1.0 for binary markets)",
    ),
    size: float = typer.Option(
        ...,
        "--size",
        "-z",
        help="Order size in shares",
    ),
    order_type: Literal["limit", "market"] = typer.Option(
        "limit",
        "--order-type",
        "-o",
        help="Order type: limit or market",
    ),
    dry_run: bool = typer.Option(
        True,
        "--dry-run/--live",
        help="Dry run mode (default: true)",
    ),
) -> None:
    """Place an order on Polymarket.

    Examples:
        # Dry run on BTC market (default)
        python scripts/place_order.py --btc-mode --side buy --price 0.55 --size 10

        # Live order
        python scripts/place_order.py --btc-mode --side buy --price 0.55 --size 10 --live

        # Manual market/token IDs
        python scripts/place_order.py \
            --market-id 0x123... \
            --token-id 0x456... \
            --side sell \
            --price 0.60 \
            --size 5
    """
    # Setup logging
    settings = get_settings()
    setup_logging(settings.log_level, settings.log_json)

    logger.info(f"Starting place_order in {settings.mode} mode")

    try:
        # Validate trading mode
        if settings.mode != "trading":
            raise ConfigError(
                "TRADING mode required for placing orders. "
                "Set POLYMARKET_MODE=trading in .env or environment."
            )

        # Initialize client
        client = PolymarketClient()

        # Resolve market and token IDs
        if btc_mode and not market_id:
            console.print("[cyan]Discovering BTC 15-min market...[/cyan]")
            market = client.discover_btc_15min_market()
            market_id = market.id

            # Get token IDs
            token_ids = market.get_token_ids()
            if not token_ids:
                raise ValidationError("Market has no token IDs available")

            # Default to YES token (index 0) for buy, NO (index 1) for sell
            if token_id is None:
                if side.lower() == "buy":
                    token_id = token_ids[0]
                    console.print(f"[green]Using YES token: {token_id}[/green]")
                else:
                    # For sell, default to YES token (selling what you own)
                    token_id = token_ids[0]
                    console.print(f"[green]Using token: {token_id}[/green]")

        if not market_id:
            raise ValidationError("Either --btc-mode or --market-id is required")

        if not token_id:
            raise ValidationError("Token ID is required (use --btc-mode or --token-id)")

        # Create order request
        request = OrderRequest(
            token_id=token_id,
            side=side.upper(),
            price=price,
            size=size,
            order_type=order_type.lower(),
        )

        # Preflight checks
        console.print("\n[bold]Order Details:[/bold]")
        console.print(f"  Market ID: {market_id}")
        console.print(f"  Token ID: {token_id}")
        console.print(f"  Side: {request.side}")
        console.print(f"  Price: {price:.4f}")
        console.print(f"  Size: {size}")
        console.print(f"  Type: {request.order_type}")
        console.print(f"  Dry Run: {dry_run}")
        console.print()

        if order_type == "market":
            console.print("[yellow]WARNING: MARKET orders are emulated via aggressive LIMIT orders.[/yellow]")

        if dry_run:
            console.print("[cyan]DRY RUN MODE - No order will be submitted[/cyan]\n")

        # Confirm for live orders
        if not dry_run:
            confirm = typer.confirm("Submit this order?")
            if not confirm:
                console.print("[yellow]Order cancelled.[/yellow]")
                raise typer.Exit(code=0)

        # Submit order
        response = client.create_order(request, dry_run=dry_run)

        # Display result
        if response.accepted:
            console.print(f"[green]Order {response.status}[/green]")
            console.print(f"  Order ID: {response.order_id}")
        else:
            console.print(f"[red]Order rejected[/red]")
            if response.error_message:
                console.print(f"  Error: {response.error_message}")

        logger.info(f"Order completed with status: {response.status}")

    except ValidationError as e:
        logger.error(f"Validation error: {e}")
        console.print(f"[red]Validation Error: {e}[/red]")
        raise typer.Exit(code=1)
    except ConfigError as e:
        logger.error(f"Configuration error: {e}")
        console.print(f"[red]Configuration Error: {e}[/red]")
        raise typer.Exit(code=1)
    except PolymarketError as e:
        logger.error(f"Polymarket error: {e}")
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(code=1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        console.print(f"[red]Unexpected error: {e}[/red]")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
