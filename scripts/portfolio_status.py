#!/usr/bin/env python3
"""portfolio_status.py - Check open orders and positions.

Usage:
    python scripts/portfolio_status.py
    python scripts/portfolio_status.py --market-id 0x...
    python scripts/portfolio_status.py --json
"""

import sys
import json
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import typer
from rich.console import Console
from rich.table import Table

from polymarket.config import get_settings
from polymarket.auth import reset_auth_manager
from polymarket.client import PolymarketClient
from polymarket.exceptions import PolymarketError
from polymarket.utils.logging import setup_logging, get_logger

app = typer.Typer(
    name="portfolio-status",
    help="Check open orders and positions on Polymarket",
    add_completion=False,
)

console = Console()
logger = get_logger(__name__)


@app.command()
def main(
    market_id: str | None = typer.Option(
        None,
        "--market-id",
        "-m",
        help="Filter by specific market ID",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output as JSON instead of formatted table",
    ),
) -> None:
    """Check portfolio status including open orders and positions.

    Examples:
        # Check all open orders
        python scripts/portfolio_status.py

        # Filter by market
        python scripts/portfolio_status.py --market-id 0x...

        # JSON output
        python scripts/portfolio_status.py --json
    """
    # Setup logging
    settings = get_settings()
    setup_logging(settings.log_level, settings.log_json)

    logger.info(f"Starting portfolio_status in {settings.mode} mode")

    try:
        client = PolymarketClient()

        if settings.mode == "read_only":
            console.print("[yellow]READ_ONLY mode - no portfolio data available[/yellow]")
            console.print("Set POLYMARKET_MODE=trading to view portfolio\n")

            if json_output:
                console.print_json(json.dumps({"open_orders": [], "positions": {}}))

            return  # Use return instead of typer.Exit to avoid exception handling issues

        # Fetch portfolio
        portfolio = client.get_portfolio_summary()

        if json_output:
            output = {
                "open_orders": portfolio.open_orders,
                "total_notional": portfolio.total_notional,
                "positions": portfolio.positions,
                "total_exposure": portfolio.total_exposure,
                "trades": portfolio.trades,
            }
            console.print_json(json.dumps(output))
        else:
            _print_portfolio_summary(portfolio, market_id)

        logger.info(f"Portfolio summary: {len(portfolio.open_orders)} open orders, {len(portfolio.trades)} trades")

    except PolymarketError as e:
        logger.error(f"Polymarket error: {e}")
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(code=1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        console.print(f"[red]Unexpected error: {e}[/red]")
        raise typer.Exit(code=1)


def _print_portfolio_summary(portfolio, market_filter: str | None = None) -> None:
    """Print portfolio summary in a nice format."""
    console.print("\n[bold cyan]Portfolio Summary[/bold cyan]\n")

    # Summary stats
    console.print(f"Open Orders: [cyan]{len(portfolio.open_orders)}[/cyan]")
    console.print(f"Total Notional: [cyan]${portfolio.total_notional:,.2f}[/cyan]")
    console.print(f"Total Exposure: [cyan]${portfolio.total_exposure:,.2f}[/cyan]")

    # Positions
    if portfolio.positions:
        console.print(f"\n[bold]Current Positions:[/bold] ({len(portfolio.positions)} tokens)")
        for token_id, quantity in portfolio.positions.items():
            side_str = "[green]LONG[/green]" if quantity > 0 else "[red]SHORT[/red]"
            console.print(f"  {side_str} {abs(quantity):.4f} shares (Token: {token_id[:20]}...)")
    else:
        console.print("\n[dim]No current positions[/dim]")

    # Open orders
    if not portfolio.open_orders:
        console.print("\n[dim]No open orders[/dim]")
    else:
        # Open orders table
        table = Table(title="\nOpen Orders")
        table.add_column("Order ID", style="dim", max_width=12)
        table.add_column("Token ID", max_width=12)
        table.add_column("Side")
        table.add_column("Price", justify="right")
        table.add_column("Size", justify="right")
        table.add_column("Status")

        for order in portfolio.open_orders:
            # Filter by market if specified
            if market_filter and order.get("marketId") != market_filter:
                continue

            table.add_row(
                str(order.get("orderId", ""))[:10] + "...",
                str(order.get("tokenId", ""))[:10] + "...",
                order.get("side", ""),
                f"{float(order.get('price', 0)):.4f}",
                f"{float(order.get('size', 0)):.2f}",
                order.get("status", ""),
            )

        console.print(table)

    # Trade history
    if portfolio.trades:
        console.print(f"\n[bold]Trade History:[/bold] ({len(portfolio.trades)} trades)")
        trades_table = Table()
        trades_table.add_column("Market", style="dim", max_width=20)
        trades_table.add_column("Side")
        trades_table.add_column("Size", justify="right")
        trades_table.add_column("Price", justify="right")
        trades_table.add_column("Status")

        for trade in portfolio.trades:
            trades_table.add_row(
                str(trade.get("market", ""))[:20] + "...",
                trade.get("side", ""),
                f"{float(trade.get('size', 0)):.4f}",
                f"${float(trade.get('price', 0)):.4f}",
                trade.get("status", ""),
            )

        console.print(trades_table)


if __name__ == "__main__":
    app()
