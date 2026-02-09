#!/usr/bin/env python3
"""
Fetch Polymarket market data.

This script queries the Polymarket CLOB API to retrieve market information
including active BTC 15-minute markets. It supports multiple query modes:
BTC market discovery, search by query string, and direct market lookup.

Usage:
    # Fetch current BTC 15-minute market
    python scripts/fetch_markets.py --btc-mode

    # Search markets by query
    python scripts/fetch_markets.py --search "bitcoin" --limit 50

    # Get market by ID
    python scripts/fetch_markets.py --market-id 0x123...

    # JSON output for agent processing
    python scripts/fetch_markets.py --btc-mode --json

Arguments:
    --btc-mode: Fetch current BTC 15-min market (auto-discovers timestamp)
    --market-id: Specific market ID to query
    --search: Search markets by query string
    --limit: Maximum results to return (default: 20)
    --json: Output as JSON instead of formatted table
    --min-volume: Filter by minimum volume in USDC

Returns:
    Market data including:
        - token_id: Token identifier for trading
        - market_id: Market identifier
        - title: Market title/description
        - price: Current price (0-1 for binary markets)
        - volume: Trading volume in USDC
        - expiry_time: ISO timestamp when market expires
        - accepting_orders: Whether market accepts new orders

Examples:
    # Get current BTC market with table output
    $ python scripts/fetch_markets.py --btc-mode
    +-----------------------------+--------+-----------+---------+
    | Market                      | Price  | Volume    | Expires |
    +-----------------------------+--------+-----------+---------+
    | BTC Up or Down 15 Minutes   | 0.52   | $12,450   | 14:15   |
    +-----------------------------+--------+-----------+---------+

    # Get BTC market as JSON
    $ python scripts/fetch_markets.py --btc-mode --json
    {"token_id": "0x...", "price": 0.52, "volume": 12450.0, ...}

Exit codes:
    0: Success
    1: API error or network failure
    2: Invalid arguments
    3: Market not found

Notes:
    - BTC markets use slug pattern: btc-updown-15m-{epoch_timestamp}
    - Timestamp represents interval start time (floored to 15-min)
    - Requires POLYMARKET_MODE env var (read_only or trading)
    - For read_only mode, no credentials are required
"""

import sys
import json
from pathlib import Path
from typing import Literal

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import typer
from rich.console import Console
from rich.table import Table

from polymarket.config import get_settings, reset_settings
from polymarket.auth import reset_auth_manager
from polymarket.client import PolymarketClient
from polymarket.exceptions import PolymarketError
from polymarket.utils.logging import setup_logging, get_logger

app = typer.Typer(
    name="fetch-markets",
    help="Fetch market data from Polymarket",
    add_completion=False,
)

console = Console()
logger = get_logger(__name__)


@app.command()
def main(
    btc_mode: bool = typer.Option(
        False,
        "--btc-mode",
        help="Discover and show the current BTC 15-min market",
    ),
    search: str | None = typer.Option(
        None,
        "--search",
        "-s",
        help="Search query for markets",
    ),
    limit: int = typer.Option(
        25,
        "--limit",
        "-l",
        help="Maximum number of markets to return",
    ),
    active_only: bool = typer.Option(
        True,
        "--active-only/--all",
        help="Only show active, tradeable markets",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output as JSON instead of formatted table",
    ),
) -> None:
    """Fetch markets from Polymarket.

    Examples:
        # Fetch active BTC 15-min market
        python scripts/fetch_markets.py --btc-mode

        # Search for markets
        python scripts/fetch_markets.py --search "bitcoin" --limit 50

        # JSON output
        python scripts/fetch_markets.py --btc-mode --json
    """
    # Setup logging
    settings = get_settings()
    setup_logging(settings.log_level, settings.log_json)

    logger.info(f"Starting fetch_markets in {settings.mode} mode")

    try:
        client = PolymarketClient()

        if btc_mode:
            # Discover BTC 15-min market
            market = client.discover_btc_15min_market()

            if json_output:
                output = {
                    "market": {
                        "id": market.id,
                        "question": market.question,
                        "slug": market.slug,
                        "condition_id": market.condition_id,
                        "active": market.active,
                        "accepting_orders": market.accepting_orders,
                        "end_date": market.end_date.isoformat() if market.end_date else None,
                        "best_bid": market.best_bid,
                        "best_ask": market.best_ask,
                        "volume": market.volume_num,
                        "token_ids": market.get_token_ids(),
                    }
                }
                console.print_json(json.dumps(output))
            else:
                _print_btc_market(market)

        else:
            # Fetch markets with optional search
            markets = client.get_markets(
                search=search,
                limit=limit,
                active_only=active_only,
            )

            if json_output:
                output = [
                    {
                        "id": m.id,
                        "question": m.question,
                        "slug": m.slug,
                        "active": m.active,
                        "accepting_orders": m.accepting_orders,
                        "best_bid": m.best_bid,
                        "best_ask": m.best_ask,
                        "volume": m.volume_num,
                    }
                    for m in markets
                ]
                console.print_json(json.dumps(output))
            else:
                _print_markets_table(markets)

        logger.info(f"Successfully fetched {1 if btc_mode else len(markets)} market(s)")

    except PolymarketError as e:
        logger.error(f"Polymarket error: {e}")
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(code=1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        console.print(f"[red]Unexpected error: {e}[/red]")
        raise typer.Exit(code=1)


def _print_btc_market(market) -> None:
    """Print BTC market details in a nice format."""
    console.print("\n[bold cyan]BTC Up/Down 15-Minute Market[/bold cyan]\n")

    info_table = Table(show_header=False, box=None)
    info_table.add_column("Field", style="dim")
    info_table.add_column("Value")

    info_table.add_row("Market ID", market.id)
    info_table.add_row("Question", market.question or "N/A")
    info_table.add_row("Slug", market.slug or "N/A")
    info_table.add_row("Active", "[green]Yes[/green]" if market.active else "[red]No[/red]")
    info_table.add_row("Accepting Orders", "[green]Yes[/green]" if market.accepting_orders else "[red]No[/red]")

    if market.end_date:
        info_table.add_row("End Date", market.end_date.strftime("%Y-%m-%d %H:%M:%S UTC"))

    info_table.add_row("Best Bid", f"{market.best_bid:.4f}" if market.best_bid else "N/A")
    info_table.add_row("Best Ask", f"{market.best_ask:.4f}" if market.best_ask else "N/A")
    info_table.add_row("Volume", f"{market.volume_num:,.0f}" if market.volume_num else "N/A")

    console.print(info_table)

    # Token IDs
    token_ids = market.get_token_ids()
    if token_ids:
        console.print("\n[bold]Token IDs:[/bold]")
        for i, token_id in enumerate(token_ids):
            outcome = "Yes" if i == 0 else "No"
            console.print(f"  {outcome}: {token_id}")


def _print_markets_table(markets: list) -> None:
    """Print markets as a formatted table."""
    table = Table(title=f"Markets ({len(markets)})")

    table.add_column("ID", style="dim", max_width=12)
    table.add_column("Question", max_width=50)
    table.add_column("Active", justify="center")
    table.add_column("Bid", justify="right")
    table.add_column("Ask", justify="right")
    table.add_column("Volume", justify="right")

    for market in markets:
        table.add_row(
            market.id[:8] + "...",
            market.question or "N/A",
            "[green]Y[/green]" if market.active else "[red]N[/red]",
            f"{market.best_bid:.3f}" if market.best_bid else "-",
            f"{market.best_ask:.3f}" if market.best_ask else "-",
            f"{market.volume_num:,.0f}" if market.volume_num else "-",
        )

    console.print(table)


if __name__ == "__main__":
    app()
