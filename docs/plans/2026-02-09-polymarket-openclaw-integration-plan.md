# Polymarket OpenClaw Integration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create comprehensive documentation and OpenClaw agent integration for Polymarket BTC 15-min trading scripts, then push to GitHub.

**Architecture:** Agent-optimized repository structure with layered documentation (user → API → internals), openclaw/ folder for agent-specific resources, and comprehensive docstrings across all Python files.

**Tech Stack:** Python 3.10+, py-clob-client, pytest, GitHub, YAML for skill definitions

---

## Task 1: Create openclaw/ directory structure

**Files:**
- Create: `openclaw/README.md`
- Create: `openclaw/prompts/fetch_markets.txt`
- Create: `openclaw/prompts/place_order.txt`
- Create: `openclaw/prompts/portfolio_check.txt`
- Create: `openclaw/skills/polymarket_fetch.yaml`
- Create: `openclaw/skills/polymarket_trade.yaml`
- Create: `openclaw/skills/polymarket_status.yaml`
- Create: `openclaw/examples/daily_trading_flow.md`
- Create: `openclaw/examples/monitoring_workflow.md`

**Step 1: Create openclaw/README.md**

Write the agent quick-start guide:

```markdown
# Polymarket Scripts for OpenClaw Agents

Quick-start guide for OpenClaw agents to use Polymarket trading scripts.

## Quick Start

1. Copy `.env.example` to `.env` and add credentials
2. Install dependencies: `pip install -r requirements.txt`
3. Run a command: `python scripts/fetch_markets.py --btc-mode`

## Available Skills

### polymarket_fetch
Fetch Polymarket market data, including BTC 15-minute markets.

### polymarket_trade
Place buy/sell orders on Polymarket markets.

### polymarket_status
Check portfolio status, open orders, and positions.

## Agent Prompts

See `openclaw/prompts/` for reusable prompt templates:
- `fetch_markets.txt` - Fetch current market data
- `place_order.txt` - Place trading orders
- `portfolio_check.txt` - Check portfolio status

## Example Workflows

See `openclaw/examples/` for complete agent workflows:
- `daily_trading_flow.md` - Daily trading workflow
- `monitoring_workflow.md` - Monitoring workflow

## Environment Setup

Required environment variables in `.env`:

```bash
POLYMARKET_MODE=trading
POLYMARKET_PRIVATE_KEY=0x...
POLYMARKET_API_KEY=...
POLYMARKET_API_SECRET=...
POLYMARKET_API_PASSPHRASE=...
POLYMARKET_FUNDER=0x...
POLYMARKET_SIGNATURE_TYPE=1
```

## Common Commands

```bash
# Fetch BTC 15-min market
python scripts/fetch_markets.py --btc-mode

# Check portfolio
python scripts/portfolio_status.py

# Place order (dry run)
python scripts/place_order.py --btc-mode --side buy --price 0.55 --size 10 --dry-run true
```

## Troubleshooting

- Authentication failed: Verify API credentials are current
- Market not found: Check market is active and accepting orders
- Order rejected: Verify price range (0-1) and minimum order size
```

**Step 2: Create openclaw/prompts/fetch_markets.txt**

```text
Fetch the current BTC 15-minute market from Polymarket.

Command to run:
python scripts/fetch_markets.py --btc-mode --json

Extract from output:
- token_id: The market token identifier
- price: Current price (0-1 for binary markets)
- volume: Trading volume
- expiry_time: When the market expires
- market_id: The market identifier

Return the data in a structured format for analysis.
```

**Step 3: Create openclaw/prompts/place_order.txt**

```text
Place an order on Polymarket.

First, determine the parameters:
- side: "BUY" or "SELL"
- price: Order price from 0 to 1
- size: Number of shares
- dry_run: Set to "true" for testing, "false" for live trading

Command template:
python scripts/place_order.py --btc-mode --side {side} --price {price} --size {size} --dry-run {dry_run}

Examples:
# Test buy order
python scripts/place_order.py --btc-mode --side buy --price 0.55 --size 10 --dry-run true

# Live sell order
python scripts/place_order.py --btc-mode --side sell --price 0.60 --size 5 --dry-run false

ALWAYS use --dry-run true first to verify the order works.
```

**Step 4: Create openclaw/prompts/portfolio_check.txt**

```text
Check the Polymarket portfolio status.

Command to run:
python scripts/portfolio_status.py --json

Extract from output:
- open_orders: List of pending orders
- total_value: Total portfolio value (cash + positions)
- usdc_balance: Available USDC for trading
- positions_value: Value of open positions
- trades: Recent trade history

Return a summary of current positions and any actions needed.
```

**Step 5: Create openclaw/skills/polymarket_fetch.yaml**

```yaml
name: polymarket_fetch
description: Fetch Polymarket BTC 15-minute market data including current price, volume, and expiry time
category: trading
version: 1.0.0

command:
  template: "python scripts/fetch_markets.py --btc-mode --json"
  timeout: 30

outputs:
  - name: token_id
    type: string
    description: The market token identifier for trading
  - name: price
    type: float
    description: Current market price (0-1 for binary)
  - name: volume
    type: float
    description: Trading volume in USDC
  - name: expiry_time
    type: string
    description: ISO timestamp when market expires
  - name: market_id
    type: string
    description: The market identifier

requirements:
  - python3
  - py-clob-client
  - Valid Polymarket credentials in .env

examples:
  - command: python scripts/fetch_markets.py --btc-mode
    description: Get current BTC market data

errors:
  - code: MARKET_NOT_FOUND
    message: No active BTC 15-minute market found
    resolution: Verify market is active or use manual market-id
  - code: AUTH_ERROR
    message: Authentication failed
    resolution: Check .env credentials
```

**Step 6: Create openclaw/skills/polymarket_trade.yaml**

```yaml
name: polymarket_trade
description: Place buy or sell orders on Polymarket markets with market order execution (FOK)
category: trading
version: 1.0.0

command:
  template: "python scripts/place_order.py --btc-mode --side {side} --price {price} --size {size} --dry-run {dry_run}"
  timeout: 30

inputs:
  - name: side
    type: enum
    values: [BUY, SELL]
    required: true
    description: Order side (buy or sell)
  - name: price
    type: float
    required: true
    min: 0.0
    max: 1.0
    description: Order price (0-1 for binary markets)
  - name: size
    type: float
    required: true
    min: 0.01
    description: Number of shares to trade
  - name: dry_run
    type: boolean
    default: true
    description: Test mode (true) or live trading (false)

outputs:
  - name: order_id
    type: string
    description: The placed order ID
  - name: status
    type: string
    description: Order status (pending, filled, rejected)

requirements:
  - python3
  - py-clob-client
  - Valid Polymarket credentials in .env

warnings:
  - ALWAYS use dry_run=true first to verify
  - Market orders use FOK (Fill-Or-Kill) for immediate execution
  - Minimum order size applies

examples:
  - command: python scripts/place_order.py --btc-mode --side buy --price 0.55 --size 10 --dry-run true
    description: Test buy order

errors:
  - code: INSUFFICIENT_BALANCE
    message: Not enough USDC to place order
    resolution: Check portfolio status and deposit funds
  - code: MARKET_CLOSED
    message: Market is not accepting orders
    resolution: Verify market is active
  - code: PRICE_OUT_OF_RANGE
    message: Price must be between 0 and 1
    resolution: Check price parameter
```

**Step 7: Create openclaw/skills/polymarket_status.yaml**

```yaml
name: polymarket_status
description: Check Polymarket portfolio status including open orders, positions, and total value
category: trading
version: 1.0.0

command:
  template: "python scripts/portfolio_status.py --json"
  timeout: 30

outputs:
  - name: open_orders
    type: array
    description: List of pending orders
  - name: total_value
    type: float
    description: Total portfolio value in USDC
  - name: usdc_balance
    type: float
    description: Available USDC for trading
  - name: positions_value
    type: float
    description: Value of open positions
  - name: trades
    type: array
    description: Recent trade history

requirements:
  - python3
  - py-clob-client
  - Valid Polymarket credentials in .env

examples:
  - command: python scripts/portfolio_status.py
    description: Get full portfolio status
  - command: python scripts/portfolio_status.py --json
    description: Get portfolio as JSON

errors:
  - code: AUTH_ERROR
    message: Authentication failed
    resolution: Check .env credentials
```

**Step 8: Create openclaw/examples/daily_trading_flow.md**

```markdown
# Daily Trading Flow

This workflow demonstrates how an agent can perform daily trading on the BTC 15-minute market.

## Workflow Steps

### 1. Fetch Current Market Data

Get the active BTC 15-minute market:

```bash
python scripts/fetch_markets.py --btc-mode --json
```

Extract:
- `token_id`: For placing orders
- `price`: Current market price
- `volume`: Liquidity check
- `expiry_time`: Market closing time

### 2. Check Portfolio Status

Review current positions and balance:

```bash
python scripts/portfolio_status.py --json
```

Extract:
- `usdc_balance`: Available trading capital
- `open_orders`: Any pending orders
- `total_value`: Overall portfolio value

### 3. Analyze and Decide

Based on market data and portfolio:
- Assess market direction
- Determine position size
- Decide buy/sell/no action

### 4. Place Test Order (Dry Run)

Always test first:

```bash
python scripts/place_order.py \\
  --btc-mode \\
  --side buy \\
  --price 0.55 \\
  --size 10 \\
  --dry-run true
```

Verify:
- Order parameters are correct
- No errors returned
- Sufficient balance

### 5. Place Live Order

If dry run succeeds:

```bash
python scripts/place_order.py \\
  --btc-mode \\
  --side buy \\
  --price 0.55 \\
  --size 10 \\
  --dry-run false
```

### 6. Verify Order

Check that order was placed:

```bash
python scripts/portfolio_status.py
```

Confirm order appears in open orders.

## Example Agent Prompt

```
1. Fetch the current BTC 15-minute market from Polymarket
2. Check my portfolio status
3. If I have more than $100 available and the market price is below 0.50, place a buy order for $50 worth of shares at the current price
4. Use dry-run mode first, then execute if the test succeeds
5. Report the final portfolio status
```

## Risk Management

- Never trade more than 10% of portfolio per trade
- Always use dry-run mode before live orders
- Check market expiry before trading
- Verify order was placed successfully
```

**Step 9: Create openclaw/examples/monitoring_workflow.md**

```markdown
# Monitoring Workflow

This workflow demonstrates periodic monitoring of Polymarket positions and orders.

## Workflow Steps

### 1. Check Portfolio Status

```bash
python scripts/portfolio_status.py --json
```

### 2. Analyze Open Orders

Check for:
- Stale orders (older than 1 hour)
- Orders at unfavorable prices
- Orders that should be cancelled

### 3. Check Market Prices

```bash
python scripts/fetch_markets.py --btc-mode --json
```

### 4. Decision Points

**If open orders exist:**
- Are they still relevant?
- Should any be cancelled?

**If positions exist:**
- Should profit be taken?
- Should stop-loss be triggered?

**If cash available:**
- Are there good entry opportunities?

## Scheduling

Run this workflow every 5 minutes for active trading:

```bash
# Every 5 minutes via cron
*/5 * * * * cd /path/to/polymarket-scripts && python scripts/portfolio_status.py >> monitoring.log
```

## Alerting

Set up alerts for:
- Total portfolio value changes > 10%
- New orders filled
- Orders rejected
- Authentication failures
```

**Step 10: Commit openclaw/ directory**

```bash
git add openclaw/
git commit -m "feat: add OpenClaw agent integration with prompts, skills, and examples"
```

---

## Task 2: Add comprehensive docstrings to scripts

**Files:**
- Modify: `scripts/fetch_markets.py`
- Modify: `scripts/place_order.py`
- Modify: `scripts/portfolio_status.py`

**Step 1: Update scripts/fetch_markets.py with comprehensive docstring**

Add module docstring at the top of the file after imports:

```python
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
    ┌─────────────────────────────┬────────┬───────────┬─────────┐
    │ Market                      │ Price  │ Volume    │ Expires │
    ├─────────────────────────────┼────────┼───────────┼─────────┤
    │ BTC Up or Down 15 Minutes   │ 0.52   │ $12,450   │ 14:15  │
    └─────────────────────────────┴────────┴───────────┴─────────┘

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
```

**Step 2: Update scripts/place_order.py with comprehensive docstring**

```python
"""
Place orders on Polymarket markets.

This script places buy/sell orders on Polymarket using the CLOB API.
It supports market orders (FOK - Fill-Or-Kill) for immediate execution
and limit orders (GTC - Good-Til-Cancelled) for price control.

Usage:
    # Place a test buy order (dry run - recommended first)
    python scripts/place_order.py \\
        --btc-mode \\
        --side buy \\
        --price 0.55 \\
        --size 10 \\
        --dry-run true

    # Place a live sell order
    python scripts/place_order.py \\
        --btc-mode \\
        --side sell \\
        --price 0.60 \\
        --size 5 \\
        --dry-run false

    # Place order with explicit market/token IDs
    python scripts/place_order.py \\
        --market-id 0x123... \\
        --token-id 0x456... \\
        --side buy \\
        --price 0.50 \\
        --size 20

Arguments:
    --btc-mode: Use BTC 15-min market (auto-discovers current market)
    --market-id: Specific market ID (required if not using --btc-mode)
    --token-id: Specific token ID (required if not using --btc-mode)
    --side: Order side - "BUY" or "SELL" (required)
    --price: Order price from 0.0 to 1.0 (required)
    --size: Number of shares to trade (required, min: 0.01)
    --dry-run: Test mode (true) or live trading (false)
    --order-type: "market" (default, FOK) or "limit" (GTC)

Returns:
    Order confirmation including:
        - order_id: Unique order identifier
        - status: Order status (pending, filled, rejected)
        - side: BUY or SELL
        - price: Executed price
        - size: Number of shares

Examples:
    # Test order first (always recommended)
    $ python scripts/place_order.py --btc-mode --side buy --price 0.55 --size 10 --dry-run true
    [DRY RUN] Would place BUY order: 10 shares @ $0.55

    # Live market order (immediate execution or cancel)
    $ python scripts/place_order.py --btc-mode --side buy --price 0.55 --size 10 --dry-run false
    Order placed: ID=0xabc123, Status=pending

Exit codes:
    0: Success (order placed or dry-run verified)
    1: API error or network failure
    2: Invalid arguments (missing required, price out of range, etc.)
    3: Insufficient balance
    4: Market not accepting orders
    5: Order rejected by exchange

Important Notes:
    - Market orders use FOK (Fill-Or-Kill) - execute immediately or cancel
    - FOK is recommended for 15-min markets due to rapid price movement
    - Limit orders may not fill before market expiry
    - Always use dry-run=true first to verify parameters
    - Minimum order size applies (check market requirements)
    - Price must be between 0.0 and 1.0 for binary markets

Security:
    - Never commit .env file with credentials
    - Use read_only mode when possible for testing
    - Verify all parameters before live trading
"""
```

**Step 3: Update scripts/portfolio_status.py with comprehensive docstring**

```python
"""
Check Polymarket portfolio status and open orders.

This script queries the Polymarket CLOB API to retrieve current portfolio
information including open orders, positions, total value, and trade history.

Usage:
    # Check full portfolio status
    python scripts/portfolio_status.py

    # Filter by specific market
    python scripts/portfolio_status.py --market-id 0x123...

    # Output as JSON for agent processing
    python scripts/portfolio_status.py --json

    # Show trades only
    python scripts/portfolio_status.py --trades-only

Returns:
    Portfolio summary including:
        - total_value: Total portfolio value (cash + positions)
        - usdc_balance: Available USDC for trading
        - positions_value: Total value of open positions
        - open_orders: List of pending orders
        - positions: Dictionary of token holdings
        - trades: Recent trade history

Examples:
    # Full portfolio with table output
    $ python scripts/portfolio_status.py
    Portfolio Summary
    ================
    Total Value: $108.50
    Available: $107.12 USDC
    Positions: $1.38

    Open Orders: 2
    ┌──────────────┬────────┬────────┬─────────┐
    │ Market       │ Side   │ Price  │ Size    │
    ├──────────────┼────────┼────────┼─────────┤
    │ BTC UP 15m   │ BUY    │ 0.52   │ 10      │
    └──────────────┴────────┴────────┴─────────┘

    # JSON output for agents
    $ python scripts/portfolio_status.py --json
    {"total_value": 108.50, "usdc_balance": 107.12, ...}

Exit codes:
    0: Success
    1: API error or authentication failure
    2: Invalid arguments

Notes:
    - Requires trading mode with valid credentials
    - Position values calculated at current market prices
    - Trade history includes all historical trades
    - Open orders show pending unfilled orders
"""
```

**Step 4: Commit script documentation**

```bash
git add scripts/fetch_markets.py scripts/place_order.py scripts/portfolio_status.py
git commit -m "docs: add comprehensive docstrings to all scripts"
```

---

## Task 3: Add API docstrings to polymarket/ library

**Files:**
- Modify: `polymarket/client.py`
- Modify: `polymarket/auth.py`
- Modify: `polymarket/config.py`
- Modify: `polymarket/models.py`
- Modify: `polymarket/exceptions.py`
- Modify: `polymarket/utils/logging.py`
- Modify: `polymarket/utils/retry.py`

**Step 1: Update polymarket/client.py docstrings**

Add module docstring:

```python
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
    >>> markets = client.fetch_markets(query="btc")
    >>> orders = client.get_portfolio()

Authentication:
    - L1 (Private key): Required for signing requests
    - L2 (API credentials): Required for trading operations
    - Supports both Web3 wallets and Gmail/Magic Link accounts

Note:
    Market orders use FOK (Fill-Or-Kill) for immediate execution.
    Limit orders use GTC (Good-Til-Cancelled) and may not fill
    before 15-min markets expire.
"""
```

Update `PolymarketClient` class docstring:

```python
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
```

**Step 2: Update polymarket/auth.py docstrings**

```python
"""
Polymarket authentication handling.

This module manages authentication for the Polymarket CLOB API including
L1 (private key signing) and L2 (API key credentials) authentication.
It supports both Web3 wallets and Gmail/Magic Link custodial accounts.

Classes:
    PolymarketAuth: Handles authentication credential derivation

Constants:
    CLOB_HOST: Production CLOB API endpoint
    CLOB_CHAIN_ID: Polygon chain ID (137)

Example:
    >>> from polymarket.auth import PolymarketAuth
    >>> auth = PolymarketAuth(private_key="0x...")
    >>> credentials = auth.derive_credentials()

Account Types:
    - Web3 Wallets (MetaMask, etc.): signature_type=2
    - Gmail/Magic Link: signature_type=1, requires funder address
"""
```

**Step 3: Update polymarket/config.py docstrings**

```python
"""
Polymarket configuration from environment variables.

This module defines the configuration dataclass that reads settings from
environment variables. It supports both read_only and trading modes.

Classes:
    Settings: Configuration container with environment variable loading

Environment Variables:
    POLYMARKET_MODE: "read_only" or "trading"
    POLYMARKET_PRIVATE_KEY: Wallet private key (trading mode)
    POLYMARKET_API_KEY: API key for L2 auth (trading mode)
    POLYMARKET_API_SECRET: API secret for L2 auth (trading mode)
    POLYMARKET_API_PASSPHRASE: API passphrase (trading mode)
    POLYMARKET_FUNDER: Proxy wallet address (Gmail accounts)
    POLYMARKET_SIGNATURE_TYPE: 1 for Magic, 2 for Web3 (default: 1)

Example:
    >>> from polymarket.config import Settings
    >>> settings = Settings()
    >>> if settings.mode == Mode.TRADING:
    ...     print("Trading mode enabled")
"""
```

**Step 4: Update polymarket/models.py docstrings**

```python
"""
Data models for Polymarket API requests and responses.

This module defines Pydantic models for type-safe API interactions including
order requests, market data, portfolio summaries, and error responses.

Classes:
    OrderRequest: Request model for placing orders
    PortfolioSummary: Portfolio status summary
    MarketInfo: Market information model

Example:
    >>> from polymarket.models import OrderRequest
    >>> request = OrderRequest(
    ...     token_id="0x...",
    ...     side="BUY",
    ...     price=0.55,
    ...     size=10
    ... )
"""
```

**Step 5: Update polymarket/exceptions.py docstrings**

```python
"""
Polymarket-specific exceptions.

This module defines custom exceptions for Polymarket API errors including
authentication failures, order rejections, and network issues.

Classes:
    PolymarketError: Base exception for all Polymarket errors
    AuthenticationError: Raised when authentication fails
    OrderError: Raised when order placement fails
    MarketNotFoundError: Raised when market is not found

Example:
    >>> try:
    ...     client.place_order(...)
    ... except AuthenticationError:
    ...     print("Check your credentials")
"""
```

**Step 6: Update polymarket/utils/logging.py docstrings**

```python
"""
Logging configuration for Polymarket scripts.

This module sets up structured logging with appropriate levels and formats
for both development and production use.

Functions:
    setup_logging: Configure logging based on environment and verbosity

Example:
    >>> from polymarket.utils.logging import setup_logging
    >>> setup_logging(verbose=True)
    >>> logger = logging.getLogger(__name__)
    >>> logger.info("Starting operation")
"""
```

**Step 7: Update polymarket/utils/retry.py docstrings**

```python
"""
Retry logic for Polymarket API requests.

This module provides retry decorators for handling transient API failures
including network errors, rate limits, and temporary server issues.

Functions:
    retry_with_backoff: Retry function with exponential backoff

Example:
    >>> from polymarket.utils.retry import retry_with_backoff
    >>> @retry_with_backoff(max_attempts=3)
    ... def api_call():
    ...     return client.get_markets()
"""
```

**Step 8: Commit library documentation**

```bash
git add polymarket/
git commit -m "docs: add comprehensive API docstrings to polymarket library"
```

---

## Task 4: Create architecture documentation

**Files:**
- Create: `docs/architecture.md`

**Step 1: Create docs/architecture.md**

```markdown
# Polymarket Scripts Architecture

This document describes the architecture, design decisions, and implementation details of the Polymarket trading scripts.

## System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        Scripts Layer                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │ fetch_      │  │ place_      │  │ portfolio_          │ │
│  │ markets.py  │  │ order.py    │  │ status.py           │ │
│  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘ │
└─────────┼──────────────────┼────────────────────┼───────────┘
          │                  │                    │
          └──────────────────┼────────────────────┘
                             │
┌────────────────────────────┼─────────────────────────────────┐
│                    Polymarket Library                        │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              PolymarketClient                         │   │
│  │  - fetch_markets()    - place_order()                 │   │
│  │  - get_portfolio()    - get_btc_15min_market()        │   │
│  └──────────────────────────┬───────────────────────────┘   │
│                             │                                │
│  ┌──────────┐  ┌───────────┴──────────┐  ┌─────────────┐  │
│  │   Auth   │  │      Config          │  │   Models    │  │
│  └──────────┘  └──────────────────────┘  └─────────────┘  │
└──────────────────────────────┬───────────────────────────────┘
                               │
┌──────────────────────────────┼───────────────────────────────┐
│                    py-clob-client                          │
│  - ClobClient             - OrderArgs/MarketOrderArgs      │
│  - L1/L2 Authentication    - Order types (FOK/GTC)         │
└──────────────────────────────┬───────────────────────────────┘
                               │
┌──────────────────────────────┼───────────────────────────────┐
│                  Polymarket CLOB API                        │
│         https://clob.polymarket.com                         │
└──────────────────────────────────────────────────────────────┘
```

## Authentication Flow

```
User Credentials (.env)
         │
         ▼
┌─────────────────┐
│   Settings      │ Load environment variables
│   (config.py)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  PolymarketAuth │ Derive API credentials if needed
│   (auth.py)     │ (L1: private key, L2: API key/secret)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  ClobClient     │ Initialize py-clob-client
│ (py_clob_client)│ with credentials
└────────┬────────┘
         │
         ▼
   API Requests
```

### Account Type Handling

**Web3 Wallets (MetaMask, WalletConnect):**
- `POLYMARKET_SIGNATURE_TYPE=2`
- Uses Safe proxy wallet
- Private key directly controls funds

**Gmail/Magic Link Accounts:**
- `POLYMARKET_SIGNATURE_TYPE=1`
- Uses custodial proxy wallet
- Requires `POLYMARKET_FUNDER` (proxy wallet address)
- Private key authorizes but doesn't directly control funds

## Order Flow

```
User Input (CLI)
       │
       ▼
┌─────────────────┐
│  OrderRequest   │ Validate: side, price (0-1), size > 0
│   (models.py)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ place_order()   │ Determine order type:
│  (client.py)    │ - Market: FOK (default)
│                 │ - Limit: GTC
└────────┬────────┘
         │
         ▼
┌─────────────────────────────┐
│ Market Order (FOK)          │
│  - Create MarketOrderArgs   │
│  - Use amount (price*size)  │
│  - Immediate or cancel      │
│                             │
│ Limit Order (GTC)           │
│  - Create OrderArgs         │
│  - Use size directly        │
│  - Rests on book            │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│ create_and_post_order()     │ Send to CLOB API
│   (py_clob_client)          │
└──────────┬──────────────────┘
           │
           ▼
     Order Response
     - order_id
     - status
     - executed price/size
```

### Market Order (FOK) Rationale

For 15-minute expiry markets, we use FOK (Fill-Or-Kill) market orders:
1. **Rapid price movement:** BTC can swing significantly in 15 minutes
2. **Time decay:** Limit orders often expire worthless
3. **Liquidity:** BTC markets have sufficient depth for market orders
4. **Simplicity:** No need to monitor and cancel stale orders

## Market Discovery (BTC 15-Minute)

```
Current Time
     │
     ▼
┌─────────────────┐
│ Floor to 15-min │ Example: 10:09 → 10:00
│    interval     │ Using: (now // 900) * 900
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Build slug     │ btc-updown-15m-{timestamp}
│  Pattern        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Query API      │ GET /markets?slug=btc-updown-15m-...
│  for slug       │
└────────┬────────┘
         │
         ▼
    Market Data
    - token_id
    - market_id
    - price, volume
    - expiry time
```

### Slug Format

- Pattern: `btc-updown-15m-{epoch_seconds}`
- Timestamp: Interval start time (floor)
- Interval: 15 minutes (900 seconds)
- Example: `btc-updown-15m-1770608700` (for interval starting at 10:00 UTC)

## Error Handling

```
API Request
     │
     ▼
┌─────────────────┐
│  Try API Call   │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌────────┐ ┌──────────────────┐
│ Success│ │    Error         │
└────┬───┘ └────┬─────────────┘
     │          │
     │          ▼
     │   ┌─────────────────────┐
     │   │  Check Error Type   │
     │   └──────────┬──────────┘
     │              │
     │    ┌─────────┼─────────┐
     │    │         │         │
     │    ▼         ▼         ▼
     │ ┌──────┐ ┌────────┐ ┌──────────┐
     │ │ Auth │ │ Rate   │ │ Network  │
     │ │Error │ │ Limit  │ │ Error    │
     │ └──────┘ └────┬───┘ └──────┬───┘
     │             │             │
     │             ▼             ▼
     │      ┌──────────┐  ┌─────────────┐
     │      │ Backoff  │  │ Retry       │
     │      │ Retry    │  │ (max 3x)    │
     │      └──────────┘  └─────────────┘
     │
     └──────────────────┘
               │
               ▼
         Return Result
```

### Retry Strategy

- **Network errors:** Exponential backoff (1s, 2s, 4s)
- **Rate limits (429):** Backoff with Retry-After header
- **Auth errors:** No retry (credential issue)
- **Order rejections:** No retry (user intervention needed)

## Data Models

### OrderRequest
```python
token_id: str      # Token to trade
side: BUY|SELL     # Order direction
price: float       # Price (0-1 for binary)
size: float        # Number of shares
order_type: market|limit  # Order execution type
```

### PortfolioSummary
```python
open_orders: list[dict]      # Pending orders
total_notional: float         # Total order value
positions: dict[str, float]   # token_id -> quantity
total_exposure: float         # Position exposure
trades: list[dict]            # Trade history
usdc_balance: float           # Available USDC
positions_value: float        # Position value at market
total_value: float            # Portfolio total
```

## Security Considerations

1. **Credential Storage:**
   - Never commit `.env` file
   - Use `.env.example` with placeholders
   - Environment variable only (no hardcoded secrets)

2. **Order Safety:**
   - Dry-run mode by default
   - Price validation (0-1 range)
   - Size validation (> 0)
   - Market checks (accepting_orders)

3. **Logging:**
   - No private keys in logs
   - No signatures in logs
   - Sanitized API responses

## Testing

Tests are located in `tests/` and use pytest:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=polymarket --cov=scripts

# Run specific test
pytest tests/test_client.py::test_place_order
```

## Future Enhancements

- WebSocket streaming for real-time prices
- Advanced order types (stop-loss, take-profit)
- Multi-legged orders (conditional orders)
- Order book depth visualization
- Performance analytics
- Backtesting framework
"""
```

**Step 2: Commit architecture documentation**

```bash
git add docs/architecture.md
git commit -m "docs: add comprehensive architecture documentation"
```

---

## Task 5: Update README with OpenClaw section

**Files:**
- Modify: `README.md`

**Step 1: Add OpenClaw section to README.md**

Add before "Known Limitations" section:

```markdown
## OpenClaw Agent Integration

This repository includes agent-specific resources in the `openclaw/` directory for use with OpenClaw agents.

### Available Skills

- **polymarket_fetch**: Fetch market data
- **polymarket_trade**: Place orders
- **polymarket_status**: Check portfolio

### Quick Start for Agents

1. Copy `.env.example` to `.env` and add credentials
2. Install: `pip install -r requirements.txt`
3. Run: `python scripts/fetch_markets.py --btc-mode`

See [`openclaw/README.md`](openclaw/README.md) for detailed agent documentation.

### Example Agent Workflow

```bash
# 1. Fetch market data
python scripts/fetch_markets.py --btc-mode

# 2. Check portfolio
python scripts/portfolio_status.py

# 3. Place order (dry run first)
python scripts/place_order.py --btc-mode --side buy --price 0.55 --size 10 --dry-run true

# 4. Place live order
python scripts/place_order.py --btc-mode --side buy --price 0.55 --size 10 --dry-run false
```

See `openclaw/examples/` for complete workflow examples.
```

**Step 2: Commit README update**

```bash
git add README.md
git commit -m "docs: add OpenClaw agent integration section to README"
```

---

## Task 6: Security scan and .gitignore verification

**Files:**
- Verify: `.gitignore`
- Scan: All Python files for secrets

**Step 1: Verify .gitignore contents**

Check that `.gitignore` excludes:
```
.env
__pycache__/
*.pyc
.pytest_cache/
.claude/
*.log
*.egg-info/
dist/
build/
.DS_Store
```

Run:
```bash
cat .gitignore
```

Expected: All sensitive and temp files listed above

**Step 2: Scan for hardcoded secrets**

```bash
# Check for private keys
grep -r "0x[a-fA-F0-9]\{64\}" --include="*.py" | grep -v "example\|test\|TODO"

# Check for GitHub tokens
grep -r "ghp_" --include="*.py"

# Check for API credentials in comments
grep -r "API_KEY\|API_SECRET" --include="*.py" | grep -v "example\|import\|env"

# Check .env file exists but has no real values
if [ -f .env ]; then
    grep -E "PRIVATE_KEY|API_KEY|API_SECRET|ghp_" .env | grep -v "0x\.\.\.|example"
fi
```

Expected: No real credentials found

**Step 3: Verify .env.example has placeholders**

```bash
cat .env.example
```

Expected: All values are placeholders like `0x...`, `your_key_here`, etc.

**Step 4: Run tests to ensure nothing broke**

```bash
pytest tests/ -v
```

Expected: All tests pass (54 tests)

**Step 5: Security scan commit**

```bash
# Add/commit .gitignore if updated
git add .gitignore .env.example
git commit -m "chore: verify .gitignore and security scan complete"
```

---

## Task 7: Configure git and push to GitHub

**Files:**
- Configure: Git user settings
- Add: Remote origin
- Push: To GitHub

**Step 1: Configure git user**

```bash
git config user.name "timchrist49"
git config user.email "timothy.christ49@gmail.com"
```

Verify:
```bash
git config user.name && git config user.email
```

Expected: "timchrist49" and "timothy.christ49@gmail.com"

**Step 2: Add remote origin**

```bash
git remote add origin https://github.com/timchrist49/polymarket-scripts.git
```

Verify:
```bash
git remote -v
```

Expected: Shows origin with GitHub URL

**Step 3: Verify status before push**

```bash
git status
```

Expected: Shows "On branch master" with committed changes ready to push

**Step 4: Push to GitHub**

```bash
git push -u origin master
```

Expected: Successful push, shows repo URL

**Step 5: Verify push on GitHub**

Check https://github.com/timchrist49/polymarket-scripts

Expected:
- Repository exists with all files
- No .env file (gitignored)
- .env.example present with placeholders
- openclaw/ directory exists
- Documentation is comprehensive

**Step 6: Push complete commit**

```bash
# Already pushed, but can add a tag if desired
git tag -a v1.0.0 -m "Initial release with OpenClaw integration"
git push origin v1.0.0
```

---

## Summary

This plan creates:

1. **openclaw/** directory with agent-specific resources
2. **Comprehensive docstrings** on all Python files
3. **Architecture documentation** explaining design decisions
4. **Updated README** with OpenClaw section
5. **Security verification** before pushing to GitHub

Total tasks: 7
Total commits: ~8
Estimated time: 30-45 minutes
