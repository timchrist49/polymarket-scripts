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
│  │  - discover_btc_15min_market()                        │   │
│  │  - create_order()                                     │   │
│  │  - get_portfolio_summary()                           │   │
│  └──────────────────────────┬───────────────────────────┘   │
│                             │                                │
│  ┌──────────┐  ┌───────────┴──────────┐  ┌─────────────┐  │
│  │   Auth   │  │      Config          │  │   Models    │  │
│  │Manager   │  │   (Settings)         │  │ (Pydantic)  │  │
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
│                  Polymarket APIs                            │
│  CLOB API:  https://clob.polymarket.com                    │
│  Gamma API: https://gamma-api.polymarket.com               │
└──────────────────────────────────────────────────────────────┘
```

## Authentication Flow

```
User Credentials (.env)
         │
         ▼
┌─────────────────┐
│   Settings      │ Load environment variables
│   (config.py)   │ - POLYMARKET_MODE
└────────┬────────┘ - POLYMARKET_PRIVATE_KEY
         │                - POLYMARKET_SIGNATURE_TYPE
         ▼
┌─────────────────┐
│  AuthManager    │ Validate credentials based on mode
│   (auth.py)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  get_clob_      │ Prepare kwargs for ClobClient:
│  client_kwargs()│ - key (private key)
│                 │ - creds (API key/secret/secret)
│                 │ - signature_type (1 or 2)
│                 │ - funder (optional, for Gmail)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  ClobClient     │ Initialize py-clob-client
│ (py_clob_client)│ with derived credentials
└────────┬────────┘
         │
         ▼
   API Requests
```

### Account Type Handling

**Web3 Wallets (MetaMask, WalletConnect, Rabby):**
- `POLYMARKET_SIGNATURE_TYPE=2`
- Uses Safe proxy wallet (deployed contract)
- Private key directly controls funds
- No `POLYMARKET_FUNDER` needed

**Gmail/Magic Link Accounts:**
- `POLYMARKET_SIGNATURE_TYPE=1`
- Uses custodial proxy wallet (managed by Polymarket)
- Requires `POLYMARKET_FUNDER` (proxy wallet address)
- Private key authorizes but doesn't directly control funds

### L1 vs L2 Authentication

- **L1 (Private Key):** Required for signing all requests
  - Used for: Order placement, portfolio queries
  - Derived from: `POLYMARKET_PRIVATE_KEY`

- **L2 (API Credentials):** Optional, can be auto-derived from private key
  - Used for: Enhanced rate limits, some trading operations
  - Manually set via: `POLYMARKET_API_KEY`, `POLYMARKET_API_SECRET`, `POLYMARKET_API_PASSPHRASE`
  - Auto-derived via: `ClobClient.create_or_derive_api_creds()`

## Order Flow

```
User Input (CLI)
       │
       ▼
┌─────────────────┐
│  OrderRequest   │ Validate: side (BUY/SELL), price (0-1), size > 0
│   (models.py)   │ Determine order_type (market/limit)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ create_order()  │ Check mode (trading required)
│  (client.py)    │ Check dry_run flag
└────────┬────────┘
         │
         ▼
┌─────────────────────────────┐
│ Market Order (FOK)          │
│  - Create MarketOrderArgs   │
│  - Use amount (price*size)  │
│  - Immediate or cancel      │
│  - OrderType.FOK            │
│                             │
│ Limit Order (GTC)           │
│  - Create OrderArgs         │
│  - Use size directly        │
│  - Rests on book            │
│  - Default OrderType        │
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
     - status ("posted", "failed", "dry_run")
     - accepted (bool)
     - raw_response (dict)
```

### Market Order (FOK) Rationale

For 15-minute expiry markets, we use FOK (Fill-Or-Kill) market orders by default:

1. **Rapid price movement:** BTC can swing significantly in 15 minutes
2. **Time decay:** Limit orders often expire worthless as the market closes
3. **Liquidity:** BTC markets have sufficient depth for immediate fills
4. **Simplicity:** No need to monitor and cancel stale orders
5. **Predictability:** Get executed price immediately or reject

### Order Types Implementation

**Market Orders (default):**
- Uses `MarketOrderArgs` from py-clob-client
- `amount` = price * size for BUY, size for SELL
- `OrderType.FOK` for immediate execution
- Optional price limit to prevent slippage

**Limit Orders:**
- Uses `OrderArgs` from py-clob-client
- `size` = number of shares
- Default `OrderType` (GTC - Good-Til-Cancelled)
- May not fill before market expiry

## Market Discovery (BTC 15-Minute)

```
Current Time (UTC)
     │
     ▼
┌─────────────────┐
│ Floor to 15-min │ Example: 10:09 → 10:00
│    interval     │ Using: (minute // 15) * 15
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Build slug     │ btc-updown-15m-{timestamp}
│  Pattern        │ Timestamp = interval start epoch
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Try Slug       │ Query Gamma API: GET /markets?slug=...
│  Discovery      │ Try current ± 15min intervals
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌────────┐ ┌──────────────────┐
│ Found? │ │  Fallback:       │
│ Return │ │  Search Query    │
└────────┘ └────┬─────────────┘
               │ Search: "Bitcoin Up or Down"
               │ Filter: active, accepting_orders
               │ Priority: 15-min markets > other BTC
               ▼
         Market Data
         - token_id (clobTokenIds)
         - market_id (conditionId)
         - best_bid, best_ask
         - volume, liquidity
```

### Slug Format

- **Pattern:** `btc-updown-15m-{epoch_seconds}`
- **Timestamp:** Unix epoch of interval START time (floor)
- **Interval:** 15 minutes (900 seconds)
- **Example:** `btc-updown-15m-1770608700` (for interval starting at 10:00 UTC)

### Discovery Strategy

1. **Primary:** Slug-based lookup (most reliable for 15-min markets)
   - Try current interval slug
   - Try ± 15, ± 30 minute offsets (for clock skew)
   - Return first tradeable market found

2. **Secondary:** Search query (fallback)
   - Search for "Bitcoin Up or Down"
   - Filter by: active=True, accepting_orders=True
   - Prioritize 15-minute markets
   - Fall back to any BTC market if no 15-min found

3. **Failure:** Raise `MarketDiscoveryError`
   - User can manually specify `--market-id`

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
     │ └──┬───┘ └────┬───┘ └──────┬───┘
     │    │No        │Retry  │Retry    │
     │    │Retry     │w/ back│w/ back  │
     │    │          │off    │off      │
     │
     └──────────────────┘
               │
               ▼
         Return Result
```

### Retry Strategy

The `@retry` decorator provides exponential backoff:

- **Network errors:** Retry with backoff (1s, 2s, 4s, 8s, 16s)
  - `NetworkError`: Connection issues, timeouts
  - `UpstreamAPIError`: 5xx server errors

- **Rate limits (429):** No automatic retry (manual intervention)
  - `RateLimitError`: HTTP 429 from API
  - User should wait and retry

- **Auth errors:** No retry (credential issue)
  - `AuthError`: Invalid credentials
  - `ValidationError`: Invalid parameters

- **Order rejections:** No retry (user intervention needed)
  - Market not accepting orders
  - Insufficient balance
  - Invalid price/size

### Exception Hierarchy

```
PolymarketError (base)
├── ConfigError         - Missing/invalid environment
├── AuthError           - Authentication failure
├── ValidationError     - Input validation failed
├── RateLimitError      - HTTP 429 rate limit
├── UpstreamAPIError    - 5xx server errors
├── NetworkError        - Connection issues
└── MarketDiscoveryError - No BTC market found
```

## Data Models

### OrderRequest
```python
token_id: str              # Token to trade (from clobTokenIds)
side: BUY|SELL            # Order direction
price: float              # Price (0-1 for binary options)
size: float               # Number of shares
order_type: market|limit  # Order execution type (default: market)
```

### OrderResponse
```python
order_id: str             # Order ID from exchange
status: str               # "posted", "failed", "dry_run"
accepted: bool            # True if order accepted
raw_response: dict        # Full API response
error_message: str|None   # Error details if failed
```

### Market
```python
id: str                   # Market ID
condition_id: str         # Condition ID for CLOB
question: str|None        # Market question
slug: str|None            # URL slug
active: bool|None         # Is market active
closed: bool|None         # Is market closed
accepting_orders: bool|None  # Can place orders
clob_token_ids: str|None  # JSON array of token IDs
best_bid: float|None      # Current bid price
best_ask: float|None      # Current ask price
last_trade_price: float|None  # Last trade price
volume_num: float|None    # Trading volume
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
   - Never commit `.env` file (in `.gitignore`)
   - Use `.env.example` with placeholders
   - Environment variables only (no hardcoded secrets)

2. **Order Safety:**
   - Dry-run mode by default (`DRY_RUN=true`)
   - Price validation (0-1 range for binary options)
   - Size validation (> 0)
   - Market checks (`accepting_orders` flag)
   - Trade mode confirmation (`POLYMARKET_MODE=trading`)

3. **Logging:**
   - No private keys in logs (use `get_masked_key()`)
   - No signatures in logs
   - Sanitized API responses
   - Configurable log level (`LOG_LEVEL`)

4. **Network Security:**
   - HTTPS only for API endpoints
   - Timeout on all requests (30s default)
   - Retry with exponential backoff
   - No credential caching beyond session

## Testing

Tests are located in `tests/` and use pytest:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=polymarket --cov=scripts

# Run specific test
pytest tests/test_client.py::test_create_order

# Run with verbose output
pytest -v

# Run only fast tests
pytest -m "not slow"
```

### Test Structure

```
tests/
├── conftest.py              # Shared fixtures
├── test_auth.py            # AuthManager tests
├── test_client.py          # PolymarketClient tests
├── test_config.py          # Settings tests
├── test_models.py          # Pydantic model tests
├── test_fetch_markets_script.py
├── test_place_order_script.py
├── test_portfolio_status_script.py
└── test_*.py               # Utility tests
```

### Key Fixtures

- `mock_settings`: Mocked configuration
- `mock_auth_manager`: Mocked auth manager
- `mock_clob_client`: Mocked CLOB client
- `sample_market`: Sample market data

## Configuration

### Environment Variables

```bash
# Required for trading
POLYMARKET_MODE=trading
POLYMARKET_PRIVATE_KEY=0x...

# Account type
POLYMARKET_SIGNATURE_TYPE=2  # 1=Gmail/Magic, 2=Web3 wallet

# Optional: For Gmail/Magic accounts
POLYMARKET_FUNDER=0x...  # Proxy wallet address

# Optional: L2 API credentials (auto-derived if not set)
POLYMARKET_API_KEY=...
POLYMARKET_API_SECRET=...
POLYMARKET_API_PASSPHRASE=...

# API endpoints (usually defaults are fine)
POLYMARKET_CLOB_URL=https://clob.polymarket.com
POLYMARKET_GAMMA_URL=https://gamma-api.polymarket.com
POLYMARKET_CHAIN_ID=137

# Runtime options
DRY_RUN=true              # Default: true
LOG_LEVEL=INFO            # DEBUG, INFO, WARNING, ERROR
LOG_JSON=false            # Structured logging
```

### Mode Selection

- **read_only:** Fetch market data only (no credentials needed)
- **trading:** Full trading capabilities (requires `POLYMARKET_PRIVATE_KEY`)

## Scripts

### fetch_markets.py
Fetch and display market data for BTC 15-minute markets.

```bash
# Discover current BTC 15-min market
python scripts/fetch_markets.py

# Search for specific market
python scripts/fetch_markets.py --search "Bitcoin"

# Show detailed market info
python scripts/fetch_markets.py --market-id <id>
```

### place_order.py
Place orders on Polymarket.

```bash
# Dry run (default)
python scripts/place_order.py --side BUY --price 0.55 --size 10

# Live order
python scripts/place_order.py --side BUY --price 0.55 --size 10 --live

# Limit order
python scripts/place_order.py --side SELL --price 0.60 --size 10 --type limit
```

### portfolio_status.py
Display portfolio summary.

```bash
# Show portfolio
python scripts/portfolio_status.py

# Show only positions
python scripts/portfolio_status.py --positions-only

# Show only trades
python scripts/portfolio_status.py --trades-only
```

## Design Decisions

### Why Pydantic Models?

- Type safety for API interactions
- Automatic validation of input data
- Clear field documentation
- Easy serialization/deserialization
- Better error messages

### Why Separate AuthManager?

- Centralized credential management
- Easy to test with mocks
- Clear separation of concerns
- Supports both account types transparently

### Why Retry Decorator?

- Reusable across all API calls
- Configurable backoff strategy
- Jitter prevents thundering herd
- Clear retry logging

### Why Dry-Run by Default?

- Prevents accidental trades
- Allows testing without risk
- Clear indication when live trading
- Easy to toggle with `--live` flag

## Future Enhancements

- [ ] WebSocket streaming for real-time prices
- [ ] Advanced order types (stop-loss, take-profit)
- [ ] Multi-legged orders (conditional orders)
- [ ] Order book depth visualization
- [ ] Performance analytics and P&L tracking
- [ ] Backtesting framework
- [ ] Automated trading strategies
- [ ] Price alerts and notifications
- [ ] Position hedging tools
- [ ] Market analysis indicators

## API References

- **Polymarket CLOB API:** https://clob.polymarket.com
- **Gamma API:** https://gamma-api.polymarket.com
- **py-clob-client:** https://github.com/0xPlaygrounds/polymarket-clob-client-python

## Support

For issues or questions:
1. Check the test files for examples
2. Review the exception messages
3. Enable DEBUG logging for more details
4. Check Polymarket dashboard for market status
