# Polymarket CLOB WebSocket Research

## Overview
Research findings for Polymarket CLOB WebSocket endpoint, subscription format, and message schemas for the BTC 15-min prediction market trading bot.

**Date:** 2026-02-10
**Status:** Complete
**Sources:** Polymarket Official Documentation, GitHub Repositories

---

## WebSocket Endpoint URLs

### Primary CLOB Market Channel
```
wss://ws-subscriptions-clob.polymarket.com/ws
```

### CLOB WebSocket with Trailing Slash (Alternative)
```
wss://ws-subscriptions-clob.polymarket.com/ws/
```

### User Channel (Requires Authentication)
```
wss://ws-subscriptions-clob.polymarket.com/ws/user
```

### Real-Time Data Service (Broader Market Data)
```
wss://ws-live-data.polymarket.com
```

**Important:** Market data subscriptions (prices, orderbook) do NOT require authentication. User subscriptions (orders, trades) REQUIRE CLOB authentication.

---

## Subscription Format

### Initial Connection JSON (Market Channel - No Auth)

```json
{
  "type": "MARKET",
  "assets_ids": ["array_of_asset_ids"],
  "custom_feature_enabled": false
}
```

### Dynamic Subscription Management (After Connection)

```json
{
  "assets_ids": ["asset_id_1", "asset_id_2"],
  "markets": ["condition_id_1"],
  "operation": "subscribe",
  "custom_feature_enabled": false
}
```

Or to unsubscribe:

```json
{
  "assets_ids": ["asset_id_1"],
  "operation": "unsubscribe"
}
```

---

## Message Types & Schemas

### 1. **book** - Orderbook Snapshots
Sent on subscription and after trades execute.

```json
{
  "event_type": "book",
  "asset_id": "string",
  "market": "condition_id",
  "timestamp": 1707398400000,
  "hash": "string",
  "buys": [
    {
      "price": "0.45",
      "size": "100"
    }
  ],
  "sells": [
    {
      "price": "0.55",
      "size": "200"
    }
  ]
}
```

### 2. **price_change** - Order Placement/Cancellation Updates
Emitted when orders are placed or cancelled.

```json
{
  "event_type": "price_change",
  "market": "condition_id",
  "timestamp": 1707398400000,
  "price_changes": [
    {
      "asset_id": "string",
      "price": "0.50",
      "size": "150",
      "side": "BUY",
      "hash": "string",
      "best_bid": "0.49",
      "best_ask": "0.51"
    }
  ]
}
```

**⚠️ Breaking Change:** Schema updates scheduled for September 15, 2025.

### 3. **tick_size_change** - Minimum Tick Size Adjustments
Triggered when price > 0.96 or < 0.04.

```json
{
  "event_type": "tick_size_change",
  "asset_id": "string",
  "market": "condition_id",
  "old_tick_size": "0.01",
  "new_tick_size": "0.001",
  "side": "BUY",
  "timestamp": 1707398400000
}
```

### 4. **last_trade_price** - Trade Execution Events
Emitted when trades execute.

```json
{
  "event_type": "last_trade_price",
  "asset_id": "string",
  "market": "condition_id",
  "price": "0.52",
  "side": "BUY",
  "size": "500",
  "fee_rate_bps": 50,
  "timestamp": 1707398400000
}
```

### 5. **best_bid_ask** - Best Price Changes (Feature-Flagged)
```json
{
  "event_type": "best_bid_ask",
  "market": "condition_id",
  "asset_id": "string",
  "best_bid": "0.49",
  "best_ask": "0.51",
  "spread": "0.02",
  "timestamp": 1707398400000
}
```

### 6. **new_market** - Market Creation (Feature-Flagged)
```json
{
  "event_type": "new_market",
  "id": "market_id",
  "question": "string",
  "market": "condition_id",
  "slug": "market_slug",
  "description": "string",
  "assets_ids": ["asset_1", "asset_2"],
  "outcomes": ["YES", "NO"],
  "event_message": {
    "key": "value"
  },
  "timestamp": 1707398400000
}
```

### 7. **market_resolved** - Market Resolution (Feature-Flagged)
```json
{
  "event_type": "market_resolved",
  "market": "condition_id",
  "winning_asset_id": "string",
  "winning_outcome": "YES",
  "timestamp": 1707398400000
}
```

---

## Key Field Reference

### Common Fields Across Messages
- **event_type** (string): Message type identifier
- **asset_id** (string): Unique asset identifier
- **market** (string): Condition ID / Market identifier
- **timestamp** (number): Unix milliseconds (always)
- **side** (string): "BUY" or "SELL"

### Price & Size Fields
- **price** (string): Decimal price (e.g., "0.52")
- **size** (string): Decimal quantity (e.g., "500")
- **best_bid** (string): Current best bid price
- **best_ask** (string): Current best ask price

### Order Book Fields
- **buys** (array): Array of {price, size} objects
- **sells** (array): Array of {price, size} objects
- **hash** (string): Transaction/order hash

### Fee Information
- **fee_rate_bps** (number): Fee rate in basis points (0.01% = 1 bps)

---

## Message Delivery Guarantees

1. **Initial Data:** Book messages sent immediately upon subscription
2. **Real-time Updates:** All message types streamed immediately when events occur
3. **Timestamps:** All timestamps are in Unix milliseconds
4. **No Filtering:** Market channel sends all events for subscribed assets

---

## Implementation Notes for BTC 15-min Market

### For Tracking Last Trade Price
Use `last_trade_price` messages to:
- Capture execution prices
- Track fee rates
- Update price changes immediately

### For Orderbook Analysis
Use `book` messages to:
- Get complete orderbook snapshots
- Identify support/resistance levels
- Calculate spread and depth

### For Momentum Signals
Use `price_change` messages to:
- Detect order flow imbalances
- Monitor best bid/ask changes
- Identify directional momentum

### Connection Strategy
1. Connect to `wss://ws-subscriptions-clob.polymarket.com/ws`
2. Subscribe with BTC market's asset_id and condition_id
3. No authentication needed for market channel
4. Buffer initial `book` message before processing trades
5. Handle feature-flagged events gracefully (may not be enabled)

---

## Related Resources

- **Official Docs:** https://docs.polymarket.com/developers/CLOB/websocket/market-channel
- **GitHub Clients:**
  - TypeScript: https://github.com/Polymarket/real-time-data-client
  - Rust: https://github.com/Polymarket/rs-clob-client
- **Community Examples:** https://github.com/nevuamarkets/poly-websockets

---

## Next Steps

1. ✓ WebSocket endpoint identified: `wss://ws-subscriptions-clob.polymarket.com/ws`
2. ✓ Subscription format documented
3. ✓ Message types cataloged with field schemas
4. ⏳ Implement WebSocket data collector (Task 2)
5. ⏳ Build market data parser
6. ⏳ Integrate with scoring system

---

*Research completed by Claude Code*
*Document created: 2026-02-10*
