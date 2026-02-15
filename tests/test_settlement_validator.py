"""
Legacy settlement validator tests - OBSOLETE

These tests were for the old validation behavior (fetch from multiple sources,
check agreement, return average). As of 2026-02-15, SettlementPriceValidator
was changed to use a 3-tier fallback hierarchy instead:

1. Chainlink buffer (primary, matches Polymarket settlement)
2. CoinGecko API (secondary)
3. Binance API (last resort)

The new fallback behavior is tested in test_settlement_validator_fallback.py.

Original tests removed:
- test_validate_prices_agree: Tested price averaging (no longer relevant)
- test_validate_prices_disagree: Tested tolerance checking (no longer relevant)
- test_validate_insufficient_sources: Tested MIN_SOURCES check (removed)
- test_calculate_spread: Tested _calculate_spread() method (removed)
- test_fetch_coingecko_at_timestamp_integration: Tested individual fetch (still works via fallback tests)

See:
- test_settlement_validator_fallback.py for current behavior tests
- docs/plans/2026-02-15-price-source-hierarchy-fix.md for design rationale
"""

# This file is intentionally empty - kept for historical reference only.
# All active tests are in test_settlement_validator_fallback.py
