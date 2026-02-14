"""
Tests to verify settlement bug fix.

These tests ensure that:
1. parse_market_start() returns START time (not close time)
2. Settlement compares START vs END prices correctly
3. Outcomes are determined correctly based on price movement
"""

import pytest
from datetime import datetime, timezone
from polymarket.trading.market_tracker import MarketTracker
from polymarket.performance.settler import TradeSettler
from polymarket.config import get_settings


def test_parse_market_start_returns_start_time():
    """
    Critical test: Verify parse_market_start returns START time, not close time.

    Market slug: btc-updown-15m-1771051500
    - Timestamp 1771051500 = 2026-02-14 06:45:00 (CLOSE time)
    - Should return: 2026-02-14 06:30:00 (START time = close - 15 min)
    """
    settings = get_settings()
    tracker = MarketTracker(settings)

    # Market closes at 06:45:00
    slug = "btc-updown-15m-1771051500"
    start_time = tracker.parse_market_start(slug)

    # Should return START time (06:30:00), NOT close time (06:45:00)
    expected_start = datetime(2026, 2, 14, 6, 30, 0, tzinfo=timezone.utc)

    assert start_time == expected_start, \
        f"Expected START time {expected_start}, but got {start_time}"

    # Verify it's NOT the close time
    close_time = datetime(2026, 2, 14, 6, 45, 0, tzinfo=timezone.utc)
    assert start_time != close_time, \
        f"parse_market_start() returned CLOSE time {close_time}, should return START time"


def test_parse_market_start_subtraction():
    """Verify the 15-minute subtraction is correct."""
    settings = get_settings()
    tracker = MarketTracker(settings)

    slug = "btc-updown-15m-1771051500"
    start_time = tracker.parse_market_start(slug)

    # Calculate expected: close - 900 seconds (15 minutes)
    close_timestamp = 1771051500
    expected_timestamp = close_timestamp - 900  # 1771050600

    assert int(start_time.timestamp()) == expected_timestamp, \
        f"Expected timestamp {expected_timestamp}, got {int(start_time.timestamp())}"


def test_settlement_outcome_btc_goes_up():
    """
    Test that settlement correctly identifies YES win when BTC goes UP.

    Scenario: BTC moves from $68,847.89 to $68,849.03 (+$1.14)
    Expected: YES (UP) wins
    """
    # Create a mock settler (no need for actual DB/BTC service for this test)
    settler = type('MockSettler', (), {
        '_determine_outcome': TradeSettler._determine_outcome.__get__(None, TradeSettler)
    })()

    # BTC went UP
    btc_start = 68847.89
    btc_close = 68849.03

    outcome = settler._determine_outcome(btc_close, btc_start)

    assert outcome == "YES", \
        f"BTC went UP from ${btc_start} to ${btc_close}, YES should win, got {outcome}"


def test_settlement_outcome_btc_goes_down():
    """
    Test that settlement correctly identifies NO win when BTC goes DOWN.

    Scenario: BTC moves from $69,000 to $68,900 (-$100)
    Expected: NO (DOWN) wins
    """
    settler = type('MockSettler', (), {
        '_determine_outcome': TradeSettler._determine_outcome.__get__(None, TradeSettler)
    })()

    # BTC went DOWN
    btc_start = 69000.00
    btc_close = 68900.00

    outcome = settler._determine_outcome(btc_close, btc_start)

    assert outcome == "NO", \
        f"BTC went DOWN from ${btc_start} to ${btc_close}, NO should win, got {outcome}"


def test_settlement_outcome_no_movement():
    """
    Test that settlement identifies NO win when BTC doesn't move (tie).

    Scenario: BTC stays at $68,850
    Expected: NO wins (ties go to DOWN)
    """
    settler = type('MockSettler', (), {
        '_determine_outcome': TradeSettler._determine_outcome.__get__(None, TradeSettler)
    })()

    # No movement
    btc_start = 68850.00
    btc_close = 68850.00

    outcome = settler._determine_outcome(btc_close, btc_start)

    assert outcome == "NO", \
        f"BTC didn't move (${btc_start} = ${btc_close}), NO should win, got {outcome}"


def test_profit_loss_yes_trade_wins():
    """Test P&L calculation when YES trade wins."""
    settler = type('MockSettler', (), {
        '_calculate_profit_loss': TradeSettler._calculate_profit_loss.__get__(None, TradeSettler)
    })()

    # YES trade @ $0.30, position $10, outcome YES
    action = "YES"
    outcome = "YES"
    position_size = 10.00
    executed_price = 0.30

    profit_loss, is_win = settler._calculate_profit_loss(
        action, outcome, position_size, executed_price
    )

    # Shares = 10 / 0.30 = 33.33
    # Payout = 33.33 * $1.00 = $33.33
    # P&L = $33.33 - $10.00 = $23.33

    assert is_win == True, "YES trade with YES outcome should WIN"
    assert abs(profit_loss - 23.33) < 0.01, \
        f"Expected P&L ~$23.33, got ${profit_loss:.2f}"


def test_profit_loss_no_trade_loses():
    """Test P&L calculation when NO trade loses."""
    settler = type('MockSettler', (), {
        '_calculate_profit_loss': TradeSettler._calculate_profit_loss.__get__(None, TradeSettler)
    })()

    # NO trade @ $0.13, position $7, outcome YES (BTC went UP)
    action = "NO"
    outcome = "YES"
    position_size = 7.00
    executed_price = 0.13

    profit_loss, is_win = settler._calculate_profit_loss(
        action, outcome, position_size, executed_price
    )

    assert is_win == False, "NO trade with YES outcome should LOSE"
    assert profit_loss == -7.00, \
        f"Expected P&L = -${position_size:.2f}, got ${profit_loss:.2f}"


def test_real_world_trade_228():
    """
    Test the actual Trade #228 scenario that exposed the bug.

    Market: btc-updown-15m-1771051500 (06:30:00 to 06:45:00)
    BTC: $68,847.89 → $68,849.03 (+$1.14 UP)
    Trade: NO @ $0.130, $6.98

    Expected: YES wins, NO trade loses $6.98
    Bug caused: Thought NO won, calculated +$46.71 profit
    """
    settings = get_settings()
    tracker = MarketTracker(settings)

    slug = "btc-updown-15m-1771051500"

    # Parse should return START time
    start_time = tracker.parse_market_start(slug)
    assert start_time == datetime(2026, 2, 14, 6, 30, 0, tzinfo=timezone.utc), \
        "START time incorrect"

    # Simulate settlement
    settler = type('MockSettler', (), {
        '_determine_outcome': TradeSettler._determine_outcome.__get__(None, TradeSettler),
        '_calculate_profit_loss': TradeSettler._calculate_profit_loss.__get__(None, TradeSettler)
    })()

    btc_start = 68847.89  # Correct START price
    btc_close = 68849.03  # END price

    # Determine outcome
    outcome = settler._determine_outcome(btc_close, btc_start)
    assert outcome == "YES", "BTC went UP, YES should win"

    # Calculate P&L for NO trade
    action = "NO"
    position_size = 6.98
    executed_price = 0.130

    profit_loss, is_win = settler._calculate_profit_loss(
        action, outcome, position_size, executed_price
    )

    assert is_win == False, "NO trade should LOSE when YES wins"
    assert profit_loss == -6.98, f"Should lose position_size, got ${profit_loss:.2f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
