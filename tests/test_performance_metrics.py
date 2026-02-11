# tests/test_performance_metrics.py
import pytest
from datetime import datetime, timedelta
from polymarket.performance.metrics import MetricsCalculator
from polymarket.performance.database import PerformanceDatabase

@pytest.fixture
def db_with_trades():
    """Create database with sample trades."""
    db = PerformanceDatabase(":memory:")

    # Add 10 sample trades
    base_time = datetime(2026, 2, 11, 10, 0, 0)

    for i in range(10):
        trade_data = {
            "timestamp": base_time + timedelta(minutes=i*15),
            "market_slug": f"btc-updown-15m-{i}",
            "action": "NO" if i % 2 == 0 else "YES",
            "confidence": 0.7 + (i * 0.02),
            "position_size": 5.0,
            "btc_price": 66000.0 + (i * 100),
            "signal_type": "STRONG_BEARISH" if i < 5 else "STRONG_BULLISH",
            "is_end_phase": i > 7,
            "executed_price": 0.50
        }
        trade_id = db.log_trade(trade_data)

        # Set outcomes for first 8 trades
        if i < 8:
            is_win = (i % 3 != 0)  # 6 wins, 2 losses
            db.update_outcome(
                market_slug=f"btc-updown-15m-{i}",
                actual_outcome="DOWN" if is_win == (trade_data["action"] == "NO") else "UP",
                profit_loss=4.0 if is_win else -5.0
            )

    yield db
    db.close()

def test_calculate_win_rate(db_with_trades):
    """Test win rate calculation."""
    calc = MetricsCalculator(db_with_trades)

    win_rate = calc.calculate_win_rate()
    assert win_rate == 0.625  # 5 wins / 8 trades (i % 3 != 0 gives 5 wins)

def test_calculate_profit_loss(db_with_trades):
    """Test total profit/loss calculation."""
    calc = MetricsCalculator(db_with_trades)

    total_profit = calc.calculate_total_profit()
    assert total_profit == 5.0  # (5 * 4.0) - (3 * 5.0) = 20 - 15 = 5.0

def test_signal_performance(db_with_trades):
    """Test win rate by signal type."""
    calc = MetricsCalculator(db_with_trades)

    signal_perf = calc.calculate_signal_performance()

    assert "STRONG_BEARISH" in signal_perf
    assert "STRONG_BULLISH" in signal_perf
    assert all(0 <= perf["win_rate"] <= 1 for perf in signal_perf.values())
