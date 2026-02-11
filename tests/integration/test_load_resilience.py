# tests/integration/test_load_resilience.py
"""Load testing and error resilience tests."""

import pytest
import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, AsyncMock, patch
import tracemalloc
import time

from polymarket.performance.database import PerformanceDatabase
from polymarket.performance.tracker import PerformanceTracker
from polymarket.performance.reflection import ReflectionEngine
from polymarket.performance.adjuster import ParameterAdjuster
from polymarket.config import Settings
from polymarket.models import (
    TradingDecision, BTCPriceData, TechnicalIndicators,
    AggregatedSentiment, SocialSentiment, MarketSignals
)


@pytest.fixture
def mock_settings():
    """Mock settings."""
    settings = Mock(spec=Settings)
    settings.openai_api_key = "test-key"
    settings.openai_model = "gpt-4o-mini"
    settings.bot_confidence_threshold = 0.75
    settings.bot_max_position_dollars = 10.0
    settings.bot_max_exposure_percent = 0.50
    return settings


def create_sample_trade_data(index: int):
    """Create sample trade data for testing."""
    now = datetime.now()

    market = {
        "id": 1362391 + index,
        "question": f"Will BTC go up? {index}",
        "best_bid": 0.50,
        "best_ask": 0.51
    }

    decision = TradingDecision(
        action="YES" if index % 2 == 0 else "NO",
        confidence=0.8,
        reasoning="Test trade",
        token_id="test",
        position_size=Decimal("5.0"),
        stop_loss_threshold=0.40
    )

    btc_data = BTCPriceData(
        price=Decimal("66000.0"),
        timestamp=now - timedelta(minutes=index),
        source="binance",
        volume_24h=Decimal("1000.0")
    )

    technical = TechnicalIndicators(
        rsi=60.1,
        macd_value=1.74,
        macd_signal=1.50,
        macd_histogram=0.24,
        ema_short=66950.0,
        ema_long=66900.0,
        sma_50=66800.0,
        volume_change=5.0,
        price_velocity=2.0,
        trend="BULLISH"
    )

    social = SocialSentiment(
        score=-0.10,
        confidence=1.0,
        fear_greed=45,
        is_trending=False,
        vote_up_pct=48.0,
        vote_down_pct=52.0,
        signal_type="STRONG_BEARISH",
        sources_available=["fear_greed"],
        timestamp=now
    )

    market_signals = MarketSignals(
        score=-0.21,
        confidence=1.0,
        order_book_score=0.0,
        whale_score=-0.15,
        volume_score=-0.10,
        momentum_score=-0.20,
        order_book_bias="N/A",
        whale_direction="SELLING",
        whale_count=2,
        volume_ratio=0.9,
        momentum_direction="DOWN",
        signal_type="STRONG_BEARISH",
        timestamp=now
    )

    aggregated = AggregatedSentiment(
        social=social,
        market=market_signals,
        final_score=-0.17,
        final_confidence=1.0,
        agreement_multiplier=1.47,
        signal_type="STRONG_BEARISH",
        timestamp=now
    )

    return market, decision, btc_data, technical, aggregated


@pytest.mark.asyncio
async def test_high_volume_trade_logging():
    """Test logging 100+ trades without performance degradation."""
    db = PerformanceDatabase(":memory:")
    tracker = PerformanceTracker(db_path=":memory:")
    tracker.db = db  # Share database

    # Track memory
    tracemalloc.start()
    start_time = time.time()

    # Log 100 trades
    trade_count = 100
    for i in range(trade_count):
        market, decision, btc_data, technical, aggregated = create_sample_trade_data(i)

        trade_id = await tracker.log_decision(
            market=market,
            decision=decision,
            btc_data=btc_data,
            technical=technical,
            aggregated=aggregated,
            price_to_beat=Decimal("65826.14"),
            time_remaining_seconds=480,
            is_end_phase=False
        )

        assert trade_id > 0

    # Check performance
    elapsed = time.time() - start_time
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Verify all trades logged
    cursor = db.conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM trades")
    count = cursor.fetchone()[0]
    assert count == trade_count

    # Performance assertions
    avg_time_per_trade = elapsed / trade_count
    assert avg_time_per_trade < 0.1  # Less than 100ms per trade
    assert peak < 50 * 1024 * 1024  # Less than 50MB peak memory

    print(f"\nLoad Test Results:")
    print(f"  Trades logged: {trade_count}")
    print(f"  Total time: {elapsed:.2f}s")
    print(f"  Avg per trade: {avg_time_per_trade*1000:.1f}ms")
    print(f"  Peak memory: {peak / 1024 / 1024:.1f}MB")

    db.close()
    tracker.close()


@pytest.mark.asyncio
async def test_openai_failure_graceful_degradation(mock_settings):
    """Test reflection engine handles OpenAI failures gracefully."""
    db = PerformanceDatabase(":memory:")
    reflection = ReflectionEngine(db, mock_settings)

    # Mock OpenAI to raise exception
    with patch.object(reflection, '_call_openai', side_effect=Exception("API Error")):
        insights = await reflection.analyze_performance(
            trigger_type="test",
            trades_analyzed=10
        )

    # Should return empty insights instead of crashing
    assert insights is not None
    assert insights["insights"] == []
    assert insights["recommendations"] == []

    db.close()


@pytest.mark.asyncio
async def test_telegram_failure_doesnt_block_trading(mock_settings):
    """Test that Telegram failures don't block parameter adjustments."""
    from polymarket.telegram.bot import TelegramBot

    db = PerformanceDatabase(":memory:")

    # Create Telegram bot that always fails
    telegram = Mock(spec=TelegramBot)
    telegram._send_message = AsyncMock(side_effect=Exception("Telegram API Error"))

    adjuster = ParameterAdjuster(mock_settings, db=db, telegram=telegram)

    # Apply adjustment - should succeed despite Telegram failure
    from polymarket.performance.adjuster import AdjustmentTier

    result = await adjuster.apply_adjustment(
        parameter_name="bot_confidence_threshold",
        old_value=0.75,
        new_value=0.7275,
        reason="Test",
        tier=AdjustmentTier.TIER_1_AUTO
    )

    # Adjustment should succeed
    assert result is True
    assert mock_settings.bot_confidence_threshold == 0.7275

    db.close()


@pytest.mark.asyncio
async def test_database_concurrent_access():
    """Test multiple concurrent database operations."""
    db = PerformanceDatabase(":memory:")

    async def log_trades(start_idx: int, count: int):
        """Log trades concurrently."""
        for i in range(start_idx, start_idx + count):
            trade_data = {
                "timestamp": datetime.now(),
                "market_slug": f"test-{i}",
                "action": "YES",
                "confidence": 0.8,
                "position_size": 5.0,
                "btc_price": 66000.0,
            }
            db.log_trade(trade_data)

    # Run 5 concurrent tasks
    tasks = [
        log_trades(i * 10, 10)
        for i in range(5)
    ]

    await asyncio.gather(*tasks)

    # Verify all trades logged
    cursor = db.conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM trades")
    count = cursor.fetchone()[0]
    assert count == 50  # 5 tasks * 10 trades each

    db.close()


@pytest.mark.asyncio
async def test_archival_under_load():
    """Test archival system handles large datasets."""
    from polymarket.performance.archival import ArchivalManager

    db = PerformanceDatabase(":memory:")

    # Add 1000 old trades
    now = datetime.now()
    for i in range(1000):
        trade_data = {
            "timestamp": now - timedelta(days=60 + i % 100),
            "market_slug": f"old-{i}",
            "action": "YES" if i % 2 == 0 else "NO",
            "confidence": 0.8,
            "position_size": 5.0,
            "btc_price": 66000.0,
        }
        db.log_trade(trade_data)

    # Archive old trades
    archival = ArchivalManager(db, archive_dir="/tmp/test_archives")
    start_time = time.time()

    archived_count = archival.archive_old_trades(days_threshold=30)

    elapsed = time.time() - start_time

    assert archived_count > 0
    assert elapsed < 5.0  # Should complete in < 5 seconds

    print(f"\nArchival Performance:")
    print(f"  Trades archived: {archived_count}")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Rate: {archived_count/elapsed:.0f} trades/sec")

    db.close()


@pytest.mark.asyncio
async def test_reflection_with_invalid_data(mock_settings):
    """Test metrics calculator handles missing/corrupted data gracefully."""
    db = PerformanceDatabase(":memory:")

    # Add trade with missing outcome (not yet resolved)
    cursor = db.conn.cursor()
    cursor.execute("""
        INSERT INTO trades (
            timestamp, market_slug, action, confidence, position_size,
            btc_price, rsi, trend
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (datetime.now(), "test", "YES", 0.8, 5.0, 66000.0, 60.0, "BULLISH"))
    db.conn.commit()

    # Metrics should handle gracefully (no outcomes yet)
    from polymarket.performance.metrics import MetricsCalculator
    metrics = MetricsCalculator(db)

    # Should not crash even with no resolved trades
    win_rate = metrics.calculate_win_rate()
    assert isinstance(win_rate, float)
    assert win_rate == 0.0  # No resolved trades yet

    # Total profit should also work
    total_profit = metrics.calculate_total_profit()
    assert isinstance(total_profit, float)
    assert total_profit == 0.0

    # Test with extreme values
    cursor.execute("""
        INSERT INTO trades (
            timestamp, market_slug, action, confidence, position_size,
            btc_price, rsi, trend, actual_outcome, profit_loss, is_win
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (datetime.now(), "extreme", "YES", 1.0, 1000000.0, 1000000.0, 100.0, "BULLISH", "YES", 999999.0, 1))
    db.conn.commit()

    # Should handle extreme values gracefully
    win_rate_after = metrics.calculate_win_rate()
    assert win_rate_after == 1.0  # 1 win out of 1 resolved trade

    total_profit_after = metrics.calculate_total_profit()
    assert total_profit_after == 999999.0  # Extreme value handled

    db.close()
