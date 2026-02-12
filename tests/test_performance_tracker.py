# tests/test_performance_tracker.py
import pytest
from datetime import datetime
from decimal import Decimal
from polymarket.performance.tracker import PerformanceTracker
from polymarket.models import TradingDecision, BTCPriceData, TechnicalIndicators, AggregatedSentiment, SocialSentiment, MarketSignals, Market

@pytest.fixture
def tracker():
    """Create performance tracker with in-memory DB."""
    tracker = PerformanceTracker(db_path=":memory:")
    yield tracker
    tracker.close()

@pytest.fixture
def sample_market():
    """Sample market data."""
    return Market(
        id="1362391",
        condition_id="test",
        question="Will BTC go up?",
        outcomes=["Up", "Down"],
        best_bid=0.50,
        best_ask=0.51,
        active=True
    )

@pytest.fixture
def sample_decision():
    """Sample trading decision."""
    return TradingDecision(
        action="NO",
        confidence=1.0,
        reasoning="Bearish signals aligned",
        token_id="test",
        position_size=Decimal("5.0"),
        stop_loss_threshold=0.40
    )

@pytest.fixture
def sample_btc_data():
    """Sample BTC price data."""
    return BTCPriceData(
        price=Decimal("66940.0"),
        timestamp=datetime(2026, 2, 11, 10, 30, 0),
        source="binance",
        volume_24h=Decimal("1000.0")
    )

@pytest.fixture
def sample_technical():
    """Sample technical indicators."""
    return TechnicalIndicators(
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

@pytest.fixture
def sample_aggregated():
    """Sample aggregated sentiment."""
    social = SocialSentiment(
        score=-0.10,
        confidence=1.0,
        fear_greed=45,
        is_trending=False,
        vote_up_pct=48.0,
        vote_down_pct=52.0,
        signal_type="STRONG_BEARISH",
        sources_available=["fear_greed", "votes"],
        timestamp=datetime(2026, 2, 11, 10, 30, 0)
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
        timestamp=datetime(2026, 2, 11, 10, 30, 0)
    )

    return AggregatedSentiment(
        social=social,
        market=market_signals,
        final_score=-0.17,
        final_confidence=1.0,
        agreement_multiplier=1.47,
        signal_type="STRONG_BEARISH",
        timestamp=datetime(2026, 2, 11, 10, 30, 0)
    )

@pytest.mark.asyncio
async def test_log_decision(
    tracker,
    sample_market,
    sample_decision,
    sample_btc_data,
    sample_technical,
    sample_aggregated
):
    """Test logging a trading decision."""
    trade_id = await tracker.log_decision(
        market=sample_market,
        decision=sample_decision,
        btc_data=sample_btc_data,
        technical=sample_technical,
        aggregated=sample_aggregated,
        price_to_beat=Decimal("66826.14"),
        time_remaining_seconds=480,
        is_end_phase=False
    )

    assert trade_id > 0

    # Verify data in database
    cursor = tracker.db.conn.cursor()
    cursor.execute("SELECT * FROM trades WHERE id = ?", (trade_id,))
    row = cursor.fetchone()

    assert row is not None
    assert row['action'] == 'NO'
    assert row['confidence'] == 1.0
    assert row['market_id'] == 1362391
