# tests/integration/test_e2e_self_reflection.py
"""End-to-end integration test for self-reflection system."""

import pytest
import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, AsyncMock, patch

from polymarket.performance.database import PerformanceDatabase
from polymarket.performance.tracker import PerformanceTracker
from polymarket.performance.metrics import MetricsCalculator
from polymarket.performance.reflection import ReflectionEngine
from polymarket.performance.adjuster import ParameterAdjuster, AdjustmentTier
from polymarket.telegram.bot import TelegramBot
from polymarket.config import Settings
from polymarket.models import (
    TradingDecision, BTCPriceData, TechnicalIndicators,
    AggregatedSentiment, SocialSentiment, MarketSignals, Market
)


@pytest.fixture
def mock_settings():
    """Mock settings for testing."""
    settings = Mock(spec=Settings)
    settings.openai_api_key = "test-key"
    settings.openai_model = "gpt-4o-mini"
    settings.bot_confidence_threshold = 0.75
    settings.bot_max_position_dollars = 10.0
    settings.bot_max_exposure_percent = 0.50
    settings.telegram_enabled = False
    settings.telegram_bot_token = None
    settings.telegram_chat_id = None
    settings.emergency_pause_enabled = False
    return settings


@pytest.fixture
def integration_components(mock_settings):
    """Create all system components."""
    # Database (shared instance)
    db = PerformanceDatabase(":memory:")

    # Tracker (using shared database)
    tracker = PerformanceTracker(db=db)

    # Telegram (mocked)
    telegram = Mock(spec=TelegramBot)
    telegram._send_message = AsyncMock()
    telegram.request_approval = AsyncMock(return_value=True)

    # Reflection engine
    reflection = ReflectionEngine(db, mock_settings)

    # Parameter adjuster
    adjuster = ParameterAdjuster(mock_settings, db=db, telegram=telegram)

    # Metrics calculator
    metrics = MetricsCalculator(db)

    yield {
        "db": db,
        "tracker": tracker,
        "telegram": telegram,
        "reflection": reflection,
        "adjuster": adjuster,
        "metrics": metrics,
        "settings": mock_settings
    }

    # Cleanup
    tracker.close()
    db.close()


@pytest.mark.asyncio
async def test_complete_workflow_tier1(integration_components):
    """Test complete workflow: trade → log → reflect → tier1 adjust."""
    components = integration_components

    # Step 1: Simulate trades being logged
    now = datetime.now()
    market_slugs = []
    for i in range(10):
        # Create sample trade data
        question = f"Will BTC go up? {i}"
        # Generate slug the same way tracker does
        market_slug = question.lower().replace(" ", "-").replace("?", "")[:50]
        market_slugs.append(market_slug)

        market = Market(
            id=str(1362391 + i),
            condition_id="test",
            question=question,
            slug=market_slug,
            best_bid=0.50,
            best_ask=0.51
        )

        decision = TradingDecision(
            action="YES" if i % 2 == 0 else "NO",
            confidence=0.8,
            reasoning="Test trade",
            token_id="test",
            position_size=Decimal("5.0"),
            stop_loss_threshold=0.40
        )

        btc_data = BTCPriceData(
            price=Decimal("66000.0"),
            timestamp=now - timedelta(minutes=i*15),
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

        # Log trade
        trade_id = await components["tracker"].log_decision(
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

        # Update outcomes for first 8 trades
        if i < 8:
            is_win = (i % 3 != 0)
            components["db"].update_outcome(
                market_slug=market_slugs[i],
                actual_outcome="UP" if is_win else "DOWN",
                profit_loss=4.0 if is_win else -5.0
            )

    # Step 2: Verify metrics calculation
    metrics = components["metrics"]
    win_rate = metrics.calculate_win_rate()
    assert 0.0 <= win_rate <= 1.0

    total_profit = metrics.calculate_total_profit()
    assert isinstance(total_profit, float)

    signal_perf = metrics.calculate_signal_performance()
    assert "STRONG_BEARISH" in signal_perf

    # Step 3: Run reflection (mock OpenAI)
    mock_insights = {
        "insights": ["Test insight 1", "Test insight 2"],
        "patterns": {
            "winning": ["Pattern 1"],
            "losing": ["Pattern 2"]
        },
        "recommendations": [
            {
                "parameter": "bot_confidence_threshold",
                "current": 0.75,
                "recommended": 0.7275,  # 3% decrease (Tier 1)
                "reason": "Win rate is good, can be slightly more aggressive",
                "tier": 1,
                "expected_impact": "Slightly increased trade frequency"
            }
        ]
    }

    with patch.object(components["reflection"], '_call_openai',
                     new_callable=AsyncMock, return_value=mock_insights):
        insights = await components["reflection"].analyze_performance(
            trigger_type="10_trades",
            trades_analyzed=10
        )

    assert insights is not None
    assert len(insights["recommendations"]) > 0

    # Step 4: Apply Tier 1 adjustment
    recommendation = insights["recommendations"][0]

    result = await components["adjuster"].apply_adjustment(
        parameter_name=recommendation["parameter"],
        old_value=recommendation["current"],
        new_value=recommendation["recommended"],
        reason=recommendation["reason"],
        tier=AdjustmentTier.TIER_1_AUTO
    )

    assert result is True
    assert components["settings"].bot_confidence_threshold == 0.7275

    # Step 5: Verify database consistency
    cursor = components["db"].conn.cursor()

    # Check trades logged
    cursor.execute("SELECT COUNT(*) FROM trades")
    trade_count = cursor.fetchone()[0]
    assert trade_count == 10

    # Check reflections logged
    cursor.execute("SELECT COUNT(*) FROM reflections")
    reflection_count = cursor.fetchone()[0]
    assert reflection_count == 1

    # Check parameter history logged
    cursor.execute("SELECT COUNT(*) FROM parameter_history")
    param_count = cursor.fetchone()[0]
    assert param_count == 1

    # Verify parameter history details
    cursor.execute("SELECT * FROM parameter_history ORDER BY timestamp DESC LIMIT 1")
    row = cursor.fetchone()
    assert row['parameter_name'] == 'bot_confidence_threshold'
    assert row['old_value'] == 0.75
    assert row['new_value'] == 0.7275
    assert row['approval_method'] == 'tier_1_auto'


@pytest.mark.asyncio
async def test_complete_workflow_tier2_approved(integration_components):
    """Test workflow with Tier 2 approval."""
    components = integration_components

    # Mock OpenAI to return Tier 2 recommendation
    mock_insights = {
        "insights": ["Significant performance issue detected"],
        "patterns": {"winning": [], "losing": ["Too conservative"]},
        "recommendations": [{
            "parameter": "bot_confidence_threshold",
            "current": 0.75,
            "recommended": 0.675,  # 10% decrease (Tier 2)
            "reason": "Need more aggressive trading",
            "tier": 2,
            "expected_impact": "Increased trade volume"
        }]
    }

    with patch.object(components["reflection"], '_call_openai',
                     new_callable=AsyncMock, return_value=mock_insights):
        insights = await components["reflection"].analyze_performance(
            trigger_type="3_losses",
            trades_analyzed=3
        )

    # Apply Tier 2 adjustment (should request approval)
    recommendation = insights["recommendations"][0]

    result = await components["adjuster"].apply_adjustment(
        parameter_name=recommendation["parameter"],
        old_value=recommendation["current"],
        new_value=recommendation["recommended"],
        reason=recommendation["reason"],
        tier=AdjustmentTier.TIER_2_APPROVAL
    )

    # Should succeed because telegram.request_approval returns True
    assert result is True
    assert components["telegram"].request_approval.called


@pytest.mark.asyncio
async def test_complete_workflow_tier3_rejected(integration_components):
    """Test workflow with Tier 3 emergency pause."""
    components = integration_components

    # Mock OpenAI to return dangerous Tier 3 recommendation
    mock_insights = {
        "insights": ["Critical performance failure"],
        "patterns": {"winning": [], "losing": ["Major issue"]},
        "recommendations": [{
            "parameter": "bot_confidence_threshold",
            "current": 0.75,
            "recommended": 0.525,  # 30% decrease (Tier 3)
            "reason": "Emergency adjustment needed",
            "tier": 3,
            "expected_impact": "Major change"
        }]
    }

    with patch.object(components["reflection"], '_call_openai',
                     new_callable=AsyncMock, return_value=mock_insights):
        insights = await components["reflection"].analyze_performance(
            trigger_type="critical",
            trades_analyzed=5
        )

    # Apply Tier 3 adjustment (should be rejected and trigger pause)
    recommendation = insights["recommendations"][0]

    result = await components["adjuster"].apply_adjustment(
        parameter_name=recommendation["parameter"],
        old_value=recommendation["current"],
        new_value=recommendation["recommended"],
        reason=recommendation["reason"],
        tier=AdjustmentTier.TIER_3_PAUSE
    )

    # Should be rejected
    assert result is False

    # Emergency pause file should exist
    from pathlib import Path
    pause_file = Path("/root/polymarket-scripts/.emergency_pause")
    assert pause_file.exists(), "Emergency pause file should be created"

    # Clean up pause file
    pause_file.unlink()

    # Should log to database with emergency flag
    cursor = components["db"].conn.cursor()
    cursor.execute("SELECT * FROM parameter_history ORDER BY timestamp DESC LIMIT 1")
    row = cursor.fetchone()
    assert row['approval_method'] == 'tier_3_emergency_pause'
