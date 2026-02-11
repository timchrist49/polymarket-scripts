# tests/test_performance_reflection.py
import pytest
from unittest.mock import Mock, AsyncMock, patch
from polymarket.performance.reflection import ReflectionEngine
from polymarket.performance.database import PerformanceDatabase
from polymarket.config import Settings

@pytest.fixture
def mock_settings():
    """Mock settings with OpenAI key."""
    settings = Mock(spec=Settings)
    settings.openai_api_key = "test-key"
    settings.openai_model = "gpt-4o-mini"
    settings.bot_confidence_threshold = 0.75
    settings.bot_max_position_dollars = 10.0
    settings.bot_max_exposure_percent = 0.50
    return settings

@pytest.fixture
def db_with_trades():
    """Database with sample trades for analysis."""
    db = PerformanceDatabase(":memory:")
    # Add some trades for testing
    from datetime import datetime
    trade_data = {
        "timestamp": datetime(2026, 2, 11, 10, 0, 0),
        "market_slug": "test-market",
        "action": "YES",
        "confidence": 0.8,
        "position_size": 5.0,
        "btc_price": 66000.0,
    }
    db.log_trade(trade_data)
    return db

@pytest.mark.asyncio
async def test_generate_reflection_prompt(mock_settings, db_with_trades):
    """Test reflection prompt generation."""
    engine = ReflectionEngine(db_with_trades, mock_settings)

    prompt = await engine._generate_prompt(trade_count=10)

    assert "trading performance" in prompt.lower()
    assert "win rate" in prompt.lower()
    assert "recommendations" in prompt.lower()
    assert "0.75" in prompt  # confidence threshold

@pytest.mark.asyncio
async def test_analyze_performance(mock_settings, db_with_trades):
    """Test full reflection analysis."""
    engine = ReflectionEngine(db_with_trades, mock_settings)

    # Mock OpenAI response
    mock_response = {
        "insights": ["Test insight 1", "Test insight 2"],
        "patterns": {
            "winning": ["Pattern 1"],
            "losing": ["Pattern 2"]
        },
        "recommendations": [
            {
                "parameter": "bot_confidence_threshold",
                "current": 0.75,
                "recommended": 0.70,
                "reason": "Test reason",
                "tier": 2,
                "expected_impact": "Test impact"
            }
        ]
    }

    with patch.object(engine, '_call_openai', new_callable=AsyncMock) as mock_openai:
        mock_openai.return_value = mock_response

        insights = await engine.analyze_performance(
            trigger_type="10_trades",
            trades_analyzed=10
        )

        assert insights is not None
        assert "insights" in insights
        assert len(insights["insights"]) == 2
        assert len(insights["recommendations"]) == 1
