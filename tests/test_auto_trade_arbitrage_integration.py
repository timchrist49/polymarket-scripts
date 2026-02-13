"""Integration tests for arbitrage system in auto_trade.py."""
import pytest
from unittest.mock import Mock, patch
from datetime import datetime, timezone
from decimal import Decimal


def test_arbitrage_imports_successful():
    """Test that arbitrage components can be imported in auto_trade."""
    # This test verifies the integration at the import level
    from scripts import auto_trade

    # Verify the imports exist in the module
    assert hasattr(auto_trade, 'ProbabilityCalculator'), \
        "auto_trade should import ProbabilityCalculator"
    assert hasattr(auto_trade, 'ArbitrageDetector'), \
        "auto_trade should import ArbitrageDetector"
    assert hasattr(auto_trade, 'SmartOrderExecutor'), \
        "auto_trade should import SmartOrderExecutor"


@pytest.mark.asyncio
async def test_arbitrage_code_paths_exist():
    """Test that arbitrage code paths are integrated into _process_market."""
    from scripts.auto_trade import AutoTrader
    import inspect

    # Get the source code of _process_market
    source = inspect.getsource(AutoTrader._process_market)

    # Verify arbitrage-related code exists
    assert 'ProbabilityCalculator' in source, \
        "_process_market should instantiate ProbabilityCalculator"
    assert 'calculate_directional_probability' in source, \
        "_process_market should call calculate_directional_probability"
    assert 'ArbitrageDetector' in source, \
        "_process_market should instantiate ArbitrageDetector"
    assert 'detect_arbitrage' in source, \
        "_process_market should call detect_arbitrage"
    assert 'arbitrage_opportunity=' in source, \
        "_process_market should pass arbitrage_opportunity to AI"

    # Get the source code of _execute_trade
    exec_source = inspect.getsource(AutoTrader._execute_trade)

    # Verify SmartOrderExecutor is used
    assert 'SmartOrderExecutor' in exec_source, \
        "_execute_trade should instantiate SmartOrderExecutor"
    assert 'execute_smart_order' in exec_source, \
        "_execute_trade should call execute_smart_order"
    assert 'arbitrage_opportunity' in exec_source, \
        "_execute_trade should handle arbitrage_opportunity parameter"


@pytest.mark.asyncio
async def test_arbitrage_data_flow():
    """Test that arbitrage data flows through the decision pipeline."""
    from scripts.auto_trade import AutoTrader
    from polymarket.config import Settings
    from unittest.mock import AsyncMock, MagicMock

    # Create mock settings
    settings = Mock()
    settings.openai_api_key = "test-key"
    settings.openai_model = "gpt-4o-mini"
    settings.bot_confidence_threshold = 0.75
    settings.bot_max_position_dollars = 10.0
    settings.bot_log_decisions = True
    settings.mode = "read_only"  # Avoid actual trading
    settings.btc_cache_stale_max_age = 30
    settings.btc_cache_size = 1000
    settings.social_sentiment_enabled = False
    settings.telegram_enabled = False
    settings.btc_fetch_max_retries = 3
    settings.btc_fetch_retry_delay = 1.0
    settings.btc_fetch_backoff_multiplier = 2.0

    trader = AutoTrader(settings, interval=180)

    # Mock the services
    trader.btc_service = Mock()
    trader.btc_service.get_price_at = AsyncMock(side_effect=[66000.0, 65900.0])
    trader.btc_service.calculate_15min_volatility = Mock(return_value=0.005)

    trader.ai_service = Mock()
    trader.ai_service.make_decision = AsyncMock(return_value=Mock(
        action="HOLD",
        confidence=0.60,
        reasoning="Test",
        position_size=Decimal("0.0")
    ))

    trader.client = Mock()
    trader.client.get_token_ids = Mock(return_value=["token-123", "token-456"])
    trader.client.get_orderbook = Mock(return_value=MagicMock())

    trader.market_tracker = Mock()
    trader.performance_tracker = Mock()
    trader.performance_tracker.log_decision = AsyncMock(return_value=1)

    # Mock market
    mock_market = Mock()
    mock_market.id = "test-market"
    mock_market.slug = "btc-updown-15m-1234567890"
    mock_market.best_ask = 0.56
    mock_market.best_bid = 0.54
    mock_market.active = True
    mock_market.question = "Will BTC be up?"
    mock_market.outcomes = ["Up", "Down"]

    mock_btc_data = Mock()
    mock_btc_data.price = Decimal("66200.0")

    mock_indicators = Mock()
    mock_aggregated = Mock()
    mock_aggregated.final_confidence = 0.75

    # Patch arbitrage components to track calls
    with patch('scripts.auto_trade.ProbabilityCalculator') as MockProbCalc, \
         patch('scripts.auto_trade.ArbitrageDetector') as MockArbDet, \
         patch('polymarket.trading.orderbook_analyzer.OrderbookAnalyzer') as MockOrderbookAnalyzer:

        # Setup mocks
        mock_prob_calc = MockProbCalc.return_value
        mock_prob_calc.calculate_directional_probability = Mock(return_value=0.70)

        mock_arb_det = MockArbDet.return_value
        mock_arb_opportunity = Mock(
            edge_percentage=0.15,
            recommended_action="BUY_YES",
            urgency="HIGH",
            confidence_boost=0.20
        )
        mock_arb_det.detect_arbitrage = Mock(return_value=mock_arb_opportunity)

        mock_orderbook_analyzer = MockOrderbookAnalyzer.return_value
        mock_orderbook_analyzer.analyze_orderbook = Mock(return_value=Mock(
            spread_bps=100,
            liquidity_score=0.8,
            order_imbalance=0.1,
            can_fill_order=True
        ))

        # Call _process_market at a time during trading hours (11 AM UTC)
        cycle_time = datetime(2026, 2, 13, 11, 0, 0, tzinfo=timezone.utc)

        await trader._process_market(
            market=mock_market,
            btc_data=mock_btc_data,
            indicators=mock_indicators,
            aggregated_sentiment=mock_aggregated,
            portfolio_value=Decimal("1000"),
            btc_momentum=None,
            cycle_start_time=cycle_time,
            volume_data=None,
            timeframe_analysis=None,
            regime=None
        )

        # Verify probability calculator was called
        assert mock_prob_calc.calculate_directional_probability.called, \
            "ProbabilityCalculator.calculate_directional_probability should be called"

        # Verify arbitrage detector was called
        assert mock_arb_det.detect_arbitrage.called, \
            "ArbitrageDetector.detect_arbitrage should be called"

        # Verify AI received arbitrage context
        ai_call_kwargs = trader.ai_service.make_decision.call_args.kwargs
        assert 'arbitrage_opportunity' in ai_call_kwargs, \
            "AI decision should receive arbitrage_opportunity"
        assert ai_call_kwargs['arbitrage_opportunity'] == mock_arb_opportunity, \
            "AI should receive the actual arbitrage opportunity object"
