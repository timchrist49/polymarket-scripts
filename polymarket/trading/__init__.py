"""
Trading bot subpackage for Polymarket.

This package contains all trading bot components:
- btc_price: BTC price data service
- social_sentiment: Social sentiment analysis (Fear/Greed, CoinGecko)
- market_microstructure: Market microstructure analysis (Binance)
- signal_aggregator: Multi-signal aggregation with dynamic confidence
- technical: Technical indicators
- ai_decision: OpenAI decision engine
- risk: Risk management
"""

from polymarket.trading.btc_price import BTCPriceService
from polymarket.trading.social_sentiment import SocialSentimentService
from polymarket.trading.market_microstructure import MarketMicrostructureService
from polymarket.trading.signal_aggregator import SignalAggregator
from polymarket.trading.technical import TechnicalAnalysis
from polymarket.trading.ai_decision import AIDecisionService
from polymarket.trading.risk import RiskManager

__all__ = [
    "BTCPriceService",
    "SocialSentimentService",
    "MarketMicrostructureService",
    "SignalAggregator",
    "TechnicalAnalysis",
    "AIDecisionService",
    "RiskManager",
]
