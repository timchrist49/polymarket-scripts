"""
Trading bot subpackage for Polymarket.

This package contains all trading bot components:
- btc_price: BTC price data service
- sentiment: Market sentiment analysis
- technical: Technical indicators
- ai_decision: OpenAI decision engine
- risk: Risk management
"""

from polymarket.trading.btc_price import BTCPriceService
from polymarket.trading.sentiment import SentimentService
from polymarket.trading.technical import TechnicalAnalysis
from polymarket.trading.ai_decision import AIDecisionService
from polymarket.trading.risk import RiskManager

__all__ = [
    "BTCPriceService",
    "SentimentService",
    "TechnicalAnalysis",
    "AIDecisionService",
    "RiskManager",
]
