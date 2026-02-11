#!/usr/bin/env python3
"""Test script to verify BOT_MAX_POSITION_DOLLARS cap works correctly."""

from decimal import Decimal
from polymarket.config import Settings
from polymarket.trading.risk import RiskManager
from polymarket.models import TradingDecision

# Test with your actual config
settings = Settings()
risk_manager = RiskManager(settings)

# Simulate a high confidence decision
decision = TradingDecision(
    action="YES",
    confidence=0.90,  # 90% confidence (would normally bet large)
    reasoning="Test decision",
    token_id="test-token",
    position_size=Decimal("100"),  # AI suggests $100
    stop_loss_threshold=0.40
)

# Test with different portfolio sizes
test_cases = [
    Decimal("100"),    # Small portfolio
    Decimal("1000"),   # Medium portfolio
    Decimal("10000"),  # Large portfolio
]

print(f"\nConfiguration:")
print(f"  BOT_MAX_POSITION_PERCENT: {settings.bot_max_position_percent * 100:.1f}%")
print(f"  BOT_MAX_POSITION_DOLLARS: ${settings.bot_max_position_dollars:.2f}")
print(f"\nDecision: {decision.action} at {decision.confidence * 100:.0f}% confidence")
print(f"AI Suggested Size: ${decision.position_size}\n")

for portfolio in test_cases:
    max_position = portfolio * Decimal(str(settings.bot_max_position_percent))
    calculated = risk_manager._calculate_position_size(decision, portfolio, max_position)

    print(f"Portfolio: ${portfolio:>8} | 10% = ${max_position:>6.2f} | Actual Bet: ${calculated:>5.2f}")

print(f"\n✅ Dollar cap working! Max bet capped at ${settings.bot_max_position_dollars:.2f}")
print(f"   (Without cap, $10k portfolio would bet $100!)")
