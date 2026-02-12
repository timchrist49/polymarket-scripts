# Odds-Adjusted Position Sizing Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add odds-based scaling to RiskManager to prevent large bets on low-probability trades (<0.50 odds).

**Architecture:** Modify `RiskManager._calculate_position_size()` to accept odds parameter and apply linear scaling from 100% at 0.50 odds down to 50% at 0.25 odds. Reject bets below 0.25 odds entirely.

**Tech Stack:** Python 3.11+, pytest, structlog

---

## Task 1: Write Test for Odds Multiplier Calculation

**Files:**
- Create: `tests/test_risk.py`

**Step 1: Write the failing test**

Create test file with odds multiplier test cases:

```python
"""Tests for RiskManager odds-adjusted position sizing."""

import pytest
from decimal import Decimal
from polymarket.trading.risk import RiskManager
from polymarket.config import Settings
from polymarket.models import TradingDecision


class TestOddsMultiplier:
    """Test odds-based position scaling."""

    @pytest.fixture
    def risk_manager(self):
        """Create RiskManager with test settings."""
        settings = Settings()
        return RiskManager(settings)

    def test_high_odds_no_scaling(self, risk_manager):
        """Odds >= 0.50 should have 1.0x multiplier (no scaling)."""
        assert risk_manager._calculate_odds_multiplier(0.83) == 1.0
        assert risk_manager._calculate_odds_multiplier(0.70) == 1.0
        assert risk_manager._calculate_odds_multiplier(0.50) == 1.0

    def test_low_odds_linear_scaling(self, risk_manager):
        """Odds < 0.50 should scale linearly from 1.0 to 0.5."""
        # 0.40 odds = 60% between 0.25 and 0.50
        # multiplier = 0.5 + 0.6 * 0.5 = 0.8
        assert risk_manager._calculate_odds_multiplier(0.40) == pytest.approx(0.8, abs=0.01)

        # 0.31 odds = 24% between 0.25 and 0.50
        # multiplier = 0.5 + 0.24 * 0.5 = 0.62
        assert risk_manager._calculate_odds_multiplier(0.31) == pytest.approx(0.62, abs=0.01)

        # 0.25 odds = 0% (minimum)
        # multiplier = 0.5
        assert risk_manager._calculate_odds_multiplier(0.25) == 0.5

    def test_below_minimum_odds_rejected(self, risk_manager):
        """Odds < 0.25 should return 0.0 (bet rejected)."""
        assert risk_manager._calculate_odds_multiplier(0.24) == 0.0
        assert risk_manager._calculate_odds_multiplier(0.20) == 0.0
        assert risk_manager._calculate_odds_multiplier(0.10) == 0.0
```

**Step 2: Run test to verify it fails**

```bash
cd /root/polymarket-scripts
pytest tests/test_risk.py::TestOddsMultiplier -v
```

**Expected output:**
```
AttributeError: 'RiskManager' object has no attribute '_calculate_odds_multiplier'
```

**Step 3: Commit the failing test**

```bash
git add tests/test_risk.py
git commit -m "test: add odds multiplier tests (failing)

Tests for linear odds scaling from 1.0x at 0.50 odds down to 0.5x at 0.25 odds.
Rejects bets below 0.25 odds with 0.0 multiplier.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 2: Implement Odds Multiplier Method

**Files:**
- Modify: `polymarket/trading/risk.py:142` (add method after `_calculate_position_size`)

**Step 1: Add the odds multiplier method**

Add this method after line 141 (after `_calculate_position_size`):

```python
    def _calculate_odds_multiplier(self, odds: float) -> float:
        """
        Calculate position size multiplier based on odds.

        Scales down position size for low-odds bets to limit risk.

        Logic:
        - odds >= 0.50: No scaling (1.0x multiplier)
        - 0.25 <= odds < 0.50: Linear scale from 0.5x to 1.0x
        - odds < 0.25: Reject bet (0.0x multiplier)

        Examples:
            0.83 odds → 1.0x (no reduction)
            0.50 odds → 1.0x (breakeven)
            0.40 odds → 0.8x (20% reduction)
            0.31 odds → 0.62x (38% reduction)
            0.25 odds → 0.5x (50% reduction, minimum)
            0.20 odds → 0.0x (rejected)

        Args:
            odds: Market odds (price) for the side being bet (0.0-1.0)

        Returns:
            Multiplier to apply to position size (0.0-1.0)
        """
        MINIMUM_ODDS = 0.25
        SCALE_THRESHOLD = 0.50

        # Reject bets below minimum threshold
        if odds < MINIMUM_ODDS:
            return 0.0

        # No scaling needed for odds above threshold
        if odds >= SCALE_THRESHOLD:
            return 1.0

        # Linear interpolation between 0.5x and 1.0x
        # Formula: 0.5 + (odds - 0.25) / (0.50 - 0.25) * 0.5
        # At odds=0.50: 0.5 + 0.25/0.25 * 0.5 = 1.0
        # At odds=0.25: 0.5 + 0.00/0.25 * 0.5 = 0.5
        multiplier = 0.5 + ((odds - MINIMUM_ODDS) / (SCALE_THRESHOLD - MINIMUM_ODDS)) * 0.5

        return multiplier
```

**Step 2: Run tests to verify they pass**

```bash
pytest tests/test_risk.py::TestOddsMultiplier -v
```

**Expected output:**
```
tests/test_risk.py::TestOddsMultiplier::test_high_odds_no_scaling PASSED
tests/test_risk.py::TestOddsMultiplier::test_low_odds_linear_scaling PASSED
tests/test_risk.py::TestOddsMultiplier::test_below_minimum_odds_rejected PASSED
```

**Step 3: Commit the implementation**

```bash
git add polymarket/trading/risk.py
git commit -m "feat: add odds multiplier for position sizing

Implement linear odds-based scaling:
- Odds >= 0.50: No scaling (1.0x)
- Odds 0.25-0.50: Scale from 0.5x to 1.0x
- Odds < 0.25: Reject (0.0x)

Prevents large bets on low-probability trades.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 3: Add Helper Method to Extract Odds from Market

**Files:**
- Modify: `polymarket/trading/risk.py:27` (add method after `__init__`)

**Step 1: Write the failing test**

Add to `tests/test_risk.py`:

```python
class TestOddsExtraction:
    """Test extracting odds from market data."""

    @pytest.fixture
    def risk_manager(self):
        """Create RiskManager with test settings."""
        settings = Settings()
        return RiskManager(settings)

    def test_extract_yes_odds(self, risk_manager):
        """Extract YES odds from market data."""
        market = {"yes_price": 0.31, "no_price": 0.69}
        odds = risk_manager._extract_odds_for_action("YES", market)
        assert odds == 0.31

    def test_extract_no_odds(self, risk_manager):
        """Extract NO odds from market data."""
        market = {"yes_price": 0.17, "no_price": 0.83}
        odds = risk_manager._extract_odds_for_action("NO", market)
        assert odds == 0.83

    def test_extract_odds_missing_data(self, risk_manager):
        """Default to 0.50 if market data missing."""
        market = {}
        odds = risk_manager._extract_odds_for_action("YES", market)
        assert odds == 0.50

    def test_extract_odds_hold_action(self, risk_manager):
        """HOLD action defaults to 0.50."""
        market = {"yes_price": 0.31, "no_price": 0.69}
        odds = risk_manager._extract_odds_for_action("HOLD", market)
        assert odds == 0.50
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_risk.py::TestOddsExtraction -v
```

**Expected output:**
```
AttributeError: 'RiskManager' object has no attribute '_extract_odds_for_action'
```

**Step 3: Implement the odds extraction method**

Add this method after line 26 (after `__init__`):

```python
    def _extract_odds_for_action(self, action: str, market: dict) -> float:
        """
        Extract the market odds (price) for the side being bet.

        Args:
            action: Trading action - "YES", "NO", or "HOLD"
            market: Market data dict with yes_price and no_price

        Returns:
            Odds (price) for the side being bet (0.0-1.0)
            Defaults to 0.50 if data missing
        """
        if action == "YES":
            return float(market.get("yes_price", 0.50))
        elif action == "NO":
            return float(market.get("no_price", 0.50))
        else:
            # HOLD or invalid action defaults to neutral odds
            return 0.50
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_risk.py::TestOddsExtraction -v
```

**Expected output:**
```
tests/test_risk.py::TestOddsExtraction::test_extract_yes_odds PASSED
tests/test_risk.py::TestOddsExtraction::test_extract_no_odds PASSED
tests/test_risk.py::TestOddsExtraction::test_extract_odds_missing_data PASSED
tests/test_risk.py::TestOddsExtraction::test_extract_odds_hold_action PASSED
```

**Step 5: Commit**

```bash
git add tests/test_risk.py polymarket/trading/risk.py
git commit -m "feat: add odds extraction from market data

Extract yes_price or no_price based on trading action.
Defaults to 0.50 if market data missing.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 4: Update validate_decision to Extract and Pass Odds

**Files:**
- Modify: `polymarket/trading/risk.py:28-106` (update `validate_decision` method)

**Step 1: Write integration test**

Add to `tests/test_risk.py`:

```python
class TestPositionSizingWithOdds:
    """Integration test for position sizing with odds adjustment."""

    @pytest.fixture
    def risk_manager(self):
        """Create RiskManager with test settings."""
        settings = Settings()
        return RiskManager(settings)

    @pytest.mark.asyncio
    async def test_low_odds_bet_scaled_down(self, risk_manager):
        """Low-odds bets should be scaled down."""
        decision = TradingDecision(
            action="YES",
            confidence=0.85,
            position_size=9.56,
            reasoning="Test",
            token_id="test_token"
        )
        market = {
            "yes_price": 0.31,
            "no_price": 0.69,
            "active": True
        }

        result = await risk_manager.validate_decision(
            decision=decision,
            portfolio_value=Decimal("100.0"),
            market=market,
            open_positions=[]
        )

        assert result.approved is True
        # 0.31 odds → 0.62x multiplier
        # Original ~9.56 → scaled to ~5.93
        assert result.adjusted_position < Decimal("6.00")
        assert result.adjusted_position > Decimal("5.50")

    @pytest.mark.asyncio
    async def test_high_odds_bet_unchanged(self, risk_manager):
        """High-odds bets should not be scaled."""
        decision = TradingDecision(
            action="NO",
            confidence=0.73,
            position_size=5.0,
            reasoning="Test",
            token_id="test_token"
        )
        market = {
            "yes_price": 0.17,
            "no_price": 0.83,
            "active": True
        }

        result = await risk_manager.validate_decision(
            decision=decision,
            portfolio_value=Decimal("100.0"),
            market=market,
            open_positions=[]
        )

        assert result.approved is True
        # 0.83 odds → 1.0x multiplier (no scaling)
        assert result.adjusted_position == Decimal("5.0")

    @pytest.mark.asyncio
    async def test_below_minimum_odds_rejected(self, risk_manager):
        """Bets below 0.25 odds should be rejected."""
        decision = TradingDecision(
            action="YES",
            confidence=0.90,
            position_size=10.0,
            reasoning="Test",
            token_id="test_token"
        )
        market = {
            "yes_price": 0.20,  # Below 0.25 minimum
            "no_price": 0.80,
            "active": True
        }

        result = await risk_manager.validate_decision(
            decision=decision,
            portfolio_value=Decimal("100.0"),
            market=market,
            open_positions=[]
        )

        assert result.approved is False
        assert "odds" in result.reason.lower()
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_risk.py::TestPositionSizingWithOdds -v
```

**Expected output:** Tests fail because `validate_decision` doesn't pass odds to `_calculate_position_size`

**Step 3: Update validate_decision method**

Modify lines 53-57 in `polymarket/trading/risk.py`:

**OLD CODE:**
```python
        # Check 3: Calculate position size
        max_position = portfolio_value * Decimal(str(self.settings.bot_max_position_percent))
        suggested_size = self._calculate_position_size(
            decision, portfolio_value, max_position
        )
```

**NEW CODE:**
```python
        # Check 3: Calculate position size
        max_position = portfolio_value * Decimal(str(self.settings.bot_max_position_percent))

        # Extract odds for the action being taken
        odds = self._extract_odds_for_action(decision.action, market)

        suggested_size = self._calculate_position_size(
            decision, portfolio_value, max_position, odds
        )

        # Check 3a: Reject if odds below minimum threshold
        if suggested_size == Decimal("0"):
            return ValidationResult(
                approved=False,
                reason=f"Odds {odds:.2f} below minimum threshold 0.25",
                adjusted_position=None
            )
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_risk.py::TestPositionSizingWithOdds -v
```

**Expected output:** All tests pass

**Step 5: Commit**

```bash
git add tests/test_risk.py polymarket/trading/risk.py
git commit -m "feat: integrate odds extraction into validate_decision

Extract odds from market data and pass to position sizing.
Reject bets below 0.25 odds with clear error message.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 5: Update _calculate_position_size Signature and Logic

**Files:**
- Modify: `polymarket/trading/risk.py:108-141` (update `_calculate_position_size` method)

**Step 1: Update method signature and apply odds scaling**

Replace the `_calculate_position_size` method (lines 108-141) with:

```python
    def _calculate_position_size(
        self,
        decision: TradingDecision,
        portfolio_value: Decimal,
        max_position: Decimal,
        odds: float
    ) -> Decimal:
        """
        Calculate position size based on confidence and odds.

        Applies confidence-based sizing, then scales by odds to limit
        exposure on low-probability bets.

        Args:
            decision: AI trading decision
            portfolio_value: Total portfolio value
            max_position: Maximum allowed position size
            odds: Market odds for the side being bet (0.0-1.0)

        Returns:
            Adjusted position size in USDC
        """
        base_size = portfolio_value * Decimal(str(self.settings.bot_max_position_percent))

        # Scale by confidence
        confidence = decision.confidence

        if 0.75 <= confidence < 0.80:
            multiplier = Decimal("0.5")
        elif 0.80 <= confidence < 0.90:
            multiplier = Decimal("0.75")
        elif confidence >= 0.90:
            multiplier = Decimal("1.0")
        else:
            multiplier = Decimal("0.0")

        calculated = base_size * multiplier

        # Apply absolute dollar cap
        dollar_cap = Decimal(str(self.settings.bot_max_position_dollars))
        calculated = min(calculated, dollar_cap)
        max_position = min(max_position, dollar_cap)

        # Use AI-suggested size if provided and reasonable
        if decision.position_size > 0:
            ai_size = min(decision.position_size, max_position)
            calculated = min(ai_size, calculated)
        else:
            calculated = min(calculated, max_position)

        # Apply odds-based scaling
        odds_multiplier = self._calculate_odds_multiplier(odds)

        if odds_multiplier == 0.0:
            logger.info(
                "Bet rejected - odds below minimum threshold",
                odds=odds,
                minimum=0.25
            )
            return Decimal("0")

        final_size = calculated * Decimal(str(odds_multiplier))

        logger.info(
            "Position sized with odds adjustment",
            original_size=float(calculated),
            odds=odds,
            odds_multiplier=odds_multiplier,
            final_size=float(final_size)
        )

        return final_size
```

**Step 2: Run all tests to verify**

```bash
pytest tests/test_risk.py -v
```

**Expected output:** All tests pass

**Step 3: Commit**

```bash
git add polymarket/trading/risk.py
git commit -m "feat: apply odds scaling to position size

Add odds parameter to _calculate_position_size.
Apply odds multiplier after confidence and dollar caps.
Log position adjustment with odds details.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 6: Run Full Test Suite and Verify

**Step 1: Run all tests**

```bash
pytest tests/ -v --tb=short
```

**Expected output:** All existing tests still pass (no regressions)

**Step 2: Run risk tests specifically**

```bash
pytest tests/test_risk.py -v --cov=polymarket.trading.risk --cov-report=term-missing
```

**Expected output:**
- All odds multiplier tests pass
- All odds extraction tests pass
- All integration tests pass
- Coverage shows new methods are tested

**Step 3: Check for any import errors**

```bash
cd /root/polymarket-scripts
python3 -c "from polymarket.trading.risk import RiskManager; print('Import OK')"
```

**Expected output:** `Import OK`

**Step 4: Commit if any test fixes needed**

If any tests needed fixes:

```bash
git add tests/
git commit -m "test: fix test compatibility with odds-adjusted sizing

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 7: Manual Testing with Bot

**Step 1: Backup current database**

```bash
cd /root/polymarket-scripts
cp data/performance.db data/performance.db.backup
```

**Step 2: Start bot in dry-run mode**

Update `.env` to enable dry-run:
```bash
# Temporarily set to read_only for testing
sed -i 's/POLYMARKET_MODE=trading/POLYMARKET_MODE=read_only/' .env
```

**Step 3: Start bot and monitor logs**

```bash
python3 scripts/auto_trade.py 2>&1 | tee logs/odds-scaling-test.log
```

**Step 4: Verify odds scaling in logs**

Watch for log entries like:
```
Position sized with odds adjustment
  original_size=9.56
  odds=0.31
  odds_multiplier=0.62
  final_size=5.93
```

**Step 5: Let bot run for 3-5 cycles**

Monitor that:
- Low-odds bets (<0.40) show reduced position sizes
- High-odds bets (>0.60) show original position sizes
- Sub-0.25 odds are rejected with clear messages

**Step 6: Stop bot and restore settings**

```bash
# Ctrl+C to stop bot
# Restore trading mode
sed -i 's/POLYMARKET_MODE=read_only/POLYMARKET_MODE=trading/' .env
```

---

## Task 8: Documentation and Final Commit

**Step 1: Update risk.py docstring**

Update module docstring at top of `polymarket/trading/risk.py`:

```python
"""
Risk Management Module

Handles position sizing, stop-loss evaluation, and portfolio safety.
Validates all trading decisions before execution.

Position Sizing:
- Confidence-based scaling (0.5x to 1.0x based on confidence)
- Odds-based scaling (0.5x to 1.0x for odds 0.25-0.50)
- Rejects bets with odds below 0.25 (too risky)
- Applies dollar caps and portfolio percentage limits

The odds scaling prevents large bets on low-probability outcomes,
which have asymmetric downside (lose entire stake on Polymarket).
"""
```

**Step 2: Add implementation notes to design doc**

Append to `docs/plans/2026-02-12-odds-adjusted-position-sizing-design.md`:

```markdown
---

## Implementation Complete

**Date:** 2026-02-12
**Commits:** 8 atomic commits following TDD
**Tests:** 10 new test cases, all passing
**Files modified:**
- `polymarket/trading/risk.py` - Added 3 new methods
- `tests/test_risk.py` - Created with comprehensive tests

**Manual testing:** Verified with 5 bot cycles in dry-run mode
**Status:** ✅ Ready for production deployment
```

**Step 3: Final commit**

```bash
git add polymarket/trading/risk.py docs/plans/2026-02-12-odds-adjusted-position-sizing-design.md
git commit -m "docs: update risk.py module docstring and design status

Document odds-based scaling in module header.
Mark design as implementation complete.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

**Step 4: Create summary of changes**

Run:
```bash
git log --oneline --since="today" | head -n 10
```

Should show 8 commits following TDD process.

---

## Deployment Checklist

Before deploying to production:

- [ ] All tests passing (`pytest tests/test_risk.py -v`)
- [ ] No regressions (`pytest tests/ -v`)
- [ ] Manual testing completed (3-5 bot cycles)
- [ ] Logs show correct odds multiplier application
- [ ] Database backed up
- [ ] Dry-run testing successful
- [ ] Design document updated
- [ ] Code reviewed by user

**To deploy:**
1. Ensure bot is stopped
2. Pull latest code (if remote)
3. Run tests one more time
4. Start bot in production mode
5. Monitor first 5 trades closely for correct behavior

---

## Rollback Plan

If issues arise:

```bash
# Stop bot
pkill -f auto_trade

# Restore previous version
git revert HEAD~8..HEAD

# Restore database backup
cp data/performance.db.backup data/performance.db

# Restart bot
python3 scripts/auto_trade.py > logs/trading.log 2>&1 &
```

---

## Success Metrics

After 10 trades with fix deployed:

- Low-odds bets (<0.40) should show 60-80% position sizes
- High-odds bets (>0.60) should show 100% position sizes
- No bets accepted below 0.25 odds
- Net P&L improvement vs pre-fix baseline
- Average loss per losing trade reduced by 30-40%
