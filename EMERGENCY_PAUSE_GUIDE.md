# Emergency Pause System - Implementation Guide

## Overview

The Tier 3 emergency pause system stops trading when dangerous parameter recommendations (>20% changes) are detected.

## How It Works

### 1. Trigger Mechanism

When a Tier 3 adjustment is detected (>20% change):
- `ParameterAdjuster.apply_adjustment()` rejects the change
- Writes `.emergency_pause` file to project root
- Sends urgent Telegram alert
- Logs to database with `tier_3_emergency_pause` flag

### 2. Trading Bot Detection

Before each trading cycle, `AutoTrader.run_cycle()`:
- Checks `EMERGENCY_PAUSE_ENABLED` environment variable
- Checks for `.emergency_pause` file existence
- If either is true: stops trading and exits

### 3. Recovery Process

To resume trading after emergency pause:

1. Review the Telegram alert for details
2. Investigate why the dangerous adjustment was recommended
3. Manually adjust parameters if needed
4. Delete the `.emergency_pause` file: `rm .emergency_pause`
5. Set `EMERGENCY_PAUSE_ENABLED=false` in `.env`
6. Restart the bot

**IMPORTANT**: Do not resume without understanding the root cause.

## Files Modified

### Configuration
- `.env.example`: Added `EMERGENCY_PAUSE_ENABLED` flag
- `polymarket/config.py`: Added `emergency_pause_enabled` setting

### Core Logic
- `polymarket/performance/adjuster.py`:
  - Tier 3 triggers emergency pause
  - `_set_emergency_pause()`: Writes pause file
  - Sends alert via Telegram
  - Logs to database

- `scripts/auto_trade.py`:
  - `_check_emergency_pause()`: Checks for pause flag/file
  - Stops trading if pause detected

### Notifications
- `polymarket/telegram/bot.py`:
  - `send_emergency_alert()`: Sends urgent formatted alert

### Tests
- `tests/test_performance_adjuster.py`:
  - `test_trigger_emergency_pause()`: Verifies pause mechanism
  - `test_emergency_pause_file_stops_trading()`: Verifies file creation

## Telegram Alert Format

```
🚨 EMERGENCY PAUSE TRIGGERED 🚨

⚠️ Dangerous parameter adjustment detected

Parameter: bot_confidence_threshold
Current: 0.7500
Proposed: 0.5250
Change: -30.0%

Reason: Win rate too low, need stricter threshold

🛑 BOT HAS BEEN PAUSED

This change exceeds the ±20% safety threshold.
Manual review required before resuming trading.

To resume:
1. Review the recommendation carefully
2. Manually adjust parameters if needed
3. Set EMERGENCY_PAUSE_ENABLED=false
4. Restart the bot

Do not resume without understanding why this was triggered.
```

## Testing

Run tests with:
```bash
pytest tests/test_performance_adjuster.py -v
```

All 15 tests pass, including:
- `test_trigger_emergency_pause`: Verifies Tier 3 triggers pause
- `test_emergency_pause_file_stops_trading`: Verifies file mechanism

## Example Usage

```python
from polymarket.performance.adjuster import ParameterAdjuster, AdjustmentTier

adjuster = ParameterAdjuster(settings, db=db, telegram=telegram)

# This will trigger emergency pause (30% decrease)
result = await adjuster.apply_adjustment(
    parameter_name="bot_confidence_threshold",
    old_value=0.75,
    new_value=0.525,  # 30% decrease
    reason="Win rate critically low",
    tier=AdjustmentTier.TIER_3_PAUSE
)

# result = False (rejected)
# .emergency_pause file created
# Telegram alert sent
# Trading bot will stop on next cycle
```

## Security Considerations

1. **File Permissions**: The `.emergency_pause` file should be readable by the trading bot process
2. **Environment Variables**: `EMERGENCY_PAUSE_ENABLED` takes precedence over file
3. **Manual Override**: Both file and env var must be cleared to resume
4. **Database Audit**: All Tier 3 attempts are logged with `tier_3_emergency_pause` flag

## Monitoring

Check for emergency pauses in database:
```sql
SELECT * FROM parameter_history 
WHERE approval_method = 'tier_3_emergency_pause'
ORDER BY timestamp DESC;
```

## Limitations

1. File-based mechanism requires filesystem access
2. No automatic recovery (manual intervention required)
3. Assumes single bot instance per project directory

## Future Enhancements

- Add email notifications in addition to Telegram
- Implement automatic diagnostics report generation
- Add emergency pause history dashboard
- Support distributed pause coordination (multiple bots)

---

**Implementation Date**: 2026-02-11
**Commit**: 04380bb0ec094a167bb7e0330813badde74d1fed
**Tests**: 15/15 passing
