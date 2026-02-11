# Self-Reflection System Design
**Date:** 2026-02-11
**Status:** Approved
**Goal:** Balanced approach - maximize profit, minimize risk, learn and improve over time

## Overview

A comprehensive self-healing and self-reflection system for the Polymarket trading bot that tracks performance, analyzes losses/missed opportunities, provides AI-powered insights, and adapts parameters autonomously within safe bounds.

## Design Principles

- **Balanced Approach:** Equal focus on profit, risk, and learning
- **Tiered Autonomy:** Small adjustments auto-applied, large changes require approval
- **Adaptive Reflection:** Triggered by smart conditions (losses, batch, daily)
- **Comprehensive Metrics:** Track all 6 metric categories (Win Rate, Risk, Signals, Missed Opportunities, Time Patterns, Execution)
- **Storage Management:** Hybrid SQLite + JSON with weekly cleanup
- **Telegram Integration:** Real-time notifications and interactive controls

---

## Architecture

### High-Level Components

```
┌─────────────────────────────────────────────────────────────┐
│                     TRADING BOT                              │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
         ┌─────────────────────┐
         │ Performance Tracker │ (Real-time)
         └──────────┬──────────┘
                    │
                    ▼
              ┌─────────┐
              │ SQLite  │ (Trade records)
              └────┬────┘
                   │
                   ▼
         ┌──────────────────┐
         │ Reflection Engine│ (Adaptive triggers)
         └────────┬─────────┘
                  │
                  ├─→ Analyzes data
                  ├─→ Generates insights (OpenAI)
                  └─→ Stores to JSON
                  │
                  ▼
         ┌──────────────────────┐
         │ Parameter Adjuster   │ (Tiered autonomy)
         └─────────┬────────────┘
                   │
                   ├─→ Tier 1: Auto-adjust (±5%)
                   ├─→ Tier 2: Ask via Telegram
                   └─→ Tier 3: Pause & alert
                   │
                   ▼
         ┌──────────────────┐
         │  Telegram Bot    │ (Notifications & Control)
         └──────────────────┘
```

---

## Component 1: Performance Tracker

### Database Schema (SQLite: `performance.db`)

**Table: `trades`**
```sql
CREATE TABLE trades (
    id INTEGER PRIMARY KEY,
    timestamp DATETIME,
    market_slug TEXT,
    market_id INTEGER,

    -- Decision
    action TEXT,  -- 'YES', 'NO', 'HOLD'
    confidence REAL,
    position_size REAL,
    reasoning TEXT,

    -- Market Context
    btc_price REAL,
    price_to_beat REAL,
    time_remaining_seconds INTEGER,
    is_end_phase BOOLEAN,

    -- Signals
    social_score REAL,
    market_score REAL,
    final_score REAL,
    final_confidence REAL,
    signal_type TEXT,

    -- Technical Indicators
    rsi REAL,
    macd REAL,
    trend TEXT,

    -- Pricing
    yes_price REAL,
    no_price REAL,
    executed_price REAL,

    -- Outcome (filled after market closes)
    actual_outcome TEXT,  -- 'UP', 'DOWN'
    profit_loss REAL,
    is_win BOOLEAN,
    is_missed_opportunity BOOLEAN
);

CREATE INDEX idx_trades_timestamp ON trades(timestamp);
CREATE INDEX idx_trades_signal_type ON trades(signal_type);
CREATE INDEX idx_trades_is_win ON trades(is_win);
```

**Table: `reflections`**
```sql
CREATE TABLE reflections (
    id INTEGER PRIMARY KEY,
    timestamp DATETIME,
    trigger_type TEXT,  -- '3_losses', '10_trades', 'end_of_day'
    trades_analyzed INTEGER,
    insights TEXT,  -- JSON of AI recommendations
    adjustments_made TEXT  -- JSON of parameter changes
);
```

**Table: `parameter_history`**
```sql
CREATE TABLE parameter_history (
    id INTEGER PRIMARY KEY,
    timestamp DATETIME,
    parameter_name TEXT,
    old_value REAL,
    new_value REAL,
    reason TEXT,
    approval_method TEXT  -- 'auto', 'telegram_approved', 'manual'
);
```

### Integration Point

In `scripts/auto_trade.py`, after every decision:

```python
await performance_tracker.log_decision(
    market=market,
    decision=decision,
    btc_data=btc_data,
    technical=technical_indicators,
    aggregated=aggregated_sentiment
)
```

After market closes (15 min later):
```python
await performance_tracker.update_outcome(
    market_slug=market_slug,
    actual_outcome=actual_outcome,
    profit_loss=profit_loss
)
```

---

## Component 2: Reflection Engine

### Trigger Conditions

1. **Emergency Trigger:** 3 consecutive losses → immediate reflection
2. **Batch Trigger:** Every 10 trades → batch analysis
3. **Daily Trigger:** End of day (11:59 PM) → comprehensive review

### Analysis Process

**Step 1: Aggregate Metrics**
```python
metrics = {
    # Win Rate & Profit (A)
    "win_rate": calculate_win_rate(),
    "total_profit": sum(profits),
    "avg_profit_per_trade": mean(profits),
    "roi": total_profit / total_invested,

    # Risk Metrics (B)
    "max_drawdown": calculate_max_drawdown(),
    "consecutive_losses": max_consecutive_losses(),
    "risk_adjusted_return": sharpe_ratio(),

    # Signal Performance (C)
    "best_signal_type": "STRONG_BEARISH",
    "worst_signal_type": "STRONG_NEUTRAL",
    "signal_accuracy": signal_win_rates_by_type(),

    # Missed Opportunities (D)
    "missed_count": count_holds_with_regret(),
    "missed_profit": sum_missed_profits(),
    "regret_rate": missed_count / total_holds,

    # Time-Based Patterns (E)
    "best_hour": analyze_by_hour(),
    "early_vs_end_performance": compare_market_phases(),

    # Execution Quality (F)
    "avg_slippage": mean(executed_price - expected_price),
    "fill_rate": orders_filled / orders_placed
}
```

**Step 2: OpenAI Reflection Prompt**
```python
prompt = f"""
You are analyzing your own Polymarket trading performance as a self-improving AI.

**Recent Performance:**
- Trades: {trade_count}
- Win Rate: {win_rate}%
- Total Profit: ${total_profit}
- Max Drawdown: ${max_drawdown}
- Missed Opportunities: {missed_count} (regret rate: {regret_rate}%)

**Signal Performance:**
{signal_breakdown}

**Time Patterns:**
{time_based_analysis}

**Missed Opportunities:**
{hold_decisions_with_regret}

**Current Parameters:**
- Confidence Threshold: {current_threshold}
- Max Position: ${max_position}
- Max Exposure: {max_exposure}%

**Analysis Tasks:**
1. What patterns led to winning trades?
2. What mistakes are being repeated?
3. Should confidence threshold be adjusted? Why?
4. Are validation rules too strict? (contradiction holds regret rate: {regret_rate}%)
5. Which signal types should be trusted more/less?
6. Are there time-based patterns to exploit?
7. Recommend 2-3 specific parameter adjustments with detailed reasoning.

**Output Format (JSON):**
{{
  "insights": [
    "End-of-market trades have 80% win rate vs 60% early trades",
    "Contradiction holds have 67% regret rate - validation may be too strict"
  ],
  "patterns": {{
    "winning": ["STRONG_BEARISH in end-phase", "High agreement (>1.4x)"],
    "losing": ["STRONG_NEUTRAL trades", "Early market entries"]
  }},
  "recommendations": [
    {{
      "parameter": "bot_confidence_threshold",
      "current": 0.75,
      "recommended": 0.70,
      "reason": "Win rate is 72%, being too conservative. Lower threshold to capture more opportunities while maintaining quality.",
      "tier": 2,
      "expected_impact": "Increase trade frequency by ~30%, maintain win rate ~70%"
    }}
  ]
}}
"""
```

**Step 3: Store & Act**
- Parse OpenAI response
- Save to `insights/YYYY-MM-DD-HH-MM.json`
- Store in `reflections` table
- Send formatted summary to Telegram
- Trigger Parameter Adjuster with recommendations

---

## Component 3: Parameter Adjuster

### Safety Bounds

```python
SAFE_ADJUSTMENTS = {
    "bot_confidence_threshold": {
        "min": 0.65,
        "max": 0.85,
        "baseline": 0.75,
        "step": 0.05
    },
    "bot_max_position_dollars": {
        "min": 5.0,
        "max": 20.0,
        "baseline": 10.0,
        "step": 2.0
    },
    "bot_max_exposure_percent": {
        "min": 0.30,
        "max": 0.70,
        "baseline": 0.50,
        "step": 0.05
    },
    "stop_loss_odds_threshold": {
        "min": 0.30,
        "max": 0.50,
        "baseline": 0.40,
        "step": 0.05
    }
}
```

### Tier 1: Auto-Adjust (No Approval)

**Conditions:**
- Small adjustments (±1 step)
- Max 1 adjustment per hour
- Max ±0.10 deviation from baseline

**Example:**
```python
if win_rate < 0.50 and current_threshold < 0.80:
    new_threshold = min(current_threshold + 0.05, 0.80)

    apply_adjustment(
        parameter="bot_confidence_threshold",
        new_value=new_threshold,
        reason=f"Win rate {win_rate}% - raising threshold for quality",
        tier=1
    )

    telegram_notify(f"""
    🔧 Auto-Adjustment Applied

    Parameter: confidence_threshold
    {current_threshold} → {new_threshold}

    Reason: Win rate {win_rate}% below target, raising bar for trade quality

    /rollback to undo
    /parameter_history for details
    """)
```

### Tier 2: Approval Required (Telegram Vote)

**Conditions:**
- Medium adjustments (>1 step)
- Significant pattern detected
- 3+ consecutive losses

**Workflow:**
```python
recommendation = {
    "parameter": "bot_confidence_threshold",
    "current": 0.75,
    "recommended": 0.65,
    "reason": "Win rate is 75% with avg profit $4.50...",
    "expected_impact": "Increase trade frequency ~40%, maintain 70%+ win rate"
}

telegram_msg = f"""
⚠️ **Parameter Adjustment Recommendation**

**Parameter:** {recommendation['parameter']}
**Current:** {recommendation['current']}
**Recommended:** {recommendation['recommended']}

**Reason:**
{recommendation['reason']}

**Expected Impact:**
{recommendation['expected_impact']}

**Approve this change?**
👍 Yes, apply it
👎 No, keep current
⏸ Pause trading for review
"""

response = await telegram_bot.wait_for_approval(
    recommendation,
    timeout_seconds=3600  # 1 hour
)

if response == "approved":
    apply_adjustment(recommendation, tier=2)
elif response == "rejected":
    log_rejection(recommendation)
elif response == "timeout":
    log_timeout(recommendation)
    # Keep current, try again next cycle
```

### Tier 3: Emergency Pause (Critical Issues)

**Triggers:**
- 5+ consecutive losses
- Drawdown > 20%
- Win rate < 30% over 20 trades
- Unusual patterns (e.g., all trades failing same way)

**Actions:**
```python
if consecutive_losses >= 5:
    pause_trading()

    telegram_alert(f"""
    🚨 **EMERGENCY PAUSE ACTIVATED**

    **Trigger:** 5 consecutive losses
    **Total Loss:** ${abs(total_loss)}
    **Current Drawdown:** {drawdown}%

    **Recent Trades:**
    {format_recent_trades(last_5)}

    **Bot Status:** PAUSED

    **Actions Available:**
    /insights - View detailed analysis
    /adjust - Manual parameter adjustment
    /rollback - Revert recent changes
    /resume - Resume trading (use with caution)
    /baseline - Reset all parameters to baseline
    """)
```

---

## Component 4: Telegram Bot

### Setup

```bash
# .env additions
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here
TELEGRAM_ENABLED=true
```

### Notification Types

**1. Trade Alerts** (Real-time)
```
🎯 Trade Executed

Market: btc-updown-15m-1770799500
Action: NO (DOWN)
Confidence: 100%
Position: $5.00 @ 0.52

Reasoning: Bearish signals aligned with BTC movement (-0.14%)

Expected profit: ~$4.80 if correct
```

**2. Reflection Insights**
```
💡 Reflection Complete
Trigger: 10 trades analyzed
Time: 2026-02-11 14:30

**Performance:**
Win Rate: 65% (6W-4L)
Profit: +$18.50
Max Drawdown: -$7.20

**Key Insights:**
✅ End-of-market trades: 80% win rate
⚠️ Contradiction holds: 60% regret rate
📊 STRONG_BEARISH signals: 70% accuracy

**Patterns:**
Winning: High agreement scores (>1.4x), End-phase entries
Losing: STRONG_NEUTRAL signals, Early entries

View full report: /insights latest
```

**3. Auto-Adjustments** (Tier 1)
```
🔧 Auto-Adjustment Applied

Parameter: confidence_threshold
Changed: 0.75 → 0.70

Reason: Win rate is 72%, being too conservative. Lower threshold to capture more opportunities.

Impact: Expect ~30% more trades
Quality: Win rate should remain ~70%

History: /parameter_history
Rollback: /rollback
```

**4. Approval Requests** (Tier 2)
```
⚠️ Parameter Adjustment Recommendation

Parameter: bot_max_position_dollars
Current: $10.00
Recommended: $15.00

Reason: Win rate is 75% with avg profit $4.50. Increasing position size could improve returns while maintaining 0.50 exposure limit.

Expected Impact: +50% profit per trade, same win rate

Approve this change?
👍 Yes, apply it (1 hour to decide)
👎 No, keep current
⏸ Pause trading for review
```

**5. Emergency Alerts** (Tier 3)
```
🚨 EMERGENCY PAUSE ACTIVATED

Trigger: 5 consecutive losses
Total Loss: -$23.50
Current Drawdown: -18.2%

Recent Trades:
1. NO @ 0.55 → LOST $5.00 (STRONG_BEARISH)
2. YES @ 0.48 → LOST $4.20 (STRONG_BULLISH)
3. NO @ 0.52 → LOST $5.30 (STRONG_BEARISH)
4. NO @ 0.60 → LOST $4.50 (STRONG_BEARISH)
5. YES @ 0.45 → LOST $4.50 (STRONG_NEUTRAL)

Pattern: STRONG_BEARISH signals failing (3/5)

Bot Status: PAUSED

Actions:
/insights - View detailed analysis
/adjust confidence_threshold 0.80 - Raise bar
/rollback - Revert recent changes
/resume - Resume trading (confirm first)
/baseline - Reset all to baseline
```

### Interactive Commands

**Performance Queries:**
```python
/stats           # Overall statistics
/today           # Today's performance
/week            # Weekly breakdown
/insights [id]   # View specific reflection (latest if no id)
/missed          # Missed opportunities analysis
/signals         # Signal type performance breakdown
```

**Control Commands:**
```python
/pause           # Pause trading immediately
/resume          # Resume trading
/status          # Bot status + current parameters
/parameter_history [limit]  # Recent adjustments (default 10)
```

**Configuration:**
```python
/adjust [param] [value]      # Manual adjustment
/rollback [steps]            # Rollback last N changes (default 1)
/reset                       # Reset all parameters to baseline
/config                      # View current configuration
```

**Examples:**
```
User: /stats
Bot:
📊 Overall Statistics

Total Trades: 47
Win Rate: 68.1% (32W-15L)
Total Profit: +$142.30
Max Drawdown: -$18.50

Avg Profit/Trade: $3.03
Best Signal: STRONG_BEARISH (75% win rate)
Worst Signal: STRONG_NEUTRAL (45% win rate)

Missed Opportunities: 8 (40% regret rate)
Potential Missed: $31.50

/week for weekly breakdown
/signals for signal analysis
```

```
User: /insights latest
Bot:
💡 Latest Reflection (2026-02-11 14:30)

Trades Analyzed: 10
Trigger: Batch (every 10 trades)

**Insights:**
1. End-of-market trades (< 3 min) have 80% win rate vs 60% early trades
2. Contradiction holds have 67% regret rate - validation may be too strict
3. STRONG_BEARISH signals perform best with 1.4x+ agreement

**Recommendations:**
1. Lower confidence threshold 0.75 → 0.70 (capture more opportunities)
2. Reduce end-phase threshold 3 min → 2 min (more end-phase trades)
3. Relax contradiction validation when agreement > 1.4x

View full: insights/2026-02-11-14-30.json
```

---

## Component 5: Data Retention & Cleanup

### Retention Strategy

**Hot Data** (SQLite - Immediate access)
- **0-30 days:** All individual trades with full details
- **30-90 days:** Aggregated daily summaries only
- **Purpose:** Recent analysis, active pattern detection

**Warm Data** (Compressed JSON Archives)
- **30-90 days old:** Individual trades compressed to JSON
- **90-180 days old:** Weekly summaries only
- **Purpose:** Historical trends, long-term patterns

**Cold Data** (Permanent Summaries)
- **180+ days:** Delete individual trades
- **Keep forever:** Monthly performance summaries
- **Purpose:** Long-term tracking without bloat

### Weekly Cleanup Job

Runs every **Sunday at 2:00 AM**:

```python
async def weekly_cleanup():
    """
    1. Archive old trades to compressed JSON
    2. Summarize and delete ancient trades
    3. Vacuum database
    4. Report storage status
    """

    # Step 1: Archive trades 30-90 days old
    trades_30_90 = get_trades(age_days=(30, 90))
    archive_path = f"archives/{today.year}/{today.month}/"

    for chunk in chunk_list(trades_30_90, size=1000):
        filename = f"{archive_path}/trades-{chunk[0].timestamp.date()}.json.gz"
        save_compressed_json(chunk, filename)

    # Step 2: Summarize trades 90-180 days old
    trades_90_180 = get_trades(age_days=(90, 180))
    weekly_summaries = aggregate_by_week(trades_90_180)

    for week_summary in weekly_summaries:
        save_summary(week_summary)

    # Delete from database
    delete_from_db(trades_90_180)

    # Step 3: Delete archives older than 180 days (keep monthly summaries)
    for archive_file in glob("archives/**/*.json.gz"):
        if file_age(archive_file) > 180:
            if not is_monthly_summary(archive_file):
                delete_file(archive_file)

    # Step 4: Vacuum database
    vacuum_database()

    # Step 5: Report
    db_size_mb = get_db_size() / 1024 / 1024
    free_space_gb = get_free_disk_space() / 1024 / 1024 / 1024

    telegram_notify(f"""
    📦 Weekly Cleanup Complete

    Archived: {len(trades_30_90)} trades → {archive_size_mb:.1f} MB
    Summarized: {len(trades_90_180)} trades → {len(weekly_summaries)} weeks
    Deleted: {deleted_archives_count} old archives

    Storage Status:
    Database: {db_size_mb:.1f} MB
    Archives: {archive_total_mb:.1f} MB
    Free Space: {free_space_gb:.1f} GB

    Next cleanup: {next_sunday_2am}
    """)
```

### Storage Estimates

**Per Trade Record:**
- SQLite row: ~2 KB
- JSON (compressed): ~500 bytes

**Expected Sizes:**
- **30 days:** ~2,880 trades × 2 KB = **5.8 MB** (SQLite)
- **90 days:** ~8,640 trades (archived) × 0.5 KB = **4.3 MB** (compressed JSON)
- **180+ days:** Monthly summaries only = **~100 KB**
- **Total:** ~10-15 MB for active data, ~5 MB/month archives

**Max Database Size:** 20 MB (with cleanup)

### Emergency Cleanup

If **free disk space < 1 GB**:

```python
if get_free_disk_space() < 1_000_000_000:  # 1 GB
    telegram_alert("🚨 Low Disk Space - Emergency Cleanup")

    # Aggressive cleanup
    delete_trades(age_days=7)  # Keep only 7 days
    compress_all_archives()
    vacuum_database()

    telegram_alert(f"✅ Emergency cleanup complete. Free space: {get_free_disk_space_gb():.1f} GB")
```

---

## Error Handling & Resilience

### Failure Modes

**Performance Tracker Failures:**
```python
try:
    await performance_tracker.log_decision(...)
except DatabaseError as e:
    logger.error("DB write failed", error=str(e))
    # Fallback: append to JSON file
    append_to_fallback_log(decision_data)
except Exception as e:
    logger.error("Performance tracking failed", error=str(e))
    # Continue trading - don't block on logging
```

**Reflection Engine Failures:**
```python
try:
    insights = await reflection_engine.analyze()
except OpenAITimeout:
    logger.warning("Reflection timeout, skipping")
    schedule_retry(delay_minutes=30)
except OpenAIError as e:
    logger.error("OpenAI API failed", error=str(e))
    # Continue trading without reflection
except Exception as e:
    logger.error("Reflection failed", error=str(e))
    # Continue trading
```

**Telegram Bot Failures:**
```python
try:
    await telegram_bot.send_alert(message)
except NetworkError:
    # Queue for retry
    retry_queue.append(message)
    schedule_retry(delay_seconds=60)
except TelegramAPIError as e:
    logger.error("Telegram API failed", error=str(e))
    # Continue trading without notifications
```

**Parameter Adjuster Failures:**
```python
try:
    apply_adjustment(param, value)
except ValidationError as e:
    logger.error("Invalid adjustment", param=param, value=value, error=str(e))
    rollback_to_previous()
    telegram_alert("🚨 Invalid adjustment rejected - rolled back")
except Exception as e:
    logger.error("Adjustment failed", error=str(e))
    rollback_to_previous()
```

### Rollback Mechanisms

```python
# Manual rollback via Telegram
/rollback          # Revert last adjustment
/rollback 3        # Revert last 3 adjustments
/baseline          # Reset all to baseline

# Automatic rollback on error
if adjustment_causes_error():
    rollback_to_previous()
    telegram_alert("🚨 Adjustment caused error - auto-rolled back")

# Rollback history preserved
SELECT * FROM parameter_history
WHERE approval_method = 'rollback'
ORDER BY timestamp DESC;
```

---

## Testing Strategy

### Unit Tests

```python
# Database operations
test_log_decision()
test_update_outcome()
test_calculate_win_rate()
test_aggregate_metrics()

# Parameter validation
test_safe_adjustment_bounds()
test_adjustment_validation()
test_rollback_logic()

# Reflection logic
test_generate_reflection_prompt()
test_parse_openai_response()
test_trigger_conditions()

# Telegram formatting
test_format_trade_alert()
test_format_reflection_summary()
test_approval_workflow()
```

### Integration Tests

```python
# End-to-end workflows
test_trade_to_reflection_to_adjustment()
test_telegram_approval_workflow()
test_emergency_pause_trigger()
test_cleanup_job_execution()

# Error handling
test_database_failure_recovery()
test_openai_timeout_handling()
test_telegram_network_failure()
test_invalid_adjustment_rollback()
```

### Live Testing (Phased Rollout)

**Phase 1: Logging Only** (Week 1)
- ✅ Enable Performance Tracker
- ❌ Disable Reflection Engine
- ❌ Disable Parameter Adjuster
- **Goal:** Verify data quality, no side effects

**Phase 2: Reflection Engine** (Week 2)
- ✅ Enable Performance Tracker
- ✅ Enable Reflection Engine (insights only)
- ❌ Disable Parameter Adjuster
- **Goal:** Test OpenAI prompts, verify insights quality

**Phase 3: Telegram Notifications** (Week 3)
- ✅ Enable Performance Tracker
- ✅ Enable Reflection Engine
- ✅ Enable Telegram Bot (alerts only)
- ❌ Disable Auto-Adjustments
- **Goal:** Test notification workflow, no parameter changes

**Phase 4: Tier 1 Auto-Adjustments** (Week 4)
- ✅ Enable all above
- ✅ Enable Tier 1 adjustments (small auto-changes)
- ❌ Disable Tier 2 approvals
- **Goal:** Test self-healing with minimal risk

**Phase 5: Full System** (Week 5+)
- ✅ Enable Tier 1 auto-adjustments
- ✅ Enable Tier 2 approval workflow
- ✅ Enable Tier 3 emergency pause
- ✅ Enable weekly cleanup
- **Goal:** Full autonomy

---

## Implementation Checklist

### Phase 1: Database & Logging
- [ ] Create `polymarket/performance/tracker.py`
- [ ] Implement SQLite schema
- [ ] Add logging hook in `auto_trade.py`
- [ ] Add outcome update hook (15 min after trade)
- [ ] Write unit tests
- [ ] Deploy and monitor (1 week)

### Phase 2: Reflection Engine
- [ ] Create `polymarket/performance/reflection.py`
- [ ] Implement trigger conditions
- [ ] Design OpenAI reflection prompt
- [ ] Add metric aggregation functions
- [ ] Write integration tests
- [ ] Deploy insights-only mode (1 week)

### Phase 3: Telegram Integration
- [ ] Create `polymarket/telegram/bot.py`
- [ ] Implement notification handlers
- [ ] Add interactive commands
- [ ] Test approval workflow (mock)
- [ ] Deploy notifications-only (1 week)

### Phase 4: Parameter Adjuster
- [ ] Create `polymarket/performance/adjuster.py`
- [ ] Implement tier system (1, 2, 3)
- [ ] Add parameter validation
- [ ] Implement rollback mechanism
- [ ] Write safety tests
- [ ] Deploy Tier 1 only (1 week)

### Phase 5: Cleanup & Monitoring
- [ ] Create `polymarket/performance/cleanup.py`
- [ ] Implement weekly job (cron)
- [ ] Add storage monitoring
- [ ] Test emergency cleanup
- [ ] Deploy full system

### Phase 6: Full Deployment
- [ ] Enable Tier 2 approvals
- [ ] Enable Tier 3 pauses
- [ ] Monitor for 2 weeks
- [ ] Iterate based on performance

---

## Monitoring & Observability

### Daily Health Check (via Telegram, 9 AM)

```
🏥 Daily Health Report

**System Status:**
✅ Performance Tracker: Active (47 trades logged)
✅ Reflection Engine: Last run 12h ago
✅ Telegram Bot: Connected
✅ Parameter Adjuster: 2 adjustments this week

**Storage:**
Database: 12.5 MB / 20 MB (62%)
Archives: 8.3 MB
Free Space: 42.1 GB
Next Cleanup: Sunday 2:00 AM

**Recent Activity:**
- Last trade: 3 min ago (NO @ 0.52)
- Last reflection: 12h ago (10 trades)
- Last adjustment: 2 days ago (confidence_threshold)

/status for details
```

### Weekly Performance Report (via Telegram, Sunday 9 AM)

```
📊 Weekly Performance Report
Week of 2026-02-04 to 2026-02-10

**Trading Performance:**
Total Trades: 47
Win Rate: 68.1% (32W-15L)
Profit: +$142.30
Max Drawdown: -$18.50
Sharpe Ratio: 1.85

**Signal Performance:**
🥇 STRONG_BEARISH: 75% (12/16)
🥈 STRONG_BULLISH: 70% (7/10)
🥉 STRONG_NEUTRAL: 45% (9/20)

**Time Patterns:**
Best Hour: 14:00-15:00 UTC (80% win rate)
Worst Hour: 08:00-09:00 UTC (50% win rate)
End-Phase Trades: 80% win rate (16/20)
Early Trades: 60% win rate (16/27)

**Missed Opportunities:**
Total: 8 HOLD decisions
Regret Rate: 40%
Potential Profit: $31.50

**Parameter Adjustments:**
- 2/6: confidence_threshold 0.75 → 0.70 (auto)
- 2/8: max_position 10 → 12 (approved)

**Recommendations:**
✅ Continue current strategy
⚠️ Consider reducing contradiction validation
📈 Increase end-phase exposure

/insights for details
```

---

## Success Metrics

### Short-term (Week 1-4)

- [ ] All trades logged successfully (100% capture rate)
- [ ] Reflection engine runs on schedule (0 missed triggers)
- [ ] Telegram notifications delivered (<5% failure rate)
- [ ] No trading interruptions from self-reflection system

### Medium-term (Month 2-3)

- [ ] Win rate improvement: +5% vs baseline
- [ ] Profit improvement: +10% vs baseline
- [ ] Missed opportunity reduction: -30% vs baseline
- [ ] Parameter adjustments showing positive impact

### Long-term (Month 4+)

- [ ] Self-optimizing parameters without human intervention
- [ ] Adaptive to different market conditions
- [ ] Consistent profitability across market cycles
- [ ] Comprehensive historical performance database

---

## Security & Safety Considerations

1. **Parameter Bounds:** All adjustments constrained to safe ranges
2. **Tier System:** Large changes require human approval
3. **Emergency Pause:** Automatic pause on critical issues
4. **Rollback:** Easy revert of any adjustment
5. **Telegram Security:** Bot token + chat ID kept secret
6. **Database Backups:** Weekly backups before cleanup
7. **Audit Trail:** All adjustments logged with reasoning

---

## Future Enhancements (Out of Scope)

- Multi-market support (track performance across different markets)
- A/B testing framework (test parameter changes on subset)
- Machine learning models (complement OpenAI with ML predictions)
- Web dashboard (visual analytics beyond Telegram)
- Portfolio optimization (Kelly criterion, risk-parity)
- Advanced risk management (stop-loss automation, hedging)

---

## Conclusion

This self-reflection system provides:
- ✅ Comprehensive performance tracking (all 6 metric categories)
- ✅ AI-powered insights (adaptive reflection triggers)
- ✅ Safe self-healing (tiered autonomy with rollback)
- ✅ Real-time control (Telegram notifications & commands)
- ✅ Storage efficiency (hybrid SQLite + JSON with cleanup)
- ✅ Phased rollout (minimize risk, validate each component)

**Next Steps:**
1. Review and approve this design
2. Create implementation plan (detailed task breakdown)
3. Set up git worktree for isolated development
4. Begin Phase 1 implementation (Database & Logging)
