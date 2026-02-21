"""Tests for AI analysis v2 hard rules, gap_z computation, phase detection."""
import pytest
from AI_analysis_upgrade.ai_analysis_v2 import (
    compute_gap_z,
    apply_hard_rules,
    get_phase,
    HardRuleResult,
)


def test_negative_gap_hard_rule():
    """gap < -$20 AND ≤180s remaining must return NO (BTC below PTB, no time to recover)."""
    result = apply_hard_rules(
        gap_usd=-45.0,   # BTC is $45 BELOW price-to-beat
        gap_z=-2.1,
        clob_yes=0.72,
        time_remaining_seconds=90,  # ≤180s
    )
    assert result is not None
    assert result.action == "NO"
    assert result.rule == "negative_gap"


def test_negative_gap_no_rule_with_time_remaining():
    """gap < -$20 but >180s remaining → BTC has time to recover → no hard rule."""
    result = apply_hard_rules(
        gap_usd=-45.0,   # BTC is $45 BELOW price-to-beat
        gap_z=-2.1,
        clob_yes=0.72,
        time_remaining_seconds=300,  # >180s — BTC has time to recover
    )
    assert result is None  # Let the ensemble decide


def test_large_positive_gap_hard_rule():
    """gap_z > 2.5 with < 120s remaining must return YES (BTC far above PTB, can't fall back)."""
    result = apply_hard_rules(
        gap_usd=120.0,   # BTC is $120 ABOVE price-to-beat
        gap_z=2.8,
        clob_yes=0.68,
        time_remaining_seconds=80,
    )
    assert result is not None
    assert result.action == "YES"
    assert result.rule == "gap_z_certainty"


def test_no_hard_rule_in_ambiguous_zone():
    """Normal conditions → no hard rule → AI decides."""
    result = apply_hard_rules(
        gap_usd=15.0,
        gap_z=0.9,
        clob_yes=0.65,
        time_remaining_seconds=300,
    )
    assert result is None  # no hard rule fires


def test_compute_gap_z_positive():
    """Positive gap with low vol → large positive z-score."""
    z = compute_gap_z(gap_usd=50.0, realized_vol_per_min=20.0, time_remaining_minutes=2.0)
    assert z > 1.5


def test_compute_gap_z_zero_vol():
    """Zero vol → return 0 (safety guard, no division by zero)."""
    z = compute_gap_z(gap_usd=50.0, realized_vol_per_min=0.0, time_remaining_minutes=2.0)
    assert z == 0.0


def test_phase_is_early():
    """At 10min remaining in a 15min market, phase is 'early'."""
    # 15min market = 900s total. 600s remaining = 67% remaining → early (>33%)
    phase = get_phase(time_remaining_seconds=600, market_duration_seconds=900)
    assert phase == "early"


def test_phase_is_late():
    """At 2min remaining in a 15min market, phase is 'late'."""
    phase = get_phase(time_remaining_seconds=120, market_duration_seconds=900)
    assert phase == "late"
