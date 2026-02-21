"""Tests for Telegram comparison alert formatting."""
from AI_analysis_upgrade.telegram_alerts import format_comparison_alert


def test_format_v2_wins():
    """When v2 is right and v1 is wrong, alert shows v2 winning."""
    msg = format_comparison_alert(
        market_slug="btc-usd-1700",
        v1_action="YES",
        v1_confidence=0.74,
        v2_action="NO",
        v2_confidence=0.68,
        actual_outcome="NO",
        trend_score=-0.35,
        fear_greed=28,
        gap_usd=-42.0,
        gap_z=-1.9,
        phase="late",
        v1_total_wins=10,
        v2_total_wins=12,
        total_markets=22,
    )
    assert "❌" in msg    # v1 wrong
    assert "✅" in msg    # v2 right
    assert "v2" in msg.lower() or "V2" in msg
    assert "12" in msg   # v2 wins count


def test_format_both_correct():
    """Both right → two checkmarks."""
    msg = format_comparison_alert(
        market_slug="btc-usd-1715",
        v1_action="NO",
        v1_confidence=0.71,
        v2_action="NO",
        v2_confidence=0.69,
        actual_outcome="NO",
        trend_score=-0.2,
        fear_greed=40,
        gap_usd=-15.0,
        gap_z=-0.8,
        phase="early",
        v1_total_wins=11,
        v2_total_wins=13,
        total_markets=23,
    )
    assert msg.count("✅") == 2


def test_format_both_wrong():
    """Both wrong → two X marks."""
    msg = format_comparison_alert(
        market_slug="btc-usd-1730",
        v1_action="YES",
        v1_confidence=0.68,
        v2_action="YES",
        v2_confidence=0.62,
        actual_outcome="NO",
        trend_score=0.1,
        fear_greed=55,
        gap_usd=10.0,
        gap_z=0.5,
        phase="late",
        v1_total_wins=11,
        v2_total_wins=13,
        total_markets=24,
    )
    assert msg.count("❌") == 2
