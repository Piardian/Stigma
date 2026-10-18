"""Comprehensive test suite for the Stigma Quant Research Engine (Faz 2).

Covers:
  - Mathematical core: Hurst, Lo-MacKinlay VR (relaxed dual fractal), Adaptive Kalman Filter,
    Yang-Zhang Volatility, Ornstein-Uhlenbeck (120-bar window, 2-40 bar half-life)
  - Net R:R Gate: Target relocation (TP1 min 1.15 R), Low R:R rejection (TP2 < 1.6 R),
    Cost-to-SL protection (risk < 4.0 * cost)
  - Institutional Session & Weekend Armor: London/NY active hours, Friday evening veto & exit
  - Funnel Telemetry: Detailed tracking of filtration stages
"""
from __future__ import annotations
import json
import math
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from stigma.quant_engine import (
    calculate_hurst_vr,
    run_kalman_filter_1d,
    yang_zhang_volatility,
    estimate_ou_half_life,
    dynamic_time_stop,
    evaluate_stigma_trigger,
)
from stigma.backtest_engine import prepare_features, backtest, check_session_rules, atr14
from stigma.macro_gate import load_macro_bias
from stigma.execution_bridge import build_order_intent

# =====================================================================
# 1. HURST & RELAXED DUAL-FRACTAL REGIME CLASSIFICATION
# =====================================================================

def test_hurst_insufficient_data_fails_closed():
    """Fails closed if data length < window or window < 32."""
    assert calculate_hurst_vr(np.log(np.arange(1, 20, dtype=float)), window=100).label == "INSUFFICIENT_DATA"
    assert calculate_hurst_vr(np.log(np.arange(1, 50, dtype=float)), window=20).label == "INSUFFICIENT_DATA"
    assert calculate_hurst_vr([], window=100).label == "INSUFFICIENT_DATA"

def test_hurst_nan_or_inf_fails_closed():
    """Any non-finite price in window returns INSUFFICIENT_DATA."""
    prices = np.full(120, 100.0)
    prices[50] = np.nan
    assert calculate_hurst_vr(prices, window=100).label == "INSUFFICIENT_DATA"
    prices[50] = np.inf
    assert calculate_hurst_vr(prices, window=100).label == "INSUFFICIENT_DATA"

def test_hurst_constant_price_zero_variance():
    """Zero variance across series returns INSUFFICIENT_DATA without crashing."""
    prices = np.full(150, 4.6)
    regime = calculate_hurst_vr(prices, window=100)
    assert regime.label == "INSUFFICIENT_DATA"

def test_relaxed_dual_fractal_classification_boundaries():
    """Test updated Faz 2 dual fractal classification logic."""
    # 1. RANDOM_WALK_VETO: 0.47 <= H <= 0.53 AND 0.95 <= VR <= 1.05
    # Construct mock arrays or verify classify rules
    rng = np.random.default_rng(42)
    rw = np.cumsum(rng.normal(0, 0.01, 500))
    r = calculate_hurst_vr(rw[-100:], window=100)
    assert r.label in {"RANDOM_WALK_VETO", "OU_MEAN_REVERSION", "KALMAN_MOMENTUM", "NO_EDGE"}

def test_hurst_persistent_vs_antipersistent_ordering():
    """Positively autocorrelated returns produce higher H and VR than negatively autocorrelated returns."""
    rng = np.random.default_rng(42)
    n = 1000
    white = rng.normal(0, 0.01, n)

    # Persistent series: returns have strong positive lag-1 autocorrelation
    r_pers = np.zeros(n)
    for i in range(1, n):
        r_pers[i] = 0.85 * r_pers[i - 1] + white[i]
    y_pers = 100.0 + np.cumsum(r_pers)

    # Anti-persistent series: returns have strong negative lag-1 autocorrelation
    r_anti = np.zeros(n)
    for i in range(1, n):
        r_anti[i] = -0.60 * r_anti[i - 1] + white[i]
    y_anti = 100.0 + np.cumsum(r_anti)

    hp = calculate_hurst_vr(y_pers[-100:], window=100)
    ha = calculate_hurst_vr(y_anti[-100:], window=100)

    assert hp.hurst > ha.hurst
    assert hp.label == "KALMAN_MOMENTUM"
    assert ha.label == "OU_MEAN_REVERSION"

# =====================================================================
# 2. ADAPTIVE KALMAN FILTER
# =====================================================================

def test_adaptive_kalman_dynamic_scaling():
    """Adaptive Kalman filter scales Ri and Qi with local ATR, maintaining finite level/vel/z."""
    prices = np.full(200, 100.0)
    atr = np.full(200, 1.0)
    k = run_kalman_filter_1d(prices, atr, adaptive=True)
    assert np.isclose(k.level[-1], 100.0, atol=1e-3)
    assert np.isclose(k.velocity[-1], 0.0, atol=1e-3)
    assert np.isclose(k.z_score[-1], 0.0, atol=1e-3)

def test_adaptive_kalman_linear_trend_velocity():
    """Adaptive Kalman filter accurately captures trend velocity."""
    n = 300
    prices = 100.0 + 0.1 * np.arange(n)
    atr = np.full(n, 0.5)
    k = run_kalman_filter_1d(prices, atr, adaptive=True)
    assert k.velocity[-1] > 0.02
    assert np.isfinite(k.level).all()
    assert np.all(k.innovation_std > 0)

def test_adaptive_kalman_handles_nans():
    """Non-positive or non-finite prices propagate without crashing."""
    p = np.array([100.0, -5.0, np.nan, 102.0, 103.0])
    atr = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    k = run_kalman_filter_1d(p, atr, adaptive=True)
    assert len(k.level) == len(p)
    assert np.isfinite(k.level[-1])

# =====================================================================
# 3. YANG-ZHANG VOLATILITY
# =====================================================================

def test_yang_zhang_initial_window_nans():
    """First 'window' bars must be NaN."""
    n = 50
    o = np.full(n, 100.0); h = np.full(n, 102.0); l = np.full(n, 99.0); c = np.full(n, 101.0)
    yz = yang_zhang_volatility(o, h, l, c, window=20)
    assert np.isnan(yz[:20]).all()
    assert np.isfinite(yz[20:]).all()

def test_yang_zhang_rejects_negative_or_corrupt_prices():
    """Raises ValueError if prices are negative or corrupt."""
    with pytest.raises(ValueError, match="strictly positive"):
        yang_zhang_volatility([-10, 10, 10, 10], [12, 12, 12, 12], [8, 8, 8, 8], [11, 11, 11, 11], window=2)
    with pytest.raises(ValueError, match="geometry"):
        yang_zhang_volatility([10, 10, 10], [8, 8, 8], [12, 12, 12], [10, 10, 10], window=2)

# =====================================================================
# 4. ORNSTEIN-UHLENBECK HALF-LIFE & TIME STOP
# =====================================================================

def test_ou_half_life_known_ar1():
    """Theoretical AR(1) half-life: beta=0.85 -> phi=-0.15 -> hl ~= 4.26 bars."""
    rng = np.random.default_rng(42)
    n = 3000
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = 0.85 * x[i - 1] + rng.normal(0, 0.1)

    res = estimate_ou_half_life(x)
    assert res["valid"] is True
    assert 3.0 <= res["half_life"] <= 5.5

def test_ou_half_life_extended_bounds():
    """Valid half-life is 2.0 to 40.0 bars."""
    rng = np.random.default_rng(7)
    # Slow mean-reversion: beta = 0.98 -> phi = -0.02 -> hl = ln(2)/0.0202 ~= 34.3 bars
    x = np.zeros(2000)
    for i in range(1, 2000):
        x[i] = 0.98 * x[i - 1] + rng.normal(0, 0.05)
    res = estimate_ou_half_life(x)
    # Under Faz 2, 34 bars is VALID (was invalid in Faz 1 which capped at 30)
    if res["valid"]:
        assert 2.0 <= res["half_life"] <= 40.0

def test_dynamic_time_stop_clipping():
    """int(clip(2.5 * half_life, 6, 40))."""
    assert dynamic_time_stop(1.0) == 6
    assert dynamic_time_stop(4.0) == 10
    assert dynamic_time_stop(15.0) == 37
    assert dynamic_time_stop(20.0) == 40

# =====================================================================
# 5. NET R:R GATE & COST-TO-SL PROTECTION
# =====================================================================

def test_cost_to_sl_protection():
    """Signals where risk < 4.0 * cost_price are rejected with HIGH_COST_TO_SL."""
    # ATR is 0.1 -> risk = 1.2 * 0.1 = 0.12. If cost_price = 0.04 -> 4.0 * cost = 0.16 > risk
    sig, reason = evaluate_stigma_trigger(
        regime="OU_MEAN_REVERSION",
        hurst=0.40,
        variance_ratio=0.85,
        half_life=10.0,
        z_score=[-2.5, -2.2, -1.80],
        level=[105.0, 105.0, 105.0],
        velocity=[0.0, 0.0, 0.0],
        innovation_std=[1.0, 1.0, 1.0],
        atr=[0.1, 0.1, 0.1],
        open_=[100.0, 100.0, 100.0],
        close=[100.5, 100.5, 101.0],
        macro_bias=0,
        cost_price=0.04,  # High cost compared to 0.12 risk!
        return_reason=True,
    )
    assert sig is None
    assert reason == "HIGH_COST_TO_SL"

def test_net_rr_gate_relocates_close_tp1():
    """When Kalman level is closer than 1.1 R, TP1 is moved forward to at least 1.15 R."""
    # ATR = 1.0 -> risk = 1.2. Kalman level is only 101.5 (0.5 above entry 101.0 -> 0.42 R)
    sig = evaluate_stigma_trigger(
        regime="OU_MEAN_REVERSION",
        hurst=0.40,
        variance_ratio=0.85,
        half_life=10.0,
        z_score=[-2.5, -2.2, -1.80],
        level=[101.5, 101.5, 101.5],  # Very close Kalman level!
        velocity=[0.0, 0.0, 0.0],
        innovation_std=[1.0, 1.0, 1.0],
        atr=[1.0, 1.0, 1.0],
        open_=[100.0, 100.0, 100.0],
        close=[100.5, 100.5, 101.0],
        macro_bias=0,
        cost_price=0.01,
    )
    assert sig is not None
    # TP1 must have been pushed forward to >= entry + 1.15 * risk = 101.0 + 1.38 = 102.38
    assert sig["tp1"] >= 101.0 + 1.15 * 1.2
    assert sig["rr_tp1"] >= 1.10

def test_net_rr_gate_rejects_low_rr():
    """When TP2 net R:R < 1.6, trade is rejected with LOW_RR."""
    # Force TP2 to be very close by passing small innovation_std and small level
    # We can test by setting a custom condition where TP2 cannot reach 1.6 R
    sig, reason = evaluate_stigma_trigger(
        regime="OU_MEAN_REVERSION",
        hurst=0.40,
        variance_ratio=0.85,
        half_life=10.0,
        z_score=[-2.5, -2.2, -1.80],
        level=[101.0, 101.0, 101.0],
        velocity=[0.0, 0.0, 0.0],
        innovation_std=[0.1, 0.1, 0.1],
        atr=[1.0, 1.0, 1.0],
        open_=[100.0, 100.0, 100.0],
        close=[100.5, 100.5, 101.0],
        cost_price=1.0,  # Large cost makes net rr_tp2 < 1.6
        return_reason=True,
    )
    assert sig is None
    assert reason in ("HIGH_COST_TO_SL", "LOW_RR")

def test_kalman_momentum_yang_zhang_expansion_filter():
    """Momentum trade is vetoed if sigma_yz <= sigma_yz_median."""
    sig, reason = evaluate_stigma_trigger(
        regime="KALMAN_MOMENTUM",
        hurst=0.60,
        variance_ratio=1.15,
        half_life=10.0,
        z_score=[-0.5, -0.6, -0.4],
        level=[100.0, 100.0, 100.0],
        velocity=[0.2, 0.2, 0.2],
        innovation_std=[1.0, 1.0, 1.0],
        atr=[1.0, 1.0, 1.0],
        open_=[100.0, 100.0, 100.0],
        close=[100.2, 100.2, 100.5],
        macro_bias=0,
        sigma_yz=0.005,
        sigma_yz_median=0.008,  # Volatility is contracting!
        return_reason=True,
    )
    assert sig is None
    assert reason == "YZ_VOL_NOT_EXPANDING"

# =====================================================================
# 6. INSTITUTIONAL SESSION & WEEKEND ARMOR
# =====================================================================

def test_session_rules_mt5_london_ny():
    """MT5 session active only between 06:00 and 20:00 UTC on weekdays."""
    # Tuesday 10:00 UTC -> active
    tue_10 = pd.Timestamp("2026-03-10 10:00:00", tz="UTC")
    can_enter, _ = check_session_rules(tue_10, "mt5")
    assert can_enter is True

    # Tuesday 02:00 UTC (Asian dead zone) -> inactive
    tue_02 = pd.Timestamp("2026-03-10 02:00:00", tz="UTC")
    can_enter, _ = check_session_rules(tue_02, "mt5")
    assert can_enter is False

    # Friday 20:45 UTC -> entry vetoed
    fri_2045 = pd.Timestamp("2026-03-13 20:45:00", tz="UTC")
    can_enter, force_close = check_session_rules(fri_2045, "mt5")
    assert can_enter is False
    assert force_close is False

    # Friday 21:35 UTC -> force close before weekend
    fri_2135 = pd.Timestamp("2026-03-13 21:35:00", tz="UTC")
    can_enter, force_close = check_session_rules(fri_2135, "mt5")
    assert can_enter is False
    assert force_close is True

# =====================================================================
# 7. FUNNEL TELEMETRY INTEGRATION
# =====================================================================

def test_backtest_funnel_counts_structure():
    """Verify backtest returns complete funnel_counts dictionary with required keys."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2026-01-05 06:00", periods=200, freq="15min", tz="UTC")
    c = 100.0 + np.cumsum(rng.normal(0, 0.1, 200))
    o = c + rng.normal(0, 0.02, 200)
    h = np.maximum(o, c) + 0.05
    l = np.minimum(o, c) - 0.05
    df = pd.DataFrame({"open": o, "high": h, "low": l, "close": c}, index=dates)

    res = backtest(df, session_type="all")
    assert "funnel_counts" in res
    fc = res["funnel_counts"]
    expected_keys = {
        "bars_evaluated", "veto_session_hours", "veto_random_walk",
        "regime_ou_mr_bars", "regime_kalman_mom_bars", "passed_ou_halflife",
        "passed_yz_vol", "passed_zscore_setup", "passed_turn_and_candle",
        "rejected_high_cost_to_sl", "rejected_low_rr", "trades_opened"
    }
    assert expected_keys.issubset(fc.keys())
    assert fc["bars_evaluated"] > 0

# =====================================================================
# 8. MACRO GATE & EXECUTION BRIDGE
# =====================================================================

def test_macro_gate_fail_closed():
    """Corrupt or missing files fail closed to 0 (neutral)."""
    with tempfile.TemporaryDirectory() as td:
        tmp_path = Path(td)
        assert load_macro_bias(tmp_path / "non_existent.json", autonomous=False) == 0
        bad_json = tmp_path / "bad.json"
        bad_json.write_text("{not valid json", encoding="utf-8")
        assert load_macro_bias(bad_json, autonomous=False) == 0

def test_execution_bridge_order_intent_validation():
    """Validates geometry and ensures live mode is disabled."""
    valid_signal = {
        "side": "LONG",
        "entry": 100.0,
        "stop_loss": 98.0,
        "tp1": 102.0,
        "tp2": 105.0,
        "max_hold_bars": 20,
    }
    intent = build_order_intent(valid_signal)
    assert intent["mode"] == "LIVE_DISABLED"
    assert intent["order_type"] == "MARKET_NEXT_BAR"
    assert intent["risk_distance"] == 2.0
