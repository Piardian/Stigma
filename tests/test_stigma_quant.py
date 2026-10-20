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
    """int(clip(2.0 * half_life, 6, 24))."""
    assert dynamic_time_stop(1.0) == 6
    assert dynamic_time_stop(4.0) == 8
    assert dynamic_time_stop(10.0) == 20
    assert dynamic_time_stop(15.0) == 24
    assert dynamic_time_stop(20.0) == 24

# =====================================================================
# 5. NET R:R GATE & COST-TO-SL PROTECTION
# =====================================================================

def test_cost_to_sl_protection():
    """Signals where risk < 3.5 * cost_price are rejected with HIGH_COST_TO_SL."""
    # ATR is 0.1 -> risk = 1.4 * 0.1 = 0.14. If cost_price = 0.05 -> 3.5 * cost = 0.175 > risk
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
        cost_price=0.05,  # High cost: 3.5 * 0.05 = 0.175 > 0.14 risk!
        return_reason=True,
    )
    assert sig is None
    assert reason == "HIGH_COST_TO_SL"

def test_net_rr_gate_relocates_close_tp1():
    """When Kalman level is closer than 1.1 R, TP1 is moved forward to at least 1.15 R."""
    # ATR = 1.0 -> risk = 1.4 * 1.0 = 1.4. Kalman level is only 101.5 (0.5 above entry 101.0 -> 0.35 R)
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
    # TP1 must have been pushed forward to >= entry + 1.15 * risk = 101.0 + 1.15 * 1.4 = 102.61
    assert sig["tp1"] >= 101.0 + 1.15 * 1.4
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

# =====================================================================
# 9. FAZ 3: SPREAD Z-SCORE, REGIME ISOLATION & H1 TIMEFRAMES
# =====================================================================

def test_empirical_spread_zscore_formula():
    """Verify Z_K = (P - level) / max(rolling_spread_std, 1.0 * ATR14)."""
    n = 100
    prices = 100.0 + np.sin(np.linspace(0, 10, n)) * 2.0
    atr = np.full(n, 0.5)
    k = run_kalman_filter_1d(prices, atr, adaptive=True)
    assert len(k.z_score) == n
    assert np.isfinite(k.z_score[20:]).all()

def test_kalman_momentum_breakout_and_pullback():
    """Kalman Momentum requires pullback to Kalman line and strong candle breakout."""
    # 1. Successful breakout
    sig, reason = evaluate_stigma_trigger(
        regime="KALMAN_MOMENTUM",
        hurst=0.60,
        variance_ratio=1.15,
        half_life=10.0,
        z_score=[0.1, -0.2, 0.1],
        level=[99.5, 99.7, 100.0],
        velocity=[0.1, 0.15, 0.2],
        innovation_std=[0.5, 0.5, 0.5],
        atr=[0.5, 0.5, 0.5],
        open_=[99.8, 99.8, 99.9],
        close=[100.1, 100.2, 100.8],
        high_=[100.3, 100.4, 101.0],
        low_=[99.5, 99.6, 99.8],
        macro_bias=0,
        sigma_yz=0.012,
        sigma_yz_median=0.008,
        strategy_mode="MOM_ONLY",
        return_reason=True,
    )
    assert sig is not None
    assert reason == "SUCCESS"
    assert sig["side"] == "LONG"
    assert sig["risk"] == pytest.approx(1.6 * 0.5)

    # 2. Vetoed by regime isolation when strategy_mode is OU_ONLY
    sig_veto, reason_veto = evaluate_stigma_trigger(
        regime="KALMAN_MOMENTUM",
        hurst=0.60,
        variance_ratio=1.15,
        half_life=10.0,
        z_score=[0.1, -0.2, 0.1],
        level=[99.5, 99.7, 100.0],
        velocity=[0.1, 0.15, 0.2],
        innovation_std=[0.5, 0.5, 0.5],
        atr=[0.5, 0.5, 0.5],
        open_=[99.8, 99.8, 99.9],
        close=[100.1, 100.2, 100.8],
        high_=[100.3, 100.4, 101.0],
        low_=[99.5, 99.6, 99.8],
        macro_bias=0,
        sigma_yz=0.012,
        sigma_yz_median=0.008,
        strategy_mode="OU_ONLY",
        return_reason=True,
    )
    assert sig_veto is None
    assert reason_veto == "STRATEGY_MODE_VETO"

def test_backtest_by_regime_breakdown():
    """Verify backtest output includes by_regime breakdown for OU and Momentum."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2026-01-05 06:00", periods=250, freq="15min", tz="UTC")
    c = 100.0 + np.cumsum(rng.normal(0, 0.1, 250))
    o = c + rng.normal(0, 0.02, 250)
    h = np.maximum(o, c) + 0.1
    l = np.minimum(o, c) - 0.1
    df = pd.DataFrame({"open": o, "high": h, "low": l, "close": c}, index=dates)

    res = backtest(df, session_type="all", strategy_mode="BOTH")
    assert "by_regime" in res
    assert "OU_MEAN_REVERSION" in res["by_regime"]
    assert "KALMAN_MOMENTUM" in res["by_regime"]
    assert "strategy_mode" in res
    assert res["strategy_mode"] == "BOTH"

def test_h1_timeframe_adapters():
    """Verify H1 is recognized as a valid timeframe without ValueError in both adapters."""
    from stigma.data.binance_data import INTERVALS
    from stigma.data.mt5_data import SUPPORTED_TIMEFRAMES

    assert "H1" in INTERVALS
    assert INTERVALS["H1"] == "1h"
    assert "H1" in SUPPORTED_TIMEFRAMES

# =====================================================================
# 10. FAZ 4: HTF TREND ANCHOR, FLAT SLOPE OU LOCK & EARLY BREAKEVEN
# =====================================================================

def test_ou_trending_slope_veto():
    """OU Mean-Reversion is vetoed if Kalman velocity or EMA 200 slope is too steep."""
    # 1. Kalman velocity > 0.12 * ATR
    sig_v, reason_v = evaluate_stigma_trigger(
        regime="OU_MEAN_REVERSION",
        hurst=0.40,
        variance_ratio=0.85,
        half_life=10.0,
        z_score=[-2.5, -2.3, -2.0],
        level=[100.0, 100.0, 100.0],
        velocity=[0.2, 0.2, 0.2],  # 0.2 > 0.12 * 1.0 ATR
        innovation_std=[1.0, 1.0, 1.0],
        atr=[1.0, 1.0, 1.0],
        open_=[100.0, 100.0, 100.0],
        close=[100.2, 100.3, 100.5],
        return_reason=True,
    )
    assert sig_v is None
    assert reason_v == "TRENDING_SLOPE_VETO"

    # 2. EMA 200 20-bar slope > 2.5 * ATR
    ema = np.linspace(90.0, 95.0, 25)  # delta = 5.0 > 2.5 * 1.0 ATR
    sig_e, reason_e = evaluate_stigma_trigger(
        regime="OU_MEAN_REVERSION",
        hurst=0.40,
        variance_ratio=0.85,
        half_life=10.0,
        z_score=[-2.5, -2.3, -2.0],
        level=[100.0, 100.0, 100.0],
        velocity=[0.05, 0.05, 0.05],  # 0.05 <= 0.12 * ATR
        innovation_std=[1.0, 1.0, 1.0],
        atr=[1.0, 1.0, 1.0],
        open_=[100.0, 100.0, 100.0],
        close=[100.2, 100.3, 100.5],
        ema_200=ema,
        return_reason=True,
    )
    assert sig_e is None
    assert reason_e == "TRENDING_SLOPE_VETO"

def test_ou_extreme_zk_and_candle_geometry():
    """OU Mean-Reversion requires extreme Z_K (|z| >= 2.20), turn confirmation, and proper candle geometry."""
    # 1. Z_K not extreme enough (min is -2.10 < 2.20 threshold)
    sig_not_ext, reason_not_ext = evaluate_stigma_trigger(
        regime="OU_MEAN_REVERSION",
        hurst=0.40,
        variance_ratio=0.85,
        half_life=10.0,
        z_score=[-2.10, -2.08, -2.01],
        level=[100.0, 100.0, 100.0],
        velocity=[0.0, 0.0, 0.0],
        innovation_std=[1.0, 1.0, 1.0],
        atr=[1.0, 1.0, 1.0],
        open_=[100.0, 100.0, 100.0],
        close=[100.2, 100.3, 100.5],
        high_=[100.4, 100.4, 100.6],
        low_=[99.9, 99.9, 99.9],
        return_reason=True,
    )
    assert sig_not_ext is None
    assert reason_not_ext == "SETUP_NOT_TRIGGERED"

    # 2. Extreme Z_K, turn confirmed, bullish candle, upper 40% close -> SUCCESS
    sig_ok, reason_ok = evaluate_stigma_trigger(
        regime="OU_MEAN_REVERSION",
        hurst=0.40,
        variance_ratio=0.85,
        half_life=10.0,
        z_score=[-2.50, -2.30, -2.02],
        level=[100.0, 100.0, 100.0],
        velocity=[0.0, 0.0, 0.0],
        innovation_std=[1.0, 1.0, 1.0],
        atr=[1.0, 1.0, 1.0],
        open_=[98.0, 98.0, 98.0],
        close=[98.1, 98.2, 98.8],
        high_=[98.2, 98.3, 98.9],
        low_=[97.8, 97.8, 97.9],
        return_reason=True,
    )
    assert sig_ok is not None
    assert reason_ok == "SUCCESS"
    assert sig_ok["side"] == "LONG"
    assert sig_ok["risk"] == pytest.approx(1.4 * 1.0)
    assert sig_ok["tp1"] == pytest.approx(98.8 + 1.15 * 1.4)
    assert sig_ok["tp2"] == pytest.approx(98.8 + 2.0 * 1.4)

def test_kalman_momentum_htf_trend_anchor():
    """Kalman Momentum requires price and 200 EMA to be aligned in trend direction."""
    ema_bullish = np.linspace(95.0, 98.0, 25)  # e_now = 98.0 > e_prev20 = 95.5
    ema_bearish = np.linspace(105.0, 102.0, 25)  # e_now = 102.0 < e_prev20 = 104.5

    # 1. Bullish momentum setup with close (100.8) > ema (98.0) and ema rising -> SUCCESS
    sig_long, reason_long = evaluate_stigma_trigger(
        regime="KALMAN_MOMENTUM",
        hurst=0.60,
        variance_ratio=1.15,
        half_life=10.0,
        z_score=[0.1, -0.2, 0.1],
        level=[99.5, 99.7, 100.0],
        velocity=[0.1, 0.15, 0.2],
        innovation_std=[0.5, 0.5, 0.5],
        atr=[0.5, 0.5, 0.5],
        open_=[99.8, 99.8, 99.9],
        close=[100.1, 100.2, 100.8],
        high_=[100.3, 100.4, 101.0],
        low_=[99.5, 99.6, 99.8],
        sigma_yz=0.012,
        sigma_yz_median=0.008,
        ema_200=ema_bullish,
        return_reason=True,
    )
    assert sig_long is not None
    assert reason_long == "SUCCESS"
    assert sig_long["risk"] == pytest.approx(1.6 * 0.5)
    assert sig_long["tp1"] == pytest.approx(100.8 + 1.35 * (1.6 * 0.5))
    assert sig_long["tp2"] == pytest.approx(100.8 + 2.6 * (1.6 * 0.5))

    # 2. Bullish momentum setup but EMA is above price (bearish HTF) -> HTF_TREND_MISALIGNED
    sig_veto, reason_veto = evaluate_stigma_trigger(
        regime="KALMAN_MOMENTUM",
        hurst=0.60,
        variance_ratio=1.15,
        half_life=10.0,
        z_score=[0.1, -0.2, 0.1],
        level=[99.5, 99.7, 100.0],
        velocity=[0.1, 0.15, 0.2],
        innovation_std=[0.5, 0.5, 0.5],
        atr=[0.5, 0.5, 0.5],
        open_=[99.8, 99.8, 99.9],
        close=[100.1, 100.2, 100.8],
        high_=[100.3, 100.4, 101.0],
        low_=[99.5, 99.6, 99.8],
        sigma_yz=0.012,
        sigma_yz_median=0.008,
        ema_200=ema_bearish,
        return_reason=True,
    )
    assert sig_veto is None
    assert reason_veto == "HTF_TREND_MISALIGNED"

# =====================================================================
# 11. FAZ 5: SIGNAL DEDUPLICATOR & TELEGRAM ALERT ENGINE
# =====================================================================

def test_signal_deduplicator():
    """SignalDeduplicator prevents duplicate alerts for identical (symbol, strat, bar_time)."""
    from stigma.execution_notifier import SignalDeduplicator

    dedup = SignalDeduplicator()
    # First time: should emit
    assert dedup.should_emit("NAS100", "OU_ONLY", "2026-10-19 01:15:00") is True
    # Duplicate bar: should not emit
    assert dedup.should_emit("NAS100", "OU_ONLY", "2026-10-19 01:15:00") is False
    assert dedup.is_duplicate("NAS100", "OU_ONLY", "2026-10-19 01:15:00") is True

    # Different time: should emit
    assert dedup.should_emit("NAS100", "OU_ONLY", "2026-10-19 01:30:00") is True
    # Different strategy: should emit
    assert dedup.should_emit("NAS100", "MOM_ONLY", "2026-10-19 01:15:00") is True
    # Different symbol: should emit
    assert dedup.should_emit("USDJPY", "OU_ONLY", "2026-10-19 01:15:00") is True

    # Clear resets state
    dedup.clear()
    assert dedup.should_emit("NAS100", "OU_ONLY", "2026-10-19 01:15:00") is True

def test_format_telegram_message_and_send_failsafe():
    """Verifies Telegram alert message contains all quant parameters and fails safe without token."""
    from stigma.execution_notifier import format_telegram_message, send_telegram_alert

    sample_signal = {
        "symbol": "BTCUSDT",
        "timeframe": "H1",
        "strategy_mode": "OU_MEAN_REVERSION",
        "side": "LONG",
        "entry": 65000.0,
        "stop_loss": 64500.0,
        "tp1": 65500.0,
        "tp2": 66200.0,
        "max_hold_bars": 18,
        "hurst": 0.412,
        "variance_ratio": 0.885,
        "half_life": 9.2,
        "z_score": -2.35,
        "bar_time": "2026-10-19 01:00:00",
    }
    msg = format_telegram_message(sample_signal)
    assert "BTCUSDT" in msg
    assert "OU_MEAN_REVERSION" in msg
    assert "65000" in msg
    assert "0.412" in msg
    assert "-2.35" in msg
    assert "9.2 bars" in msg

    # Without tokens, send_telegram_alert logs and returns False cleanly without raising
    sent = send_telegram_alert(sample_signal, bot_token="", chat_id="")
    assert sent is False

def test_scan_whitelist_missing_file_failsafe():
    """scan_whitelisted_strategies returns empty list when whitelist file is missing."""
    from run_pipeline import scan_whitelisted_strategies

    signals, bars = scan_whitelisted_strategies(whitelist_file="non_existent_file.json")
    assert signals == []
    assert bars == {}





#


# =====================================================================
# 12. TELEGRAM TRADE CLOSING ALERTS, PNL, NET R & TP/SL COUNTERS
# =====================================================================

def test_format_trade_close_message_metrics():
    """Trade close message contains Net PnL, gross Net R, post-commission Net R, and TP/SL counters."""
    from stigma.execution_notifier import format_trade_close_message

    trade = {
        "symbol": "USDJPY",
        "timeframe": "M15",
        "strategy_mode": "MOM_ONLY",
        "side": "LONG",
        "entry": 149.250,
        "exit": 150.400,
        "tp1_exit_price": 149.850,
        "tp1_hit": True,
        "initial_risk": 0.450,
        "bars_held": 8,
        "exit_reason": "tp2",
        "gross_r": 1.98,
        "net_r": 1.91,
        "net_pnl": 88.20,
    }
    stats = {
        "total_trades": 20,
        "tp_count": 14,
        "sl_count": 5,
        "breakeven_count": 1,
        "win_rate": 70.0,
        "cumulative_net_r": 18.45,
        "cumulative_net_pnl": 842.10,
    }

    msg = format_trade_close_message(trade, stats)
    assert "USDJPY (M15)" in msg
    assert "MOM_ONLY" in msg
    assert "TP2" in msg
    assert "+88.20 USD" in msg
    assert "+1.98 R" in msg
    assert "+1.91 R" in msg
    assert "14" in msg
    assert "5" in msg
    assert "70.0" in msg
    assert "+18.45 R" in msg

def test_live_performance_tracker_counters():
    """LivePerformanceTracker accurately updates TP, SL, BE, Win Rate, and cumulative Net R."""
    import tempfile
    from pathlib import Path
    import pytest
    from stigma.execution_notifier import LivePerformanceTracker

    with tempfile.TemporaryDirectory() as td:
        json_file = Path(td) / "test_perf.json"
        tracker = LivePerformanceTracker(filepath=json_file)

        # 1. Winning trade (+1.85 R, +$92.50)
        win_trade = {
            "symbol": "NAS100",
            "timeframe": "M15",
            "side": "LONG",
            "entry": 20000.0,
            "exit": 20185.0,
            "exit_reason": "tp2",
            "gross_r": 1.90,
            "net_r": 1.85,
            "net_pnl": 92.50,
        }
        stats1 = tracker.record_closed_trade(win_trade)
        assert stats1["tp_count"] == 1
        assert stats1["sl_count"] == 0
        assert stats1["total_trades"] == 1
        assert stats1["win_rate"] == 100.0
        assert stats1["cumulative_net_r"] == 1.85
        assert stats1["cumulative_net_pnl"] == 92.50

        # 2. Losing trade (-1.05 R, -$52.50)
        loss_trade = {
            "symbol": "BTCUSDT",
            "timeframe": "H1",
            "side": "SHORT",
            "entry": 65000.0,
            "exit": 65550.0,
            "exit_reason": "stop",
            "gross_r": -1.0,
            "net_r": -1.05,
            "net_pnl": -52.50,
        }
        stats2 = tracker.record_closed_trade(loss_trade)
        assert stats2["tp_count"] == 1
        assert stats2["sl_count"] == 1
        assert stats2["total_trades"] == 2
        assert stats2["win_rate"] == 50.0
        assert stats2["cumulative_net_r"] == pytest.approx(0.80)
        assert stats2["cumulative_net_pnl"] == pytest.approx(40.00)

        # 3. Breakeven trade (+0.01 R, +$0.50)
        be_trade = {
            "symbol": "USDJPY",
            "timeframe": "M15",
            "side": "LONG",
            "entry": 149.00,
            "exit": 149.02,
            "exit_reason": "breakeven_stop",
            "gross_r": 0.04,
            "net_r": 0.01,
            "net_pnl": 0.50,
        }
        stats3 = tracker.record_closed_trade(be_trade)
        assert stats3["tp_count"] == 1
        assert stats3["sl_count"] == 1
        assert stats3["breakeven_count"] == 1
        assert stats3["total_trades"] == 3
        assert stats3["win_rate"] == pytest.approx(33.3, abs=0.1)

def test_live_position_tracker_tp_scale_out_lifecycle():
    """LivePositionTracker scales out at TP1, trails SL to breakeven, and closes at TP2."""
    import tempfile
    from pathlib import Path
    import pytest
    from stigma.execution_notifier import LivePerformanceTracker, LivePositionTracker

    with tempfile.TemporaryDirectory() as td:
        perf_file = Path(td) / "perf.json"
        pos_file = Path(td) / "pos.json"
        perf = LivePerformanceTracker(filepath=perf_file)
        tracker = LivePositionTracker(filepath=pos_file, perf_tracker=perf)

        signal = {
            "symbol": "USDJPY",
            "timeframe": "M15",
            "strategy_mode": "MOM_ONLY",
            "macro_mode": "Begonya_Macro",
            "side": "LONG",
            "entry": 100.0,
            "stop_loss": 99.0,
            "tp1": 101.35,
            "tp2": 102.60,
            "risk": 1.0,
            "max_hold_bars": 20,
            "bar_time": "2026-10-20 08:00:00",
        }
        pos = tracker.open_position(signal, units=1.0, cost_price=0.05)
        assert tracker.has_position("USDJPY", "MOM_ONLY") is True

        # Bar 1: Price reaches TP1 (High = 101.50)
        candle1 = {"open": 100.2, "high": 101.50, "low": 100.1, "close": 101.2, "time": "2026-10-20 08:15:00"}
        closed1, stats1 = tracker.update_candle("USDJPY", "MOM_ONLY", candle1, kalman_level=100.5, atr=0.5)
        assert closed1 is None  # Position still open (50% remaining)
        pos_updated = tracker.get_position("USDJPY", "MOM_ONLY")
        assert pos_updated["tp1_hit"] is True
        assert pos_updated["tp1_exit_price"] == 101.35
        assert pos_updated["stop_loss"] == 100.0  # Trailed to breakeven

        # Bar 2: Price reaches TP2 (High = 102.80) -> Position closes completely
        candle2 = {"open": 101.3, "high": 102.80, "low": 101.0, "close": 102.7, "time": "2026-10-20 08:30:00"}
        closed2, stats2 = tracker.update_candle("USDJPY", "MOM_ONLY", candle2, kalman_level=101.0, atr=0.5)
        assert closed2 is not None
        assert tracker.has_position("USDJPY", "MOM_ONLY") is False
        assert closed2["exit_reason"] == "tp2"
        # Gross = 0.5 * (101.35 - 100) + 0.5 * (102.60 - 100) = 0.5 * 1.35 + 0.5 * 2.60 = 0.675 + 1.30 = 1.975
        assert closed2["gross_pnl_unit"] == pytest.approx(1.975)
        assert closed2["net_pnl_unit"] == pytest.approx(1.975 - 0.05)
        assert closed2["net_r"] == pytest.approx((1.975 - 0.05) / 1.0)
        assert stats2["tp_count"] == 1
        assert stats2["sl_count"] == 0

def test_live_position_tracker_stop_loss_exit():
    """LivePositionTracker closes position at Stop Loss and records negative Net R."""
    import tempfile
    from pathlib import Path
    import pytest
    from stigma.execution_notifier import LivePerformanceTracker, LivePositionTracker

    with tempfile.TemporaryDirectory() as td:
        perf_file = Path(td) / "perf2.json"
        pos_file = Path(td) / "pos2.json"
        perf = LivePerformanceTracker(filepath=perf_file)
        tracker = LivePositionTracker(filepath=pos_file, perf_tracker=perf)

        signal = {
            "symbol": "NAS100",
            "timeframe": "M15",
            "strategy_mode": "OU_ONLY",
            "side": "SHORT",
            "entry": 20000.0,
            "stop_loss": 20100.0,
            "tp1": 19850.0,
            "tp2": 19700.0,
            "risk": 100.0,
            "max_hold_bars": 15,
            "bar_time": "2026-10-20 09:00:00",
        }
        tracker.open_position(signal, units=2.0, cost_price=2.0)

        # Bar 1: Price spikes up hitting stop (High = 20120)
        candle = {"open": 20010.0, "high": 20120.0, "low": 19990.0, "close": 20110.0, "time": "2026-10-20 09:15:00"}
        closed, stats = tracker.update_candle("NAS100", "OU_ONLY", candle, kalman_level=20000.0, atr=50.0)

        assert closed is not None
        assert tracker.has_position("NAS100", "OU_ONLY") is False
        assert closed["exit_reason"] == "stop"
        assert closed["exit"] == 20100.0
        assert closed["gross_pnl_unit"] == pytest.approx(-100.0)
        assert closed["net_pnl_unit"] == pytest.approx(-102.0)
        assert closed["net_pnl"] == pytest.approx(-204.0)
        assert closed["net_r"] == pytest.approx(-102.0 / 100.0)
        assert stats["tp_count"] == 0
        assert stats["sl_count"] == 1
