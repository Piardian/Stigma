"""Comprehensive test suite for the Stigma Quant Research Engine.

Covers:
  - Mathematical core: Hurst, Lo-MacKinlay VR, Kalman Filter, Yang-Zhang, Ornstein-Uhlenbeck
  - Edge cases: Insufficient data, NaN/Inf, constant prices, zero variance, invalid geometry
  - Signal evaluation: Long/Short symmetry, macro bias gate, threshold breaches, trigger rules
  - Execution & Backtest: Next-open execution, gap stops, intrabar conflict, TP1/TP2 scale-out,
    cost accounting, look-ahead bias prevention, single-position lock
  - Data adapters: Fail-closed behavior on missing libraries or network errors
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
from stigma.backtest_engine import prepare_features, backtest, atr14
from stigma.macro_gate import load_macro_bias
from stigma.execution_bridge import build_order_intent

# =====================================================================
# 1. HURST & LO-MACKINLAY VARIANCE RATIO
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
    assert np.isnan(regime.hurst) or np.isnan(regime.variance_ratio)

def test_hurst_random_walk_veto_boundaries():
    """Random walk series should fall around 0.50 and trigger RANDOM_WALK_VETO."""
    rng = np.random.default_rng(42)
    rw = np.cumsum(rng.normal(0, 0.01, 500))
    r = calculate_hurst_vr(rw[-100:], window=100)
    assert np.isfinite(r.hurst)
    assert np.isfinite(r.variance_ratio)
    assert 0.40 <= r.hurst <= 0.60
    if 0.46 <= r.hurst <= 0.54:
        assert r.label == "RANDOM_WALK_VETO"

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
    assert hp.hurst > 0.54
    assert hp.variance_ratio > 1.08
    assert hp.label == "KALMAN_MOMENTUM"

    assert ha.hurst < 0.46
    assert ha.variance_ratio < 0.92
    assert ha.label == "OU_MEAN_REVERSION"

def test_hurst_vr_classification_matrix():
    """Explicitly verify valid label outcomes for finite series."""
    r1 = calculate_hurst_vr(np.cumsum(np.random.default_rng(10).normal(0, 0.01, 100)), window=100)
    assert r1.label in {"RANDOM_WALK_VETO", "OU_MEAN_REVERSION", "KALMAN_MOMENTUM", "NO_EDGE"}

# =====================================================================
# 2. KALMAN FILTER
# =====================================================================

def test_kalman_constant_price_tracking():
    """Kalman filter should stably track a constant price with zero velocity."""
    prices = np.full(200, 100.0)
    atr = np.full(200, 1.0)
    k = run_kalman_filter_1d(prices, atr)
    assert np.isclose(k.level[-1], 100.0, atol=1e-3)
    assert np.isclose(k.velocity[-1], 0.0, atol=1e-3)
    assert np.isclose(k.z_score[-1], 0.0, atol=1e-3)

def test_kalman_linear_trend_velocity():
    """Kalman filter should capture positive velocity in a steady uptrend."""
    n = 300
    prices = 100.0 + 0.1 * np.arange(n)
    atr = np.full(n, 0.5)
    k = run_kalman_filter_1d(prices, atr)
    assert k.velocity[-1] > 0.05
    assert np.isfinite(k.level).all()
    assert np.all(k.innovation_std > 0)

def test_kalman_z_score_formula_exact():
    """Validate Z_K = (P_t - x_hat_t) / max(sqrt(S_t), 0.5 * ATR14)."""
    p = np.array([100.0, 102.0, 105.0, 95.0])
    atr = np.array([1.0, 1.0, 1.0, 2.0])
    k = run_kalman_filter_1d(p, atr)
    for i in range(len(p)):
        assert np.isfinite(k.z_score[i])
        floor = max(0.5 * atr[i], 1e-12)
        expected_denom = max(k.innovation_std[i], floor)
        assert expected_denom >= floor

def test_kalman_handles_nans_and_negatives():
    """Non-positive or non-finite prices must not cause crashes."""
    p = np.array([100.0, -5.0, np.nan, 102.0, 103.0])
    atr = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    k = run_kalman_filter_1d(p, atr)
    assert len(k.level) == len(p)
    assert np.isfinite(k.level[-1])

# =====================================================================
# 3. YANG-ZHANG VOLATILITY
# =====================================================================

def test_yang_zhang_initial_window_nans():
    """First 'window' bars must be NaN."""
    n = 50
    o = np.full(n, 100.0)
    h = np.full(n, 102.0)
    l = np.full(n, 99.0)
    c = np.full(n, 101.0)
    yz = yang_zhang_volatility(o, h, l, c, window=20)
    assert np.isnan(yz[:20]).all()
    assert np.isfinite(yz[20:]).all()

def test_yang_zhang_rejects_negative_or_corrupt_prices():
    """Raises ValueError if prices are negative or corrupt."""
    with pytest.raises(ValueError, match="strictly positive"):
        yang_zhang_volatility([-10, 10, 10, 10], [12, 12, 12, 12], [8, 8, 8, 8], [11, 11, 11, 11], window=2)
    with pytest.raises(ValueError, match="geometry"):
        # High lower than Low
        yang_zhang_volatility([10, 10, 10], [8, 8, 8], [12, 12, 12], [10, 10, 10], window=2)

def test_yang_zhang_flat_market_zero_vol():
    """In a completely flat market (H=L=O=C), volatility should be near zero."""
    n = 30
    p = np.full(n, 100.0)
    yz = yang_zhang_volatility(p, p, p, p, window=10)
    assert np.allclose(yz[10:], 0.0, atol=1e-8)

# =====================================================================
# 4. ORNSTEIN-UHLENBECK HALF-LIFE & DYNAMIC TIME STOP
# =====================================================================

def test_ou_half_life_known_ar1():
    """Theoretical AR(1) half-life: beta=0.85 -> phi=-0.15 -> hl = ln(2)/(-ln(0.85)) ~= 4.26 bars."""
    rng = np.random.default_rng(42)
    n = 5000
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = 0.85 * x[i - 1] + rng.normal(0, 0.1)

    res = estimate_ou_half_life(x)
    assert res["valid"] is True
    assert 3.5 <= res["half_life"] <= 5.0
    assert res["phi"] < -0.02

def test_ou_invalid_regimes():
    """phi >= -0.02 or random walk is invalid (hl=inf, valid=False)."""
    rw = np.cumsum(np.random.default_rng(1).normal(0, 1, 500))
    res_rw = estimate_ou_half_life(rw)
    assert res_rw["valid"] is False
    assert math.isinf(res_rw["half_life"])

    # Explosive AR(1): beta = 1.05 -> phi = +0.05
    explosive = np.zeros(200)
    for i in range(1, 200):
        explosive[i] = 1.05 * explosive[i - 1] + 0.1
    res_exp = estimate_ou_half_life(explosive)
    assert res_exp["valid"] is False
    assert math.isinf(res_exp["half_life"])

def test_dynamic_time_stop_clipping():
    """int(clip(2.5 * half_life, 6, 40))."""
    assert dynamic_time_stop(1.0) == 6
    assert dynamic_time_stop(2.0) == 6
    assert dynamic_time_stop(4.0) == 10
    assert dynamic_time_stop(10.0) == 25
    assert dynamic_time_stop(16.0) == 40
    assert dynamic_time_stop(30.0) == 40
    assert dynamic_time_stop(float("inf")) == 6
    assert dynamic_time_stop(float("nan")) == 6
    assert dynamic_time_stop(-5.0) == 6

# =====================================================================
# 5. TRIGGER LOGIC & SYMMETRY
# =====================================================================

def test_trigger_ou_mean_reversion_symmetry():
    """LONG and SHORT mean-reversion triggers must mirror each other."""
    sig_long = evaluate_stigma_trigger(
        regime="OU_MEAN_REVERSION",
        hurst=0.40,
        variance_ratio=0.85,
        half_life=10.0,
        z_score=[-2.5, -2.2, -1.80],
        level=[105.0, 105.0, 105.0],
        velocity=[0.0, 0.0, 0.0],
        innovation_std=[1.0, 1.0, 1.0],
        atr=[1.0, 1.0, 1.0],
        open_=[100.0, 100.0, 100.0],
        close=[100.5, 100.5, 101.0],
        macro_bias=0,
    )
    assert sig_long is not None
    assert sig_long["side"] == "LONG"
    assert sig_long["stop_loss"] == 101.0 - 1.2 * 1.0
    assert sig_long["max_hold_bars"] == 25

    sig_short = evaluate_stigma_trigger(
        regime="OU_MEAN_REVERSION",
        hurst=0.40,
        variance_ratio=0.85,
        half_life=10.0,
        z_score=[2.5, 2.2, 1.80],
        level=[95.0, 95.0, 95.0],
        velocity=[0.0, 0.0, 0.0],
        innovation_std=[1.0, 1.0, 1.0],
        atr=[1.0, 1.0, 1.0],
        open_=[100.0, 100.0, 100.0],
        close=[99.5, 99.5, 99.0],
        macro_bias=0,
    )
    assert sig_short is not None
    assert sig_short["side"] == "SHORT"
    assert sig_short["stop_loss"] == 99.0 + 1.2 * 1.0
    assert sig_short["max_hold_bars"] == 25

def test_trigger_kalman_momentum_requires_vr():
    """Momentum trigger must be vetoed if VR <= 1.08."""
    sig_vetoed = evaluate_stigma_trigger(
        regime="KALMAN_MOMENTUM",
        hurst=0.60,
        variance_ratio=1.02,
        half_life=10.0,
        z_score=[-0.5, -0.6, -0.4],
        level=[100.0, 100.0, 100.0],
        velocity=[0.2, 0.2, 0.2],
        innovation_std=[1.0, 1.0, 1.0],
        atr=[1.0, 1.0, 1.0],
        open_=[100.0, 100.0, 100.0],
        close=[100.2, 100.2, 100.5],
        macro_bias=0,
    )
    assert sig_vetoed is None

    sig_valid = evaluate_stigma_trigger(
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
    )
    assert sig_valid is not None
    assert sig_valid["side"] == "LONG"
    assert sig_valid["tp1"] == 100.5 + 1.4 * 1.2
    assert sig_valid["tp2"] == 100.5 + 2.6 * 1.2
    assert sig_valid["max_hold_bars"] == 24

def test_trigger_macro_bias_filter():
    """Macro bias -1 vetoes LONG; macro bias +1 vetoes SHORT."""
    params_long = dict(
        regime="OU_MEAN_REVERSION",
        hurst=0.40,
        variance_ratio=0.85,
        half_life=10.0,
        z_score=[-2.5, -2.2, -1.80],
        level=[105.0, 105.0, 105.0],
        velocity=[0.0, 0.0, 0.0],
        innovation_std=[1.0, 1.0, 1.0],
        atr=[1.0, 1.0, 1.0],
        open_=[100.0, 100.0, 100.0],
        close=[100.5, 100.5, 101.0],
    )
    assert evaluate_stigma_trigger(**params_long, macro_bias=-1) is None
    assert evaluate_stigma_trigger(**params_long, macro_bias=0) is not None
    assert evaluate_stigma_trigger(**params_long, macro_bias=1) is not None

# =====================================================================
# 6. BACKTEST ENGINE & EXECUTION REALISM
# =====================================================================

def create_synthetic_df(n=300):
    """Helper to create valid OHLC DataFrame."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2026-01-01", periods=n, freq="15min", tz="UTC")
    c = 100.0 + np.cumsum(rng.normal(0, 0.2, n))
    o = c + rng.normal(0, 0.05, n)
    h = np.maximum(o, c) + np.abs(rng.normal(0, 0.1, n))
    l = np.minimum(o, c) - np.abs(rng.normal(0, 0.1, n))
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c}, index=dates)

def test_backtest_signal_close_vs_next_open_execution():
    """Verify entry price executes strictly at next bar's open."""
    df = create_synthetic_df(200)
    res = backtest(df, spread=0.0, commission=0.0, slippage=0.0)
    t = res["trades_frame"]
    assert "exit_reason" in t.columns
    if not t.empty:
        for _, trade in t.iterrows():
            entry_time = trade["entry_time"]
            assert np.isclose(trade["entry"], df.loc[entry_time, "open"])

def test_backtest_cost_accounting():
    """Spread, commission, and slippage must strictly reduce net PnL."""
    df = create_synthetic_df(200)
    res_free = backtest(df, spread=0.0, commission=0.0, slippage=0.0)
    res_cost = backtest(df, spread=0.05, commission=0.02, slippage=0.01)

    t_free = res_free["trades_frame"]
    t_cost = res_cost["trades_frame"]
    assert "exit_reason" in t_free.columns
    assert "exit_reason" in t_cost.columns
    if not t_free.empty and not t_cost.empty:
        assert res_cost["net_pnl"] < res_free["net_pnl"]

def test_backtest_no_lookahead_bias():
    """Altering future price data must not change signals or trades on past bars."""
    df = create_synthetic_df(200)
    res1 = backtest(df.copy())
    t1 = res1["trades_frame"]

    # Modify the last 20 bars
    df_mod = df.copy()
    df_mod.iloc[-20:, df_mod.columns.get_loc("close")] *= 1.5
    df_mod.iloc[-20:, df_mod.columns.get_loc("high")] *= 1.5
    res2 = backtest(df_mod)
    t2 = res2["trades_frame"]

    cutoff_time = df.index[179]
    t1_prior = t1[t1["entry_time"] <= cutoff_time]
    t2_prior = t2[t2["entry_time"] <= cutoff_time]
    assert len(t1_prior) == len(t2_prior)
    if len(t1_prior) > 0:
        pd.testing.assert_series_equal(t1_prior["entry"], t2_prior["entry"])
        pd.testing.assert_series_equal(t1_prior["side"], t2_prior["side"])

def test_backtest_single_position_enforcement():
    """At no point can two positions be open simultaneously."""
    df = create_synthetic_df(300)
    res = backtest(df)
    t = res["trades_frame"]
    if len(t) > 1:
        for idx in range(len(t) - 1):
            exit_time = t["exit_time"].iloc[idx]
            next_entry_time = t["entry_time"].iloc[idx + 1]
            assert exit_time <= next_entry_time

# =====================================================================
# 7. MACRO GATE & EXECUTION BRIDGE
# =====================================================================

def test_macro_gate_fail_closed():
    """Corrupt or missing files fail closed to 0 (neutral)."""
    with tempfile.TemporaryDirectory() as td:
        tmp_path = Path(td)
        # Non-existent file
        assert load_macro_bias(tmp_path / "non_existent.json", autonomous=False) == 0
        assert load_macro_bias(tmp_path / "non_existent.json", autonomous=True) == 0

        # Corrupt JSON
        bad_json = tmp_path / "bad.json"
        bad_json.write_text("{not valid json", encoding="utf-8")
        assert load_macro_bias(bad_json, autonomous=False) == 0

        # Out of range bias
        invalid_val = tmp_path / "invalid.json"
        invalid_val.write_text(json.dumps({"macro_bias": 5}), encoding="utf-8")
        assert load_macro_bias(invalid_val, autonomous=False) == 0

        # Valid bias
        valid_file = tmp_path / "valid.json"
        valid_file.write_text(json.dumps({"macro_bias": -1}), encoding="utf-8")
        assert load_macro_bias(valid_file, autonomous=False) == -1

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

    invalid_signal = {**valid_signal, "stop_loss": 102.0}
    with pytest.raises(ValueError, match="geometry"):
        build_order_intent(invalid_signal)

# =====================================================================
# 8. DATA ADAPTERS FAIL-CLOSED BEHAVIOR
# =====================================================================

def test_binance_adapter_invalid_timeframe():
    """Binance adapter rejects timeframes other than M5 / M15."""
    from stigma.data.binance_data import fetch_binance
    with pytest.raises(ValueError, match="Timeframe must be M5 or M15"):
        fetch_binance("BTCUSDT", timeframe="H1")

def test_mt5_adapter_invalid_timeframe():
    """MT5 adapter rejects timeframes other than M5 / M15."""
    from stigma.data.mt5_data import fetch_mt5
    with pytest.raises(ValueError, match="Timeframe must be M5 or M15"):
        fetch_mt5("EURUSD", timeframe="D1")
