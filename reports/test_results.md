# Stigma Quant Engine — Automated Test Suite Results (Phase 3)

## 1. Test Execution Summary

* **Date:** 2026-10-18
* **Test Runner:** `pytest 9.1.1` (Python 3.12.10, 64-bit Windows)
* **Configuration:** `pyproject.toml` (with causal testpaths and `pythonpath = ["."]`)
* **Total Tests Executed:** 25
* **Passed:** 25
* **Failed:** 0
* **Errors:** 0
* **Execution Time:** ~1.19 seconds

```powershell
python -m pytest tests/test_stigma_quant.py -v
# Result: 25 passed in 1.19s (100% pass rate)
```

---

## 2. Phase 3 Test Catalog and Verification Details

| # | Test Identifier | Module / Feature | Mathematical & Behavioral Invariants Verified | Result |
|---|-----------------|------------------|-----------------------------------------------|--------|
| 1 | `test_hurst_insufficient_data_fails_closed` | Hurst / VR | Returns `INSUFFICIENT_DATA` when $N < 32$ or inputs are corrupt | PASSED |
| 2 | `test_hurst_nan_or_inf_fails_closed` | Hurst / VR | Returns `INSUFFICIENT_DATA` if non-finite prices exist | PASSED |
| 3 | `test_hurst_constant_price_zero_variance` | Hurst / VR | Zero variance price series fails closed cleanly | PASSED |
| 4 | `test_relaxed_dual_fractal_classification_boundaries` | Hurst / VR | Validates boundary classification into state spaces | PASSED |
| 5 | `test_hurst_persistent_vs_antipersistent_ordering` | Hurst / VR | Validates directional regime partitioning across synthetic series | PASSED |
| 6 | `test_adaptive_kalman_dynamic_scaling` | Adaptive Kalman | Dynamically scales $R_i \propto \text{ATR}_i^2$ and $Q_i = 0.0005 R_i$ | PASSED |
| 7 | `test_adaptive_kalman_linear_trend_velocity` | Adaptive Kalman | Extracts steady trend velocity component accurately | PASSED |
| 8 | `test_adaptive_kalman_handles_nans` | Adaptive Kalman | Non-positive or non-finite inputs propagate without throwing | PASSED |
| 9 | `test_yang_zhang_initial_window_nans` | Yang–Zhang | Initial $w$ bars are strictly masked as NaN | PASSED |
| 10 | `test_yang_zhang_rejects_negative_or_corrupt_prices` | Yang–Zhang | Raises `ValueError` for negative prices or invalid bar geometry | PASSED |
| 11 | `test_ou_half_life_known_ar1` | Ornstein–Uhlenbeck | AR(1) series matches theoretical half-life value | PASSED |
| 12 | `test_ou_half_life_extended_bounds` | Ornstein–Uhlenbeck | Calibrated to 120-bar window and bounds valid mean-reversion within $[2.0, 40.0]$ bars | PASSED |
| 13 | `test_dynamic_time_stop_clipping` | Time-Stop | Validates $\text{int}(\text{clip}(2.5 \times T_{1/2}, 6, 40))$ | PASSED |
| 14 | `test_cost_to_sl_protection` | Cost Defense | Vetoes entries with `"HIGH_COST_TO_SL"` when $\text{risk} < 3.5 \times \text{cost\_price}$ | PASSED |
| 15 | `test_net_rr_gate_relocates_close_tp1` | Net R:R Gate | Relocates $\text{TP1} \ge 1.15 R$ when raw Kalman level is too close ($< 1.1 R$) | PASSED |
| 16 | `test_net_rr_gate_rejects_low_rr` | Net R:R Gate | Vetoes entries with `"LOW_RR"` when net $\text{TP2} < 1.6 R$ | PASSED |
| 17 | `test_kalman_momentum_yang_zhang_expansion_filter` | Momentum Trigger | Vetoes momentum if Yang–Zhang volatility is not expanding ($\le \text{median}_{50}$) | PASSED |
| 18 | `test_session_rules_mt5_london_ny` | Session Armor | Rejects entries outside London/NY UTC hours (MT5: 06-20 UTC, Binance: 08-22 UTC) | PASSED |
| 19 | `test_backtest_funnel_counts_structure` | Funnel Diagnostics | Accurately accounts for every bar evaluated and logs filter rejection tallies | PASSED |
| 20 | `test_macro_gate_fail_closed` | Macro Gate | Corrupt or missing macro JSON fails closed to neutral (`0`) | PASSED |
| 21 | `test_execution_bridge_order_intent_validation` | Execution Bridge | Verifies price geometry constraints and ensures live mode remains permanently disabled | PASSED |
| 22 | `test_empirical_spread_zscore_formula` | Kalman Z-Score | Validates $Z_{K, t} = (P_t - \hat{x}_t) / \max(\text{rolling\_spread\_std}_t, 1.0 \times \text{ATR}_{14, t})$ | PASSED |
| 23 | `test_kalman_momentum_breakout_and_pullback` | Momentum Trigger | Enforces pullback $-1.0 \le Z_K \le 0.25$ and candle breakout $C > H_{i-1}$ | PASSED |
| 24 | `test_backtest_by_regime_breakdown` | Regime Isolation | Asserts `by_regime` metrics exist for both `OU_MEAN_REVERSION` and `KALMAN_MOMENTUM` | PASSED |
| 25 | `test_h1_timeframe_adapters` | Timeframe Adapters | Confirms `H1` is supported in both MT5 and Binance without errors | PASSED |

---

## 3. Conclusion

All 25 unit tests pass deterministically in 1.19s with zero failures. All Phase 3 mathematical calibrations, empirical Z-score formulations, regime isolation filters, and higher-timeframe adapters are 100% verified.
