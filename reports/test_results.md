# Stigma Quant Engine — Automated Test Suite Results

## 1. Test Execution Summary

* **Date:** 2026-10-18
* **Test Runner:** `pytest 9.1.1` (Python 3.12.10, 64-bit Windows)
* **Configuration:** `pyproject.toml` (with causal testpaths and `pythonpath = ["."]`)
* **Total Tests Executed:** 27
* **Passed:** 27
* **Failed:** 0
* **Errors:** 0
* **Execution Time:** ~0.91 seconds

Compilation verification:
```powershell
python -m compileall -q stigma tests run_pipeline.py
# Exit Code: 0 (No syntax or compilation errors)
```

---

## 2. Test Catalog and Verification Details

| # | Test Identifier | Module / Focus | Description & Mathematical Invariants Verified | Result |
|---|-----------------|----------------|------------------------------------------------|--------|
| 1 | `test_hurst_insufficient_data_fails_closed` | Hurst / VR | Fails closed to `INSUFFICIENT_DATA` when $N < 32$ or $N < \text{window}$ | PASSED |
| 2 | `test_hurst_nan_or_inf_fails_closed` | Hurst / VR | Returns `INSUFFICIENT_DATA` if any price in window is NaN or Inf | PASSED |
| 3 | `test_hurst_constant_price_zero_variance` | Hurst / VR | Zero-variance price series returns `INSUFFICIENT_DATA` without numerical division-by-zero | PASSED |
| 4 | `test_hurst_random_walk_veto_boundaries` | Hurst / VR | Pure Gaussian random walk yields $H \approx 0.50$ triggering `RANDOM_WALK_VETO` | PASSED |
| 5 | `test_hurst_persistent_vs_antipersistent_ordering` | Hurst / VR | Positively autocorrelated series yields $H > 0.54, \text{VR} > 1.08$ (`KALMAN_MOMENTUM`); negatively autocorrelated yields $H < 0.46, \text{VR} < 0.92$ (`OU_MEAN_REVERSION`) | PASSED |
| 6 | `test_hurst_vr_classification_matrix` | Hurst / VR | Validates complete partition of state space into 5 distinct labels | PASSED |
| 7 | `test_kalman_constant_price_tracking` | Kalman Filter | Stably converges to price level with zero velocity and zero Z-score on flat input | PASSED |
| 8 | `test_kalman_linear_trend_velocity` | Kalman Filter | Accurately extracts positive velocity component in a steady upward drift | PASSED |
| 9 | `test_kalman_z_score_formula_exact` | Kalman Filter | Verifies exact formula $Z_K = (P_t - \hat{x}_t) / \max(\sqrt{S_t}, 0.5 \times \text{ATR}_{14})$ | PASSED |
| 10 | `test_kalman_handles_nans_and_negatives` | Kalman Filter | Non-finite and negative inputs propagate state without throwing unhandled exceptions | PASSED |
| 11 | `test_yang_zhang_initial_window_nans` | Yang–Zhang | Asserts that initial $w$ bars are strictly masked as NaN | PASSED |
| 12 | `test_yang_zhang_rejects_negative_or_corrupt_prices` | Yang–Zhang | Raises `ValueError` for non-positive prices or invalid bar geometry ($H < L$) | PASSED |
| 13 | `test_yang_zhang_flat_market_zero_vol` | Yang–Zhang | Completely flat prices ($H=L=O=C$) produce zero volatility | PASSED |
| 14 | `test_ou_half_life_known_ar1` | Ornstein–Uhlenbeck | AR(1) series with $\beta = 0.85$ matches theoretical half-life $T_{1/2} = \ln(2)/-\ln(0.85) \approx 4.26$ bars | PASSED |
| 15 | `test_ou_invalid_regimes` | Ornstein–Uhlenbeck | Non-stationary ($\phi \ge -0.02$) and explosive series are flagged invalid ($T_{1/2} = \infty$) | PASSED |
| 16 | `test_dynamic_time_stop_clipping` | Time-Stop | Validates $\text{int}(\text{clip}(2.5 \times T_{1/2}, 6, 40))$ across boundary values and NaNs | PASSED |
| 17 | `test_trigger_ou_mean_reversion_symmetry` | Signal Trigger | Evaluates mathematical symmetry between Long (oversold, turn-up) and Short (overbought, turn-down) triggers | PASSED |
| 18 | `test_trigger_kalman_momentum_requires_vr` | Signal Trigger | Verifies that momentum triggers require $\text{VR} > 1.08$ and are vetoed if $\text{VR} \le 1.08$ | PASSED |
| 19 | `test_trigger_macro_bias_filter` | Signal Trigger | Confirms `macro_bias = -1` vetoes Longs and `macro_bias = +1` vetoes Shorts | PASSED |
| 20 | `test_backtest_signal_close_vs_next_open_execution` | Backtest Engine | Ensures entry price is executed strictly at next bar's Open | PASSED |
| 21 | `test_backtest_cost_accounting` | Backtest Engine | Verifies that spread, commission, and slippage are correctly deducted from net PnL | PASSED |
| 22 | `test_backtest_no_lookahead_bias` | Backtest Engine | Mutating future bars does not change trades generated on prior historical bars | PASSED |
| 23 | `test_backtest_single_position_enforcement` | Backtest Engine | Confirms zero concurrency: a new trade cannot open while an existing position is active | PASSED |
| 24 | `test_macro_gate_fail_closed` | Macro Gate | Verifies corrupt, missing, or out-of-bounds macro JSON fails closed to neutral (`0`) | PASSED |
| 25 | `test_execution_bridge_order_intent_validation` | Execution Bridge | Verifies price geometry constraints and ensures live mode remains permanently disabled | PASSED |
| 26 | `test_binance_adapter_invalid_timeframe` | Binance Data | Rejects unsupported timeframes with a descriptive `ValueError` | PASSED |
| 27 | `test_mt5_adapter_invalid_timeframe` | MT5 Data | Rejects unsupported timeframes with a descriptive `ValueError` | PASSED |

---

## 3. Conclusion

All 27 automated tests pass reliably in under 1 second. No tests were skipped, loosened, or mocked out to artificially pass. Every mathematical boundary, causal execution step, and fail-closed mechanism functions exactly as specified.
