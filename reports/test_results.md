# Stigma Quant Engine — Automated Test Suite Results (Phase 2)

## 1. Test Execution Summary

* **Date:** 2026-10-18
* **Test Runner:** `pytest 9.1.1` (Python 3.12.10, 64-bit Windows)
* **Configuration:** `pyproject.toml` (with causal testpaths and `pythonpath = ["."]`)
* **Total Tests Executed:** 21
* **Passed:** 21
* **Failed:** 0
* **Errors:** 0
* **Execution Time:** ~0.81 seconds

```powershell
python -m pytest tests/test_stigma_quant.py -v
# Result: 21 passed in 0.81s (100% pass rate)
```

---

## 2. Phase 2 Test Catalog and Verification Details

| # | Test Identifier | Module / Feature | Mathematical & Behavioral Invariants Verified | Result |
|---|-----------------|------------------|-----------------------------------------------|--------|
| 1 | `test_hurst_insufficient_data_fails_closed` | Hurst / VR | Returns `INSUFFICIENT_DATA` when $N < 32$ or inputs are corrupt | PASSED |
| 2 | `test_hurst_random_walk_veto_boundaries` | Hurst / VR | Enforces veto when $0.47 \le H \le 0.53$ and $0.95 \le VR \le 1.05$ | PASSED |
| 3 | `test_hurst_persistent_vs_antipersistent_ordering` | Hurst / VR | Validates directional regime partitioning across synthetic series | PASSED |
| 4 | `test_adaptive_kalman_dynamic_noise` | Adaptive Kalman | Dynamically scales $R_i \propto \text{ATR}_i^2$ and $Q_i = 0.02 R_i$ across volatility changes | PASSED |
| 5 | `test_adaptive_kalman_constant_tracking` | Adaptive Kalman | Smoothly converges to static price with zero velocity and stable innovations | PASSED |
| 6 | `test_adaptive_kalman_zscore_formula` | Adaptive Kalman | Exact verification of $Z_K = (P_t - \hat{x}_t) / \max(\sqrt{S_t}, 0.5 \times \text{ATR}_{14})$ | PASSED |
| 7 | `test_yang_zhang_initial_window_nans` | Yang–Zhang | Asserts strict causal initialization where initial $w$ bars are NaN | PASSED |
| 8 | `test_yang_zhang_rejects_corrupt_prices` | Yang–Zhang | Throws `ValueError` for negative prices or invalid bar geometry | PASSED |
| 9 | `test_ou_half_life_120_window` | Ornstein–Uhlenbeck | Calibrated to 120-bar window and bounds valid mean-reversion within $[2.0, 40.0]$ bars | PASSED |
| 10 | `test_trigger_net_rr_gate_rejects_low_rr` | Net R:R Gate | Vetoes entries with `"LOW_RR"` when net $\text{TP2} < 1.6 R$ | PASSED |
| 11 | `test_trigger_high_cost_to_sl_rejection` | Cost Defense | Vetoes entries with `"HIGH_COST_TO_SL"` when $\text{risk} < 4.0 \times \text{cost\_price}$ | PASSED |
| 12 | `test_trigger_dynamic_tp1_relocation` | Net R:R Gate | Relocates $\text{TP1} \ge 1.15 R$ when raw Kalman level is too close ($< 1.1 R$) | PASSED |
| 13 | `test_trigger_kalman_momentum_requires_expanding_yz` | Momentum Trigger | Vetoes momentum if Yang–Zhang volatility is not expanding ($\le \text{median}_{50}$) | PASSED |
| 14 | `test_session_filter_mt5_and_binance` | Session Armor | Rejects entries outside London/NY UTC hours (MT5: 06-20 UTC, Binance: 08-22 UTC) | PASSED |
| 15 | `test_weekend_close_forced_exit` | Weekend Armor | Forces closure of open MT5 positions on Friday at 21:30 UTC | PASSED |
| 16 | `test_backtest_funnel_counts_telemetry` | Funnel Diagnostics | Accurately accounts for every bar evaluated and logs filter rejection tallies | PASSED |
| 17 | `test_backtest_causal_next_open_execution` | Execution Causal | Strictly executes entries at the next bar's Open, eliminating lookahead bias | PASSED |
| 18 | `test_macro_gate_fail_closed` | Macro Gate | Corrupt or missing macro JSON fails closed to neutral (`0`) | PASSED |
| 19 | `test_execution_bridge_order_intent_validation` | Execution Bridge | Verifies price geometry constraints and ensures live mode remains permanently disabled | PASSED |
| 20 | `test_binance_adapter_invalid_timeframe` | Binance Data | Rejects unsupported timeframes with descriptive error | PASSED |
| 21 | `test_mt5_adapter_invalid_timeframe` | MT5 Data | Rejects unsupported timeframes with descriptive error | PASSED |

---

## 3. Conclusion

All 21 unit tests pass deterministically in 0.81s. All 4 structural defects identified in Phase 1 (Net R:R deficiency, non-adaptive Kalman, Hurst/VR dual-fractal bottleneck, missing session/weekend armor) and complete funnel telemetry are 100% validated.
