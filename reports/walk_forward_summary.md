# Stigma Quant Engine — Walk-Forward Matrix & Whitelist Audit

## 1. Executive Summary

A comprehensive 28-combination walk-forward backtest matrix was conducted using **100% real closed-bar market data**:
* **MetaTrader 5 (MT5):** 50,000 bars each for `EURUSD`, `CADCHF`, `USDJPY`, `XAUUSD`, and `NAS100` across `M5` and `M15` timeframes.
* **Binance REST API:** 100,000 closed klines each for `BTCUSDT` and `SOLUSDT` across `M5` and `M15` timeframes.
* **Execution Modes:** Both **Autonomous Mode** (`macro_bias = 0`) and **Begonya Macro-Anchored Mode** were tested across all 14 instrument/timeframe combinations ($14 \times 2 = 28$ test runs).

All simulations enforced:
1. Strict chronologically separated splits: **Train (60%)**, **Validation (20%)**, and **Untouched Out-Of-Sample (20%)**.
2. Realistic broker transaction friction: live spreads, broker commission fees, and adverse execution slippage.
3. Causal next-bar Open fills with dynamic SL/TP entry shifts, gap stops, and conservative intrabar conflict priority.

---

## 2. Walk-Forward Matrix Results (28 Combinations)

| # | Combination | Source | TF | Mode | Total Bars | OOS Trades | OOS Win% | OOS Profit Factor | OOS Net PnL | Full Net PnL | Status |
|---|-------------|--------|----|------|------------|------------|----------|-------------------|-------------|--------------|--------|
| 1 | `EURUSD_M5_Autonomous` | MT5 | M5 | Autonomous | 49,999 | 5 | 40.0% | 0.27 | -0.02 | -0.03 | **INCONCLUSIVE** |
| 2 | `EURUSD_M5_Begonya_Macro` | MT5 | M5 | Begonya_Macro | 49,999 | 4 | 50.0% | 0.41 | -0.01 | -0.01 | **INCONCLUSIVE** |
| 3 | `EURUSD_M15_Autonomous` | MT5 | M15 | Autonomous | 49,999 | 14 | 42.9% | 1.51 | +0.03 | -0.03 | **INCONCLUSIVE** |
| 4 | `EURUSD_M15_Begonya_Macro` | MT5 | M15 | Begonya_Macro | 49,999 | 5 | 80.0% | 6.83 | +0.03 | -0.04 | **INCONCLUSIVE** |
| 5 | `CADCHF_M5_Autonomous` | MT5 | M5 | Autonomous | 49,999 | 7 | 14.3% | 0.01 | -0.02 | -0.06 | **INCONCLUSIVE** |
| 6 | `CADCHF_M5_Begonya_Macro` | MT5 | M5 | Begonya_Macro | 49,999 | 3 | 0.0% | 0.00 | -0.01 | -0.03 | **INCONCLUSIVE** |
| 7 | `CADCHF_M15_Autonomous` | MT5 | M15 | Autonomous | 49,999 | 0 | 0.0% | 0.00 | 0.00 | -0.05 | **INCONCLUSIVE** |
| 8 | `CADCHF_M15_Begonya_Macro` | MT5 | M15 | Begonya_Macro | 49,999 | 0 | 0.0% | 0.00 | 0.00 | -0.01 | **INCONCLUSIVE** |
| 9 | `USDJPY_M5_Autonomous` | MT5 | M5 | Autonomous | 49,999 | 7 | 14.3% | 0.28 | -3.33 | -31.68 | **INCONCLUSIVE** |
| 10 | `USDJPY_M5_Begonya_Macro` | MT5 | M5 | Begonya_Macro | 49,999 | 6 | 16.7% | 0.32 | -2.79 | -27.92 | **INCONCLUSIVE** |
| 11 | `USDJPY_M15_Autonomous` | MT5 | M15 | Autonomous | 49,999 | 11 | 36.4% | 0.66 | -3.77 | -6.44 | **INCONCLUSIVE** |
| 12 | `USDJPY_M15_Begonya_Macro` | MT5 | M15 | Begonya_Macro | 49,999 | 9 | 44.4% | 0.99 | -0.09 | -8.36 | **INCONCLUSIVE** |
| 13 | `XAUUSD_M5_Autonomous` | MT5 | M5 | Autonomous | 49,999 | 16 | 31.2% | 0.52 | -305.09 | -714.88 | **REJECTED** |
| 14 | `XAUUSD_M5_Begonya_Macro` | MT5 | M5 | Begonya_Macro | 49,999 | 8 | 12.5% | 0.06 | -394.04 | -596.98 | **INCONCLUSIVE** |
| 15 | `XAUUSD_M15_Autonomous` | MT5 | M15 | Autonomous | 49,999 | 11 | 45.5% | 1.12 | +77.16 | +1,236.79 | **INCONCLUSIVE** |
| 16 | `XAUUSD_M15_Begonya_Macro` | MT5 | M15 | Begonya_Macro | 49,999 | 6 | 50.0% | 1.49 | +152.46 | +893.93 | **INCONCLUSIVE** |
| 17 | `NAS100_M5_Autonomous` | MT5 | M5 | Autonomous | 49,999 | 15 | 46.7% | 1.37 | +332.06 | -899.23 | **REJECTED** |
| 18 | `NAS100_M5_Begonya_Macro` | MT5 | M5 | Begonya_Macro | 49,999 | 6 | 33.3% | 0.89 | -44.32 | -441.69 | **INCONCLUSIVE** |
| 19 | `NAS100_M15_Autonomous` | MT5 | M15 | Autonomous | 49,999 | 14 | 57.1% | 1.63 | +420.08 | -3,007.98 | **INCONCLUSIVE** |
| 20 | `NAS100_M15_Begonya_Macro` | MT5 | M15 | Begonya_Macro | 49,999 | 3 | 33.3% | 0.89 | -23.79 | -1,495.20 | **INCONCLUSIVE** |
| 21 | `BTCUSDT_M5_Autonomous` | BINANCE | M5 | Autonomous | 100,000 | 22 | 31.8% | 0.56 | -757.91 | -6,206.16 | **REJECTED** |
| 22 | `BTCUSDT_M5_Begonya_Macro` | BINANCE | M5 | Begonya_Macro | 100,000 | 8 | 25.0% | 0.27 | -555.40 | -2,086.54 | **INCONCLUSIVE** |
| 23 | `BTCUSDT_M15_Autonomous` | BINANCE | M15 | Autonomous | 99,999 | 21 | 28.6% | 0.30 | -1,486.79 | -4,240.34 | **REJECTED** |
| 24 | `BTCUSDT_M15_Begonya_Macro` | BINANCE | M15 | Begonya_Macro | 99,999 | 11 | 27.3% | 0.28 | -806.73 | -803.40 | **INCONCLUSIVE** |
| 25 | `SOLUSDT_M5_Autonomous` | BINANCE | M5 | Autonomous | 99,999 | 11 | 9.1% | 0.01 | -44.26 | -153.08 | **INCONCLUSIVE** |
| 26 | `SOLUSDT_M5_Begonya_Macro` | BINANCE | M5 | Begonya_Macro | 99,999 | 5 | 0.0% | 0.00 | -21.85 | -68.89 | **INCONCLUSIVE** |
| 27 | `SOLUSDT_M15_Autonomous` | BINANCE | M15 | Autonomous | 99,999 | 33 | 30.3% | 0.28 | -92.04 | -272.37 | **REJECTED** |
| 28 | `SOLUSDT_M15_Begonya_Macro` | BINANCE | M15 | Begonya_Macro | 99,999 | 15 | 40.0% | 0.27 | -37.52 | -145.15 | **REJECTED** |

---

## 3. Whitelist Decision & Quantitative Analysis

### A. Strict Whitelisting Thresholds (Phase 7)
According to the specification:
1. **Out-of-sample positive expectation** after real broker spreads, commissions, and slippage.
2. **Acceptable drawdown** ($< 25\%$).
3. **Adequate trade count** ($\ge 20$ trades out-of-sample).
4. **Walk-forward stability:** No extreme divergence between Train, Validation, and OOS periods. A strategy that loses heavily in Train/Val and only gains in OOS must be rejected as an unstationary fluke / multi-trial bias.

### B. In-Depth Evaluation of Notable Candidates

#### 1. `NAS100_M5_Autonomous` (Rejected)
* **Out-of-Sample Performance:** 15 trades, Net PnL $+332.06$, Profit Factor $1.37$.
* **Train & Validation Performance:**
  * Train Net PnL: $-1,149.16$ (54 trades, $\text{PF} = 0.69$).
  * Validation Net PnL: $-261.54$ (10 trades, $\text{PF} = 0.63$).
  * Full Historical PnL: $-899.23$.
* **Audit Verdict:** **REJECTED**. The positive OOS performance was a localized regime anomaly over a small sample of 15 trades. Over the preceding 64 trades (Train + Validation), the strategy consistently eroded capital. Passing only a single fold while failing prior folds violates walk-forward stability.

#### 2. `XAUUSD_M15_Autonomous` & `XAUUSD_M15_Begonya_Macro` (Inconclusive)
* **Performance:** Positive full PnL ($+1,236.79$ full, $+77.16$ OOS).
* **Sample Size Limitation:** Produced only 11 trades in OOS (Autonomous) and 6 trades (Macro).
* **Audit Verdict:** **INCONCLUSIVE**. With only 6–11 out-of-sample trades across nearly a year of M15 data, the sample size is statistically inadequate to reject the null hypothesis of zero edge.

#### 3. Cryptocurrency Feeds (`BTCUSDT`, `SOLUSDT`) (Rejected / Inconclusive)
* **Frictional Drag:** On M5 and M15 crypto data, 0.04% taker fee each way (0.08% round-trip) imposes an immense hurdle. With average trade holding periods of 8–15 bars, transaction costs consumed $100\%$ of gross returns, leading to steep negative profit factors ($0.27 - 0.56$).
* **Audit Verdict:** **REJECTED** on high-sample candidates (e.g. `BTCUSDT_M5_Autonomous`, `BTCUSDT_M15_Autonomous`, `SOLUSDT_M15`).

### C. Final Whitelist State
```json
"whitelisted_strategies": []
```
* **Status:** `EMPTY_WHITELIST`
* **Integrity Guarantee:** In accordance with Phase 7 ("Yeterli kanıt yoksa boş whitelist kabul edilebilir"), Stigma maintains strict institutional standards. Zero strategies are falsely promoted to whitelist status without multi-period, statistically significant, cost-adjusted proof.
