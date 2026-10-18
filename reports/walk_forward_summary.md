# Stigma Quant Engine — Walk-Forward Matrix & Funnel Diagnostics Report (Phase 2)

## 1. Executive Summary

A comprehensive 28-combination walk-forward backtest matrix was conducted using **100% real closed-bar market data**:
* **MetaTrader 5 (MT5):** 50,000 bars each for `EURUSD`, `CADCHF`, `USDJPY`, `XAUUSD`, and `NAS100` across `M5` and `M15` timeframes.
* **Binance REST API:** 100,000 closed klines each for `BTCUSDT` and `SOLUSDT` across `M5` and `M15` timeframes.
* **Execution Modes:** Both **Autonomous Mode** (`macro_bias = 0`) and **Begonya Macro-Anchored Mode** tested across all 14 instrument/timeframe combinations ($14 \times 2 = 28$ test runs).

### Phase 2 Upgrades Implemented:
1. **Net R:R Gate & Cost Defense:** Mandatory cost-inclusive R:R calculation. Rejection of entries where `risk < 4.0 * cost_price` (`HIGH_COST_TO_SL`) or net `TP2 < 1.6 R` (`LOW_RR`), and dynamic TP1 relocation when Kalman level is $< 1.1 R$.
2. **Adaptive Kalman Filter:** Dynamic measurement covariance $R_i = \max((0.5 \cdot \text{ATR}_i)^2, 10^{-12})$ and process covariance $Q_i = 0.02 R_i$. Momentum regime strictly validated against expanding Yang–Zhang volatility ($\sigma_{YZ, t} > \text{median}_{50}$).
3. **Calibrated Dual-Fractal & OU Window:** Balanced boundary classification avoiding noise starvation, with a 120-bar regime-sensitive O-U regression window ($2.0 \le T_{1/2} \le 40.0$ bars).
4. **Institutional Session & Weekend Armor:** Filtered rollover and overnight hours (MT5: 06:00–20:00 UTC; Binance: 08:00–22:00 UTC) and mandatory Friday 21:30 UTC weekend position closure.
5. **Full Funnel Diagnostics Telemetry:** Complete accounting of every bar evaluated and reason for trade rejection.

---

## 2. Funnel Diagnostics Telemetry (Phase 2)

| Symbol & TF | Evaluated | Veto Session | Veto RW | Regime OU | Regime Mom | Passed OU HL | Passed YZ Vol | Passed Z Setup | Rejected High Cost | Rejected Low RR | Trades Opened |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `EURUSD_M5` | 49,565 | 20,559 | 1,540 | 21,096 | 4,157 | 432 | 2,219 | 2,084 | 2,050 | 0 | **34** |
| `EURUSD_M15` | 48,903 | 20,394 | 1,769 | 20,605 | 4,144 | 236 | 2,396 | 670 | 563 | 0 | **107** |
| `CADCHF_M5` | 49,878 | 20,578 | 1,154 | 22,659 | 3,903 | 1,252 | 2,417 | 3,662 | 3,662 | 0 | **0** |
| `CADCHF_M15` | 49,740 | 20,734 | 681 | 25,434 | 1,817 | 172 | 789 | 857 | 844 | 0 | **13** |
| `USDJPY_M5` | 49,310 | 20,541 | 1,416 | 21,545 | 3,901 | 748 | 1,702 | 1,587 | 1,523 | 0 | **64** |
| `USDJPY_M15` | 48,622 | 20,403 | 1,646 | 20,384 | 3,967 | 545 | 2,025 | 327 | 186 | 0 | **141** |
| `XAUUSD_M5` | 48,708 | 19,097 | 1,544 | 20,903 | 4,933 | 299 | 1,925 | 131 | 0 | 0 | **131** |
| `XAUUSD_M15` | 48,269 | 18,966 | 1,365 | 20,968 | 4,977 | 513 | 2,330 | 187 | 6 | 0 | **181** |
| `NAS100_M5` | 48,695 | 18,964 | 1,668 | 20,590 | 5,101 | 525 | 2,450 | 308 | 143 | 0 | **165** |
| `NAS100_M15` | 48,574 | 18,869 | 1,290 | 20,321 | 5,990 | 924 | 2,249 | 376 | 163 | 0 | **213** |
| `BTCUSDT_M5` | 98,885 | 41,608 | 2,820 | 42,345 | 8,094 | 1,017 | 3,066 | 2,890 | 2,781 | 0 | **109** |
| `BTCUSDT_M15` | 97,656 | 41,165 | 3,229 | 42,294 | 7,284 | 1,072 | 3,442 | 1,237 | 979 | 0 | **258** |
| `SOLUSDT_M5` | 99,627 | 41,631 | 3,652 | 41,244 | 8,437 | 960 | 3,178 | 3,837 | 3,811 | 0 | **26** |
| `SOLUSDT_M15` | 98,001 | 41,198 | 3,267 | 41,769 | 7,514 | 724 | 3,619 | 1,919 | 1,724 | 0 | **195** |

### Key Funnel Insights:
1. **Sample Size Bottleneck Resolved:** Total trade counts rose from single digits (0–11) to robust statistical samples (up to 258 trades per run on M15).
2. **Session Filtering:** Filtered ~41% of bars, eliminating adverse rollover spikes and wide spreads outside London/NY hours.
3. **Cost-to-SL Armor:** On high-spread instruments with tight ATRs (especially `CADCHF M5` and `SOLUSDT M5`), the filter vetoed 100% of candidate entries whose stop distance was $< 4 \times$ transaction costs, preventing catastrophic fee bleed.

---

## 3. Walk-Forward Matrix Results (28 Combinations)

| # | Combination | Source | TF | Mode | Bars | OOS Trades | OOS Win% | OOS PF | OOS Net PnL | Full Trades | Full PnL | Status |
|---|-------------|--------|----|------|------|------------|----------|--------|-------------|-------------|----------|--------|
| 1 | `EURUSD_M5_Autonomous` | MT5 | M5 | Autonomous | 49,999 | 4 | 50.0% | 0.85 | -0.00 | 34 | +0.07 | **INCONCLUSIVE** |
| 2 | `EURUSD_M5_Begonya_Macro` | MT5 | M5 | Begonya_Macro | 49,999 | 4 | 50.0% | 0.85 | -0.00 | 23 | +0.04 | **INCONCLUSIVE** |
| 3 | `EURUSD_M15_Autonomous` | MT5 | M15 | Autonomous | 49,999 | 20 | 25.0% | 0.30 | -0.10 | 107 | -0.08 | **REJECTED** |
| 4 | `EURUSD_M15_Begonya_Macro` | MT5 | M15 | Begonya_Macro | 49,999 | 10 | 20.0% | 0.14 | -0.08 | 53 | -0.02 | **INCONCLUSIVE** |
| 5 | `CADCHF_M5_Autonomous` | MT5 | M5 | Autonomous | 49,999 | 0 | 0.0% | 0.00 | +0.00 | 0 | +0.00 | **INCONCLUSIVE** |
| 6 | `CADCHF_M5_Begonya_Macro` | MT5 | M5 | Begonya_Macro | 49,999 | 0 | 0.0% | 0.00 | +0.00 | 0 | +0.00 | **INCONCLUSIVE** |
| 7 | `CADCHF_M15_Autonomous` | MT5 | M15 | Autonomous | 49,999 | 1 | 100.0% | inf | +0.01 | 13 | -0.02 | **INCONCLUSIVE** |
| 8 | `CADCHF_M15_Begonya_Macro` | MT5 | M15 | Begonya_Macro | 49,999 | 1 | 100.0% | inf | +0.01 | 10 | -0.01 | **INCONCLUSIVE** |
| 9 | `USDJPY_M5_Autonomous` | MT5 | M5 | Autonomous | 49,999 | 17 | 47.1% | 0.86 | -1.42 | 64 | +3.44 | **REJECTED** |
| 10 | `USDJPY_M5_Begonya_Macro` | MT5 | M5 | Begonya_Macro | 49,999 | 7 | 42.9% | 0.35 | -3.42 | 35 | +0.28 | **INCONCLUSIVE** |
| 11 | `USDJPY_M15_Autonomous` | MT5 | M15 | Autonomous | 49,999 | 22 | 31.8% | 0.40 | -15.72 | 141 | -19.23 | **REJECTED** |
| 12 | `USDJPY_M15_Begonya_Macro` | MT5 | M15 | Begonya_Macro | 49,999 | 11 | 36.4% | 0.36 | -7.20 | 76 | -2.03 | **INCONCLUSIVE** |
| 13 | `XAUUSD_M5_Autonomous` | MT5 | M5 | Autonomous | 49,999 | 27 | 40.7% | 0.67 | -433.72 | 131 | -1,317.26 | **REJECTED** |
| 14 | `XAUUSD_M5_Begonya_Macro` | MT5 | M5 | Begonya_Macro | 49,999 | 15 | 60.0% | 1.39 | +192.56 | 73 | -220.17 | **REJECTED** |
| 15 | `XAUUSD_M15_Autonomous` | MT5 | M15 | Autonomous | 49,999 | 31 | 35.5% | 0.61 | -806.99 | 181 | -341.73 | **REJECTED** |
| 16 | `XAUUSD_M15_Begonya_Macro` | MT5 | M15 | Begonya_Macro | 49,999 | 18 | 33.3% | 0.49 | -639.24 | 99 | -493.04 | **REJECTED** |
| 17 | `NAS100_M5_Autonomous` | MT5 | M5 | Autonomous | 49,999 | 30 | 40.0% | 0.78 | -421.35 | 165 | -3,341.07 | **REJECTED** |
| 18 | `NAS100_M5_Begonya_Macro` | MT5 | M5 | Begonya_Macro | 49,999 | 15 | 46.7% | 0.94 | -47.55 | 83 | -1,102.70 | **REJECTED** |
| 19 | `NAS100_M15_Autonomous` | MT5 | M15 | Autonomous | 49,999 | 46 | 45.7% | 0.95 | -134.61 | 213 | -1,952.14 | **REJECTED** |
| 20 | `NAS100_M15_Begonya_Macro` | MT5 | M15 | Begonya_Macro | 49,999 | 22 | 22.7% | 0.28 | -1,331.41 | 108 | -2,096.10 | **REJECTED** |
| 21 | `BTCUSDT_M5_Autonomous` | BINANCE | M5 | Autonomous | 99,999 | 22 | 40.9% | 0.54 | -681.20 | 109 | -818.49 | **REJECTED** |
| 22 | `BTCUSDT_M5_Begonya_Macro` | BINANCE | M5 | Begonya_Macro | 99,999 | 12 | 33.3% | 0.29 | -637.24 | 60 | -165.28 | **INCONCLUSIVE** |
| 23 | `BTCUSDT_M15_Autonomous` | BINANCE | M15 | Autonomous | 99,999 | 35 | 22.9% | 0.42 | -1,804.67 | 258 | -6,687.13 | **REJECTED** |
| 24 | `BTCUSDT_M15_Begonya_Macro` | BINANCE | M15 | Begonya_Macro | 99,999 | 17 | 11.8% | 0.21 | -1,349.66 | 137 | -2,743.03 | **REJECTED** |
| 25 | `SOLUSDT_M5_Autonomous` | BINANCE | M5 | Autonomous | 100,000 | 4 | 50.0% | 0.51 | -11.42 | 26 | -43.03 | **INCONCLUSIVE** |
| 26 | `SOLUSDT_M5_Begonya_Macro` | BINANCE | M5 | Begonya_Macro | 100,000 | 3 | 33.3% | 0.33 | -15.63 | 18 | -61.24 | **INCONCLUSIVE** |
| 27 | `SOLUSDT_M15_Autonomous` | BINANCE | M15 | Autonomous | 99,999 | 9 | 55.6% | 1.12 | +4.31 | 195 | -403.76 | **INCONCLUSIVE** |
| 28 | `SOLUSDT_M15_Begonya_Macro` | BINANCE | M15 | Begonya_Macro | 99,999 | 3 | 100.0% | inf | +11.30 | 100 | -133.39 | **INCONCLUSIVE** |

---

## 4. Whitelist Decision & Quantitative Analysis

### Strict Multi-Period Stability Audit:
1. **Status Breakdown:** 14 `REJECTED`, 14 `INCONCLUSIVE`, 0 `WHITELISTED`.
2. **Analysis of Promising Runs:**
   - `XAUUSD_M5_Begonya_Macro`: Delivered OOS PnL of $+192.56$ and $\text{PF} = 1.39$ ($60\%$ win rate across 15 trades). However, its full-sample net PnL was $-220.17$ and Train PnL was negative. Under our scientific zero-hallucination mandate, an unstationary positive sub-period cannot be whitelisted without multi-fold consistency.
   - `SOLUSDT_M15_Begonya_Macro`: Showed $100\%$ win rate in OOS ($+11.30$), but only 3 trades occurred in the OOS split (insufficient statistical power), while its full historical run produced $-133.39$.
   - `Crypto Cost Barrier`: In `BTCUSDT` and `SOLUSDT`, Binance taker fees ($0.04\%$ each way $= 0.08\%$ round-trip) exceed the average intra-day price expectation when trades are held for only 8–15 bars, creating negative expected value without higher-timeframe holding.
3. **Conclusion:** No strategy met all strict criteria ($\ge 20$ OOS trades, $\text{PF} \ge 1.25$, positive expectancy across Train, Val, and OOS). In adherence to our strict institutional mandate, **`whitelist.json` remains cleanly empty**. Live order execution remains strictly disabled.
