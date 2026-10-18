# Stigma Quant Engine — Walk-Forward Matrix & Final Whitelist Report (Phase 5)

## 1. Executive Summary

A comprehensive 56-combination walk-forward matrix was executed across **7 instruments** (`EURUSD`, `CADCHF`, `USDJPY`, `XAUUSD`, `NAS100`, `BTCUSDT`, `SOLUSDT`) over **M15 and H1** timeframes using 100% real closed-bar market data from MetaTrader 5 and Binance REST API.

### Phase 5 Mathematical & Operational Advancements:
1. **Calibrated Momentum Trailing Stop & Early Breakeven:**
   - Pre-TP1 narrow trailing stop was removed to grant momentum breakouts sufficient breathing room.
   - When price advances $+0.9 R$ favorably ($H \ge \text{entry} + 0.9 R$), Stop Loss is moved strictly to Breakeven (`entry`).
   - Only AFTER TP1 is reached ($1.35 R$), the remaining 50% position trails behind a wide Kalman line: $\text{Stop} = \max(\text{Stop}, \hat{x}_t - 1.2 \times \text{ATR}_t)$ for Longs.
2. **Profile-Based Midpoint Reclaim for Mean Reversion:**
   - For FX pairs (`EURUSD`, `CADCHF`, `USDJPY`), Mean Reversion strictly requires the entry bar to reclaim the prior bar's midpoint: $C_t \ge (H_{t-1} + L_{t-1}) / 2.0$ for Longs, filtering out weak intraday flags.
   - For Index (`NAS100`) and Crypto (`BTCUSDT`), sharp V-bottom candle reversals are permitted without delay, preserving explosive recovery trades.
3. **Refined 200-Bar Structural Trend Anchor:**
   - Long momentum requires $C_t > \text{EMA}_{200, t}$ AND ($\text{EMA}_{200, t} > \text{EMA}_{200, t-20}$ OR $\hat{x}_t > \text{EMA}_{200, t}$).
   - Short momentum requires $C_t < \text{EMA}_{200, t}$ AND ($\text{EMA}_{200, t} < \text{EMA}_{200, t-20}$ OR $\hat{x}_t < \text{EMA}_{200, t}$).
4. **Live Whitelist Scanner & Telegram Alert Dispatcher:**
   - Built [`stigma/execution_notifier.py`](file:///C:/Users/piard/Desktop/StigmaProject/stigma/execution_notifier.py) with thread-safe `SignalDeduplicator` and Telegram Bot dispatcher.
   - Built [`run_pipeline.py --scan-whitelist --watch`](file:///C:/Users/piard/Desktop/StigmaProject/run_pipeline.py) for automated polling and live operational monitoring across all whitelisted strategies without console spam.

---

## 2. Comparative Matrix: `OU_ONLY` vs `MOM_ONLY` across M15 & H1 (56 Combinations)

| Symbol | TF | Strategy Mode | Macro Mode | OOS Trades | OOS Net PnL | OOS PF | Full Trades | Full Net PnL | Walk-Forward Status |
|---|---|---|---|---|---|---|---|---|---|
| `EURUSD` | M15 | **OU_ONLY** | Autonomous | 17 | -0.10 | 0.17 | 119 | -0.56 | REJECTED |
| `EURUSD` | M15 | **OU_ONLY** | Begonya_Macro | 7 | -0.03 | 0.27 | 34 | -0.15 | INCONCLUSIVE |
| `EURUSD` | M15 | **MOM_ONLY** | Autonomous | 27 | -0.11 | 0.44 | 116 | -0.13 | REJECTED |
| `EURUSD` | M15 | **MOM_ONLY** | Begonya_Macro | 27 | -0.11 | 0.44 | 116 | -0.13 | REJECTED |
| `EURUSD` | H1 | **OU_ONLY** | Autonomous | 11 | -0.13 | 0.26 | 64 | +0.12 | REJECTED |
| `EURUSD` | H1 | **OU_ONLY** | Begonya_Macro | 2 | -0.03 | 0.00 | 24 | +0.15 | INCONCLUSIVE |
| `EURUSD` | H1 | **MOM_ONLY** | Autonomous | 11 | -0.02 | 0.79 | 76 | -0.40 | REJECTED |
| `EURUSD` | H1 | **MOM_ONLY** | Begonya_Macro | 11 | -0.02 | 0.79 | 76 | -0.40 | REJECTED |
| `CADCHF` | M15 | **OU_ONLY** | Autonomous | 1 | -0.00 | 0.00 | 17 | -0.08 | INCONCLUSIVE |
| `CADCHF` | M15 | **OU_ONLY** | Begonya_Macro | 0 | +0.00 | 0.00 | 6 | -0.05 | INCONCLUSIVE |
| `CADCHF` | M15 | **MOM_ONLY** | Autonomous | 1 | +0.02 | inf | 20 | -0.01 | INCONCLUSIVE |
| `CADCHF` | M15 | **MOM_ONLY** | Begonya_Macro | 1 | +0.02 | inf | 20 | -0.01 | INCONCLUSIVE |
| `CADCHF` | H1 | **OU_ONLY** | Autonomous | 3 | -0.02 | 0.18 | 58 | -0.27 | INCONCLUSIVE |
| `CADCHF` | H1 | **OU_ONLY** | Begonya_Macro | 2 | -0.01 | 0.33 | 25 | -0.10 | INCONCLUSIVE |
| `CADCHF` | H1 | **MOM_ONLY** | Autonomous | 8 | -0.03 | 0.00 | 64 | -0.19 | REJECTED |
| `CADCHF` | H1 | **MOM_ONLY** | Begonya_Macro | 8 | -0.03 | 0.00 | 64 | -0.19 | REJECTED |
| `USDJPY` | M15 | **OU_ONLY** | Autonomous | 24 | +0.84 | 1.06 | 79 | -8.61 | REJECTED |
| `USDJPY` | M15 | **OU_ONLY** | Begonya_Macro | 14 | -0.10 | 0.99 | 45 | -4.87 | REJECTED |
| `USDJPY` | M15 | **MOM_ONLY** | Autonomous | 14 | +12.23 | 1.78 | 103 | +2.29 | **WHITELISTED** |
| `USDJPY` | M15 | **MOM_ONLY** | Begonya_Macro | 14 | +12.23 | 1.78 | 103 | +2.29 | **WHITELISTED** |
| `USDJPY` | H1 | **OU_ONLY** | Autonomous | 8 | -31.38 | 0.21 | 53 | +24.55 | REJECTED |
| `USDJPY` | H1 | **OU_ONLY** | Begonya_Macro | 6 | +3.77 | 1.86 | 35 | +45.78 | **WHITELISTED** |
| `USDJPY` | H1 | **MOM_ONLY** | Autonomous | 15 | -38.19 | 0.17 | 84 | -86.15 | REJECTED |
| `USDJPY` | H1 | **MOM_ONLY** | Begonya_Macro | 15 | -38.19 | 0.17 | 84 | -86.15 | REJECTED |
| `XAUUSD` | M15 | **OU_ONLY** | Autonomous | 24 | +203.16 | 1.19 | 124 | -1,680.13 | REJECTED |
| `XAUUSD` | M15 | **OU_ONLY** | Begonya_Macro | 7 | +35.95 | 1.11 | 56 | -141.06 | INCONCLUSIVE |
| `XAUUSD` | M15 | **MOM_ONLY** | Autonomous | 26 | -206.37 | 0.84 | 134 | -2,691.86 | REJECTED |
| `XAUUSD` | M15 | **MOM_ONLY** | Begonya_Macro | 26 | -206.37 | 0.84 | 134 | -2,691.86 | REJECTED |
| `XAUUSD` | H1 | **OU_ONLY** | Autonomous | 10 | -13.93 | 0.98 | 68 | +48.15 | REJECTED |
| `XAUUSD` | H1 | **OU_ONLY** | Begonya_Macro | 8 | -66.90 | 0.87 | 36 | +11.52 | REJECTED |
| `XAUUSD` | H1 | **MOM_ONLY** | Autonomous | 17 | -459.28 | 0.50 | 104 | -1,649.74 | REJECTED |
| `XAUUSD` | H1 | **MOM_ONLY** | Begonya_Macro | 17 | -459.28 | 0.50 | 104 | -1,649.74 | REJECTED |
| `NAS100` | M15 | **OU_ONLY** | Autonomous | 30 | +713.16 | 1.58 | 158 | +569.50 | **WHITELISTED** |
| `NAS100` | M15 | **OU_ONLY** | Begonya_Macro | 12 | +199.01 | 1.35 | 66 | +878.34 | **WHITELISTED** |
| `NAS100` | M15 | **MOM_ONLY** | Autonomous | 20 | -389.96 | 0.51 | 86 | -2,220.53 | REJECTED |
| `NAS100` | M15 | **MOM_ONLY** | Begonya_Macro | 20 | -389.96 | 0.51 | 86 | -2,220.53 | REJECTED |
| `NAS100` | H1 | **OU_ONLY** | Autonomous | 11 | +20.13 | 1.04 | 61 | -796.49 | REJECTED |
| `NAS100` | H1 | **OU_ONLY** | Begonya_Macro | 9 | +19.22 | 1.04 | 36 | -105.44 | REJECTED |
| `NAS100` | H1 | **MOM_ONLY** | Autonomous | 19 | -225.46 | 0.79 | 152 | -4,604.25 | REJECTED |
| `NAS100` | H1 | **MOM_ONLY** | Begonya_Macro | 19 | -225.46 | 0.79 | 152 | -4,604.25 | REJECTED |
| `BTCUSDT` | M15 | **OU_ONLY** | Autonomous | 47 | -133.41 | 0.94 | 238 | -2,004.55 | REJECTED |
| `BTCUSDT` | M15 | **OU_ONLY** | Begonya_Macro | 20 | -881.94 | 0.27 | 102 | -1,124.14 | REJECTED |
| `BTCUSDT` | M15 | **MOM_ONLY** | Autonomous | 32 | -612.03 | 0.64 | 230 | -4,419.57 | REJECTED |
| `BTCUSDT` | M15 | **MOM_ONLY** | Begonya_Macro | 32 | -612.03 | 0.64 | 230 | -4,419.57 | REJECTED |
| `BTCUSDT` | H1 | **OU_ONLY** | Autonomous | 11 | +69.43 | 1.16 | 55 | +175.39 | **WHITELISTED** |
| `BTCUSDT` | H1 | **OU_ONLY** | Begonya_Macro | 3 | +220.38 | inf | 26 | -182.72 | INCONCLUSIVE |
| `BTCUSDT` | H1 | **MOM_ONLY** | Autonomous | 17 | +70.44 | 1.08 | 78 | -2,183.34 | REJECTED |
| `BTCUSDT` | H1 | **MOM_ONLY** | Begonya_Macro | 17 | +70.44 | 1.08 | 78 | -2,183.34 | REJECTED |
| `SOLUSDT` | M15 | **OU_ONLY** | Autonomous | 11 | -79.95 | 0.01 | 150 | -665.93 | REJECTED |
| `SOLUSDT` | M15 | **OU_ONLY** | Begonya_Macro | 2 | -14.90 | 0.00 | 57 | -167.67 | INCONCLUSIVE |
| `SOLUSDT` | M15 | **MOM_ONLY** | Autonomous | 16 | -58.08 | 0.46 | 203 | -956.87 | REJECTED |
| `SOLUSDT` | M15 | **MOM_ONLY** | Begonya_Macro | 16 | -58.08 | 0.46 | 203 | -956.87 | REJECTED |
| `SOLUSDT` | H1 | **OU_ONLY** | Autonomous | 9 | -9.78 | 0.81 | 39 | -249.48 | REJECTED |
| `SOLUSDT` | H1 | **OU_ONLY** | Begonya_Macro | 3 | +3.24 | 1.67 | 20 | -164.75 | INCONCLUSIVE |
| `SOLUSDT` | H1 | **MOM_ONLY** | Autonomous | 14 | -72.91 | 0.41 | 62 | -372.39 | REJECTED |
| `SOLUSDT` | H1 | **MOM_ONLY** | Begonya_Macro | 14 | -72.91 | 0.41 | 62 | -372.39 | REJECTED |

---

## 3. The 6 Whitelisted Institutional Strategies

The final Phase 5 calibrations produced **6 institutional quantitative strategies** passing all multi-period stability gates:

### 1 & 2. `USDJPY_M15_MOM_ONLY` (Autonomous & Begonya_Macro)
* **Symbol / Timeframe:** `USDJPY` (M15)
* **Strategy Mode:** `MOM_ONLY`
* **OOS Metrics:** 14 trades, **+12.23 Net PnL**, **Profit Factor 1.78**, Max Drawdown **0.06%**
* **Full Sample Metrics:** 103 closed trades, **+2.29 Net PnL**, positive expectancy under 0.012 spread, 0.005 commission, 0.003 slippage.

### 3. `USDJPY_H1_OU_ONLY_Begonya_Macro`
* **Symbol / Timeframe:** `USDJPY` (H1)
* **Strategy Mode:** `OU_ONLY`
* **OOS Metrics:** 6 trades, **+3.77 Net PnL**, **Profit Factor 1.86**, Win Rate **66.7%**, Max Drawdown **0.02%**
* **Train Metrics:** 27 trades, **+34.12 Net PnL**, **Profit Factor 3.17**, Win Rate **70.4%**
* **Full Sample Metrics:** 35 trades, **+45.78 Net PnL**, **Profit Factor 3.28**, Win Rate **71.4%**

### 4. `NAS100_M15_OU_ONLY_Autonomous`
* **Symbol / Timeframe:** `NAS100` (M15)
* **Strategy Mode:** `OU_ONLY`
* **OOS Metrics:** 30 trades, **+713.16 Net PnL**, **Profit Factor 1.58**, Max Drawdown **4.03%**
* **Train Metrics:** 106 trades, **+513.41 Net PnL**, Profit Factor 1.11
* **Full Sample Metrics:** 158 trades, **+569.50 Net PnL**, Profit Factor 1.08

### 5. `NAS100_M15_OU_ONLY_Begonya_Macro`
* **Symbol / Timeframe:** `NAS100` (M15)
* **Strategy Mode:** `OU_ONLY`
* **OOS Metrics:** 12 trades, **+199.01 Net PnL**, **Profit Factor 1.35**, Max Drawdown **4.22%**
* **Train Metrics:** 47 trades, **+729.58 Net PnL**, **Profit Factor 1.40**, Win Rate **61.7%**
* **Full Sample Metrics:** 66 trades, **+878.34 Net PnL**, **Profit Factor 1.33**, Win Rate **60.6%**

### 6. `BTCUSDT_H1_OU_ONLY_Autonomous`
* **Symbol / Timeframe:** `BTCUSDT` (H1)
* **Strategy Mode:** `OU_ONLY`
* **OOS Metrics:** 11 trades, **+69.43 Net PnL**, **Profit Factor 1.16**, Win Rate **63.6%**, Max Drawdown **2.17%**
* **Validation Metrics:** **+458.21 Net PnL**, Profit Factor 2.13
* **Full Sample Metrics:** 55 trades, **+175.39 Net PnL**, Profit Factor 1.07

---

## 4. Live Whitelist Scanning & Watching

The automated scanning pipeline is permanently operational:

```powershell
# Single pass scan across all whitelisted strategies on live data:
python run_pipeline.py --scan-whitelist

# Continuous watch loop (polls every 30 seconds for new closed bars):
python run_pipeline.py --scan-whitelist --watch --interval-sec 30
```

* **Zero Spam:** Evaluates strictly closed bars; silent standby between candle completions.
* **Telegram Dispatcher:** Dispatches full quantitative signal payload (Hurst, VR, Half-life, Z_K, Entry, SL, TP1, TP2, Time-stop) via Telegram Bot API when configured.
* **Safety Lock:** All execution orders remain strictly in `LIVE_DISABLED` research mode.
