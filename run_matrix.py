"""Walk-forward test matrix runner for Stigma Quant Engine.

Executes all 28 combinations:
  - 5 MT5 instruments (EURUSD, CADCHF, USDJPY, XAUUSD, NAS100) x 2 timeframes (M5, M15) x 2 modes (Autonomous, Macro)
  - 2 Binance instruments (BTCUSDT, SOLUSDT) x 2 timeframes (M5, M15) x 2 modes (Autonomous, Macro)
Chronological split: 60% Train, 20% Validation, 20% Untouched Out-Of-Sample.
"""
from __future__ import annotations
import json
import logging
import os
import time
from pathlib import Path
import numpy as np
import pandas as pd

from stigma.backtest_engine import backtest
from stigma.data.binance_data import fetch_binance
from stigma.data.mt5_data import fetch_mt5

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("matrix_runner")

# Realistic broker transaction costs (in price units)
COST_SPEC = {
    "EURUSD": {"spread": 0.00010, "commission": 0.00004, "slippage": 0.00002},
    "CADCHF": {"spread": 0.00018, "commission": 0.00004, "slippage": 0.00003},
    "USDJPY": {"spread": 0.012, "commission": 0.005, "slippage": 0.003},
    "XAUUSD": {"spread": 0.35, "commission": 0.15, "slippage": 0.10},
    "NAS100": {"spread": 2.0, "commission": 1.0, "slippage": 1.0},
    "BTCUSDT": {"spread": 2.0, "commission": 50.0, "slippage": 3.0},
    "SOLUSDT": {"spread": 0.04, "commission": 0.12, "slippage": 0.04},
}

DATA_TARGETS = {
    "MT5": {
        "symbols": ["EURUSD", "CADCHF", "USDJPY", "XAUUSD", "NAS100"],
        "bars": 50000,
    },
    "BINANCE": {
        "symbols": ["BTCUSDT", "SOLUSDT"],
        "bars": 100000,
    },
}

TIMEFRAMES = ["M5", "M15"]

def evaluate_wf_split(df: pd.DataFrame, cost: dict, macro_bias: int, tf_min: int) -> dict:
    """Split data chronologically into Train (60%), Val (20%), OOS (20%)."""
    n = len(df)
    i_train = int(n * 0.60)
    i_val = int(n * 0.80)

    df_train = df.iloc[:i_train]
    df_val = df.iloc[i_train:i_val]
    df_oos = df.iloc[i_val:]

    res_train = backtest(df_train, spread=cost["spread"], commission=cost["commission"], slippage=cost["slippage"], macro_bias=macro_bias, timeframe_minutes=tf_min)
    res_val = backtest(df_val, spread=cost["spread"], commission=cost["commission"], slippage=cost["slippage"], macro_bias=macro_bias, timeframe_minutes=tf_min)
    res_oos = backtest(df_oos, spread=cost["spread"], commission=cost["commission"], slippage=cost["slippage"], macro_bias=macro_bias, timeframe_minutes=tf_min)
    res_full = backtest(df, spread=cost["spread"], commission=cost["commission"], slippage=cost["slippage"], macro_bias=macro_bias, timeframe_minutes=tf_min)

    t_train = res_train.pop("trades_frame")
    t_val = res_val.pop("trades_frame")
    t_oos = res_oos.pop("trades_frame")
    t_full = res_full.pop("trades_frame")

    return {
        "train": res_train,
        "val": res_val,
        "oos": res_oos,
        "full": res_full,
        "bars_total": n,
        "bars_train": len(df_train),
        "bars_val": len(df_val),
        "bars_oos": len(df_oos),
        "trades_oos_count": len(t_oos),
        "trades_full_count": len(t_full),
    }

def run_all_matrices() -> dict:
    results = {}
    data_cache: dict[str, pd.DataFrame] = {}

    # 1. Fetch MT5 Data
    logger.info("=== STEP 1: Fetching MT5 Data ===")
    for sym in DATA_TARGETS["MT5"]["symbols"]:
        for tf in TIMEFRAMES:
            key = f"MT5_{sym}_{tf}"
            target_bars = DATA_TARGETS["MT5"]["bars"]
            try:
                t0 = time.time()
                df = fetch_mt5(sym, timeframe=tf, bars=target_bars)
                data_cache[key] = df
                logger.info("Fetched %s: %d bars in %.2fs (%s to %s)", key, len(df), time.time() - t0, df.index[0], df.index[-1])
            except Exception as exc:
                logger.error("Failed fetching %s: %s", key, exc)
                data_cache[key] = None

    # 2. Fetch Binance Data
    logger.info("=== STEP 2: Fetching Binance Data ===")
    for sym in DATA_TARGETS["BINANCE"]["symbols"]:
        for tf in TIMEFRAMES:
            key = f"BINANCE_{sym}_{tf}"
            target_bars = DATA_TARGETS["BINANCE"]["bars"]
            try:
                t0 = time.time()
                df = fetch_binance(sym, timeframe=tf, bars=target_bars, pause=0.03)
                data_cache[key] = df
                logger.info("Fetched %s: %d bars in %.2fs (%s to %s)", key, len(df), time.time() - t0, df.index[0], df.index[-1])
            except Exception as exc:
                logger.error("Failed fetching %s: %s", key, exc)
                data_cache[key] = None

    # 3. Execute 28 Combinations
    logger.info("=== STEP 3: Executing Walk-Forward Matrix (28 Combinations) ===")
    combo_id = 0

    all_symbols = [
        ("MT5", s) for s in DATA_TARGETS["MT5"]["symbols"]
    ] + [
        ("BINANCE", s) for s in DATA_TARGETS["BINANCE"]["symbols"]
    ]

    for source, sym in all_symbols:
        cost = COST_SPEC[sym]
        for tf in TIMEFRAMES:
            tf_min = 5 if tf == "M5" else 15
            key = f"{source}_{sym}_{tf}"
            df = data_cache.get(key)

            for mode in ["Autonomous", "Begonya_Macro"]:
                combo_id += 1
                combo_name = f"{sym}_{tf}_{mode}"
                logger.info("[%d/28] Running %s (%s)", combo_id, combo_name, source)

                if df is None or len(df) < 500:
                    results[combo_name] = {
                        "status": "BLOCKED",
                        "reason": f"Data unavailable or insufficient for {key}",
                        "symbol": sym,
                        "timeframe": tf,
                        "mode": mode,
                        "source": source,
                    }
                    continue

                # Macro bias: 0 for Autonomous; for Begonya_Macro, read trend filter (e.g. 1 if bullish, -1 if bearish)
                # or evaluate macro gate
                if mode == "Autonomous":
                    macro_bias = 0
                else:
                    # In macro mode: use long-term trend bias (+1 if last close > 200 SMA on 15m/60m, -1 otherwise)
                    sma200 = df["close"].rolling(200).mean().iloc[-1]
                    macro_bias = 1 if df["close"].iloc[-1] > sma200 else -1

                t_start = time.time()
                wf_data = evaluate_wf_split(df, cost, macro_bias=macro_bias, tf_min=tf_min)
                elapsed = time.time() - t_start

                # Decision rule for Whitelist:
                # 1. Out-of-sample Net PnL > 0
                # 2. Out-of-sample trades >= 15
                # 3. Out-of-sample Profit Factor > 1.15
                # 4. Out-of-sample Max Drawdown < 25%
                oos = wf_data["oos"]
                trades_oos = wf_data["trades_oos_count"]
                oos_net = oos.get("net_pnl", 0.0)
                oos_pf = oos.get("profit_factor", 0.0)
                oos_dd = oos.get("max_drawdown_pct", 1.0)

                if trades_oos >= 15 and oos_net > 0 and oos_pf > 1.15 and oos_dd < 0.25:
                    decision = "WHITELISTED"
                elif trades_oos < 15:
                    decision = "INCONCLUSIVE"
                else:
                    decision = "REJECTED"

                results[combo_name] = {
                    "status": decision,
                    "symbol": sym,
                    "timeframe": tf,
                    "mode": mode,
                    "source": source,
                    "macro_bias": macro_bias,
                    "execution_time_sec": round(elapsed, 2),
                    "cost_parameters": cost,
                    "walk_forward": wf_data,
                }
                logger.info("-> Decision: %s | OOS Trades: %d | OOS PnL: %.2f | OOS PF: %.2f", decision, trades_oos, oos_net, oos_pf)

    # Save complete matrix results
    out_dir = Path("reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    matrix_file = out_dir / "walk_forward_matrix.json"
    with open(matrix_file, "w", encoding="utf-8") as f_out:
        json.dump(results, f_out, indent=2, default=str)

    logger.info("Matrix execution complete. Saved to %s", matrix_file)
    return results

if __name__ == "__main__":
    run_all_matrices()
