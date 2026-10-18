"""Walk-forward test matrix runner for Stigma Quant Engine (Faz 3).

Evaluates 7 instruments (EURUSD, CADCHF, USDJPY, XAUUSD, NAS100, BTCUSDT, SOLUSDT)
across M15 and H1 timeframes under:
  - Regime Isolation: OU_ONLY (Mean Reversion) vs MOM_ONLY (Kalman Momentum)
  - Execution Modes: Autonomous vs Begonya Macro-Anchored
Chronological split: 60% Train, 20% Validation, 20% Untouched Out-Of-Sample.
Enforces realistic broker costs (spread, commission, slippage), session armor, and full funnel telemetry.
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
        "bars": {"M15": 50000, "H1": 35000},
    },
    "BINANCE": {
        "symbols": ["BTCUSDT", "SOLUSDT"],
        "bars": {"M15": 100000, "H1": 35000},
    },
}

TIMEFRAMES = ["M15", "H1"]
STRATEGY_MODES = ["OU_ONLY", "MOM_ONLY"]
MACRO_MODES = ["Autonomous", "Begonya_Macro"]

def evaluate_wf_split(
    df: pd.DataFrame,
    cost: dict,
    macro_bias: int,
    tf_min: int,
    session_type: str = "auto",
    strategy_mode: str = "BOTH",
) -> dict:
    """Split data chronologically into Train (60%), Val (20%), OOS (20%) with Funnel Diagnostics."""
    n = len(df)
    i_train = int(n * 0.60)
    i_val = int(n * 0.80)

    df_train = df.iloc[:i_train]
    df_val = df.iloc[i_train:i_val]
    df_oos = df.iloc[i_val:]

    res_train = backtest(
        df_train,
        spread=cost["spread"],
        commission=cost["commission"],
        slippage=cost["slippage"],
        macro_bias=macro_bias,
        timeframe_minutes=tf_min,
        session_type=session_type,
        strategy_mode=strategy_mode,
    )
    res_val = backtest(
        df_val,
        spread=cost["spread"],
        commission=cost["commission"],
        slippage=cost["slippage"],
        macro_bias=macro_bias,
        timeframe_minutes=tf_min,
        session_type=session_type,
        strategy_mode=strategy_mode,
    )
    res_oos = backtest(
        df_oos,
        spread=cost["spread"],
        commission=cost["commission"],
        slippage=cost["slippage"],
        macro_bias=macro_bias,
        timeframe_minutes=tf_min,
        session_type=session_type,
        strategy_mode=strategy_mode,
    )
    res_full = backtest(
        df,
        spread=cost["spread"],
        commission=cost["commission"],
        slippage=cost["slippage"],
        macro_bias=macro_bias,
        timeframe_minutes=tf_min,
        session_type=session_type,
        strategy_mode=strategy_mode,
    )

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
        "funnel_counts": res_full.get("funnel_counts", {}),
    }

def run_all_matrices() -> dict:
    results = {}
    data_cache: dict[str, pd.DataFrame] = {}

    # 1. Fetch MT5 Data
    logger.info("=== STEP 1: Fetching MT5 Data (M15 & H1) ===")
    for sym in DATA_TARGETS["MT5"]["symbols"]:
        for tf in TIMEFRAMES:
            key = f"MT5_{sym}_{tf}"
            target_bars = DATA_TARGETS["MT5"]["bars"][tf]
            try:
                t0 = time.time()
                df = fetch_mt5(sym, timeframe=tf, bars=target_bars)
                data_cache[key] = df
                logger.info("Fetched %s: %d bars in %.2fs (%s to %s)", key, len(df), time.time() - t0, df.index[0], df.index[-1])
            except Exception as exc:
                logger.error("Failed fetching %s: %s", key, exc)
                data_cache[key] = None

    # 2. Fetch Binance Data
    logger.info("=== STEP 2: Fetching Binance Data (M15 & H1) ===")
    for sym in DATA_TARGETS["BINANCE"]["symbols"]:
        for tf in TIMEFRAMES:
            key = f"BINANCE_{sym}_{tf}"
            target_bars = DATA_TARGETS["BINANCE"]["bars"][tf]
            try:
                t0 = time.time()
                df = fetch_binance(sym, timeframe=tf, bars=target_bars, pause=0.03)
                data_cache[key] = df
                logger.info("Fetched %s: %d bars in %.2fs (%s to %s)", key, len(df), time.time() - t0, df.index[0], df.index[-1])
            except Exception as exc:
                logger.error("Failed fetching %s: %s", key, exc)
                data_cache[key] = None

    # 3. Execute Matrix Combinations
    total_combos = len(DATA_TARGETS["MT5"]["symbols"] + DATA_TARGETS["BINANCE"]["symbols"]) * len(TIMEFRAMES) * len(STRATEGY_MODES) * len(MACRO_MODES)
    logger.info("=== STEP 3: Executing Walk-Forward Matrix (%d Combinations) ===", total_combos)
    combo_id = 0

    all_symbols = [
        ("MT5", s) for s in DATA_TARGETS["MT5"]["symbols"]
    ] + [
        ("BINANCE", s) for s in DATA_TARGETS["BINANCE"]["symbols"]
    ]

    for source, sym in all_symbols:
        cost = COST_SPEC[sym]
        session_type = "binance" if source == "BINANCE" else "mt5"
        for tf in TIMEFRAMES:
            tf_min = 15 if tf == "M15" else 60
            key = f"{source}_{sym}_{tf}"
            df = data_cache.get(key)

            for strat in STRATEGY_MODES:
                for mode in MACRO_MODES:
                    combo_id += 1
                    combo_name = f"{sym}_{tf}_{strat}_{mode}"
                    logger.info("[%d/%d] Running %s (%s)", combo_id, total_combos, combo_name, source)

                    if df is None or len(df) < 500:
                        results[combo_name] = {
                            "status": "BLOCKED",
                            "reason": f"Data unavailable or insufficient for {key}",
                            "symbol": sym,
                            "timeframe": tf,
                            "strategy_mode": strat,
                            "mode": mode,
                            "source": source,
                        }
                        continue

                    if mode == "Autonomous":
                        macro_bias = 0
                    else:
                        # Begonya macro anchored: 200 SMA long-term directional anchor
                        sma200 = df["close"].rolling(200).mean().iloc[-1]
                        macro_bias = 1 if df["close"].iloc[-1] > sma200 else -1

                    t_start = time.time()
                    wf_data = evaluate_wf_split(
                        df,
                        cost,
                        macro_bias=macro_bias,
                        tf_min=tf_min,
                        session_type=session_type,
                        strategy_mode=strat,
                    )
                    elapsed = time.time() - t_start

                    # Whitelist Multi-Period Decision Rules:
                    # 1. Out-of-sample trades >= 10
                    # 2. Out-of-sample Net PnL > 0
                    # 3. Out-of-sample Profit Factor > 1.15
                    # 4. Out-of-sample Max Drawdown < 25%
                    # 5. Multi-period stability: Train Net PnL >= 0 and Full Net PnL >= 0
                    oos = wf_data["oos"]
                    train = wf_data["train"]
                    full = wf_data["full"]

                    trades_oos = wf_data["trades_oos_count"]
                    oos_net = oos.get("net_pnl", 0.0)
                    oos_pf = oos.get("profit_factor", 0.0)
                    oos_dd = oos.get("max_drawdown_pct", 1.0)
                    train_net = train.get("net_pnl", 0.0)
                    full_net = full.get("net_pnl", 0.0)

                    if trades_oos >= 10 and oos_net > 0 and oos_pf > 1.15 and oos_dd < 0.25 and full_net > 0 and train_net > 0:
                        decision = "WHITELISTED"
                    elif trades_oos < 10:
                        decision = "INCONCLUSIVE"
                    else:
                        decision = "REJECTED"

                    fc = wf_data.get("funnel_counts", {})
                    results[combo_name] = {
                        "status": decision,
                        "symbol": sym,
                        "timeframe": tf,
                        "strategy_mode": strat,
                        "mode": mode,
                        "source": source,
                        "macro_bias": macro_bias,
                        "execution_time_sec": round(elapsed, 2),
                        "cost_parameters": cost,
                        "walk_forward": wf_data,
                        "funnel_summary": {
                            "bars_evaluated": fc.get("bars_evaluated", 0),
                            "veto_session_hours": fc.get("veto_session_hours", 0),
                            "veto_random_walk": fc.get("veto_random_walk", 0),
                            "regime_ou_mr": fc.get("regime_ou_mr_bars", 0),
                            "regime_kalman_mom": fc.get("regime_kalman_mom_bars", 0),
                            "passed_ou_hl": fc.get("passed_ou_halflife", 0),
                            "passed_yz_vol": fc.get("passed_yz_vol", 0),
                            "passed_z_setup": fc.get("passed_zscore_setup", 0),
                            "rejected_high_cost_to_sl": fc.get("rejected_high_cost_to_sl", 0),
                            "rejected_low_rr": fc.get("rejected_low_rr", 0),
                            "trades_opened": fc.get("trades_opened", 0),
                        },
                    }
                    logger.info(
                        "-> Decision: %s | OOS Trades: %d | OOS PnL: %.2f | OOS PF: %.2f | Full PnL: %.2f | Funnel Open: %d",
                        decision, trades_oos, oos_net, oos_pf, full_net, fc.get("trades_opened", 0)
                    )

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
