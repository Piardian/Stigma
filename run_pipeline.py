"""Run a local research backtest, walk-forward matrix, or live whitelist scanner.

Usage:
  - Standard Backtest: python run_pipeline.py --source mt5 --symbol EURUSD --timeframe M15
  - Walk-Forward Split: python run_pipeline.py --source binance --symbol BTCUSDT --walk-forward
  - Live Whitelist Scan: python run_pipeline.py --scan-whitelist
  - Live Whitelist Watch: python run_pipeline.py --scan-whitelist --watch --interval-sec 30
"""
from __future__ import annotations
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

from stigma.backtest_engine import backtest, prepare_features, check_session_rules
from stigma.data.binance_data import fetch_binance
from stigma.data.mt5_data import fetch_mt5
from stigma.execution_bridge import build_order_intent
from stigma.execution_notifier import SignalDeduplicator, send_telegram_alert
from stigma.macro_gate import load_macro_bias
from stigma.quant_engine import calculate_hurst_vr, estimate_ou_half_life, evaluate_stigma_trigger

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_pipeline")

def run_walk_forward(
    df: pd.DataFrame,
    *,
    spread: float = 0.0,
    commission: float = 0.0,
    slippage: float = 0.0,
    macro_bias: int = 0,
    timeframe_minutes: int = 15,
    train_ratio: float = 0.60,
    val_ratio: float = 0.20,
) -> dict:
    """Chronological, untouched 3-way split: Train (60%), Validation (20%), Out-of-Sample (20%)."""
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    df_train = df.iloc[:train_end]
    df_val = df.iloc[train_end:val_end]
    df_oos = df.iloc[val_end:]

    res_train = backtest(df_train, spread=spread, commission=commission, slippage=slippage, macro_bias=macro_bias, timeframe_minutes=timeframe_minutes)
    res_val = backtest(df_val, spread=spread, commission=commission, slippage=slippage, macro_bias=macro_bias, timeframe_minutes=timeframe_minutes)
    res_oos = backtest(df_oos, spread=spread, commission=commission, slippage=slippage, macro_bias=macro_bias, timeframe_minutes=timeframe_minutes)

    # Pop trades frames to avoid huge JSON output
    train_trades = res_train.pop("trades_frame")
    val_trades = res_val.pop("trades_frame")
    oos_trades = res_oos.pop("trades_frame")

    return {
        "train": res_train,
        "val": res_val,
        "oos": res_oos,
        "train_bars": len(df_train),
        "val_bars": len(df_val),
        "oos_bars": len(df_oos),
        "all_trades": pd.concat([train_trades, val_trades, oos_trades], ignore_index=True),
    }

def scan_whitelisted_strategies(
    whitelist_file: str | Path = "reports/whitelist.json",
    deduplicator: SignalDeduplicator | None = None,
    send_alerts: bool = True,
) -> tuple[list[dict], dict[str, str]]:
    """Scan all approved strategies from whitelist.json against live closed-bar market data."""
    wl_path = Path(whitelist_file)
    if not wl_path.exists():
        logger.error("Whitelist file not found: %s", wl_path)
        return [], {}

    with open(wl_path, "r", encoding="utf-8") as f:
        wl_data = json.load(f)

    strategies = wl_data.get("whitelisted_strategies", [])
    if not strategies:
        logger.warning("No whitelisted strategies found in %s", wl_path)
        return [], {}

    # Group strategies by (source, symbol, timeframe) to fetch market data once per asset/tf
    groups: dict[tuple[str, str, str], list[dict]] = {}
    for st in strategies:
        key = (st["source"].upper(), st["symbol"].upper(), st["timeframe"].upper())
        groups.setdefault(key, []).append(st)

    data_cache: dict[tuple[str, str, str], pd.DataFrame] = {}
    last_bar_times: dict[str, str] = {}
    active_signals: list[dict] = []
    summary_rows: list[dict] = []

    # 1. Fetch data
    for (src, sym, tf) in groups.keys():
        bars_needed = 350
        try:
            if src == "BINANCE":
                df = fetch_binance(sym, tf, bars_needed, pause=0.02)
            else:
                df = fetch_mt5(sym, tf, bars_needed)
            if df is not None and len(df) >= 150:
                data_cache[(src, sym, tf)] = df
                last_bar_times[f"{src}:{sym}:{tf}"] = str(df.index[-1])
            else:
                logger.warning("Insufficient data for %s %s %s", src, sym, tf)
        except Exception as exc:
            logger.error("Error fetching data for %s %s %s: %s", src, sym, tf, exc)

    # 2. Evaluate each whitelisted strategy
    for (src, sym, tf), strat_list in groups.items():
        df = data_cache.get((src, sym, tf))
        if df is None:
            for st in strat_list:
                summary_rows.append({
                    "symbol": sym,
                    "tf": tf,
                    "strat": st["strategy_mode"],
                    "mode": st["mode"],
                    "regime": "DATA_ERROR",
                    "z_score": "N/A",
                    "signal": "NO_DATA",
                    "bar_time": "N/A",
                })
            continue

        # Prepare features causally
        feat = prepare_features(df)
        n = len(feat)
        closes = feat["close"].values
        highs = feat["high"].values
        lows = feat["low"].values
        opens = feat["open"].values
        atrs = feat["atr14"].values
        levels = feat["kalman_level"].values
        velocities = feat["kalman_velocity"].values
        z_scores = feat["z_k"].values
        innovations = feat["innovation_std"].values
        sigmas_yz = feat["sigma_yz"].values
        sigmas_yz_med = feat["sigma_yz_median50"].values
        ema_200s = feat["ema_200"].values

        # Latest closed bar index
        i = n - 1
        last_time = df.index[-1]
        session_type = "binance" if src == "BINANCE" else "mt5"
        can_enter, _ = check_session_rules(last_time, session_type)

        # Hurst & VR on last 100 closed bars
        hist_close = closes[max(0, i - 99):i + 1]
        regime = calculate_hurst_vr(np.log(hist_close), window=min(100, len(hist_close)))

        # Evaluate each strategy variant
        for st in strat_list:
            strat_mode = st["strategy_mode"]
            macro_mode = st["mode"]
            cost_spec = st.get("cost_parameters", {"spread": 0.0, "commission": 0.0, "slippage": 0.0})
            total_cost = cost_spec.get("spread", 0.0) + cost_spec.get("commission", 0.0) + cost_spec.get("slippage", 0.0)

            # Regime gate
            regime_match = (
                (strat_mode == "OU_ONLY" and regime.label == "OU_MEAN_REVERSION") or
                (strat_mode == "MOM_ONLY" and regime.label == "KALMAN_MOMENTUM") or
                (strat_mode == "BOTH" and regime.label in ("OU_MEAN_REVERSION", "KALMAN_MOMENTUM"))
            )

            half_life_val = float("nan")
            if regime.label == "OU_MEAN_REVERSION":
                start_ou = max(0, i - 119)
                spread_series = closes[start_ou:i + 1] - levels[start_ou:i + 1]
                ou = estimate_ou_half_life(spread_series)
                if ou["valid"]:
                    half_life_val = ou["half_life"]

            # Macro bias calculation
            eff_macro = (1 if closes[i] > ema_200s[i] else -1) if macro_mode == "Begonya_Macro" else 0

            signal_label = "NO_TRIGGER"
            trigger_sig = None

            if not can_enter:
                signal_label = "SESSION_CLOSED"
            elif not regime_match:
                signal_label = f"REGIME_MISMATCH ({regime.label})"
            else:
                sig, reason = evaluate_stigma_trigger(
                    regime=regime.label,
                    hurst=regime.hurst,
                    variance_ratio=regime.variance_ratio,
                    half_life=half_life_val,
                    z_score=z_scores[:i + 1],
                    level=levels[:i + 1],
                    velocity=velocities[:i + 1],
                    innovation_std=innovations[:i + 1],
                    atr=atrs[:i + 1],
                    open_=opens[:i + 1],
                    close=closes[:i + 1],
                    high_=highs[:i + 1],
                    low_=lows[:i + 1],
                    ema_200=ema_200s[:i + 1],
                    macro_bias=eff_macro,
                    cost_price=total_cost,
                    sigma_yz=sigmas_yz[i],
                    sigma_yz_median=sigmas_yz_med[i],
                    strategy_mode=strat_mode,
                    return_reason=True,
                )
                if sig is not None:
                    signal_label = f"TRIGGER_{sig['side']}"
                    trigger_sig = sig
                else:
                    signal_label = reason

            summary_rows.append({
                "symbol": sym,
                "tf": tf,
                "strat": strat_mode,
                "mode": macro_mode,
                "regime": regime.label,
                "z_score": f"{z_scores[i]:.2f}",
                "signal": signal_label,
                "bar_time": str(last_time),
            })

            if trigger_sig is not None:
                payload = {
                    "symbol": sym,
                    "timeframe": tf,
                    "strategy_mode": strat_mode,
                    "macro_mode": macro_mode,
                    "side": trigger_sig["side"],
                    "entry": trigger_sig["entry"],
                    "stop_loss": trigger_sig["stop_loss"],
                    "tp1": trigger_sig["tp1"],
                    "tp2": trigger_sig["tp2"],
                    "max_hold_bars": trigger_sig["max_hold_bars"],
                    "hurst": regime.hurst,
                    "variance_ratio": regime.variance_ratio,
                    "half_life": half_life_val,
                    "z_score": z_scores[i],
                    "bar_time": str(last_time),
                }
                active_signals.append(payload)

                # Deduplication and notification
                is_new = True
                if deduplicator is not None:
                    is_new = deduplicator.should_emit(sym, strat_mode, str(last_time))

                if is_new:
                    intent = build_order_intent(payload)
                    logger.info("VALIDATED ORDER INTENT: %s", intent)
                    if send_alerts:
                        send_telegram_alert(payload)

    # Print scan report table
    print("\n" + "=" * 96)
    print(f"  STIGMA QUANT ENGINE -- LIVE WHITELIST SCAN REPORT ({len(summary_rows)} Strategies)")
    print("=" * 96)
    print(f"{'SYMBOL':<10} {'TF':<5} {'STRATEGY':<12} {'MACRO':<15} {'REGIME':<20} {'Z_K':<7} {'STATUS / TRIGGER':<20}")
    print("-" * 96)
    for r in summary_rows:
        print(f"{r['symbol']:<10} {r['tf']:<5} {r['strat']:<12} {r['mode']:<15} {r['regime']:<20} {r['z_score']:<7} {r['signal']:<20}")
    print("=" * 96)
    if active_signals:
        print(f"[*] ACTIVE SIGNALS DETECTED: {len(active_signals)}")
        for s in active_signals:
            print(f"  -> {s['symbol']} {s['timeframe']} {s['side']} @ {s['entry']:.5f} (SL: {s['stop_loss']:.5f}, TP1: {s['tp1']:.5f}, TP2: {s['tp2']:.5f})")
    else:
        print("[OK] All whitelisted strategies actively monitored. Zero false positives on current closed bar.")
    print("=" * 96 + "\n")

    return active_signals, last_bar_times

def watch_whitelisted_strategies(
    whitelist_file: str | Path = "reports/whitelist.json",
    interval_sec: int = 30,
    send_alerts: bool = True,
) -> None:
    """Continuously monitor whitelisted strategies. Re-scans silently on each new closed bar."""
    logger.info("Starting Stigma Whitelist Watch loop (interval: %ds, whitelist: %s)...", interval_sec, whitelist_file)
    dedup = SignalDeduplicator()
    previous_bars: dict[str, str] = {}

    try:
        while True:
            # Run scan
            _, current_bars = scan_whitelisted_strategies(whitelist_file, deduplicator=dedup, send_alerts=send_alerts)
            previous_bars = current_bars

            # Wait for next poll
            logger.info("Watch loop standing by (%ds). Waiting for next closed candle...", interval_sec)
            time.sleep(interval_sec)

    except KeyboardInterrupt:
        logger.info("Whitelist Watch loop stopped by user.")

def main():
    p = argparse.ArgumentParser(description="Stigma Quant Engine Pipeline & Live Whitelist Runner")
    p.add_argument("--scan-whitelist", action="store_true", help="Scan all approved strategies from whitelist.json")
    p.add_argument("--watch", action="store_true", help="Run continuous monitoring loop for whitelisted strategies")
    p.add_argument("--interval-sec", type=int, default=30, help="Polling interval in seconds for --watch")
    p.add_argument("--whitelist-file", default="reports/whitelist.json", help="Path to whitelist.json")
    p.add_argument("--source", choices=["csv", "mt5", "binance"], default="csv")
    p.add_argument("--file", help="CSV file with time,open,high,low,close columns")
    p.add_argument("--symbol", default="EURUSD")
    p.add_argument("--timeframe", choices=["M5", "M15", "H1"], default="M15")
    p.add_argument("--bars", type=int, default=50000)
    p.add_argument("--macro-bias", type=int, choices=[-1, 0, 1], default=0)
    p.add_argument("--macro-file", help="Path to Begonya macro JSON file")
    p.add_argument("--spread", type=float, default=0.0)
    p.add_argument("--commission", type=float, default=0.0)
    p.add_argument("--slippage", type=float, default=0.0)
    p.add_argument("--walk-forward", action="store_true", help="Run chronological train/val/oos split")
    p.add_argument("--output", default="reports/trades.csv")
    args = p.parse_args()

    # Mode 1: Whitelist Scan or Watch
    if args.scan_whitelist:
        if args.watch:
            watch_whitelisted_strategies(
                whitelist_file=args.whitelist_file,
                interval_sec=args.interval_sec,
                send_alerts=True,
            )
        else:
            scan_whitelisted_strategies(
                whitelist_file=args.whitelist_file,
                send_alerts=True,
            )
        return

    # Mode 2: Standard Backtest or Walk-Forward Split
    if args.macro_file:
        macro_bias = load_macro_bias(args.macro_file, autonomous=False)
        logger.info("Loaded macro bias %d from %s", macro_bias, args.macro_file)
    else:
        macro_bias = args.macro_bias

    if args.source == "csv":
        if not args.file:
            p.error("--file is required when source is 'csv'")
        df = pd.read_csv(args.file)
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"], utc=True)
            df = df.set_index("time")
    elif args.source == "mt5":
        df = fetch_mt5(args.symbol, args.timeframe, args.bars)
    else:
        df = fetch_binance(args.symbol, args.timeframe, args.bars)

    tf_min = 5 if args.timeframe == "M5" else (60 if args.timeframe == "H1" else 15)

    if args.walk_forward:
        wf_res = run_walk_forward(
            df,
            spread=args.spread,
            commission=args.commission,
            slippage=args.slippage,
            macro_bias=macro_bias,
            timeframe_minutes=tf_min,
        )
        trades = wf_res.pop("all_trades")
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        trades.to_csv(out_path, index=False)
        print(json.dumps({
            "status": "COMPLETED",
            "symbol": args.symbol,
            "timeframe": args.timeframe,
            "macro_bias": macro_bias,
            "walk_forward": wf_res,
            "trades_file": str(out_path),
        }, indent=2))
    else:
        result = backtest(
            df,
            spread=args.spread,
            commission=args.commission,
            slippage=args.slippage,
            macro_bias=macro_bias,
            timeframe_minutes=tf_min,
        )
        trades = result.pop("trades_frame")
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        trades.to_csv(out_path, index=False)
        print(json.dumps({
            "status": "COMPLETED",
            "symbol": args.symbol,
            "timeframe": args.timeframe,
            "macro_bias": macro_bias,
            "metrics": result,
            "trades_file": str(out_path),
        }, indent=2))

if __name__ == "__main__":
    main()
