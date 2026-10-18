"""Run a local research backtest or walk-forward matrix from CSV, MT5, or Binance."""
from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path
import pandas as pd
from stigma.backtest_engine import backtest
from stigma.data.binance_data import fetch_binance
from stigma.data.mt5_data import fetch_mt5
from stigma.macro_gate import load_macro_bias

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

def main():
    p = argparse.ArgumentParser(description="Stigma Quant Engine Pipeline Runner")
    p.add_argument("--source", choices=["csv", "mt5", "binance"], default="csv")
    p.add_argument("--file", help="CSV file with time,open,high,low,close columns")
    p.add_argument("--symbol", default="EURUSD")
    p.add_argument("--timeframe", choices=["M5", "M15"], default="M15")
    p.add_argument("--bars", type=int, default=50000)
    p.add_argument("--macro-bias", type=int, choices=[-1, 0, 1], default=0)
    p.add_argument("--macro-file", help="Path to Begonya macro JSON file")
    p.add_argument("--spread", type=float, default=0.0)
    p.add_argument("--commission", type=float, default=0.0)
    p.add_argument("--slippage", type=float, default=0.0)
    p.add_argument("--walk-forward", action="store_true", help="Run chronological train/val/oos split")
    p.add_argument("--output", default="reports/trades.csv")
    args = p.parse_args()

    # Determine macro bias
    if args.macro_file:
        macro_bias = load_macro_bias(args.macro_file, autonomous=False)
        logger.info("Loaded macro bias %d from %s", macro_bias, args.macro_file)
    else:
        macro_bias = args.macro_bias

    # Fetch data
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

    tf_min = 5 if args.timeframe == "M5" else 15

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
