"""Run a local research backtest from CSV or a public Binance feed."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from stigma.backtest_engine import backtest
from stigma.data.binance_data import fetch_binance
from stigma.data.mt5_data import fetch_mt5

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source",choices=["csv","mt5","binance"],default="csv")
    p.add_argument("--file",help="CSV with time,open,high,low,close columns")
    p.add_argument("--symbol",default="EURUSD"); p.add_argument("--timeframe",choices=["M5","M15"],default="M15")
    p.add_argument("--bars",type=int,default=50000); p.add_argument("--macro-bias",type=int,choices=[-1,0,1],default=0)
    p.add_argument("--spread",type=float,default=0.0); p.add_argument("--commission",type=float,default=0.0); p.add_argument("--slippage",type=float,default=0.0)
    p.add_argument("--output",default="reports/trades.csv")
    args=p.parse_args()
    if args.source=="csv":
        if not args.file: p.error("--file is required for csv source")
        df=pd.read_csv(args.file)
        if "time" in df: df["time"]=pd.to_datetime(df["time"],utc=True); df=df.set_index("time")
    elif args.source=="mt5": df=fetch_mt5(args.symbol,args.timeframe,args.bars)
    else: df=fetch_binance(args.symbol,args.timeframe,args.bars)
    result=backtest(df,spread=args.spread,commission=args.commission,slippage=args.slippage,macro_bias=args.macro_bias,timeframe_minutes=5 if args.timeframe=="M5" else 15)
    trades=result.pop("trades_frame")
    Path(args.output).parent.mkdir(parents=True,exist_ok=True); trades.to_csv(args.output,index=False)
    print({**result,"trades_file":args.output})
if __name__=="__main__": main()
