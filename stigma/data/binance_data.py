"""Binance public kline adapter with pagination; tick volume is retained only as metadata."""
from __future__ import annotations
import time
import requests
import pandas as pd

BASE="https://api.binance.com/api/v3/klines"
INTERVALS={"M5":"5m","M15":"15m"}

def fetch_binance(symbol: str, timeframe: str="M15", bars: int=100000, pause: float=0.15) -> pd.DataFrame:
    if timeframe not in INTERVALS: raise ValueError("timeframe must be M5 or M15")
    rows=[]; end=None
    while len(rows)<bars:
        limit=min(1000,bars-len(rows))
        params={"symbol":symbol.upper(),"interval":INTERVALS[timeframe],"limit":limit}
        if end is not None: params["endTime"]=end
        response=requests.get(BASE,params=params,timeout=20); response.raise_for_status()
        batch=response.json()
        if not batch: break
        rows=batch+rows
        end=int(batch[0][0])-1
        if len(batch)<limit: break
        time.sleep(pause)
    rows=rows[-bars:]
    if not rows: raise RuntimeError(f"No Binance klines for {symbol}")
    df=pd.DataFrame(rows,columns=["open_time","open","high","low","close","volume","close_time","quote_volume","trades","taker_buy_base","taker_buy_quote","ignore"])
    for col in ["open","high","low","close","volume"]: df[col]=pd.to_numeric(df[col],errors="coerce")
    df["time"]=pd.to_datetime(df["open_time"],unit="ms",utc=True)
    return df.set_index("time")[["open","high","low","close","volume"]].sort_index()
