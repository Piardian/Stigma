"""MT5 terminal adapter. Requires MetaTrader5 package and a logged-in terminal."""
from __future__ import annotations
import pandas as pd

def fetch_mt5(symbol: str, timeframe: str="M15", bars: int=50000) -> pd.DataFrame:
    try: import MetaTrader5 as mt5
    except ImportError as exc: raise RuntimeError("Install MetaTrader5 on Windows and log in to the terminal first") from exc
    mapping={"M5":mt5.TIMEFRAME_M5,"M15":mt5.TIMEFRAME_M15}
    if timeframe not in mapping: raise ValueError("timeframe must be M5 or M15")
    if not mt5.initialize(): raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")
    try:
        if not mt5.symbol_select(symbol,True): raise RuntimeError(f"symbol unavailable: {symbol}")
        rates=mt5.copy_rates_from_pos(symbol,mapping[timeframe],0,bars)
        if rates is None or len(rates)==0: raise RuntimeError(f"no MT5 rates: {mt5.last_error()}")
        df=pd.DataFrame(rates); df["time"]=pd.to_datetime(df["time"],unit="s",utc=True); df=df.set_index("time")
        return df.rename(columns={"tick_volume":"tick_volume"}).loc[:,["open","high","low","close","tick_volume"]].sort_index()
    finally: mt5.shutdown()
