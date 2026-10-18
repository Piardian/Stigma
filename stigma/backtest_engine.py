"""Causal, bar-by-bar walk-forward backtest utilities."""
from __future__ import annotations
import numpy as np
import pandas as pd
from stigma.quant_engine import calculate_hurst_vr, run_kalman_filter_1d, yang_zhang_volatility, estimate_ou_half_life, evaluate_stigma_trigger

def atr14(df: pd.DataFrame) -> np.ndarray:
    prev=df["close"].shift(1)
    tr=pd.concat([(df["high"]-df["low"]).abs(),(df["high"]-prev).abs(),(df["low"]-prev).abs()],axis=1).max(axis=1)
    return tr.rolling(14,min_periods=14).mean().to_numpy()

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    required={"open","high","low","close"}
    if not required.issubset(df.columns): raise ValueError(f"missing OHLC columns: {required-set(df.columns)}")
    out=df.copy().sort_index()
    if out.index.has_duplicates: raise ValueError("duplicate timestamps")
    if not np.isfinite(out[["open","high","low","close"]].to_numpy()).all(): raise ValueError("non-finite OHLC")
    if (out[["open","high","low","close"]]<=0).any().any(): raise ValueError("OHLC prices must be positive")
    out["atr14"]=atr14(out)
    out["sigma_yz"]=yang_zhang_volatility(out.open.to_numpy(),out.high.to_numpy(),out.low.to_numpy(),out.close.to_numpy())
    k=run_kalman_filter_1d(out.close.to_numpy(),out.atr14.to_numpy())
    out["kalman_level"]=k.level; out["kalman_velocity"]=k.velocity
    out["innovation_std"]=k.innovation_std; out["z_k"]=k.z_score
    return out

def backtest(df: pd.DataFrame, *, spread: float=0.0, commission: float=0.0, slippage: float=0.0, macro_bias: int=0, timeframe_minutes: int=15) -> dict:
    """Single-position simulator; signal on close, enter next open.

    Stop gaps (including weekend gaps) execute at the observed open rather than at a stale
    stop level. Cost inputs are in price units. OHLC cannot determine intrabar event order
    when stop and target both fall inside the same candle; this simulator checks stop first.
    """
    f=prepare_features(df)
    trades=[]; position=None; warm=120
    for i in range(warm,len(f)-1):
        row=f.iloc[i]; nxt=f.iloc[i+1]
        if position:
            position["bars"]+=1
            op=float(nxt.open); hi=float(nxt.high); lo=float(nxt.low)
            side=position["side"]; stop=position["stop_loss"]; target=position["tp2"]
            reason=None; exit_price=None
            if side=="LONG":
                if op<=stop: exit_price=op; reason="gap_stop"
                elif lo<=stop: exit_price=stop; reason="stop"
                elif hi>=target: exit_price=target; reason="tp2"
            else:
                if op>=stop: exit_price=op; reason="gap_stop"
                elif hi>=stop: exit_price=stop; reason="stop"
                elif lo<=target: exit_price=target; reason="tp2"
            if reason is None and position["bars"]>=position["max_hold_bars"]:
                exit_price=float(nxt.close); reason="time_stop"
            if reason:
                gross=(exit_price-position["entry"])*(1 if side=="LONG" else -1)
                trades.append({**position,"exit":exit_price,"gross_pnl":gross,"net_pnl":gross-spread-commission-slippage,"exit_reason":reason,"exit_time":f.index[i+1]})
                position=None
            continue
        hist=f.iloc[max(0,i-99):i+1]
        regime=calculate_hurst_vr(np.log(hist.close.to_numpy()),window=min(100,len(hist)))
        if regime.label not in ("OU_MEAN_REVERSION","KALMAN_MOMENTUM"): continue
        start=max(0,i-250)
        spread_series=f.close.to_numpy()[start:i+1]-f.kalman_level.to_numpy()[start:i+1]
        ou=estimate_ou_half_life(spread_series)
        sig=evaluate_stigma_trigger(regime=regime.label,hurst=regime.hurst,half_life=ou["half_life"],z_score=f.z_k.to_numpy()[:i+1],level=f.kalman_level.to_numpy()[:i+1],velocity=f.kalman_velocity.to_numpy()[:i+1],innovation_std=f.innovation_std.to_numpy()[:i+1],atr=f.atr14.to_numpy()[:i+1],open_=f.open.to_numpy()[:i+1],close=f.close.to_numpy()[:i+1],macro_bias=macro_bias)
        if sig:
            entry=float(nxt.open); shift=entry-float(row.close)
            sig["entry"]=entry; sig["stop_loss"]+=shift; sig["tp1"]+=shift; sig["tp2"]+=shift
            sig.update({"entry_time":f.index[i+1],"bars":0}); position=sig
    if position:
        last=f.iloc[-1]; side=position["side"]; exit_price=float(last.close)
        gross=(exit_price-position["entry"])*(1 if side=="LONG" else -1)
        trades.append({**position,"exit":exit_price,"gross_pnl":gross,"net_pnl":gross-spread-commission-slippage,"exit_reason":"end_of_data","exit_time":f.index[-1]})
    t=pd.DataFrame(trades)
    if t.empty: return {"trades":0,"win_rate":0.,"profit_factor":0.,"net_pnl":0.,"trades_frame":t}
    wins=t.loc[t.net_pnl>0,"net_pnl"].sum(); losses=-t.loc[t.net_pnl<0,"net_pnl"].sum()
    pf=float(wins/losses) if losses>0 else float("inf") if wins>0 else 0.
    return {"trades":len(t),"win_rate":float((t.net_pnl>0).mean()),"profit_factor":pf,"net_pnl":float(t.net_pnl.sum()),"trades_frame":t}
