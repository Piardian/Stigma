"""Price-only statistical features and trigger logic.

All signal functions consume chronological, closed-bar data. No tick-volume feature is used.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np

@dataclass
class Regime:
    hurst: float
    variance_ratio: float
    label: str

@dataclass
class KalmanResult:
    level: np.ndarray
    velocity: np.ndarray
    innovation_std: np.ndarray
    z_score: np.ndarray

def calculate_hurst_vr(log_prices, window: int = 100, lags=(2, 4, 8, 16)) -> Regime:
    """Estimate H from lagged-difference variance scaling and a Lo–MacKinlay-style VR."""
    y = np.asarray(log_prices, dtype=float)
    if window < 32 or len(y) < window or not np.isfinite(y[-window:]).all():
        return Regime(float("nan"), float("nan"), "INSUFFICIENT_DATA")
    y = y[-window:]
    taus = np.asarray([int(t) for t in lags if 2 <= int(t) < window // 2])
    if len(taus) < 2:
        return Regime(float("nan"), float("nan"), "INSUFFICIENT_DATA")
    variances = np.array([np.var(y[t:] - y[:-t], ddof=1) for t in taus])
    good = np.isfinite(variances) & (variances > 0)
    if good.sum() < 2:
        return Regime(float("nan"), float("nan"), "INSUFFICIENT_DATA")
    slope = np.polyfit(np.log(taus[good]), np.log(variances[good]), 1)[0]
    hurst = float(np.clip(slope / 2.0, 0.0, 1.0))
    r = np.diff(y); q = int(taus[0])
    if len(r) <= q + 2 or np.var(r, ddof=1) <= 0:
        return Regime(hurst, float("nan"), "INSUFFICIENT_DATA")
    centered = r - np.mean(r)
    gamma0 = np.dot(centered, centered) / len(r)
    if gamma0 <= 0:
        return Regime(hurst, float("nan"), "INSUFFICIENT_DATA")
    vr = 1.0 + 2.0 * sum((1.0-k/q) * (np.dot(centered[k:], centered[:-k]) / len(r)) / gamma0 for k in range(1,q))
    if 0.46 <= hurst <= 0.54: label = "RANDOM_WALK_VETO"
    elif hurst < 0.46 and vr < 0.92: label = "OU_MEAN_REVERSION"
    elif hurst > 0.54 and vr > 1.08: label = "KALMAN_MOMENTUM"
    else: label = "NO_EDGE"
    return Regime(hurst, float(vr), label)

def run_kalman_filter_1d(prices, atr, process_noise: float=1e-4, measurement_noise: float=1e-3) -> KalmanResult:
    """Constant-velocity two-state Kalman filter, O(N), causal."""
    p=np.asarray(prices,dtype=float); a=np.broadcast_to(np.asarray(atr,dtype=float),p.shape); n=len(p)
    level=np.full(n,np.nan); velocity=np.full(n,np.nan); std=np.full(n,np.nan); z=np.full(n,np.nan)
    finite=np.flatnonzero(np.isfinite(p))
    if not len(finite): return KalmanResult(level,velocity,std,z)
    i0=int(finite[0]); x=np.array([p[i0],0.]); cov=np.eye(2)
    F=np.array([[1.,1.],[0.,1.]])
    Q=process_noise*np.array([[.25,.5],[.5,1.]])
    for i in range(i0,n):
        if not np.isfinite(p[i]): continue
        if i>i0: x=F@x; cov=F@cov@F.T+Q
        innovation=float(p[i]-x[0]); S=float(cov[0,0]+max(measurement_noise,1e-12))
        K=cov[:,0]/S; x=x+K*innovation; cov=(np.eye(2)-np.outer(K,np.array([1.,0.])))@cov
        level[i],velocity[i]=x; std[i]=np.sqrt(max(S,0.))
        floor=max(.5*a[i],1e-12) if np.isfinite(a[i]) else 1e-12
        z[i]=innovation/max(std[i],floor)
    return KalmanResult(level,velocity,std,z)

def yang_zhang_volatility(open_,high,low,close,window:int=20)->np.ndarray:
    """Rolling Yang–Zhang volatility in log-return units (not annualized)."""
    o,h,l,c=[np.asarray(v,dtype=float) for v in (open_,high,low,close)]
    if not (len(o)==len(h)==len(l)==len(c)): raise ValueError("OHLC arrays must have equal length")
    if window<2: raise ValueError("window must be >= 2")
    n=len(c); out=np.full(n,np.nan); log=lambda x:np.log(np.maximum(x,np.finfo(float).tiny))
    overnight=log(o[1:]/c[:-1]); intraday=log(c/o)
    ho=log(h/o); lo=log(l/o); hc=log(h/c); lc=log(l/c)
    rs_all=ho*hc+lo*lc
    k=.34/(1.34+(window+1)/(window-1))
    for i in range(window,n):
        ov=overnight[i-window:i]; oc=intraday[i-window+1:i+1]; rsw=rs_all[i-window+1:i+1]
        if not (np.isfinite(ov).all() and np.isfinite(oc).all() and np.isfinite(rsw).all()): continue
        vo=np.var(ov,ddof=1); vc=np.var(oc,ddof=1); vr=np.mean(rsw)
        out[i]=np.sqrt(max(vo+k*vc+(1-k)*vr,0.))
    return out

def estimate_ou_half_life(spread,min_phi:float=-.02)->dict[str,float|bool]:
    """OLS ΔX = alpha + phi X[t-1]; valid half-life is 2–30 bars."""
    x=np.asarray(spread,dtype=float); x=x[np.isfinite(x)]
    if len(x)<20: return {"alpha":float("nan"),"phi":float("nan"),"half_life":float("inf"),"valid":False}
    lag=x[:-1]; delta=np.diff(x); design=np.column_stack([np.ones(len(lag)),lag])
    try: alpha,phi=np.linalg.lstsq(design,delta,rcond=None)[0]
    except np.linalg.LinAlgError: return {"alpha":float("nan"),"phi":float("nan"),"half_life":float("inf"),"valid":False}
    if not np.isfinite(phi) or phi>=min_phi or 1.+phi<=0: hl=float("inf")
    else:
        theta=-np.log1p(phi); hl=float(np.log(2)/theta) if theta>0 else float("inf")
    return {"alpha":float(alpha),"phi":float(phi),"half_life":hl,"valid":bool(2.<=hl<=30.)}

def dynamic_time_stop(half_life:float)->int:
    if not np.isfinite(half_life): return 6
    return int(np.clip(2.5*half_life,6,40))

def evaluate_stigma_trigger(*,regime:str,hurst:float,half_life:float,z_score,level,velocity,innovation_std,atr,open_,close,macro_bias:int=0)->dict[str,Any]|None:
    arrays=[np.asarray(v,dtype=float) for v in (z_score,level,velocity,innovation_std,atr,open_,close)]
    z,x,v,s,a,o,c=arrays; n=len(c)
    if n<3 or any(len(q)!=n for q in arrays) or not all(np.isfinite(q[-2:]).all() for q in arrays): return None
    if a[-1]<=0 or not np.isfinite(hurst): return None
    side=None; entry=float(c[-1]); stop=tp1=tp2=None; hold=24
    if regime=="OU_MEAN_REVERSION" and hurst<.46 and 2<=half_life<=30:
        hold=dynamic_time_stop(half_life)
        if macro_bias>=0 and np.min(z[-3:])<=-2 and z[-1]>z[-2] and z[-1]>=-1.85 and c[-1]>o[-1]:
            side="LONG"; stop=entry-1.2*a[-1]; risk=entry-stop
            tp1=max(float(x[-1]),entry+1.1*risk); tp2=max(float(x[-1]+1.2*s[-1]),entry+2*risk)
        elif macro_bias<=0 and np.max(z[-3:])>=2 and z[-1]<z[-2] and z[-1]<=1.85 and c[-1]<o[-1]:
            side="SHORT"; stop=entry+1.2*a[-1]; risk=stop-entry
            tp1=min(float(x[-1]),entry-1.1*risk); tp2=min(float(x[-1]-1.2*s[-1]),entry-2*risk)
    elif regime=="KALMAN_MOMENTUM" and hurst>.54 and abs(v[-1])>.05*a[-1]:
        if (macro_bias==1 or macro_bias==0) and v[-1]>0 and -1.2<=z[-1]<=-.15 and z[-1]>z[-2] and c[-1]>o[-1]:
            side="LONG"; stop=entry-1.2*a[-1]; risk=entry-stop; tp1=entry+1.4*risk; tp2=entry+2.6*risk
        elif (macro_bias==-1 or macro_bias==0) and v[-1]<0 and .15<=z[-1]<=1.2 and z[-1]<z[-2] and c[-1]<o[-1]:
            side="SHORT"; stop=entry+1.2*a[-1]; risk=stop-entry; tp1=entry-1.4*risk; tp2=entry-2.6*risk
    if side is None: return None
    return {"side":side,"entry":entry,"stop_loss":float(stop),"tp1":float(tp1),"tp2":float(tp2),"max_hold_bars":hold,"regime":regime,"hurst":float(hurst),"half_life":float(half_life)}
