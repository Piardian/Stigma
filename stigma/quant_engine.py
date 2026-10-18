"""Price-only statistical features and trigger logic.

All signal functions consume chronological, closed-bar data. No tick-volume feature is used.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
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
    """Estimate H from lagged-difference variance scaling and Lo-MacKinlay-style VR.

    Returns RANDOM_WALK_VETO for the specified neutral H band. Insufficient/degenerate
    inputs return INSUFFICIENT_DATA rather than fabricating a regime.
    """
    y = np.asarray(log_prices, dtype=float)
    y = y[np.isfinite(y)]
    if window < 32 or len(y) < window or np.any(~np.isfinite(y[-window:])):
        return Regime(float("nan"), float("nan"), "INSUFFICIENT_DATA")
    y = y[-window:]
    taus = np.asarray([int(t) for t in lags if int(t) >= 2 and int(t) < window // 2])
    if len(taus) < 2:
        return Regime(float("nan"), float("nan"), "INSUFFICIENT_DATA")
    variances = np.array([np.var(y[t:] - y[:-t], ddof=1) for t in taus])
    good = np.isfinite(variances) & (variances > 0)
    if good.sum() < 2:
        return Regime(float("nan"), float("nan"), "INSUFFICIENT_DATA")
    slope = np.polyfit(np.log(taus[good]), np.log(variances[good]), 1)[0]
    hurst = float(np.clip(slope / 2.0, 0.0, 1.0))
    r = np.diff(y)
    q = int(taus[0])
    if len(r) <= q + 2 or np.var(r, ddof=1) <= 0:
        return Regime(hurst, float("nan"), "INSUFFICIENT_DATA")
    # Variance ratio for q-period sums, with finite-sample overlapping-return correction.
    mu = np.mean(r)
    centered = r - mu
    gamma0 = np.dot(centered, centered) / len(r)
    if gamma0 <= 0:
        return Regime(hurst, float("nan"), "INSUFFICIENT_DATA")
    vr = 1.0 + 2.0 * sum((1.0 - k / q) * (np.dot(centered[k:], centered[:-k]) / len(r)) / gamma0 for k in range(1, q))
    if 0.46 <= hurst <= 0.54:
        label = "RANDOM_WALK_VETO"
    elif hurst < 0.46 and vr < 0.92:
        label = "OU_MEAN_REVERSION"
    elif hurst > 0.54 and vr > 1.08:
        label = "KALMAN_MOMENTUM"
    else:
        label = "NO_EDGE"
    return Regime(hurst, float(vr), label)

def run_kalman_filter_1d(prices, atr, process_noise: float = 1e-4, measurement_noise: float = 1e-3) -> KalmanResult:
    """Constant-velocity two-state Kalman filter; O(N) and causal."""
    p = np.asarray(prices, dtype=float)
    a = np.broadcast_to(np.asarray(atr, dtype=float), p.shape)
    n = len(p)
    level = np.full(n, np.nan); velocity = np.full(n, np.nan)
    std = np.full(n, np.nan); z = np.full(n, np.nan)
    if n == 0: return KalmanResult(level, velocity, std, z)
    finite = np.flatnonzero(np.isfinite(p))
    if not len(finite): return KalmanResult(level, velocity, std, z)
    i0 = int(finite[0]); x = np.array([p[i0], 0.0]); cov = np.eye(2)
    F = np.array([[1., 1.], [0., 1.]])
    Q = process_noise * np.array([[0.25, 0.5], [0.5, 1.]])
    Hm = np.array([[1., 0.]])
    for i in range(i0, n):
        if not np.isfinite(p[i]): continue
        if i > i0:
            x = F @ x; cov = F @ cov @ F.T + Q
        R = max(float(measurement_noise), 1e-12)
        innovation = p[i] - float(Hm @ x)
        S = float(Hm @ cov @ Hm.T) + R
        K = (cov @ Hm.T / S).reshape(2)
        x = x + K * innovation
        cov = (np.eye(2) - np.outer(K, Hm.reshape(2))) @ cov
        level[i], velocity[i] = x
        std[i] = np.sqrt(max(S, 0.0))
        floor = max(0.5 * a[i], 1e-12) if np.isfinite(a[i]) else 1e-12
        z[i] = innovation / max(std[i], floor)
    return KalmanResult(level, velocity, std, z)

def yang_zhang_volatility(open_, high, low, close, window: int = 20) -> np.ndarray:
    """Rolling annualization-free Yang–Zhang volatility estimate in log-return units."""
    o,h,l,c = [np.asarray(v, dtype=float) for v in (open_, high, low, close)]
    if not (len(o)==len(h)==len(l)==len(c)): raise ValueError("OHLC arrays must have equal length")
    n=len(c); out=np.full(n,np.nan)
    if window < 2: raise ValueError("window must be >= 2")
    log = lambda x: np.log(np.maximum(x, np.finfo(float).tiny))
    co = log(o[1:]/c[:-1]); oo = log(o[1:]/o[:-1])
    rs = log(h/l)*log(h/o[1:]) + log(h/o[1:])*0 + log(l/o[1:])*log(l/c[1:])
    # Rogers–Satchell per bar; overnight and open-to-close components are separate.
    ho=log(h/o); lo=log(l/o); hc=log(h/c); lc=log(l/c)
    rs_all = ho*hc + lo*lc
    overnight = log(o[1:]/c[:-1]); intraday = log(c/o)
    for i in range(window, n):
        sl=slice(i-window+1, i+1)
        ov=overnight[i-window:i]
        oc=intraday[sl]
        rsw=rs_all[sl]
        if not (np.all(np.isfinite(ov)) and np.all(np.isfinite(oc)) and np.all(np.isfinite(rsw))): continue
        vo=np.var(ov, ddof=1); vc=np.var(oc, ddof=1); vr=np.mean(rsw)
        k=0.34/(1.34+(window+1)/(window-1))
        out[i]=np.sqrt(max(vo+k*vc+(1-k)*vr,0.0))
    return out

def estimate_ou_half_life(spread, min_phi: float = -0.02) -> dict[str, float | bool]:
    """OLS ΔX = alpha + phi*X[t-1]; return half-life only for mean-reverting phi."""
    x=np.asarray(spread,dtype=float)
    x=x[np.isfinite(x)]
    if len(x)<20: return {"alpha":float("nan"),"phi":float("nan"),"half_life":float("inf"),"valid":False}
    lag=x[:-1]; delta=np.diff(x)
    design=np.column_stack([np.ones(len(lag)),lag])
    try: alpha,phi=np.linalg.lstsq(design,delta,rcond=None)[0]
    except np.linalg.LinAlgError: return {"alpha":float("nan"),"phi":float("nan"),"half_life":float("inf"),"valid":False}
    if not np.isfinite(phi) or phi >= min_phi or 1.0+phi <= 0:
        hl=float("inf")
    else:
        theta=-np.log1p(phi); hl=float(np.log(2)/theta) if theta>0 else float("inf")
    return {"alpha":float(alpha),"phi":float(phi),"half_life":hl,"valid":bool(2.0 <= hl <= 30.0)}

def dynamic_time_stop(half_life: float) -> int:
    if not np.isfinite(half_life): return 6
    return int(np.clip(2.5 * half_life, 6, 40))

def evaluate_stigma_trigger(*, regime: str, hurst: float, half_life: float, z_score, level, velocity, innovation_std, atr, open_, close, macro_bias: int = 0) -> dict[str, Any] | None:
    """Evaluate the latest closed bar; return a risk-defined setup or None."""
    arrays=[np.asarray(v,dtype=float) for v in (z_score,level,velocity,innovation_std,atr,open_,close)]
    z,x,v,s,a,o,c=arrays
    n=len(c)
    if n<3 or any(len(q)!=n for q in arrays) or not all(np.isfinite(q[-2:]).all() for q in arrays): return None
    if a[-1] <= 0 or not np.isfinite(hurst): return None
    side=None; entry=float(c[-1]); stop=None; tp1=None; tp2=None; hold=24
    if regime=="OU_MEAN_REVERSION" and hurst<0.46 and 2<=half_life<=30:
        hold=dynamic_time_stop(half_life)
        if macro_bias>=0 and np.min(z[-3:])<=-2 and z[-1]>z[-2] and z[-1]>=-1.85 and c[-1]>o[-1]:
            side="LONG"; stop=entry-1.2*a[-1]; risk=entry-stop
            tp1=max(float(x[-1]),entry+1.1*risk); tp2=max(float(x[-1]+1.2*s[-1]),entry+2*risk)
        elif macro_bias<=0 and np.max(z[-3:])>=2 and z[-1]<z[-2] and z[-1]<=1.85 and c[-1]<o[-1]:
            side="SHORT"; stop=entry+1.2*a[-1]; risk=stop-entry
            tp1=min(float(x[-1]),entry-1.1*risk); tp2=min(float(x[-1]-1.2*s[-1]),entry-2*risk)
    elif regime=="KALMAN_MOMENTUM" and hurst>0.54 and abs(v[-1])>0.05*a[-1]:
        hold=24
        if (macro_bias==1 or macro_bias==0) and v[-1]>0 and -1.2<=z[-1]<=-0.15 and z[-1]>z[-2] and c[-1]>o[-1]:
            side="LONG"; stop=entry-1.2*a[-1]; risk=entry-stop; tp1=entry+1.4*risk; tp2=entry+2.6*risk
        elif (macro_bias==-1 or macro_bias==0) and v[-1]<0 and 0.15<=z[-1]<=1.2 and z[-1]<z[-2] and c[-1]<o[-1]:
            side="SHORT"; stop=entry+1.2*a[-1]; risk=stop-entry; tp1=entry-1.4*risk; tp2=entry-2.6*risk
    if side is None: return None
    return {"side":side,"entry":entry,"stop_loss":float(stop),"tp1":float(tp1),"tp2":float(tp2),"max_hold_bars":hold,"regime":regime,"hurst":float(hurst),"half_life":float(half_life)}
