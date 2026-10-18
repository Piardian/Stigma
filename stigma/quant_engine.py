"""Price-only statistical features and econometric quant research engine.

All signal functions consume chronological, closed-bar data.
Tick volume and trade volume are retained strictly for broker/exchange telemetry
and are NEVER used as statistical features in signal generation.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd

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

def calculate_hurst_vr(
    log_prices,
    window: int = 100,
    lags: tuple[int, ...] = (2, 4, 8, 16),
    vr_lag: int = 2,
) -> Regime:
    """Estimate Hurst exponent from lagged-difference variance scaling and Lo-MacKinlay VR.

    Mathematical framework:
      - Log-prices y = ln(P).
      - For lag tau in lags: Var(y_{t+tau} - y_t) ~ tau^{2H}.
      - Slope of log(Var) vs log(tau) yields 2H via OLS regression.
      - Finite sample properties: In 100-bar windows, Hurst estimation carries sample
        variance; strict multi-hurdle classification is applied.
      - Lo-MacKinlay Variance Ratio:
          VR(q) = 1 + 2 * sum_{k=1}^{q-1} (1 - k/q) * rho_k
        where rho_k is the sample autocorrelation of log-returns.
      - Classification:
          0.46 <= H <= 0.54                -> RANDOM_WALK_VETO
          H < 0.46 and VR < 0.92           -> OU_MEAN_REVERSION
          H > 0.54 and VR > 1.08           -> KALMAN_MOMENTUM
          Otherwise                        -> NO_EDGE
          Insufficient data / non-finite   -> INSUFFICIENT_DATA
    """
    y = np.asarray(log_prices, dtype=float)
    if y.ndim != 1 or window < 32 or len(y) < window:
        return Regime(float("nan"), float("nan"), "INSUFFICIENT_DATA")

    sub = y[-window:]
    if not np.isfinite(sub).all():
        return Regime(float("nan"), float("nan"), "INSUFFICIENT_DATA")

    taus = np.asarray([int(t) for t in lags if 2 <= int(t) < window // 2])
    if len(taus) < 2:
        return Regime(float("nan"), float("nan"), "INSUFFICIENT_DATA")

    # Lagged variance scaling
    variances = np.array([np.var(sub[t:] - sub[:-t], ddof=1) for t in taus])
    good = np.isfinite(variances) & (variances > 1e-14)
    if good.sum() < 2:
        return Regime(float("nan"), float("nan"), "INSUFFICIENT_DATA")

    # Fast closed-form OLS when all 4 standard lags are valid
    if good.sum() == len(taus) and len(taus) == 4 and np.array_equal(taus, [2, 4, 8, 16]):
        # ln_taus = ln(2)*[1, 2, 3, 4], mean=2.5*ln(2), denom=5*(ln(2))^2
        # weights = [-1.5, -0.5, 0.5, 1.5] / (5 * ln(2))
        inv_denom = 1.0 / (5.0 * np.log(2.0))
        ln_vars = np.log(variances)
        slope = (-1.5 * ln_vars[0] - 0.5 * ln_vars[1] + 0.5 * ln_vars[2] + 1.5 * ln_vars[3]) * inv_denom
    else:
        slope = np.polyfit(np.log(taus[good]), np.log(variances[good]), 1)[0]
    hurst = float(np.clip(slope / 2.0, 0.0, 1.0))

    # Lo-MacKinlay Variance Ratio
    r = np.diff(sub)
    q = int(vr_lag if vr_lag >= 2 else taus[0])
    if len(r) <= q + 2 or np.var(r, ddof=1) <= 1e-14:
        return Regime(hurst, float("nan"), "INSUFFICIENT_DATA")

    centered = r - np.mean(r)
    gamma0 = np.dot(centered, centered) / len(r)
    if gamma0 <= 1e-14 or not np.isfinite(gamma0):
        return Regime(hurst, float("nan"), "INSUFFICIENT_DATA")

    autocovs = [np.dot(centered[k:], centered[:-k]) / len(r) for k in range(1, q)]
    vr = 1.0 + 2.0 * sum((1.0 - k / q) * (autocovs[k - 1] / gamma0) for k in range(1, q))
    vr = float(vr)

    if not (np.isfinite(hurst) and np.isfinite(vr)):
        label = "INSUFFICIENT_DATA"
    elif 0.46 <= hurst <= 0.54:
        label = "RANDOM_WALK_VETO"
    elif hurst < 0.46 and vr < 0.92:
        label = "OU_MEAN_REVERSION"
    elif hurst > 0.54 and vr > 1.08:
        label = "KALMAN_MOMENTUM"
    else:
        label = "NO_EDGE"

    return Regime(hurst, vr, label)

def run_kalman_filter_1d(
    prices,
    atr,
    process_noise: float = 1e-4,
    measurement_noise: float = 1e-3,
    scale_invariant: bool = True,
) -> KalmanResult:
    """Constant-velocity two-state Kalman filter, O(N), causal.

    State vector: x_t = [level, velocity]^T
    Transition:   F = [[1, 1], [0, 1]]
    Process noise covariance: Q = process_noise * [[0.25, 0.5], [0.5, 1.0]]
    Measurement model: H = [1, 0], R = max(measurement_noise, 1e-12)

    Innovation Z-score:
        Z_K = (P_t - x_hat_{t|t-1}[0]) / max(sqrt(S_t), 0.5 * ATR14)
    Only uses past and current closed bars.
    """
    p = np.asarray(prices, dtype=float)
    a = np.broadcast_to(np.asarray(atr, dtype=float), p.shape)
    n = len(p)
    level = np.full(n, np.nan)
    velocity = np.full(n, np.nan)
    std = np.full(n, np.nan)
    z = np.full(n, np.nan)

    if n == 0:
        return KalmanResult(level, velocity, std, z)

    finite = np.flatnonzero(np.isfinite(p) & (p > 0))
    if len(finite) == 0:
        return KalmanResult(level, velocity, std, z)

    i0 = int(finite[0])
    x0 = float(p[i0])
    x1 = 0.0

    # Asset-class scale invariance: normalize nominal price units to base 100
    scale = (x0 / 100.0) ** 2 if (scale_invariant and x0 > 0) else 1.0

    c00 = 1.0 * scale
    c01 = 0.0
    c11 = 1.0 * scale

    r = float(max(measurement_noise * scale, 1e-12))
    q00 = 0.25 * process_noise * scale
    q01 = 0.5 * process_noise * scale
    q11 = 1.0 * process_noise * scale

    for i in range(i0, n):
        pi = p[i]
        if not np.isfinite(pi) or pi <= 0:
            # Propagate unobserved state forward without measurement update
            x0 = x0 + x1
            c00 = c00 + 2.0 * c01 + c11 + q00
            c01 = c01 + c11 + q01
            c11 = c11 + q11
            level[i] = x0
            velocity[i] = x1
            continue

        if i > i0:
            # Predict step
            x0 = x0 + x1
            c00 = c00 + 2.0 * c01 + c11 + q00
            c01 = c01 + c11 + q01
            c11 = c11 + q11

        innovation = pi - x0
        S = c00 + r
        s_val = max(S, 1e-12)

        k0 = c00 / s_val
        k1 = c01 / s_val

        # Update state
        x0 += k0 * innovation
        x1 += k1 * innovation

        # Update covariance (Joseph-stabilized symmetric form)
        c11 = max(c11 - k1 * c01, 1e-14)
        c00 = max((1.0 - k0) * c00, 1e-14)
        c01 = (1.0 - k0) * c01

        level[i] = x0
        velocity[i] = x1
        std_val = np.sqrt(s_val)
        std[i] = std_val

        ai = a[i]
        floor = max(0.5 * ai, 1e-12) if np.isfinite(ai) and ai > 0 else 1e-12
        z[i] = innovation / max(std_val, floor)

    return KalmanResult(level, velocity, std, z)

def yang_zhang_volatility(
    open_,
    high,
    low,
    close,
    window: int = 20,
) -> np.ndarray:
    """Rolling Yang–Zhang volatility in log-return units (not annualized).

    Yang-Zhang (2000) estimator combines overnight jump variance, open-to-close
    variance, and Rogers-Satchell drift-independent variance:
        V_YZ = V_overnight + k * V_open_to_close + (1 - k) * V_RS
        k = 0.34 / (1.34 + (window + 1) / (window - 1))
        sigma_YZ = sqrt(V_YZ)
    """
    o, h, l, c = [np.asarray(v, dtype=float) for v in (open_, high, low, close)]
    if not (len(o) == len(h) == len(l) == len(c)):
        raise ValueError("OHLC arrays must have equal length")
    if window < 2:
        raise ValueError("window must be >= 2")

    # Validate strictly positive prices
    if (o <= 0).any() or (h <= 0).any() or (l <= 0).any() or (c <= 0).any():
        raise ValueError("OHLC prices must be strictly positive")
    if not (np.isfinite(o).all() and np.isfinite(h).all() and np.isfinite(l).all() and np.isfinite(c).all()):
        raise ValueError("OHLC prices must be finite")

    # Geometry validation: high >= max(open, close, low), low <= min(open, close, high)
    if (h < l).any() or (h < o).any() or (h < c).any() or (l > o).any() or (l > c).any():
        raise ValueError("Invalid OHLC bar geometry: high must be >= open, close, low")

    n = len(c)
    out = np.full(n, np.nan)
    if n < window + 1:
        return out

    # Component returns
    overnight = np.log(o[1:] / c[:-1])
    intraday = np.log(c / o)
    ho = np.log(h / o)
    lo = np.log(l / o)
    hc = np.log(h / c)
    lc = np.log(l / c)
    rs_all = ho * hc + lo * lc

    k = 0.34 / (1.34 + (window + 1) / (window - 1))

    # Vectorized rolling calculations
    s_overnight = pd.Series(overnight)
    s_intraday = pd.Series(intraday)
    s_rs = pd.Series(rs_all)

    vo_vec = np.full(n, np.nan)
    vo_vec[1:] = s_overnight.rolling(window, min_periods=window).var(ddof=1).to_numpy()
    vc_vec = s_intraday.rolling(window, min_periods=window).var(ddof=1).to_numpy()
    vr_vec = s_rs.rolling(window, min_periods=window).mean().to_numpy()

    total_var = vo_vec + k * vc_vec + (1.0 - k) * vr_vec
    out = np.sqrt(np.maximum(total_var, 0.0))
    # Ensure first window bars remain NaN
    out[:window] = np.nan
    return out

def estimate_ou_half_life(
    spread,
    min_phi: float = -0.02,
) -> dict[str, float | bool]:
    """OLS Delta X_t = alpha + phi * X_{t-1} + epsilon_t; valid half-life is 2–30 bars.

    Theoretical Ornstein-Uhlenbeck formulation:
      theta = -ln(1 + phi)
      half_life = ln(2) / theta
    Valid mean-reversion requires phi < min_phi (default -0.02) and 2 <= half_life <= 30.
    """
    x = np.asarray(spread, dtype=float)
    if len(x) < 20 or not np.isfinite(x).all():
        return {"alpha": float("nan"), "phi": float("nan"), "half_life": float("inf"), "valid": False}

    lag = x[:-1]
    delta = np.diff(x)
    var_lag = np.var(lag)
    if not np.isfinite(var_lag) or var_lag <= 1e-14:
        return {"alpha": float("nan"), "phi": float("nan"), "half_life": float("inf"), "valid": False}

    mean_lag = np.mean(lag)
    mean_delta = np.mean(delta)
    sxx = np.dot(lag - mean_lag, lag - mean_lag)
    if sxx <= 1e-14 or not np.isfinite(sxx):
        return {"alpha": float("nan"), "phi": float("nan"), "half_life": float("inf"), "valid": False}
    sxy = np.dot(lag - mean_lag, delta - mean_delta)
    phi = sxy / sxx
    alpha = mean_delta - phi * mean_lag

    if not np.isfinite(phi) or phi >= min_phi or 1.0 + phi <= 0:
        hl = float("inf")
    else:
        theta = -np.log1p(phi)
        hl = float(np.log(2.0) / theta) if theta > 0 else float("inf")

    valid = bool(np.isfinite(hl) and 2.0 <= hl <= 30.0)
    return {
        "alpha": float(alpha) if np.isfinite(alpha) else float("nan"),
        "phi": float(phi) if np.isfinite(phi) else float("nan"),
        "half_life": hl,
        "valid": valid,
    }

def dynamic_time_stop(half_life: float) -> int:
    """Dynamic bar time-stop: int(clip(2.5 * half_life, 6, 40))."""
    if not np.isfinite(half_life) or half_life <= 0:
        return 6
    return int(np.clip(2.5 * half_life, 6, 40))

def evaluate_stigma_trigger(
    *,
    regime: str,
    hurst: float,
    half_life: float,
    z_score,
    level,
    velocity,
    innovation_std,
    atr,
    open_,
    close,
    variance_ratio: float = float("nan"),
    macro_bias: int = 0,
) -> dict[str, Any] | None:
    """Evaluate causal trigger conditions for OU_MEAN_REVERSION and KALMAN_MOMENTUM."""
    arrays = [np.asarray(v, dtype=float) for v in (z_score, level, velocity, innovation_std, atr, open_, close)]
    z, x, v, s, a, o, c = arrays
    n = len(c)

    if n < 3 or any(len(q) != n for q in arrays):
        return None
    if not all(np.isfinite(q[-3:]).all() for q in arrays):
        return None
    if a[-1] <= 0 or not np.isfinite(hurst) or (o[-1] <= 0) or (c[-1] <= 0):
        return None
    if macro_bias not in (-1, 0, 1):
        return None

    side = None
    entry = float(c[-1])
    stop = tp1 = tp2 = None
    hold = 24
    risk = 1.2 * float(a[-1])

    # Mean-reversion regime
    if regime == "OU_MEAN_REVERSION" and hurst < 0.46 and (np.isnan(variance_ratio) or variance_ratio < 0.92) and (2.0 <= half_life <= 30.0):
        hold = dynamic_time_stop(half_life)
        if macro_bias >= 0 and np.min(z[-3:]) <= -2.0 and z[-1] > z[-2] and z[-1] >= -1.85 and c[-1] > o[-1]:
            side = "LONG"
            stop = entry - risk
            tp1 = max(float(x[-1]), entry + 1.1 * risk)
            tp2 = max(float(x[-1] + 1.2 * s[-1]), entry + 2.0 * risk)
        elif macro_bias <= 0 and np.max(z[-3:]) >= 2.0 and z[-1] < z[-2] and z[-1] <= 1.85 and c[-1] < o[-1]:
            side = "SHORT"
            stop = entry + risk
            tp1 = min(float(x[-1]), entry - 1.1 * risk)
            tp2 = min(float(x[-1] - 1.2 * s[-1]), entry - 2.0 * risk)

    # Kalman momentum regime (VR > 1.08 strictly enforced!)
    elif regime == "KALMAN_MOMENTUM" and hurst > 0.54 and (np.isfinite(variance_ratio) and variance_ratio > 1.08) and abs(v[-1]) > 0.05 * a[-1]:
        hold = 24
        if macro_bias >= 0 and v[-1] > 0 and -1.2 <= z[-1] <= -0.15 and z[-1] > z[-2] and c[-1] > o[-1]:
            side = "LONG"
            stop = entry - risk
            tp1 = entry + 1.4 * risk
            tp2 = entry + 2.6 * risk
        elif macro_bias <= 0 and v[-1] < 0 and 0.15 <= z[-1] <= 1.2 and z[-1] < z[-2] and c[-1] < o[-1]:
            side = "SHORT"
            stop = entry + risk
            tp1 = entry - 1.4 * risk
            tp2 = entry - 2.6 * risk

    if side is None or stop is None or tp1 is None or tp2 is None:
        return None

    return {
        "side": side,
        "entry": entry,
        "stop_loss": float(stop),
        "tp1": float(tp1),
        "tp2": float(tp2),
        "max_hold_bars": hold,
        "regime": regime,
        "hurst": float(hurst),
        "variance_ratio": float(variance_ratio),
        "half_life": float(half_life),
        "risk": float(risk),
    }
