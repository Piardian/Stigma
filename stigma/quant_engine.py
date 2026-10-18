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
      - Lo-MacKinlay Variance Ratio:
          VR(q) = 1 + 2 * sum_{k=1}^{q-1} (1 - k/q) * rho_k
        where rho_k is the sample autocorrelation of log-returns.
      - Classification rules:
          0.47 <= H <= 0.53 and 0.95 <= VR <= 1.05 -> RANDOM_WALK_VETO (pure noise)
          H < 0.47 or (H < 0.50 and VR < 0.90)     -> OU_MEAN_REVERSION
          H > 0.53 or (H > 0.50 and VR > 1.10)     -> KALMAN_MOMENTUM
          Otherwise                                -> NO_EDGE
          Insufficient data / non-finite           -> INSUFFICIENT_DATA
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
    elif 0.47 <= hurst <= 0.53 and 0.95 <= vr <= 1.05:
        label = "RANDOM_WALK_VETO"
    elif hurst < 0.47 or (hurst < 0.50 and vr < 0.90):
        label = "OU_MEAN_REVERSION"
    elif hurst > 0.53 or (hurst > 0.50 and vr > 1.10):
        label = "KALMAN_MOMENTUM"
    else:
        label = "NO_EDGE"

    return Regime(hurst, vr, label)

def run_kalman_filter_1d(
    prices,
    atr,
    process_noise: float = 1e-4,
    measurement_noise: float = 1e-3,
    adaptive: bool = True,
) -> KalmanResult:
    """Adaptive heavy-line or static constant-velocity two-state Kalman filter, O(N), causal.

    In adaptive mode:
      - Local measurement noise R_i = max(atr[i] ** 2, 1e-12)
      - Heavy institutional line process noise Q_i = 0.0005 * R_i
      - Eliminates nominal asset price distortion across FX, Metals, Indices, and Crypto.

    Empirical Spread Z-Score:
      - Spread S_t = P_t - x_hat_t
      - 60-bar rolling spread standard deviation: rolling_spread_std
      - Z_K,t = (P_t - x_hat_t) / max(rolling_spread_std, 1.0 * ATR14)

    State vector: x_t = [level, velocity]^T
    Transition:   F = [[1, 1], [0, 1]]
    Observation:  H = [1, 0]
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

    if adaptive:
        ai0 = a[i0] if (np.isfinite(a[i0]) and a[i0] > 0) else 0.001 * x0
        r_init = max(ai0 ** 2, 1e-12)
        c00 = r_init
        c01 = 0.0
        c11 = r_init
    else:
        c00 = 1.0
        c01 = 0.0
        c11 = 1.0

    level[i0] = x0
    velocity[i0] = 0.0
    std[i0] = np.sqrt(c00)
    z[i0] = 0.0

    for i in range(i0 + 1, n):
        pi = p[i]
        ai = a[i]

        if adaptive:
            # Dynamic variance scaling based on local ATR: R_i = ATR^2, Q_i = 0.0005 * R_i
            sig_i = ai if (np.isfinite(ai) and ai > 0) else (a[i - 1] if (np.isfinite(a[i - 1]) and a[i - 1] > 0) else 1e-4)
            r_val = max(sig_i ** 2, 1e-12)
            q_val = 0.0005 * r_val
            q00 = 0.25 * q_val
            q01 = 0.5 * q_val
            q11 = 1.0 * q_val
        else:
            r_val = float(max(measurement_noise, 1e-12))
            q00 = 0.25 * process_noise
            q01 = 0.5 * process_noise
            q11 = 1.0 * process_noise

        if not np.isfinite(pi) or pi <= 0:
            # Propagate unobserved state forward without measurement update
            x0 = x0 + x1
            c00 = c00 + 2.0 * c01 + c11 + q00
            c01 = c01 + c11 + q01
            c11 = c11 + q11
            level[i] = x0
            velocity[i] = x1
            continue

        # Predict step
        x0 = x0 + x1
        c00_pred = c00 + 2.0 * c01 + c11 + q00
        c01_pred = c01 + c11 + q01
        c11_pred = c11 + q11

        # Innovation step
        innovation = pi - x0
        S = c00_pred + r_val
        s_val = max(S, 1e-12)

        k0 = c00_pred / s_val
        k1 = c01_pred / s_val

        # Update state
        x0 += k0 * innovation
        x1 += k1 * innovation

        # Update covariance (Joseph stabilized symmetric form)
        c11 = max(c11_pred - k1 * c01_pred, 1e-14)
        c00 = max((1.0 - k0) * c00_pred, 1e-14)
        c01 = (1.0 - k0) * c01_pred

        level[i] = x0
        velocity[i] = x1
        std[i] = np.sqrt(s_val)

    # -------------------------------------------------------------
    # Empirical Spread Z-Score:
    # S_t = P_t - level_t
    # rolling_spread_std over trailing 60 bars (min_periods=20)
    # Z_K = S_t / max(rolling_spread_std_t, 1.0 * ATR14_t)
    # -------------------------------------------------------------
    spread = p - level
    s_spread = pd.Series(spread).rolling(60, min_periods=20).std().to_numpy()
    s_atr = np.where(np.isfinite(a) & (a > 0), a, 1e-4)
    denom = np.where(np.isfinite(s_spread) & (s_spread > 0), np.maximum(s_spread, s_atr), s_atr)
    z = np.where(np.isfinite(spread), spread / np.maximum(denom, 1e-12), 0.0)

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
    out[:window] = np.nan
    return out

def estimate_ou_half_life(
    spread,
    min_phi: float = -0.02,
) -> dict[str, float | bool]:
    """OLS Delta X_t = alpha + phi * X_{t-1} + epsilon_t; valid half-life is 2–40 bars.

    Theoretical Ornstein-Uhlenbeck formulation:
      theta = -ln(1 + phi)
      half_life = ln(2) / theta
    Valid mean-reversion requires phi < min_phi (default -0.02) and 2.0 <= half_life <= 40.0.
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

    valid = bool(np.isfinite(hl) and 2.0 <= hl <= 40.0)
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
    high_=None,
    low_=None,
    variance_ratio: float = float("nan"),
    macro_bias: int = 0,
    cost_price: float = 0.0,
    sigma_yz: float = float("nan"),
    sigma_yz_median: float = float("nan"),
    strategy_mode: str = "BOTH",
    return_reason: bool = False,
) -> dict[str, Any] | tuple[dict[str, Any] | None, str]:
    """Evaluate causal trigger conditions with Net R:R Gate and Cost-to-SL protection."""
    def fail(reason: str):
        return (None, reason) if return_reason else None

    # Regime isolation filter
    mode_upper = strategy_mode.upper()
    if mode_upper == "OU_ONLY" and regime != "OU_MEAN_REVERSION":
        return fail("STRATEGY_MODE_VETO")
    elif mode_upper == "MOM_ONLY" and regime != "KALMAN_MOMENTUM":
        return fail("STRATEGY_MODE_VETO")

    arrays = [np.asarray(v, dtype=float) for v in (z_score, level, velocity, innovation_std, atr, open_, close)]
    z, x, v, s, a, o, c = arrays
    n = len(c)

    h = np.asarray(high_, dtype=float) if high_ is not None else None
    l = np.asarray(low_, dtype=float) if low_ is not None else None

    if n < 3 or any(len(q) != n for q in arrays):
        return fail("INSUFFICIENT_BARS")
    if not all(np.isfinite(q[-3:]).all() for q in arrays):
        return fail("NON_FINITE_DATA")
    if a[-1] <= 0 or not np.isfinite(hurst) or (o[-1] <= 0) or (c[-1] <= 0):
        return fail("INVALID_PRICES_OR_ATR")
    if macro_bias not in (-1, 0, 1):
        return fail("INVALID_MACRO_BIAS")

    entry = float(c[-1])
    risk = 1.6 * float(a[-1])

    # Minimum SL / Spread protection (cost must not exceed 28.5% of stop distance: risk >= 3.5 * cost)
    if cost_price > 0 and risk < 3.5 * cost_price:
        return fail("HIGH_COST_TO_SL")

    side = None
    stop = tp1 = tp2 = None
    hold = 24

    # 1. OU Mean-Reversion Regime
    if regime == "OU_MEAN_REVERSION":
        if not (2.0 <= half_life <= 40.0):
            return fail("OU_HALFLIFE_OUT_OF_BOUNDS")

        hold = dynamic_time_stop(half_life)

        # Long Mean-Reversion setup
        if np.min(z[-3:]) <= -2.0 and z[-1] > z[-2] and z[-1] >= -1.85 and c[-1] > o[-1]:
            if macro_bias < 0:
                return fail("MACRO_VETO_LONG")
            side = "LONG"
            stop = entry - risk
            eff_entry = entry + 0.5 * cost_price

            # Target 1: Kalman level; if closer than 1.1 R to effective entry, push forward to minimum 1.15 R
            tp1_raw = float(x[-1])
            if (tp1_raw - eff_entry) < 1.1 * risk:
                tp1 = max(float(x[-1] + 0.5 * s[-1]), entry + 1.15 * risk)
            else:
                tp1 = tp1_raw

            # Target 2: Mean-reversion extension
            tp2 = max(float(x[-1] + 1.2 * s[-1]), entry + 2.0 * risk)

            # Net R:R verification for TP2
            rr_tp2 = (tp2 - eff_entry) / risk
            if rr_tp2 < 1.6:
                return fail("LOW_RR")

        # Short Mean-Reversion setup
        elif np.max(z[-3:]) >= 2.0 and z[-1] < z[-2] and z[-1] <= 1.85 and c[-1] < o[-1]:
            if macro_bias > 0:
                return fail("MACRO_VETO_SHORT")
            side = "SHORT"
            stop = entry + risk
            eff_entry = entry - 0.5 * cost_price

            tp1_raw = float(x[-1])
            if (eff_entry - tp1_raw) < 1.1 * risk:
                tp1 = min(float(x[-1] - 0.5 * s[-1]), entry - 1.15 * risk)
            else:
                tp1 = tp1_raw

            tp2 = min(float(x[-1] - 1.2 * s[-1]), entry - 2.0 * risk)

            rr_tp2 = (eff_entry - tp2) / risk
            if rr_tp2 < 1.6:
                return fail("LOW_RR")

    # 2. Kalman Momentum Regime
    elif regime == "KALMAN_MOMENTUM":
        # Check Yang-Zhang volatility expansion
        if np.isfinite(sigma_yz) and np.isfinite(sigma_yz_median) and sigma_yz <= sigma_yz_median:
            return fail("YZ_VOL_NOT_EXPANDING")

        if abs(v[-1]) <= 0.05 * a[-1]:
            return fail("INSUFFICIENT_VELOCITY")

        hold = 24

        # Long Momentum:
        # 1. Kalman slope positive (v[-1] > 0)
        # 2. Kalman level trend positive (x[-1] > x[-10])
        # 3. Pullback to Kalman line (-1.0 <= Z_K <= +0.25)
        # 4. Strong breakout candle (c[-1] > o[-1] and c[-1] > high[i-1])
        pullback_long = (-1.0 <= z[-1] <= 0.25) or (-1.0 <= np.min(z[-3:]) <= 0.2 and z[-1] <= 0.8)
        trend_long = (x[-1] > x[-10]) if n >= 10 else (x[-1] > x[0])
        breakout_long = (c[-1] > o[-1]) and (c[-1] > h[-2] if (h is not None and len(h) >= 2) else True)

        if v[-1] > 0 and trend_long and pullback_long and breakout_long:
            if macro_bias < 0:
                return fail("MACRO_VETO_LONG")
            side = "LONG"
            stop = entry - risk
            eff_entry = entry + 0.5 * cost_price
            tp1 = entry + 1.5 * risk
            tp2 = entry + 2.8 * risk

            rr_tp2 = (tp2 - eff_entry) / risk
            if rr_tp2 < 1.6:
                return fail("LOW_RR")

        # Short Momentum:
        # 1. Kalman slope negative (v[-1] < 0)
        # 2. Kalman level trend negative (x[-1] < x[-10])
        # 3. Pullback to Kalman line (-0.25 <= Z_K <= +1.0)
        # 4. Strong breakdown candle (c[-1] < o[-1] and c[-1] < low[i-1])
        pullback_short = (-0.25 <= z[-1] <= 1.0) or (-0.2 <= np.max(z[-3:]) <= 1.0 and z[-1] >= -0.8)
        trend_short = (x[-1] < x[-10]) if n >= 10 else (x[-1] < x[0])
        breakdown_short = (c[-1] < o[-1]) and (c[-1] < l[-2] if (l is not None and len(l) >= 2) else True)

        if v[-1] < 0 and trend_short and pullback_short and breakdown_short:
            if macro_bias > 0:
                return fail("MACRO_VETO_SHORT")
            side = "SHORT"
            stop = entry + risk
            eff_entry = entry - 0.5 * cost_price
            tp1 = entry - 1.5 * risk
            tp2 = entry - 2.8 * risk

            rr_tp2 = (eff_entry - tp2) / risk
            if rr_tp2 < 1.6:
                return fail("LOW_RR")

    if side is None or stop is None or tp1 is None or tp2 is None:
        return fail("SETUP_NOT_TRIGGERED")

    eff_entry = entry + (0.5 * cost_price if side == "LONG" else -0.5 * cost_price)
    rr1 = abs(tp1 - eff_entry) / risk
    rr2 = abs(tp2 - eff_entry) / risk

    sig_data = {
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
        "rr_tp1": float(rr1),
        "rr_tp2": float(rr2),
    }
    return (sig_data, "SUCCESS") if return_reason else sig_data
