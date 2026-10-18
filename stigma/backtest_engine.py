"""Causal, bar-by-bar walk-forward backtest utilities with Funnel Diagnostics.

Features:
  - Adaptive 2-state Kalman Filter with dynamic variance scaling.
  - Relaxed double-fractal regime classification (Hurst + Lo-MacKinlay VR).
  - Net R:R Gate: Pre-trade evaluation of effective entry, relocation of close TP1 (min 1.15 R),
    and rejection of unfavorable geometries (TP2 < 1.6 R).
  - Cost-to-SL Protection: Rejects trades where cost exceeds 25% of stop-loss distance.
  - Institutional Session & Weekend Guard: Active trading hours (London/NY, Crypto windows)
    with Friday evening entry veto and weekend forced close.
  - Complete Funnel Diagnostics (funnel_counts) tracking bar-by-bar filtration.
"""
from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
from stigma.quant_engine import (
    calculate_hurst_vr,
    run_kalman_filter_1d,
    yang_zhang_volatility,
    estimate_ou_half_life,
    evaluate_stigma_trigger,
)

def atr14(df: pd.DataFrame) -> np.ndarray:
    """14-period standard Average True Range (causal rolling SMA)."""
    prev = df["close"].shift(1)
    tr = pd.concat([
        (df["high"] - df["low"]).abs(),
        (df["high"] - prev).abs(),
        (df["low"] - prev).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(14, min_periods=14).mean().to_numpy()

def prepare_features(df: pd.DataFrame, *, adaptive_kalman: bool = True) -> pd.DataFrame:
    """Precompute causal features: ATR14, Yang-Zhang sigma, 50-bar median YZ, Adaptive Kalman state."""
    required = {"open", "high", "low", "close"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing required OHLC columns: {required - set(df.columns)}")

    out = df.copy().sort_index()
    if out.index.has_duplicates:
        raise ValueError("Duplicate timestamps detected in price index")

    arr = out[["open", "high", "low", "close"]].to_numpy(dtype=float)
    if not np.isfinite(arr).all():
        raise ValueError("Non-finite values (NaN / Inf) found in OHLC prices")
    if (arr <= 0).any():
        raise ValueError("OHLC prices must be strictly positive")

    # Geometry verification
    o, h, l, c = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
    if (h < l).any() or (h < o).any() or (h < c).any() or (l > o).any() or (l > c).any():
        raise ValueError("Corrupt bar geometry: High must be >= Low, Open, Close")

    out["atr14"] = atr14(out)
    out["sigma_yz"] = yang_zhang_volatility(o, h, l, c)
    out["sigma_yz_median50"] = out["sigma_yz"].rolling(50, min_periods=20).median()

    k = run_kalman_filter_1d(c, out["atr14"].to_numpy(), adaptive=adaptive_kalman)
    out["kalman_level"] = k.level
    out["kalman_velocity"] = k.velocity
    out["innovation_std"] = k.innovation_std
    out["z_k"] = k.z_score
    out["rolling_spread_std"] = pd.Series(c - k.level).rolling(60, min_periods=20).std()
    out["ema_200"] = out["close"].ewm(span=200, adjust=False).mean()

    return out

def check_session_rules(
    dt: pd.Timestamp,
    session_type: str = "auto",
) -> tuple[bool, bool]:
    """Check if session is active for new entry and if Friday weekend exit is required.

    Returns:
      (can_enter, should_force_close_weekend)
    """
    if not isinstance(dt, pd.Timestamp):
        return True, False

    hour = dt.hour
    minute = dt.minute
    dow = dt.dayofweek  # 0=Monday, 4=Friday, 5=Saturday, 6=Sunday

    st = session_type.lower()
    if st == "all":
        return True, False

    is_crypto = (st in ("binance", "crypto"))

    if is_crypto:
        # Binance active hours: 08:00 - 22:00 UTC (high liquidity window)
        can_enter = (8 <= hour < 22)
        return can_enter, False
    else:
        # MT5 (Forex, Metals, Indices): London & NY sessions 06:00 - 20:00 UTC
        # Weekend: Saturday / Sunday closed
        if dow in (5, 6):
            return False, False

        # Friday restrictions:
        # After 20:30 UTC: no new entries
        # After 21:30 UTC: force close open positions before weekend
        if dow == 4:
            if hour > 21 or (hour == 21 and minute >= 30):
                return False, True
            if hour > 20 or (hour == 20 and minute >= 30):
                return False, False

        can_enter = (6 <= hour < 20)
        return can_enter, False

def backtest(
    df: pd.DataFrame,
    *,
    spread: float = 0.0,
    commission: float = 0.0,
    slippage: float = 0.0,
    macro_bias: int = 0,
    timeframe_minutes: int = 15,
    initial_capital: float = 10000.0,
    risk_per_trade_pct: float = 0.01,
    max_position_size: float = 10.0,
    tp_mode: str = "scale_out",
    session_type: str = "auto",
    adaptive_kalman: bool = True,
    strategy_mode: str = "BOTH",
) -> dict[str, Any]:
    """Causal, realistic single-position simulator with Funnel Telemetry."""
    f = prepare_features(df, adaptive_kalman=adaptive_kalman)
    n = len(f)
    trades: list[dict[str, Any]] = []
    position: dict[str, Any] | None = None
    warm = 120

    # Auto-detect session type if 'auto'
    if session_type == "auto":
        # If Saturday or Sunday present in data, it is crypto (24/7)
        dows = f.index.dayofweek
        if (dows == 5).any() or (dows == 6).any():
            eff_session = "binance"
        else:
            eff_session = "mt5"
    else:
        eff_session = session_type

    funnel = {
        "bars_evaluated": 0,
        "veto_session_hours": 0,
        "veto_random_walk": 0,
        "regime_ou_mr_bars": 0,
        "regime_kalman_mom_bars": 0,
        "passed_ou_halflife": 0,
        "passed_yz_vol": 0,
        "passed_zscore_setup": 0,
        "passed_turn_and_candle": 0,
        "rejected_high_cost_to_sl": 0,
        "rejected_low_rr": 0,
        "trades_opened": 0,
    }

    closes = f.close.to_numpy()
    opens = f.open.to_numpy()
    highs = f.high.to_numpy()
    lows = f.low.to_numpy()
    levels = f.kalman_level.to_numpy()
    velocities = f.kalman_velocity.to_numpy()
    innovation_stds = f.innovation_std.to_numpy()
    z_scores = f.z_k.to_numpy()
    atrs = f.atr14.to_numpy()
    sigmas_yz = f.sigma_yz.to_numpy()
    sigmas_yz_med = f.sigma_yz_median50.to_numpy()
    ema_200s = f.ema_200.to_numpy()
    times = f.index

    total_cost = spread + commission + slippage

    for i in range(warm, n - 1):
        nxt_open = float(opens[i + 1])
        nxt_high = float(highs[i + 1])
        nxt_low = float(lows[i + 1])
        nxt_close = float(closes[i + 1])
        nxt_time = times[i + 1]

        can_enter, should_force_close_weekend = check_session_rules(nxt_time, eff_session)

        # -----------------------------------------------------------------
        # 1. Manage Active Position (Single-position lock)
        # -----------------------------------------------------------------
        if position is not None:
            position["bars"] += 1
            side = position["side"]

            # Early Breakeven (+0.9 R) and Kalman line trailing for Momentum
            if position.get("regime") == "KALMAN_MOMENTUM":
                init_r = position["initial_risk"]
                if side == "LONG":
                    if not position.get("breakeven_triggered", False):
                        if nxt_high >= position["entry"] + 0.9 * init_r:
                            position["breakeven_triggered"] = True
                            position["stop_loss"] = max(position["stop_loss"], position["entry"])
                    if position.get("breakeven_triggered", False):
                        kalman_trail = levels[i] - 0.5 * atrs[i]
                        if kalman_trail > position["stop_loss"]:
                            position["stop_loss"] = kalman_trail
                else:  # SHORT
                    if not position.get("breakeven_triggered", False):
                        if nxt_low <= position["entry"] - 0.9 * init_r:
                            position["breakeven_triggered"] = True
                            position["stop_loss"] = min(position["stop_loss"], position["entry"])
                    if position.get("breakeven_triggered", False):
                        kalman_trail = levels[i] + 0.5 * atrs[i]
                        if kalman_trail < position["stop_loss"]:
                            position["stop_loss"] = kalman_trail

            stop = float(position["stop_loss"])
            tp1 = float(position["tp1"])
            tp2 = float(position["tp2"])
            reason = None
            exit_price = None

            # Friday weekend force close
            if should_force_close_weekend:
                exit_price = nxt_open
                reason = "weekend_close"

            elif side == "LONG":
                # Conservative rule: check stop BEFORE target
                if nxt_open <= stop:
                    exit_price = nxt_open
                    reason = "gap_stop"
                elif nxt_low <= stop:
                    exit_price = stop
                    reason = "stop"
                else:
                    if tp_mode == "scale_out":
                        if not position["tp1_hit"]:
                            if nxt_open >= tp1:
                                position["tp1_hit"] = True
                                position["tp1_exit_price"] = nxt_open
                                position["tp1_exit_bar"] = position["bars"]
                                position["stop_loss"] = position["entry"]
                                if nxt_open >= tp2:
                                    exit_price = nxt_open
                                    reason = "gap_tp2"
                                elif nxt_high >= tp2:
                                    exit_price = tp2
                                    reason = "tp2"
                            elif nxt_high >= tp1:
                                position["tp1_hit"] = True
                                position["tp1_exit_price"] = tp1
                                position["tp1_exit_bar"] = position["bars"]
                                position["stop_loss"] = position["entry"]
                                if nxt_high >= tp2:
                                    exit_price = tp2
                                    reason = "tp2"
                        else:
                            if nxt_open <= position["stop_loss"]:
                                exit_price = nxt_open
                                reason = "breakeven_gap_stop"
                            elif nxt_low <= position["stop_loss"]:
                                exit_price = position["stop_loss"]
                                reason = "breakeven_stop"
                            elif nxt_open >= tp2:
                                exit_price = nxt_open
                                reason = "gap_tp2"
                            elif nxt_high >= tp2:
                                exit_price = tp2
                                reason = "tp2"
                    else:  # tp2_only
                        if nxt_open >= tp2:
                            exit_price = nxt_open
                            reason = "gap_tp2"
                        elif nxt_high >= tp2:
                            exit_price = tp2
                            reason = "tp2"

            else:  # SHORT
                if nxt_open >= stop:
                    exit_price = nxt_open
                    reason = "gap_stop"
                elif nxt_high >= stop:
                    exit_price = stop
                    reason = "stop"
                else:
                    if tp_mode == "scale_out":
                        if not position["tp1_hit"]:
                            if nxt_open <= tp1:
                                position["tp1_hit"] = True
                                position["tp1_exit_price"] = nxt_open
                                position["tp1_exit_bar"] = position["bars"]
                                position["stop_loss"] = position["entry"]
                                if nxt_open <= tp2:
                                    exit_price = nxt_open
                                    reason = "gap_tp2"
                                elif nxt_low <= tp2:
                                    exit_price = tp2
                                    reason = "tp2"
                            elif nxt_low <= tp1:
                                position["tp1_hit"] = True
                                position["tp1_exit_price"] = tp1
                                position["tp1_exit_bar"] = position["bars"]
                                position["stop_loss"] = position["entry"]
                                if nxt_low <= tp2:
                                    exit_price = tp2
                                    reason = "tp2"
                        else:
                            if nxt_open >= position["stop_loss"]:
                                exit_price = nxt_open
                                reason = "breakeven_gap_stop"
                            elif nxt_high >= position["stop_loss"]:
                                exit_price = position["stop_loss"]
                                reason = "breakeven_stop"
                            elif nxt_open <= tp2:
                                exit_price = nxt_open
                                reason = "gap_tp2"
                            elif nxt_low <= tp2:
                                exit_price = tp2
                                reason = "tp2"
                    else:
                        if nxt_open <= tp2:
                            exit_price = nxt_open
                            reason = "gap_tp2"
                        elif nxt_low <= tp2:
                            exit_price = tp2
                            reason = "tp2"

            if reason is None and position["bars"] >= position["max_hold_bars"]:
                exit_price = nxt_close
                reason = "time_stop"

            if reason is not None and exit_price is not None:
                entry = position["entry"]
                direction = 1 if side == "LONG" else -1
                units = position["units"]
                initial_risk = position["initial_risk"]

                if position["tp1_hit"]:
                    pnl1 = (position["tp1_exit_price"] - entry) * direction
                    pnl2 = (exit_price - entry) * direction
                    gross = 0.5 * pnl1 + 0.5 * pnl2
                else:
                    gross = (exit_price - entry) * direction

                net_unit = gross - total_cost
                net_pnl = net_unit * units
                return_r = net_unit / initial_risk if initial_risk > 0 else 0.0

                trades.append({
                    **position,
                    "exit": float(exit_price),
                    "gross_pnl_unit": float(gross),
                    "net_pnl_unit": float(net_unit),
                    "net_pnl": float(net_pnl),
                    "return_r": float(return_r),
                    "exit_reason": reason,
                    "exit_time": nxt_time,
                })
                position = None
            continue

        # -----------------------------------------------------------------
        # 2. Evaluate Signals (Only when flat) & Funnel Diagnostics
        # -----------------------------------------------------------------
        funnel["bars_evaluated"] += 1

        if not can_enter:
            funnel["veto_session_hours"] += 1
            continue

        hist_close = closes[max(0, i - 99):i + 1]
        regime = calculate_hurst_vr(np.log(hist_close), window=min(100, len(hist_close)))

        if regime.label == "RANDOM_WALK_VETO":
            funnel["veto_random_walk"] += 1
            continue

        mode_upper = strategy_mode.upper()
        if mode_upper == "OU_ONLY" and regime.label != "OU_MEAN_REVERSION":
            continue
        elif mode_upper == "MOM_ONLY" and regime.label != "KALMAN_MOMENTUM":
            continue

        if regime.label == "OU_MEAN_REVERSION":
            funnel["regime_ou_mr_bars"] += 1
            start = max(0, i - 119)  # 120-bar local window
            spread_series = closes[start:i + 1] - levels[start:i + 1]
            ou = estimate_ou_half_life(spread_series)
            if not ou["valid"]:
                continue
            funnel["passed_ou_halflife"] += 1
            half_life_val = ou["half_life"]
        elif regime.label == "KALMAN_MOMENTUM":
            funnel["regime_kalman_mom_bars"] += 1
            # Yang-Zhang volatility expansion check: current sigma > 50-bar median
            curr_yz = sigmas_yz[i]
            med_yz = sigmas_yz_med[i]
            if not (np.isfinite(curr_yz) and np.isfinite(med_yz) and curr_yz > med_yz):
                continue
            funnel["passed_yz_vol"] += 1
            half_life_val = float("nan")
        else:
            continue

        # Dynamic or static macro bias
        eff_macro = (1 if closes[i] > ema_200s[i] else -1) if (str(macro_bias).lower() == "dynamic" or macro_bias == 999) else int(macro_bias)

        # Evaluate trigger conditions with reason reporting
        sig, reason = evaluate_stigma_trigger(
            regime=regime.label,
            hurst=regime.hurst,
            variance_ratio=regime.variance_ratio,
            half_life=half_life_val,
            z_score=z_scores[:i + 1],
            level=levels[:i + 1],
            velocity=velocities[:i + 1],
            innovation_std=innovation_stds[:i + 1],
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
            strategy_mode=strategy_mode,
            return_reason=True,
        )

        if reason in (
            "SETUP_NOT_TRIGGERED",
            "INSUFFICIENT_VELOCITY",
            "MACRO_VETO_LONG",
            "MACRO_VETO_SHORT",
            "STRATEGY_MODE_VETO",
            "TRENDING_SLOPE_VETO",
            "HTF_TREND_MISALIGNED",
            "OU_HALFLIFE_OUT_OF_BOUNDS",
            "YZ_VOL_NOT_EXPANDING",
        ):
            continue

        funnel["passed_zscore_setup"] += 1
        funnel["passed_turn_and_candle"] += 1

        if reason == "HIGH_COST_TO_SL":
            funnel["rejected_high_cost_to_sl"] += 1
            continue
        elif reason == "LOW_RR":
            funnel["rejected_low_rr"] += 1
            continue

        if sig is not None and reason == "SUCCESS":
            funnel["trades_opened"] += 1

            # Order executes on NEXT bar open
            entry = nxt_open
            shift = entry - float(closes[i])
            stop = float(sig["stop_loss"] + shift)
            tp1 = float(sig["tp1"] + shift)
            tp2 = float(sig["tp2"] + shift)
            initial_risk = abs(entry - stop)

            risk_capital = initial_capital * risk_per_trade_pct
            units = min(risk_capital / initial_risk, max_position_size) if initial_risk > 1e-12 else 1.0

            position = {
                **sig,
                "entry": entry,
                "stop_loss": stop,
                "tp1": tp1,
                "tp2": tp2,
                "initial_risk": initial_risk,
                "units": float(units),
                "tp1_hit": False,
                "tp1_exit_price": None,
                "tp1_exit_bar": None,
                "entry_time": nxt_time,
                "bars": 0,
            }

    # Exit any lingering open position at final close
    if position is not None:
        exit_price = float(closes[-1])
        direction = 1 if position["side"] == "LONG" else -1
        units = position["units"]
        initial_risk = position["initial_risk"]

        if position["tp1_hit"]:
            pnl1 = (position["tp1_exit_price"] - position["entry"]) * direction
            pnl2 = (exit_price - position["entry"]) * direction
            gross = 0.5 * pnl1 + 0.5 * pnl2
        else:
            gross = (exit_price - position["entry"]) * direction

        net_unit = gross - total_cost
        net_pnl = net_unit * units
        return_r = net_unit / initial_risk if initial_risk > 0 else 0.0

        trades.append({
            **position,
            "exit": exit_price,
            "gross_pnl_unit": float(gross),
            "net_pnl_unit": float(net_unit),
            "net_pnl": float(net_pnl),
            "return_r": float(return_r),
            "exit_reason": "end_of_data",
            "exit_time": times[-1],
        })

    trade_columns = [
        "side", "entry", "stop_loss", "tp1", "tp2", "max_hold_bars", "regime",
        "hurst", "variance_ratio", "half_life", "risk", "initial_risk", "units",
        "tp1_hit", "tp1_exit_price", "tp1_exit_bar", "entry_time", "bars",
        "exit", "gross_pnl_unit", "net_pnl_unit", "net_pnl", "return_r",
        "exit_reason", "exit_time"
    ]
    t = pd.DataFrame(trades) if trades else pd.DataFrame(columns=trade_columns)

    by_regime: dict[str, Any] = {}
    for reg in ["OU_MEAN_REVERSION", "KALMAN_MOMENTUM"]:
        sub_t = t.loc[t.regime == reg] if (not t.empty and "regime" in t.columns) else pd.DataFrame()
        if sub_t.empty:
            by_regime[reg] = {
                "trades": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "net_pnl": 0.0,
            }
        else:
            w = sub_t.loc[sub_t.net_pnl > 0, "net_pnl"].sum()
            l = -sub_t.loc[sub_t.net_pnl < 0, "net_pnl"].sum()
            pf_reg = float(w / l) if l > 0 else (float("inf") if w > 0 else 0.0)
            by_regime[reg] = {
                "trades": int(len(sub_t)),
                "win_rate": float((sub_t.net_pnl > 0).mean()),
                "profit_factor": pf_reg,
                "net_pnl": float(sub_t.net_pnl.sum()),
            }

    if t.empty:
        return {
            "trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "net_pnl": 0.0,
            "avg_r": 0.0,
            "expectancy": 0.0,
            "max_drawdown": 0.0,
            "max_drawdown_pct": 0.0,
            "sharpe_ratio": 0.0,
            "avg_trade_bars": 0.0,
            "long_trades": 0,
            "short_trades": 0,
            "strategy_mode": strategy_mode,
            "by_regime": by_regime,
            "trades_frame": t,
            "funnel_counts": funnel,
        }

    wins = t.loc[t.net_pnl > 0, "net_pnl"].sum()
    losses = -t.loc[t.net_pnl < 0, "net_pnl"].sum()
    pf = float(wins / losses) if losses > 0 else float("inf") if wins > 0 else 0.0

    cum_pnl = t.net_pnl.cumsum()
    peak = cum_pnl.cummax()
    drawdowns = peak - cum_pnl
    max_dd = float(drawdowns.max())
    max_dd_pct = float((drawdowns / (initial_capital + peak).clip(lower=1.0)).max())

    r_multiples = t.return_r.to_numpy()
    std_r = float(np.std(r_multiples, ddof=1)) if len(r_multiples) > 1 else 0.0
    sharpe = float(np.mean(r_multiples) / std_r * np.sqrt(252 * (1440 // timeframe_minutes))) if std_r > 0 else 0.0

    return {
        "trades": len(t),
        "win_rate": float((t.net_pnl > 0).mean()),
        "profit_factor": pf,
        "net_pnl": float(t.net_pnl.sum()),
        "avg_r": float(np.mean(r_multiples)),
        "expectancy": float(t.net_pnl.mean()),
        "max_drawdown": max_dd,
        "max_drawdown_pct": max_dd_pct,
        "sharpe_ratio": sharpe,
        "avg_trade_bars": float(t.bars.mean()),
        "long_trades": int((t.side == "LONG").sum()),
        "short_trades": int((t.side == "SHORT").sum()),
        "strategy_mode": strategy_mode,
        "by_regime": by_regime,
        "trades_frame": t,
        "funnel_counts": funnel,
    }
