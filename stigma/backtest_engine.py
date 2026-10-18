"""Causal, bar-by-bar walk-forward backtest utilities.

All calculations strictly causal:
  - Features evaluated on closed bar i.
  - Entry order executed on bar i+1 Open.
  - Geometry: Entry shifts with Next Open, preserving exact stop and target distance.
  - Gap Stop: If bar opens beyond stop, execution occurs at Open (not stale stop).
  - Intrabar Conflict: If stop and target both fall in the same candle, stop is checked first (conservative).
  - TP1 / TP2 Management: Institutional scale-out; 50% locked in at TP1, stop trailed to breakeven (entry), 50% targeting TP2.
  - Single-Position: Strictly one position at a time (no concurrent positions, no signal leakage).
  - Realistic Costs: Spread, commission, and slippage explicitly deducted.
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

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Precompute causal features: ATR14, Yang-Zhang sigma, Kalman filter state."""
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

    k = run_kalman_filter_1d(c, out["atr14"].to_numpy())
    out["kalman_level"] = k.level
    out["kalman_velocity"] = k.velocity
    out["innovation_std"] = k.innovation_std
    out["z_k"] = k.z_score

    return out

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
) -> dict[str, Any]:
    """Causal, realistic single-position simulator.

    Parameters:
      df: DataFrame with DatetimeIndex and open, high, low, close columns.
      spread: Round-trip spread in price units.
      commission: Round-trip commission in price units.
      slippage: Adverse slippage per round-trip in price units.
      macro_bias: -1, 0, or +1.
      timeframe_minutes: Bar interval (5 or 15).
      initial_capital: Account starting equity.
      risk_per_trade_pct: Fraction of capital risked per trade (0.01 = 1%).
      max_position_size: Maximum allowed position units.
      tp_mode: 'scale_out' (50% at TP1 with BE stop trailing, 50% at TP2) or 'tp2_only'.
    """
    f = prepare_features(df)
    n = len(f)
    trades: list[dict[str, Any]] = []
    position: dict[str, Any] | None = None
    warm = 120

    closes = f.close.to_numpy()
    opens = f.open.to_numpy()
    highs = f.high.to_numpy()
    lows = f.low.to_numpy()
    levels = f.kalman_level.to_numpy()
    velocities = f.kalman_velocity.to_numpy()
    innovation_stds = f.innovation_std.to_numpy()
    z_scores = f.z_k.to_numpy()
    atrs = f.atr14.to_numpy()
    times = f.index

    for i in range(warm, n - 1):
        nxt_open = float(opens[i + 1])
        nxt_high = float(highs[i + 1])
        nxt_low = float(lows[i + 1])
        nxt_close = float(closes[i + 1])
        nxt_time = times[i + 1]

        # -----------------------------------------------------------------
        # 1. Manage Active Position (Single-position lock)
        # -----------------------------------------------------------------
        if position is not None:
            position["bars"] += 1
            side = position["side"]
            stop = float(position["stop_loss"])
            tp1 = float(position["tp1"])
            tp2 = float(position["tp2"])
            reason = None
            exit_price = None

            if side == "LONG":
                # Conservative rule: check stop BEFORE target
                if nxt_open <= stop:
                    exit_price = nxt_open
                    reason = "gap_stop"
                elif nxt_low <= stop:
                    exit_price = stop
                    reason = "stop"
                else:
                    # Check targets
                    if tp_mode == "scale_out":
                        if not position["tp1_hit"]:
                            if nxt_open >= tp1:
                                position["tp1_hit"] = True
                                position["tp1_exit_price"] = nxt_open
                                position["tp1_exit_bar"] = position["bars"]
                                position["stop_loss"] = position["entry"]  # Move stop to breakeven
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
                                position["stop_loss"] = position["entry"]  # Move stop to breakeven
                                if nxt_high >= tp2:
                                    exit_price = tp2
                                    reason = "tp2"
                        else:
                            # TP1 already hit on an earlier bar; stop is at breakeven
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
                # Conservative rule: check stop BEFORE target
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
                                position["stop_loss"] = position["entry"]  # Move stop to breakeven
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
                                position["stop_loss"] = position["entry"]  # Move stop to breakeven
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

            # Check time-stop if position has not exited
            if reason is None and position["bars"] >= position["max_hold_bars"]:
                exit_price = nxt_close
                reason = "time_stop"

            # If position exited, record trade
            if reason is not None and exit_price is not None:
                entry = position["entry"]
                direction = 1 if side == "LONG" else -1
                total_cost = spread + commission + slippage
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
        # 2. Evaluate Signals (Only when flat)
        # -----------------------------------------------------------------
        hist_close = closes[max(0, i - 99):i + 1]
        regime = calculate_hurst_vr(np.log(hist_close), window=min(100, len(hist_close)))
        if regime.label not in ("OU_MEAN_REVERSION", "KALMAN_MOMENTUM"):
            continue

        start = max(0, i - 250)
        spread_series = closes[start:i + 1] - levels[start:i + 1]
        ou = estimate_ou_half_life(spread_series)

        sig = evaluate_stigma_trigger(
            regime=regime.label,
            hurst=regime.hurst,
            variance_ratio=regime.variance_ratio,
            half_life=ou["half_life"],
            z_score=z_scores[:i + 1],
            level=levels[:i + 1],
            velocity=velocities[:i + 1],
            innovation_std=innovation_stds[:i + 1],
            atr=atrs[:i + 1],
            open_=opens[:i + 1],
            close=closes[:i + 1],
            macro_bias=macro_bias,
        )

        if sig is not None:
            # Entry executes on NEXT bar open
            entry = nxt_open
            # Shift SL and TP distances to match execution price
            shift = entry - float(closes[i])
            stop = float(sig["stop_loss"] + shift)
            tp1 = float(sig["tp1"] + shift)
            tp2 = float(sig["tp2"] + shift)
            initial_risk = abs(entry - stop)

            # Position sizing
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

    # If position remains open at the end of the data, exit at final close
    if position is not None:
        exit_price = float(closes[-1])
        direction = 1 if position["side"] == "LONG" else -1
        total_cost = spread + commission + slippage
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
            "trades_frame": t,
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
        "trades_frame": t,
    }
