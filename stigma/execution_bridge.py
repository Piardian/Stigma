"""Fail-closed signal handoff. Live order placement is intentionally disabled.

Produces a validated order intent representation for research, verification,
and simulation without sending orders to live broker accounts.
"""
from __future__ import annotations
from typing import Any

def build_order_intent(
    signal: dict[str, Any] | None,
    *,
    allow_live: bool = False,
) -> dict[str, Any] | None:
    """Validate signal and return an order intent structure.

    Live execution is permanently disabled by default.
    Raises ValueError on corrupt or incomplete signal geometry.
    """
    if not signal:
        return None

    required = {"side", "entry", "stop_loss", "tp1", "tp2", "max_hold_bars"}
    missing = required - set(signal.keys())
    if missing:
        raise ValueError(f"Signal is missing required fields: {missing}")

    side = signal["side"]
    if side not in ("LONG", "SHORT"):
        raise ValueError(f"Invalid signal side: '{side}'")

    entry = float(signal["entry"])
    stop = float(signal["stop_loss"])
    tp1 = float(signal["tp1"])
    tp2 = float(signal["tp2"])
    hold = int(signal["max_hold_bars"])

    if entry <= 0 or stop <= 0 or tp1 <= 0 or tp2 <= 0:
        raise ValueError("Prices in signal must be strictly positive")

    if side == "LONG":
        if not (stop < entry < tp1 <= tp2):
            raise ValueError(
                f"Invalid LONG geometry: stop ({stop}) < entry ({entry}) < tp1 ({tp1}) <= tp2 ({tp2}) violated"
            )
    else:  # SHORT
        if not (stop > entry > tp1 >= tp2):
            raise ValueError(
                f"Invalid SHORT geometry: stop ({stop}) > entry ({entry}) > tp1 ({tp1}) >= tp2 ({tp2}) violated"
            )

    if hold < 1 or hold > 100:
        raise ValueError(f"Invalid max_hold_bars: {hold}")

    return {
        "mode": "LIVE_DISABLED" if not allow_live else "INTENT_ONLY",
        "order_type": "MARKET_NEXT_BAR",
        "side": side,
        "reference_entry": entry,
        "stop_loss": stop,
        "tp1": tp1,
        "tp2": tp2,
        "max_hold_bars": hold,
        "regime": signal.get("regime", "UNKNOWN"),
        "hurst": float(signal.get("hurst", float("nan"))),
        "variance_ratio": float(signal.get("variance_ratio", float("nan"))),
        "risk_distance": abs(entry - stop),
        "requires_external_risk_checks": True,
    }
