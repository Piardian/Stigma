"""Fail-closed signal handoff. Live order placement is intentionally not implemented."""
from __future__ import annotations
from typing import Any

def build_order_intent(signal: dict[str, Any] | None, *, allow_live: bool=False) -> dict[str, Any] | None:
    if not signal: return None
    required={"side","entry","stop_loss","tp1","tp2","max_hold_bars"}
    if not required.issubset(signal): raise ValueError(f"signal missing fields: {required-set(signal)}")
    if signal["side"] not in ("LONG","SHORT"): raise ValueError("invalid side")
    entry=float(signal["entry"]); stop=float(signal["stop_loss"])
    if entry<=0 or stop<=0 or (signal["side"]=="LONG" and stop>=entry) or (signal["side"]=="SHORT" and stop<=entry):
        raise ValueError("invalid entry/stop geometry")
    return {"mode":"LIVE_DISABLED" if not allow_live else "INTENT_ONLY","order_type":"MARKET_NEXT_BAR","side":signal["side"],"reference_entry":entry,"stop_loss":stop,"tp1":float(signal["tp1"]),"tp2":float(signal["tp2"]),"max_hold_bars":int(signal["max_hold_bars"]),"requires_external_risk_checks":True}
