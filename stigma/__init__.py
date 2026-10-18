"""Stigma statistical/econometric quant research engine."""
from .quant_engine import calculate_hurst_vr, run_kalman_filter_1d, yang_zhang_volatility, estimate_ou_half_life, evaluate_stigma_trigger
from .execution_notifier import SignalDeduplicator, send_telegram_alert

__all__ = [
    "calculate_hurst_vr",
    "run_kalman_filter_1d",
    "yang_zhang_volatility",
    "estimate_ou_half_life",
    "evaluate_stigma_trigger",
    "SignalDeduplicator",
    "send_telegram_alert",
]
