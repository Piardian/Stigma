"""Stigma statistical/econometric quant research engine."""
from .quant_engine import (
    calculate_hurst_vr,
    run_kalman_filter_1d,
    yang_zhang_volatility,
    estimate_ou_half_life,
    evaluate_stigma_trigger,
)
from .execution_notifier import (
    SignalDeduplicator,
    LivePerformanceTracker,
    LivePositionTracker,
    send_telegram_alert,
    send_telegram_close_alert,
    send_telegram_message,
    format_trade_open_message,
    format_trade_close_message,
    discover_telegram_chat_id,
)

__all__ = [
    "calculate_hurst_vr",
    "run_kalman_filter_1d",
    "yang_zhang_volatility",
    "estimate_ou_half_life",
    "evaluate_stigma_trigger",
    "SignalDeduplicator",
    "LivePerformanceTracker",
    "LivePositionTracker",
    "send_telegram_alert",
    "send_telegram_close_alert",
    "send_telegram_message",
    "format_trade_open_message",
    "format_trade_close_message",
    "discover_telegram_chat_id",
]
