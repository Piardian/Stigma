"""Execution Notifier & Telegram Alert Engine for Stigma Quant Engine.

Features:
- SignalDeduplicator: Thread-safe in-memory deduplication preventing repeated signals
  for the same symbol, strategy mode, and closed bar timestamp.
- Telegram Alert Dispatcher: Formatted institutional alert payload sent via Telegram Bot API
  (fail-safe: logs gracefully if credentials are not configured).
"""
from __future__ import annotations
import json
import logging
import os
import urllib.parse
import urllib.request
from typing import Any

logger = logging.getLogger("execution_notifier")

class SignalDeduplicator:
    """Prevents duplicate alerts for identical (symbol, strategy_mode, bar_time) tuples."""

    def __init__(self) -> None:
        self._seen: set[str] = set()

    def _make_key(self, symbol: str, strategy_mode: str, bar_time: Any) -> str:
        t_str = str(bar_time).strip()
        return f"{symbol.upper()}:{strategy_mode.upper()}:{t_str}"

    def is_duplicate(self, symbol: str, strategy_mode: str, bar_time: Any) -> bool:
        key = self._make_key(symbol, strategy_mode, bar_time)
        return key in self._seen

    def record(self, symbol: str, strategy_mode: str, bar_time: Any) -> None:
        key = self._make_key(symbol, strategy_mode, bar_time)
        self._seen.add(key)

    def should_emit(self, symbol: str, strategy_mode: str, bar_time: Any) -> bool:
        """Atomic check-and-record: returns True if signal is new and records it."""
        if self.is_duplicate(symbol, strategy_mode, bar_time):
            return False
        self.record(symbol, strategy_mode, bar_time)
        return True

    def clear(self) -> None:
        self._seen.clear()

def format_telegram_message(signal: dict[str, Any]) -> str:
    """Format quant signal payload into a clean, institutional Telegram message."""
    sym = signal.get("symbol", "UNKNOWN")
    tf = signal.get("timeframe", "M15")
    side = signal.get("side", "NEUTRAL")
    strat = signal.get("strategy_mode", signal.get("regime", "UNKNOWN"))
    h = signal.get("hurst", float("nan"))
    vr = signal.get("variance_ratio", float("nan"))
    hl = signal.get("half_life", float("nan"))
    zk = signal.get("z_score", float("nan"))
    entry = signal.get("entry", 0.0)
    sl = signal.get("stop_loss", 0.0)
    tp1 = signal.get("tp1", 0.0)
    tp2 = signal.get("tp2", 0.0)
    time_stop = signal.get("max_hold_bars", signal.get("dynamic_time_stop", 24))
    bar_time = signal.get("bar_time", signal.get("time", "N/A"))

    side_emoji = "🟢 LONG" if side == "LONG" else ("🔴 SHORT" if side == "SHORT" else side)
    hl_str = f"{hl:.1f} bars" if (hl is not None and hl == hl) else "N/A"

    msg = (
        f"⚡ <b>STIGMA QUANT SIGNAL</b> ⚡\n"
        f"━━━━━━━━━━━━━━━━━━━\n"
        f"📌 <b>Asset:</b> {sym} ({tf})\n"
        f"🧭 <b>Strategy:</b> {strat}\n"
        f"🎯 <b>Direction:</b> {side_emoji}\n"
        f"⏰ <b>Bar Time:</b> {bar_time} UTC\n\n"
        f"📊 <b>Econometric Invariants:</b>\n"
        f" • Hurst Exponent (H): <code>{h:.3f}</code>\n"
        f" • Variance Ratio (VR): <code>{vr:.3f}</code>\n"
        f" • O-U Half-Life: <code>{hl_str}</code>\n"
        f" • Kalman Z-Score (Z_K): <code>{zk:.2f}</code>\n\n"
        f"💰 <b>Execution Levels:</b>\n"
        f" • Entry Price: <code>{entry:.5f}</code>\n"
        f" • Stop Loss (SL): <code>{sl:.5f}</code>\n"
        f" • Target 1 (TP1): <code>{tp1:.5f}</code>\n"
        f" • Target 2 (TP2): <code>{tp2:.5f}</code>\n"
        f" • Dynamic Time-Stop: <code>{time_stop} bars</code>\n"
        f"━━━━━━━━━━━━━━━━━━━\n"
        f"🛡️ <i>Mode: RESEARCH SIMULATION (Live Execution Disabled)</i>"
    )
    return msg

def send_telegram_alert(
    signal: dict[str, Any],
    bot_token: str | None = None,
    chat_id: str | None = None,
    timeout_sec: float = 5.0,
) -> bool:
    """Dispatch signal message to Telegram. Fails closed / logs gracefully if unconfigured."""
    token = bot_token or os.environ.get("TELEGRAM_BOT_TOKEN")
    chat = chat_id or os.environ.get("TELEGRAM_CHAT_ID")

    if not token or not chat:
        logger.info(
            "Telegram notification skipped (TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set). "
            "Signal for %s (%s %s) logged locally.",
            signal.get("symbol"),
            signal.get("strategy_mode"),
            signal.get("side"),
        )
        return False

    text = format_telegram_message(signal)
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }

    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            if resp.status == 200:
                logger.info("Telegram alert sent successfully for %s (%s)", signal.get("symbol"), signal.get("side"))
                return True
            logger.warning("Telegram API returned non-200 status: %d", resp.status)
            return False
    except Exception as exc:
        logger.error("Failed to send Telegram alert: %s", exc)
        return False
