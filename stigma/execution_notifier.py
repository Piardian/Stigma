"""Execution Notifier, Live Position Tracking & Telegram Alert Engine for Stigma Quant Engine.

Features:
- SignalDeduplicator: Thread-safe in-memory deduplication preventing repeated signals
  for the same symbol, strategy mode, and closed bar timestamp.
- LivePerformanceTracker: Persistent JSON tracker (`reports/performance_tracker.json`)
  maintaining cumulative TP/SL counters, Win Rate, Net PnL, and Cumulative Net R.
- LivePositionTracker: Persistent active position tracker (`reports/live_positions.json`)
  evaluating live bar updates against scale-out TP1, trailing stop, TP2, SL, and time stop.
- Telegram Alert Dispatcher: Formatted institutional alert payloads for trade open and close
  (fail-safe: logs gracefully if credentials are not configured).
- Automatic .env loading and chat_id discovery.
"""
from __future__ import annotations
import json
import logging
import os
import time
from pathlib import Path
from typing import Any
import urllib.parse
import urllib.request

logger = logging.getLogger("execution_notifier")

# Default paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"
PERFORMANCE_TRACKER_PATH = PROJECT_ROOT / "reports" / "performance_tracker.json"
LIVE_POSITIONS_PATH = PROJECT_ROOT / "reports" / "live_positions.json"


def load_env_file(env_file: Path | str | None = None) -> dict[str, str]:
    """Parse key=value pairs from .env and inject into os.environ if not already set."""
    path = Path(env_file) if env_file else ENV_PATH
    loaded: dict[str, str] = {}
    if not path.exists():
        return loaded

    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip("'\"")
                if k:
                    loaded[k] = v
                    if k not in os.environ or not os.environ[k]:
                        os.environ[k] = v
    except Exception as exc:
        logger.warning("Could not read .env file at %s: %s", path, exc)
    return loaded


# Auto-load on import
load_env_file()


def save_env_var(key: str, value: str, env_file: Path | str | None = None) -> None:
    """Safely append or update a key=value pair in .env file."""
    path = Path(env_file) if env_file else ENV_PATH
    lines: list[str] = []
    found = False

    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except Exception:
            lines = []

    new_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(f"{key}=") or stripped.startswith(f"{key} ="):
            new_lines.append(f"{key}={value}\n")
            found = True
        else:
            new_lines.append(line)

    if not found:
        if new_lines and not new_lines[-1].endswith("\n"):
            new_lines[-1] += "\n"
        new_lines.append(f"{key}={value}\n")

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(new_lines)
        os.environ[key] = value
        logger.info("Saved %s to %s", key, path)
    except Exception as exc:
        logger.error("Failed to save %s to %s: %s", key, path, exc)


def discover_telegram_chat_id(
    bot_token: str | None = None,
    timeout_sec: float = 3.0,
    save_to_env: bool = True,
) -> tuple[str | None, str | None]:
    """Query Telegram getUpdates to auto-discover user chat_id and username."""
    token = bot_token or os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        return None, None

    url = f"https://api.telegram.org/bot{token}/getUpdates"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            if not data.get("ok"):
                return None, None
            results = data.get("result", [])
            if not results:
                return None, None

            # Get the most recent message
            for upd in reversed(results):
                msg = upd.get("message") or upd.get("channel_post") or upd.get("my_chat_member")
                if msg and "chat" in msg:
                    chat_id = str(msg["chat"]["id"])
                    user_name = msg["chat"].get("first_name") or msg["chat"].get("title") or msg["chat"].get("username", "Trader")
                    if save_to_env:
                        save_env_var("TELEGRAM_CHAT_ID", chat_id)
                    return chat_id, user_name
    except Exception as exc:
        logger.debug("getUpdates error: %s", exc)
    return None, None


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


def format_trade_open_message(signal: dict[str, Any]) -> str:
    """Format quant entry signal payload into an institutional Telegram open alert."""
    sym = signal.get("symbol", "UNKNOWN")
    tf = signal.get("timeframe", "M15")
    side = signal.get("side", "NEUTRAL")
    strat = signal.get("strategy_mode", signal.get("regime", "UNKNOWN"))
    macro = signal.get("macro_mode", "Autonomous")
    h = signal.get("hurst", float("nan"))
    vr = signal.get("variance_ratio", float("nan"))
    hl = signal.get("half_life", float("nan"))
    zk = signal.get("z_score", float("nan"))
    entry = signal.get("entry", 0.0)
    sl = signal.get("stop_loss", 0.0)
    tp1 = signal.get("tp1", 0.0)
    tp2 = signal.get("tp2", 0.0)
    risk = signal.get("risk", abs(entry - sl) if sl > 0 else 0.0)
    time_stop = signal.get("max_hold_bars", signal.get("dynamic_time_stop", 24))
    bar_time = signal.get("bar_time", signal.get("time", "N/A"))

    side_emoji = "🟢 LONG (ALIŞ)" if side == "LONG" else ("🔴 SHORT (SATIŞ)" if side == "SHORT" else side)
    hl_str = f"{hl:.1f} bars" if (hl is not None and hl == hl) else "N/A"

    msg = (
        f"⚡ <b>STIGMA QUANT — İŞLEM AÇILDI</b> ⚡\n"
        f"━━━━━━━━━━━━━━━━━━━\n"
        f"📌 <b>Varlık:</b> {sym} ({tf})\n"
        f"🧭 <b>Strateji:</b> {strat} | {macro}\n"
        f"🎯 <b>Yön:</b> {side_emoji}\n"
        f"⏰ <b>Bar Zamanı:</b> {bar_time} UTC\n\n"
        f"💰 <b>İşlem ve Risk Seviyeleri:</b>\n"
        f" • Giriş Fiyatı: <code>{entry:.5f}</code>\n"
        f" • Stop Loss (SL): <code>{sl:.5f}</code>\n"
        f" • Hedef 1 (TP1 - %50): <code>{tp1:.5f}</code>\n"
        f" • Hedef 2 (TP2 - %50): <code>{tp2:.5f}</code>\n"
        f" • 1.0 R Risk Mesafesi: <code>{risk:.5f}</code>\n"
        f" • Dinamik Zaman Stopu: <code>{time_stop} bar</code>\n\n"
        f"📊 <b>Ekonometrik Doğrulama:</b>\n"
        f" • Hurst Exponent (H): <code>{h:.3f}</code>\n"
        f" • Variance Ratio (VR): <code>{vr:.3f}</code>\n"
        f" • O-U Half-Life: <code>{hl_str}</code>\n"
        f" • Kalman Z-Score (Z_K): <code>{zk:.2f}</code>\n"
        f"━━━━━━━━━━━━━━━━━━━\n"
        f"🛡️ <i>Mod: RESEARCH SIMULATION (Stigma Live Monitor)</i>"
    )
    return msg


# Backward compatibility
format_telegram_message = format_trade_open_message


def format_trade_close_message(trade: dict[str, Any], stats: dict[str, Any]) -> str:
    """Format trade close payload including Net PnL, Net R, Post-Commission Net R, and TP/SL counters."""
    sym = trade.get("symbol", "UNKNOWN")
    tf = trade.get("timeframe", "M15")
    side = trade.get("side", "LONG")
    strat = trade.get("strategy_mode", "OU_ONLY")
    entry = trade.get("entry", 0.0)
    exit_p = trade.get("exit", 0.0)
    tp1_exit = trade.get("tp1_exit_price")
    tp1_hit = trade.get("tp1_hit", False)
    risk = trade.get("initial_risk", 1.0)
    bars_held = trade.get("bars_held", trade.get("bars", 0))
    reason = trade.get("exit_reason", "exit")

    net_pnl = trade.get("net_pnl", 0.0)
    gross_r = trade.get("gross_r", trade.get("return_r", 0.0))
    net_r = trade.get("net_r", trade.get("return_r", 0.0))

    # Determine human readable exit reason and icon
    reason_map = {
        "tp2": "🎯 HEDEF 2 (TP2) ALINDI — Tam Hedef Kapatıldı",
        "gap_tp2": "🎯 HEDEF 2 (TP2) GAP İLE ALINDI",
        "breakeven_stop": "⚖️ BAŞABAŞ (BREAKEVEN) STOP — TP1 Kârı Korundu",
        "breakeven_gap_stop": "⚖️ BAŞABAŞ GAP STOP — TP1 Kârı Korundu",
        "stop": "🛑 STOP LOSS ÇALIŞTI",
        "gap_stop": "🛑 GAP İLE STOP LOSS ÇALIŞTI",
        "trailing_stop": "📈 KALMAN İZ SÜREN STOP (Trailing Stop)",
        "time_stop": "⏱️ DİNAMİK ZAMAN STOPU — Maksimum Bar Vadesi Doldu",
        "weekend_close": "🛡️ HAFTA SONU RİSK KORUMASI — Cuma Seans Sonu",
    }
    reason_text = reason_map.get(str(reason).lower(), f"🔔 {reason.upper()}")

    # Sign prefixes
    pnl_prefix = "+" if net_pnl >= 0 else ""
    gross_r_prefix = "+" if gross_r >= 0 else ""
    net_r_prefix = "+" if net_r >= 0 else ""

    # Stats counters
    total_trades = stats.get("total_trades", 0)
    tp_count = stats.get("tp_count", 0)
    sl_count = stats.get("sl_count", 0)
    be_count = stats.get("breakeven_count", 0)
    win_rate = stats.get("win_rate", 0.0)
    cum_net_r = stats.get("cumulative_net_r", 0.0)
    cum_pnl = stats.get("cumulative_net_pnl", 0.0)
    cum_r_prefix = "+" if cum_net_r >= 0 else ""
    cum_pnl_prefix = "+" if cum_pnl >= 0 else ""

    tp1_detail = f" • TP1 Çıkış (%50): <code>{tp1_exit:.5f}</code>\n" if (tp1_hit and tp1_exit) else ""

    msg = (
        f"🏁 <b>STIGMA QUANT — İŞLEM KAPANDI</b> 🏁\n"
        f"━━━━━━━━━━━━━━━━━━━\n"
        f"📌 <b>Varlık:</b> {sym} ({tf})\n"
        f"🧭 <b>Strateji:</b> {strat}\n"
        f"🎯 <b>Yön:</b> {'🟢 LONG' if side == 'LONG' else '🔴 SHORT'}\n"
        f"🚪 <b>Kapanış Nedeni:</b> {reason_text}\n"
        f"⏱️ <b>Pozisyon Süresi:</b> {bars_held} bar\n\n"
        f"💵 <b>Fiyat Özeti:</b>\n"
        f" • Giriş Fiyatı: <code>{entry:.5f}</code>\n"
        f"{tp1_detail}"
        f" • Nihai Çıkış Fiyatı: <code>{exit_p:.5f}</code>\n"
        f" • 1.0 R Risk Mesafesi: <code>{risk:.5f}</code>\n\n"
        f"📊 <b>FİNANSAL GETİRİ (PNL & R):</b>\n"
        f" • <b>Net PnL:</b> <code>{pnl_prefix}{net_pnl:.2f} USD</code>\n"
        f" • <b>Net R (Brüt / Komisyonsuz):</b> <code>{gross_r_prefix}{gross_r:.2f} R</code>\n"
        f" • <b>Komisyon Sonrası Net R:</b> <code>{net_r_prefix}{net_r:.2f} R</code>\n\n"
        f"🏆 <b>KÜMÜLATİF PERFORMANS SAYACI:</b>\n"
        f" 🎯 <b>TP Sayacı (Kazanç):</b> <code>{tp_count}</code>\n"
        f" 🛑 <b>SL Sayacı (Kayıp):</b> <code>{sl_count}</code>\n"
        f" ⚖️ <b>Başabaş (BE):</b> <code>{be_count}</code>\n"
        f" 📊 <b>Kazanma Oranı (Win Rate):</b> <code>%{win_rate:.1f}</code>\n"
        f" 📈 <b>Kümülatif Net R:</b> <code>{cum_r_prefix}{cum_net_r:.2f} R</code>\n"
        f" 💰 <b>Toplam Net Kâr:</b> <code>{cum_pnl_prefix}{cum_pnl:.2f} USD</code>\n"
        f"━━━━━━━━━━━━━━━━━━━\n"
        f"🛡️ <i>Stigma Quant Engine v5.0 — Live Execution Monitor</i>"
    )
    return msg


def send_telegram_message(
    text: str,
    bot_token: str | None = None,
    chat_id: str | None = None,
    timeout_sec: float = 5.0,
) -> bool:
    """Send arbitrary HTML message to Telegram API. Fails closed / logs gracefully."""
    token = bot_token or os.environ.get("TELEGRAM_BOT_TOKEN")
    chat = chat_id or os.environ.get("TELEGRAM_CHAT_ID")

    # If chat_id is missing, attempt quick discovery once
    if token and not chat:
        chat, _ = discover_telegram_chat_id(token, timeout_sec=2.0)

    if not token or not chat:
        logger.info("Telegram notification skipped (TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not configured).")
        return False

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
                return True
            logger.warning("Telegram API returned non-200 status: %d", resp.status)
            return False
    except Exception as exc:
        logger.error("Failed to send Telegram message: %s", exc)
        return False


def send_telegram_alert(
    signal: dict[str, Any],
    bot_token: str | None = None,
    chat_id: str | None = None,
    timeout_sec: float = 5.0,
) -> bool:
    """Dispatch trade entry message to Telegram."""
    text = format_trade_open_message(signal)
    success = send_telegram_message(text, bot_token=bot_token, chat_id=chat_id, timeout_sec=timeout_sec)
    if success:
        logger.info("Telegram open alert sent for %s (%s)", signal.get("symbol"), signal.get("side"))
    return success


def send_telegram_close_alert(
    trade: dict[str, Any],
    stats: dict[str, Any],
    bot_token: str | None = None,
    chat_id: str | None = None,
    timeout_sec: float = 5.0,
) -> bool:
    """Dispatch trade closing message to Telegram."""
    text = format_trade_close_message(trade, stats)
    success = send_telegram_message(text, bot_token=bot_token, chat_id=chat_id, timeout_sec=timeout_sec)
    if success:
        logger.info("Telegram close alert sent for %s (%s, reason=%s)", trade.get("symbol"), trade.get("side"), trade.get("exit_reason"))
    return success


class LivePerformanceTracker:
    """Maintains persistent cumulative trade performance statistics and counters."""

    def __init__(self, filepath: Path | str | None = None) -> None:
        self.filepath = Path(filepath) if filepath else PERFORMANCE_TRACKER_PATH
        self.stats: dict[str, Any] = {
            "total_trades": 0,
            "tp_count": 0,
            "sl_count": 0,
            "breakeven_count": 0,
            "time_stop_count": 0,
            "cumulative_gross_r": 0.0,
            "cumulative_net_r": 0.0,
            "cumulative_net_pnl": 0.0,
            "win_rate": 0.0,
            "history": [],
        }
        self.load()

    def load(self) -> None:
        if self.filepath.exists():
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.stats.update(data)
            except Exception as exc:
                logger.warning("Failed to load performance tracker from %s: %s", self.filepath, exc)

    def save(self) -> None:
        try:
            self.filepath.parent.mkdir(parents=True, exist_ok=True)
            with open(self.filepath, "w", encoding="utf-8") as f:
                json.dump(self.stats, f, indent=2)
        except Exception as exc:
            logger.error("Failed to save performance tracker to %s: %s", self.filepath, exc)

    def record_closed_trade(self, trade: dict[str, Any]) -> dict[str, Any]:
        """Update cumulative counters based on closed trade outcome."""
        net_r = float(trade.get("net_r", 0.0))
        gross_r = float(trade.get("gross_r", net_r))
        net_pnl = float(trade.get("net_pnl", 0.0))
        reason = str(trade.get("exit_reason", "")).lower()

        # Counter classification:
        # If net_r > +0.05 or explicit TP2 hit -> TP
        # If net_r < -0.05 or explicit SL hit -> SL
        # Otherwise -> Breakeven
        if net_r > 0.05 or "tp2" in reason:
            self.stats["tp_count"] += 1
        elif net_r < -0.05 or "stop" in reason and "breakeven" not in reason:
            self.stats["sl_count"] += 1
        else:
            self.stats["breakeven_count"] += 1

        if "time_stop" in reason:
            self.stats["time_stop_count"] += 1

        self.stats["total_trades"] = (
            self.stats["tp_count"] + self.stats["sl_count"] + self.stats["breakeven_count"]
        )
        self.stats["cumulative_gross_r"] = round(self.stats["cumulative_gross_r"] + gross_r, 4)
        self.stats["cumulative_net_r"] = round(self.stats["cumulative_net_r"] + net_r, 4)
        self.stats["cumulative_net_pnl"] = round(self.stats["cumulative_net_pnl"] + net_pnl, 2)

        tot = self.stats["total_trades"]
        self.stats["win_rate"] = round((self.stats["tp_count"] / tot * 100.0), 1) if tot > 0 else 0.0

        # Compact history record
        hist_entry = {
            "symbol": trade.get("symbol"),
            "tf": trade.get("timeframe"),
            "side": trade.get("side"),
            "entry": trade.get("entry"),
            "exit": trade.get("exit"),
            "net_pnl": round(net_pnl, 2),
            "gross_r": round(gross_r, 2),
            "net_r": round(net_r, 2),
            "exit_reason": trade.get("exit_reason"),
            "time": trade.get("exit_time", str(time.strftime("%Y-%m-%d %H:%M:%S"))),
        }
        self.stats["history"].append(hist_entry)
        if len(self.stats["history"]) > 500:
            self.stats["history"] = self.stats["history"][-500:]

        self.save()
        return self.get_stats()

    def get_stats(self) -> dict[str, Any]:
        return dict(self.stats)

    def reset(self) -> None:
        self.stats = {
            "total_trades": 0,
            "tp_count": 0,
            "sl_count": 0,
            "breakeven_count": 0,
            "time_stop_count": 0,
            "cumulative_gross_r": 0.0,
            "cumulative_net_r": 0.0,
            "cumulative_net_pnl": 0.0,
            "win_rate": 0.0,
            "history": [],
        }
        self.save()


class LivePositionTracker:
    """Tracks active open positions and evaluates them bar-by-bar against scale-out, trailing stop, and TP/SL."""

    def __init__(
        self,
        filepath: Path | str | None = None,
        perf_tracker: LivePerformanceTracker | None = None,
    ) -> None:
        self.filepath = Path(filepath) if filepath else LIVE_POSITIONS_PATH
        self.perf_tracker = perf_tracker or LivePerformanceTracker()
        self.active_positions: dict[str, dict[str, Any]] = {}
        self.load()

    def _make_key(self, symbol: str, strategy_mode: str) -> str:
        return f"{symbol.upper()}:{strategy_mode.upper()}"

    def load(self) -> None:
        if self.filepath.exists():
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    self.active_positions = json.load(f)
            except Exception as exc:
                logger.warning("Failed to load active positions from %s: %s", self.filepath, exc)

    def save(self) -> None:
        try:
            self.filepath.parent.mkdir(parents=True, exist_ok=True)
            with open(self.filepath, "w", encoding="utf-8") as f:
                json.dump(self.active_positions, f, indent=2)
        except Exception as exc:
            logger.error("Failed to save active positions to %s: %s", self.filepath, exc)

    def has_position(self, symbol: str, strategy_mode: str) -> bool:
        return self._make_key(symbol, strategy_mode) in self.active_positions

    def get_position(self, symbol: str, strategy_mode: str) -> dict[str, Any] | None:
        return self.active_positions.get(self._make_key(symbol, strategy_mode))

    def open_position(self, signal: dict[str, Any], units: float = 1.0, cost_price: float = 0.0) -> dict[str, Any]:
        """Register a new open simulated/paper position."""
        sym = signal["symbol"].upper()
        strat = signal.get("strategy_mode", "OU_ONLY").upper()
        key = self._make_key(sym, strat)

        entry = float(signal["entry"])
        sl = float(signal["stop_loss"])
        tp1 = float(signal["tp1"])
        tp2 = float(signal["tp2"])
        risk = float(signal.get("risk", abs(entry - sl)))
        max_bars = int(signal.get("max_hold_bars", signal.get("dynamic_time_stop", 24)))

        pos = {
            "key": key,
            "symbol": sym,
            "timeframe": signal.get("timeframe", "M15"),
            "strategy_mode": strat,
            "macro_mode": signal.get("macro_mode", "Autonomous"),
            "side": signal["side"],
            "entry": entry,
            "stop_loss": sl,
            "initial_stop_loss": sl,
            "tp1": tp1,
            "tp2": tp2,
            "initial_risk": max(risk, 1e-8),
            "max_hold_bars": max_bars,
            "bars_held": 0,
            "tp1_hit": False,
            "tp1_exit_price": None,
            "breakeven_triggered": False,
            "cost_price": cost_price,
            "units": units,
            "entry_time": signal.get("bar_time", str(time.strftime("%Y-%m-%d %H:%M:%S"))),
        }
        self.active_positions[key] = pos
        self.save()
        logger.info("Opened live position: %s %s %s @ %.5f (SL: %.5f, TP1: %.5f, TP2: %.5f)", sym, strat, pos["side"], entry, sl, tp1, tp2)
        return pos

    def update_candle(
        self,
        symbol: str,
        strategy_mode: str,
        candle: dict[str, Any],
        kalman_level: float = 0.0,
        atr: float = 0.0,
        should_weekend_close: bool = False,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        """Evaluate latest closed candle against active position.

        Returns (closed_trade_dict, stats_dict) if position closed, else (None, None).
        """
        key = self._make_key(symbol, strategy_mode)
        pos = self.active_positions.get(key)
        if pos is None:
            return None, None

        pos["bars_held"] += 1
        sym = pos["symbol"]
        side = pos["side"]
        entry = pos["entry"]
        init_r = pos["initial_risk"]
        strat = pos["strategy_mode"]
        cost_price = pos["cost_price"]
        units = pos["units"]

        c_open = float(candle.get("open", entry))
        c_high = float(candle.get("high", entry))
        c_low = float(candle.get("low", entry))
        c_close = float(candle.get("close", entry))
        c_time = str(candle.get("time", candle.get("name", "N/A")))

        # 1. Momentum Breakeven & Trailing Stop Dynamics
        if strat in ("MOM_ONLY", "KALMAN_MOMENTUM"):
            if not pos["tp1_hit"]:
                # Early Breakeven at +0.9 R favorable excursion
                if not pos["breakeven_triggered"]:
                    if side == "LONG" and c_high >= entry + 0.9 * init_r:
                        pos["breakeven_triggered"] = True
                        pos["stop_loss"] = max(pos["stop_loss"], entry)
                    elif side == "SHORT" and c_low <= entry - 0.9 * init_r:
                        pos["breakeven_triggered"] = True
                        pos["stop_loss"] = min(pos["stop_loss"], entry)
            else:
                # Post-TP1 Wide Kalman Trailing Stop: kalman_level ∓ 1.2 * ATR
                if atr > 0:
                    if side == "LONG":
                        trail = kalman_level - 1.2 * atr
                        if trail > pos["stop_loss"]:
                            pos["stop_loss"] = trail
                    else:
                        trail = kalman_level + 1.2 * atr
                        if trail < pos["stop_loss"]:
                            pos["stop_loss"] = trail

        stop = pos["stop_loss"]
        tp1 = pos["tp1"]
        tp2 = pos["tp2"]
        exit_price = None
        reason = None

        # 2. Check Exits (matching backtest conservative resolution)
        if should_weekend_close:
            exit_price = c_open
            reason = "weekend_close"

        elif side == "LONG":
            # Conservative rule: check stop BEFORE target
            if c_open <= stop:
                exit_price = c_open
                reason = "gap_stop"
            elif c_low <= stop:
                exit_price = stop
                reason = "stop" if not pos["breakeven_triggered"] and not pos["tp1_hit"] else "breakeven_stop"
            else:
                if not pos["tp1_hit"]:
                    if c_open >= tp1:
                        pos["tp1_hit"] = True
                        pos["tp1_exit_price"] = c_open
                        pos["stop_loss"] = entry
                        if c_open >= tp2:
                            exit_price = c_open
                            reason = "gap_tp2"
                        elif c_high >= tp2:
                            exit_price = tp2
                            reason = "tp2"
                    elif c_high >= tp1:
                        pos["tp1_hit"] = True
                        pos["tp1_exit_price"] = tp1
                        pos["stop_loss"] = entry
                        if c_high >= tp2:
                            exit_price = tp2
                            reason = "tp2"
                else:
                    if c_open <= pos["stop_loss"]:
                        exit_price = c_open
                        reason = "breakeven_gap_stop"
                    elif c_low <= pos["stop_loss"]:
                        exit_price = pos["stop_loss"]
                        reason = "breakeven_stop" if not pos.get("tp1_hit") else "trailing_stop"
                    elif c_open >= tp2:
                        exit_price = c_open
                        reason = "gap_tp2"
                    elif c_high >= tp2:
                        exit_price = tp2
                        reason = "tp2"

        else:  # SHORT
            if c_open >= stop:
                exit_price = c_open
                reason = "gap_stop"
            elif c_high >= stop:
                exit_price = stop
                reason = "stop" if not pos["breakeven_triggered"] and not pos["tp1_hit"] else "breakeven_stop"
            else:
                if not pos["tp1_hit"]:
                    if c_open <= tp1:
                        pos["tp1_hit"] = True
                        pos["tp1_exit_price"] = c_open
                        pos["stop_loss"] = entry
                        if c_open <= tp2:
                            exit_price = c_open
                            reason = "gap_tp2"
                        elif c_low <= tp2:
                            exit_price = tp2
                            reason = "tp2"
                    elif c_low <= tp1:
                        pos["tp1_hit"] = True
                        pos["tp1_exit_price"] = tp1
                        pos["stop_loss"] = entry
                        if c_low <= tp2:
                            exit_price = tp2
                            reason = "tp2"
                else:
                    if c_open >= pos["stop_loss"]:
                        exit_price = c_open
                        reason = "breakeven_gap_stop"
                    elif c_high >= pos["stop_loss"]:
                        exit_price = pos["stop_loss"]
                        reason = "breakeven_stop" if not pos.get("tp1_hit") else "trailing_stop"
                    elif c_open <= tp2:
                        exit_price = c_open
                        reason = "gap_tp2"
                    elif c_low <= tp2:
                        exit_price = tp2
                        reason = "tp2"

        # Time-Stop fallback
        if reason is None and pos["bars_held"] >= pos["max_hold_bars"]:
            exit_price = c_close
            reason = "time_stop"

        # 3. Position Closed: Calculate Financials & Update Stats
        if reason is not None and exit_price is not None:
            direction = 1 if side == "LONG" else -1
            if pos["tp1_hit"]:
                pnl1 = (pos["tp1_exit_price"] - entry) * direction
                pnl2 = (exit_price - entry) * direction
                gross_unit = 0.5 * pnl1 + 0.5 * pnl2
            else:
                gross_unit = (exit_price - entry) * direction

            net_unit = gross_unit - cost_price
            gross_r = gross_unit / init_r if init_r > 0 else 0.0
            net_r = net_unit / init_r if init_r > 0 else 0.0
            net_pnl = net_unit * units

            closed_trade = {
                **pos,
                "exit": float(exit_price),
                "exit_reason": reason,
                "exit_time": c_time,
                "gross_pnl_unit": float(gross_unit),
                "net_pnl_unit": float(net_unit),
                "gross_r": float(gross_r),
                "net_r": float(net_r),
                "net_pnl": float(net_pnl),
            }

            # Update stats counter
            stats = self.perf_tracker.record_closed_trade(closed_trade)

            # Remove from active
            del self.active_positions[key]
            self.save()

            logger.info(
                "Closed position %s %s: reason=%s, net_pnl=%.2f, net_r=%.2f (TP Count: %d, SL Count: %d)",
                sym, strat, reason, net_pnl, net_r, stats["tp_count"], stats["sl_count"],
            )
            return closed_trade, stats

        self.save()
        return None, None
