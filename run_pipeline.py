"""Run a local research backtest, walk-forward matrix, or live whitelist scanner & Telegram monitor.

Usage:
  - Standard Backtest: python run_pipeline.py --source mt5 --symbol EURUSD --timeframe M15
  - Walk-Forward Split: python run_pipeline.py --source binance --symbol BTCUSDT --walk-forward
  - Live Whitelist Scan: python run_pipeline.py --scan-whitelist
  - Live Whitelist Watch: python run_pipeline.py --scan-whitelist --watch --interval-sec 30
  - Telegram Pairing Wizard: python run_pipeline.py --pair-telegram
  - Telegram Alert Test: python run_pipeline.py --test-telegram
"""
from __future__ import annotations
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

from stigma.backtest_engine import backtest, prepare_features, check_session_rules
from stigma.data.binance_data import fetch_binance
from stigma.data.mt5_data import fetch_mt5
from stigma.execution_bridge import build_order_intent
from stigma.execution_notifier import (
    SignalDeduplicator,
    LivePerformanceTracker,
    LivePositionTracker,
    send_telegram_alert,
    send_telegram_close_alert,
    send_telegram_message,
    discover_telegram_chat_id,
    load_env_file,
)
from stigma.macro_gate import load_macro_bias
from stigma.quant_engine import calculate_hurst_vr, estimate_ou_half_life, evaluate_stigma_trigger

# Ensure .env is loaded
load_env_file()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_pipeline")

# Persistent file logging for background/autostart watchers
try:
    _log_dir = Path(__file__).resolve().parent / "reports"
    _log_dir.mkdir(parents=True, exist_ok=True)
    _fh = logging.FileHandler(_log_dir / "watcher.log", encoding="utf-8")
    _fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logging.getLogger().addHandler(_fh)
except Exception:
    pass

def run_walk_forward(
    df: pd.DataFrame,
    *,
    spread: float = 0.0,
    commission: float = 0.0,
    slippage: float = 0.0,
    macro_bias: int = 0,
    timeframe_minutes: int = 15,
    train_ratio: float = 0.60,
    val_ratio: float = 0.20,
) -> dict:
    """Chronological, untouched 3-way split: Train (60%), Validation (20%), Out-of-Sample (20%)."""
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    df_train = df.iloc[:train_end]
    df_val = df.iloc[train_end:val_end]
    df_oos = df.iloc[val_end:]

    res_train = backtest(df_train, spread=spread, commission=commission, slippage=slippage, macro_bias=macro_bias, timeframe_minutes=timeframe_minutes)
    res_val = backtest(df_val, spread=spread, commission=commission, slippage=slippage, macro_bias=macro_bias, timeframe_minutes=timeframe_minutes)
    res_oos = backtest(df_oos, spread=spread, commission=commission, slippage=slippage, macro_bias=macro_bias, timeframe_minutes=timeframe_minutes)

    # Pop trades frames to avoid huge JSON output
    train_trades = res_train.pop("trades_frame")
    val_trades = res_val.pop("trades_frame")
    oos_trades = res_oos.pop("trades_frame")

    return {
        "train": res_train,
        "val": res_val,
        "oos": res_oos,
        "train_bars": len(df_train),
        "val_bars": len(df_val),
        "oos_bars": len(df_oos),
        "all_trades": pd.concat([train_trades, val_trades, oos_trades], ignore_index=True),
    }

def scan_whitelisted_strategies(
    whitelist_file: str | Path = "reports/whitelist.json",
    deduplicator: SignalDeduplicator | None = None,
    position_tracker: LivePositionTracker | None = None,
    send_alerts: bool = True,
) -> tuple[list[dict], dict[str, str]]:
    """Scan all approved strategies from whitelist.json against live closed-bar market data."""
    wl_path = Path(whitelist_file)
    if not wl_path.exists():
        logger.error("Whitelist file not found: %s", wl_path)
        return [], {}

    with open(wl_path, "r", encoding="utf-8") as f:
        wl_data = json.load(f)

    strategies = wl_data.get("whitelisted_strategies", [])
    if not strategies:
        logger.warning("No whitelisted strategies found in %s", wl_path)
        return [], {}

    # Group strategies by (source, symbol, timeframe) to fetch market data once per asset/tf
    groups: dict[tuple[str, str, str], list[dict]] = {}
    for st in strategies:
        key = (st["source"].upper(), st["symbol"].upper(), st["timeframe"].upper())
        groups.setdefault(key, []).append(st)

    data_cache: dict[tuple[str, str, str], pd.DataFrame] = {}
    last_bar_times: dict[str, str] = {}
    active_signals: list[dict] = []
    summary_rows: list[dict] = []

    # 1. Fetch data
    for (src, sym, tf) in groups.keys():
        bars_needed = 350
        try:
            if src == "BINANCE":
                df = fetch_binance(sym, tf, bars_needed, pause=0.02)
            else:
                df = fetch_mt5(sym, tf, bars_needed)
            if df is not None and len(df) >= 150:
                data_cache[(src, sym, tf)] = df
                last_bar_times[f"{src}:{sym}:{tf}"] = str(df.index[-1])
            else:
                logger.warning("Insufficient data for %s %s %s", src, sym, tf)
        except Exception as exc:
            logger.error("Error fetching data for %s %s %s: %s", src, sym, tf, exc)

    # 2. Evaluate each whitelisted strategy
    for (src, sym, tf), strat_list in groups.items():
        df = data_cache.get((src, sym, tf))
        if df is None:
            for st in strat_list:
                summary_rows.append({
                    "symbol": sym,
                    "tf": tf,
                    "strat": st["strategy_mode"],
                    "mode": st["mode"],
                    "regime": "DATA_ERROR",
                    "z_score": "N/A",
                    "signal": "NO_DATA",
                    "bar_time": "N/A",
                })
            continue

        # Prepare features causally
        feat = prepare_features(df)
        n = len(feat)
        closes = feat["close"].values
        highs = feat["high"].values
        lows = feat["low"].values
        opens = feat["open"].values
        atrs = feat["atr14"].values
        levels = feat["kalman_level"].values
        velocities = feat["kalman_velocity"].values
        z_scores = feat["z_k"].values
        innovations = feat["innovation_std"].values
        sigmas_yz = feat["sigma_yz"].values
        sigmas_yz_med = feat["sigma_yz_median50"].values
        ema_200s = feat["ema_200"].values

        # Latest closed bar index
        i = n - 1
        last_time = df.index[-1]
        session_type = "binance" if src == "BINANCE" else "mt5"
        can_enter, should_force_close_weekend = check_session_rules(last_time, session_type)

        latest_candle = {
            "open": opens[i],
            "high": highs[i],
            "low": lows[i],
            "close": closes[i],
            "time": str(last_time),
        }

        # Hurst & VR on last 100 closed bars
        hist_close = closes[max(0, i - 99):i + 1]
        regime = calculate_hurst_vr(np.log(hist_close), window=min(100, len(hist_close)))

        # Evaluate each strategy variant
        for st in strat_list:
            strat_mode = st["strategy_mode"]
            macro_mode = st["mode"]
            cost_spec = st.get("cost_parameters", {"spread": 0.0, "commission": 0.0, "slippage": 0.0})
            total_cost = cost_spec.get("spread", 0.0) + cost_spec.get("commission", 0.0) + cost_spec.get("slippage", 0.0)

            # Check if position tracker has active position for this strategy
            if position_tracker is not None and position_tracker.has_position(sym, strat_mode):
                pos = position_tracker.get_position(sym, strat_mode)
                if pos and pos.get("entry_time") != str(last_time):
                    closed_trade, stats = position_tracker.update_candle(
                        sym,
                        strat_mode,
                        latest_candle,
                        kalman_level=levels[i],
                        atr=atrs[i],
                        should_weekend_close=should_force_close_weekend,
                    )
                    if closed_trade is not None and stats is not None:
                        logger.info(
                            "[TRADE CLOSED] %s %s: Reason=%s, Net PnL=%.2f, Net R=%.2f (TP:%d, SL:%d)",
                            sym, strat_mode, closed_trade["exit_reason"], closed_trade["net_pnl"],
                            closed_trade["net_r"], stats["tp_count"], stats["sl_count"],
                        )
                        summary_rows.append({
                            "symbol": sym,
                            "tf": tf,
                            "strat": strat_mode,
                            "mode": macro_mode,
                            "regime": regime.label,
                            "z_score": f"{z_scores[i]:.2f}",
                            "signal": f"CLOSED_{closed_trade['exit_reason'].upper()} ({closed_trade['net_r']:+.2f}R)",
                            "bar_time": str(last_time),
                        })
                        if send_alerts:
                            send_telegram_close_alert(closed_trade, stats)
                        continue
                    else:
                        summary_rows.append({
                            "symbol": sym,
                            "tf": tf,
                            "strat": strat_mode,
                            "mode": macro_mode,
                            "regime": regime.label,
                            "z_score": f"{z_scores[i]:.2f}",
                            "signal": f"IN_POSITION ({pos['side']} bar {pos['bars_held']})",
                            "bar_time": str(last_time),
                        })
                        continue

            # Regime gate
            regime_match = (
                (strat_mode == "OU_ONLY" and regime.label == "OU_MEAN_REVERSION") or
                (strat_mode == "MOM_ONLY" and regime.label == "KALMAN_MOMENTUM") or
                (strat_mode == "BOTH" and regime.label in ("OU_MEAN_REVERSION", "KALMAN_MOMENTUM"))
            )

            half_life_val = float("nan")
            if regime.label == "OU_MEAN_REVERSION":
                start_ou = max(0, i - 119)
                spread_series = closes[start_ou:i + 1] - levels[start_ou:i + 1]
                ou = estimate_ou_half_life(spread_series)
                if ou["valid"]:
                    half_life_val = ou["half_life"]

            # Macro bias calculation
            eff_macro = (1 if closes[i] > ema_200s[i] else -1) if macro_mode == "Begonya_Macro" else 0

            signal_label = "NO_TRIGGER"
            trigger_sig = None

            if not can_enter:
                signal_label = "SESSION_CLOSED"
            elif not regime_match:
                signal_label = f"REGIME_MISMATCH ({regime.label})"
            else:
                sig, reason = evaluate_stigma_trigger(
                    regime=regime.label,
                    hurst=regime.hurst,
                    variance_ratio=regime.variance_ratio,
                    half_life=half_life_val,
                    z_score=z_scores[:i + 1],
                    level=levels[:i + 1],
                    velocity=velocities[:i + 1],
                    innovation_std=innovations[:i + 1],
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
                    strategy_mode=strat_mode,
                    return_reason=True,
                )
                if sig is not None:
                    signal_label = f"TRIGGER_{sig['side']}"
                    trigger_sig = sig
                else:
                    signal_label = reason

            summary_rows.append({
                "symbol": sym,
                "tf": tf,
                "strat": strat_mode,
                "mode": macro_mode,
                "regime": regime.label,
                "z_score": f"{z_scores[i]:.2f}",
                "signal": signal_label,
                "bar_time": str(last_time),
            })

            if trigger_sig is not None:
                payload = {
                    "symbol": sym,
                    "timeframe": tf,
                    "strategy_mode": strat_mode,
                    "macro_mode": macro_mode,
                    "side": trigger_sig["side"],
                    "entry": trigger_sig["entry"],
                    "stop_loss": trigger_sig["stop_loss"],
                    "tp1": trigger_sig["tp1"],
                    "tp2": trigger_sig["tp2"],
                    "risk": trigger_sig.get("risk", abs(trigger_sig["entry"] - trigger_sig["stop_loss"])),
                    "max_hold_bars": trigger_sig["max_hold_bars"],
                    "hurst": regime.hurst,
                    "variance_ratio": regime.variance_ratio,
                    "half_life": half_life_val,
                    "z_score": z_scores[i],
                    "bar_time": str(last_time),
                }
                active_signals.append(payload)

                # Deduplication and notification
                is_new = True
                if deduplicator is not None:
                    is_new = deduplicator.should_emit(sym, strat_mode, str(last_time))

                if is_new:
                    intent = build_order_intent(payload)
                    logger.info("VALIDATED ORDER INTENT: %s", intent)
                    if position_tracker is not None:
                        position_tracker.open_position(payload, units=1.0, cost_price=total_cost)
                    if send_alerts:
                        send_telegram_alert(payload)

    # Print scan report table
    print("\n" + "=" * 96)
    print(f"  STIGMA QUANT ENGINE -- LIVE WHITELIST SCAN REPORT ({len(summary_rows)} Strategies)")
    print("=" * 96)
    print(f"{'SYMBOL':<10} {'TF':<5} {'STRATEGY':<12} {'MACRO':<15} {'REGIME':<20} {'Z_K':<7} {'STATUS / TRIGGER':<20}")
    print("-" * 96)
    for r in summary_rows:
        print(f"{r['symbol']:<10} {r['tf']:<5} {r['strat']:<12} {r['mode']:<15} {r['regime']:<20} {r['z_score']:<7} {r['signal']:<20}")
    print("=" * 96)
    if active_signals:
        print(f"[*] ACTIVE SIGNALS DETECTED: {len(active_signals)}")
        for s in active_signals:
            print(f"  -> {s['symbol']} {s['timeframe']} {s['side']} @ {s['entry']:.5f} (SL: {s['stop_loss']:.5f}, TP1: {s['tp1']:.5f}, TP2: {s['tp2']:.5f})")
    else:
        print("[OK] All whitelisted strategies actively monitored. Zero false positives on current closed bar.")
    print("=" * 96 + "\n")

    return active_signals, last_bar_times

def watch_whitelisted_strategies(
    whitelist_file: str | Path = "reports/whitelist.json",
    interval_sec: int = 30,
    send_alerts: bool = True,
) -> None:
    """Continuously monitor whitelisted strategies. Re-scans silently on each new closed bar."""
    logger.info("Starting Stigma Whitelist Watch loop (interval: %ds, whitelist: %s)...", interval_sec, whitelist_file)
    if send_alerts:
        startup_msg = (
            "🚀 <b>STIGMA QUANT ENGINE — NÖBETÇİ AKTİF</b> 🚀\n"
            "━━━━━━━━━━━━━━━━━━━\n"
            "📡 <b>Durum:</b> Arka plan canlı piyasa nöbetçisi devrede.\n"
            f"⏱️ <b>Tarama Periyodu:</b> Her {interval_sec} saniyede bir.\n"
            "🛡️ <b>İzlenen Stratejiler:</b> 6 Onaylı Whitelist Stratejisi.\n"
            "━━━━━━━━━━━━━━━━━━━\n"
            "🔔 <i>Açılan/kapanan işlemler bu kanala anlık olarak raporlanacaktır.</i>"
        )
        send_telegram_message(startup_msg)
    dedup = SignalDeduplicator()
    pos_tracker = LivePositionTracker()
    previous_bars: dict[str, str] = {}

    try:
        while True:
            # Run scan with position tracker
            _, current_bars = scan_whitelisted_strategies(
                whitelist_file,
                deduplicator=dedup,
                position_tracker=pos_tracker,
                send_alerts=send_alerts,
            )
            previous_bars = current_bars

            # Wait for next poll
            logger.info("Watch loop standing by (%ds). Waiting for next closed candle...", interval_sec)
            time.sleep(interval_sec)

    except KeyboardInterrupt:
        logger.info("Whitelist Watch loop stopped by user.")

def pair_telegram_wizard(timeout_sec: int = 45) -> bool:
    """Interactive wizard to pair Telegram bot and save TELEGRAM_CHAT_ID to .env."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        print("\n[ERROR] TELEGRAM_BOT_TOKEN is not configured in .env or environment!")
        return False

    print("\n" + "=" * 80)
    print("  STIGMA QUANT ENGINE -- TELEGRAM BOT ESLESTIRME SIHIRBAZI")
    print("=" * 80)
    print("  1. Telegram uygulamanizda botunuzu acin: https://t.me/SStigma_bot")
    print("  2. Bota /start yazin veya herhangi bir mesaj gonderin.")
    print(f"  3. Botunuz gelen mesaji dinliyor (Zaman asimi: {timeout_sec} saniye)...")
    print("=" * 80 + "\n")

    start_time = time.time()
    discovered_chat_id = None
    discovered_user = None

    while time.time() - start_time < timeout_sec:
        chat_id, user_name = discover_telegram_chat_id(token, timeout_sec=2.0, save_to_env=True)
        if chat_id:
            discovered_chat_id = chat_id
            discovered_user = user_name
            break
        time.sleep(2.0)
        sys.stdout.write(".")
        sys.stdout.flush()

    print("\n")
    if not discovered_chat_id:
        print("[!] Zaman asimi! Botunuza henuz mesaj gelmedi.")
        print("    Lutfen https://t.me/SStigma_bot adresine girip /start gonderdikten sonra komutu tekrar calistirin.")
        return False

    print(f"[OK] Tebrikler {discovered_user}! Chat ID basariyla tespit edildi: {discovered_chat_id}")
    print("[OK] .env dosyasina TELEGRAM_CHAT_ID kaydedildi.")

    welcome_text = (
        f"✅ <b>STIGMA QUANT ENGINE — TELEGRAM BAĞLANTISI AKTİF</b> ✅\n"
        f"━━━━━━━━━━━━━━━━━━━\n"
        f"👤 <b>Kullanıcı:</b> {discovered_user}\n"
        f"🆔 <b>Chat ID:</b> <code>{discovered_chat_id}</code>\n"
        f"📡 <b>Aktif Whitelist:</b> 6 Kurumsal Strateji\n"
        f"━━━━━━━━━━━━━━━━━━━\n"
        f"🔔 <i>Açılan ve kapanan tüm kurumsal işlemler buraya anlık olarak iletilecektir.</i>\n"
        f"<i>Kapanış mesajlarında Net PnL, Net R, Komisyon Sonrası Net R ve SL/TP sayacı yer alacaktır.</i>"
    )
    sent = send_telegram_message(welcome_text, bot_token=token, chat_id=discovered_chat_id)
    if sent:
        print("[OK] Telegram'a onay mesaji gonderildi!")
    else:
        print("[!] Onay mesaji gonderilemedi.")
    print("=" * 80 + "\n")
    return True

def test_telegram_alert() -> bool:
    """Send test open and close alert messages to verify formatting and Telegram delivery."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("\n[ERROR] TELEGRAM_BOT_TOKEN veya TELEGRAM_CHAT_ID bulunamadi!")
        print("Lutfen once 'python run_pipeline.py --pair-telegram' calistirin.\n")
        return False

    print("\n[*] Telegram'a ornek ISLEM ACILIS mesaji gonderiliyor...")
    test_signal = {
        "symbol": "USDJPY",
        "timeframe": "M15",
        "strategy_mode": "MOM_ONLY",
        "macro_mode": "Begonya_Macro",
        "side": "LONG",
        "entry": 149.250,
        "stop_loss": 148.800,
        "tp1": 149.850,
        "tp2": 150.400,
        "risk": 0.450,
        "max_hold_bars": 24,
        "hurst": 0.582,
        "variance_ratio": 1.145,
        "half_life": 0.0,
        "z_score": 0.12,
        "bar_time": "2026-10-20 08:30:00",
    }
    open_ok = send_telegram_alert(test_signal, bot_token=token, chat_id=chat_id)
    print(f"[{'OK' if open_ok else 'FAIL'}] Acilis mesaji: {open_ok}")

    print("[*] Telegram'a ornek ISLEM KAPANIS (Net PnL, Net R, SL/TP Sayaci) mesaji gonderiliyor...")
    test_trade = {
        "symbol": "USDJPY",
        "timeframe": "M15",
        "strategy_mode": "MOM_ONLY",
        "side": "LONG",
        "entry": 149.250,
        "exit": 150.400,
        "tp1_exit_price": 149.850,
        "tp1_hit": True,
        "initial_risk": 0.450,
        "bars_held": 8,
        "exit_reason": "tp2",
        "gross_r": 1.98,
        "net_r": 1.91,
        "net_pnl": 88.20,
    }
    perf = LivePerformanceTracker()
    test_stats = perf.get_stats()
    if test_stats["total_trades"] == 0:
        test_stats.update({
            "total_trades": 1,
            "tp_count": 1,
            "sl_count": 0,
            "breakeven_count": 0,
            "win_rate": 100.0,
            "cumulative_net_r": 1.91,
            "cumulative_net_pnl": 88.20,
        })
    close_ok = send_telegram_close_alert(test_trade, test_stats, bot_token=token, chat_id=chat_id)
    print(f"[{'OK' if close_ok else 'FAIL'}] Kapanis mesaji: {close_ok}\n")
    return open_ok and close_ok

def install_autostart(interval_sec: int = 30) -> bool:
    """Create a Windows Startup VBScript that launches Stigma watcher silently on login."""
    startup_dir = Path(os.environ.get("APPDATA", "")) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup"
    if not startup_dir.exists():
        print(f"[ERROR] Windows Startup directory not found: {startup_dir}")
        return False

    vbs_path = startup_dir / "StigmaWatcher.vbs"
    project_dir = Path(__file__).resolve().parent
    python_exe = sys.executable
    pythonw_exe = Path(python_exe).parent / "pythonw.exe"
    runner_exe = str(pythonw_exe) if pythonw_exe.exists() else str(python_exe)
    script_path = project_dir / "run_pipeline.py"

    vbs_content = (
        'Set WshShell = CreateObject("WScript.Shell")\n'
        f'WshShell.CurrentDirectory = "{project_dir}"\n'
        f'WshShell.Run """{runner_exe}"" ""{script_path}"" --scan-whitelist --watch --interval-sec {interval_sec}", 0, False\n'
    )
    try:
        with open(vbs_path, "w", encoding="utf-8") as f:
            f.write(vbs_content)
        print("\n" + "=" * 80)
        print("  STIGMA QUANT ENGINE -- OTOMATIK BASLANGIC KURULDU")
        print("=" * 80)
        print(f"  [OK] Windows Baslangic Dosyasi: {vbs_path}")
        print(f"  [OK] Calistirilacak: pythonw.exe (Sessiz arka plan, CMD penceresi acilmaz)")
        print(f"  [OK] Interval: Her {interval_sec} saniyede bir canli piyasa taramasi")
        print("  [OK] Bilgisayariniz her acildiginda Stigma Watcher otomatik baslayacak!")
        print("=" * 80 + "\n")
        return True
    except Exception as exc:
        print(f"[ERROR] Otomatik baslangic kurulamadi: {exc}")
        return False

def uninstall_autostart() -> bool:
    """Remove Stigma watcher from Windows Startup."""
    startup_dir = Path(os.environ.get("APPDATA", "")) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup"
    vbs_path = startup_dir / "StigmaWatcher.vbs"
    if vbs_path.exists():
        try:
            vbs_path.unlink()
            print("\n[OK] Stigma Watcher Windows baslangicindan basariyla kaldirildi.\n")
            return True
        except Exception as exc:
            print(f"[ERROR] Kaldirma hatasi: {exc}")
            return False
    else:
        print("\n[*] Otomatik baslangic zaten yuklu degil.\n")
        return True

def main():
    p = argparse.ArgumentParser(description="Stigma Quant Engine Pipeline & Live Whitelist Runner")
    p.add_argument("--scan-whitelist", action="store_true", help="Scan all approved strategies from whitelist.json")
    p.add_argument("--watch", action="store_true", help="Run continuous monitoring loop for whitelisted strategies")
    p.add_argument("--interval-sec", type=int, default=30, help="Polling interval in seconds for --watch")
    p.add_argument("--whitelist-file", default="reports/whitelist.json", help="Path to whitelist.json")
    p.add_argument("--pair-telegram", action="store_true", help="Run Telegram bot pairing wizard to detect chat_id")
    p.add_argument("--test-telegram", action="store_true", help="Send sample open and close alerts to Telegram")
    p.add_argument("--install-autostart", action="store_true", help="Configure Windows to start watcher automatically on PC boot")
    p.add_argument("--uninstall-autostart", action="store_true", help="Remove Stigma watcher from Windows startup")
    p.add_argument("--source", choices=["csv", "mt5", "binance"], default="csv")
    p.add_argument("--file", help="CSV file with time,open,high,low,close columns")
    p.add_argument("--symbol", default="EURUSD")
    p.add_argument("--timeframe", choices=["M5", "M15", "H1"], default="M15")
    p.add_argument("--bars", type=int, default=50000)
    p.add_argument("--macro-bias", type=int, choices=[-1, 0, 1], default=0)
    p.add_argument("--macro-file", help="Path to Begonya macro JSON file")
    p.add_argument("--spread", type=float, default=0.0)
    p.add_argument("--commission", type=float, default=0.0)
    p.add_argument("--slippage", type=float, default=0.0)
    p.add_argument("--walk-forward", action="store_true", help="Run chronological train/val/oos split")
    p.add_argument("--output", default="reports/trades.csv")
    args = p.parse_args()

    # Mode 0: Telegram Utilities
    if args.pair_telegram:
        pair_telegram_wizard()
        return

    if args.test_telegram:
        test_telegram_alert()
        return

    if args.install_autostart:
        install_autostart(interval_sec=args.interval_sec)
        return

    if args.uninstall_autostart:
        uninstall_autostart()
        return

    # Mode 1: Whitelist Scan or Watch
    if args.scan_whitelist:
        if args.watch:
            watch_whitelisted_strategies(
                whitelist_file=args.whitelist_file,
                interval_sec=args.interval_sec,
                send_alerts=True,
            )
        else:
            scan_whitelisted_strategies(
                whitelist_file=args.whitelist_file,
                send_alerts=True,
            )
        return

    # Mode 2: Standard Backtest or Walk-Forward Split
    if args.macro_file:
        macro_bias = load_macro_bias(args.macro_file, autonomous=False)
        logger.info("Loaded macro bias %d from %s", macro_bias, args.macro_file)
    else:
        macro_bias = args.macro_bias

    if args.source == "csv":
        if not args.file:
            p.error("--file is required when source is 'csv'")
        df = pd.read_csv(args.file)
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"], utc=True)
            df = df.set_index("time")
    elif args.source == "mt5":
        df = fetch_mt5(args.symbol, args.timeframe, args.bars)
    else:
        df = fetch_binance(args.symbol, args.timeframe, args.bars)

    tf_min = 5 if args.timeframe == "M5" else (60 if args.timeframe == "H1" else 15)

    if args.walk_forward:
        wf_res = run_walk_forward(
            df,
            spread=args.spread,
            commission=args.commission,
            slippage=args.slippage,
            macro_bias=macro_bias,
            timeframe_minutes=tf_min,
        )
        trades = wf_res.pop("all_trades")
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        trades.to_csv(out_path, index=False)
        print(json.dumps({
            "status": "COMPLETED",
            "symbol": args.symbol,
            "timeframe": args.timeframe,
            "macro_bias": macro_bias,
            "walk_forward": wf_res,
            "trades_file": str(out_path),
        }, indent=2))
    else:
        result = backtest(
            df,
            spread=args.spread,
            commission=args.commission,
            slippage=args.slippage,
            macro_bias=macro_bias,
            timeframe_minutes=tf_min,
        )
        trades = result.pop("trades_frame")
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        trades.to_csv(out_path, index=False)
        print(json.dumps({
            "status": "COMPLETED",
            "symbol": args.symbol,
            "timeframe": args.timeframe,
            "macro_bias": macro_bias,
            "metrics": result,
            "trades_file": str(out_path),
        }, indent=2))

if __name__ == "__main__":
    main()
