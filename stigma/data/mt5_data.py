"""MT5 terminal adapter. Requires MetaTrader5 package and a logged-in terminal."""
from __future__ import annotations
import logging
import pandas as pd

logger = logging.getLogger(__name__)

SUPPORTED_SYMBOLS = {"EURUSD", "CADCHF", "USDJPY", "XAUUSD", "NAS100"}
SUPPORTED_TIMEFRAMES = {"M5", "M15", "H1"}

def resolve_broker_symbol(mt5, symbol: str) -> str:
    """Find exact broker symbol name, resolving broker-specific suffixes/prefixes."""
    # 1. Direct check
    info = mt5.symbol_info(symbol)
    if info is not None:
        return symbol

    # 2. Common broker aliases / suffixes
    candidates = [
        symbol,
        f"{symbol}m",
        f"{symbol}.r",
        f"{symbol}_i",
        f"{symbol}.cash",
        f"{symbol}.pro",
        f"{symbol}#",
    ]
    if symbol == "NAS100":
        candidates.extend(["US100", "USTEC", "US100.cash", "NAS100.cash", "USTEC.cash", "NDX"])

    for cand in candidates:
        if mt5.symbol_info(cand) is not None:
            logger.info("Resolved symbol '%s' to broker symbol '%s'", symbol, cand)
            return cand

    # 3. Fuzzy search via symbols_get
    matches = mt5.symbols_get(f"*{symbol}*")
    if matches:
        resolved = matches[0].name
        logger.info("Resolved symbol '%s' to match '%s'", symbol, resolved)
        return resolved

    raise RuntimeError(f"Symbol '{symbol}' not found on connected MT5 broker terminal")

def fetch_mt5(
    symbol: str,
    timeframe: str = "M15",
    bars: int = 50000,
) -> pd.DataFrame:
    """Fetch closed OHLC bars from MetaTrader5.

    Reports requested vs actual bars retrieved.
    Tick volume is preserved strictly as broker metadata and never passed to quant features.
    """
    try:
        import MetaTrader5 as mt5
    except ImportError as exc:
        raise RuntimeError("Install MetaTrader5 on Windows and log in to the terminal first") from exc

    tf_upper = timeframe.upper()
    if tf_upper not in SUPPORTED_TIMEFRAMES:
        raise ValueError(f"Timeframe must be M5, M15, or H1, got '{timeframe}'")

    mapping = {
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "H1": mt5.TIMEFRAME_H1,
    }

    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialization failed: {mt5.last_error()}")

    try:
        broker_symbol = resolve_broker_symbol(mt5, symbol.upper())
        if not mt5.symbol_select(broker_symbol, True):
            raise RuntimeError(f"Failed to select symbol '{broker_symbol}' in Market Watch: {mt5.last_error()}")

        rates = mt5.copy_rates_from_pos(broker_symbol, mapping[tf_upper], 0, bars)
        actual_bars = len(rates) if rates is not None else 0

        logger.info(
            "MT5 fetch for %s (%s): requested=%d bars, retrieved=%d bars",
            broker_symbol, tf_upper, bars, actual_bars
        )

        if rates is None or actual_bars == 0:
            raise RuntimeError(f"No MT5 rates returned for {broker_symbol} ({tf_upper}): {mt5.last_error()}")

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.set_index("time").sort_index()

        # Deduplicate timestamps if any
        if df.index.has_duplicates:
            df = df[~df.index.duplicated(keep="last")]

        # Drop currently forming (unclosed) last bar if timestamp matches live bar
        # In copy_rates_from_pos pos=0 is the current (unclosed) candle, pos=1..bars is closed
        # To strictly use closed bars: if bar count > 1, exclude bar 0 (the last element)
        # However, if pos=1 was used: copy_rates_from_pos(sym, tf, 1, bars) gives closed bars only!
        # Let's inspect: rates from pos=0 has the currently forming candle at the end.
        # Dropping the last bar ensures 100% causal closed-bar integrity:
        if len(df) > 1:
            df = df.iloc[:-1]

        result = df.loc[:, ["open", "high", "low", "close", "tick_volume"]]
        # Validate positive prices
        if (result[["open", "high", "low", "close"]] <= 0).any().any():
            raise ValueError("Corrupt price data: negative or zero price in MT5 rates")

        return result
    finally:
        mt5.shutdown()
