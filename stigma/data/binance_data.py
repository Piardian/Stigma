"""Binance public kline adapter with pagination, rate limiting, and closed-bar verification.

Trade volume is retained strictly as metadata and is NEVER used in signal generation.
"""
from __future__ import annotations
import logging
import time
import requests
import pandas as pd

logger = logging.getLogger(__name__)

BASE = "https://api.binance.com/api/v3/klines"
INTERVALS = {"M5": "5m", "M15": "15m", "H1": "1h"}
SUPPORTED_SYMBOLS = {"BTCUSDT", "SOLUSDT"}

def fetch_binance(
    symbol: str,
    timeframe: str = "M15",
    bars: int = 100000,
    pause: float = 0.05,
    max_retries: int = 5,
) -> pd.DataFrame:
    """Fetch closed klines from Binance public REST API with pagination.

    - Paginates backwards using endTime.
    - Handles rate limits and 429 backoff.
    - Deduplicates timestamps.
    - Filters out currently forming (unclosed) final candle.
    - Reports requested vs actual bars retrieved.
    """
    tf_upper = timeframe.upper()
    if tf_upper not in INTERVALS:
        raise ValueError(f"Timeframe must be M5, M15, or H1, got '{timeframe}'")

    sym_upper = symbol.upper()
    rows: list[list] = []
    end: int | None = None
    retries = 0

    logger.info("Starting Binance fetch for %s (%s), requested %d bars", sym_upper, tf_upper, bars)

    while len(rows) < bars:
        limit = min(1000, bars - len(rows))
        params: dict[str, str | int] = {
            "symbol": sym_upper,
            "interval": INTERVALS[tf_upper],
            "limit": limit,
        }
        if end is not None:
            params["endTime"] = end

        try:
            response = requests.get(BASE, params=params, timeout=20)
            if response.status_code in (418, 429):
                retry_after = int(response.headers.get("Retry-After", 2))
                logger.warning("Binance rate limit reached (HTTP %d). Waiting %d seconds.", response.status_code, retry_after)
                time.sleep(retry_after)
                retries += 1
                if retries > max_retries:
                    raise RuntimeError("Exceeded maximum retries due to Binance rate limits")
                continue

            response.raise_for_status()
            batch = response.json()
            if not batch:
                logger.info("Binance returned empty batch, pagination complete.")
                break

            rows = batch + rows
            end = int(batch[0][0]) - 1

            if len(batch) < limit:
                logger.info("Received partial batch (%d < %d), reached start of history.", len(batch), limit)
                break

            time.sleep(pause)
        except requests.RequestException as exc:
            retries += 1
            if retries > max_retries:
                raise RuntimeError(f"Binance fetch failed after {max_retries} retries: {exc}") from exc
            wait_time = 0.5 * (2 ** retries)
            logger.warning("Binance request error: %s. Retrying in %.1f seconds.", exc, wait_time)
            time.sleep(wait_time)

    if not rows:
        raise RuntimeError(f"No Binance klines returned for symbol '{sym_upper}'")

    df = pd.DataFrame(
        rows,
        columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_base",
            "taker_buy_quote", "ignore",
        ],
    )

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop any row with non-finite prices
    df = df.dropna(subset=["open", "high", "low", "close"])

    # Deduplicate by open_time
    df = df.drop_duplicates(subset=["open_time"])

    # Exclude currently open (unclosed) candle:
    # A candle is fully closed only when close_time <= current time
    now_ms = int(time.time() * 1000)
    df = df[df["close_time"] <= now_ms]

    df["time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.set_index("time").sort_index()

    # Trim to requested bars
    df = df.iloc[-bars:]
    actual_bars = len(df)
    logger.info("Binance fetch complete for %s (%s): requested=%d, retrieved=%d", sym_upper, tf_upper, bars, actual_bars)

    # Validate strictly positive prices
    if (df[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("Corrupt price data: negative or zero price in Binance klines")

    return df[["open", "high", "low", "close", "volume"]]
