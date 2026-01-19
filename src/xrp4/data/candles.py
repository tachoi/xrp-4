"""Candle data loader for xrp-4.

Standalone implementation - no dependency on xrp3.
"""

from datetime import datetime
from typing import Optional

import pandas as pd

from .loader import load_ohlcv_sync, resample_to_15m


def load_candles(
    symbol: str,
    timeframe: str,
    start: str | datetime,
    end: str | datetime,
    validate: bool = True,
    fix_issues: bool = True,
) -> pd.DataFrame:
    """Load OHLCV candles from TimescaleDB.

    Args:
        symbol: Trading symbol (e.g., "XRPUSDT")
        timeframe: Timeframe string (must be "3m")
        start: Start date/datetime
        end: End date/datetime
        validate: If True, run validation checks
        fix_issues: If True, automatically fix common issues

    Returns:
        DataFrame with columns:
            - timestamp: UTC datetime (naive)
            - open: Opening price
            - high: High price
            - low: Low price
            - close: Closing price
            - volume: Trading volume
            - symbol: Symbol string
            - timeframe: Timeframe string

    Raises:
        ValueError: If timeframe is not "3m"
    """
    # Convert string dates to datetime if needed
    if isinstance(start, str):
        start = datetime.fromisoformat(start)
    if isinstance(end, str):
        end = datetime.fromisoformat(end)

    # Load data using local loader
    df = load_ohlcv_sync(
        symbol=symbol,
        start=start,
        end=end,
        validate=validate,
        fix_issues=fix_issues,
    )

    return df


def compute_atr(
    df: pd.DataFrame,
    period: int = 14,
    column_prefix: str = "",
) -> pd.Series:
    """Compute Average True Range (ATR) indicator.

    Uses Wilder's smoothing method.

    Args:
        df: DataFrame with OHLCV data (must have high, low, close columns)
        period: ATR period (default 14)
        column_prefix: Optional prefix for column names (e.g., "3m_" -> "3m_high")

    Returns:
        Series with ATR values (same index as input df)
    """
    high_col = f"{column_prefix}high" if column_prefix else "high"
    low_col = f"{column_prefix}low" if column_prefix else "low"
    close_col = f"{column_prefix}close" if column_prefix else "close"

    if high_col not in df.columns or low_col not in df.columns or close_col not in df.columns:
        raise ValueError(f"Missing required OHLCV columns with prefix '{column_prefix}'")

    high = df[high_col]
    low = df[low_col]
    close = df[close_col]

    # True Range components
    h_l = high - low
    h_pc = (high - close.shift(1)).abs()
    l_pc = (low - close.shift(1)).abs()

    # True Range
    tr = pd.concat([h_l, h_pc, l_pc], axis=1).max(axis=1)

    # ATR using Wilder's smoothing (exponential moving average with alpha = 1/period)
    atr = tr.ewm(alpha=1.0 / period, adjust=False).mean()

    return atr


def resample_candles(
    df: pd.DataFrame,
    target_timeframe: str,
    base_timeframe: str = "3m",
) -> pd.DataFrame:
    """Resample 3m candles to higher timeframe.

    Args:
        df: DataFrame with 3m OHLCV data
        target_timeframe: Target timeframe (e.g., "15m", "1h")
        base_timeframe: Base timeframe (default "3m")

    Returns:
        Resampled DataFrame with target timeframe
    """
    if base_timeframe != "3m":
        raise ValueError("Base timeframe must be 3m")

    # Mapping of timeframe strings to pandas frequency strings
    TF_MAP = {
        "3m": "3min",
        "5m": "5min",
        "15m": "15min",
        "30m": "30min",
        "1h": "1h",
        "4h": "4h",
        "1d": "1D",
    }

    if target_timeframe not in TF_MAP:
        raise ValueError(f"Unsupported target timeframe: {target_timeframe}")

    if target_timeframe == "15m":
        return resample_to_15m(df)

    df = df.copy()
    df = df.set_index("timestamp")

    # Resample OHLCV
    resampled = df.resample(TF_MAP[target_timeframe]).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    })

    # Drop rows with NaN (incomplete periods)
    resampled = resampled.dropna()

    # Reset index and add metadata
    resampled = resampled.reset_index()
    resampled["symbol"] = df["symbol"].iloc[0] if "symbol" in df.columns else "XRPUSDT"
    resampled["timeframe"] = target_timeframe

    return resampled
