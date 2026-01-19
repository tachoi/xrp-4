"""
Standalone TimescaleDB data loader for xrp-4.
No dependency on xrp3.
"""

import asyncio
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import structlog

# Load environment variables from .env file
def load_env():
    """Load environment variables from .env file."""
    env_paths = [
        Path(__file__).parent.parent.parent.parent / ".env",  # xrp-4/.env
        Path.home() / ".xrp4" / ".env",  # ~/.xrp4/.env
    ]

    for env_path in env_paths:
        if env_path.exists():
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
                        if key not in os.environ:
                            os.environ[key] = value
            break

# Load env on module import
load_env()

logger = structlog.get_logger()


def get_db_config() -> dict:
    """Get database configuration from environment variables."""
    return {
        "host": os.getenv("TIMESCALE_HOST", "localhost"),
        "port": int(os.getenv("TIMESCALE_PORT", "5432")),
        "database": os.getenv("TIMESCALE_DB", "xrp_data"),
        "user": os.getenv("TIMESCALE_USER", "xrp_user"),
        "password": os.getenv("TIMESCALE_PASSWORD", ""),
    }


async def load_ohlcv_async(
    symbol: str,
    start: datetime | str,
    end: datetime | str,
    timeframe: str = "3m",
    validate: bool = True,
    fix_issues: bool = True,
) -> pd.DataFrame:
    """
    Load OHLCV data from TimescaleDB asynchronously.

    Args:
        symbol: Trading symbol (e.g., "XRPUSDT")
        start: Start datetime
        end: End datetime
        timeframe: Timeframe (default "3m")
        validate: Run validation checks
        fix_issues: Fix common data issues

    Returns:
        DataFrame with OHLCV data
    """
    import asyncpg

    # Convert string dates to datetime
    if isinstance(start, str):
        start = datetime.fromisoformat(start)
    if isinstance(end, str):
        end = datetime.fromisoformat(end)

    logger.info(
        "Loading OHLCV data",
        symbol=symbol,
        timeframe=timeframe,
        start=str(start),
        end=str(end),
    )

    db_config = get_db_config()

    try:
        pool = await asyncpg.create_pool(
            host=db_config["host"],
            port=db_config["port"],
            database=db_config["database"],
            user=db_config["user"],
            password=db_config["password"],
            min_size=2,
            max_size=10,
        )
        logger.info("Connected to TimescaleDB", min_size=2, max_size=10)
    except Exception as e:
        logger.error("Failed to connect to TimescaleDB", error=str(e))
        raise

    try:
        # Query for OHLCV data
        query = """
            SELECT
                time as timestamp,
                open,
                high,
                low,
                close,
                volume
            FROM ohlcv
            WHERE symbol = $1
              AND timeframe = $4
              AND time >= $2
              AND time < $3
            ORDER BY time ASC
        """

        async with pool.acquire() as conn:
            rows = await conn.fetch(query, symbol, start, end, timeframe)

        if not rows:
            logger.warning("No data found", symbol=symbol, start=str(start), end=str(end))
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])

        # Convert types
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)

        # Add metadata columns
        df["symbol"] = symbol
        df["timeframe"] = timeframe

        logger.info(
            "Fetched OHLCV data",
            symbol=symbol,
            timeframe=timeframe,
            rows=len(df),
            start=str(df["timestamp"].iloc[0]) if len(df) > 0 else None,
            end=str(df["timestamp"].iloc[-1]) if len(df) > 0 else None,
        )

        # Validate data
        if validate and len(df) > 0:
            df = _validate_ohlcv(df, fix_issues)

        logger.info(
            "Data loaded and validated",
            rows=len(df),
            start=str(df["timestamp"].iloc[0]) if len(df) > 0 else None,
            end=str(df["timestamp"].iloc[-1]) if len(df) > 0 else None,
        )

        return df

    finally:
        await pool.close()


def load_ohlcv_sync(
    symbol: str,
    start: datetime | str,
    end: datetime | str,
    timeframe: str = "3m",
    validate: bool = True,
    fix_issues: bool = True,
) -> pd.DataFrame:
    """
    Synchronous wrapper for load_ohlcv_async.

    Args:
        symbol: Trading symbol (e.g., "XRPUSDT")
        start: Start datetime
        end: End datetime
        timeframe: Timeframe (default "3m")
        validate: Run validation checks
        fix_issues: Fix common data issues

    Returns:
        DataFrame with OHLCV data
    """
    return asyncio.run(load_ohlcv_async(
        symbol=symbol,
        start=start,
        end=end,
        timeframe=timeframe,
        validate=validate,
        fix_issues=fix_issues,
    ))


def _validate_ohlcv(df: pd.DataFrame, fix_issues: bool = True) -> pd.DataFrame:
    """
    Validate OHLCV data and optionally fix issues.

    Args:
        df: DataFrame with OHLCV data
        fix_issues: If True, fix common issues

    Returns:
        Validated (and optionally fixed) DataFrame
    """
    df = df.copy()

    # Check for duplicate timestamps
    duplicates = df.duplicated(subset=["timestamp"], keep="first")
    if duplicates.any():
        logger.warning(f"Found {duplicates.sum()} duplicate timestamps")
        if fix_issues:
            df = df[~duplicates]
            logger.info(f"Removed duplicates, {len(df)} rows remaining")

    # Check for gaps in data
    expected_freq = pd.Timedelta("3min")
    time_diffs = df["timestamp"].diff()
    gaps = time_diffs[time_diffs > expected_freq * 1.5]
    if len(gaps) > 0:
        logger.warning(f"Found {len(gaps)} gaps in data")

    # Validate OHLC consistency
    invalid_ohlc = (
        (df["high"] < df["low"]) |
        (df["high"] < df["open"]) |
        (df["high"] < df["close"]) |
        (df["low"] > df["open"]) |
        (df["low"] > df["close"])
    )
    if invalid_ohlc.any():
        logger.warning(f"Found {invalid_ohlc.sum()} rows with invalid OHLC values")
        if fix_issues:
            # Fix by recalculating high/low
            df.loc[invalid_ohlc, "high"] = df.loc[invalid_ohlc, ["open", "high", "low", "close"]].max(axis=1)
            df.loc[invalid_ohlc, "low"] = df.loc[invalid_ohlc, ["open", "high", "low", "close"]].min(axis=1)

    # Check for zero or negative prices
    invalid_prices = (
        (df["open"] <= 0) |
        (df["high"] <= 0) |
        (df["low"] <= 0) |
        (df["close"] <= 0)
    )
    if invalid_prices.any():
        logger.warning(f"Found {invalid_prices.sum()} rows with invalid prices")
        if fix_issues:
            df = df[~invalid_prices]

    return df


def resample_to_15m(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample 3m candles to 15m timeframe.

    Args:
        df: DataFrame with 3m OHLCV data

    Returns:
        DataFrame with 15m OHLCV data
    """
    df = df.copy()
    df = df.set_index("timestamp")

    resampled = df.resample("15min").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    })

    # Drop incomplete bars
    resampled = resampled.dropna()

    # Reset index
    resampled = resampled.reset_index()
    resampled["symbol"] = df["symbol"].iloc[0] if "symbol" in df.columns else "XRPUSDT"
    resampled["timeframe"] = "15m"

    return resampled
