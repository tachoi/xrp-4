"""Feature building for 3m OHLCV data.

Generates all features defined in feature_spec from 3m OHLCV data.
Supports multi-timeframe features (15m, 1h) and MongoDB caching.
"""

import asyncio
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import structlog

from xrp3.core.constants import BASE_TIMEFRAME, DEFAULT_SYMBOL
from xrp3.data.loader import load_ohlcv
from xrp3.features.feature_spec import (
    FEATURE_COLUMNS,
    FEATURE_COLUMNS_WITH_MULTITF,
    REQUIRED_SOURCE_COLUMNS,
    validate_feature_columns,
)
from xrp3.features.indicators import (
    calculate_adx,
    calculate_choppiness,
    calculate_directional_consistency,
    calculate_ema_slope,
    calculate_trend_to_range_ratio,
    calculate_vol_ratio,
    calculate_z_score,
)

logger = structlog.get_logger(__name__)

# Default parameters
DEFAULT_EWM_SPAN = 20
DEFAULT_ATR_PERIOD = 14
DEFAULT_BB_PERIOD = 20
DEFAULT_BB_STD = 2.0
DEFAULT_VOL_Z_PERIOD = 20

# Gate feature parameters
DEFAULT_ADX_PERIOD = 14
DEFAULT_CHOP_PERIOD = 14
DEFAULT_DIRCONS_PERIOD = 14
DEFAULT_TRR_PERIOD = 20
DEFAULT_VOLRATIO_PERIOD = 20
DEFAULT_Z_SCORE_PERIOD = 20
DEFAULT_EMA_PERIOD = 20


async def build_features(
    symbol: str = DEFAULT_SYMBOL,
    start: Optional[str | datetime] = None,
    end: Optional[str | datetime] = None,
    df: Optional[pd.DataFrame] = None,
    ewm_span: int = DEFAULT_EWM_SPAN,
    atr_period: int = DEFAULT_ATR_PERIOD,
    bb_period: int = DEFAULT_BB_PERIOD,
    bb_std: float = DEFAULT_BB_STD,
    vol_z_period: int = DEFAULT_VOL_Z_PERIOD,
    drop_na: bool = True,
    include_multitf: bool = False,
    save_to_mongo: bool = False,
    load_from_mongo: bool = True,
    adx_period: int = DEFAULT_ADX_PERIOD,
    chop_period: int = DEFAULT_CHOP_PERIOD,
    dircons_period: int = DEFAULT_DIRCONS_PERIOD,
    trr_period: int = DEFAULT_TRR_PERIOD,
    volratio_period: int = DEFAULT_VOLRATIO_PERIOD,
    z_score_period: int = DEFAULT_Z_SCORE_PERIOD,
    ema_period: int = DEFAULT_EMA_PERIOD,
) -> pd.DataFrame:
    """Build features from 3m OHLCV data.

    If df is not provided, loads data from database.
    Supports MongoDB caching and multi-timeframe features.

    Args:
        symbol: Trading symbol
        start: Start datetime
        end: End datetime
        df: Pre-loaded OHLCV DataFrame (optional)
        ewm_span: EWM span for return smoothing
        atr_period: Period for ATR calculation
        bb_period: Period for Bollinger Bands
        bb_std: Standard deviation multiplier for BB
        vol_z_period: Period for volume Z-score
        drop_na: If True, drop rows with NaN values
        include_multitf: If True, include 15m/1h features
        save_to_mongo: If True, save features to MongoDB
        load_from_mongo: If True, try to load from MongoDB first
        adx_period: Period for ADX calculation
        chop_period: Period for Choppiness Index
        dircons_period: Period for directional consistency
        trr_period: Period for trend-to-range ratio
        volratio_period: Period for volume ratio
        z_score_period: Period for Z-score calculation
        ema_period: Period for EMA slope

    Returns:
        DataFrame with OHLCV data and all feature columns
    """
    # Try loading from MongoDB first (if requested and no df provided)
    if load_from_mongo and df is None and start and end:
        try:
            from xrp3.db.mongodb import get_mongo_store_async

            mongo_store = await get_mongo_store_async()
            if mongo_store:
                # Convert start/end to datetime if needed
                start_dt = pd.to_datetime(start) if isinstance(start, str) else start
                end_dt = pd.to_datetime(end) if isinstance(end, str) else end

                # Try to load features from MongoDB
                cached_df = await mongo_store.load_features(symbol, start_dt, end_dt)

                if not cached_df.empty:
                    # Check if all required features exist
                    expected_features = FEATURE_COLUMNS_WITH_MULTITF if include_multitf else FEATURE_COLUMNS
                    missing_features = [col for col in expected_features if col not in cached_df.columns]

                    if not missing_features:
                        logger.info(
                            "Loaded features from MongoDB cache",
                            symbol=symbol,
                            rows=len(cached_df),
                            features=len(expected_features)
                        )
                        return cached_df
                    else:
                        logger.debug(
                            "MongoDB cache missing some features, recalculating",
                            missing=missing_features[:5]
                        )
        except Exception as e:
            logger.warning("Error loading from MongoDB, will calculate fresh", error=str(e))

    # Load data if not provided
    if df is None:
        df = await load_ohlcv(symbol, start, end)

    if df.empty:
        logger.warning("Empty DataFrame provided for feature building")
        return df

    # Ensure numeric columns are float (convert from Decimal if needed)
    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].astype(float)

    # Validate source columns
    missing_source = [col for col in REQUIRED_SOURCE_COLUMNS if col not in df.columns]
    if missing_source:
        raise ValueError(f"Missing required source columns: {missing_source}")

    logger.info(
        "Building features",
        symbol=symbol,
        rows=len(df),
        start=df["timestamp"].min(),
        end=df["timestamp"].max(),
    )

    # Create working copy
    result = df.copy()

    # Build base features
    result = _build_return_features(result, ewm_span)
    result = _build_volatility_features(result, atr_period, bb_period, bb_std)
    result = _build_candle_features(result)
    result = _build_volume_features(result, vol_z_period)

    # Build Gate features (3m)
    result = _build_gate_features(
        result,
        adx_period=adx_period,
        chop_period=chop_period,
        dircons_period=dircons_period,
        trr_period=trr_period,
        volratio_period=volratio_period,
        z_score_period=z_score_period,
    )

    # Build multi-timeframe features (optional)
    if include_multitf:
        logger.info("Building multi-timeframe features (15m/1h)")
        result = _build_multitf_features(
            result,
            chop_period=chop_period,
            volratio_period=volratio_period,
            dircons_period=dircons_period,
            z_score_period=z_score_period,
            ema_period=ema_period,
        )

    # Validate all features were created
    expected_features = FEATURE_COLUMNS_WITH_MULTITF if include_multitf else FEATURE_COLUMNS
    missing = [col for col in expected_features if col not in result.columns]
    if missing:
        raise ValueError(f"Failed to create features: {missing}")

    # Handle NaN values
    feature_cols_to_check = expected_features
    nan_counts = result[feature_cols_to_check].isna().sum()
    total_nans = nan_counts.sum()

    if drop_na and total_nans > 0:
        original_len = len(result)
        result = result.dropna(subset=feature_cols_to_check).reset_index(drop=True)
        dropped = original_len - len(result)
        logger.info(
            "Dropped rows with NaN values",
            dropped=dropped,
            remaining=len(result),
        )

    logger.info(
        "Features built successfully",
        feature_count=len(expected_features),
        rows=len(result),
        multitf=include_multitf,
    )

    # Save to MongoDB if requested
    if save_to_mongo:
        try:
            from xrp3.db.mongodb import get_mongo_store_async

            mongo_store = await get_mongo_store_async()
            if mongo_store:
                saved_count = await mongo_store.save_features(symbol, result, overwrite=True)
                logger.info(
                    "Saved features to MongoDB",
                    symbol=symbol,
                    documents=saved_count
                )
        except Exception as e:
            logger.error("Error saving features to MongoDB", error=str(e))

    return result


def build_features_sync(
    symbol: str = DEFAULT_SYMBOL,
    start: Optional[str | datetime] = None,
    end: Optional[str | datetime] = None,
    df: Optional[pd.DataFrame] = None,
    **kwargs,
) -> pd.DataFrame:
    """Synchronous wrapper for build_features.

    Args:
        symbol: Trading symbol
        start: Start datetime
        end: End datetime
        df: Pre-loaded DataFrame
        **kwargs: Additional parameters for build_features

    Returns:
        DataFrame with features
    """
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(
        build_features(symbol, start, end, df, **kwargs)
    )


def _build_return_features(df: pd.DataFrame, ewm_span: int) -> pd.DataFrame:
    """Build return-based features.

    Features:
        - ret_3m: Simple return
        - logret_3m: Log return
        - ewm_ret_3m: EWM smoothed return
    """
    # Simple return
    df["ret_3m"] = df["close"].pct_change()

    # Log return
    df["logret_3m"] = np.log(df["close"]) - np.log(df["close"].shift(1))

    # EWM smoothed return
    df["ewm_ret_3m"] = df["ret_3m"].ewm(span=ewm_span, adjust=False).mean()

    return df


def _build_volatility_features(
    df: pd.DataFrame,
    atr_period: int,
    bb_period: int,
    bb_std: float,
) -> pd.DataFrame:
    """Build volatility-based features.

    Features:
        - atr_pct_3m: ATR as percentage of close price
        - ewm_std_ret_3m: EWM standard deviation of returns
        - bb_width_3m: Bollinger Band width as percentage
    """
    # ATR calculation
    high = df["high"]
    low = df["low"]
    close = df["close"]

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window=atr_period).mean()
    df["atr_pct_3m"] = atr / close * 100

    # EWM standard deviation of returns
    if "ret_3m" not in df.columns:
        df["ret_3m"] = close.pct_change()
    df["ewm_std_ret_3m"] = df["ret_3m"].ewm(span=bb_period, adjust=False).std()

    # Bollinger Band width
    sma = close.rolling(window=bb_period).mean()
    std = close.rolling(window=bb_period).std()
    upper_band = sma + bb_std * std
    lower_band = sma - bb_std * std
    df["bb_width_3m"] = (upper_band - lower_band) / sma * 100

    return df


def _build_candle_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build candlestick pattern features.

    Features:
        - body_ratio: |close-open| / (high-low)
        - upper_wick_ratio: upper_wick / (high-low)
        - lower_wick_ratio: lower_wick / (high-low)
    """
    high = df["high"]
    low = df["low"]
    open_ = df["open"]
    close = df["close"]

    # Total range (avoid division by zero)
    total_range = high - low
    total_range = total_range.replace(0, np.nan)

    # Body
    body = (close - open_).abs()
    df["body_ratio"] = body / total_range

    # Upper wick
    upper_wick = high - pd.concat([open_, close], axis=1).max(axis=1)
    df["upper_wick_ratio"] = upper_wick / total_range

    # Lower wick
    lower_wick = pd.concat([open_, close], axis=1).min(axis=1) - low
    df["lower_wick_ratio"] = lower_wick / total_range

    return df


def _build_volume_features(df: pd.DataFrame, vol_z_period: int) -> pd.DataFrame:
    """Build volume-based features.

    Features:
        - vol_z_3m: Volume Z-score
    """
    volume = df["volume"]

    vol_mean = volume.rolling(window=vol_z_period).mean()
    vol_std = volume.rolling(window=vol_z_period).std()

    # Avoid division by zero
    vol_std = vol_std.replace(0, np.nan)

    df["vol_z_3m"] = (volume - vol_mean) / vol_std

    return df


def _build_gate_features(
    df: pd.DataFrame,
    adx_period: int = DEFAULT_ADX_PERIOD,
    chop_period: int = DEFAULT_CHOP_PERIOD,
    dircons_period: int = DEFAULT_DIRCONS_PERIOD,
    trr_period: int = DEFAULT_TRR_PERIOD,
    volratio_period: int = DEFAULT_VOLRATIO_PERIOD,
    z_score_period: int = DEFAULT_Z_SCORE_PERIOD,
) -> pd.DataFrame:
    """Build Gate-required features (3m only).

    Features:
        - ret_abs_z_3m: Absolute return Z-score
        - vol_ratio_3m: Volume ratio (current / average)
        - adx_3m: Average Directional Index
        - directional_consistency_3m: Directional consistency
        - chop_3m: Choppiness Index
        - trend_to_range_ratio_3m: Trend-to-range ratio
    """
    # ret_abs_z_3m
    if "ret_3m" not in df.columns:
        df["ret_3m"] = df["close"].pct_change()
    df["ret_abs_z_3m"] = calculate_z_score(df["ret_3m"].abs(), period=z_score_period)

    # vol_ratio_3m
    df["vol_ratio_3m"] = calculate_vol_ratio(df, period=volratio_period)

    # adx_3m
    df["adx_3m"] = calculate_adx(df, period=adx_period)

    # directional_consistency_3m
    df["directional_consistency_3m"] = calculate_directional_consistency(df, period=dircons_period)

    # chop_3m
    df["chop_3m"] = calculate_choppiness(df, period=chop_period)

    # trend_to_range_ratio_3m
    df["trend_to_range_ratio_3m"] = calculate_trend_to_range_ratio(df, period=trr_period)

    return df


def _resample_to_timeframe(df_3m: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Resample 3m OHLCV to higher timeframe (15m/1h).

    Args:
        df_3m: DataFrame with 3m OHLCV data (must have timestamp index or column)
        timeframe: Target timeframe ('15min' or '1h')

    Returns:
        Resampled DataFrame with OHLCV columns
    """
    # Ensure timestamp is index
    if "timestamp" in df_3m.columns:
        df = df_3m.set_index("timestamp")
    else:
        df = df_3m.copy()

    # Resample OHLCV data
    resampled = df.resample(timeframe).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    })

    # Drop rows with NaN (incomplete periods)
    resampled = resampled.dropna(subset=["close"])

    return resampled.reset_index()


def _build_multitf_features(
    df_3m: pd.DataFrame,
    chop_period: int = DEFAULT_CHOP_PERIOD,
    volratio_period: int = DEFAULT_VOLRATIO_PERIOD,
    dircons_period: int = DEFAULT_DIRCONS_PERIOD,
    z_score_period: int = DEFAULT_Z_SCORE_PERIOD,
    ema_period: int = DEFAULT_EMA_PERIOD,
) -> pd.DataFrame:
    """Build multi-timeframe features and merge into 3m DataFrame.

    Args:
        df_3m: DataFrame with 3m OHLCV data
        chop_period: Period for Choppiness Index
        volratio_period: Period for volume ratio
        dircons_period: Period for directional consistency
        z_score_period: Period for Z-score
        ema_period: Period for EMA slope

    Returns:
        DataFrame with 3m data plus 15m/1h features (forward-filled)
    """
    # Resample to 15m and 1h
    df_15m = _resample_to_timeframe(df_3m.copy(), "15min")
    df_1h = _resample_to_timeframe(df_3m.copy(), "1h")

    # Calculate 15m features
    df_15m["chop_15m"] = calculate_choppiness(df_15m, period=chop_period)
    df_15m["vol_ratio_15m"] = calculate_vol_ratio(df_15m, period=volratio_period)
    df_15m["directional_consistency_15m"] = calculate_directional_consistency(
        df_15m, period=dircons_period
    )

    # Calculate 1h features
    df_1h["ret_1h"] = df_1h["close"].pct_change()
    df_1h["ret_abs_z_1h"] = calculate_z_score(df_1h["ret_1h"].abs(), period=z_score_period)
    df_1h["vol_ratio_1h"] = calculate_vol_ratio(df_1h, period=volratio_period)
    df_1h["ema_slope_1h"] = calculate_ema_slope(df_1h, period=ema_period)

    # Merge back to 3m (forward-fill higher TF values)
    df_result = df_3m.set_index("timestamp") if "timestamp" in df_3m.columns else df_3m.copy()

    # Select only the feature columns for joining
    df_15m_features = df_15m.set_index("timestamp")[
        ["chop_15m", "vol_ratio_15m", "directional_consistency_15m"]
    ]
    df_1h_features = df_1h.set_index("timestamp")[
        ["ret_abs_z_1h", "vol_ratio_1h", "ema_slope_1h"]
    ]

    # Join features
    df_result = df_result.join(df_15m_features, how="left")
    df_result = df_result.join(df_1h_features, how="left")

    # Forward-fill higher timeframe values
    df_result[["chop_15m", "vol_ratio_15m", "directional_consistency_15m"]] = df_result[
        ["chop_15m", "vol_ratio_15m", "directional_consistency_15m"]
    ].ffill()

    df_result[["ret_abs_z_1h", "vol_ratio_1h", "ema_slope_1h"]] = df_result[
        ["ret_abs_z_1h", "vol_ratio_1h", "ema_slope_1h"]
    ].ffill()

    return df_result.reset_index() if "timestamp" not in df_result.columns else df_result.reset_index(drop=False)
