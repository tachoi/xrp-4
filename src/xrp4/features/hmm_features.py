"""HMM Feature builder for xrp-4.

Builds features required for HMM training from OHLCV candle data.
Uses the feature contract defined in configs/hmm_features.yaml.

Supports multi-timeframe features:
- Fast HMM (3m): Uses 3m + 15m features
- Mid HMM (15m): Uses 15m + 1h features
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from xrp4.features.indicators import (
    calculate_adx,
    calculate_atr,
    calculate_bollinger_bands,
    calculate_choppiness,
    calculate_directional_consistency,
    calculate_ema,
    calculate_ema_slope,
    calculate_macd,
    calculate_rsi,
    calculate_trend_strength,
    calculate_vol_ratio,
    calculate_z_score,
    calculate_price_z_from_ema,
)


def build_hmm_features(
    df: pd.DataFrame,
    feature_list: List[str],
    drop_na: bool = True,
) -> Tuple[np.ndarray, pd.DatetimeIndex]:
    """Build HMM features from OHLCV data.

    Args:
        df: DataFrame with OHLCV columns (open, high, low, close, volume, timestamp)
        feature_list: List of feature names to build (from feature contract)
        drop_na: If True, drop rows with NaN values

    Returns:
        Tuple of (feature_matrix, timestamps)
    """
    # Ensure numeric columns
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = df[col].astype(float)

    # Build each feature
    result = df.copy()
    result = _build_all_features(result)

    # Select only requested features
    missing = [f for f in feature_list if f not in result.columns]
    if missing:
        raise ValueError(f"Missing features: {missing}")

    # Get feature matrix
    features = result[feature_list].values

    # Get timestamps
    if "timestamp" in result.columns:
        timestamps = pd.DatetimeIndex(result["timestamp"])
    else:
        timestamps = result.index

    # Handle NaN
    if drop_na:
        mask = ~np.isnan(features).any(axis=1)
        features = features[mask]
        timestamps = timestamps[mask]

    return features, timestamps


def _build_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build all possible features for HMM.

    Args:
        df: OHLCV DataFrame

    Returns:
        DataFrame with all features added
    """
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # Returns
    df["ret_3m"] = close.pct_change()
    df["logret_3m"] = np.log(close) - np.log(close.shift(1))
    df["ewm_ret_3m"] = df["ret_3m"].ewm(span=20, adjust=False).mean()

    # Absolute return
    df["ret_abs_3m"] = df["ret_3m"].abs()

    # Z-scores
    df["ret_z_3m"] = calculate_z_score(df["ret_3m"], period=20)
    df["ret_abs_z_3m"] = calculate_z_score(df["ret_abs_3m"], period=20)

    # ATR
    atr = calculate_atr(high, low, close, period=14)
    df["atr_3m"] = atr
    df["atr_pct_3m"] = atr / close * 100

    # ATR Z-score
    df["atr_z_3m"] = calculate_z_score(df["atr_pct_3m"], period=20)

    # Volatility
    df["ewm_std_ret_3m"] = df["ret_3m"].ewm(span=20, adjust=False).std()

    # Bollinger Bands
    upper, middle, lower, pct_b, bandwidth = calculate_bollinger_bands(close, period=20)
    df["bb_width_3m"] = bandwidth
    df["bb_pct_b_3m"] = pct_b

    # RSI
    df["rsi_3m"] = calculate_rsi(close, period=14)
    df["rsi_z_3m"] = calculate_z_score(df["rsi_3m"], period=20)

    # MACD
    macd_line, signal_line, histogram = calculate_macd(close)
    df["macd_hist_3m"] = histogram
    df["macd_hist_z_3m"] = calculate_z_score(histogram, period=20)

    # ADX
    df["adx_3m"] = calculate_adx(df, period=14)

    # Choppiness
    df["chop_3m"] = calculate_choppiness(df, period=14)

    # Directional consistency
    df["directional_consistency_3m"] = calculate_directional_consistency(df, period=14)

    # Trend strength
    df["trend_strength_3m"] = calculate_trend_strength(df, period=14)

    # EMA slope
    df["ema_slope_3m"] = calculate_ema_slope(df, period=20)

    # Volume features
    df["vol_ratio_3m"] = calculate_vol_ratio(df, period=20)
    vol_mean = volume.rolling(20).mean()
    vol_std = volume.rolling(20).std().replace(0, np.nan)
    df["vol_z_3m"] = (volume - vol_mean) / vol_std

    # Price position relative to range
    rolling_high = high.rolling(20).max()
    rolling_low = low.rolling(20).min()
    range_size = (rolling_high - rolling_low).replace(0, np.nan)
    df["price_position_3m"] = (close - rolling_low) / range_size

    # Body ratio
    body = (close - df["open"]).abs()
    total_range = (high - low).replace(0, np.nan)
    df["body_ratio"] = body / total_range

    return df


def resample_to_15m(df_3m: pd.DataFrame) -> pd.DataFrame:
    """Resample 3m OHLCV data to 15m.

    Args:
        df_3m: DataFrame with 3m OHLCV data

    Returns:
        Resampled 15m DataFrame
    """
    if "timestamp" in df_3m.columns:
        df = df_3m.set_index("timestamp")
    else:
        df = df_3m.copy()

    resampled = df.resample("15min").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna(subset=["close"])

    return resampled.reset_index()


def build_mid_hmm_features(
    df_15m: pd.DataFrame,
    feature_list: List[str],
    drop_na: bool = True,
) -> Tuple[np.ndarray, pd.DatetimeIndex]:
    """Build features for Mid HMM (15m timeframe).

    Args:
        df_15m: DataFrame with 15m OHLCV data
        feature_list: List of feature names to build
        drop_na: If True, drop rows with NaN values

    Returns:
        Tuple of (feature_matrix, timestamps)
    """
    # Ensure numeric
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df_15m.columns:
            df_15m[col] = df_15m[col].astype(float)

    result = df_15m.copy()
    close = result["close"]
    high = result["high"]
    low = result["low"]
    volume = result["volume"]

    # Returns
    result["ret_15m"] = close.pct_change()
    result["logret_15m"] = np.log(close) - np.log(close.shift(1))
    result["ewm_ret_15m"] = result["ret_15m"].ewm(span=20, adjust=False).mean()

    # Absolute return
    result["ret_abs_15m"] = result["ret_15m"].abs()

    # Z-scores
    result["ret_z_15m"] = calculate_z_score(result["ret_15m"], period=20)
    result["ret_abs_z_15m"] = calculate_z_score(result["ret_abs_15m"], period=20)

    # ATR
    atr = calculate_atr(high, low, close, period=14)
    result["atr_15m"] = atr
    result["atr_pct_15m"] = atr / close * 100
    result["atr_z_15m"] = calculate_z_score(result["atr_pct_15m"], period=20)

    # Volatility
    result["ewm_std_ret_15m"] = result["ret_15m"].ewm(span=20, adjust=False).std()

    # Bollinger Bands
    upper, middle, lower, pct_b, bandwidth = calculate_bollinger_bands(close, period=20)
    result["bb_width_15m"] = bandwidth

    # RSI
    result["rsi_15m"] = calculate_rsi(close, period=14)
    result["rsi_z_15m"] = calculate_z_score(result["rsi_15m"], period=20)

    # ADX
    result["adx_15m"] = calculate_adx(result, period=14)

    # Choppiness
    result["chop_15m"] = calculate_choppiness(result, period=14)

    # Directional consistency
    result["directional_consistency_15m"] = calculate_directional_consistency(result, period=14)

    # Trend strength
    result["trend_strength_15m"] = calculate_trend_strength(result, period=14)

    # EMA slope
    result["ema_slope_15m"] = calculate_ema_slope(result, period=20)

    # Volume features
    result["vol_ratio_15m"] = calculate_vol_ratio(result, period=20)
    vol_mean = volume.rolling(20).mean()
    vol_std = volume.rolling(20).std().replace(0, np.nan)
    result["vol_z_15m"] = (volume - vol_mean) / vol_std

    # Price position
    rolling_high = high.rolling(20).max()
    rolling_low = low.rolling(20).min()
    range_size = (rolling_high - rolling_low).replace(0, np.nan)
    result["price_position_15m"] = (close - rolling_low) / range_size

    # MACD
    macd_line, signal_line, histogram = calculate_macd(close)
    result["macd_hist_15m"] = histogram
    result["macd_hist_z_15m"] = calculate_z_score(histogram, period=20)

    # Select requested features
    missing = [f for f in feature_list if f not in result.columns]
    if missing:
        raise ValueError(f"Missing 15m features: {missing}")

    features = result[feature_list].values

    if "timestamp" in result.columns:
        timestamps = pd.DatetimeIndex(result["timestamp"])
    else:
        timestamps = result.index

    if drop_na:
        mask = ~np.isnan(features).any(axis=1)
        features = features[mask]
        timestamps = timestamps[mask]

    return features, timestamps


def build_fast_hmm_features_v2(
    df_3m: pd.DataFrame,
    feature_list: List[str],
    drop_na: bool = True,
) -> Tuple[np.ndarray, pd.DatetimeIndex]:
    """Build features for Fast HMM (3m) with multi-timeframe support.

    According to hmm_features.yaml v2, Fast HMM uses:
    - 3m candle features (body_ratio, wick ratios, range_pct, ret_3m)
    - 15m volatility features (ewm_ret_15m, atr_pct_15m, ewm_std_ret_15m, bb_width_15m)

    Args:
        df_3m: DataFrame with 3m OHLCV data
        feature_list: List of feature names from contract
        drop_na: If True, drop rows with NaN values

    Returns:
        Tuple of (feature_matrix, timestamps)
    """
    # Copy first to avoid SettingWithCopyWarning
    result = df_3m.copy()

    # Ensure numeric
    for col in ["open", "high", "low", "close", "volume"]:
        if col in result.columns:
            result[col] = result[col].astype(float)
    close = result["close"]
    high = result["high"]
    low = result["low"]

    # ===== 3m features =====
    # Returns
    result["ret_3m"] = close.pct_change()

    # Candle micro-structure
    body = (close - result["open"]).abs()
    total_range = (high - low).replace(0, np.nan)
    result["body_ratio"] = body / total_range

    # Upper wick
    upper_wick = high - pd.concat([result["open"], close], axis=1).max(axis=1)
    result["upper_wick_ratio"] = upper_wick / total_range

    # Lower wick
    lower_wick = pd.concat([result["open"], close], axis=1).min(axis=1) - low
    result["lower_wick_ratio"] = lower_wick / total_range

    # Range as percentage of close
    result["range_pct"] = (high - low) / close.replace(0, np.nan) * 100

    # ===== 15m features (resampled from 3m) =====
    df_15m = resample_to_15m(df_3m)
    close_15m = df_15m["close"]
    high_15m = df_15m["high"]
    low_15m = df_15m["low"]

    # 15m returns
    df_15m["ret_15m"] = close_15m.pct_change()
    df_15m["ewm_ret_15m"] = df_15m["ret_15m"].ewm(span=20, adjust=False).mean()

    # 15m ATR
    atr_15m = calculate_atr(high_15m, low_15m, close_15m, period=14)
    df_15m["atr_pct_15m"] = atr_15m / close_15m * 100

    # 15m volatility
    df_15m["ewm_std_ret_15m"] = df_15m["ret_15m"].ewm(span=20, adjust=False).std()

    # 15m Bollinger width
    upper, middle, lower, pct_b, bandwidth = calculate_bollinger_bands(close_15m, period=20)
    df_15m["bb_width_15m"] = bandwidth

    # Merge 15m features back to 3m (forward fill)
    if "timestamp" in result.columns:
        result = result.set_index("timestamp")
    if "timestamp" in df_15m.columns:
        df_15m = df_15m.set_index("timestamp")

    features_15m = df_15m[["ewm_ret_15m", "atr_pct_15m", "ewm_std_ret_15m", "bb_width_15m"]]
    result = result.join(features_15m, how="left")
    result[["ewm_ret_15m", "atr_pct_15m", "ewm_std_ret_15m", "bb_width_15m"]] = result[
        ["ewm_ret_15m", "atr_pct_15m", "ewm_std_ret_15m", "bb_width_15m"]
    ].ffill()

    result = result.reset_index()

    # Select requested features
    missing = [f for f in feature_list if f not in result.columns]
    if missing:
        raise ValueError(f"Missing Fast HMM features: {missing}")

    features = result[feature_list].values

    if "timestamp" in result.columns:
        timestamps = pd.DatetimeIndex(result["timestamp"])
    else:
        timestamps = result.index

    if drop_na:
        mask = ~np.isnan(features).any(axis=1)
        features = features[mask]
        timestamps = timestamps[mask]

    return features, timestamps


def build_mid_hmm_features_v2(
    df_15m: pd.DataFrame,
    feature_list: List[str],
    drop_na: bool = True,
) -> Tuple[np.ndarray, pd.DatetimeIndex]:
    """Build features for Mid HMM (15m) with 1h context.

    According to hmm_features.yaml v2, Mid HMM uses:
    - 15m features (ret_15m, ewm_ret_15m, atr_pct_15m, etc.)
    - 1h context features (ret_1h, ewm_ret_1h, atr_pct_1h, etc.)

    Args:
        df_15m: DataFrame with 15m OHLCV data
        feature_list: List of feature names from contract
        drop_na: If True, drop rows with NaN values

    Returns:
        Tuple of (feature_matrix, timestamps)
    """
    # Copy first to avoid SettingWithCopyWarning
    result = df_15m.copy()

    # Ensure numeric
    for col in ["open", "high", "low", "close", "volume"]:
        if col in result.columns:
            result[col] = result[col].astype(float)

    close = result["close"]
    high = result["high"]
    low = result["low"]
    volume = result["volume"]

    # ===== 15m features =====
    result["ret_15m"] = close.pct_change()
    result["ewm_ret_15m"] = result["ret_15m"].ewm(span=20, adjust=False).mean()

    atr = calculate_atr(high, low, close, period=14)
    result["atr_pct_15m"] = atr / close * 100
    result["ewm_std_ret_15m"] = result["ret_15m"].ewm(span=20, adjust=False).std()

    upper, middle, lower, pct_b, bandwidth = calculate_bollinger_bands(close, period=20)
    result["bb_width_15m"] = bandwidth

    vol_mean = volume.rolling(20).mean()
    vol_std = volume.rolling(20).std().replace(0, np.nan)
    result["vol_z_15m"] = (volume - vol_mean) / vol_std

    # ===== 1h features (resampled from 15m) =====
    df_1h = resample_to_1h(df_15m)
    close_1h = df_1h["close"]
    high_1h = df_1h["high"]
    low_1h = df_1h["low"]

    df_1h["ret_1h"] = close_1h.pct_change()
    df_1h["ewm_ret_1h"] = df_1h["ret_1h"].ewm(span=20, adjust=False).mean()

    atr_1h = calculate_atr(high_1h, low_1h, close_1h, period=14)
    df_1h["atr_pct_1h"] = atr_1h / close_1h * 100
    df_1h["ewm_std_ret_1h"] = df_1h["ret_1h"].ewm(span=20, adjust=False).std()

    # Price Z from EMA (1h)
    df_1h["price_z_from_ema_1h"] = calculate_price_z_from_ema(close_1h, ema_period=20, z_period=20)

    # Merge 1h features back to 15m
    if "timestamp" in result.columns:
        result = result.set_index("timestamp")
    if "timestamp" in df_1h.columns:
        df_1h = df_1h.set_index("timestamp")

    features_1h = df_1h[["ret_1h", "ewm_ret_1h", "atr_pct_1h", "ewm_std_ret_1h", "price_z_from_ema_1h"]]
    # Drop existing 1h columns if they exist to avoid overlap error
    cols_to_drop = [c for c in features_1h.columns if c in result.columns]
    if cols_to_drop:
        result = result.drop(columns=cols_to_drop)
    result = result.join(features_1h, how="left")
    result[["ret_1h", "ewm_ret_1h", "atr_pct_1h", "ewm_std_ret_1h", "price_z_from_ema_1h"]] = result[
        ["ret_1h", "ewm_ret_1h", "atr_pct_1h", "ewm_std_ret_1h", "price_z_from_ema_1h"]
    ].ffill()

    result = result.reset_index()

    # Select requested features
    missing = [f for f in feature_list if f not in result.columns]
    if missing:
        raise ValueError(f"Missing Mid HMM features: {missing}")

    features = result[feature_list].values

    if "timestamp" in result.columns:
        timestamps = pd.DatetimeIndex(result["timestamp"])
    else:
        timestamps = result.index

    if drop_na:
        mask = ~np.isnan(features).any(axis=1)
        features = features[mask]
        timestamps = timestamps[mask]

    return features, timestamps


def resample_to_1h(df_15m: pd.DataFrame) -> pd.DataFrame:
    """Resample 15m OHLCV data to 1h.

    Args:
        df_15m: DataFrame with 15m OHLCV data

    Returns:
        Resampled 1h DataFrame
    """
    if "timestamp" in df_15m.columns:
        df = df_15m.set_index("timestamp")
    else:
        df = df_15m.copy()

    resampled = df.resample("1h").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna(subset=["close"])

    return resampled.reset_index()
