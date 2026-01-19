"""Technical indicators for feature engineering.

Provides functions to calculate technical indicators used by the Gate Layer,
XGB models, HMM regime detection, and other trading strategies.

Includes:
- Momentum: RSI, MACD, Stochastic, OBV
- Volatility: ATR, Bollinger Bands
- Moving Averages: EMA, SMA
- Trend: ADX, Choppiness, Trend Strength, Composite Trend Strength
- Volume: Volume Ratio, Z-score
"""

from typing import Tuple

import numpy as np
import pandas as pd


def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average Directional Index (ADX).

    ADX measures trend strength on a scale of 0-100.
    - 0-20: Weak or no trend
    - 20-25: Emerging trend
    - 25-50: Strong trend
    - 50-75: Very strong trend
    - 75-100: Extremely strong trend

    Args:
        df: DataFrame with 'high', 'low', 'close' columns
        period: Lookback period (default: 14)

    Returns:
        Series with ADX values
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]

    # Calculate True Range (TR)
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Calculate +DM and -DM
    high_diff = high - high.shift(1)
    low_diff = low.shift(1) - low

    plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
    minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)

    # Smooth using Wilder's smoothing (similar to EWM with alpha=1/period)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr)

    # Calculate DX (Directional Movement Index)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)

    # ADX is smoothed DX
    adx = dx.ewm(alpha=1/period, adjust=False).mean()

    return adx


def calculate_choppiness(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Choppiness Index.

    Measures market choppiness/consolidation on a scale of 0-100.
    - 0-38.2: Trending market
    - 38.2-61.8: Choppy/ranging market (most common threshold: 50)
    - 61.8-100: Very choppy/consolidating market

    Args:
        df: DataFrame with 'high', 'low', 'close' columns
        period: Lookback period (default: 14)

    Returns:
        Series with Choppiness Index values
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]

    # Calculate True Range
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Sum of True Range over period
    atr_sum = tr.rolling(window=period).sum()

    # High-Low range over period
    high_high = high.rolling(window=period).max()
    low_low = low.rolling(window=period).min()
    range_hl = high_high - low_low

    # Choppiness formula
    # CHOP = 100 * LOG10(SUM(ATR, n) / (MAX(HIGH, n) - MIN(LOW, n))) / LOG10(n)
    chop = 100 * np.log10(atr_sum / range_hl) / np.log10(period)

    return chop


def calculate_directional_consistency(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate directional consistency metric.

    Measures how consistently price moves in one direction.
    - Ranges from -1 to +1
    - +1: Perfectly consistent upward movement
    - -1: Perfectly consistent downward movement
    - 0: No directional consistency (choppy)

    Args:
        df: DataFrame with 'close' column
        period: Lookback period (default: 14)

    Returns:
        Series with directional consistency values
    """
    close = df["close"]

    # Calculate returns
    returns = close.pct_change()

    # Count positive and negative returns in the window
    positive_returns = (returns > 0).rolling(window=period).sum()
    negative_returns = (returns < 0).rolling(window=period).sum()

    # Directional consistency
    # If all returns are positive: +1
    # If all returns are negative: -1
    # If mixed: somewhere between
    total_returns = positive_returns + negative_returns
    directional_consistency = (positive_returns - negative_returns) / total_returns.replace(0, np.nan)

    return directional_consistency


def calculate_ema_slope(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Calculate EMA slope (rate of change).

    Measures the slope of the EMA to identify trend direction and strength.
    - Positive: Uptrend
    - Negative: Downtrend
    - Near zero: Flat/no trend

    Args:
        df: DataFrame with 'close' column
        period: EMA period (default: 20)

    Returns:
        Series with EMA slope values (percentage change per bar)
    """
    close = df["close"]

    # Calculate EMA
    ema = close.ewm(span=period, adjust=False).mean()

    # Calculate slope as percentage change
    # slope = (EMA_t - EMA_{t-1}) / EMA_{t-1}
    ema_slope = ema.pct_change()

    return ema_slope


def calculate_z_score(series: pd.Series, period: int = 20) -> pd.Series:
    """Calculate rolling Z-score.

    Measures how many standard deviations a value is from the mean.
    - 0: At the mean
    - +2/-2: Two standard deviations from mean
    - +3/-3: Three standard deviations from mean (rare events)

    Args:
        series: Data series
        period: Lookback period for calculating mean and std

    Returns:
        Series with Z-score values
    """
    rolling_mean = series.rolling(window=period).mean()
    rolling_std = series.rolling(window=period).std()

    # Avoid division by zero
    rolling_std = rolling_std.replace(0, np.nan)

    z_score = (series - rolling_mean) / rolling_std

    return z_score


def calculate_trend_to_range_ratio(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Calculate trend-to-range ratio.

    Measures the proportion of directional movement vs total range.
    - High values (>1.5): Strong trending
    - Low values (<1.0): Ranging/choppy

    Args:
        df: DataFrame with 'high', 'low', 'close' columns
        period: Lookback period (default: 20)

    Returns:
        Series with trend-to-range ratio values
    """
    close = df["close"]
    high = df["high"]
    low = df["low"]

    # Directional movement (absolute net change)
    directional_move = (close - close.shift(period)).abs()

    # Total range (sum of high-low ranges)
    ranges = high - low
    total_range = ranges.rolling(window=period).sum()

    # Avoid division by zero
    total_range = total_range.replace(0, np.nan)

    # Ratio
    trr = directional_move / total_range

    return trr


def calculate_vol_ratio(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Calculate volume ratio (current volume vs average volume).

    Identifies volume spikes and unusual activity.
    - 1.0: Average volume
    - >1.5: High volume (above average)
    - >2.0: Very high volume spike
    - <0.5: Low volume (below average)

    Args:
        df: DataFrame with 'volume' column
        period: Lookback period for average (default: 20)

    Returns:
        Series with volume ratio values
    """
    volume = df["volume"]

    # Calculate rolling average volume
    avg_volume = volume.rolling(window=period).mean()

    # Avoid division by zero
    avg_volume = avg_volume.replace(0, np.nan)

    # Ratio
    vol_ratio = volume / avg_volume

    return vol_ratio


def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index (RSI).

    RSI measures momentum on a scale of 0-100.
    - 0-30: Oversold
    - 30-70: Neutral
    - 70-100: Overbought

    Args:
        close: Close price series
        period: Lookback period (default: 14)

    Returns:
        Series with RSI values
    """
    delta = close.diff()

    # Separate gains and losses
    gains = delta.where(delta > 0, 0.0)
    losses = (-delta).where(delta < 0, 0.0)

    # Calculate average gains and losses using Wilder's smoothing
    avg_gains = gains.ewm(alpha=1/period, adjust=False).mean()
    avg_losses = losses.ewm(alpha=1/period, adjust=False).mean()

    # Avoid division by zero
    avg_losses = avg_losses.replace(0, np.nan)

    # Calculate RS and RSI
    rs = avg_gains / avg_losses
    rsi = 100 - (100 / (1 + rs))

    return rsi


def calculate_macd(
    close: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate MACD (Moving Average Convergence Divergence).

    Returns MACD line, signal line, and histogram.

    Args:
        close: Close price series
        fast_period: Fast EMA period (default: 12)
        slow_period: Slow EMA period (default: 26)
        signal_period: Signal line EMA period (default: 9)

    Returns:
        Tuple of (macd_line, signal_line, histogram) Series
    """
    # Calculate EMAs
    ema_fast = close.ewm(span=fast_period, adjust=False).mean()
    ema_slow = close.ewm(span=slow_period, adjust=False).mean()

    # MACD line
    macd_line = ema_fast - ema_slow

    # Signal line
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()

    # Histogram
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


def calculate_stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = 14,
    d_period: int = 3
) -> Tuple[pd.Series, pd.Series]:
    """Calculate Stochastic Oscillator (%K and %D).

    Stochastic measures momentum on a scale of 0-100.
    - 0-20: Oversold
    - 80-100: Overbought

    Args:
        high: High price series
        low: Low price series
        close: Close price series
        k_period: %K lookback period (default: 14)
        d_period: %D smoothing period (default: 3)

    Returns:
        Tuple of (stoch_k, stoch_d) Series
    """
    # Lowest low and highest high over period
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()

    # Avoid division by zero
    range_hl = highest_high - lowest_low
    range_hl = range_hl.replace(0, np.nan)

    # %K
    stoch_k = ((close - lowest_low) / range_hl) * 100

    # %D (SMA of %K)
    stoch_d = stoch_k.rolling(window=d_period).mean()

    return stoch_k, stoch_d


def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Calculate On Balance Volume (OBV).

    OBV is a cumulative volume indicator based on price direction.

    Args:
        close: Close price series
        volume: Volume series

    Returns:
        Series with OBV values
    """
    # Direction based on close price change
    direction = np.sign(close.diff())

    # OBV is cumulative sum of signed volume
    obv = (direction * volume).cumsum()

    return obv


def calculate_ema(close: pd.Series, period: int) -> pd.Series:
    """Calculate Exponential Moving Average (EMA).

    Args:
        close: Close price series
        period: EMA period

    Returns:
        Series with EMA values
    """
    return close.ewm(span=period, adjust=False).mean()


def calculate_sma(close: pd.Series, period: int) -> pd.Series:
    """Calculate Simple Moving Average (SMA).

    Args:
        close: Close price series
        period: SMA period

    Returns:
        Series with SMA values
    """
    return close.rolling(window=period).mean()


def calculate_bollinger_bands(
    close: pd.Series,
    period: int = 20,
    std_dev: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Calculate Bollinger Bands.

    Returns upper band, middle band (SMA), lower band, %B, and bandwidth.

    Args:
        close: Close price series
        period: SMA period (default: 20)
        std_dev: Standard deviation multiplier (default: 2.0)

    Returns:
        Tuple of (upper, middle, lower, pct_b, bandwidth) Series
    """
    # Middle band (SMA)
    middle = close.rolling(window=period).mean()

    # Standard deviation
    std = close.rolling(window=period).std()

    # Upper and lower bands
    upper = middle + std_dev * std
    lower = middle - std_dev * std

    # %B (where is price relative to bands)
    band_range = upper - lower
    band_range = band_range.replace(0, np.nan)
    pct_b = (close - lower) / band_range

    # Bandwidth (as percentage of middle)
    middle_safe = middle.replace(0, np.nan)
    bandwidth = (upper - lower) / middle_safe * 100

    return upper, middle, lower, pct_b, bandwidth


def calculate_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14
) -> pd.Series:
    """Calculate Average True Range (ATR).

    ATR measures volatility based on true range.

    Args:
        high: High price series
        low: Low price series
        close: Close price series
        period: ATR period (default: 14)

    Returns:
        Series with ATR values
    """
    # True Range components
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()

    # True Range is max of the three
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # ATR is smoothed true range (Wilder's smoothing)
    atr = true_range.ewm(alpha=1/period, adjust=False).mean()

    return atr


def calculate_range_pct(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series
) -> pd.Series:
    """Calculate range as percentage of close price.

    Args:
        high: High price series
        low: Low price series
        close: Close price series

    Returns:
        Series with range percentage values
    """
    close_safe = close.replace(0, np.nan)
    return (high - low) / close_safe * 100


def calculate_trend_strength(
    df: pd.DataFrame,
    period: int = 15
) -> pd.Series:
    """Calculate trend strength based on directional movement.

    Measures how strongly price is trending vs ranging.
    - Values close to 1: Strong trend
    - Values close to 0: Ranging/choppy

    Args:
        df: DataFrame with 'close' column
        period: Lookback period (default: 15)

    Returns:
        Series with trend strength values (0 to 1)
    """
    close = df["close"]

    # Net price change over period
    net_change = (close - close.shift(period)).abs()

    # Sum of absolute bar-by-bar changes
    abs_changes = close.diff().abs().rolling(window=period).sum()

    # Trend strength = net / total (0 if no movement)
    abs_changes_safe = abs_changes.replace(0, np.nan)
    trend_strength = net_change / abs_changes_safe

    return trend_strength


def calculate_composite_trend_strength(df: pd.DataFrame) -> pd.Series:
    """Calculate composite trend strength from multiple indicators.

    Combines ADX, directional consistency, and price momentum
    into a single trend strength metric.

    Args:
        df: DataFrame with OHLCV data

    Returns:
        Series with composite trend strength (0 to 1)
    """
    # ADX component (normalized to 0-1)
    adx = calculate_adx(df, period=14)
    adx_norm = adx / 100

    # Directional consistency component (absolute value, 0-1)
    dir_cons = calculate_directional_consistency(df, period=14).abs()

    # Trend strength component
    trend_str = calculate_trend_strength(df, period=14)

    # Composite: weighted average
    composite = (0.4 * adx_norm + 0.3 * dir_cons + 0.3 * trend_str)

    # Clip to 0-1 range
    return composite.clip(0, 1)


def calculate_price_z_from_ema(
    close: pd.Series,
    ema_period: int = 20,
    z_period: int = 20
) -> pd.Series:
    """Calculate Z-score of price distance from EMA.

    Measures how far price is from its EMA in standard deviations.

    Args:
        close: Close price series
        ema_period: EMA period (default: 20)
        z_period: Z-score calculation period (default: 20)

    Returns:
        Series with Z-score values
    """
    # Calculate EMA
    ema = close.ewm(span=ema_period, adjust=False).mean()

    # Distance from EMA (as percentage)
    ema_safe = ema.replace(0, np.nan)
    distance = (close - ema) / ema_safe * 100

    # Z-score of the distance
    return calculate_z_score(distance, period=z_period)
