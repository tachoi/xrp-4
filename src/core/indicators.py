"""
Technical indicators for the XRP Core Trading System.
No external TA libraries - pure Python with optional numpy.
All calculations are deterministic.
"""
from typing import List, Optional, Tuple
from .types import Candle

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


def sma(values: List[float], period: int) -> List[float]:
    """
    Simple Moving Average.

    Args:
        values: List of price values
        period: SMA period

    Returns:
        List of SMA values (first period-1 values are 0.0)
    """
    n = len(values)
    result = [0.0] * n

    if n < period:
        return result

    # First valid SMA
    window_sum = sum(values[:period])
    result[period - 1] = window_sum / period

    # Rolling calculation
    for i in range(period, n):
        window_sum = window_sum - values[i - period] + values[i]
        result[i] = window_sum / period

    return result


def ema(values: List[float], period: int) -> List[float]:
    """
    Exponential Moving Average.

    Args:
        values: List of price values
        period: EMA period

    Returns:
        List of EMA values (first period-1 values are 0.0)
    """
    n = len(values)
    result = [0.0] * n

    if n < period:
        return result

    # Initialize with SMA
    initial_sma = sum(values[:period]) / period
    result[period - 1] = initial_sma

    # EMA multiplier
    mult = 2.0 / (period + 1)

    # Calculate EMA
    for i in range(period, n):
        result[i] = (values[i] - result[i - 1]) * mult + result[i - 1]

    return result


def rsi(values: List[float], period: int = 14) -> List[float]:
    """
    Relative Strength Index.

    Args:
        values: List of price values (typically close prices)
        period: RSI period (default 14)

    Returns:
        List of RSI values (0-100), first period values are 50.0
    """
    n = len(values)
    result = [50.0] * n  # Default neutral RSI

    if n < period + 1:
        return result

    # Calculate price changes
    changes = [0.0] * n
    for i in range(1, n):
        changes[i] = values[i] - values[i - 1]

    # Separate gains and losses
    gains = [max(c, 0.0) for c in changes]
    losses = [max(-c, 0.0) for c in changes]

    # Initial average gain/loss (SMA)
    avg_gain = sum(gains[1:period + 1]) / period
    avg_loss = sum(losses[1:period + 1]) / period

    # First RSI value
    if avg_loss == 0:
        result[period] = 100.0 if avg_gain > 0 else 50.0
    else:
        rs = avg_gain / avg_loss
        result[period] = 100.0 - (100.0 / (1.0 + rs))

    # Smoothed RSI (Wilder's smoothing)
    for i in range(period + 1, n):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

        if avg_loss == 0:
            result[i] = 100.0 if avg_gain > 0 else 50.0
        else:
            rs = avg_gain / avg_loss
            result[i] = 100.0 - (100.0 / (1.0 + rs))

    return result


def atr(candles: List[Candle], period: int = 14) -> List[float]:
    """
    Average True Range.

    Args:
        candles: List of OHLCV candles
        period: ATR period (default 14)

    Returns:
        List of ATR values (first period values are 0.0)
    """
    n = len(candles)
    result = [0.0] * n

    if n < 2:
        return result

    # Calculate True Range
    tr = [0.0] * n
    tr[0] = candles[0].high - candles[0].low  # First candle TR

    for i in range(1, n):
        high = candles[i].high
        low = candles[i].low
        prev_close = candles[i - 1].close

        tr[i] = max(
            high - low,
            abs(high - prev_close),
            abs(low - prev_close)
        )

    if n < period:
        return result

    # Initial ATR (SMA of TR)
    result[period - 1] = sum(tr[:period]) / period

    # Smoothed ATR (Wilder's smoothing)
    for i in range(period, n):
        result[i] = (result[i - 1] * (period - 1) + tr[i]) / period

    return result


def atr_from_arrays(highs: List[float], lows: List[float],
                    closes: List[float], period: int = 14) -> List[float]:
    """
    ATR from separate arrays.

    Args:
        highs: High prices
        lows: Low prices
        closes: Close prices
        period: ATR period

    Returns:
        List of ATR values
    """
    n = len(closes)
    result = [0.0] * n

    if n < 2:
        return result

    # Calculate True Range
    tr = [0.0] * n
    tr[0] = highs[0] - lows[0]

    for i in range(1, n):
        tr[i] = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1])
        )

    if n < period:
        return result

    # Initial ATR (SMA of TR)
    result[period - 1] = sum(tr[:period]) / period

    # Smoothed ATR
    for i in range(period, n):
        result[i] = (result[i - 1] * (period - 1) + tr[i]) / period

    return result


def candle_anatomy(candle: Candle) -> dict:
    """
    Compute candle anatomy metrics.

    Args:
        candle: OHLCV candle

    Returns:
        Dict with body, range, wicks, ratios
    """
    body = abs(candle.close - candle.open)
    range_ = max(candle.high - candle.low, 1e-10)
    lower_wick = min(candle.open, candle.close) - candle.low
    upper_wick = candle.high - max(candle.open, candle.close)

    return {
        'body': body,
        'range': range_,
        'lower_wick': lower_wick,
        'upper_wick': upper_wick,
        'lower_wick_ratio': lower_wick / range_,
        'upper_wick_ratio': upper_wick / range_,
        'body_ratio': body / range_,
        'is_bullish': candle.close >= candle.open,
    }


def bodies(candles: List[Candle]) -> List[float]:
    """Extract body sizes from candles."""
    return [c.body for c in candles]


def ranges(candles: List[Candle]) -> List[float]:
    """Extract ranges from candles."""
    return [c.range for c in candles]


def closes(candles: List[Candle]) -> List[float]:
    """Extract close prices from candles."""
    return [c.close for c in candles]


def volumes(candles: List[Candle]) -> List[float]:
    """Extract volumes from candles."""
    return [c.volume for c in candles]


def find_swing_low(candles: List[Candle], lookback: int = 3,
                   current_idx: Optional[int] = None) -> Optional[float]:
    """
    Find most recent swing low.

    Args:
        candles: List of candles
        lookback: Number of candles to consider
        current_idx: Current index (default: last candle)

    Returns:
        Swing low price or None if not found
    """
    if current_idx is None:
        current_idx = len(candles) - 1

    if current_idx < lookback:
        return None

    start = max(0, current_idx - lookback)
    lows = [c.low for c in candles[start:current_idx + 1]]

    if not lows:
        return None

    return min(lows)


def find_swing_high(candles: List[Candle], lookback: int = 3,
                    current_idx: Optional[int] = None) -> Optional[float]:
    """
    Find most recent swing high.

    Args:
        candles: List of candles
        lookback: Number of candles to consider
        current_idx: Current index (default: last candle)

    Returns:
        Swing high price or None if not found
    """
    if current_idx is None:
        current_idx = len(candles) - 1

    if current_idx < lookback:
        return None

    start = max(0, current_idx - lookback)
    highs = [c.high for c in candles[start:current_idx + 1]]

    if not highs:
        return None

    return max(highs)


def detect_divergence(prices: List[float], rsi_values: List[float],
                      lookback: int = 5, bullish: bool = True) -> bool:
    """
    Detect RSI divergence.

    Args:
        prices: Price values (typically lows for bullish, highs for bearish)
        rsi_values: RSI values
        lookback: Number of bars to check
        bullish: True for bullish divergence (lower lows, higher RSI)

    Returns:
        True if divergence detected
    """
    if len(prices) < lookback + 1 or len(rsi_values) < lookback + 1:
        return False

    current_idx = len(prices) - 1
    start_idx = current_idx - lookback

    if bullish:
        # Bullish: price makes lower low, RSI makes higher low
        price_lower = prices[current_idx] < prices[start_idx]
        rsi_higher = rsi_values[current_idx] > rsi_values[start_idx]
        return price_lower and rsi_higher
    else:
        # Bearish: price makes higher high, RSI makes lower high
        price_higher = prices[current_idx] > prices[start_idx]
        rsi_lower = rsi_values[current_idx] < rsi_values[start_idx]
        return price_higher and rsi_lower


def is_rsi_turning_up(rsi_values: List[float], lookback: int = 2) -> bool:
    """Check if RSI is turning up (rising from recent values)."""
    if len(rsi_values) < lookback + 1:
        return False

    current = rsi_values[-1]
    prev = rsi_values[-2]

    return current > prev


def is_rsi_turning_down(rsi_values: List[float], lookback: int = 2) -> bool:
    """Check if RSI is turning down (falling from recent values)."""
    if len(rsi_values) < lookback + 1:
        return False

    current = rsi_values[-1]
    prev = rsi_values[-2]

    return current < prev


def mean(values: List[float]) -> float:
    """Calculate mean of values, returns 0 for empty list."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def safe_divide(numerator: float, denominator: float,
                default: float = 0.0) -> float:
    """Safe division avoiding zero division."""
    if denominator == 0 or abs(denominator) < 1e-10:
        return default
    return numerator / denominator
