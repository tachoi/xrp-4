"""Feature specification for XRP4.

Defines the exact feature columns that will be generated.
All feature names include the base timeframe suffix (_3m).

IMPORTANT: These feature names are the single source of truth.
Do not change them without updating all dependent code.
"""

from typing import List, Set

# Base timeframe for xrp-4
BASE_TIMEFRAME = "3m"

# Timeframe suffix for all features
_TF_SUFFIX = f"_{BASE_TIMEFRAME}"

# =============================================================================
# Feature Column Definitions
# =============================================================================

# Return features
RETURN_FEATURES = [
    f"ret{_TF_SUFFIX}",  # Simple return
    f"logret{_TF_SUFFIX}",  # Log return
    f"ewm_ret{_TF_SUFFIX}",  # EWM smoothed return
]

# Volatility features
VOLATILITY_FEATURES = [
    f"atr_pct{_TF_SUFFIX}",  # ATR as percentage of price
    f"ewm_std_ret{_TF_SUFFIX}",  # EWM standard deviation of returns
    f"bb_width{_TF_SUFFIX}",  # Bollinger Band width
]

# Candlestick pattern features
CANDLE_FEATURES = [
    "body_ratio",  # |close-open| / (high-low)
    "upper_wick_ratio",  # Upper wick / (high-low)
    "lower_wick_ratio",  # Lower wick / (high-low)
]

# Volume features
VOLUME_FEATURES = [
    f"vol_z{_TF_SUFFIX}",  # Volume Z-score
]

# Gate features (3m) - Required for Gate Layer
GATE_3M_FEATURES = [
    f"ret_abs_z{_TF_SUFFIX}",              # Absolute return Z-score
    f"vol_ratio{_TF_SUFFIX}",              # Volume ratio (current / average)
    f"adx{_TF_SUFFIX}",                    # Average Directional Index
    f"directional_consistency{_TF_SUFFIX}",  # Directional consistency
    f"chop{_TF_SUFFIX}",                   # Choppiness Index
    f"trend_to_range_ratio{_TF_SUFFIX}",   # Trend-to-range ratio
]

# Gate features (15m) - Multi-timeframe context
GATE_15M_FEATURES = [
    "chop_15m",                      # Choppiness Index (15m)
    "vol_ratio_15m",                 # Volume ratio (15m)
    "directional_consistency_15m",   # Directional consistency (15m)
]

# Gate features (1h) - Multi-timeframe context
GATE_1H_FEATURES = [
    "ret_abs_z_1h",    # Absolute return Z-score (1h)
    "vol_ratio_1h",    # Volume ratio (1h)
    "ema_slope_1h",    # EMA slope (1h)
]

# =============================================================================
# Complete Feature List
# =============================================================================

# All feature columns that will be generated (base 3m features)
FEATURE_COLUMNS: List[str] = (
    RETURN_FEATURES
    + VOLATILITY_FEATURES
    + CANDLE_FEATURES
    + VOLUME_FEATURES
    + GATE_3M_FEATURES
)

# All features including multi-timeframe (optional)
FEATURE_COLUMNS_WITH_MULTITF: List[str] = (
    FEATURE_COLUMNS
    + GATE_15M_FEATURES
    + GATE_1H_FEATURES
)

# Feature columns as a frozen set for validation
FEATURE_COLUMNS_SET: Set[str] = frozenset(FEATURE_COLUMNS)

# Minimum required columns from source data
REQUIRED_SOURCE_COLUMNS = frozenset([
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
])


def get_feature_names() -> List[str]:
    """Get the list of feature column names.

    Returns:
        List of feature column names
    """
    return FEATURE_COLUMNS.copy()


def validate_feature_columns(df_columns: List[str]) -> List[str]:
    """Validate that all required feature columns are present.

    Args:
        df_columns: List of column names in DataFrame

    Returns:
        List of missing feature columns (empty if all present)
    """
    df_cols_set = set(df_columns)
    missing = [col for col in FEATURE_COLUMNS if col not in df_cols_set]
    return missing


def get_feature_groups() -> dict:
    """Get features grouped by category.

    Returns:
        Dictionary mapping category names to feature lists
    """
    return {
        "returns": RETURN_FEATURES.copy(),
        "volatility": VOLATILITY_FEATURES.copy(),
        "candle": CANDLE_FEATURES.copy(),
        "volume": VOLUME_FEATURES.copy(),
        "gate_3m": GATE_3M_FEATURES.copy(),
        "gate_15m": GATE_15M_FEATURES.copy(),
        "gate_1h": GATE_1H_FEATURES.copy(),
    }
