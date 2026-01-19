"""
Vectorized backtesting for fast parameter optimization.
Uses numpy/pandas for batch processing instead of sequential candle-by-candle.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List, Optional
from dataclasses import dataclass

from .config import CoreConfig


@dataclass
class VectorizedResult:
    """Result of vectorized backtest."""
    total_pnl_pct: float
    win_rate: float
    total_trades: int
    wins: int
    losses: int
    anchors_detected: int
    entries_long: int
    entries_short: int
    avg_trade_pnl: float
    max_drawdown_pct: float
    profit_factor: float


def compute_indicators_vectorized(df: pd.DataFrame, config: CoreConfig) -> pd.DataFrame:
    """
    Compute all indicators in vectorized manner.

    Args:
        df: DataFrame with columns [timestamp, open, high, low, close, volume]
        config: Trading configuration

    Returns:
        DataFrame with added indicator columns
    """
    df = df.copy()

    # Convert Decimal to float if needed
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            df[col] = df[col].astype(float)

    # EMA
    df['ema20'] = df['close'].ewm(span=config.fsm.ema_period, adjust=False).mean()

    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/config.fsm.rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/config.fsm.rsi_period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.inf)
    df['rsi14'] = 100 - (100 / (1 + rs))
    df['rsi14'] = df['rsi14'].fillna(50)

    # ATR
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr14'] = tr.ewm(alpha=1/config.fsm.atr_period, adjust=False).mean()

    # Volume SMA
    df['vol_sma'] = df['volume'].rolling(window=config.fsm.anchor_vol_sma_n, min_periods=1).mean()

    # Candle anatomy
    df['body'] = (df['close'] - df['open']).abs()
    df['range'] = df['high'] - df['low']
    df['range'] = df['range'].replace(0, 1e-10)
    df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
    df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['lower_wick_ratio'] = df['lower_wick'] / df['range']
    df['is_bullish'] = df['close'] >= df['open']

    # Body and range means for phenomena
    k = config.phenomena.lookback_k
    df['body_mean'] = df['body'].rolling(window=k, min_periods=1).mean().shift(1)
    df['range_mean'] = df['range'].rolling(window=k, min_periods=1).mean().shift(1)

    # EMA distance
    df['ema_dist'] = (df['close'] - df['ema20']).abs() / df['atr14'].replace(0, np.inf)

    # RSI turning
    df['rsi_prev'] = df['rsi14'].shift(1)
    df['rsi_turning_up'] = df['rsi14'] > df['rsi_prev']
    df['rsi_turning_down'] = df['rsi14'] < df['rsi_prev']

    return df


def detect_anchors_vectorized(df: pd.DataFrame, config: CoreConfig) -> pd.Series:
    """
    Detect anchor candles vectorized.

    Returns:
        Series with anchor direction (+1 bullish, -1 bearish, 0 no anchor)
    """
    vol_mult = df['volume'] / df['vol_sma'].replace(0, np.inf)
    body_atr_mult = df['body'] / df['atr14'].replace(0, np.inf)

    # Anchor conditions
    vol_ok = vol_mult >= config.fsm.anchor_vol_mult
    body_ok = body_atr_mult >= config.fsm.anchor_body_atr_mult
    chase_ok = df['ema_dist'] <= config.fsm.chase_dist_max_atr

    is_anchor = vol_ok & body_ok & chase_ok

    # Direction
    direction = np.where(df['is_bullish'], 1, -1)

    return pd.Series(np.where(is_anchor, direction, 0), index=df.index)


def check_phenomena_vectorized(df: pd.DataFrame, config: CoreConfig) -> pd.Series:
    """
    Check phenomena conditions vectorized.

    Returns:
        Series with count of satisfied conditions
    """
    # A) Low fail: min(low[t], low[t-1]) >= min(low[t-2], low[t-3])
    min_recent = df['low'].rolling(2).min()
    min_older = df['low'].shift(2).rolling(2).min()
    low_fail = min_recent >= min_older

    # B) Lower wick ratio
    wick_ok = df['lower_wick_ratio'] >= config.phenomena.lower_wick_ratio_min

    # C) Body shrink
    body_shrink = df['body'] <= config.phenomena.body_shrink_factor * df['body_mean']

    # D) Range shrink
    range_shrink = df['range'] <= config.phenomena.range_shrink_factor * df['range_mean']

    # Count conditions
    count = low_fail.astype(int) + wick_ok.astype(int) + body_shrink.astype(int) + range_shrink.astype(int)

    return count.fillna(0)


def detect_zones_simple(df_15m: pd.DataFrame, config: CoreConfig) -> pd.DataFrame:
    """
    Simple zone detection using pivot points.
    Returns DataFrame with zone levels.
    """
    L = config.zone.pivot_lookback

    # Pivot highs
    df_15m['pivot_high'] = df_15m['high'][
        (df_15m['high'] == df_15m['high'].rolling(2*L+1, center=True).max())
    ]

    # Pivot lows
    df_15m['pivot_low'] = df_15m['low'][
        (df_15m['low'] == df_15m['low'].rolling(2*L+1, center=True).min())
    ]

    # Get zone levels (last N pivots)
    resistance_levels = df_15m['pivot_high'].dropna().tail(config.zone.max_per_side).values
    support_levels = df_15m['pivot_low'].dropna().tail(config.zone.max_per_side).values

    return resistance_levels, support_levels


def check_in_zone(price: float, levels: np.ndarray, radius: float) -> Tuple[bool, bool]:
    """
    Check if price is in CORE or NEAR zone.

    Returns:
        (is_core, is_near)
    """
    if len(levels) == 0:
        return False, False

    distances = np.abs(levels - price)
    min_dist = distances.min()

    is_core = min_dist <= radius
    is_near = min_dist <= radius * 1.5  # pad

    return is_core, is_near


def run_vectorized_backtest(df_3m: pd.DataFrame,
                            df_15m: pd.DataFrame,
                            config: CoreConfig) -> VectorizedResult:
    """
    Run vectorized backtest.

    This is a simplified but fast implementation that:
    1. Pre-computes all indicators
    2. Detects anchors in batch
    3. Checks phenomena in batch
    4. Simulates trades with simplified entry/exit logic
    """
    # Compute indicators
    df = compute_indicators_vectorized(df_3m, config)

    # Detect zones from 15m data
    if len(df_15m) > 0:
        df_15m_ind = compute_indicators_vectorized(df_15m, config)
        atr_15m = df_15m_ind['atr14'].iloc[-1] if len(df_15m_ind) > 0 else df['atr14'].mean()
        resistance_levels, support_levels = detect_zones_simple(df_15m_ind, config)
        zone_radius = atr_15m * config.zone.radius_multiplier
    else:
        resistance_levels = np.array([])
        support_levels = np.array([])
        zone_radius = df['atr14'].mean() * config.zone.radius_multiplier

    all_zones = np.concatenate([resistance_levels, support_levels]) if len(resistance_levels) > 0 or len(support_levels) > 0 else np.array([])

    # Detect anchors
    anchor_dirs = detect_anchors_vectorized(df, config)

    # Check phenomena
    phenomena_count = check_phenomena_vectorized(df, config)
    phenomena_ok = phenomena_count >= config.phenomena.requirements_min_count

    # Simulation state
    trades = []
    in_trade = False
    entry_price = 0.0
    entry_idx = 0
    trade_dir = 0  # +1 long, -1 short
    stop_price = 0.0
    anchor_idx = -100
    anchor_mid = 0.0

    n = len(df)
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    atrs = df['atr14'].values
    rsi_up = df['rsi_turning_up'].values
    rsi_down = df['rsi_turning_down'].values
    rsis = df['rsi14'].values

    # Count anchors in zones
    anchors_in_zone = 0

    for i in range(50, n):  # Skip warmup period
        price = closes[i]

        # Check zone
        is_core, is_near = check_in_zone(price, all_zones, zone_radius)

        if in_trade:
            # Check exit conditions
            bars_in_trade = i - entry_idx

            # Hard stop
            if trade_dir == 1 and lows[i] <= stop_price:
                pnl = (stop_price - entry_price) / entry_price * 100
                trades.append(pnl)
                in_trade = False
                continue
            elif trade_dir == -1 and highs[i] >= stop_price:
                pnl = (entry_price - stop_price) / entry_price * 100
                trades.append(pnl)
                in_trade = False
                continue

            # Time stop
            if bars_in_trade >= config.fsm.hold_max_candles:
                if trade_dir == 1:
                    pnl = (price - entry_price) / entry_price * 100
                else:
                    pnl = (entry_price - price) / entry_price * 100
                trades.append(pnl)
                in_trade = False
                continue

            # RSI exit
            if trade_dir == 1 and rsis[i] >= 70 and rsi_down[i]:
                pnl = (price - entry_price) / entry_price * 100
                trades.append(pnl)
                in_trade = False
                continue
            elif trade_dir == -1 and rsis[i] <= 30 and rsi_up[i]:
                pnl = (entry_price - price) / entry_price * 100
                trades.append(pnl)
                in_trade = False
                continue

        else:
            # Not in trade - look for entry
            anchor = anchor_dirs.iloc[i]

            if anchor != 0 and (is_core or is_near):
                anchors_in_zone += 1
                anchor_idx = i
                anchor_mid = (df['open'].iloc[i] + df['close'].iloc[i]) / 2
                trade_dir = anchor

            # Check for entry after anchor
            elif anchor_idx >= 0 and (i - anchor_idx) <= config.fsm.anchor_expire_candles:
                # Check pullback to anchor_mid
                pullback_tol = config.fsm.pullback_tolerance_atr * atrs[i]
                near_mid = abs(price - anchor_mid) <= pullback_tol

                # Entry: pullback near anchor mid + phenomena ok (zone already checked at anchor)
                if near_mid and phenomena_ok.iloc[i]:
                    # Check RSI confirmation
                    rsi_confirmed = (trade_dir == 1 and rsi_up[i]) or (trade_dir == -1 and rsi_down[i])

                    if rsi_confirmed:
                        # Entry trigger - simplified: just enter on confirmation
                        in_trade = True
                        entry_price = price
                        entry_idx = i

                        # Set stop
                        if trade_dir == 1:
                            stop_price = min(lows[max(0,i-5):i+1]) if i >= 5 else lows[i]
                        else:
                            stop_price = max(highs[max(0,i-5):i+1]) if i >= 5 else highs[i]

                        anchor_idx = -100  # Reset

    # Calculate results
    if len(trades) == 0:
        return VectorizedResult(
            total_pnl_pct=0.0,
            win_rate=0.0,
            total_trades=0,
            wins=0,
            losses=0,
            anchors_detected=anchors_in_zone,
            entries_long=0,
            entries_short=0,
            avg_trade_pnl=0.0,
            max_drawdown_pct=0.0,
            profit_factor=0.0,
        )

    trades_arr = np.array(trades)
    wins = np.sum(trades_arr > 0)
    losses = np.sum(trades_arr <= 0)
    total_pnl = np.sum(trades_arr)

    # Max drawdown
    cumsum = np.cumsum(trades_arr)
    running_max = np.maximum.accumulate(cumsum)
    drawdown = running_max - cumsum
    max_dd = np.max(drawdown) if len(drawdown) > 0 else 0

    # Profit factor
    gross_profit = np.sum(trades_arr[trades_arr > 0])
    gross_loss = abs(np.sum(trades_arr[trades_arr <= 0]))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (2.0 if gross_profit > 0 else 0.0)

    return VectorizedResult(
        total_pnl_pct=total_pnl,
        win_rate=wins / len(trades) * 100 if len(trades) > 0 else 0,
        total_trades=len(trades),
        wins=int(wins),
        losses=int(losses),
        anchors_detected=anchors_in_zone,
        entries_long=0,  # Not tracked in simplified version
        entries_short=0,
        avg_trade_pnl=total_pnl / len(trades) if len(trades) > 0 else 0,
        max_drawdown_pct=max_dd,
        profit_factor=profit_factor,
    )
