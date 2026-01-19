#!/usr/bin/env python
"""Advanced strategy tuning with multiple entry/exit strategies.

Tests different trading strategies:
1. Pullback strategy (original)
2. Momentum breakout strategy
3. Regime change strategy
4. Conservative filtered strategy

Usage:
    python scripts/tune_strategy.py
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd
from hmmlearn import hmm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from xrp4.regime.confirm import RegimeConfirmLayer, ConfirmConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


class EntryStrategy(Enum):
    PULLBACK = "pullback"           # Enter on pullback in trend direction
    MOMENTUM = "momentum"           # Enter on momentum confirmation
    BREAKOUT = "breakout"           # Enter on breakout from range
    REGIME_CHANGE = "regime_change" # Enter when regime changes
    CONSERVATIVE = "conservative"   # Only high-confidence setups


@dataclass
class StrategyParams:
    """Strategy parameters."""
    entry_strategy: EntryStrategy = EntryStrategy.PULLBACK
    min_confidence: float = 0.7

    # Position sizing
    risk_per_trade: float = 0.01
    fee_rate: float = 0.001

    # Trend params
    sl_mult_trend: float = 2.0
    tp_mult_trend: float = 4.0

    # Range params
    sl_mult_range: float = 1.5
    tp_mult_range: float = 2.0

    # Momentum filter
    momentum_threshold: float = 0.0  # Min absolute ewm_ret for entry

    # Volume filter
    volume_z_threshold: float = 0.0  # Min volume z-score

    # Trend strength filter
    trend_strength_threshold: float = 0.0  # Min abs(ema_slope)

    # Time exit (max bars in trade)
    max_hold_bars: int = 0  # 0 = disabled

    # Trailing stop
    use_trailing_stop: bool = False
    trailing_atr_mult: float = 2.0


class SingleHMM:
    """Single HMM model for regime detection."""

    def __init__(
        self,
        n_states: int = 5,
        covariance_type: str = "full",
        n_iter: int = 200,
        random_state: int = 42,
    ):
        self.n_states = n_states
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.random_state = random_state
        self.model: Optional[hmm.GaussianHMM] = None
        self.feature_names: List[str] = []
        self.state_labels: Dict[int, str] = {}
        self._is_trained = False
        self._feature_mean: Optional[np.ndarray] = None
        self._feature_std: Optional[np.ndarray] = None

    def _create_model(self) -> hmm.GaussianHMM:
        return hmm.GaussianHMM(
            n_components=self.n_states,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=self.random_state,
            verbose=False,
        )

    def train(self, features: np.ndarray, feature_names: List[str]) -> float:
        n_samples, n_features = features.shape

        if np.any(np.isnan(features)):
            features = features.copy()
            col_means = np.nanmean(features, axis=0)
            for i in range(n_features):
                mask = np.isnan(features[:, i])
                features[mask, i] = col_means[i]

        self._feature_mean = np.mean(features, axis=0)
        self._feature_std = np.std(features, axis=0)
        self._feature_std[self._feature_std == 0] = 1.0
        features_scaled = (features - self._feature_mean) / self._feature_std

        self.model = self._create_model()
        self.model.fit(features_scaled)
        self.feature_names = feature_names
        log_likelihood = self.model.score(features_scaled)

        self._map_states_to_labels(features_scaled)
        self._is_trained = True
        return log_likelihood

    def _map_states_to_labels(self, features: np.ndarray) -> None:
        if self.model is None:
            return

        means = self.model.means_

        if self.covariance_type == "full":
            variances = np.array([np.trace(c) for c in self.model.covars_])
        elif self.covariance_type == "diag":
            variances = np.sum(self.model.covars_, axis=1)
        else:
            variances = np.ones(self.n_states)

        var_order = np.argsort(variances)

        return_idx = 0
        for i, name in enumerate(self.feature_names):
            if name == "ret_15m":
                return_idx = i
                break

        state_returns = means[:, return_idx]

        self.state_labels = {}
        self.state_labels[var_order[0]] = "RANGE"
        self.state_labels[var_order[-1]] = "TRANSITION"
        self.state_labels[var_order[-2]] = "HIGH_VOL"

        remaining = list(var_order[1:-2])
        if len(remaining) >= 2:
            ret_sorted = sorted(remaining, key=lambda s: state_returns[s])
            self.state_labels[ret_sorted[0]] = "TREND_DOWN"
            self.state_labels[ret_sorted[-1]] = "TREND_UP"
        elif len(remaining) == 1:
            s = remaining[0]
            self.state_labels[s] = "TREND_UP" if state_returns[s] > 0 else "TREND_DOWN"

    def predict_sequence(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self._is_trained:
            raise RuntimeError("Model not trained")

        if np.any(np.isnan(features)):
            features = features.copy()
            for i in range(features.shape[1]):
                mask = np.isnan(features[:, i])
                features[mask, i] = self._feature_mean[i]

        features_scaled = (features - self._feature_mean) / self._feature_std
        _, state_seq = self.model.decode(features_scaled, algorithm="viterbi")
        state_probs = self.model.predict_proba(features_scaled)
        labels = np.array([self.state_labels.get(s, "UNKNOWN") for s in state_seq])
        return state_seq, state_probs, labels


def resample_to_15m(df_3m: pd.DataFrame) -> pd.DataFrame:
    df = df_3m.copy()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")

    resampled = df.resample("15min").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum",
    })
    resampled = resampled.dropna(subset=["close"])
    return resampled.reset_index()


def _resample_to_1h(df_3m: pd.DataFrame) -> pd.DataFrame:
    df = df_3m.copy()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")

    resampled = df.resample("1h").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum",
    })
    resampled = resampled.dropna(subset=["close"])
    return resampled.reset_index()


def build_features(df_3m: pd.DataFrame) -> Tuple[np.ndarray, List[str], pd.DataFrame]:
    """Build multi-timeframe features for HMM."""
    df_15m = resample_to_15m(df_3m)
    df_1h = _resample_to_1h(df_3m)

    close_15m = df_15m["close"].astype(float)
    high_15m = df_15m["high"].astype(float)
    low_15m = df_15m["low"].astype(float)
    open_15m = df_15m["open"].astype(float)
    volume_15m = df_15m["volume"].astype(float)

    df_15m["ret_15m"] = close_15m.pct_change()
    df_15m["ewm_ret_15m"] = df_15m["ret_15m"].ewm(span=20, adjust=False).mean()

    ema_20 = close_15m.ewm(span=20, adjust=False).mean()
    df_15m["ema_slope_15m"] = (ema_20 - ema_20.shift(1)) / close_15m * 100

    tr_15m = pd.concat([
        high_15m - low_15m,
        (high_15m - close_15m.shift(1)).abs(),
        (low_15m - close_15m.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr_15m = tr_15m.ewm(alpha=1/14, adjust=False).mean()
    df_15m["atr_pct_15m"] = atr_15m / close_15m * 100

    df_15m["ewm_std_ret_15m"] = df_15m["ret_15m"].ewm(span=20, adjust=False).std()

    sma_15m = close_15m.rolling(20).mean()
    std_15m = close_15m.rolling(20).std()
    df_15m["bb_width_15m"] = (2 * std_15m / sma_15m) * 100

    df_15m["bb_width_15m_pct"] = df_15m["bb_width_15m"].rolling(50).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if x.max() != x.min() else 0.5
    )

    rolling_range = (high_15m.rolling(20).max() - low_15m.rolling(20).min())
    recent_range = high_15m - low_15m
    df_15m["range_comp_15m"] = 1 - (recent_range / rolling_range.replace(0, np.nan))
    df_15m["range_comp_15m"] = df_15m["range_comp_15m"].clip(0, 1)

    total_range = (high_15m - low_15m).replace(0, np.nan)
    body = (close_15m - open_15m).abs()
    df_15m["body_ratio"] = body / total_range

    upper_wick = high_15m - pd.concat([open_15m, close_15m], axis=1).max(axis=1)
    df_15m["upper_wick_ratio"] = upper_wick / total_range

    lower_wick = pd.concat([open_15m, close_15m], axis=1).min(axis=1) - low_15m
    df_15m["lower_wick_ratio"] = lower_wick / total_range

    vol_mean = volume_15m.rolling(20).mean()
    vol_std = volume_15m.rolling(20).std().replace(0, np.nan)
    df_15m["vol_z_15m"] = (volume_15m - vol_mean) / vol_std

    # Rolling high/low for breakout detection
    df_15m["rolling_high_20"] = high_15m.rolling(20).max()
    df_15m["rolling_low_20"] = low_15m.rolling(20).min()

    close_1h = df_1h["close"].astype(float)
    high_1h = df_1h["high"].astype(float)
    low_1h = df_1h["low"].astype(float)

    df_1h["ret_1h"] = close_1h.pct_change()
    df_1h["ewm_ret_1h"] = df_1h["ret_1h"].ewm(span=20, adjust=False).mean()

    ema_20_1h = close_1h.ewm(span=20, adjust=False).mean()
    ema_std_1h = (close_1h - ema_20_1h).rolling(20).std().replace(0, np.nan)
    df_1h["price_z_from_ema_1h"] = (close_1h - ema_20_1h) / ema_std_1h

    tr_1h = pd.concat([
        high_1h - low_1h,
        (high_1h - close_1h.shift(1)).abs(),
        (low_1h - close_1h.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr_1h = tr_1h.ewm(alpha=1/14, adjust=False).mean()
    df_1h["atr_pct_1h"] = atr_1h / close_1h * 100
    df_1h["ewm_std_ret_1h"] = df_1h["ret_1h"].ewm(span=20, adjust=False).std()

    df_15m = df_15m.set_index("timestamp")
    df_1h = df_1h.set_index("timestamp")

    features_1h = df_1h[["ret_1h", "ewm_ret_1h", "price_z_from_ema_1h", "atr_pct_1h", "ewm_std_ret_1h"]]
    df_result = df_15m.join(features_1h, how="left")
    df_result[["ret_1h", "ewm_ret_1h", "price_z_from_ema_1h", "atr_pct_1h", "ewm_std_ret_1h"]] = \
        df_result[["ret_1h", "ewm_ret_1h", "price_z_from_ema_1h", "atr_pct_1h", "ewm_std_ret_1h"]].ffill()
    df_result = df_result.reset_index()

    hmm_features = [
        "ret_15m", "ret_1h", "ewm_ret_15m", "ewm_ret_1h",
        "ema_slope_15m", "price_z_from_ema_1h",
        "atr_pct_15m", "atr_pct_1h", "ewm_std_ret_15m", "ewm_std_ret_1h",
        "bb_width_15m", "range_comp_15m", "bb_width_15m_pct",
        "body_ratio", "upper_wick_ratio", "lower_wick_ratio", "vol_z_15m",
    ]

    df_clean = df_result.dropna(subset=hmm_features)
    feature_matrix = df_clean[hmm_features].values

    return feature_matrix, hmm_features, df_clean


def run_strategy_backtest(
    df: pd.DataFrame,
    hmm_model: SingleHMM,
    features: np.ndarray,
    params: StrategyParams,
    initial_capital: float = 10000.0,
) -> Dict:
    """Run backtest with specific strategy."""

    # Create Confirm Layer
    confirm_cfg = ConfirmConfig(
        HIGH_VOL_LAMBDA_ON=2.0,
        HIGH_VOL_COOLDOWN_BARS_15M=6,
    )

    state_seq, state_probs, labels = hmm_model.predict_sequence(features)

    df_bt = df.copy()
    df_bt["regime_raw"] = labels
    df_bt["confidence"] = [probs.max() for probs in state_probs]

    close = df_bt["close"].astype(float)
    high = df_bt["high"].astype(float)
    low = df_bt["low"].astype(float)

    tr = pd.concat([
        high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False).mean()
    df_bt["atr"] = atr

    # Apply Confirm Layer
    confirm_layer = RegimeConfirmLayer(confirm_cfg)
    confirm_state = None

    confirmed_regimes = []
    for idx in range(len(df_bt)):
        row = df_bt.iloc[idx].to_dict()
        regime_raw = row["regime_raw"]

        if idx >= 32:
            hist = df_bt.iloc[max(0, idx-96):idx+1]
            result, confirm_state = confirm_layer.confirm(
                regime_raw=regime_raw, row_15m=row, hist_15m=hist, state=confirm_state,
            )
            confirmed_regimes.append(result.confirmed_regime)
        else:
            confirmed_regimes.append(regime_raw)

    df_bt["regime"] = confirmed_regimes

    # Backtest
    equity = initial_capital
    position = None
    trades = []
    equity_curve = [equity]

    for idx in range(3, len(df_bt)):
        bar = df_bt.iloc[idx]
        prev_bar = df_bt.iloc[idx - 1]
        prev2_bar = df_bt.iloc[idx - 2]
        prev3_bar = df_bt.iloc[idx - 3]

        regime = bar["regime"]
        prev_regime = prev_bar["regime"]
        conf = bar["confidence"]
        current_atr = bar["atr"]

        if pd.isna(current_atr) or current_atr <= 0:
            equity_curve.append(equity)
            continue

        # Position management
        if position is not None:
            hit_sl = False
            hit_tp = False
            exit_price = None
            bars_held = idx - position["entry_idx"]

            # Time-based exit
            if params.max_hold_bars > 0 and bars_held >= params.max_hold_bars:
                exit_price = float(bar["close"])
                exit_reason = "time_exit"
            else:
                # Trailing stop update
                if params.use_trailing_stop:
                    if position["side"] == "long":
                        new_trail_sl = bar["high"] - current_atr * params.trailing_atr_mult
                        position["sl"] = max(position["sl"], new_trail_sl)
                    else:
                        new_trail_sl = bar["low"] + current_atr * params.trailing_atr_mult
                        position["sl"] = min(position["sl"], new_trail_sl)

                # Check SL/TP
                if position["side"] == "long":
                    if bar["low"] <= position["sl"]:
                        hit_sl = True
                        exit_price = position["sl"]
                    elif bar["high"] >= position["tp"]:
                        hit_tp = True
                        exit_price = position["tp"]
                else:
                    if bar["high"] >= position["sl"]:
                        hit_sl = True
                        exit_price = position["sl"]
                    elif bar["low"] <= position["tp"]:
                        hit_tp = True
                        exit_price = position["tp"]

                if hit_sl:
                    exit_reason = "sl"
                elif hit_tp:
                    exit_reason = "tp"

            if exit_price is not None:
                if position["side"] == "long":
                    pnl = (exit_price - position["entry"]) * position["size"]
                else:
                    pnl = (position["entry"] - exit_price) * position["size"]

                fee = exit_price * position["size"] * params.fee_rate
                pnl -= fee
                equity += pnl

                trades.append({
                    "pnl": pnl,
                    "pnl_pct": pnl / position["entry_equity"] * 100,
                    "exit_reason": exit_reason,
                    "regime": position["regime"],
                    "bars_held": bars_held,
                })
                position = None

        # Entry signals based on strategy
        if position is None and conf >= params.min_confidence and equity > 0:
            signal = None
            sl_mult = 2.0
            tp_mult = 3.0

            # Get feature values
            ewm_ret = bar.get("ewm_ret_15m", 0) if "ewm_ret_15m" in bar else 0
            vol_z = bar.get("vol_z_15m", 0) if "vol_z_15m" in bar else 0
            ema_slope = bar.get("ema_slope_15m", 0) if "ema_slope_15m" in bar else 0

            # Apply filters
            passes_momentum = abs(ewm_ret) >= params.momentum_threshold
            passes_volume = vol_z >= params.volume_z_threshold
            passes_trend_strength = abs(ema_slope) >= params.trend_strength_threshold

            if params.entry_strategy == EntryStrategy.PULLBACK:
                # Original pullback strategy
                if regime == "TREND_UP" and passes_momentum:
                    if prev_bar["close"] < prev2_bar["close"]:  # Pullback
                        signal = "long"
                        sl_mult = params.sl_mult_trend
                        tp_mult = params.tp_mult_trend

                elif regime == "TREND_DOWN" and passes_momentum:
                    if prev_bar["close"] > prev2_bar["close"]:  # Bounce
                        signal = "short"
                        sl_mult = params.sl_mult_trend
                        tp_mult = params.tp_mult_trend

                elif regime == "RANGE":
                    roll_high = high.iloc[max(0, idx-20):idx].max()
                    roll_low = low.iloc[max(0, idx-20):idx].min()
                    mid_price = (roll_high + roll_low) / 2

                    if bar["close"] < mid_price and prev_bar["close"] < prev2_bar["close"]:
                        signal = "long"
                        sl_mult = params.sl_mult_range
                        tp_mult = params.tp_mult_range
                    elif bar["close"] > mid_price and prev_bar["close"] > prev2_bar["close"]:
                        signal = "short"
                        sl_mult = params.sl_mult_range
                        tp_mult = params.tp_mult_range

            elif params.entry_strategy == EntryStrategy.MOMENTUM:
                # Enter on momentum continuation
                if regime == "TREND_UP" and passes_trend_strength:
                    # Enter when momentum is accelerating
                    if ewm_ret > 0 and bar["close"] > prev_bar["close"] > prev2_bar["close"]:
                        signal = "long"
                        sl_mult = params.sl_mult_trend
                        tp_mult = params.tp_mult_trend

                elif regime == "TREND_DOWN" and passes_trend_strength:
                    if ewm_ret < 0 and bar["close"] < prev_bar["close"] < prev2_bar["close"]:
                        signal = "short"
                        sl_mult = params.sl_mult_trend
                        tp_mult = params.tp_mult_trend

            elif params.entry_strategy == EntryStrategy.BREAKOUT:
                # Enter on range breakout
                if "rolling_high_20" in bar and "rolling_low_20" in bar:
                    roll_high = bar["rolling_high_20"]
                    roll_low = bar["rolling_low_20"]

                    if regime in ["TREND_UP", "TRANSITION"]:
                        if bar["close"] > roll_high and prev_bar["close"] <= roll_high:
                            signal = "long"
                            sl_mult = 2.5
                            tp_mult = 4.0

                    if regime in ["TREND_DOWN", "TRANSITION"]:
                        if bar["close"] < roll_low and prev_bar["close"] >= roll_low:
                            signal = "short"
                            sl_mult = 2.5
                            tp_mult = 4.0

            elif params.entry_strategy == EntryStrategy.REGIME_CHANGE:
                # Enter when regime changes
                if regime != prev_regime:
                    if regime == "TREND_UP" and prev_regime in ["RANGE", "TRANSITION"]:
                        signal = "long"
                        sl_mult = params.sl_mult_trend
                        tp_mult = params.tp_mult_trend

                    elif regime == "TREND_DOWN" and prev_regime in ["RANGE", "TRANSITION"]:
                        signal = "short"
                        sl_mult = params.sl_mult_trend
                        tp_mult = params.tp_mult_trend

            elif params.entry_strategy == EntryStrategy.CONSERVATIVE:
                # Only enter with multiple confirmations
                if regime == "TREND_UP" and conf >= 0.85:
                    if passes_momentum and passes_volume and passes_trend_strength:
                        if ema_slope > 0 and ewm_ret > 0:
                            signal = "long"
                            sl_mult = 1.5
                            tp_mult = 3.0

                elif regime == "TREND_DOWN" and conf >= 0.85:
                    if passes_momentum and passes_volume and passes_trend_strength:
                        if ema_slope < 0 and ewm_ret < 0:
                            signal = "short"
                            sl_mult = 1.5
                            tp_mult = 3.0

            # Execute signal
            if signal is not None:
                entry_price = float(bar["close"])
                sl_distance = current_atr * sl_mult
                tp_distance = current_atr * tp_mult

                risk_amount = equity * params.risk_per_trade
                size = risk_amount / sl_distance
                max_size = equity / entry_price * 0.5
                size = min(size, max_size)

                if size > 0:
                    fee = entry_price * size * params.fee_rate
                    equity -= fee

                    if signal == "long":
                        sl = entry_price - sl_distance
                        tp = entry_price + tp_distance
                    else:
                        sl = entry_price + sl_distance
                        tp = entry_price - tp_distance

                    position = {
                        "side": signal,
                        "entry": entry_price,
                        "size": size,
                        "sl": sl,
                        "tp": tp,
                        "entry_idx": idx,
                        "entry_equity": equity,
                        "regime": regime,
                    }

        equity_curve.append(equity)

    # Close remaining position
    if position is not None:
        bar = df_bt.iloc[-1]
        exit_price = float(bar["close"])

        if position["side"] == "long":
            pnl = (exit_price - position["entry"]) * position["size"]
        else:
            pnl = (position["entry"] - exit_price) * position["size"]

        fee = exit_price * position["size"] * params.fee_rate
        pnl -= fee
        equity += pnl

        trades.append({
            "pnl": pnl,
            "pnl_pct": pnl / position["entry_equity"] * 100,
            "exit_reason": "eod",
            "regime": position["regime"],
            "bars_held": len(df_bt) - 1 - position["entry_idx"],
        })

    # Calculate metrics
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()

    results = {
        "final_equity": equity,
        "total_return_pct": (equity - initial_capital) / initial_capital * 100,
        "n_trades": len(trades),
    }

    if len(trades) > 0:
        wins = trades_df[trades_df["pnl"] > 0]
        losses = trades_df[trades_df["pnl"] <= 0]

        results["win_rate"] = len(wins) / len(trades) * 100
        results["profit_factor"] = (
            wins["pnl"].sum() / abs(losses["pnl"].sum())
            if len(losses) > 0 and losses["pnl"].sum() != 0
            else float("inf") if len(wins) > 0 else 0
        )
        results["avg_bars_held"] = trades_df["bars_held"].mean()

        eq_series = pd.Series(equity_curve)
        peak = eq_series.expanding().max()
        drawdown = (eq_series - peak) / peak * 100
        results["max_drawdown_pct"] = drawdown.min()

        # Exit reason breakdown
        exit_breakdown = trades_df.groupby("exit_reason").agg({
            "pnl": ["count", "sum"]
        })
        results["exit_breakdown"] = exit_breakdown.to_dict()
    else:
        results["win_rate"] = 0
        results["profit_factor"] = 0
        results["max_drawdown_pct"] = 0
        results["avg_bars_held"] = 0

    return results


def load_ohlcv_from_timescaledb(
    symbol: str, start: datetime, end: datetime, timeframe: str = "3m",
) -> pd.DataFrame:
    import psycopg2

    conn = psycopg2.connect(
        host="localhost", port=5432, database="xrp_timeseries",
        user="xrp_user", password="xrp_password_change_me",
    )

    query = """
        SELECT time as timestamp, open, high, low, close, volume
        FROM ohlcv
        WHERE symbol = %s AND timeframe = %s
          AND time >= %s AND time < %s
        ORDER BY time ASC
    """

    df = pd.read_sql(query, conn, params=(symbol, timeframe, start, end))
    conn.close()

    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)

    return df


def main():
    parser = argparse.ArgumentParser(description="Strategy tuning")
    parser.add_argument("--start", type=str, default="2023-01-01")
    parser.add_argument("--end", type=str, default="2024-12-31")
    parser.add_argument("--symbol", type=str, default="XRPUSDT")
    parser.add_argument("--n_states", type=int, default=5)
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/strategy_tune"))

    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("ADVANCED STRATEGY TUNING")
    logger.info("=" * 70)

    # Load data
    logger.info("Loading data...")
    start_dt = datetime.fromisoformat(args.start)
    end_dt = datetime.fromisoformat(args.end)
    df_3m = load_ohlcv_from_timescaledb(args.symbol, start_dt, end_dt, "3m")

    if df_3m.empty:
        logger.error("No data!")
        sys.exit(1)

    logger.info(f"Loaded {len(df_3m)} 3m bars")

    # Build features
    logger.info("Building features...")
    features, feature_names, df_features = build_features(df_3m)
    logger.info(f"Feature matrix shape: {features.shape}")

    # Split data
    n_total = len(features)
    n_train = int(n_total * 0.6)
    n_val = int(n_total * 0.2)

    train_features = features[:n_train]
    val_features = features[n_train:n_train + n_val]
    test_features = features[n_train + n_val:]

    train_df = df_features.iloc[:n_train].reset_index(drop=True)
    val_df = df_features.iloc[n_train:n_train + n_val].reset_index(drop=True)
    test_df = df_features.iloc[n_train + n_val:].reset_index(drop=True)

    logger.info(f"Train: {len(train_features)}, Val: {len(val_features)}, Test: {len(test_features)}")

    # Train HMM
    logger.info("Training HMM...")
    hmm_model = SingleHMM(n_states=args.n_states)
    hmm_model.train(train_features, feature_names)
    logger.info(f"State labels: {hmm_model.state_labels}")

    # Test different strategies
    strategies_to_test = [
        # Strategy 1: Original pullback
        StrategyParams(
            entry_strategy=EntryStrategy.PULLBACK,
            min_confidence=0.6,
            sl_mult_trend=2.5,
            tp_mult_trend=5.0,
        ),
        # Strategy 2: Momentum with trailing stop
        StrategyParams(
            entry_strategy=EntryStrategy.MOMENTUM,
            min_confidence=0.7,
            sl_mult_trend=2.0,
            tp_mult_trend=4.0,
            trend_strength_threshold=0.02,
            use_trailing_stop=True,
            trailing_atr_mult=2.0,
        ),
        # Strategy 3: Breakout
        StrategyParams(
            entry_strategy=EntryStrategy.BREAKOUT,
            min_confidence=0.6,
            sl_mult_trend=2.5,
            tp_mult_trend=4.0,
        ),
        # Strategy 4: Regime change
        StrategyParams(
            entry_strategy=EntryStrategy.REGIME_CHANGE,
            min_confidence=0.7,
            sl_mult_trend=2.0,
            tp_mult_trend=3.0,
        ),
        # Strategy 5: Conservative
        StrategyParams(
            entry_strategy=EntryStrategy.CONSERVATIVE,
            min_confidence=0.85,
            momentum_threshold=0.001,
            volume_z_threshold=0.5,
            trend_strength_threshold=0.03,
        ),
        # Strategy 6: Momentum with filters
        StrategyParams(
            entry_strategy=EntryStrategy.MOMENTUM,
            min_confidence=0.6,
            sl_mult_trend=2.0,
            tp_mult_trend=5.0,
            momentum_threshold=0.001,
            volume_z_threshold=0.0,
            trend_strength_threshold=0.01,
        ),
        # Strategy 7: Pullback with time exit
        StrategyParams(
            entry_strategy=EntryStrategy.PULLBACK,
            min_confidence=0.6,
            sl_mult_trend=2.5,
            tp_mult_trend=5.0,
            max_hold_bars=20,
        ),
        # Strategy 8: Tight stops
        StrategyParams(
            entry_strategy=EntryStrategy.PULLBACK,
            min_confidence=0.6,
            sl_mult_trend=1.5,
            tp_mult_trend=3.0,
            sl_mult_range=1.0,
            tp_mult_range=1.5,
        ),
        # Strategy 9: Wide RR
        StrategyParams(
            entry_strategy=EntryStrategy.MOMENTUM,
            min_confidence=0.7,
            sl_mult_trend=1.5,
            tp_mult_trend=6.0,
            trend_strength_threshold=0.02,
        ),
        # Strategy 10: Higher risk
        StrategyParams(
            entry_strategy=EntryStrategy.PULLBACK,
            min_confidence=0.6,
            risk_per_trade=0.02,
            sl_mult_trend=2.0,
            tp_mult_trend=4.0,
        ),
    ]

    logger.info("\n" + "=" * 70)
    logger.info("TESTING STRATEGIES ON VALIDATION SET")
    logger.info("=" * 70)

    val_results = []

    for i, params in enumerate(strategies_to_test, 1):
        logger.info(f"\nStrategy {i}: {params.entry_strategy.value}")

        results = run_strategy_backtest(val_df, hmm_model, val_features, params)

        val_results.append({
            "strategy_id": i,
            "strategy": params.entry_strategy.value,
            "params": params,
            "results": results,
        })

        logger.info(
            f"  Return: {results['total_return_pct']:.2f}%, "
            f"WR: {results['win_rate']:.1f}%, "
            f"PF: {results['profit_factor']:.2f}, "
            f"DD: {results['max_drawdown_pct']:.2f}%, "
            f"Trades: {results['n_trades']}"
        )

    # Sort by return
    val_results_sorted = sorted(val_results, key=lambda x: x["results"]["total_return_pct"], reverse=True)

    logger.info("\n" + "=" * 70)
    logger.info("VALIDATION RESULTS RANKING")
    logger.info("=" * 70)

    for rank, item in enumerate(val_results_sorted, 1):
        r = item["results"]
        logger.info(
            f"#{rank}: Strategy {item['strategy_id']} ({item['strategy']}) | "
            f"Return: {r['total_return_pct']:.2f}%, "
            f"WR: {r['win_rate']:.1f}%, "
            f"PF: {r['profit_factor']:.2f}"
        )

    # Test top 3 on test set
    logger.info("\n" + "=" * 70)
    logger.info("FINAL TEST ON OUT-OF-SAMPLE DATA (TOP 3 STRATEGIES)")
    logger.info("=" * 70)

    test_results = []

    for item in val_results_sorted[:3]:
        params = item["params"]
        strategy_id = item["strategy_id"]

        results = run_strategy_backtest(test_df, hmm_model, test_features, params)

        test_results.append({
            "strategy_id": strategy_id,
            "strategy": params.entry_strategy.value,
            "val_return": item["results"]["total_return_pct"],
            "test_results": results,
        })

        logger.info(f"\nStrategy {strategy_id} ({params.entry_strategy.value}):")
        logger.info(f"  Validation return: {item['results']['total_return_pct']:.2f}%")
        logger.info(f"  Test return: {results['total_return_pct']:.2f}%")
        logger.info(f"  Test win rate: {results['win_rate']:.1f}%")
        logger.info(f"  Test profit factor: {results['profit_factor']:.2f}")
        logger.info(f"  Test max DD: {results['max_drawdown_pct']:.2f}%")
        logger.info(f"  Test trades: {results['n_trades']}")

    # Save results
    args.output_dir.mkdir(parents=True, exist_ok=True)

    best_strategy = val_results_sorted[0]
    best_test = test_results[0]

    summary = {
        "best_strategy": {
            "id": best_strategy["strategy_id"],
            "type": best_strategy["strategy"],
            "params": {
                "entry_strategy": best_strategy["params"].entry_strategy.value,
                "min_confidence": best_strategy["params"].min_confidence,
                "sl_mult_trend": best_strategy["params"].sl_mult_trend,
                "tp_mult_trend": best_strategy["params"].tp_mult_trend,
                "use_trailing_stop": best_strategy["params"].use_trailing_stop,
                "momentum_threshold": best_strategy["params"].momentum_threshold,
                "volume_z_threshold": best_strategy["params"].volume_z_threshold,
                "trend_strength_threshold": best_strategy["params"].trend_strength_threshold,
                "max_hold_bars": best_strategy["params"].max_hold_bars,
            },
        },
        "validation_results": best_strategy["results"],
        "test_results": best_test["test_results"],
    }

    with open(args.output_dir / "best_strategy.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info(f"\nResults saved to {args.output_dir}")
    logger.info("\nDone!")


if __name__ == "__main__":
    main()
