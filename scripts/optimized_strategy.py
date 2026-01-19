#!/usr/bin/env python
"""Optimized HMM trading strategy with multi-timeframe alignment.

Key improvements:
1. Multi-timeframe trend alignment (15m + 1h)
2. Reduced false signals with stricter filters
3. Asymmetric risk management (smaller stops, wider targets)
4. Exit on regime change

Usage:
    python scripts/optimized_strategy.py
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

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


@dataclass
class OptimizedParams:
    """Optimized strategy parameters."""
    # Entry filters
    min_confidence: float = 0.65
    require_1h_alignment: bool = True  # 1h trend must align with trade direction
    min_ewm_ret_15m: float = 0.0005   # Minimum momentum
    min_ewm_ret_1h: float = 0.0002    # Minimum 1h momentum

    # Position sizing
    risk_per_trade: float = 0.01
    fee_rate: float = 0.001

    # Risk management (asymmetric RR)
    sl_atr_mult: float = 1.5
    tp_atr_mult: float = 4.0

    # Trade management
    exit_on_regime_change: bool = True
    max_consecutive_losses: int = 3
    cooldown_after_loss: int = 2  # Bars to wait after losing streak

    # Trailing stop
    use_trailing_stop: bool = True
    trail_activation_r: float = 1.5  # Activate after 1.5R profit
    trail_atr_mult: float = 1.5


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
    df_15m["ema_20_15m"] = ema_20

    tr_15m = pd.concat([
        high_15m - low_15m,
        (high_15m - close_15m.shift(1)).abs(),
        (low_15m - close_15m.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr_15m = tr_15m.ewm(alpha=1/14, adjust=False).mean()
    df_15m["atr_pct_15m"] = atr_15m / close_15m * 100
    df_15m["atr_15m"] = atr_15m

    df_15m["ewm_std_ret_15m"] = df_15m["ret_15m"].ewm(span=20, adjust=False).std()

    sma_15m = close_15m.rolling(20).mean()
    std_15m = close_15m.rolling(20).std()
    df_15m["bb_width_15m"] = (2 * std_15m / sma_15m) * 100
    df_15m["bb_upper_15m"] = sma_15m + 2 * std_15m
    df_15m["bb_lower_15m"] = sma_15m - 2 * std_15m

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

    close_1h = df_1h["close"].astype(float)
    high_1h = df_1h["high"].astype(float)
    low_1h = df_1h["low"].astype(float)

    df_1h["ret_1h"] = close_1h.pct_change()
    df_1h["ewm_ret_1h"] = df_1h["ret_1h"].ewm(span=20, adjust=False).mean()

    ema_20_1h = close_1h.ewm(span=20, adjust=False).mean()
    ema_std_1h = (close_1h - ema_20_1h).rolling(20).std().replace(0, np.nan)
    df_1h["price_z_from_ema_1h"] = (close_1h - ema_20_1h) / ema_std_1h
    df_1h["ema_20_1h"] = ema_20_1h

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

    features_1h = df_1h[["ret_1h", "ewm_ret_1h", "price_z_from_ema_1h", "atr_pct_1h", "ewm_std_ret_1h", "ema_20_1h"]]
    df_result = df_15m.join(features_1h, how="left")
    cols_to_ffill = ["ret_1h", "ewm_ret_1h", "price_z_from_ema_1h", "atr_pct_1h", "ewm_std_ret_1h", "ema_20_1h"]
    df_result[cols_to_ffill] = df_result[cols_to_ffill].ffill()
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


def run_optimized_backtest(
    df: pd.DataFrame,
    hmm_model: SingleHMM,
    features: np.ndarray,
    params: OptimizedParams,
    initial_capital: float = 10000.0,
    verbose: bool = False,
) -> Dict:
    """Run optimized backtest with multi-timeframe alignment."""

    confirm_cfg = ConfirmConfig(
        HIGH_VOL_LAMBDA_ON=2.0,
        HIGH_VOL_COOLDOWN_BARS_15M=6,
    )

    state_seq, state_probs, labels = hmm_model.predict_sequence(features)

    df_bt = df.copy()
    df_bt["regime_raw"] = labels
    df_bt["confidence"] = [probs.max() for probs in state_probs]

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

    # Backtest state
    equity = initial_capital
    position = None
    trades = []
    equity_curve = [equity]
    consecutive_losses = 0
    cooldown_bars = 0

    for idx in range(3, len(df_bt)):
        bar = df_bt.iloc[idx]
        prev_bar = df_bt.iloc[idx - 1]

        regime = bar["regime"]
        prev_regime = prev_bar["regime"]
        conf = bar["confidence"]
        atr = bar.get("atr_15m", 0)

        if pd.isna(atr) or atr <= 0:
            equity_curve.append(equity)
            continue

        # Position management
        if position is not None:
            hit_sl = False
            hit_tp = False
            exit_price = None
            exit_reason = None

            # Exit on regime change (optional)
            if params.exit_on_regime_change:
                if position["side"] == "long" and regime not in ["TREND_UP", "RANGE"]:
                    exit_price = float(bar["close"])
                    exit_reason = "regime_change"
                elif position["side"] == "short" and regime not in ["TREND_DOWN", "RANGE"]:
                    exit_price = float(bar["close"])
                    exit_reason = "regime_change"

            if exit_price is None:
                # Trailing stop update
                if params.use_trailing_stop:
                    current_profit_r = 0
                    if position["side"] == "long":
                        current_profit_r = (bar["high"] - position["entry"]) / position["risk_per_unit"]
                    else:
                        current_profit_r = (position["entry"] - bar["low"]) / position["risk_per_unit"]

                    if current_profit_r >= params.trail_activation_r:
                        if position["side"] == "long":
                            new_sl = bar["high"] - atr * params.trail_atr_mult
                            position["sl"] = max(position["sl"], new_sl)
                        else:
                            new_sl = bar["low"] + atr * params.trail_atr_mult
                            position["sl"] = min(position["sl"], new_sl)

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

                # Track consecutive losses
                if pnl < 0:
                    consecutive_losses += 1
                    if consecutive_losses >= params.max_consecutive_losses:
                        cooldown_bars = params.cooldown_after_loss
                else:
                    consecutive_losses = 0

                trades.append({
                    "pnl": pnl,
                    "pnl_pct": pnl / position["entry_equity"] * 100,
                    "exit_reason": exit_reason,
                    "regime": position["regime"],
                    "bars_held": idx - position["entry_idx"],
                })
                position = None

        # Cooldown check
        if cooldown_bars > 0:
            cooldown_bars -= 1
            equity_curve.append(equity)
            continue

        # Entry signals
        if position is None and conf >= params.min_confidence and equity > 0:
            signal = None

            # Get momentum values
            ewm_ret_15m = bar.get("ewm_ret_15m", 0) if "ewm_ret_15m" in bar.index else 0
            ewm_ret_1h = bar.get("ewm_ret_1h", 0) if "ewm_ret_1h" in bar.index else 0
            close = float(bar["close"])
            ema_1h = bar.get("ema_20_1h", close) if "ema_20_1h" in bar.index else close

            # Check 1h alignment
            price_above_1h_ema = close > ema_1h
            price_below_1h_ema = close < ema_1h

            # TREND_UP entry conditions
            if regime == "TREND_UP":
                momentum_ok = ewm_ret_15m >= params.min_ewm_ret_15m
                h1_aligned = (not params.require_1h_alignment) or (price_above_1h_ema and ewm_ret_1h >= params.min_ewm_ret_1h)

                if momentum_ok and h1_aligned:
                    # Enter on confirmation (2 green bars)
                    if prev_bar["close"] > df_bt.iloc[idx - 2]["close"]:
                        signal = "long"

            # TREND_DOWN entry conditions
            elif regime == "TREND_DOWN":
                momentum_ok = ewm_ret_15m <= -params.min_ewm_ret_15m
                h1_aligned = (not params.require_1h_alignment) or (price_below_1h_ema and ewm_ret_1h <= -params.min_ewm_ret_1h)

                if momentum_ok and h1_aligned:
                    if prev_bar["close"] < df_bt.iloc[idx - 2]["close"]:
                        signal = "short"

            # Execute signal
            if signal is not None:
                entry_price = float(bar["close"])
                sl_distance = atr * params.sl_atr_mult
                tp_distance = atr * params.tp_atr_mult

                risk_amount = equity * params.risk_per_trade
                size = risk_amount / sl_distance
                max_size = equity / entry_price * 0.3  # Max 30% equity
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
                        "risk_per_unit": sl_distance,
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
        "initial_capital": initial_capital,
        "final_equity": equity,
        "total_return_pct": (equity - initial_capital) / initial_capital * 100,
        "n_trades": len(trades),
    }

    if len(trades) > 0:
        wins = trades_df[trades_df["pnl"] > 0]
        losses = trades_df[trades_df["pnl"] <= 0]

        results["win_rate"] = len(wins) / len(trades) * 100
        results["avg_win_pct"] = wins["pnl_pct"].mean() if len(wins) > 0 else 0
        results["avg_loss_pct"] = abs(losses["pnl_pct"].mean()) if len(losses) > 0 else 0
        results["profit_factor"] = (
            wins["pnl"].sum() / abs(losses["pnl"].sum())
            if len(losses) > 0 and losses["pnl"].sum() != 0
            else float("inf") if len(wins) > 0 else 0
        )
        results["expectancy"] = trades_df["pnl_pct"].mean()
        results["avg_bars_held"] = trades_df["bars_held"].mean()

        eq_series = pd.Series(equity_curve)
        peak = eq_series.expanding().max()
        drawdown = (eq_series - peak) / peak * 100
        results["max_drawdown_pct"] = drawdown.min()

        # Exit reason breakdown
        results["exit_breakdown"] = trades_df.groupby("exit_reason")["pnl"].agg(["count", "sum", "mean"]).to_dict()

        # Regime breakdown
        results["regime_breakdown"] = trades_df.groupby("regime")["pnl"].agg(["count", "sum", "mean"]).to_dict()
    else:
        results["win_rate"] = 0
        results["profit_factor"] = 0
        results["max_drawdown_pct"] = 0
        results["expectancy"] = 0

    results["trades_df"] = trades_df

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


def grid_search_optimized(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    train_features: np.ndarray,
    val_features: np.ndarray,
    hmm_model: SingleHMM,
) -> Tuple[OptimizedParams, Dict]:
    """Grid search over optimized strategy parameters."""

    param_grid = {
        "min_confidence": [0.6, 0.65, 0.7],
        "require_1h_alignment": [True, False],
        "min_ewm_ret_15m": [0.0003, 0.0005, 0.0008],
        "sl_atr_mult": [1.0, 1.5, 2.0],
        "tp_atr_mult": [3.0, 4.0, 5.0],
        "exit_on_regime_change": [True, False],
        "use_trailing_stop": [True, False],
    }

    best_score = -float("inf")
    best_params = None
    best_results = None

    # Test key combinations (reduced grid)
    key_combos = [
        # Baseline
        {"min_confidence": 0.65, "require_1h_alignment": True, "min_ewm_ret_15m": 0.0005,
         "sl_atr_mult": 1.5, "tp_atr_mult": 4.0, "exit_on_regime_change": True, "use_trailing_stop": True},

        # Lower confidence
        {"min_confidence": 0.6, "require_1h_alignment": True, "min_ewm_ret_15m": 0.0005,
         "sl_atr_mult": 1.5, "tp_atr_mult": 4.0, "exit_on_regime_change": True, "use_trailing_stop": True},

        # No 1h alignment
        {"min_confidence": 0.65, "require_1h_alignment": False, "min_ewm_ret_15m": 0.0005,
         "sl_atr_mult": 1.5, "tp_atr_mult": 4.0, "exit_on_regime_change": True, "use_trailing_stop": True},

        # Tighter stops
        {"min_confidence": 0.65, "require_1h_alignment": True, "min_ewm_ret_15m": 0.0005,
         "sl_atr_mult": 1.0, "tp_atr_mult": 3.0, "exit_on_regime_change": True, "use_trailing_stop": True},

        # Wider targets
        {"min_confidence": 0.65, "require_1h_alignment": True, "min_ewm_ret_15m": 0.0005,
         "sl_atr_mult": 1.5, "tp_atr_mult": 5.0, "exit_on_regime_change": True, "use_trailing_stop": True},

        # No regime exit
        {"min_confidence": 0.65, "require_1h_alignment": True, "min_ewm_ret_15m": 0.0005,
         "sl_atr_mult": 1.5, "tp_atr_mult": 4.0, "exit_on_regime_change": False, "use_trailing_stop": True},

        # No trailing stop
        {"min_confidence": 0.65, "require_1h_alignment": True, "min_ewm_ret_15m": 0.0005,
         "sl_atr_mult": 1.5, "tp_atr_mult": 4.0, "exit_on_regime_change": True, "use_trailing_stop": False},

        # Higher momentum filter
        {"min_confidence": 0.65, "require_1h_alignment": True, "min_ewm_ret_15m": 0.0008,
         "sl_atr_mult": 1.5, "tp_atr_mult": 4.0, "exit_on_regime_change": True, "use_trailing_stop": True},

        # Lower momentum filter
        {"min_confidence": 0.65, "require_1h_alignment": True, "min_ewm_ret_15m": 0.0003,
         "sl_atr_mult": 1.5, "tp_atr_mult": 4.0, "exit_on_regime_change": True, "use_trailing_stop": True},

        # High RR
        {"min_confidence": 0.6, "require_1h_alignment": True, "min_ewm_ret_15m": 0.0003,
         "sl_atr_mult": 1.0, "tp_atr_mult": 5.0, "exit_on_regime_change": True, "use_trailing_stop": True},

        # Conservative
        {"min_confidence": 0.7, "require_1h_alignment": True, "min_ewm_ret_15m": 0.0008,
         "sl_atr_mult": 1.5, "tp_atr_mult": 4.0, "exit_on_regime_change": True, "use_trailing_stop": True},

        # Aggressive
        {"min_confidence": 0.6, "require_1h_alignment": False, "min_ewm_ret_15m": 0.0003,
         "sl_atr_mult": 2.0, "tp_atr_mult": 5.0, "exit_on_regime_change": False, "use_trailing_stop": False},
    ]

    logger.info(f"Testing {len(key_combos)} parameter combinations...")

    for i, combo in enumerate(key_combos):
        params = OptimizedParams(**combo)

        results = run_optimized_backtest(val_df, hmm_model, val_features, params)

        # Score: prioritize positive returns with good RR
        n_trades = results["n_trades"]
        if n_trades < 20:
            score = -1000
        else:
            ret = results["total_return_pct"]
            pf = min(results["profit_factor"], 5)
            wr = results["win_rate"]
            mdd = abs(results["max_drawdown_pct"])

            if ret > 0:
                score = ret * np.sqrt(pf) * (wr / 100) / (1 + mdd / 50)
            else:
                score = ret * (1 + mdd / 50)

        if score > best_score:
            best_score = score
            best_params = params
            best_results = results

        logger.info(
            f"  [{i+1}/{len(key_combos)}] "
            f"Ret={results['total_return_pct']:.2f}%, "
            f"WR={results['win_rate']:.1f}%, "
            f"PF={results['profit_factor']:.2f}, "
            f"Trades={n_trades}"
        )

    return best_params, best_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=str, default="2023-01-01")
    parser.add_argument("--end", type=str, default="2024-12-31")
    parser.add_argument("--symbol", type=str, default="XRPUSDT")
    parser.add_argument("--n_states", type=int, default=5)
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/optimized"))

    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("OPTIMIZED HMM TRADING STRATEGY")
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

    # Grid search on validation
    logger.info("\n" + "=" * 70)
    logger.info("GRID SEARCH ON VALIDATION SET")
    logger.info("=" * 70)

    best_params, best_val_results = grid_search_optimized(
        train_df, val_df, train_features, val_features, hmm_model
    )

    logger.info("\n" + "=" * 70)
    logger.info("BEST VALIDATION RESULTS")
    logger.info("=" * 70)
    logger.info(f"Return: {best_val_results['total_return_pct']:.2f}%")
    logger.info(f"Win Rate: {best_val_results['win_rate']:.1f}%")
    logger.info(f"Profit Factor: {best_val_results['profit_factor']:.2f}")
    logger.info(f"Max DD: {best_val_results['max_drawdown_pct']:.2f}%")
    logger.info(f"N Trades: {best_val_results['n_trades']}")
    logger.info(f"Expectancy: {best_val_results.get('expectancy', 0):.3f}%")

    logger.info("\nBest parameters:")
    logger.info(f"  min_confidence: {best_params.min_confidence}")
    logger.info(f"  require_1h_alignment: {best_params.require_1h_alignment}")
    logger.info(f"  min_ewm_ret_15m: {best_params.min_ewm_ret_15m}")
    logger.info(f"  sl_atr_mult: {best_params.sl_atr_mult}")
    logger.info(f"  tp_atr_mult: {best_params.tp_atr_mult}")
    logger.info(f"  exit_on_regime_change: {best_params.exit_on_regime_change}")
    logger.info(f"  use_trailing_stop: {best_params.use_trailing_stop}")

    # Test on out-of-sample
    logger.info("\n" + "=" * 70)
    logger.info("FINAL TEST ON OUT-OF-SAMPLE DATA")
    logger.info("=" * 70)

    test_results = run_optimized_backtest(test_df, hmm_model, test_features, best_params)

    logger.info(f"\nTest Results:")
    logger.info(f"Return: {test_results['total_return_pct']:.2f}%")
    logger.info(f"Win Rate: {test_results['win_rate']:.1f}%")
    logger.info(f"Profit Factor: {test_results['profit_factor']:.2f}")
    logger.info(f"Max DD: {test_results['max_drawdown_pct']:.2f}%")
    logger.info(f"N Trades: {test_results['n_trades']}")
    logger.info(f"Expectancy: {test_results.get('expectancy', 0):.3f}%")

    if test_results["n_trades"] > 0:
        logger.info(f"Avg Win: {test_results.get('avg_win_pct', 0):.3f}%")
        logger.info(f"Avg Loss: {test_results.get('avg_loss_pct', 0):.3f}%")
        logger.info(f"Avg Bars Held: {test_results.get('avg_bars_held', 0):.1f}")

    # Save results
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "best_params": {
            "min_confidence": best_params.min_confidence,
            "require_1h_alignment": best_params.require_1h_alignment,
            "min_ewm_ret_15m": best_params.min_ewm_ret_15m,
            "sl_atr_mult": best_params.sl_atr_mult,
            "tp_atr_mult": best_params.tp_atr_mult,
            "exit_on_regime_change": best_params.exit_on_regime_change,
            "use_trailing_stop": best_params.use_trailing_stop,
        },
        "validation_results": {
            "return_pct": best_val_results["total_return_pct"],
            "win_rate": best_val_results["win_rate"],
            "profit_factor": best_val_results["profit_factor"],
            "max_drawdown_pct": best_val_results["max_drawdown_pct"],
            "n_trades": best_val_results["n_trades"],
        },
        "test_results": {
            "return_pct": test_results["total_return_pct"],
            "win_rate": test_results["win_rate"],
            "profit_factor": test_results["profit_factor"],
            "max_drawdown_pct": test_results["max_drawdown_pct"],
            "n_trades": test_results["n_trades"],
        },
    }

    with open(args.output_dir / "optimized_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\nResults saved to {args.output_dir}")
    logger.info("\nDone!")


if __name__ == "__main__":
    main()
