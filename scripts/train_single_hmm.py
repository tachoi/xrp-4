#!/usr/bin/env python
"""Train Single HMM model and run backtest for xrp-4.

Simplified single HMM approach (no multi-HMM fusion).
Loads feature data from MongoDB.

Usage:
    python scripts/train_single_hmm.py --start 2023-01-01 --end 2024-12-31
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from hmmlearn import hmm
from scipy.stats import entropy
from pymongo import MongoClient

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


class SingleHMM:
    """Single HMM model for regime detection on 15m timeframe.

    5-state model:
    - RANGE: Low volatility, mean-reverting
    - TREND_UP: Positive momentum
    - TREND_DOWN: Negative momentum
    - HIGH_VOL: High volatility without clear direction
    - TRANSITION: Regime change / uncertainty
    """

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

        # Scaling params
        self._feature_mean: Optional[np.ndarray] = None
        self._feature_std: Optional[np.ndarray] = None

        # Training metadata
        self.n_samples: int = 0
        self.log_likelihood: float = 0.0

    def _create_model(self) -> hmm.GaussianHMM:
        return hmm.GaussianHMM(
            n_components=self.n_states,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=self.random_state,
            verbose=False,
        )

    def train(
        self,
        features: np.ndarray,
        feature_names: List[str],
    ) -> float:
        """Train the HMM model."""
        if features.ndim != 2:
            raise ValueError(f"Features must be 2D, got {features.shape}")

        n_samples, n_features = features.shape
        if n_samples < self.n_states * 10:
            raise ValueError(f"Insufficient samples: {n_samples}")

        # Handle NaN
        if np.any(np.isnan(features)):
            logger.warning("NaN in features, filling with mean")
            col_means = np.nanmean(features, axis=0)
            for i in range(n_features):
                mask = np.isnan(features[:, i])
                features[mask, i] = col_means[i]

        # Standardize
        self._feature_mean = np.mean(features, axis=0)
        self._feature_std = np.std(features, axis=0)
        self._feature_std[self._feature_std == 0] = 1.0
        features_scaled = (features - self._feature_mean) / self._feature_std

        # Train
        self.model = self._create_model()
        self.model.fit(features_scaled)

        self.feature_names = feature_names
        self.n_samples = n_samples
        self.log_likelihood = self.model.score(features_scaled)

        # Map states to labels
        self._map_states_to_labels(features_scaled)

        self._is_trained = True
        logger.info(
            f"HMM trained: {n_samples} samples, {n_features} features, "
            f"log_likelihood={self.log_likelihood:.2f}"
        )

        return self.log_likelihood

    def _map_states_to_labels(self, features: np.ndarray) -> None:
        """Map HMM states to regime labels for 5-state model.

        Labels:
        - RANGE: Lowest variance state
        - TREND_UP: Positive return with moderate variance
        - TREND_DOWN: Negative return with moderate variance
        - HIGH_VOL: High variance but not highest
        - TRANSITION: Highest variance (regime change)
        """
        if self.model is None:
            return

        means = self.model.means_

        # Get state variances
        if self.covariance_type == "full":
            variances = np.array([np.trace(c) for c in self.model.covars_])
        elif self.covariance_type == "diag":
            variances = np.sum(self.model.covars_, axis=1)
        else:
            variances = np.ones(self.n_states)

        var_order = np.argsort(variances)  # Low to high variance

        # Find return feature index
        return_idx = 0
        for i, name in enumerate(self.feature_names):
            if name == "ret":
                return_idx = i
                break

        state_returns = means[:, return_idx]

        # Find volatility feature index (ewm_std_ret or atr_pct)
        vol_idx = None
        for i, name in enumerate(self.feature_names):
            if "std" in name.lower() or "atr" in name.lower():
                vol_idx = i
                break

        # Assign labels for 5-state model
        self.state_labels = {}

        # Lowest variance -> RANGE
        self.state_labels[var_order[0]] = "RANGE"

        # Highest variance -> TRANSITION
        self.state_labels[var_order[-1]] = "TRANSITION"

        # Second highest variance -> HIGH_VOL
        self.state_labels[var_order[-2]] = "HIGH_VOL"

        # Remaining 2 states -> TREND_UP and TREND_DOWN based on return direction
        remaining = list(var_order[1:-2])

        if len(remaining) >= 2:
            ret_sorted = sorted(remaining, key=lambda s: state_returns[s])
            self.state_labels[ret_sorted[0]] = "TREND_DOWN"
            self.state_labels[ret_sorted[-1]] = "TREND_UP"
        elif len(remaining) == 1:
            s = remaining[0]
            self.state_labels[s] = "TREND_UP" if state_returns[s] > 0 else "TREND_DOWN"

        # Log state characteristics
        logger.info("State characteristics:")
        for state_idx in range(self.n_states):
            label = self.state_labels.get(state_idx, "UNKNOWN")
            var = variances[state_idx]
            ret = state_returns[state_idx]
            logger.info(f"  State {state_idx} ({label}): var={var:.4f}, ret={ret:.6f}")

        logger.info(f"State labels: {self.state_labels}")

    def predict_sequence(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict regime for sequence.

        Returns:
            state_seq: State indices
            state_probs: State probabilities for each sample
            labels: Regime labels
        """
        if not self._is_trained:
            raise RuntimeError("Model not trained")

        # Handle NaN
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

    def save(self, path: Path) -> None:
        """Save model to JSON."""
        if not self._is_trained:
            raise RuntimeError("Model not trained")

        data = {
            "n_states": self.n_states,
            "covariance_type": self.covariance_type,
            "feature_names": self.feature_names,
            "means": self.model.means_.tolist(),
            "covars": self.model.covars_.tolist(),
            "transmat": self.model.transmat_.tolist(),
            "startprob": self.model.startprob_.tolist(),
            "feature_mean": self._feature_mean.tolist(),
            "feature_std": self._feature_std.tolist(),
            "state_labels": {int(k): v for k, v in self.state_labels.items()},
            "n_samples": self.n_samples,
            "log_likelihood": self.log_likelihood,
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Model saved to {path}")

    def load(self, path: Path) -> None:
        """Load model from JSON."""
        with open(path) as f:
            data = json.load(f)

        self.n_states = data["n_states"]
        self.covariance_type = data["covariance_type"]
        self.feature_names = data["feature_names"]
        self.state_labels = {int(k): v for k, v in data["state_labels"].items()}
        self._feature_mean = np.array(data["feature_mean"])
        self._feature_std = np.array(data["feature_std"])
        self.n_samples = data["n_samples"]
        self.log_likelihood = data["log_likelihood"]

        self.model = self._create_model()
        self.model.means_ = np.array(data["means"])
        self.model.covars_ = np.array(data["covars"])
        self.model.transmat_ = np.array(data["transmat"])
        self.model.startprob_ = np.array(data["startprob"])

        self._is_trained = True
        logger.info(f"Model loaded from {path}")


def resample_to_15m(df_3m: pd.DataFrame) -> pd.DataFrame:
    """Resample 3m OHLCV data to 15m."""
    df = df_3m.copy()

    # Ensure timestamp is datetime
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")

    # Resample
    resampled = df.resample("15min").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    })

    # Drop incomplete bars
    resampled = resampled.dropna(subset=["close"])
    resampled = resampled.reset_index()

    return resampled


def build_features(df_3m: pd.DataFrame) -> Tuple[np.ndarray, List[str], pd.DataFrame]:
    """Build multi-timeframe features for HMM.

    Uses 15m and 1h features for regime detection.

    Returns:
        feature_matrix: numpy array of features
        feature_names: list of feature names
        df_result: DataFrame with all features and timestamps
    """
    # Resample to 15m and 1h
    df_15m = resample_to_15m(df_3m)
    df_1h = _resample_to_1h(df_3m)

    # Build 15m features
    close_15m = df_15m["close"].astype(float)
    high_15m = df_15m["high"].astype(float)
    low_15m = df_15m["low"].astype(float)
    open_15m = df_15m["open"].astype(float)
    volume_15m = df_15m["volume"].astype(float)

    # Direction features (15m)
    df_15m["ret_15m"] = close_15m.pct_change()
    df_15m["ewm_ret_15m"] = df_15m["ret_15m"].ewm(span=20, adjust=False).mean()

    # EMA slope (15m)
    ema_20 = close_15m.ewm(span=20, adjust=False).mean()
    df_15m["ema_slope_15m"] = (ema_20 - ema_20.shift(1)) / close_15m * 100

    # Volatility features (15m)
    tr_15m = pd.concat([
        high_15m - low_15m,
        (high_15m - close_15m.shift(1)).abs(),
        (low_15m - close_15m.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr_15m = tr_15m.ewm(alpha=1/14, adjust=False).mean()
    df_15m["atr_pct_15m"] = atr_15m / close_15m * 100

    df_15m["ewm_std_ret_15m"] = df_15m["ret_15m"].ewm(span=20, adjust=False).std()

    # Bollinger Band Width (15m)
    sma_15m = close_15m.rolling(20).mean()
    std_15m = close_15m.rolling(20).std()
    df_15m["bb_width_15m"] = (2 * std_15m / sma_15m) * 100

    # BB width percentile (recent 50 bars)
    df_15m["bb_width_15m_pct"] = df_15m["bb_width_15m"].rolling(50).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if x.max() != x.min() else 0.5
    )

    # Range compression (15m) - inverse of range expansion
    rolling_range = (high_15m.rolling(20).max() - low_15m.rolling(20).min())
    recent_range = high_15m - low_15m
    df_15m["range_comp_15m"] = 1 - (recent_range / rolling_range.replace(0, np.nan))
    df_15m["range_comp_15m"] = df_15m["range_comp_15m"].clip(0, 1)

    # Candle structure (15m)
    total_range = (high_15m - low_15m).replace(0, np.nan)
    body = (close_15m - open_15m).abs()
    df_15m["body_ratio"] = body / total_range

    upper_wick = high_15m - pd.concat([open_15m, close_15m], axis=1).max(axis=1)
    df_15m["upper_wick_ratio"] = upper_wick / total_range

    lower_wick = pd.concat([open_15m, close_15m], axis=1).min(axis=1) - low_15m
    df_15m["lower_wick_ratio"] = lower_wick / total_range

    # Volume Z-score (15m)
    vol_mean = volume_15m.rolling(20).mean()
    vol_std = volume_15m.rolling(20).std().replace(0, np.nan)
    df_15m["vol_z_15m"] = (volume_15m - vol_mean) / vol_std

    # Build 1h features
    close_1h = df_1h["close"].astype(float)
    high_1h = df_1h["high"].astype(float)
    low_1h = df_1h["low"].astype(float)

    # Direction features (1h)
    df_1h["ret_1h"] = close_1h.pct_change()
    df_1h["ewm_ret_1h"] = df_1h["ret_1h"].ewm(span=20, adjust=False).mean()

    # Price Z from EMA (1h)
    ema_20_1h = close_1h.ewm(span=20, adjust=False).mean()
    ema_std_1h = (close_1h - ema_20_1h).rolling(20).std().replace(0, np.nan)
    df_1h["price_z_from_ema_1h"] = (close_1h - ema_20_1h) / ema_std_1h

    # Volatility features (1h)
    tr_1h = pd.concat([
        high_1h - low_1h,
        (high_1h - close_1h.shift(1)).abs(),
        (low_1h - close_1h.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr_1h = tr_1h.ewm(alpha=1/14, adjust=False).mean()
    df_1h["atr_pct_1h"] = atr_1h / close_1h * 100

    df_1h["ewm_std_ret_1h"] = df_1h["ret_1h"].ewm(span=20, adjust=False).std()

    # Merge 1h features to 15m (forward fill)
    df_15m = df_15m.set_index("timestamp")
    df_1h = df_1h.set_index("timestamp")

    features_1h = df_1h[["ret_1h", "ewm_ret_1h", "price_z_from_ema_1h", "atr_pct_1h", "ewm_std_ret_1h"]]
    df_result = df_15m.join(features_1h, how="left")
    df_result[["ret_1h", "ewm_ret_1h", "price_z_from_ema_1h", "atr_pct_1h", "ewm_std_ret_1h"]] = \
        df_result[["ret_1h", "ewm_ret_1h", "price_z_from_ema_1h", "atr_pct_1h", "ewm_std_ret_1h"]].ffill()

    df_result = df_result.reset_index()

    # Select HMM features
    hmm_features = [
        # Direction
        "ret_15m", "ret_1h",
        "ewm_ret_15m", "ewm_ret_1h",
        "ema_slope_15m", "price_z_from_ema_1h",
        # Volatility
        "atr_pct_15m", "atr_pct_1h",
        "ewm_std_ret_15m", "ewm_std_ret_1h",
        "bb_width_15m",
        # Range / Compression
        "range_comp_15m", "bb_width_15m_pct",
        # Candle structure
        "body_ratio", "upper_wick_ratio", "lower_wick_ratio",
        # Volume
        "vol_z_15m",
    ]

    # Check missing features
    missing = [f for f in hmm_features if f not in df_result.columns]
    if missing:
        raise ValueError(f"Missing features: {missing}")

    # Drop NaN
    df_clean = df_result.dropna(subset=hmm_features)

    feature_matrix = df_clean[hmm_features].values
    feature_names = hmm_features

    return feature_matrix, feature_names, df_clean


def _resample_to_1h(df_3m: pd.DataFrame) -> pd.DataFrame:
    """Resample 3m OHLCV data to 1h."""
    df = df_3m.copy()

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")

    resampled = df.resample("1h").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    })

    resampled = resampled.dropna(subset=["close"])
    resampled = resampled.reset_index()

    return resampled


def run_hmm_backtest(
    df: pd.DataFrame,
    hmm_model: SingleHMM,
    features: np.ndarray,
    initial_capital: float = 10000.0,
    risk_per_trade: float = 0.01,
    fee_rate: float = 0.001,
    use_confirm_layer: bool = True,
) -> Dict:
    """Run backtest with HMM regime filter and Confirm Layer.

    Strategy:
    - TREND_UP: Long on pullback
    - TREND_DOWN: Short on bounce
    - RANGE: Mean reversion
    - HIGH_VOL: No trade (too volatile)
    - TRANSITION: No trade (unless confirmed)
    - NO_TRADE: Skip (cooldown)
    """
    from xrp4.regime.confirm import RegimeConfirmLayer, ConfirmConfig

    # Get regime predictions
    state_seq, state_probs, labels = hmm_model.predict_sequence(features)

    # Align with df
    df_bt = df.copy()
    df_bt["regime_raw"] = labels
    df_bt["confidence"] = [probs.max() for probs in state_probs]

    # Calculate ATR for position sizing
    close = df_bt["close"].astype(float)
    high = df_bt["high"].astype(float)
    low = df_bt["low"].astype(float)

    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False).mean()
    df_bt["atr"] = atr

    # Initialize Confirm Layer
    confirm_layer = RegimeConfirmLayer(ConfirmConfig()) if use_confirm_layer else None
    confirm_state = None

    # Apply Confirm Layer to get confirmed regimes
    confirmed_regimes = []
    confirm_reasons = []

    for idx in range(len(df_bt)):
        row = df_bt.iloc[idx].to_dict()
        regime_raw = row["regime_raw"]

        if confirm_layer and idx >= 32:  # Need history for confirm
            hist = df_bt.iloc[max(0, idx-96):idx+1]
            result, confirm_state = confirm_layer.confirm(
                regime_raw=regime_raw,
                row_15m=row,
                hist_15m=hist,
                state=confirm_state,
            )
            confirmed_regimes.append(result.confirmed_regime)
            confirm_reasons.append(result.reason)
        else:
            confirmed_regimes.append(regime_raw)
            confirm_reasons.append("NO_CONFIRM_LAYER")

    df_bt["regime"] = confirmed_regimes
    df_bt["confirm_reason"] = confirm_reasons

    # Backtest state
    equity = initial_capital
    position = None
    trades = []
    equity_curve = [equity]

    # Minimum confidence to trade
    MIN_CONF = 0.8

    for idx in range(2, len(df_bt)):
        bar = df_bt.iloc[idx]
        prev_bar = df_bt.iloc[idx - 1]
        prev2_bar = df_bt.iloc[idx - 2]

        regime = bar["regime"]
        conf = bar["confidence"]
        current_atr = bar["atr"]

        # Skip if ATR is invalid
        if pd.isna(current_atr) or current_atr <= 0:
            equity_curve.append(equity)
            continue

        # Update position - check SL/TP
        if position is not None:
            hit_sl = False
            hit_tp = False
            exit_price = None

            if position["side"] == "long":
                if bar["low"] <= position["sl"]:
                    hit_sl = True
                    exit_price = position["sl"]
                elif bar["high"] >= position["tp"]:
                    hit_tp = True
                    exit_price = position["tp"]
            else:  # short
                if bar["high"] >= position["sl"]:
                    hit_sl = True
                    exit_price = position["sl"]
                elif bar["low"] <= position["tp"]:
                    hit_tp = True
                    exit_price = position["tp"]

            if hit_sl or hit_tp:
                # Close position
                if position["side"] == "long":
                    pnl = (exit_price - position["entry"]) * position["size"]
                else:
                    pnl = (position["entry"] - exit_price) * position["size"]

                fee = exit_price * position["size"] * fee_rate
                pnl -= fee
                equity += pnl

                trades.append({
                    "entry_idx": position["entry_idx"],
                    "exit_idx": idx,
                    "entry_time": position["entry_time"],
                    "exit_time": bar["timestamp"],
                    "side": position["side"],
                    "entry_price": position["entry"],
                    "exit_price": exit_price,
                    "size": position["size"],
                    "pnl": pnl,
                    "pnl_pct": pnl / position["entry_equity"] * 100,
                    "exit_reason": "sl" if hit_sl else "tp",
                    "regime_raw": position["regime_raw"],
                    "regime": position["regime"],
                    "confirm_reason": position["confirm_reason"],
                })
                position = None

        # Generate signals (only if no position and high confidence)
        # Skip NO_TRADE, HIGH_VOL, unconfirmed TRANSITION
        if position is None and conf >= MIN_CONF and equity > 0:
            signal = None
            sl_mult = 2.0
            tp_mult = 3.0

            # TREND_UP: Long on pullback (confirmed)
            if regime == "TREND_UP":
                if prev_bar["close"] < prev2_bar["close"]:
                    signal = "long"
                    sl_mult = 2.0
                    tp_mult = 4.0

            # TREND_DOWN: Short on bounce (confirmed)
            elif regime == "TREND_DOWN":
                if prev_bar["close"] > prev2_bar["close"]:
                    signal = "short"
                    sl_mult = 2.0
                    tp_mult = 4.0

            # RANGE: Mean reversion
            elif regime == "RANGE":
                roll_high = high.iloc[max(0, idx-20):idx].max()
                roll_low = low.iloc[max(0, idx-20):idx].min()
                mid_price = (roll_high + roll_low) / 2

                if bar["close"] < mid_price and prev_bar["close"] < prev2_bar["close"]:
                    signal = "long"
                    sl_mult = 1.5
                    tp_mult = 2.0
                elif bar["close"] > mid_price and prev_bar["close"] > prev2_bar["close"]:
                    signal = "short"
                    sl_mult = 1.5
                    tp_mult = 2.0

            # HIGH_VOL, TRANSITION, NO_TRADE: Skip

            # Execute signal
            if signal is not None:
                entry_price = float(bar["close"])
                sl_distance = current_atr * sl_mult
                tp_distance = current_atr * tp_mult

                risk_amount = equity * risk_per_trade
                size = risk_amount / sl_distance

                max_size = equity / entry_price * 0.5
                size = min(size, max_size)

                if size > 0:
                    fee = entry_price * size * fee_rate
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
                        "entry_time": bar["timestamp"],
                        "entry_equity": equity,
                        "regime_raw": bar["regime_raw"],
                        "regime": regime,
                        "confirm_reason": bar["confirm_reason"],
                    }

        equity_curve.append(equity)

    # Close any remaining position
    if position is not None:
        bar = df_bt.iloc[-1]
        exit_price = float(bar["close"])

        if position["side"] == "long":
            pnl = (exit_price - position["entry"]) * position["size"]
        else:
            pnl = (position["entry"] - exit_price) * position["size"]

        fee = exit_price * position["size"] * fee_rate
        pnl -= fee
        equity += pnl

        trades.append({
            "entry_idx": position["entry_idx"],
            "exit_idx": len(df_bt) - 1,
            "entry_time": position["entry_time"],
            "exit_time": bar["timestamp"],
            "side": position["side"],
            "entry_price": position["entry"],
            "exit_price": exit_price,
            "size": position["size"],
            "pnl": pnl,
            "pnl_pct": pnl / position["entry_equity"] * 100,
            "exit_reason": "eod",
            "regime_raw": position.get("regime_raw", position["regime"]),
            "regime": position["regime"],
            "confirm_reason": position.get("confirm_reason", ""),
        })

    # Calculate metrics
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()

    results = {
        "initial_capital": initial_capital,
        "final_equity": equity,
        "total_return_pct": (equity - initial_capital) / initial_capital * 100,
        "n_trades": len(trades),
        "trades": trades_df,
        "equity_curve": equity_curve,
    }

    if len(trades) > 0:
        wins = trades_df[trades_df["pnl"] > 0]
        losses = trades_df[trades_df["pnl"] <= 0]

        results["win_rate"] = len(wins) / len(trades) * 100
        results["avg_win"] = wins["pnl"].mean() if len(wins) > 0 else 0
        results["avg_loss"] = abs(losses["pnl"].mean()) if len(losses) > 0 else 0
        results["profit_factor"] = (
            wins["pnl"].sum() / abs(losses["pnl"].sum())
            if len(losses) > 0 and losses["pnl"].sum() != 0
            else float("inf")
        )
        results["avg_pnl_pct"] = trades_df["pnl_pct"].mean()

        # Max drawdown
        eq_series = pd.Series(equity_curve)
        peak = eq_series.expanding().max()
        drawdown = (eq_series - peak) / peak * 100
        results["max_drawdown_pct"] = drawdown.min()

        # Regime breakdown
        regime_stats = {}
        for regime in ["TREND_UP", "TREND_DOWN", "RANGE"]:
            regime_trades = trades_df[trades_df["regime"] == regime]
            if len(regime_trades) > 0:
                regime_wins = regime_trades[regime_trades["pnl"] > 0]
                regime_stats[regime] = {
                    "n_trades": len(regime_trades),
                    "win_rate": len(regime_wins) / len(regime_trades) * 100,
                    "total_pnl": regime_trades["pnl"].sum(),
                    "avg_pnl_pct": regime_trades["pnl_pct"].mean(),
                }
        results["regime_stats"] = regime_stats

        # Exit reason breakdown
        exit_stats = trades_df.groupby("exit_reason").agg({
            "pnl": ["count", "sum", "mean"]
        }).to_dict()
        results["exit_stats"] = exit_stats

        # Confirm reason breakdown
        if "confirm_reason" in trades_df.columns:
            confirm_stats = {}
            for reason in trades_df["confirm_reason"].unique():
                reason_trades = trades_df[trades_df["confirm_reason"] == reason]
                if len(reason_trades) > 0:
                    reason_wins = reason_trades[reason_trades["pnl"] > 0]
                    confirm_stats[reason] = {
                        "n_trades": len(reason_trades),
                        "win_rate": len(reason_wins) / len(reason_trades) * 100,
                        "total_pnl": reason_trades["pnl"].sum(),
                    }
            results["confirm_stats"] = confirm_stats

    # Confirm layer summary from df_bt
    confirm_summary = df_bt["regime"].value_counts().to_dict()
    results["confirm_regime_distribution"] = confirm_summary

    raw_summary = df_bt["regime_raw"].value_counts().to_dict()
    results["raw_regime_distribution"] = raw_summary

    return results


def analyze_regime_distribution(
    df: pd.DataFrame,
    hmm_model: SingleHMM,
    features: np.ndarray,
) -> Dict:
    """Analyze regime distribution without trading.

    Returns regime statistics and distribution.
    """
    # Get regime predictions
    state_seq, state_probs, labels = hmm_model.predict_sequence(features)

    # Align features with df
    n_warmup = len(df) - len(features)
    df_aligned = df.iloc[n_warmup:].reset_index(drop=True)

    # Add regime labels
    df_aligned = df_aligned.copy()
    df_aligned["regime"] = labels
    df_aligned["state_idx"] = state_seq
    df_aligned["confidence"] = [probs.max() for probs in state_probs]

    # Calculate returns for each bar
    df_aligned["ret"] = df_aligned["close"].pct_change()

    # Regime distribution
    regime_counts = df_aligned["regime"].value_counts()
    total_bars = len(df_aligned)

    results = {
        "total_bars": total_bars,
        "regime_distribution": {},
        "regime_returns": {},
        "transition_matrix": {},
        "confidence_stats": {},
    }

    # Distribution and returns by regime
    for regime in regime_counts.index:
        count = regime_counts[regime]
        pct = count / total_bars * 100
        regime_data = df_aligned[df_aligned["regime"] == regime]

        # Average return in this regime
        avg_ret = regime_data["ret"].mean() * 100  # in percent
        std_ret = regime_data["ret"].std() * 100
        avg_conf = regime_data["confidence"].mean()

        results["regime_distribution"][regime] = {
            "count": int(count),
            "pct": pct,
        }
        results["regime_returns"][regime] = {
            "avg_ret_pct": avg_ret,
            "std_ret_pct": std_ret,
            "sharpe": avg_ret / std_ret if std_ret > 0 else 0,
        }
        results["confidence_stats"][regime] = {
            "avg_confidence": avg_conf,
            "high_conf_pct": (regime_data["confidence"] >= 0.7).mean() * 100,
        }

    # Transition analysis
    transitions = {}
    for i in range(1, len(df_aligned)):
        prev_regime = df_aligned.iloc[i-1]["regime"]
        curr_regime = df_aligned.iloc[i]["regime"]
        key = f"{prev_regime}->{curr_regime}"
        transitions[key] = transitions.get(key, 0) + 1

    # Sort by frequency
    results["transitions"] = dict(sorted(transitions.items(), key=lambda x: -x[1])[:10])

    # Regime persistence (average bars in same regime)
    regime_runs = []
    current_regime = df_aligned.iloc[0]["regime"]
    run_length = 1

    for i in range(1, len(df_aligned)):
        if df_aligned.iloc[i]["regime"] == current_regime:
            run_length += 1
        else:
            regime_runs.append((current_regime, run_length))
            current_regime = df_aligned.iloc[i]["regime"]
            run_length = 1
    regime_runs.append((current_regime, run_length))

    # Average persistence by regime
    persistence = {}
    for regime in regime_counts.index:
        runs = [r[1] for r in regime_runs if r[0] == regime]
        if runs:
            persistence[regime] = {
                "avg_bars": np.mean(runs),
                "max_bars": max(runs),
                "n_runs": len(runs),
            }
    results["persistence"] = persistence

    return results, df_aligned


def load_ohlcv_from_timescaledb(
    symbol: str,
    start: datetime,
    end: datetime,
    timeframe: str = "3m",
) -> pd.DataFrame:
    """Load OHLCV data from TimescaleDB."""
    import psycopg2

    conn = psycopg2.connect(
        host="localhost",
        port=5432,
        database="xrp_timeseries",
        user="xrp_user",
        password="xrp_password_change_me",
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
    parser = argparse.ArgumentParser(description="Train single HMM and backtest")
    parser.add_argument("--start", type=str, default="2023-01-01", help="Start date")
    parser.add_argument("--end", type=str, default="2024-12-31", help="End date")
    parser.add_argument("--symbol", type=str, default="XRPUSDT", help="Symbol")
    parser.add_argument("--train_split", type=float, default=0.7, help="Train/test split ratio")
    parser.add_argument("--n_states", type=int, default=4, help="Number of HMM states")
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/hmm_backtest"))

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("SINGLE HMM TRAINING & BACKTEST")
    logger.info("=" * 60)
    logger.info(f"Symbol: {args.symbol}")
    logger.info(f"Period: {args.start} to {args.end}")
    logger.info(f"Train split: {args.train_split}")
    logger.info(f"N states: {args.n_states}")

    # Load OHLCV data from TimescaleDB
    logger.info("Loading OHLCV data from TimescaleDB...")
    start_dt = datetime.fromisoformat(args.start)
    end_dt = datetime.fromisoformat(args.end)

    df_3m = load_ohlcv_from_timescaledb(args.symbol, start_dt, end_dt, "3m")

    if df_3m.empty:
        logger.error("No data loaded from TimescaleDB!")
        sys.exit(1)

    logger.info(f"Loaded {len(df_3m)} 3m bars from TimescaleDB")
    logger.info(f"Date range: {df_3m['timestamp'].min()} to {df_3m['timestamp'].max()}")

    # Build multi-timeframe features (15m + 1h)
    logger.info("Building multi-timeframe HMM features (15m + 1h)...")
    features, feature_names, df_features = build_features(df_3m)
    logger.info(f"Features ({len(feature_names)}): {feature_names}")
    logger.info(f"Feature matrix shape: {features.shape}")

    # Train/test split
    n_train = int(len(features) * args.train_split)
    train_features = features[:n_train]
    test_features = features[n_train:]

    train_df = df_features.iloc[:n_train].reset_index(drop=True)
    test_df = df_features.iloc[n_train:].reset_index(drop=True)

    logger.info(f"Train samples: {len(train_features)}")
    logger.info(f"Test samples: {len(test_features)}")

    # Train HMM
    logger.info("=" * 60)
    logger.info("TRAINING HMM")
    logger.info("=" * 60)

    hmm_model = SingleHMM(n_states=args.n_states)
    log_likelihood = hmm_model.train(train_features, feature_names)

    logger.info(f"Log-likelihood: {log_likelihood:.2f}")
    logger.info(f"State labels: {hmm_model.state_labels}")

    # Print transition matrix
    logger.info("\nTransition matrix:")
    for i in range(args.n_states):
        row = hmm_model.model.transmat_[i]
        label = hmm_model.state_labels.get(i, "UNK")
        probs_str = " ".join([f"{p:.3f}" for p in row])
        logger.info(f"  {label:12s}: {probs_str}")

    # Save model
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "hmm_model.json"
    hmm_model.save(model_path)

    # Analyze regime distribution on train set
    logger.info("=" * 60)
    logger.info("REGIME DISTRIBUTION - TRAIN SET")
    logger.info("=" * 60)

    train_results, train_df_labeled = analyze_regime_distribution(train_df, hmm_model, train_features)

    logger.info(f"Total bars: {train_results['total_bars']}")
    logger.info("\nRegime Distribution:")
    for regime, stats in sorted(train_results["regime_distribution"].items(), key=lambda x: -x[1]["pct"]):
        logger.info(f"  {regime:12s}: {stats['count']:6d} bars ({stats['pct']:5.1f}%)")

    logger.info("\nRegime Returns (per bar):")
    for regime, stats in train_results["regime_returns"].items():
        logger.info(
            f"  {regime:12s}: avg={stats['avg_ret_pct']:+.4f}%, "
            f"std={stats['std_ret_pct']:.4f}%, sharpe={stats['sharpe']:.3f}"
        )

    logger.info("\nRegime Confidence:")
    for regime, stats in train_results["confidence_stats"].items():
        logger.info(
            f"  {regime:12s}: avg_conf={stats['avg_confidence']:.3f}, "
            f"high_conf={stats['high_conf_pct']:.1f}%"
        )

    logger.info("\nRegime Persistence (avg bars per run):")
    for regime, stats in train_results["persistence"].items():
        logger.info(
            f"  {regime:12s}: avg={stats['avg_bars']:.1f}, "
            f"max={stats['max_bars']}, runs={stats['n_runs']}"
        )

    logger.info("\nTop Transitions:")
    for trans, count in list(train_results["transitions"].items())[:10]:
        logger.info(f"  {trans}: {count}")

    # Analyze regime distribution on test set
    logger.info("=" * 60)
    logger.info("REGIME DISTRIBUTION - TEST SET (Out-of-Sample)")
    logger.info("=" * 60)

    test_results, test_df_labeled = analyze_regime_distribution(test_df, hmm_model, test_features)

    logger.info(f"Total bars: {test_results['total_bars']}")
    logger.info("\nRegime Distribution:")
    for regime, stats in sorted(test_results["regime_distribution"].items(), key=lambda x: -x[1]["pct"]):
        logger.info(f"  {regime:12s}: {stats['count']:6d} bars ({stats['pct']:5.1f}%)")

    logger.info("\nRegime Returns (per bar):")
    for regime, stats in test_results["regime_returns"].items():
        logger.info(
            f"  {regime:12s}: avg={stats['avg_ret_pct']:+.4f}%, "
            f"std={stats['std_ret_pct']:.4f}%, sharpe={stats['sharpe']:.3f}"
        )

    logger.info("\nRegime Confidence:")
    for regime, stats in test_results["confidence_stats"].items():
        logger.info(
            f"  {regime:12s}: avg_conf={stats['avg_confidence']:.3f}, "
            f"high_conf={stats['high_conf_pct']:.1f}%"
        )

    logger.info("\nRegime Persistence (avg bars per run):")
    for regime, stats in test_results["persistence"].items():
        logger.info(
            f"  {regime:12s}: avg={stats['avg_bars']:.1f}, "
            f"max={stats['max_bars']}, runs={stats['n_runs']}"
        )

    # Save labeled data
    output_path = args.output_dir / "regimes_test.csv"
    test_df_labeled.to_csv(output_path, index=False)
    logger.info(f"\nLabeled data saved to {output_path}")

    # Run backtest on train set
    logger.info("=" * 60)
    logger.info("BACKTEST - TRAIN SET")
    logger.info("=" * 60)

    train_bt = run_hmm_backtest(train_df, hmm_model, train_features)

    logger.info(f"Initial capital: ${train_bt['initial_capital']:,.2f}")
    logger.info(f"Final equity: ${train_bt['final_equity']:,.2f}")
    logger.info(f"Total return: {train_bt['total_return_pct']:.2f}%")
    logger.info(f"N trades: {train_bt['n_trades']}")

    if train_bt["n_trades"] > 0:
        logger.info(f"Win rate: {train_bt['win_rate']:.1f}%")
        logger.info(f"Profit factor: {train_bt['profit_factor']:.2f}")
        logger.info(f"Max drawdown: {train_bt['max_drawdown_pct']:.2f}%")
        logger.info(f"Avg PnL per trade: {train_bt['avg_pnl_pct']:.3f}%")
        logger.info("\nRegime breakdown:")
        for regime, stats in train_bt.get("regime_stats", {}).items():
            logger.info(
                f"  {regime:12s}: {stats['n_trades']:4d} trades, "
                f"WR={stats['win_rate']:.1f}%, PnL=${stats['total_pnl']:.2f}"
            )

    # Run backtest on test set
    logger.info("=" * 60)
    logger.info("BACKTEST - TEST SET (Out-of-Sample)")
    logger.info("=" * 60)

    test_bt = run_hmm_backtest(test_df, hmm_model, test_features)

    logger.info(f"Initial capital: ${test_bt['initial_capital']:,.2f}")
    logger.info(f"Final equity: ${test_bt['final_equity']:,.2f}")
    logger.info(f"Total return: {test_bt['total_return_pct']:.2f}%")
    logger.info(f"N trades: {test_bt['n_trades']}")

    if test_bt["n_trades"] > 0:
        logger.info(f"Win rate: {test_bt['win_rate']:.1f}%")
        logger.info(f"Profit factor: {test_bt['profit_factor']:.2f}")
        logger.info(f"Max drawdown: {test_bt['max_drawdown_pct']:.2f}%")
        logger.info(f"Avg PnL per trade: {test_bt['avg_pnl_pct']:.3f}%")
        logger.info("\nRegime breakdown:")
        for regime, stats in test_bt.get("regime_stats", {}).items():
            logger.info(
                f"  {regime:12s}: {stats['n_trades']:4d} trades, "
                f"WR={stats['win_rate']:.1f}%, PnL=${stats['total_pnl']:.2f}"
            )

        # Save trades
        trades_path = args.output_dir / "trades_test.csv"
        test_bt["trades"].to_csv(trades_path, index=False)
        logger.info(f"\nTrades saved to {trades_path}")

    # Summary
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"{'Metric':<20} {'Train':>12} {'Test':>12}")
    logger.info("-" * 46)
    logger.info(f"{'Return %':<20} {train_bt['total_return_pct']:>12.2f} {test_bt['total_return_pct']:>12.2f}")
    logger.info(f"{'N Trades':<20} {train_bt['n_trades']:>12d} {test_bt['n_trades']:>12d}")

    if train_bt["n_trades"] > 0 and test_bt["n_trades"] > 0:
        logger.info(f"{'Win Rate %':<20} {train_bt['win_rate']:>12.1f} {test_bt['win_rate']:>12.1f}")
        logger.info(f"{'Profit Factor':<20} {train_bt['profit_factor']:>12.2f} {test_bt['profit_factor']:>12.2f}")
        logger.info(f"{'Max DD %':<20} {train_bt['max_drawdown_pct']:>12.2f} {test_bt['max_drawdown_pct']:>12.2f}")

    # Confirm Layer statistics
    logger.info("\n" + "=" * 60)
    logger.info("CONFIRM LAYER STATISTICS (Test Set)")
    logger.info("=" * 60)

    logger.info("\nRaw -> Confirmed Regime Distribution:")
    raw_dist = test_bt.get("raw_regime_distribution", {})
    conf_dist = test_bt.get("confirm_regime_distribution", {})
    all_regimes = set(raw_dist.keys()) | set(conf_dist.keys())

    logger.info(f"{'Regime':<15} {'Raw':>8} {'Confirmed':>10} {'Change':>10}")
    logger.info("-" * 45)
    for regime in sorted(all_regimes):
        raw_count = raw_dist.get(regime, 0)
        conf_count = conf_dist.get(regime, 0)
        change = conf_count - raw_count
        logger.info(f"{regime:<15} {raw_count:>8} {conf_count:>10} {change:>+10}")

    # Confirm reason breakdown for trades
    if test_bt["n_trades"] > 0 and "confirm_stats" in test_bt:
        logger.info("\nTrades by Confirm Reason:")
        for reason, stats in sorted(test_bt["confirm_stats"].items(), key=lambda x: -x[1]["n_trades"]):
            logger.info(
                f"  {reason:<30}: {stats['n_trades']:4d} trades, "
                f"WR={stats['win_rate']:.1f}%, PnL=${stats['total_pnl']:.2f}"
            )

    logger.info("\nDone!")


if __name__ == "__main__":
    main()
