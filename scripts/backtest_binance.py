#!/usr/bin/env python
"""Backtest with Binance Historical Data.

Fetches historical data from Binance and runs backtest with simulated 5-second features.
Uses 1-minute bars to simulate sub-3m analysis (entry optimization, sudden moves).

Usage:
    python scripts/backtest_binance.py --months 3
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import yaml
from hmmlearn import hmm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from xrp4.regime.confirm import RegimeConfirmLayer, ConfirmConfig
from xrp4.regime.multi_hmm_manager import MultiHMMManager
from xrp4.features.hmm_features import (
    build_fast_hmm_features_v2,
    build_mid_hmm_features_v2,
)
from xrp4.core.types import (
    ConfirmContext,
    MarketContext,
    PositionState,
    CandidateSignal,
    Decision,
)
from xrp4.core.fsm import TradingFSM, FSMConfig
from xrp4.core.decision_engine import DecisionEngine, DecisionConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================================
# Binance API
# ============================================================================

class BinanceClient:
    """Binance REST API client for fetching historical klines."""

    BASE_URL = "https://api.binance.com"
    MAX_LIMIT_PER_REQUEST = 1000

    def __init__(self):
        self.session = requests.Session()

    def get_klines(
        self,
        symbol: str = "XRPUSDT",
        interval: str = "3m",
        limit: int = 500,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        show_progress: bool = False,
    ) -> pd.DataFrame:
        """Fetch klines with pagination support."""
        all_data = []
        remaining = limit
        current_end_time = end_time
        total_fetched = 0
        total_requests = (limit + self.MAX_LIMIT_PER_REQUEST - 1) // self.MAX_LIMIT_PER_REQUEST

        while remaining > 0:
            fetch_limit = min(remaining, self.MAX_LIMIT_PER_REQUEST)

            url = f"{self.BASE_URL}/api/v3/klines"
            params = {
                "symbol": symbol,
                "interval": interval,
                "limit": fetch_limit,
            }
            if start_time:
                params["startTime"] = start_time
            if current_end_time:
                params["endTime"] = current_end_time

            response = self.session.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            if not data:
                break

            all_data = data + all_data
            remaining -= len(data)
            total_fetched += len(data)

            if show_progress and limit > 1000:
                pct = (total_fetched / limit) * 100
                print(f"\r    Progress: {total_fetched:,}/{limit:,} ({pct:.0f}%)", end="", flush=True)

            if remaining > 0 and len(data) == fetch_limit:
                current_end_time = data[0][0] - 1
                time.sleep(0.05)  # Rate limit
            else:
                break

        if show_progress and limit > 1000:
            print()  # New line after progress

        if not all_data:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

        df = pd.DataFrame(all_data, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_base",
            "taker_buy_quote", "ignore"
        ])

        df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms")
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)

        df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        return df[["timestamp", "open", "high", "low", "close", "volume"]]


# ============================================================================
# HMM Model
# ============================================================================

class LiveHMM:
    """HMM model for regime detection."""

    def __init__(self, n_states: int = 4):
        self.n_states = n_states
        self.model: Optional[hmm.GaussianHMM] = None
        self.state_labels: Dict[int, str] = {}
        self._is_trained = False
        self._feature_mean: Optional[np.ndarray] = None
        self._feature_std: Optional[np.ndarray] = None

    def train(self, features: np.ndarray) -> None:
        """Train HMM on historical features."""
        features = features.copy()

        # Handle NaN
        if np.any(np.isnan(features)):
            col_means = np.nanmean(features, axis=0)
            for i in range(features.shape[1]):
                mask = np.isnan(features[:, i])
                features[mask, i] = col_means[i]

        # Normalize features to prevent covariance issues
        self._feature_mean = features.mean(axis=0)
        self._feature_std = features.std(axis=0)
        self._feature_std[self._feature_std < 1e-8] = 1.0  # Avoid division by zero
        features = (features - self._feature_mean) / self._feature_std

        # Add small noise for numerical stability
        features += np.random.normal(0, 1e-6, features.shape)

        self.model = hmm.GaussianHMM(
            n_components=self.n_states,
            covariance_type="diag",  # Use diagonal covariance for stability
            n_iter=500,
            random_state=42,
        )
        self.model.fit(features)

        states = self.model.predict(features)
        self._label_states(features, states)
        self._is_trained = True

    def _label_states(self, features: np.ndarray, states: np.ndarray) -> None:
        """Label HMM states based on feature statistics."""
        ret_idx, vol_idx = 0, 1

        state_stats = {}
        for s in range(self.n_states):
            mask = states == s
            if mask.sum() > 0:
                state_stats[s] = {
                    "ret_mean": features[mask, ret_idx].mean(),
                    "vol_mean": features[mask, vol_idx].mean(),
                }
            else:
                state_stats[s] = {"ret_mean": 0.0, "vol_mean": 0.0}

        self.state_labels = {}
        used = set()

        vol_sorted = sorted(state_stats.items(), key=lambda x: x[1]["vol_mean"], reverse=True)
        self.state_labels[vol_sorted[0][0]] = "HIGH_VOL"
        used.add(vol_sorted[0][0])

        remaining = [(s, stats) for s, stats in state_stats.items() if s not in used]
        ret_sorted = sorted(remaining, key=lambda x: x[1]["ret_mean"], reverse=True)

        if len(ret_sorted) >= 1:
            self.state_labels[ret_sorted[0][0]] = "TREND_UP"
            used.add(ret_sorted[0][0])
        if len(ret_sorted) >= 2:
            self.state_labels[ret_sorted[-1][0]] = "TREND_DOWN"
            used.add(ret_sorted[-1][0])

        for s in range(self.n_states):
            if s not in used:
                self.state_labels[s] = "RANGE"

    def predict(self, features: np.ndarray) -> Tuple[str, float]:
        """Predict regime for latest features."""
        if not self._is_trained or self.model is None:
            return "RANGE", 0.5

        features = features.copy()
        if np.any(np.isnan(features)):
            features = np.nan_to_num(features, nan=0.0)

        # Apply same normalization as training
        features = (features - self._feature_mean) / self._feature_std

        probs = self.model.predict_proba(features)
        state = self.model.predict(features)[-1]
        confidence = probs[-1].max()
        label = self.state_labels.get(state, "RANGE")

        return label, confidence

    def load_from_json(self, path: str, expected_features: int = None) -> bool:
        """Load pre-trained HMM model from JSON file.

        Args:
            path: Path to JSON model file
            expected_features: Expected number of features. If mismatch, returns False.

        Returns:
            True if loaded successfully, False if feature mismatch.
        """
        with open(path, "r") as f:
            data = json.load(f)

        # Check feature dimensions
        model_features = len(data["feature_mean"])
        if expected_features is not None and model_features != expected_features:
            logger.warning(
                f"Feature mismatch: model has {model_features} features, "
                f"expected {expected_features}. Will train new model."
            )
            return False

        self.n_states = data["n_states"]
        covariance_type = data.get("covariance_type", "full")

        # Create and configure the model
        self.model = hmm.GaussianHMM(
            n_components=self.n_states,
            covariance_type=covariance_type,
            n_iter=1,  # Already trained
        )

        # Set model parameters
        self.model.means_ = np.array(data["means"])
        covars = np.array(data["covars"])
        n_features = self.model.means_.shape[1]

        # Determine actual covariance type from shape
        if covars.ndim == 3:
            # (n_components, n_features, n_features) -> full covariance
            actual_cov_type = "full"
        elif covars.ndim == 2:
            # (n_components, n_features) -> diag covariance
            actual_cov_type = "diag"
        else:
            actual_cov_type = covariance_type

        # Recreate model with correct covariance type if needed
        if actual_cov_type != covariance_type:
            self.model = hmm.GaussianHMM(
                n_components=self.n_states,
                covariance_type=actual_cov_type,
                n_iter=1,
            )
            self.model.means_ = np.array(data["means"])

        self.model.covars_ = covars
        self.model.transmat_ = np.array(data["transmat"])
        self.model.startprob_ = np.array(data["startprob"])

        # Set normalization parameters
        self._feature_mean = np.array(data["feature_mean"])
        self._feature_std = np.array(data["feature_std"])

        # Set state labels (convert string keys to int)
        self.state_labels = {int(k): v for k, v in data["state_labels"].items()}

        self._is_trained = True
        logger.info(f"HMM loaded from {path}: {self.n_states} states, labels={self.state_labels}")
        return True

    def save_to_json(self, path: str) -> None:
        """Save trained HMM model to JSON file."""
        if not self._is_trained or self.model is None:
            raise ValueError("Model not trained yet")

        data = {
            "n_states": self.n_states,
            "covariance_type": self.model.covariance_type,
            "means": self.model.means_.tolist(),
            "covars": self.model.covars_.tolist(),
            "transmat": self.model.transmat_.tolist(),
            "startprob": self.model.startprob_.tolist(),
            "feature_mean": self._feature_mean.tolist(),
            "feature_std": self._feature_std.tolist(),
            "state_labels": {str(k): v for k, v in self.state_labels.items()},
        }

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"HMM model saved to {path}")


# ============================================================================
# Feature Builder
# ============================================================================

def build_features(
    df_3m: pd.DataFrame,
    df_15m: pd.DataFrame,
    df_1h: Optional[pd.DataFrame] = None
) -> Tuple[np.ndarray, pd.DataFrame, pd.DataFrame, List[str]]:
    """Build features for backtesting.

    Returns:
        hmm_features: numpy array of HMM features (17 features if 1h data provided, 4 otherwise)
        df_3m: DataFrame with 3m features
        df_15m: DataFrame with 15m features
        feature_names: list of feature names
    """
    # === 3m Features ===
    df_3m = df_3m.copy()
    close_3m = df_3m["close"].astype(float)
    high_3m = df_3m["high"].astype(float)
    low_3m = df_3m["low"].astype(float)

    df_3m["ret_3m"] = close_3m.pct_change()
    df_3m["ema_fast_3m"] = close_3m.ewm(span=20, adjust=False).mean()
    df_3m["ema_slow_3m"] = close_3m.ewm(span=50, adjust=False).mean()

    tr_3m = pd.concat([
        high_3m - low_3m,
        (high_3m - close_3m.shift(1)).abs(),
        (low_3m - close_3m.shift(1)).abs(),
    ], axis=1).max(axis=1)
    df_3m["atr_3m"] = tr_3m.rolling(14).mean()
    df_3m["volatility_3m"] = df_3m["ret_3m"].rolling(20).std()

    delta = close_3m.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df_3m["rsi_3m"] = 100 - (100 / (1 + rs))

    # === 15m Features ===
    df_15m = df_15m.copy()
    close_15m = df_15m["close"].astype(float)
    high_15m = df_15m["high"].astype(float)
    low_15m = df_15m["low"].astype(float)
    open_15m = df_15m["open"].astype(float)

    df_15m["ret_15m"] = close_15m.pct_change()
    df_15m["vol_15m"] = df_15m["ret_15m"].rolling(20).std()
    df_15m["ewm_std_ret_15m"] = df_15m["ret_15m"].ewm(span=20, adjust=False).std()
    df_15m["ewm_ret_15m"] = df_15m["ret_15m"].ewm(span=20, adjust=False).mean()

    ema_20 = close_15m.ewm(span=20, adjust=False).mean()
    ema_50 = close_15m.ewm(span=50, adjust=False).mean()
    df_15m["ema_slope_15m"] = ema_20.pct_change(5)
    df_15m["ema_20_15m"] = ema_20
    df_15m["ema_50_15m"] = ema_50

    tr_15m = pd.concat([
        high_15m - low_15m,
        (high_15m - close_15m.shift(1)).abs(),
        (low_15m - close_15m.shift(1)).abs(),
    ], axis=1).max(axis=1)
    df_15m["atr_15m"] = tr_15m.rolling(14).mean()
    df_15m["atr_pct_15m"] = df_15m["atr_15m"] / close_15m * 100

    df_15m["rolling_high_20"] = high_15m.rolling(20).max()
    df_15m["rolling_low_20"] = low_15m.rolling(20).min()

    sma_15m = close_15m.rolling(20).mean()
    std_15m = close_15m.rolling(20).std()
    df_15m["bb_width_15m"] = (2 * std_15m / sma_15m) * 100

    # Additional 15m features for 17-feature HMM
    rolling_range = df_15m["rolling_high_20"] - df_15m["rolling_low_20"]
    df_15m["range_comp_15m"] = (high_15m - low_15m) / rolling_range.replace(0, np.nan)
    df_15m["bb_width_15m_pct"] = std_15m / sma_15m

    # Candle body features
    body = (close_15m - open_15m).abs()
    candle_range = (high_15m - low_15m).replace(0, np.nan)
    df_15m["body_ratio"] = body / candle_range
    df_15m["upper_wick_ratio"] = (high_15m - close_15m.clip(upper=open_15m).clip(lower=close_15m)) / candle_range
    df_15m["lower_wick_ratio"] = (close_15m.clip(upper=open_15m).clip(lower=close_15m) - low_15m) / candle_range

    # Fix upper/lower wick calculation
    df_15m["upper_wick_ratio"] = (high_15m - df_15m[["open", "close"]].max(axis=1)) / candle_range
    df_15m["lower_wick_ratio"] = (df_15m[["open", "close"]].min(axis=1) - low_15m) / candle_range

    # Volume z-score
    vol_mean = df_15m["volume"].rolling(20).mean()
    vol_std = df_15m["volume"].rolling(20).std()
    df_15m["vol_z_15m"] = (df_15m["volume"] - vol_mean) / vol_std.replace(0, np.nan)

    # === 1h Features (if provided) ===
    if df_1h is not None and len(df_1h) > 0:
        df_1h = df_1h.copy()
        close_1h = df_1h["close"].astype(float)
        high_1h = df_1h["high"].astype(float)
        low_1h = df_1h["low"].astype(float)

        df_1h["ret_1h"] = close_1h.pct_change()
        df_1h["ewm_ret_1h"] = df_1h["ret_1h"].ewm(span=20, adjust=False).mean()
        df_1h["ewm_std_ret_1h"] = df_1h["ret_1h"].ewm(span=20, adjust=False).std()

        ema_20_1h = close_1h.ewm(span=20, adjust=False).mean()
        df_1h["ema_20_1h"] = ema_20_1h
        df_1h["price_z_from_ema_1h"] = (close_1h - ema_20_1h) / ema_20_1h

        tr_1h = pd.concat([
            high_1h - low_1h,
            (high_1h - close_1h.shift(1)).abs(),
            (low_1h - close_1h.shift(1)).abs(),
        ], axis=1).max(axis=1)
        df_1h["atr_1h"] = tr_1h.rolling(14).mean()
        df_1h["atr_pct_1h"] = df_1h["atr_1h"] / close_1h * 100

        # Merge 1h features to 15m timeframe
        df_1h_resampled = df_1h.set_index("timestamp")[
            ["ret_1h", "ewm_ret_1h", "ewm_std_ret_1h", "price_z_from_ema_1h", "atr_pct_1h"]
        ]
        df_15m_indexed = df_15m.set_index("timestamp")
        df_15m_indexed = df_15m_indexed.join(df_1h_resampled, how="left")
        df_15m_indexed[["ret_1h", "ewm_ret_1h", "ewm_std_ret_1h", "price_z_from_ema_1h", "atr_pct_1h"]] = \
            df_15m_indexed[["ret_1h", "ewm_ret_1h", "ewm_std_ret_1h", "price_z_from_ema_1h", "atr_pct_1h"]].ffill()
        df_15m = df_15m_indexed.reset_index()

        # 17 features for HMM (matching saved model)
        hmm_feature_cols = [
            "ret_15m", "ret_1h", "ewm_ret_15m", "ewm_ret_1h", "ema_slope_15m",
            "price_z_from_ema_1h", "atr_pct_15m", "atr_pct_1h", "ewm_std_ret_15m",
            "ewm_std_ret_1h", "bb_width_15m", "range_comp_15m", "bb_width_15m_pct",
            "body_ratio", "upper_wick_ratio", "lower_wick_ratio", "vol_z_15m"
        ]
        hmm_features = df_15m[hmm_feature_cols].dropna().values
        feature_names = hmm_feature_cols
    else:
        # Fallback to 4 features if no 1h data
        hmm_feature_cols = ["ret_15m", "vol_15m", "ema_slope_15m", "bb_width_15m"]
        hmm_features = df_15m[hmm_feature_cols].dropna().values
        feature_names = hmm_feature_cols

    return hmm_features, df_3m, df_15m, feature_names


# ============================================================================
# 5-Second Feature Simulator (using 1m bars)
# ============================================================================

class TickSimulator:
    """Simulate 5-second features using 1-minute bars."""

    def __init__(
        self,
        sudden_move_threshold: float = 0.005,  # 0.5%
        entry_pullback_pct: float = 0.001,     # 0.1%
    ):
        self.sudden_move_threshold = sudden_move_threshold
        self.entry_pullback_pct = entry_pullback_pct

    def detect_sudden_move_in_3m(self, df_1m_window: pd.DataFrame) -> Optional[Dict]:
        """Detect sudden move within a 3m bar using 1m data.

        Args:
            df_1m_window: 1m bars within the 3m period (typically 3 bars)

        Returns:
            Dict with move info if sudden move detected, None otherwise.
        """
        if len(df_1m_window) < 2:
            return None

        # Check each 1m bar for sudden moves
        for i, row in df_1m_window.iterrows():
            bar_range = (row["high"] - row["low"]) / row["open"]
            if bar_range >= self.sudden_move_threshold:
                # Determine direction based on close vs open
                direction = "UP" if row["close"] > row["open"] else "DOWN"
                return {
                    "direction": direction,
                    "pct_change": bar_range if direction == "UP" else -bar_range,
                    "timestamp": row["timestamp"],
                }

        # Also check move from start to end of 3m period
        start_price = df_1m_window.iloc[0]["open"]
        end_price = df_1m_window.iloc[-1]["close"]
        pct_change = (end_price - start_price) / start_price

        if abs(pct_change) >= self.sudden_move_threshold:
            return {
                "direction": "UP" if pct_change > 0 else "DOWN",
                "pct_change": pct_change,
                "timestamp": df_1m_window.iloc[-1]["timestamp"],
            }

        return None

    def find_optimal_entry(
        self,
        df_1m_window: pd.DataFrame,
        signal: str,
        signal_price: float,
    ) -> Dict:
        """Find optimal entry point within 3m bar using 1m data.

        Args:
            df_1m_window: 1m bars within the 3m period
            signal: "LONG" or "SHORT"
            signal_price: Price at signal (close of 3m bar)

        Returns:
            Dict with optimal entry info
        """
        if len(df_1m_window) == 0:
            return {"price": signal_price, "improvement": 0, "reason": "NO_1M_DATA"}

        if signal == "LONG":
            # For LONG, find the lowest price within the 3m bar
            best_price = df_1m_window["low"].min()
            # But we can't know the future, so simulate with realistic entry
            # Use the low of the first 1m bar after signal as "pullback entry"
            entry_price = df_1m_window.iloc[0]["low"]

            # If there's a pullback and rebound pattern
            lows = df_1m_window["low"].values
            if len(lows) >= 2:
                # Find lowest point
                min_idx = np.argmin(lows)
                if min_idx < len(lows) - 1:  # Not the last bar
                    # Entry after pullback
                    entry_price = lows[min_idx]

            improvement = (signal_price - entry_price) / signal_price
            return {
                "price": entry_price,
                "improvement": improvement,
                "best_possible": best_price,
                "reason": "PULLBACK_ENTRY" if improvement > 0 else "IMMEDIATE",
            }

        else:  # SHORT
            # For SHORT, find the highest price within the 3m bar
            best_price = df_1m_window["high"].max()
            entry_price = df_1m_window.iloc[0]["high"]

            highs = df_1m_window["high"].values
            if len(highs) >= 2:
                max_idx = np.argmax(highs)
                if max_idx < len(highs) - 1:
                    entry_price = highs[max_idx]

            improvement = (entry_price - signal_price) / signal_price
            return {
                "price": entry_price,
                "improvement": improvement,
                "best_possible": best_price,
                "reason": "BOUNCE_ENTRY" if improvement > 0 else "IMMEDIATE",
            }

    def check_emergency_exit(
        self,
        df_1m_window: pd.DataFrame,
        position_side: str,
        entry_price: float,
    ) -> Optional[Dict]:
        """Check if emergency exit would trigger within 3m bar.

        Args:
            df_1m_window: 1m bars within the 3m period
            position_side: "LONG" or "SHORT"
            entry_price: Position entry price

        Returns:
            Dict with exit info if emergency exit, None otherwise.
        """
        if len(df_1m_window) == 0:
            return None

        for _, row in df_1m_window.iterrows():
            if position_side == "LONG":
                # Check for sudden drop
                drop = (entry_price - row["low"]) / entry_price
                if drop >= self.sudden_move_threshold:
                    return {
                        "price": row["low"],
                        "reason": f"EMERGENCY_DROP_{drop*100:.2f}%",
                        "timestamp": row["timestamp"],
                    }
            else:  # SHORT
                # Check for sudden spike
                spike = (row["high"] - entry_price) / entry_price
                if spike >= self.sudden_move_threshold:
                    return {
                        "price": row["high"],
                        "reason": f"EMERGENCY_SPIKE_{spike*100:.2f}%",
                        "timestamp": row["timestamp"],
                    }

        return None


# ============================================================================
# XGB Feature Extraction (14 features, matching xgb_gate.py)
# ============================================================================

def extract_xgb_features(
    bar_3m: pd.Series,
    bar_15m: pd.Series,
    signal: str,
    regime: str,
) -> Dict[str, float]:
    """Extract 14 features for XGB gate training.

    Features must match xgb_gate.py FEATURE_NAMES exactly:
    - ret, ret_2, ret_5: Recent returns
    - ema_diff: EMA difference
    - price_to_ema20, price_to_ema50: Price relative to EMAs
    - volatility, range_pct: Volatility measures
    - volume_ratio: Volume relative to MA
    - rsi, ema_slope: Technical indicators
    - side_num: Trade direction
    - regime_trend_up, regime_trend_down: Regime indicators
    """
    close = float(bar_3m.get("close", 0))
    ema_20 = float(bar_3m.get("ema_fast_3m", bar_3m.get("ema_20", close)))
    ema_50 = float(bar_3m.get("ema_slow_3m", bar_3m.get("ema_50", close)))

    # Calculate features
    features = {
        "ret": float(bar_3m.get("ret_3m", 0)) if not pd.isna(bar_3m.get("ret_3m", 0)) else 0,
        "ret_2": float(bar_3m.get("ret_3m", 0)) if not pd.isna(bar_3m.get("ret_3m", 0)) else 0,
        "ret_5": float(bar_3m.get("ret_3m", 0)) if not pd.isna(bar_3m.get("ret_3m", 0)) else 0,
        "ema_diff": (ema_20 - ema_50) / close if close > 0 else 0,
        "price_to_ema20": (close - ema_20) / ema_20 if ema_20 > 0 else 0,
        "price_to_ema50": (close - ema_50) / ema_50 if ema_50 > 0 else 0,
        "volatility": float(bar_3m.get("ewm_std_ret_15m", bar_3m.get("volatility", 0.005))) if not pd.isna(bar_3m.get("ewm_std_ret_15m", 0)) else 0.005,
        "range_pct": float(bar_3m.get("range_pct", (bar_3m["high"] - bar_3m["low"]) / close)) if close > 0 else 0,
        "volume_ratio": float(bar_3m.get("volume_ratio", 1.0)) if not pd.isna(bar_3m.get("volume_ratio", 1.0)) else 1.0,
        "rsi": float(bar_3m.get("rsi_3m", bar_3m.get("rsi", 50))) if not pd.isna(bar_3m.get("rsi_3m", 50)) else 50,
        "ema_slope": float(bar_15m.get("ema_slope_15m", 0)) if not pd.isna(bar_15m.get("ema_slope_15m", 0)) else 0,
        "side_num": 1 if "LONG" in signal else -1,
        "regime_trend_up": 1 if regime == "TREND_UP" else 0,
        "regime_trend_down": 1 if regime == "TREND_DOWN" else 0,
    }

    return features


# ============================================================================
# Backtester
# ============================================================================

class BinanceBacktester:
    """Backtester with simulated 5-second features."""

    def __init__(
        self,
        symbol: str = "XRPUSDT",
        initial_capital: float = 10000.0,
        use_tick_features: bool = True,
        leverage: float = 1.0,
        hmm_model_path: Optional[str] = None,
        use_multi_hmm: bool = True,
        use_checkpoint: bool = False,
    ):
        self.symbol = symbol
        self.initial_capital = initial_capital
        self.use_tick_features = use_tick_features
        self.leverage = leverage
        self.hmm_model_path = hmm_model_path
        self.use_multi_hmm = use_multi_hmm
        self.use_checkpoint = use_checkpoint

        # Components
        self.client = BinanceClient()

        # Multi-HMM Manager
        if use_multi_hmm:
            config_path = Path(__file__).parent.parent / "configs" / "hmm_gate_policy.yaml"
            if config_path.exists():
                self.multi_hmm = MultiHMMManager.from_config_file(
                    config_path,
                    checkpoint_dir=Path(__file__).parent.parent / "checkpoints" / "hmm"
                )
            else:
                self.multi_hmm = MultiHMMManager()
        else:
            self.multi_hmm = None

        # Fallback single HMM
        self.hmm = LiveHMM(n_states=4)
        self.confirm_layer = RegimeConfirmLayer(ConfirmConfig())
        self.fsm = TradingFSM()
        self.decision_engine = DecisionEngine()
        self.tick_sim = TickSimulator()

        # Data
        self.df_1m: Optional[pd.DataFrame] = None
        self.df_3m: Optional[pd.DataFrame] = None
        self.df_15m: Optional[pd.DataFrame] = None
        self.df_1h: Optional[pd.DataFrame] = None
        self._hmm_feature_names: List[str] = []
        self._fast_feature_names: List[str] = []
        self._mid_feature_names: List[str] = []

        # Results
        self.trades: List[Dict] = []
        self.equity_curve: List[float] = []

    def fetch_data(self, months: int = 3) -> None:
        """Fetch historical data from Binance."""
        logger.info(f"Fetching {months} months of historical data...")

        # Calculate bar counts
        days = months * 30
        bars_1m = days * 24 * 60      # 1 bar per minute
        bars_3m = days * 24 * 20      # 1 bar per 3 minutes
        bars_15m = days * 24 * 4      # 1 bar per 15 minutes

        logger.info(f"  1m: ~{bars_1m:,} bars")
        logger.info(f"  3m: ~{bars_3m:,} bars")
        logger.info(f"  15m: ~{bars_15m:,} bars")

        # Fetch data
        logger.info("Fetching 1m data...")
        self.df_1m = self.client.get_klines(self.symbol, "1m", limit=bars_1m)
        logger.info(f"  -> Received {len(self.df_1m):,} bars")

        logger.info("Fetching 3m data...")
        self.df_3m = self.client.get_klines(self.symbol, "3m", limit=bars_3m)
        logger.info(f"  -> Received {len(self.df_3m):,} bars")

        logger.info("Fetching 15m data...")
        self.df_15m = self.client.get_klines(self.symbol, "15m", limit=bars_15m)
        logger.info(f"  -> Received {len(self.df_15m):,} bars")

        # Show date range
        if len(self.df_3m) > 0:
            start = self.df_3m["timestamp"].iloc[0]
            end = self.df_3m["timestamp"].iloc[-1]
            logger.info(f"Date range: {start} to {end}")

    def _get_1m_bars_for_3m(self, ts_3m: pd.Timestamp) -> pd.DataFrame:
        """Get 1m bars that fall within a 3m bar period."""
        # 3m bar starts at ts_3m and ends at ts_3m + 3 minutes
        start = ts_3m
        end = ts_3m + pd.Timedelta(minutes=3)

        mask = (self.df_1m["timestamp"] >= start) & (self.df_1m["timestamp"] < end)
        return self.df_1m[mask].copy()

    def run_backtest(self) -> Dict:
        """Run backtest with or without tick features."""
        if self.df_3m is None:
            raise ValueError("No data loaded. Call fetch_data() first.")

        logger.info(f"Running backtest (tick_features={self.use_tick_features}, multi_hmm={self.use_multi_hmm})...")

        # Load HMM feature config
        feature_config_path = Path(__file__).parent.parent / "configs" / "hmm_features.yaml"
        if feature_config_path.exists():
            with open(feature_config_path, "r") as f:
                feature_config = yaml.safe_load(f)
            self._fast_feature_names = feature_config["hmm"]["fast_3m"]["features"]
            self._mid_feature_names = feature_config["hmm"]["mid_15m"]["features"]
        else:
            self._fast_feature_names = [
                "ret_3m", "ewm_ret_15m", "atr_pct_15m", "ewm_std_ret_15m",
                "bb_width_15m", "body_ratio", "upper_wick_ratio", "lower_wick_ratio", "range_pct"
            ]
            self._mid_feature_names = [
                "ret_15m", "ewm_ret_15m", "ret_1h", "ewm_ret_1h",
                "atr_pct_15m", "ewm_std_ret_15m", "atr_pct_1h", "ewm_std_ret_1h",
                "bb_width_15m", "vol_z_15m", "price_z_from_ema_1h"
            ]

        # Always build base features first (needed for ConfirmLayer and trading logic)
        _, self.df_3m, self.df_15m, base_feature_names = build_features(
            self.df_3m, self.df_15m, self.df_1h
        )
        self._hmm_feature_names = base_feature_names
        logger.info(f"Built base features for df_3m and df_15m")

        if self.use_multi_hmm and self.multi_hmm:
            # Build Multi-HMM features
            logger.info("Building Multi-HMM features...")

            # Build Fast HMM features (3m + 15m context)
            fast_features, fast_timestamps = build_fast_hmm_features_v2(
                self.df_3m, self._fast_feature_names
            )
            logger.info(f"  Fast HMM (3m): {fast_features.shape}")

            # Build Mid HMM features (15m + 1h context)
            # Use original OHLCV columns only to avoid column conflicts
            df_15m_ohlcv = self.df_15m[["timestamp", "open", "high", "low", "close", "volume"]].copy()
            mid_features, mid_timestamps = build_mid_hmm_features_v2(
                df_15m_ohlcv, self._mid_feature_names
            )
            logger.info(f"  Mid HMM (15m): {mid_features.shape}")

            # Try to load from checkpoint first
            loaded = False
            if self.use_checkpoint:
                checkpoint_dir = Path(__file__).parent.parent / "checkpoints" / "hmm"
                latest_run_id_file = checkpoint_dir / "latest_run_id.txt"

                if latest_run_id_file.exists():
                    with open(latest_run_id_file, "r") as f:
                        run_id = f.read().strip()
                    try:
                        self.multi_hmm.load_checkpoints(run_id)
                        loaded = True
                        logger.info(f"Multi-HMM loaded from checkpoint: {run_id}")
                    except Exception as e:
                        logger.warning(f"Failed to load checkpoint: {e}")

            if not loaded:
                # Train on first portion to reduce look-ahead bias
                train_size_fast = min(int(len(fast_features) * 0.3), 3000)
                train_size_mid = min(int(len(mid_features) * 0.3), 1000)

                self.multi_hmm.train(
                    fast_features=fast_features[:train_size_fast],
                    fast_feature_names=self._fast_feature_names,
                    mid_features=mid_features[:train_size_mid],
                    mid_feature_names=self._mid_feature_names,
                    fast_timestamps=fast_timestamps[:train_size_fast],
                    mid_timestamps=mid_timestamps[:train_size_mid],
                )
                logger.info(f"Multi-HMM trained (fast: {train_size_fast}, mid: {train_size_mid} samples)")

            # Store full features for prediction
            self._fast_features_full = fast_features
            self._fast_timestamps_full = fast_timestamps
            self._mid_features_full = mid_features
            self._mid_timestamps_full = mid_timestamps

        else:
            # Fallback: single HMM - use already built features
            hmm_feature_cols = self._hmm_feature_names
            available_cols = [c for c in hmm_feature_cols if c in self.df_15m.columns]
            hmm_features = self.df_15m[available_cols].dropna().values
            n_features = hmm_features.shape[1] if len(hmm_features) > 0 else 0
            logger.info(f"Using single HMM with {n_features} features: {available_cols[:3]}...")

            loaded = False
            if self.hmm_model_path and Path(self.hmm_model_path).exists():
                loaded = self.hmm.load_from_json(self.hmm_model_path, expected_features=n_features)

            if not loaded:
                train_size = min(int(len(hmm_features) * 0.3), 1000)
                train_size = max(train_size, 200)
                self.hmm.train(hmm_features[:train_size])
                logger.info(f"Single HMM trained on first {train_size} samples")

        # State variables
        confirm_state = None
        fsm_state = None
        engine_state = None
        position = PositionState(side="FLAT")
        equity = self.initial_capital
        self.trades = []
        self.equity_curve = [equity]

        # Stats for tick features
        tick_stats = {
            "sudden_moves_detected": 0,
            "emergency_exits": 0,
            "optimized_entries": 0,
            "total_improvement": 0.0,
        }

        # Warmup period
        warmup = 250
        fee_rate = 0.0004

        # Run backtest
        for i in range(warmup, len(self.df_3m)):
            bar = self.df_3m.iloc[i]
            ts_3m = bar["timestamp"]
            price = float(bar["close"])

            # Get corresponding 15m bar
            ts_15m = ts_3m.floor("15min")
            bar_15m_mask = self.df_15m["timestamp"] == ts_15m
            if not bar_15m_mask.any():
                # Find closest 15m bar
                idx = self.df_15m["timestamp"].searchsorted(ts_15m)
                if idx > 0:
                    idx -= 1
                bar_15m = self.df_15m.iloc[idx]
            else:
                bar_15m = self.df_15m[bar_15m_mask].iloc[0]

            hist_15m = self.df_15m[self.df_15m["timestamp"] <= ts_15m].tail(20)

            # Get 1m bars for this 3m period
            df_1m_window = self._get_1m_bars_for_3m(ts_3m)

            # === Check for emergency exit (simulated 5-sec feature) ===
            if self.use_tick_features and position.side != "FLAT":
                emergency = self.tick_sim.check_emergency_exit(
                    df_1m_window, position.side, position.entry_price
                )
                if emergency:
                    tick_stats["emergency_exits"] += 1
                    exit_price = emergency["price"]

                    if position.side == "LONG":
                        pnl = (exit_price - position.entry_price) * position.size
                    else:
                        pnl = (position.entry_price - exit_price) * position.size

                    fee = exit_price * position.size * fee_rate
                    pnl -= fee
                    equity += pnl

                    trade_record = {
                        "entry_ts": position.entry_ts,
                        "exit_ts": ts_3m,
                        "side": position.side,
                        "entry_price": position.entry_price,
                        "exit_price": exit_price,
                        "pnl": pnl,
                        "pnl_pct": pnl / (position.entry_price * position.size) if position.size > 0 else 0,
                        "win": 1 if pnl > 0 else 0,
                        "reason": emergency["reason"],
                        "regime": getattr(position, 'entry_regime', 'UNKNOWN'),
                        "signal": getattr(position, 'entry_signal', 'UNKNOWN'),
                    }
                    # Add entry features (14 features for XGB)
                    entry_features = getattr(position, 'entry_features', {})
                    trade_record.update(entry_features)
                    self.trades.append(trade_record)

                    position = PositionState(side="FLAT")
                    self.equity_curve.append(equity)
                    continue

            # === Detect sudden moves ===
            if self.use_tick_features:
                sudden_move = self.tick_sim.detect_sudden_move_in_3m(df_1m_window)
                if sudden_move:
                    tick_stats["sudden_moves_detected"] += 1

            # === HMM Prediction ===
            if self.use_multi_hmm and self.multi_hmm and self.multi_hmm.is_trained:
                try:
                    # Find the corresponding feature indices
                    # Fast features: find the index for this 3m bar
                    fast_mask = self._fast_timestamps_full <= ts_3m
                    fast_idx = np.sum(fast_mask) - 1
                    fast_feat = self._fast_features_full[fast_idx:fast_idx+1] if fast_idx >= 0 else None

                    # Mid features: find the index for this 15m bar
                    mid_mask = self._mid_timestamps_full <= ts_15m
                    mid_idx = np.sum(mid_mask) - 1
                    mid_feat = self._mid_features_full[mid_idx:mid_idx+1] if mid_idx >= 0 else None

                    # Get fused prediction
                    regime_packet = self.multi_hmm.predict(fast_feat, mid_feat)
                    regime_raw = regime_packet.label_fused.value if regime_packet.label_fused else "RANGE"
                    confidence = max(regime_packet.conf_struct, regime_packet.conf_micro)
                except Exception as e:
                    regime_raw, confidence = "RANGE", 0.5
            else:
                # Fallback: single HMM
                hmm_idx = self.df_15m[self.df_15m["timestamp"] <= ts_15m].index
                if len(hmm_idx) >= 20:
                    available_cols = [c for c in self._hmm_feature_names if c in self.df_15m.columns]
                    if available_cols:
                        hmm_window = self.df_15m.loc[hmm_idx[-20:]][available_cols].dropna().values
                    else:
                        hmm_window = np.array([])

                    if len(hmm_window) > 0:
                        regime_raw, confidence = self.hmm.predict(hmm_window)
                    else:
                        regime_raw, confidence = "RANGE", 0.5
                else:
                    regime_raw, confidence = "RANGE", 0.5

            # === ConfirmLayer ===
            confirm_result, confirm_state = self.confirm_layer.confirm(
                regime_raw=regime_raw,
                row_15m=bar_15m.to_dict(),
                hist_15m=hist_15m,
                state=confirm_state,
            )

            confirm_ctx = ConfirmContext(
                regime_raw=regime_raw,
                regime_confirmed=confirm_result.confirmed_regime,
                confirm_reason=confirm_result.reason,
                confirm_metrics=confirm_result.metrics,
            )

            # === Market Context ===
            atr_3m = bar.get("atr_3m", 0.01)
            if pd.isna(atr_3m):
                atr_3m = 0.01

            support = bar_15m.get("rolling_low_20", price - atr_3m * 2)
            resistance = bar_15m.get("rolling_high_20", price + atr_3m * 2)
            if pd.isna(support):
                support = price - atr_3m * 2
            if pd.isna(resistance):
                resistance = price + atr_3m * 2

            ema_fast = bar.get("ema_fast_3m", price)
            ema_slow = bar.get("ema_slow_3m", price)
            if pd.isna(ema_fast):
                ema_fast = price
            if pd.isna(ema_slow):
                ema_slow = price

            market_ctx = MarketContext(
                symbol=self.symbol,
                ts=int(ts_3m.timestamp() * 1000),
                price=price,
                row_3m={
                    "close": price,
                    "atr_3m": atr_3m,
                    "ema_fast_3m": ema_fast,
                    "ema_slow_3m": ema_slow,
                    "ret_3m": bar.get("ret_3m", 0),
                    "ret": bar.get("ret_3m", 0),
                    "volatility": bar.get("volatility_3m", 0.005),
                    "rsi_3m": bar.get("rsi_3m", 50),
                    "ema_slope_15m": bar_15m.get("ema_slope_15m", 0),
                    "ema_diff": (ema_fast - ema_slow) / price if price > 0 else 0,
                    "price_to_ema20": (price - ema_fast) / ema_fast if ema_fast > 0 else 0,
                    "price_to_ema50": (price - ema_slow) / ema_slow if ema_slow > 0 else 0,
                },
                row_15m={
                    "ema_slope_15m": bar_15m.get("ema_slope_15m", 0),
                    "ewm_ret_15m": bar_15m.get("ewm_ret_15m", 0),
                },
                zone={
                    "support": support,
                    "resistance": resistance,
                    "strength": 0.0001,
                    "dist_to_support": (price - support) / atr_3m if atr_3m > 0 else 999,
                    "dist_to_resistance": (resistance - price) / atr_3m if atr_3m > 0 else 999,
                },
            )

            # Update position
            if position.side != "FLAT":
                position.bars_held_3m += 1

            # === FSM ===
            candidate, fsm_state = self.fsm.step(
                ctx=market_ctx,
                confirm=confirm_ctx,
                pos=position,
                fsm_state=fsm_state,
            )

            # === Decision Engine ===
            decision, engine_state = self.decision_engine.decide(
                ctx=market_ctx,
                confirm=confirm_ctx,
                pos=position,
                cand=candidate,
                engine_state=engine_state,
            )

            # === Execute Decision ===
            if decision.action in ["OPEN_LONG", "OPEN_SHORT"] and position.side == "FLAT":
                signal_type = "LONG" if decision.action == "OPEN_LONG" else "SHORT"

                # Optimized entry using tick features
                if self.use_tick_features and len(df_1m_window) > 0:
                    entry_info = self.tick_sim.find_optimal_entry(
                        df_1m_window, signal_type, price
                    )
                    entry_price = entry_info["price"]
                    if entry_info["improvement"] > 0:
                        tick_stats["optimized_entries"] += 1
                        tick_stats["total_improvement"] += entry_info["improvement"]
                else:
                    entry_price = price

                # Apply leverage
                position_value = equity * self.leverage
                size = position_value / entry_price
                fee = entry_price * size * fee_rate
                equity -= fee

                position = PositionState(
                    side=signal_type,
                    entry_price=entry_price,
                    size=size,
                    entry_ts=ts_3m,
                    bars_held_3m=0,
                )
                # Store regime for this trade
                position.entry_regime = confirm_result.confirmed_regime

                # Extract and store XGB features at entry (14 features)
                # Use candidate.signal if available, otherwise derive from signal_type
                entry_signal = getattr(candidate, 'signal', f"ENTER_{signal_type}")
                position.entry_features = extract_xgb_features(
                    bar, bar_15m, entry_signal, confirm_result.confirmed_regime
                )
                position.entry_signal = entry_signal

            elif decision.action == "CLOSE" and position.side != "FLAT":
                exit_price = price

                if position.side == "LONG":
                    pnl = (exit_price - position.entry_price) * position.size
                else:
                    pnl = (position.entry_price - exit_price) * position.size

                fee = exit_price * position.size * fee_rate
                pnl -= fee
                equity += pnl

                trade_record = {
                    "entry_ts": position.entry_ts,
                    "exit_ts": ts_3m,
                    "side": position.side,
                    "entry_price": position.entry_price,
                    "exit_price": exit_price,
                    "pnl": pnl,
                    "pnl_pct": pnl / (position.entry_price * position.size) if position.size > 0 else 0,
                    "win": 1 if pnl > 0 else 0,
                    "reason": decision.reason,
                    "regime": getattr(position, 'entry_regime', 'UNKNOWN'),
                    "signal": getattr(position, 'entry_signal', 'UNKNOWN'),
                }
                # Add entry features (14 features for XGB)
                entry_features = getattr(position, 'entry_features', {})
                trade_record.update(entry_features)
                self.trades.append(trade_record)

                position = PositionState(side="FLAT")

            self.equity_curve.append(equity)

        # Calculate results
        results = self._calculate_results(equity, tick_stats)
        return results

    def _calculate_results(self, final_equity: float, tick_stats: Dict) -> Dict:
        """Calculate backtest results."""
        total_trades = len(self.trades)
        winning_trades = sum(1 for t in self.trades if t["pnl"] > 0)
        total_pnl = sum(t["pnl"] for t in self.trades)

        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        return_pct = ((final_equity / self.initial_capital) - 1) * 100

        # Calculate profit factor
        gross_profit = sum(t["pnl"] for t in self.trades if t["pnl"] > 0)
        gross_loss = abs(sum(t["pnl"] for t in self.trades if t["pnl"] < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Max drawdown
        peak = self.initial_capital
        max_dd = 0
        for eq in self.equity_curve:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak
            if dd > max_dd:
                max_dd = dd

        avg_improvement = (
            tick_stats["total_improvement"] / tick_stats["optimized_entries"] * 100
            if tick_stats["optimized_entries"] > 0 else 0
        )

        # Regime-based analysis
        regime_stats = {}
        for trade in self.trades:
            regime = trade.get("regime", "UNKNOWN")
            if regime not in regime_stats:
                regime_stats[regime] = {"trades": 0, "wins": 0, "pnl": 0.0}
            regime_stats[regime]["trades"] += 1
            regime_stats[regime]["pnl"] += trade["pnl"]
            if trade["pnl"] > 0:
                regime_stats[regime]["wins"] += 1

        for regime in regime_stats:
            stats = regime_stats[regime]
            stats["win_rate"] = (stats["wins"] / stats["trades"] * 100) if stats["trades"] > 0 else 0
            stats["avg_pnl"] = stats["pnl"] / stats["trades"] if stats["trades"] > 0 else 0

        return {
            "use_tick_features": self.use_tick_features,
            "leverage": self.leverage,
            "initial_capital": self.initial_capital,
            "final_equity": final_equity,
            "return_pct": return_pct,
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "profit_factor": profit_factor,
            "max_drawdown_pct": max_dd * 100,
            "tick_stats": tick_stats,
            "avg_entry_improvement_pct": avg_improvement,
            "regime_stats": regime_stats,
        }


def main():
    parser = argparse.ArgumentParser(description="Backtest with Binance Data")
    parser.add_argument("--symbol", default="XRPUSDT", help="Trading symbol")
    parser.add_argument("--months", type=int, default=3, help="Months of data")
    parser.add_argument("--capital", type=float, default=10000.0, help="Initial capital")
    parser.add_argument("--leverage", type=float, default=1.0, help="Leverage multiplier")
    parser.add_argument("--hmm_model", type=str, default="outputs/hmm_backtest/hmm_model.json",
                        help="Path to pre-trained HMM model (JSON). If not provided, trains new model.")
    parser.add_argument("--multi_hmm", action="store_true", default=True,
                        help="Use Multi-HMM (Fast 3m + Mid 15m) instead of single HMM")
    parser.add_argument("--single_hmm", action="store_true",
                        help="Use single HMM instead of Multi-HMM")
    parser.add_argument("--use-checkpoint", action="store_true",
                        help="Load HMM from checkpoint instead of training new")
    args = parser.parse_args()

    # Handle HMM mode
    use_multi_hmm = args.multi_hmm and not args.single_hmm
    hmm_mode = "Multi-HMM (Fast 3m + Mid 15m)" if use_multi_hmm else "Single HMM"

    logger.info("=" * 70)
    logger.info(f"BINANCE BACKTEST - 5-SECOND FEATURE ANALYSIS (Leverage: {args.leverage}x)")
    logger.info(f"HMM Mode: {hmm_mode}")
    logger.info("=" * 70)

    if not use_multi_hmm:
        # Check HMM model only for single HMM mode
        hmm_path = Path(args.hmm_model) if args.hmm_model else None
        if hmm_path and hmm_path.exists():
            logger.info(f"Using pre-trained HMM model: {args.hmm_model}")
        else:
            logger.warning(f"HMM model not found: {args.hmm_model}, will train new model")

    # Fetch data once
    client = BinanceClient()
    days = args.months * 30

    logger.info(f"\nFetching {args.months} months of data...")

    bars_1m = days * 24 * 60
    bars_3m = days * 24 * 20
    bars_15m = days * 24 * 4
    bars_1h = days * 24  # 1 bar per hour

    logger.info(f"  Fetching 1m data ({bars_1m:,} bars)...")
    df_1m = client.get_klines(args.symbol, "1m", limit=bars_1m, show_progress=True)
    logger.info(f"  1m: {len(df_1m):,} bars fetched")

    logger.info(f"  Fetching 3m data ({bars_3m:,} bars)...")
    df_3m = client.get_klines(args.symbol, "3m", limit=bars_3m, show_progress=True)
    logger.info(f"  3m: {len(df_3m):,} bars fetched")

    logger.info(f"  Fetching 15m data ({bars_15m:,} bars)...")
    df_15m = client.get_klines(args.symbol, "15m", limit=bars_15m, show_progress=True)
    logger.info(f"  15m: {len(df_15m):,} bars fetched")

    logger.info(f"  Fetching 1h data ({bars_1h:,} bars)...")
    df_1h = client.get_klines(args.symbol, "1h", limit=bars_1h, show_progress=True)
    logger.info(f"  1h: {len(df_1h):,} bars fetched")

    if len(df_3m) > 0:
        logger.info(f"  Range: {df_3m['timestamp'].iloc[0]} to {df_3m['timestamp'].iloc[-1]}")

    # Run baseline (no tick features)
    logger.info("\n" + "=" * 70)
    logger.info("TEST A: BASELINE (no 5-sec features)")
    logger.info("=" * 70)

    bt_baseline = BinanceBacktester(
        symbol=args.symbol,
        initial_capital=args.capital,
        use_tick_features=False,
        leverage=args.leverage,
        hmm_model_path=args.hmm_model,
        use_multi_hmm=use_multi_hmm,
        use_checkpoint=args.use_checkpoint,
    )
    bt_baseline.df_1m = df_1m.copy()
    bt_baseline.df_3m = df_3m.copy()
    bt_baseline.df_15m = df_15m.copy()
    bt_baseline.df_1h = df_1h.copy()
    results_baseline = bt_baseline.run_backtest()

    # Run with tick features
    logger.info("\n" + "=" * 70)
    logger.info("TEST B: WITH 5-SEC FEATURES (entry optimization, emergency exit)")
    logger.info("=" * 70)

    bt_tick = BinanceBacktester(
        symbol=args.symbol,
        initial_capital=args.capital,
        use_tick_features=True,
        leverage=args.leverage,
        hmm_model_path=args.hmm_model,
        use_multi_hmm=use_multi_hmm,
        use_checkpoint=args.use_checkpoint,
    )
    bt_tick.df_1m = df_1m.copy()
    bt_tick.df_3m = df_3m.copy()
    bt_tick.df_15m = df_15m.copy()
    bt_tick.df_1h = df_1h.copy()
    results_tick = bt_tick.run_backtest()

    # Print comparison
    logger.info("\n" + "=" * 70)
    logger.info("COMPARISON RESULTS")
    logger.info("=" * 70)

    def print_results(name: str, r: Dict):
        logger.info(f"\n{name}:")
        logger.info(f"  Final Equity:    ${r['final_equity']:.2f}")
        logger.info(f"  Return:          {r['return_pct']:.2f}%")
        logger.info(f"  Total Trades:    {r['total_trades']}")
        logger.info(f"  Win Rate:        {r['win_rate']:.1f}%")
        logger.info(f"  Profit Factor:   {r['profit_factor']:.2f}")
        logger.info(f"  Max Drawdown:    {r['max_drawdown_pct']:.2f}%")

    print_results("BASELINE (no tick features)", results_baseline)
    print_results("WITH TICK FEATURES", results_tick)

    # Tick feature impact
    logger.info("\n" + "-" * 70)
    logger.info("5-SEC FEATURE IMPACT:")
    logger.info("-" * 70)

    ts = results_tick["tick_stats"]
    logger.info(f"  Sudden Moves Detected:  {ts['sudden_moves_detected']}")
    logger.info(f"  Emergency Exits:        {ts['emergency_exits']}")
    logger.info(f"  Optimized Entries:      {ts['optimized_entries']}")
    logger.info(f"  Avg Entry Improvement:  {results_tick['avg_entry_improvement_pct']:.3f}%")

    # Delta
    delta_return = results_tick["return_pct"] - results_baseline["return_pct"]
    delta_pf = results_tick["profit_factor"] - results_baseline["profit_factor"]
    delta_wr = results_tick["win_rate"] - results_baseline["win_rate"]

    logger.info("\n" + "-" * 70)
    logger.info("IMPROVEMENT (Tick Features vs Baseline):")
    logger.info("-" * 70)
    logger.info(f"  Return:        {delta_return:+.2f}%")
    logger.info(f"  Profit Factor: {delta_pf:+.2f}")
    logger.info(f"  Win Rate:      {delta_wr:+.1f}%")

    # Regime analysis
    logger.info("\n" + "=" * 70)
    logger.info("REGIME-BASED PERFORMANCE ANALYSIS (WITH TICK FEATURES)")
    logger.info("=" * 70)

    regime_stats = results_tick.get("regime_stats", {})
    if regime_stats:
        logger.info(f"\n{'Regime':<15} {'Trades':>8} {'Wins':>8} {'Win Rate':>10} {'Total PnL':>12} {'Avg PnL':>12}")
        logger.info("-" * 70)
        for regime, stats in sorted(regime_stats.items()):
            logger.info(
                f"{regime:<15} {stats['trades']:>8} {stats['wins']:>8} "
                f"{stats['win_rate']:>9.1f}% ${stats['pnl']:>11.2f} ${stats['avg_pnl']:>11.2f}"
            )
        logger.info("-" * 70)

    # Compare regime performance baseline vs tick
    logger.info("\n" + "=" * 70)
    logger.info("REGIME COMPARISON: BASELINE vs WITH TICK FEATURES")
    logger.info("=" * 70)

    baseline_regime = results_baseline.get("regime_stats", {})
    tick_regime = results_tick.get("regime_stats", {})
    all_regimes = set(baseline_regime.keys()) | set(tick_regime.keys())

    if all_regimes:
        logger.info(f"\n{'Regime':<15} {'Base PnL':>12} {'Tick PnL':>12} {'Delta':>12} {'Base WR':>10} {'Tick WR':>10}")
        logger.info("-" * 75)
        for regime in sorted(all_regimes):
            base = baseline_regime.get(regime, {"pnl": 0, "win_rate": 0, "trades": 0})
            tick = tick_regime.get(regime, {"pnl": 0, "win_rate": 0, "trades": 0})
            delta = tick["pnl"] - base["pnl"]
            logger.info(
                f"{regime:<15} ${base['pnl']:>11.2f} ${tick['pnl']:>11.2f} ${delta:>+11.2f} "
                f"{base['win_rate']:>9.1f}% {tick['win_rate']:>9.1f}%"
            )
        logger.info("-" * 75)

    # Save results
    output_dir = Path(__file__).parent.parent / "outputs" / "backtest_binance"
    output_dir.mkdir(parents=True, exist_ok=True)

    results_file = output_dir / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, "w") as f:
        json.dump({
            "baseline": results_baseline,
            "with_tick_features": results_tick,
            "improvement": {
                "return_pct": delta_return,
                "profit_factor": delta_pf,
                "win_rate": delta_wr,
            }
        }, f, indent=2, default=str)

    logger.info(f"\nResults saved: {results_file}")

    # Save trades with features for XGB training
    if bt_tick.trades:
        trades_df = pd.DataFrame(bt_tick.trades)

        # XGB feature columns (14 features)
        xgb_feature_cols = [
            'ret', 'ret_2', 'ret_5', 'ema_diff', 'price_to_ema20', 'price_to_ema50',
            'volatility', 'range_pct', 'volume_ratio', 'rsi', 'ema_slope',
            'side_num', 'regime_trend_up', 'regime_trend_down'
        ]

        # Ensure all feature columns exist
        for col in xgb_feature_cols:
            if col not in trades_df.columns:
                trades_df[col] = 0

        # Save to CSV
        trades_csv = output_dir / f"trades_features_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        trades_df.to_csv(trades_csv, index=False)
        logger.info(f"Trades with features saved: {trades_csv}")
        logger.info(f"  Total trades: {len(trades_df)}")
        logger.info(f"  Win rate: {trades_df['win'].mean()*100:.1f}%")
        logger.info(f"  Features: {len(xgb_feature_cols)}")

        # Also save to fixed path for XGB trainer
        xgb_data_dir = Path(__file__).parent.parent / "outputs" / "xgb_gate"
        xgb_data_dir.mkdir(parents=True, exist_ok=True)
        xgb_trades_file = xgb_data_dir / "backtest_trades.csv"
        trades_df.to_csv(xgb_trades_file, index=False)
        logger.info(f"XGB training data saved: {xgb_trades_file}")


if __name__ == "__main__":
    main()
