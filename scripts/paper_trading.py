#!/usr/bin/env python
"""Paper Trading with Binance Real-time Data.

Fetches data from Binance every 5 seconds and runs the FSM pipeline.
Simulates trading without real money.

Usage:
    python scripts/paper_trading.py
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
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Force unbuffered output
import functools
print = functools.partial(print, flush=True)


# ============================================================================
# Binance API
# ============================================================================

class BinanceClient:
    """Simple Binance REST API client for fetching klines."""

    # USDT-M Futures API (changed from Spot API)
    BASE_URL = "https://fapi.binance.com"
    MAX_LIMIT_PER_REQUEST = 1500  # Futures allows up to 1500

    def __init__(self):
        self.session = requests.Session()

    def get_klines(
        self,
        symbol: str = "XRPUSDT",
        interval: str = "3m",
        limit: int = 500,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> pd.DataFrame:
        """Fetch klines (candlestick) data from Binance.

        Args:
            symbol: Trading pair (e.g., "XRPUSDT")
            interval: Kline interval (e.g., "1m", "3m", "5m", "15m", "1h")
            limit: Number of klines to fetch (max 1000 per request, will paginate if more)
            start_time: Start time in milliseconds
            end_time: End time in milliseconds

        Returns:
            DataFrame with OHLCV data
        """
        all_data = []
        remaining = limit
        current_end_time = end_time

        while remaining > 0:
            fetch_limit = min(remaining, self.MAX_LIMIT_PER_REQUEST)

            url = f"{self.BASE_URL}/fapi/v1/klines"
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

            all_data = data + all_data  # Prepend (older data first)
            remaining -= len(data)

            if remaining > 0 and len(data) == fetch_limit:
                # Move end time to before the oldest fetched bar
                current_end_time = data[0][0] - 1  # open_time - 1ms
            else:
                break

        if not all_data:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

        df = pd.DataFrame(all_data, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_base",
            "taker_buy_quote", "ignore"
        ])

        df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms")
        df["open"] = df["open"].astype(float)
        df["high"] = df["high"].astype(float)
        df["low"] = df["low"].astype(float)
        df["close"] = df["close"].astype(float)
        df["volume"] = df["volume"].astype(float)

        # Remove duplicates and sort
        df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

        return df[["timestamp", "open", "high", "low", "close", "volume"]]

    def get_ticker_price(self, symbol: str = "XRPUSDT") -> float:
        """Get current ticker price."""
        url = f"{self.BASE_URL}/fapi/v1/ticker/price"
        params = {"symbol": symbol}
        response = self.session.get(url, params=params)
        response.raise_for_status()
        return float(response.json()["price"])


# ============================================================================
# HMM Model (simplified for paper trading)
# ============================================================================

class LiveHMM:
    """HMM model for live regime detection."""

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

        # Label states
        states = self.model.predict(features)
        self._label_states(features, states)
        self._is_trained = True
        logger.info(f"HMM trained with {features.shape[1]} features, labels: {self.state_labels}")

    def _label_states(self, features: np.ndarray, states: np.ndarray) -> None:
        """Label HMM states based on feature statistics."""
        # Assume features: [ret, vol, ema_slope, ...]
        ret_idx = 0
        vol_idx = 1

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

        # HIGH_VOL = highest volatility
        vol_sorted = sorted(state_stats.items(), key=lambda x: x[1]["vol_mean"], reverse=True)
        self.state_labels[vol_sorted[0][0]] = "HIGH_VOL"
        used.add(vol_sorted[0][0])

        # TREND_UP/DOWN from remaining
        remaining = [(s, stats) for s, stats in state_stats.items() if s not in used]
        ret_sorted = sorted(remaining, key=lambda x: x[1]["ret_mean"], reverse=True)

        if len(ret_sorted) >= 1:
            self.state_labels[ret_sorted[0][0]] = "TREND_UP"
            used.add(ret_sorted[0][0])
        if len(ret_sorted) >= 2:
            self.state_labels[ret_sorted[-1][0]] = "TREND_DOWN"
            used.add(ret_sorted[-1][0])

        # Remaining = RANGE
        for s in range(self.n_states):
            if s not in used:
                self.state_labels[s] = "RANGE"

    def predict(self, features: np.ndarray) -> Tuple[str, float]:
        """Predict regime for latest features."""
        if not self._is_trained or self.model is None:
            return "RANGE", 0.5

        features = features.copy()

        # Handle NaN
        if np.any(np.isnan(features)):
            features = np.nan_to_num(features, nan=0.0)

        # Apply same normalization as training
        if self._feature_mean is not None and self._feature_std is not None:
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

        # Determine actual covariance type from shape
        if covars.ndim == 3:
            actual_cov_type = "full"
        elif covars.ndim == 2:
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
        logger.info(f"HMM loaded from {path}: {self.n_states} states, {model_features} features")
        return True


# ============================================================================
# Feature Builder
# ============================================================================

def build_features_live(
    df_3m: pd.DataFrame,
    df_15m: pd.DataFrame,
    df_1h: Optional[pd.DataFrame] = None
) -> Tuple[np.ndarray, pd.DataFrame, pd.DataFrame, List[str]]:
    """Build features for live trading.

    Returns:
        - HMM features (17 features if 1h data provided, 4 otherwise)
        - df_3m with features
        - df_15m with features
        - feature_names list
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

    # Volatility
    df_3m["volatility_3m"] = df_3m["ret_3m"].rolling(20).std()

    # RSI
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

    df_15m["ret_15m"] = close_15m.pct_change()
    df_15m["vol_15m"] = df_15m["ret_15m"].rolling(20).std()
    df_15m["ewm_std_ret_15m"] = df_15m["ret_15m"].ewm(span=20, adjust=False).std()  # Required by ConfirmLayer

    ema_20 = close_15m.ewm(span=20, adjust=False).mean()
    ema_50 = close_15m.ewm(span=50, adjust=False).mean()
    df_15m["ema_slope_15m"] = ema_20.pct_change(5)
    df_15m["ema_20_15m"] = ema_20
    df_15m["ema_50_15m"] = ema_50
    df_15m["ewm_ret_15m"] = df_15m["ret_15m"].ewm(span=20, adjust=False).mean()

    tr_15m = pd.concat([
        high_15m - low_15m,
        (high_15m - close_15m.shift(1)).abs(),
        (low_15m - close_15m.shift(1)).abs(),
    ], axis=1).max(axis=1)
    df_15m["atr_15m"] = tr_15m.rolling(14).mean()
    df_15m["atr_pct_15m"] = df_15m["atr_15m"] / close_15m * 100  # ATR as percentage

    # Rolling high/low for zones
    df_15m["rolling_high_20"] = high_15m.rolling(20).max()
    df_15m["rolling_low_20"] = low_15m.rolling(20).min()

    # BB width
    sma_15m = close_15m.rolling(20).mean()
    std_15m = close_15m.rolling(20).std()
    df_15m["bb_width_15m"] = (2 * std_15m / sma_15m) * 100

    # Additional 15m features for 17-feature HMM
    open_15m = df_15m["open"].astype(float)
    rolling_range = df_15m["rolling_high_20"] - df_15m["rolling_low_20"]
    df_15m["range_comp_15m"] = (high_15m - low_15m) / rolling_range.replace(0, np.nan)
    df_15m["bb_width_15m_pct"] = std_15m / sma_15m

    # Candle body features
    body = (close_15m - open_15m).abs()
    candle_range = (high_15m - low_15m).replace(0, np.nan)
    df_15m["body_ratio"] = body / candle_range
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
        # Drop existing 1h columns if they exist to avoid overlap error
        cols_to_drop = [c for c in df_1h_resampled.columns if c in df_15m_indexed.columns]
        if cols_to_drop:
            df_15m_indexed = df_15m_indexed.drop(columns=cols_to_drop)
        df_15m_indexed = df_15m_indexed.join(df_1h_resampled, how="left")
        df_15m_indexed[["ret_1h", "ewm_ret_1h", "ewm_std_ret_1h", "price_z_from_ema_1h", "atr_pct_1h"]] = \
            df_15m_indexed[["ret_1h", "ewm_ret_1h", "ewm_std_ret_1h", "price_z_from_ema_1h", "atr_pct_1h"]].ffill()
        df_15m = df_15m_indexed.reset_index()

        # 17 features for HMM
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
# Paper Trading Engine
# ============================================================================

class PaperTradingEngine:
    """Paper trading engine with FSM pipeline."""

    # Data requirements for proper initialization
    # Based on indicator calculations and HMM training needs
    DATA_REQUIREMENTS = {
        # 3m timeframe requirements
        "3m": {
            "ema_slow": 50,          # EMA 50 needs 50 bars
            "ema_warmup": 150,       # 3x EMA period for stability
            "atr": 14,               # ATR 14 periods
            "volatility": 20,        # 20-bar rolling std
            "rsi": 14,               # RSI 14 periods
            "backtest_warmup": 250,  # Match backtest warmup
        },
        # 15m timeframe requirements
        "15m": {
            "ema_slow": 50,                 # EMA 50 needs 50 bars
            "ema_warmup": 150,              # 3x EMA period for stability
            "vol_base_lookback": 96,        # ConfirmLayer: 24h baseline
            "box_lookback": 32,             # ConfirmLayer: 6-8 hours
            "hmm_training_min": 500,        # Minimum for robust HMM training
            "hmm_training_optimal": 2000,   # Optimal for HMM training (~20 days)
        },
        # 1h timeframe requirements (for 17-feature HMM)
        "1h": {
            "ema_20": 20,                   # EMA 20 needs 20 bars
            "ema_warmup": 60,               # 3x EMA period for stability
            "atr": 14,                      # ATR 14 periods
            "optimal": 500,                 # ~20 days of 1h data
        },
    }

    def __init__(
        self,
        symbol: str = "XRPUSDT",
        initial_capital: float = 10000.0,
        poll_interval: int = 5,
        hmm_model_path: Optional[str] = None,
        leverage: float = 1.0,
    ):
        self.symbol = symbol
        self.initial_capital = initial_capital
        self.poll_interval = poll_interval
        self.hmm_model_path = hmm_model_path
        self.leverage = leverage

        # Components
        self.client = BinanceClient()

        # Multi-HMM Manager
        config_path = Path(__file__).parent.parent / "configs" / "hmm_gate_policy.yaml"
        if config_path.exists():
            self.multi_hmm = MultiHMMManager.from_config_file(
                config_path,
                checkpoint_dir=Path(__file__).parent.parent / "checkpoints" / "hmm"
            )
            self._use_multi_hmm = True
            logger.info("Using Multi-HMM (Fast 3m + Mid 15m)")
        else:
            self.multi_hmm = None
            self._use_multi_hmm = False
            logger.warning("Multi-HMM config not found, falling back to single HMM")

        # Fallback single HMM (for compatibility)
        self.hmm = LiveHMM(n_states=4)
        # ConfirmConfig matching backtest_fsm_pipeline.py exactly
        self.confirm_layer = RegimeConfirmLayer(ConfirmConfig(
            HIGH_VOL_LAMBDA_ON=5.5,      # Match backtest: 상위 3.6%에서만 트리거
            HIGH_VOL_LAMBDA_OFF=3.5,     # Match backtest: 적절한 퇴장 조건
            HIGH_VOL_COOLDOWN_BARS_15M=4,  # Match backtest
        ))
        self.fsm = TradingFSM()
        self.decision_engine = DecisionEngine()

        # Log XGB gate status
        if self.decision_engine.cfg.XGB_ENABLED:
            logger.info(f"XGB Gate: ENABLED (threshold={self.decision_engine.cfg.XGB_PMIN_TREND})")
        else:
            logger.info("XGB Gate: DISABLED")

        # State
        self.df_3m: Optional[pd.DataFrame] = None
        self.df_15m: Optional[pd.DataFrame] = None
        self.df_1h: Optional[pd.DataFrame] = None
        self._hmm_feature_names: List[str] = []
        self._fast_feature_names: List[str] = []
        self._mid_feature_names: List[str] = []
        self.confirm_state = None
        self.fsm_state = None
        self.engine_state = None

        # Trading state
        self.equity = initial_capital
        self.position = PositionState(side="FLAT")
        self.trades: List[Dict] = []
        self.last_3m_ts: Optional[pd.Timestamp] = None

        # Stats
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0

        # Regime tracking for change detection
        self.last_regime: Optional[str] = None

        # === 5-second tick data tracking ===
        self.tick_history: List[Dict] = []  # Recent 5-sec price ticks
        self.tick_history_max = 60  # Keep last 60 ticks (5 minutes)

        # Sudden move detection parameters
        self.sudden_move_threshold = 0.005  # 0.5% move in short time
        self.sudden_move_window = 6  # 6 ticks = 30 seconds

        # Pending signal for entry optimization
        self.pending_signal: Optional[Dict] = None
        self.pending_signal_timeout = 12  # 12 ticks = 60 seconds max wait

        # Entry optimization parameters
        self.entry_pullback_pct = 0.001  # Wait for 0.1% pullback for better entry
        self.entry_momentum_confirm = True  # Confirm momentum before entry

        # Entry timing statistics (5-sec based)
        self.entry_5s_stats = {
            "total": 0,           # Total entry attempts
            "skipped": 0,         # Skipped due to chasing
            "improved": 0,        # Got better price than signal
            "executed": 0,        # Actually executed
            "total_saved_pct": 0, # Total percentage saved
        }

        # === 5-second Trailing Stop (matching backtest's 1m trailing stop) ===
        # Parameters matching backtest_fsm_pipeline.py exactly (2026-01-19 sync)
        self.TRAILING_STOP_ACTIVATION_PCT = 0.10   # Activate when profit reaches 0.10%
        self.TRAILING_STOP_PRESERVE_PCT = 0.60     # Preserve 60% of max profit
        self.TRAILING_STOP_MIN_PRESERVE_PCT = 0.05 # Minimum 0.05% profit to preserve (must be < activation)

        # === Break-even Stop (손실 방지) ===
        # 작은 수익 도달 시 손익분기점으로 스탑 이동
        self.BREAKEVEN_ACTIVATION_PCT = 0.10   # 0.10% = TRAILING과 동일 (BREAKEVEN 구간 제거)
        self.BREAKEVEN_BUFFER_PCT = 0.05       # 손익분기 + 0.05% (수수료 0.04% 커버 + 여유)
        self.breakeven_activated = False        # Break-even 활성화 상태

        # Tracking variables
        self.max_unrealized_pnl = 0.0
        self.max_unrealized_pnl_pct = 0.0
        self.entry_regime = None
        self.entry_signal = None

        # Trailing stop statistics
        self.trailing_stop_stats = {
            "triggered": 0,
            "total_preserved_pct": 0.0,
            "breakeven_triggered": 0,  # Break-even으로 청산된 횟수
        }

        # Output files
        self.output_dir = Path(__file__).parent.parent / "outputs" / "paper_trading"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create session-specific files with timestamp
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.trades_file = self.output_dir / f"trades_{self.session_id}.csv"
        self.signals_file = self.output_dir / f"signals_{self.session_id}.csv"
        self.summary_file = self.output_dir / f"summary_{self.session_id}.json"

        # Initialize CSV headers
        self._init_csv_files()

    def _init_csv_files(self) -> None:
        """Initialize CSV files with headers."""
        # Trades CSV
        with open(self.trades_file, "w") as f:
            f.write("timestamp,side,entry_price,exit_price,size,pnl,pnl_pct,bars_held,equity_after\n")

        # Signals CSV - only regime changes and trades
        with open(self.signals_file, "w") as f:
            f.write("timestamp,event_type,price,regime_raw,regime_confirmed,confirm_reason,"
                    "signal,signal_reason,signal_score,action,decision_reason,"
                    "position,unrealized_pnl,ema_fast,ema_slow,atr,rsi\n")

        logger.info(f"Trade log: {self.trades_file}")
        logger.info(f"Signal log: {self.signals_file}")

    def _save_trade(self, trade: Dict) -> None:
        """Append a trade to the trades CSV file."""
        with open(self.trades_file, "a") as f:
            f.write(
                f"{trade['timestamp']},"
                f"{trade['side']},"
                f"{trade['entry_price']:.4f},"
                f"{trade['exit_price']:.4f},"
                f"{trade['size']:.4f},"
                f"{trade['pnl']:.2f},"
                f"{trade['pnl_pct']:.2f},"
                f"{trade['bars_held']},"
                f"{trade['equity_after']:.2f}\n"
            )

    def _save_signal(
        self,
        timestamp: str,
        price: float,
        regime_raw: str,
        regime_confirmed: str,
        confirm_reason: str,
        signal: str,
        signal_reason: str,
        signal_score: float,
        action: str,
        decision_reason: str,
        market_ctx: Optional[MarketContext] = None,
        event_type: str = "UNKNOWN",
    ) -> None:
        """Append a detailed signal to the signals CSV file.

        Only called on regime changes or trade actions.
        """
        pos = self.position.side
        unrealized = self.position.unrealized_pnl if pos != "FLAT" else 0.0

        # Extract market features
        ema_fast = market_ctx.row_3m.get("ema_fast_3m", 0) if market_ctx else 0
        ema_slow = market_ctx.row_3m.get("ema_slow_3m", 0) if market_ctx else 0
        atr = market_ctx.row_3m.get("atr_3m", 0) if market_ctx else 0
        rsi = market_ctx.row_3m.get("rsi_3m", 50) if market_ctx else 50

        # Escape commas in reason strings
        confirm_reason = confirm_reason.replace(",", ";") if confirm_reason else ""
        signal_reason = signal_reason.replace(",", ";") if signal_reason else ""
        decision_reason = decision_reason.replace(",", ";") if decision_reason else ""

        with open(self.signals_file, "a") as f:
            f.write(
                f"{timestamp},{event_type},{price:.4f},{regime_raw},{regime_confirmed},{confirm_reason},"
                f"{signal},{signal_reason},{signal_score:.3f},{action},{decision_reason},"
                f"{pos},{unrealized:.2f},{ema_fast:.4f},{ema_slow:.4f},{atr:.6f},{rsi:.1f}\n"
            )

    def _save_summary(self) -> None:
        """Save trading session summary to JSON file."""
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        return_pct = ((self.equity / self.initial_capital) - 1) * 100

        # Calculate entry timing stats
        stats = self.entry_5s_stats
        skip_rate = (stats["skipped"] / max(1, stats["total"])) * 100
        improve_rate = (stats["improved"] / max(1, stats["executed"])) * 100
        avg_saved = stats["total_saved_pct"] / max(1, stats["improved"])

        summary = {
            "session_id": self.session_id,
            "symbol": self.symbol,
            "start_time": self.trades[0]["timestamp"] if self.trades else None,
            "end_time": datetime.now().isoformat(),
            "initial_capital": self.initial_capital,
            "final_equity": self.equity,
            "return_pct": return_pct,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.total_trades - self.winning_trades,
            "win_rate": win_rate,
            "total_pnl": self.total_pnl,
            "open_position": {
                "side": self.position.side,
                "entry_price": self.position.entry_price if self.position.side != "FLAT" else None,
                "size": self.position.size if self.position.side != "FLAT" else None,
            } if self.position.side != "FLAT" else None,
            "entry_timing_5s": {
                "total_attempts": stats["total"],
                "skipped_chasing": stats["skipped"],
                "skip_rate_pct": skip_rate,
                "executed": stats["executed"],
                "improved_entry": stats["improved"],
                "improve_rate_pct": improve_rate,
                "avg_saved_pct": avg_saved,
            },
            "trades": self.trades,
        }

        with open(self.summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Summary saved: {self.summary_file}")

    # =========================================================================
    # 5-Second Tick Data Methods
    # =========================================================================

    def _update_tick_history(self, price: float) -> None:
        """Update tick history with current price."""
        tick = {
            "timestamp": datetime.now(),
            "price": price,
        }
        self.tick_history.append(tick)

        # Keep only recent ticks
        if len(self.tick_history) > self.tick_history_max:
            self.tick_history = self.tick_history[-self.tick_history_max:]

    def _detect_sudden_move(self) -> Optional[Dict]:
        """Detect sudden price movement in recent ticks.

        Returns:
            Dict with move info if sudden move detected, None otherwise.
            {"direction": "UP" or "DOWN", "pct_change": float, "duration_secs": int}
        """
        if len(self.tick_history) < self.sudden_move_window:
            return None

        recent = self.tick_history[-self.sudden_move_window:]
        start_price = recent[0]["price"]
        end_price = recent[-1]["price"]

        pct_change = (end_price - start_price) / start_price

        if abs(pct_change) >= self.sudden_move_threshold:
            duration = (recent[-1]["timestamp"] - recent[0]["timestamp"]).total_seconds()
            return {
                "direction": "UP" if pct_change > 0 else "DOWN",
                "pct_change": pct_change,
                "duration_secs": int(duration),
                "start_price": start_price,
                "end_price": end_price,
            }

        return None

    def _check_emergency_exit(self, current_price: float) -> Optional[str]:
        """Check if emergency exit is needed due to adverse sudden move.

        Returns:
            Exit reason string if should exit, None otherwise.
        """
        if self.position.side == "FLAT":
            return None

        sudden_move = self._detect_sudden_move()
        if sudden_move is None:
            return None

        # Check if move is against our position
        if self.position.side == "LONG" and sudden_move["direction"] == "DOWN":
            if abs(sudden_move["pct_change"]) >= self.sudden_move_threshold:
                return f"EMERGENCY_EXIT_SUDDEN_DROP_{abs(sudden_move['pct_change'])*100:.2f}%"

        elif self.position.side == "SHORT" and sudden_move["direction"] == "UP":
            if abs(sudden_move["pct_change"]) >= self.sudden_move_threshold:
                return f"EMERGENCY_EXIT_SUDDEN_SPIKE_{abs(sudden_move['pct_change'])*100:.2f}%"

        return None

    def _update_max_profit_5s(self, current_price: float) -> None:
        """Track maximum unrealized profit using 5-second tick data.

        Matching backtest's 1m max profit tracking but using 5s ticks.
        """
        if self.position.side == "FLAT":
            return

        # Track max unrealized profit from tick history
        if len(self.tick_history) > 0:
            recent_ticks = self.tick_history[-12:]  # Last 60 seconds (12 ticks)
            prices = [t["price"] for t in recent_ticks]

            if self.position.side == "LONG":
                # For LONG, best price is highest
                best_price_5s = max(prices) if prices else current_price
                best_pnl = (best_price_5s - self.position.entry_price) * self.position.size
                best_pnl_pct = (best_price_5s - self.position.entry_price) / self.position.entry_price * 100
            else:  # SHORT
                # For SHORT, best price is lowest
                best_price_5s = min(prices) if prices else current_price
                best_pnl = (self.position.entry_price - best_price_5s) * self.position.size
                best_pnl_pct = (self.position.entry_price - best_price_5s) / self.position.entry_price * 100

            if best_pnl > self.max_unrealized_pnl:
                self.max_unrealized_pnl = best_pnl
                self.max_unrealized_pnl_pct = best_pnl_pct

        # Also track using current price
        if self.position.side == "LONG":
            current_pnl = (current_price - self.position.entry_price) * self.position.size
            current_pnl_pct = (current_price - self.position.entry_price) / self.position.entry_price * 100
        else:
            current_pnl = (self.position.entry_price - current_price) * self.position.size
            current_pnl_pct = (self.position.entry_price - current_price) / self.position.entry_price * 100

        if current_pnl > self.max_unrealized_pnl:
            self.max_unrealized_pnl = current_pnl
            self.max_unrealized_pnl_pct = current_pnl_pct

    def _check_breakeven_stop_5s(self, current_price: float) -> Optional[Tuple[float, str]]:
        """Check if break-even stop should trigger (손실 방지).

        Break-even Stop 로직:
        - 0.05% 수익 도달 시 활성화
        - 손익분기점 + 0.02% (수수료 커버)에서 청산

        Returns:
            Tuple of (exit_price, reason) if should exit, None otherwise.
        """
        if self.position.side == "FLAT":
            return None

        # Check if break-even should be activated
        if self.max_unrealized_pnl_pct >= self.BREAKEVEN_ACTIVATION_PCT:
            self.breakeven_activated = True

        if not self.breakeven_activated:
            return None

        # If trailing stop is activated (higher threshold), skip break-even check
        # Trailing stop takes priority once activated
        if self.max_unrealized_pnl_pct >= self.TRAILING_STOP_ACTIVATION_PCT:
            return None  # Let trailing stop handle it

        # Break-even price (entry + small buffer for fees)
        if self.position.side == "LONG":
            breakeven_price = self.position.entry_price * (1 + self.BREAKEVEN_BUFFER_PCT / 100)
            current_pnl_pct = (current_price - self.position.entry_price) / self.position.entry_price * 100

            # Check if price dropped below break-even
            if current_price <= breakeven_price:
                return (breakeven_price, f"BREAKEVEN_STOP (max={self.max_unrealized_pnl_pct:.2f}%, exit at BE+{self.BREAKEVEN_BUFFER_PCT}%)")

            # Check recent ticks
            if len(self.tick_history) >= 2:
                recent_low = min(t["price"] for t in self.tick_history[-6:])
                if recent_low <= breakeven_price:
                    return (breakeven_price, f"BREAKEVEN_STOP (max={self.max_unrealized_pnl_pct:.2f}%, exit at BE+{self.BREAKEVEN_BUFFER_PCT}%)")

        else:  # SHORT
            breakeven_price = self.position.entry_price * (1 - self.BREAKEVEN_BUFFER_PCT / 100)
            current_pnl_pct = (self.position.entry_price - current_price) / self.position.entry_price * 100

            # Check if price rose above break-even
            if current_price >= breakeven_price:
                return (breakeven_price, f"BREAKEVEN_STOP (max={self.max_unrealized_pnl_pct:.2f}%, exit at BE+{self.BREAKEVEN_BUFFER_PCT}%)")

            # Check recent ticks
            if len(self.tick_history) >= 2:
                recent_high = max(t["price"] for t in self.tick_history[-6:])
                if recent_high >= breakeven_price:
                    return (breakeven_price, f"BREAKEVEN_STOP (max={self.max_unrealized_pnl_pct:.2f}%, exit at BE+{self.BREAKEVEN_BUFFER_PCT}%)")

        return None

    def _check_trailing_stop_5s(self, current_price: float) -> Optional[Tuple[float, str]]:
        """Check if 5-second trailing stop should trigger.

        Matching backtest's 1m trailing stop exactly:
        - Activation: 0.10% profit
        - Preserve: 60% of max profit, minimum 0.15%

        Returns:
            Tuple of (exit_price, reason) if should exit, None otherwise.
        """
        if self.position.side == "FLAT":
            return None

        if self.max_unrealized_pnl_pct < self.TRAILING_STOP_ACTIVATION_PCT:
            return None  # Not activated yet

        # Calculate preserve level
        preserve_pct = max(
            self.max_unrealized_pnl_pct * self.TRAILING_STOP_PRESERVE_PCT,
            self.TRAILING_STOP_MIN_PRESERVE_PCT
        )

        # Check current profit level
        if self.position.side == "LONG":
            current_pnl_pct = (current_price - self.position.entry_price) / self.position.entry_price * 100
            preserve_price = self.position.entry_price * (1 + preserve_pct / 100)

            # Check if price dropped below preserve level
            if current_price <= preserve_price:
                exit_price = max(current_price, preserve_price)
                return (exit_price, f"TRAILING_STOP_5S (max={self.max_unrealized_pnl_pct:.2f}%, preserved={preserve_pct:.2f}%)")

            # Also check recent ticks for faster detection
            if len(self.tick_history) >= 2:
                recent_low = min(t["price"] for t in self.tick_history[-6:])  # Last 30 seconds
                if recent_low <= preserve_price:
                    exit_price = max(recent_low, preserve_price)
                    return (exit_price, f"TRAILING_STOP_5S (max={self.max_unrealized_pnl_pct:.2f}%, preserved={preserve_pct:.2f}%)")

        else:  # SHORT
            current_pnl_pct = (self.position.entry_price - current_price) / self.position.entry_price * 100
            preserve_price = self.position.entry_price * (1 - preserve_pct / 100)

            # Check if price rose above preserve level
            if current_price >= preserve_price:
                exit_price = min(current_price, preserve_price)
                return (exit_price, f"TRAILING_STOP_5S (max={self.max_unrealized_pnl_pct:.2f}%, preserved={preserve_pct:.2f}%)")

            # Also check recent ticks
            if len(self.tick_history) >= 2:
                recent_high = max(t["price"] for t in self.tick_history[-6:])
                if recent_high >= preserve_price:
                    exit_price = min(recent_high, preserve_price)
                    return (exit_price, f"TRAILING_STOP_5S (max={self.max_unrealized_pnl_pct:.2f}%, preserved={preserve_pct:.2f}%)")

        return None

    def _execute_trailing_stop_exit(self, exit_price: float, reason: str) -> None:
        """Execute trailing stop exit at given price."""
        if self.position.side == "FLAT":
            return

        fee_rate = 0.0004

        if self.position.side == "LONG":
            pnl = (exit_price - self.position.entry_price) * self.position.size
        else:
            pnl = (self.position.entry_price - exit_price) * self.position.size

        fee = exit_price * self.position.size * fee_rate
        pnl -= fee
        self.equity += pnl
        self.total_pnl += pnl
        self.total_trades += 1

        if pnl > 0:
            self.winning_trades += 1

        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        final_pnl_pct = pnl / (self.position.entry_price * self.position.size) * 100 if self.position.size > 0 else 0

        # Update statistics
        self.trailing_stop_stats["triggered"] += 1
        self.trailing_stop_stats["total_preserved_pct"] += final_pnl_pct

        logger.info(
            f"[TRAILING_STOP] {self.position.side} "
            f"Entry: ${self.position.entry_price:.4f}, Exit: ${exit_price:.4f}, "
            f"PnL: ${pnl:.2f} ({final_pnl_pct:.2f}%), Max: {self.max_unrealized_pnl_pct:.2f}%, "
            f"Reason: {reason}"
        )

        # Save trade
        trade = {
            "timestamp": datetime.now().isoformat(),
            "side": self.position.side,
            "entry_price": self.position.entry_price,
            "exit_price": exit_price,
            "size": self.position.size,
            "pnl": pnl,
            "pnl_pct": final_pnl_pct,
            "bars_held": self.position.bars_held_3m,
            "equity_after": self.equity,
            "exit_reason": reason,
            "max_profit_pct": self.max_unrealized_pnl_pct,
            "regime": self.entry_regime,
            "signal": self.entry_signal,
        }
        self.trades.append(trade)
        self._save_trade(trade)

        # Reset position and tracking
        self.position = PositionState(side="FLAT")
        self.max_unrealized_pnl = 0.0
        self.max_unrealized_pnl_pct = 0.0
        self.breakeven_activated = False  # Reset break-even flag
        self.entry_regime = None
        self.entry_signal = None

    def _set_pending_signal(self, signal: str, decision: Decision, market_ctx: MarketContext) -> None:
        """Set a pending signal for entry optimization using 5s analysis."""
        self.entry_5s_stats["total"] += 1

        side = "LONG" if decision.action == "OPEN_LONG" else "SHORT"
        entry_analysis = self._analyze_5s_entry(side)

        # Check if we should skip this trade due to chasing
        if entry_analysis["should_skip"]:
            self.entry_5s_stats["skipped"] += 1
            logger.info(f"[SKIP] {decision.action} - {entry_analysis['skip_reason']}")
            return  # Don't create pending signal

        self.pending_signal = {
            "signal": signal,
            "decision": decision,
            "market_ctx": market_ctx,
            "signal_price": market_ctx.price,
            "ticks_waited": 0,
            "best_price": market_ctx.price,
            "created_at": datetime.now(),
            "entry_analysis": entry_analysis,
        }
        logger.info(f"[PENDING] {decision.action} @ ${market_ctx.price:.4f} - waiting for optimal entry")

    def _check_entry_timing(self, current_price: float) -> Optional[Dict]:
        """Check if now is a good time to execute pending signal.

        Returns optimal entry info or None to keep waiting.
        """
        if self.pending_signal is None:
            return None

        self.pending_signal["ticks_waited"] += 1
        signal_price = self.pending_signal["signal_price"]
        action = self.pending_signal["decision"].action

        # Calculate price improvement
        if action == "OPEN_LONG":
            # For LONG, lower price is better
            if current_price < self.pending_signal["best_price"]:
                self.pending_signal["best_price"] = current_price

            improvement = (signal_price - current_price) / signal_price
            is_pullback = current_price < signal_price

            # Entry conditions for LONG:
            # 1. Price pulled back (better entry) and now rebounding
            # 2. Or timeout reached
            # 3. Or momentum confirmed (price moving up after pullback)
            if self.pending_signal["ticks_waited"] >= 2:  # Wait at least 2 ticks
                recent_prices = [t["price"] for t in self.tick_history[-3:]] if len(self.tick_history) >= 3 else []

                # Check for rebound (price going up after pullback)
                if len(recent_prices) >= 3:
                    is_rebounding = recent_prices[-1] > recent_prices[-2] > recent_prices[-3]
                    if is_pullback and is_rebounding:
                        return {
                            "execute": True,
                            "reason": f"PULLBACK_REBOUND (saved {improvement*100:.3f}%)",
                            "price": current_price,
                        }

                # Timeout - execute at best available price
                if self.pending_signal["ticks_waited"] >= self.pending_signal_timeout:
                    return {
                        "execute": True,
                        "reason": f"TIMEOUT (best: ${self.pending_signal['best_price']:.4f})",
                        "price": current_price,
                    }

                # Price running away - don't miss the trade
                if current_price > signal_price * 1.002:  # >0.2% above signal price
                    return {
                        "execute": True,
                        "reason": "PRICE_RUNNING",
                        "price": current_price,
                    }

        elif action == "OPEN_SHORT":
            # For SHORT, higher price is better
            if current_price > self.pending_signal["best_price"]:
                self.pending_signal["best_price"] = current_price

            improvement = (current_price - signal_price) / signal_price
            is_bounce = current_price > signal_price

            if self.pending_signal["ticks_waited"] >= 2:
                recent_prices = [t["price"] for t in self.tick_history[-3:]] if len(self.tick_history) >= 3 else []

                # Check for rejection (price going down after bounce)
                if len(recent_prices) >= 3:
                    is_rejecting = recent_prices[-1] < recent_prices[-2] < recent_prices[-3]
                    if is_bounce and is_rejecting:
                        return {
                            "execute": True,
                            "reason": f"BOUNCE_REJECTION (saved {improvement*100:.3f}%)",
                            "price": current_price,
                        }

                # Timeout
                if self.pending_signal["ticks_waited"] >= self.pending_signal_timeout:
                    return {
                        "execute": True,
                        "reason": f"TIMEOUT (best: ${self.pending_signal['best_price']:.4f})",
                        "price": current_price,
                    }

                # Price running away
                if current_price < signal_price * 0.998:  # >0.2% below signal price
                    return {
                        "execute": True,
                        "reason": "PRICE_RUNNING",
                        "price": current_price,
                    }

        return None  # Keep waiting

    def _cancel_pending_signal(self, reason: str) -> None:
        """Cancel pending signal."""
        if self.pending_signal:
            logger.info(f"[CANCELLED] Pending {self.pending_signal['decision'].action} - {reason}")
            self.pending_signal = None

    def _analyze_5s_entry(self, side: str) -> Dict:
        """Analyze 5-second tick data for optimal entry timing.

        Similar to backtest's analyze_1m_entry() but uses 5-second ticks.
        Uses recent 36 ticks (180 seconds = 3 minutes, matching 3m bar period).

        Args:
            side: "LONG" or "SHORT"

        Returns:
            dict with:
                - optimal_price: Best entry price from recent ticks
                - has_spike: Whether a sudden spike/drop occurred
                - spike_direction: "UP", "DOWN", or None
                - max_ret_5s: Maximum 5s return
                - min_ret_5s: Minimum 5s return
                - should_skip: Whether to skip due to chasing
                - skip_reason: Reason for skipping
        """
        # Need at least 6 ticks for meaningful analysis (30 seconds)
        if len(self.tick_history) < 6:
            return {
                "optimal_price": None,
                "has_spike": False,
                "spike_direction": None,
                "max_ret_5s": 0,
                "min_ret_5s": 0,
                "should_skip": False,
                "skip_reason": None,
            }

        # Get recent ticks (up to 36 = 3 minutes equivalent)
        recent_ticks = self.tick_history[-min(36, len(self.tick_history)):]
        prices = [t["price"] for t in recent_ticks]

        # Calculate returns
        returns = []
        for i in range(1, len(prices)):
            ret = (prices[i] - prices[i-1]) / prices[i-1]
            returns.append(ret)

        if not returns:
            return {
                "optimal_price": None,
                "has_spike": False,
                "spike_direction": None,
                "max_ret_5s": 0,
                "min_ret_5s": 0,
                "should_skip": False,
                "skip_reason": None,
            }

        max_ret = max(returns)
        min_ret = min(returns)

        # Spike detection (>0.3% move in 5s is significant - scaled from 0.5% in 1m)
        spike_threshold = 0.003
        has_spike = abs(max_ret) > spike_threshold or abs(min_ret) > spike_threshold

        # Determine spike direction
        spike_direction = None
        if max_ret > spike_threshold:
            spike_direction = "UP"
        elif min_ret < -spike_threshold:
            spike_direction = "DOWN"

        # Should we skip this trade? (Don't chase after big moves)
        # Use same 1% threshold as backtest (cumulative over 3m window is comparable)
        chase_threshold = 0.01  # 1% - matching backtest's 1m anti-chasing
        should_skip = False
        skip_reason = None

        # Check cumulative move over recent ticks
        cumulative_ret = (prices[-1] - prices[0]) / prices[0]

        if side == "LONG":
            # Skip LONG if price just spiked up significantly
            if cumulative_ret > chase_threshold:
                should_skip = True
                skip_reason = f"CHASING_UP_{cumulative_ret*100:.2f}%"
        elif side == "SHORT":
            # Skip SHORT if price just dropped significantly
            if cumulative_ret < -chase_threshold:
                should_skip = True
                skip_reason = f"CHASING_DOWN_{abs(cumulative_ret)*100:.2f}%"

        # Find optimal entry price from recent ticks
        if side == "LONG":
            # For LONG, lower price is better - use minimum
            optimal_price = min(prices)
        else:
            # For SHORT, higher price is better - use maximum
            optimal_price = max(prices)

        return {
            "optimal_price": optimal_price,
            "has_spike": has_spike,
            "spike_direction": spike_direction,
            "max_ret_5s": max_ret,
            "min_ret_5s": min_ret,
            "should_skip": should_skip,
            "skip_reason": skip_reason,
            "cumulative_ret": cumulative_ret,
        }

    def _execute_pending_signal(self, current_price: float, reason: str) -> None:
        """Execute the pending signal at current price with 5s entry timing."""
        if self.pending_signal is None:
            return

        decision = self.pending_signal["decision"]
        signal_price = self.pending_signal["signal_price"]
        action = decision.action

        # Get fresh 5s entry analysis for optimal price
        side = "LONG" if action == "OPEN_LONG" else "SHORT"
        entry_analysis = self._analyze_5s_entry(side)

        # Use optimal price from recent 5s ticks if available
        if entry_analysis["optimal_price"] is not None:
            if side == "LONG" and entry_analysis["optimal_price"] < current_price:
                # For LONG, use the lower price (dip) if recent
                current_price = entry_analysis["optimal_price"]
            elif side == "SHORT" and entry_analysis["optimal_price"] > current_price:
                # For SHORT, use the higher price (bounce) if recent
                current_price = entry_analysis["optimal_price"]

        # Calculate price improvement
        if action == "OPEN_LONG":
            improvement = (signal_price - current_price) / signal_price * 100
            got_better_price = current_price < signal_price
        else:  # OPEN_SHORT
            improvement = (current_price - signal_price) / signal_price * 100
            got_better_price = current_price > signal_price

        # Update statistics
        self.entry_5s_stats["executed"] += 1
        if got_better_price and improvement > 0:
            self.entry_5s_stats["improved"] += 1
            self.entry_5s_stats["total_saved_pct"] += improvement

        fee_rate = 0.0004

        if action == "OPEN_LONG":
            # Position size = equity * leverage / price
            position_value = self.equity * self.leverage
            size = position_value / current_price
            fee = current_price * size * fee_rate
            self.equity -= fee

            self.position = PositionState(
                side="LONG",
                entry_price=current_price,
                size=size,
                entry_ts=int(datetime.now().timestamp() * 1000),
                bars_held_3m=0,
            )

            # Reset trailing stop and break-even tracking
            self.max_unrealized_pnl = 0.0
            self.max_unrealized_pnl_pct = 0.0
            self.breakeven_activated = False  # Reset break-even flag
            self.entry_regime = self.pending_signal.get("market_ctx").row_3m.get("regime_confirmed") if self.pending_signal.get("market_ctx") else None
            self.entry_signal = self.pending_signal.get("signal")

            logger.info(
                f"[OPEN_LONG] Price: ${current_price:.4f} (signal: ${signal_price:.4f}, "
                f"saved: {improvement:.3f}%), Size: {size:.2f} XRP (${position_value:.0f}), Entry: {reason}"
            )

        elif action == "OPEN_SHORT":
            # Position size = equity * leverage / price
            position_value = self.equity * self.leverage
            size = position_value / current_price
            fee = current_price * size * fee_rate
            self.equity -= fee

            self.position = PositionState(
                side="SHORT",
                entry_price=current_price,
                size=size,
                entry_ts=int(datetime.now().timestamp() * 1000),
                bars_held_3m=0,
            )

            # Reset trailing stop and break-even tracking
            self.max_unrealized_pnl = 0.0
            self.max_unrealized_pnl_pct = 0.0
            self.breakeven_activated = False  # Reset break-even flag
            self.entry_regime = self.pending_signal.get("market_ctx").row_3m.get("regime_confirmed") if self.pending_signal.get("market_ctx") else None
            self.entry_signal = self.pending_signal.get("signal")

            logger.info(
                f"[OPEN_SHORT] Price: ${current_price:.4f} (signal: ${signal_price:.4f}, "
                f"saved: {improvement:.3f}%), Size: {size:.2f} XRP (${position_value:.0f}), Entry: {reason}"
            )

        # Clear pending signal
        self.pending_signal = None

    def _execute_emergency_exit(self, current_price: float, reason: str) -> None:
        """Execute emergency exit at current price."""
        if self.position.side == "FLAT":
            return

        fee_rate = 0.0004

        if self.position.side == "LONG":
            pnl = (current_price - self.position.entry_price) * self.position.size
        else:
            pnl = (self.position.entry_price - current_price) * self.position.size

        fee = current_price * self.position.size * fee_rate
        pnl -= fee
        self.equity += pnl
        self.total_pnl += pnl
        self.total_trades += 1

        if pnl > 0:
            self.winning_trades += 1

        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0

        logger.info(
            f"[EMERGENCY_CLOSE_{self.position.side}] "
            f"Entry: ${self.position.entry_price:.4f}, Exit: ${current_price:.4f}, "
            f"PnL: ${pnl:.2f}, Reason: {reason}, "
            f"Total PnL: ${self.total_pnl:.2f}, WR: {win_rate:.1f}%"
        )

        # Calculate PnL percentage
        pnl_pct = (pnl / (self.position.entry_price * self.position.size)) * 100
        profit_given_back = self.max_unrealized_pnl_pct - pnl_pct if self.max_unrealized_pnl_pct > 0 else 0

        trade = {
            "timestamp": datetime.now().isoformat(),
            "side": self.position.side,
            "entry_price": self.position.entry_price,
            "exit_price": current_price,
            "size": self.position.size,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "bars_held": self.position.bars_held_3m,
            "equity_after": self.equity,
            "exit_reason": reason,
            "max_profit_pct": self.max_unrealized_pnl_pct,  # Match backtest
            "profit_given_back": profit_given_back,         # Match backtest
            "regime": self.entry_regime,
            "signal": self.entry_signal,
        }
        self.trades.append(trade)
        self._save_trade(trade)

        # Reset position and trailing stop tracking
        self.position = PositionState(side="FLAT")
        self.max_unrealized_pnl = 0.0
        self.max_unrealized_pnl_pct = 0.0
        self.breakeven_activated = False  # Reset break-even flag
        self.entry_regime = None
        self.entry_signal = None

    def _calculate_required_bars(self) -> Tuple[int, int, int]:
        """Calculate required number of bars for each timeframe.

        Returns:
            (required_3m_bars, required_15m_bars, required_1h_bars)
        """
        req_3m = self.DATA_REQUIREMENTS["3m"]
        req_15m = self.DATA_REQUIREMENTS["15m"]
        req_1h = self.DATA_REQUIREMENTS["1h"]

        # 3m: max of all requirements
        bars_3m = max(
            req_3m["ema_warmup"],
            req_3m["backtest_warmup"],
            req_3m["volatility"] * 3,  # 3x for stability
        )

        # 15m: max of all requirements, prioritize HMM training
        bars_15m = max(
            req_15m["ema_warmup"],
            req_15m["vol_base_lookback"] + 50,  # Extra buffer
            req_15m["hmm_training_optimal"],     # Prioritize HMM quality
        )

        # 1h: for 17-feature HMM
        bars_1h = max(
            req_1h["ema_warmup"],
            req_1h["optimal"],
        )

        return bars_3m, bars_15m, bars_1h

    def initialize(self) -> None:
        """Initialize with historical data and train HMM.

        Fetches sufficient historical data based on indicator requirements:
        - 3m: 500+ bars for EMA/ATR warmup and backtest compatibility
        - 15m: 2000+ bars for robust HMM training (~20 days)
        - 1h: 500+ bars for 17-feature HMM (~20 days)
        """
        logger.info("Initializing paper trading engine...")
        logger.info("Calculating required historical data...")

        bars_3m, bars_15m, bars_1h = self._calculate_required_bars()

        # Calculate time spans
        hours_3m = bars_3m * 3 / 60  # bars * 3min / 60min
        days_15m = bars_15m * 15 / 60 / 24  # bars * 15min / 60min / 24h
        days_1h = bars_1h / 24  # bars / 24h

        logger.info(f"Required data:")
        logger.info(f"  - 3m: {bars_3m} bars (~{hours_3m:.1f} hours)")
        logger.info(f"  - 15m: {bars_15m} bars (~{days_15m:.1f} days)")
        logger.info(f"  - 1h: {bars_1h} bars (~{days_1h:.1f} days)")

        # Fetch 3m data
        logger.info(f"Fetching historical 3m data ({bars_3m} bars)...")
        self.df_3m = self.client.get_klines(self.symbol, "3m", limit=bars_3m)
        logger.info(f"  -> Received {len(self.df_3m)} bars")

        # Fetch 15m data
        logger.info(f"Fetching historical 15m data ({bars_15m} bars)...")
        self.df_15m = self.client.get_klines(self.symbol, "15m", limit=bars_15m)
        logger.info(f"  -> Received {len(self.df_15m)} bars")

        # Fetch 1h data for 17-feature HMM
        logger.info(f"Fetching historical 1h data ({bars_1h} bars)...")
        self.df_1h = self.client.get_klines(self.symbol, "1h", limit=bars_1h)
        logger.info(f"  -> Received {len(self.df_1h)} bars")

        # Validate data
        if len(self.df_3m) < self.DATA_REQUIREMENTS["3m"]["backtest_warmup"]:
            logger.warning(f"Insufficient 3m data: {len(self.df_3m)} < {self.DATA_REQUIREMENTS['3m']['backtest_warmup']}")

        if len(self.df_15m) < self.DATA_REQUIREMENTS["15m"]["hmm_training_min"]:
            logger.warning(f"Insufficient 15m data for HMM: {len(self.df_15m)} < {self.DATA_REQUIREMENTS['15m']['hmm_training_min']}")

        # Show data range
        if len(self.df_3m) > 0:
            start_3m = self.df_3m["timestamp"].iloc[0]
            end_3m = self.df_3m["timestamp"].iloc[-1]
            logger.info(f"  3m data range: {start_3m} to {end_3m}")

        if len(self.df_15m) > 0:
            start_15m = self.df_15m["timestamp"].iloc[0]
            end_15m = self.df_15m["timestamp"].iloc[-1]
            logger.info(f"  15m data range: {start_15m} to {end_15m}")

        if len(self.df_1h) > 0:
            start_1h = self.df_1h["timestamp"].iloc[0]
            end_1h = self.df_1h["timestamp"].iloc[-1]
            logger.info(f"  1h data range: {start_1h} to {end_1h}")

        # Load HMM feature config
        feature_config_path = Path(__file__).parent.parent / "configs" / "hmm_features.yaml"
        if feature_config_path.exists():
            with open(feature_config_path, "r") as f:
                feature_config = yaml.safe_load(f)
            self._fast_feature_names = feature_config["hmm"]["fast_3m"]["features"]
            self._mid_feature_names = feature_config["hmm"]["mid_15m"]["features"]
        else:
            # Fallback feature lists
            self._fast_feature_names = [
                "ret_3m", "ewm_ret_15m", "atr_pct_15m", "ewm_std_ret_15m",
                "bb_width_15m", "body_ratio", "upper_wick_ratio", "lower_wick_ratio", "range_pct"
            ]
            self._mid_feature_names = [
                "ret_15m", "ewm_ret_15m", "ret_1h", "ewm_ret_1h",
                "atr_pct_15m", "ewm_std_ret_15m", "atr_pct_1h", "ewm_std_ret_1h",
                "bb_width_15m", "vol_z_15m", "price_z_from_ema_1h"
            ]

        # Build features for Multi-HMM
        if self._use_multi_hmm:
            logger.info("Building Multi-HMM features...")
            logger.info(f"  Fast HMM (3m): {len(self._fast_feature_names)} features")
            logger.info(f"  Mid HMM (15m): {len(self._mid_feature_names)} features")

            # Build Fast HMM features (3m + 15m context)
            fast_features, fast_timestamps = build_fast_hmm_features_v2(
                self.df_3m, self._fast_feature_names
            )
            logger.info(f"  Fast features: {fast_features.shape}")

            # Build Mid HMM features (15m + 1h context)
            # Use OHLCV columns only to avoid column conflicts
            df_15m_ohlcv = self.df_15m[["timestamp", "open", "high", "low", "close", "volume"]].copy()
            mid_features, mid_timestamps = build_mid_hmm_features_v2(
                df_15m_ohlcv, self._mid_feature_names
            )
            logger.info(f"  Mid features: {mid_features.shape}")

            # Try to load checkpoint or train new
            loaded = False
            checkpoint_dir = Path(__file__).parent.parent / "checkpoints" / "hmm"

            if checkpoint_dir.exists():
                # First, try to load from latest_run_id.txt
                latest_run_id_file = checkpoint_dir / "latest_run_id.txt"
                run_id = None

                if latest_run_id_file.exists():
                    with open(latest_run_id_file, "r") as f:
                        run_id = f.read().strip()
                    logger.info(f"Found latest run ID: {run_id}")
                else:
                    # Fallback: find the latest checkpoint file
                    checkpoints = list(checkpoint_dir.glob("fast_hmm_*.json"))
                    if checkpoints:
                        latest = sorted(checkpoints)[-1]
                        run_id = latest.stem.replace("fast_hmm_", "")

                if run_id:
                    try:
                        self.multi_hmm.load_checkpoints(run_id)
                        loaded = True
                        logger.info(f"Multi-HMM loaded from checkpoint: {run_id}")
                    except Exception as e:
                        logger.warning(f"Failed to load checkpoint: {e}")

            if not loaded:
                logger.info("Training new Multi-HMM models...")
                self.multi_hmm.train(
                    fast_features=fast_features,
                    fast_feature_names=self._fast_feature_names,
                    mid_features=mid_features,
                    mid_feature_names=self._mid_feature_names,
                    fast_timestamps=fast_timestamps,
                    mid_timestamps=mid_timestamps,
                )
                # Save checkpoints
                self.multi_hmm.save_checkpoints()
                logger.info("Multi-HMM training complete")

        else:
            # Fallback: Build features for single HMM
            logger.info("Building single HMM features (fallback)...")
            hmm_features, self.df_3m, self.df_15m, self._hmm_feature_names = build_features_live(
                self.df_3m, self.df_15m, self.df_1h
            )

            n_features = len(self._hmm_feature_names)
            logger.info(f"HMM features: {n_features} ({self._hmm_feature_names[:3]}...)")

            # Load pre-trained HMM or train new one
            loaded = False
            if self.hmm_model_path and Path(self.hmm_model_path).exists():
                loaded = self.hmm.load_from_json(self.hmm_model_path, expected_features=n_features)

            if not loaded:
                logger.info(f"Training new HMM model with {len(hmm_features)} samples...")
                if len(hmm_features) >= self.DATA_REQUIREMENTS["15m"]["hmm_training_min"]:
                    self.hmm.train(hmm_features)
                    logger.info("HMM training complete")
                else:
                    logger.warning("Not enough data for robust HMM training, using available data")
                    if len(hmm_features) >= 100:
                        self.hmm.train(hmm_features)

        # Use second-to-last bar as the last "closed" bar
        # The last bar (iloc[-1]) might be the current incomplete bar
        self.last_3m_ts = self.df_3m["timestamp"].iloc[-2]
        logger.info(f"Initialization complete. Last closed 3m bar: {self.last_3m_ts}")
        logger.info(f"Current price: ${self.df_3m['close'].iloc[-1]:.4f}")
        logger.info("Starting trading loop... (heartbeat every 30s)")
        sys.stdout.flush()

    def update_data(self) -> bool:
        """Fetch latest data and check for new 3m bar.

        Returns:
            True if a new 3m bar closed
        """
        # Fetch latest klines
        new_3m = self.client.get_klines(self.symbol, "3m", limit=10)
        new_15m = self.client.get_klines(self.symbol, "15m", limit=10)
        new_1h = self.client.get_klines(self.symbol, "1h", limit=5)

        # Check for new 3m bar
        latest_ts = new_3m["timestamp"].iloc[-2]  # -1 is current (incomplete), -2 is last closed

        if self.last_3m_ts is None or latest_ts > self.last_3m_ts:
            # New bar closed - update dataframes
            # Merge new data
            self.df_3m = pd.concat([self.df_3m, new_3m]).drop_duplicates(subset=["timestamp"]).tail(500)
            self.df_15m = pd.concat([self.df_15m, new_15m]).drop_duplicates(subset=["timestamp"]).tail(500)
            self.df_1h = pd.concat([self.df_1h, new_1h]).drop_duplicates(subset=["timestamp"]).tail(500)

            # Rebuild features with 1h data for 17 features
            _, self.df_3m, self.df_15m, self._hmm_feature_names = build_features_live(
                self.df_3m, self.df_15m, self.df_1h
            )

            self.last_3m_ts = latest_ts
            return True

        return False

    def run_pipeline(self) -> Optional[Decision]:
        """Run the full pipeline on current data."""
        if self.df_3m is None or self.df_15m is None:
            return None

        bar = self.df_3m.iloc[-1]
        bar_15m = self.df_15m.iloc[-1]
        hist_15m = self.df_15m.tail(20)

        # Get HMM prediction
        if self._use_multi_hmm and self.multi_hmm and self.multi_hmm.is_trained:
            # Build features for prediction
            try:
                # Fast features (latest 3m bar)
                fast_features, _ = build_fast_hmm_features_v2(
                    self.df_3m.tail(100), self._fast_feature_names
                )
                fast_feat = fast_features[-1:] if len(fast_features) > 0 else None

                # Mid features (latest 15m bar)
                # Use OHLCV columns only to avoid column conflicts
                df_15m_ohlcv = self.df_15m[["timestamp", "open", "high", "low", "close", "volume"]].tail(100).copy()
                mid_features, _ = build_mid_hmm_features_v2(
                    df_15m_ohlcv, self._mid_feature_names
                )
                mid_feat = mid_features[-1:] if len(mid_features) > 0 else None

                # Get fused prediction
                regime_packet = self.multi_hmm.predict(fast_feat, mid_feat)
                regime_raw = regime_packet.label_fused.value if regime_packet.label_fused else "RANGE"
                confidence = max(regime_packet.conf_struct, regime_packet.conf_micro)

            except Exception as e:
                logger.warning(f"Multi-HMM prediction failed: {e}")
                regime_raw, confidence = "RANGE", 0.5
        else:
            # Fallback: single HMM
            available_cols = [c for c in self._hmm_feature_names if c in self.df_15m.columns]
            if available_cols:
                hmm_features = self.df_15m[available_cols].tail(20).dropna().values
            else:
                hmm_features = np.array([])

            if len(hmm_features) > 0:
                regime_raw, confidence = self.hmm.predict(hmm_features)
            else:
                regime_raw, confidence = "RANGE", 0.5

        # ConfirmLayer - Include ret_3m for faster spike detection (matching backtest)
        row_15m_dict = bar_15m.to_dict()
        row_15m_dict["ret_3m"] = bar.get("ret_3m", 0)  # Add 3m return for spike detection
        row_15m_dict["ret_15m"] = bar_15m.get("ret_15m", 0)  # Add 15m return for spike detection

        confirm_result, self.confirm_state = self.confirm_layer.confirm(
            regime_raw=regime_raw,
            row_15m=row_15m_dict,
            hist_15m=hist_15m,
            state=self.confirm_state,
        )

        confirm_ctx = ConfirmContext(
            regime_raw=regime_raw,
            regime_confirmed=confirm_result.confirmed_regime,
            confirm_reason=confirm_result.reason,
            confirm_metrics=confirm_result.metrics,
        )

        # Market context
        price = float(bar["close"])
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
        ret_3m = bar.get("ret_3m", 0)
        ema_slope_15m = bar_15m.get("ema_slope_15m", 0)

        if pd.isna(ema_fast):
            ema_fast = price
        if pd.isna(ema_slow):
            ema_slow = price
        if pd.isna(ret_3m):
            ret_3m = 0
        if pd.isna(ema_slope_15m):
            ema_slope_15m = 0

        # Get historical close prices for multi-bar returns (XGB feature fix)
        close_2 = self.df_3m.iloc[-3]["close"] if len(self.df_3m) >= 3 else price
        close_5 = self.df_3m.iloc[-6]["close"] if len(self.df_3m) >= 6 else price

        # Get volume data for volume_ratio (XGB feature fix)
        volume = bar.get("volume", 1.0)
        volume_ma = self.df_3m["volume"].tail(20).mean() if len(self.df_3m) >= 20 else volume

        # Get high/low for range_pct (XGB feature fix)
        high_3m = float(bar.get("high", price))
        low_3m = float(bar.get("low", price))

        market_ctx = MarketContext(
            symbol=self.symbol,
            ts=int(bar["timestamp"].timestamp() * 1000),
            price=price,
            row_3m={
                # Core OHLCV
                "close": price,
                "high": high_3m,
                "low": low_3m,
                "volume": volume,
                "volume_ma_20": volume_ma,  # Match backtest key name
                # Historical closes for multi-bar returns (XGB)
                "close_2": close_2,
                "close_5": close_5,
                # Technical indicators
                "atr_3m": atr_3m,
                "ema_fast_3m": ema_fast,
                "ema_slow_3m": ema_slow,
                "ret_3m": ret_3m,
                "ret": ret_3m,
                "volatility": bar.get("volatility_3m", 0.005),
                "rsi_3m": bar.get("rsi_3m", 50),
                "ema_slope_15m": ema_slope_15m,
                "ema_diff": (ema_fast - ema_slow) / price if price > 0 else 0,
                "price_to_ema20": (price - ema_fast) / ema_fast if ema_fast > 0 else 0,
                "price_to_ema50": (price - ema_slow) / ema_slow if ema_slow > 0 else 0,
            },
            row_15m={
                # Price data (needed for box/breakout calculations)
                "close": float(bar_15m.get("close", price)),
                "high": float(bar_15m.get("high", price)),
                "low": float(bar_15m.get("low", price)),
                # Spike detection (matching backtest)
                "ret_3m": bar.get("ret_3m", 0),  # For faster spike detection
                "ret_15m": bar_15m.get("ret_15m", 0),  # For 15m spike detection
                # EMA and trend indicators (needed for TREND/RANGE validation)
                "ema_slope_15m": ema_slope_15m,
                "ewm_ret_15m": bar_15m.get("ewm_ret_15m", 0),
                # Volatility metrics (needed for RANGE and HIGH_VOL validation)
                "ewm_std_ret_15m": bar_15m.get("ewm_std_ret_15m", 0.005),
                "atr_pct_15m": bar_15m.get("atr_pct_15m", 1.0),
                # Structure metrics (needed for HIGH_VOL exit)
                "bb_width_15m": bar_15m.get("bb_width_15m", 0.02),
                "range_comp_15m": bar_15m.get("range_comp_15m", 0.5),
            },
            zone={
                "support": support,
                "resistance": resistance,
                "strength": 0.0001,
                "dist_to_support": (price - support) / atr_3m if atr_3m > 0 else 999,
                "dist_to_resistance": (resistance - price) / atr_3m if atr_3m > 0 else 999,
            },
        )

        # Update position state
        if self.position.side != "FLAT":
            self.position.bars_held_3m += 1
            if self.position.side == "LONG":
                self.position.unrealized_pnl = (price - self.position.entry_price) * self.position.size
            else:
                self.position.unrealized_pnl = (self.position.entry_price - price) * self.position.size

        # FSM
        candidate_signal, self.fsm_state = self.fsm.step(
            ctx=market_ctx,
            confirm=confirm_ctx,
            pos=self.position,
            fsm_state=self.fsm_state,
        )

        # DecisionEngine
        decision, self.engine_state = self.decision_engine.decide(
            ctx=market_ctx,
            confirm=confirm_ctx,
            pos=self.position,
            cand=candidate_signal,
            engine_state=self.engine_state,
        )

        return decision, market_ctx, confirm_ctx, candidate_signal

    def execute_decision(self, decision: Decision, market_ctx: MarketContext) -> None:
        """Execute trading decision (paper)."""
        price = market_ctx.price
        fee_rate = 0.0004  # 0.04% taker fee

        if decision.action == "OPEN_LONG" and self.position.side == "FLAT":
            # Position size = equity * leverage / price
            position_value = self.equity * self.leverage
            size = position_value / price
            fee = price * size * fee_rate
            self.equity -= fee

            self.position = PositionState(
                side="LONG",
                entry_price=price,
                size=size,
                entry_ts=market_ctx.ts,
                bars_held_3m=0,
            )

            logger.info(f"[OPEN_LONG] Price: ${price:.4f}, Size: {size:.2f} XRP (${position_value:.0f}), Reason: {decision.reason}")

        elif decision.action == "OPEN_SHORT" and self.position.side == "FLAT":
            # Position size = equity * leverage / price
            position_value = self.equity * self.leverage
            size = position_value / price
            fee = price * size * fee_rate
            self.equity -= fee

            self.position = PositionState(
                side="SHORT",
                entry_price=price,
                size=size,
                entry_ts=market_ctx.ts,
                bars_held_3m=0,
            )

            logger.info(f"[OPEN_SHORT] Price: ${price:.4f}, Size: {size:.2f} XRP (${position_value:.0f}), Reason: {decision.reason}")

        elif decision.action == "CLOSE" and self.position.side != "FLAT":
            if self.position.side == "LONG":
                pnl = (price - self.position.entry_price) * self.position.size
            else:
                pnl = (self.position.entry_price - price) * self.position.size

            fee = price * self.position.size * fee_rate
            pnl -= fee
            self.equity += pnl
            self.total_pnl += pnl
            self.total_trades += 1

            if pnl > 0:
                self.winning_trades += 1

            win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0

            logger.info(
                f"[CLOSE_{self.position.side}] "
                f"Entry: ${self.position.entry_price:.4f}, Exit: ${price:.4f}, "
                f"PnL: ${pnl:.2f}, Bars: {self.position.bars_held_3m}, "
                f"Total PnL: ${self.total_pnl:.2f}, WR: {win_rate:.1f}%"
            )

            # Calculate PnL percentage
            pnl_pct = (pnl / (self.position.entry_price * self.position.size)) * 100
            profit_given_back = self.max_unrealized_pnl_pct - pnl_pct if self.max_unrealized_pnl_pct > 0 else 0

            trade = {
                "timestamp": datetime.now().isoformat(),
                "side": self.position.side,
                "entry_price": self.position.entry_price,
                "exit_price": price,
                "size": self.position.size,
                "pnl": pnl,
                "pnl_pct": pnl_pct,
                "bars_held": self.position.bars_held_3m,
                "equity_after": self.equity,
                "exit_reason": decision.reason,
                "max_profit_pct": self.max_unrealized_pnl_pct,  # Match backtest
                "profit_given_back": profit_given_back,         # Match backtest
                "regime": self.entry_regime,
                "signal": self.entry_signal,
            }
            self.trades.append(trade)

            # Save trade to file
            self._save_trade(trade)

            # Reset position and trailing stop tracking
            self.position = PositionState(side="FLAT")
            self.max_unrealized_pnl = 0.0
            self.max_unrealized_pnl_pct = 0.0
            self.breakeven_activated = False  # Reset break-even flag
            self.entry_regime = None
            self.entry_signal = None

    def run(self) -> None:
        """Main trading loop."""
        logger.info("=" * 60)
        logger.info("PAPER TRADING STARTED")
        logger.info(f"Symbol: {self.symbol}")
        logger.info(f"Initial Capital: ${self.initial_capital:.2f}")
        logger.info(f"Leverage: {self.leverage}x")
        logger.info(f"Max Position Size: ${self.initial_capital * self.leverage:.2f}")
        logger.info(f"Poll Interval: {self.poll_interval}s")
        logger.info("=" * 60)

        self.initialize()

        poll_count = 0
        last_heartbeat = 0

        while True:
            try:
                poll_count += 1

                # === Always fetch current price for tick tracking ===
                current_price = self.client.get_ticker_price(self.symbol)
                self._update_tick_history(current_price)

                # === Check for emergency exit (5-sec based) ===
                if self.position.side != "FLAT":
                    emergency_reason = self._check_emergency_exit(current_price)
                    if emergency_reason:
                        logger.info(f"[EMERGENCY] {emergency_reason}")
                        self._execute_emergency_exit(current_price, emergency_reason)

                # === 5-second Profit Protection (Break-even + Trailing Stop) ===
                if self.position.side != "FLAT":
                    # Update max profit tracking
                    self._update_max_profit_5s(current_price)

                    # Priority 1: Check break-even stop (손실 방지)
                    breakeven_result = self._check_breakeven_stop_5s(current_price)
                    if breakeven_result is not None:
                        exit_price, reason = breakeven_result
                        self.trailing_stop_stats["breakeven_triggered"] += 1
                        self._execute_trailing_stop_exit(exit_price, reason)
                    else:
                        # Priority 2: Check trailing stop (수익 보존)
                        trailing_result = self._check_trailing_stop_5s(current_price)
                        if trailing_result is not None:
                            exit_price, reason = trailing_result
                            self._execute_trailing_stop_exit(exit_price, reason)

                # === Check pending signal for optimized entry (5-sec based) ===
                if self.pending_signal is not None:
                    entry_result = self._check_entry_timing(current_price)
                    if entry_result and entry_result["execute"]:
                        logger.info(f"[OPTIMIZED ENTRY] {entry_result['reason']} @ ${current_price:.4f}")
                        self._execute_pending_signal(current_price, entry_result["reason"])

                # === Check for new 3m bar ===
                new_bar = self.update_data()

                if new_bar:
                    # Cancel pending signal on new bar (regime may have changed)
                    if self.pending_signal:
                        self._cancel_pending_signal("NEW_BAR")

                    # Run pipeline
                    result = self.run_pipeline()

                    if result:
                        decision, market_ctx, confirm_ctx, signal = result

                        # Log status
                        price = market_ctx.price
                        regime = confirm_ctx.regime_confirmed

                        # Detect sudden moves
                        sudden_move = self._detect_sudden_move()
                        sudden_info = ""
                        if sudden_move:
                            sudden_info = f" | SUDDEN_{sudden_move['direction']}({sudden_move['pct_change']*100:.2f}%)"

                        status = f"[NEW BAR] "
                        status += f"Price: ${price:.4f} | Regime: {regime} | "
                        status += f"Signal: {signal.signal} | Action: {decision.action}"
                        status += sudden_info

                        if self.position.side != "FLAT":
                            status += f" | Pos: {self.position.side} (${self.position.unrealized_pnl:.2f})"

                        logger.info(status)
                        sys.stdout.flush()

                        # Check if regime changed or trade action occurred
                        is_first_bar = (self.last_regime is None)
                        regime_changed = (not is_first_bar and regime != self.last_regime)
                        trade_action = decision.action in ["OPEN_LONG", "OPEN_SHORT", "CLOSE"]

                        # Log initial regime or regime change
                        if is_first_bar:
                            logger.info(f"[INITIAL REGIME] {regime}")
                        elif regime_changed:
                            logger.info(f"[REGIME CHANGE] {self.last_regime} -> {regime}")

                        # Increment bar counter for periodic logging
                        if not hasattr(self, '_bar_counter'):
                            self._bar_counter = 0
                        self._bar_counter += 1
                        periodic_log = (self._bar_counter % 20 == 0)  # Log every 20 bars

                        # Save to file on: initial bar, regime change, trade, or periodic
                        should_log = is_first_bar or regime_changed or trade_action or periodic_log
                        if should_log:
                            if trade_action:
                                event_type = "TRADE"
                            elif regime_changed:
                                event_type = "REGIME_CHANGE"
                            elif is_first_bar:
                                event_type = "INITIAL"
                            else:
                                event_type = "PERIODIC"
                            self._save_signal(
                                timestamp=datetime.now().isoformat(),
                                price=price,
                                regime_raw=confirm_ctx.regime_raw,
                                regime_confirmed=confirm_ctx.regime_confirmed,
                                confirm_reason=confirm_ctx.confirm_reason,
                                signal=signal.signal,
                                signal_reason=signal.reason,
                                signal_score=signal.score,
                                action=decision.action,
                                decision_reason=decision.reason,
                                market_ctx=market_ctx,
                                event_type=event_type,
                            )

                        # Update last regime
                        self.last_regime = regime

                        # Handle trade actions with entry optimization
                        if decision.action in ["OPEN_LONG", "OPEN_SHORT"]:
                            # Use pending signal for entry optimization
                            self._set_pending_signal(signal.signal, decision, market_ctx)
                        elif decision.action == "CLOSE":
                            # Execute close immediately
                            self.execute_decision(decision, market_ctx)
                else:
                    # Show heartbeat every 30 seconds
                    current_time = int(time.time())
                    if current_time - last_heartbeat >= 30:
                        last_heartbeat = current_time
                        pos_info = ""
                        pending_info = ""
                        if self.position.side != "FLAT":
                            if self.position.side == "LONG":
                                unrealized = (current_price - self.position.entry_price) * self.position.size
                            else:
                                unrealized = (self.position.entry_price - current_price) * self.position.size
                            pos_info = f" | {self.position.side}: ${unrealized:.2f}"
                        if self.pending_signal:
                            pending_info = f" | PENDING: {self.pending_signal['decision'].action}"

                        # Total PnL info
                        if self.total_pnl >= 0:
                            pnl_info = f" | PnL: +${self.total_pnl:.2f} ({self.total_trades}T)"
                        else:
                            pnl_info = f" | PnL: -${abs(self.total_pnl):.2f} ({self.total_trades}T)"
                        logger.info(f"[HEARTBEAT] Price: ${current_price:.4f}{pos_info}{pending_info}{pnl_info}")
                        sys.stdout.flush()

                time.sleep(self.poll_interval)

            except KeyboardInterrupt:
                logger.info("\n" + "=" * 60)
                logger.info("PAPER TRADING STOPPED")
                logger.info("=" * 60)
                self._print_summary()
                break
            except Exception as e:
                logger.error(f"Error: {e}")
                time.sleep(self.poll_interval)

    def _print_summary(self) -> None:
        """Print trading summary."""
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0

        logger.info(f"Total Trades: {self.total_trades}")
        logger.info(f"Winning Trades: {self.winning_trades}")
        logger.info(f"Win Rate: {win_rate:.1f}%")
        logger.info(f"Total PnL: ${self.total_pnl:.2f}")
        logger.info(f"Final Equity: ${self.equity:.2f}")
        logger.info(f"Return: {((self.equity / self.initial_capital) - 1) * 100:.2f}%")

        if self.position.side != "FLAT":
            logger.info(f"Open Position: {self.position.side} @ ${self.position.entry_price:.4f}")

        # 5-second entry timing statistics
        logger.info("")
        logger.info("5-Second Entry Timing Statistics:")
        stats = self.entry_5s_stats
        skip_rate = (stats["skipped"] / max(1, stats["total"])) * 100
        improve_rate = (stats["improved"] / max(1, stats["executed"])) * 100
        avg_saved = stats["total_saved_pct"] / max(1, stats["improved"])

        logger.info(f"  Entry Attempts: {stats['total']}")
        logger.info(f"  Skipped (chasing): {stats['skipped']} ({skip_rate:.1f}%)")
        logger.info(f"  Executed: {stats['executed']}")
        logger.info(f"  Improved Entry: {stats['improved']} ({improve_rate:.1f}%)")
        logger.info(f"  Avg Saved: {avg_saved:.3f}%")

        # Save summary to file
        self._save_summary()


def main():
    parser = argparse.ArgumentParser(description="Paper Trading with Binance")
    parser.add_argument("--symbol", default="XRPUSDT", help="Trading symbol")
    parser.add_argument("--capital", type=float, default=10000.0, help="Initial capital (USDT)")
    parser.add_argument("--leverage", type=float, default=5.0, help="Leverage multiplier (default: 5x)")
    parser.add_argument("--interval", type=int, default=5, help="Poll interval (seconds)")
    parser.add_argument("--hmm_model", type=str, default="outputs/backtest_binance/hmm_model_17feat.json",
                        help="Path to pre-trained HMM model (JSON)")
    args = parser.parse_args()

    # Check HMM model
    hmm_path = Path(args.hmm_model) if args.hmm_model else None
    if hmm_path and hmm_path.exists():
        logger.info(f"Using pre-trained HMM model: {args.hmm_model}")
    else:
        logger.warning(f"HMM model not found: {args.hmm_model}, will train new model")

    engine = PaperTradingEngine(
        symbol=args.symbol,
        initial_capital=args.capital,
        poll_interval=args.interval,
        hmm_model_path=args.hmm_model,
        leverage=args.leverage,
    )
    engine.run()


if __name__ == "__main__":
    main()
