#!/usr/bin/env python
"""Parameter tuning for HMM trading strategy.

Optimizes:
1. Trading strategy parameters (confidence, SL/TP multipliers)
2. Confirm Layer parameters (thresholds)
3. HMM parameters (n_states)

Usage:
    python scripts/tune_parameters.py
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

import numpy as np
import pandas as pd
from hmmlearn import hmm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from xrp4.regime.confirm import RegimeConfirmLayer, ConfirmConfig, ConfirmResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class TradingParams:
    """Trading strategy parameters."""
    min_confidence: float = 0.8
    sl_mult_trend: float = 2.0
    tp_mult_trend: float = 4.0
    sl_mult_range: float = 1.5
    tp_mult_range: float = 2.0
    risk_per_trade: float = 0.01
    fee_rate: float = 0.001


@dataclass
class ConfirmParams:
    """Confirm Layer parameters."""
    trend_confirm_b_atr: float = 0.8
    trend_confirm_s_atr: float = 0.07
    trend_confirm_ewm_ret_sigma: float = 0.20
    trend_confirm_consec_bars: int = 2
    high_vol_lambda_on: float = 1.50
    high_vol_lambda_off: float = 0.50
    high_vol_cooldown_bars: int = 4


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

        # Handle NaN
        if np.any(np.isnan(features)):
            features = features.copy()
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
        log_likelihood = self.model.score(features_scaled)

        # Map states to labels
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
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    })

    resampled = resampled.dropna(subset=["close"])
    resampled = resampled.reset_index()

    return resampled


def _resample_to_1h(df_3m: pd.DataFrame) -> pd.DataFrame:
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


def build_features(df_3m: pd.DataFrame) -> Tuple[np.ndarray, List[str], pd.DataFrame]:
    """Build multi-timeframe features for HMM."""
    df_15m = resample_to_15m(df_3m)
    df_1h = _resample_to_1h(df_3m)

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

    # BB width percentile
    df_15m["bb_width_15m_pct"] = df_15m["bb_width_15m"].rolling(50).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if x.max() != x.min() else 0.5
    )

    # Range compression (15m)
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

    # Merge 1h features to 15m
    df_15m = df_15m.set_index("timestamp")
    df_1h = df_1h.set_index("timestamp")

    features_1h = df_1h[["ret_1h", "ewm_ret_1h", "price_z_from_ema_1h", "atr_pct_1h", "ewm_std_ret_1h"]]
    df_result = df_15m.join(features_1h, how="left")
    df_result[["ret_1h", "ewm_ret_1h", "price_z_from_ema_1h", "atr_pct_1h", "ewm_std_ret_1h"]] = \
        df_result[["ret_1h", "ewm_ret_1h", "price_z_from_ema_1h", "atr_pct_1h", "ewm_std_ret_1h"]].ffill()

    df_result = df_result.reset_index()

    hmm_features = [
        "ret_15m", "ret_1h",
        "ewm_ret_15m", "ewm_ret_1h",
        "ema_slope_15m", "price_z_from_ema_1h",
        "atr_pct_15m", "atr_pct_1h",
        "ewm_std_ret_15m", "ewm_std_ret_1h",
        "bb_width_15m",
        "range_comp_15m", "bb_width_15m_pct",
        "body_ratio", "upper_wick_ratio", "lower_wick_ratio",
        "vol_z_15m",
    ]

    df_clean = df_result.dropna(subset=hmm_features)
    feature_matrix = df_clean[hmm_features].values

    return feature_matrix, hmm_features, df_clean


def run_backtest_with_params(
    df: pd.DataFrame,
    hmm_model: SingleHMM,
    features: np.ndarray,
    trading_params: TradingParams,
    confirm_params: ConfirmParams,
    initial_capital: float = 10000.0,
) -> Dict:
    """Run backtest with specific parameters."""

    # Create Confirm Layer with custom params
    confirm_cfg = ConfirmConfig(
        TREND_CONFIRM_B_ATR=confirm_params.trend_confirm_b_atr,
        TREND_CONFIRM_S_ATR=confirm_params.trend_confirm_s_atr,
        TREND_CONFIRM_EWM_RET_SIGMA=confirm_params.trend_confirm_ewm_ret_sigma,
        TREND_CONFIRM_CONSEC_BARS=confirm_params.trend_confirm_consec_bars,
        HIGH_VOL_LAMBDA_ON=confirm_params.high_vol_lambda_on,
        HIGH_VOL_LAMBDA_OFF=confirm_params.high_vol_lambda_off,
        HIGH_VOL_COOLDOWN_BARS_15M=confirm_params.high_vol_cooldown_bars,
    )

    # Get regime predictions
    state_seq, state_probs, labels = hmm_model.predict_sequence(features)

    df_bt = df.copy()
    df_bt["regime_raw"] = labels
    df_bt["confidence"] = [probs.max() for probs in state_probs]

    # Calculate ATR
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
                regime_raw=regime_raw,
                row_15m=row,
                hist_15m=hist,
                state=confirm_state,
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

    MIN_CONF = trading_params.min_confidence

    for idx in range(2, len(df_bt)):
        bar = df_bt.iloc[idx]
        prev_bar = df_bt.iloc[idx - 1]
        prev2_bar = df_bt.iloc[idx - 2]

        regime = bar["regime"]
        conf = bar["confidence"]
        current_atr = bar["atr"]

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
            else:
                if bar["high"] >= position["sl"]:
                    hit_sl = True
                    exit_price = position["sl"]
                elif bar["low"] <= position["tp"]:
                    hit_tp = True
                    exit_price = position["tp"]

            if hit_sl or hit_tp:
                if position["side"] == "long":
                    pnl = (exit_price - position["entry"]) * position["size"]
                else:
                    pnl = (position["entry"] - exit_price) * position["size"]

                fee = exit_price * position["size"] * trading_params.fee_rate
                pnl -= fee
                equity += pnl

                trades.append({
                    "pnl": pnl,
                    "pnl_pct": pnl / position["entry_equity"] * 100,
                    "exit_reason": "sl" if hit_sl else "tp",
                    "regime": position["regime"],
                })
                position = None

        # Generate signals
        if position is None and conf >= MIN_CONF and equity > 0:
            signal = None
            sl_mult = 2.0
            tp_mult = 3.0

            # TREND_UP: Long on pullback
            if regime == "TREND_UP":
                if prev_bar["close"] < prev2_bar["close"]:
                    signal = "long"
                    sl_mult = trading_params.sl_mult_trend
                    tp_mult = trading_params.tp_mult_trend

            # TREND_DOWN: Short on bounce
            elif regime == "TREND_DOWN":
                if prev_bar["close"] > prev2_bar["close"]:
                    signal = "short"
                    sl_mult = trading_params.sl_mult_trend
                    tp_mult = trading_params.tp_mult_trend

            # RANGE: Mean reversion
            elif regime == "RANGE":
                roll_high = high.iloc[max(0, idx-20):idx].max()
                roll_low = low.iloc[max(0, idx-20):idx].min()
                mid_price = (roll_high + roll_low) / 2

                if bar["close"] < mid_price and prev_bar["close"] < prev2_bar["close"]:
                    signal = "long"
                    sl_mult = trading_params.sl_mult_range
                    tp_mult = trading_params.tp_mult_range
                elif bar["close"] > mid_price and prev_bar["close"] > prev2_bar["close"]:
                    signal = "short"
                    sl_mult = trading_params.sl_mult_range
                    tp_mult = trading_params.tp_mult_range

            # Execute signal
            if signal is not None:
                entry_price = float(bar["close"])
                sl_distance = current_atr * sl_mult
                tp_distance = current_atr * tp_mult

                risk_amount = equity * trading_params.risk_per_trade
                size = risk_amount / sl_distance

                max_size = equity / entry_price * 0.5
                size = min(size, max_size)

                if size > 0:
                    fee = entry_price * size * trading_params.fee_rate
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

        fee = exit_price * position["size"] * trading_params.fee_rate
        pnl -= fee
        equity += pnl

        trades.append({
            "pnl": pnl,
            "pnl_pct": pnl / position["entry_equity"] * 100,
            "exit_reason": "eod",
            "regime": position["regime"],
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

        # Max drawdown
        eq_series = pd.Series(equity_curve)
        peak = eq_series.expanding().max()
        drawdown = (eq_series - peak) / peak * 100
        results["max_drawdown_pct"] = drawdown.min()
    else:
        results["win_rate"] = 0
        results["profit_factor"] = 0
        results["max_drawdown_pct"] = 0

    return results


def calculate_score(results: Dict) -> float:
    """Calculate optimization score from backtest results.

    Score = Return * sqrt(ProfitFactor) * WinRate / (1 + abs(MaxDD))
    """
    ret = results.get("total_return_pct", -100)
    pf = results.get("profit_factor", 0)
    wr = results.get("win_rate", 0)
    mdd = abs(results.get("max_drawdown_pct", -100))
    n_trades = results.get("n_trades", 0)

    # Minimum trades required
    if n_trades < 50:
        return -1000

    # Calculate score
    pf_adj = min(pf, 5)  # Cap profit factor

    # Prioritize positive returns with reasonable drawdown
    if ret > 0:
        score = ret * np.sqrt(pf_adj) * (wr / 100) / (1 + mdd / 100)
    else:
        score = ret * (1 + mdd / 100)  # Penalize negative returns more with high DD

    return score


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
    parser = argparse.ArgumentParser(description="Parameter tuning for HMM strategy")
    parser.add_argument("--start", type=str, default="2023-01-01", help="Start date")
    parser.add_argument("--end", type=str, default="2024-12-31", help="End date")
    parser.add_argument("--symbol", type=str, default="XRPUSDT", help="Symbol")
    parser.add_argument("--n_states", type=int, default=5, help="Number of HMM states")
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/tune_results"))

    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("PARAMETER TUNING FOR HMM TRADING STRATEGY")
    logger.info("=" * 70)

    # Load data
    logger.info("Loading data from TimescaleDB...")
    start_dt = datetime.fromisoformat(args.start)
    end_dt = datetime.fromisoformat(args.end)

    df_3m = load_ohlcv_from_timescaledb(args.symbol, start_dt, end_dt, "3m")

    if df_3m.empty:
        logger.error("No data loaded!")
        sys.exit(1)

    logger.info(f"Loaded {len(df_3m)} 3m bars")

    # Build features
    logger.info("Building features...")
    features, feature_names, df_features = build_features(df_3m)
    logger.info(f"Feature matrix shape: {features.shape}")

    # Split: 60% train, 20% validation, 20% test
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
    logger.info("Training HMM model...")
    hmm_model = SingleHMM(n_states=args.n_states)
    hmm_model.train(train_features, feature_names)
    logger.info(f"State labels: {hmm_model.state_labels}")

    # Define parameter grid
    param_grid = {
        # Trading parameters
        "min_confidence": [0.6, 0.7, 0.8],
        "sl_mult_trend": [1.5, 2.0, 2.5, 3.0],
        "tp_mult_trend": [2.0, 3.0, 4.0, 5.0],
        "sl_mult_range": [1.0, 1.5, 2.0],
        "tp_mult_range": [1.5, 2.0, 2.5],

        # Confirm Layer parameters
        "trend_confirm_b_atr": [0.5, 0.8, 1.0, 1.2],
        "high_vol_lambda_on": [1.2, 1.5, 1.8, 2.0],
        "high_vol_cooldown_bars": [2, 4, 6, 8],
    }

    # Simplified grid for faster search (subset of key parameters)
    key_params = {
        "min_confidence": [0.6, 0.7, 0.8],
        "sl_mult_trend": [1.5, 2.0, 2.5],
        "tp_mult_trend": [3.0, 4.0, 5.0],
        "trend_confirm_b_atr": [0.5, 0.8, 1.0],
        "high_vol_lambda_on": [1.2, 1.5, 2.0],
        "high_vol_cooldown_bars": [2, 4, 6],
    }

    # Calculate total combinations
    total_combos = 1
    for k, v in key_params.items():
        total_combos *= len(v)

    logger.info(f"Total parameter combinations to test: {total_combos}")

    # Grid search
    logger.info("\n" + "=" * 70)
    logger.info("GRID SEARCH ON VALIDATION SET")
    logger.info("=" * 70)

    best_score = -float("inf")
    best_params = None
    best_results = None
    all_results = []

    param_keys = list(key_params.keys())
    param_values = [key_params[k] for k in param_keys]

    for i, combo in enumerate(product(*param_values)):
        params = dict(zip(param_keys, combo))

        # Create parameter objects
        trading_params = TradingParams(
            min_confidence=params["min_confidence"],
            sl_mult_trend=params["sl_mult_trend"],
            tp_mult_trend=params["tp_mult_trend"],
            sl_mult_range=1.5,  # Fixed for now
            tp_mult_range=2.0,  # Fixed for now
        )

        confirm_params = ConfirmParams(
            trend_confirm_b_atr=params["trend_confirm_b_atr"],
            high_vol_lambda_on=params["high_vol_lambda_on"],
            high_vol_cooldown_bars=params["high_vol_cooldown_bars"],
        )

        # Run backtest on validation set
        results = run_backtest_with_params(
            val_df, hmm_model, val_features,
            trading_params, confirm_params,
        )

        score = calculate_score(results)

        all_results.append({
            "params": params.copy(),
            "results": results.copy(),
            "score": score,
        })

        if score > best_score:
            best_score = score
            best_params = params.copy()
            best_results = results.copy()

        # Progress update
        if (i + 1) % 50 == 0:
            logger.info(
                f"Progress: {i+1}/{total_combos} | "
                f"Best score: {best_score:.2f} | "
                f"Best return: {best_results['total_return_pct']:.2f}%"
            )

    logger.info("\n" + "=" * 70)
    logger.info("GRID SEARCH COMPLETE")
    logger.info("=" * 70)

    logger.info(f"\nBest validation score: {best_score:.2f}")
    logger.info(f"\nBest parameters:")
    for k, v in best_params.items():
        logger.info(f"  {k}: {v}")

    logger.info(f"\nValidation results with best params:")
    logger.info(f"  Return: {best_results['total_return_pct']:.2f}%")
    logger.info(f"  Win Rate: {best_results['win_rate']:.1f}%")
    logger.info(f"  Profit Factor: {best_results['profit_factor']:.2f}")
    logger.info(f"  Max DD: {best_results['max_drawdown_pct']:.2f}%")
    logger.info(f"  N Trades: {best_results['n_trades']}")

    # Evaluate on test set with best parameters
    logger.info("\n" + "=" * 70)
    logger.info("FINAL EVALUATION ON TEST SET")
    logger.info("=" * 70)

    trading_params_best = TradingParams(
        min_confidence=best_params["min_confidence"],
        sl_mult_trend=best_params["sl_mult_trend"],
        tp_mult_trend=best_params["tp_mult_trend"],
        sl_mult_range=1.5,
        tp_mult_range=2.0,
    )

    confirm_params_best = ConfirmParams(
        trend_confirm_b_atr=best_params["trend_confirm_b_atr"],
        high_vol_lambda_on=best_params["high_vol_lambda_on"],
        high_vol_cooldown_bars=best_params["high_vol_cooldown_bars"],
    )

    test_results = run_backtest_with_params(
        test_df, hmm_model, test_features,
        trading_params_best, confirm_params_best,
    )

    logger.info(f"\nTest set results:")
    logger.info(f"  Return: {test_results['total_return_pct']:.2f}%")
    logger.info(f"  Win Rate: {test_results['win_rate']:.1f}%")
    logger.info(f"  Profit Factor: {test_results['profit_factor']:.2f}")
    logger.info(f"  Max DD: {test_results['max_drawdown_pct']:.2f}%")
    logger.info(f"  N Trades: {test_results['n_trades']}")

    # Also test with default parameters for comparison
    logger.info("\n" + "-" * 40)
    logger.info("Comparison with default parameters:")

    default_trading = TradingParams()
    default_confirm = ConfirmParams()

    default_test_results = run_backtest_with_params(
        test_df, hmm_model, test_features,
        default_trading, default_confirm,
    )

    logger.info(f"  Return: {default_test_results['total_return_pct']:.2f}%")
    logger.info(f"  Win Rate: {default_test_results['win_rate']:.1f}%")
    logger.info(f"  Profit Factor: {default_test_results['profit_factor']:.2f}")
    logger.info(f"  Max DD: {default_test_results['max_drawdown_pct']:.2f}%")

    improvement = test_results['total_return_pct'] - default_test_results['total_return_pct']
    logger.info(f"\nImprovement: {improvement:+.2f}%")

    # Save results
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Save best parameters
    best_params_file = args.output_dir / "best_params.json"
    with open(best_params_file, "w") as f:
        json.dump({
            "trading_params": {
                "min_confidence": best_params["min_confidence"],
                "sl_mult_trend": best_params["sl_mult_trend"],
                "tp_mult_trend": best_params["tp_mult_trend"],
                "sl_mult_range": 1.5,
                "tp_mult_range": 2.0,
            },
            "confirm_params": {
                "trend_confirm_b_atr": best_params["trend_confirm_b_atr"],
                "high_vol_lambda_on": best_params["high_vol_lambda_on"],
                "high_vol_cooldown_bars": best_params["high_vol_cooldown_bars"],
            },
            "validation_results": best_results,
            "test_results": test_results,
            "default_test_results": default_test_results,
        }, f, indent=2)

    logger.info(f"\nBest parameters saved to {best_params_file}")

    # Top 10 parameter sets
    logger.info("\n" + "=" * 70)
    logger.info("TOP 10 PARAMETER SETS")
    logger.info("=" * 70)

    sorted_results = sorted(all_results, key=lambda x: x["score"], reverse=True)

    for rank, item in enumerate(sorted_results[:10], 1):
        logger.info(
            f"\n#{rank}: Score={item['score']:.2f}, "
            f"Return={item['results']['total_return_pct']:.2f}%, "
            f"WR={item['results']['win_rate']:.1f}%, "
            f"PF={item['results']['profit_factor']:.2f}"
        )
        logger.info(f"   Params: {item['params']}")

    logger.info("\nDone!")


if __name__ == "__main__":
    main()
