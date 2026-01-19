#!/usr/bin/env python
"""Backtest with full FSM + DecisionEngine pipeline.

Pipeline:
Features -> HMM(raw) -> ConfirmLayer(confirmed_regime) -> FSM(candidate signal) -> DecisionEngine(final action) -> Execution

Usage:
    python scripts/backtest_fsm_pipeline.py
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from xrp4.regime.confirm import RegimeConfirmLayer, ConfirmConfig
from xrp4.core.types import (
    ConfirmContext,
    MarketContext,
    PositionState,
    CandidateSignal,
    Decision,
)
from xrp4.core.fsm import TradingFSM, FSMConfig
from xrp4.core.decision_engine import DecisionEngine, DecisionConfig

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def generate_backtest_chart(
    df_3m: pd.DataFrame,
    trades_df: pd.DataFrame,
    signals_df: pd.DataFrame,
    output_path: Path,
    title: str = "Backtest Results"
) -> None:
    """Generate TradingView-style interactive chart from backtest results."""
    if not PLOTLY_AVAILABLE:
        logger.warning("Plotly not installed. Skipping chart generation.")
        return

    logger.info("Generating backtest chart...")

    # Sample data for faster rendering
    max_points = 5000
    if len(df_3m) > max_points:
        step = len(df_3m) // max_points
        df_plot = df_3m.iloc[::step].copy()
    else:
        df_plot = df_3m.copy()

    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3],
        subplot_titles=('Price & Trades', 'Cumulative PnL')
    )

    # 1. Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df_plot['timestamp'],
            open=df_plot['open'],
            high=df_plot['high'],
            low=df_plot['low'],
            close=df_plot['close'],
            name='XRPUSDT',
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350',
        ),
        row=1, col=1
    )

    # 2. Regime background colors
    if 'regime_confirmed' in signals_df.columns:
        regime_colors = {
            'TREND_UP': 'rgba(0, 255, 0, 0.1)',
            'TREND_DOWN': 'rgba(255, 0, 0, 0.1)',
            'RANGE': 'rgba(0, 0, 255, 0.08)',
            'TRANSITION': 'rgba(255, 255, 0, 0.1)',
            'HIGH_VOL': 'rgba(255, 165, 0, 0.15)',
        }
        signals_sampled = signals_df.iloc[::max(1, len(signals_df)//1000)]
        signals_sampled = signals_sampled.copy()
        signals_sampled['regime_group'] = (signals_sampled['regime_confirmed'] != signals_sampled['regime_confirmed'].shift()).cumsum()

        for _, group in signals_sampled.groupby('regime_group'):
            regime = group['regime_confirmed'].iloc[0]
            color = regime_colors.get(regime, 'rgba(200, 200, 200, 0.05)')
            fig.add_vrect(
                x0=group['timestamp'].iloc[0],
                x1=group['timestamp'].iloc[-1],
                fillcolor=color,
                layer="below",
                line_width=0,
                row=1, col=1
            )

    # 3. Trade markers
    if len(trades_df) > 0 and 'timestamp' in signals_df.columns:
        signals_df_ts = signals_df.set_index('idx') if 'idx' in signals_df.columns else signals_df

        for _, trade in trades_df.iterrows():
            exit_idx = trade['exit_idx']
            bars_held = int(trade.get('bars_held', 1))
            entry_idx = exit_idx - bars_held

            # Get timestamps
            if exit_idx in signals_df_ts.index:
                exit_time = signals_df_ts.loc[exit_idx, 'timestamp']
            else:
                continue
            if entry_idx in signals_df_ts.index:
                entry_time = signals_df_ts.loc[entry_idx, 'timestamp']
            else:
                entry_time = exit_time

            side = trade['side']
            pnl = trade['pnl']

            # Entry marker
            marker_color = 'green' if side == 'LONG' else 'red'
            marker_symbol = 'triangle-up' if side == 'LONG' else 'triangle-down'
            fig.add_trace(
                go.Scatter(
                    x=[entry_time],
                    y=[trade['entry_price']],
                    mode='markers',
                    marker=dict(symbol=marker_symbol, size=10, color=marker_color),
                    name=f'{side} Entry',
                    showlegend=False,
                    hovertemplate=f'{side} Entry<br>Price: ${trade["entry_price"]:.4f}<extra></extra>',
                ),
                row=1, col=1
            )

            # Exit marker
            exit_color = 'lime' if pnl > 0 else 'salmon'
            fig.add_trace(
                go.Scatter(
                    x=[exit_time],
                    y=[trade['exit_price']],
                    mode='markers',
                    marker=dict(symbol='circle', size=8, color=exit_color),
                    name='Exit',
                    showlegend=False,
                    hovertemplate=f'Exit<br>Price: ${trade["exit_price"]:.4f}<br>PnL: ${pnl:.2f}<extra></extra>',
                ),
                row=1, col=1
            )

    # 4. Cumulative PnL
    if len(trades_df) > 0:
        trades_sorted = trades_df.copy()
        trades_sorted['cum_pnl'] = trades_sorted['pnl'].cumsum()

        pnl_times = []
        pnl_values = []
        signals_df_ts = signals_df.set_index('idx') if 'idx' in signals_df.columns else signals_df
        for _, trade in trades_sorted.iterrows():
            if trade['exit_idx'] in signals_df_ts.index:
                pnl_times.append(signals_df_ts.loc[trade['exit_idx'], 'timestamp'])
                pnl_values.append(trade['cum_pnl'])

        if pnl_times:
            fig.add_trace(
                go.Scatter(
                    x=pnl_times,
                    y=pnl_values,
                    mode='lines+markers',
                    line=dict(color='blue', width=2),
                    marker=dict(size=4),
                    name='Cumulative PnL',
                ),
                row=2, col=1
            )
            fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)

    # 5. Statistics annotation
    total_trades = len(trades_df)
    if total_trades > 0:
        wins = len(trades_df[trades_df['pnl'] > 0])
        win_rate = wins / total_trades * 100
        total_pnl = trades_df['pnl'].sum()
        gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
        pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        stats_text = (
            f"<b>Stats</b><br>"
            f"Trades: {total_trades}<br>"
            f"WR: {win_rate:.1f}%<br>"
            f"PF: {pf:.2f}<br>"
            f"PnL: ${total_pnl:.2f}"
        )
        fig.add_annotation(
            xref="paper", yref="paper", x=0.01, y=0.99,
            text=stats_text, showarrow=False,
            font=dict(size=10, family="monospace"),
            align="left", bgcolor="rgba(255,255,255,0.8)",
            bordercolor="gray", borderwidth=1,
        )

    # Layout
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=16)),
        xaxis_rangeslider_visible=False,
        height=800,
        showlegend=False,
        hovermode='x unified',
    )
    fig.update_xaxes(title_text="Time", row=2, col=1)
    fig.update_yaxes(title_text="Price (USDT)", row=1, col=1)
    fig.update_yaxes(title_text="PnL ($)", row=2, col=1)

    # Save
    fig.write_html(str(output_path))
    logger.info(f"Chart saved to: {output_path}")


class SingleHMM:
    """Single HMM model for regime detection.

    Matches tuning script (tune_trend_pullback.py) exactly:
    - No z-score scaling
    - n_iter=500
    - State labeling by vol_mean (not variance)
    """

    def __init__(
        self,
        n_states: int = 5,
        covariance_type: str = "full",
        n_iter: int = 500,  # Match tuning: 500 iterations
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
        # Store raw feature stats for state labeling (no scaling)
        self._train_features: Optional[np.ndarray] = None

    def _create_model(self) -> hmm.GaussianHMM:
        return hmm.GaussianHMM(
            n_components=self.n_states,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=self.random_state,
            verbose=False,
        )

    def train(self, features: np.ndarray, feature_names: List[str]) -> float:
        """Train HMM without scaling (matches tuning script)."""
        n_samples, n_features = features.shape

        # Handle NaN by filling with column means
        if np.any(np.isnan(features)):
            features = features.copy()
            col_means = np.nanmean(features, axis=0)
            for i in range(n_features):
                mask = np.isnan(features[:, i])
                features[mask, i] = col_means[i]

        # NO SCALING - matches tuning script exactly!
        # Tuning: hmm.fit(X) where X is raw values
        self._train_features = features.copy()

        self.model = self._create_model()
        self.model.fit(features)  # Raw features, no scaling!
        self.feature_names = feature_names
        log_likelihood = self.model.score(features)

        # Get state predictions for labeling
        states = self.model.predict(features)
        self._map_states_to_labels_tuning_style(features, states)
        self._is_trained = True
        return log_likelihood

    def _map_states_to_labels_tuning_style(self, features: np.ndarray, states: np.ndarray) -> None:
        """State labeling that matches tuning script exactly.

        Tuning approach:
        1. Compute ret_mean and vol_mean for each state
        2. HIGH_VOL = state with highest vol_mean
        3. TREND_UP = state with highest ret_mean (excluding HIGH_VOL)
        4. TREND_DOWN = state with lowest ret_mean (excluding HIGH_VOL)
        5. Remaining = TRANSITION, RANGE
        """
        if self.model is None:
            return

        # Find ret and vol feature indices
        ret_idx = 0
        vol_idx = 1
        for i, name in enumerate(self.feature_names):
            if name in ("ret", "ret_15m"):
                ret_idx = i
            elif name in ("vol", "vol_15m"):
                vol_idx = i

        # Compute state statistics (like tuning script)
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

        # 1. HIGH_VOL = highest vol_mean
        vol_sorted = sorted(state_stats.items(), key=lambda x: x[1]["vol_mean"], reverse=True)
        self.state_labels[vol_sorted[0][0]] = "HIGH_VOL"
        used.add(vol_sorted[0][0])

        # 2. TREND_UP = highest ret_mean (excluding HIGH_VOL)
        # 3. TREND_DOWN = lowest ret_mean (excluding HIGH_VOL)
        ret_sorted = sorted(
            [(k, v) for k, v in state_stats.items() if k not in used],
            key=lambda x: x[1]["ret_mean"],
            reverse=True
        )
        if len(ret_sorted) >= 1:
            self.state_labels[ret_sorted[0][0]] = "TREND_UP"
            used.add(ret_sorted[0][0])
        if len(ret_sorted) >= 2:
            self.state_labels[ret_sorted[-1][0]] = "TREND_DOWN"
            used.add(ret_sorted[-1][0])

        # 4. Remaining = TRANSITION, RANGE (in that order, like tuning)
        remaining = [k for k in range(self.n_states) if k not in used]
        if len(remaining) >= 2:
            self.state_labels[remaining[0]] = "TRANSITION"
            self.state_labels[remaining[1]] = "RANGE"
        elif len(remaining) == 1:
            self.state_labels[remaining[0]] = "RANGE"

    @classmethod
    def from_saved_model(cls, model_path: Path) -> "SingleHMM":
        """Load a pre-trained HMM model from saved bundle.

        This ensures consistency between tuning and backtesting by using
        the exact same model with the same state labels.
        """
        import pickle

        with open(model_path, "rb") as f:
            data = pickle.load(f)

        # Handle dict format (new) - more portable
        if isinstance(data, dict):
            n_states = data["n_states"]
            model = data["model"]
            state_labels = data["state_labels"]
            feature_names = data["feature_names"]
            train_start = data["train_start"]
            train_end = data["train_end"]
            created_at = data["created_at"]
        else:
            # Old format - dataclass object
            n_states = data.n_states
            model = data.model
            state_labels = data.state_labels
            feature_names = data.feature_names
            train_start = data.train_start
            train_end = data.train_end
            created_at = data.created_at

        # Create instance and populate
        instance = cls(n_states=n_states)
        instance.model = model
        instance.state_labels = state_labels
        instance.feature_names = feature_names
        instance._is_trained = True
        instance._train_features = np.zeros((1, len(feature_names)))

        logger.info(f"Loaded HMM model from: {model_path}")
        logger.info(f"  Train period: {train_start} ~ {train_end}")
        logger.info(f"  State labels: {state_labels}")
        logger.info(f"  Created at: {created_at}")

        return instance

    def predict_sequence(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict regime sequence (no scaling, matches tuning)."""
        if not self._is_trained:
            raise RuntimeError("Model not trained")

        # Handle NaN
        if np.any(np.isnan(features)):
            features = features.copy()
            train_means = np.mean(self._train_features, axis=0)
            for i in range(features.shape[1]):
                mask = np.isnan(features[:, i])
                features[mask, i] = train_means[i]

        # NO SCALING - raw features
        _, state_seq = self.model.decode(features, algorithm="viterbi")
        state_probs = self.model.predict_proba(features)
        labels = np.array([self.state_labels.get(s, "UNKNOWN") for s in state_seq])
        return state_seq, state_probs, labels


class MultiHMM:
    """Multi-timeframe HMM: 3m + 15m with weighted fusion.

    Combines short-term (3m) and structural (15m) HMM predictions
    using weighted fusion for more robust regime detection.
    """

    def __init__(
        self,
        n_states_3m: int = 4,
        n_states_15m: int = 4,
        fast_weight: float = 0.4,
        mid_weight: float = 0.6,
        n_iter: int = 500,
        random_state: int = 42,
    ):
        self.n_states_3m = n_states_3m
        self.n_states_15m = n_states_15m
        self.fast_weight = fast_weight
        self.mid_weight = mid_weight
        self.n_iter = n_iter
        self.random_state = random_state

        self.model_3m: Optional[hmm.GaussianHMM] = None
        self.model_15m: Optional[hmm.GaussianHMM] = None
        self.state_labels_3m: Dict[int, str] = {}
        self.state_labels_15m: Dict[int, str] = {}
        self.feature_names_3m: List[str] = []
        self.feature_names_15m: List[str] = []
        self._is_trained = False

    def train(
        self,
        features_3m: np.ndarray,
        feature_names_3m: List[str],
        features_15m: np.ndarray,
        feature_names_15m: List[str],
    ) -> Dict[str, float]:
        """Train both HMM models."""
        self.feature_names_3m = feature_names_3m
        self.feature_names_15m = feature_names_15m

        # Train 3m HMM
        features_3m = self._handle_nan(features_3m)
        self.model_3m = hmm.GaussianHMM(
            n_components=self.n_states_3m,
            covariance_type="full",
            n_iter=self.n_iter,
            random_state=self.random_state,
            verbose=False,
        )
        self.model_3m.fit(features_3m)
        ll_3m = self.model_3m.score(features_3m)
        states_3m = self.model_3m.predict(features_3m)
        self.state_labels_3m = self._map_states(features_3m, states_3m, feature_names_3m, self.n_states_3m)

        # Train 15m HMM
        features_15m = self._handle_nan(features_15m)
        self.model_15m = hmm.GaussianHMM(
            n_components=self.n_states_15m,
            covariance_type="full",
            n_iter=self.n_iter,
            random_state=self.random_state,
            verbose=False,
        )
        self.model_15m.fit(features_15m)
        ll_15m = self.model_15m.score(features_15m)
        states_15m = self.model_15m.predict(features_15m)
        self.state_labels_15m = self._map_states(features_15m, states_15m, feature_names_15m, self.n_states_15m)

        self._is_trained = True
        logger.info(f"MultiHMM trained: 3m states={self.state_labels_3m}, 15m states={self.state_labels_15m}")
        return {"ll_3m": ll_3m, "ll_15m": ll_15m}

    def _handle_nan(self, features: np.ndarray) -> np.ndarray:
        """Handle NaN values."""
        if np.any(np.isnan(features)):
            features = features.copy()
            col_means = np.nanmean(features, axis=0)
            for i in range(features.shape[1]):
                mask = np.isnan(features[:, i])
                features[mask, i] = col_means[i]
        return features

    def _map_states(
        self, features: np.ndarray, states: np.ndarray, feature_names: List[str], n_states: int
    ) -> Dict[int, str]:
        """Map states to labels based on ret_mean and vol_mean."""
        ret_idx, vol_idx = 0, 1
        for i, name in enumerate(feature_names):
            if "ret" in name.lower():
                ret_idx = i
            elif "vol" in name.lower():
                vol_idx = i

        state_stats = {}
        for s in range(n_states):
            mask = states == s
            if mask.sum() > 0:
                state_stats[s] = {
                    "ret_mean": features[mask, ret_idx].mean(),
                    "vol_mean": features[mask, vol_idx].mean(),
                }
            else:
                state_stats[s] = {"ret_mean": 0.0, "vol_mean": 0.0}

        labels = {}
        used = set()

        # HIGH_VOL = highest vol
        vol_sorted = sorted(state_stats.items(), key=lambda x: x[1]["vol_mean"], reverse=True)
        labels[vol_sorted[0][0]] = "HIGH_VOL"
        used.add(vol_sorted[0][0])

        # TREND_UP = highest ret, TREND_DOWN = lowest ret
        ret_sorted = sorted(
            [(k, v) for k, v in state_stats.items() if k not in used],
            key=lambda x: x[1]["ret_mean"],
            reverse=True
        )
        if len(ret_sorted) >= 1:
            labels[ret_sorted[0][0]] = "TREND_UP"
            used.add(ret_sorted[0][0])
        if len(ret_sorted) >= 2:
            labels[ret_sorted[-1][0]] = "TREND_DOWN"
            used.add(ret_sorted[-1][0])

        # Remaining = RANGE
        for k in range(n_states):
            if k not in used:
                labels[k] = "RANGE"

        return labels

    def predict_sequence(
        self,
        features_3m: np.ndarray,
        features_15m: np.ndarray,
        timestamps_3m: pd.DatetimeIndex,
        timestamps_15m: pd.DatetimeIndex,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict fused regime labels (returns same format as SingleHMM)."""
        features_3m = self._handle_nan(features_3m)
        features_15m = self._handle_nan(features_15m)

        # Predict states for 3m
        states_3m = self.model_3m.predict(features_3m)
        posteriors_3m = self.model_3m.predict_proba(features_3m)
        labels_3m = [self.state_labels_3m.get(s, "UNKNOWN") for s in states_3m]
        conf_3m = posteriors_3m.max(axis=1)

        # Predict states for 15m
        states_15m = self.model_15m.predict(features_15m)
        posteriors_15m = self.model_15m.predict_proba(features_15m)
        labels_15m = [self.state_labels_15m.get(s, "UNKNOWN") for s in states_15m]
        conf_15m = posteriors_15m.max(axis=1)

        # Build 15m lookup by timestamp
        mid_by_ts = {}
        for i, ts in enumerate(timestamps_15m):
            mid_by_ts[ts] = (labels_15m[i], conf_15m[i], posteriors_15m[i])

        # Fuse for each 3m bar
        fused_labels = []
        fused_probs = []

        for i, ts in enumerate(timestamps_3m):
            ts_floor = ts.floor("15min")

            label_3m = labels_3m[i]
            c_3m = conf_3m[i]
            probs_3m = posteriors_3m[i]

            if ts_floor in mid_by_ts:
                label_15m, c_15m, probs_15m = mid_by_ts[ts_floor]
            else:
                # No 15m data - use 3m only
                fused_labels.append(label_3m)
                fused_probs.append(probs_3m)
                continue

            # Weighted fusion
            if label_3m == label_15m:
                fused_labels.append(label_3m)
                fused_probs.append(probs_3m)
            else:
                # Conflict resolution: weighted confidence
                score_3m = c_3m * self.fast_weight
                score_15m = c_15m * self.mid_weight

                # Mid priority: 15m wins unless 3m is much stronger
                if score_3m > score_15m * 1.5:
                    fused_labels.append(label_3m)
                    fused_probs.append(probs_3m)
                else:
                    fused_labels.append(label_15m)
                    # Create probability array matching 3m shape
                    fused_probs.append(probs_3m)

        # Convert to numpy arrays (match SingleHMM format)
        fused_labels = np.array(fused_labels)
        fused_probs = np.array(fused_probs)
        # State sequence not meaningful for fusion, use 3m states
        return states_3m, fused_probs, fused_labels


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


def build_features(df_3m: pd.DataFrame, df_15m_native: pd.DataFrame = None) -> Tuple[np.ndarray, List[str], pd.DataFrame, pd.DataFrame, np.ndarray, List[str], pd.DataFrame]:
    """Build multi-timeframe features for HMM + FSM.

    Args:
        df_3m: 3-minute OHLCV data
        df_15m_native: 15-minute OHLCV data loaded directly from DB (critical for matching tuning).
                      If None, will resample from 3m (NOT recommended - causes regime mismatch).

    Returns:
        - HMM feature matrix (15m resolution)
        - Feature names (15m)
        - df_15m_clean (15m data for HMM training/prediction)
        - df_3m_full (3m data with 15m features forward-filled for backtest)
        - HMM feature matrix (3m resolution) for MultiHMM
        - Feature names (3m)
        - df_3m_hmm (3m data with HMM features)
    """
    # === 3m Features (for FSM) ===
    df_3m_feat = df_3m.copy()
    if "timestamp" in df_3m_feat.columns:
        df_3m_feat["timestamp"] = pd.to_datetime(df_3m_feat["timestamp"])
        df_3m_feat = df_3m_feat.set_index("timestamp")

    close_3m = df_3m_feat["close"].astype(float)
    high_3m = df_3m_feat["high"].astype(float)
    low_3m = df_3m_feat["low"].astype(float)

    # 3m returns
    df_3m_feat["ret_3m"] = close_3m.pct_change()

    # 3m EMA
    df_3m_feat["ema_fast_3m"] = close_3m.ewm(span=20, adjust=False).mean()
    df_3m_feat["ema_slow_3m"] = close_3m.ewm(span=50, adjust=False).mean()

    # 3m ATR - Use rolling mean to match tuning script
    tr_3m = pd.concat([
        high_3m - low_3m,
        (high_3m - close_3m.shift(1)).abs(),
        (low_3m - close_3m.shift(1)).abs(),
    ], axis=1).max(axis=1)
    df_3m_feat["atr_3m"] = tr_3m.rolling(14).mean()  # Changed from ewm to rolling

    # 3m RSI (14-period) - Added for exit conditions
    delta = close_3m.diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, np.nan)
    df_3m_feat["rsi_3m"] = 100 - (100 / (1 + rs))
    df_3m_feat["rsi_3m"] = df_3m_feat["rsi_3m"].fillna(50)  # Default to neutral

    # Add close_2, close_5 for XGB ret_2, ret_5 calculation
    df_3m_feat["close_2"] = close_3m.shift(2)
    df_3m_feat["close_5"] = close_3m.shift(5)

    # Add volume_ma for XGB volume_ratio calculation
    if "volume" in df_3m_feat.columns:
        df_3m_feat["volume_ma_20"] = df_3m_feat["volume"].rolling(20).mean()

    # Keep 3m index for later joining
    df_3m_indexed = df_3m_feat.copy()

    # === 15m Features (for HMM) ===
    # CRITICAL: Use native 15m data from DB to match tuning script exactly
    if df_15m_native is not None:
        logger.info("Using native 15m data from DB (matches tuning)")
        df_15m = df_15m_native.copy()
        if "timestamp" in df_15m.columns:
            df_15m["timestamp"] = pd.to_datetime(df_15m["timestamp"])
    else:
        logger.warning("No native 15m data - resampling from 3m (may cause regime mismatch!)")
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

    # EMA (15m)
    ema_20 = close_15m.ewm(span=20, adjust=False).mean()
    ema_50 = close_15m.ewm(span=50, adjust=False).mean()
    df_15m["ema_slope_15m"] = ema_20.pct_change(5)  # Fixed: 5-bar pct change (consistent with tuning)
    df_15m["ema_20_15m"] = ema_20
    df_15m["ema_50_15m"] = ema_50

    # ATR (15m) - Use rolling mean to match tuning script
    tr_15m = pd.concat([
        high_15m - low_15m,
        (high_15m - close_15m.shift(1)).abs(),
        (low_15m - close_15m.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr_15m = tr_15m.rolling(14).mean()  # Changed from ewm to rolling (match tuning)
    df_15m["atr_pct_15m"] = atr_15m / close_15m * 100
    df_15m["atr_15m"] = atr_15m

    df_15m["ewm_std_ret_15m"] = df_15m["ret_15m"].ewm(span=20, adjust=False).std()

    # Bollinger Band Width (15m)
    sma_15m = close_15m.rolling(20).mean()
    std_15m = close_15m.rolling(20).std()
    df_15m["bb_width_15m"] = (2 * std_15m / sma_15m) * 100

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

    # Zone calculation (simple support/resistance)
    df_15m["rolling_high_20"] = high_15m.rolling(20).max()
    df_15m["rolling_low_20"] = low_15m.rolling(20).min()

    # 1h features
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

    # Merge 15m with 1h
    df_15m = df_15m.set_index("timestamp")
    df_1h = df_1h.set_index("timestamp")

    features_1h = df_1h[["ret_1h", "ewm_ret_1h", "price_z_from_ema_1h", "atr_pct_1h", "ewm_std_ret_1h", "ema_20_1h"]]
    df_15m_result = df_15m.join(features_1h, how="left")
    cols_to_ffill = ["ret_1h", "ewm_ret_1h", "price_z_from_ema_1h", "atr_pct_1h", "ewm_std_ret_1h", "ema_20_1h"]
    df_15m_result[cols_to_ffill] = df_15m_result[cols_to_ffill].ffill()

    # === Simple HMM Features (6 features - same as tuning script) ===
    # Add box/breakout features for HMM (matching tuning script)
    df_15m_result["vol_15m"] = df_15m_result["ret_15m"].rolling(20).std()
    df_15m_result["box_high"] = df_15m_result["high"].rolling(32).max()
    df_15m_result["box_low"] = df_15m_result["low"].rolling(32).min()
    df_15m_result["box_range"] = (df_15m_result["box_high"] - df_15m_result["box_low"]) / df_15m_result["atr_15m"]
    df_15m_result["B_up"] = (df_15m_result["close"] - df_15m_result["box_high"].shift(1)) / df_15m_result["atr_15m"]
    df_15m_result["B_dn"] = (df_15m_result["box_low"].shift(1) - df_15m_result["close"]) / df_15m_result["atr_15m"]

    # Use same 6 features as tuning script
    hmm_features = [
        "ret_15m", "vol_15m", "ema_slope_15m",
        "box_range", "B_up", "B_dn",
    ]

    df_15m_clean = df_15m_result.dropna(subset=hmm_features)
    feature_matrix = df_15m_clean[hmm_features].values

    # === 3m Data with 15m Features Forward-Filled (for backtest) ===
    # Forward fill 15m features to 3m timestamps - include all needed for RANGE validation
    features_15m_for_3m = df_15m_result[[
        "ema_slope_15m", "ewm_ret_15m", "ewm_std_ret_15m", "atr_15m", "atr_pct_15m",
        "ema_20_15m", "ema_50_15m", "rolling_high_20", "rolling_low_20",
        "bb_width_15m", "range_comp_15m"
    ]]
    df_3m_full = df_3m_indexed.join(features_15m_for_3m, how="left")

    # Forward fill 15m features to cover all 3m bars
    cols_15m_ffill = ["ema_slope_15m", "ewm_ret_15m", "ewm_std_ret_15m", "atr_15m", "atr_pct_15m",
                      "ema_20_15m", "ema_50_15m", "rolling_high_20", "rolling_low_20",
                      "bb_width_15m", "range_comp_15m"]
    df_3m_full[cols_15m_ffill] = df_3m_full[cols_15m_ffill].ffill()

    # Drop rows without required features
    df_3m_full = df_3m_full.dropna(subset=["ret_3m", "ema_fast_3m", "ema_slope_15m"])
    df_3m_full = df_3m_full.reset_index()

    # === 3m HMM Features (for MultiHMM) ===
    df_3m_hmm = df_3m_indexed.copy()
    df_3m_hmm["vol_3m"] = df_3m_hmm["ret_3m"].rolling(20).std()
    ema_20_3m = df_3m_hmm["close"].ewm(span=20, adjust=False).mean()
    df_3m_hmm["ema_slope_3m"] = ema_20_3m.pct_change(5)
    df_3m_hmm["box_high_3m"] = df_3m_hmm["high"].rolling(32).max()
    df_3m_hmm["box_low_3m"] = df_3m_hmm["low"].rolling(32).min()
    df_3m_hmm["box_range_3m"] = (df_3m_hmm["box_high_3m"] - df_3m_hmm["box_low_3m"]) / (df_3m_hmm["atr_3m"] + 1e-8)
    df_3m_hmm["B_up_3m"] = (df_3m_hmm["close"] - df_3m_hmm["box_high_3m"].shift(1)) / (df_3m_hmm["atr_3m"] + 1e-8)
    df_3m_hmm["B_dn_3m"] = (df_3m_hmm["box_low_3m"].shift(1) - df_3m_hmm["close"]) / (df_3m_hmm["atr_3m"] + 1e-8)

    hmm_features_3m = ["ret_3m", "vol_3m", "ema_slope_3m", "box_range_3m", "B_up_3m", "B_dn_3m"]
    df_3m_hmm_clean = df_3m_hmm.dropna(subset=hmm_features_3m)
    feature_matrix_3m = df_3m_hmm_clean[hmm_features_3m].values
    df_3m_hmm_clean = df_3m_hmm_clean.reset_index()

    return feature_matrix, hmm_features, df_15m_clean.reset_index(), df_3m_full, feature_matrix_3m, hmm_features_3m, df_3m_hmm_clean


def run_fsm_backtest(
    df_3m: pd.DataFrame,
    df_15m: pd.DataFrame,
    hmm_model,  # SingleHMM or MultiHMM
    features: np.ndarray,
    initial_capital: float = 10000.0,
    fee_rate: float = 0.001,
    # Additional params for MultiHMM
    features_3m: np.ndarray = None,
    df_3m_hmm: pd.DataFrame = None,
    # 1m data for precise execution
    df_1m: pd.DataFrame = None,
) -> Dict:
    """Run backtest with full FSM + DecisionEngine pipeline at 3m resolution.

    Uses 1m data for precise profit tracking if available.

    Args:
        df_3m: 3m data with 15m features forward-filled
        df_15m: 15m data (used for ConfirmLayer)
        hmm_model: Trained HMM model (SingleHMM or MultiHMM)
        features: HMM feature matrix (15m resolution)
        features_3m: HMM feature matrix (3m resolution) for MultiHMM
        df_3m_hmm: 3m data with HMM features for MultiHMM
        df_1m: 1m data indexed by timestamp for precise profit tracking
    """

    # Initialize components
    confirm_layer = RegimeConfirmLayer(ConfirmConfig(
        HIGH_VOL_LAMBDA_ON=5.5,      # 상위 3.6%에서만 트리거 (was 2.0)
        HIGH_VOL_LAMBDA_OFF=3.5,     # 적절한 퇴장 조건
        HIGH_VOL_COOLDOWN_BARS_15M=4,
    ))
    fsm = TradingFSM()
    decision_engine = DecisionEngine()

    # Get HMM predictions
    is_multi_hmm = isinstance(hmm_model, MultiHMM)

    if is_multi_hmm and features_3m is not None and df_3m_hmm is not None:
        # MultiHMM: Use fused 3m+15m predictions
        timestamps_3m = pd.to_datetime(df_3m_hmm["timestamp"])
        timestamps_15m = pd.to_datetime(df_15m["timestamp"])
        state_seq, state_probs, labels = hmm_model.predict_sequence(
            features_3m, features, timestamps_3m, timestamps_15m
        )
        # Labels are already at 3m resolution for MultiHMM
        df_3m_bt = df_3m.copy()
        # Match timestamps between df_3m and df_3m_hmm
        df_3m_bt = df_3m_bt.set_index("timestamp")
        df_3m_hmm_indexed = df_3m_hmm.set_index("timestamp")
        df_3m_hmm_indexed["regime_raw"] = labels
        df_3m_hmm_indexed["confidence"] = [probs.max() for probs in state_probs]
        df_3m_bt = df_3m_bt.join(df_3m_hmm_indexed[["regime_raw", "confidence"]], how="left")
        df_3m_bt[["regime_raw", "confidence"]] = df_3m_bt[["regime_raw", "confidence"]].ffill()
        df_3m_bt = df_3m_bt.dropna(subset=["regime_raw"])
        df_3m_bt = df_3m_bt.reset_index()
    else:
        # SingleHMM: Get HMM predictions at 15m resolution
        state_seq, state_probs, labels = hmm_model.predict_sequence(features)

        # Map 15m regime predictions to 3m bars
        df_15m_bt = df_15m.copy()
        df_15m_bt["regime_raw"] = labels
        df_15m_bt["confidence"] = [probs.max() for probs in state_probs]
        df_15m_bt = df_15m_bt.set_index("timestamp")

        # Join 15m regime to 3m data (forward fill)
        df_3m_bt = df_3m.copy()
        df_3m_bt = df_3m_bt.set_index("timestamp")
        df_3m_bt = df_3m_bt.join(df_15m_bt[["regime_raw", "confidence"]], how="left")
        df_3m_bt[["regime_raw", "confidence"]] = df_3m_bt[["regime_raw", "confidence"]].ffill()
        df_3m_bt = df_3m_bt.dropna(subset=["regime_raw"])
        df_3m_bt = df_3m_bt.reset_index()

    # Create 15m history lookup for ConfirmLayer
    df_15m_indexed = df_15m.copy()
    if "timestamp" in df_15m_indexed.columns:
        df_15m_indexed["timestamp"] = pd.to_datetime(df_15m_indexed["timestamp"])

    # State tracking
    confirm_state = None
    fsm_state = None
    engine_state = None

    # Backtest state
    equity = initial_capital
    position = PositionState(side="FLAT")
    trades = []
    equity_curve = [equity]
    signals_log = []
    entry_regime = None  # Track entry regime for accurate reporting
    entry_signal = None

    # Track max unrealized profit for profit-to-loss analysis
    max_unrealized_pnl = 0.0
    max_unrealized_pnl_pct = 0.0
    entry_idx = None

    # Track last 15m bar index for confirm layer
    last_15m_idx = 0

    for idx in range(250, len(df_3m_bt)):  # 250 bars warmup for 3m
        bar = df_3m_bt.iloc[idx]
        ts = bar["timestamp"]

        # Find corresponding 15m bar for ConfirmLayer
        matching_15m = df_15m_indexed[df_15m_indexed["timestamp"] <= ts]
        if len(matching_15m) == 0:
            continue
        current_15m_idx = len(matching_15m) - 1
        bar_15m = matching_15m.iloc[-1]
        hist_15m = matching_15m.iloc[max(0, current_15m_idx-96):current_15m_idx+1]

        # Build contexts
        regime_raw = bar["regime_raw"]
        ret_3m = bar.get("ret_3m", 0)

        # Step 1: ConfirmLayer (runs at 15m conceptually, but we update per 3m bar)
        # Include ret_3m for faster spike detection
        row_15m_dict = bar_15m.to_dict()
        row_15m_dict["ret_3m"] = ret_3m  # Add 3m return for spike detection

        confirm_result, confirm_state = confirm_layer.confirm(
            regime_raw=regime_raw,
            row_15m=row_15m_dict,
            hist_15m=hist_15m,
            state=confirm_state,
        )

        confirm_ctx = ConfirmContext(
            regime_raw=regime_raw,
            regime_confirmed=confirm_result.confirmed_regime,
            confirm_reason=confirm_result.reason,
            confirm_metrics=confirm_result.metrics,
        )

        # Build market context using actual 3m data
        price = float(bar["close"])
        atr_3m = bar.get("atr_3m", 0.01)
        atr_15m = bar.get("atr_15m", atr_3m)

        # Zone from 15m data
        support = bar.get("rolling_low_20", price - atr_15m * 2)
        resistance = bar.get("rolling_high_20", price + atr_15m * 2)
        dist_to_support = (price - support) / atr_3m if atr_3m > 0 else 999
        dist_to_resistance = (resistance - price) / atr_3m if atr_3m > 0 else 999

        # Get EMA values
        ema_fast = bar.get("ema_fast_3m", price)
        ema_slow = bar.get("ema_slow_3m", price)
        # ret_3m already defined above for spike detection

        # Get high/low for XGB range_pct calculation
        high_3m = float(bar.get("high", price))
        low_3m = float(bar.get("low", price))

        # Get close_2, close_5 for XGB ret_2, ret_5
        close_2 = bar.get("close_2", None)
        close_5 = bar.get("close_5", None)

        # Get volume data for XGB
        volume = bar.get("volume", 1.0)
        volume_ma = bar.get("volume_ma_20", volume)

        market_ctx = MarketContext(
            symbol="XRPUSDT",
            ts=int(ts.timestamp() * 1000) if hasattr(ts, "timestamp") else 0,
            price=price,
            row_3m={
                # Basic features
                "close": price,
                "high": high_3m,
                "low": low_3m,
                "atr_3m": atr_3m,
                "ema_fast_3m": ema_fast,
                "ema_slow_3m": ema_slow,
                "ret_3m": ret_3m,
                # XGB features (derived)
                "ret": ret_3m,
                "close_2": close_2,  # For XGB ret_2 calculation
                "close_5": close_5,  # For XGB ret_5 calculation
                "volatility": bar.get("volatility_3m", bar.get("ret_std_3m", 0.005)),
                "rsi_3m": bar.get("rsi_3m", 50),
                "ema_slope_15m": bar.get("ema_slope_15m", 0),
                "volume": volume,
                "volume_ma_20": volume_ma,
                # Calculated features for XGB
                "ema_diff": (ema_fast - ema_slow) / price if price > 0 else 0,
                "price_to_ema20": (price - ema_fast) / ema_fast if ema_fast > 0 else 0,
                "price_to_ema50": (price - ema_slow) / ema_slow if ema_slow > 0 else 0,
            },
            row_15m={
                "close": price,
                "high": high_3m,
                "low": low_3m,
                "ret_3m": ret_3m,  # For faster spike detection
                "ret_15m": bar.get("ret_15m", 0),  # For 15m spike detection
                "ema_slope_15m": bar.get("ema_slope_15m", 0),
                "ewm_ret_15m": bar.get("ewm_ret_15m", 0),
                "ewm_std_ret_15m": bar.get("ewm_std_ret_15m", 0.005),
                "atr_pct_15m": bar.get("atr_pct_15m", 1.0),
                "bb_width_15m": bar.get("bb_width_15m", 0.02),
                "range_comp_15m": bar.get("range_comp_15m", 0.5),
            },
            zone={
                "support": support,
                "resistance": resistance,
                "strength": 0.0001,
                "dist_to_support": dist_to_support,
                "dist_to_resistance": dist_to_resistance,
            },
        )

        # Update position state
        if position.side != "FLAT":
            position.bars_held_3m += 1
            if position.side == "LONG":
                position.unrealized_pnl = (price - position.entry_price) * position.size
            else:
                position.unrealized_pnl = (position.entry_price - price) * position.size

            # Track maximum unrealized profit using 1m data if available
            if df_1m is not None and len(df_1m) > 0:
                # Get 1m bars within this 3m period
                ts_start = ts - pd.Timedelta(minutes=3)
                try:
                    bars_1m = df_1m.loc[ts_start:ts]
                    if len(bars_1m) > 0:
                        if position.side == "LONG":
                            # For LONG, best price is highest high
                            best_price_1m = bars_1m["high"].max()
                            best_pnl = (best_price_1m - position.entry_price) * position.size
                            best_pnl_pct = (best_price_1m - position.entry_price) / position.entry_price * 100
                        else:
                            # For SHORT, best price is lowest low
                            best_price_1m = bars_1m["low"].min()
                            best_pnl = (position.entry_price - best_price_1m) * position.size
                            best_pnl_pct = (position.entry_price - best_price_1m) / position.entry_price * 100

                        if best_pnl > max_unrealized_pnl:
                            max_unrealized_pnl = best_pnl
                            max_unrealized_pnl_pct = best_pnl_pct
                except (KeyError, TypeError):
                    pass  # Fallback to 3m tracking below

            # Fallback: Track using 3m close price
            current_pnl_pct = position.unrealized_pnl / (position.entry_price * position.size) * 100 if position.size > 0 else 0
            if position.unrealized_pnl > max_unrealized_pnl:
                max_unrealized_pnl = position.unrealized_pnl
                max_unrealized_pnl_pct = current_pnl_pct

        # === Break-even Stop (손실 방지) ===
        # 0.08% 수익 도달 시 활성화, 손익분기+0.02%에서 청산
        breakeven_stop_triggered = False
        breakeven_stop_exit_price = None
        BREAKEVEN_ACTIVATION_PCT = 0.08   # Activate when profit reaches 0.08% (raised from 0.05%)
        BREAKEVEN_BUFFER_PCT = 0.02       # Exit at break-even + 0.02% (covers fees)

        if position.side != "FLAT" and df_1m is not None and len(df_1m) > 0:
            # Check if break-even should be activated (profit reached 0.05%)
            if max_unrealized_pnl_pct >= BREAKEVEN_ACTIVATION_PCT and max_unrealized_pnl_pct < TRAILING_STOP_ACTIVATION_PCT:
                # Break-even is active, trailing stop not yet - check for break-even exit
                breakeven_pct = BREAKEVEN_BUFFER_PCT
                ts_start = ts - pd.Timedelta(minutes=3)
                try:
                    bars_1m = df_1m.loc[ts_start:ts]
                    if len(bars_1m) > 0:
                        for _, bar_1m in bars_1m.iterrows():
                            if position.side == "LONG":
                                bar_low = bar_1m["low"]
                                breakeven_price = position.entry_price * (1 + breakeven_pct / 100)
                                if bar_low <= breakeven_price:
                                    breakeven_stop_triggered = True
                                    breakeven_stop_exit_price = breakeven_price
                                    break
                            else:  # SHORT
                                bar_high = bar_1m["high"]
                                breakeven_price = position.entry_price * (1 - breakeven_pct / 100)
                                if bar_high >= breakeven_price:
                                    breakeven_stop_triggered = True
                                    breakeven_stop_exit_price = breakeven_price
                                    break
                except (KeyError, TypeError):
                    pass

        # === 1m-level Trailing Stop Check (higher precision than 3m FSM) ===
        # Fee consideration: ~0.2% round-trip fees, need profit > 0.2% to be worthwhile
        # Activation: 0.10% profit (ensures meaningful profit to protect)
        # Preserve: 80% in low volatility, 60% otherwise (dynamic based on market conditions)
        trailing_stop_triggered = False
        trailing_stop_exit_price = None
        TRAILING_STOP_ACTIVATION_PCT = 0.10   # Activate when profit reaches 0.10%
        TRAILING_STOP_MIN_PRESERVE_PCT = 0.05 # Minimum 0.05% profit to preserve (must be < activation)

        # Dynamic preserve percentage based on volatility
        LOW_VOLATILITY_THRESHOLD = 0.30  # 0.30% volatility threshold
        current_volatility = bar.get("volatility_3m", bar.get("ret_std_3m", 0.005)) * 100  # Convert to %
        if current_volatility < LOW_VOLATILITY_THRESHOLD:
            TRAILING_STOP_PRESERVE_PCT = 0.80     # Low volatility: preserve 80% of max profit
        else:
            TRAILING_STOP_PRESERVE_PCT = 0.60     # Normal: preserve 60% of max profit

        if position.side != "FLAT" and df_1m is not None and len(df_1m) > 0 and not breakeven_stop_triggered:
            if max_unrealized_pnl_pct >= TRAILING_STOP_ACTIVATION_PCT:
                # Calculate preserve level: max of (60% of peak, minimum 0.05%)
                preserve_pct = max(
                    max_unrealized_pnl_pct * TRAILING_STOP_PRESERVE_PCT,
                    TRAILING_STOP_MIN_PRESERVE_PCT
                )

                # Trailing stop is activated - check 1m bars for exit
                ts_start = ts - pd.Timedelta(minutes=3)
                try:
                    bars_1m = df_1m.loc[ts_start:ts]
                    if len(bars_1m) > 0:
                        # Check each 1m bar for trailing stop trigger
                        for _, bar_1m in bars_1m.iterrows():
                            if position.side == "LONG":
                                # For LONG, check if low drops below preserve level
                                bar_low = bar_1m["low"]
                                preserve_price = position.entry_price * (1 + preserve_pct / 100)
                                if bar_low <= preserve_price:
                                    trailing_stop_triggered = True
                                    trailing_stop_exit_price = max(bar_low, preserve_price)
                                    break
                            else:  # SHORT
                                # For SHORT, check if high rises above preserve level
                                bar_high = bar_1m["high"]
                                preserve_price = position.entry_price * (1 - preserve_pct / 100)
                                if bar_high >= preserve_price:
                                    trailing_stop_triggered = True
                                    trailing_stop_exit_price = min(bar_high, preserve_price)
                                    break
                except (KeyError, TypeError):
                    pass

        # Step 2: FSM
        candidate_signal, fsm_state = fsm.step(
            ctx=market_ctx,
            confirm=confirm_ctx,
            pos=position,
            fsm_state=fsm_state,
        )

        # Step 3: DecisionEngine
        decision, engine_state = decision_engine.decide(
            ctx=market_ctx,
            confirm=confirm_ctx,
            pos=position,
            cand=candidate_signal,
            engine_state=engine_state,
        )

        # Log signal
        signals_log.append({
            "idx": idx,
            "timestamp": bar["timestamp"],
            "price": price,
            "regime_raw": regime_raw,
            "regime_confirmed": confirm_result.confirmed_regime,
            "confirm_reason": confirm_result.reason,
            "signal": candidate_signal.signal,
            "signal_reason": candidate_signal.reason,
            "signal_score": candidate_signal.score,
            "action": decision.action,
            "decision_reason": decision.reason,
            "decision_size": decision.size,
        })

        # Step 4: Execute action
        # Priority 1: Break-even stop (손실 방지)
        if breakeven_stop_triggered and position.side != "FLAT":
            exit_price = breakeven_stop_exit_price if breakeven_stop_exit_price else price

            if position.side == "LONG":
                pnl = (exit_price - position.entry_price) * position.size
            else:
                pnl = (position.entry_price - exit_price) * position.size

            fee = exit_price * position.size * fee_rate
            pnl -= fee
            equity += pnl

            final_pnl_pct = pnl / (position.entry_price * position.size) * 100 if position.size > 0 else 0
            profit_given_back = max_unrealized_pnl_pct - final_pnl_pct if max_unrealized_pnl_pct > 0 else 0

            trades.append({
                "entry_idx": entry_idx,
                "exit_idx": idx,
                "side": position.side,
                "entry_price": position.entry_price,
                "exit_price": exit_price,
                "size": position.size,
                "pnl": pnl,
                "pnl_pct": final_pnl_pct,
                "max_profit_pct": max_unrealized_pnl_pct,
                "max_profit_usd": max_unrealized_pnl,
                "profit_given_back": profit_given_back,
                "bars_held": position.bars_held_3m,
                "exit_reason": "BREAKEVEN_STOP_1M",
                "regime": entry_regime,
                "signal": entry_signal,
            })

            position = PositionState(side="FLAT", entry_price=0, size=0, entry_ts=0, bars_held_3m=0, unrealized_pnl=0)
            max_unrealized_pnl = 0.0
            max_unrealized_pnl_pct = 0.0
            entry_regime = None
            entry_signal = None
            entry_idx = None

        # Priority 2: Trailing stop (수익 보존)
        elif trailing_stop_triggered and position.side != "FLAT":
            # Use trailing stop exit price (from 1m data)
            exit_price = trailing_stop_exit_price if trailing_stop_exit_price else price

            if position.side == "LONG":
                pnl = (exit_price - position.entry_price) * position.size
            else:
                pnl = (position.entry_price - exit_price) * position.size

            fee = exit_price * position.size * fee_rate
            pnl -= fee  # Deduct exit fee from PnL (consistent with regular exits)
            equity += pnl

            final_pnl_pct = pnl / (position.entry_price * position.size) * 100 if position.size > 0 else 0
            profit_given_back = max_unrealized_pnl_pct - final_pnl_pct if max_unrealized_pnl_pct > 0 else 0

            trades.append({
                "entry_idx": entry_idx,
                "exit_idx": idx,
                "side": position.side,
                "entry_price": position.entry_price,
                "exit_price": exit_price,
                "size": position.size,
                "pnl": pnl,
                "pnl_pct": final_pnl_pct,
                "max_profit_pct": max_unrealized_pnl_pct,
                "max_profit_usd": max_unrealized_pnl,
                "profit_given_back": profit_given_back,
                "bars_held": position.bars_held_3m,
                "exit_reason": "TRAILING_STOP_1M",
                "regime": entry_regime,
                "signal": entry_signal,
            })

            position = PositionState(side="FLAT", entry_price=0, size=0, entry_ts=0, bars_held_3m=0, unrealized_pnl=0)
            max_unrealized_pnl = 0.0
            max_unrealized_pnl_pct = 0.0
            entry_regime = None
            entry_signal = None
            entry_idx = None

        elif decision.action == "OPEN_LONG" and position.side == "FLAT":
            # 1m Entry Analysis: Spike detection + optimal entry
            entry_price = price
            skip_entry = False
            entry_improvement = 0.0

            # Low volatility filter: Skip trades when volatility < 0.30%
            LOW_VOL_ENTRY_THRESHOLD = 0.30  # 0.30%
            current_vol_pct = bar.get("volatility_3m", bar.get("ret_std_3m", 0.005)) * 100
            if current_vol_pct < LOW_VOL_ENTRY_THRESHOLD:
                skip_entry = True
                signals_log[-1]["action"] = "SKIP_LOW_VOLATILITY"

            if df_1m is not None and len(df_1m) > 0 and not skip_entry:
                ts_start = ts - pd.Timedelta(minutes=3)
                try:
                    bars_1m = df_1m.loc[ts_start:ts]
                    if len(bars_1m) > 0:
                        # Anti-chasing: Skip if large UP spike (>1%) before LONG
                        max_ret_1m = ((bars_1m["high"] - bars_1m["open"]) / bars_1m["open"]).max()
                        if max_ret_1m > 0.01:  # 1% spike up
                            skip_entry = True
                            signals_log[-1]["action"] = "SKIP_CHASING_UP"

                        # Optimal entry: Use pullback low instead of 3m close
                        if not skip_entry and len(bars_1m) >= 2:
                            lows = bars_1m["low"].values
                            min_idx = np.argmin(lows)
                            if min_idx < len(lows) - 1:  # Pullback pattern (not last bar)
                                entry_price = lows[min_idx]
                                entry_improvement = (price - entry_price) / price * 100
                except (KeyError, TypeError):
                    pass

            if not skip_entry:
                size = decision.size / entry_price
                fee = entry_price * size * fee_rate
                equity -= fee

                position = PositionState(
                    side="LONG",
                    entry_price=entry_price,
                    size=size,
                    entry_ts=market_ctx.ts,
                    bars_held_3m=0,
                    unrealized_pnl=0,
                )
                entry_regime = confirm_result.confirmed_regime
                entry_signal = candidate_signal.signal
                entry_idx = idx
                max_unrealized_pnl = 0.0
                max_unrealized_pnl_pct = 0.0

        elif decision.action == "OPEN_SHORT" and position.side == "FLAT":
            # 1m Entry Analysis: Spike detection + optimal entry
            entry_price = price
            skip_entry = False
            entry_improvement = 0.0

            # Low volatility filter: Skip trades when volatility < 0.30%
            LOW_VOL_ENTRY_THRESHOLD = 0.30  # 0.30%
            current_vol_pct = bar.get("volatility_3m", bar.get("ret_std_3m", 0.005)) * 100
            if current_vol_pct < LOW_VOL_ENTRY_THRESHOLD:
                skip_entry = True
                signals_log[-1]["action"] = "SKIP_LOW_VOLATILITY"

            if df_1m is not None and len(df_1m) > 0 and not skip_entry:
                ts_start = ts - pd.Timedelta(minutes=3)
                try:
                    bars_1m = df_1m.loc[ts_start:ts]
                    if len(bars_1m) > 0:
                        # Anti-chasing: Skip if large DOWN spike (>1%) before SHORT
                        min_ret_1m = ((bars_1m["low"] - bars_1m["open"]) / bars_1m["open"]).min()
                        if min_ret_1m < -0.01:  # 1% spike down
                            skip_entry = True
                            signals_log[-1]["action"] = "SKIP_CHASING_DOWN"

                        # Optimal entry: Use bounce high instead of 3m close
                        if not skip_entry and len(bars_1m) >= 2:
                            highs = bars_1m["high"].values
                            max_idx = np.argmax(highs)
                            if max_idx < len(highs) - 1:  # Bounce pattern (not last bar)
                                entry_price = highs[max_idx]
                                entry_improvement = (entry_price - price) / price * 100
                except (KeyError, TypeError):
                    pass

            if not skip_entry:
                size = decision.size / entry_price
                fee = entry_price * size * fee_rate
                equity -= fee
                entry_regime = confirm_result.confirmed_regime
                entry_signal = candidate_signal.signal
                entry_idx = idx
                max_unrealized_pnl = 0.0
                max_unrealized_pnl_pct = 0.0

                position = PositionState(
                    side="SHORT",
                    entry_price=entry_price,
                    size=size,
                    entry_ts=market_ctx.ts,
                    bars_held_3m=0,
                    unrealized_pnl=0,
                )

        elif decision.action == "CLOSE" and position.side != "FLAT":
            exit_price = price

            if position.side == "LONG":
                pnl = (exit_price - position.entry_price) * position.size
            else:
                pnl = (position.entry_price - exit_price) * position.size

            fee = exit_price * position.size * fee_rate
            pnl -= fee
            equity += pnl

            final_pnl_pct = pnl / (position.entry_price * position.size) * 100 if position.size > 0 else 0
            profit_given_back = max_unrealized_pnl_pct - final_pnl_pct if max_unrealized_pnl_pct > 0 else 0

            trades.append({
                "entry_idx": entry_idx,
                "exit_idx": idx,
                "side": position.side,
                "entry_price": position.entry_price,
                "exit_price": exit_price,
                "size": position.size,
                "pnl": pnl,
                "pnl_pct": final_pnl_pct,
                "max_profit_pct": max_unrealized_pnl_pct,  # NEW: max unrealized profit
                "max_profit_usd": max_unrealized_pnl,      # NEW: max unrealized profit in USD
                "profit_given_back": profit_given_back,    # NEW: how much profit was lost
                "bars_held": position.bars_held_3m,
                "exit_reason": decision.reason,
                "regime": entry_regime,  # Use entry regime, not exit regime
                "signal": entry_signal,
            })

            position = PositionState(side="FLAT")
            max_unrealized_pnl = 0.0
            max_unrealized_pnl_pct = 0.0

        equity_curve.append(equity)

    # Close any remaining position
    if position.side != "FLAT":
        bar = df_3m_bt.iloc[-1]
        exit_price = float(bar["close"])

        if position.side == "LONG":
            pnl = (exit_price - position.entry_price) * position.size
        else:
            pnl = (position.entry_price - exit_price) * position.size

        fee = exit_price * position.size * fee_rate
        pnl -= fee
        equity += pnl

        trades.append({
            "entry_idx": None,
            "exit_idx": len(df_3m_bt) - 1,
            "side": position.side,
            "entry_price": position.entry_price,
            "exit_price": exit_price,
            "size": position.size,
            "pnl": pnl,
            "pnl_pct": pnl / (position.entry_price * position.size) * 100 if position.size > 0 else 0,
            "bars_held": position.bars_held_3m,
            "exit_reason": "EOD",
            "regime": "EOD",
        })

    # Calculate metrics
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    signals_df = pd.DataFrame(signals_log) if signals_log else pd.DataFrame()

    results = {
        "initial_capital": initial_capital,
        "final_equity": equity,
        "total_return_pct": (equity - initial_capital) / initial_capital * 100,
        "n_trades": len(trades),
        "trades_df": trades_df,
        "signals_df": signals_df,
        "equity_curve": equity_curve,
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

        # Breakdown by regime
        regime_stats = {}
        for regime in trades_df["regime"].unique():
            regime_trades = trades_df[trades_df["regime"] == regime]
            if len(regime_trades) > 0:
                regime_wins = regime_trades[regime_trades["pnl"] > 0]
                regime_stats[regime] = {
                    "n_trades": len(regime_trades),
                    "win_rate": len(regime_wins) / len(regime_trades) * 100,
                    "total_pnl": regime_trades["pnl"].sum(),
                }
        results["regime_stats"] = regime_stats

        # Breakdown by exit reason
        exit_stats = {}
        for reason in trades_df["exit_reason"].unique():
            reason_trades = trades_df[trades_df["exit_reason"] == reason]
            if len(reason_trades) > 0:
                reason_wins = reason_trades[reason_trades["pnl"] > 0]
                exit_stats[reason] = {
                    "n_trades": len(reason_trades),
                    "win_rate": len(reason_wins) / len(reason_trades) * 100,
                    "total_pnl": reason_trades["pnl"].sum(),
                }
        results["exit_stats"] = exit_stats
    else:
        results["win_rate"] = 0
        results["profit_factor"] = 0
        results["max_drawdown_pct"] = 0
        results["expectancy"] = 0

    # Signal statistics
    if len(signals_df) > 0:
        signal_counts = signals_df["signal"].value_counts().to_dict()
        action_counts = signals_df["action"].value_counts().to_dict()
        regime_counts = signals_df["regime_confirmed"].value_counts().to_dict()

        results["signal_counts"] = signal_counts
        results["action_counts"] = action_counts
        results["regime_counts"] = regime_counts

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
        # Remove timezone to ensure consistent tz-naive timestamps
        if df["timestamp"].dt.tz is not None:
            df["timestamp"] = df["timestamp"].dt.tz_localize(None)
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=str, default="2023-01-01")
    parser.add_argument("--end", type=str, default="2024-12-31")
    parser.add_argument("--symbol", type=str, default="XRPUSDT")
    parser.add_argument("--n_states", type=int, default=5)
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/fsm_backtest"))
    parser.add_argument("--hmm-model", type=Path, default=None,
                       help="Path to pre-trained HMM model (pkl). If not provided, trains new model.")
    parser.add_argument("--hmm-type", type=str, default="multi", choices=["single", "multi"],
                       help="HMM type: 'single' (5-state 15m only) or 'multi' (3m+15m fusion)")
    parser.add_argument("--use-checkpoint", type=str, default=None,
                       help="Use pre-trained HMM checkpoint (run_id). Skips training and uses all data as test.")

    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("FSM + DecisionEngine BACKTEST")
    logger.info("=" * 70)
    logger.info("Pipeline: HMM -> ConfirmLayer -> FSM -> DecisionEngine -> Execution")

    # Load data
    logger.info("\nLoading data...")
    start_dt = datetime.fromisoformat(args.start)
    end_dt = datetime.fromisoformat(args.end)

    # Load 3m, 15m AND 1m data directly from DB
    df_3m = load_ohlcv_from_timescaledb(args.symbol, start_dt, end_dt, "3m")
    df_15m_native = load_ohlcv_from_timescaledb(args.symbol, start_dt, end_dt, "15m")
    df_1m = load_ohlcv_from_timescaledb(args.symbol, start_dt, end_dt, "1m")

    if df_3m.empty:
        logger.error("No 3m data!")
        sys.exit(1)
    if df_15m_native.empty:
        logger.warning("No native 15m data - will resample from 3m (may cause regime mismatch)")
        df_15m_native = None

    logger.info(f"Loaded {len(df_3m)} 3m bars")
    if df_15m_native is not None:
        logger.info(f"Loaded {len(df_15m_native)} 15m bars from DB (native)")
    if not df_1m.empty:
        logger.info(f"Loaded {len(df_1m)} 1m bars for precise execution")
        # Index 1m data by timestamp for fast lookup
        df_1m = df_1m.set_index("timestamp").sort_index()
        # Remove timezone to match 3m data (tz-naive)
        if df_1m.index.tz is not None:
            df_1m.index = df_1m.index.tz_localize(None)
    else:
        logger.warning("No 1m data - using 3m data for execution")
        df_1m = None
    logger.info(f"HMM Type: {args.hmm_type}")

    # Build features
    logger.info("Building features...")
    features, feature_names, df_15m, df_3m_full, features_3m, feature_names_3m, df_3m_hmm = build_features(df_3m, df_15m_native)
    logger.info(f"Feature matrix shape (15m): {features.shape}")
    logger.info(f"Feature matrix shape (3m): {features_3m.shape}")
    logger.info(f"3m data shape: {df_3m_full.shape}")

    # Split data (by 15m for HMM, corresponding 3m data split by timestamp)
    n_total_15m = len(features)
    split_ratio = float(os.environ.get("SPLIT_RATIO", "0.7"))  # Allow override via env
    n_train_15m = int(n_total_15m * split_ratio)

    train_features = features[:n_train_15m]
    test_features = features[n_train_15m:]

    train_df_15m = df_15m.iloc[:n_train_15m].reset_index(drop=True)
    test_df_15m = df_15m.iloc[n_train_15m:].reset_index(drop=True)

    # Split 3m data by timestamp
    split_ts = train_df_15m["timestamp"].iloc[-1]
    train_df_3m = df_3m_full[df_3m_full["timestamp"] <= split_ts].reset_index(drop=True)
    test_df_3m = df_3m_full[df_3m_full["timestamp"] > split_ts].reset_index(drop=True)

    # Split 3m HMM data
    train_df_3m_hmm = df_3m_hmm[df_3m_hmm["timestamp"] <= split_ts].reset_index(drop=True)
    test_df_3m_hmm = df_3m_hmm[df_3m_hmm["timestamp"] > split_ts].reset_index(drop=True)
    train_features_3m = train_df_3m_hmm[feature_names_3m].values
    test_features_3m = test_df_3m_hmm[feature_names_3m].values

    logger.info(f"Train: {len(train_features)} 15m bars, {len(train_df_3m)} 3m bars")
    logger.info(f"Test: {len(test_features)} 15m bars, {len(test_df_3m)} 3m bars")

    # Check if using checkpoint (test-only mode)
    use_checkpoint = args.use_checkpoint
    if use_checkpoint:
        logger.info(f"\n*** CHECKPOINT MODE: Using pre-trained HMM from {use_checkpoint} ***")
        logger.info("*** All data will be used as TEST (no training) ***")

        # Build features using checkpoint-compatible feature builders
        from xrp4.features.hmm_features import build_fast_hmm_features_v2, build_mid_hmm_features_v2

        # Fast HMM features (9 features) - from checkpoint
        _fast_feature_names = [
            "ret_3m", "ewm_ret_15m", "atr_pct_15m", "ewm_std_ret_15m",
            "bb_width_15m", "body_ratio", "upper_wick_ratio", "lower_wick_ratio", "range_pct",
        ]
        # Mid HMM features (11 features) - from checkpoint
        _mid_feature_names = [
            "ret_15m", "ewm_ret_15m", "ret_1h", "ewm_ret_1h", "atr_pct_15m",
            "ewm_std_ret_15m", "atr_pct_1h", "ewm_std_ret_1h", "bb_width_15m",
            "vol_z_15m", "price_z_from_ema_1h",
        ]

        logger.info("Building checkpoint-compatible features...")
        df_3m_ohlcv = df_3m[["timestamp", "open", "high", "low", "close", "volume"]].copy()
        df_15m_ohlcv = df_15m_native[["timestamp", "open", "high", "low", "close", "volume"]].copy()

        test_features_3m, fast_timestamps = build_fast_hmm_features_v2(df_3m_ohlcv, _fast_feature_names)
        test_features_15m, mid_timestamps = build_mid_hmm_features_v2(df_15m_ohlcv, _mid_feature_names)

        # Remove timezone info for compatibility
        if fast_timestamps.tz is not None:
            fast_timestamps = fast_timestamps.tz_localize(None)
        if mid_timestamps.tz is not None:
            mid_timestamps = mid_timestamps.tz_localize(None)

        logger.info(f"  Fast HMM features (3m): {test_features_3m.shape}")
        logger.info(f"  Mid HMM features (15m): {test_features_15m.shape}")

        # Create df_3m_hmm with timestamps
        test_df_3m_hmm = df_3m_ohlcv.copy()
        test_df_3m_hmm = test_df_3m_hmm.iloc[-len(fast_timestamps):].reset_index(drop=True)
        test_df_3m_hmm["timestamp"] = fast_timestamps.values

        # Use all data
        test_features = test_features_15m
        # Use df_15m (with features like ema_slope_15m) instead of df_15m_native
        test_df_15m = df_15m.iloc[-len(mid_timestamps):].reset_index(drop=True)
        test_df_3m = df_3m_full.iloc[-len(fast_timestamps):].reset_index(drop=True)

        # Ensure timestamps are tz-naive
        test_df_3m["timestamp"] = pd.to_datetime(test_df_3m["timestamp"]).dt.tz_localize(None)
        test_df_15m["timestamp"] = pd.to_datetime(test_df_15m["timestamp"]).dt.tz_localize(None)
        test_df_3m_hmm["timestamp"] = pd.to_datetime(test_df_3m_hmm["timestamp"]).dt.tz_localize(None)

        feature_names_3m = _fast_feature_names
        feature_names = _mid_feature_names

    # Train or load HMM based on type
    if args.hmm_type == "multi":
        if use_checkpoint:
            # Load from checkpoint
            from xrp4.regime.multi_hmm_manager import MultiHMMManager
            checkpoint_dir = Path("checkpoints/hmm")
            logger.info(f"\nLoading MultiHMM checkpoint: {use_checkpoint}")
            manager = MultiHMMManager(checkpoint_dir=checkpoint_dir)
            manager.load_checkpoints(use_checkpoint)

            # Create a wrapper that mimics MultiHMM interface but uses proper scaling
            class CheckpointMultiHMM(MultiHMM):
                def __init__(self, manager):
                    # Don't call parent __init__, just set attributes directly
                    self.manager = manager
                    self.fast_hmm = manager.fast_hmm  # Use full FastHMM object
                    self.mid_hmm = manager.mid_hmm    # Use full MidHMM object
                    self.model_3m = manager.fast_hmm.model
                    self.model_15m = manager.mid_hmm.model
                    self.state_labels_3m = {i: label.value for i, label in manager.fast_hmm.state_labels.items()}
                    self.state_labels_15m = {i: label.value for i, label in manager.mid_hmm.state_labels.items()}
                    self.fast_weight = 0.4
                    self.mid_weight = 0.6
                    self._is_trained = True

                def predict_sequence(
                    self,
                    features_3m: np.ndarray,
                    features_15m: np.ndarray,
                    timestamps_3m: pd.DatetimeIndex,
                    timestamps_15m: pd.DatetimeIndex,
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
                    """Predict fused regime labels using checkpoint models with proper scaling."""
                    # Use FastHMM and MidHMM predict methods which apply scaling internally
                    labels_3m = []
                    conf_3m = []
                    probs_3m = []

                    for i in range(len(features_3m)):
                        pred = self.fast_hmm.predict(features_3m[i])
                        labels_3m.append(pred.label.value)
                        conf_3m.append(pred.confidence)
                        probs_3m.append(pred.state_probs)

                    labels_15m = []
                    conf_15m = []
                    probs_15m = []

                    for i in range(len(features_15m)):
                        pred = self.mid_hmm.predict(features_15m[i])
                        labels_15m.append(pred.label.value)
                        conf_15m.append(pred.confidence)
                        probs_15m.append(pred.state_probs)

                    conf_3m = np.array(conf_3m)
                    conf_15m = np.array(conf_15m)

                    # Build 15m lookup by timestamp
                    mid_by_ts = {}
                    for i, ts in enumerate(timestamps_15m):
                        mid_by_ts[ts] = (labels_15m[i], conf_15m[i], probs_15m[i])

                    # Fuse for each 3m bar
                    fused_labels = []
                    fused_probs = []

                    for i, ts in enumerate(timestamps_3m):
                        ts_floor = ts.floor("15min")

                        label_3m = labels_3m[i]
                        c_3m = conf_3m[i]
                        p_3m = probs_3m[i]

                        if ts_floor in mid_by_ts:
                            label_15m, c_15m, p_15m = mid_by_ts[ts_floor]
                        else:
                            fused_labels.append(label_3m)
                            fused_probs.append(p_3m)
                            continue

                        # Weighted fusion
                        if label_3m == label_15m:
                            fused_labels.append(label_3m)
                            fused_probs.append(p_3m)
                        else:
                            score_3m = c_3m * self.fast_weight
                            score_15m = c_15m * self.mid_weight
                            if score_3m > score_15m * 1.5:
                                fused_labels.append(label_3m)
                                fused_probs.append(p_3m)
                            else:
                                fused_labels.append(label_15m)
                                fused_probs.append(p_3m)

                    fused_labels = np.array(fused_labels)
                    fused_probs = np.array(fused_probs)
                    states_3m = np.zeros(len(fused_labels), dtype=int)  # Placeholder

                    return states_3m, fused_probs, fused_labels

            hmm_model = CheckpointMultiHMM(manager)
            logger.info(f"State labels 3m: {hmm_model.state_labels_3m}")
            logger.info(f"State labels 15m: {hmm_model.state_labels_15m}")
        else:
            logger.info("\nTraining MultiHMM (3m + 15m fusion)...")
            hmm_model = MultiHMM(n_states_3m=4, n_states_15m=4, fast_weight=0.4, mid_weight=0.6)
            hmm_model.train(train_features_3m, feature_names_3m, train_features, feature_names)
            logger.info(f"State labels 3m: {hmm_model.state_labels_3m}")
            logger.info(f"State labels 15m: {hmm_model.state_labels_15m}")
    else:
        # Train or load SingleHMM
        if args.hmm_model and args.hmm_model.exists():
            logger.info(f"\nLoading pre-trained HMM from: {args.hmm_model}")
            hmm_model = SingleHMM.from_saved_model(args.hmm_model)
        else:
            if args.hmm_model:
                logger.warning(f"HMM model file not found: {args.hmm_model}, training new model...")
            logger.info("\nTraining SingleHMM...")
            hmm_model = SingleHMM(n_states=args.n_states)
            hmm_model.train(train_features, feature_names)
        logger.info(f"State labels: {hmm_model.state_labels}")

    # Run backtest on train (skip if using checkpoint)
    train_results = None
    if not use_checkpoint:
        logger.info("\n" + "=" * 70)
        logger.info("BACKTEST - TRAIN SET")
        logger.info("=" * 70)

        if args.hmm_type == "multi":
            train_results = run_fsm_backtest(
                train_df_3m, train_df_15m, hmm_model, train_features,
                features_3m=train_features_3m, df_3m_hmm=train_df_3m_hmm, df_1m=df_1m
            )
        else:
            train_results = run_fsm_backtest(train_df_3m, train_df_15m, hmm_model, train_features, df_1m=df_1m)

        logger.info(f"\nTrain Results:")
        logger.info(f"  Return: {train_results['total_return_pct']:.2f}%")
        logger.info(f"  Win Rate: {train_results['win_rate']:.1f}%")
        logger.info(f"  Profit Factor: {train_results['profit_factor']:.2f}")
        logger.info(f"  Max DD: {train_results['max_drawdown_pct']:.2f}%")
        logger.info(f"  N Trades: {train_results['n_trades']}")

        if train_results['n_trades'] > 0:
            logger.info(f"  Expectancy: {train_results.get('expectancy', 0):.4f}%")
            logger.info(f"  Avg Win: {train_results.get('avg_win_pct', 0):.3f}%")
            logger.info(f"  Avg Loss: {train_results.get('avg_loss_pct', 0):.3f}%")

    # Run backtest on test
    logger.info("\n" + "=" * 70)
    logger.info("BACKTEST - TEST SET (Out-of-Sample)")
    logger.info("=" * 70)

    if args.hmm_type == "multi":
        test_results = run_fsm_backtest(
            test_df_3m, test_df_15m, hmm_model, test_features,
            features_3m=test_features_3m, df_3m_hmm=test_df_3m_hmm, df_1m=df_1m
        )
    else:
        test_results = run_fsm_backtest(test_df_3m, test_df_15m, hmm_model, test_features, df_1m=df_1m)

    logger.info(f"\nTest Results:")
    logger.info(f"  Return: {test_results['total_return_pct']:.2f}%")
    logger.info(f"  Win Rate: {test_results['win_rate']:.1f}%")
    logger.info(f"  Profit Factor: {test_results['profit_factor']:.2f}")
    logger.info(f"  Max DD: {test_results['max_drawdown_pct']:.2f}%")
    logger.info(f"  N Trades: {test_results['n_trades']}")

    if test_results['n_trades'] > 0:
        logger.info(f"  Expectancy: {test_results.get('expectancy', 0):.4f}%")
        logger.info(f"  Avg Win: {test_results.get('avg_win_pct', 0):.3f}%")
        logger.info(f"  Avg Loss: {test_results.get('avg_loss_pct', 0):.3f}%")
        logger.info(f"  Avg Bars Held: {test_results.get('avg_bars_held', 0):.1f}")

    # Signal statistics
    if "signal_counts" in test_results:
        logger.info("\nSignal Distribution (Test):")
        for signal, count in sorted(test_results["signal_counts"].items(), key=lambda x: -x[1]):
            logger.info(f"  {signal}: {count}")

    if "action_counts" in test_results:
        logger.info("\nAction Distribution (Test):")
        for action, count in sorted(test_results["action_counts"].items(), key=lambda x: -x[1]):
            logger.info(f"  {action}: {count}")

    if "regime_counts" in test_results:
        logger.info("\nRegime Distribution (Test):")
        for regime, count in sorted(test_results["regime_counts"].items(), key=lambda x: -x[1]):
            logger.info(f"  {regime}: {count}")

    # Regime breakdown
    if "regime_stats" in test_results and test_results["regime_stats"]:
        logger.info("\nTrade Breakdown by Regime (Test):")
        for regime, stats in test_results["regime_stats"].items():
            logger.info(
                f"  {regime}: {stats['n_trades']} trades, "
                f"WR={stats['win_rate']:.1f}%, "
                f"PnL=${stats['total_pnl']:.2f}"
            )

    # Exit reason breakdown
    if "exit_stats" in test_results and test_results["exit_stats"]:
        logger.info("\nTrade Breakdown by Exit Reason (Test):")
        for reason, stats in test_results["exit_stats"].items():
            logger.info(
                f"  {reason}: {stats['n_trades']} trades, "
                f"WR={stats['win_rate']:.1f}%, "
                f"PnL=${stats['total_pnl']:.2f}"
            )

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)

    if train_results:
        logger.info(f"{'Metric':<20} {'Train':>12} {'Test':>12}")
        logger.info("-" * 46)
        logger.info(f"{'Return %':<20} {train_results['total_return_pct']:>12.2f} {test_results['total_return_pct']:>12.2f}")
        logger.info(f"{'N Trades':<20} {train_results['n_trades']:>12d} {test_results['n_trades']:>12d}")

        if train_results["n_trades"] > 0 and test_results["n_trades"] > 0:
            logger.info(f"{'Win Rate %':<20} {train_results['win_rate']:>12.1f} {test_results['win_rate']:>12.1f}")
            logger.info(f"{'Profit Factor':<20} {train_results['profit_factor']:>12.2f} {test_results['profit_factor']:>12.2f}")
            logger.info(f"{'Max DD %':<20} {train_results['max_drawdown_pct']:>12.2f} {test_results['max_drawdown_pct']:>12.2f}")
    else:
        # Checkpoint mode: test only
        logger.info(f"{'Metric':<20} {'Test':>12}")
        logger.info("-" * 34)
        logger.info(f"{'Return %':<20} {test_results['total_return_pct']:>12.2f}")
        logger.info(f"{'N Trades':<20} {test_results['n_trades']:>12d}")

        if test_results["n_trades"] > 0:
            logger.info(f"{'Win Rate %':<20} {test_results['win_rate']:>12.1f}")
            logger.info(f"{'Profit Factor':<20} {test_results['profit_factor']:>12.2f}")
            logger.info(f"{'Max DD %':<20} {test_results['max_drawdown_pct']:>12.2f}")
            logger.info(f"{'Expectancy %':<20} {test_results.get('expectancy', 0):>12.4f}")

    # Save results
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "test_results": {
            "return_pct": test_results["total_return_pct"],
            "n_trades": test_results["n_trades"],
            "win_rate": test_results.get("win_rate", 0),
            "profit_factor": test_results.get("profit_factor", 0),
            "max_drawdown_pct": test_results.get("max_drawdown_pct", 0),
            "expectancy": test_results.get("expectancy", 0),
        },
    }

    if train_results:
        summary["train_results"] = {
            "return_pct": train_results["total_return_pct"],
            "n_trades": train_results["n_trades"],
            "win_rate": train_results.get("win_rate", 0),
            "profit_factor": train_results.get("profit_factor", 0),
            "max_drawdown_pct": train_results.get("max_drawdown_pct", 0),
        }

    with open(args.output_dir / "fsm_backtest_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Save trades
    if test_results["n_trades"] > 0:
        test_results["trades_df"].to_csv(args.output_dir / "trades_test.csv", index=False)

    # Save signals
    if "signals_df" in test_results and len(test_results["signals_df"]) > 0:
        test_results["signals_df"].to_csv(args.output_dir / "signals_test.csv", index=False)

    logger.info(f"\nResults saved to {args.output_dir}")
    logger.info("\nDone!")


if __name__ == "__main__":
    main()
