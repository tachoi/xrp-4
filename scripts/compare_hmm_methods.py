#!/usr/bin/env python
"""Compare SingleHMM vs MultiHMM performance.

Tests:
1. SingleHMM (current): 5-state HMM on 3m data only
2. MultiHMM: 3m HMM + 15m HMM with weighted fusion

Usage:
    python scripts/compare_hmm_methods.py
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from hmmlearn import hmm
import psycopg2

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================================
# SingleHMM (Current Implementation)
# ============================================================================

class SingleHMM:
    """Single HMM model for regime detection (current approach)."""

    def __init__(self, n_states: int = 5, n_iter: int = 500, random_state: int = 42):
        self.n_states = n_states
        self.n_iter = n_iter
        self.random_state = random_state
        self.model: Optional[hmm.GaussianHMM] = None
        self.feature_names: List[str] = []
        self.state_labels: Dict[int, str] = {}
        self._is_trained = False

    def train(self, features: np.ndarray, feature_names: List[str]) -> float:
        """Train HMM without scaling."""
        if np.any(np.isnan(features)):
            features = features.copy()
            col_means = np.nanmean(features, axis=0)
            for i in range(features.shape[1]):
                mask = np.isnan(features[:, i])
                features[mask, i] = col_means[i]

        self.model = hmm.GaussianHMM(
            n_components=self.n_states,
            covariance_type="full",
            n_iter=self.n_iter,
            random_state=self.random_state,
            verbose=False,
        )
        self.model.fit(features)
        self.feature_names = feature_names
        log_likelihood = self.model.score(features)

        states = self.model.predict(features)
        self._map_states_to_labels(features, states)
        self._is_trained = True
        return log_likelihood

    def _map_states_to_labels(self, features: np.ndarray, states: np.ndarray) -> None:
        """Map HMM states to regime labels."""
        ret_idx, vol_idx = 0, 1
        for i, name in enumerate(self.feature_names):
            if name in ("ret", "ret_15m"):
                ret_idx = i
            elif name in ("vol", "vol_15m"):
                vol_idx = i

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

        remaining = [k for k in range(self.n_states) if k not in used]
        if len(remaining) >= 2:
            self.state_labels[remaining[0]] = "TRANSITION"
            self.state_labels[remaining[1]] = "RANGE"
        elif len(remaining) == 1:
            self.state_labels[remaining[0]] = "RANGE"

    def predict(self, features: np.ndarray) -> List[str]:
        """Predict regime labels for feature sequences."""
        if np.any(np.isnan(features)):
            features = features.copy()
            col_means = np.nanmean(features, axis=0)
            for i in range(features.shape[1]):
                mask = np.isnan(features[:, i])
                features[mask, i] = col_means[i]

        states = self.model.predict(features)
        return [self.state_labels.get(s, "UNKNOWN") for s in states]


# ============================================================================
# MultiHMM (New Implementation)
# ============================================================================

class MultiHMM:
    """Multi-timeframe HMM: 3m + 15m with weighted fusion."""

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
        self._is_trained = False

    def train(
        self,
        features_3m: np.ndarray,
        feature_names_3m: List[str],
        features_15m: np.ndarray,
        feature_names_15m: List[str],
    ) -> Dict[str, float]:
        """Train both HMM models."""
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
            if "ret" in name:
                ret_idx = i
            elif "vol" in name:
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

    def predict(
        self,
        features_3m: np.ndarray,
        features_15m: np.ndarray,
        timestamps_3m: pd.DatetimeIndex,
        timestamps_15m: pd.DatetimeIndex,
    ) -> List[str]:
        """Predict fused regime labels."""
        features_3m = self._handle_nan(features_3m)
        features_15m = self._handle_nan(features_15m)

        # Predict states
        states_3m = self.model_3m.predict(features_3m)
        posteriors_3m = self.model_3m.predict_proba(features_3m)
        labels_3m = [self.state_labels_3m.get(s, "UNKNOWN") for s in states_3m]
        conf_3m = posteriors_3m.max(axis=1)

        states_15m = self.model_15m.predict(features_15m)
        posteriors_15m = self.model_15m.predict_proba(features_15m)
        labels_15m = [self.state_labels_15m.get(s, "UNKNOWN") for s in states_15m]
        conf_15m = posteriors_15m.max(axis=1)

        # Build 15m lookup
        mid_by_ts = {}
        for i, ts in enumerate(timestamps_15m):
            mid_by_ts[ts] = (labels_15m[i], conf_15m[i])

        # Fuse for each 3m bar
        fused_labels = []
        for i, ts in enumerate(timestamps_3m):
            ts_floor = ts.floor("15min")

            label_3m = labels_3m[i]
            c_3m = conf_3m[i]

            if ts_floor in mid_by_ts:
                label_15m, c_15m = mid_by_ts[ts_floor]
            else:
                # No 15m data - use 3m only
                fused_labels.append(label_3m)
                continue

            # Weighted fusion
            if label_3m == label_15m:
                fused_labels.append(label_3m)
            else:
                # Conflict resolution: weighted confidence
                score_3m = c_3m * self.fast_weight
                score_15m = c_15m * self.mid_weight

                # Mid priority: 15m wins unless 3m is much stronger
                if score_3m > score_15m * 1.5:
                    fused_labels.append(label_3m)
                else:
                    fused_labels.append(label_15m)

        return fused_labels


# ============================================================================
# Data Loading
# ============================================================================

def load_data(start_date: str, end_date: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load 3m and 15m OHLCV data from TimescaleDB."""
    conn = psycopg2.connect(
        host="localhost",
        port=5432,
        database="xrp_timeseries",
        user="xrp_user",
        password="xrp_password_change_me",
    )

    # Load 3m data
    query_3m = f"""
    SELECT time as ts, open, high, low, close, volume
    FROM ohlcv
    WHERE symbol = 'XRPUSDT' AND timeframe = '3m'
    AND time >= '{start_date}' AND time < '{end_date}'
    ORDER BY time
    """
    df_3m = pd.read_sql(query_3m, conn, parse_dates=["ts"])
    df_3m.set_index("ts", inplace=True)

    # Load 15m data
    query_15m = f"""
    SELECT time as ts, open, high, low, close, volume
    FROM ohlcv
    WHERE symbol = 'XRPUSDT' AND timeframe = '15m'
    AND time >= '{start_date}' AND time < '{end_date}'
    ORDER BY time
    """
    df_15m = pd.read_sql(query_15m, conn, parse_dates=["ts"])
    df_15m.set_index("ts", inplace=True)

    conn.close()

    logger.info(f"Loaded {len(df_3m)} 3m bars, {len(df_15m)} 15m bars")
    return df_3m, df_15m


def build_features(df: pd.DataFrame, prefix: str = "") -> Tuple[np.ndarray, List[str]]:
    """Build HMM features from OHLCV data."""
    # Returns
    df["ret"] = df["close"].pct_change()

    # Volatility (rolling std of returns)
    df["vol"] = df["ret"].rolling(20).std()

    # EMA slope
    ema20 = df["close"].ewm(span=20).mean()
    df["ema_slope"] = ema20.pct_change(5)

    # Box range (normalized)
    hh = df["high"].rolling(32).max()
    ll = df["low"].rolling(32).min()
    df["box_range"] = (hh - ll) / df["close"]

    # Breakout indicators
    df["B_up"] = (df["close"] - hh.shift(1)) / (df["vol"] * df["close"] + 1e-8)
    df["B_dn"] = (ll.shift(1) - df["close"]) / (df["vol"] * df["close"] + 1e-8)

    # Clean up
    df = df.dropna()

    feature_names = ["ret", "vol", "ema_slope", "box_range", "B_up", "B_dn"]
    if prefix:
        feature_names = [f"{prefix}_{n}" for n in feature_names]

    features = df[["ret", "vol", "ema_slope", "box_range", "B_up", "B_dn"]].values

    return features, feature_names, df.index


def build_features_15m(df: pd.DataFrame) -> Tuple[np.ndarray, List[str], pd.DatetimeIndex]:
    """Build features for 15m timeframe."""
    # Returns
    df["ret"] = df["close"].pct_change()

    # Volatility
    df["vol"] = df["ret"].rolling(20).std()

    # EMA slope
    ema20 = df["close"].ewm(span=20).mean()
    df["ema_slope"] = ema20.pct_change(5)

    # Box range
    hh = df["high"].rolling(32).max()
    ll = df["low"].rolling(32).min()
    df["box_range"] = (hh - ll) / df["close"]

    # Clean up
    df = df.dropna()

    feature_names = ["ret_15m", "vol_15m", "ema_slope_15m", "box_range_15m"]
    features = df[["ret", "vol", "ema_slope", "box_range"]].values

    return features, feature_names, df.index


# ============================================================================
# Backtest Functions
# ============================================================================

def run_backtest_with_regimes(
    df_3m: pd.DataFrame,
    regimes: List[str],
    timestamps: pd.DatetimeIndex,
) -> Dict:
    """Run backtest with given regime predictions."""
    # Align data
    df_3m = df_3m.loc[timestamps].copy()
    df_3m["regime"] = regimes

    # Build additional features needed for FSM
    df_3m["ret_3m"] = df_3m["close"].pct_change()
    df_3m["ema_fast"] = df_3m["close"].ewm(span=20).mean()
    df_3m["ema_slow"] = df_3m["close"].ewm(span=50).mean()
    df_3m["atr_3m"] = (df_3m["high"] - df_3m["low"]).ewm(span=14).mean()
    df_3m["volatility"] = df_3m["ret_3m"].rolling(20).std()
    df_3m["rsi_3m"] = 50.0  # Simplified
    df_3m["ema_slope_15m"] = df_3m["ema_fast"].pct_change(5)

    # EWM features for confirm layer
    df_3m["ewm_ret_15m"] = df_3m["ret_3m"].ewm(span=5).mean()
    df_3m["ewm_std_ret_15m"] = df_3m["ret_3m"].ewm(span=20).std()

    # Box features
    df_3m["HH_32"] = df_3m["high"].rolling(32).max()
    df_3m["LL_32"] = df_3m["low"].rolling(32).min()

    df_3m = df_3m.dropna()

    # Initialize components
    confirm_layer = RegimeConfirmLayer(ConfirmConfig())
    fsm = TradingFSM(FSMConfig())
    de = DecisionEngine(DecisionConfig(XGB_ENABLED=False))  # Disable XGB for fair comparison

    pos = PositionState()
    fsm_state = {}
    engine_state = {}

    trades = []

    for i, (ts, row) in enumerate(df_3m.iterrows()):
        price = row["close"]
        regime_raw = row["regime"]

        # Build confirm context
        row_15m = {
            "ewm_ret_15m": row["ewm_ret_15m"],
            "ewm_std_ret_15m": row["ewm_std_ret_15m"],
            "ema_slope_15m": row["ema_slope_15m"],
        }

        confirm_metrics = {
            "B_up": (price - row["HH_32"]) / (row["atr_3m"] + 1e-8) if row["HH_32"] else 0,
            "B_dn": (row["LL_32"] - price) / (row["atr_3m"] + 1e-8) if row["LL_32"] else 0,
        }

        # Simplified confirm: pass through regime
        regime_confirmed = regime_raw
        if regime_raw == "HIGH_VOL":
            regime_confirmed = "HIGH_VOL"

        confirm_ctx = ConfirmContext(
            regime_raw=regime_raw,
            regime_confirmed=regime_confirmed,
            confirm_reason="PASSTHROUGH",
            confirm_metrics=confirm_metrics,
        )

        # Market context
        ema_fast = row["ema_fast"]
        ema_slow = row["ema_slow"]

        ctx = MarketContext(
            symbol="XRPUSDT",
            ts=int(ts.timestamp() * 1000),
            price=price,
            row_3m={
                "close": price,
                "atr_3m": row["atr_3m"],
                "ema_fast_3m": ema_fast,
                "ema_slow_3m": ema_slow,
                "ret_3m": row["ret_3m"],
                "ret": row["ret_3m"],
                "volatility": row["volatility"],
                "rsi_3m": row["rsi_3m"],
                "ema_slope_15m": row["ema_slope_15m"],
                "ema_diff": (ema_fast - ema_slow) / price if price > 0 else 0,
                "price_to_ema20": (price - ema_fast) / ema_fast if ema_fast > 0 else 0,
                "price_to_ema50": (price - ema_slow) / ema_slow if ema_slow > 0 else 0,
            },
            row_15m=row_15m,
            zone={},
        )

        # Update position
        if pos.side != "FLAT":
            pos.bars_held_3m += 1
            if pos.side == "LONG":
                pos.unrealized_pnl = (price - pos.entry_price) / pos.entry_price
            else:
                pos.unrealized_pnl = (pos.entry_price - price) / pos.entry_price

        # FSM step
        cand, fsm_state = fsm.step(ctx, confirm_ctx, pos, fsm_state)

        # Decision engine
        decision, engine_state = de.decide(ctx, confirm_ctx, pos, cand, engine_state)

        # Execute decision
        if decision.action == "OPEN_LONG" and pos.side == "FLAT":
            pos = PositionState(
                side="LONG",
                entry_price=price,
                size=decision.size,
                entry_ts=int(ts.timestamp() * 1000),
                bars_held_3m=0,
            )
        elif decision.action == "OPEN_SHORT" and pos.side == "FLAT":
            pos = PositionState(
                side="SHORT",
                entry_price=price,
                size=decision.size,
                entry_ts=int(ts.timestamp() * 1000),
                bars_held_3m=0,
            )
        elif decision.action == "CLOSE" and pos.side != "FLAT":
            pnl_pct = pos.unrealized_pnl
            pnl_usdt = pnl_pct * pos.size
            trades.append({
                "entry_ts": pos.entry_ts,
                "exit_ts": int(ts.timestamp() * 1000),
                "side": pos.side,
                "entry_price": pos.entry_price,
                "exit_price": price,
                "pnl_pct": pnl_pct,
                "pnl_usdt": pnl_usdt,
                "bars_held": pos.bars_held_3m,
                "regime": regime_confirmed,
            })
            pos = PositionState()

    # Calculate metrics
    if trades:
        df_trades = pd.DataFrame(trades)
        n_trades = len(df_trades)
        wins = (df_trades["pnl_usdt"] > 0).sum()
        win_rate = wins / n_trades * 100
        total_pnl = df_trades["pnl_usdt"].sum()

        gross_profit = df_trades[df_trades["pnl_usdt"] > 0]["pnl_usdt"].sum()
        gross_loss = abs(df_trades[df_trades["pnl_usdt"] < 0]["pnl_usdt"].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        return {
            "n_trades": n_trades,
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "profit_factor": profit_factor,
            "avg_pnl": total_pnl / n_trades,
            "trades": trades,
        }
    else:
        return {
            "n_trades": 0,
            "win_rate": 0,
            "total_pnl": 0,
            "profit_factor": 0,
            "avg_pnl": 0,
            "trades": [],
        }


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Compare SingleHMM vs MultiHMM")
    parser.add_argument("--train-start", default="2024-07-01", help="Training start date")
    parser.add_argument("--train-end", default="2025-07-01", help="Training end date")
    parser.add_argument("--test-start", default="2025-07-01", help="Test start date")
    parser.add_argument("--test-end", default="2026-01-15", help="Test end date")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("SingleHMM vs MultiHMM Comparison")
    logger.info("=" * 60)

    # Load data
    logger.info(f"\nLoading training data: {args.train_start} ~ {args.train_end}")
    df_3m_train, df_15m_train = load_data(args.train_start, args.train_end)

    logger.info(f"\nLoading test data: {args.test_start} ~ {args.test_end}")
    df_3m_test, df_15m_test = load_data(args.test_start, args.test_end)

    # Build features
    logger.info("\nBuilding features...")

    # 3m features
    features_3m_train, names_3m, ts_3m_train = build_features(df_3m_train.copy())
    features_3m_test, _, ts_3m_test = build_features(df_3m_test.copy())

    # 15m features
    features_15m_train, names_15m, ts_15m_train = build_features_15m(df_15m_train.copy())
    features_15m_test, _, ts_15m_test = build_features_15m(df_15m_test.copy())

    logger.info(f"3m train: {features_3m_train.shape}, test: {features_3m_test.shape}")
    logger.info(f"15m train: {features_15m_train.shape}, test: {features_15m_test.shape}")

    # ========================================================================
    # Method 1: SingleHMM (5-state, 3m only)
    # ========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("Method 1: SingleHMM (5-state, 3m only)")
    logger.info("=" * 60)

    single_hmm = SingleHMM(n_states=5)
    ll_single = single_hmm.train(features_3m_train, names_3m)
    logger.info(f"SingleHMM trained, log-likelihood: {ll_single:.2f}")
    logger.info(f"State labels: {single_hmm.state_labels}")

    # Predict on test
    regimes_single = single_hmm.predict(features_3m_test)

    # Regime distribution
    regime_counts_single = pd.Series(regimes_single).value_counts()
    logger.info(f"\nSingleHMM regime distribution (test):")
    for regime, count in regime_counts_single.items():
        pct = count / len(regimes_single) * 100
        logger.info(f"  {regime}: {count} ({pct:.1f}%)")

    # Backtest
    results_single = run_backtest_with_regimes(df_3m_test.copy(), regimes_single, ts_3m_test)
    logger.info(f"\nSingleHMM Backtest Results:")
    logger.info(f"  Trades: {results_single['n_trades']}")
    logger.info(f"  Win Rate: {results_single['win_rate']:.1f}%")
    logger.info(f"  Total PnL: ${results_single['total_pnl']:.2f}")
    logger.info(f"  Profit Factor: {results_single['profit_factor']:.2f}")

    # ========================================================================
    # Method 2: MultiHMM (3m + 15m fusion)
    # ========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("Method 2: MultiHMM (3m + 15m fusion)")
    logger.info("=" * 60)

    multi_hmm = MultiHMM(
        n_states_3m=4,
        n_states_15m=4,
        fast_weight=0.4,
        mid_weight=0.6,
    )
    lls = multi_hmm.train(features_3m_train, names_3m, features_15m_train, names_15m)
    logger.info(f"MultiHMM trained, log-likelihood 3m: {lls['ll_3m']:.2f}, 15m: {lls['ll_15m']:.2f}")
    logger.info(f"State labels 3m: {multi_hmm.state_labels_3m}")
    logger.info(f"State labels 15m: {multi_hmm.state_labels_15m}")

    # Predict on test (need aligned timestamps)
    regimes_multi = multi_hmm.predict(features_3m_test, features_15m_test, ts_3m_test, ts_15m_test)

    # Regime distribution
    regime_counts_multi = pd.Series(regimes_multi).value_counts()
    logger.info(f"\nMultiHMM regime distribution (test):")
    for regime, count in regime_counts_multi.items():
        pct = count / len(regimes_multi) * 100
        logger.info(f"  {regime}: {count} ({pct:.1f}%)")

    # Backtest
    results_multi = run_backtest_with_regimes(df_3m_test.copy(), regimes_multi, ts_3m_test)
    logger.info(f"\nMultiHMM Backtest Results:")
    logger.info(f"  Trades: {results_multi['n_trades']}")
    logger.info(f"  Win Rate: {results_multi['win_rate']:.1f}%")
    logger.info(f"  Total PnL: ${results_multi['total_pnl']:.2f}")
    logger.info(f"  Profit Factor: {results_multi['profit_factor']:.2f}")

    # ========================================================================
    # Comparison Summary
    # ========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("COMPARISON SUMMARY")
    logger.info("=" * 60)

    print(f"\n{'Metric':<20} {'SingleHMM':>15} {'MultiHMM':>15} {'Diff':>15}")
    print("-" * 65)
    print(f"{'Trades':<20} {results_single['n_trades']:>15} {results_multi['n_trades']:>15} {results_multi['n_trades'] - results_single['n_trades']:>+15}")
    print(f"{'Win Rate':<20} {results_single['win_rate']:>14.1f}% {results_multi['win_rate']:>14.1f}% {results_multi['win_rate'] - results_single['win_rate']:>+14.1f}%")
    print(f"{'Total PnL':<20} ${results_single['total_pnl']:>13.2f} ${results_multi['total_pnl']:>13.2f} ${results_multi['total_pnl'] - results_single['total_pnl']:>+13.2f}")
    print(f"{'Profit Factor':<20} {results_single['profit_factor']:>15.2f} {results_multi['profit_factor']:>15.2f} {results_multi['profit_factor'] - results_single['profit_factor']:>+15.2f}")

    # Regime agreement analysis
    if len(regimes_single) == len(regimes_multi):
        agreement = sum(1 for s, m in zip(regimes_single, regimes_multi) if s == m)
        agreement_pct = agreement / len(regimes_single) * 100
        logger.info(f"\nRegime Agreement: {agreement}/{len(regimes_single)} ({agreement_pct:.1f}%)")

    # Save results
    output_dir = Path("outputs/hmm_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "test_period": f"{args.test_start} ~ {args.test_end}",
        "single_hmm": {
            "n_trades": results_single["n_trades"],
            "win_rate": results_single["win_rate"],
            "total_pnl": results_single["total_pnl"],
            "profit_factor": results_single["profit_factor"],
        },
        "multi_hmm": {
            "n_trades": results_multi["n_trades"],
            "win_rate": results_multi["win_rate"],
            "total_pnl": results_multi["total_pnl"],
            "profit_factor": results_multi["profit_factor"],
        },
    }

    with open(output_dir / "comparison_results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
