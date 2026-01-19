#!/usr/bin/env python3
"""Diagnose exact differences between tuning and backtest results.

Step-by-step comparison:
1. HMM training and regime classification
2. ConfirmLayer behavior
3. FSM signal generation
4. DecisionEngine behavior
5. Data alignment
"""

import sys
import logging
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from collections import Counter

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from xrp4.core.types import (
    MarketContext,
    ConfirmContext,
    PositionState,
    CandidateSignal,
)
from xrp4.core.decision_engine import DecisionEngine
from xrp4.core.fsm import TradingFSM
from xrp4.regime.confirm import RegimeConfirmLayer, ConfirmConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_data():
    """Load data from TimescaleDB."""
    import psycopg2
    conn = psycopg2.connect(
        host="localhost", port=5432, database="xrp_timeseries",
        user="xrp_user", password="xrp_password_change_me",
    )
    query = """
        SELECT time as timestamp, open, high, low, close, volume
        FROM ohlcv WHERE symbol = %s AND timeframe = %s
          AND time >= %s AND time < %s
        ORDER BY time ASC
    """
    start = datetime(2023, 1, 1)
    end = datetime(2024, 12, 31)

    df_3m = pd.read_sql(query, conn, params=("XRPUSDT", "3m", start, end))
    df_3m["timestamp"] = pd.to_datetime(df_3m["timestamp"])
    df_3m.set_index("timestamp", inplace=True)

    df_15m = pd.read_sql(query, conn, params=("XRPUSDT", "15m", start, end))
    df_15m["timestamp"] = pd.to_datetime(df_15m["timestamp"])
    df_15m.set_index("timestamp", inplace=True)

    for col in ["open", "high", "low", "close", "volume"]:
        df_3m[col] = df_3m[col].astype(float)
        df_15m[col] = df_15m[col].astype(float)

    conn.close()
    return df_3m, df_15m


def add_features_tuning(df, suffix=""):
    """Add features - TUNING SCRIPT VERSION."""
    df = df.copy()
    df[f"ret{suffix}"] = df["close"].pct_change()
    df[f"ema_20{suffix}"] = df["close"].ewm(span=20, adjust=False).mean()
    df[f"ema_50{suffix}"] = df["close"].ewm(span=50, adjust=False).mean()
    df[f"ema_fast{suffix}"] = df[f"ema_20{suffix}"]
    df[f"ema_slow{suffix}"] = df[f"ema_50{suffix}"]
    df[f"ema_slope{suffix}"] = df[f"ema_20{suffix}"].pct_change(5)
    df[f"ewm_ret{suffix}"] = df[f"ret{suffix}"].ewm(span=20, adjust=False).mean()
    df[f"ewm_std_ret{suffix}"] = df[f"ret{suffix}"].ewm(span=20, adjust=False).std()

    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df[f"atr{suffix}"] = tr.rolling(14).mean()

    return df


def train_hmm_tuning(df_15m):
    """Train HMM - TUNING SCRIPT VERSION (6 features)."""
    df = df_15m.copy()
    df["ret"] = df["close"].pct_change()
    df["vol"] = df["ret"].rolling(20).std()
    df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema_slope"] = df["ema_20"].pct_change(5)

    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()

    df["box_high"] = df["high"].rolling(32).max()
    df["box_low"] = df["low"].rolling(32).min()
    df["box_range"] = (df["box_high"] - df["box_low"]) / df["atr"]
    df["B_up"] = (df["close"] - df["box_high"].shift(1)) / df["atr"]
    df["B_dn"] = (df["box_low"].shift(1) - df["close"]) / df["atr"]

    features = df[["ret", "vol", "ema_slope", "box_range", "B_up", "B_dn"]].dropna()
    X = features.values

    hmm = GaussianHMM(n_components=5, covariance_type="full", n_iter=500, random_state=42)
    hmm.fit(X)
    states = hmm.predict(X)

    state_stats = {}
    for s in range(5):
        mask = states == s
        state_stats[s] = {
            "ret_mean": features["ret"].values[mask].mean(),
            "vol_mean": features["vol"].values[mask].mean(),
        }

    state_labels = {}
    used = set()

    vol_sorted = sorted(state_stats.items(), key=lambda x: x[1]["vol_mean"], reverse=True)
    state_labels[vol_sorted[0][0]] = "HIGH_VOL"
    used.add(vol_sorted[0][0])

    ret_sorted = sorted([(k, v) for k, v in state_stats.items() if k not in used],
                        key=lambda x: x[1]["ret_mean"], reverse=True)
    state_labels[ret_sorted[0][0]] = "TREND_UP"
    used.add(ret_sorted[0][0])
    state_labels[ret_sorted[-1][0]] = "TREND_DOWN"
    used.add(ret_sorted[-1][0])

    remaining = [k for k in range(5) if k not in used]
    if len(remaining) >= 2:
        state_labels[remaining[0]] = "TRANSITION"
        state_labels[remaining[1]] = "RANGE"
    elif remaining:
        state_labels[remaining[0]] = "RANGE"

    return states, state_labels, features.index, df


def resample_to_15m(df_3m):
    """Resample 3m to 15m - BACKTEST VERSION."""
    df = df_3m.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")

    resampled = df.resample("15min").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum",
    })
    resampled = resampled.dropna(subset=["close"])
    return resampled


def train_hmm_backtest(df_3m):
    """Train HMM - BACKTEST VERSION (from resampled 3m data)."""
    df_15m = resample_to_15m(df_3m)

    close = df_15m["close"]
    high = df_15m["high"]
    low = df_15m["low"]

    df_15m["ret"] = close.pct_change()
    df_15m["vol"] = df_15m["ret"].rolling(20).std()
    ema_20 = close.ewm(span=20, adjust=False).mean()
    df_15m["ema_slope"] = ema_20.pct_change(5)

    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False).mean()

    df_15m["box_high"] = high.rolling(32).max()
    df_15m["box_low"] = low.rolling(32).min()
    df_15m["box_range"] = (df_15m["box_high"] - df_15m["box_low"]) / atr
    df_15m["B_up"] = (close - df_15m["box_high"].shift(1)) / atr
    df_15m["B_dn"] = (df_15m["box_low"].shift(1) - close) / atr

    features = df_15m[["ret", "vol", "ema_slope", "box_range", "B_up", "B_dn"]].dropna()
    X = features.values

    hmm = GaussianHMM(n_components=5, covariance_type="full", n_iter=500, random_state=42)
    hmm.fit(X)
    states = hmm.predict(X)

    state_stats = {}
    for s in range(5):
        mask = states == s
        state_stats[s] = {
            "ret_mean": features["ret"].values[mask].mean(),
            "vol_mean": features["vol"].values[mask].mean(),
        }

    state_labels = {}
    used = set()

    vol_sorted = sorted(state_stats.items(), key=lambda x: x[1]["vol_mean"], reverse=True)
    state_labels[vol_sorted[0][0]] = "HIGH_VOL"
    used.add(vol_sorted[0][0])

    ret_sorted = sorted([(k, v) for k, v in state_stats.items() if k not in used],
                        key=lambda x: x[1]["ret_mean"], reverse=True)
    state_labels[ret_sorted[0][0]] = "TREND_UP"
    used.add(ret_sorted[0][0])
    state_labels[ret_sorted[-1][0]] = "TREND_DOWN"
    used.add(ret_sorted[-1][0])

    remaining = [k for k in range(5) if k not in used]
    if len(remaining) >= 2:
        state_labels[remaining[0]] = "TRANSITION"
        state_labels[remaining[1]] = "RANGE"
    elif remaining:
        state_labels[remaining[0]] = "RANGE"

    return states, state_labels, features.index, df_15m


def main():
    logger.info("=" * 70)
    logger.info("DIAGNOSTIC: TUNING vs BACKTEST DIFFERENCES")
    logger.info("=" * 70)

    # Load data
    logger.info("\n[1] LOADING DATA...")
    df_3m, df_15m_raw = load_data()
    logger.info(f"  3m from DB: {len(df_3m)} bars, {df_3m.index[0]} ~ {df_3m.index[-1]}")
    logger.info(f"  15m from DB: {len(df_15m_raw)} bars, {df_15m_raw.index[0]} ~ {df_15m_raw.index[-1]}")

    # Compare 15m data sources
    df_15m_resampled = resample_to_15m(df_3m)
    logger.info(f"  15m resampled from 3m: {len(df_15m_resampled)} bars")

    # Check if timestamps align
    common_ts = df_15m_raw.index.intersection(df_15m_resampled.index)
    logger.info(f"  Common timestamps: {len(common_ts)}")

    if len(common_ts) > 0:
        # Check price differences
        close_diff = (df_15m_raw.loc[common_ts, "close"] - df_15m_resampled.loc[common_ts, "close"]).abs()
        logger.info(f"  Close price diff: max={close_diff.max():.6f}, mean={close_diff.mean():.6f}")

    # HMM Training comparison
    logger.info("\n[2] HMM TRAINING COMPARISON...")

    # Tuning version (uses raw 15m from DB)
    df_15m_tuning = add_features_tuning(df_15m_raw, "_15m")
    states_tuning, labels_tuning, idx_tuning, df_hmm_tuning = train_hmm_tuning(df_15m_raw)
    logger.info(f"  TUNING HMM: {len(states_tuning)} samples")
    logger.info(f"    State labels: {labels_tuning}")

    # Backtest version (uses resampled 15m from 3m)
    states_backtest, labels_backtest, idx_backtest, df_hmm_backtest = train_hmm_backtest(df_3m)
    logger.info(f"  BACKTEST HMM: {len(states_backtest)} samples")
    logger.info(f"    State labels: {labels_backtest}")

    # Compare feature values
    logger.info("\n[3] FEATURE VALUE COMPARISON...")
    common_idx = idx_tuning.intersection(idx_backtest)
    logger.info(f"  Common feature indices: {len(common_idx)}")

    if len(common_idx) > 100:
        sample_idx = common_idx[:1000]

        for feat in ["ret", "vol", "ema_slope", "box_range", "B_up", "B_dn"]:
            tuning_vals = df_hmm_tuning.loc[sample_idx, feat]
            backtest_vals = df_hmm_backtest.loc[sample_idx, feat]
            diff = (tuning_vals - backtest_vals).abs()
            corr = tuning_vals.corr(backtest_vals)
            logger.info(f"  {feat}: max_diff={diff.max():.6f}, mean_diff={diff.mean():.6f}, corr={corr:.4f}")

    # Regime distribution comparison
    logger.info("\n[4] REGIME DISTRIBUTION COMPARISON...")

    # Map states to labels
    regimes_tuning = [labels_tuning.get(s, "UNKNOWN") for s in states_tuning]
    regimes_backtest = [labels_backtest.get(s, "UNKNOWN") for s in states_backtest]

    tuning_dist = Counter(regimes_tuning)
    backtest_dist = Counter(regimes_backtest)

    logger.info("  TUNING regime distribution:")
    for regime, count in sorted(tuning_dist.items(), key=lambda x: -x[1]):
        pct = count / len(regimes_tuning) * 100
        logger.info(f"    {regime}: {count} ({pct:.1f}%)")

    logger.info("  BACKTEST regime distribution:")
    for regime, count in sorted(backtest_dist.items(), key=lambda x: -x[1]):
        pct = count / len(regimes_backtest) * 100
        logger.info(f"    {regime}: {count} ({pct:.1f}%)")

    # Check regime agreement on common timestamps
    logger.info("\n[5] REGIME AGREEMENT ON COMMON TIMESTAMPS...")

    # Create regime series
    regime_tuning_series = pd.Series(regimes_tuning, index=idx_tuning)
    regime_backtest_series = pd.Series(regimes_backtest, index=idx_backtest)

    common_regime_idx = regime_tuning_series.index.intersection(regime_backtest_series.index)
    if len(common_regime_idx) > 0:
        tuning_common = regime_tuning_series.loc[common_regime_idx]
        backtest_common = regime_backtest_series.loc[common_regime_idx]

        agreement = (tuning_common == backtest_common).sum()
        agreement_pct = agreement / len(common_regime_idx) * 100
        logger.info(f"  Common indices: {len(common_regime_idx)}")
        logger.info(f"  Agreement: {agreement} ({agreement_pct:.1f}%)")

        # Confusion matrix
        logger.info("\n  Confusion (rows=tuning, cols=backtest):")
        confusion = pd.crosstab(tuning_common, backtest_common)
        logger.info(f"\n{confusion}")

    # ATR calculation comparison
    logger.info("\n[6] ATR CALCULATION COMPARISON...")
    logger.info("  TUNING: tr.rolling(14).mean()")
    logger.info("  BACKTEST: tr.ewm(alpha=1/14, adjust=False).mean()")

    # Calculate both ATR methods on same data
    df_test = df_15m_raw.copy()
    high = df_test["high"]
    low = df_test["low"]
    close = df_test["close"]

    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)

    atr_rolling = tr.rolling(14).mean()
    atr_ewm = tr.ewm(alpha=1/14, adjust=False).mean()

    valid_idx = atr_rolling.dropna().index
    atr_diff = (atr_rolling.loc[valid_idx] - atr_ewm.loc[valid_idx]).abs()
    atr_corr = atr_rolling.loc[valid_idx].corr(atr_ewm.loc[valid_idx])

    logger.info(f"  ATR diff: max={atr_diff.max():.6f}, mean={atr_diff.mean():.6f}")
    logger.info(f"  ATR correlation: {atr_corr:.6f}")

    # Test split comparison
    logger.info("\n[7] TEST SPLIT COMPARISON...")

    # Tuning split
    n_tuning = len(states_tuning)
    split_tuning = int(n_tuning * 0.7)
    test_start_tuning = idx_tuning[split_tuning]
    logger.info(f"  TUNING: {n_tuning} total, split at {split_tuning}, test starts {test_start_tuning}")

    # Backtest split
    n_backtest = len(states_backtest)
    split_backtest = int(n_backtest * 0.7)
    test_start_backtest = idx_backtest[split_backtest]
    logger.info(f"  BACKTEST: {n_backtest} total, split at {split_backtest}, test starts {test_start_backtest}")

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY OF DIFFERENCES")
    logger.info("=" * 70)

    differences = []

    # 1. Data source
    if len(df_15m_raw) != len(df_15m_resampled):
        differences.append(f"15m data count: DB={len(df_15m_raw)} vs Resampled={len(df_15m_resampled)}")

    # 2. ATR method
    if atr_diff.mean() > 0.0001:
        differences.append(f"ATR calculation: rolling vs ewm (mean diff={atr_diff.mean():.6f})")

    # 3. Regime agreement
    if len(common_regime_idx) > 0 and agreement_pct < 90:
        differences.append(f"Regime agreement: only {agreement_pct:.1f}%")

    # 4. State labels
    if labels_tuning != labels_backtest:
        differences.append(f"State labels differ: {labels_tuning} vs {labels_backtest}")

    if differences:
        logger.info("\nKEY DIFFERENCES FOUND:")
        for i, diff in enumerate(differences, 1):
            logger.info(f"  {i}. {diff}")
    else:
        logger.info("\nNo major differences found - check FSM/DE logic")


if __name__ == "__main__":
    main()
