#!/usr/bin/env python3
"""Debug TREND pullback signals - find out why no trades."""

import sys
from pathlib import Path
from datetime import datetime
from collections import Counter

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from xrp4.core.types import MarketContext, ConfirmContext, PositionState, CandidateSignal


def load_data():
    """Load data from TimescaleDB."""
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

    start = datetime(2023, 1, 1)
    end = datetime(2024, 12, 31)

    df_3m = pd.read_sql(query, conn, params=("XRPUSDT", "3m", start, end))
    df_3m["timestamp"] = pd.to_datetime(df_3m["timestamp"])
    df_3m.set_index("timestamp", inplace=True)

    df_15m = pd.read_sql(query, conn, params=("XRPUSDT", "15m", start, end))
    df_15m["timestamp"] = pd.to_datetime(df_15m["timestamp"])
    df_15m.set_index("timestamp", inplace=True)

    conn.close()
    return df_3m, df_15m


def add_features(df, suffix=""):
    """Add technical indicators."""
    df = df.copy()
    df[f"ret{suffix}"] = df["close"].pct_change()
    df[f"ema_20{suffix}"] = df["close"].ewm(span=20, adjust=False).mean()
    df[f"ema_50{suffix}"] = df["close"].ewm(span=50, adjust=False).mean()
    df[f"ema_fast{suffix}"] = df[f"ema_20{suffix}"]
    df[f"ema_slow{suffix}"] = df[f"ema_50{suffix}"]
    df[f"ema_slope{suffix}"] = df[f"ema_20{suffix}"].pct_change(5)

    # ATR
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df[f"atr{suffix}"] = tr.rolling(14).mean()

    # RSI
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, 1e-10)
    df[f"rsi{suffix}"] = 100 - (100 / (1 + rs))

    df[f"vol{suffix}"] = df[f"ret{suffix}"].rolling(20).std()
    return df


def train_hmm(df_15m):
    """Train HMM and get states."""
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

    lookback = 32
    df["box_high"] = df["high"].rolling(lookback).max()
    df["box_low"] = df["low"].rolling(lookback).min()
    df["box_range"] = (df["box_high"] - df["box_low"]) / df["atr"]
    df["B_up"] = (df["close"] - df["box_high"].shift(1)) / df["atr"]
    df["B_dn"] = (df["box_low"].shift(1) - df["close"]) / df["atr"]

    features = df[["ret", "vol", "ema_slope", "box_range", "B_up", "B_dn"]].dropna()
    X = features.values

    hmm = GaussianHMM(n_components=5, covariance_type="full", n_iter=500, random_state=42)
    hmm.fit(X)
    states = hmm.predict(X)

    # Label states
    state_stats = {}
    for s in range(5):
        mask = states == s
        state_stats[s] = {
            "ret_mean": features["ret"].values[mask].mean(),
            "vol_mean": features["vol"].values[mask].mean(),
            "ema_slope_mean": features["ema_slope"].values[mask].mean(),
            "count": mask.sum(),
        }

    state_labels = {}
    used = set()

    # HIGH_VOL
    vol_sorted = sorted(state_stats.items(), key=lambda x: x[1]["vol_mean"], reverse=True)
    state_labels[vol_sorted[0][0]] = "HIGH_VOL"
    used.add(vol_sorted[0][0])

    # TREND_UP
    ret_sorted = sorted([(k, v) for k, v in state_stats.items() if k not in used],
                        key=lambda x: x[1]["ret_mean"], reverse=True)
    state_labels[ret_sorted[0][0]] = "TREND_UP"
    used.add(ret_sorted[0][0])

    # TREND_DOWN
    state_labels[ret_sorted[-1][0]] = "TREND_DOWN"
    used.add(ret_sorted[-1][0])

    # TRANSITION & RANGE
    remaining = [k for k in range(5) if k not in used]
    if len(remaining) >= 2:
        state_labels[remaining[0]] = "TRANSITION"
        state_labels[remaining[1]] = "RANGE"
    elif remaining:
        state_labels[remaining[0]] = "RANGE"

    return states, state_labels, features.index


def main():
    print("=" * 70)
    print("DEBUG TREND PULLBACK SIGNALS")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    df_3m, df_15m = load_data()
    print(f"3m bars: {len(df_3m)}, 15m bars: {len(df_15m)}")

    # Add features
    print("Adding features...")
    df_3m = add_features(df_3m, "_3m")
    df_15m = add_features(df_15m, "_15m")

    # Train HMM
    print("Training HMM...")
    states_15m, state_labels, hmm_index = train_hmm(df_15m)
    print(f"State labels: {state_labels}")

    # Map states to 15m dataframe
    state_df = pd.DataFrame({"state": states_15m}, index=hmm_index)

    # Align 15m to 3m
    df_15m_aligned = df_15m.reindex(df_3m.index, method="ffill")
    state_3m = state_df.reindex(df_3m.index, method="ffill")

    # Get regime labels
    inv_labels = {v: k for k, v in state_labels.items()}
    trend_up_state = inv_labels.get("TREND_UP")
    trend_down_state = inv_labels.get("TREND_DOWN")

    print(f"\nTREND_UP state: {trend_up_state}")
    print(f"TREND_DOWN state: {trend_down_state}")

    # Count regime distribution
    regime_counts = state_3m["state"].value_counts()
    print("\nRegime distribution (3m bars):")
    for state_id, count in regime_counts.items():
        label = state_labels.get(int(state_id), "UNKNOWN")
        print(f"  {label} ({state_id}): {count} ({count/len(state_3m)*100:.1f}%)")

    # Split test set
    n_total = len(df_3m)
    test_start = int(n_total * 0.7)
    test_df_3m = df_3m.iloc[test_start:]
    test_df_15m = df_15m_aligned.iloc[test_start:]
    test_states = state_3m.iloc[test_start:]

    print(f"\nTest set size: {len(test_df_3m)} bars")

    # Count TREND bars in test
    trend_up_count = (test_states["state"] == trend_up_state).sum()
    trend_down_count = (test_states["state"] == trend_down_state).sum()
    print(f"TREND_UP bars in test: {trend_up_count}")
    print(f"TREND_DOWN bars in test: {trend_down_count}")

    # Check signal conditions
    print("\n" + "=" * 70)
    print("CHECKING SIGNAL CONDITIONS")
    print("=" * 70)

    # Parameters
    TREND_PULLBACK_TO_EMA_ATR = 1.0
    TREND_MIN_EMA_SLOPE_15M = 0.03
    MIN_REBOUND_RET = 0.0

    signal_reasons = Counter()
    signals_generated = []

    for i in range(len(test_df_3m)):
        row_3m = test_df_3m.iloc[i]
        row_15m = test_df_15m.iloc[i]
        state = test_states.iloc[i]["state"]

        if pd.isna(state):
            signal_reasons["na_state"] += 1
            continue

        regime = state_labels.get(int(state), "UNKNOWN")

        if regime not in ["TREND_UP", "TREND_DOWN"]:
            signal_reasons[f"not_trend_{regime}"] += 1
            continue

        # Get values
        price = row_3m["close"]
        atr = row_3m.get("atr_3m", 0.01)
        ema_fast = row_3m.get("ema_fast_3m", row_3m.get("ema_20_3m", price))
        ret = row_3m.get("ret_3m", 0)
        ema_slope_15m = row_15m.get("ema_slope_15m", row_15m.get("ema_slope", 0))

        if pd.isna(atr) or pd.isna(ema_fast) or pd.isna(ret) or pd.isna(ema_slope_15m):
            signal_reasons["nan_features"] += 1
            continue

        # Filter 1: EMA slope
        if abs(ema_slope_15m) < TREND_MIN_EMA_SLOPE_15M:
            signal_reasons["weak_slope"] += 1
            continue

        # Filter 2: Pullback
        if regime == "TREND_UP":
            threshold = ema_fast + TREND_PULLBACK_TO_EMA_ATR * atr
            is_pullback = price <= threshold
            is_rebound = ret > MIN_REBOUND_RET

            if not is_pullback:
                signal_reasons["no_pullback_up"] += 1
                continue
            if not is_rebound:
                signal_reasons["no_rebound_up"] += 1
                continue

            signals_generated.append(("LONG_TREND_PULLBACK", i, price, ema_fast, atr, ret, ema_slope_15m))

        else:  # TREND_DOWN
            threshold = ema_fast - TREND_PULLBACK_TO_EMA_ATR * atr
            is_pullback = price >= threshold
            is_rejection = ret < -MIN_REBOUND_RET

            if not is_pullback:
                signal_reasons["no_pullback_down"] += 1
                continue
            if not is_rejection:
                signal_reasons["no_rejection_down"] += 1
                continue

            signals_generated.append(("SHORT_TREND_PULLBACK", i, price, ema_fast, atr, ret, ema_slope_15m))

    print(f"\nSignal filter reasons:")
    for reason, count in signal_reasons.most_common():
        print(f"  {reason}: {count}")

    print(f"\nTotal signals generated: {len(signals_generated)}")

    if signals_generated:
        print("\nFirst 10 signals:")
        for sig in signals_generated[:10]:
            print(f"  {sig[0]}: idx={sig[1]}, price={sig[2]:.4f}, ema={sig[3]:.4f}, "
                  f"atr={sig[4]:.4f}, ret={sig[5]:.6f}, slope={sig[6]:.6f}")

    # Test with relaxed parameters
    print("\n" + "=" * 70)
    print("TESTING RELAXED PARAMETERS")
    print("=" * 70)

    for slope_thresh in [0.01, 0.02, 0.03]:
        for pullback_atr in [1.0, 2.0, 3.0]:
            signals = 0
            for i in range(len(test_df_3m)):
                row_3m = test_df_3m.iloc[i]
                row_15m = test_df_15m.iloc[i]
                state = test_states.iloc[i]["state"]

                if pd.isna(state):
                    continue

                regime = state_labels.get(int(state), "UNKNOWN")
                if regime not in ["TREND_UP", "TREND_DOWN"]:
                    continue

                price = row_3m["close"]
                atr = row_3m.get("atr_3m", 0.01)
                ema_fast = row_3m.get("ema_fast_3m", row_3m.get("ema_20_3m", price))
                ret = row_3m.get("ret_3m", 0)
                ema_slope_15m = row_15m.get("ema_slope_15m", row_15m.get("ema_slope", 0))

                if pd.isna(atr) or pd.isna(ema_fast) or pd.isna(ret) or pd.isna(ema_slope_15m):
                    continue

                if abs(ema_slope_15m) < slope_thresh:
                    continue

                if regime == "TREND_UP":
                    threshold = ema_fast + pullback_atr * atr
                    if price <= threshold and ret > 0:
                        signals += 1
                else:
                    threshold = ema_fast - pullback_atr * atr
                    if price >= threshold and ret < 0:
                        signals += 1

            print(f"slope={slope_thresh}, pullback_atr={pullback_atr}: {signals} signals")


if __name__ == "__main__":
    main()
