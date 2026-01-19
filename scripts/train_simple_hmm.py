#!/usr/bin/env python3
"""
Train and save simple HMM model for regime detection.

This script trains a single HMM model (6 features) and saves it for consistent use across:
- tune_short_params.py
- backtest_fsm_pipeline.py
- Live trading

Usage:
    python scripts/train_simple_hmm.py --start 2024-01-01 --end 2025-01-01
    python scripts/train_simple_hmm.py --start 2024-07-01 --end 2025-01-01 --output models/hmm_2024h2.pkl
"""

import argparse
import logging
import pickle
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import psycopg2
from hmmlearn import hmm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class SimpleHMMBundle:
    """Bundle containing trained HMM model and metadata."""
    model: hmm.GaussianHMM
    state_labels: Dict[int, str]
    feature_names: List[str]
    train_start: str
    train_end: str
    n_states: int
    state_stats: Dict[int, Dict]
    created_at: str
    
    # For feature normalization (optional)
    feature_means: Optional[np.ndarray] = None
    feature_stds: Optional[np.ndarray] = None

    def get_regime(self, state_idx: int) -> str:
        """Get regime label for state index."""
        return self.state_labels.get(state_idx, "UNKNOWN")
    
    def predict(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict states and get probabilities."""
        states = self.model.predict(features)
        probs = self.model.predict_proba(features)
        return states, probs
    
    def get_labels(self, states: np.ndarray) -> np.ndarray:
        """Convert state indices to regime labels."""
        return np.array([self.state_labels.get(s, "UNKNOWN") for s in states])


def load_15m_data(start: str, end: str, symbol: str = "XRPUSDT") -> pd.DataFrame:
    """Load 15m data from TimescaleDB."""
    conn = psycopg2.connect(
        host="localhost", port=5432, database="xrp_timeseries",
        user="xrp_user", password="xrp_password_change_me",
    )

    query = """
        SELECT time as timestamp, open, high, low, close, volume
        FROM ohlcv
        WHERE symbol = %s AND timeframe = %s AND time >= %s AND time < %s
        ORDER BY time ASC
    """

    df_15m = pd.read_sql(query, conn, params=(symbol, "15m", start, end))
    conn.close()

    df_15m["timestamp"] = pd.to_datetime(df_15m["timestamp"])
    logger.info(f"Loaded {len(df_15m)} 15m bars from TimescaleDB")
    return df_15m


def compute_features(df_15m: pd.DataFrame) -> Tuple[np.ndarray, List[str], pd.Index]:
    """Compute HMM features from 15m data.

    Features (6 total) - MUST match backtest_fsm_pipeline.py and tune_short_params.py:
    - ret: 15m return
    - vol: rolling 20-bar volatility (std of returns)
    - ema_slope: EMA20 slope (5-bar pct change)
    - box_range: 32-bar box range / ATR
    - B_up: breakout up (close - box_high) / ATR
    - B_dn: breakout down (box_low - close) / ATR
    """
    df = df_15m.copy()
    df = df.set_index("timestamp")

    # Basic features
    df["ret"] = df["close"].pct_change()
    df["vol"] = df["ret"].rolling(20).std()
    df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema_slope"] = df["ema_20"].pct_change(5)

    # ATR (True Range)
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()

    # Box/Breakout features
    df["box_high"] = df["high"].rolling(32).max()
    df["box_low"] = df["low"].rolling(32).min()
    df["box_range"] = (df["box_high"] - df["box_low"]) / df["atr"]
    df["B_up"] = (df["close"] - df["box_high"].shift(1)) / df["atr"]
    df["B_dn"] = (df["box_low"].shift(1) - df["close"]) / df["atr"]

    # Feature selection - SAME ORDER as other scripts
    feature_names = ["ret", "vol", "ema_slope", "box_range", "B_up", "B_dn"]
    features_df = df[feature_names].dropna()

    return features_df.values, feature_names, features_df.index


def train_hmm_model(features: np.ndarray, feature_names: List[str], n_states: int = 5) -> Tuple[hmm.GaussianHMM, np.ndarray, Dict]:
    """Train HMM model.

    Args:
        features: Feature matrix (n_samples, n_features)
        feature_names: List of feature names
        n_states: Number of hidden states

    Returns:
        model: Trained HMM model
        states: Predicted state sequence
        state_stats: Statistics for each state
    """
    logger.info(f"Training HMM with {n_states} states on {len(features)} samples...")

    model = hmm.GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=500,
        random_state=42,  # IMPORTANT: Fixed seed for reproducibility
    )
    model.fit(features)
    states = model.predict(features)

    # Compute state statistics
    ret_idx = feature_names.index("ret")
    vol_idx = feature_names.index("vol")

    state_stats = {}
    for s in range(n_states):
        mask = states == s
        if mask.sum() > 0:
            state_stats[s] = {
                "count": int(mask.sum()),
                "pct": float(mask.sum() / len(states) * 100),
                "ret_mean": float(features[mask, ret_idx].mean()),
                "vol_mean": float(features[mask, vol_idx].mean()),
            }
        else:
            state_stats[s] = {"count": 0, "pct": 0, "ret_mean": 0, "vol_mean": 0}

    return model, states, state_stats


def label_states(state_stats: Dict, n_states: int = 5) -> Dict[int, str]:
    """Assign regime labels to states based on statistics.

    Labeling logic (MUST match other scripts):
    1. HIGH_VOL: highest vol_mean
    2. TREND_UP: highest ret_mean (excluding HIGH_VOL)
    3. TREND_DOWN: lowest ret_mean (excluding HIGH_VOL)
    4. TRANSITION, RANGE: remaining states
    """
    state_labels = {}
    used = set()

    # 1. HIGH_VOL = highest volatility
    vol_sorted = sorted(state_stats.items(), key=lambda x: x[1]["vol_mean"], reverse=True)
    state_labels[vol_sorted[0][0]] = "HIGH_VOL"
    used.add(vol_sorted[0][0])

    # 2. TREND_UP = highest return (excluding HIGH_VOL)
    # 3. TREND_DOWN = lowest return (excluding HIGH_VOL)
    ret_sorted = sorted(
        [(k, v) for k, v in state_stats.items() if k not in used],
        key=lambda x: x[1]["ret_mean"],
        reverse=True
    )

    if len(ret_sorted) >= 1:
        state_labels[ret_sorted[0][0]] = "TREND_UP"
        used.add(ret_sorted[0][0])

    if len(ret_sorted) >= 2:
        state_labels[ret_sorted[-1][0]] = "TREND_DOWN"
        used.add(ret_sorted[-1][0])

    # 4. Remaining = TRANSITION, RANGE
    remaining = [k for k in range(n_states) if k not in used]
    if len(remaining) >= 2:
        state_labels[remaining[0]] = "TRANSITION"
        state_labels[remaining[1]] = "RANGE"
    elif len(remaining) == 1:
        state_labels[remaining[0]] = "RANGE"

    return state_labels


def save_model(bundle: SimpleHMMBundle, output_path: Path):
    """Save HMM model bundle to file as a plain dict (for portability)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save as dict to avoid pickle class issues
    save_dict = {
        "model": bundle.model,  # HMM model (hmmlearn object)
        "state_labels": bundle.state_labels,
        "feature_names": bundle.feature_names,
        "train_start": bundle.train_start,
        "train_end": bundle.train_end,
        "n_states": bundle.n_states,
        "state_stats": bundle.state_stats,
        "created_at": bundle.created_at,
        "feature_means": bundle.feature_means,
        "feature_stds": bundle.feature_stds,
    }

    with open(output_path, "wb") as f:
        pickle.dump(save_dict, f)

    logger.info(f"Model saved to: {output_path}")


def load_hmm_model(model_path: Path) -> SimpleHMMBundle:
    """Load HMM model bundle from file.

    This function can be imported by other scripts:
        from train_simple_hmm import load_hmm_model
        bundle = load_hmm_model(Path("models/hmm_model.pkl"))
    """
    with open(model_path, "rb") as f:
        data = pickle.load(f)

    # Handle both old (dataclass) and new (dict) formats
    if isinstance(data, dict):
        bundle = SimpleHMMBundle(
            model=data["model"],
            state_labels=data["state_labels"],
            feature_names=data["feature_names"],
            train_start=data["train_start"],
            train_end=data["train_end"],
            n_states=data["n_states"],
            state_stats=data["state_stats"],
            created_at=data["created_at"],
            feature_means=data.get("feature_means"),
            feature_stds=data.get("feature_stds"),
        )
    else:
        # Old format - dataclass directly
        bundle = data

    return bundle


def main():
    parser = argparse.ArgumentParser(description="Train and save simple HMM model")
    parser.add_argument("--start", type=str, default="2024-01-01", help="Training start date")
    parser.add_argument("--end", type=str, default="2025-01-01", help="Training end date")
    parser.add_argument("--output", type=str, default="models/hmm_simple.pkl", help="Output path")
    parser.add_argument("--n-states", type=int, default=5, help="Number of HMM states")
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("SIMPLE HMM MODEL TRAINING")
    logger.info("=" * 70)
    logger.info(f"Period: {args.start} ~ {args.end}")
    logger.info(f"Output: {args.output}")

    # Load data
    logger.info("\nLoading 15m data...")
    df_15m = load_15m_data(args.start, args.end)
    logger.info(f"15m bars: {len(df_15m)}")

    # Compute features
    logger.info("\nComputing features...")
    features, feature_names, feature_index = compute_features(df_15m)
    logger.info(f"Features: {feature_names}")
    logger.info(f"Samples: {len(features)}")

    # Train HMM
    model, states, state_stats = train_hmm_model(features, feature_names, args.n_states)

    # Label states
    state_labels = label_states(state_stats, args.n_states)

    # Print state info
    logger.info("\n" + "=" * 70)
    logger.info("STATE LABELS")
    logger.info("=" * 70)
    for state_idx in range(args.n_states):
        label = state_labels.get(state_idx, "UNKNOWN")
        stats = state_stats[state_idx]
        logger.info(f"  State {state_idx} -> {label:12s}: {stats['count']:5d} bars ({stats['pct']:5.1f}%), "
                   f"ret={stats['ret_mean']:+.6f}, vol={stats['vol_mean']:.6f}")

    # Create bundle
    bundle = SimpleHMMBundle(
        model=model,
        state_labels=state_labels,
        feature_names=feature_names,
        train_start=args.start,
        train_end=args.end,
        n_states=args.n_states,
        state_stats=state_stats,
        created_at=datetime.now().isoformat(),
        feature_means=features.mean(axis=0),
        feature_stds=features.std(axis=0),
    )

    # Save
    output_path = Path(args.output)
    save_model(bundle, output_path)

    # Verification
    logger.info("\n" + "=" * 70)
    logger.info("VERIFICATION")
    logger.info("=" * 70)
    loaded = load_hmm_model(output_path)
    logger.info(f"Loaded model from: {output_path}")
    logger.info(f"  State labels: {loaded.state_labels}")
    logger.info(f"  Train period: {loaded.train_start} ~ {loaded.train_end}")
    logger.info(f"  Features: {loaded.feature_names}")
    logger.info(f"  Created at: {loaded.created_at}")

    # Test prediction
    test_states, test_probs = loaded.predict(features[:10])
    test_labels = loaded.get_labels(test_states)
    logger.info(f"\nTest prediction (first 10 bars):")
    logger.info(f"  States: {test_states}")
    logger.info(f"  Labels: {test_labels}")

    logger.info("\nDone!")
    logger.info(f"\nTo use this model in other scripts:")
    logger.info(f"  from train_simple_hmm import load_hmm_model")
    logger.info(f"  bundle = load_hmm_model(Path('{args.output}'))")


if __name__ == "__main__":
    main()
