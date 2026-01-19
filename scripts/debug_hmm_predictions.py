#!/usr/bin/env python
"""Debug script to analyze HMM prediction issues."""

import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from xrp4.regime.multi_hmm_manager import MultiHMMManager
from xrp4.features.hmm_features import build_fast_hmm_features_v2, build_mid_hmm_features_v2
import yaml
import requests
import time

def get_klines(symbol="XRPUSDT", interval="3m", limit=500):
    """Fetch klines from Binance."""
    all_data = []
    remaining = limit
    current_end_time = None

    while remaining > 0:
        fetch_limit = min(remaining, 1000)
        url = "https://api.binance.com/api/v3/klines"
        params = {"symbol": symbol, "interval": interval, "limit": fetch_limit}
        if current_end_time:
            params["endTime"] = current_end_time

        resp = requests.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()

        if not data:
            break

        all_data = data + all_data
        remaining -= len(data)

        if remaining > 0 and len(data) == fetch_limit:
            current_end_time = data[0][0] - 1
            time.sleep(0.05)
        else:
            break

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore"
    ])
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    return df[["timestamp", "open", "high", "low", "close", "volume"]]


def main():
    checkpoint_dir = Path(__file__).parent.parent / "checkpoints" / "hmm"

    # Load latest run_id
    latest_file = checkpoint_dir / "latest_run_id.txt"
    if not latest_file.exists():
        print("No checkpoint found!")
        return

    with open(latest_file, "r") as f:
        run_id = f.read().strip()

    print(f"Loading checkpoint: {run_id}")

    # Load HMM
    policy_config_path = Path(__file__).parent.parent / "configs" / "hmm_gate_policy.yaml"
    multi_hmm = MultiHMMManager.from_config_file(policy_config_path, checkpoint_dir=checkpoint_dir)
    multi_hmm.load_checkpoints(run_id)

    # Check scaling parameters
    print("\n=== FastHMM Scaling Parameters ===")
    print(f"Feature mean: {multi_hmm.fast_hmm._feature_mean}")
    print(f"Feature std: {multi_hmm.fast_hmm._feature_std}")
    print(f"Feature names: {multi_hmm.fast_hmm.feature_names}")

    print("\n=== MidHMM Scaling Parameters ===")
    print(f"Feature mean: {multi_hmm.mid_hmm._feature_mean}")
    print(f"Feature std: {multi_hmm.mid_hmm._feature_std}")
    print(f"Feature names: {multi_hmm.mid_hmm.feature_names}")

    # Check state labels
    print("\n=== FastHMM State Labels ===")
    for state, label in multi_hmm.fast_hmm.state_labels.items():
        print(f"  State {state}: {label.value}")

    print("\n=== MidHMM State Labels ===")
    for state, label in multi_hmm.mid_hmm.state_labels.items():
        print(f"  State {state}: {label.value}")

    # Fetch some test data
    print("\n=== Fetching Test Data ===")
    df_3m = get_klines("XRPUSDT", "3m", 1000)
    df_15m = get_klines("XRPUSDT", "15m", 200)
    print(f"3m data: {len(df_3m)} bars, {df_3m['timestamp'].iloc[0]} to {df_3m['timestamp'].iloc[-1]}")
    print(f"15m data: {len(df_15m)} bars, {df_15m['timestamp'].iloc[0]} to {df_15m['timestamp'].iloc[-1]}")

    # Load feature config
    config_path = Path(__file__).parent.parent / "configs" / "hmm_features.yaml"
    with open(config_path, "r") as f:
        feature_config = yaml.safe_load(f)
    fast_feature_names = feature_config["hmm"]["fast_3m"]["features"]
    mid_feature_names = feature_config["hmm"]["mid_15m"]["features"]

    # Build features
    fast_features, fast_timestamps = build_fast_hmm_features_v2(df_3m, fast_feature_names)
    df_15m_ohlcv = df_15m[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    mid_features, mid_timestamps = build_mid_hmm_features_v2(df_15m_ohlcv, mid_feature_names)

    print(f"\nFast features shape: {fast_features.shape}")
    print(f"Mid features shape: {mid_features.shape}")

    # Check feature statistics
    print("\n=== Fast Feature Statistics (first 100 rows) ===")
    for i, name in enumerate(fast_feature_names):
        vals = fast_features[:100, i]
        print(f"  {name}: mean={np.mean(vals):.6f}, std={np.std(vals):.6f}, min={np.min(vals):.6f}, max={np.max(vals):.6f}")

    # Make predictions
    print("\n=== Making Predictions ===")
    predictions = {"RANGE": 0, "TREND_UP": 0, "TREND_DOWN": 0, "TRANSITION": 0, "UNKNOWN": 0}
    fast_state_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    mid_state_counts = {0: 0, 1: 0, 2: 0, 3: 0}

    sample_preds = []

    for i in range(min(500, len(fast_features))):
        # Find matching 15m index
        fast_ts = fast_timestamps[i]
        mid_idx = np.searchsorted(mid_timestamps, fast_ts)
        if mid_idx >= len(mid_features):
            mid_idx = len(mid_features) - 1

        packet = multi_hmm.predict(
            fast_features[i:i+1],
            mid_features[mid_idx:mid_idx+1]
        )

        label = packet.label_fused.value
        predictions[label] = predictions.get(label, 0) + 1

        # Track state indices
        if packet.fast_pred:
            fast_state_counts[packet.fast_pred.state_idx] = fast_state_counts.get(packet.fast_pred.state_idx, 0) + 1
        if packet.mid_pred:
            mid_state_counts[packet.mid_pred.state_idx] = mid_state_counts.get(packet.mid_pred.state_idx, 0) + 1

        # Sample first 10 predictions
        if i < 10:
            sample_preds.append({
                "idx": i,
                "fused": label,
                "fast_label": packet.fast_pred.label.value if packet.fast_pred else "N/A",
                "fast_state": packet.fast_pred.state_idx if packet.fast_pred else -1,
                "fast_conf": f"{packet.fast_pred.confidence:.3f}" if packet.fast_pred else "N/A",
                "fast_probs": packet.fast_pred.state_probs.tolist() if packet.fast_pred else [],
                "mid_label": packet.mid_pred.label.value if packet.mid_pred else "N/A",
                "mid_state": packet.mid_pred.state_idx if packet.mid_pred else -1,
                "mid_conf": f"{packet.mid_pred.confidence:.3f}" if packet.mid_pred else "N/A",
            })

    print("\n=== Prediction Distribution (500 samples) ===")
    for label, count in sorted(predictions.items(), key=lambda x: -x[1]):
        pct = count / 500 * 100
        print(f"  {label}: {count} ({pct:.1f}%)")

    print("\n=== FastHMM State Distribution ===")
    for state, count in sorted(fast_state_counts.items()):
        label = multi_hmm.fast_hmm.state_labels.get(state, "UNKNOWN")
        pct = count / 500 * 100
        print(f"  State {state} ({label.value if hasattr(label, 'value') else label}): {count} ({pct:.1f}%)")

    print("\n=== MidHMM State Distribution ===")
    for state, count in sorted(mid_state_counts.items()):
        label = multi_hmm.mid_hmm.state_labels.get(state, "UNKNOWN")
        pct = count / 500 * 100
        print(f"  State {state} ({label.value if hasattr(label, 'value') else label}): {count} ({pct:.1f}%)")

    print("\n=== Sample Predictions (first 10) ===")
    for pred in sample_preds:
        print(f"  [{pred['idx']}] fused={pred['fused']}, "
              f"fast={pred['fast_label']}(s{pred['fast_state']}, c={pred['fast_conf']}), "
              f"mid={pred['mid_label']}(s{pred['mid_state']}, c={pred['mid_conf']})")
        if pred['fast_probs']:
            print(f"        fast_probs={[f'{p:.3f}' for p in pred['fast_probs']]}")

    # Check HMM model parameters
    print("\n=== FastHMM Model Parameters ===")
    print(f"Transition matrix:\n{multi_hmm.fast_hmm.model.transmat_}")
    print(f"Start probabilities: {multi_hmm.fast_hmm.model.startprob_}")
    print(f"Means (first 2 features per state):")
    for state in range(4):
        print(f"  State {state}: {multi_hmm.fast_hmm.model.means_[state, :2]}")

    # Check scaled features
    print("\n=== Scaled Feature Analysis ===")
    sample_features = fast_features[:10]
    fast_mean = multi_hmm.fast_hmm._feature_mean
    fast_std = multi_hmm.fast_hmm._feature_std
    scaled = (sample_features - fast_mean) / fast_std

    print(f"Sample raw features (first row):")
    for i, name in enumerate(fast_feature_names):
        print(f"  {name}: raw={sample_features[0, i]:.6f}, scaled={scaled[0, i]:.3f}, training_mean={fast_mean[i]:.6f}")

    print(f"\nScaled feature stats:")
    for i, name in enumerate(fast_feature_names):
        vals = scaled[:, i]
        print(f"  {name}: mean={np.mean(vals):.2f}, std={np.std(vals):.2f}, range=[{np.min(vals):.2f}, {np.max(vals):.2f}]")

    # Check emission probabilities
    print("\n=== Emission Log-Probabilities (first sample) ===")
    from scipy.stats import multivariate_normal
    scaled_sample = scaled[0:1]
    for state in range(4):
        mean = multi_hmm.fast_hmm.model.means_[state]
        covar = multi_hmm.fast_hmm.model.covars_[state]
        # For diagonal covariance
        log_prob = -0.5 * np.sum(((scaled_sample[0] - mean) ** 2) / covar)
        log_prob -= 0.5 * np.sum(np.log(2 * np.pi * covar))
        print(f"  State {state}: log_prob = {log_prob:.2f}")


if __name__ == "__main__":
    main()
