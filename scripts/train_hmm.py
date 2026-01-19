#!/usr/bin/env python
"""Train HMM models for xrp-4.

Trains Fast HMM (3m) and Mid HMM (15m) from raw candle data.

Usage:
    python scripts/train_hmm.py --start 2023-01-01 --end 2024-12-31
    python scripts/train_hmm.py --start 2023-01-01 --end 2024-12-31 --run_id v1
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from xrp4.data.candles import load_candles, resample_candles
from xrp4.features.hmm_features import (
    build_fast_hmm_features_v2,
    build_mid_hmm_features_v2,
    resample_to_15m,
)
from xrp4.regime.multi_hmm_manager import MultiHMMManager
from xrp4.regime.feature_contract import get_contract, validate_training_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(config_path: Path):
    """Load YAML config file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train HMM models for xrp-4")
    parser.add_argument(
        "--start",
        type=str,
        default="2023-01-01",
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default="2024-12-31",
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="XRPUSDT",
        help="Trading symbol",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=Path,
        default=Path("checkpoints/hmm"),
        help="Checkpoint output directory",
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="Run ID for checkpoints (default: timestamp)",
    )
    parser.add_argument(
        "--features_config",
        type=Path,
        default=Path("../configs/hmm_features.yaml"),
        help="HMM features config path (feature contract)",
    )
    parser.add_argument(
        "--policy",
        type=Path,
        default=Path("../configs/hmm_gate_policy.yaml"),
        help="HMM Gate policy config path",
    )

    args = parser.parse_args()

    # Resolve paths relative to script location
    script_dir = Path(__file__).resolve().parent
    if not args.features_config.is_absolute():
        args.features_config = script_dir / args.features_config
    if not args.policy.is_absolute():
        args.policy = script_dir / args.policy
    if not args.checkpoint_dir.is_absolute():
        args.checkpoint_dir = script_dir / args.checkpoint_dir

    # Create checkpoint directory
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("XRP-4 HMM TRAINING")
    logger.info("=" * 60)
    logger.info(f"Symbol: {args.symbol}")
    logger.info(f"Period: {args.start} to {args.end}")
    logger.info(f"Features config: {args.features_config}")
    logger.info(f"Policy config: {args.policy}")

    # Get feature lists from contract
    logger.info("Loading feature contract...")
    contract = get_contract(args.features_config)
    fast_features = contract.get_fast_features()
    mid_features = contract.get_mid_features()

    logger.info(f"Fast HMM features ({len(fast_features)}): {fast_features}")
    logger.info(f"Mid HMM features ({len(mid_features)}): {mid_features}")

    # Load raw candle data
    logger.info("Loading 3m candle data...")
    df_3m = load_candles(
        symbol=args.symbol,
        timeframe="3m",
        start=args.start,
        end=args.end,
    )

    if df_3m.empty:
        logger.error("No 3m candle data loaded!")
        sys.exit(1)

    logger.info(f"Loaded {len(df_3m)} 3m bars")
    logger.info(f"Date range: {df_3m['timestamp'].min()} to {df_3m['timestamp'].max()}")

    # Build Fast HMM features (3m with 15m context)
    logger.info("Building Fast HMM features (3m + 15m context)...")
    try:
        fast_matrix, fast_timestamps = build_fast_hmm_features_v2(
            df_3m.copy(),
            fast_features,
            drop_na=True,
        )
        logger.info(f"Fast HMM samples: {len(fast_matrix)}")
    except Exception as e:
        logger.error(f"Failed to build Fast HMM features: {e}")
        sys.exit(1)

    # Resample to 15m for Mid HMM
    logger.info("Resampling to 15m...")
    df_15m = resample_to_15m(df_3m.copy())
    logger.info(f"15m bars: {len(df_15m)}")

    # Build Mid HMM features (15m with 1h context)
    logger.info("Building Mid HMM features (15m + 1h context)...")
    try:
        mid_matrix, mid_timestamps = build_mid_hmm_features_v2(
            df_15m.copy(),
            mid_features,
            drop_na=True,
        )
        logger.info(f"Mid HMM samples: {len(mid_matrix)}")
    except Exception as e:
        logger.error(f"Failed to build Mid HMM features: {e}")
        sys.exit(1)

    # Initialize MultiHMMManager
    logger.info("Initializing MultiHMMManager...")

    # Try to load policy config, use defaults if not available
    try:
        manager = MultiHMMManager.from_config_file(
            args.policy,
            checkpoint_dir=args.checkpoint_dir,
        )
    except Exception as e:
        logger.warning(f"Could not load policy config: {e}")
        logger.info("Using default HMM configuration...")
        manager = MultiHMMManager(
            config={},
            checkpoint_dir=args.checkpoint_dir,
        )

    # Generate run_id
    run_id = args.run_id or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    logger.info(f"Training run_id: {run_id}")

    # Train models
    logger.info("=" * 60)
    logger.info("TRAINING HMM MODELS")
    logger.info("=" * 60)

    results = manager.train(
        fast_features=fast_matrix,
        fast_feature_names=fast_features,
        mid_features=mid_matrix,
        mid_feature_names=mid_features,
        fast_timestamps=fast_timestamps,
        mid_timestamps=mid_timestamps,
        run_id=run_id,
    )

    logger.info(f"Fast HMM log-likelihood: {results['fast_log_likelihood']:.2f}")
    logger.info(f"Mid HMM log-likelihood: {results['mid_log_likelihood']:.2f}")

    # Save checkpoints
    logger.info("=" * 60)
    logger.info("SAVING CHECKPOINTS")
    logger.info("=" * 60)

    saved_paths = manager.save_checkpoints(run_id)
    for name, path in saved_paths.items():
        logger.info(f"Saved {name}: {path}")

    # Print training report
    logger.info("=" * 60)
    logger.info("TRAINING REPORT")
    logger.info("=" * 60)

    report = manager.get_training_report()

    print("\n--- Fast HMM (3m) ---")
    fast_report = report["fast_hmm"]
    print(f"  States: {fast_report['n_states']}")
    print(f"  Features: {fast_report['n_features']}")
    print(f"  Samples: {fast_report['n_samples']}")
    print(f"  Log-likelihood: {fast_report['log_likelihood']:.2f}")
    print(f"  State labels: {fast_report['state_labels']}")

    print("\n--- Mid HMM (15m) ---")
    mid_report = report["mid_hmm"]
    print(f"  States: {mid_report['n_states']}")
    print(f"  Features: {mid_report['n_features']}")
    print(f"  Samples: {mid_report['n_samples']}")
    print(f"  Log-likelihood: {mid_report['log_likelihood']:.2f}")
    print(f"  State labels: {mid_report['state_labels']}")

    print("\n--- Fusion ---")
    fusion_report = report["fusion"]
    print(f"  Method: {fusion_report['method']}")
    print(f"  Fast weight: {fusion_report['fast_weight']}")
    print(f"  Mid weight: {fusion_report['mid_weight']}")

    print(f"\nCheckpoint run_id: {run_id}")
    print(f"Checkpoint directory: {args.checkpoint_dir}")

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
