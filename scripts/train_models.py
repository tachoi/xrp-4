#!/usr/bin/env python
"""Train HMM and XGB models for trading system.

This script trains:
1. Multi-HMM (Fast 3m + Mid 15m) with optimal data periods
2. XGB Gate model using backtest trade results

Usage:
    # Train both HMM and XGB
    python scripts/train_models.py

    # Train only HMM
    python scripts/train_models.py --hmm-only

    # Train only XGB
    python scripts/train_models.py --xgb-only
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from xrp4.regime.multi_hmm_manager import MultiHMMManager
from xrp4.features.hmm_features import (
    build_fast_hmm_features_v2,
    build_mid_hmm_features_v2,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================================
# Binance API Client
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
                print(f"\r  Progress: {total_fetched:,}/{limit:,} ({pct:.0f}%)", end="", flush=True)

            if remaining > 0 and len(data) == fetch_limit:
                current_end_time = data[0][0] - 1
                time.sleep(0.05)
            else:
                break

        if show_progress and limit > 1000:
            print()

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
# HMM Training
# ============================================================================

def train_hmm(
    symbol: str = "XRPUSDT",
    fast_weeks: int = 3,
    mid_months: int = 2,
    checkpoint_dir: Optional[Path] = None,
) -> Tuple[MultiHMMManager, str]:
    """Train Multi-HMM with optimal data periods.

    Args:
        symbol: Trading symbol
        fast_weeks: Weeks of data for Fast HMM (3m)
        mid_months: Months of data for Mid HMM (15m)
        checkpoint_dir: Directory to save checkpoints

    Returns:
        Tuple of (trained MultiHMMManager, run_id)
    """
    logger.info("=" * 70)
    logger.info("TRAINING MULTI-HMM")
    logger.info("=" * 70)

    client = BinanceClient()

    # Calculate bar counts
    # Fast HMM: 3 weeks of 3m data
    fast_bars = fast_weeks * 7 * 24 * 20  # 20 bars per hour for 3m
    # Mid HMM: 2 months of 15m data
    mid_bars = mid_months * 30 * 24 * 4  # 4 bars per hour for 15m

    logger.info(f"Data requirements:")
    logger.info(f"  Fast HMM (3m): {fast_weeks} weeks = ~{fast_bars:,} bars")
    logger.info(f"  Mid HMM (15m): {mid_months} months = ~{mid_bars:,} bars")

    # Fetch 3m data
    logger.info(f"\nFetching 3m data ({fast_bars:,} bars)...")
    df_3m = client.get_klines(symbol, "3m", limit=fast_bars, show_progress=True)
    logger.info(f"  -> Received {len(df_3m):,} bars")
    logger.info(f"  -> Range: {df_3m['timestamp'].iloc[0]} to {df_3m['timestamp'].iloc[-1]}")

    # Fetch 15m data
    logger.info(f"\nFetching 15m data ({mid_bars:,} bars)...")
    df_15m = client.get_klines(symbol, "15m", limit=mid_bars, show_progress=True)
    logger.info(f"  -> Received {len(df_15m):,} bars")
    logger.info(f"  -> Range: {df_15m['timestamp'].iloc[0]} to {df_15m['timestamp'].iloc[-1]}")

    # Load feature config
    config_path = Path(__file__).parent.parent / "configs" / "hmm_features.yaml"
    import yaml
    if config_path.exists():
        with open(config_path, "r") as f:
            feature_config = yaml.safe_load(f)
        fast_feature_names = feature_config["hmm"]["fast_3m"]["features"]
        mid_feature_names = feature_config["hmm"]["mid_15m"]["features"]
    else:
        fast_feature_names = [
            "ret_3m", "ewm_ret_15m", "atr_pct_15m", "ewm_std_ret_15m",
            "bb_width_15m", "body_ratio", "upper_wick_ratio", "lower_wick_ratio", "range_pct"
        ]
        mid_feature_names = [
            "ret_15m", "ewm_ret_15m", "ret_1h", "ewm_ret_1h",
            "atr_pct_15m", "ewm_std_ret_15m", "atr_pct_1h", "ewm_std_ret_1h",
            "bb_width_15m", "vol_z_15m", "price_z_from_ema_1h"
        ]

    logger.info(f"\nBuilding features...")
    logger.info(f"  Fast HMM features: {fast_feature_names}")
    logger.info(f"  Mid HMM features: {mid_feature_names}")

    # Build features
    fast_features, fast_timestamps = build_fast_hmm_features_v2(
        df_3m, fast_feature_names
    )
    logger.info(f"  Fast features shape: {fast_features.shape}")

    # Use OHLCV only for mid features to avoid column conflicts
    df_15m_ohlcv = df_15m[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    mid_features, mid_timestamps = build_mid_hmm_features_v2(
        df_15m_ohlcv, mid_feature_names
    )
    logger.info(f"  Mid features shape: {mid_features.shape}")

    # Initialize MultiHMMManager
    if checkpoint_dir is None:
        checkpoint_dir = Path(__file__).parent.parent / "checkpoints" / "hmm"

    policy_config_path = Path(__file__).parent.parent / "configs" / "hmm_gate_policy.yaml"
    if policy_config_path.exists():
        multi_hmm = MultiHMMManager.from_config_file(
            policy_config_path,
            checkpoint_dir=checkpoint_dir
        )
    else:
        multi_hmm = MultiHMMManager(checkpoint_dir=checkpoint_dir)

    # Train
    logger.info(f"\nTraining Multi-HMM...")
    logger.info(f"  Fast HMM: {len(fast_features):,} samples")
    logger.info(f"  Mid HMM: {len(mid_features):,} samples")

    multi_hmm.train(
        fast_features=fast_features,
        fast_feature_names=fast_feature_names,
        mid_features=mid_features,
        mid_feature_names=mid_feature_names,
        fast_timestamps=fast_timestamps,
        mid_timestamps=mid_timestamps,
    )

    # Get run_id from the manager
    run_id = multi_hmm.run_id

    # Save checkpoints explicitly
    multi_hmm.save_checkpoints(run_id)

    logger.info(f"\nMulti-HMM training complete!")
    logger.info(f"  Run ID: {run_id}")
    logger.info(f"  Checkpoint dir: {checkpoint_dir}")

    # Verify checkpoint files
    checkpoint_files = list(checkpoint_dir.glob(f"*_{run_id}*"))
    logger.info(f"  Saved files: {len(checkpoint_files)}")
    for f in checkpoint_files:
        logger.info(f"    - {f.name}")

    return multi_hmm, run_id


# ============================================================================
# XGB Training
# ============================================================================

def train_xgb(
    symbol: str = "XRPUSDT",
    months: int = 3,
    leverage: float = 5.0,
    output_dir: Optional[Path] = None,
    from_csv: bool = False,
    csv_path: Optional[Path] = None,
) -> Optional[Path]:
    """Train XGB Gate model using backtest trade results.

    Args:
        symbol: Trading symbol
        months: Months of data for backtest
        leverage: Leverage for backtest
        output_dir: Directory to save XGB model
        from_csv: If True, train from existing CSV (recommended)
        csv_path: Path to CSV file (if from_csv=True)

    Returns:
        Path to saved XGB model or None if training failed
    """
    logger.info("=" * 70)
    logger.info("TRAINING XGB GATE MODEL")
    logger.info("=" * 70)

    # Check if XGB training module exists
    xgb_trainer_path = Path(__file__).parent.parent / "src" / "xrp4" / "gate" / "xgb_trainer.py"

    if not xgb_trainer_path.exists():
        logger.warning("XGB trainer module not found.")
        return None

    # If XGB trainer exists, use it
    try:
        from xrp4.gate.xgb_trainer import XGBTrainer

        trainer = XGBTrainer()

        if from_csv:
            # Train from CSV - uses real Multi-HMM system trades
            logger.info("Training from CSV data (real Multi-HMM trades)...")
            model_path = trainer.train_from_csv(csv_path=csv_path)
        else:
            # Train using internal strategy (less accurate)
            logger.info("Training using internal strategy...")
            model_path = trainer.train(months=months, leverage=leverage)

        logger.info(f"\nXGB model trained and saved to: {model_path}")
        return model_path

    except FileNotFoundError as e:
        logger.error(f"CSV file not found: {e}")
        logger.info("\nTo generate CSV data, run backtest first:")
        logger.info("  python scripts/backtest_binance.py --months 6 --use-checkpoint")
        return None
    except Exception as e:
        logger.error(f"XGB training failed: {e}")
        return None


# ============================================================================
# Save Latest Run ID
# ============================================================================

def save_latest_run_id(run_id: str, checkpoint_dir: Path):
    """Save the latest run ID for easy loading."""
    latest_file = checkpoint_dir / "latest_run_id.txt"
    with open(latest_file, "w") as f:
        f.write(run_id)
    logger.info(f"Saved latest run ID: {run_id}")


def load_latest_run_id(checkpoint_dir: Path) -> Optional[str]:
    """Load the latest run ID."""
    latest_file = checkpoint_dir / "latest_run_id.txt"
    if latest_file.exists():
        with open(latest_file, "r") as f:
            return f.read().strip()
    return None


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train HMM and XGB models")
    parser.add_argument("--symbol", default="XRPUSDT", help="Trading symbol")
    parser.add_argument("--hmm-only", action="store_true", help="Train only HMM")
    parser.add_argument("--xgb-only", action="store_true", help="Train only XGB")
    parser.add_argument("--fast-weeks", type=int, default=3, help="Weeks of data for Fast HMM")
    parser.add_argument("--mid-months", type=int, default=2, help="Months of data for Mid HMM")
    parser.add_argument("--xgb-months", type=int, default=3, help="Months of data for XGB training")
    parser.add_argument("--leverage", type=float, default=5.0, help="Leverage for XGB backtest")
    parser.add_argument("--from-csv", action="store_true",
                        help="Train XGB from backtest CSV (recommended)")
    parser.add_argument("--csv-path", type=str, default=None,
                        help="Path to CSV file for XGB training")
    args = parser.parse_args()

    checkpoint_dir = Path(__file__).parent.parent / "checkpoints" / "hmm"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_hmm_flag = not args.xgb_only
    train_xgb_flag = not args.hmm_only

    logger.info("=" * 70)
    logger.info("MODEL TRAINING")
    logger.info("=" * 70)
    logger.info(f"Symbol: {args.symbol}")
    logger.info(f"Train HMM: {train_hmm_flag}")
    logger.info(f"Train XGB: {train_xgb_flag}")
    logger.info("")

    results = {}

    # Train HMM
    if train_hmm_flag:
        try:
            multi_hmm, run_id = train_hmm(
                symbol=args.symbol,
                fast_weeks=args.fast_weeks,
                mid_months=args.mid_months,
                checkpoint_dir=checkpoint_dir,
            )
            save_latest_run_id(run_id, checkpoint_dir)
            results["hmm"] = {
                "status": "success",
                "run_id": run_id,
                "checkpoint_dir": str(checkpoint_dir),
            }
        except Exception as e:
            logger.error(f"HMM training failed: {e}")
            results["hmm"] = {"status": "failed", "error": str(e)}

    # Train XGB
    if train_xgb_flag:
        try:
            csv_path = Path(args.csv_path) if args.csv_path else None
            xgb_path = train_xgb(
                symbol=args.symbol,
                months=args.xgb_months,
                leverage=args.leverage,
                from_csv=args.from_csv,
                csv_path=csv_path,
            )
            results["xgb"] = {
                "status": "success" if xgb_path else "skipped",
                "model_path": str(xgb_path) if xgb_path else None,
            }
        except Exception as e:
            logger.error(f"XGB training failed: {e}")
            results["xgb"] = {"status": "failed", "error": str(e)}

    # Summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 70)

    for model, result in results.items():
        status = result.get("status", "unknown")
        if status == "success":
            logger.info(f"{model.upper()}: SUCCESS")
            if "run_id" in result:
                logger.info(f"  Run ID: {result['run_id']}")
            if "model_path" in result:
                logger.info(f"  Model path: {result['model_path']}")
        elif status == "skipped":
            logger.info(f"{model.upper()}: SKIPPED")
        else:
            logger.info(f"{model.upper()}: FAILED - {result.get('error', 'unknown error')}")

    # Save results
    results_file = checkpoint_dir / f"training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\nResults saved to: {results_file}")

    logger.info("")
    logger.info("To use trained models:")
    logger.info("  Paper trading: python scripts/paper_trading.py")
    logger.info("  Backtest: python scripts/backtest_binance.py --use-checkpoint")


if __name__ == "__main__":
    main()
