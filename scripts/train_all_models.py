#!/usr/bin/env python
"""Comprehensive Model Training Script for XRP-4 Trading System.

This script trains all models in the correct sequence:
1. HMM Models (Fast 3m + Mid 15m) with composite scoring
2. Backtest to generate trade data for XGB
3. XGB Gate model

Usage:
    # Train all models
    python scripts/train_all_models.py

    # Train only HMM
    python scripts/train_all_models.py --hmm-only

    # Train only XGB (requires existing backtest data)
    python scripts/train_all_models.py --xgb-only

    # Full training with longer data period
    python scripts/train_all_models.py --fast-weeks 4 --mid-months 3 --backtest-months 6
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
        show_progress: bool = False,
    ) -> pd.DataFrame:
        """Fetch klines with pagination support."""
        all_data = []
        remaining = limit
        current_end_time = end_time
        total_fetched = 0

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
# Phase 1: HMM Training
# ============================================================================

def train_hmm(
    symbol: str = "XRPUSDT",
    fast_weeks: int = 3,
    mid_months: int = 2,
    checkpoint_dir: Optional[Path] = None,
) -> Tuple[MultiHMMManager, str]:
    """Train Multi-HMM with optimal data periods.

    Uses the fixed composite scoring for state labeling.

    Args:
        symbol: Trading symbol
        fast_weeks: Weeks of data for Fast HMM (3m)
        mid_months: Months of data for Mid HMM (15m)
        checkpoint_dir: Directory to save checkpoints

    Returns:
        Tuple of (trained MultiHMMManager, run_id)
    """
    logger.info("=" * 70)
    logger.info("PHASE 1: TRAINING MULTI-HMM")
    logger.info("=" * 70)

    client = BinanceClient()

    # Calculate bar counts
    fast_bars = fast_weeks * 7 * 24 * 20  # 20 bars per hour for 3m
    mid_bars = mid_months * 30 * 24 * 4   # 4 bars per hour for 15m

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

    df_15m_ohlcv = df_15m[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    mid_features, mid_timestamps = build_mid_hmm_features_v2(
        df_15m_ohlcv, mid_feature_names
    )
    logger.info(f"  Mid features shape: {mid_features.shape}")

    # Initialize MultiHMMManager
    if checkpoint_dir is None:
        checkpoint_dir = Path(__file__).parent.parent / "checkpoints" / "hmm"

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    policy_config_path = Path(__file__).parent.parent / "configs" / "hmm_gate_policy.yaml"
    if policy_config_path.exists():
        multi_hmm = MultiHMMManager.from_config_file(
            policy_config_path,
            checkpoint_dir=checkpoint_dir
        )
    else:
        multi_hmm = MultiHMMManager(checkpoint_dir=checkpoint_dir)

    # Train
    logger.info(f"\nTraining Multi-HMM with composite scoring...")
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

    run_id = multi_hmm.run_id
    multi_hmm.save_checkpoints(run_id)

    # Save latest_run_id
    latest_file = checkpoint_dir / "latest_run_id.txt"
    with open(latest_file, "w") as f:
        f.write(run_id)

    # Print training report
    logger.info(f"\n--- HMM Training Report ---")
    report = multi_hmm.get_training_report()

    fast_report = report.get("fast_hmm", {})
    logger.info(f"Fast HMM (3m):")
    logger.info(f"  States: {fast_report.get('n_states', 'N/A')}")
    logger.info(f"  Log-likelihood: {fast_report.get('log_likelihood', 'N/A'):.2f}")
    logger.info(f"  State labels: {fast_report.get('state_labels', 'N/A')}")

    mid_report = report.get("mid_hmm", {})
    logger.info(f"Mid HMM (15m):")
    logger.info(f"  States: {mid_report.get('n_states', 'N/A')}")
    logger.info(f"  Log-likelihood: {mid_report.get('log_likelihood', 'N/A'):.2f}")
    logger.info(f"  State labels: {mid_report.get('state_labels', 'N/A')}")

    logger.info(f"\nMulti-HMM training complete!")
    logger.info(f"  Run ID: {run_id}")
    logger.info(f"  Checkpoint dir: {checkpoint_dir}")

    return multi_hmm, run_id


# ============================================================================
# Phase 2: Backtest for XGB Training Data
# ============================================================================

def run_backtest_for_xgb(
    symbol: str = "XRPUSDT",
    months: int = 3,
    checkpoint_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Run backtest to generate trade data for XGB training.

    Args:
        symbol: Trading symbol
        months: Months of data for backtest
        checkpoint_dir: Directory with HMM checkpoints
        output_dir: Directory for output CSV

    Returns:
        Path to trades CSV or None if failed
    """
    logger.info("=" * 70)
    logger.info("PHASE 2: BACKTEST FOR XGB TRAINING DATA")
    logger.info("=" * 70)

    if checkpoint_dir is None:
        checkpoint_dir = Path(__file__).parent.parent / "checkpoints" / "hmm"

    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "outputs" / "xgb_gate"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load Multi-HMM
    latest_file = checkpoint_dir / "latest_run_id.txt"
    if not latest_file.exists():
        logger.error("No HMM checkpoint found! Train HMM first.")
        return None

    with open(latest_file, "r") as f:
        run_id = f.read().strip()

    logger.info(f"Loading Multi-HMM (run_id: {run_id})...")

    policy_config_path = Path(__file__).parent.parent / "configs" / "hmm_gate_policy.yaml"
    if policy_config_path.exists():
        multi_hmm = MultiHMMManager.from_config_file(
            policy_config_path,
            checkpoint_dir=checkpoint_dir
        )
    else:
        multi_hmm = MultiHMMManager(checkpoint_dir=checkpoint_dir)

    multi_hmm.load_checkpoints(run_id)

    # Fetch data for backtest
    client = BinanceClient()
    bars_3m = months * 30 * 24 * 20  # 20 bars per hour for 3m
    bars_15m = months * 30 * 24 * 4  # 4 bars per hour for 15m
    bars_1m = months * 30 * 24 * 60  # 60 bars per hour for 1m (for entry timing)

    logger.info(f"\nFetching backtest data ({months} months)...")
    df_3m = client.get_klines(symbol, "3m", limit=bars_3m, show_progress=True)
    df_15m = client.get_klines(symbol, "15m", limit=bars_15m, show_progress=True)
    df_1m = client.get_klines(symbol, "1m", limit=bars_1m, show_progress=True)

    logger.info(f"  3m bars: {len(df_3m):,}")
    logger.info(f"  15m bars: {len(df_15m):,}")
    logger.info(f"  1m bars: {len(df_1m):,}")

    # Prepare 1m data for entry timing analysis
    df_1m["timestamp"] = pd.to_datetime(df_1m["timestamp"])
    df_1m.set_index("timestamp", inplace=True)
    df_1m["ret_1m"] = df_1m["close"].pct_change()
    df_1m["spike"] = df_1m["ret_1m"].abs() > 0.005  # >0.5% move in 1m = spike

    # Import backtest components
    from xrp4.regime.confirm import RegimeConfirmLayer, ConfirmConfig
    from xrp4.core.types import ConfirmContext, MarketContext, PositionState
    from xrp4.core.fsm import TradingFSM
    from xrp4.core.decision_engine import DecisionEngine

    # Build features for regime detection
    config_path = Path(__file__).parent.parent / "configs" / "hmm_features.yaml"
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

    logger.info("Building features for backtest...")
    fast_features, fast_timestamps = build_fast_hmm_features_v2(df_3m.copy(), fast_feature_names)
    df_15m_ohlcv = df_15m[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    mid_features, mid_timestamps = build_mid_hmm_features_v2(df_15m_ohlcv, mid_feature_names)

    logger.info(f"  Fast features: {fast_features.shape}")
    logger.info(f"  Mid features: {mid_features.shape}")

    # Initialize components
    # NOTE: Disable XGB during backtest to collect raw trade candidates for training
    confirm_layer = RegimeConfirmLayer(ConfirmConfig())
    fsm = TradingFSM()
    decision_engine = DecisionEngine({
        "DECISION": {"XGB_ENABLED": False},  # Disable XGB during data collection
        "RISK": {},
        "FSM": {},
    })

    # Run simplified backtest
    logger.info("\nRunning backtest...")
    trades = []
    pos = PositionState(side="FLAT", entry_price=0, size=0, bars_held_3m=0)
    fsm_state = None
    engine_state = None
    entry_row = None
    entry_idx = None

    # Create regime predictions
    regime_preds = []
    for i in range(len(fast_features)):
        try:
            # Find matching 15m index
            fast_ts = fast_timestamps[i]
            mid_idx = np.searchsorted(mid_timestamps, fast_ts)
            if mid_idx >= len(mid_features):
                mid_idx = len(mid_features) - 1

            packet = multi_hmm.predict(
                fast_features[i:i+1],
                mid_features[mid_idx:mid_idx+1]
            )
            regime_preds.append(packet.label_fused.value)
        except Exception:
            regime_preds.append("UNKNOWN")

    # Add features to df_3m
    df_3m = df_3m.iloc[-len(fast_features):].reset_index(drop=True)
    df_3m["regime"] = regime_preds

    # Calculate features needed for FSM and XGB
    # Returns (XGB needs ret, ret_2, ret_5; FSM needs ret_3m)
    df_3m["ret"] = df_3m["close"].pct_change().fillna(0)
    df_3m["ret_3m"] = df_3m["ret"]  # Alias for FSM
    df_3m["ret_2"] = df_3m["close"].pct_change(2).fillna(0)
    df_3m["ret_5"] = df_3m["close"].pct_change(5).fillna(0)

    # EMAs for FSM (required names: ema_fast_3m, ema_slow_3m)
    df_3m["ema_fast_3m"] = df_3m["close"].ewm(span=20, adjust=False).mean()
    df_3m["ema_slow_3m"] = df_3m["close"].ewm(span=50, adjust=False).mean()
    df_3m["ema_20"] = df_3m["ema_fast_3m"]  # Alias for XGB
    df_3m["ema_50"] = df_3m["ema_slow_3m"]  # Alias for XGB

    # ATR for FSM (required: atr_3m)
    df_3m["tr"] = np.maximum(
        df_3m["high"] - df_3m["low"],
        np.maximum(
            abs(df_3m["high"] - df_3m["close"].shift(1)),
            abs(df_3m["low"] - df_3m["close"].shift(1))
        )
    )
    df_3m["atr_3m"] = df_3m["tr"].rolling(14).mean().fillna(df_3m["tr"])

    # EMA-based features
    df_3m["ema_diff"] = (df_3m["ema_fast_3m"] - df_3m["ema_slow_3m"]) / df_3m["close"]
    df_3m["price_to_ema20"] = (df_3m["close"] - df_3m["ema_fast_3m"]) / df_3m["ema_fast_3m"]
    df_3m["price_to_ema50"] = (df_3m["close"] - df_3m["ema_slow_3m"]) / df_3m["ema_slow_3m"]

    # EMA slope (required for FSM: ema_slope_15m)
    df_3m["ema_slope"] = df_3m["ema_fast_3m"].pct_change(5).fillna(0)
    df_3m["ema_slope_15m"] = df_3m["ema_slope"]  # FSM uses this

    # Volatility
    df_3m["volatility"] = df_3m["ret"].rolling(20).std().fillna(0.005)
    df_3m["vol"] = df_3m["volatility"]  # Alias
    df_3m["range_pct"] = (df_3m["high"] - df_3m["low"]) / df_3m["close"]

    # Volume
    df_3m["volume_ma"] = df_3m["volume"].rolling(20).mean()
    df_3m["volume_ratio"] = (df_3m["volume"] / df_3m["volume_ma"].replace(0, 1)).fillna(1.0)

    # RSI
    delta = df_3m["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, 1e-10)
    df_3m["rsi"] = 100 - (100 / (1 + rs))
    df_3m["rsi_3m"] = df_3m["rsi"]

    df_3m = df_3m.fillna(0)

    # === 1m Entry Timing Helper ===
    def analyze_1m_entry(bar_3m_ts, side, df_1m):
        """Analyze 1m data within a 3m bar for optimal entry timing.

        Args:
            bar_3m_ts: Timestamp of the 3m bar
            side: "LONG" or "SHORT"
            df_1m: 1m dataframe (indexed by timestamp)

        Returns:
            dict with:
                - entry_price: Optimal entry price within the 3m bar
                - has_spike: Whether a sudden spike/drop occurred
                - spike_direction: "UP", "DOWN", or None
                - max_ret_1m: Maximum 1m return (for spike detection)
                - min_ret_1m: Minimum 1m return (for drop detection)
                - should_skip: Whether to skip this trade due to adverse conditions
        """
        # Get the 3 1m bars within this 3m bar
        bar_start = bar_3m_ts - pd.Timedelta(minutes=2)  # 3m bar covers [ts-2m, ts]
        bar_end = bar_3m_ts

        mask = (df_1m.index > bar_start) & (df_1m.index <= bar_end)
        bars_1m = df_1m.loc[mask]

        if len(bars_1m) == 0:
            return {
                "entry_price": None,
                "has_spike": False,
                "spike_direction": None,
                "max_ret_1m": 0,
                "min_ret_1m": 0,
                "should_skip": False,
            }

        # Calculate spike indicators
        max_ret = bars_1m["ret_1m"].max()
        min_ret = bars_1m["ret_1m"].min()
        has_spike = bars_1m["spike"].any()

        # Determine spike direction
        spike_direction = None
        if max_ret > 0.005:
            spike_direction = "UP"
        elif min_ret < -0.005:
            spike_direction = "DOWN"

        # Determine if we should skip this trade
        # Skip LONG if there was a spike UP (chasing), or SHORT if spike DOWN
        should_skip = False
        if side == "LONG" and spike_direction == "UP" and max_ret > 0.01:
            should_skip = True  # Don't chase after big up move
        elif side == "SHORT" and spike_direction == "DOWN" and min_ret < -0.01:
            should_skip = True  # Don't chase after big down move

        # Find optimal entry price within the 1m bars
        if side == "LONG":
            # For LONG, try to enter at lower price (dip within the bar)
            entry_price = bars_1m["low"].min()
        else:
            # For SHORT, try to enter at higher price (bounce within the bar)
            entry_price = bars_1m["high"].max()

        return {
            "entry_price": entry_price,
            "has_spike": has_spike,
            "spike_direction": spike_direction,
            "max_ret_1m": max_ret,
            "min_ret_1m": min_ret,
            "should_skip": should_skip,
        }

    # Debug counters
    regime_counts = {}
    signal_reasons = {}
    entry_1m_stats = {"total": 0, "skipped": 0, "improved": 0}

    # Simple backtest loop
    warmup = 100
    for i in range(warmup, len(df_3m)):
        row = df_3m.iloc[i]
        price = row["close"]
        regime = row["regime"]

        # Count regimes
        regime_counts[regime] = regime_counts.get(regime, 0) + 1

        # Create confirm context (simplified)
        confirm_ctx = ConfirmContext(
            regime_raw=regime,
            regime_confirmed=regime,
            confirm_reason="PASSTHROUGH",
            confirm_metrics={},
        )

        # Create market context
        market_ctx = MarketContext(
            symbol=symbol,
            ts=int(row["timestamp"].timestamp() * 1000),
            price=price,
            row_3m=row.to_dict(),
            row_15m={"ema_slope_15m": row.get("ema_slope", 0)},
            zone={"support": 0, "resistance": 0, "strength": 0,
                  "dist_to_support": 999, "dist_to_resistance": 999},
        )

        # FSM step
        cand, fsm_state = fsm.step(market_ctx, confirm_ctx, pos, fsm_state)

        # Track signal reasons
        signal_reasons[cand.reason] = signal_reasons.get(cand.reason, 0) + 1

        # Decision
        decision, engine_state = decision_engine.decide(
            market_ctx, confirm_ctx, pos, cand, engine_state
        )

        # Position management with 1m entry timing
        if decision.action == "OPEN_LONG":
            entry_1m_stats["total"] += 1
            # Analyze 1m data for optimal entry
            entry_analysis = analyze_1m_entry(row["timestamp"], "LONG", df_1m)

            if entry_analysis["should_skip"]:
                # Skip this trade due to adverse 1m conditions
                entry_1m_stats["skipped"] += 1
                continue

            # Use optimized entry price if available
            if entry_analysis["entry_price"] is not None:
                actual_entry = entry_analysis["entry_price"]
                if actual_entry < price:
                    entry_1m_stats["improved"] += 1
            else:
                actual_entry = price

            pos = PositionState(side="LONG", entry_price=actual_entry, size=decision.size, bars_held_3m=0)
            entry_row = row.copy()
            entry_row["entry_1m_analysis"] = entry_analysis
            entry_idx = i

        elif decision.action == "OPEN_SHORT":
            entry_1m_stats["total"] += 1
            # Analyze 1m data for optimal entry
            entry_analysis = analyze_1m_entry(row["timestamp"], "SHORT", df_1m)

            if entry_analysis["should_skip"]:
                # Skip this trade due to adverse 1m conditions
                entry_1m_stats["skipped"] += 1
                continue

            # Use optimized entry price if available
            if entry_analysis["entry_price"] is not None:
                actual_entry = entry_analysis["entry_price"]
                if actual_entry > price:
                    entry_1m_stats["improved"] += 1
            else:
                actual_entry = price

            pos = PositionState(side="SHORT", entry_price=actual_entry, size=decision.size, bars_held_3m=0)
            entry_row = row.copy()
            entry_row["entry_1m_analysis"] = entry_analysis
            entry_idx = i
        elif decision.action == "CLOSE" and pos.side != "FLAT":
            # Calculate PnL
            if pos.side == "LONG":
                pnl_pct = (price - pos.entry_price) / pos.entry_price
            else:
                pnl_pct = (pos.entry_price - price) / pos.entry_price

            pnl = pnl_pct * pos.size

            trade = {
                "entry_idx": entry_idx,
                "exit_idx": i,
                "side": pos.side,
                "entry_price": pos.entry_price,
                "exit_price": price,
                "pnl": pnl,
                "pnl_pct": pnl_pct * 100,
                "bars_held": pos.bars_held_3m,
                "regime": entry_row["regime"] if entry_row is not None else "UNKNOWN",
                # Entry features for XGB
                "ret": entry_row["ret"] if entry_row is not None else 0,
                "ret_2": entry_row["ret_2"] if entry_row is not None else 0,
                "ret_5": entry_row["ret_5"] if entry_row is not None else 0,
                "ema_diff": entry_row["ema_diff"] if entry_row is not None else 0,
                "price_to_ema20": entry_row["price_to_ema20"] if entry_row is not None else 0,
                "price_to_ema50": entry_row["price_to_ema50"] if entry_row is not None else 0,
                "volatility": entry_row["volatility"] if entry_row is not None else 0,
                "range_pct": entry_row["range_pct"] if entry_row is not None else 0,
                "volume_ratio": entry_row["volume_ratio"] if entry_row is not None else 0,
                "rsi": entry_row["rsi"] if entry_row is not None else 50,
                "ema_slope": entry_row["ema_slope"] if entry_row is not None else 0,
            }
            trades.append(trade)

            pos = PositionState(side="FLAT", entry_price=0, size=0, bars_held_3m=0)
            entry_row = None
            entry_idx = None
        elif pos.side != "FLAT":
            pos = PositionState(
                side=pos.side,
                entry_price=pos.entry_price,
                size=pos.size,
                bars_held_3m=pos.bars_held_3m + 1,
            )

        if i % 5000 == 0:
            logger.info(f"  Progress: {i:,}/{len(df_3m):,} ({i/len(df_3m)*100:.0f}%)")

    trades_df = pd.DataFrame(trades)
    csv_path = output_dir / "backtest_trades.csv"
    trades_df.to_csv(csv_path, index=False)

    # Debug output
    logger.info(f"\nRegime distribution:")
    for reg, cnt in sorted(regime_counts.items(), key=lambda x: -x[1]):
        logger.info(f"  {reg}: {cnt} bars ({cnt/len(df_3m)*100:.1f}%)")

    logger.info(f"\nSignal reasons (top 10):")
    for reason, cnt in sorted(signal_reasons.items(), key=lambda x: -x[1])[:10]:
        logger.info(f"  {reason}: {cnt}")

    logger.info(f"\n1m Entry Timing Statistics:")
    logger.info(f"  Total entry attempts: {entry_1m_stats['total']}")
    logger.info(f"  Skipped (chasing): {entry_1m_stats['skipped']} ({entry_1m_stats['skipped']/max(1,entry_1m_stats['total'])*100:.1f}%)")
    logger.info(f"  Improved entry: {entry_1m_stats['improved']} ({entry_1m_stats['improved']/max(1,entry_1m_stats['total']-entry_1m_stats['skipped'])*100:.1f}%)")

    logger.info(f"\nBacktest complete!")
    logger.info(f"  Total trades: {len(trades_df)}")
    if len(trades_df) > 0:
        win_rate = (trades_df["pnl"] > 0).mean() * 100
        total_pnl = trades_df["pnl"].sum()
        logger.info(f"  Win rate: {win_rate:.1f}%")
        logger.info(f"  Total PnL: ${total_pnl:.2f}")
    logger.info(f"  Saved to: {csv_path}")

    return csv_path


# ============================================================================
# Phase 3: XGB Training
# ============================================================================

def train_xgb(
    csv_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    test_ratio: float = 0.2,
) -> Optional[Path]:
    """Train XGB Gate model from backtest trade data.

    Args:
        csv_path: Path to trades CSV
        output_dir: Directory to save XGB model
        test_ratio: Ratio of data for testing

    Returns:
        Path to saved XGB model or None if failed
    """
    logger.info("=" * 70)
    logger.info("PHASE 3: TRAINING XGB GATE MODEL")
    logger.info("=" * 70)

    try:
        import xgboost as xgb
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    except ImportError:
        logger.error("xgboost or sklearn not installed!")
        return None

    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "outputs" / "xgb_gate"

    output_dir.mkdir(parents=True, exist_ok=True)

    if csv_path is None:
        csv_path = output_dir / "backtest_trades.csv"

    if not csv_path.exists():
        logger.error(f"Trades CSV not found: {csv_path}")
        return None

    # Load trades
    trades = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(trades)} trades from {csv_path}")

    if len(trades) < 50:
        logger.error("Not enough trades for XGB training!")
        return None

    # Feature columns
    feature_cols = [
        "ret", "ret_2", "ret_5",
        "ema_diff", "price_to_ema20", "price_to_ema50",
        "volatility", "range_pct", "volume_ratio",
        "rsi", "ema_slope",
    ]

    # Add side and regime features
    trades["side_num"] = trades["side"].apply(lambda x: 1 if x == "LONG" else -1)
    trades["regime_trend_up"] = (trades["regime"] == "TREND_UP").astype(int)
    trades["regime_trend_down"] = (trades["regime"] == "TREND_DOWN").astype(int)
    trades["target"] = (trades["pnl"] > 0).astype(int)

    all_features = feature_cols + ["side_num", "regime_trend_up", "regime_trend_down"]

    # Split data (time-based)
    split_idx = int(len(trades) * (1 - test_ratio))
    train_df = trades.iloc[:split_idx]
    test_df = trades.iloc[split_idx:]

    X_train = train_df[all_features].fillna(0).values
    y_train = train_df["target"].values
    X_test = test_df[all_features].fillna(0).values
    y_test = test_df["target"].values

    logger.info(f"\nTrain: {len(X_train)} samples, {y_train.sum()} wins ({y_train.mean()*100:.1f}%)")
    logger.info(f"Test: {len(X_test)} samples, {y_test.sum()} wins ({y_test.mean()*100:.1f}%)")

    # Train XGBoost
    logger.info("\nTraining XGBoost...")

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "max_depth": 4,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "seed": 42,
    }

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=100,
        evals=[(dtrain, "train"), (dtest, "test")],
        early_stopping_rounds=20,
        verbose_eval=20,
    )

    # Evaluate
    y_pred_proba = model.predict(dtest)
    y_pred = (y_pred_proba >= 0.5).astype(int)

    logger.info(f"\nTest Metrics:")
    logger.info(f"  Accuracy: {accuracy_score(y_test, y_pred)*100:.1f}%")
    logger.info(f"  Precision: {precision_score(y_test, y_pred, zero_division=0)*100:.1f}%")
    logger.info(f"  Recall: {recall_score(y_test, y_pred, zero_division=0)*100:.1f}%")
    logger.info(f"  F1 Score: {f1_score(y_test, y_pred, zero_division=0)*100:.1f}%")
    try:
        logger.info(f"  ROC AUC: {roc_auc_score(y_test, y_pred_proba)*100:.1f}%")
    except:
        pass

    # Feature importance
    importance = model.get_score(importance_type="gain")
    logger.info("\nFeature Importance (top 10):")
    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
    for fname, score in sorted_imp:
        idx = int(fname.replace("f", ""))
        if idx < len(all_features):
            logger.info(f"  {all_features[idx]}: {score:.2f}")

    # Trading simulation with XGB filter
    logger.info("\n--- Trading Simulation with XGB Filter ---")
    test_df["p_win"] = y_pred_proba

    baseline = {
        "n_trades": len(test_df),
        "win_rate": test_df["target"].mean() * 100,
        "total_pnl": test_df["pnl"].sum(),
    }

    logger.info(f"\n{'Threshold':<12} {'Trades':<10} {'WinRate':<10} {'TotalPnL':<12} {'Filter%':<10}")
    logger.info("-" * 60)
    logger.info(f"{'No Filter':<12} {baseline['n_trades']:<10} {baseline['win_rate']:.1f}%{'':<5} ${baseline['total_pnl']:.2f}")

    best_threshold = 0.5
    best_pnl = baseline["total_pnl"]

    for threshold in [0.40, 0.45, 0.50, 0.55, 0.60]:
        filtered = test_df[test_df["p_win"] >= threshold]
        if len(filtered) == 0:
            continue
        win_rate = filtered["target"].mean() * 100
        total_pnl = filtered["pnl"].sum()
        filter_pct = (1 - len(filtered) / len(test_df)) * 100
        logger.info(f"{threshold:<12.2f} {len(filtered):<10} {win_rate:.1f}%{'':<5} ${total_pnl:.2f}{'':<7} {filter_pct:.0f}%")

        if total_pnl > best_pnl:
            best_pnl = total_pnl
            best_threshold = threshold

    logger.info(f"\nOptimal threshold: {best_threshold}")

    # Save model
    model_path = output_dir / "xgb_model.json"
    model.save_model(str(model_path))
    logger.info(f"\nModel saved to: {model_path}")

    # Save results
    results = {
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "baseline": baseline,
        "best_threshold": best_threshold,
        "feature_cols": all_features,
    }

    results_path = output_dir / "xgb_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    return model_path


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train all XRP-4 models")
    parser.add_argument("--symbol", default="XRPUSDT", help="Trading symbol")
    parser.add_argument("--hmm-only", action="store_true", help="Train only HMM")
    parser.add_argument("--xgb-only", action="store_true", help="Train only XGB")
    parser.add_argument("--fast-weeks", type=int, default=3, help="Weeks of data for Fast HMM")
    parser.add_argument("--mid-months", type=int, default=2, help="Months of data for Mid HMM")
    parser.add_argument("--backtest-months", type=int, default=3, help="Months for backtest/XGB training")
    args = parser.parse_args()

    checkpoint_dir = Path(__file__).parent.parent / "checkpoints" / "hmm"
    output_dir = Path(__file__).parent.parent / "outputs" / "xgb_gate"

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_hmm_flag = not args.xgb_only
    train_xgb_flag = not args.hmm_only

    logger.info("=" * 70)
    logger.info("XRP-4 COMPREHENSIVE MODEL TRAINING")
    logger.info("=" * 70)
    logger.info(f"Symbol: {args.symbol}")
    logger.info(f"Train HMM: {train_hmm_flag}")
    logger.info(f"Train XGB: {train_xgb_flag}")
    logger.info(f"Fast HMM data: {args.fast_weeks} weeks")
    logger.info(f"Mid HMM data: {args.mid_months} months")
    logger.info(f"Backtest data: {args.backtest_months} months")
    logger.info("")

    results = {}

    # Phase 1: Train HMM
    if train_hmm_flag:
        try:
            multi_hmm, run_id = train_hmm(
                symbol=args.symbol,
                fast_weeks=args.fast_weeks,
                mid_months=args.mid_months,
                checkpoint_dir=checkpoint_dir,
            )
            results["hmm"] = {
                "status": "success",
                "run_id": run_id,
            }
        except Exception as e:
            logger.error(f"HMM training failed: {e}")
            import traceback
            traceback.print_exc()
            results["hmm"] = {"status": "failed", "error": str(e)}

    # Phase 2 & 3: Backtest and XGB
    if train_xgb_flag:
        try:
            # Phase 2: Backtest
            csv_path = run_backtest_for_xgb(
                symbol=args.symbol,
                months=args.backtest_months,
                checkpoint_dir=checkpoint_dir,
                output_dir=output_dir,
            )

            if csv_path is None:
                raise Exception("Backtest failed to generate trades")

            # Phase 3: Train XGB
            model_path = train_xgb(
                csv_path=csv_path,
                output_dir=output_dir,
            )

            results["xgb"] = {
                "status": "success" if model_path else "failed",
                "model_path": str(model_path) if model_path else None,
            }
        except Exception as e:
            logger.error(f"XGB training failed: {e}")
            import traceback
            traceback.print_exc()
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
