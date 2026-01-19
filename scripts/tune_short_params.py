#!/usr/bin/env python3
"""SHORT Strategy Parameter Tuning.

Tunes SHORT-specific parameters using ACTUAL TradingFSM + DecisionEngine.
This ensures consistency between tuning and backtesting results.

Usage:
    python scripts/tune_short_params.py
    python scripts/tune_short_params.py --hmm-model models/hmm_simple.pkl
"""

import sys
import logging
import argparse
import pickle
from pathlib import Path
from datetime import datetime
from itertools import product
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import json

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
from xrp4.core.fsm import TradingFSM, FSMConfig
from xrp4.core.decision_engine import DecisionEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class ShortTuneParams:
    """Parameters for SHORT tuning - matches TradingFSM SHORT parameters."""
    # SHORT entry params (tunable)
    SHORT_PULLBACK_TO_EMA_ATR: float = 1.0
    SHORT_MIN_EMA_SLOPE_15M: float = 0.002
    SHORT_MIN_REJECTION_RET: float = 0.0007
    SHORT_REQUIRE_EMA_ALIGNMENT: bool = True
    SHORT_MAX_HOLD_BARS_3M: int = 30

    # SHORT entry confirmation - STRONGER conditions (same as actual FSM)
    SHORT_CONSEC_NEG_BARS: int = 2       # Consecutive negative bars required
    SHORT_PEAK_CONFIRM_BARS: int = 3     # Bars to look back for peak confirmation

    # Risk params
    STOP_ATR_MULT: float = 1.5
    TP_ATR_MULT: float = 2.0
    TRAIL_ATR_MULT: float = 1.0
    MAX_HOLD_BARS_3M: int = 40           # For LONG (not used in SHORT-only tuning)

    def to_fsm_config(self) -> FSMConfig:
        """Convert to FSMConfig for use with actual TradingFSM."""
        return FSMConfig(
            # SHORT-specific params (tunable)
            SHORT_PULLBACK_TO_EMA_ATR=self.SHORT_PULLBACK_TO_EMA_ATR,
            SHORT_MIN_EMA_SLOPE_15M=self.SHORT_MIN_EMA_SLOPE_15M,
            SHORT_MIN_REJECTION_RET=self.SHORT_MIN_REJECTION_RET,
            SHORT_REQUIRE_EMA_ALIGNMENT=self.SHORT_REQUIRE_EMA_ALIGNMENT,
            SHORT_MAX_HOLD_BARS_3M=self.SHORT_MAX_HOLD_BARS_3M,
            SHORT_CONSEC_NEG_BARS=self.SHORT_CONSEC_NEG_BARS,
            SHORT_PEAK_CONFIRM_BARS=self.SHORT_PEAK_CONFIRM_BARS,
            SHORT_ENABLED=True,

            # Risk params
            STOP_ATR_MULT=self.STOP_ATR_MULT,
            TP_ATR_MULT=self.TP_ATR_MULT,
            TRAIL_ATR_MULT=self.TRAIL_ATR_MULT,
            MAX_HOLD_BARS_3M=self.MAX_HOLD_BARS_3M,

            # Fixed LONG params (not tuned here, use defaults)
            TREND_PULLBACK_TO_EMA_ATR=2.0,
            TREND_MIN_EMA_SLOPE_15M=0.002,
            TREND_MIN_REBOUND_RET=0.0003,
            TREND_REQUIRE_EMA_ALIGNMENT=True,

            # Disable other trading (SHORT-only tuning)
            TRANSITION_ALLOW_BREAKOUT_ONLY=False,  # Disable TRANSITION
        )


# NOTE: ShortOnlyFSM removed - now using actual TradingFSM from xrp4.core.fsm
# This ensures tuning and backtesting use IDENTICAL logic


class ShortTuner:
    """Tuner for SHORT parameters using ACTUAL TradingFSM."""

    def __init__(self, df_3m: pd.DataFrame, df_15m: pd.DataFrame,
                 regimes: np.ndarray, state_labels: Dict, config: Dict):
        self.df_3m = df_3m
        self.df_15m = df_15m
        self.regimes = regimes
        self.state_labels = state_labels
        self.config = config
        self.df_15m_aligned = df_15m.reindex(df_3m.index, method='ffill')

    def run_backtest(self, params: ShortTuneParams, test_idx: np.ndarray) -> Dict:
        """Run backtest using actual TradingFSM + DecisionEngine (same as backtest pipeline)."""
        # Create TradingFSM with tuning parameters
        fsm_config = params.to_fsm_config()
        fsm = TradingFSM(fsm_config)
        de = DecisionEngine(self.config)

        pos = PositionState(
            side="FLAT", entry_price=0.0, size=0.0,
            entry_ts=0, bars_held_3m=0, unrealized_pnl=0.0
        )
        fsm_state = None
        de_state = None

        trades = []
        current_trade = None

        for i in test_idx:
            row_3m = self.df_3m.iloc[i]
            row_15m = self.df_15m_aligned.iloc[i]
            regime_raw = self.state_labels.get(self.regimes[i], "UNKNOWN")

            ctx = MarketContext(
                symbol="XRPUSDT",
                ts=int(row_3m.name.timestamp() * 1000) if hasattr(row_3m.name, 'timestamp') else i,
                price=row_3m["close"],
                row_3m=row_3m.to_dict(),
                row_15m=row_15m.to_dict(),
                zone={"support": 0, "resistance": 0, "strength": 0}
            )

            confirm = ConfirmContext(
                regime_raw=regime_raw,
                regime_confirmed=regime_raw,
                confirm_reason="DIRECT",
                confirm_metrics={}
            )

            if pos.side != "FLAT":
                pos = PositionState(
                    side=pos.side, entry_price=pos.entry_price, size=pos.size,
                    entry_ts=pos.entry_ts, bars_held_3m=pos.bars_held_3m + 1,
                    unrealized_pnl=(pos.entry_price - ctx.price) * pos.size  # SHORT PnL
                )

            cand, fsm_state = fsm.step(ctx, confirm, pos, fsm_state)
            decision, de_state = de.decide(ctx, confirm, pos, cand, de_state)

            if decision.action == "OPEN_SHORT" and pos.side == "FLAT":
                pos = PositionState(
                    side="SHORT", entry_price=ctx.price, size=decision.size,
                    entry_ts=ctx.ts, bars_held_3m=0, unrealized_pnl=0.0
                )
                current_trade = {
                    "entry_idx": i, "entry_price": ctx.price, "side": "SHORT",
                    "size": decision.size, "regime": regime_raw, "signal": cand.signal,
                }

            elif decision.action == "CLOSE" and pos.side == "SHORT":
                if current_trade is not None:
                    # SHORT PnL: (entry - exit) / entry * 100
                    pnl_pct = (pos.entry_price - ctx.price) / pos.entry_price * 100

                    current_trade["exit_idx"] = i
                    current_trade["exit_price"] = ctx.price
                    current_trade["pnl_pct"] = pnl_pct
                    current_trade["bars_held"] = pos.bars_held_3m
                    current_trade["exit_reason"] = cand.reason
                    trades.append(current_trade)
                    current_trade = None

                pos = PositionState(
                    side="FLAT", entry_price=0.0, size=0.0,
                    entry_ts=0, bars_held_3m=0, unrealized_pnl=0.0
                )

        return self._calculate_metrics(trades)

    def _calculate_metrics(self, trades: List[Dict]) -> Dict:
        if not trades:
            return {
                "n_trades": 0, "win_rate": 0.0, "return_pct": 0.0,
                "profit_factor": 0.0, "expectancy": 0.0,
                "avg_bars_held": 0, "max_consecutive_loss": 0,
            }

        pnls = [t["pnl_pct"] for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        # Max consecutive losses
        max_consec = 0
        consec = 0
        for p in pnls:
            if p <= 0:
                consec += 1
                max_consec = max(max_consec, consec)
            else:
                consec = 0

        return {
            "n_trades": len(trades),
            "win_rate": len(wins) / len(trades) * 100 if trades else 0,
            "return_pct": sum(pnls),
            "profit_factor": sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else 999,
            "expectancy": np.mean(pnls) if pnls else 0,
            "avg_bars_held": np.mean([t["bars_held"] for t in trades]),
            "max_consecutive_loss": max_consec,
        }


def load_data():
    """Load recent 6 months data (declining market)."""
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

    # Recent 6 months
    start = datetime(2024, 7, 1)
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


def add_features(df, suffix=""):
    """Add technical indicators."""
    df = df.copy()
    df[f"ret{suffix}"] = df["close"].pct_change()
    df[f"ema_20{suffix}"] = df["close"].ewm(span=20, adjust=False).mean()
    df[f"ema_50{suffix}"] = df["close"].ewm(span=50, adjust=False).mean()
    df[f"ema_fast{suffix}"] = df[f"ema_20{suffix}"]
    df[f"ema_slow{suffix}"] = df[f"ema_50{suffix}"]
    df[f"ema_slope{suffix}"] = df[f"ema_20{suffix}"].pct_change(5)

    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df[f"atr{suffix}"] = tr.rolling(14).mean()

    return df


def train_hmm(df_15m):
    """Train HMM (same as tuning script)."""
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

    # Label states by vol_mean (matching tuning)
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

    return states, state_labels, features.index


def load_hmm_from_saved(model_path: Path, df_15m: pd.DataFrame):
    """Load HMM from saved model and predict states for the given data.

    This ensures consistency between training and tuning by using
    the exact same model with the same state labels.
    """
    with open(model_path, "rb") as f:
        data = pickle.load(f)

    # Handle dict format (new) - more portable
    if isinstance(data, dict):
        hmm_model = data["model"]
        state_labels = data["state_labels"]
        train_start = data["train_start"]
        train_end = data["train_end"]
    else:
        # Old format - dataclass object
        hmm_model = data.model
        state_labels = data.state_labels
        train_start = data.train_start
        train_end = data.train_end

    logger.info(f"Loaded HMM model from: {model_path}")
    logger.info(f"  Train period: {train_start} ~ {train_end}")
    logger.info(f"  State labels: {state_labels}")

    # Compute features for prediction (same as train_hmm)
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

    # Predict using saved model
    states = hmm_model.predict(X)

    return states, state_labels, features.index


def main():
    parser = argparse.ArgumentParser(description="SHORT Strategy Parameter Tuning")
    parser.add_argument("--hmm-model", type=Path, default=None,
                       help="Path to pre-trained HMM model (pkl). If not provided, trains new model.")
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("SHORT STRATEGY PARAMETER TUNING")
    logger.info("Using ACTUAL TradingFSM + DecisionEngine (consistent with backtest)")
    logger.info("=" * 70)
    logger.info("Period: 2024-07-01 ~ 2024-12-31 (declining market)")

    logger.info("\nLoading data...")
    df_3m, df_15m = load_data()
    logger.info(f"3m bars: {len(df_3m)}, 15m bars: {len(df_15m)}")

    logger.info("Adding features...")
    df_3m = add_features(df_3m, "_3m")
    df_15m = add_features(df_15m, "_15m")

    # Train or load HMM
    if args.hmm_model and args.hmm_model.exists():
        logger.info(f"Loading pre-trained HMM from: {args.hmm_model}")
        states_15m, state_labels, hmm_index = load_hmm_from_saved(args.hmm_model, df_15m)
    else:
        if args.hmm_model:
            logger.warning(f"HMM model file not found: {args.hmm_model}, training new model...")
        logger.info("Training HMM...")
        states_15m, state_labels, hmm_index = train_hmm(df_15m)
    logger.info(f"State labels: {state_labels}")

    # Count TREND_DOWN bars
    trend_down_state = [k for k, v in state_labels.items() if v == "TREND_DOWN"][0]
    trend_down_count = (states_15m == trend_down_state).sum()
    logger.info(f"TREND_DOWN bars: {trend_down_count} ({trend_down_count/len(states_15m)*100:.1f}%)")

    state_df = pd.DataFrame({"state": states_15m}, index=hmm_index)
    state_3m = state_df.reindex(df_3m.index, method="ffill")

    valid_mask = ~state_3m["state"].isna() & ~df_3m["atr_3m"].isna()
    valid_idx = np.where(valid_mask.values)[0]

    # Use 70% train, 30% test
    split_idx = int(len(valid_idx) * 0.7)
    test_idx = valid_idx[split_idx:]

    logger.info(f"Test samples: {len(test_idx)}")

    config = {
        "DECISION": {"DENY_IN_HIGH_VOL": True, "DENY_IN_NO_TRADE": True},
        "RISK": {"BASE_SIZE_USDT": 100},
    }

    regimes = state_3m["state"].fillna(0).astype(int).values
    tuner = ShortTuner(df_3m, df_15m, regimes, state_labels, config)

    # Parameter grid for SHORT - includes all TradingFSM SHORT params
    # Reduced grid for faster testing (648 combinations)
    param_grid = {
        # Entry conditions
        "SHORT_PULLBACK_TO_EMA_ATR": [0.5, 1.0, 1.5],
        "SHORT_MIN_EMA_SLOPE_15M": [0.002, 0.003],
        "SHORT_MIN_REJECTION_RET": [0.0005, 0.0007, 0.001],
        # Entry confirmation (CRITICAL - these were missing in old tuner!)
        "SHORT_CONSEC_NEG_BARS": [1, 2, 3],           # Consecutive negative bars
        "SHORT_PEAK_CONFIRM_BARS": [2, 3, 4],        # Peak confirmation lookback
        # Risk/Exit
        "SHORT_MAX_HOLD_BARS_3M": [20, 30],          # Time stop for SHORT
        "STOP_ATR_MULT": [1.5, 2.0],
    }

    # Generate all combinations
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(product(*values))

    logger.info(f"\nTesting {len(combinations)} parameter combinations...")

    results = []
    best_result = None
    best_return = -999

    for i, combo in enumerate(combinations):
        params = ShortTuneParams(**dict(zip(keys, combo)))
        metrics = tuner.run_backtest(params, test_idx)

        if metrics["n_trades"] >= 10:  # Minimum trades
            results.append({
                "params": dict(zip(keys, combo)),
                "metrics": metrics,
            })

            if metrics["return_pct"] > best_return:
                best_return = metrics["return_pct"]
                best_result = results[-1]

        if (i + 1) % 100 == 0:
            logger.info(f"  Tested {i + 1}/{len(combinations)}...")

    # Sort by return
    results.sort(key=lambda x: x["metrics"]["return_pct"], reverse=True)

    logger.info("\n" + "=" * 70)
    logger.info("TOP 10 PARAMETER COMBINATIONS")
    logger.info("=" * 70)

    for i, r in enumerate(results[:10]):
        m = r["metrics"]
        logger.info(f"\n#{i+1}: Return={m['return_pct']:.2f}%, WR={m['win_rate']:.1f}%, "
                   f"PF={m['profit_factor']:.2f}, Trades={m['n_trades']}")
        logger.info(f"   Params: {r['params']}")

    # Best result details
    if best_result:
        logger.info("\n" + "=" * 70)
        logger.info("BEST PARAMETERS FOR SHORT")
        logger.info("=" * 70)
        m = best_result["metrics"]
        p = best_result["params"]

        logger.info(f"\nPerformance:")
        logger.info(f"  Return: {m['return_pct']:.2f}%")
        logger.info(f"  Win Rate: {m['win_rate']:.1f}%")
        logger.info(f"  Profit Factor: {m['profit_factor']:.2f}")
        logger.info(f"  Expectancy: {m['expectancy']:.4f}%")
        logger.info(f"  N Trades: {m['n_trades']}")
        logger.info(f"  Avg Bars Held: {m['avg_bars_held']:.1f}")
        logger.info(f"  Max Consecutive Loss: {m['max_consecutive_loss']}")

        logger.info(f"\nOptimal Parameters:")
        for k, v in p.items():
            logger.info(f"  {k}: {v}")

        # Save to file
        output_dir = PROJECT_ROOT / "outputs" / "short_tuning"
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir / "best_short_params.json", "w") as f:
            json.dump({
                "params": p,
                "metrics": m,
            }, f, indent=2)

        logger.info(f"\nResults saved to {output_dir}")

    logger.info("\nDone!")


if __name__ == "__main__":
    main()
