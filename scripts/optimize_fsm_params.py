#!/usr/bin/env python
"""FSM Parameter Optimization via Grid Search.

Optimizes key FSM parameters using grid search on train set,
then validates on test set.

Usage:
    python scripts/optimize_fsm_params.py
"""

import argparse
import itertools
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from backtest_fsm_pipeline import (
    MultiHMM,
    build_features,
    load_ohlcv_from_timescaledb,
)

from xrp4.regime.confirm import RegimeConfirmLayer, ConfirmConfig
from xrp4.core.types import (
    ConfirmContext,
    MarketContext,
    PositionState,
)
from xrp4.core.fsm import TradingFSM, FSMConfig
from xrp4.core.decision_engine import DecisionEngine, DecisionConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def run_backtest_with_params(
    df_3m: pd.DataFrame,
    df_15m: pd.DataFrame,
    hmm_model: MultiHMM,
    features_15m: np.ndarray,
    features_3m: np.ndarray,
    df_3m_hmm: pd.DataFrame,
    fsm_params: Dict,
    initial_capital: float = 10000.0,
    fee_rate: float = 0.001,
) -> Dict:
    """Run backtest with specific FSM parameters."""

    confirm_layer = RegimeConfirmLayer(ConfirmConfig(
        HIGH_VOL_LAMBDA_ON=2.0,
        HIGH_VOL_COOLDOWN_BARS_15M=6,
    ))

    # Create FSMConfig with custom parameters
    fsm_config = FSMConfig(
        RANGE_ENABLED=False,
        TRANSITION_ALLOW_BREAKOUT_ONLY=False,
        **fsm_params
    )
    fsm = TradingFSM(fsm_config)
    de_config = DecisionConfig(XGB_ENABLED=True)
    decision_engine = DecisionEngine(de_config)

    # Get MultiHMM predictions
    timestamps_3m = pd.to_datetime(df_3m_hmm["timestamp"])
    timestamps_15m = pd.to_datetime(df_15m["timestamp"])
    state_seq, state_probs, labels = hmm_model.predict_sequence(
        features_3m, features_15m, timestamps_3m, timestamps_15m
    )

    # Match timestamps
    df_3m_bt = df_3m.copy()
    df_3m_bt = df_3m_bt.set_index("timestamp")
    df_3m_hmm_indexed = df_3m_hmm.set_index("timestamp")
    df_3m_hmm_indexed["regime_raw"] = labels
    df_3m_hmm_indexed["confidence"] = [probs.max() for probs in state_probs]
    df_3m_bt = df_3m_bt.join(df_3m_hmm_indexed[["regime_raw", "confidence"]], how="left")
    df_3m_bt[["regime_raw", "confidence"]] = df_3m_bt[["regime_raw", "confidence"]].ffill()
    df_3m_bt = df_3m_bt.dropna(subset=["regime_raw"])
    df_3m_bt = df_3m_bt.reset_index()

    df_15m_indexed = df_15m.copy()
    if "timestamp" in df_15m_indexed.columns:
        df_15m_indexed["timestamp"] = pd.to_datetime(df_15m_indexed["timestamp"])

    confirm_state = None
    fsm_state = None
    engine_state = None

    equity = initial_capital
    position = PositionState(side="FLAT")
    trades = []
    entry_regime = None

    for idx in range(250, len(df_3m_bt)):
        bar = df_3m_bt.iloc[idx]
        ts = bar["timestamp"]

        matching_15m = df_15m_indexed[df_15m_indexed["timestamp"] <= ts]
        if len(matching_15m) == 0:
            continue
        current_15m_idx = len(matching_15m) - 1
        bar_15m = matching_15m.iloc[-1]
        hist_15m = matching_15m.iloc[max(0, current_15m_idx-96):current_15m_idx+1]

        regime_raw = bar["regime_raw"]

        confirm_result, confirm_state = confirm_layer.confirm(
            regime_raw=regime_raw,
            row_15m=bar_15m.to_dict(),
            hist_15m=hist_15m,
            state=confirm_state,
        )

        confirm_ctx = ConfirmContext(
            regime_raw=regime_raw,
            regime_confirmed=confirm_result.confirmed_regime,
            confirm_reason=confirm_result.reason,
            confirm_metrics=confirm_result.metrics,
        )

        price = float(bar["close"])
        atr_3m = bar.get("atr_3m", 0.01)
        atr_15m = bar.get("atr_15m", atr_3m)
        support = bar.get("rolling_low_20", price - atr_15m * 2)
        resistance = bar.get("rolling_high_20", price + atr_15m * 2)
        dist_to_support = (price - support) / atr_3m if atr_3m > 0 else 999
        dist_to_resistance = (resistance - price) / atr_3m if atr_3m > 0 else 999
        ema_fast = bar.get("ema_fast_3m", price)
        ema_slow = bar.get("ema_slow_3m", price)
        ret_3m = bar.get("ret_3m", 0)
        volatility = bar.get("volatility_3m", bar.get("ret_std_3m", 0.005))

        market_ctx = MarketContext(
            symbol="XRPUSDT",
            ts=int(ts.timestamp() * 1000) if hasattr(ts, "timestamp") else 0,
            price=price,
            row_3m={
                "close": price,
                "atr_3m": atr_3m,
                "ema_fast_3m": ema_fast,
                "ema_slow_3m": ema_slow,
                "ret_3m": ret_3m,
                "ret": ret_3m,
                "volatility": volatility,
                "rsi_3m": bar.get("rsi_3m", 50),
                "ema_slope_15m": bar.get("ema_slope_15m", 0),
                "ema_diff": (ema_fast - ema_slow) / price if price > 0 else 0,
                "price_to_ema20": (price - ema_fast) / ema_fast if ema_fast > 0 else 0,
                "price_to_ema50": (price - ema_slow) / ema_slow if ema_slow > 0 else 0,
            },
            row_15m={
                "ema_slope_15m": bar.get("ema_slope_15m", 0),
                "ewm_ret_15m": bar.get("ewm_ret_15m", 0),
            },
            zone={
                "support": support,
                "resistance": resistance,
                "strength": 0.0001,
                "dist_to_support": dist_to_support,
                "dist_to_resistance": dist_to_resistance,
            },
        )

        if position.side != "FLAT":
            position.bars_held_3m += 1
            if position.side == "LONG":
                position.unrealized_pnl = (price - position.entry_price) * position.size
            else:
                position.unrealized_pnl = (position.entry_price - price) * position.size

        candidate_signal, fsm_state = fsm.step(
            ctx=market_ctx,
            confirm=confirm_ctx,
            pos=position,
            fsm_state=fsm_state,
        )

        decision, engine_state = decision_engine.decide(
            ctx=market_ctx,
            confirm=confirm_ctx,
            pos=position,
            cand=candidate_signal,
            engine_state=engine_state,
        )

        position_size_usd = equity * 0.95

        if decision.action == "OPEN_LONG" and position.side == "FLAT":
            entry_price = price
            size = position_size_usd / entry_price
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

        elif decision.action == "OPEN_SHORT" and position.side == "FLAT":
            entry_price = price
            size = position_size_usd / entry_price
            fee = entry_price * size * fee_rate
            equity -= fee
            position = PositionState(
                side="SHORT",
                entry_price=entry_price,
                size=size,
                entry_ts=market_ctx.ts,
                bars_held_3m=0,
                unrealized_pnl=0,
            )
            entry_regime = confirm_result.confirmed_regime

        elif decision.action == "CLOSE" and position.side != "FLAT":
            exit_price = price
            if position.side == "LONG":
                pnl = (exit_price - position.entry_price) * position.size
            else:
                pnl = (position.entry_price - exit_price) * position.size
            fee = exit_price * position.size * fee_rate
            pnl -= fee
            equity += pnl

            trades.append({
                "pnl": pnl,
                "pnl_pct": pnl / (position.entry_price * position.size) * 100 if position.size > 0 else 0,
                "bars_held": position.bars_held_3m,
                "regime": entry_regime,
            })
            position = PositionState(side="FLAT")

    # Close remaining position
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
            "pnl": pnl,
            "pnl_pct": pnl / (position.entry_price * position.size) * 100 if position.size > 0 else 0,
            "bars_held": position.bars_held_3m,
            "regime": entry_regime,
        })

    # Calculate metrics
    if not trades:
        return {
            "n_trades": 0,
            "total_pnl": 0,
            "return_pct": 0,
            "win_rate": 0,
            "profit_factor": 0,
            "final_equity": equity,
        }

    df = pd.DataFrame(trades)
    wins = df[df["pnl"] > 0]
    losses = df[df["pnl"] <= 0]

    return {
        "n_trades": len(trades),
        "total_pnl": df["pnl"].sum(),
        "return_pct": (equity - initial_capital) / initial_capital * 100,
        "win_rate": len(wins) / len(trades) * 100,
        "profit_factor": (
            wins["pnl"].sum() / abs(losses["pnl"].sum())
            if len(losses) > 0 and losses["pnl"].sum() != 0
            else float("inf") if len(wins) > 0 else 0
        ),
        "avg_win_pct": wins["pnl_pct"].mean() if len(wins) > 0 else 0,
        "avg_loss_pct": losses["pnl_pct"].mean() if len(losses) > 0 else 0,
        "final_equity": equity,
    }


def main():
    parser = argparse.ArgumentParser(description="Optimize FSM parameters")
    parser.add_argument("--start", default="2024-07-01", help="Start date")
    parser.add_argument("--end", default="2026-01-15", help="End date")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Train ratio")
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("FSM Parameter Optimization")
    logger.info("=" * 70)

    # Load data
    logger.info(f"\nLoading data: {args.start} ~ {args.end}")
    start_dt = datetime.fromisoformat(args.start)
    end_dt = datetime.fromisoformat(args.end)

    df_3m = load_ohlcv_from_timescaledb("XRPUSDT", start_dt, end_dt, "3m")
    df_15m = load_ohlcv_from_timescaledb("XRPUSDT", start_dt, end_dt, "15m")

    if df_3m.empty or df_15m.empty:
        logger.error("No data!")
        sys.exit(1)

    logger.info(f"Loaded {len(df_3m)} 3m bars, {len(df_15m)} 15m bars")

    # Build features
    logger.info("Building features...")
    features_15m, names_15m, df_15m_clean, df_3m_full, features_3m, names_3m, df_3m_hmm = build_features(df_3m, df_15m)

    # Train/Test split
    train_size = int(len(df_3m_full) * args.train_ratio)
    df_3m_train = df_3m_full.iloc[:train_size].copy()
    df_3m_test = df_3m_full.iloc[train_size:].copy()

    train_size_15m = int(len(df_15m_clean) * args.train_ratio)
    df_15m_train = df_15m_clean.iloc[:train_size_15m].copy()
    df_15m_test = df_15m_clean.iloc[train_size_15m:].copy()

    train_size_hmm = int(len(df_3m_hmm) * args.train_ratio)
    df_3m_hmm_train = df_3m_hmm.iloc[:train_size_hmm].copy()
    df_3m_hmm_test = df_3m_hmm.iloc[train_size_hmm:].copy()

    features_15m_train = features_15m[:train_size_15m]
    features_15m_test = features_15m[train_size_15m:]
    features_3m_train = features_3m[:train_size_hmm]
    features_3m_test = features_3m[train_size_hmm:]

    logger.info(f"Train: {len(df_3m_train)} 3m bars, Test: {len(df_3m_test)} 3m bars")

    # Train MultiHMM on train data
    logger.info("Training MultiHMM on train data...")
    hmm_model = MultiHMM(n_states_3m=4, n_states_15m=4, fast_weight=0.4, mid_weight=0.6)
    hmm_model.train(features_3m_train, names_3m, features_15m_train, names_15m)

    # Define parameter grid (simplified - key parameters only)
    param_grid = {
        "TREND_PULLBACK_TO_EMA_ATR": [1.5, 2.0, 2.5, 3.0],
        "MAX_HOLD_BARS_3M": [30, 40, 50, 60],
    }

    # Generate all combinations
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(itertools.product(*values))

    logger.info(f"\nGrid Search: {len(combinations)} combinations")
    logger.info(f"Parameters: {keys}")

    # Run grid search on train set
    results = []
    best_pf = 0
    best_params = None
    best_result = None

    for i, combo in enumerate(combinations):
        params = dict(zip(keys, combo))

        if (i + 1) % 10 == 0:
            logger.info(f"Progress: {i+1}/{len(combinations)}")

        result = run_backtest_with_params(
            df_3m_train, df_15m_train, hmm_model,
            features_15m_train, features_3m_train, df_3m_hmm_train,
            params
        )

        result["params"] = params
        results.append(result)

        # Track best by profit factor (with minimum trades)
        if result["n_trades"] >= 100 and result["profit_factor"] > best_pf:
            best_pf = result["profit_factor"]
            best_params = params
            best_result = result

    logger.info("\n" + "=" * 70)
    logger.info("GRID SEARCH RESULTS (Train Set)")
    logger.info("=" * 70)

    # Sort by profit factor
    results_sorted = sorted(results, key=lambda x: x["profit_factor"] if x["n_trades"] >= 100 else 0, reverse=True)

    logger.info("\nTop 5 Parameter Sets:")
    for i, r in enumerate(results_sorted[:5]):
        logger.info(f"\n{i+1}. PF={r['profit_factor']:.2f}, WR={r['win_rate']:.1f}%, Trades={r['n_trades']}")
        logger.info(f"   Params: {r['params']}")

    # Validate best params on test set
    if best_params:
        logger.info("\n" + "=" * 70)
        logger.info("VALIDATION ON TEST SET")
        logger.info("=" * 70)
        logger.info(f"\nBest Params: {best_params}")

        test_result = run_backtest_with_params(
            df_3m_test, df_15m_test, hmm_model,
            features_15m_test, features_3m_test, df_3m_hmm_test,
            best_params
        )

        logger.info(f"\nTrain Results:")
        logger.info(f"  Trades: {best_result['n_trades']}")
        logger.info(f"  Win Rate: {best_result['win_rate']:.1f}%")
        logger.info(f"  Profit Factor: {best_result['profit_factor']:.2f}")
        logger.info(f"  Return: {best_result['return_pct']:.2f}%")

        logger.info(f"\nTest Results:")
        logger.info(f"  Trades: {test_result['n_trades']}")
        logger.info(f"  Win Rate: {test_result['win_rate']:.1f}%")
        logger.info(f"  Profit Factor: {test_result['profit_factor']:.2f}")
        logger.info(f"  Return: {test_result['return_pct']:.2f}%")

        # Compare with current params
        current_params = {
            "TREND_PULLBACK_TO_EMA_ATR": 2.0,
            "TREND_MIN_EMA_SLOPE_15M": 0.002,
            "TREND_MIN_REBOUND_RET": 0.0003,
            "MAX_HOLD_BARS_3M": 40,
        }

        current_train = run_backtest_with_params(
            df_3m_train, df_15m_train, hmm_model,
            features_15m_train, features_3m_train, df_3m_hmm_train,
            current_params
        )
        current_test = run_backtest_with_params(
            df_3m_test, df_15m_test, hmm_model,
            features_15m_test, features_3m_test, df_3m_hmm_test,
            current_params
        )

        logger.info("\n" + "=" * 70)
        logger.info("COMPARISON: Current vs Optimized")
        logger.info("=" * 70)

        print(f"\n{'Metric':<20} {'Current (Train)':>15} {'Optimized (Train)':>18}")
        print("-" * 55)
        print(f"{'Trades':<20} {current_train['n_trades']:>15} {best_result['n_trades']:>18}")
        print(f"{'Win Rate':<20} {current_train['win_rate']:>14.1f}% {best_result['win_rate']:>17.1f}%")
        print(f"{'Profit Factor':<20} {current_train['profit_factor']:>15.2f} {best_result['profit_factor']:>18.2f}")
        print(f"{'Return':<20} {current_train['return_pct']:>14.2f}% {best_result['return_pct']:>17.2f}%")

        print(f"\n{'Metric':<20} {'Current (Test)':>15} {'Optimized (Test)':>18}")
        print("-" * 55)
        print(f"{'Trades':<20} {current_test['n_trades']:>15} {test_result['n_trades']:>18}")
        print(f"{'Win Rate':<20} {current_test['win_rate']:>14.1f}% {test_result['win_rate']:>17.1f}%")
        print(f"{'Profit Factor':<20} {current_test['profit_factor']:>15.2f} {test_result['profit_factor']:>18.2f}")
        print(f"{'Return':<20} {current_test['return_pct']:>14.2f}% {test_result['return_pct']:>17.2f}%")

        # Save results
        output_dir = Path("outputs/fsm_optimization")
        output_dir.mkdir(parents=True, exist_ok=True)

        output_data = {
            "best_params": best_params,
            "current_params": current_params,
            "train_comparison": {
                "current": current_train,
                "optimized": best_result,
            },
            "test_comparison": {
                "current": current_test,
                "optimized": test_result,
            },
            "all_results": results_sorted[:20],
        }

        with open(output_dir / "optimization_results.json", "w") as f:
            json.dump(output_data, f, indent=2, default=str)

        logger.info(f"\nResults saved to {output_dir}")

        # Print recommendation
        logger.info("\n" + "=" * 70)
        logger.info("RECOMMENDATION")
        logger.info("=" * 70)

        if test_result['profit_factor'] > current_test['profit_factor']:
            logger.info(f"\n✅ Optimized params improve Test PF: {current_test['profit_factor']:.2f} -> {test_result['profit_factor']:.2f}")
            logger.info(f"\nRecommended params to apply:")
            for k, v in best_params.items():
                logger.info(f"  {k}: {v}")
        else:
            logger.info(f"\n⚠️ Optimized params don't improve Test PF")
            logger.info(f"   Current: {current_test['profit_factor']:.2f}, Optimized: {test_result['profit_factor']:.2f}")
            logger.info(f"   Keep current parameters.")


if __name__ == "__main__":
    main()
