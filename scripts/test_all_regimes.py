#!/usr/bin/env python
"""Test with all regimes enabled (RANGE, TRANSITION, TREND).

Compares:
1. Current (RANGE/TRANSITION disabled)
2. All regimes enabled

Usage:
    python scripts/test_all_regimes.py
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# Import the backtest pipeline components
from backtest_fsm_pipeline import (
    MultiHMM,
    SingleHMM,
    build_features,
    load_ohlcv_from_timescaledb,
)

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


def run_backtest_with_config(
    df_3m: pd.DataFrame,
    df_15m: pd.DataFrame,
    hmm_model: MultiHMM,
    features_15m: np.ndarray,
    features_3m: np.ndarray,
    df_3m_hmm: pd.DataFrame,
    fsm_config: FSMConfig,
    de_config: DecisionConfig,
    initial_capital: float = 10000.0,
    fee_rate: float = 0.001,
) -> Dict:
    """Run backtest with specific FSM and DE configuration."""

    # Initialize components with custom config
    confirm_layer = RegimeConfirmLayer(ConfirmConfig(
        HIGH_VOL_LAMBDA_ON=2.0,
        HIGH_VOL_COOLDOWN_BARS_15M=6,
    ))
    fsm = TradingFSM(fsm_config)
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

    # State tracking
    confirm_state = None
    fsm_state = None
    engine_state = None

    # Backtest state
    equity = initial_capital
    position = PositionState(side="FLAT")
    trades = []
    entry_regime = None
    entry_signal = None

    for idx in range(250, len(df_3m_bt)):
        bar = df_3m_bt.iloc[idx]
        ts = bar["timestamp"]

        # Find corresponding 15m bar
        matching_15m = df_15m_indexed[df_15m_indexed["timestamp"] <= ts]
        if len(matching_15m) == 0:
            continue
        current_15m_idx = len(matching_15m) - 1
        bar_15m = matching_15m.iloc[-1]
        hist_15m = matching_15m.iloc[max(0, current_15m_idx-96):current_15m_idx+1]

        regime_raw = bar["regime_raw"]

        # ConfirmLayer
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
                "volatility": bar.get("volatility_3m", bar.get("ret_std_3m", 0.005)),
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

        # Update position state
        if position.side != "FLAT":
            position.bars_held_3m += 1
            if position.side == "LONG":
                position.unrealized_pnl = (price - position.entry_price) * position.size
            else:
                position.unrealized_pnl = (position.entry_price - price) * position.size

        # FSM
        candidate_signal, fsm_state = fsm.step(
            ctx=market_ctx,
            confirm=confirm_ctx,
            pos=position,
            fsm_state=fsm_state,
        )

        # DecisionEngine
        decision, engine_state = decision_engine.decide(
            ctx=market_ctx,
            confirm=confirm_ctx,
            pos=position,
            cand=candidate_signal,
            engine_state=engine_state,
        )

        # Execute
        if decision.action == "OPEN_LONG" and position.side == "FLAT":
            entry_price = price
            size = decision.size / entry_price
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
            entry_signal = candidate_signal.signal

        elif decision.action == "OPEN_SHORT" and position.side == "FLAT":
            entry_price = price
            size = decision.size / entry_price
            fee = entry_price * size * fee_rate
            equity -= fee
            entry_regime = confirm_result.confirmed_regime
            entry_signal = candidate_signal.signal
            position = PositionState(
                side="SHORT",
                entry_price=entry_price,
                size=size,
                entry_ts=market_ctx.ts,
                bars_held_3m=0,
                unrealized_pnl=0,
            )

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
                "side": position.side,
                "entry_price": position.entry_price,
                "exit_price": exit_price,
                "size": position.size,
                "pnl": pnl,
                "pnl_pct": pnl / (position.entry_price * position.size) * 100 if position.size > 0 else 0,
                "bars_held": position.bars_held_3m,
                "regime": entry_regime,
                "signal": entry_signal,
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
            "side": position.side,
            "entry_price": position.entry_price,
            "exit_price": exit_price,
            "size": position.size,
            "pnl": pnl,
            "pnl_pct": pnl / (position.entry_price * position.size) * 100 if position.size > 0 else 0,
            "bars_held": position.bars_held_3m,
            "regime": entry_regime,
            "signal": "EOD",
        })

    # Calculate metrics
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()

    results = {
        "n_trades": len(trades),
        "total_pnl": sum(t["pnl"] for t in trades),
        "total_return_pct": (equity - initial_capital) / initial_capital * 100,
    }

    if len(trades) > 0:
        wins = trades_df[trades_df["pnl"] > 0]
        losses = trades_df[trades_df["pnl"] <= 0]
        results["win_rate"] = len(wins) / len(trades) * 100
        results["profit_factor"] = (
            wins["pnl"].sum() / abs(losses["pnl"].sum())
            if len(losses) > 0 and losses["pnl"].sum() != 0
            else float("inf") if len(wins) > 0 else 0
        )

        # Regime breakdown
        regime_stats = {}
        for regime in trades_df["regime"].unique():
            regime_trades = trades_df[trades_df["regime"] == regime]
            if len(regime_trades) > 0:
                regime_wins = regime_trades[regime_trades["pnl"] > 0]
                regime_stats[regime] = {
                    "n_trades": len(regime_trades),
                    "win_rate": len(regime_wins) / len(regime_trades) * 100,
                    "total_pnl": regime_trades["pnl"].sum(),
                }
        results["regime_stats"] = regime_stats

        # Signal breakdown
        signal_stats = {}
        for signal in trades_df["signal"].unique():
            signal_trades = trades_df[trades_df["signal"] == signal]
            if len(signal_trades) > 0:
                signal_wins = signal_trades[signal_trades["pnl"] > 0]
                signal_stats[signal] = {
                    "n_trades": len(signal_trades),
                    "win_rate": len(signal_wins) / len(signal_trades) * 100,
                    "total_pnl": signal_trades["pnl"].sum(),
                }
        results["signal_stats"] = signal_stats
    else:
        results["win_rate"] = 0
        results["profit_factor"] = 0
        results["regime_stats"] = {}
        results["signal_stats"] = {}

    return results


def main():
    parser = argparse.ArgumentParser(description="Test with all regimes enabled")
    parser.add_argument("--start", default="2025-07-01", help="Test start date")
    parser.add_argument("--end", default="2026-01-15", help="Test end date")
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("Testing All Regimes (RANGE, TRANSITION, TREND)")
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

    # Train MultiHMM
    logger.info("Training MultiHMM...")
    hmm_model = MultiHMM(n_states_3m=4, n_states_15m=4, fast_weight=0.4, mid_weight=0.6)
    hmm_model.train(features_3m, names_3m, features_15m, names_15m)

    # ========================================================================
    # Test 1: Current config (RANGE/TRANSITION disabled)
    # ========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("Test 1: Current Config (RANGE/TRANSITION disabled)")
    logger.info("=" * 70)

    fsm_config_current = FSMConfig(
        RANGE_ENABLED=False,
        TRANSITION_ALLOW_BREAKOUT_ONLY=False,
    )
    de_config_current = DecisionConfig(XGB_ENABLED=True)

    results_current = run_backtest_with_config(
        df_3m_full, df_15m_clean, hmm_model, features_15m, features_3m, df_3m_hmm,
        fsm_config_current, de_config_current
    )

    logger.info(f"\nCurrent Config Results:")
    logger.info(f"  Trades: {results_current['n_trades']}")
    logger.info(f"  Win Rate: {results_current['win_rate']:.1f}%")
    logger.info(f"  Total PnL: ${results_current['total_pnl']:.2f}")
    logger.info(f"  Profit Factor: {results_current['profit_factor']:.2f}")

    if results_current.get("regime_stats"):
        logger.info("\n  Regime Breakdown:")
        for regime, stats in results_current["regime_stats"].items():
            logger.info(f"    {regime}: {stats['n_trades']} trades, WR={stats['win_rate']:.1f}%, PnL=${stats['total_pnl']:.2f}")

    # ========================================================================
    # Test 2: All regimes enabled
    # ========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("Test 2: All Regimes Enabled (RANGE + TRANSITION + TREND)")
    logger.info("=" * 70)

    fsm_config_all = FSMConfig(
        RANGE_ENABLED=True,
        TRANSITION_ALLOW_BREAKOUT_ONLY=True,
    )
    de_config_all = DecisionConfig(XGB_ENABLED=True)

    results_all = run_backtest_with_config(
        df_3m_full, df_15m_clean, hmm_model, features_15m, features_3m, df_3m_hmm,
        fsm_config_all, de_config_all
    )

    logger.info(f"\nAll Regimes Results:")
    logger.info(f"  Trades: {results_all['n_trades']}")
    logger.info(f"  Win Rate: {results_all['win_rate']:.1f}%")
    logger.info(f"  Total PnL: ${results_all['total_pnl']:.2f}")
    logger.info(f"  Profit Factor: {results_all['profit_factor']:.2f}")

    if results_all.get("regime_stats"):
        logger.info("\n  Regime Breakdown:")
        for regime, stats in results_all["regime_stats"].items():
            logger.info(f"    {regime}: {stats['n_trades']} trades, WR={stats['win_rate']:.1f}%, PnL=${stats['total_pnl']:.2f}")

    if results_all.get("signal_stats"):
        logger.info("\n  Signal Breakdown:")
        for signal, stats in results_all["signal_stats"].items():
            logger.info(f"    {signal}: {stats['n_trades']} trades, WR={stats['win_rate']:.1f}%, PnL=${stats['total_pnl']:.2f}")

    # ========================================================================
    # Comparison
    # ========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("COMPARISON")
    logger.info("=" * 70)

    print(f"\n{'Metric':<20} {'Current':>15} {'All Regimes':>15} {'Diff':>15}")
    print("-" * 65)
    print(f"{'Trades':<20} {results_current['n_trades']:>15} {results_all['n_trades']:>15} {results_all['n_trades'] - results_current['n_trades']:>+15}")
    print(f"{'Win Rate':<20} {results_current['win_rate']:>14.1f}% {results_all['win_rate']:>14.1f}% {results_all['win_rate'] - results_current['win_rate']:>+14.1f}%")
    print(f"{'Total PnL':<20} ${results_current['total_pnl']:>13.2f} ${results_all['total_pnl']:>13.2f} ${results_all['total_pnl'] - results_current['total_pnl']:>+13.2f}")
    pf_diff = results_all['profit_factor'] - results_current['profit_factor']
    print(f"{'Profit Factor':<20} {results_current['profit_factor']:>15.2f} {results_all['profit_factor']:>15.2f} {pf_diff:>+15.2f}")

    # Save results
    output_dir = Path("outputs/all_regimes_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "test_period": f"{args.start} ~ {args.end}",
        "current_config": {
            "n_trades": results_current["n_trades"],
            "win_rate": results_current["win_rate"],
            "total_pnl": results_current["total_pnl"],
            "profit_factor": results_current["profit_factor"],
            "regime_stats": results_current.get("regime_stats", {}),
        },
        "all_regimes": {
            "n_trades": results_all["n_trades"],
            "win_rate": results_all["win_rate"],
            "total_pnl": results_all["total_pnl"],
            "profit_factor": results_all["profit_factor"],
            "regime_stats": results_all.get("regime_stats", {}),
            "signal_stats": results_all.get("signal_stats", {}),
        },
    }

    with open(output_dir / "comparison_results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
