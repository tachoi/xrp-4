#!/usr/bin/env python
"""Test with independent positions per regime type.

Each regime (RANGE, TREND) has its own position slot.
This eliminates interference between strategies.

Usage:
    python scripts/test_independent_positions.py
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


def run_backtest_independent_positions(
    df_3m: pd.DataFrame,
    df_15m: pd.DataFrame,
    hmm_model: MultiHMM,
    features_15m: np.ndarray,
    features_3m: np.ndarray,
    df_3m_hmm: pd.DataFrame,
    capital_allocation: Dict[str, float] = None,
    fee_rate: float = 0.001,
) -> Dict:
    """Run backtest with independent positions per regime type.

    Args:
        capital_allocation: Dict mapping regime type to capital fraction.
            e.g., {"RANGE": 0.3, "TREND": 0.7}
            Default: {"RANGE": 0.3, "TREND": 0.7}
    """
    if capital_allocation is None:
        capital_allocation = {"RANGE": 0.3, "TREND": 0.7}

    total_capital = 10000.0

    # Initialize components - all regimes enabled
    confirm_layer = RegimeConfirmLayer(ConfirmConfig(
        HIGH_VOL_LAMBDA_ON=2.0,
        HIGH_VOL_COOLDOWN_BARS_15M=6,
    ))
    fsm_config = FSMConfig(
        RANGE_ENABLED=True,
        TRANSITION_ALLOW_BREAKOUT_ONLY=True,
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

    # State tracking
    confirm_state = None
    fsm_state = None
    engine_state = None

    # Independent position tracking per regime type
    # "RANGE" = RANGE regime positions
    # "TREND" = TREND_UP + TREND_DOWN positions
    positions = {
        "RANGE": PositionState(side="FLAT"),
        "TREND": PositionState(side="FLAT"),
    }
    equity = {
        "RANGE": total_capital * capital_allocation["RANGE"],
        "TREND": total_capital * capital_allocation["TREND"],
    }
    trades = {
        "RANGE": [],
        "TREND": [],
    }
    entry_info = {
        "RANGE": {"regime": None, "signal": None},
        "TREND": {"regime": None, "signal": None},
    }

    def get_regime_type(regime: str) -> Optional[str]:
        """Map confirmed regime to position type."""
        if regime == "RANGE":
            return "RANGE"
        elif regime in ("TREND_UP", "TREND_DOWN"):
            return "TREND"
        return None

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

        # Get current regime type for position routing
        current_regime_type = get_regime_type(confirm_result.confirmed_regime)

        # Use the position for the current regime type (or TREND as default for FSM)
        active_regime_type = current_regime_type if current_regime_type else "TREND"
        active_position = positions[active_regime_type]

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

        # Update all positions' bars held and unrealized PnL
        for rtype in positions:
            pos = positions[rtype]
            if pos.side != "FLAT":
                pos.bars_held_3m += 1
                if pos.side == "LONG":
                    pos.unrealized_pnl = (price - pos.entry_price) * pos.size
                else:
                    pos.unrealized_pnl = (pos.entry_price - price) * pos.size

        # FSM - pass the active position for this regime
        candidate_signal, fsm_state = fsm.step(
            ctx=market_ctx,
            confirm=confirm_ctx,
            pos=active_position,
            fsm_state=fsm_state,
        )

        # DecisionEngine
        decision, engine_state = decision_engine.decide(
            ctx=market_ctx,
            confirm=confirm_ctx,
            pos=active_position,
            cand=candidate_signal,
            engine_state=engine_state,
        )

        # Execute - route to correct position based on regime type
        if current_regime_type is None:
            continue

        pos = positions[current_regime_type]
        eq = equity[current_regime_type]

        if decision.action == "OPEN_LONG" and pos.side == "FLAT":
            entry_price = price
            # Use allocated capital for this regime type
            position_size_usd = eq * 0.95  # Use 95% of allocated capital
            size = position_size_usd / entry_price
            fee = entry_price * size * fee_rate
            equity[current_regime_type] -= fee

            positions[current_regime_type] = PositionState(
                side="LONG",
                entry_price=entry_price,
                size=size,
                entry_ts=market_ctx.ts,
                bars_held_3m=0,
                unrealized_pnl=0,
            )
            entry_info[current_regime_type] = {
                "regime": confirm_result.confirmed_regime,
                "signal": candidate_signal.signal,
            }

        elif decision.action == "OPEN_SHORT" and pos.side == "FLAT":
            entry_price = price
            position_size_usd = eq * 0.95
            size = position_size_usd / entry_price
            fee = entry_price * size * fee_rate
            equity[current_regime_type] -= fee

            positions[current_regime_type] = PositionState(
                side="SHORT",
                entry_price=entry_price,
                size=size,
                entry_ts=market_ctx.ts,
                bars_held_3m=0,
                unrealized_pnl=0,
            )
            entry_info[current_regime_type] = {
                "regime": confirm_result.confirmed_regime,
                "signal": candidate_signal.signal,
            }

        elif decision.action == "CLOSE" and pos.side != "FLAT":
            exit_price = price
            if pos.side == "LONG":
                pnl = (exit_price - pos.entry_price) * pos.size
            else:
                pnl = (pos.entry_price - exit_price) * pos.size
            fee = exit_price * pos.size * fee_rate
            pnl -= fee
            equity[current_regime_type] += pnl

            trades[current_regime_type].append({
                "side": pos.side,
                "entry_price": pos.entry_price,
                "exit_price": exit_price,
                "size": pos.size,
                "pnl": pnl,
                "pnl_pct": pnl / (pos.entry_price * pos.size) * 100 if pos.size > 0 else 0,
                "bars_held": pos.bars_held_3m,
                "regime": entry_info[current_regime_type]["regime"],
                "signal": entry_info[current_regime_type]["signal"],
            })
            positions[current_regime_type] = PositionState(side="FLAT")

    # Close remaining positions
    for rtype in positions:
        pos = positions[rtype]
        if pos.side != "FLAT":
            bar = df_3m_bt.iloc[-1]
            exit_price = float(bar["close"])
            if pos.side == "LONG":
                pnl = (exit_price - pos.entry_price) * pos.size
            else:
                pnl = (pos.entry_price - exit_price) * pos.size
            fee = exit_price * pos.size * fee_rate
            pnl -= fee
            equity[rtype] += pnl
            trades[rtype].append({
                "side": pos.side,
                "entry_price": pos.entry_price,
                "exit_price": exit_price,
                "size": pos.size,
                "pnl": pnl,
                "pnl_pct": pnl / (pos.entry_price * pos.size) * 100 if pos.size > 0 else 0,
                "bars_held": pos.bars_held_3m,
                "regime": entry_info[rtype]["regime"],
                "signal": "EOD",
            })

    # Calculate metrics
    all_trades = trades["RANGE"] + trades["TREND"]

    def calc_metrics(trade_list, initial_cap):
        if not trade_list:
            return {
                "n_trades": 0,
                "total_pnl": 0,
                "win_rate": 0,
                "profit_factor": 0,
                "regime_stats": {},
            }

        df = pd.DataFrame(trade_list)
        wins = df[df["pnl"] > 0]
        losses = df[df["pnl"] <= 0]

        regime_stats = {}
        for regime in df["regime"].unique():
            regime_trades = df[df["regime"] == regime]
            if len(regime_trades) > 0:
                regime_wins = regime_trades[regime_trades["pnl"] > 0]
                regime_stats[regime] = {
                    "n_trades": len(regime_trades),
                    "win_rate": len(regime_wins) / len(regime_trades) * 100,
                    "total_pnl": regime_trades["pnl"].sum(),
                }

        return {
            "n_trades": len(trade_list),
            "total_pnl": sum(t["pnl"] for t in trade_list),
            "win_rate": len(wins) / len(trade_list) * 100 if trade_list else 0,
            "profit_factor": (
                wins["pnl"].sum() / abs(losses["pnl"].sum())
                if len(losses) > 0 and losses["pnl"].sum() != 0
                else float("inf") if len(wins) > 0 else 0
            ),
            "regime_stats": regime_stats,
        }

    results = {
        "range": calc_metrics(trades["RANGE"], total_capital * capital_allocation["RANGE"]),
        "trend": calc_metrics(trades["TREND"], total_capital * capital_allocation["TREND"]),
        "combined": calc_metrics(all_trades, total_capital),
        "equity": {
            "range": equity["RANGE"],
            "trend": equity["TREND"],
            "total": equity["RANGE"] + equity["TREND"],
        },
        "capital_allocation": capital_allocation,
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="Test with independent positions per regime")
    parser.add_argument("--start", default="2024-07-01", help="Test start date")
    parser.add_argument("--end", default="2026-01-15", help="Test end date")
    parser.add_argument("--range-alloc", type=float, default=0.3, help="Capital allocation for RANGE (0-1)")
    parser.add_argument("--trend-alloc", type=float, default=0.7, help="Capital allocation for TREND (0-1)")
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("Testing Independent Positions per Regime Type")
    logger.info("=" * 70)
    logger.info(f"Capital allocation: RANGE={args.range_alloc*100:.0f}%, TREND={args.trend_alloc*100:.0f}%")

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

    # Run backtest with independent positions
    logger.info("\nRunning backtest with independent positions...")

    capital_allocation = {
        "RANGE": args.range_alloc,
        "TREND": args.trend_alloc,
    }

    results = run_backtest_independent_positions(
        df_3m_full, df_15m_clean, hmm_model, features_15m, features_3m, df_3m_hmm,
        capital_allocation=capital_allocation,
    )

    # Print results
    logger.info("\n" + "=" * 70)
    logger.info("RESULTS - Independent Positions")
    logger.info("=" * 70)

    logger.info(f"\nRANGE Strategy (Capital: ${10000 * args.range_alloc:.0f}):")
    logger.info(f"  Trades: {results['range']['n_trades']}")
    logger.info(f"  Win Rate: {results['range']['win_rate']:.1f}%")
    logger.info(f"  Total PnL: ${results['range']['total_pnl']:.2f}")
    logger.info(f"  Profit Factor: {results['range']['profit_factor']:.2f}")
    if results['range'].get('regime_stats'):
        for regime, stats in results['range']['regime_stats'].items():
            logger.info(f"    {regime}: {stats['n_trades']} trades, WR={stats['win_rate']:.1f}%, PnL=${stats['total_pnl']:.2f}")

    logger.info(f"\nTREND Strategy (Capital: ${10000 * args.trend_alloc:.0f}):")
    logger.info(f"  Trades: {results['trend']['n_trades']}")
    logger.info(f"  Win Rate: {results['trend']['win_rate']:.1f}%")
    logger.info(f"  Total PnL: ${results['trend']['total_pnl']:.2f}")
    logger.info(f"  Profit Factor: {results['trend']['profit_factor']:.2f}")
    if results['trend'].get('regime_stats'):
        for regime, stats in results['trend']['regime_stats'].items():
            logger.info(f"    {regime}: {stats['n_trades']} trades, WR={stats['win_rate']:.1f}%, PnL=${stats['total_pnl']:.2f}")

    logger.info(f"\nCOMBINED (Total Capital: $10000):")
    logger.info(f"  Total Trades: {results['combined']['n_trades']}")
    logger.info(f"  Win Rate: {results['combined']['win_rate']:.1f}%")
    logger.info(f"  Total PnL: ${results['combined']['total_pnl']:.2f}")
    logger.info(f"  Profit Factor: {results['combined']['profit_factor']:.2f}")
    logger.info(f"  Final Equity: ${results['equity']['total']:.2f}")

    # Compare with previous results
    logger.info("\n" + "=" * 70)
    logger.info("COMPARISON with Previous Tests")
    logger.info("=" * 70)

    print(f"\n{'Configuration':<35} {'Trades':>10} {'Win Rate':>10} {'Total PnL':>12} {'PF':>8}")
    print("-" * 75)
    print(f"{'Previous: TREND only':<35} {'1490':>10} {'49.3%':>10} {'$195.32':>12} {'1.74':>8}")
    print(f"{'Previous: All Regimes (shared pos)':<35} {'2468':>10} {'52.1%':>10} {'$127.64':>12} {'1.24':>8}")
    print(f"{'NEW: Independent Positions':<35} {results['combined']['n_trades']:>10} {results['combined']['win_rate']:>9.1f}% ${results['combined']['total_pnl']:>10.2f} {results['combined']['profit_factor']:>8.2f}")

    # Save results
    output_dir = Path("outputs/independent_positions_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_data = {
        "test_period": f"{args.start} ~ {args.end}",
        "capital_allocation": capital_allocation,
        "range_results": results["range"],
        "trend_results": results["trend"],
        "combined_results": results["combined"],
        "final_equity": results["equity"],
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(output_data, f, indent=2, default=str)

    logger.info(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
