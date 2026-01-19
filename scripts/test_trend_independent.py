#!/usr/bin/env python
"""Test TREND only with independent positions for TREND_UP and TREND_DOWN.

Each trend direction has its own position slot.
- TREND_UP: Long positions
- TREND_DOWN: Short positions

Usage:
    python scripts/test_trend_independent.py
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

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


def run_backtest_single_position(
    df_3m: pd.DataFrame,
    df_15m: pd.DataFrame,
    hmm_model: MultiHMM,
    features_15m: np.ndarray,
    features_3m: np.ndarray,
    df_3m_hmm: pd.DataFrame,
    initial_capital: float = 10000.0,
    fee_rate: float = 0.001,
) -> Dict:
    """Run backtest with single position (current behavior)."""

    confirm_layer = RegimeConfirmLayer(ConfirmConfig(
        HIGH_VOL_LAMBDA_ON=2.0,
        HIGH_VOL_COOLDOWN_BARS_15M=6,
    ))
    fsm_config = FSMConfig(
        RANGE_ENABLED=False,
        TRANSITION_ALLOW_BREAKOUT_ONLY=False,
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
    entry_signal = None

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

        # Use consistent position sizing: 95% of capital
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
            entry_signal = candidate_signal.signal

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
            entry_signal = candidate_signal.signal

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

    return calc_metrics(trades, initial_capital, equity)


def run_backtest_independent_trend(
    df_3m: pd.DataFrame,
    df_15m: pd.DataFrame,
    hmm_model: MultiHMM,
    features_15m: np.ndarray,
    features_3m: np.ndarray,
    df_3m_hmm: pd.DataFrame,
    capital_allocation: Dict[str, float] = None,
    fee_rate: float = 0.001,
) -> Dict:
    """Run backtest with independent positions for TREND_UP and TREND_DOWN.

    Args:
        capital_allocation: {"TREND_UP": 0.5, "TREND_DOWN": 0.5}
    """
    if capital_allocation is None:
        capital_allocation = {"TREND_UP": 0.5, "TREND_DOWN": 0.5}

    total_capital = 10000.0

    confirm_layer = RegimeConfirmLayer(ConfirmConfig(
        HIGH_VOL_LAMBDA_ON=2.0,
        HIGH_VOL_COOLDOWN_BARS_15M=6,
    ))
    fsm_config = FSMConfig(
        RANGE_ENABLED=False,
        TRANSITION_ALLOW_BREAKOUT_ONLY=False,
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

    # Independent positions for each trend direction
    positions = {
        "TREND_UP": PositionState(side="FLAT"),
        "TREND_DOWN": PositionState(side="FLAT"),
    }
    equity = {
        "TREND_UP": total_capital * capital_allocation["TREND_UP"],
        "TREND_DOWN": total_capital * capital_allocation["TREND_DOWN"],
    }
    trades = {
        "TREND_UP": [],
        "TREND_DOWN": [],
    }
    entry_info = {
        "TREND_UP": {"regime": None, "signal": None},
        "TREND_DOWN": {"regime": None, "signal": None},
    }

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

        # Determine which position slot to use based on confirmed regime
        current_regime = confirm_result.confirmed_regime
        if current_regime not in ("TREND_UP", "TREND_DOWN"):
            # Update bars held for existing positions
            for rtype in positions:
                pos = positions[rtype]
                if pos.side != "FLAT":
                    pos.bars_held_3m += 1
                    if pos.side == "LONG":
                        pos.unrealized_pnl = (price - pos.entry_price) * pos.size
                    else:
                        pos.unrealized_pnl = (pos.entry_price - price) * pos.size
            continue

        active_position = positions[current_regime]

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

        # Update all positions
        for rtype in positions:
            pos = positions[rtype]
            if pos.side != "FLAT":
                pos.bars_held_3m += 1
                if pos.side == "LONG":
                    pos.unrealized_pnl = (price - pos.entry_price) * pos.size
                else:
                    pos.unrealized_pnl = (pos.entry_price - pos.size) * pos.size

        candidate_signal, fsm_state = fsm.step(
            ctx=market_ctx,
            confirm=confirm_ctx,
            pos=active_position,
            fsm_state=fsm_state,
        )

        decision, engine_state = decision_engine.decide(
            ctx=market_ctx,
            confirm=confirm_ctx,
            pos=active_position,
            cand=candidate_signal,
            engine_state=engine_state,
        )

        pos = positions[current_regime]
        eq = equity[current_regime]
        position_size_usd = eq * 0.95

        if decision.action == "OPEN_LONG" and pos.side == "FLAT":
            entry_price = price
            size = position_size_usd / entry_price
            fee = entry_price * size * fee_rate
            equity[current_regime] -= fee

            positions[current_regime] = PositionState(
                side="LONG",
                entry_price=entry_price,
                size=size,
                entry_ts=market_ctx.ts,
                bars_held_3m=0,
                unrealized_pnl=0,
            )
            entry_info[current_regime] = {
                "regime": current_regime,
                "signal": candidate_signal.signal,
            }

        elif decision.action == "OPEN_SHORT" and pos.side == "FLAT":
            entry_price = price
            size = position_size_usd / entry_price
            fee = entry_price * size * fee_rate
            equity[current_regime] -= fee

            positions[current_regime] = PositionState(
                side="SHORT",
                entry_price=entry_price,
                size=size,
                entry_ts=market_ctx.ts,
                bars_held_3m=0,
                unrealized_pnl=0,
            )
            entry_info[current_regime] = {
                "regime": current_regime,
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
            equity[current_regime] += pnl

            trades[current_regime].append({
                "side": pos.side,
                "entry_price": pos.entry_price,
                "exit_price": exit_price,
                "size": pos.size,
                "pnl": pnl,
                "pnl_pct": pnl / (pos.entry_price * pos.size) * 100 if pos.size > 0 else 0,
                "bars_held": pos.bars_held_3m,
                "regime": entry_info[current_regime]["regime"],
                "signal": entry_info[current_regime]["signal"],
            })
            positions[current_regime] = PositionState(side="FLAT")

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

    all_trades = trades["TREND_UP"] + trades["TREND_DOWN"]
    total_equity = equity["TREND_UP"] + equity["TREND_DOWN"]

    return {
        "trend_up": calc_metrics(trades["TREND_UP"], total_capital * capital_allocation["TREND_UP"], equity["TREND_UP"]),
        "trend_down": calc_metrics(trades["TREND_DOWN"], total_capital * capital_allocation["TREND_DOWN"], equity["TREND_DOWN"]),
        "combined": calc_metrics(all_trades, total_capital, total_equity),
        "equity": {
            "trend_up": equity["TREND_UP"],
            "trend_down": equity["TREND_DOWN"],
            "total": total_equity,
        },
        "capital_allocation": capital_allocation,
    }


def calc_metrics(trade_list, initial_cap, final_equity):
    """Calculate trading metrics."""
    if not trade_list:
        return {
            "n_trades": 0,
            "total_pnl": 0,
            "return_pct": 0,
            "win_rate": 0,
            "profit_factor": 0,
            "avg_win": 0,
            "avg_loss": 0,
            "final_equity": final_equity,
            "regime_stats": {},
        }

    df = pd.DataFrame(trade_list)
    wins = df[df["pnl"] > 0]
    losses = df[df["pnl"] <= 0]

    total_pnl = df["pnl"].sum()

    regime_stats = {}
    for regime in df["regime"].dropna().unique():
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
        "total_pnl": total_pnl,
        "return_pct": (final_equity - initial_cap) / initial_cap * 100,
        "win_rate": len(wins) / len(trade_list) * 100 if trade_list else 0,
        "profit_factor": (
            wins["pnl"].sum() / abs(losses["pnl"].sum())
            if len(losses) > 0 and losses["pnl"].sum() != 0
            else float("inf") if len(wins) > 0 else 0
        ),
        "avg_win": wins["pnl_pct"].mean() if len(wins) > 0 else 0,
        "avg_loss": losses["pnl_pct"].mean() if len(losses) > 0 else 0,
        "final_equity": final_equity,
        "regime_stats": regime_stats,
    }


def main():
    parser = argparse.ArgumentParser(description="Test TREND with independent positions")
    parser.add_argument("--start", default="2024-07-01", help="Test start date")
    parser.add_argument("--end", default="2026-01-15", help="Test end date")
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("TREND Only: Single vs Independent Positions")
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
    # Test 1: Single position (current behavior)
    # ========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("Test 1: Single Position (Current Behavior)")
    logger.info("=" * 70)

    results_single = run_backtest_single_position(
        df_3m_full, df_15m_clean, hmm_model, features_15m, features_3m, df_3m_hmm,
    )

    logger.info(f"\nSingle Position Results:")
    logger.info(f"  Trades: {results_single['n_trades']}")
    logger.info(f"  Win Rate: {results_single['win_rate']:.1f}%")
    logger.info(f"  Total PnL: ${results_single['total_pnl']:.2f}")
    logger.info(f"  Return: {results_single['return_pct']:.2f}%")
    logger.info(f"  Profit Factor: {results_single['profit_factor']:.2f}")
    logger.info(f"  Avg Win: {results_single['avg_win']:.2f}%")
    logger.info(f"  Avg Loss: {results_single['avg_loss']:.2f}%")
    if results_single.get('regime_stats'):
        logger.info("  Regime Breakdown:")
        for regime, stats in results_single['regime_stats'].items():
            logger.info(f"    {regime}: {stats['n_trades']} trades, WR={stats['win_rate']:.1f}%, PnL=${stats['total_pnl']:.2f}")

    # ========================================================================
    # Test 2: Independent positions (TREND_UP and TREND_DOWN separate)
    # ========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("Test 2: Independent Positions (TREND_UP / TREND_DOWN)")
    logger.info("=" * 70)
    logger.info("Capital: TREND_UP=50%, TREND_DOWN=50%")

    results_indep = run_backtest_independent_trend(
        df_3m_full, df_15m_clean, hmm_model, features_15m, features_3m, df_3m_hmm,
        capital_allocation={"TREND_UP": 0.5, "TREND_DOWN": 0.5},
    )

    logger.info(f"\nTREND_UP (Capital: $5000):")
    logger.info(f"  Trades: {results_indep['trend_up']['n_trades']}")
    logger.info(f"  Win Rate: {results_indep['trend_up']['win_rate']:.1f}%")
    logger.info(f"  Total PnL: ${results_indep['trend_up']['total_pnl']:.2f}")
    logger.info(f"  Profit Factor: {results_indep['trend_up']['profit_factor']:.2f}")

    logger.info(f"\nTREND_DOWN (Capital: $5000):")
    logger.info(f"  Trades: {results_indep['trend_down']['n_trades']}")
    logger.info(f"  Win Rate: {results_indep['trend_down']['win_rate']:.1f}%")
    logger.info(f"  Total PnL: ${results_indep['trend_down']['total_pnl']:.2f}")
    logger.info(f"  Profit Factor: {results_indep['trend_down']['profit_factor']:.2f}")

    logger.info(f"\nCOMBINED (Total Capital: $10000):")
    logger.info(f"  Total Trades: {results_indep['combined']['n_trades']}")
    logger.info(f"  Win Rate: {results_indep['combined']['win_rate']:.1f}%")
    logger.info(f"  Total PnL: ${results_indep['combined']['total_pnl']:.2f}")
    logger.info(f"  Profit Factor: {results_indep['combined']['profit_factor']:.2f}")
    logger.info(f"  Final Equity: ${results_indep['equity']['total']:.2f}")

    # ========================================================================
    # Comparison
    # ========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("COMPARISON")
    logger.info("=" * 70)

    print(f"\n{'Configuration':<30} {'Trades':>10} {'Win Rate':>10} {'Total PnL':>15} {'PF':>8}")
    print("-" * 73)
    print(f"{'Single Position':<30} {results_single['n_trades']:>10} {results_single['win_rate']:>9.1f}% ${results_single['total_pnl']:>13.2f} {results_single['profit_factor']:>8.2f}")
    print(f"{'Independent (UP+DOWN)':<30} {results_indep['combined']['n_trades']:>10} {results_indep['combined']['win_rate']:>9.1f}% ${results_indep['combined']['total_pnl']:>13.2f} {results_indep['combined']['profit_factor']:>8.2f}")

    # Difference
    pnl_diff = results_indep['combined']['total_pnl'] - results_single['total_pnl']
    trade_diff = results_indep['combined']['n_trades'] - results_single['n_trades']
    print(f"\n{'Difference':<30} {trade_diff:>+10} {results_indep['combined']['win_rate'] - results_single['win_rate']:>+9.1f}% ${pnl_diff:>+13.2f}")

    # Save results
    output_dir = Path("outputs/trend_independent_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_data = {
        "test_period": f"{args.start} ~ {args.end}",
        "single_position": results_single,
        "independent_positions": {
            "trend_up": results_indep["trend_up"],
            "trend_down": results_indep["trend_down"],
            "combined": results_indep["combined"],
            "equity": results_indep["equity"],
        },
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(output_data, f, indent=2, default=str)

    logger.info(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
