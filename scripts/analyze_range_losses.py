#!/usr/bin/env python
"""Analyze RANGE strategy losses in detail.

Investigates why RANGE has 55% win rate but PF=0.67 (losing money).
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

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


def run_range_analysis(
    df_3m: pd.DataFrame,
    df_15m: pd.DataFrame,
    hmm_model: MultiHMM,
    features_15m: np.ndarray,
    features_3m: np.ndarray,
    df_3m_hmm: pd.DataFrame,
    initial_capital: float = 10000.0,
    fee_rate: float = 0.001,
) -> Dict:
    """Run RANGE-only backtest with detailed trade tracking."""

    confirm_layer = RegimeConfirmLayer(ConfirmConfig(
        HIGH_VOL_LAMBDA_ON=2.0,
        HIGH_VOL_COOLDOWN_BARS_15M=6,
    ))

    # Enable RANGE only
    fsm_config = FSMConfig(
        RANGE_ENABLED=True,
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

    # Track entry details
    entry_info = {
        "regime": None,
        "signal": None,
        "reason": None,
        "support": None,
        "resistance": None,
        "mid": None,
        "dist_to_zone": None,
        "zone_strength": None,
        "atr_at_entry": None,
        "volatility_at_entry": None,
    }

    # Only process RANGE regime trades
    range_signals_generated = 0
    range_entries = 0

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

        # Calculate zone metrics
        zone_width = resistance - support
        zone_width_atr = zone_width / atr_3m if atr_3m > 0 else 0
        mid = (support + resistance) / 2
        zone_strength = bar.get("zone_strength", 0.0001)

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
                "strength": zone_strength,
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

        # Track RANGE signals
        if confirm_result.confirmed_regime == "RANGE" and candidate_signal.signal not in ["HOLD", "EXIT"]:
            range_signals_generated += 1

        decision, engine_state = decision_engine.decide(
            ctx=market_ctx,
            confirm=confirm_ctx,
            pos=position,
            cand=candidate_signal,
            engine_state=engine_state,
        )

        # Only track RANGE regime trades
        if confirm_result.confirmed_regime != "RANGE" and position.side == "FLAT":
            continue

        position_size_usd = equity * 0.95

        if decision.action == "OPEN_LONG" and position.side == "FLAT" and confirm_result.confirmed_regime == "RANGE":
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
            entry_info = {
                "regime": "RANGE",
                "signal": candidate_signal.signal,
                "reason": candidate_signal.reason,
                "support": support,
                "resistance": resistance,
                "mid": mid,
                "zone_width_atr": zone_width_atr,
                "dist_to_support": dist_to_support,
                "dist_to_resistance": dist_to_resistance,
                "zone_strength": zone_strength,
                "atr_at_entry": atr_3m,
                "volatility_at_entry": volatility,
                "price_position": (price - support) / zone_width if zone_width > 0 else 0.5,
            }
            range_entries += 1

        elif decision.action == "OPEN_SHORT" and position.side == "FLAT" and confirm_result.confirmed_regime == "RANGE":
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
            entry_info = {
                "regime": "RANGE",
                "signal": candidate_signal.signal,
                "reason": candidate_signal.reason,
                "support": support,
                "resistance": resistance,
                "mid": mid,
                "zone_width_atr": zone_width_atr,
                "dist_to_support": dist_to_support,
                "dist_to_resistance": dist_to_resistance,
                "zone_strength": zone_strength,
                "atr_at_entry": atr_3m,
                "volatility_at_entry": volatility,
                "price_position": (price - support) / zone_width if zone_width > 0 else 0.5,
            }
            range_entries += 1

        elif decision.action == "CLOSE" and position.side != "FLAT":
            exit_price = price
            if position.side == "LONG":
                pnl = (exit_price - position.entry_price) * position.size
                price_move_pct = (exit_price - position.entry_price) / position.entry_price * 100
            else:
                pnl = (position.entry_price - exit_price) * position.size
                price_move_pct = (position.entry_price - exit_price) / position.entry_price * 100
            fee = exit_price * position.size * fee_rate
            pnl -= fee
            equity += pnl

            # Determine exit reason
            exit_reason = "UNKNOWN"
            if candidate_signal.signal == "EXIT":
                exit_reason = candidate_signal.reason
            elif decision.reason:
                exit_reason = decision.reason

            # Calculate how far price moved toward target
            if entry_info["mid"] and entry_info["support"] and entry_info["resistance"]:
                if position.side == "LONG":
                    target_dist = entry_info["mid"] - position.entry_price
                    actual_dist = exit_price - position.entry_price
                else:
                    target_dist = position.entry_price - entry_info["mid"]
                    actual_dist = position.entry_price - exit_price
                target_reached_pct = (actual_dist / target_dist * 100) if target_dist != 0 else 0
            else:
                target_reached_pct = 0

            trades.append({
                "side": position.side,
                "entry_price": position.entry_price,
                "exit_price": exit_price,
                "size": position.size,
                "pnl": pnl,
                "pnl_pct": pnl / (position.entry_price * position.size) * 100 if position.size > 0 else 0,
                "price_move_pct": price_move_pct,
                "bars_held": position.bars_held_3m,
                "exit_reason": exit_reason,
                "signal": entry_info["signal"],
                "support": entry_info["support"],
                "resistance": entry_info["resistance"],
                "mid": entry_info["mid"],
                "zone_width_atr": entry_info["zone_width_atr"],
                "dist_to_zone": entry_info["dist_to_support"] if position.side == "LONG" else entry_info["dist_to_resistance"],
                "zone_strength": entry_info["zone_strength"],
                "atr_at_entry": entry_info["atr_at_entry"],
                "volatility_at_entry": entry_info["volatility_at_entry"],
                "price_position": entry_info["price_position"],
                "target_reached_pct": target_reached_pct,
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
            "exit_reason": "EOD",
            "signal": entry_info.get("signal"),
        })

    return {
        "trades": trades,
        "range_signals_generated": range_signals_generated,
        "range_entries": range_entries,
        "final_equity": equity,
    }


def analyze_trades(trades: List[Dict]) -> Dict:
    """Analyze trade statistics."""
    if not trades:
        return {}

    df = pd.DataFrame(trades)

    wins = df[df["pnl"] > 0]
    losses = df[df["pnl"] <= 0]

    analysis = {
        "total_trades": len(df),
        "win_rate": len(wins) / len(df) * 100,
        "total_pnl": df["pnl"].sum(),
        "profit_factor": wins["pnl"].sum() / abs(losses["pnl"].sum()) if len(losses) > 0 and losses["pnl"].sum() != 0 else float("inf"),

        # Win/Loss size comparison
        "avg_win_pnl": wins["pnl"].mean() if len(wins) > 0 else 0,
        "avg_loss_pnl": losses["pnl"].mean() if len(losses) > 0 else 0,
        "avg_win_pct": wins["pnl_pct"].mean() if len(wins) > 0 else 0,
        "avg_loss_pct": losses["pnl_pct"].mean() if len(losses) > 0 else 0,
        "max_win": wins["pnl"].max() if len(wins) > 0 else 0,
        "max_loss": losses["pnl"].min() if len(losses) > 0 else 0,

        # Holding period
        "avg_bars_held": df["bars_held"].mean(),
        "avg_bars_winners": wins["bars_held"].mean() if len(wins) > 0 else 0,
        "avg_bars_losers": losses["bars_held"].mean() if len(losses) > 0 else 0,

        # By signal type
        "signal_breakdown": {},

        # By exit reason
        "exit_breakdown": {},

        # Zone analysis
        "zone_analysis": {},
    }

    # Signal breakdown
    for signal in df["signal"].dropna().unique():
        signal_trades = df[df["signal"] == signal]
        signal_wins = signal_trades[signal_trades["pnl"] > 0]
        analysis["signal_breakdown"][signal] = {
            "n_trades": len(signal_trades),
            "win_rate": len(signal_wins) / len(signal_trades) * 100 if len(signal_trades) > 0 else 0,
            "total_pnl": signal_trades["pnl"].sum(),
            "avg_pnl": signal_trades["pnl"].mean(),
        }

    # Exit reason breakdown
    for reason in df["exit_reason"].dropna().unique():
        reason_trades = df[df["exit_reason"] == reason]
        reason_wins = reason_trades[reason_trades["pnl"] > 0]
        analysis["exit_breakdown"][reason] = {
            "n_trades": len(reason_trades),
            "win_rate": len(reason_wins) / len(reason_trades) * 100 if len(reason_trades) > 0 else 0,
            "total_pnl": reason_trades["pnl"].sum(),
            "avg_pnl": reason_trades["pnl"].mean(),
        }

    # Zone width analysis
    if "zone_width_atr" in df.columns:
        df_with_zone = df[df["zone_width_atr"].notna()]
        if len(df_with_zone) > 0:
            analysis["zone_analysis"]["avg_zone_width_atr"] = df_with_zone["zone_width_atr"].mean()
            analysis["zone_analysis"]["avg_zone_width_atr_winners"] = wins["zone_width_atr"].mean() if "zone_width_atr" in wins.columns and len(wins) > 0 else 0
            analysis["zone_analysis"]["avg_zone_width_atr_losers"] = losses["zone_width_atr"].mean() if "zone_width_atr" in losses.columns and len(losses) > 0 else 0

    # Volatility analysis
    if "volatility_at_entry" in df.columns:
        df_with_vol = df[df["volatility_at_entry"].notna()]
        if len(df_with_vol) > 0:
            analysis["zone_analysis"]["avg_volatility"] = df_with_vol["volatility_at_entry"].mean()
            analysis["zone_analysis"]["avg_volatility_winners"] = wins["volatility_at_entry"].mean() if "volatility_at_entry" in wins.columns and len(wins) > 0 else 0
            analysis["zone_analysis"]["avg_volatility_losers"] = losses["volatility_at_entry"].mean() if "volatility_at_entry" in losses.columns and len(losses) > 0 else 0

    # Target reached analysis
    if "target_reached_pct" in df.columns:
        df_with_target = df[df["target_reached_pct"].notna()]
        if len(df_with_target) > 0:
            analysis["zone_analysis"]["avg_target_reached_pct"] = df_with_target["target_reached_pct"].mean()
            analysis["zone_analysis"]["avg_target_reached_winners"] = wins["target_reached_pct"].mean() if "target_reached_pct" in wins.columns and len(wins) > 0 else 0
            analysis["zone_analysis"]["avg_target_reached_losers"] = losses["target_reached_pct"].mean() if "target_reached_pct" in losses.columns and len(losses) > 0 else 0

    return analysis


def main():
    parser = argparse.ArgumentParser(description="Analyze RANGE strategy losses")
    parser.add_argument("--start", default="2024-07-01", help="Start date")
    parser.add_argument("--end", default="2026-01-15", help="End date")
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("RANGE Strategy Loss Analysis")
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

    # Run RANGE analysis
    logger.info("\nRunning RANGE-only backtest with detailed tracking...")
    results = run_range_analysis(
        df_3m_full, df_15m_clean, hmm_model, features_15m, features_3m, df_3m_hmm,
    )

    trades = results["trades"]
    analysis = analyze_trades(trades)

    # Print results
    logger.info("\n" + "=" * 70)
    logger.info("RANGE STRATEGY ANALYSIS")
    logger.info("=" * 70)

    logger.info(f"\nBasic Statistics:")
    logger.info(f"  Total Trades: {analysis.get('total_trades', 0)}")
    logger.info(f"  Win Rate: {analysis.get('win_rate', 0):.1f}%")
    logger.info(f"  Total PnL: ${analysis.get('total_pnl', 0):.2f}")
    logger.info(f"  Profit Factor: {analysis.get('profit_factor', 0):.2f}")

    logger.info(f"\n" + "=" * 70)
    logger.info("KEY FINDING: Win/Loss Size Comparison")
    logger.info("=" * 70)
    logger.info(f"  Average Win: ${analysis.get('avg_win_pnl', 0):.2f} ({analysis.get('avg_win_pct', 0):.2f}%)")
    logger.info(f"  Average Loss: ${analysis.get('avg_loss_pnl', 0):.2f} ({analysis.get('avg_loss_pct', 0):.2f}%)")
    logger.info(f"  Max Win: ${analysis.get('max_win', 0):.2f}")
    logger.info(f"  Max Loss: ${analysis.get('max_loss', 0):.2f}")

    # Win/Loss ratio
    avg_win = abs(analysis.get('avg_win_pct', 0))
    avg_loss = abs(analysis.get('avg_loss_pct', 0))
    if avg_loss > 0:
        win_loss_ratio = avg_win / avg_loss
        logger.info(f"  Win/Loss Ratio: {win_loss_ratio:.2f} (need > 1.0 for 50% WR)")
        logger.info(f"  Required WR for breakeven: {100 / (1 + win_loss_ratio):.1f}%")

    logger.info(f"\n" + "=" * 70)
    logger.info("Holding Period Analysis")
    logger.info("=" * 70)
    logger.info(f"  Avg Bars Held (All): {analysis.get('avg_bars_held', 0):.1f}")
    logger.info(f"  Avg Bars Held (Winners): {analysis.get('avg_bars_winners', 0):.1f}")
    logger.info(f"  Avg Bars Held (Losers): {analysis.get('avg_bars_losers', 0):.1f}")

    logger.info(f"\n" + "=" * 70)
    logger.info("Signal Type Breakdown")
    logger.info("=" * 70)
    for signal, stats in analysis.get("signal_breakdown", {}).items():
        logger.info(f"  {signal}:")
        logger.info(f"    Trades: {stats['n_trades']}, WR: {stats['win_rate']:.1f}%, PnL: ${stats['total_pnl']:.2f}")

    logger.info(f"\n" + "=" * 70)
    logger.info("Exit Reason Breakdown")
    logger.info("=" * 70)
    for reason, stats in analysis.get("exit_breakdown", {}).items():
        logger.info(f"  {reason}:")
        logger.info(f"    Trades: {stats['n_trades']}, WR: {stats['win_rate']:.1f}%, PnL: ${stats['total_pnl']:.2f}, Avg: ${stats['avg_pnl']:.2f}")

    logger.info(f"\n" + "=" * 70)
    logger.info("Zone Analysis")
    logger.info("=" * 70)
    zone = analysis.get("zone_analysis", {})
    if zone:
        logger.info(f"  Avg Zone Width (ATR): {zone.get('avg_zone_width_atr', 0):.2f}")
        logger.info(f"    Winners: {zone.get('avg_zone_width_atr_winners', 0):.2f}")
        logger.info(f"    Losers: {zone.get('avg_zone_width_atr_losers', 0):.2f}")
        logger.info(f"  Avg Volatility at Entry: {zone.get('avg_volatility', 0):.4f}")
        logger.info(f"    Winners: {zone.get('avg_volatility_winners', 0):.4f}")
        logger.info(f"    Losers: {zone.get('avg_volatility_losers', 0):.4f}")
        logger.info(f"  Avg Target Reached %: {zone.get('avg_target_reached_pct', 0):.1f}%")
        logger.info(f"    Winners: {zone.get('avg_target_reached_winners', 0):.1f}%")
        logger.info(f"    Losers: {zone.get('avg_target_reached_losers', 0):.1f}%")

    # Root cause analysis
    logger.info(f"\n" + "=" * 70)
    logger.info("ROOT CAUSE ANALYSIS")
    logger.info("=" * 70)

    if avg_loss > 0:
        win_loss_ratio = avg_win / avg_loss
        win_rate = analysis.get('win_rate', 0)

        logger.info(f"\n  Problem: Win/Loss Ratio = {win_loss_ratio:.2f}")
        logger.info(f"  With {win_rate:.1f}% win rate, need W/L ratio > {(100 - win_rate) / win_rate:.2f} to profit")

        if win_loss_ratio < 1:
            logger.info(f"\n  DIAGNOSIS: Losses are {1/win_loss_ratio:.1f}x larger than wins")
            logger.info(f"  Despite {win_rate:.1f}% win rate, large losses overwhelm small wins")

    # Save detailed trades for further analysis
    output_dir = Path("outputs/range_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    if trades:
        pd.DataFrame(trades).to_csv(output_dir / "range_trades.csv", index=False)
        logger.info(f"\nDetailed trades saved to {output_dir / 'range_trades.csv'}")


if __name__ == "__main__":
    main()
