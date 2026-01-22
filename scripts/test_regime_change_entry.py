#!/usr/bin/env python
"""Test regime change entry signals.

Compare two strategies:
1. Current: FSM pullback-based entry (wait for pullback after regime detected)
2. New: Regime change entry (enter immediately when regime changes)

Usage:
    python scripts/test_regime_change_entry.py
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

from xrp4.regime.confirm import RegimeConfirmLayer, ConfirmConfig
from xrp4.core.types import ConfirmContext, MarketContext, PositionState
from xrp4.core.fsm import TradingFSM, FSMConfig
from xrp4.core.decision_engine import DecisionEngine, DecisionConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================================
# Data Loading (reuse from backtest_fsm_pipeline)
# ============================================================================

def load_data(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load 1m, 3m, 15m data from parquet files."""
    df_1m = pd.read_parquet(data_dir / "df_1m.parquet")
    df_3m = pd.read_parquet(data_dir / "df_3m.parquet")
    df_15m = pd.read_parquet(data_dir / "df_15m.parquet")

    logger.info(f"Loaded data: 1m={len(df_1m)}, 3m={len(df_3m)}, 15m={len(df_15m)}")
    return df_1m, df_3m, df_15m


def load_hmm_predictions(data_dir: Path, df_3m: pd.DataFrame) -> pd.DataFrame:
    """Load or compute HMM predictions."""
    hmm_path = data_dir / "hmm_predictions.parquet"

    if hmm_path.exists():
        df_hmm = pd.read_parquet(hmm_path)
        logger.info(f"Loaded HMM predictions: {len(df_hmm)} rows")

        # Merge with 3m data
        df_3m = df_3m.copy()
        df_3m = df_3m.merge(
            df_hmm[["timestamp", "regime_raw", "confidence"]],
            on="timestamp",
            how="left"
        )
        df_3m[["regime_raw", "confidence"]] = df_3m[["regime_raw", "confidence"]].ffill()
        return df_3m.dropna(subset=["regime_raw"])
    else:
        raise FileNotFoundError(f"HMM predictions not found at {hmm_path}")


# ============================================================================
# Regime Change Entry Strategy
# ============================================================================

class RegimeChangeStrategy:
    """Strategy that enters on regime changes."""

    def __init__(
        self,
        initial_capital: float = 10000.0,
        leverage: float = 5.0,
        fee_rate: float = 0.0004,
        trailing_stop_pct: float = 0.10,  # 0.10% trailing stop
        max_hold_bars: int = 20,  # Max hold time in 3m bars (1 hour)
    ):
        self.initial_capital = initial_capital
        self.leverage = leverage
        self.fee_rate = fee_rate
        self.trailing_stop_pct = trailing_stop_pct
        self.max_hold_bars = max_hold_bars

        self.equity = initial_capital
        self.position = PositionState(side="FLAT")
        self.trades: List[Dict] = []

        self.prev_regime: Optional[str] = None
        self.max_unrealized_pnl_pct = 0.0

    def reset(self):
        """Reset strategy state."""
        self.equity = self.initial_capital
        self.position = PositionState(side="FLAT")
        self.trades = []
        self.prev_regime = None
        self.max_unrealized_pnl_pct = 0.0

    def on_bar(
        self,
        ts: pd.Timestamp,
        price: float,
        high: float,
        low: float,
        regime: str,
        confidence: float,
    ) -> Optional[str]:
        """Process a bar and return action taken."""
        action = None

        # Track regime change
        regime_changed = self.prev_regime is not None and regime != self.prev_regime

        # Update position
        if self.position.side != "FLAT":
            self.position.bars_held_3m += 1

            # Calculate unrealized PnL
            if self.position.side == "LONG":
                unrealized_pnl_pct = (price - self.position.entry_price) / self.position.entry_price * 100
                # Check high for max profit
                max_pnl_pct = (high - self.position.entry_price) / self.position.entry_price * 100
            else:
                unrealized_pnl_pct = (self.position.entry_price - price) / self.position.entry_price * 100
                # Check low for max profit
                max_pnl_pct = (self.position.entry_price - low) / self.position.entry_price * 100

            self.max_unrealized_pnl_pct = max(self.max_unrealized_pnl_pct, max_pnl_pct)

            # Exit conditions
            should_exit = False
            exit_reason = ""
            exit_price = price

            # 1. Trailing stop (preserve profit)
            if self.max_unrealized_pnl_pct >= self.trailing_stop_pct:
                preserve_pct = self.max_unrealized_pnl_pct * 0.6  # Preserve 60%
                if unrealized_pnl_pct <= preserve_pct:
                    should_exit = True
                    exit_reason = f"TRAILING_STOP (max={self.max_unrealized_pnl_pct:.2f}%)"

            # 2. Regime reversal
            if regime_changed:
                if self.position.side == "LONG" and regime == "TREND_DOWN":
                    should_exit = True
                    exit_reason = "REGIME_REVERSAL_DOWN"
                elif self.position.side == "SHORT" and regime == "TREND_UP":
                    should_exit = True
                    exit_reason = "REGIME_REVERSAL_UP"

            # 3. Max hold time
            if self.position.bars_held_3m >= self.max_hold_bars:
                should_exit = True
                exit_reason = "MAX_HOLD_TIME"

            # 4. Stop loss (-0.5%)
            if unrealized_pnl_pct <= -0.5:
                should_exit = True
                exit_reason = "STOP_LOSS"

            if should_exit:
                self._close_position(ts, exit_price, exit_reason)
                action = f"CLOSE_{self.trades[-1]['side']}"

        # Entry on regime change (only if flat)
        if self.position.side == "FLAT" and regime_changed and confidence >= 0.5:
            # RANGE -> TREND_UP: Enter LONG
            if self.prev_regime == "RANGE" and regime == "TREND_UP":
                self._open_position(ts, price, "LONG", "RANGE_TO_TREND_UP")
                action = "OPEN_LONG"

            # RANGE -> TREND_DOWN: Enter SHORT
            elif self.prev_regime == "RANGE" and regime == "TREND_DOWN":
                self._open_position(ts, price, "SHORT", "RANGE_TO_TREND_DOWN")
                action = "OPEN_SHORT"

            # TREND_DOWN -> TREND_UP: Enter LONG (reversal)
            elif self.prev_regime == "TREND_DOWN" and regime == "TREND_UP":
                self._open_position(ts, price, "LONG", "TREND_REVERSAL_UP")
                action = "OPEN_LONG"

            # TREND_UP -> TREND_DOWN: Enter SHORT (reversal)
            elif self.prev_regime == "TREND_UP" and regime == "TREND_DOWN":
                self._open_position(ts, price, "SHORT", "TREND_REVERSAL_DOWN")
                action = "OPEN_SHORT"

        self.prev_regime = regime
        return action

    def _open_position(self, ts: pd.Timestamp, price: float, side: str, reason: str):
        """Open a new position."""
        position_value = self.equity * self.leverage
        size = position_value / price
        fee = price * size * self.fee_rate
        self.equity -= fee

        self.position = PositionState(
            side=side,
            entry_price=price,
            size=size,
            entry_ts=int(ts.timestamp() * 1000),
            bars_held_3m=0,
        )
        self.position.entry_reason = reason
        self.max_unrealized_pnl_pct = 0.0

    def _close_position(self, ts: pd.Timestamp, price: float, reason: str):
        """Close current position."""
        if self.position.side == "LONG":
            pnl = (price - self.position.entry_price) * self.position.size
        else:
            pnl = (self.position.entry_price - price) * self.position.size

        fee = price * self.position.size * self.fee_rate
        pnl -= fee
        self.equity += pnl

        pnl_pct = pnl / (self.position.entry_price * self.position.size) * 100

        self.trades.append({
            "entry_ts": pd.Timestamp(self.position.entry_ts, unit="ms"),
            "exit_ts": ts,
            "side": self.position.side,
            "entry_price": self.position.entry_price,
            "exit_price": price,
            "size": self.position.size,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "bars_held": self.position.bars_held_3m,
            "entry_reason": getattr(self.position, "entry_reason", ""),
            "exit_reason": reason,
            "max_profit_pct": self.max_unrealized_pnl_pct,
        })

        self.position = PositionState(side="FLAT")
        self.max_unrealized_pnl_pct = 0.0

    def get_results(self) -> Dict:
        """Calculate strategy results."""
        if not self.trades:
            return {
                "n_trades": 0,
                "win_rate": 0,
                "total_pnl": 0,
                "return_pct": 0,
                "profit_factor": 0,
                "avg_pnl": 0,
                "avg_win": 0,
                "avg_loss": 0,
            }

        df = pd.DataFrame(self.trades)
        n_trades = len(df)
        wins = df[df["pnl"] > 0]
        losses = df[df["pnl"] <= 0]

        total_win = wins["pnl"].sum() if len(wins) > 0 else 0
        total_loss = abs(losses["pnl"].sum()) if len(losses) > 0 else 0

        return {
            "n_trades": n_trades,
            "win_rate": len(wins) / n_trades * 100 if n_trades > 0 else 0,
            "total_pnl": df["pnl"].sum(),
            "return_pct": (self.equity / self.initial_capital - 1) * 100,
            "profit_factor": total_win / total_loss if total_loss > 0 else float("inf"),
            "avg_pnl": df["pnl"].mean(),
            "avg_win": wins["pnl"].mean() if len(wins) > 0 else 0,
            "avg_loss": losses["pnl"].mean() if len(losses) > 0 else 0,
            "avg_bars_held": df["bars_held"].mean(),
        }


# ============================================================================
# FSM Strategy (Current Approach)
# ============================================================================

class FSMStrategy:
    """Current FSM-based pullback strategy."""

    def __init__(
        self,
        initial_capital: float = 10000.0,
        leverage: float = 5.0,
        fee_rate: float = 0.0004,
    ):
        self.initial_capital = initial_capital
        self.leverage = leverage
        self.fee_rate = fee_rate

        self.equity = initial_capital
        self.position = PositionState(side="FLAT")
        self.trades: List[Dict] = []

        # FSM components
        self.confirm_layer = RegimeConfirmLayer(ConfirmConfig())
        self.fsm = TradingFSM(FSMConfig())
        self.decision_engine = DecisionEngine(DecisionConfig())

        self.confirm_state = None
        self.fsm_state = None
        self.engine_state = None

        self.max_unrealized_pnl_pct = 0.0

    def reset(self):
        """Reset strategy state."""
        self.equity = self.initial_capital
        self.position = PositionState(side="FLAT")
        self.trades = []
        self.confirm_state = None
        self.fsm_state = None
        self.engine_state = None
        self.max_unrealized_pnl_pct = 0.0

    def on_bar(
        self,
        ts: pd.Timestamp,
        bar_3m: pd.Series,
        bar_15m: pd.Series,
        hist_15m: pd.DataFrame,
        regime_raw: str,
    ) -> Optional[str]:
        """Process a bar using FSM pipeline."""
        action = None
        price = float(bar_3m["close"])

        # Step 1: Confirm regime
        row_15m_dict = bar_15m.to_dict()
        row_15m_dict["ret_3m"] = bar_3m.get("ret_3m", 0)

        confirm_result, self.confirm_state = self.confirm_layer.confirm(
            regime_raw=regime_raw,
            row_15m=row_15m_dict,
            hist_15m=hist_15m,
            state=self.confirm_state,
        )

        confirm_ctx = ConfirmContext(
            regime_raw=regime_raw,
            regime_confirmed=confirm_result.confirmed_regime,
            confirm_reason=confirm_result.reason,
            confirm_metrics=confirm_result.metrics,
        )

        # Build market context
        market_ctx = MarketContext(
            symbol="XRPUSDT",
            ts=int(ts.timestamp() * 1000),
            price=price,
            row_3m={
                "close": price,
                "high": float(bar_3m.get("high", price)),
                "low": float(bar_3m.get("low", price)),
                "atr_3m": bar_3m.get("atr_3m", 0.01),
                "ema_fast_3m": bar_3m.get("ema_fast_3m", price),
                "ema_slow_3m": bar_3m.get("ema_slow_3m", price),
                "ret_3m": bar_3m.get("ret_3m", 0),
                "rsi_3m": bar_3m.get("rsi_3m", 50),
            },
            row_15m={
                "ema_slope_15m": bar_15m.get("ema_slope_15m", 0),
                "ewm_ret_15m": bar_15m.get("ewm_ret_15m", 0),
            },
            zone={
                "support": price - 0.01,
                "resistance": price + 0.01,
                "strength": 0.0001,
                "dist_to_support": 1.0,
                "dist_to_resistance": 1.0,
            },
        )

        # Update position
        if self.position.side != "FLAT":
            self.position.bars_held_3m += 1
            if self.position.side == "LONG":
                self.position.unrealized_pnl = (price - self.position.entry_price) * self.position.size
                pnl_pct = (price - self.position.entry_price) / self.position.entry_price * 100
            else:
                self.position.unrealized_pnl = (self.position.entry_price - price) * self.position.size
                pnl_pct = (self.position.entry_price - price) / self.position.entry_price * 100
            self.max_unrealized_pnl_pct = max(self.max_unrealized_pnl_pct, pnl_pct)

        # Step 2: FSM
        candidate, self.fsm_state = self.fsm.step(
            ctx=market_ctx,
            confirm=confirm_ctx,
            pos=self.position,
            fsm_state=self.fsm_state,
        )

        # Step 3: Decision Engine
        decision, self.engine_state = self.decision_engine.decide(
            ctx=market_ctx,
            confirm=confirm_ctx,
            pos=self.position,
            cand=candidate,
            engine_state=self.engine_state,
        )

        # Execute
        if decision.action == "OPEN_LONG" and self.position.side == "FLAT":
            self._open_position(ts, price, "LONG", candidate.signal)
            action = "OPEN_LONG"
        elif decision.action == "OPEN_SHORT" and self.position.side == "FLAT":
            self._open_position(ts, price, "SHORT", candidate.signal)
            action = "OPEN_SHORT"
        elif decision.action == "CLOSE" and self.position.side != "FLAT":
            self._close_position(ts, price, decision.reason)
            action = f"CLOSE_{self.trades[-1]['side']}"

        return action

    def _open_position(self, ts: pd.Timestamp, price: float, side: str, reason: str):
        position_value = self.equity * self.leverage
        size = position_value / price
        fee = price * size * self.fee_rate
        self.equity -= fee

        self.position = PositionState(
            side=side,
            entry_price=price,
            size=size,
            entry_ts=int(ts.timestamp() * 1000),
            bars_held_3m=0,
        )
        self.position.entry_reason = reason
        self.max_unrealized_pnl_pct = 0.0

    def _close_position(self, ts: pd.Timestamp, price: float, reason: str):
        if self.position.side == "LONG":
            pnl = (price - self.position.entry_price) * self.position.size
        else:
            pnl = (self.position.entry_price - price) * self.position.size

        fee = price * self.position.size * self.fee_rate
        pnl -= fee
        self.equity += pnl

        pnl_pct = pnl / (self.position.entry_price * self.position.size) * 100

        self.trades.append({
            "entry_ts": pd.Timestamp(self.position.entry_ts, unit="ms"),
            "exit_ts": ts,
            "side": self.position.side,
            "entry_price": self.position.entry_price,
            "exit_price": price,
            "size": self.position.size,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "bars_held": self.position.bars_held_3m,
            "entry_reason": getattr(self.position, "entry_reason", ""),
            "exit_reason": reason,
            "max_profit_pct": self.max_unrealized_pnl_pct,
        })

        self.position = PositionState(side="FLAT")
        self.max_unrealized_pnl_pct = 0.0

    def get_results(self) -> Dict:
        if not self.trades:
            return {
                "n_trades": 0,
                "win_rate": 0,
                "total_pnl": 0,
                "return_pct": 0,
                "profit_factor": 0,
            }

        df = pd.DataFrame(self.trades)
        n_trades = len(df)
        wins = df[df["pnl"] > 0]
        losses = df[df["pnl"] <= 0]

        total_win = wins["pnl"].sum() if len(wins) > 0 else 0
        total_loss = abs(losses["pnl"].sum()) if len(losses) > 0 else 0

        return {
            "n_trades": n_trades,
            "win_rate": len(wins) / n_trades * 100 if n_trades > 0 else 0,
            "total_pnl": df["pnl"].sum(),
            "return_pct": (self.equity / self.initial_capital - 1) * 100,
            "profit_factor": total_win / total_loss if total_loss > 0 else float("inf"),
            "avg_pnl": df["pnl"].mean(),
            "avg_bars_held": df["bars_held"].mean(),
        }


# ============================================================================
# Main Backtest
# ============================================================================

def run_comparison(
    df_3m: pd.DataFrame,
    df_15m: pd.DataFrame,
    initial_capital: float = 10000.0,
    leverage: float = 5.0,
) -> Dict:
    """Run both strategies and compare results."""

    # Initialize strategies
    regime_change_strategy = RegimeChangeStrategy(
        initial_capital=initial_capital,
        leverage=leverage,
    )
    fsm_strategy = FSMStrategy(
        initial_capital=initial_capital,
        leverage=leverage,
    )

    # Prepare data
    df_3m = df_3m.copy()
    df_15m = df_15m.copy()

    # Index 15m data by timestamp for efficient lookup
    df_15m_indexed = df_15m.set_index("timestamp")

    # Warmup period
    warmup = 250

    logger.info(f"Running backtest on {len(df_3m) - warmup} bars...")

    for i in range(warmup, len(df_3m)):
        bar = df_3m.iloc[i]
        ts = bar["timestamp"]
        price = float(bar["close"])
        high = float(bar.get("high", price))
        low = float(bar.get("low", price))

        regime_raw = bar.get("regime_raw", "RANGE")
        confidence = bar.get("confidence", 0.5)

        # Get 15m data
        matching_15m = df_15m_indexed[df_15m_indexed.index <= ts]
        if len(matching_15m) == 0:
            continue
        bar_15m = matching_15m.iloc[-1]
        hist_15m = matching_15m.iloc[-20:]

        # Strategy 1: Regime Change Entry
        regime_change_strategy.on_bar(
            ts=ts,
            price=price,
            high=high,
            low=low,
            regime=regime_raw,
            confidence=confidence,
        )

        # Strategy 2: FSM Pullback Entry
        fsm_strategy.on_bar(
            ts=ts,
            bar_3m=bar,
            bar_15m=bar_15m,
            hist_15m=hist_15m,
            regime_raw=regime_raw,
        )

    # Get results
    regime_results = regime_change_strategy.get_results()
    fsm_results = fsm_strategy.get_results()

    return {
        "regime_change": regime_results,
        "fsm_pullback": fsm_results,
        "regime_trades": regime_change_strategy.trades,
        "fsm_trades": fsm_strategy.trades,
    }


def print_comparison(results: Dict):
    """Print comparison results."""
    regime = results["regime_change"]
    fsm = results["fsm_pullback"]

    print("\n" + "=" * 70)
    print("REGIME CHANGE ENTRY vs FSM PULLBACK ENTRY")
    print("=" * 70)

    print(f"\n{'Metric':<25} {'Regime Change':>20} {'FSM Pullback':>20}")
    print("-" * 70)
    print(f"{'Trades':<25} {regime['n_trades']:>20} {fsm['n_trades']:>20}")
    print(f"{'Win Rate':<25} {regime['win_rate']:>19.1f}% {fsm['win_rate']:>19.1f}%")
    print(f"{'Total PnL':<25} ${regime['total_pnl']:>18.2f} ${fsm['total_pnl']:>18.2f}")
    print(f"{'Return':<25} {regime['return_pct']:>19.2f}% {fsm['return_pct']:>19.2f}%")
    print(f"{'Profit Factor':<25} {regime['profit_factor']:>20.2f} {fsm['profit_factor']:>20.2f}")
    print(f"{'Avg PnL/Trade':<25} ${regime.get('avg_pnl', 0):>18.2f} ${fsm.get('avg_pnl', 0):>18.2f}")
    print(f"{'Avg Bars Held':<25} {regime.get('avg_bars_held', 0):>20.1f} {fsm.get('avg_bars_held', 0):>20.1f}")

    # Regime change entry breakdown
    if results["regime_trades"]:
        print("\n" + "-" * 70)
        print("REGIME CHANGE ENTRY BREAKDOWN:")
        df = pd.DataFrame(results["regime_trades"])

        for reason in df["entry_reason"].unique():
            subset = df[df["entry_reason"] == reason]
            n = len(subset)
            wr = (subset["pnl"] > 0).mean() * 100
            pnl = subset["pnl"].sum()
            print(f"  {reason:<30} {n:>5} trades, WR={wr:>5.1f}%, PnL=${pnl:>8.2f}")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Test regime change entry signals")
    parser.add_argument("--data-dir", type=str, default="outputs/backtest_data",
                        help="Directory with backtest data")
    parser.add_argument("--capital", type=float, default=10000.0,
                        help="Initial capital")
    parser.add_argument("--leverage", type=float, default=5.0,
                        help="Leverage")
    args = parser.parse_args()

    # Load data
    data_dir = Path(args.data_dir)

    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        logger.info("Please run backtest_fsm_pipeline.py first to generate data")
        sys.exit(1)

    try:
        df_1m, df_3m, df_15m = load_data(data_dir)
        df_3m = load_hmm_predictions(data_dir, df_3m)
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    # Run comparison
    results = run_comparison(
        df_3m=df_3m,
        df_15m=df_15m,
        initial_capital=args.capital,
        leverage=args.leverage,
    )

    # Print results
    print_comparison(results)

    # Save detailed results
    output_path = data_dir / "regime_change_comparison.json"
    with open(output_path, "w") as f:
        # Convert timestamps to strings for JSON
        regime_trades = [{**t, "entry_ts": str(t["entry_ts"]), "exit_ts": str(t["exit_ts"])}
                         for t in results["regime_trades"]]
        fsm_trades = [{**t, "entry_ts": str(t["entry_ts"]), "exit_ts": str(t["exit_ts"])}
                      for t in results["fsm_trades"]]
        json.dump({
            "regime_change": results["regime_change"],
            "fsm_pullback": results["fsm_pullback"],
            "regime_trades": regime_trades,
            "fsm_trades": fsm_trades,
        }, f, indent=2)
    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
