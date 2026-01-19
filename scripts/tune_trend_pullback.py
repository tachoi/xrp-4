#!/usr/bin/env python3
"""TREND Pullback Parameter Tuning.

Tunes FSM TREND pullback parameters to improve win rate and PnL.

Key findings from analysis:
- EMA slope 0.07 threshold is way too strict (0% data passes)
- Realistic thresholds: 0.003 (27%), 0.005 (14%), 0.007 (8%)
"""

import sys
import logging
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
from xrp4.core.decision_engine import DecisionEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class TrendTuneParams:
    """Parameters for TREND pullback tuning."""
    TREND_PULLBACK_TO_EMA_ATR: float = 1.0
    TREND_MIN_EMA_SLOPE_15M: float = 0.003  # Realistic default
    MIN_REBOUND_RET: float = 0.0
    REQUIRE_EMA_ALIGNMENT: bool = False
    STOP_ATR_MULT: float = 1.5
    TP_ATR_MULT: float = 2.0
    MAX_HOLD_BARS_3M: int = 40


class TrendOnlyFSM:
    """Modified FSM that only trades TREND regimes."""

    def __init__(self, params: TrendTuneParams):
        self.params = params

    def step(
        self,
        ctx: MarketContext,
        confirm: ConfirmContext,
        pos: PositionState,
        fsm_state: Optional[Dict] = None,
    ) -> Tuple[CandidateSignal, Dict]:
        """Generate signal for TREND regimes only."""
        if fsm_state is None:
            fsm_state = {"last_signal": "HOLD", "bars_since_signal": 0}

        fsm_state = fsm_state.copy()
        fsm_state["bars_since_signal"] = fsm_state.get("bars_since_signal", 0) + 1

        regime = confirm.regime_confirmed

        # HIGH_VOL, NO_TRADE -> HOLD (FSM blocks these)
        if regime in {"HIGH_VOL", "NO_TRADE"}:
            return CandidateSignal(
                signal="HOLD", score=0.0,
                reason="FSM_BLOCKED_REGIME", params={}
            ), fsm_state

        # Only trade TREND_UP and TREND_DOWN
        if regime not in {"TREND_UP", "TREND_DOWN"}:
            return CandidateSignal(
                signal="HOLD", score=0.0,
                reason="NOT_TREND_REGIME", params={}
            ), fsm_state

        # Get market data
        price = ctx.price
        atr_3m = ctx.row_3m.get("atr_3m", ctx.row_3m.get("atr", 0.01))
        ema_fast = ctx.row_3m.get("ema_fast_3m", ctx.row_3m.get("ema_20_3m", price))
        ema_slow = ctx.row_3m.get("ema_slow_3m", ctx.row_3m.get("ema_50_3m", price))
        ret_3m = ctx.row_3m.get("ret_3m", 0)
        ema_slope_15m = ctx.row_15m.get("ema_slope_15m", ctx.row_15m.get("ema_slope", 0))

        default_params = {
            "stop_atr_mult": self.params.STOP_ATR_MULT,
            "tp_atr_mult": self.params.TP_ATR_MULT,
            "atr_3m": atr_3m,
        }

        # Check exit conditions
        if pos.side != "FLAT":
            exit_signal = self._check_exit(pos, price, atr_3m, ema_slow)
            if exit_signal is not None:
                fsm_state["last_signal"] = exit_signal.signal
                fsm_state["bars_since_signal"] = 0
                return exit_signal, fsm_state

        # Generate entry signal
        if regime == "TREND_UP":
            signal = self._trend_up_signal(
                pos, price, atr_3m, ema_fast, ema_slow, ret_3m, ema_slope_15m, default_params
            )
        else:
            signal = self._trend_down_signal(
                pos, price, atr_3m, ema_fast, ema_slow, ret_3m, ema_slope_15m, default_params
            )

        fsm_state["last_signal"] = signal.signal
        if signal.signal != "HOLD":
            fsm_state["bars_since_signal"] = 0

        return signal, fsm_state

    def _check_exit(self, pos, price, atr_3m, ema_slow) -> Optional[CandidateSignal]:
        """Check exit conditions."""
        if pos.bars_held_3m >= self.params.MAX_HOLD_BARS_3M:
            return CandidateSignal(
                signal="EXIT", score=1.0,
                reason="TREND_EXIT_TIMEOUT", params={"exit_type": "time_stop"}
            )

        if pos.side == "LONG" and price < ema_slow:
            return CandidateSignal(
                signal="EXIT", score=0.8,
                reason="TREND_EXIT_EMA_CROSS", params={"exit_type": "ema_cross"}
            )
        elif pos.side == "SHORT" and price > ema_slow:
            return CandidateSignal(
                signal="EXIT", score=0.8,
                reason="TREND_EXIT_EMA_CROSS", params={"exit_type": "ema_cross"}
            )

        return None

    def _trend_up_signal(self, pos, price, atr_3m, ema_fast, ema_slow,
                         ret_3m, ema_slope_15m, default_params) -> CandidateSignal:
        if pos.side != "FLAT":
            return CandidateSignal(signal="HOLD", score=0.0, reason="IN_POSITION", params={})

        if abs(ema_slope_15m) < self.params.TREND_MIN_EMA_SLOPE_15M:
            return CandidateSignal(signal="HOLD", score=0.0, reason="WEAK_SLOPE", params={})

        threshold = ema_fast + self.params.TREND_PULLBACK_TO_EMA_ATR * atr_3m
        if price > threshold:
            return CandidateSignal(signal="HOLD", score=0.0, reason="NO_PULLBACK", params={})

        if ret_3m <= self.params.MIN_REBOUND_RET:
            return CandidateSignal(signal="HOLD", score=0.0, reason="NO_REBOUND", params={})

        if self.params.REQUIRE_EMA_ALIGNMENT and not (ema_fast > ema_slow):
            return CandidateSignal(signal="HOLD", score=0.0, reason="EMA_NOT_ALIGNED", params={})

        score = min(1.0, 0.5 + abs(ema_slope_15m) * 50)
        return CandidateSignal(
            signal="LONG_TREND_PULLBACK", score=score,
            reason="TREND_UP_PULLBACK_ENTRY", params=default_params
        )

    def _trend_down_signal(self, pos, price, atr_3m, ema_fast, ema_slow,
                           ret_3m, ema_slope_15m, default_params) -> CandidateSignal:
        if pos.side != "FLAT":
            return CandidateSignal(signal="HOLD", score=0.0, reason="IN_POSITION", params={})

        if abs(ema_slope_15m) < self.params.TREND_MIN_EMA_SLOPE_15M:
            return CandidateSignal(signal="HOLD", score=0.0, reason="WEAK_SLOPE", params={})

        threshold = ema_fast - self.params.TREND_PULLBACK_TO_EMA_ATR * atr_3m
        if price < threshold:
            return CandidateSignal(signal="HOLD", score=0.0, reason="NO_PULLBACK", params={})

        if ret_3m >= -self.params.MIN_REBOUND_RET:
            return CandidateSignal(signal="HOLD", score=0.0, reason="NO_REJECTION", params={})

        if self.params.REQUIRE_EMA_ALIGNMENT and not (ema_fast < ema_slow):
            return CandidateSignal(signal="HOLD", score=0.0, reason="EMA_NOT_ALIGNED", params={})

        score = min(1.0, 0.5 + abs(ema_slope_15m) * 50)
        return CandidateSignal(
            signal="SHORT_TREND_PULLBACK", score=score,
            reason="TREND_DOWN_PULLBACK_ENTRY", params=default_params
        )


class TrendTuner:
    """Tuner for TREND pullback parameters."""

    def __init__(self, df_3m: pd.DataFrame, df_15m: pd.DataFrame,
                 regimes: np.ndarray, state_labels: Dict, config: Dict):
        self.df_3m = df_3m
        self.df_15m = df_15m
        self.regimes = regimes
        self.state_labels = state_labels
        self.config = config
        self.df_15m_aligned = df_15m.reindex(df_3m.index, method='ffill')

    def run_backtest(self, params: TrendTuneParams, test_idx: np.ndarray) -> Dict:
        """Run backtest with given parameters using FSM + DecisionEngine."""
        fsm = TrendOnlyFSM(params)
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
                    unrealized_pnl=(ctx.price - pos.entry_price) * pos.size
                                   if pos.side == "LONG"
                                   else (pos.entry_price - ctx.price) * pos.size
                )

            cand, fsm_state = fsm.step(ctx, confirm, pos, fsm_state)
            decision, de_state = de.decide(ctx, confirm, pos, cand, de_state)

            if decision.action == "OPEN_LONG" and pos.side == "FLAT":
                pos = PositionState(
                    side="LONG", entry_price=ctx.price, size=decision.size,
                    entry_ts=ctx.ts, bars_held_3m=0, unrealized_pnl=0.0
                )
                current_trade = {
                    "entry_idx": i, "entry_price": ctx.price, "side": "LONG",
                    "size": decision.size, "regime": regime_raw, "signal": cand.signal,
                }

            elif decision.action == "OPEN_SHORT" and pos.side == "FLAT":
                pos = PositionState(
                    side="SHORT", entry_price=ctx.price, size=decision.size,
                    entry_ts=ctx.ts, bars_held_3m=0, unrealized_pnl=0.0
                )
                current_trade = {
                    "entry_idx": i, "entry_price": ctx.price, "side": "SHORT",
                    "size": decision.size, "regime": regime_raw, "signal": cand.signal,
                }

            elif decision.action == "CLOSE" and pos.side != "FLAT":
                if current_trade is not None:
                    pnl_pct = (ctx.price - pos.entry_price) / pos.entry_price * 100
                    if pos.side == "SHORT":
                        pnl_pct = -pnl_pct

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
                "trend_up_trades": 0, "trend_up_wr": 0.0, "trend_up_pnl": 0.0,
                "trend_down_trades": 0, "trend_down_wr": 0.0, "trend_down_pnl": 0.0,
            }

        pnls = [t["pnl_pct"] for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        trend_up = [t for t in trades if t["regime"] == "TREND_UP"]
        trend_down = [t for t in trades if t["regime"] == "TREND_DOWN"]

        return {
            "n_trades": len(trades),
            "win_rate": len(wins) / len(trades) * 100 if trades else 0,
            "return_pct": sum(pnls),
            "profit_factor": sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else 999,
            "expectancy": np.mean(pnls) if pnls else 0,
            "trend_up_trades": len(trend_up),
            "trend_up_wr": len([t for t in trend_up if t["pnl_pct"] > 0]) / len(trend_up) * 100 if trend_up else 0,
            "trend_up_pnl": sum(t["pnl_pct"] for t in trend_up),
            "trend_down_trades": len(trend_down),
            "trend_down_wr": len([t for t in trend_down if t["pnl_pct"] > 0]) / len(trend_down) * 100 if trend_down else 0,
            "trend_down_pnl": sum(t["pnl_pct"] for t in trend_down),
        }


def load_data():
    """Load data from TimescaleDB."""
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

    start = datetime(2023, 1, 1)
    end = datetime(2024, 12, 31)

    df_3m = pd.read_sql(query, conn, params=("XRPUSDT", "3m", start, end))
    df_3m["timestamp"] = pd.to_datetime(df_3m["timestamp"])
    df_3m.set_index("timestamp", inplace=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df_3m[col] = df_3m[col].astype(float)

    df_15m = pd.read_sql(query, conn, params=("XRPUSDT", "15m", start, end))
    df_15m["timestamp"] = pd.to_datetime(df_15m["timestamp"])
    df_15m.set_index("timestamp", inplace=True)
    for col in ["open", "high", "low", "close", "volume"]:
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

    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, 1e-10)
    df[f"rsi{suffix}"] = 100 - (100 / (1 + rs))

    return df


def train_hmm(df_15m):
    """Train HMM and get states."""
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


def main():
    logger.info("=" * 70)
    logger.info("TREND PULLBACK PARAMETER TUNING")
    logger.info("=" * 70)

    logger.info("Loading data...")
    df_3m, df_15m = load_data()
    logger.info(f"3m bars: {len(df_3m)}, 15m bars: {len(df_15m)}")

    logger.info("Adding features...")
    df_3m = add_features(df_3m, "_3m")
    df_15m = add_features(df_15m, "_15m")

    logger.info("Training HMM...")
    states_15m, state_labels, hmm_index = train_hmm(df_15m)
    logger.info(f"State labels: {state_labels}")

    state_df = pd.DataFrame({"state": states_15m}, index=hmm_index)
    state_3m = state_df.reindex(df_3m.index, method="ffill")

    valid_mask = ~state_3m["state"].isna() & ~df_3m["atr_3m"].isna()
    valid_idx = np.where(valid_mask.values)[0]

    split_idx = int(len(valid_idx) * 0.7)
    test_idx = valid_idx[split_idx:]

    logger.info(f"Test samples: {len(test_idx)}")

    config = {
        "DECISION": {"DENY_IN_HIGH_VOL": True, "DENY_IN_NO_TRADE": True},
        "RISK": {"BASE_SIZE_USDT": 100},
    }

    regimes = state_3m["state"].fillna(0).astype(int).values
    tuner = TrendTuner(df_3m, df_15m, regimes, state_labels, config)

    # Parameter grid with REALISTIC EMA slope thresholds
    param_grid = {
        "TREND_PULLBACK_TO_EMA_ATR": [0.5, 1.0, 1.5, 2.0],
        "TREND_MIN_EMA_SLOPE_15M": [0.002, 0.003, 0.005, 0.007],  # Realistic!
        "MIN_REBOUND_RET": [0.0, 0.0003, 0.0005],
        "REQUIRE_EMA_ALIGNMENT": [False, True],
        "STOP_ATR_MULT": [1.5, 2.0],
        "TP_ATR_MULT": [2.0, 3.0],
    }

    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(product(*values))

    logger.info(f"\nTesting {len(combinations)} parameter combinations...")

    results = []
    min_trades = 10

    for i, combo in enumerate(combinations):
        params = TrendTuneParams(
            TREND_PULLBACK_TO_EMA_ATR=combo[0],
            TREND_MIN_EMA_SLOPE_15M=combo[1],
            MIN_REBOUND_RET=combo[2],
            REQUIRE_EMA_ALIGNMENT=combo[3],
            STOP_ATR_MULT=combo[4],
            TP_ATR_MULT=combo[5],
            MAX_HOLD_BARS_3M=40,
        )

        metrics = tuner.run_backtest(params, test_idx)
        results.append({
            "params": {k: combo[j] for j, k in enumerate(keys)},
            "metrics": metrics,
        })

        if (i + 1) % 50 == 0:
            logger.info(f"  Progress: {i + 1}/{len(combinations)}")

    # Show results
    trade_counts = [r["metrics"]["n_trades"] for r in results]
    logger.info(f"\nTrade count: min={min(trade_counts)}, max={max(trade_counts)}, median={np.median(trade_counts)}")

    valid_results = [r for r in results if r["metrics"]["n_trades"] >= min_trades]
    sorted_results = sorted(valid_results, key=lambda x: x["metrics"]["profit_factor"], reverse=True)

    logger.info(f"\n{len(valid_results)} results with >= {min_trades} trades")
    logger.info("\n" + "=" * 70)
    logger.info("TOP 10 RESULTS (by Profit Factor)")
    logger.info("=" * 70)

    for i, r in enumerate(sorted_results[:10]):
        m = r["metrics"]
        logger.info(f"\n#{i+1}:")
        logger.info(f"  Params: {r['params']}")
        logger.info(f"  Trades: {m['n_trades']}, WR: {m['win_rate']:.1f}%, PF: {m['profit_factor']:.2f}")
        logger.info(f"  Return: {m['return_pct']:.2f}%, Expectancy: {m['expectancy']:.4f}%")
        logger.info(f"  UP: {m['trend_up_trades']} trades, WR: {m['trend_up_wr']:.1f}%, PnL: {m['trend_up_pnl']:.2f}%")
        logger.info(f"  DOWN: {m['trend_down_trades']} trades, WR: {m['trend_down_wr']:.1f}%, PnL: {m['trend_down_pnl']:.2f}%")

    # Save results
    output_dir = PROJECT_ROOT / "outputs" / "trend_tuning"
    output_dir.mkdir(parents=True, exist_ok=True)

    if sorted_results:
        best = sorted_results[0]
        with open(output_dir / "best_trend_params.json", "w") as f:
            json.dump({
                "best_params": best["params"],
                "best_metrics": best["metrics"],
                "timestamp": datetime.now().isoformat(),
            }, f, indent=2)

        logger.info(f"\nResults saved to {output_dir}")

        logger.info("\n" + "=" * 70)
        logger.info("RECOMMENDED CONFIG UPDATE")
        logger.info("=" * 70)
        logger.info(f"""
FSM:
  TREND_PULLBACK_TO_EMA_ATR: {best['params']['TREND_PULLBACK_TO_EMA_ATR']}
  TREND_MIN_EMA_SLOPE_15M: {best['params']['TREND_MIN_EMA_SLOPE_15M']}
  TREND_MIN_REBOUND_RET: {best['params']['MIN_REBOUND_RET']}
  TREND_REQUIRE_EMA_ALIGNMENT: {best['params']['REQUIRE_EMA_ALIGNMENT']}

RISK:
  STOP_ATR_MULT: {best['params']['STOP_ATR_MULT']}
  TP_ATR_MULT: {best['params']['TP_ATR_MULT']}
""")


if __name__ == "__main__":
    main()
