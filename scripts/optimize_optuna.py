#!/usr/bin/env python
"""Optuna-based Parameter Optimization for XRP Trading System.

Uses Bayesian optimization to efficiently search the parameter space.
Supports:
- Single objective (maximize Profit Factor or Sharpe Ratio)
- Train/Test split validation
- Walk-forward validation
- Multi-objective optimization (optional)

Usage:
    # Basic optimization
    python scripts/optimize_optuna.py --trials 100

    # With walk-forward validation
    python scripts/optimize_optuna.py --trials 100 --walk-forward --n-folds 5

    # Optimize specific parameter groups
    python scripts/optimize_optuna.py --trials 100 --groups fsm risk

    # Resume from previous study
    python scripts/optimize_optuna.py --trials 100 --resume
"""

import argparse
import json
import logging
import pickle
import sys
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Suppress warnings during optimization
warnings.filterwarnings("ignore")

try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
except ImportError:
    print("Optuna not installed. Run: pip install optuna")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from xrp4.regime.confirm import RegimeConfirmLayer, ConfirmConfig
from xrp4.regime.multi_hmm_manager import MultiHMMManager
from xrp4.features.hmm_features import (
    build_fast_hmm_features_v2,
    build_mid_hmm_features_v2,
)
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

# Suppress Optuna's verbose logging during optimization
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ============================================================================
# Parameter Definitions
# ============================================================================

@dataclass
class ParameterSpace:
    """Defines the search space for each parameter group."""

    # FSM Parameters (Entry Conditions)
    FSM_PARAMS = {
        # LONG entry parameters
        "TREND_PULLBACK_TO_EMA_ATR": {"type": "float", "low": 1.0, "high": 4.0, "step": 0.5},
        "TREND_MIN_EMA_SLOPE_15M": {"type": "float", "low": 0.001, "high": 0.01, "step": 0.001},
        "TREND_MIN_REBOUND_RET": {"type": "float", "low": 0.0001, "high": 0.001, "step": 0.0001},

        # SHORT entry parameters
        "SHORT_PULLBACK_TO_EMA_ATR": {"type": "float", "low": 0.5, "high": 2.5, "step": 0.5},
        "SHORT_MIN_EMA_SLOPE_15M": {"type": "float", "low": 0.001, "high": 0.01, "step": 0.001},
        "SHORT_MIN_REJECTION_RET": {"type": "float", "low": 0.0003, "high": 0.002, "step": 0.0001},
        "SHORT_CONSEC_NEG_BARS": {"type": "int", "low": 1, "high": 3},
        "SHORT_PEAK_CONFIRM_BARS": {"type": "int", "low": 1, "high": 4},

        # RANGE parameters
        "RANGE_PULLBACK_ATR": {"type": "float", "low": 0.3, "high": 1.0, "step": 0.1},
        "RANGE_MIN_ZONE_STRENGTH": {"type": "float", "low": 0.000005, "high": 0.00005, "log": True},
    }

    # Risk Management Parameters
    RISK_PARAMS = {
        "STOP_ATR_MULT": {"type": "float", "low": 1.0, "high": 3.0, "step": 0.25},
        "TP_ATR_MULT": {"type": "float", "low": 1.5, "high": 4.0, "step": 0.25},
        "TRAIL_ATR_MULT": {"type": "float", "low": 0.5, "high": 2.0, "step": 0.25},
        "MAX_HOLD_BARS_3M": {"type": "int", "low": 20, "high": 80, "step": 10},
        "SHORT_MAX_HOLD_BARS_3M": {"type": "int", "low": 15, "high": 50, "step": 5},
    }

    # Confirm Layer Parameters
    CONFIRM_PARAMS = {
        "TREND_CONFIRM_B_ATR": {"type": "float", "low": 0.5, "high": 1.5, "step": 0.1},
        "TREND_CONFIRM_S_ATR": {"type": "float", "low": 0.03, "high": 0.15, "step": 0.01},
        "TREND_CONFIRM_EWM_RET_SIGMA": {"type": "float", "low": 0.1, "high": 0.4, "step": 0.05},
        "TREND_CONFIRM_CONSEC_BARS": {"type": "int", "low": 1, "high": 4},
        "HIGH_VOL_LAMBDA_ON": {"type": "float", "low": 1.0, "high": 2.5, "step": 0.25},
        "HIGH_VOL_LAMBDA_OFF": {"type": "float", "low": 0.25, "high": 1.0, "step": 0.25},
        "HIGH_VOL_COOLDOWN_BARS_15M": {"type": "int", "low": 2, "high": 8},
    }

    # XGB Gate Parameters
    XGB_PARAMS = {
        "XGB_PMIN_TREND": {"type": "float", "low": 0.30, "high": 0.60, "step": 0.05},
        "XGB_PMIN_RANGE": {"type": "float", "low": 0.30, "high": 0.60, "step": 0.05},
    }


# ============================================================================
# Backtest Engine (Optimized for Speed)
# ============================================================================

class FastBacktestEngine:
    """Optimized backtest engine for parameter optimization."""

    def __init__(
        self,
        df_3m: pd.DataFrame,
        df_15m: pd.DataFrame,
        df_1h: pd.DataFrame,
        multi_hmm: MultiHMMManager,
        fast_features: np.ndarray,
        fast_timestamps: np.ndarray,
        mid_features: np.ndarray,
        mid_timestamps: np.ndarray,
        initial_capital: float = 10000.0,
        leverage: float = 1.0,
        fee_rate: float = 0.0004,
    ):
        self.df_3m = df_3m
        self.df_15m = df_15m
        self.df_1h = df_1h
        self.multi_hmm = multi_hmm
        self.fast_features = fast_features
        self.fast_timestamps = fast_timestamps
        self.mid_features = mid_features
        self.mid_timestamps = mid_timestamps
        self.initial_capital = initial_capital
        self.leverage = leverage
        self.fee_rate = fee_rate

        # Pre-compute regime predictions for speed
        self._precompute_regimes()

    def _precompute_regimes(self) -> None:
        """Pre-compute HMM regime predictions."""
        if self.multi_hmm and self.multi_hmm.is_trained:
            regime_packets = self.multi_hmm.predict_sequence(
                self.fast_features,
                self.mid_features,
                self.fast_timestamps,
                self.mid_timestamps,
            )
            # Create timestamp -> regime mapping from RegimePacket objects
            self.ts_to_regime = {}
            for i, ts in enumerate(self.fast_timestamps):
                if i < len(regime_packets):
                    packet = regime_packets[i]
                    label = packet.label_fused.value if hasattr(packet.label_fused, 'value') else str(packet.label_fused)
                    confidence = packet.fast_pred.confidence if packet.fast_pred else 0.5
                    self.ts_to_regime[ts] = (label, confidence)
        else:
            self.ts_to_regime = {}

    def run(
        self,
        fsm_params: Dict[str, Any],
        risk_params: Dict[str, Any],
        confirm_params: Dict[str, Any],
        xgb_params: Dict[str, Any],
    ) -> Dict[str, float]:
        """Run backtest with given parameters."""

        # Create components with custom parameters
        confirm_config = ConfirmConfig(
            TREND_CONFIRM_B_ATR=confirm_params.get("TREND_CONFIRM_B_ATR", 0.8),
            TREND_CONFIRM_S_ATR=confirm_params.get("TREND_CONFIRM_S_ATR", 0.07),
            TREND_CONFIRM_EWM_RET_SIGMA=confirm_params.get("TREND_CONFIRM_EWM_RET_SIGMA", 0.20),
            TREND_CONFIRM_CONSEC_BARS=confirm_params.get("TREND_CONFIRM_CONSEC_BARS", 2),
            HIGH_VOL_LAMBDA_ON=confirm_params.get("HIGH_VOL_LAMBDA_ON", 1.50),
            HIGH_VOL_LAMBDA_OFF=confirm_params.get("HIGH_VOL_LAMBDA_OFF", 0.50),
            HIGH_VOL_COOLDOWN_BARS_15M=confirm_params.get("HIGH_VOL_COOLDOWN_BARS_15M", 4),
        )
        confirm_layer = RegimeConfirmLayer(confirm_config)

        fsm_config = FSMConfig(
            # LONG params
            TREND_PULLBACK_TO_EMA_ATR=fsm_params.get("TREND_PULLBACK_TO_EMA_ATR", 2.0),
            TREND_MIN_EMA_SLOPE_15M=fsm_params.get("TREND_MIN_EMA_SLOPE_15M", 0.002),
            TREND_MIN_REBOUND_RET=fsm_params.get("TREND_MIN_REBOUND_RET", 0.0003),
            # SHORT params
            SHORT_PULLBACK_TO_EMA_ATR=fsm_params.get("SHORT_PULLBACK_TO_EMA_ATR", 1.0),
            SHORT_MIN_EMA_SLOPE_15M=fsm_params.get("SHORT_MIN_EMA_SLOPE_15M", 0.002),
            SHORT_MIN_REJECTION_RET=fsm_params.get("SHORT_MIN_REJECTION_RET", 0.0007),
            SHORT_CONSEC_NEG_BARS=fsm_params.get("SHORT_CONSEC_NEG_BARS", 1),
            SHORT_PEAK_CONFIRM_BARS=fsm_params.get("SHORT_PEAK_CONFIRM_BARS", 2),
            SHORT_MAX_HOLD_BARS_3M=risk_params.get("SHORT_MAX_HOLD_BARS_3M", 30),
            # RANGE params
            RANGE_PULLBACK_ATR=fsm_params.get("RANGE_PULLBACK_ATR", 0.6),
            RANGE_MIN_ZONE_STRENGTH=fsm_params.get("RANGE_MIN_ZONE_STRENGTH", 0.00001),
            RANGE_ENABLED=True,
            # Risk params
            STOP_ATR_MULT=risk_params.get("STOP_ATR_MULT", 1.5),
            TP_ATR_MULT=risk_params.get("TP_ATR_MULT", 2.0),
            TRAIL_ATR_MULT=risk_params.get("TRAIL_ATR_MULT", 1.0),
            MAX_HOLD_BARS_3M=risk_params.get("MAX_HOLD_BARS_3M", 40),
            # Other
            TRANSITION_ALLOW_BREAKOUT_ONLY=False,
        )
        fsm = TradingFSM(fsm_config)

        decision_config = DecisionConfig(
            XGB_ENABLED=xgb_params.get("XGB_ENABLED", True),
            XGB_PMIN_TREND=xgb_params.get("XGB_PMIN_TREND", 0.40),
            XGB_PMIN_RANGE=xgb_params.get("XGB_PMIN_RANGE", 0.40),
        )
        decision_engine = DecisionEngine(decision_config)

        # Run backtest
        confirm_state = None
        fsm_state = None
        engine_state = None
        position = PositionState(side="FLAT")
        equity = self.initial_capital
        trades = []
        equity_curve = [equity]

        warmup = 250

        for i in range(warmup, len(self.df_3m)):
            bar = self.df_3m.iloc[i]
            ts_3m = bar["timestamp"]
            price = float(bar["close"])

            # Get 15m bar
            ts_15m = ts_3m.floor("15min")
            bar_15m_mask = self.df_15m["timestamp"] == ts_15m
            if not bar_15m_mask.any():
                idx = self.df_15m["timestamp"].searchsorted(ts_15m)
                if idx > 0:
                    idx -= 1
                bar_15m = self.df_15m.iloc[idx]
            else:
                bar_15m = self.df_15m[bar_15m_mask].iloc[0]

            hist_15m = self.df_15m[self.df_15m["timestamp"] <= ts_15m].tail(20)

            # Get regime from pre-computed
            regime_raw = "RANGE"
            confidence = 0.5
            if ts_3m in self.ts_to_regime:
                regime_raw, confidence = self.ts_to_regime[ts_3m]

            # Confirm regime
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

            # Market context
            atr_3m = bar.get("atr_3m", 0.01)
            if pd.isna(atr_3m) or atr_3m <= 0:
                atr_3m = 0.01
            atr_15m = bar_15m.get("atr_15m", atr_3m)
            if pd.isna(atr_15m) or atr_15m <= 0:
                atr_15m = atr_3m

            support = bar_15m.get("rolling_low_20", price - atr_15m * 2)
            resistance = bar_15m.get("rolling_high_20", price + atr_15m * 2)
            if pd.isna(support):
                support = price - atr_15m * 2
            if pd.isna(resistance):
                resistance = price + atr_15m * 2

            dist_to_support = (price - support) / atr_3m if atr_3m > 0 else 999
            dist_to_resistance = (resistance - price) / atr_3m if atr_3m > 0 else 999

            ema_fast = bar.get("ema_fast_3m", price)
            ema_slow = bar.get("ema_slow_3m", price)
            ret_3m = bar.get("ret_3m", 0)
            volatility = bar.get("volatility_3m", bar.get("ret_std_3m", 0.005))

            if pd.isna(ema_fast):
                ema_fast = price
            if pd.isna(ema_slow):
                ema_slow = price
            if pd.isna(ret_3m):
                ret_3m = 0
            if pd.isna(volatility):
                volatility = 0.005

            market_ctx = MarketContext(
                symbol="XRPUSDT",
                ts=int(ts_3m.timestamp() * 1000) if hasattr(ts_3m, "timestamp") else 0,
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
                    "ema_slope_15m": bar_15m.get("ema_slope_15m", 0),
                    "ema_diff": (ema_fast - ema_slow) / price if price > 0 else 0,
                    "price_to_ema20": (price - ema_fast) / ema_fast if ema_fast > 0 else 0,
                    "price_to_ema50": (price - ema_slow) / ema_slow if ema_slow > 0 else 0,
                },
                row_15m={
                    "ema_slope_15m": bar_15m.get("ema_slope_15m", 0),
                    "ewm_ret_15m": bar_15m.get("ewm_ret_15m", 0),
                },
                zone={
                    "support": support,
                    "resistance": resistance,
                    "strength": 0.0001,
                    "dist_to_support": dist_to_support,
                    "dist_to_resistance": dist_to_resistance,
                },
            )

            # Update position
            if position.side != "FLAT":
                position.bars_held_3m += 1
                if position.side == "LONG":
                    position.unrealized_pnl = (price - position.entry_price) * position.size
                else:
                    position.unrealized_pnl = (position.entry_price - price) * position.size

            # FSM step
            candidate_signal, fsm_state = fsm.step(
                ctx=market_ctx,
                confirm=confirm_ctx,
                pos=position,
                fsm_state=fsm_state,
            )

            # Decision
            decision, engine_state = decision_engine.decide(
                ctx=market_ctx,
                confirm=confirm_ctx,
                pos=position,
                cand=candidate_signal,
                engine_state=engine_state,
            )

            # Position sizing with leverage
            position_value = equity * self.leverage

            # Execute decision
            if decision.action == "OPEN_LONG" and position.side == "FLAT":
                entry_price = price
                size = position_value / entry_price
                fee = entry_price * size * self.fee_rate
                equity -= fee
                position = PositionState(
                    side="LONG",
                    entry_price=entry_price,
                    size=size,
                    entry_ts=market_ctx.ts,
                    bars_held_3m=0,
                    unrealized_pnl=0,
                )

            elif decision.action == "OPEN_SHORT" and position.side == "FLAT":
                entry_price = price
                size = position_value / entry_price
                fee = entry_price * size * self.fee_rate
                equity -= fee
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

                fee = exit_price * position.size * self.fee_rate
                pnl -= fee
                equity += pnl

                trades.append({
                    "pnl": pnl,
                    "pnl_pct": pnl / (position.entry_price * position.size) * 100 if position.size > 0 else 0,
                    "bars_held": position.bars_held_3m,
                    "side": position.side,
                })
                position = PositionState(side="FLAT")

            equity_curve.append(equity)

        # Close remaining position
        if position.side != "FLAT":
            bar = self.df_3m.iloc[-1]
            exit_price = float(bar["close"])
            if position.side == "LONG":
                pnl = (exit_price - position.entry_price) * position.size
            else:
                pnl = (position.entry_price - exit_price) * position.size
            fee = exit_price * position.size * self.fee_rate
            pnl -= fee
            equity += pnl
            trades.append({
                "pnl": pnl,
                "pnl_pct": pnl / (position.entry_price * position.size) * 100 if position.size > 0 else 0,
                "bars_held": position.bars_held_3m,
                "side": position.side,
            })

        # Calculate metrics
        return self._calculate_metrics(trades, equity, equity_curve)

    def _calculate_metrics(
        self,
        trades: List[Dict],
        final_equity: float,
        equity_curve: List[float],
    ) -> Dict[str, float]:
        """Calculate performance metrics."""
        if not trades:
            return {
                "n_trades": 0,
                "profit_factor": 0.0,
                "win_rate": 0.0,
                "return_pct": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "avg_trade_pnl": 0.0,
                "final_equity": final_equity,
            }

        df = pd.DataFrame(trades)
        wins = df[df["pnl"] > 0]
        losses = df[df["pnl"] <= 0]

        # Profit factor
        gross_profit = wins["pnl"].sum() if len(wins) > 0 else 0
        gross_loss = abs(losses["pnl"].sum()) if len(losses) > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else (999 if gross_profit > 0 else 0)

        # Win rate
        win_rate = len(wins) / len(trades) * 100

        # Return
        return_pct = (final_equity - self.initial_capital) / self.initial_capital * 100

        # Sharpe ratio (annualized, assuming 3m bars)
        equity_arr = np.array(equity_curve)
        returns = np.diff(equity_arr) / equity_arr[:-1]
        if len(returns) > 1 and returns.std() > 0:
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(365 * 24 * 20)  # 20 bars per hour
        else:
            sharpe_ratio = 0.0

        # Max drawdown
        peak = np.maximum.accumulate(equity_arr)
        drawdown = (peak - equity_arr) / peak
        max_drawdown = drawdown.max() * 100

        return {
            "n_trades": len(trades),
            "profit_factor": profit_factor,
            "win_rate": win_rate,
            "return_pct": return_pct,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "avg_trade_pnl": df["pnl"].mean(),
            "final_equity": final_equity,
        }


# ============================================================================
# Data Loading
# ============================================================================

def load_data_from_binance(
    symbol: str = "XRPUSDT",
    months: int = 6,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load historical data from Binance."""
    import requests
    import time

    BASE_URL = "https://api.binance.com"

    def fetch_klines(interval: str, limit: int, end_time: Optional[int] = None) -> pd.DataFrame:
        all_data = []
        remaining = limit
        current_end = end_time

        while remaining > 0:
            fetch_limit = min(remaining, 1000)
            params = {"symbol": symbol, "interval": interval, "limit": fetch_limit}
            if current_end:
                params["endTime"] = current_end

            resp = requests.get(f"{BASE_URL}/api/v3/klines", params=params)
            resp.raise_for_status()
            data = resp.json()

            if not data:
                break

            all_data = data + all_data
            remaining -= len(data)

            if remaining > 0 and len(data) == fetch_limit:
                current_end = data[0][0] - 1
                time.sleep(0.05)
            else:
                break

        if not all_data:
            return pd.DataFrame()

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

    # Calculate limits
    bars_per_month_3m = 30 * 24 * 20  # ~14400
    bars_per_month_15m = 30 * 24 * 4  # ~2880
    bars_per_month_1h = 30 * 24       # ~720

    logger.info(f"Fetching {months} months of data from Binance...")

    df_3m = fetch_klines("3m", bars_per_month_3m * months)
    logger.info(f"  3m: {len(df_3m)} bars")

    df_15m = fetch_klines("15m", bars_per_month_15m * months)
    logger.info(f"  15m: {len(df_15m)} bars")

    df_1h = fetch_klines("1h", bars_per_month_1h * months)
    logger.info(f"  1h: {len(df_1h)} bars")

    return df_3m, df_15m, df_1h


def build_features(
    df_3m: pd.DataFrame,
    df_15m: pd.DataFrame,
    df_1h: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build features for backtest."""
    df_3m = df_3m.copy()
    df_15m = df_15m.copy()
    df_1h = df_1h.copy()

    # 3m features
    df_3m["ret_3m"] = df_3m["close"].pct_change()
    df_3m["ema_fast_3m"] = df_3m["close"].ewm(span=20, adjust=False).mean()
    df_3m["ema_slow_3m"] = df_3m["close"].ewm(span=50, adjust=False).mean()
    df_3m["atr_3m"] = (
        df_3m["high"].rolling(14).max() - df_3m["low"].rolling(14).min()
    ).rolling(14).mean()
    df_3m["volatility_3m"] = df_3m["ret_3m"].rolling(20).std()

    # RSI
    delta = df_3m["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, 1e-10)
    df_3m["rsi_3m"] = 100 - (100 / (1 + rs))

    # 15m features
    df_15m["ret_15m"] = df_15m["close"].pct_change()
    df_15m["ewm_ret_15m"] = df_15m["ret_15m"].ewm(span=10, adjust=False).mean()
    df_15m["ewm_std_ret_15m"] = df_15m["ret_15m"].ewm(span=10, adjust=False).std()
    df_15m["atr_15m"] = (
        df_15m["high"].rolling(14).max() - df_15m["low"].rolling(14).min()
    ).rolling(14).mean()
    df_15m["atr_pct_15m"] = df_15m["atr_15m"] / df_15m["close"] * 100
    df_15m["rolling_high_20"] = df_15m["high"].rolling(20).max()
    df_15m["rolling_low_20"] = df_15m["low"].rolling(20).min()
    df_15m["ema_15m"] = df_15m["close"].ewm(span=20, adjust=False).mean()
    df_15m["ema_slope_15m"] = df_15m["ema_15m"].diff() / df_15m["close"]
    df_15m["bb_mid"] = df_15m["close"].rolling(20).mean()
    df_15m["bb_std"] = df_15m["close"].rolling(20).std()
    df_15m["bb_width_15m"] = df_15m["bb_std"] / df_15m["bb_mid"]

    # Candle patterns
    df_15m["body_ratio"] = abs(df_15m["close"] - df_15m["open"]) / (df_15m["high"] - df_15m["low"]).replace(0, 1e-10)
    df_15m["upper_wick_ratio"] = (df_15m["high"] - df_15m[["open", "close"]].max(axis=1)) / (df_15m["high"] - df_15m["low"]).replace(0, 1e-10)
    df_15m["lower_wick_ratio"] = (df_15m[["open", "close"]].min(axis=1) - df_15m["low"]) / (df_15m["high"] - df_15m["low"]).replace(0, 1e-10)

    # Range composition
    range_15m = df_15m["rolling_high_20"] - df_15m["rolling_low_20"]
    df_15m["range_comp_15m"] = range_15m / range_15m.rolling(20).mean().replace(0, 1)
    df_15m["vol_z_15m"] = (df_15m["volume"] - df_15m["volume"].rolling(20).mean()) / df_15m["volume"].rolling(20).std().replace(0, 1)

    # 1h features
    df_1h["ret_1h"] = df_1h["close"].pct_change()
    df_1h["ewm_ret_1h"] = df_1h["ret_1h"].ewm(span=10, adjust=False).mean()
    df_1h["ewm_std_ret_1h"] = df_1h["ret_1h"].ewm(span=10, adjust=False).std()
    df_1h["atr_pct_1h"] = (
        (df_1h["high"].rolling(14).max() - df_1h["low"].rolling(14).min()).rolling(14).mean() / df_1h["close"] * 100
    )
    df_1h["ema_1h"] = df_1h["close"].ewm(span=20, adjust=False).mean()
    df_1h["price_z_from_ema_1h"] = (df_1h["close"] - df_1h["ema_1h"]) / df_1h["ema_1h"].replace(0, 1)

    # Merge 15m features to 3m
    df_3m["ts_15m"] = df_3m["timestamp"].dt.floor("15min")
    df_15m_indexed = df_15m.set_index("timestamp")
    df_3m = df_3m.join(df_15m_indexed[["ewm_ret_15m", "ewm_std_ret_15m", "atr_15m", "atr_pct_15m",
                                        "rolling_high_20", "rolling_low_20", "ema_slope_15m",
                                        "bb_width_15m", "body_ratio", "upper_wick_ratio",
                                        "lower_wick_ratio", "range_comp_15m", "vol_z_15m"]],
                       on="ts_15m", how="left")
    df_3m = df_3m.drop(columns=["ts_15m"])

    # Add range_pct for Fast HMM
    df_3m["range_pct"] = (df_3m["high"] - df_3m["low"]) / df_3m["close"]

    # Merge 1h to 15m for Mid HMM
    df_15m["ts_1h"] = df_15m["timestamp"].dt.floor("1h")
    df_1h_indexed = df_1h.set_index("timestamp")
    cols_1h = ["ret_1h", "ewm_ret_1h", "ewm_std_ret_1h", "atr_pct_1h", "price_z_from_ema_1h"]
    df_15m = df_15m.join(df_1h_indexed[cols_1h], on="ts_1h", how="left")
    df_15m = df_15m.drop(columns=["ts_1h"])

    # Forward fill and drop NaN
    df_3m = df_3m.ffill().dropna()
    df_15m = df_15m.ffill().dropna()

    return df_3m, df_15m


# ============================================================================
# Optuna Optimization
# ============================================================================

class OptunaOptimizer:
    """Optuna-based parameter optimizer."""

    def __init__(
        self,
        df_3m: pd.DataFrame,
        df_15m: pd.DataFrame,
        df_1h: pd.DataFrame,
        train_ratio: float = 0.7,
        leverage: float = 1.0,
        objective_metric: str = "profit_factor",
        param_groups: List[str] = None,
        min_trades: int = 50,
    ):
        self.df_3m = df_3m
        self.df_15m = df_15m
        self.df_1h = df_1h
        self.train_ratio = train_ratio
        self.leverage = leverage
        self.objective_metric = objective_metric
        self.param_groups = param_groups or ["fsm", "risk", "confirm", "xgb"]
        self.min_trades = min_trades

        # Split data
        self._split_data()

        # Build features
        self._build_features()

        # Train HMM
        self._train_hmm()

        # Create backtest engine
        self.engine = FastBacktestEngine(
            df_3m=self.df_3m_train,
            df_15m=self.df_15m_train,
            df_1h=self.df_1h_train,
            multi_hmm=self.multi_hmm,
            fast_features=self.fast_features_train,
            fast_timestamps=self.fast_timestamps_train,
            mid_features=self.mid_features_train,
            mid_timestamps=self.mid_timestamps_train,
            leverage=leverage,
        )

        # Best params
        self.best_params = None
        self.best_value = None

    def _split_data(self) -> None:
        """Split data into train/test."""
        train_size_3m = int(len(self.df_3m) * self.train_ratio)
        train_size_15m = int(len(self.df_15m) * self.train_ratio)
        train_size_1h = int(len(self.df_1h) * self.train_ratio)

        self.df_3m_train = self.df_3m.iloc[:train_size_3m].copy()
        self.df_3m_test = self.df_3m.iloc[train_size_3m:].copy()

        self.df_15m_train = self.df_15m.iloc[:train_size_15m].copy()
        self.df_15m_test = self.df_15m.iloc[train_size_15m:].copy()

        self.df_1h_train = self.df_1h.iloc[:train_size_1h].copy()
        self.df_1h_test = self.df_1h.iloc[train_size_1h:].copy()

        logger.info(f"Train: {len(self.df_3m_train)} 3m bars, Test: {len(self.df_3m_test)} 3m bars")

    def _build_features(self) -> None:
        """Build features for train and test sets."""
        logger.info("Building features...")
        self.df_3m_train, self.df_15m_train = build_features(
            self.df_3m_train, self.df_15m_train, self.df_1h_train
        )
        self.df_3m_test, self.df_15m_test = build_features(
            self.df_3m_test, self.df_15m_test, self.df_1h_test
        )

    def _train_hmm(self) -> None:
        """Train Multi-HMM model."""
        logger.info("Training Multi-HMM...")

        # Feature names
        fast_feature_names = [
            "ret_3m", "ewm_ret_15m", "atr_pct_15m", "ewm_std_ret_15m",
            "bb_width_15m", "body_ratio", "upper_wick_ratio", "lower_wick_ratio", "range_pct"
        ]
        mid_feature_names = [
            "ret_15m", "ewm_ret_15m", "ret_1h", "ewm_ret_1h",
            "atr_pct_15m", "ewm_std_ret_15m", "atr_pct_1h", "ewm_std_ret_1h",
            "bb_width_15m", "vol_z_15m", "price_z_from_ema_1h"
        ]

        # Build HMM features - use only OHLCV columns to avoid column overlap
        df_3m_ohlcv = self.df_3m_train[["timestamp", "open", "high", "low", "close", "volume"]].copy()
        self.fast_features_train, self.fast_timestamps_train = build_fast_hmm_features_v2(
            df_3m_ohlcv, fast_feature_names
        )
        df_15m_ohlcv = self.df_15m_train[["timestamp", "open", "high", "low", "close", "volume"]].copy()
        self.mid_features_train, self.mid_timestamps_train = build_mid_hmm_features_v2(
            df_15m_ohlcv, mid_feature_names
        )

        # Train HMM
        self.multi_hmm = MultiHMMManager()
        train_size_fast = min(int(len(self.fast_features_train) * 0.3), 3000)
        train_size_mid = min(int(len(self.mid_features_train) * 0.3), 1000)

        self.multi_hmm.train(
            fast_features=self.fast_features_train[:train_size_fast],
            fast_feature_names=fast_feature_names,
            mid_features=self.mid_features_train[:train_size_mid],
            mid_feature_names=mid_feature_names,
            fast_timestamps=self.fast_timestamps_train[:train_size_fast],
            mid_timestamps=self.mid_timestamps_train[:train_size_mid],
        )
        logger.info(f"Multi-HMM trained (fast: {train_size_fast}, mid: {train_size_mid})")

    def _suggest_params(self, trial: optuna.Trial) -> Dict[str, Dict]:
        """Suggest parameters from Optuna trial."""
        fsm_params = {}
        risk_params = {}
        confirm_params = {}
        xgb_params = {}

        space = ParameterSpace()

        # FSM params
        if "fsm" in self.param_groups:
            for name, spec in space.FSM_PARAMS.items():
                if spec["type"] == "float":
                    if spec.get("log"):
                        fsm_params[name] = trial.suggest_float(name, spec["low"], spec["high"], log=True)
                    elif spec.get("step"):
                        fsm_params[name] = trial.suggest_float(name, spec["low"], spec["high"], step=spec["step"])
                    else:
                        fsm_params[name] = trial.suggest_float(name, spec["low"], spec["high"])
                elif spec["type"] == "int":
                    fsm_params[name] = trial.suggest_int(name, spec["low"], spec["high"], step=spec.get("step", 1))

        # Risk params
        if "risk" in self.param_groups:
            for name, spec in space.RISK_PARAMS.items():
                if spec["type"] == "float":
                    risk_params[name] = trial.suggest_float(name, spec["low"], spec["high"], step=spec.get("step"))
                elif spec["type"] == "int":
                    risk_params[name] = trial.suggest_int(name, spec["low"], spec["high"], step=spec.get("step", 1))

        # Confirm params
        if "confirm" in self.param_groups:
            for name, spec in space.CONFIRM_PARAMS.items():
                if spec["type"] == "float":
                    confirm_params[name] = trial.suggest_float(name, spec["low"], spec["high"], step=spec.get("step"))
                elif spec["type"] == "int":
                    confirm_params[name] = trial.suggest_int(name, spec["low"], spec["high"])

        # XGB params
        if "xgb" in self.param_groups:
            for name, spec in space.XGB_PARAMS.items():
                xgb_params[name] = trial.suggest_float(name, spec["low"], spec["high"], step=spec.get("step"))
            xgb_params["XGB_ENABLED"] = True

        return {
            "fsm": fsm_params,
            "risk": risk_params,
            "confirm": confirm_params,
            "xgb": xgb_params,
        }

    def objective(self, trial: optuna.Trial) -> float:
        """Optuna objective function."""
        params = self._suggest_params(trial)

        try:
            result = self.engine.run(
                fsm_params=params["fsm"],
                risk_params=params["risk"],
                confirm_params=params["confirm"],
                xgb_params=params["xgb"],
            )

            # Penalize if too few trades
            if result["n_trades"] < self.min_trades:
                return 0.0

            # Return objective metric
            value = result.get(self.objective_metric, 0.0)

            # Report intermediate values for pruning
            trial.report(value, 0)

            # Store additional metrics
            trial.set_user_attr("n_trades", result["n_trades"])
            trial.set_user_attr("win_rate", result["win_rate"])
            trial.set_user_attr("return_pct", result["return_pct"])
            trial.set_user_attr("max_drawdown", result["max_drawdown"])

            return value

        except Exception as e:
            logger.warning(f"Trial failed: {e}")
            return 0.0

    def optimize(
        self,
        n_trials: int = 100,
        timeout: Optional[int] = None,
        study_name: str = "xrp_optimization",
        storage: Optional[str] = None,
        resume: bool = False,
    ) -> optuna.Study:
        """Run optimization."""
        logger.info(f"\nStarting Optuna optimization: {n_trials} trials")
        logger.info(f"Objective: maximize {self.objective_metric}")
        logger.info(f"Parameter groups: {self.param_groups}")

        # Create or load study
        sampler = TPESampler(seed=42)
        pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=0)

        if storage:
            study = optuna.create_study(
                study_name=study_name,
                storage=storage,
                load_if_exists=resume,
                direction="maximize",
                sampler=sampler,
                pruner=pruner,
            )
        else:
            study = optuna.create_study(
                direction="maximize",
                sampler=sampler,
                pruner=pruner,
            )

        # Run optimization
        study.optimize(
            self.objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True,
        )

        # Store best params
        self.best_params = study.best_params
        self.best_value = study.best_value

        return study

    def validate_on_test(self, params: Dict[str, Any] = None) -> Dict[str, float]:
        """Validate parameters on test set."""
        if params is None:
            params = self.best_params

        if params is None:
            raise ValueError("No parameters to validate")

        # Separate params by group
        space = ParameterSpace()
        fsm_params = {k: v for k, v in params.items() if k in space.FSM_PARAMS}
        risk_params = {k: v for k, v in params.items() if k in space.RISK_PARAMS}
        confirm_params = {k: v for k, v in params.items() if k in space.CONFIRM_PARAMS}
        xgb_params = {k: v for k, v in params.items() if k in space.XGB_PARAMS}
        xgb_params["XGB_ENABLED"] = True

        # Build test features
        fast_feature_names = [
            "ret_3m", "ewm_ret_15m", "atr_pct_15m", "ewm_std_ret_15m",
            "bb_width_15m", "body_ratio", "upper_wick_ratio", "lower_wick_ratio", "range_pct"
        ]
        mid_feature_names = [
            "ret_15m", "ewm_ret_15m", "ret_1h", "ewm_ret_1h",
            "atr_pct_15m", "ewm_std_ret_15m", "atr_pct_1h", "ewm_std_ret_1h",
            "bb_width_15m", "vol_z_15m", "price_z_from_ema_1h"
        ]

        # Use only OHLCV columns to avoid column overlap
        df_3m_test_ohlcv = self.df_3m_test[["timestamp", "open", "high", "low", "close", "volume"]].copy()
        fast_features_test, fast_timestamps_test = build_fast_hmm_features_v2(
            df_3m_test_ohlcv, fast_feature_names
        )
        df_15m_test_ohlcv = self.df_15m_test[["timestamp", "open", "high", "low", "close", "volume"]].copy()
        mid_features_test, mid_timestamps_test = build_mid_hmm_features_v2(
            df_15m_test_ohlcv, mid_feature_names
        )

        # Create test engine
        test_engine = FastBacktestEngine(
            df_3m=self.df_3m_test,
            df_15m=self.df_15m_test,
            df_1h=self.df_1h_test,
            multi_hmm=self.multi_hmm,
            fast_features=fast_features_test,
            fast_timestamps=fast_timestamps_test,
            mid_features=mid_features_test,
            mid_timestamps=mid_timestamps_test,
            leverage=self.leverage,
        )

        return test_engine.run(
            fsm_params=fsm_params,
            risk_params=risk_params,
            confirm_params=confirm_params,
            xgb_params=xgb_params,
        )


# ============================================================================
# Walk-Forward Optimization
# ============================================================================

def walk_forward_optimization(
    df_3m: pd.DataFrame,
    df_15m: pd.DataFrame,
    df_1h: pd.DataFrame,
    n_folds: int = 5,
    trials_per_fold: int = 50,
    leverage: float = 1.0,
    objective_metric: str = "profit_factor",
    param_groups: List[str] = None,
) -> Dict:
    """Perform walk-forward optimization."""
    logger.info(f"\nWalk-Forward Optimization: {n_folds} folds, {trials_per_fold} trials each")

    # Calculate fold sizes
    fold_size_3m = len(df_3m) // (n_folds + 1)
    fold_size_15m = len(df_15m) // (n_folds + 1)
    fold_size_1h = len(df_1h) // (n_folds + 1)

    results = []
    all_params = []

    for fold in range(n_folds):
        logger.info(f"\n{'='*60}")
        logger.info(f"Fold {fold + 1}/{n_folds}")
        logger.info("="*60)

        # Define train/test ranges for this fold
        train_end_3m = (fold + 1) * fold_size_3m
        test_end_3m = train_end_3m + fold_size_3m

        train_end_15m = (fold + 1) * fold_size_15m
        test_end_15m = train_end_15m + fold_size_15m

        train_end_1h = (fold + 1) * fold_size_1h
        test_end_1h = train_end_1h + fold_size_1h

        # Extract fold data
        df_3m_fold = df_3m.iloc[:test_end_3m].copy()
        df_15m_fold = df_15m.iloc[:test_end_15m].copy()
        df_1h_fold = df_1h.iloc[:test_end_1h].copy()

        # Set train ratio for this fold
        train_ratio = train_end_3m / test_end_3m

        logger.info(f"Train: {train_end_3m} bars, Test: {test_end_3m - train_end_3m} bars")

        # Create optimizer
        optimizer = OptunaOptimizer(
            df_3m=df_3m_fold,
            df_15m=df_15m_fold,
            df_1h=df_1h_fold,
            train_ratio=train_ratio,
            leverage=leverage,
            objective_metric=objective_metric,
            param_groups=param_groups,
        )

        # Run optimization
        study = optimizer.optimize(n_trials=trials_per_fold)

        # Validate on test set
        test_result = optimizer.validate_on_test()

        fold_result = {
            "fold": fold + 1,
            "train_value": study.best_value,
            "test_result": test_result,
            "best_params": study.best_params,
        }
        results.append(fold_result)
        all_params.append(study.best_params)

        logger.info(f"\nFold {fold + 1} Results:")
        logger.info(f"  Train {objective_metric}: {study.best_value:.3f}")
        logger.info(f"  Test {objective_metric}: {test_result[objective_metric]:.3f}")
        logger.info(f"  Test trades: {test_result['n_trades']}, WR: {test_result['win_rate']:.1f}%")

    # Aggregate results
    test_metrics = [r["test_result"][objective_metric] for r in results]

    summary = {
        "n_folds": n_folds,
        "objective_metric": objective_metric,
        "test_mean": np.mean(test_metrics),
        "test_std": np.std(test_metrics),
        "test_min": np.min(test_metrics),
        "test_max": np.max(test_metrics),
        "fold_results": results,
    }

    logger.info(f"\n{'='*60}")
    logger.info("Walk-Forward Summary")
    logger.info("="*60)
    logger.info(f"Test {objective_metric}: {summary['test_mean']:.3f} +/- {summary['test_std']:.3f}")
    logger.info(f"Range: [{summary['test_min']:.3f}, {summary['test_max']:.3f}]")

    return summary


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Optuna Parameter Optimization")
    parser.add_argument("--months", type=int, default=6, help="Months of data")
    parser.add_argument("--trials", type=int, default=100, help="Number of trials")
    parser.add_argument("--timeout", type=int, default=None, help="Timeout in seconds")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Train ratio")
    parser.add_argument("--leverage", type=float, default=1.0, help="Leverage")
    parser.add_argument("--objective", type=str, default="profit_factor",
                       choices=["profit_factor", "sharpe_ratio", "return_pct"],
                       help="Optimization objective")
    parser.add_argument("--groups", nargs="+", default=["fsm", "risk"],
                       choices=["fsm", "risk", "confirm", "xgb"],
                       help="Parameter groups to optimize")
    parser.add_argument("--min-trades", type=int, default=50, help="Minimum trades required")
    parser.add_argument("--walk-forward", action="store_true", help="Use walk-forward validation")
    parser.add_argument("--n-folds", type=int, default=5, help="Number of folds for walk-forward")
    parser.add_argument("--resume", action="store_true", help="Resume from previous study")
    parser.add_argument("--db", type=str, default=None, help="SQLite database path for study storage")
    args = parser.parse_args()

    logger.info("="*70)
    logger.info("Optuna Parameter Optimization")
    logger.info("="*70)

    # Load data
    df_3m, df_15m, df_1h = load_data_from_binance(months=args.months)

    if args.walk_forward:
        # Walk-forward optimization
        results = walk_forward_optimization(
            df_3m=df_3m,
            df_15m=df_15m,
            df_1h=df_1h,
            n_folds=args.n_folds,
            trials_per_fold=args.trials // args.n_folds,
            leverage=args.leverage,
            objective_metric=args.objective,
            param_groups=args.groups,
        )

        # Save results
        output_dir = Path("outputs/optuna_optimization")
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"walk_forward_{timestamp}.json"

        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"\nResults saved to {output_file}")

    else:
        # Standard optimization
        storage = f"sqlite:///{args.db}" if args.db else None

        optimizer = OptunaOptimizer(
            df_3m=df_3m,
            df_15m=df_15m,
            df_1h=df_1h,
            train_ratio=args.train_ratio,
            leverage=args.leverage,
            objective_metric=args.objective,
            param_groups=args.groups,
            min_trades=args.min_trades,
        )

        study = optimizer.optimize(
            n_trials=args.trials,
            timeout=args.timeout,
            storage=storage,
            resume=args.resume,
        )

        # Validate on test set
        logger.info("\n" + "="*70)
        logger.info("Validation on Test Set")
        logger.info("="*70)

        test_result = optimizer.validate_on_test()

        logger.info(f"\nTrain {args.objective}: {study.best_value:.3f}")
        logger.info(f"Test {args.objective}: {test_result[args.objective]:.3f}")
        logger.info(f"Test trades: {test_result['n_trades']}")
        logger.info(f"Test win rate: {test_result['win_rate']:.1f}%")
        logger.info(f"Test return: {test_result['return_pct']:.2f}%")
        logger.info(f"Test max drawdown: {test_result['max_drawdown']:.2f}%")

        # Print best parameters
        logger.info("\n" + "="*70)
        logger.info("Best Parameters")
        logger.info("="*70)
        for name, value in study.best_params.items():
            logger.info(f"  {name}: {value}")

        # Save results
        output_dir = Path("outputs/optuna_optimization")
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save study
        output_data = {
            "objective": args.objective,
            "param_groups": args.groups,
            "best_params": study.best_params,
            "train_value": study.best_value,
            "test_result": test_result,
            "n_trials": len(study.trials),
            "train_ratio": args.train_ratio,
            "leverage": args.leverage,
        }

        with open(output_dir / f"optimization_{timestamp}.json", "w") as f:
            json.dump(output_data, f, indent=2, default=str)

        # Save study object
        with open(output_dir / f"study_{timestamp}.pkl", "wb") as f:
            pickle.dump(study, f)

        logger.info(f"\nResults saved to {output_dir}")

        # Print recommendation
        logger.info("\n" + "="*70)
        logger.info("Recommendation")
        logger.info("="*70)

        if test_result[args.objective] > study.best_value * 0.8:
            logger.info("\n✅ Test performance is consistent with train performance.")
            logger.info("   Consider applying these parameters to configs/base.yaml")
        else:
            logger.info("\n⚠️ Test performance is significantly lower than train.")
            logger.info("   Parameters may be overfitted. Consider:")
            logger.info("   1. Using more data (--months)")
            logger.info("   2. Walk-forward validation (--walk-forward)")
            logger.info("   3. Fewer parameter groups (--groups)")


if __name__ == "__main__":
    main()
