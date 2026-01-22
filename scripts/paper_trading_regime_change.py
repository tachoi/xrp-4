#!/usr/bin/env python
"""Paper Trading with Regime Change Entry Strategy.

Uses regime change signals (RANGE->TREND, TREND reversal) for entry,
with trailing stop for exit. Based on backtest results showing 706% return
with 94.8% win rate.

Runs separately from the existing FSM-based paper trading system.

Usage:
    python scripts/paper_trading_regime_change.py
    python scripts/paper_trading_regime_change.py --port 8081  # For web UI
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
from hmmlearn import hmm

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Force unbuffered output
import functools
print = functools.partial(print, flush=True)


# ============================================================================
# Binance API
# ============================================================================

class BinanceClient:
    """Simple Binance REST API client for fetching klines."""

    BASE_URL = "https://fapi.binance.com"
    MAX_LIMIT_PER_REQUEST = 1500

    def __init__(self):
        self.session = requests.Session()

    def get_klines(
        self,
        symbol: str = "XRPUSDT",
        interval: str = "3m",
        limit: int = 500,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> pd.DataFrame:
        """Fetch klines (candlestick) data from Binance."""
        all_data = []
        remaining = limit
        current_end_time = end_time

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

            if remaining > 0 and len(data) == fetch_limit:
                current_end_time = data[0][0] - 1
            else:
                break

        if not all_data:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

        df = pd.DataFrame(all_data, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_base",
            "taker_buy_quote", "ignore"
        ])

        df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms")
        df["open"] = df["open"].astype(float)
        df["high"] = df["high"].astype(float)
        df["low"] = df["low"].astype(float)
        df["close"] = df["close"].astype(float)
        df["volume"] = df["volume"].astype(float)

        df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

        return df[["timestamp", "open", "high", "low", "close", "volume"]]

    def get_ticker_price(self, symbol: str = "XRPUSDT") -> float:
        """Get current ticker price."""
        url = f"{self.BASE_URL}/fapi/v1/ticker/price"
        params = {"symbol": symbol}
        response = self.session.get(url, params=params)
        response.raise_for_status()
        return float(response.json()["price"])


# ============================================================================
# Feature Builder
# ============================================================================

def build_features_live(
    df_3m: pd.DataFrame,
    df_15m: pd.DataFrame,
    df_1h: Optional[pd.DataFrame] = None
) -> Tuple[np.ndarray, pd.DataFrame, pd.DataFrame, List[str]]:
    """Build features for live trading."""
    # === 3m Features ===
    df_3m = df_3m.copy()
    close_3m = df_3m["close"].astype(float)
    high_3m = df_3m["high"].astype(float)
    low_3m = df_3m["low"].astype(float)

    df_3m["ret_3m"] = close_3m.pct_change()
    df_3m["ema_fast_3m"] = close_3m.ewm(span=20, adjust=False).mean()
    df_3m["ema_slow_3m"] = close_3m.ewm(span=50, adjust=False).mean()

    tr_3m = pd.concat([
        high_3m - low_3m,
        (high_3m - close_3m.shift(1)).abs(),
        (low_3m - close_3m.shift(1)).abs(),
    ], axis=1).max(axis=1)
    df_3m["atr_3m"] = tr_3m.rolling(14).mean()
    df_3m["volatility_3m"] = df_3m["ret_3m"].rolling(20).std()

    # === 15m Features ===
    df_15m = df_15m.copy()
    close_15m = df_15m["close"].astype(float)
    high_15m = df_15m["high"].astype(float)
    low_15m = df_15m["low"].astype(float)

    df_15m["ret_15m"] = close_15m.pct_change()
    df_15m["vol_15m"] = df_15m["ret_15m"].rolling(20).std()
    df_15m["ewm_std_ret_15m"] = df_15m["ret_15m"].ewm(span=20, adjust=False).std()

    ema_20 = close_15m.ewm(span=20, adjust=False).mean()
    ema_50 = close_15m.ewm(span=50, adjust=False).mean()
    df_15m["ema_slope_15m"] = ema_20.pct_change(5)
    df_15m["ema_20_15m"] = ema_20
    df_15m["ema_50_15m"] = ema_50
    df_15m["ewm_ret_15m"] = df_15m["ret_15m"].ewm(span=20, adjust=False).mean()

    tr_15m = pd.concat([
        high_15m - low_15m,
        (high_15m - close_15m.shift(1)).abs(),
        (low_15m - close_15m.shift(1)).abs(),
    ], axis=1).max(axis=1)
    df_15m["atr_15m"] = tr_15m.rolling(14).mean()
    df_15m["atr_pct_15m"] = df_15m["atr_15m"] / close_15m * 100

    # Rolling high/low
    df_15m["rolling_high_20"] = high_15m.rolling(20).max()
    df_15m["rolling_low_20"] = low_15m.rolling(20).min()

    # BB width
    sma_15m = close_15m.rolling(20).mean()
    std_15m = close_15m.rolling(20).std()
    df_15m["bb_width_15m"] = (2 * std_15m / sma_15m) * 100

    # Additional features
    open_15m = df_15m["open"].astype(float)
    rolling_range = df_15m["rolling_high_20"] - df_15m["rolling_low_20"]
    df_15m["range_comp_15m"] = (high_15m - low_15m) / rolling_range.replace(0, np.nan)
    df_15m["bb_width_15m_pct"] = std_15m / sma_15m

    body = (close_15m - open_15m).abs()
    candle_range = (high_15m - low_15m).replace(0, np.nan)
    df_15m["body_ratio"] = body / candle_range
    df_15m["upper_wick_ratio"] = (high_15m - df_15m[["open", "close"]].max(axis=1)) / candle_range
    df_15m["lower_wick_ratio"] = (df_15m[["open", "close"]].min(axis=1) - low_15m) / candle_range

    vol_mean = df_15m["volume"].rolling(20).mean()
    vol_std = df_15m["volume"].rolling(20).std()
    df_15m["vol_z_15m"] = (df_15m["volume"] - vol_mean) / vol_std.replace(0, np.nan)

    # === 1h Features ===
    if df_1h is not None and len(df_1h) > 0:
        df_1h = df_1h.copy()
        close_1h = df_1h["close"].astype(float)
        high_1h = df_1h["high"].astype(float)
        low_1h = df_1h["low"].astype(float)

        df_1h["ret_1h"] = close_1h.pct_change()
        df_1h["ewm_ret_1h"] = df_1h["ret_1h"].ewm(span=20, adjust=False).mean()
        df_1h["ewm_std_ret_1h"] = df_1h["ret_1h"].ewm(span=20, adjust=False).std()

        ema_20_1h = close_1h.ewm(span=20, adjust=False).mean()
        df_1h["ema_20_1h"] = ema_20_1h
        df_1h["price_z_from_ema_1h"] = (close_1h - ema_20_1h) / ema_20_1h

        tr_1h = pd.concat([
            high_1h - low_1h,
            (high_1h - close_1h.shift(1)).abs(),
            (low_1h - close_1h.shift(1)).abs(),
        ], axis=1).max(axis=1)
        df_1h["atr_1h"] = tr_1h.rolling(14).mean()
        df_1h["atr_pct_1h"] = df_1h["atr_1h"] / close_1h * 100

        # Merge 1h features to 15m
        df_1h_resampled = df_1h.set_index("timestamp")[
            ["ret_1h", "ewm_ret_1h", "ewm_std_ret_1h", "price_z_from_ema_1h", "atr_pct_1h"]
        ]
        df_15m_indexed = df_15m.set_index("timestamp")
        cols_to_drop = [c for c in df_1h_resampled.columns if c in df_15m_indexed.columns]
        if cols_to_drop:
            df_15m_indexed = df_15m_indexed.drop(columns=cols_to_drop)
        df_15m_indexed = df_15m_indexed.join(df_1h_resampled, how="left")
        df_15m_indexed[["ret_1h", "ewm_ret_1h", "ewm_std_ret_1h", "price_z_from_ema_1h", "atr_pct_1h"]] = \
            df_15m_indexed[["ret_1h", "ewm_ret_1h", "ewm_std_ret_1h", "price_z_from_ema_1h", "atr_pct_1h"]].ffill()
        df_15m = df_15m_indexed.reset_index()

        hmm_feature_cols = [
            "ret_15m", "ret_1h", "ewm_ret_15m", "ewm_ret_1h", "ema_slope_15m",
            "price_z_from_ema_1h", "atr_pct_15m", "atr_pct_1h", "ewm_std_ret_15m",
            "ewm_std_ret_1h", "bb_width_15m", "range_comp_15m", "bb_width_15m_pct",
            "body_ratio", "upper_wick_ratio", "lower_wick_ratio", "vol_z_15m"
        ]
        hmm_features = df_15m[hmm_feature_cols].dropna().values
        feature_names = hmm_feature_cols
    else:
        hmm_feature_cols = ["ret_15m", "vol_15m", "ema_slope_15m", "bb_width_15m"]
        hmm_features = df_15m[hmm_feature_cols].dropna().values
        feature_names = hmm_feature_cols

    return hmm_features, df_3m, df_15m, feature_names


# ============================================================================
# Regime Change Paper Trading Engine
# ============================================================================

class RegimeChangePaperTrader:
    """Paper trading engine using regime change entry strategy.

    Entry: Based on regime changes (RANGE->TREND, TREND reversal)
    Exit: Trailing stop (matching backtest: 0.10% activation, 60% preserve)
    """

    DATA_REQUIREMENTS = {
        "3m": {"backtest_warmup": 250},
        "15m": {"hmm_training_optimal": 2000},
        "1h": {"optimal": 500},
    }

    def __init__(
        self,
        symbol: str = "XRPUSDT",
        initial_capital: float = 10000.0,
        poll_interval: int = 5,
        leverage: float = 1.0,
    ):
        self.symbol = symbol
        self.initial_capital = initial_capital
        self.poll_interval = poll_interval
        self.leverage = leverage

        # Components
        self.client = BinanceClient()

        # Multi-HMM Manager
        config_path = Path(__file__).parent.parent / "configs" / "hmm_gate_policy.yaml"
        if config_path.exists():
            self.multi_hmm = MultiHMMManager.from_config_file(
                config_path,
                checkpoint_dir=Path(__file__).parent.parent / "checkpoints" / "hmm"
            )
            self._use_multi_hmm = True
            logger.info("Using Multi-HMM (Fast 3m + Mid 15m)")
        else:
            self.multi_hmm = None
            self._use_multi_hmm = False

        # ConfirmLayer
        self.confirm_layer = RegimeConfirmLayer(ConfirmConfig(
            HIGH_VOL_LAMBDA_ON=5.5,
            HIGH_VOL_LAMBDA_OFF=3.5,
            HIGH_VOL_COOLDOWN_BARS_15M=4,
        ))

        # State
        self.df_3m: Optional[pd.DataFrame] = None
        self.df_15m: Optional[pd.DataFrame] = None
        self.df_1h: Optional[pd.DataFrame] = None
        self._fast_feature_names: List[str] = []
        self._mid_feature_names: List[str] = []
        self._hmm_feature_names: List[str] = []
        self.confirm_state = None
        self.last_3m_ts: Optional[pd.Timestamp] = None

        # Trading state
        self.equity = initial_capital
        self.position = PositionState(side="FLAT")
        self.trades: List[Dict] = []

        # Stats
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0

        # Regime tracking for change detection
        self.prev_regime_confirmed: Optional[str] = None

        # Tick history for 5-second tracking
        self.tick_history: List[Dict] = []
        self.tick_history_max = 60

        # Sudden move detection parameters (급등/급락 감지)
        self.sudden_move_threshold = 0.005  # 0.5% move in short time
        self.sudden_move_window = 6  # 6 ticks = 30 seconds

        # Trailing stop parameters (matching backtest)
        self.TRAILING_STOP_ACTIVATION_PCT = 0.10
        self.TRAILING_STOP_PRESERVE_PCT = 0.60
        self.TRAILING_STOP_MIN_PRESERVE_PCT = 0.05

        # Break-even stop
        self.BREAKEVEN_ACTIVATION_PCT = 0.10
        self.BREAKEVEN_BUFFER_PCT = 0.05
        self.breakeven_activated = False

        # Tracking
        self.max_unrealized_pnl = 0.0
        self.max_unrealized_pnl_pct = 0.0
        self.entry_regime = None
        self.entry_signal = None

        # Pending signal for entry optimization
        self.pending_signal: Optional[Dict] = None
        self.pending_signal_timeout = 12

        # Output files
        self.output_dir = Path(__file__).parent.parent / "outputs" / "paper_trading_regime_change"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.trades_file = self.output_dir / f"trades_{self.session_id}.csv"
        self.signals_file = self.output_dir / f"signals_{self.session_id}.csv"

        self._init_csv_files()

    def _init_csv_files(self) -> None:
        """Initialize CSV files."""
        with open(self.trades_file, "w") as f:
            f.write("timestamp,side,entry_price,exit_price,size,pnl,pnl_pct,bars_held,equity_after,exit_reason,regime,signal\n")

        with open(self.signals_file, "w") as f:
            f.write("timestamp,event_type,price,prev_regime,new_regime,signal,position\n")

        logger.info(f"Trade log: {self.trades_file}")
        logger.info(f"Signal log: {self.signals_file}")

    def _save_trade(self, trade: Dict) -> None:
        """Append a trade to CSV."""
        with open(self.trades_file, "a") as f:
            f.write(
                f"{trade['timestamp']},"
                f"{trade['side']},"
                f"{trade['entry_price']:.4f},"
                f"{trade['exit_price']:.4f},"
                f"{trade['size']:.4f},"
                f"{trade['pnl']:.2f},"
                f"{trade['pnl_pct']:.2f},"
                f"{trade['bars_held']},"
                f"{trade['equity_after']:.2f},"
                f"{trade.get('exit_reason', '')},"
                f"{trade.get('regime', '')},"
                f"{trade.get('signal', '')}\n"
            )

    def _save_signal(self, price: float, prev_regime: str, new_regime: str, signal: str, event_type: str = "REGIME_CHANGE") -> None:
        """Log regime change signal."""
        with open(self.signals_file, "a") as f:
            f.write(
                f"{datetime.now().isoformat()},{event_type},{price:.4f},"
                f"{prev_regime},{new_regime},{signal},{self.position.side}\n"
            )

    def _update_tick_history(self, price: float) -> None:
        """Update tick history."""
        self.tick_history.append({
            "timestamp": datetime.now(),
            "price": price,
        })
        if len(self.tick_history) > self.tick_history_max:
            self.tick_history = self.tick_history[-self.tick_history_max:]

    def _detect_sudden_move(self) -> Optional[Dict]:
        """Detect sudden price movement in recent ticks.

        Returns:
            Dict with move info if sudden move detected, None otherwise.
            {"direction": "UP" or "DOWN", "pct_change": float, "duration_secs": int}
        """
        if len(self.tick_history) < self.sudden_move_window:
            return None

        recent = self.tick_history[-self.sudden_move_window:]
        start_price = recent[0]["price"]
        end_price = recent[-1]["price"]

        pct_change = (end_price - start_price) / start_price

        if abs(pct_change) >= self.sudden_move_threshold:
            duration = (recent[-1]["timestamp"] - recent[0]["timestamp"]).total_seconds()
            return {
                "direction": "UP" if pct_change > 0 else "DOWN",
                "pct_change": pct_change,
                "duration_secs": int(duration),
            }

        return None

    def _check_emergency_exit(self, current_price: float) -> Optional[str]:
        """Check if emergency exit is needed due to adverse sudden move.

        Returns:
            Exit reason string if should exit, None otherwise.
        """
        if self.position.side == "FLAT":
            return None

        sudden_move = self._detect_sudden_move()
        if sudden_move is None:
            return None

        # Check if move is against our position
        if self.position.side == "LONG" and sudden_move["direction"] == "DOWN":
            if abs(sudden_move["pct_change"]) >= self.sudden_move_threshold:
                return f"EMERGENCY_EXIT_SUDDEN_DROP_{abs(sudden_move['pct_change'])*100:.2f}%"

        elif self.position.side == "SHORT" and sudden_move["direction"] == "UP":
            if abs(sudden_move["pct_change"]) >= self.sudden_move_threshold:
                return f"EMERGENCY_EXIT_SUDDEN_SPIKE_{abs(sudden_move['pct_change'])*100:.2f}%"

        return None

    def _execute_emergency_exit(self, current_price: float, reason: str) -> None:
        """Execute emergency exit at current price."""
        if self.position.side == "FLAT":
            return

        fee_rate = 0.0004

        if self.position.side == "LONG":
            pnl = (current_price - self.position.entry_price) * self.position.size
        else:
            pnl = (self.position.entry_price - current_price) * self.position.size

        fee = current_price * self.position.size * fee_rate
        pnl -= fee
        self.equity += pnl
        self.total_pnl += pnl
        self.total_trades += 1

        if pnl > 0:
            self.winning_trades += 1

        pnl_pct = pnl / (self.position.entry_price * self.position.size) * 100 if self.position.size > 0 else 0

        logger.info(
            f"[EMERGENCY EXIT] {self.position.side} @ ${current_price:.4f}, "
            f"PnL: ${pnl:.2f} ({pnl_pct:.2f}%), Reason: {reason}"
        )

        trade = {
            "timestamp": datetime.now().isoformat(),
            "side": self.position.side,
            "entry_price": self.position.entry_price,
            "exit_price": current_price,
            "size": self.position.size,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "bars_held": self.position.bars_held_3m,
            "equity_after": self.equity,
            "exit_reason": reason,
            "regime": self.entry_regime,
            "signal": self.entry_signal,
        }
        self.trades.append(trade)
        self._save_trade(trade)

        # Reset position
        self.position = PositionState(side="FLAT")
        self.max_unrealized_pnl = 0.0
        self.max_unrealized_pnl_pct = 0.0
        self.breakeven_activated = False
        self.entry_regime = None
        self.entry_signal = None

    def _update_max_profit(self, current_price: float) -> None:
        """Update max unrealized profit tracking."""
        if self.position.side == "FLAT":
            return

        if self.position.side == "LONG":
            current_pnl = (current_price - self.position.entry_price) * self.position.size
            current_pnl_pct = (current_price - self.position.entry_price) / self.position.entry_price * 100
        else:
            current_pnl = (self.position.entry_price - current_price) * self.position.size
            current_pnl_pct = (self.position.entry_price - current_price) / self.position.entry_price * 100

        self.position.unrealized_pnl = current_pnl

        if current_pnl > self.max_unrealized_pnl:
            self.max_unrealized_pnl = current_pnl
            self.max_unrealized_pnl_pct = current_pnl_pct

        # Activate breakeven
        if current_pnl_pct >= self.BREAKEVEN_ACTIVATION_PCT and not self.breakeven_activated:
            self.breakeven_activated = True

    def _check_breakeven_stop(self, current_price: float) -> Optional[Tuple[float, str]]:
        """Check break-even stop."""
        if self.position.side == "FLAT" or not self.breakeven_activated:
            return None

        if self.max_unrealized_pnl_pct >= self.TRAILING_STOP_ACTIVATION_PCT:
            return None  # Let trailing stop handle it

        if self.position.side == "LONG":
            breakeven_price = self.position.entry_price * (1 + self.BREAKEVEN_BUFFER_PCT / 100)
            if current_price <= breakeven_price:
                return (breakeven_price, f"BREAKEVEN_STOP (max={self.max_unrealized_pnl_pct:.2f}%)")
        else:
            breakeven_price = self.position.entry_price * (1 - self.BREAKEVEN_BUFFER_PCT / 100)
            if current_price >= breakeven_price:
                return (breakeven_price, f"BREAKEVEN_STOP (max={self.max_unrealized_pnl_pct:.2f}%)")

        return None

    def _check_trailing_stop(self, current_price: float) -> Optional[Tuple[float, str]]:
        """Check trailing stop."""
        if self.position.side == "FLAT":
            return None

        if self.max_unrealized_pnl_pct < self.TRAILING_STOP_ACTIVATION_PCT:
            return None

        preserve_pct = max(
            self.max_unrealized_pnl_pct * self.TRAILING_STOP_PRESERVE_PCT,
            self.TRAILING_STOP_MIN_PRESERVE_PCT
        )

        if self.position.side == "LONG":
            preserve_price = self.position.entry_price * (1 + preserve_pct / 100)
            if current_price <= preserve_price:
                exit_price = max(current_price, preserve_price)
                return (exit_price, f"TRAILING_STOP (max={self.max_unrealized_pnl_pct:.2f}%, preserved={preserve_pct:.2f}%)")
        else:
            preserve_price = self.position.entry_price * (1 - preserve_pct / 100)
            if current_price >= preserve_price:
                exit_price = min(current_price, preserve_price)
                return (exit_price, f"TRAILING_STOP (max={self.max_unrealized_pnl_pct:.2f}%, preserved={preserve_pct:.2f}%)")

        return None

    def _execute_exit(self, exit_price: float, reason: str) -> None:
        """Execute position exit."""
        if self.position.side == "FLAT":
            return

        fee_rate = 0.0004

        if self.position.side == "LONG":
            pnl = (exit_price - self.position.entry_price) * self.position.size
        else:
            pnl = (self.position.entry_price - exit_price) * self.position.size

        fee = exit_price * self.position.size * fee_rate
        pnl -= fee
        self.equity += pnl
        self.total_pnl += pnl
        self.total_trades += 1

        if pnl > 0:
            self.winning_trades += 1

        final_pnl_pct = pnl / (self.position.entry_price * self.position.size) * 100 if self.position.size > 0 else 0

        logger.info(
            f"[EXIT] {self.position.side} Entry: ${self.position.entry_price:.4f}, "
            f"Exit: ${exit_price:.4f}, PnL: ${pnl:.2f} ({final_pnl_pct:.2f}%), "
            f"Max: {self.max_unrealized_pnl_pct:.2f}%, Reason: {reason}"
        )

        trade = {
            "timestamp": datetime.now().isoformat(),
            "side": self.position.side,
            "entry_price": self.position.entry_price,
            "exit_price": exit_price,
            "size": self.position.size,
            "pnl": pnl,
            "pnl_pct": final_pnl_pct,
            "bars_held": self.position.bars_held_3m,
            "equity_after": self.equity,
            "exit_reason": reason,
            "regime": self.entry_regime,
            "signal": self.entry_signal,
        }
        self.trades.append(trade)
        self._save_trade(trade)

        # Reset
        self.position = PositionState(side="FLAT")
        self.max_unrealized_pnl = 0.0
        self.max_unrealized_pnl_pct = 0.0
        self.breakeven_activated = False
        self.entry_regime = None
        self.entry_signal = None

    def _execute_entry(self, side: str, price: float, signal: str, regime: str) -> None:
        """Execute position entry."""
        if self.position.side != "FLAT":
            return

        fee_rate = 0.0004
        max_position_value = self.equity * self.leverage
        size = max_position_value / price

        fee = price * size * fee_rate
        self.equity -= fee

        self.position = PositionState(
            side=side,
            entry_price=price,
            size=size,
            entry_ts=int(datetime.now().timestamp() * 1000),
            bars_held_3m=0,
            unrealized_pnl=0,
        )

        self.entry_regime = regime
        self.entry_signal = signal
        self.max_unrealized_pnl = 0.0
        self.max_unrealized_pnl_pct = 0.0
        self.breakeven_activated = False

        logger.info(
            f"[ENTRY] {side} @ ${price:.4f}, Size: {size:.2f} XRP (${max_position_value:.0f}), "
            f"Signal: {signal}, Regime: {regime}"
        )

    def _set_pending_signal(self, side: str, price: float, signal: str, regime: str) -> None:
        """Set pending signal for entry optimization."""
        self.pending_signal = {
            "side": side,
            "signal_price": price,
            "signal": signal,
            "regime": regime,
            "ticks_waited": 0,
            "best_price": price,
            "created_at": datetime.now(),
        }
        logger.info(f"[PENDING] {side} @ ${price:.4f} - Signal: {signal}, waiting for optimal entry")

    def _check_pending_entry(self, current_price: float) -> bool:
        """Check if pending signal should execute. Returns True if executed."""
        if self.pending_signal is None:
            return False

        self.pending_signal["ticks_waited"] += 1
        signal_price = self.pending_signal["signal_price"]
        side = self.pending_signal["side"]

        # Update best price
        if side == "LONG":
            if current_price < self.pending_signal["best_price"]:
                self.pending_signal["best_price"] = current_price
            improvement = (signal_price - current_price) / signal_price
        else:
            if current_price > self.pending_signal["best_price"]:
                self.pending_signal["best_price"] = current_price
            improvement = (current_price - signal_price) / signal_price

        # Entry conditions
        execute = False
        reason = ""

        if self.pending_signal["ticks_waited"] >= 2:
            # Check for pullback rebound / bounce rejection
            if len(self.tick_history) >= 3:
                recent_prices = [t["price"] for t in self.tick_history[-3:]]

                if side == "LONG":
                    is_pullback = current_price < signal_price
                    is_rebounding = recent_prices[-1] > recent_prices[-2] > recent_prices[-3]
                    if is_pullback and is_rebounding:
                        execute = True
                        reason = f"PULLBACK_REBOUND (saved {improvement*100:.3f}%)"
                else:
                    is_bounce = current_price > signal_price
                    is_rejecting = recent_prices[-1] < recent_prices[-2] < recent_prices[-3]
                    if is_bounce and is_rejecting:
                        execute = True
                        reason = f"BOUNCE_REJECTION (saved {improvement*100:.3f}%)"

            # Timeout
            if not execute and self.pending_signal["ticks_waited"] >= self.pending_signal_timeout:
                execute = True
                reason = "TIMEOUT"

            # Price running away
            if not execute:
                if side == "LONG" and current_price > signal_price * 1.002:
                    execute = True
                    reason = "PRICE_RUNNING"
                elif side == "SHORT" and current_price < signal_price * 0.998:
                    execute = True
                    reason = "PRICE_RUNNING"

        if execute:
            logger.info(f"[OPTIMIZED ENTRY] {reason} @ ${current_price:.4f}")
            self._execute_entry(
                side=self.pending_signal["side"],
                price=current_price,
                signal=self.pending_signal["signal"],
                regime=self.pending_signal["regime"],
            )
            self.pending_signal = None
            return True

        return False

    def initialize(self) -> None:
        """Initialize with historical data."""
        logger.info("Initializing Regime Change Paper Trading Engine...")

        bars_3m = 250
        bars_15m = 2000
        bars_1h = 500

        logger.info(f"Fetching historical 3m data ({bars_3m} bars)...")
        self.df_3m = self.client.get_klines(self.symbol, "3m", limit=bars_3m)
        logger.info(f"  -> Received {len(self.df_3m)} bars")

        logger.info(f"Fetching historical 15m data ({bars_15m} bars)...")
        self.df_15m = self.client.get_klines(self.symbol, "15m", limit=bars_15m)
        logger.info(f"  -> Received {len(self.df_15m)} bars")

        logger.info(f"Fetching historical 1h data ({bars_1h} bars)...")
        self.df_1h = self.client.get_klines(self.symbol, "1h", limit=bars_1h)
        logger.info(f"  -> Received {len(self.df_1h)} bars")

        # Load feature config
        feature_config_path = Path(__file__).parent.parent / "configs" / "hmm_features.yaml"
        if feature_config_path.exists():
            with open(feature_config_path, "r") as f:
                feature_config = yaml.safe_load(f)
            self._fast_feature_names = feature_config["hmm"]["fast_3m"]["features"]
            self._mid_feature_names = feature_config["hmm"]["mid_15m"]["features"]
        else:
            self._fast_feature_names = [
                "ret_3m", "ewm_ret_15m", "atr_pct_15m", "ewm_std_ret_15m",
                "bb_width_15m", "body_ratio", "upper_wick_ratio", "lower_wick_ratio", "range_pct"
            ]
            self._mid_feature_names = [
                "ret_15m", "ewm_ret_15m", "ret_1h", "ewm_ret_1h",
                "atr_pct_15m", "ewm_std_ret_15m", "atr_pct_1h", "ewm_std_ret_1h",
                "bb_width_15m", "vol_z_15m", "price_z_from_ema_1h"
            ]

        # Build features and load HMM
        if self._use_multi_hmm:
            logger.info("Building Multi-HMM features...")
            fast_features, fast_timestamps = build_fast_hmm_features_v2(
                self.df_3m, self._fast_feature_names
            )
            df_15m_ohlcv = self.df_15m[["timestamp", "open", "high", "low", "close", "volume"]].copy()
            mid_features, mid_timestamps = build_mid_hmm_features_v2(
                df_15m_ohlcv, self._mid_feature_names
            )

            # Load checkpoint
            checkpoint_dir = Path(__file__).parent.parent / "checkpoints" / "hmm"
            if checkpoint_dir.exists():
                latest_run_id_file = checkpoint_dir / "latest_run_id.txt"
                if latest_run_id_file.exists():
                    with open(latest_run_id_file, "r") as f:
                        run_id = f.read().strip()
                    try:
                        self.multi_hmm.load_checkpoints(run_id)
                        logger.info(f"Multi-HMM loaded from checkpoint: {run_id}")
                    except Exception as e:
                        logger.warning(f"Failed to load checkpoint: {e}")

        # Build features for live use
        _, self.df_3m, self.df_15m, self._hmm_feature_names = build_features_live(
            self.df_3m, self.df_15m, self.df_1h
        )

        self.last_3m_ts = self.df_3m["timestamp"].iloc[-2]
        logger.info(f"Initialization complete. Last closed 3m bar: {self.last_3m_ts}")
        logger.info(f"Current price: ${self.df_3m['close'].iloc[-1]:.4f}")

    def update_data(self) -> bool:
        """Fetch latest data. Returns True if new 3m bar closed."""
        new_3m = self.client.get_klines(self.symbol, "3m", limit=10)
        new_15m = self.client.get_klines(self.symbol, "15m", limit=10)
        new_1h = self.client.get_klines(self.symbol, "1h", limit=5)

        latest_ts = new_3m["timestamp"].iloc[-2]

        if self.last_3m_ts is None or latest_ts > self.last_3m_ts:
            self.df_3m = pd.concat([self.df_3m, new_3m]).drop_duplicates(subset=["timestamp"]).tail(500)
            self.df_15m = pd.concat([self.df_15m, new_15m]).drop_duplicates(subset=["timestamp"]).tail(500)
            self.df_1h = pd.concat([self.df_1h, new_1h]).drop_duplicates(subset=["timestamp"]).tail(500)

            _, self.df_3m, self.df_15m, self._hmm_feature_names = build_features_live(
                self.df_3m, self.df_15m, self.df_1h
            )

            self.last_3m_ts = latest_ts
            return True

        return False

    def detect_regime(self) -> Tuple[str, str, str]:
        """Detect current regime using Multi-HMM.

        Returns:
            (regime_raw, regime_confirmed, confirm_reason)
        """
        if self._use_multi_hmm and self.multi_hmm:
            # Build features
            fast_features, fast_ts = build_fast_hmm_features_v2(
                self.df_3m.tail(100), self._fast_feature_names
            )
            df_15m_ohlcv = self.df_15m[["timestamp", "open", "high", "low", "close", "volume"]].copy()
            mid_features, mid_ts = build_mid_hmm_features_v2(
                df_15m_ohlcv.tail(200), self._mid_feature_names
            )

            if len(fast_features) == 0 or len(mid_features) == 0:
                return "RANGE", "RANGE", "NO_FEATURES"

            # Predict
            result = self.multi_hmm.predict(fast_features[-1:], mid_features[-1:])
            regime_raw = result.label_fused.value
        else:
            regime_raw = "RANGE"

        # Confirm layer - match CLI paper_trading.py behavior
        bar = self.df_3m.iloc[-1]
        bar_15m = self.df_15m.iloc[-1]
        hist_15m = self.df_15m.tail(20)  # Match CLI (20 bars, not 96)

        # Build row_15m with full data (matching CLI)
        row_15m_dict = bar_15m.to_dict()
        row_15m_dict["ret_3m"] = bar.get("ret_3m", 0)  # Add 3m return for spike detection
        row_15m_dict["ret_15m"] = bar_15m.get("ret_15m", 0)  # Add 15m return for spike detection

        confirm_result, self.confirm_state = self.confirm_layer.confirm(
            regime_raw=regime_raw,
            row_15m=row_15m_dict,
            hist_15m=hist_15m,
            state=self.confirm_state,
        )

        return regime_raw, confirm_result.confirmed_regime, confirm_result.reason

    def detect_regime_change(self, current_regime: str) -> Optional[str]:
        """Detect regime change and return entry signal if applicable.

        Returns:
            Entry signal name or None
        """
        if self.prev_regime_confirmed is None:
            return None

        if current_regime == self.prev_regime_confirmed:
            return None

        signal = None

        # RANGE -> TREND transitions
        if self.prev_regime_confirmed == "RANGE" and current_regime == "TREND_UP":
            signal = "REGIME_LONG_RANGE_TO_UP"
        elif self.prev_regime_confirmed == "RANGE" and current_regime == "TREND_DOWN":
            signal = "REGIME_SHORT_RANGE_TO_DOWN"
        # TREND reversals
        elif self.prev_regime_confirmed == "TREND_DOWN" and current_regime == "TREND_UP":
            signal = "REGIME_LONG_REVERSAL"
        elif self.prev_regime_confirmed == "TREND_UP" and current_regime == "TREND_DOWN":
            signal = "REGIME_SHORT_REVERSAL"

        return signal

    def run(self) -> None:
        """Main trading loop."""
        logger.info("=" * 60)
        logger.info("REGIME CHANGE PAPER TRADING STARTED")
        logger.info(f"Symbol: {self.symbol}")
        logger.info(f"Initial Capital: ${self.initial_capital:.2f}")
        logger.info(f"Leverage: {self.leverage}x")
        logger.info(f"Strategy: Regime Change Entry + Trailing Stop Exit")
        logger.info("=" * 60)

        self.initialize()

        poll_count = 0
        last_heartbeat = 0

        while True:
            try:
                poll_count += 1

                # Get current price
                current_price = self.client.get_ticker_price(self.symbol)
                self._update_tick_history(current_price)

                # Check stops if in position
                if self.position.side != "FLAT":
                    self._update_max_profit(current_price)

                    # Emergency exit (급등/급락)
                    emergency_reason = self._check_emergency_exit(current_price)
                    if emergency_reason:
                        logger.info(f"[EMERGENCY] {emergency_reason}")
                        self._execute_emergency_exit(current_price, emergency_reason)
                    else:
                        # Break-even stop
                        be_result = self._check_breakeven_stop(current_price)
                        if be_result:
                            self._execute_exit(be_result[0], be_result[1])
                        else:
                            # Trailing stop
                            ts_result = self._check_trailing_stop(current_price)
                            if ts_result:
                                self._execute_exit(ts_result[0], ts_result[1])

                # Check pending signal
                if self.pending_signal is not None:
                    self._check_pending_entry(current_price)

                # Check for new 3m bar
                new_bar = self.update_data()

                if new_bar:
                    # Cancel pending signal on new bar
                    if self.pending_signal:
                        logger.info("[CANCELLED] Pending signal - new bar")
                        self.pending_signal = None

                    # Detect regime
                    regime_raw, regime_confirmed, confirm_reason = self.detect_regime()
                    price = float(self.df_3m.iloc[-1]["close"])

                    # Detect regime change
                    regime_signal = self.detect_regime_change(regime_confirmed)

                    # Log regime change
                    if regime_signal:
                        logger.info(
                            f"[REGIME CHANGE] {self.prev_regime_confirmed} -> {regime_confirmed} | "
                            f"Signal: {regime_signal} | Price: ${price:.4f}"
                        )
                        self._save_signal(price, self.prev_regime_confirmed or "NONE", regime_confirmed, regime_signal)

                        # Create entry signal if flat
                        if self.position.side == "FLAT":
                            if "LONG" in regime_signal:
                                self._set_pending_signal("LONG", price, regime_signal, regime_confirmed)
                            elif "SHORT" in regime_signal:
                                self._set_pending_signal("SHORT", price, regime_signal, regime_confirmed)

                    # Update previous regime
                    self.prev_regime_confirmed = regime_confirmed

                    # Update position bars held
                    if self.position.side != "FLAT":
                        self.position.bars_held_3m += 1

                # Heartbeat every 30 seconds
                if poll_count * self.poll_interval >= last_heartbeat + 30:
                    last_heartbeat = poll_count * self.poll_interval
                    win_rate = (self.winning_trades / max(1, self.total_trades)) * 100
                    status = f"[HEARTBEAT] Price: ${current_price:.4f}"
                    if self.prev_regime_confirmed:
                        status += f" | Regime: {self.prev_regime_confirmed}"
                    status += f" | Equity: ${self.equity:.2f} ({(self.equity/self.initial_capital-1)*100:+.2f}%)"
                    status += f" | Trades: {self.total_trades} ({win_rate:.0f}% WR)"
                    if self.position.side != "FLAT":
                        status += f" | Pos: {self.position.side} (${self.position.unrealized_pnl:.2f})"
                    logger.info(status)

                time.sleep(self.poll_interval)

            except KeyboardInterrupt:
                logger.info("\n" + "=" * 60)
                logger.info("PAPER TRADING STOPPED")
                logger.info("=" * 60)
                self._print_summary()
                break

            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(10)

    def _print_summary(self) -> None:
        """Print trading session summary."""
        win_rate = (self.winning_trades / max(1, self.total_trades)) * 100
        return_pct = (self.equity / self.initial_capital - 1) * 100

        logger.info(f"Final Equity: ${self.equity:.2f}")
        logger.info(f"Total Return: {return_pct:+.2f}%")
        logger.info(f"Total Trades: {self.total_trades}")
        logger.info(f"Win Rate: {win_rate:.1f}%")
        logger.info(f"Total PnL: ${self.total_pnl:.2f}")

    # === Properties for web interface ===
    def get_stats(self) -> Dict:
        """Get current trading statistics."""
        win_rate = (self.winning_trades / max(1, self.total_trades)) * 100
        return {
            "equity": self.equity,
            "initial_capital": self.initial_capital,
            "total_pnl": self.equity - self.initial_capital,
            "pnl_pct": (self.equity / self.initial_capital - 1) * 100,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "win_rate": win_rate,
        }

    def get_position(self) -> Dict:
        """Get current position."""
        if self.position.side == "FLAT":
            return {"side": "FLAT", "size": 0, "entry_price": 0, "unrealized_pnl": 0}
        return {
            "side": self.position.side,
            "size": self.position.size,
            "entry_price": self.position.entry_price,
            "unrealized_pnl": self.position.unrealized_pnl,
            "unrealized_pnl_pct": self.position.unrealized_pnl / (self.position.entry_price * self.position.size) * 100 if self.position.size > 0 else 0,
            "bars_held": self.position.bars_held_3m,
            "max_profit_pct": self.max_unrealized_pnl_pct,
        }


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Regime Change Entry Paper Trading")
    parser.add_argument("--symbol", type=str, default="XRPUSDT", help="Trading symbol")
    parser.add_argument("--capital", type=float, default=10000.0, help="Initial capital")
    parser.add_argument("--leverage", type=float, default=1.0, help="Leverage")
    parser.add_argument("--poll", type=int, default=5, help="Poll interval in seconds")
    args = parser.parse_args()

    trader = RegimeChangePaperTrader(
        symbol=args.symbol,
        initial_capital=args.capital,
        leverage=args.leverage,
        poll_interval=args.poll,
    )

    trader.run()


if __name__ == "__main__":
    main()
