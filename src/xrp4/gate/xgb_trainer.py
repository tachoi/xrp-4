"""XGBoost Gate Trainer for trade filtering.

This module trains an XGBoost classifier to filter out bad trades
based on features at the time of trade entry.

Usage:
    from xrp4.gate.xgb_trainer import XGBTrainer

    trainer = XGBTrainer()
    model_path = trainer.train(months=4, leverage=5.0)
"""

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
import xgboost as xgb
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class BinanceClient:
    """Binance REST API client for fetching historical klines."""

    BASE_URL = "https://api.binance.com"
    MAX_LIMIT_PER_REQUEST = 1000

    def __init__(self):
        self.session = requests.Session()

    def get_klines(
        self,
        symbol: str = "XRPUSDT",
        interval: str = "3m",
        limit: int = 500,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        show_progress: bool = False,
    ) -> pd.DataFrame:
        """Fetch klines with pagination support."""
        all_data = []
        remaining = limit
        current_end_time = end_time
        total_fetched = 0

        while remaining > 0:
            fetch_limit = min(remaining, self.MAX_LIMIT_PER_REQUEST)

            url = f"{self.BASE_URL}/api/v3/klines"
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
            total_fetched += len(data)

            if show_progress and limit > 1000:
                pct = (total_fetched / limit) * 100
                print(f"\r  Progress: {total_fetched:,}/{limit:,} ({pct:.0f}%)", end="", flush=True)

            if remaining > 0 and len(data) == fetch_limit:
                current_end_time = data[0][0] - 1
                time.sleep(0.05)
            else:
                break

        if show_progress and limit > 1000:
            print()

        if not all_data:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

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


class XGBTrainer:
    """XGBoost trainer for gate model.

    Trains an XGBoost classifier to predict whether a trade will be profitable,
    allowing the system to filter out likely losing trades.
    """

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        checkpoint_dir: Optional[Path] = None,
    ):
        """Initialize XGBTrainer.

        Args:
            output_dir: Directory to save trained model
            checkpoint_dir: Directory for HMM checkpoints
        """
        if output_dir is None:
            output_dir = Path(__file__).parent.parent.parent.parent / "outputs" / "xgb_gate"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if checkpoint_dir is None:
            checkpoint_dir = Path(__file__).parent.parent.parent.parent / "checkpoints" / "hmm"
        self.checkpoint_dir = Path(checkpoint_dir)

        self.client = BinanceClient()
        self.model: Optional[xgb.XGBClassifier] = None
        self.feature_cols: List[str] = []
        self.best_threshold: float = 0.5

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build 60+ features for XGB training.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with added feature columns
        """
        df = df.copy()

        # =========================================================================
        # 1. Returns (multiple timeframes) - 10 features
        # =========================================================================
        for period in [1, 2, 3, 5, 10, 15, 20, 30, 50, 100]:
            df[f"ret_{period}"] = df["close"].pct_change(period)

        # Log returns
        df["log_ret_1"] = np.log(df["close"] / df["close"].shift(1))
        df["log_ret_5"] = np.log(df["close"] / df["close"].shift(5))

        # =========================================================================
        # 2. EMAs and relationships - 12 features
        # =========================================================================
        for span in [5, 10, 20, 50, 100, 200]:
            df[f"ema_{span}"] = df["close"].ewm(span=span, adjust=False).mean()

        # EMA differences (normalized)
        df["ema_5_10_diff"] = (df["ema_5"] - df["ema_10"]) / df["ema_10"]
        df["ema_10_20_diff"] = (df["ema_10"] - df["ema_20"]) / df["ema_20"]
        df["ema_20_50_diff"] = (df["ema_20"] - df["ema_50"]) / df["ema_50"]
        df["ema_50_100_diff"] = (df["ema_50"] - df["ema_100"]) / df["ema_100"]
        df["ema_100_200_diff"] = (df["ema_100"] - df["ema_200"]) / df["ema_200"]

        # Price to EMA
        df["price_to_ema5"] = (df["close"] - df["ema_5"]) / df["ema_5"]
        df["price_to_ema10"] = (df["close"] - df["ema_10"]) / df["ema_10"]
        df["price_to_ema20"] = (df["close"] - df["ema_20"]) / df["ema_20"]
        df["price_to_ema50"] = (df["close"] - df["ema_50"]) / df["ema_50"]
        df["price_to_ema100"] = (df["close"] - df["ema_100"]) / df["ema_100"]
        df["price_to_ema200"] = (df["close"] - df["ema_200"]) / df["ema_200"]

        # EMA slopes
        df["ema_slope_5"] = df["ema_10"].pct_change(5)
        df["ema_slope_10"] = df["ema_20"].pct_change(10)
        df["ema_slope_20"] = df["ema_50"].pct_change(20)

        # =========================================================================
        # 3. Volatility features - 8 features
        # =========================================================================
        df["volatility_5"] = df["ret_1"].rolling(5).std()
        df["volatility_10"] = df["ret_1"].rolling(10).std()
        df["volatility_20"] = df["ret_1"].rolling(20).std()
        df["volatility_50"] = df["ret_1"].rolling(50).std()

        df["volatility_ratio"] = df["volatility_5"] / (df["volatility_20"] + 1e-10)
        df["range_pct"] = (df["high"] - df["low"]) / df["close"]
        df["range_pct_ma5"] = df["range_pct"].rolling(5).mean()
        df["range_pct_ma20"] = df["range_pct"].rolling(20).mean()

        # ATR variations
        df["atr_14"] = self._calculate_atr(df, period=14)
        df["atr_20"] = self._calculate_atr(df, period=20)
        df["atr_pct_14"] = df["atr_14"] / df["close"]
        df["atr_pct_20"] = df["atr_20"] / df["close"]
        df["atr_ratio"] = df["atr_14"] / (df["atr_20"] + 1e-10)

        # =========================================================================
        # 4. Volume features - 6 features
        # =========================================================================
        df["volume_ma5"] = df["volume"].rolling(5).mean()
        df["volume_ma20"] = df["volume"].rolling(20).mean()
        df["volume_ma50"] = df["volume"].rolling(50).mean()

        df["volume_ratio_5"] = df["volume"] / (df["volume_ma5"] + 1e-10)
        df["volume_ratio_20"] = df["volume"] / (df["volume_ma20"] + 1e-10)
        df["volume_ratio_50"] = df["volume"] / (df["volume_ma50"] + 1e-10)

        df["volume_trend"] = df["volume_ma5"] / (df["volume_ma20"] + 1e-10)

        # =========================================================================
        # 5. RSI features - 4 features
        # =========================================================================
        df["rsi_7"] = self._calculate_rsi(df["close"], period=7)
        df["rsi_14"] = self._calculate_rsi(df["close"], period=14)
        df["rsi_21"] = self._calculate_rsi(df["close"], period=21)
        df["rsi_slope"] = df["rsi_14"] - df["rsi_14"].shift(5)

        # =========================================================================
        # 6. MACD features - 4 features
        # =========================================================================
        ema_12 = df["close"].ewm(span=12, adjust=False).mean()
        ema_26 = df["close"].ewm(span=26, adjust=False).mean()
        df["macd"] = ema_12 - ema_26
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]
        df["macd_hist_slope"] = df["macd_hist"] - df["macd_hist"].shift(3)

        # =========================================================================
        # 7. Bollinger Bands features - 6 features
        # =========================================================================
        df["bb_mid"] = df["close"].rolling(20).mean()
        df["bb_std"] = df["close"].rolling(20).std()
        df["bb_upper"] = df["bb_mid"] + 2 * df["bb_std"]
        df["bb_lower"] = df["bb_mid"] - 2 * df["bb_std"]
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / (df["bb_mid"] + 1e-10)
        df["bb_position"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-10)
        df["bb_width_ma"] = df["bb_width"].rolling(10).mean()
        df["bb_squeeze"] = df["bb_width"] / (df["bb_width_ma"] + 1e-10)

        # =========================================================================
        # 8. Stochastic features - 4 features
        # =========================================================================
        low_14 = df["low"].rolling(14).min()
        high_14 = df["high"].rolling(14).max()
        df["stoch_k"] = 100 * (df["close"] - low_14) / (high_14 - low_14 + 1e-10)
        df["stoch_d"] = df["stoch_k"].rolling(3).mean()
        df["stoch_diff"] = df["stoch_k"] - df["stoch_d"]
        df["stoch_slope"] = df["stoch_k"] - df["stoch_k"].shift(3)

        # =========================================================================
        # 9. Candle pattern features - 8 features
        # =========================================================================
        df["body_size"] = abs(df["close"] - df["open"])
        df["body_ratio"] = df["body_size"] / (df["high"] - df["low"] + 1e-10)
        df["upper_wick"] = (df["high"] - df[["open", "close"]].max(axis=1)) / (df["high"] - df["low"] + 1e-10)
        df["lower_wick"] = (df[["open", "close"]].min(axis=1) - df["low"]) / (df["high"] - df["low"] + 1e-10)
        df["candle_direction"] = np.sign(df["close"] - df["open"])
        df["body_to_atr"] = df["body_size"] / (df["atr_14"] + 1e-10)

        # Consecutive candles
        df["consec_up"] = (df["candle_direction"] > 0).rolling(5).sum()
        df["consec_down"] = (df["candle_direction"] < 0).rolling(5).sum()

        # =========================================================================
        # 10. Momentum features - 6 features
        # =========================================================================
        df["momentum_3"] = df["close"] - df["close"].shift(3)
        df["momentum_5"] = df["close"] - df["close"].shift(5)
        df["momentum_10"] = df["close"] - df["close"].shift(10)
        df["momentum_20"] = df["close"] - df["close"].shift(20)

        # Rate of change
        df["roc_5"] = (df["close"] - df["close"].shift(5)) / (df["close"].shift(5) + 1e-10) * 100
        df["roc_10"] = (df["close"] - df["close"].shift(10)) / (df["close"].shift(10) + 1e-10) * 100

        # =========================================================================
        # 11. Trend strength features - 4 features
        # =========================================================================
        # ADX-like calculation (simplified)
        plus_dm = df["high"].diff()
        minus_dm = -df["low"].diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0

        tr = pd.concat([
            df["high"] - df["low"],
            abs(df["high"] - df["close"].shift(1)),
            abs(df["low"] - df["close"].shift(1))
        ], axis=1).max(axis=1)

        atr_14_adx = tr.rolling(14).mean()
        plus_di = 100 * (plus_dm.rolling(14).mean() / (atr_14_adx + 1e-10))
        minus_di = 100 * (minus_dm.rolling(14).mean() / (atr_14_adx + 1e-10))
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        df["adx"] = dx.rolling(14).mean()
        df["plus_di"] = plus_di
        df["minus_di"] = minus_di
        df["di_diff"] = plus_di - minus_di

        # =========================================================================
        # 12. Price position features - 4 features
        # =========================================================================
        df["high_20"] = df["high"].rolling(20).max()
        df["low_20"] = df["low"].rolling(20).min()
        df["price_position_20"] = (df["close"] - df["low_20"]) / (df["high_20"] - df["low_20"] + 1e-10)

        df["high_50"] = df["high"].rolling(50).max()
        df["low_50"] = df["low"].rolling(50).min()
        df["price_position_50"] = (df["close"] - df["low_50"]) / (df["high_50"] - df["low_50"] + 1e-10)

        df["dist_from_high_20"] = (df["high_20"] - df["close"]) / df["close"]
        df["dist_from_low_20"] = (df["close"] - df["low_20"]) / df["close"]

        # =========================================================================
        # 13. Z-score features - 4 features
        # =========================================================================
        df["price_zscore_20"] = (df["close"] - df["close"].rolling(20).mean()) / (df["close"].rolling(20).std() + 1e-10)
        df["price_zscore_50"] = (df["close"] - df["close"].rolling(50).mean()) / (df["close"].rolling(50).std() + 1e-10)
        df["volume_zscore"] = (df["volume"] - df["volume"].rolling(20).mean()) / (df["volume"].rolling(20).std() + 1e-10)
        df["volatility_zscore"] = (df["volatility_20"] - df["volatility_20"].rolling(50).mean()) / (df["volatility_20"].rolling(50).std() + 1e-10)

        return df

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high = df["high"]
        low = df["low"]
        close = df["close"].shift(1)

        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))

    def _generate_trades(
        self,
        df_3m: pd.DataFrame,
        df_15m: pd.DataFrame,
        leverage: float = 5.0,
    ) -> List[Dict]:
        """Generate trades using simplified strategy for training data.

        Uses EMA crossover + RSI for signal generation.

        Args:
            df_3m: 3-minute OHLCV data with features
            df_15m: 15-minute OHLCV data with features
            leverage: Leverage multiplier

        Returns:
            List of trade dictionaries with features and PnL
        """
        trades = []
        position = None
        fee_rate = 0.0004

        # Build features
        df = self._build_features(df_3m.copy())

        # Warmup (need more for 200 EMA)
        warmup = 250

        # Define all feature columns to collect
        feature_cols = self._get_feature_cols()

        for i in range(warmup, len(df)):
            bar = df.iloc[i]
            price = float(bar["close"])
            ts = bar["timestamp"]

            # Skip if key features have NaN
            if pd.isna(bar.get("ema_10_20_diff")) or pd.isna(bar.get("rsi_14")):
                continue

            # Get corresponding 15m features
            ts_15m = ts.floor("15min")
            df_15m_hist = df_15m[df_15m["timestamp"] <= ts_15m].tail(1)

            if len(df_15m_hist) == 0:
                continue

            bar_15m = df_15m_hist.iloc[-1]

            # Determine regime using multiple indicators
            ema_diff_short = float(bar.get("ema_5_10_diff", 0) or 0)
            ema_diff_mid = float(bar.get("ema_10_20_diff", 0) or 0)
            ema_slope = float(bar.get("ema_slope_5", 0) or 0)
            rsi = float(bar.get("rsi_14", 50) or 50)
            macd_hist = float(bar.get("macd_hist", 0) or 0)
            adx = float(bar.get("adx", 20) or 20)
            di_diff = float(bar.get("di_diff", 0) or 0)

            # Trend detection (multi-factor)
            trend_score = 0
            if ema_diff_short > 0.0005:
                trend_score += 1
            if ema_diff_mid > 0.0005:
                trend_score += 1
            if ema_slope > 0:
                trend_score += 1
            if macd_hist > 0:
                trend_score += 1
            if di_diff > 0:
                trend_score += 1

            is_uptrend = trend_score >= 3 and adx > 20
            is_downtrend = trend_score <= 2 and adx > 20

            # Entry signals with RSI filter
            long_signal = is_uptrend and 30 < rsi < 70
            short_signal = is_downtrend and 30 < rsi < 70

            # Exit conditions
            if position is not None:
                bars_held = i - position["entry_idx"]
                unrealized_pnl = (price - position["entry_price"]) / position["entry_price"]
                if position["side"] == "SHORT":
                    unrealized_pnl = -unrealized_pnl
                unrealized_pnl *= leverage

                # Exit conditions
                should_exit = False
                if unrealized_pnl > 0.025:  # Take profit 2.5%
                    should_exit = True
                elif unrealized_pnl < -0.012:  # Stop loss 1.2%
                    should_exit = True
                elif bars_held > 50:  # Max hold 50 bars
                    should_exit = True
                elif position["side"] == "LONG" and trend_score <= 1:
                    should_exit = True
                elif position["side"] == "SHORT" and trend_score >= 4:
                    should_exit = True

                if should_exit:
                    # Calculate PnL
                    pnl_pct = (price - position["entry_price"]) / position["entry_price"]
                    if position["side"] == "SHORT":
                        pnl_pct = -pnl_pct
                    pnl_pct = pnl_pct * leverage - fee_rate * 2

                    # Record trade with features
                    trade = {
                        "timestamp": position["timestamp"],
                        "side": position["side"],
                        "entry_price": position["entry_price"],
                        "exit_price": price,
                        "bars_held": bars_held,
                        "pnl_pct": pnl_pct,
                        "win": 1 if pnl_pct > 0 else 0,
                        # Features at entry
                        **position["features"],
                    }
                    trades.append(trade)
                    position = None

            # Entry
            if position is None:
                side = None
                if long_signal:
                    side = "LONG"
                elif short_signal:
                    side = "SHORT"

                if side:
                    # Collect all features at entry
                    features = {}
                    for col in feature_cols:
                        if col == "side_num":
                            features[col] = 1 if side == "LONG" else -1
                        elif col in bar.index:
                            val = bar[col]
                            features[col] = float(val) if not pd.isna(val) else 0.0
                        else:
                            features[col] = 0.0

                    position = {
                        "timestamp": ts,
                        "side": side,
                        "entry_price": price,
                        "entry_idx": i,
                        "features": features,
                    }

        return trades

    def _get_feature_cols(self) -> List[str]:
        """Get list of all feature columns (60+)."""
        return [
            # Returns (12)
            "ret_1", "ret_2", "ret_3", "ret_5", "ret_10", "ret_15",
            "ret_20", "ret_30", "ret_50", "ret_100",
            "log_ret_1", "log_ret_5",
            # EMA relationships (9)
            "ema_5_10_diff", "ema_10_20_diff", "ema_20_50_diff",
            "ema_50_100_diff", "ema_100_200_diff",
            "price_to_ema5", "price_to_ema10", "price_to_ema20", "price_to_ema50",
            # price_to_ema100, price_to_ema200
            # EMA slopes (3)
            "ema_slope_5", "ema_slope_10", "ema_slope_20",
            # Volatility (8)
            "volatility_5", "volatility_10", "volatility_20", "volatility_50",
            "volatility_ratio", "range_pct", "range_pct_ma5", "range_pct_ma20",
            # ATR (5)
            "atr_pct_14", "atr_pct_20", "atr_ratio",
            # Volume (5)
            "volume_ratio_5", "volume_ratio_20", "volume_ratio_50", "volume_trend",
            # RSI (4)
            "rsi_7", "rsi_14", "rsi_21", "rsi_slope",
            # MACD (4)
            "macd", "macd_signal", "macd_hist", "macd_hist_slope",
            # Bollinger (4)
            "bb_width", "bb_position", "bb_width_ma", "bb_squeeze",
            # Stochastic (4)
            "stoch_k", "stoch_d", "stoch_diff", "stoch_slope",
            # Candle (6)
            "body_ratio", "upper_wick", "lower_wick",
            "candle_direction", "body_to_atr", "consec_up",
            # Momentum (6)
            "momentum_3", "momentum_5", "momentum_10", "momentum_20",
            "roc_5", "roc_10",
            # Trend strength (4)
            "adx", "plus_di", "minus_di", "di_diff",
            # Price position (4)
            "price_position_20", "price_position_50",
            "dist_from_high_20", "dist_from_low_20",
            # Z-scores (4)
            "price_zscore_20", "price_zscore_50", "volume_zscore", "volatility_zscore",
            # Side (1)
            "side_num",
        ]

    def train(
        self,
        months: int = 4,
        leverage: float = 5.0,
        symbol: str = "XRPUSDT",
        test_ratio: float = 0.3,
    ) -> Path:
        """Train XGB gate model.

        Args:
            months: Months of historical data to use
            leverage: Leverage for trade simulation
            symbol: Trading symbol
            test_ratio: Ratio of data for testing

        Returns:
            Path to saved model
        """
        logger.info("=" * 70)
        logger.info("XGB GATE TRAINER")
        logger.info("=" * 70)
        logger.info(f"Symbol: {symbol}")
        logger.info(f"Months: {months}")
        logger.info(f"Leverage: {leverage}x")

        # Fetch data
        days = months * 30
        bars_3m = days * 24 * 20
        bars_15m = days * 24 * 4

        logger.info(f"\nFetching data...")
        logger.info(f"  3m bars: {bars_3m:,}")

        df_3m = self.client.get_klines(symbol, "3m", limit=bars_3m, show_progress=True)
        df_15m = self.client.get_klines(symbol, "15m", limit=bars_15m, show_progress=True)

        logger.info(f"  Received: 3m={len(df_3m):,}, 15m={len(df_15m):,}")

        if len(df_3m) > 0:
            start_date = df_3m["timestamp"].iloc[0].strftime("%Y-%m-%d")
            end_date = df_3m["timestamp"].iloc[-1].strftime("%Y-%m-%d")
            logger.info(f"  Range: {start_date} to {end_date}")

        # Build features for 15m
        df_15m = self._build_features(df_15m)

        # Generate trades
        logger.info(f"\nGenerating trades...")
        trades = self._generate_trades(df_3m, df_15m, leverage=leverage)
        logger.info(f"  Total trades: {len(trades)}")

        if len(trades) < 100:
            raise ValueError(f"Not enough trades for training: {len(trades)}")

        # Create DataFrame
        df_trades = pd.DataFrame(trades)

        # Define feature columns (60+)
        self.feature_cols = self._get_feature_cols()
        logger.info(f"  Using {len(self.feature_cols)} features")

        # Prepare features and labels
        # Only use columns that exist in the trades
        available_cols = [c for c in self.feature_cols if c in df_trades.columns]
        missing_cols = [c for c in self.feature_cols if c not in df_trades.columns]
        if missing_cols:
            logger.warning(f"  Missing {len(missing_cols)} features: {missing_cols[:5]}...")

        self.feature_cols = available_cols
        X = df_trades[self.feature_cols].values
        y = df_trades["win"].values

        # Handle NaN/inf
        X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)

        # Split train/test by time (not random)
        split_idx = int(len(X) * (1 - test_ratio))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        logger.info(f"\nTraining XGBoost...")
        logger.info(f"  Train: {len(X_train)} trades")
        logger.info(f"  Test: {len(X_test)} trades")

        # Train XGBoost with more estimators for 60+ features
        self.model = xgb.XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.7,
            min_child_weight=3,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            use_label_encoder=False,
            eval_metric="logloss",
        )

        self.model.fit(X_train, y_train)

        # Predict probabilities
        y_prob_test = self.model.predict_proba(X_test)[:, 1]

        # Find optimal threshold
        logger.info(f"\nFinding optimal threshold...")
        best_threshold, best_result = self._find_optimal_threshold(
            y_test, y_prob_test, df_trades.iloc[split_idx:]["pnl_pct"].values
        )
        self.best_threshold = best_threshold

        # Baseline results
        baseline_pnl = df_trades.iloc[split_idx:]["pnl_pct"].sum()
        baseline_win_rate = y_test.mean() * 100

        logger.info(f"\n" + "=" * 50)
        logger.info("RESULTS")
        logger.info("=" * 50)
        logger.info(f"Baseline (all trades):")
        logger.info(f"  Trades: {len(y_test)}")
        logger.info(f"  Win Rate: {baseline_win_rate:.1f}%")
        logger.info(f"  Total PnL: {baseline_pnl:.2f}%")
        logger.info(f"\nWith XGB Filter (threshold={best_threshold}):")
        logger.info(f"  Trades: {best_result['n_trades']} ({best_result['filter_pct']:.1f}% filtered)")
        logger.info(f"  Win Rate: {best_result['win_rate']:.1f}%")
        logger.info(f"  Total PnL: {best_result['total_pnl']:.2f}%")

        # Feature importance
        logger.info(f"\nTop 10 Features (out of {len(self.feature_cols)}):")
        importance = self.model.feature_importances_
        feat_imp = sorted(zip(self.feature_cols, importance), key=lambda x: x[1], reverse=True)
        for feat, imp in feat_imp[:10]:
            logger.info(f"  {feat}: {imp:.4f}")

        # Save model
        model_path = self.output_dir / "xgb_model.json"
        self.model.save_model(str(model_path))
        logger.info(f"\nModel saved: {model_path}")

        # Save results
        train_start = df_trades["timestamp"].iloc[0].strftime("%Y-%m-%d")
        train_end = df_trades["timestamp"].iloc[split_idx-1].strftime("%Y-%m-%d")
        test_start = df_trades["timestamp"].iloc[split_idx].strftime("%Y-%m-%d")
        test_end = df_trades["timestamp"].iloc[-1].strftime("%Y-%m-%d")

        results = {
            "train_period": f"{train_start} ~ {train_end}",
            "test_period": f"{test_start} ~ {test_end}",
            "train_trades": int(len(X_train)),
            "test_trades": int(len(X_test)),
            "n_features": len(self.feature_cols),
            "baseline": {
                "n_trades": int(len(y_test)),
                "win_rate": float(baseline_win_rate),
                "total_pnl": float(baseline_pnl),
                "avg_pnl": float(baseline_pnl / len(y_test)),
            },
            "best_threshold": float(best_threshold),
            "best_result": best_result,
            "feature_cols": self.feature_cols,
            "feature_importance": {k: float(v) for k, v in feat_imp[:20]},  # Top 20
        }

        results_path = self.output_dir / "xgb_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved: {results_path}")

        return model_path

    def _find_optimal_threshold(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        pnl: np.ndarray,
    ) -> Tuple[float, Dict]:
        """Find optimal probability threshold.

        Args:
            y_true: True labels (win/loss)
            y_prob: Predicted probabilities
            pnl: Actual PnL for each trade

        Returns:
            Tuple of (best_threshold, best_result_dict)
        """
        best_threshold = 0.5
        best_pnl = float("-inf")
        best_result = {}

        for threshold in np.arange(0.35, 0.70, 0.05):
            mask = y_prob >= threshold
            if mask.sum() < 10:
                continue

            filtered_pnl = pnl[mask]
            filtered_wins = y_true[mask]

            total_pnl = filtered_pnl.sum()
            win_rate = filtered_wins.mean() * 100
            n_trades = mask.sum()
            filter_pct = (1 - n_trades / len(y_true)) * 100

            if total_pnl > best_pnl:
                best_pnl = total_pnl
                best_threshold = threshold
                best_result = {
                    "threshold": float(threshold),
                    "n_trades": int(n_trades),
                    "win_rate": float(win_rate),
                    "total_pnl": float(total_pnl),
                    "avg_pnl": float(total_pnl / n_trades),
                    "filter_pct": float(filter_pct),
                }

        return best_threshold, best_result

    def train_from_csv(
        self,
        csv_path: Optional[Path] = None,
        test_ratio: float = 0.3,
    ) -> Path:
        """Train XGB gate model from backtest CSV data.

        This uses real Multi-HMM system trades with 14 features matching xgb_gate.py.

        Args:
            csv_path: Path to CSV file with trades and features.
                      If None, uses default path: outputs/xgb_gate/backtest_trades.csv
            test_ratio: Ratio of data for testing

        Returns:
            Path to saved model
        """
        logger.info("=" * 70)
        logger.info("XGB GATE TRAINER (from CSV)")
        logger.info("=" * 70)

        # Default CSV path
        if csv_path is None:
            csv_path = self.output_dir / "backtest_trades.csv"

        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(
                f"CSV file not found: {csv_path}\n"
                "Run backtest first: python scripts/backtest_binance.py --months 6 --use-checkpoint"
            )

        logger.info(f"Loading trades from: {csv_path}")
        df_trades = pd.read_csv(csv_path)
        logger.info(f"  Loaded {len(df_trades)} trades")

        if len(df_trades) < 100:
            raise ValueError(f"Not enough trades for training: {len(df_trades)}")

        # XGB feature columns (14 features, matching xgb_gate.py)
        self.feature_cols = [
            'ret', 'ret_2', 'ret_5', 'ema_diff', 'price_to_ema20', 'price_to_ema50',
            'volatility', 'range_pct', 'volume_ratio', 'rsi', 'ema_slope',
            'side_num', 'regime_trend_up', 'regime_trend_down'
        ]

        # Check available columns
        available_cols = [c for c in self.feature_cols if c in df_trades.columns]
        missing_cols = [c for c in self.feature_cols if c not in df_trades.columns]

        if missing_cols:
            logger.warning(f"Missing columns: {missing_cols}")
            # Fill missing columns with defaults
            for col in missing_cols:
                df_trades[col] = 0

        logger.info(f"  Using {len(self.feature_cols)} features")

        # Prepare features and labels
        X = df_trades[self.feature_cols].values
        y = df_trades["win"].values
        pnl = df_trades["pnl_pct"].values if "pnl_pct" in df_trades.columns else df_trades["pnl"].values

        # Handle NaN/inf
        X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)

        # Split train/test by time (not random)
        split_idx = int(len(X) * (1 - test_ratio))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        pnl_test = pnl[split_idx:]

        logger.info(f"\nTraining XGBoost...")
        logger.info(f"  Train: {len(X_train)} trades")
        logger.info(f"  Test: {len(X_test)} trades")

        # Train XGBoost (simpler model for 14 features)
        self.model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=2,
            reg_alpha=0.05,
            reg_lambda=0.5,
            random_state=42,
            use_label_encoder=False,
            eval_metric="logloss",
        )

        self.model.fit(X_train, y_train)

        # Predict probabilities
        y_prob_test = self.model.predict_proba(X_test)[:, 1]

        # Find optimal threshold
        logger.info(f"\nFinding optimal threshold...")
        best_threshold, best_result = self._find_optimal_threshold(
            y_test, y_prob_test, pnl_test
        )
        self.best_threshold = best_threshold

        # Baseline results
        baseline_pnl = pnl_test.sum()
        baseline_win_rate = y_test.mean() * 100

        logger.info(f"\n" + "=" * 50)
        logger.info("RESULTS")
        logger.info("=" * 50)
        logger.info(f"Baseline (all trades):")
        logger.info(f"  Trades: {len(y_test)}")
        logger.info(f"  Win Rate: {baseline_win_rate:.1f}%")
        logger.info(f"  Total PnL: {baseline_pnl:.4f}")
        logger.info(f"\nWith XGB Filter (threshold={best_threshold}):")
        logger.info(f"  Trades: {best_result['n_trades']} ({best_result['filter_pct']:.1f}% filtered)")
        logger.info(f"  Win Rate: {best_result['win_rate']:.1f}%")
        logger.info(f"  Total PnL: {best_result['total_pnl']:.4f}")

        # Feature importance
        logger.info(f"\nTop Features (out of {len(self.feature_cols)}):")
        importance = self.model.feature_importances_
        feat_imp = sorted(zip(self.feature_cols, importance), key=lambda x: x[1], reverse=True)
        for feat, imp in feat_imp[:10]:
            logger.info(f"  {feat}: {imp:.4f}")

        # Save model
        model_path = self.output_dir / "xgb_model.json"
        self.model.save_model(str(model_path))
        logger.info(f"\nModel saved: {model_path}")

        # Get timestamps for period info
        if "entry_ts" in df_trades.columns:
            train_start = str(df_trades["entry_ts"].iloc[0])[:10]
            train_end = str(df_trades["entry_ts"].iloc[split_idx-1])[:10]
            test_start = str(df_trades["entry_ts"].iloc[split_idx])[:10]
            test_end = str(df_trades["entry_ts"].iloc[-1])[:10]
        else:
            train_start = train_end = test_start = test_end = "unknown"

        # Save results
        results = {
            "train_period": f"{train_start} ~ {train_end}",
            "test_period": f"{test_start} ~ {test_end}",
            "train_trades": int(len(X_train)),
            "test_trades": int(len(X_test)),
            "n_features": len(self.feature_cols),
            "data_source": "backtest_csv",
            "baseline": {
                "n_trades": int(len(y_test)),
                "win_rate": float(baseline_win_rate),
                "total_pnl": float(baseline_pnl),
                "avg_pnl": float(baseline_pnl / len(y_test)) if len(y_test) > 0 else 0,
            },
            "best_threshold": float(best_threshold),
            "best_result": best_result,
            "feature_cols": self.feature_cols,
            "feature_importance": {k: float(v) for k, v in feat_imp},
        }

        results_path = self.output_dir / "xgb_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved: {results_path}")

        return model_path


def main():
    """Run XGB training as standalone script."""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser(description="Train XGB Gate Model")
    parser.add_argument("--months", type=int, default=4, help="Months of data (for generate mode)")
    parser.add_argument("--leverage", type=float, default=5.0, help="Leverage (for generate mode)")
    parser.add_argument("--symbol", default="XRPUSDT", help="Symbol")
    parser.add_argument("--from-csv", action="store_true",
                        help="Train from backtest CSV (recommended)")
    parser.add_argument("--csv-path", type=str, default=None,
                        help="Path to CSV file (default: outputs/xgb_gate/backtest_trades.csv)")
    args = parser.parse_args()

    trainer = XGBTrainer()

    if args.from_csv:
        # Train from backtest CSV (recommended)
        csv_path = Path(args.csv_path) if args.csv_path else None
        model_path = trainer.train_from_csv(csv_path=csv_path)
    else:
        # Train with generated trades (not recommended)
        model_path = trainer.train(
            months=args.months,
            leverage=args.leverage,
            symbol=args.symbol,
        )

    print(f"\nModel saved to: {model_path}")


if __name__ == "__main__":
    main()
