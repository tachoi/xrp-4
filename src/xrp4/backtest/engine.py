"""Backtest engine for Step0 baseline strategy."""

from typing import List, Optional, Tuple

import pandas as pd
import structlog

from xrp4.config import Step0Config
from xrp4.data.candles import compute_atr, resample_candles
from xrp4.filters.box_strength import BoxFilterConfig, BoxStrengthFilter
from xrp4.strategy.executor import Executor, Position, Trade
from xrp4.strategy.risk import RiskManager
from xrp4.strategy.signals import SignalGenerator, SignalType
from xrp4.zones.zone_builder import ZoneBuilder
from xrp4.zones.zone_models import Zone

logger = structlog.get_logger(__name__)


class BacktestEngine:
    """Backtest engine for zone-based bounce/breakout strategy."""

    def __init__(self, config: Step0Config):
        """Initialize backtest engine.

        Args:
            config: Step0 configuration
        """
        self.config = config

        # Initialize components
        self.zone_builder = ZoneBuilder(
            pivot_left=config.pivot_left,
            pivot_right=config.pivot_right,
            zone_width=0.001,  # Will be updated dynamically
            max_zones=config.max_zones,
        )

        self.signal_generator = SignalGenerator(
            touch_band_atr_k=config.touch_band_atr_k,
            break_band_atr_k=config.break_band_atr_k,
            sl_band_atr_k=config.sl_band_atr_k,
            rr_bounce=config.rr_bounce,
            rr_breakout=config.rr_breakout,
            confirm_mode=config.confirm_mode,
            signal_mode=config.signal_mode,
        )

        self.risk_manager = RiskManager(
            risk_per_trade=config.risk_per_trade,
            max_position=config.max_position,
        )

        self.executor = Executor(
            fee_rate=config.fee_rate,
            slippage_rate=config.slippage_rate,
        )

        # Box filter (Step 0.1)
        self.box_filter = BoxStrengthFilter(
            config=config.box_filter,
            fee_bps=config.fee_bps,
            slippage_bps=config.slippage_bps,
        )

        # State
        self.equity = config.initial_capital
        self.position: Optional[Position] = None
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = [config.initial_capital]
        self.cooldown_until: int = 0

        # Zones (updated periodically)
        self.support_zones: List[Zone] = []
        self.resistance_zones: List[Zone] = []

    def run(self, df: pd.DataFrame) -> Tuple[List[Trade], pd.Series]:
        """Run backtest on OHLCV data.

        Args:
            df: DataFrame with 3m OHLCV data

        Returns:
            Tuple of (trades list, equity curve series)
        """
        logger.info(
            "Starting backtest",
            bars=len(df),
            start=df["timestamp"].iloc[0],
            end=df["timestamp"].iloc[-1],
        )

        # Compute ATR on 3m data
        atr_3m = compute_atr(df, period=14)

        # Try to resample to higher timeframe for ATR (if configured)
        if self.config.atr_tf != "3m":
            try:
                df_higher = resample_candles(df, self.config.atr_tf)
                atr_higher = compute_atr(df_higher, period=14)
                # We'll use atr_3m for now since we need bar-level ATR
                # In production, you'd map higher TF ATR back to 3m bars
                atr = atr_3m
            except:
                logger.warning(f"Failed to resample to {self.config.atr_tf}, using 3m ATR")
                atr = atr_3m
        else:
            atr = atr_3m

        # Precompute box filter data (Step 0.1)
        if self.config.box_filter.enabled:
            self.box_filter.precompute(df, atr)
            logger.info("Box filter precomputed", min_height_atr=self.config.box_filter.min_height_atr)

        # Main backtest loop
        zone_rebuild_counter = 0

        for idx in range(len(df)):
            current_bar = df.iloc[idx]
            bar_idx = idx

            # Skip if we don't have enough data yet
            if idx < max(self.config.pivot_left + self.config.pivot_right, 14):
                self.equity_curve.append(self.equity)
                continue

            # Get current ATR
            current_atr = atr.iloc[idx] if not pd.isna(atr.iloc[idx]) else atr.iloc[idx - 1]

            # Rebuild zones periodically
            if zone_rebuild_counter == 0:
                self._rebuild_zones(df.iloc[:idx + 1], current_atr)
                zone_rebuild_counter = self.config.zone_rebuild_freq

            zone_rebuild_counter -= 1

            # Update position (check exits and MFE/MAE)
            if self.position is not None:
                self._update_position(bar_idx, current_bar)

            # Generate signals (only if no position and not in cooldown)
            if self.position is None and bar_idx >= self.cooldown_until:
                if idx > 0:
                    prev_bar = df.iloc[idx - 1]
                    signal = self.signal_generator.generate_signals(
                        bar_idx=bar_idx,
                        current_bar=current_bar,
                        prev_bar=prev_bar,
                        support_zones=self.support_zones,
                        resistance_zones=self.resistance_zones,
                        atr=current_atr,
                    )

                    if signal.signal_type != SignalType.NO_SIGNAL:
                        # Apply box filter (Step 0.1)
                        should_open = True
                        if self.config.box_filter.enabled:
                            allowed, reason, debug = self.box_filter.allow_trade(
                                df=df,
                                idx=idx,
                                side=signal.side,
                                zone=signal.zone,
                            )
                            if not allowed:
                                logger.debug(
                                    "Trade blocked by box filter",
                                    reason=reason,
                                    box_height_atr=debug.get("box_height_atr", 0),
                                )
                                should_open = False

                        if should_open:
                            self._open_position(bar_idx, current_bar, signal)

            # Record equity
            self.equity_curve.append(self.equity)

        # Close any remaining position at end
        if self.position is not None:
            self._force_close_position(len(df) - 1, df.iloc[-1])

        logger.info(
            "Backtest completed",
            total_trades=len(self.trades),
            final_equity=self.equity,
        )

        # Trim equity curve to match df length (remove initial capital entry)
        equity_curve_trimmed = self.equity_curve[1:] if len(self.equity_curve) > len(df) else self.equity_curve
        equity_series = pd.Series(equity_curve_trimmed, index=df.index)
        return self.trades, equity_series

    def _rebuild_zones(self, df_history: pd.DataFrame, atr: float) -> None:
        """Rebuild zones from historical data.

        Args:
            df_history: Historical OHLCV data up to current bar
            atr: Current ATR value
        """
        self.support_zones, self.resistance_zones = self.zone_builder.build_zones(
            df=df_history,
            lookback_bars=self.config.zone_lookback_bars,
            atr=atr,
            atr_k=self.config.zone_width_atr_k,
        )

        logger.debug(
            "Zones rebuilt",
            support_zones=len(self.support_zones),
            resistance_zones=len(self.resistance_zones),
        )

    def _open_position(self, bar_idx: int, current_bar: pd.Series, signal) -> None:
        """Open a new position from signal.

        Args:
            bar_idx: Current bar index
            current_bar: Current bar data
            signal: Trading signal
        """
        # Check if we can open position
        if not self.risk_manager.can_open_position(0):  # 0 current positions
            return

        # Calculate position size
        size = self.risk_manager.calculate_position_size(
            equity=self.equity,
            entry_price=signal.entry_price,
            stop_loss=signal.stop_loss,
            side=signal.side,
        )

        if size <= 0:
            return

        # Execute entry
        self.position = self.executor.execute_entry(
            bar_idx=bar_idx,
            bar_time=current_bar["timestamp"],
            side=signal.side,
            entry_price=signal.entry_price,
            size=size,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            signal_type=signal.signal_type.value,
        )

        # Track box_height_atr for this position (Step 0.1)
        if self.config.box_filter.enabled:
            box_height_atr = self.box_filter.get_current_box_height_atr()
            self.position.box_height_atr = box_height_atr

        logger.debug(
            "Position opened",
            bar_idx=bar_idx,
            side=signal.side,
            entry=signal.entry_price,
            size=size,
            signal=signal.signal_type.value,
        )

    def _update_position(self, bar_idx: int, current_bar: pd.Series) -> None:
        """Update position (check exits, update MFE/MAE).

        Args:
            bar_idx: Current bar index
            current_bar: Current bar data
        """
        if self.position is None:
            return

        # Update MFE/MAE
        self.executor.update_mfe_mae(self.position, current_bar)

        # Check for exit
        should_exit, exit_reason, exit_price = self.executor.check_exit(
            self.position, current_bar
        )

        if should_exit:
            self._close_position(bar_idx, current_bar, exit_price, exit_reason)

    def _close_position(
        self,
        bar_idx: int,
        current_bar: pd.Series,
        exit_price: float,
        exit_reason: str,
    ) -> None:
        """Close current position.

        Args:
            bar_idx: Current bar index
            current_bar: Current bar data
            exit_price: Exit price
            exit_reason: Reason for exit
        """
        if self.position is None:
            return

        # Execute exit
        trade = self.executor.execute_exit(
            position=self.position,
            bar_idx=bar_idx,
            bar_time=current_bar["timestamp"],
            exit_price=exit_price,
            exit_reason=exit_reason,
        )

        # Update equity
        self.equity += trade.pnl

        # Record trade
        self.trades.append(trade)

        # Track for box filter diagnostics (Step 0.1)
        if self.config.box_filter.enabled:
            self.box_filter.diagnostics.add_bin_trade(
                trade.box_height_atr,
                trade.to_dict(),
            )

        # Set cooldown
        self.cooldown_until = bar_idx + self.config.cooldown_bars

        logger.debug(
            "Position closed",
            bar_idx=bar_idx,
            pnl=trade.pnl,
            reason=exit_reason,
            equity=self.equity,
        )

        # Clear position
        self.position = None

    def _force_close_position(self, bar_idx: int, current_bar: pd.Series) -> None:
        """Force close position at end of data.

        Args:
            bar_idx: Current bar index
            current_bar: Current bar data
        """
        if self.position is None:
            return

        self._close_position(
            bar_idx=bar_idx,
            current_bar=current_bar,
            exit_price=float(current_bar["close"]),
            exit_reason="eod",
        )
