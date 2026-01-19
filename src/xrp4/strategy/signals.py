"""Signal generation for zone-based bounce/breakout strategy."""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import pandas as pd

from xrp4.zones.zone_models import Zone, ZoneType


class SignalType(Enum):
    """Signal type."""
    BOUNCE_LONG = "bounce_long"
    BOUNCE_SHORT = "bounce_short"
    BREAKOUT_LONG = "breakout_long"
    BREAKOUT_SHORT = "breakout_short"
    NO_SIGNAL = "no_signal"


@dataclass
class Signal:
    """Trading signal.

    Attributes:
        signal_type: Type of signal
        zone: Zone that generated the signal
        entry_price: Expected entry price
        stop_loss: Stop loss price
        take_profit: Take profit price
        bar_index: Bar index where signal was generated
    """
    signal_type: SignalType
    zone: Optional[Zone] = None
    entry_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    bar_index: int = 0

    @property
    def side(self) -> str:
        """Get trade side (long/short)."""
        if self.signal_type in [SignalType.BOUNCE_LONG, SignalType.BREAKOUT_LONG]:
            return "long"
        elif self.signal_type in [SignalType.BOUNCE_SHORT, SignalType.BREAKOUT_SHORT]:
            return "short"
        else:
            return "none"

    @property
    def risk(self) -> float:
        """Calculate risk amount (entry - stop_loss for long)."""
        if self.side == "long":
            return abs(self.entry_price - self.stop_loss)
        elif self.side == "short":
            return abs(self.stop_loss - self.entry_price)
        else:
            return 0.0

    @property
    def reward(self) -> float:
        """Calculate reward amount (take_profit - entry for long)."""
        if self.side == "long":
            return abs(self.take_profit - self.entry_price)
        elif self.side == "short":
            return abs(self.entry_price - self.take_profit)
        else:
            return 0.0


class SignalGenerator:
    """Generate bounce and breakout signals from zones."""

    def __init__(
        self,
        touch_band_atr_k: float = 0.2,
        break_band_atr_k: float = 0.2,
        sl_band_atr_k: float = 0.2,
        rr_bounce: float = 1.2,
        rr_breakout: float = 1.5,
        confirm_mode: str = "next_close",
        signal_mode: str = "all",
    ):
        """Initialize signal generator.

        Args:
            touch_band_atr_k: ATR multiplier for touch band
            break_band_atr_k: ATR multiplier for breakout band
            sl_band_atr_k: ATR multiplier for stop loss band
            rr_bounce: Risk:Reward ratio for bounce trades
            rr_breakout: Risk:Reward ratio for breakout trades
            confirm_mode: Confirmation mode ("next_close" for Step0)
            signal_mode: Signal filter mode ("all", "bounce_only", "breakout_only")
        """
        self.touch_band_atr_k = touch_band_atr_k
        self.break_band_atr_k = break_band_atr_k
        self.sl_band_atr_k = sl_band_atr_k
        self.rr_bounce = rr_bounce
        self.rr_breakout = rr_breakout
        self.confirm_mode = confirm_mode
        self.signal_mode = signal_mode

    def generate_signals(
        self,
        bar_idx: int,
        current_bar: pd.Series,
        prev_bar: pd.Series,
        support_zones: List[Zone],
        resistance_zones: List[Zone],
        atr: float,
    ) -> Signal:
        """Generate trading signal for current bar.

        Args:
            bar_idx: Current bar index
            current_bar: Current candle data
            prev_bar: Previous candle data
            support_zones: List of support zones
            resistance_zones: List of resistance zones
            atr: Current ATR value

        Returns:
            Signal object (may be NO_SIGNAL)
        """
        bounce_signal = Signal(signal_type=SignalType.NO_SIGNAL, bar_index=bar_idx)
        breakout_signal = Signal(signal_type=SignalType.NO_SIGNAL, bar_index=bar_idx)

        # Check bounce signals (unless breakout_only mode)
        if self.signal_mode != "breakout_only":
            bounce_signal = self._check_bounce_signals(
                bar_idx, current_bar, prev_bar, support_zones, resistance_zones, atr
            )

        # Check breakout signals (unless bounce_only mode)
        if self.signal_mode != "bounce_only":
            breakout_signal = self._check_breakout_signals(
                bar_idx, current_bar, prev_bar, support_zones, resistance_zones, atr
            )

        # Conflict resolution: if both bounce and breakout, return NO_SIGNAL
        if (bounce_signal.signal_type != SignalType.NO_SIGNAL and
            breakout_signal.signal_type != SignalType.NO_SIGNAL):
            return Signal(signal_type=SignalType.NO_SIGNAL, bar_index=bar_idx)

        # Return the active signal (or NO_SIGNAL if neither)
        if bounce_signal.signal_type != SignalType.NO_SIGNAL:
            return bounce_signal
        elif breakout_signal.signal_type != SignalType.NO_SIGNAL:
            return breakout_signal
        else:
            return Signal(signal_type=SignalType.NO_SIGNAL, bar_index=bar_idx)

    def _check_bounce_signals(
        self,
        bar_idx: int,
        current_bar: pd.Series,
        prev_bar: pd.Series,
        support_zones: List[Zone],
        resistance_zones: List[Zone],
        atr: float,
    ) -> Signal:
        """Check for bounce signals.

        Args:
            bar_idx: Current bar index
            current_bar: Current candle
            prev_bar: Previous candle
            support_zones: Support zones
            resistance_zones: Resistance zones
            atr: ATR value

        Returns:
            Bounce signal or NO_SIGNAL
        """
        touch_band = atr * self.touch_band_atr_k
        sl_band = atr * self.sl_band_atr_k

        # Convert to float to handle Decimal types from DB
        cur_low = float(current_bar["low"])
        cur_high = float(current_bar["high"])
        cur_close = float(current_bar["close"])

        # Check LONG bounce at support
        for zone in support_zones:
            # Touch: current low touched zone (with touch_band tolerance)
            if (cur_low <= zone.high and
                cur_low >= zone.low - touch_band):

                # Confirm: current close above zone low (simple confirm for Step0)
                if cur_close > zone.low:
                    # Entry at current close (assuming next bar fills at this price)
                    entry = cur_close
                    stop_loss = zone.low - sl_band
                    risk = entry - stop_loss
                    take_profit = entry + (risk * self.rr_bounce)

                    return Signal(
                        signal_type=SignalType.BOUNCE_LONG,
                        zone=zone,
                        entry_price=entry,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        bar_index=bar_idx,
                    )

        # Check SHORT bounce at resistance
        for zone in resistance_zones:
            # Touch: current high touched zone (with touch_band tolerance)
            if (cur_high >= zone.low and
                cur_high <= zone.high + touch_band):

                # Confirm: current close below zone high
                if cur_close < zone.high:
                    entry = cur_close
                    stop_loss = zone.high + sl_band
                    risk = stop_loss - entry
                    take_profit = entry - (risk * self.rr_bounce)

                    return Signal(
                        signal_type=SignalType.BOUNCE_SHORT,
                        zone=zone,
                        entry_price=entry,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        bar_index=bar_idx,
                    )

        return Signal(signal_type=SignalType.NO_SIGNAL, bar_index=bar_idx)

    def _check_breakout_signals(
        self,
        bar_idx: int,
        current_bar: pd.Series,
        prev_bar: pd.Series,
        support_zones: List[Zone],
        resistance_zones: List[Zone],
        atr: float,
    ) -> Signal:
        """Check for breakout signals.

        Args:
            bar_idx: Current bar index
            current_bar: Current candle
            prev_bar: Previous candle
            support_zones: Support zones
            resistance_zones: Resistance zones
            atr: ATR value

        Returns:
            Breakout signal or NO_SIGNAL
        """
        break_band = atr * self.break_band_atr_k
        sl_band = atr * self.sl_band_atr_k

        # Convert to float to handle Decimal types from DB
        cur_close = float(current_bar["close"])

        # Check LONG breakout above resistance
        for zone in resistance_zones:
            breakout_level = zone.high + break_band

            # Breakout: close above breakout level
            if cur_close > breakout_level:
                # Confirm: close stayed above breakout level (simple for Step0)
                entry = cur_close
                stop_loss = zone.high - sl_band
                risk = entry - stop_loss
                take_profit = entry + (risk * self.rr_breakout)

                return Signal(
                    signal_type=SignalType.BREAKOUT_LONG,
                    zone=zone,
                    entry_price=entry,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    bar_index=bar_idx,
                )

        # Check SHORT breakout below support
        for zone in support_zones:
            breakout_level = zone.low - break_band

            # Breakout: close below breakout level
            if cur_close < breakout_level:
                entry = cur_close
                stop_loss = zone.low + sl_band
                risk = stop_loss - entry
                take_profit = entry - (risk * self.rr_breakout)

                return Signal(
                    signal_type=SignalType.BREAKOUT_SHORT,
                    zone=zone,
                    entry_price=entry,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    bar_index=bar_idx,
                )

        return Signal(signal_type=SignalType.NO_SIGNAL, bar_index=bar_idx)
