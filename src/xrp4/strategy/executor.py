"""Trade executor with fees and slippage."""

from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass
class Position:
    """Open position.

    Attributes:
        entry_bar: Bar index at entry
        entry_time: Entry timestamp
        side: Trade side ("long" or "short")
        entry_price: Entry price (after fees+slippage)
        size: Position size in base currency
        stop_loss: Stop loss price
        take_profit: Take profit price
        signal_type: Signal type that generated this position
        box_height_atr: Box height in ATR at entry (Step 0.1)
    """
    entry_bar: int
    entry_time: pd.Timestamp
    side: str
    entry_price: float
    size: float
    stop_loss: float
    take_profit: float
    signal_type: str

    # MFE/MAE tracking
    mfe: float = 0.0  # Maximum Favorable Excursion
    mae: float = 0.0  # Maximum Adverse Excursion

    # Box filter tracking (Step 0.1)
    box_height_atr: float = 0.0


@dataclass
class Trade:
    """Completed trade record.

    Attributes:
        entry_bar: Entry bar index
        exit_bar: Exit bar index
        entry_time: Entry timestamp
        exit_time: Exit timestamp
        side: Trade side
        signal_type: Signal type
        entry_price: Entry price (after fees+slippage)
        exit_price: Exit price (after fees+slippage)
        size: Position size
        pnl: Profit/Loss in quote currency
        pnl_pct: Profit/Loss as percentage of entry value
        exit_reason: Reason for exit ("tp", "sl", "eod")
        stop_loss: Stop loss price
        take_profit: Take profit price
        mfe: Maximum Favorable Excursion
        mae: Maximum Adverse Excursion
        mfe_pct: MFE as percentage
        mae_pct: MAE as percentage
        box_height_atr: Box height in ATR at entry (Step 0.1)
    """
    entry_bar: int
    exit_bar: int
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    side: str
    signal_type: str
    entry_price: float
    exit_price: float
    size: float
    pnl: float
    pnl_pct: float
    exit_reason: str
    stop_loss: float
    take_profit: float
    mfe: float = 0.0
    mae: float = 0.0
    mfe_pct: float = 0.0
    mae_pct: float = 0.0
    box_height_atr: float = 0.0  # Step 0.1

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "entry_bar": self.entry_bar,
            "exit_bar": self.exit_bar,
            "entry_time": self.entry_time,
            "exit_time": self.exit_time,
            "side": self.side,
            "signal_type": self.signal_type,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "size": self.size,
            "pnl": self.pnl,
            "pnl_pct": self.pnl_pct,
            "exit_reason": self.exit_reason,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "mfe": self.mfe,
            "mae": self.mae,
            "mfe_pct": self.mfe_pct,
            "mae_pct": self.mae_pct,
            "box_height_atr": self.box_height_atr,
        }


class Executor:
    """Execute trades with fees and slippage."""

    def __init__(
        self,
        fee_rate: float = 0.0004,      # 4 bps
        slippage_rate: float = 0.0002,  # 2 bps
    ):
        """Initialize executor.

        Args:
            fee_rate: Fee rate per trade (e.g., 0.0004 = 0.04%)
            slippage_rate: Slippage rate per trade (e.g., 0.0002 = 0.02%)
        """
        self.fee_rate = fee_rate
        self.slippage_rate = slippage_rate

    def execute_entry(
        self,
        bar_idx: int,
        bar_time: pd.Timestamp,
        side: str,
        entry_price: float,
        size: float,
        stop_loss: float,
        take_profit: float,
        signal_type: str,
    ) -> Position:
        """Execute entry with fees and slippage.

        Args:
            bar_idx: Current bar index
            bar_time: Current bar timestamp
            side: Trade side ("long" or "short")
            entry_price: Entry price (before fees/slippage)
            size: Position size
            stop_loss: Stop loss price
            take_profit: Take profit price
            signal_type: Signal type

        Returns:
            Position object
        """
        # Apply slippage (adverse for entry)
        if side == "long":
            slipped_price = entry_price * (1 + self.slippage_rate)
        else:  # short
            slipped_price = entry_price * (1 - self.slippage_rate)

        # Fees are applied on position value, not price
        # Entry price already includes slippage; fees reduce effective capital

        return Position(
            entry_bar=bar_idx,
            entry_time=bar_time,
            side=side,
            entry_price=slipped_price,
            size=size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            signal_type=signal_type,
        )

    def check_exit(
        self,
        position: Position,
        current_bar: pd.Series,
    ) -> tuple[bool, str, float]:
        """Check if position should be exited.

        Args:
            position: Current position
            current_bar: Current bar data

        Returns:
            Tuple of (should_exit, exit_reason, exit_price)
        """
        high = float(current_bar["high"])
        low = float(current_bar["low"])
        close = float(current_bar["close"])

        if position.side == "long":
            # Check stop loss
            if low <= position.stop_loss:
                return True, "sl", position.stop_loss

            # Check take profit
            if high >= position.take_profit:
                return True, "tp", position.take_profit

        else:  # short
            # Check stop loss
            if high >= position.stop_loss:
                return True, "sl", position.stop_loss

            # Check take profit
            if low <= position.take_profit:
                return True, "tp", position.take_profit

        return False, "", 0.0

    def execute_exit(
        self,
        position: Position,
        bar_idx: int,
        bar_time: pd.Timestamp,
        exit_price: float,
        exit_reason: str,
    ) -> Trade:
        """Execute exit with fees and slippage.

        Args:
            position: Position to close
            bar_idx: Current bar index
            bar_time: Current bar timestamp
            exit_price: Exit price (before fees/slippage)
            exit_reason: Reason for exit

        Returns:
            Trade object
        """
        # Apply slippage (adverse for exit)
        if position.side == "long":
            slipped_exit = exit_price * (1 - self.slippage_rate)
        else:  # short
            slipped_exit = exit_price * (1 + self.slippage_rate)

        # Calculate PnL
        if position.side == "long":
            gross_pnl = (slipped_exit - position.entry_price) * position.size
        else:  # short
            gross_pnl = (position.entry_price - slipped_exit) * position.size

        # Subtract fees (on both entry and exit)
        entry_value = position.entry_price * position.size
        exit_value = slipped_exit * position.size
        total_fees = (entry_value + exit_value) * self.fee_rate

        net_pnl = gross_pnl - total_fees

        # PnL percentage
        pnl_pct = net_pnl / entry_value if entry_value > 0 else 0.0

        # MFE/MAE percentages
        mfe_pct = position.mfe / position.entry_price if position.entry_price > 0 else 0.0
        mae_pct = position.mae / position.entry_price if position.entry_price > 0 else 0.0

        return Trade(
            entry_bar=position.entry_bar,
            exit_bar=bar_idx,
            entry_time=position.entry_time,
            exit_time=bar_time,
            side=position.side,
            signal_type=position.signal_type,
            entry_price=position.entry_price,
            exit_price=slipped_exit,
            size=position.size,
            pnl=net_pnl,
            pnl_pct=pnl_pct,
            exit_reason=exit_reason,
            stop_loss=position.stop_loss,
            take_profit=position.take_profit,
            mfe=position.mfe,
            mae=position.mae,
            mfe_pct=mfe_pct,
            mae_pct=mae_pct,
            box_height_atr=position.box_height_atr,
        )

    def update_mfe_mae(
        self,
        position: Position,
        current_bar: pd.Series,
    ) -> None:
        """Update MFE and MAE for position.

        Args:
            position: Position to update
            current_bar: Current bar data
        """
        high = float(current_bar["high"])
        low = float(current_bar["low"])

        if position.side == "long":
            # MFE: highest price reached - entry
            favorable = high - position.entry_price
            if favorable > position.mfe:
                position.mfe = favorable

            # MAE: entry - lowest price reached
            adverse = position.entry_price - low
            if adverse > position.mae:
                position.mae = adverse

        else:  # short
            # MFE: entry - lowest price reached
            favorable = position.entry_price - low
            if favorable > position.mfe:
                position.mfe = favorable

            # MAE: highest price reached - entry
            adverse = high - position.entry_price
            if adverse > position.mae:
                position.mae = adverse
