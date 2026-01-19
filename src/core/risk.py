"""
Risk management module for the XRP Core Trading System.
Handles position sizing and daily loss limits.
"""
from typing import Tuple, List
from dataclasses import dataclass

from .config import RiskConfig


@dataclass
class RiskState:
    """Current risk state."""
    daily_pnl_pct: float = 0.0
    daily_trades: int = 0
    daily_wins: int = 0
    daily_losses: int = 0
    size_multiplier: float = 1.0
    is_blocked: bool = False
    block_reason: str = ""


class RiskManager:
    """
    Risk management for position sizing and loss limits.

    Features:
    - Position sizing based on risk per trade
    - Daily max loss enforcement
    - Progressive size reduction as losses mount
    """

    def __init__(self, config: RiskConfig):
        self.config = config
        self.state = RiskState()

    def reset_daily(self) -> None:
        """Reset daily tracking (call at start of trading day)."""
        self.state = RiskState()

    def compute_position_size(self, equity: float, entry: float,
                               stop: float) -> Tuple[float, str]:
        """
        Compute position size based on risk parameters.

        Args:
            equity: Current account equity
            entry: Planned entry price
            stop: Stop loss price

        Returns:
            Tuple of (position_size, explanation)
        """
        if self.state.is_blocked:
            return 0.0, f"Blocked: {self.state.block_reason}"

        if entry <= 0 or stop <= 0:
            return 0.0, "Invalid entry or stop price"

        # Calculate stop distance
        stop_distance = abs(entry - stop)
        if stop_distance < 1e-10:
            return 0.0, "Stop distance too small"

        # Risk amount
        risk_pct = self.config.risk_per_trade_pct / 100.0
        risk_amount = equity * risk_pct

        # Apply size multiplier based on drawdown
        risk_amount *= self.state.size_multiplier

        # Position size
        size = risk_amount / stop_distance

        explanation = (
            f"Size={size:.4f}, risk={risk_amount:.2f} "
            f"({risk_pct*100:.2f}% * {self.state.size_multiplier:.1f}x), "
            f"stop_dist={stop_distance:.4f}"
        )

        return size, explanation

    def update_pnl(self, pnl_pct: float) -> None:
        """
        Update daily PnL tracking.

        Args:
            pnl_pct: Trade PnL as percentage of equity
        """
        self.state.daily_pnl_pct += pnl_pct
        self.state.daily_trades += 1

        if pnl_pct > 0:
            self.state.daily_wins += 1
        else:
            self.state.daily_losses += 1

        # Check if we should block trading
        self._check_daily_limit()

        # Update size multiplier
        self._update_size_multiplier()

    def _check_daily_limit(self) -> None:
        """Check if daily loss limit is exceeded."""
        if self.state.daily_pnl_pct <= -self.config.daily_max_loss_pct:
            self.state.is_blocked = True
            self.state.block_reason = (
                f"Daily loss limit exceeded: {self.state.daily_pnl_pct:.2f}% "
                f"(limit: -{self.config.daily_max_loss_pct:.2f}%)"
            )

    def _update_size_multiplier(self) -> None:
        """Update position size multiplier based on drawdown."""
        steps = self.config.drawdown_size_reduce_steps
        max_loss = self.config.daily_max_loss_pct

        if not steps or max_loss <= 0:
            return

        # Calculate how far into daily loss limit we are
        loss_ratio = abs(min(0, self.state.daily_pnl_pct)) / max_loss

        if loss_ratio >= 0.75:
            # More than 75% into limit - use smallest multiplier
            self.state.size_multiplier = steps[-1] if steps else 0.3
        elif loss_ratio >= 0.5:
            # 50-75% into limit - use middle multiplier
            self.state.size_multiplier = steps[0] if steps else 0.5
        else:
            self.state.size_multiplier = 1.0

    def can_trade(self) -> Tuple[bool, str]:
        """
        Check if trading is allowed.

        Returns:
            Tuple of (allowed, reason)
        """
        if self.state.is_blocked:
            return False, self.state.block_reason

        return True, "Trading allowed"

    def get_summary(self) -> str:
        """Get daily risk summary."""
        return (
            f"Daily PnL: {self.state.daily_pnl_pct:.2f}%, "
            f"Trades: {self.state.daily_trades}, "
            f"W/L: {self.state.daily_wins}/{self.state.daily_losses}, "
            f"Size mult: {self.state.size_multiplier:.1f}x, "
            f"Blocked: {self.state.is_blocked}"
        )


def compute_position_size(equity: float, entry: float, stop: float,
                          config: RiskConfig,
                          daily_pnl_pct: float = 0.0) -> Tuple[float, str]:
    """
    Convenience function for position sizing.

    Args:
        equity: Current account equity
        entry: Planned entry price
        stop: Stop loss price
        config: Risk configuration
        daily_pnl_pct: Current daily PnL percentage

    Returns:
        Tuple of (position_size, explanation)
    """
    manager = RiskManager(config)
    manager.state.daily_pnl_pct = daily_pnl_pct
    manager._update_size_multiplier()

    return manager.compute_position_size(equity, entry, stop)


def calculate_pnl_pct(entry: float, exit: float, is_long: bool) -> float:
    """
    Calculate PnL percentage.

    Args:
        entry: Entry price
        exit: Exit price
        is_long: True for long, False for short

    Returns:
        PnL as percentage
    """
    if entry <= 0:
        return 0.0

    if is_long:
        return (exit - entry) / entry * 100
    else:
        return (entry - exit) / entry * 100
