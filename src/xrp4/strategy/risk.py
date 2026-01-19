"""Risk management for fixed-risk position sizing."""


class RiskManager:
    """Fixed-risk position sizing manager."""

    def __init__(
        self,
        risk_per_trade: float = 0.0025,  # 0.25% of equity
        max_position: int = 1,
    ):
        """Initialize risk manager.

        Args:
            risk_per_trade: Fraction of equity to risk per trade (e.g., 0.0025 = 0.25%)
            max_position: Maximum number of concurrent positions
        """
        self.risk_per_trade = risk_per_trade
        self.max_position = max_position

    def calculate_position_size(
        self,
        equity: float,
        entry_price: float,
        stop_loss: float,
        side: str,
    ) -> float:
        """Calculate position size based on fixed risk.

        Args:
            equity: Current account equity
            entry_price: Entry price
            stop_loss: Stop loss price
            side: Trade side ("long" or "short")

        Returns:
            Position size in base currency units
        """
        # Calculate risk amount in quote currency
        risk_amount = equity * self.risk_per_trade

        # Calculate risk per unit
        if side == "long":
            risk_per_unit = abs(entry_price - stop_loss)
        elif side == "short":
            risk_per_unit = abs(stop_loss - entry_price)
        else:
            return 0.0

        if risk_per_unit <= 0:
            return 0.0

        # Position size = risk_amount / risk_per_unit
        position_size = risk_amount / risk_per_unit

        return position_size

    def can_open_position(self, current_positions: int) -> bool:
        """Check if can open a new position.

        Args:
            current_positions: Number of current open positions

        Returns:
            True if can open new position
        """
        return current_positions < self.max_position
