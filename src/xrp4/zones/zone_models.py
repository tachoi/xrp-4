"""Zone data models."""

from dataclasses import dataclass
from enum import Enum
from typing import List


class ZoneType(Enum):
    """Zone type classification."""
    SUPPORT = "support"
    RESISTANCE = "resistance"


@dataclass
class Zone:
    """Price zone for bounce/breakout trading.

    Attributes:
        low: Lower bound of zone
        high: Upper bound of zone
        zone_type: SUPPORT or RESISTANCE
        strength: Zone strength (based on touch count)
        touch_count: Number of times price touched this zone
        pivot_prices: List of pivot prices that formed this zone
    """
    low: float
    high: float
    zone_type: ZoneType
    strength: float = 0.0
    touch_count: int = 0
    pivot_prices: List[float] = None

    def __post_init__(self):
        if self.pivot_prices is None:
            self.pivot_prices = []

    @property
    def mid(self) -> float:
        """Zone midpoint."""
        return (self.low + self.high) / 2.0

    @property
    def width(self) -> float:
        """Zone width."""
        return self.high - self.low

    def contains(self, price: float) -> bool:
        """Check if price is within zone bounds.

        Args:
            price: Price to check

        Returns:
            True if price is within [low, high]
        """
        return self.low <= price <= self.high

    def distance_to(self, price: float) -> float:
        """Calculate distance from price to zone.

        Args:
            price: Price to calculate distance from

        Returns:
            Positive distance if price is outside zone, 0 if inside
        """
        if price < self.low:
            return self.low - price
        elif price > self.high:
            return price - self.high
        else:
            return 0.0

    def __repr__(self) -> str:
        return (
            f"Zone({self.zone_type.value}, "
            f"[{self.low:.6f}, {self.high:.6f}], "
            f"touches={self.touch_count})"
        )
