"""
Data types for the XRP Core Trading System.
"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any


@dataclass
class Candle:
    """OHLCV candle representation."""
    ts: int  # epoch milliseconds
    open: float
    high: float
    low: float
    close: float
    volume: float

    @property
    def body(self) -> float:
        """Absolute body size."""
        return abs(self.close - self.open)

    @property
    def range(self) -> float:
        """High - Low, minimum 1e-10 to avoid division by zero."""
        return max(self.high - self.low, 1e-10)

    @property
    def lower_wick(self) -> float:
        """Lower shadow length."""
        return min(self.open, self.close) - self.low

    @property
    def upper_wick(self) -> float:
        """Upper shadow length."""
        return self.high - max(self.open, self.close)

    @property
    def lower_wick_ratio(self) -> float:
        """Lower wick as fraction of range."""
        return self.lower_wick / self.range

    @property
    def upper_wick_ratio(self) -> float:
        """Upper wick as fraction of range."""
        return self.upper_wick / self.range

    @property
    def body_ratio(self) -> float:
        """Body as fraction of range."""
        return self.body / self.range

    @property
    def is_bullish(self) -> bool:
        """True if close >= open."""
        return self.close >= self.open

    @property
    def is_bearish(self) -> bool:
        """True if close < open."""
        return self.close < self.open

    def to_datetime(self) -> datetime:
        """Convert timestamp to datetime."""
        return datetime.fromtimestamp(self.ts / 1000.0)


class ZoneKind(Enum):
    """Type of zone."""
    SUPPORT = "support"
    RESISTANCE = "resistance"


@dataclass
class Zone:
    """A support or resistance zone with range."""
    id: str
    center: float  # Z
    radius: float  # R
    strength: float  # S
    kind: ZoneKind
    last_updated_ts: int
    reaction_count: int = 0

    @property
    def low(self) -> float:
        """Lower bound of zone."""
        return self.center - self.radius

    @property
    def high(self) -> float:
        """Upper bound of zone."""
        return self.center + self.radius


class ZoneStateType(Enum):
    """Zone proximity state."""
    CORE = "CORE"      # Inside zone range
    NEAR = "NEAR"      # Inside zone range + pad
    OUTSIDE = "OUTSIDE"  # Not in any zone


@dataclass
class ZoneState:
    """Current zone state with matched zone info."""
    state: ZoneStateType
    matched_zone: Optional[Zone] = None
    distance_to_zone: Optional[float] = None
    explanation: str = ""


class FSMState(Enum):
    """3m FSM states."""
    IDLE = "IDLE"
    ANCHOR_FOUND = "ANCHOR_FOUND"
    PULLBACK_WAIT = "PULLBACK_WAIT"
    ENTRY_READY = "ENTRY_READY"
    WAIT_ENTRY_PULLBACK = "WAIT_ENTRY_PULLBACK"  # Wait for pullback before entry
    IN_TRADE = "IN_TRADE"
    EXIT_COOLDOWN = "EXIT_COOLDOWN"


class Signal(Enum):
    """Trading signals."""
    NONE = "NONE"
    ARM = "ARM"
    ANCHOR = "ANCHOR"
    ENTRY_LONG = "ENTRY_LONG"
    ENTRY_SHORT = "ENTRY_SHORT"
    EXIT = "EXIT"
    COOL = "COOL"


class Side(Enum):
    """Trade side."""
    LONG = "LONG"
    SHORT = "SHORT"


class EntryType(Enum):
    """Order entry type."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"


@dataclass
class AnchorInfo:
    """Information about detected anchor candle."""
    open: float
    close: float
    mid: float
    direction: int  # +1 for bullish, -1 for bearish
    ts: int
    index: int
    volume_mult: float  # actual volume / SMA volume
    body_atr_mult: float  # body / ATR


@dataclass
class PhenomenaResult:
    """Result of phenomenon-based condition checks."""
    ok: bool
    low_fail: bool = False
    wick_ratio_ok: bool = False
    body_shrink: bool = False
    range_shrink: bool = False
    count: int = 0
    details: str = ""


@dataclass
class Decision:
    """FSM decision output with explanation."""
    signal: Signal
    prev_state: FSMState
    new_state: FSMState
    explanation: str
    debug: Dict[str, Any] = field(default_factory=dict)

    # Optional entry/exit details
    side: Optional[Side] = None
    entry_price: Optional[float] = None
    stop_price: Optional[float] = None
    anchor_info: Optional[AnchorInfo] = None
    phenomena: Optional[PhenomenaResult] = None


@dataclass
class OrderIntent:
    """Order intent for external execution."""
    side: Side
    entry_type: EntryType
    entry_price: float
    stop_price: float
    size: float
    explanation: str
    take_profit: Optional[float] = None
    debug: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IndicatorValues:
    """Pre-computed indicator values for a candle."""
    ema20: float
    rsi14: float
    atr14: float
    volume_sma: float
    rsi_prev: Optional[float] = None
    ema_distance: Optional[float] = None  # abs(close - ema) / atr

    @classmethod
    def empty(cls) -> 'IndicatorValues':
        """Return empty indicator values."""
        return cls(ema20=0.0, rsi14=50.0, atr14=0.0, volume_sma=0.0)


@dataclass
class ZoneSet:
    """Collection of zones with lookup methods."""
    support_zones: List[Zone] = field(default_factory=list)
    resistance_zones: List[Zone] = field(default_factory=list)

    @property
    def all_zones(self) -> List[Zone]:
        """All zones combined."""
        return self.support_zones + self.resistance_zones

    def find_zone_state(self, price: float, pad_ratio: float = 0.4,
                        atr: float = 0.0) -> ZoneState:
        """
        Find zone state for given price.

        Args:
            price: Current price
            pad_ratio: Pad multiplier for NEAR detection
            atr: ATR for calculating pad (if 0, uses zone radius)

        Returns:
            ZoneState indicating CORE/NEAR/OUTSIDE
        """
        # Check all zones for CORE first
        for zone in self.all_zones:
            if zone.low <= price <= zone.high:
                return ZoneState(
                    state=ZoneStateType.CORE,
                    matched_zone=zone,
                    distance_to_zone=0.0,
                    explanation=f"CORE: inside zone {zone.id} ({zone.kind.value})"
                )

        # Check for NEAR
        for zone in self.all_zones:
            pad = atr * pad_ratio if atr > 0 else zone.radius * pad_ratio
            low_with_pad = zone.low - pad
            high_with_pad = zone.high + pad

            if low_with_pad <= price <= high_with_pad:
                dist = min(abs(price - zone.low), abs(price - zone.high))
                return ZoneState(
                    state=ZoneStateType.NEAR,
                    matched_zone=zone,
                    distance_to_zone=dist,
                    explanation=f"NEAR: price near zone {zone.id} ({zone.kind.value})"
                )

        # OUTSIDE
        return ZoneState(
            state=ZoneStateType.OUTSIDE,
            matched_zone=None,
            distance_to_zone=None,
            explanation="OUTSIDE: price not within any zone+pad"
        )
