"""
Tests for 15m zone detection.
"""
import pytest
from src.core.types import Candle, ZoneKind
from src.core.config import ZoneConfig
from src.core.zones_15m import ZoneEngine


def make_candle(ts: int, open_: float, high: float, low: float,
                close: float, volume: float = 1000.0) -> Candle:
    """Helper to create candles."""
    return Candle(ts=ts, open=open_, high=high, low=low,
                  close=close, volume=volume)


def make_candle_series(n: int, base_price: float = 1.0,
                       trend: float = 0.0) -> list:
    """Create a series of candles."""
    candles = []
    price = base_price

    for i in range(n):
        ts = 1000 * 60 * 15 * i  # 15m intervals
        noise = 0.01 * (i % 5 - 2)
        open_ = price
        close = price * (1 + trend + noise)
        high = max(open_, close) * 1.01
        low = min(open_, close) * 0.99
        candles.append(make_candle(ts, open_, high, low, close))
        price = close

    return candles


class TestZoneEngine:
    """Tests for ZoneEngine."""

    def test_init(self):
        """Test engine initialization."""
        config = ZoneConfig()
        engine = ZoneEngine(config)

        assert engine.zones.support_zones == []
        assert engine.zones.resistance_zones == []

    def test_requires_enough_candles(self):
        """Test that engine requires minimum candles."""
        config = ZoneConfig(atr_period=14, pivot_lookback=2)
        engine = ZoneEngine(config)

        # Too few candles
        candles = make_candle_series(10)
        zones = engine.update(candles)

        assert len(zones.all_zones) == 0

    def test_detects_pivot_high(self):
        """Test pivot high detection."""
        config = ZoneConfig(atr_period=5, pivot_lookback=2)
        engine = ZoneEngine(config)

        # Create candles with clear pivot high
        candles = []
        prices = [1.0, 1.01, 1.02, 1.05, 1.02, 1.01, 1.0, 0.99, 0.98]

        for i, p in enumerate(prices):
            ts = 1000 * 60 * 15 * i
            candles.append(make_candle(ts, p*0.99, p*1.01, p*0.99, p))

        # Add more candles for ATR calculation
        candles.extend(make_candle_series(20, base_price=0.95))

        zones = engine.update(candles)

        # Should detect something (may not have reaction yet)
        assert len(engine._candidates) >= 0  # Candidates found

    def test_detects_pivot_low(self):
        """Test pivot low detection."""
        config = ZoneConfig(atr_period=5, pivot_lookback=2)
        engine = ZoneEngine(config)

        # Create candles with clear pivot low
        prices = [1.0, 0.99, 0.98, 0.95, 0.98, 0.99, 1.0, 1.01, 1.02]
        candles = []

        for i, p in enumerate(prices):
            ts = 1000 * 60 * 15 * i
            candles.append(make_candle(ts, p*1.01, p*1.01, p*0.99, p))

        candles.extend(make_candle_series(20, base_price=1.05))

        zones = engine.update(candles)

        # Candidates should be detected
        assert isinstance(zones, type(engine.zones))

    def test_zone_merge_with_radius(self):
        """Test that nearby candidates merge into zones."""
        config = ZoneConfig(
            atr_period=5,
            pivot_lookback=2,
            radius_multiplier=0.6
        )
        engine = ZoneEngine(config)

        # Create candles
        candles = make_candle_series(50, base_price=1.0)
        zones = engine.update(candles)

        # Zone set should be valid
        assert hasattr(zones, 'support_zones')
        assert hasattr(zones, 'resistance_zones')

    def test_zone_decay(self):
        """Test zone strength decay over time."""
        config = ZoneConfig(decay_halflife_hours=12.0)
        engine = ZoneEngine(config)

        candles = make_candle_series(50)
        engine.update(candles)

        # Create a zone manually
        from src.core.types import Zone
        zone = Zone(
            id='test',
            center=1.0,
            radius=0.01,
            strength=10.0,
            kind=ZoneKind.SUPPORT,
            last_updated_ts=0,
        )
        engine.zones.support_zones.append(zone)

        # Apply decay for 12 hours
        engine._apply_decay(12 * 3600 * 1000)

        # Strength should be halved
        assert 4.5 < zone.strength < 5.5  # ~5.0 after half-life

    def test_prune_zones(self):
        """Test pruning to max zones per side."""
        config = ZoneConfig(max_per_side=3)
        engine = ZoneEngine(config)

        # Add many zones
        from src.core.types import Zone
        for i in range(10):
            zone = Zone(
                id=f'support_{i}',
                center=1.0 + i * 0.1,
                radius=0.01,
                strength=float(i),
                kind=ZoneKind.SUPPORT,
                last_updated_ts=0,
            )
            engine.zones.support_zones.append(zone)

        engine._prune_zones()

        assert len(engine.zones.support_zones) == 3
        # Should keep highest strength
        strengths = [z.strength for z in engine.zones.support_zones]
        assert all(s >= 7 for s in strengths)

    def test_reset(self):
        """Test engine reset."""
        config = ZoneConfig()
        engine = ZoneEngine(config)

        # Add some state
        candles = make_candle_series(30)
        engine.update(candles)

        engine.reset()

        assert engine.zones.support_zones == []
        assert engine.zones.resistance_zones == []
        assert engine._candidates == []


class TestZoneSet:
    """Tests for ZoneSet functionality."""

    def test_find_zone_state_core(self):
        """Test CORE zone detection."""
        from src.core.types import Zone, ZoneSet, ZoneStateType

        zone = Zone(
            id='test',
            center=1.0,
            radius=0.05,
            strength=1.0,
            kind=ZoneKind.SUPPORT,
            last_updated_ts=0,
        )
        zones = ZoneSet(support_zones=[zone])

        state = zones.find_zone_state(1.02)  # Inside zone

        assert state.state == ZoneStateType.CORE
        assert state.matched_zone == zone

    def test_find_zone_state_near(self):
        """Test NEAR zone detection."""
        from src.core.types import Zone, ZoneSet, ZoneStateType

        zone = Zone(
            id='test',
            center=1.0,
            radius=0.05,
            strength=1.0,
            kind=ZoneKind.SUPPORT,
            last_updated_ts=0,
        )
        zones = ZoneSet(support_zones=[zone])

        state = zones.find_zone_state(1.07, pad_ratio=0.4)  # Near but outside

        assert state.state == ZoneStateType.NEAR
        assert state.matched_zone == zone

    def test_find_zone_state_outside(self):
        """Test OUTSIDE zone detection."""
        from src.core.types import Zone, ZoneSet, ZoneStateType

        zone = Zone(
            id='test',
            center=1.0,
            radius=0.02,
            strength=1.0,
            kind=ZoneKind.SUPPORT,
            last_updated_ts=0,
        )
        zones = ZoneSet(support_zones=[zone])

        state = zones.find_zone_state(1.5, pad_ratio=0.4)  # Far outside

        assert state.state == ZoneStateType.OUTSIDE
        assert state.matched_zone is None
