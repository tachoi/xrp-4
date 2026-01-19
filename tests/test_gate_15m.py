"""
Tests for 15m gate (CORE/NEAR/OUTSIDE classification).
"""
import pytest
from src.core.types import Zone, ZoneSet, ZoneState, ZoneStateType, ZoneKind
from src.core.config import ZoneConfig
from src.core.gate_15m import ZoneGate, zone_state


def make_zone(center: float, radius: float, kind: ZoneKind,
              zone_id: str = "test") -> Zone:
    """Helper to create zones."""
    return Zone(
        id=zone_id,
        center=center,
        radius=radius,
        strength=1.0,
        kind=kind,
        last_updated_ts=0,
    )


class TestZoneGate:
    """Tests for ZoneGate."""

    def test_core_inside_zone(self):
        """Test CORE when price is inside zone."""
        config = ZoneConfig(pad_multiplier=0.4)
        gate = ZoneGate(config)

        zone = make_zone(center=1.0, radius=0.05, kind=ZoneKind.SUPPORT)
        zones = ZoneSet(support_zones=[zone])

        state = gate.check(1.02, zones, atr_15m=0.1)

        assert state.state == ZoneStateType.CORE
        assert state.matched_zone == zone
        assert "CORE" in state.explanation

    def test_near_with_pad(self):
        """Test NEAR when price is in pad area."""
        config = ZoneConfig(pad_multiplier=0.4)
        gate = ZoneGate(config)

        zone = make_zone(center=1.0, radius=0.05, kind=ZoneKind.RESISTANCE)
        zones = ZoneSet(resistance_zones=[zone])

        # Price just outside zone but within pad (1.05 + 0.04 = 1.09)
        state = gate.check(1.07, zones, atr_15m=0.1)

        assert state.state == ZoneStateType.NEAR
        assert state.matched_zone == zone
        assert "NEAR" in state.explanation

    def test_outside_far_from_zone(self):
        """Test OUTSIDE when price is far from zone."""
        config = ZoneConfig(pad_multiplier=0.4)
        gate = ZoneGate(config)

        zone = make_zone(center=1.0, radius=0.05, kind=ZoneKind.SUPPORT)
        zones = ZoneSet(support_zones=[zone])

        state = gate.check(1.5, zones, atr_15m=0.1)

        assert state.state == ZoneStateType.OUTSIDE
        assert state.matched_zone is None
        assert "OUTSIDE" in state.explanation

    def test_outside_when_no_zones(self):
        """Test OUTSIDE when no zones defined."""
        config = ZoneConfig()
        gate = ZoneGate(config)

        zones = ZoneSet()
        state = gate.check(1.0, zones, atr_15m=0.1)

        assert state.state == ZoneStateType.OUTSIDE
        assert "no zones" in state.explanation.lower()

    def test_can_detect_anchor(self):
        """Test anchor detection permission."""
        config = ZoneConfig()
        gate = ZoneGate(config)

        core_state = ZoneState(state=ZoneStateType.CORE)
        near_state = ZoneState(state=ZoneStateType.NEAR)
        outside_state = ZoneState(state=ZoneStateType.OUTSIDE)

        assert gate.can_detect_anchor(core_state) is True
        assert gate.can_detect_anchor(near_state) is True
        assert gate.can_detect_anchor(outside_state) is False

    def test_can_enter_trade(self):
        """Test trade entry permission."""
        config = ZoneConfig()
        gate = ZoneGate(config)

        core_state = ZoneState(state=ZoneStateType.CORE)
        near_state = ZoneState(state=ZoneStateType.NEAR)
        outside_state = ZoneState(state=ZoneStateType.OUTSIDE)

        assert gate.can_enter_trade(core_state) is True
        assert gate.can_enter_trade(near_state) is False
        assert gate.can_enter_trade(outside_state) is False

    def test_should_force_idle(self):
        """Test force idle check."""
        config = ZoneConfig()
        gate = ZoneGate(config)

        core_state = ZoneState(state=ZoneStateType.CORE)
        near_state = ZoneState(state=ZoneStateType.NEAR)
        outside_state = ZoneState(state=ZoneStateType.OUTSIDE)

        assert gate.should_force_idle(core_state) is False
        assert gate.should_force_idle(near_state) is False
        assert gate.should_force_idle(outside_state) is True

    def test_multiple_zones_nearest(self):
        """Test that nearest zone is matched."""
        config = ZoneConfig(pad_multiplier=0.4)
        gate = ZoneGate(config)

        zone1 = make_zone(center=1.0, radius=0.05, kind=ZoneKind.SUPPORT, zone_id="z1")
        zone2 = make_zone(center=1.2, radius=0.05, kind=ZoneKind.RESISTANCE, zone_id="z2")
        zones = ZoneSet(support_zones=[zone1], resistance_zones=[zone2])

        # Price inside zone2 range [1.15, 1.25] -> CORE
        state = gate.check(1.18, zones, atr_15m=0.1)

        assert state.state == ZoneStateType.CORE
        assert state.matched_zone.id == "z2"

    def test_gate_explanation(self):
        """Test explanation generation."""
        config = ZoneConfig()
        gate = ZoneGate(config)

        zone = make_zone(center=1.0, radius=0.05, kind=ZoneKind.SUPPORT)

        core_state = ZoneState(
            state=ZoneStateType.CORE,
            matched_zone=zone,
            explanation="CORE: inside zone"
        )

        explanation = gate.get_gate_explanation(core_state, "anchor")
        assert "ALLOWED" in explanation
        assert "anchor" in explanation


class TestZoneStateFunction:
    """Tests for zone_state convenience function."""

    def test_returns_zone_state(self):
        """Test that function returns ZoneState."""
        config = ZoneConfig()
        zones = ZoneSet()

        result = zone_state(1.0, zones, config, atr_15m=0.1)

        assert isinstance(result, ZoneState)
        assert result.state == ZoneStateType.OUTSIDE
