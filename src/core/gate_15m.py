"""
15m Gate for the XRP Core Trading System.
Controls FSM operation based on zone proximity.
"""
from typing import Optional, Tuple

from .types import Zone, ZoneSet, ZoneState, ZoneStateType, ZoneKind
from .config import ZoneConfig


class ZoneGate:
    """
    15m Zone Gate.

    Controls whether the 3m FSM can operate based on
    price position relative to zones.

    Rules:
    - OUTSIDE: 3m FSM must not progress (force IDLE, no anchor detection)
    - NEAR: FSM can ARM but cannot enter trade (no ENTRY)
    - CORE: FSM can fully operate
    """

    def __init__(self, config: ZoneConfig):
        self.config = config

    def check(self, price: float, zones: ZoneSet,
              atr_15m: float = 0.0) -> ZoneState:
        """
        Determine zone state for given price.

        Args:
            price: Current price
            zones: Current zone set
            atr_15m: Current 15m ATR (for pad calculation)

        Returns:
            ZoneState with CORE/NEAR/OUTSIDE and matched zone
        """
        if not zones.all_zones:
            return ZoneState(
                state=ZoneStateType.OUTSIDE,
                matched_zone=None,
                distance_to_zone=None,
                explanation="OUTSIDE: no zones defined"
            )

        pad = self.config.pad_multiplier * atr_15m if atr_15m > 0 else 0

        # Single zone mode: only consider the nearest zone
        if self.config.single_zone_mode:
            return self._check_single_zone_mode(price, zones, atr_15m, pad)

        # Multi-zone mode (original behavior)
        return self._check_multi_zone_mode(price, zones, pad)

    def _find_nearest_zone(self, price: float, zones: ZoneSet) -> Tuple[Optional[Zone], float]:
        """Find the nearest zone to the current price."""
        nearest_zone: Optional[Zone] = None
        nearest_distance: float = float('inf')

        for zone in zones.all_zones:
            if zone.low <= price <= zone.high:
                # Inside zone - distance is 0
                return zone, 0.0
            elif price < zone.low:
                distance = zone.low - price
            else:
                distance = price - zone.high

            if distance < nearest_distance:
                nearest_distance = distance
                nearest_zone = zone

        return nearest_zone, nearest_distance

    def _check_single_zone_mode(self, price: float, zones: ZoneSet,
                                 atr_15m: float, pad: float) -> ZoneState:
        """Check zone state using only the nearest zone."""
        nearest_zone, distance = self._find_nearest_zone(price, zones)

        if nearest_zone is None:
            return ZoneState(
                state=ZoneStateType.OUTSIDE,
                matched_zone=None,
                distance_to_zone=None,
                explanation="OUTSIDE: no zones defined"
            )

        # Check if inside zone (CORE)
        if nearest_zone.low <= price <= nearest_zone.high:
            return ZoneState(
                state=ZoneStateType.CORE,
                matched_zone=nearest_zone,
                distance_to_zone=0.0,
                explanation=f"CORE: inside zone {nearest_zone.id} "
                           f"({nearest_zone.kind.value} @ {nearest_zone.center:.4f})"
            )

        # Check if near zone (within pad)
        effective_pad = pad if pad > 0 else nearest_zone.radius * self.config.pad_multiplier

        if distance <= effective_pad:
            return ZoneState(
                state=ZoneStateType.NEAR,
                matched_zone=nearest_zone,
                distance_to_zone=distance,
                explanation=f"NEAR: price near zone {nearest_zone.id} "
                           f"({nearest_zone.kind.value} @ {nearest_zone.center:.4f}), "
                           f"distance={distance:.4f}"
            )

        # OUTSIDE
        return ZoneState(
            state=ZoneStateType.OUTSIDE,
            matched_zone=None,
            distance_to_zone=distance,
            explanation=f"OUTSIDE: nearest zone {nearest_zone.id} @ {distance:.4f}"
        )

    def _check_multi_zone_mode(self, price: float, zones: ZoneSet,
                                pad: float) -> ZoneState:
        """Check zone state using all zones (original behavior)."""
        # Check for CORE first (inside zone range)
        for zone in zones.all_zones:
            if zone.low <= price <= zone.high:
                return ZoneState(
                    state=ZoneStateType.CORE,
                    matched_zone=zone,
                    distance_to_zone=0.0,
                    explanation=f"CORE: inside zone {zone.id} "
                               f"({zone.kind.value} @ {zone.center:.4f})"
                )

        # Check for NEAR (inside zone + pad)
        nearest_zone: Optional[Zone] = None
        nearest_distance: float = float('inf')

        for zone in zones.all_zones:
            # Use zone radius for pad if ATR not available
            effective_pad = pad if pad > 0 else zone.radius * self.config.pad_multiplier

            low_with_pad = zone.low - effective_pad
            high_with_pad = zone.high + effective_pad

            if low_with_pad <= price <= high_with_pad:
                # Price is near this zone
                if price < zone.low:
                    distance = zone.low - price
                else:
                    distance = price - zone.high

                if distance < nearest_distance:
                    nearest_distance = distance
                    nearest_zone = zone

        if nearest_zone is not None:
            return ZoneState(
                state=ZoneStateType.NEAR,
                matched_zone=nearest_zone,
                distance_to_zone=nearest_distance,
                explanation=f"NEAR: price near zone {nearest_zone.id} "
                           f"({nearest_zone.kind.value} @ {nearest_zone.center:.4f}), "
                           f"distance={nearest_distance:.4f}"
            )

        # OUTSIDE - not near any zone
        # Find closest zone for reference
        for zone in zones.all_zones:
            if price < zone.low:
                distance = zone.low - price
            elif price > zone.high:
                distance = price - zone.high
            else:
                distance = 0

            if distance < nearest_distance:
                nearest_distance = distance
                nearest_zone = zone

        explanation = "OUTSIDE: price not within any zone+pad"
        if nearest_zone:
            explanation += f" (nearest: {nearest_zone.id} @ {nearest_distance:.4f})"

        return ZoneState(
            state=ZoneStateType.OUTSIDE,
            matched_zone=None,
            distance_to_zone=nearest_distance if nearest_zone else None,
            explanation=explanation
        )

    def can_detect_anchor(self, zone_state: ZoneState) -> bool:
        """
        Check if anchor detection is allowed.

        Allowed in CORE or NEAR states.
        """
        return zone_state.state in (ZoneStateType.CORE, ZoneStateType.NEAR)

    def can_arm(self, zone_state: ZoneState) -> bool:
        """
        Check if FSM can ARM (prepare for entry).

        Allowed in CORE or NEAR states.
        """
        return zone_state.state in (ZoneStateType.CORE, ZoneStateType.NEAR)

    def can_enter_trade(self, zone_state: ZoneState) -> bool:
        """
        Check if trade entry is allowed.

        Only allowed in CORE state.
        """
        return zone_state.state == ZoneStateType.CORE

    def should_force_idle(self, zone_state: ZoneState) -> bool:
        """
        Check if FSM should be forced to IDLE.

        True when OUTSIDE any zone.
        """
        return zone_state.state == ZoneStateType.OUTSIDE

    def get_gate_explanation(self, zone_state: ZoneState,
                              action: str) -> str:
        """
        Generate explanation for gate decision.

        Args:
            zone_state: Current zone state
            action: Action being attempted (e.g., "anchor", "entry")

        Returns:
            Explanation string
        """
        state = zone_state.state

        if state == ZoneStateType.OUTSIDE:
            return f"Gate BLOCKED {action}: {zone_state.explanation}"

        if state == ZoneStateType.NEAR:
            if action == "entry":
                return f"Gate BLOCKED entry (NEAR only): {zone_state.explanation}"
            return f"Gate ALLOWED {action} (NEAR): {zone_state.explanation}"

        # CORE
        return f"Gate ALLOWED {action} (CORE): {zone_state.explanation}"

    def get_zone_context(self, zone_state: ZoneState) -> str:
        """
        Get context string for logging/explanation.

        Returns:
            Context description
        """
        if zone_state.matched_zone is None:
            return "No zone context"

        zone = zone_state.matched_zone
        state = zone_state.state.value

        return (f"{state} zone {zone.id}: {zone.kind.value} "
                f"[{zone.low:.4f} - {zone.high:.4f}], "
                f"strength={zone.strength:.2f}")


def zone_state(price: float, zoneset: ZoneSet,
               config: ZoneConfig, atr_15m: float = 0.0) -> ZoneState:
    """
    Convenience function to get zone state.

    Args:
        price: Current price
        zoneset: Zone set to check against
        config: Zone configuration
        atr_15m: Current 15m ATR

    Returns:
        ZoneState
    """
    gate = ZoneGate(config)
    return gate.check(price, zoneset, atr_15m)
