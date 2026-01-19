"""
15m Zone Engine for the XRP Core Trading System.
Produces support/resistance zones as ranges with strength and decay.
"""
import math
from typing import List, Optional, Tuple
from dataclasses import dataclass

from .types import Candle, Zone, ZoneKind, ZoneSet
from .config import ZoneConfig
from .indicators import atr


@dataclass
class PivotCandidate:
    """A candidate pivot point."""
    price: float
    ts: int
    index: int
    kind: ZoneKind
    reaction_count: int = 0
    reaction_strength: float = 0.0


class ZoneEngine:
    """
    15m Zone Detection Engine (Optimized).

    Detects support/resistance zones from pivot highs/lows,
    validates with reaction checks, and maintains zones with
    strength scoring and time decay.

    Performance optimizations:
    - Incremental pivot detection (only check new candles)
    - Bounded candidate list
    - Cached ATR values
    - Efficient zone lookups
    """

    # Performance tuning constants
    MAX_CANDIDATES = 100  # Maximum candidates to keep
    LOOKBACK_FOR_PIVOTS = 50  # Only check recent candles for new pivots

    def __init__(self, config: ZoneConfig):
        self.config = config
        self.zones = ZoneSet()
        self._zone_counter = 0
        self._candidates: List[PivotCandidate] = []
        self._atr_values: List[float] = []
        self._last_processed_index = 0  # Track last processed candle
        self._current_atr = 0.0  # Cached current ATR

    def reset(self) -> None:
        """Reset engine state."""
        self.zones = ZoneSet()
        self._zone_counter = 0
        self._candidates = []
        self._atr_values = []
        self._last_processed_index = 0
        self._current_atr = 0.0

    def _generate_zone_id(self) -> str:
        """Generate unique zone ID."""
        self._zone_counter += 1
        return f"zone_{self._zone_counter}"

    def update(self, candles: List[Candle]) -> ZoneSet:
        """
        Update zones based on 15m candles (optimized incremental processing).

        Args:
            candles: List of 15m candles (oldest to newest)

        Returns:
            Updated ZoneSet
        """
        n = len(candles)
        min_required = self.config.atr_period + self.config.pivot_lookback

        if n < min_required:
            return self.zones

        # Compute ATR incrementally (only recent values needed)
        self._update_atr(candles)

        # Detect pivot points incrementally (only check new candles)
        self._detect_pivots_incremental(candles)

        # Validate reactions for recent candidates only
        self._validate_reactions_incremental(candles)

        # Merge validated candidates into zones
        self._merge_to_zones(candles)

        # Prune old candidates to keep list bounded
        self._prune_candidates(candles)

        # Apply zone expiration logic
        if candles:
            current_ts = candles[-1].ts
            current_price = candles[-1].close

            # 1. Apply time decay and remove weak zones
            self._apply_decay(current_ts)

            # 2. Remove zones too far from current price
            if self._current_atr > 0:
                self._prune_by_price_distance(current_price, self._current_atr)

            # 3. Remove zones that are too old
            self._prune_by_age(current_ts)

            # 4. Remove zones that have been decisively broken
            self._check_and_remove_broken_zones(candles[-3:])

        # Keep top zones by strength
        self._prune_zones()

        # Update last processed index
        self._last_processed_index = n

        return self.zones

    def _update_atr(self, candles: List[Candle]) -> None:
        """Update ATR with simple incremental calculation."""
        n = len(candles)
        period = self.config.atr_period

        if n < 2:
            return

        # Simple ATR update: only compute TR for the last candle
        # and update running average
        curr = candles[-1]
        prev = candles[-2]

        # True Range for current candle
        tr = max(
            curr.high - curr.low,
            abs(curr.high - prev.close),
            abs(curr.low - prev.close)
        )

        if self._current_atr == 0:
            # Initial ATR: simple average of recent TRs
            if n >= period:
                trs = []
                for i in range(-period, 0):
                    c = candles[i]
                    p = candles[i - 1]
                    t = max(c.high - c.low, abs(c.high - p.close), abs(c.low - p.close))
                    trs.append(t)
                self._current_atr = sum(trs) / period
        else:
            # EMA-style update: ATR = (prev_ATR * (period-1) + TR) / period
            self._current_atr = (self._current_atr * (period - 1) + tr) / period

    def _detect_pivots_incremental(self, candles: List[Candle]) -> None:
        """Detect pivot highs and lows incrementally (only check new candles)."""
        n = len(candles)
        L = self.config.pivot_lookback

        if n < 2 * L + 1:
            return

        # Only check candles that haven't been processed yet
        # But we need L candles after the pivot to confirm it
        start_idx = max(L, self._last_processed_index - L)
        end_idx = n - L

        for i in range(start_idx, end_idx):
            candle = candles[i]

            # Check for pivot high (resistance candidate)
            is_pivot_high = True
            for j in range(i - L, i):
                if candles[j].high >= candle.high:
                    is_pivot_high = False
                    break
            if is_pivot_high:
                for j in range(i + 1, i + L + 1):
                    if candles[j].high > candle.high:
                        is_pivot_high = False
                        break

            if is_pivot_high:
                if not self._has_nearby_candidate(candle.high, ZoneKind.RESISTANCE, i):
                    self._candidates.append(PivotCandidate(
                        price=candle.high,
                        ts=candle.ts,
                        index=i,
                        kind=ZoneKind.RESISTANCE,
                    ))

            # Check for pivot low (support candidate)
            is_pivot_low = True
            for j in range(i - L, i):
                if candles[j].low <= candle.low:
                    is_pivot_low = False
                    break
            if is_pivot_low:
                for j in range(i + 1, i + L + 1):
                    if candles[j].low < candle.low:
                        is_pivot_low = False
                        break

            if is_pivot_low:
                if not self._has_nearby_candidate(candle.low, ZoneKind.SUPPORT, i):
                    self._candidates.append(PivotCandidate(
                        price=candle.low,
                        ts=candle.ts,
                        index=i,
                        kind=ZoneKind.SUPPORT,
                    ))

    def _has_nearby_candidate(self, price: float, kind: ZoneKind,
                               index: int) -> bool:
        """Check if similar candidate exists nearby (optimized)."""
        # Only check recent candidates (performance optimization)
        # Since candidates are added in order, check from the end
        check_count = min(20, len(self._candidates))
        for c in self._candidates[-check_count:]:
            if c.kind == kind and abs(c.index - index) <= 2:
                if abs(c.price - price) < 0.001:  # Near same price
                    return True
        return False

    def _validate_reactions_incremental(self, candles: List[Candle]) -> None:
        """Validate pivot candidates with reaction checks (optimized)."""
        n = len(candles)
        lookahead = self.config.reaction_lookahead
        k_atr = self.config.reaction_k_atr

        # Use cached ATR for efficiency
        atr_val = self._current_atr
        if atr_val == 0:
            return

        reaction_threshold = k_atr * atr_val

        # Only validate candidates that haven't been validated yet
        # and are old enough to have reaction data
        for candidate in self._candidates:
            # Skip if already validated
            if candidate.reaction_count > 0:
                continue

            idx = candidate.index

            # Skip if not enough candles after pivot
            if idx + lookahead > n:
                continue

            # Look ahead for reaction
            end_idx = min(idx + lookahead, n)
            for i in range(idx + 1, end_idx):
                candle = candles[i]

                if candidate.kind == ZoneKind.RESISTANCE:
                    distance = candidate.price - candle.low
                    if distance >= reaction_threshold:
                        candidate.reaction_count += 1
                        candidate.reaction_strength += distance / atr_val
                        break

                else:  # SUPPORT
                    distance = candle.high - candidate.price
                    if distance >= reaction_threshold:
                        candidate.reaction_count += 1
                        candidate.reaction_strength += distance / atr_val
                        break

    def _prune_candidates(self, candles: List[Candle]) -> None:
        """Remove old candidates and keep list bounded."""
        if not candles:
            return

        n = len(candles)
        lookahead = self.config.reaction_lookahead

        # Remove candidates that are too old (already processed or expired)
        # Keep only candidates from recent candles
        min_valid_index = max(0, n - self.LOOKBACK_FOR_PIVOTS)

        # Filter: keep only recent candidates or those with reactions (pending zones)
        self._candidates = [
            c for c in self._candidates
            if c.index >= min_valid_index or (c.reaction_count > 0 and c.index >= n - lookahead * 2)
        ]

        # If still too many, keep only the most recent
        if len(self._candidates) > self.MAX_CANDIDATES:
            self._candidates = sorted(
                self._candidates, key=lambda c: c.index, reverse=True
            )[:self.MAX_CANDIDATES]

    def _merge_to_zones(self, candles: List[Candle]) -> None:
        """Merge validated candidates into zones."""
        if not candles or self._current_atr == 0:
            return

        current_atr = self._current_atr

        radius = self.config.radius_multiplier * current_atr
        current_ts = candles[-1].ts

        # Process candidates with reactions
        valid_candidates = [c for c in self._candidates if c.reaction_count > 0]

        # Group by kind
        support_candidates = [c for c in valid_candidates if c.kind == ZoneKind.SUPPORT]
        resistance_candidates = [c for c in valid_candidates if c.kind == ZoneKind.RESISTANCE]

        # Merge support zones
        self._merge_candidates_to_zones(
            support_candidates, radius, current_ts, ZoneKind.SUPPORT
        )

        # Merge resistance zones
        self._merge_candidates_to_zones(
            resistance_candidates, radius, current_ts, ZoneKind.RESISTANCE
        )

    def _merge_candidates_to_zones(self, candidates: List[PivotCandidate],
                                    radius: float, current_ts: int,
                                    kind: ZoneKind) -> None:
        """Merge candidates into zones of given kind."""
        if not candidates:
            return

        # Sort by price
        candidates = sorted(candidates, key=lambda c: c.price)

        # Group candidates within radius
        groups: List[List[PivotCandidate]] = []
        current_group: List[PivotCandidate] = []

        for c in candidates:
            if not current_group:
                current_group.append(c)
            elif abs(c.price - current_group[-1].price) <= radius:
                current_group.append(c)
            else:
                groups.append(current_group)
                current_group = [c]

        if current_group:
            groups.append(current_group)

        # Create zones from groups
        for group in groups:
            # Weighted center (by reaction strength and recency)
            total_weight = 0.0
            weighted_sum = 0.0
            total_reactions = 0

            for c in group:
                # Weight by recency (more recent = higher weight)
                recency_weight = 1.0
                age_hours = (current_ts - c.ts) / (1000 * 3600)
                if age_hours > 0:
                    recency_weight = math.exp(-age_hours / self.config.decay_halflife_hours)

                weight = (1.0 + c.reaction_strength) * recency_weight
                weighted_sum += c.price * weight
                total_weight += weight
                total_reactions += c.reaction_count

            if total_weight > 0:
                center = weighted_sum / total_weight
            else:
                center = sum(c.price for c in group) / len(group)

            # Check if zone already exists nearby
            existing = self._find_existing_zone(center, radius, kind)

            if existing:
                # Update existing zone
                existing.center = (existing.center + center) / 2
                existing.strength += total_reactions
                existing.reaction_count += total_reactions
                existing.last_updated_ts = current_ts
            else:
                # Create new zone
                zone = Zone(
                    id=self._generate_zone_id(),
                    center=center,
                    radius=radius,
                    strength=float(total_reactions),
                    kind=kind,
                    last_updated_ts=current_ts,
                    reaction_count=total_reactions,
                )

                if kind == ZoneKind.SUPPORT:
                    self.zones.support_zones.append(zone)
                else:
                    self.zones.resistance_zones.append(zone)

    def _find_existing_zone(self, center: float, radius: float,
                            kind: ZoneKind) -> Optional[Zone]:
        """Find existing zone near the given center."""
        zones = (self.zones.support_zones if kind == ZoneKind.SUPPORT
                 else self.zones.resistance_zones)

        for zone in zones:
            if abs(zone.center - center) <= radius:
                return zone

        return None

    def _apply_decay(self, current_ts: int) -> None:
        """Apply time decay to zone strengths and remove weak zones."""
        halflife_ms = self.config.decay_halflife_hours * 3600 * 1000
        min_strength = self.config.min_strength_threshold

        zones_to_remove = []

        for zone in self.zones.all_zones:
            age_ms = current_ts - zone.last_updated_ts
            if age_ms > 0:
                decay_factor = math.pow(0.5, age_ms / halflife_ms)
                zone.strength *= decay_factor

            # Mark for removal if strength below threshold
            if zone.strength < min_strength:
                zones_to_remove.append(zone.id)

        # Remove weak zones
        for zone_id in zones_to_remove:
            self.remove_zone(zone_id)

    def _prune_by_age(self, current_ts: int) -> None:
        """Remove zones that are too old."""
        max_age_ms = self.config.max_age_hours * 3600 * 1000
        zones_to_remove = []

        for zone in self.zones.all_zones:
            age_ms = current_ts - zone.last_updated_ts
            if age_ms > max_age_ms:
                zones_to_remove.append(zone.id)

        for zone_id in zones_to_remove:
            self.remove_zone(zone_id)

    def _prune_by_price_distance(self, current_price: float, current_atr: float) -> None:
        """Remove zones that are too far from current price."""
        max_distance = self.config.price_distance_atr * current_atr
        zones_to_remove = []

        for zone in self.zones.all_zones:
            distance = abs(zone.center - current_price)
            if distance > max_distance:
                zones_to_remove.append(zone.id)

        for zone_id in zones_to_remove:
            self.remove_zone(zone_id)

    def _check_and_remove_broken_zones(self, candles: List[Candle]) -> None:
        """Check for broken zones and remove them."""
        if not self.config.break_removal_enabled or not candles:
            return

        zones_to_remove = []

        for zone in self.zones.all_zones:
            if self.check_break(candles, zone):
                zones_to_remove.append(zone.id)

        for zone_id in zones_to_remove:
            self.remove_zone(zone_id)

    def _prune_zones(self) -> None:
        """Keep only top zones by strength."""
        max_per_side = self.config.max_per_side

        # Sort by strength descending and keep top N
        self.zones.support_zones = sorted(
            self.zones.support_zones,
            key=lambda z: z.strength,
            reverse=True
        )[:max_per_side]

        self.zones.resistance_zones = sorted(
            self.zones.resistance_zones,
            key=lambda z: z.strength,
            reverse=True
        )[:max_per_side]

    def check_break(self, candles: List[Candle], zone: Zone) -> bool:
        """
        Check if zone was decisively broken.

        A break occurs when close is beyond zone boundary by > radius.

        Args:
            candles: Recent candles to check
            zone: Zone to check for break

        Returns:
            True if zone was broken
        """
        if not candles:
            return False

        latest = candles[-1]

        if zone.kind == ZoneKind.SUPPORT:
            # Support broken if close < low - radius
            return latest.close < zone.low - zone.radius
        else:
            # Resistance broken if close > high + radius
            return latest.close > zone.high + zone.radius

    def remove_zone(self, zone_id: str) -> bool:
        """Remove zone by ID."""
        for i, z in enumerate(self.zones.support_zones):
            if z.id == zone_id:
                self.zones.support_zones.pop(i)
                return True

        for i, z in enumerate(self.zones.resistance_zones):
            if z.id == zone_id:
                self.zones.resistance_zones.pop(i)
                return True

        return False

    def weaken_zone(self, zone_id: str, factor: float = 0.5) -> bool:
        """Reduce zone strength on partial break."""
        for z in self.zones.all_zones:
            if z.id == zone_id:
                z.strength *= factor
                return True
        return False
