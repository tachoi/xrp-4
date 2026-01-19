"""Zone builder using pivot clustering."""

from typing import List, Tuple

import numpy as np
import pandas as pd

from xrp4.zones.zone_models import Zone, ZoneType


class ZoneBuilder:
    """Build support/resistance zones from pivot points."""

    def __init__(
        self,
        pivot_left: int = 3,
        pivot_right: int = 3,
        zone_width: float = 0.001,  # Will be set dynamically based on ATR
        max_zones: int = 12,
    ):
        """Initialize zone builder.

        Args:
            pivot_left: Number of bars to left for pivot detection
            pivot_right: Number of bars to right for pivot detection
            zone_width: Zone width (typically ATR-based)
            max_zones: Maximum number of zones to keep
        """
        self.pivot_left = pivot_left
        self.pivot_right = pivot_right
        self.zone_width = zone_width
        self.max_zones = max_zones

    def build_zones(
        self,
        df: pd.DataFrame,
        lookback_bars: int = 500,
        atr: float = None,
        atr_k: float = 1.0,
    ) -> Tuple[List[Zone], List[Zone]]:
        """Build support and resistance zones from recent price action.

        Args:
            df: DataFrame with OHLCV data (must have 'high', 'low' columns)
            lookback_bars: Number of recent bars to analyze
            atr: ATR value for zone width calculation (if None, uses self.zone_width)
            atr_k: ATR multiplier for zone width

        Returns:
            Tuple of (support_zones, resistance_zones)
        """
        # Use recent data only
        if len(df) > lookback_bars:
            df = df.iloc[-lookback_bars:].copy()
        else:
            df = df.copy()

        # Update zone width based on ATR if provided
        if atr is not None:
            self.zone_width = atr * atr_k

        # Find pivot highs and lows
        pivot_highs = self._find_pivot_highs(df)
        pivot_lows = self._find_pivot_lows(df)

        # Cluster pivots into zones
        resistance_zones = self._cluster_pivots(pivot_highs, ZoneType.RESISTANCE)
        support_zones = self._cluster_pivots(pivot_lows, ZoneType.SUPPORT)

        # Count touches for each zone
        resistance_zones = self._count_touches(resistance_zones, df, ZoneType.RESISTANCE)
        support_zones = self._count_touches(support_zones, df, ZoneType.SUPPORT)

        # Keep top zones by touch count
        resistance_zones = sorted(resistance_zones, key=lambda z: z.touch_count, reverse=True)
        support_zones = sorted(support_zones, key=lambda z: z.touch_count, reverse=True)

        resistance_zones = resistance_zones[:self.max_zones]
        support_zones = support_zones[:self.max_zones]

        return support_zones, resistance_zones

    def _find_pivot_highs(self, df: pd.DataFrame) -> List[float]:
        """Find pivot high points.

        Args:
            df: DataFrame with 'high' column

        Returns:
            List of pivot high prices
        """
        pivots = []
        highs = df["high"].astype(float).values

        for i in range(self.pivot_left, len(highs) - self.pivot_right):
            is_pivot = True

            # Check left side
            for j in range(1, self.pivot_left + 1):
                if highs[i] <= highs[i - j]:
                    is_pivot = False
                    break

            if not is_pivot:
                continue

            # Check right side
            for j in range(1, self.pivot_right + 1):
                if highs[i] <= highs[i + j]:
                    is_pivot = False
                    break

            if is_pivot:
                pivots.append(highs[i])

        return pivots

    def _find_pivot_lows(self, df: pd.DataFrame) -> List[float]:
        """Find pivot low points.

        Args:
            df: DataFrame with 'low' column

        Returns:
            List of pivot low prices
        """
        pivots = []
        lows = df["low"].astype(float).values

        for i in range(self.pivot_left, len(lows) - self.pivot_right):
            is_pivot = True

            # Check left side
            for j in range(1, self.pivot_left + 1):
                if lows[i] >= lows[i - j]:
                    is_pivot = False
                    break

            if not is_pivot:
                continue

            # Check right side
            for j in range(1, self.pivot_right + 1):
                if lows[i] >= lows[i + j]:
                    is_pivot = False
                    break

            if is_pivot:
                pivots.append(lows[i])

        return pivots

    def _cluster_pivots(
        self,
        pivots: List[float],
        zone_type: ZoneType,
    ) -> List[Zone]:
        """Cluster nearby pivots into zones.

        Args:
            pivots: List of pivot prices
            zone_type: SUPPORT or RESISTANCE

        Returns:
            List of zones
        """
        if not pivots:
            return []

        # Sort pivots
        pivots = sorted(pivots)

        zones = []
        current_cluster = [pivots[0]]

        for i in range(1, len(pivots)):
            # Check if this pivot is within zone_width of the current cluster
            cluster_center = np.mean(current_cluster)

            if abs(pivots[i] - cluster_center) <= self.zone_width:
                # Add to current cluster
                current_cluster.append(pivots[i])
            else:
                # Create zone from current cluster
                zone = self._create_zone_from_cluster(current_cluster, zone_type)
                zones.append(zone)

                # Start new cluster
                current_cluster = [pivots[i]]

        # Don't forget the last cluster
        if current_cluster:
            zone = self._create_zone_from_cluster(current_cluster, zone_type)
            zones.append(zone)

        return zones

    def _create_zone_from_cluster(
        self,
        cluster: List[float],
        zone_type: ZoneType,
    ) -> Zone:
        """Create a zone from a cluster of pivot prices.

        Args:
            cluster: List of pivot prices in cluster
            zone_type: SUPPORT or RESISTANCE

        Returns:
            Zone object
        """
        cluster_min = min(cluster)
        cluster_max = max(cluster)
        cluster_mid = np.mean(cluster)

        # Ensure zone has minimum width (zone_width)
        if cluster_max - cluster_min < self.zone_width:
            half_width = self.zone_width / 2.0
            low = cluster_mid - half_width
            high = cluster_mid + half_width
        else:
            low = cluster_min
            high = cluster_max

        return Zone(
            low=low,
            high=high,
            zone_type=zone_type,
            pivot_prices=cluster.copy(),
        )

    def _count_touches(
        self,
        zones: List[Zone],
        df: pd.DataFrame,
        zone_type: ZoneType,
    ) -> List[Zone]:
        """Count how many times price touched each zone.

        Args:
            zones: List of zones
            df: DataFrame with OHLCV data
            zone_type: SUPPORT or RESISTANCE

        Returns:
            Updated zones with touch counts
        """
        if not zones:
            return zones

        # Vectorized approach for performance
        if zone_type == ZoneType.SUPPORT:
            prices = df["low"].astype(float).values
        else:
            prices = df["high"].astype(float).values

        for zone in zones:
            # Count touches using vectorized comparison
            touch_mask = (prices >= zone.low) & (prices <= zone.high)
            touch_count = int(touch_mask.sum())

            zone.touch_count = touch_count
            zone.strength = float(touch_count)

        return zones
