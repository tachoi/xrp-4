"""Box Strength filter for market quality assessment.

This filter blocks trades unless the current "box" (range structure) is strong enough:
- Box must be wide enough relative to volatility
- Box must have positive expected tradability after costs

NO ML - this is a structural market-quality filter only.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class BoxFilterConfig:
    """Configuration for Box Strength filter."""

    enabled: bool = True
    lookback_bars: int = 64  # ~3h on 3m
    atr_period: int = 14
    atr_tf: str = "15m"  # fallback to 3m if not available
    min_height_atr: float = 3.0  # Key threshold
    update_every_bars: int = 10  # Recompute box/strength periodically

    # Reaction filter (optional toggle)
    use_reaction_filter: bool = False
    edge_band_atr_k: float = 0.3
    reaction_lookback_touches: int = 20
    reaction_horizon_bars: int = 6
    move_target_atr_k: float = 1.0
    reaction_min_rate: float = 0.48

    # Cost filter (optional toggle)
    use_cost_filter: bool = False
    cost_safety_mult: float = 2.0


@dataclass
class BoxState:
    """Current box state at a given bar."""

    box_high: float = 0.0
    box_low: float = 0.0
    box_height: float = 0.0
    atr: float = 0.0
    box_height_atr: float = 0.0
    expected_move_atr: float = 0.0
    costs_atr: float = 0.0
    reaction_score: float = 0.0
    bar_idx: int = 0


@dataclass
class BoxFilterDiagnostics:
    """Diagnostics for box filter."""

    total_candidates: int = 0
    passed: int = 0
    failed_min_height_atr: int = 0
    failed_reaction: int = 0
    failed_cost: int = 0

    # Store box_height_atr values for passed/failed signals
    passed_box_heights: List[float] = field(default_factory=list)
    failed_box_heights: List[float] = field(default_factory=list)

    # Per-bin performance tracking
    bin_trades: Dict[str, List[dict]] = field(default_factory=dict)

    def get_pass_rate(self) -> float:
        """Get pass rate."""
        if self.total_candidates == 0:
            return 0.0
        return self.passed / self.total_candidates

    def get_top_fail_reason(self) -> str:
        """Get top failure reason."""
        reasons = {
            "FAIL_MIN_HEIGHT_ATR": self.failed_min_height_atr,
            "FAIL_REACTION": self.failed_reaction,
            "FAIL_COST": self.failed_cost,
        }
        if not any(reasons.values()):
            return "NONE"
        return max(reasons, key=reasons.get)

    def get_median_box_height(self, passed: bool) -> float:
        """Get median box_height_atr for passed or failed signals."""
        heights = self.passed_box_heights if passed else self.failed_box_heights
        if not heights:
            return 0.0
        return float(np.median(heights))

    def add_bin_trade(self, box_height_atr: float, trade_dict: dict) -> None:
        """Add trade to appropriate bin for performance analysis."""
        bin_name = self._get_bin_name(box_height_atr)
        if bin_name not in self.bin_trades:
            self.bin_trades[bin_name] = []
        self.bin_trades[bin_name].append(trade_dict)

    def _get_bin_name(self, box_height_atr: float) -> str:
        """Get bin name for box_height_atr value."""
        if box_height_atr < 2.0:
            return "[0-2)"
        elif box_height_atr < 3.0:
            return "[2-3)"
        elif box_height_atr < 4.0:
            return "[3-4)"
        elif box_height_atr < 6.0:
            return "[4-6)"
        else:
            return "[6+)"

    def compute_bin_stats(self) -> Dict[str, dict]:
        """Compute performance stats per bin."""
        stats = {}
        for bin_name, trades in self.bin_trades.items():
            if not trades:
                continue

            wins = [t for t in trades if t.get("pnl", 0) > 0]
            losses = [t for t in trades if t.get("pnl", 0) <= 0]
            total_win = sum(t.get("pnl", 0) for t in wins)
            total_loss = abs(sum(t.get("pnl", 0) for t in losses))

            stats[bin_name] = {
                "count": len(trades),
                "win_rate": len(wins) / len(trades) if trades else 0,
                "pf": total_win / total_loss if total_loss > 0 else float("inf"),
                "avg_pnl": sum(t.get("pnl", 0) for t in trades) / len(trades) if trades else 0,
            }
        return stats

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "total_candidates": self.total_candidates,
            "passed": self.passed,
            "failed_min_height_atr": self.failed_min_height_atr,
            "failed_reaction": self.failed_reaction,
            "failed_cost": self.failed_cost,
            "pass_rate": self.get_pass_rate(),
            "top_fail_reason": self.get_top_fail_reason(),
            "median_passed_box_height": self.get_median_box_height(True),
            "median_failed_box_height": self.get_median_box_height(False),
            "bin_stats": self.compute_bin_stats(),
        }


class BoxStrengthFilter:
    """Filter trades based on box strength (market quality assessment).

    This filter blocks trades unless the current "box" is strong enough,
    avoiding low-quality RANGE/LOWVOL noise zones.
    """

    def __init__(
        self,
        config: BoxFilterConfig,
        fee_bps: float = 4.0,
        slippage_bps: float = 2.0,
    ):
        """Initialize box strength filter.

        Args:
            config: Box filter configuration
            fee_bps: Fee in basis points (per side)
            slippage_bps: Slippage in basis points
        """
        self.config = config
        self.fee_bps = fee_bps
        self.slippage_bps = slippage_bps

        # Precomputed columns (set by precompute())
        self._box_high: Optional[pd.Series] = None
        self._box_low: Optional[pd.Series] = None
        self._atr: Optional[pd.Series] = None

        # Current box state
        self._current_state: Optional[BoxState] = None
        self._last_update_bar: int = -999

        # Diagnostics
        self.diagnostics = BoxFilterDiagnostics()

        logger.debug(
            "BoxStrengthFilter initialized",
            enabled=config.enabled,
            min_height_atr=config.min_height_atr,
        )

    def precompute(self, df: pd.DataFrame, atr: pd.Series) -> None:
        """Precompute rolling box high/low and ATR.

        Args:
            df: DataFrame with OHLCV data
            atr: Precomputed ATR series
        """
        L = self.config.lookback_bars

        # Precompute rolling max/min
        self._box_high = df["high"].astype(float).rolling(L, min_periods=1).max()
        self._box_low = df["low"].astype(float).rolling(L, min_periods=1).min()
        self._atr = atr

        logger.debug(
            "Box filter precomputed",
            lookback_bars=L,
            data_len=len(df),
        )

    def update(self, df: pd.DataFrame, idx: int) -> BoxState:
        """Update box state at given index.

        Args:
            df: DataFrame with OHLCV data
            idx: Current bar index

        Returns:
            BoxState at current bar
        """
        if self._box_high is None or self._box_low is None or self._atr is None:
            raise RuntimeError("Must call precompute() before update()")

        box_high = float(self._box_high.iloc[idx])
        box_low = float(self._box_low.iloc[idx])
        box_height = box_high - box_low

        atr = float(self._atr.iloc[idx]) if not pd.isna(self._atr.iloc[idx]) else 0.001
        box_height_atr = box_height / atr if atr > 0 else 0.0

        # Expected move (half-box traversal)
        expected_move_atr = box_height_atr / 2.0

        # Cost calculation
        close_price = float(df.iloc[idx]["close"])
        total_cost_bps = (self.fee_bps * 2) + (self.slippage_bps * 2)  # Round trip
        costs_atr = (total_cost_bps / 10000) * close_price / atr if atr > 0 else 0.0

        # Reaction score (if enabled)
        reaction_score = 0.0
        if self.config.use_reaction_filter:
            reaction_score = self._compute_reaction_score(df, idx, atr)

        self._current_state = BoxState(
            box_high=box_high,
            box_low=box_low,
            box_height=box_height,
            atr=atr,
            box_height_atr=box_height_atr,
            expected_move_atr=expected_move_atr,
            costs_atr=costs_atr,
            reaction_score=reaction_score,
            bar_idx=idx,
        )

        self._last_update_bar = idx
        return self._current_state

    def allow_trade(
        self,
        df: pd.DataFrame,
        idx: int,
        side: str,
        zone=None,
    ) -> Tuple[bool, str, dict]:
        """Check if trade is allowed based on box strength.

        Args:
            df: DataFrame with OHLCV data
            idx: Current bar index
            side: Trade side ("long" or "short")
            zone: Optional zone object

        Returns:
            Tuple of (allowed: bool, reason: str, debug: dict)
        """
        if not self.config.enabled:
            return True, "FILTER_DISABLED", {}

        # Update state if needed
        if self._current_state is None or idx - self._last_update_bar >= self.config.update_every_bars:
            self.update(df, idx)

        state = self._current_state
        debug = {
            "box_high": state.box_high,
            "box_low": state.box_low,
            "box_height_atr": state.box_height_atr,
            "atr": state.atr,
            "expected_move_atr": state.expected_move_atr,
            "costs_atr": state.costs_atr,
        }

        # Track as candidate
        self.diagnostics.total_candidates += 1

        # Check 1: Minimum height ATR
        if state.box_height_atr < self.config.min_height_atr:
            self.diagnostics.failed_min_height_atr += 1
            self.diagnostics.failed_box_heights.append(state.box_height_atr)
            return False, "FAIL_MIN_HEIGHT_ATR", debug

        # Check 2: Reaction filter (optional)
        if self.config.use_reaction_filter:
            if state.reaction_score < self.config.reaction_min_rate:
                self.diagnostics.failed_reaction += 1
                self.diagnostics.failed_box_heights.append(state.box_height_atr)
                debug["reaction_score"] = state.reaction_score
                return False, "FAIL_REACTION", debug

        # Check 3: Cost filter (optional)
        if self.config.use_cost_filter:
            if state.expected_move_atr < state.costs_atr * self.config.cost_safety_mult:
                self.diagnostics.failed_cost += 1
                self.diagnostics.failed_box_heights.append(state.box_height_atr)
                return False, "FAIL_COST", debug

        # All checks passed
        self.diagnostics.passed += 1
        self.diagnostics.passed_box_heights.append(state.box_height_atr)
        return True, "PASS", debug

    def get_current_box_height_atr(self) -> float:
        """Get current box_height_atr for trade tracking."""
        if self._current_state is None:
            return 0.0
        return self._current_state.box_height_atr

    def _compute_reaction_score(self, df: pd.DataFrame, idx: int, atr: float) -> float:
        """Compute reaction score at edges.

        Measures how often price bounces near box edges.

        Args:
            df: DataFrame with OHLCV data
            idx: Current bar index
            atr: Current ATR value

        Returns:
            Reaction score (0.0 to 1.0)
        """
        if idx < self.config.reaction_lookback_touches + self.config.reaction_horizon_bars:
            return 0.0

        state = self._current_state
        if state is None:
            return 0.0

        edge_band = atr * self.config.edge_band_atr_k
        move_target = atr * self.config.move_target_atr_k

        # Support band
        support_band_low = state.box_low
        support_band_high = state.box_low + edge_band

        # Resistance band
        resist_band_low = state.box_high - edge_band
        resist_band_high = state.box_high

        # Find touches and compute success rate
        support_touches = 0
        support_successes = 0
        resist_touches = 0
        resist_successes = 0

        lookback_start = max(0, idx - self.config.reaction_lookback_touches)

        for i in range(lookback_start, idx):
            low = float(df.iloc[i]["low"])
            high = float(df.iloc[i]["high"])

            # Support touch
            if low <= support_band_high and low >= support_band_low - edge_band:
                support_touches += 1
                # Check success (price moves up within horizon)
                for j in range(i + 1, min(i + self.config.reaction_horizon_bars + 1, idx)):
                    future_high = float(df.iloc[j]["high"])
                    if future_high - low >= move_target:
                        support_successes += 1
                        break

            # Resistance touch
            if high >= resist_band_low and high <= resist_band_high + edge_band:
                resist_touches += 1
                # Check success (price moves down within horizon)
                for j in range(i + 1, min(i + self.config.reaction_horizon_bars + 1, idx)):
                    future_low = float(df.iloc[j]["low"])
                    if high - future_low >= move_target:
                        resist_successes += 1
                        break

        support_rate = support_successes / support_touches if support_touches > 0 else 0.0
        resist_rate = resist_successes / resist_touches if resist_touches > 0 else 0.0

        return max(support_rate, resist_rate)

    def reset_diagnostics(self) -> None:
        """Reset diagnostics for new run."""
        self.diagnostics = BoxFilterDiagnostics()
