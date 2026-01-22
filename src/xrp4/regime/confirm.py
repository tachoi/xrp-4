"""Regime Confirm Layer for HMM output filtering.

Implements:
1. TRANSITION -> TREND_UP/TREND_DOWN confirmation rules
2. HIGH_VOL exit + cooldown + retrigger rules
3. Priority-based regime confirmation
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, List
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ConfirmResult:
    """Result of regime confirmation."""
    confirmed_regime: str  # RANGE, TREND_UP, TREND_DOWN, HIGH_VOL, TRANSITION, NO_TRADE
    reason: str            # Short reason code
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class ConfirmConfig:
    """Configuration for RegimeConfirmLayer."""
    # Box/Breakout parameters
    BOX_LOOKBACK_15M: int = 32  # 6-8 hours
    TREND_CONFIRM_B_ATR: float = 0.8  # Breakout threshold
    TREND_CONFIRM_S_ATR: float = 0.07  # Slope threshold
    TREND_CONFIRM_EWM_RET_SIGMA: float = 0.20  # Drift threshold multiplier
    TREND_CONFIRM_CONSEC_BARS: int = 2  # Consecutive bars required

    # SPIKE detection (uses 3m data for faster detection) - RELAXED 2026-01-19
    SPIKE_THRESHOLD_3M_PCT: float = 2.0   # 2% move in 3m = spike (was 1.0%)
    SPIKE_THRESHOLD_15M_PCT: float = 3.5  # 3.5% move in 15m = spike (was 2.0%)
    SPIKE_COOLDOWN_BARS_3M: int = 5       # 15 min cooldown (was 30 min)

    # HIGH_VOL parameters (only for extreme cases)
    VOL_BASE_LOOKBACK_15M: int = 96  # 24h baseline
    HIGH_VOL_LAMBDA_ON: float = 6.0   # Entry: V > base + 6.0*MAD (very extreme only)
    HIGH_VOL_LAMBDA_OFF: float = 4.0  # Exit: V < base + 4.0*MAD
    HIGH_VOL_STABLE_N: int = 2  # Stability window (fast exit)
    HIGH_VOL_STABLE_K: int = 1  # Min decreasing bars (fast exit)
    HIGH_VOL_COOLDOWN_BARS_15M: int = 1  # 15min cooldown
    HIGH_VOL_RETRIGGER_EXTEND: bool = False  # Disable re-triggering

    # Structure restoration
    RANGE_RESTORE_THRESHOLD: float = 0.8  # range_comp threshold

    # TREND validation parameters (relaxed 2026-01-19)
    TREND_MIN_EMA_SLOPE_15M: float = 0.0003  # Min slope to confirm TREND (was 0.001, 0.002)
    TREND_OVERRIDE_TO_RANGE: bool = True     # Override TREND to RANGE if slope too low
    TREND_MIN_DRIFT: float = 0.00005         # Minimum drift for TREND (was 0.0001)

    # RANGE validation parameters (relaxed 2026-01-19)
    RANGE_MAX_EMA_SLOPE: float = 0.0005     # RANGE: max EMA slope (was 0.002)
    RANGE_MAX_VOLATILITY: float = 0.008     # RANGE: max volatility (ewm_std_ret)
    RANGE_MIN_DURATION_BARS: int = 3        # Min bars to confirm RANGE


class RegimeConfirmLayer:
    """Confirm layer for HMM regime output.

    Filters raw HMM regimes through confirmation rules:
    1. HIGH_VOL handling (highest priority, can override any regime)
    2. TRANSITION -> TREND confirmation
    3. Pass-through for other regimes
    """

    def __init__(self, cfg: Optional[ConfirmConfig] = None):
        self.cfg = cfg or ConfirmConfig()

    def confirm(
        self,
        regime_raw: str,
        row_15m: dict,
        hist_15m: pd.DataFrame,
        state: Optional[dict] = None,
    ) -> Tuple[ConfirmResult, dict]:
        """Confirm regime with rules.

        Args:
            regime_raw: Raw HMM regime (RANGE, TREND_UP, TREND_DOWN, HIGH_VOL, TRANSITION)
            row_15m: Current 15m feature row as dict
            hist_15m: Historical 15m data (last N rows including current)
            state: Persistent state for cooldown/retrigger

        Returns:
            (ConfirmResult, updated_state)
        """
        # Initialize state if needed
        if state is None:
            state = {
                "high_vol_active": False,
                "high_vol_cooldown_left": 0,
                "high_vol_last_v": 0.0,
                "spike_cooldown_left": 0,  # NEW: spike cooldown
                "consec_breakout_up": 0,
                "consec_breakout_down": 0,
            }

        state = state.copy()

        # Priority 1: HIGH_VOL handling
        result, state = self._handle_high_vol(regime_raw, row_15m, hist_15m, state)
        if result is not None:
            return result, state

        # Priority 2: RANGE validation (NEW - no longer passthrough)
        if regime_raw == "RANGE":
            result, state = self._validate_range(row_15m, hist_15m, state)
            return result, state

        # Priority 3: TRANSITION -> TREND confirmation
        if regime_raw == "TRANSITION":
            result, state = self._confirm_transition(row_15m, hist_15m, state)
            return result, state

        # Priority 4: TREND validation - check if TREND should be RANGE
        if regime_raw in ("TREND_UP", "TREND_DOWN"):
            result, state = self._validate_trend(regime_raw, row_15m, state)
            return result, state

        # Priority 5: Pass-through (other regimes)
        return ConfirmResult(
            confirmed_regime=regime_raw,
            reason=f"{regime_raw}_PASSTHROUGH",
            metrics={},
        ), state

    def _handle_high_vol(
        self,
        regime_raw: str,
        row_15m: dict,
        hist_15m: pd.DataFrame,
        state: dict,
    ) -> Tuple[Optional[ConfirmResult], dict]:
        """Handle SPIKE detection and HIGH_VOL state machine.

        LOGIC (2026-01-19):
        1. SPIKE detection using 3m data: |ret_3m| >= 1% → 30 min cooldown
        2. HIGH_VOL: Disabled (only spike-based protection)

        This allows trading opportunities after spikes (30 min cooldown)
        instead of blocking for hours with HIGH_VOL.
        """
        cfg = self.cfg

        # Get current return (3m for fast spike detection, 15m as backup)
        ret_3m = row_15m.get("ret_3m", 0)
        ret_15m = row_15m.get("ret_15m", 0)
        V = row_15m.get("ewm_std_ret_15m", 0)

        # Spike thresholds
        spike_threshold_3m = cfg.SPIKE_THRESHOLD_3M_PCT / 100  # 1% in 3m
        spike_threshold_15m = cfg.SPIKE_THRESHOLD_15M_PCT / 100  # 2% in 15m

        metrics = {
            "ret_3m": ret_3m,
            "ret_15m": ret_15m,
            "V": V,
            "spike_threshold_3m": spike_threshold_3m,
            "spike_threshold_15m": spike_threshold_15m,
        }

        # ================================================================
        # PRIORITY 1: Check SPIKE cooldown (in progress)
        # ================================================================
        spike_cooldown_left = state.get("spike_cooldown_left", 0)
        if spike_cooldown_left > 0:
            state["spike_cooldown_left"] = spike_cooldown_left - 1
            if spike_cooldown_left - 1 > 0:
                return ConfirmResult(
                    confirmed_regime="NO_TRADE",
                    reason="SPIKE_COOLDOWN",
                    metrics={"cooldown_left": spike_cooldown_left - 1, **metrics},
                ), state
            else:
                # Cooldown finished → allow trading via TRANSITION
                return ConfirmResult(
                    confirmed_regime="TRANSITION",
                    reason="SPIKE_COOLDOWN_END",
                    metrics=metrics,
                ), state

        # ================================================================
        # PRIORITY 2: Detect NEW SPIKE (3m data for faster detection)
        # ================================================================
        # Check 3m spike first (faster)
        if abs(ret_3m) >= spike_threshold_3m:
            state["spike_cooldown_left"] = cfg.SPIKE_COOLDOWN_BARS_3M
            direction = "UP" if ret_3m > 0 else "DOWN"
            return ConfirmResult(
                confirmed_regime="NO_TRADE",
                reason=f"SPIKE_{direction}_3M",
                metrics={"spike_ret": ret_3m, **metrics},
            ), state

        # Check 15m spike as backup (in case 3m missed cumulative move)
        if abs(ret_15m) >= spike_threshold_15m:
            state["spike_cooldown_left"] = cfg.SPIKE_COOLDOWN_BARS_3M
            direction = "UP" if ret_15m > 0 else "DOWN"
            return ConfirmResult(
                confirmed_regime="NO_TRADE",
                reason=f"SPIKE_{direction}_15M",
                metrics={"spike_ret": ret_15m, **metrics},
            ), state

        # ================================================================
        # PRIORITY 3: Override HMM HIGH_VOL (no longer using HIGH_VOL state)
        # ================================================================
        if regime_raw == "HIGH_VOL":
            return ConfirmResult(
                confirmed_regime="TRANSITION",
                reason="HIGH_VOL_OVERRIDE_SPIKE_MODE",
                metrics=metrics,
            ), state

        # No volatility handling needed
        return None, state

    def _validate_range(
        self,
        row_15m: dict,
        hist_15m: pd.DataFrame,
        state: dict,
    ) -> Tuple[ConfirmResult, dict]:
        """Validate RANGE regime - check if it should be overridden to TREND.

        RANGE should have:
        - Low directional drift (low ewm_ret)
        - Flat EMA slope (near zero)
        - Moderate volatility (not too high)

        If EMA slope or drift is high, override to TREND_UP/DOWN.
        If volatility is too high, override to TRANSITION.
        """
        cfg = self.cfg

        # Get validation metrics
        ema_slope = row_15m.get("ema_slope_15m", 0)
        ewm_std = row_15m.get("ewm_std_ret_15m", 0.005)
        ewm_ret = row_15m.get("ewm_ret_15m", 0)

        # Handle NaN values
        if pd.isna(ema_slope):
            ema_slope = 0
        if pd.isna(ewm_std):
            ewm_std = 0.005
        if pd.isna(ewm_ret):
            ewm_ret = 0

        # Debug logging
        logger.debug(f"[RANGE_VALIDATE] ema_slope={ema_slope:.6f}, threshold={cfg.RANGE_MAX_EMA_SLOPE:.6f}, "
                    f"ewm_std={ewm_std:.6f}, ewm_ret={ewm_ret:.6f}")

        metrics = {
            "ema_slope": ema_slope,
            "ewm_std": ewm_std,
            "ewm_ret": ewm_ret,
            "ema_slope_threshold": cfg.RANGE_MAX_EMA_SLOPE,
            "vol_threshold": cfg.RANGE_MAX_VOLATILITY,
        }

        # Validation 1: EMA slope should be flat for RANGE
        # If slope is steep, this is likely a TREND
        if abs(ema_slope) > cfg.RANGE_MAX_EMA_SLOPE:
            direction = "TREND_UP" if ema_slope > 0 else "TREND_DOWN"
            logger.info(f"[RANGE->TREND] EMA slope {ema_slope:.6f} > threshold {cfg.RANGE_MAX_EMA_SLOPE:.6f} -> {direction}")
            return ConfirmResult(
                confirmed_regime=direction,
                reason="RANGE_OVERRIDE_TREND_SLOPE",
                metrics=metrics,
            ), state

        # Validation 2: Volatility should not be excessive for RANGE
        # High volatility suggests TRANSITION (uncertain market)
        if ewm_std > cfg.RANGE_MAX_VOLATILITY:
            return ConfirmResult(
                confirmed_regime="TRANSITION",
                reason="RANGE_OVERRIDE_TRANSITION_HIGH_VOL",
                metrics=metrics,
            ), state

        # Validation 3: Drift should be minimal for RANGE
        # If drift is significant, market is trending
        drift_threshold = cfg.TREND_MIN_DRIFT * 2
        if abs(ewm_ret) > drift_threshold:
            direction = "TREND_UP" if ewm_ret > 0 else "TREND_DOWN"
            return ConfirmResult(
                confirmed_regime=direction,
                reason="RANGE_OVERRIDE_TREND_DRIFT",
                metrics=metrics,
            ), state

        # RANGE validated - conditions are met
        return ConfirmResult(
            confirmed_regime="RANGE",
            reason="RANGE_VALIDATED",
            metrics=metrics,
        ), state

    def _check_high_vol_exit(
        self,
        V: float,
        V_hi_off: float,
        row_15m: dict,
        hist_15m: pd.DataFrame,
        state: dict,
    ) -> bool:
        """Check if HIGH_VOL can exit."""
        cfg = self.cfg

        # Condition 1: V below exit threshold
        if V > V_hi_off:
            return False

        # Condition 2: Stability - count decreasing V in last N bars
        if len(hist_15m) >= cfg.HIGH_VOL_STABLE_N:
            recent_v = hist_15m["ewm_std_ret_15m"].tail(cfg.HIGH_VOL_STABLE_N).values
            decreases = sum(1 for i in range(1, len(recent_v)) if recent_v[i] <= recent_v[i-1])
            if decreases < cfg.HIGH_VOL_STABLE_K:
                return False

        # Condition 3: Structure restored (RANGE or TREND)
        range_comp = row_15m.get("range_comp_15m", 0)
        bb_width = row_15m.get("bb_width_15m", 0)

        # Check RANGE restoration
        if range_comp >= cfg.RANGE_RESTORE_THRESHOLD:
            return True

        # Check if TREND would be confirmed (simplified)
        ewm_ret = row_15m.get("ewm_ret_15m", 0)
        ewm_std = row_15m.get("ewm_std_ret_15m", 0.001)
        drift_threshold = cfg.TREND_CONFIRM_EWM_RET_SIGMA * ewm_std

        if abs(ewm_ret) >= drift_threshold:
            return True

        return False

    def _validate_trend(
        self,
        regime_raw: str,
        row_15m: dict,
        state: dict,
    ) -> Tuple[ConfirmResult, dict]:
        """Validate TREND_UP/TREND_DOWN based on EMA slope.

        If EMA slope is below threshold, override to RANGE.
        This prevents false TREND signals when market is actually ranging.
        """
        cfg = self.cfg

        # Get EMA slope
        ema_slope = row_15m.get("ema_slope_15m", 0)

        metrics = {
            "ema_slope": ema_slope,
            "threshold": cfg.TREND_MIN_EMA_SLOPE_15M,
        }

        # Check if slope validates the trend direction
        if cfg.TREND_OVERRIDE_TO_RANGE:
            if regime_raw == "TREND_UP" and ema_slope < cfg.TREND_MIN_EMA_SLOPE_15M:
                return ConfirmResult(
                    confirmed_regime="RANGE",
                    reason="TREND_UP_OVERRIDE_RANGE_LOW_SLOPE",
                    metrics=metrics,
                ), state

            if regime_raw == "TREND_DOWN" and ema_slope > -cfg.TREND_MIN_EMA_SLOPE_15M:
                return ConfirmResult(
                    confirmed_regime="RANGE",
                    reason="TREND_DOWN_OVERRIDE_RANGE_LOW_SLOPE",
                    metrics=metrics,
                ), state

        # Trend validated
        return ConfirmResult(
            confirmed_regime=regime_raw,
            reason=f"{regime_raw}_VALIDATED",
            metrics=metrics,
        ), state

    def _confirm_transition(
        self,
        row_15m: dict,
        hist_15m: pd.DataFrame,
        state: dict,
    ) -> Tuple[ConfirmResult, dict]:
        """Confirm TRANSITION to TREND_UP or TREND_DOWN."""
        cfg = self.cfg

        # Get current values
        close = row_15m.get("close", 0)
        atr_pct = row_15m.get("atr_pct_15m", 0)
        ATR = atr_pct * close / 100 if atr_pct > 0 else 0.001

        ewm_ret = row_15m.get("ewm_ret_15m", 0)
        ewm_std = row_15m.get("ewm_std_ret_15m", 0.001)
        ema_slope = row_15m.get("ema_slope_15m", 0)

        # Calculate HH_n, LL_n from history (excluding current bar)
        lookback = min(cfg.BOX_LOOKBACK_15M, len(hist_15m) - 1)
        if lookback < 2:
            return ConfirmResult(
                confirmed_regime="TRANSITION",
                reason="TRANSITION_INSUFFICIENT_HISTORY",
                metrics={},
            ), state

        hist_prev = hist_15m.iloc[-(lookback+1):-1]  # Exclude current bar
        HH_n = hist_prev["high"].max()
        LL_n = hist_prev["low"].min()

        # Breakout magnitude
        B_up = (close - HH_n) / ATR if ATR > 0 else 0
        B_dn = (LL_n - close) / ATR if ATR > 0 else 0

        # Drift threshold
        theta_mu = cfg.TREND_CONFIRM_EWM_RET_SIGMA * ewm_std

        # Slope threshold
        theta_S = cfg.TREND_CONFIRM_S_ATR

        metrics = {
            "B_up": B_up,
            "B_dn": B_dn,
            "mu": ewm_ret,
            "sigma": ewm_std,
            "theta_mu": theta_mu,
            "S": ema_slope,
            "HH_n": HH_n,
            "LL_n": LL_n,
            "ATR": ATR,
        }

        # Check TREND_UP conditions
        up_breakout = B_up >= cfg.TREND_CONFIRM_B_ATR
        up_drift = ewm_ret >= theta_mu
        up_slope = ema_slope >= theta_S

        # Check TREND_DOWN conditions
        down_breakout = B_dn >= cfg.TREND_CONFIRM_B_ATR
        down_drift = ewm_ret <= -theta_mu
        down_slope = ema_slope <= -theta_S

        # Update consecutive counters
        if up_breakout:
            state["consec_breakout_up"] = state.get("consec_breakout_up", 0) + 1
        else:
            state["consec_breakout_up"] = 0

        if down_breakout:
            state["consec_breakout_down"] = state.get("consec_breakout_down", 0) + 1
        else:
            state["consec_breakout_down"] = 0

        # Check TREND_UP confirmation
        if (state["consec_breakout_up"] >= cfg.TREND_CONFIRM_CONSEC_BARS
            and up_drift and up_slope):
            return ConfirmResult(
                confirmed_regime="TREND_UP",
                reason="TRANSITION_CONFIRM_UP",
                metrics=metrics,
            ), state

        # Check TREND_DOWN confirmation
        if (state["consec_breakout_down"] >= cfg.TREND_CONFIRM_CONSEC_BARS
            and down_drift and down_slope):
            return ConfirmResult(
                confirmed_regime="TREND_DOWN",
                reason="TRANSITION_CONFIRM_DOWN",
                metrics=metrics,
            ), state

        # Not confirmed
        return ConfirmResult(
            confirmed_regime="TRANSITION",
            reason="TRANSITION_NOT_CONFIRMED",
            metrics=metrics,
        ), state
