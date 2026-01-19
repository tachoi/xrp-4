"""
3m FSM (Finite State Machine) for the XRP Core Trading System.
Handles anchor detection, pullback/retest, phenomena conditions, entry/exit.
"""
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass, field

from .types import (
    Candle, ZoneState, ZoneStateType, ZoneKind, FSMState, Signal, Decision, Side,
    AnchorInfo, PhenomenaResult, IndicatorValues
)
from .config import FSMConfig, PhenomenaConfig
from .indicators import (
    mean, find_swing_low, find_swing_high, is_rsi_turning_up,
    is_rsi_turning_down, detect_divergence
)


@dataclass
class FSMContext:
    """Internal FSM context storage."""
    anchor: Optional[AnchorInfo] = None
    candles_since_anchor: int = 0
    candles_in_trade: int = 0
    cooldown_remaining: int = 0
    entry_price: float = 0.0
    stop_price: float = 0.0
    initial_stop: float = 0.0         # Original stop for R calculation
    trade_side: Optional[Side] = None
    last_decision: Optional[Decision] = None
    max_favorable_price: float = 0.0  # Best price during trade (for trailing)
    # Pullback entry tracking
    trigger_price: float = 0.0        # Price at trigger (for pullback calculation)
    candles_since_trigger: int = 0    # Candles waiting for pullback
    mfe_pct: float = 0.0              # Max favorable excursion % during trade


class TradingFSM:
    """
    3m Trading FSM.

    States:
    - IDLE: No position, waiting for anchor
    - ANCHOR_FOUND: Anchor candle detected
    - PULLBACK_WAIT: Waiting for pullback to A_mid
    - ENTRY_READY: Pullback + phenomena confirmed, waiting for trigger
    - IN_TRADE: Position open
    - EXIT_COOLDOWN: After exit, blocking new entries

    Gate rules enforced by caller:
    - OUTSIDE: force IDLE
    - NEAR: allow up to ENTRY_READY, block IN_TRADE
    - CORE: full operation
    """

    def __init__(self, fsm_config: FSMConfig, phenomena_config: PhenomenaConfig):
        self.fsm_config = fsm_config
        self.phenomena_config = phenomena_config
        self.state = FSMState.IDLE
        self.ctx = FSMContext()
        self._candle_index = 0

    def reset(self) -> None:
        """Reset FSM to initial state."""
        self.state = FSMState.IDLE
        self.ctx = FSMContext()
        self._candle_index = 0

    def step(self, candle: Candle, indicators: IndicatorValues,
             zone_state: ZoneState, recent_candles: List[Candle],
             rsi_history: List[float]) -> Decision:
        """
        Process one 3m candle through the FSM.

        Args:
            candle: Current 3m candle
            indicators: Pre-computed indicator values
            zone_state: Current zone state from gate
            recent_candles: Recent candles for lookback
            rsi_history: RSI history for divergence detection

        Returns:
            Decision with signal and explanation
        """
        self._candle_index += 1
        prev_state = self.state

        # Handle OUTSIDE zone - force IDLE
        if zone_state.state == ZoneStateType.OUTSIDE:
            return self._force_idle(prev_state, zone_state)

        # Handle cooldown
        if self.state == FSMState.EXIT_COOLDOWN:
            return self._handle_cooldown(prev_state, zone_state)

        # Handle IN_TRADE
        if self.state == FSMState.IN_TRADE:
            return self._handle_in_trade(
                candle, indicators, zone_state, recent_candles, rsi_history
            )

        # Handle WAIT_ENTRY_PULLBACK
        if self.state == FSMState.WAIT_ENTRY_PULLBACK:
            return self._handle_wait_entry_pullback(
                candle, indicators, zone_state, recent_candles
            )

        # Handle ENTRY_READY
        if self.state == FSMState.ENTRY_READY:
            return self._handle_entry_ready(
                candle, indicators, zone_state, recent_candles
            )

        # Handle PULLBACK_WAIT
        if self.state == FSMState.PULLBACK_WAIT:
            return self._handle_pullback_wait(
                candle, indicators, zone_state, recent_candles, rsi_history
            )

        # Handle ANCHOR_FOUND (immediate transition to PULLBACK_WAIT)
        if self.state == FSMState.ANCHOR_FOUND:
            self.state = FSMState.PULLBACK_WAIT
            return Decision(
                signal=Signal.NONE,
                prev_state=prev_state,
                new_state=self.state,
                explanation="Anchor confirmed. Waiting for pullback to A_mid.",
                anchor_info=self.ctx.anchor,
            )

        # IDLE - check for anchor
        return self._handle_idle(candle, indicators, zone_state, recent_candles)

    def _force_idle(self, prev_state: FSMState,
                     zone_state: ZoneState) -> Decision:
        """Force transition to IDLE when OUTSIDE zone."""
        self.state = FSMState.IDLE
        self.ctx.anchor = None
        self.ctx.candles_since_anchor = 0

        return Decision(
            signal=Signal.NONE,
            prev_state=prev_state,
            new_state=FSMState.IDLE,
            explanation=f"Forced IDLE: {zone_state.explanation}",
            debug={'forced_idle': True, 'zone_state': zone_state.state.value}
        )

    def _handle_idle(self, candle: Candle, indicators: IndicatorValues,
                      zone_state: ZoneState,
                      recent_candles: List[Candle]) -> Decision:
        """Handle IDLE state - look for anchor candle."""
        prev_state = self.state

        # Check anchor detection conditions (now includes zone alignment check)
        anchor = self._detect_anchor(candle, indicators, recent_candles, zone_state)

        if anchor is not None:
            self.ctx.anchor = anchor
            self.ctx.candles_since_anchor = 0
            self.state = FSMState.ANCHOR_FOUND

            direction = "bullish" if anchor.direction > 0 else "bearish"

            return Decision(
                signal=Signal.ANCHOR,
                prev_state=prev_state,
                new_state=self.state,
                explanation=(
                    f"{zone_state.explanation}: Anchor found "
                    f"({direction}, vol {anchor.volume_mult:.1f}x, "
                    f"body {anchor.body_atr_mult:.1f} ATR). Waiting pullback."
                ),
                anchor_info=anchor,
                debug={
                    'anchor_open': anchor.open,
                    'anchor_close': anchor.close,
                    'anchor_mid': anchor.mid,
                }
            )

        return Decision(
            signal=Signal.NONE,
            prev_state=prev_state,
            new_state=self.state,
            explanation=f"IDLE: No anchor detected. {zone_state.explanation}",
        )

    def _detect_anchor(self, candle: Candle, indicators: IndicatorValues,
                        recent_candles: List[Candle],
                        zone_state: ZoneState) -> Optional[AnchorInfo]:
        """
        Detect anchor candle with improved filters.

        Anchor criteria:
        1. volume >= ANCHOR_VOL_MULT * SMA(volume)
        2. body >= ANCHOR_BODY_ATR_MULT * ATR
        3. Chase filter: abs(close - EMA) / ATR <= CHASE_DIST_MAX_ATR
        4. [NEW] Zone-Anchor alignment: anchor direction must match zone expectation
        5. [NEW] EMA trend alignment: anchor direction should align with EMA slope
        """
        cfg = self.fsm_config

        # Volume check
        if indicators.volume_sma <= 0:
            return None

        vol_mult = candle.volume / indicators.volume_sma
        if vol_mult < cfg.anchor_vol_mult:
            return None

        # Body check
        if indicators.atr14 <= 0:
            return None

        body_atr_mult = candle.body / indicators.atr14
        if body_atr_mult < cfg.anchor_body_atr_mult:
            return None

        # Chase filter
        if indicators.ema20 <= 0:
            return None

        ema_distance = abs(candle.close - indicators.ema20) / indicators.atr14
        if ema_distance > cfg.chase_dist_max_atr:
            return None

        # Determine anchor direction
        direction = 1 if candle.is_bullish else -1

        # [NEW] Zone-Anchor alignment check
        # Support zone expects bounce (LONG), Resistance zone expects rejection (SHORT)
        if zone_state.matched_zone is not None:
            zone_kind = zone_state.matched_zone.kind
            if zone_kind == ZoneKind.SUPPORT and direction < 0:
                # Support zone but bearish anchor - skip (expecting bounce, not breakdown)
                return None
            if zone_kind == ZoneKind.RESISTANCE and direction > 0:
                # Resistance zone but bullish anchor - skip (expecting rejection, not breakout)
                return None

        # [NEW] EMA trend alignment check
        # Check EMA slope over recent candles
        if len(recent_candles) >= 5:
            ema_slope = self._check_ema_slope(recent_candles, indicators.ema20)

            # LONG trend filter: block LONG in strong downtrend
            if direction > 0:
                threshold = cfg.long_ema_slope_threshold if cfg.long_trend_filter_enabled else -0.5
                if ema_slope < threshold:
                    # Trying to go LONG in a downtrend - skip
                    return None

            # SHORT trend filter: block SHORT in strong uptrend
            if direction < 0:
                threshold = cfg.short_ema_slope_threshold if cfg.short_trend_filter_enabled else 0.5
                if ema_slope > threshold:
                    # Trying to go SHORT in an uptrend - skip
                    return None

        mid = (candle.open + candle.close) / 2

        return AnchorInfo(
            open=candle.open,
            close=candle.close,
            mid=mid,
            direction=direction,
            ts=candle.ts,
            index=self._candle_index,
            volume_mult=vol_mult,
            body_atr_mult=body_atr_mult,
        )

    def _check_ema_slope(self, recent_candles: List[Candle], current_ema: float) -> float:
        """
        Check EMA slope direction.

        Returns normalized slope:
        - Positive = rising EMA (bullish)
        - Negative = falling EMA (bearish)
        - Value in ATR units per 5 candles
        """
        if len(recent_candles) < 5:
            return 0.0

        # Estimate past EMA by using close prices
        # This is a rough approximation
        closes = [c.close for c in recent_candles[-5:]]
        avg_recent = sum(closes[:3]) / 3
        avg_current = sum(closes[-3:]) / 3

        # Normalize by recent price range
        price_range = max(c.high for c in recent_candles[-5:]) - min(c.low for c in recent_candles[-5:])
        if price_range <= 0:
            return 0.0

        slope = (avg_current - avg_recent) / price_range * 10
        return slope

    def _handle_pullback_wait(self, candle: Candle, indicators: IndicatorValues,
                               zone_state: ZoneState,
                               recent_candles: List[Candle],
                               rsi_history: List[float]) -> Decision:
        """Handle PULLBACK_WAIT - wait for retest of A_mid."""
        prev_state = self.state
        self.ctx.candles_since_anchor += 1
        cfg = self.fsm_config

        anchor = self.ctx.anchor
        if anchor is None:
            self.state = FSMState.IDLE
            return Decision(
                signal=Signal.NONE,
                prev_state=prev_state,
                new_state=self.state,
                explanation="PULLBACK_WAIT: Anchor lost, returning to IDLE",
            )

        # Check expiry
        if self.ctx.candles_since_anchor >= cfg.anchor_expire_candles:
            self.state = FSMState.IDLE
            self.ctx.anchor = None
            return Decision(
                signal=Signal.NONE,
                prev_state=prev_state,
                new_state=self.state,
                explanation=(
                    f"Anchor expired ({self.ctx.candles_since_anchor} candles). "
                    "Returning to IDLE."
                ),
            )

        # Check invalidation
        if self._is_anchor_invalidated(candle, anchor):
            self.state = FSMState.IDLE
            self.ctx.anchor = None
            return Decision(
                signal=Signal.NONE,
                prev_state=prev_state,
                new_state=self.state,
                explanation="Anchor invalidated (price broke A_open). Returning to IDLE.",
                debug={'invalidation': 'a_open_break'}
            )

        # Check pullback near A_mid
        tolerance = cfg.pullback_tolerance_atr * indicators.atr14
        near_a_mid = abs(candle.close - anchor.mid) <= tolerance

        if not near_a_mid:
            return Decision(
                signal=Signal.NONE,
                prev_state=prev_state,
                new_state=self.state,
                explanation=(
                    f"Waiting pullback to A_mid ({anchor.mid:.4f}). "
                    f"Current: {candle.close:.4f}"
                ),
                anchor_info=anchor,
            )

        # Price is near A_mid - check phenomena
        phenomena = self._check_phenomena(recent_candles, indicators)

        if not phenomena.ok:
            return Decision(
                signal=Signal.NONE,
                prev_state=prev_state,
                new_state=self.state,
                explanation=(
                    f"Pullback at A_mid but phenomena not met "
                    f"({phenomena.count}/{self.phenomena_config.requirements_min_count}). "
                    f"{phenomena.details}"
                ),
                phenomena=phenomena,
                anchor_info=anchor,
            )

        # Check RSI confirmation (improved: checks level + direction)
        rsi_confirmed = self._check_rsi_confirmation(
            anchor.direction, rsi_history, indicators.rsi14
        )

        if not rsi_confirmed:
            return Decision(
                signal=Signal.NONE,
                prev_state=prev_state,
                new_state=self.state,
                explanation=(
                    f"Pullback + phenomena OK, but RSI not confirming. "
                    f"RSI={indicators.rsi14:.1f} (need <55 for LONG, >45 for SHORT)"
                ),
                phenomena=phenomena,
                anchor_info=anchor,
            )

        # All conditions met - transition to ENTRY_READY
        self.state = FSMState.ENTRY_READY

        return Decision(
            signal=Signal.ARM,
            prev_state=prev_state,
            new_state=self.state,
            explanation=(
                f"Phenomena OK ({phenomena.details}). RSI turned. ENTRY_READY."
            ),
            phenomena=phenomena,
            anchor_info=anchor,
            debug={'rsi': indicators.rsi14}
        )

    def _handle_entry_ready(self, candle: Candle, indicators: IndicatorValues,
                             zone_state: ZoneState,
                             recent_candles: List[Candle]) -> Decision:
        """Handle ENTRY_READY - wait for entry trigger."""
        prev_state = self.state
        self.ctx.candles_since_anchor += 1
        cfg = self.fsm_config

        anchor = self.ctx.anchor
        if anchor is None:
            self.state = FSMState.IDLE
            return Decision(
                signal=Signal.NONE,
                prev_state=prev_state,
                new_state=self.state,
                explanation="ENTRY_READY: Anchor lost, returning to IDLE",
            )

        # Check expiry
        if self.ctx.candles_since_anchor >= cfg.anchor_expire_candles:
            self.state = FSMState.IDLE
            self.ctx.anchor = None
            return Decision(
                signal=Signal.NONE,
                prev_state=prev_state,
                new_state=self.state,
                explanation="Anchor expired while waiting for entry trigger. IDLE.",
            )

        # Check invalidation
        if self._is_anchor_invalidated(candle, anchor):
            self.state = FSMState.IDLE
            self.ctx.anchor = None
            return Decision(
                signal=Signal.NONE,
                prev_state=prev_state,
                new_state=self.state,
                explanation="Anchor invalidated during ENTRY_READY. IDLE.",
            )

        # Block entry if not in CORE zone
        if zone_state.state != ZoneStateType.CORE:
            return Decision(
                signal=Signal.NONE,
                prev_state=prev_state,
                new_state=self.state,
                explanation=(
                    f"Entry blocked: zone state is {zone_state.state.value}, "
                    "need CORE for entry."
                ),
                anchor_info=anchor,
            )

        # Check entry trigger
        triggered, trigger_explanation = self._check_entry_trigger(
            candle, anchor, recent_candles
        )

        if not triggered:
            return Decision(
                signal=Signal.NONE,
                prev_state=prev_state,
                new_state=self.state,
                explanation=f"ENTRY_READY: {trigger_explanation}",
                anchor_info=anchor,
            )

        # Trigger confirmed - check if pullback entry is enabled
        side = Side.LONG if anchor.direction > 0 else Side.SHORT
        stop = self._compute_stop(side, anchor, recent_candles)
        cfg = self.fsm_config

        if cfg.entry_pullback_enabled:
            # Wait for pullback before entry
            self.state = FSMState.WAIT_ENTRY_PULLBACK
            self.ctx.trigger_price = candle.close
            self.ctx.candles_since_trigger = 0
            self.ctx.trade_side = side
            self.ctx.stop_price = stop
            self.ctx.initial_stop = stop

            return Decision(
                signal=Signal.ARM,  # Armed but waiting for pullback
                prev_state=prev_state,
                new_state=self.state,
                explanation=(
                    f"Trigger confirmed: {trigger_explanation}. "
                    f"Waiting for pullback to enter {side.value}."
                ),
                side=side,
                stop_price=stop,
                anchor_info=anchor,
            )

        # Direct entry (pullback disabled)
        self.state = FSMState.IN_TRADE
        self.ctx.candles_in_trade = 0
        self.ctx.entry_price = candle.close
        self.ctx.stop_price = stop
        self.ctx.initial_stop = stop
        self.ctx.trade_side = side
        self.ctx.max_favorable_price = candle.close
        self.ctx.mfe_pct = 0.0

        signal = Signal.ENTRY_LONG if side == Side.LONG else Signal.ENTRY_SHORT

        return Decision(
            signal=signal,
            prev_state=prev_state,
            new_state=self.state,
            explanation=(
                f"ENTRY_{side.value} confirmed: {trigger_explanation}. "
                f"Stop at {stop:.4f}"
            ),
            side=side,
            entry_price=candle.close,
            stop_price=stop,
            anchor_info=anchor,
        )

    def _handle_wait_entry_pullback(self, candle: Candle, indicators: IndicatorValues,
                                     zone_state: ZoneState,
                                     recent_candles: List[Candle]) -> Decision:
        """
        Handle WAIT_ENTRY_PULLBACK - wait for price to pull back before entry.

        This avoids chasing by entering at a better price after the trigger.

        Entry conditions:
        1. Price pulls back by at least entry_pullback_atr * ATR from trigger
        2. Price doesn't invalidate (break stop level)
        3. Max wait time not exceeded
        """
        prev_state = self.state
        self.ctx.candles_since_trigger += 1
        self.ctx.candles_since_anchor += 1
        cfg = self.fsm_config

        anchor = self.ctx.anchor
        side = self.ctx.trade_side
        stop = self.ctx.stop_price
        trigger_price = self.ctx.trigger_price
        atr = indicators.atr14

        if anchor is None or side is None:
            self.state = FSMState.IDLE
            return Decision(
                signal=Signal.NONE,
                prev_state=prev_state,
                new_state=self.state,
                explanation="WAIT_ENTRY_PULLBACK: Context lost, returning to IDLE",
            )

        # Check if stop is breached (invalidation)
        if side == Side.LONG and candle.low <= stop:
            self.state = FSMState.IDLE
            self.ctx.anchor = None
            return Decision(
                signal=Signal.NONE,
                prev_state=prev_state,
                new_state=self.state,
                explanation=f"Pullback wait cancelled: price hit stop {stop:.4f}",
            )
        if side == Side.SHORT and candle.high >= stop:
            self.state = FSMState.IDLE
            self.ctx.anchor = None
            return Decision(
                signal=Signal.NONE,
                prev_state=prev_state,
                new_state=self.state,
                explanation=f"Pullback wait cancelled: price hit stop {stop:.4f}",
            )

        # Check max wait time
        if self.ctx.candles_since_trigger >= cfg.entry_pullback_max_candles:
            # Enter at current price if still valid (accept less pullback)
            self.state = FSMState.IN_TRADE
            self.ctx.candles_in_trade = 0
            self.ctx.entry_price = candle.close
            self.ctx.max_favorable_price = candle.close
            self.ctx.mfe_pct = 0.0

            signal = Signal.ENTRY_LONG if side == Side.LONG else Signal.ENTRY_SHORT
            return Decision(
                signal=signal,
                prev_state=prev_state,
                new_state=self.state,
                explanation=(
                    f"ENTRY_{side.value}: Max pullback wait reached. "
                    f"Entering at {candle.close:.4f} (trigger was {trigger_price:.4f})"
                ),
                side=side,
                entry_price=candle.close,
                stop_price=stop,
                anchor_info=anchor,
            )

        # Check for pullback
        pullback_threshold = cfg.entry_pullback_atr * atr
        if side == Side.LONG:
            # For LONG: price should pull back (go down) from trigger
            pullback_amount = trigger_price - candle.close
            if pullback_amount >= pullback_threshold:
                # Good pullback - enter now
                self.state = FSMState.IN_TRADE
                self.ctx.candles_in_trade = 0
                self.ctx.entry_price = candle.close
                self.ctx.max_favorable_price = candle.close
                self.ctx.mfe_pct = 0.0

                improvement = (trigger_price - candle.close) / trigger_price * 100
                return Decision(
                    signal=Signal.ENTRY_LONG,
                    prev_state=prev_state,
                    new_state=self.state,
                    explanation=(
                        f"ENTRY_LONG on pullback: {candle.close:.4f} "
                        f"({improvement:.2f}% better than trigger {trigger_price:.4f})"
                    ),
                    side=side,
                    entry_price=candle.close,
                    stop_price=stop,
                    anchor_info=anchor,
                )
        else:  # SHORT
            # For SHORT: price should pull back (go up) from trigger
            pullback_amount = candle.close - trigger_price
            if pullback_amount >= pullback_threshold:
                self.state = FSMState.IN_TRADE
                self.ctx.candles_in_trade = 0
                self.ctx.entry_price = candle.close
                self.ctx.max_favorable_price = candle.close
                self.ctx.mfe_pct = 0.0

                improvement = (candle.close - trigger_price) / trigger_price * 100
                return Decision(
                    signal=Signal.ENTRY_SHORT,
                    prev_state=prev_state,
                    new_state=self.state,
                    explanation=(
                        f"ENTRY_SHORT on pullback: {candle.close:.4f} "
                        f"({improvement:.2f}% better than trigger {trigger_price:.4f})"
                    ),
                    side=side,
                    entry_price=candle.close,
                    stop_price=stop,
                    anchor_info=anchor,
                )

        # Still waiting for pullback
        return Decision(
            signal=Signal.NONE,
            prev_state=prev_state,
            new_state=self.state,
            explanation=(
                f"Waiting for pullback ({self.ctx.candles_since_trigger}/{cfg.entry_pullback_max_candles}). "
                f"Need {pullback_threshold:.4f} pullback from {trigger_price:.4f}"
            ),
            anchor_info=anchor,
            side=side,
        )

    def _handle_in_trade(self, candle: Candle, indicators: IndicatorValues,
                          zone_state: ZoneState,
                          recent_candles: List[Candle],
                          rsi_history: List[float]) -> Decision:
        """
        Handle IN_TRADE - manage position with technical analysis based exits.

        Exit priority:
        1. Hard stop - capital protection (always)
        2. Technical exits (only after min_candles AND min_mfe reached):
           - EMA cross exit
           - Reversal candle exit
           - RSI extreme exit
           - Swing break exit
        3. Time stop - only if momentum fading
        """
        prev_state = self.state
        self.ctx.candles_in_trade += 1
        cfg = self.fsm_config

        side = self.ctx.trade_side
        stop = self.ctx.stop_price
        entry = self.ctx.entry_price
        atr = indicators.atr14

        # Calculate current P&L and track MFE
        if side == Side.LONG:
            pnl_pct = (candle.close - entry) / entry * 100
            current_mfe = (candle.high - entry) / entry * 100
            self.ctx.max_favorable_price = max(self.ctx.max_favorable_price, candle.high)
        else:
            pnl_pct = (entry - candle.close) / entry * 100
            current_mfe = (entry - candle.low) / entry * 100
            self.ctx.max_favorable_price = min(self.ctx.max_favorable_price, candle.low)

        # Update MFE tracking
        self.ctx.mfe_pct = max(self.ctx.mfe_pct, current_mfe)

        # 1. Hard stop - capital protection (always check first)
        stop_hit = False
        if side == Side.LONG and candle.low <= stop:
            stop_hit = True
        elif side == Side.SHORT and candle.high >= stop:
            stop_hit = True

        if stop_hit:
            return self._exit_trade(
                prev_state, f"Hard stop at {stop:.4f}", Signal.EXIT
            )

        # 2. SHORT early take profit (SHORT loses 77.8% of profits if not taken early)
        if side == Side.SHORT and cfg.short_early_exit_enabled:
            if self.ctx.candles_in_trade >= cfg.short_min_candles:
                if pnl_pct >= cfg.short_take_profit_pct:
                    return self._exit_trade(
                        prev_state,
                        f"SHORT early TP: +{pnl_pct:.2f}% (target {cfg.short_take_profit_pct}%)",
                        Signal.EXIT
                    )

        # 3. Breakeven trailing stop - protect profits by moving stop to entry
        if cfg.breakeven_trail_enabled and self.ctx.mfe_pct >= cfg.breakeven_trigger_pct:
            # Move stop to breakeven (with small offset)
            if side == Side.LONG:
                breakeven_stop = entry * (1 + cfg.breakeven_offset_pct / 100)
                if self.ctx.stop_price < breakeven_stop:
                    self.ctx.stop_price = breakeven_stop
            else:  # SHORT
                breakeven_stop = entry * (1 - cfg.breakeven_offset_pct / 100)
                if self.ctx.stop_price > breakeven_stop:
                    self.ctx.stop_price = breakeven_stop

        # Check if technical exits are allowed
        # Requires: min candles elapsed AND trade reached min MFE
        min_candles_ok = self.ctx.candles_in_trade >= cfg.min_candles_for_exit
        min_mfe_ok = self.ctx.mfe_pct >= cfg.min_mfe_for_tech_exit

        # Technical exits only allowed after both conditions met
        tech_exits_allowed = min_candles_ok and min_mfe_ok

        if tech_exits_allowed:
            # 2. EMA cross exit - trend reversal signal (disabled for SHORT due to 0% WR)
            ema_exit_enabled = not (side == Side.SHORT and cfg.short_disable_ema_exit)
            if ema_exit_enabled and indicators.ema20 > 0 and atr > 0:
                ema_exit, ema_reason = self._check_ema_cross_exit(
                    candle, side, indicators.ema20, atr, recent_candles
                )
                if ema_exit:
                    return self._exit_trade(prev_state, ema_reason, Signal.EXIT)

            # 3. Reversal candle exit - strong opposite momentum
            if atr > 0:
                reversal_exit, reversal_reason = self._check_reversal_candle_exit(
                    candle, side, atr, entry
                )
                if reversal_exit:
                    return self._exit_trade(prev_state, reversal_reason, Signal.EXIT)

            # 4. RSI extreme exit - overbought/oversold
            rsi_exit, rsi_explanation = self._check_rsi_exit(
                side, indicators.rsi14, rsi_history
            )
            if rsi_exit:
                return self._exit_trade(prev_state, rsi_explanation, Signal.EXIT)

            # 5. Swing break exit - structure failure
            if len(recent_candles) >= 3:
                swing_exit, swing_reason = self._check_swing_break_exit(
                    candle, side, recent_candles, entry
                )
                if swing_exit:
                    return self._exit_trade(prev_state, swing_reason, Signal.EXIT)

        # 6. Time stop - only if momentum is fading (no clear direction)
        if self.ctx.candles_in_trade >= cfg.hold_max_candles:
            # Check if momentum is fading (small bodies, no progress)
            momentum_fading = self._check_momentum_fading(recent_candles, atr)
            if momentum_fading or pnl_pct < 0:
                return self._exit_trade(
                    prev_state,
                    f"Time stop ({self.ctx.candles_in_trade} candles, momentum fading)",
                    Signal.EXIT
                )

        # Still in trade
        tech_status = "exits enabled" if tech_exits_allowed else f"waiting (candles:{self.ctx.candles_in_trade}/{cfg.min_candles_for_exit}, MFE:{self.ctx.mfe_pct:.2f}%/{cfg.min_mfe_for_tech_exit}%)"
        return Decision(
            signal=Signal.NONE,
            prev_state=prev_state,
            new_state=self.state,
            explanation=(
                f"IN_TRADE {side.value}: {self.ctx.candles_in_trade} candles, "
                f"PnL {pnl_pct:.2f}%, MFE {self.ctx.mfe_pct:.2f}%, [{tech_status}]"
            ),
            side=side,
            entry_price=entry,
            stop_price=self.ctx.stop_price,
        )

    def _check_ema_cross_exit(self, candle: Candle, side: Side,
                               ema: float, atr: float,
                               recent_candles: List[Candle]) -> Tuple[bool, str]:
        """
        Check for EMA cross exit signal.

        LONG: Exit if close < EMA and previous candle also closed < EMA
        SHORT: Exit if close > EMA and previous candle also closed > EMA
        """
        if len(recent_candles) < 2:
            return False, ""

        prev_candle = recent_candles[-2]

        if side == Side.LONG:
            # Need two consecutive closes below EMA for confirmation
            if candle.close < ema and prev_candle.close < ema:
                return True, f"EMA cross exit: 2 closes below EMA {ema:.4f}"
        else:  # SHORT
            if candle.close > ema and prev_candle.close > ema:
                return True, f"EMA cross exit: 2 closes above EMA {ema:.4f}"

        return False, ""

    def _check_reversal_candle_exit(self, candle: Candle, side: Side,
                                     atr: float, entry: float) -> Tuple[bool, str]:
        """
        Check for reversal candle exit.

        Exit on strong opposite direction candle:
        - Body > 1.0 * ATR (strong reversal)
        - Closes beyond entry (against position)
        """
        # Strong body check - must be a significant candle
        if candle.body < 1.0 * atr:
            return False, ""

        if side == Side.LONG:
            # Bearish reversal: closes below entry with strong body
            if not candle.is_bullish and candle.close < entry:
                return True, f"Reversal candle exit: bearish {candle.body/atr:.1f}x ATR"
        else:  # SHORT
            # Bullish reversal: closes above entry with strong body
            if candle.is_bullish and candle.close > entry:
                return True, f"Reversal candle exit: bullish {candle.body/atr:.1f}x ATR"

        return False, ""

    def _check_swing_break_exit(self, candle: Candle, side: Side,
                                 recent_candles: List[Candle],
                                 entry: float) -> Tuple[bool, str]:
        """
        Check for swing structure break exit.

        LONG: Exit if price breaks below recent swing low (after being in profit)
        SHORT: Exit if price breaks above recent swing high (after being in profit)
        """
        if len(recent_candles) < 5:
            return False, ""

        # Get recent swing levels (excluding current candle)
        lookback = min(5, len(recent_candles) - 1)
        recent = recent_candles[-lookback-1:-1]

        if side == Side.LONG:
            # Only check if we were in profit at some point
            if self.ctx.max_favorable_price <= entry:
                return False, ""

            # Find recent swing low
            swing_low = min(c.low for c in recent)
            if candle.close < swing_low:
                return True, f"Swing break exit: broke low {swing_low:.4f}"
        else:  # SHORT
            if self.ctx.max_favorable_price >= entry:
                return False, ""

            swing_high = max(c.high for c in recent)
            if candle.close > swing_high:
                return True, f"Swing break exit: broke high {swing_high:.4f}"

        return False, ""

    def _check_momentum_fading(self, recent_candles: List[Candle],
                                atr: float) -> bool:
        """
        Check if momentum is fading (small bodies, indecision).

        Returns True if last 3 candles have small bodies (< 0.5 ATR).
        """
        if len(recent_candles) < 3 or atr <= 0:
            return False

        last_3 = recent_candles[-3:]
        small_body_count = sum(1 for c in last_3 if c.body < 0.5 * atr)

        return small_body_count >= 2

    def _exit_trade(self, prev_state: FSMState, reason: str,
                     signal: Signal) -> Decision:
        """Exit trade and enter cooldown."""
        side = self.ctx.trade_side
        entry = self.ctx.entry_price

        self.state = FSMState.EXIT_COOLDOWN
        self.ctx.cooldown_remaining = self.fsm_config.cooldown_candles
        self.ctx.anchor = None
        self.ctx.candles_in_trade = 0

        return Decision(
            signal=signal,
            prev_state=prev_state,
            new_state=self.state,
            explanation=f"EXIT: {reason}. Entering cooldown.",
            side=side,
            entry_price=entry,
            stop_price=self.ctx.stop_price,
        )

    def _handle_cooldown(self, prev_state: FSMState,
                          zone_state: ZoneState) -> Decision:
        """Handle EXIT_COOLDOWN state."""
        self.ctx.cooldown_remaining -= 1

        if self.ctx.cooldown_remaining <= 0:
            self.state = FSMState.IDLE
            self.ctx = FSMContext()  # Reset context
            return Decision(
                signal=Signal.NONE,
                prev_state=prev_state,
                new_state=self.state,
                explanation="Cooldown complete. Returning to IDLE.",
            )

        return Decision(
            signal=Signal.COOL,
            prev_state=prev_state,
            new_state=self.state,
            explanation=f"Cooldown: {self.ctx.cooldown_remaining} candles remaining.",
        )

    def _is_anchor_invalidated(self, candle: Candle,
                                anchor: AnchorInfo) -> bool:
        """Check if anchor is invalidated by price action."""
        if anchor.direction > 0:  # LONG
            # Invalid if close breaks below A_open
            return candle.close < anchor.open
        else:  # SHORT
            # Invalid if close breaks above A_open
            return candle.close > anchor.open

    def _check_phenomena(self, recent_candles: List[Candle],
                          indicators: IndicatorValues) -> PhenomenaResult:
        """
        Check phenomenon-based conditions.

        Checks:
        A) low_fail: min(low[t], low[t-1]) >= min(low[t-2], low[t-3])
        B) lower_wick_ratio >= PH_LOWER_WICK_RATIO_MIN
        C) body shrink: body[t] <= PH_BODY_SHRINK_FACTOR * mean(body over K)
        D) range shrink: range[t] <= PH_RANGE_SHRINK_FACTOR * mean(range over K)
        """
        cfg = self.phenomena_config

        if len(recent_candles) < 4:
            return PhenomenaResult(ok=False, details="Not enough candles")

        current = recent_candles[-1]
        prev = recent_candles[-2] if len(recent_candles) > 1 else current

        # A) Low fail check
        low_fail = False
        if len(recent_candles) >= 4:
            min_recent = min(current.low, prev.low)
            min_older = min(recent_candles[-3].low, recent_candles[-4].low)
            low_fail = min_recent >= min_older

        # B) Lower wick ratio
        wick_ratio_ok = current.lower_wick_ratio >= cfg.lower_wick_ratio_min

        # C) Body shrink
        k = min(cfg.lookback_k, len(recent_candles))
        bodies = [c.body for c in recent_candles[-k:]]
        mean_body = mean(bodies[:-1]) if len(bodies) > 1 else bodies[0]
        body_shrink = current.body <= cfg.body_shrink_factor * mean_body if mean_body > 0 else False

        # D) Range shrink
        ranges = [c.range for c in recent_candles[-k:]]
        mean_range = mean(ranges[:-1]) if len(ranges) > 1 else ranges[0]
        range_shrink = current.range <= cfg.range_shrink_factor * mean_range if mean_range > 0 else False

        # Count true conditions
        checks = [low_fail, wick_ratio_ok, body_shrink, range_shrink]
        count = sum(checks)
        ok = count >= cfg.requirements_min_count

        # Build details
        details_parts = []
        if low_fail:
            details_parts.append("low_fail")
        if wick_ratio_ok:
            details_parts.append(f"wick_ratio={current.lower_wick_ratio:.2f}")
        if body_shrink:
            details_parts.append("body_shrink")
        if range_shrink:
            details_parts.append("range_shrink")

        details = ", ".join(details_parts) if details_parts else "none"

        return PhenomenaResult(
            ok=ok,
            low_fail=low_fail,
            wick_ratio_ok=wick_ratio_ok,
            body_shrink=body_shrink,
            range_shrink=range_shrink,
            count=count,
            details=f"[{count}/{cfg.requirements_min_count}] {details}"
        )

    def _check_rsi_confirmation(self, direction: int,
                                 rsi_history: List[float],
                                 current_rsi: float) -> bool:
        """
        Check RSI confirms turn in anchor direction.

        Improved logic:
        - LONG: RSI should be turning up AND not overbought (< 70)
        - SHORT: RSI should be turning down AND not oversold (> 30)
        - Also checks RSI is in favorable zone (40-60 range is neutral)
        """
        if len(rsi_history) < 2:
            return True  # Not enough data, allow

        if direction > 0:  # LONG
            # RSI should be turning up
            if not is_rsi_turning_up(rsi_history):
                return False
            # RSI should not be overbought (room to grow)
            if current_rsi > 65:
                return False
            # RSI from oversold is ideal (< 45)
            # Acceptable if neutral (< 55)
            return current_rsi < 55

        else:  # SHORT
            # RSI should be turning down
            if not is_rsi_turning_down(rsi_history):
                return False
            # RSI should not be oversold (room to fall)
            if current_rsi < 35:
                return False
            # RSI from overbought is ideal (> 55)
            # Acceptable if neutral (> 45)
            return current_rsi > 45

    def _check_entry_trigger(self, candle: Candle, anchor: AnchorInfo,
                              recent_candles: List[Candle]) -> Tuple[bool, str]:
        """
        Check entry trigger condition.

        LONG: close > A_mid AND (close > prev high OR close > max(high[-1], high[-2]))
        SHORT: close < A_mid AND (close < prev low OR close < min(low[-1], low[-2]))
        """
        if len(recent_candles) < 2:
            return False, "Not enough candles for trigger check"

        prev = recent_candles[-2]

        if anchor.direction > 0:  # LONG
            if candle.close <= anchor.mid:
                return False, f"Close {candle.close:.4f} <= A_mid {anchor.mid:.4f}"

            # Check micro break
            if candle.close > prev.high:
                return True, f"close reclaimed A_mid and broke prev high {prev.high:.4f}"

            if len(recent_candles) >= 3:
                max_high = max(recent_candles[-2].high, recent_candles[-3].high)
                if candle.close > max_high:
                    return True, f"close reclaimed A_mid and broke 2-bar high {max_high:.4f}"

            return False, f"Close above A_mid but no micro break yet"

        else:  # SHORT
            if candle.close >= anchor.mid:
                return False, f"Close {candle.close:.4f} >= A_mid {anchor.mid:.4f}"

            # Check micro break
            if candle.close < prev.low:
                return True, f"close broke below A_mid and prev low {prev.low:.4f}"

            if len(recent_candles) >= 3:
                min_low = min(recent_candles[-2].low, recent_candles[-3].low)
                if candle.close < min_low:
                    return True, f"close broke below A_mid and 2-bar low {min_low:.4f}"

            return False, f"Close below A_mid but no micro break yet"

    def _compute_stop(self, side: Side, anchor: AnchorInfo,
                       recent_candles: List[Candle]) -> float:
        """
        Compute stop loss price.

        LONG: min(recent swing low, A_open)
        SHORT: max(recent swing high, A_open)
        """
        if side == Side.LONG:
            swing_low = find_swing_low(recent_candles, lookback=5)
            if swing_low is not None:
                return min(swing_low, anchor.open)
            return anchor.open
        else:
            swing_high = find_swing_high(recent_candles, lookback=5)
            if swing_high is not None:
                return max(swing_high, anchor.open)
            return anchor.open

    def _check_rsi_exit(self, side: Side, current_rsi: float,
                         rsi_history: List[float]) -> Tuple[bool, str]:
        """
        Check RSI-based exit conditions.

        LONG: RSI >= 70 and turning down
        SHORT: RSI <= 30 and turning up
        """
        if len(rsi_history) < 2:
            return False, ""

        if side == Side.LONG:
            if current_rsi >= 70 and is_rsi_turning_down(rsi_history):
                return True, f"RSI exit: RSI={current_rsi:.1f} >= 70 and turning down"

            # Check bearish divergence
            if len(rsi_history) >= 5:
                highs = [c.high for c in []]  # Would need candle data
                # Simplified - just check RSI level
                pass

        else:  # SHORT
            if current_rsi <= 30 and is_rsi_turning_up(rsi_history):
                return True, f"RSI exit: RSI={current_rsi:.1f} <= 30 and turning up"

        return False, ""

    @property
    def is_in_trade(self) -> bool:
        """Check if currently in a trade."""
        return self.state == FSMState.IN_TRADE

    @property
    def current_side(self) -> Optional[Side]:
        """Get current trade side if in trade."""
        return self.ctx.trade_side if self.is_in_trade else None

    def get_state_info(self) -> Dict[str, Any]:
        """Get current state information for debugging."""
        return {
            'state': self.state.value,
            'has_anchor': self.ctx.anchor is not None,
            'candles_since_anchor': self.ctx.candles_since_anchor,
            'candles_in_trade': self.ctx.candles_in_trade,
            'cooldown_remaining': self.ctx.cooldown_remaining,
            'trade_side': self.ctx.trade_side.value if self.ctx.trade_side else None,
            'entry_price': self.ctx.entry_price,
            'stop_price': self.ctx.stop_price,
        }
