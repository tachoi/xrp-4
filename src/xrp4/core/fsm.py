"""Trading FSM - Candidate Signal Generator.

Generates candidate signals based on confirmed regime from ConfirmLayer.

FSM design rules:
1. FSM MUST be regime-conditioned state machine
2. FSM input is confirmed_regime only (raw regime use forbidden)
3. Priority 0: HIGH_VOL/NO_TRADE -> emit HOLD
4. RANGE regime -> Bounce only
5. TREND_UP/DOWN regime -> Pullback only (no chase)
6. TRANSITION regime -> Breakout only (or HOLD)
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict

from .types import (
    ConfirmContext,
    MarketContext,
    PositionState,
    CandidateSignal,
    SignalType,
)


@dataclass
class FSMConfig:
    """FSM configuration parameters."""
    # Timeframes
    TF_ENTRY: str = "3m"
    TF_CONTEXT: str = "15m"

    # EMA/RSI/ATR lengths for 3m
    EMA_FAST_3M: int = 20
    EMA_SLOW_3M: int = 50
    RSI_LEN_3M: int = 14
    ATR_LEN_3M: int = 14

    # Range (bounce) params - DISABLED (losing money despite 61% WR)
    RANGE_ENABLED: bool = False                   # Disabled: avg loss >> avg win
    RANGE_PULLBACK_ATR: float = 0.6
    RANGE_MIN_ZONE_STRENGTH: float = 0.000010
    RANGE_MAX_SPREAD_ATR: float = 0.25

    # Trend (pullback) params - LONG - Tuned 2026-01-19 (relaxed for sideways market)
    TREND_PULLBACK_TO_EMA_ATR: float = 1.5      # Optimized: tighter pullback zone (was 2.0)
    TREND_MIN_EMA_SLOPE_15M: float = 0.0004     # Relaxed: was 0.0008 (blocked in sideways)
    TREND_MIN_REBOUND_RET: float = 0.0003       # Minimum rebound confirmation
    TREND_REQUIRE_EMA_ALIGNMENT: bool = True    # fast > slow for LONG
    TREND_CONFIRM_REQUIRED: bool = True
    TREND_CHASE_FORBIDDEN: bool = True

    # Trend (pullback) params - SHORT - Relaxed 2026-01-19 (sideways market)
    # Previous threshold (0.002) blocked 68% of downtrends - too strict
    SHORT_PULLBACK_TO_EMA_ATR: float = 1.5      # Optimal: wider pullback zone (was 1.0)
    SHORT_MIN_EMA_SLOPE_15M: float = 0.0004     # Relaxed: was 0.0008 (blocked in sideways)
    SHORT_MIN_REJECTION_RET: float = 0.0003     # Relaxed: was 0.0005
    SHORT_REQUIRE_EMA_ALIGNMENT: bool = True    # fast < slow for SHORT
    SHORT_ENABLED: bool = True                  # Can disable SHORT trading
    SHORT_MAX_HOLD_BARS_3M: int = 30            # Optimal: longer hold (was 20)

    # SHORT entry confirmation - DISABLED 2026-01-18 for LONG/SHORT symmetry
    # These extra conditions created asymmetry: LONG had no equivalent filters
    # In -17.4% bearish market, SHORT was still 35% vs LONG 65%
    SHORT_CONSEC_NEG_BARS: int = 0              # Disabled: 0 means skip check (was 1)
    SHORT_PEAK_CONFIRM_BARS: int = 0            # Disabled: 0 means skip check (was 2)
    SHORT_EXTRA_FILTERS_ENABLED: bool = False   # Master switch for SHORT extra filters

    # Transition limits - DISABLED (losing money)
    TRANSITION_ALLOW_BREAKOUT_ONLY: bool = False  # Disabled: TRANSITION losing money
    TRANSITION_SIZE_MULT: float = 0.4

    # Risk params (for targets) - Tuned 2026-01-16 (grid search optimized)
    STOP_ATR_MULT: float = 1.5
    TP_ATR_MULT: float = 2.0
    TRAIL_ATR_MULT: float = 1.0

    # Technical Indicator Exit params - Tuned 2026-01-19
    # Goal: Let winners run, cut losers faster with ATR stop
    EXIT_ON_EMA_CROSS: bool = False             # DISABLED: caused 98.6% losses
    EXIT_ON_RSI_EXTREME: bool = True            # Exit on RSI overbought/oversold
    RSI_OVERBOUGHT: float = 75.0                # RSI level for LONG exit
    RSI_OVERSOLD: float = 25.0                  # RSI level for SHORT exit
    EXIT_ON_REGIME_REVERSAL: bool = True        # Exit when regime reverses (TREND_UP -> TREND_DOWN)
    MIN_BARS_BEFORE_EXIT: int = 2               # Minimum bars before exit

    # ATR-based Stop Loss - Tuned 2026-01-19
    STOP_ATR_EXIT: bool = True                  # Enable ATR-based stop loss
    STOP_ATR_THRESHOLD: float = 2.0             # Exit if price moves against by 2.0 ATR (best WR)

    # Profit Protection - Analysis 2026-01-19 with 1m precision
    # Finding: 100% of losses had been profitable (+0.121% avg before -0.364% loss)
    # Note: Trailing stop simulation showed promise but 3m bar granularity too coarse
    # Result: At 3m bar level, trailing stop triggers AFTER damage already done
    # Solution: Use faster ATR TP instead (see TP_ATR_THRESHOLD)
    PROFIT_RETRACEMENT_EXIT: bool = False       # Disabled: 3m granularity too coarse
    RETRACEMENT_THRESHOLD: float = 0.40         # Not used when disabled
    MIN_PROFIT_FOR_RETRACEMENT: float = 0.0005  # Not used when disabled

    # Breakeven Stop: Move stop to entry price once profit reaches threshold
    # Testing showed mixed results - disabling for now
    BREAKEVEN_STOP_ENABLED: bool = False        # Disabled: mixed results in testing
    BREAKEVEN_TRIGGER_PCT: float = 0.15         # Trigger breakeven after 0.15% profit
    BREAKEVEN_BUFFER_ATR: float = 0.2           # Buffer above/below entry

    # 2. ATR-based Take Profit: Exit when price moves favorably by X ATR
    # DISABLED: Fixed TP cuts winners short in strong trends
    # Use trailing stop instead to let profits run
    TP_ATR_EXIT: bool = False                   # Disabled: let trailing stop manage exits
    TP_ATR_THRESHOLD: float = 2.0               # (unused when TP_ATR_EXIT=False)

    # Volatility filter - losers had 12% higher volatility
    MAX_VOLATILITY_FOR_ENTRY: float = 0.012     # Skip entry if volatility > 1.2%
    VOLATILITY_FILTER_ENABLED: bool = True      # Can disable for testing


class TradingFSM:
    """Trading FSM - Candidate Signal Generator.

    Generates candidate trading signals based on:
    - confirmed_regime from ConfirmLayer
    - Current market context (price, features, zones)
    - Current position state

    Pipeline: ConfirmLayer(confirmed_regime) -> FSM(candidate signal) -> DecisionEngine
    """

    def __init__(self, cfg: Optional[Dict] = None):
        """Initialize FSM with configuration.

        Args:
            cfg: Configuration dict or None for defaults.
                 Expected keys match FSMConfig fields.
        """
        if cfg is None:
            self.cfg = FSMConfig()
        elif isinstance(cfg, FSMConfig):
            self.cfg = cfg
        else:
            # Extract FSM config from dict
            fsm_cfg = cfg.get("FSM", {})
            risk_cfg = cfg.get("RISK", {})

            self.cfg = FSMConfig(
                TF_ENTRY=fsm_cfg.get("TF_ENTRY", "3m"),
                TF_CONTEXT=fsm_cfg.get("TF_CONTEXT", "15m"),
                EMA_FAST_3M=fsm_cfg.get("EMA_FAST_3M", 20),
                EMA_SLOW_3M=fsm_cfg.get("EMA_SLOW_3M", 50),
                RSI_LEN_3M=fsm_cfg.get("RSI_LEN_3M", 14),
                ATR_LEN_3M=fsm_cfg.get("ATR_LEN_3M", 14),
                RANGE_PULLBACK_ATR=fsm_cfg.get("RANGE_PULLBACK_ATR", 0.6),
                RANGE_MIN_ZONE_STRENGTH=fsm_cfg.get("RANGE_MIN_ZONE_STRENGTH", 0.000010),
                RANGE_MAX_SPREAD_ATR=fsm_cfg.get("RANGE_MAX_SPREAD_ATR", 0.25),
                # LONG params
                TREND_PULLBACK_TO_EMA_ATR=fsm_cfg.get("TREND_PULLBACK_TO_EMA_ATR", 2.0),
                TREND_MIN_EMA_SLOPE_15M=fsm_cfg.get("TREND_MIN_EMA_SLOPE_15M", 0.002),
                TREND_MIN_REBOUND_RET=fsm_cfg.get("TREND_MIN_REBOUND_RET", 0.0003),
                TREND_REQUIRE_EMA_ALIGNMENT=fsm_cfg.get("TREND_REQUIRE_EMA_ALIGNMENT", True),
                TREND_CONFIRM_REQUIRED=fsm_cfg.get("TREND_CONFIRM_REQUIRED", True),
                TREND_CHASE_FORBIDDEN=fsm_cfg.get("TREND_CHASE_FORBIDDEN", True),
                # SHORT params (tuned 2026-01-15)
                SHORT_PULLBACK_TO_EMA_ATR=fsm_cfg.get("SHORT_PULLBACK_TO_EMA_ATR", 1.0),
                SHORT_MIN_EMA_SLOPE_15M=fsm_cfg.get("SHORT_MIN_EMA_SLOPE_15M", 0.002),
                SHORT_MIN_REJECTION_RET=fsm_cfg.get("SHORT_MIN_REJECTION_RET", 0.0007),
                SHORT_REQUIRE_EMA_ALIGNMENT=fsm_cfg.get("SHORT_REQUIRE_EMA_ALIGNMENT", True),
                SHORT_ENABLED=fsm_cfg.get("SHORT_ENABLED", True),
                SHORT_MAX_HOLD_BARS_3M=fsm_cfg.get("SHORT_MAX_HOLD_BARS_3M", 30),
                # SHORT entry confirmation
                SHORT_CONSEC_NEG_BARS=fsm_cfg.get("SHORT_CONSEC_NEG_BARS", 2),
                SHORT_PEAK_CONFIRM_BARS=fsm_cfg.get("SHORT_PEAK_CONFIRM_BARS", 3),
                # Other
                TRANSITION_ALLOW_BREAKOUT_ONLY=fsm_cfg.get("TRANSITION_ALLOW_BREAKOUT_ONLY", True),
                TRANSITION_SIZE_MULT=fsm_cfg.get("TRANSITION_SIZE_MULT", 0.4),
                STOP_ATR_MULT=risk_cfg.get("STOP_ATR_MULT", 1.5),
                TP_ATR_MULT=risk_cfg.get("TP_ATR_MULT", 2.0),
                TRAIL_ATR_MULT=risk_cfg.get("TRAIL_ATR_MULT", 1.0),
                MAX_HOLD_BARS_3M=risk_cfg.get("MAX_HOLD_BARS_3M", 40),
            )

    def step(
        self,
        ctx: MarketContext,
        confirm: ConfirmContext,
        pos: PositionState,
        fsm_state: Optional[Dict] = None,
    ) -> Tuple[CandidateSignal, Dict]:
        """Generate candidate signal based on confirmed regime.

        Args:
            ctx: Current market context
            confirm: Confirmation context from ConfirmLayer
            pos: Current position state
            fsm_state: Persistent FSM state (JSON-serializable)

        Returns:
            Tuple of (CandidateSignal, updated_fsm_state)
        """
        # Initialize state if needed
        if fsm_state is None:
            fsm_state = {
                "last_signal": "HOLD",
                "bars_since_signal": 0,
                "recent_returns": [],  # Track recent returns for consecutive bar check
                "recent_highs": [],    # Track recent highs for peak confirmation
                "entry_regime": None,  # Track regime at entry for regime-change exit
                "best_price": None,    # Track best price for profit retracement
                "entry_price": None,   # Track entry price for ATR TP
                "breakeven_triggered": False,  # Track if breakeven stop is activated
            }

        fsm_state = fsm_state.copy()
        fsm_state["bars_since_signal"] = fsm_state.get("bars_since_signal", 0) + 1

        # Track best price for profit retracement exit
        if pos.side != "FLAT":
            current_price = ctx.price
            best_price = fsm_state.get("best_price")
            entry_price = fsm_state.get("entry_price")

            if pos.side == "LONG":
                # For LONG, track highest price (best for profit)
                if best_price is None or current_price > best_price:
                    fsm_state["best_price"] = current_price
                # Check if breakeven should be triggered
                if entry_price and not fsm_state.get("breakeven_triggered"):
                    profit_pct = (current_price - entry_price) / entry_price
                    if profit_pct >= self.cfg.BREAKEVEN_TRIGGER_PCT / 100:
                        fsm_state["breakeven_triggered"] = True
            else:  # SHORT
                # For SHORT, track lowest price (best for profit)
                if best_price is None or current_price < best_price:
                    fsm_state["best_price"] = current_price
                # Check if breakeven should be triggered
                if entry_price and not fsm_state.get("breakeven_triggered"):
                    profit_pct = (entry_price - current_price) / entry_price
                    if profit_pct >= self.cfg.BREAKEVEN_TRIGGER_PCT / 100:
                        fsm_state["breakeven_triggered"] = True

        # Track recent returns and highs for SHORT entry confirmation
        ret_3m = ctx.row_3m.get("ret_3m", ctx.row_3m.get("ret", 0))
        recent_returns = fsm_state.get("recent_returns", [])
        recent_returns.append(ret_3m)
        if len(recent_returns) > 10:  # Keep last 10 bars
            recent_returns = recent_returns[-10:]
        fsm_state["recent_returns"] = recent_returns

        recent_highs = fsm_state.get("recent_highs", [])
        recent_highs.append(ctx.row_3m.get("high", ctx.price))
        if len(recent_highs) > 10:
            recent_highs = recent_highs[-10:]
        fsm_state["recent_highs"] = recent_highs

        regime = confirm.regime_confirmed

        # Priority 0: Hard block regimes
        if regime in {"HIGH_VOL", "NO_TRADE"}:
            signal = CandidateSignal(
                signal="HOLD",
                score=0.0,
                reason="FSM_BLOCKED_REGIME",
                params={},
            )
            return signal, fsm_state

        # Get market data
        price = ctx.price
        atr_3m = ctx.row_3m.get("atr_3m", ctx.row_3m.get("atr", 0.01))
        ema_fast = ctx.row_3m.get("ema_fast_3m", ctx.row_3m.get("ema_20", price))
        ema_slow = ctx.row_3m.get("ema_slow_3m", ctx.row_3m.get("ema_50", price))
        ret_3m = ctx.row_3m.get("ret_3m", ctx.row_3m.get("ret", 0))
        ema_slope_15m = ctx.row_15m.get("ema_slope_15m", ctx.row_15m.get("ema_slope", 0))

        # Zone data
        support = ctx.zone.get("support", price - atr_3m * 2)
        resistance = ctx.zone.get("resistance", price + atr_3m * 2)
        zone_strength = ctx.zone.get("strength", 0)
        dist_to_support = ctx.zone.get("dist_to_support", (price - support) / atr_3m if atr_3m > 0 else 999)
        dist_to_resistance = ctx.zone.get("dist_to_resistance", (resistance - price) / atr_3m if atr_3m > 0 else 999)

        # Default params for targets
        default_params = {
            "stop_atr_mult": self.cfg.STOP_ATR_MULT,
            "tp_atr_mult": self.cfg.TP_ATR_MULT,
            "trail_atr_mult": self.cfg.TRAIL_ATR_MULT,
            "atr_3m": atr_3m,
        }

        # Check exit conditions first if in position
        if pos.side != "FLAT":
            exit_signal = self._check_exit(ctx, confirm, pos, atr_3m, ema_slow, fsm_state)
            if exit_signal is not None:
                fsm_state["last_signal"] = exit_signal.signal
                fsm_state["bars_since_signal"] = 0
                fsm_state["entry_regime"] = None  # Reset entry regime on exit
                fsm_state["best_price"] = None    # Reset profit tracking
                fsm_state["entry_price"] = None   # Reset entry price
                return exit_signal, fsm_state

        # Generate entry signals based on regime
        if regime == "RANGE":
            signal = self._generate_range_signal(
                ctx, pos, price, atr_3m, dist_to_support, dist_to_resistance,
                zone_strength, default_params
            )
        elif regime == "TREND_UP":
            signal = self._generate_trend_up_signal(
                ctx, pos, price, atr_3m, ema_fast, ema_slow, ret_3m, ema_slope_15m, default_params
            )
        elif regime == "TREND_DOWN":
            signal = self._generate_trend_down_signal(
                ctx, pos, price, atr_3m, ema_fast, ema_slow, ret_3m, ema_slope_15m, default_params, fsm_state
            )
        elif regime == "TRANSITION":
            signal = self._generate_transition_signal(
                ctx, confirm, pos, price, atr_3m, default_params
            )
        else:
            # Unknown regime -> HOLD
            signal = CandidateSignal(
                signal="HOLD",
                score=0.0,
                reason="FSM_UNKNOWN_REGIME",
                params={},
            )

        fsm_state["last_signal"] = signal.signal
        if signal.signal != "HOLD":
            fsm_state["bars_since_signal"] = 0
            # Store entry regime and price for exit logic
            if signal.signal not in {"EXIT", "HOLD"}:
                fsm_state["entry_regime"] = regime
                fsm_state["entry_price"] = price  # Store entry price for ATR TP
                fsm_state["best_price"] = price   # Initialize best price
                fsm_state["breakeven_triggered"] = False  # Reset breakeven flag

        return signal, fsm_state

    def _check_exit(
        self,
        ctx: MarketContext,
        confirm: ConfirmContext,
        pos: PositionState,
        atr_3m: float,
        ema_slow: float,
        fsm_state: Optional[Dict] = None,
    ) -> Optional[CandidateSignal]:
        """Check exit conditions for open position."""
        price = ctx.price
        entry_regime = fsm_state.get("entry_regime") if fsm_state else None

        # Technical Indicator Based Exit - Refactored 2026-01-18
        # No timeout or trailing stop - only exit on technical signals

        # Get technical indicators
        rsi = ctx.row_3m.get("rsi", ctx.row_3m.get("rsi_14", 50))
        current_regime = confirm.regime_confirmed

        # Minimum bars check - prevent premature exits
        min_bars_met = pos.bars_held_3m >= self.cfg.MIN_BARS_BEFORE_EXIT

        if not min_bars_met:
            return None  # Too early to exit

        # Exit Condition 1: Regime Reversal
        # LONG entered on TREND_UP -> exit if regime changes to TREND_DOWN
        # SHORT entered on TREND_DOWN -> exit if regime changes to TREND_UP
        if self.cfg.EXIT_ON_REGIME_REVERSAL and entry_regime is not None:
            regime_reversed = False

            if pos.side == "LONG" and entry_regime == "TREND_UP":
                if current_regime == "TREND_DOWN":
                    regime_reversed = True
            elif pos.side == "SHORT" and entry_regime == "TREND_DOWN":
                if current_regime == "TREND_UP":
                    regime_reversed = True

            if regime_reversed:
                return CandidateSignal(
                    signal="EXIT",
                    score=0.95,
                    reason="REGIME_REVERSAL_EXIT",
                    params={
                        "exit_type": "regime_reversal",
                        "entry_regime": entry_regime,
                        "current_regime": current_regime,
                    },
                )

        # Exit Condition 2: RSI Extreme
        # LONG -> exit on RSI overbought (> 70)
        # SHORT -> exit on RSI oversold (< 30)
        if self.cfg.EXIT_ON_RSI_EXTREME:
            if pos.side == "LONG" and rsi >= self.cfg.RSI_OVERBOUGHT:
                return CandidateSignal(
                    signal="EXIT",
                    score=0.85,
                    reason="RSI_OVERBOUGHT_EXIT",
                    params={
                        "exit_type": "rsi_extreme",
                        "rsi": rsi,
                        "threshold": self.cfg.RSI_OVERBOUGHT,
                    },
                )
            elif pos.side == "SHORT" and rsi <= self.cfg.RSI_OVERSOLD:
                return CandidateSignal(
                    signal="EXIT",
                    score=0.85,
                    reason="RSI_OVERSOLD_EXIT",
                    params={
                        "exit_type": "rsi_extreme",
                        "rsi": rsi,
                        "threshold": self.cfg.RSI_OVERSOLD,
                    },
                )

        # Exit Condition 3: ATR-based Take Profit (Aggressive profit-taking)
        # Exit when price moves favorably by X ATR from entry
        if self.cfg.TP_ATR_EXIT and fsm_state:
            entry_price = fsm_state.get("entry_price")
            if entry_price is not None and atr_3m > 0:
                tp_distance = self.cfg.TP_ATR_THRESHOLD * atr_3m

                if pos.side == "LONG":
                    # LONG: take profit when price > entry + X ATR
                    if price >= entry_price + tp_distance:
                        profit_pct = (price - entry_price) / entry_price * 100
                        return CandidateSignal(
                            signal="EXIT",
                            score=0.95,
                            reason="ATR_TP_EXIT",
                            params={
                                "exit_type": "atr_take_profit",
                                "entry_price": entry_price,
                                "price": price,
                                "atr_move": (price - entry_price) / atr_3m,
                                "profit_pct": profit_pct,
                            },
                        )
                elif pos.side == "SHORT":
                    # SHORT: take profit when price < entry - X ATR
                    if price <= entry_price - tp_distance:
                        profit_pct = (entry_price - price) / entry_price * 100
                        return CandidateSignal(
                            signal="EXIT",
                            score=0.95,
                            reason="ATR_TP_EXIT",
                            params={
                                "exit_type": "atr_take_profit",
                                "entry_price": entry_price,
                                "price": price,
                                "atr_move": (entry_price - price) / atr_3m,
                                "profit_pct": profit_pct,
                            },
                        )

        # Exit Condition 4: Profit Retracement (Protect profits)
        # Exit when profit drops X% from peak
        if self.cfg.PROFIT_RETRACEMENT_EXIT and fsm_state:
            entry_price = fsm_state.get("entry_price")
            best_price = fsm_state.get("best_price")

            if entry_price is not None and best_price is not None:
                if pos.side == "LONG":
                    # Calculate max profit and current profit for LONG
                    max_profit_pct = (best_price - entry_price) / entry_price
                    current_profit_pct = (price - entry_price) / entry_price

                    # Only trigger if we had meaningful profit
                    if max_profit_pct >= self.cfg.MIN_PROFIT_FOR_RETRACEMENT:
                        # Check if profit retraced by threshold
                        if max_profit_pct > 0:
                            retracement = 1 - (current_profit_pct / max_profit_pct)
                            if retracement >= self.cfg.RETRACEMENT_THRESHOLD:
                                return CandidateSignal(
                                    signal="EXIT",
                                    score=0.90,
                                    reason="PROFIT_RETRACEMENT_EXIT",
                                    params={
                                        "exit_type": "profit_retracement",
                                        "max_profit_pct": max_profit_pct * 100,
                                        "current_profit_pct": current_profit_pct * 100,
                                        "retracement_pct": retracement * 100,
                                        "best_price": best_price,
                                    },
                                )

                elif pos.side == "SHORT":
                    # Calculate max profit and current profit for SHORT
                    max_profit_pct = (entry_price - best_price) / entry_price
                    current_profit_pct = (entry_price - price) / entry_price

                    # Only trigger if we had meaningful profit
                    if max_profit_pct >= self.cfg.MIN_PROFIT_FOR_RETRACEMENT:
                        # Check if profit retraced by threshold
                        if max_profit_pct > 0:
                            retracement = 1 - (current_profit_pct / max_profit_pct)
                            if retracement >= self.cfg.RETRACEMENT_THRESHOLD:
                                return CandidateSignal(
                                    signal="EXIT",
                                    score=0.90,
                                    reason="PROFIT_RETRACEMENT_EXIT",
                                    params={
                                        "exit_type": "profit_retracement",
                                        "max_profit_pct": max_profit_pct * 100,
                                        "current_profit_pct": current_profit_pct * 100,
                                        "retracement_pct": retracement * 100,
                                        "best_price": best_price,
                                    },
                                )

        # Exit Condition 5: Breakeven Stop (protect profits)
        # Once trade reaches profit threshold, move stop to entry price
        if self.cfg.BREAKEVEN_STOP_ENABLED and fsm_state:
            entry_price = fsm_state.get("entry_price")
            breakeven_triggered = fsm_state.get("breakeven_triggered", False)

            if entry_price is not None and breakeven_triggered and atr_3m > 0:
                buffer = self.cfg.BREAKEVEN_BUFFER_ATR * atr_3m

                if pos.side == "LONG":
                    # LONG: exit if price drops back to entry (minus small buffer)
                    breakeven_level = entry_price - buffer
                    if price <= breakeven_level:
                        return CandidateSignal(
                            signal="EXIT",
                            score=0.88,
                            reason="BREAKEVEN_STOP_EXIT",
                            params={
                                "exit_type": "breakeven_stop",
                                "entry_price": entry_price,
                                "breakeven_level": breakeven_level,
                                "price": price,
                            },
                        )
                elif pos.side == "SHORT":
                    # SHORT: exit if price rises back to entry (plus small buffer)
                    breakeven_level = entry_price + buffer
                    if price >= breakeven_level:
                        return CandidateSignal(
                            signal="EXIT",
                            score=0.88,
                            reason="BREAKEVEN_STOP_EXIT",
                            params={
                                "exit_type": "breakeven_stop",
                                "entry_price": entry_price,
                                "breakeven_level": breakeven_level,
                                "price": price,
                            },
                        )

        # Exit Condition 6: ATR-based Stop Loss (replaces EMA Cross)
        # Exit if price moves against position by X ATR from entry
        if self.cfg.STOP_ATR_EXIT and fsm_state:
            entry_price = fsm_state.get("entry_price")
            if entry_price is not None and atr_3m > 0:
                stop_distance = self.cfg.STOP_ATR_THRESHOLD * atr_3m

                if pos.side == "LONG":
                    # LONG: stop loss when price < entry - X ATR
                    if price <= entry_price - stop_distance:
                        loss_pct = (price - entry_price) / entry_price * 100
                        return CandidateSignal(
                            signal="EXIT",
                            score=0.85,
                            reason="ATR_STOP_EXIT",
                            params={
                                "exit_type": "atr_stop_loss",
                                "entry_price": entry_price,
                                "price": price,
                                "atr_move": (entry_price - price) / atr_3m,
                                "loss_pct": loss_pct,
                            },
                        )
                elif pos.side == "SHORT":
                    # SHORT: stop loss when price > entry + X ATR
                    if price >= entry_price + stop_distance:
                        loss_pct = (entry_price - price) / entry_price * 100
                        return CandidateSignal(
                            signal="EXIT",
                            score=0.85,
                            reason="ATR_STOP_EXIT",
                            params={
                                "exit_type": "atr_stop_loss",
                                "entry_price": entry_price,
                                "price": price,
                                "atr_move": (price - entry_price) / atr_3m,
                                "loss_pct": loss_pct,
                            },
                        )

        # Exit Condition 6: EMA Cross (DISABLED - was causing 98.6% losses)
        # LONG -> exit if price < EMA slow (trend weakening)
        # SHORT -> exit if price > EMA slow (trend weakening)
        if self.cfg.EXIT_ON_EMA_CROSS:
            if pos.side == "LONG" and price < ema_slow:
                return CandidateSignal(
                    signal="EXIT",
                    score=0.80,
                    reason="EMA_CROSS_EXIT",
                    params={
                        "exit_type": "ema_cross",
                        "price": price,
                        "ema_slow": ema_slow,
                    },
                )
            elif pos.side == "SHORT" and price > ema_slow:
                return CandidateSignal(
                    signal="EXIT",
                    score=0.80,
                    reason="EMA_CROSS_EXIT",
                    params={
                        "exit_type": "ema_cross",
                        "price": price,
                        "ema_slow": ema_slow,
                    },
                )

        return None

    def _generate_range_signal(
        self,
        ctx: MarketContext,
        pos: PositionState,
        price: float,
        atr_3m: float,
        dist_to_support: float,
        dist_to_resistance: float,
        zone_strength: float,
        default_params: Dict,
    ) -> CandidateSignal:
        """Generate RANGE regime signal (bounce only)."""
        # Check if RANGE trading is enabled
        if not self.cfg.RANGE_ENABLED:
            return CandidateSignal(
                signal="HOLD",
                score=0.0,
                reason="RANGE_DISABLED",
                params={},
            )

        # Already in position -> HOLD
        if pos.side != "FLAT":
            return CandidateSignal(
                signal="HOLD",
                score=0.0,
                reason="RANGE_IN_POSITION",
                params={},
            )

        # Check zone strength
        if zone_strength < self.cfg.RANGE_MIN_ZONE_STRENGTH:
            return CandidateSignal(
                signal="HOLD",
                score=0.0,
                reason="RANGE_WEAK_ZONE",
                params={"zone_strength": zone_strength},
            )

        # LONG_BOUNCE: near support
        near_support = dist_to_support <= self.cfg.RANGE_PULLBACK_ATR
        near_resistance = dist_to_resistance <= self.cfg.RANGE_PULLBACK_ATR

        if near_support and near_resistance:
            # Tight box - choose higher strength side or HOLD
            return CandidateSignal(
                signal="HOLD",
                score=0.0,
                reason="RANGE_TIGHT_BOX",
                params={},
            )

        if near_support:
            score = min(1.0, 0.5 + zone_strength * 10000)
            params = default_params.copy()
            params["entry_zone"] = "support"
            params["dist_to_support"] = dist_to_support

            return CandidateSignal(
                signal="LONG_BOUNCE",
                score=score,
                reason="RANGE_LONG_BOUNCE_NEAR_SUPPORT",
                params=params,
            )

        if near_resistance:
            score = min(1.0, 0.5 + zone_strength * 10000)
            params = default_params.copy()
            params["entry_zone"] = "resistance"
            params["dist_to_resistance"] = dist_to_resistance

            return CandidateSignal(
                signal="SHORT_BOUNCE",
                score=score,
                reason="RANGE_SHORT_BOUNCE_NEAR_RESIST",
                params=params,
            )

        return CandidateSignal(
            signal="HOLD",
            score=0.0,
            reason="RANGE_NO_ENTRY_ZONE",
            params={},
        )

    def _generate_trend_up_signal(
        self,
        ctx: MarketContext,
        pos: PositionState,
        price: float,
        atr_3m: float,
        ema_fast: float,
        ema_slow: float,
        ret_3m: float,
        ema_slope_15m: float,
        default_params: Dict,
    ) -> CandidateSignal:
        """Generate TREND_UP regime signal (pullback only, no chase)."""
        # Already in position -> HOLD
        if pos.side != "FLAT":
            return CandidateSignal(
                signal="HOLD",
                score=0.0,
                reason="TREND_UP_IN_POSITION",
                params={},
            )

        # Check EMA slope
        if abs(ema_slope_15m) < self.cfg.TREND_MIN_EMA_SLOPE_15M:
            return CandidateSignal(
                signal="HOLD",
                score=0.0,
                reason="TREND_UP_WEAK_SLOPE",
                params={"ema_slope_15m": ema_slope_15m},
            )

        # Check EMA alignment (fast > slow for uptrend)
        if self.cfg.TREND_REQUIRE_EMA_ALIGNMENT and ema_fast <= ema_slow:
            return CandidateSignal(
                signal="HOLD",
                score=0.0,
                reason="TREND_UP_EMA_NOT_ALIGNED",
                params={"ema_fast": ema_fast, "ema_slow": ema_slow},
            )

        # Volatility filter - Added 2026-01-15 (loss analysis)
        # Losing trades had 12% higher volatility than winners
        if self.cfg.VOLATILITY_FILTER_ENABLED:
            volatility = ctx.row_3m.get("volatility", ctx.row_3m.get("vol", 0))
            if volatility > self.cfg.MAX_VOLATILITY_FOR_ENTRY:
                return CandidateSignal(
                    signal="HOLD",
                    score=0.0,
                    reason="TREND_UP_HIGH_VOLATILITY",
                    params={"volatility": volatility, "max": self.cfg.MAX_VOLATILITY_FOR_ENTRY},
                )

        # Check pullback condition (no chase)
        if self.cfg.TREND_CHASE_FORBIDDEN:
            pullback_threshold = ema_fast + self.cfg.TREND_PULLBACK_TO_EMA_ATR * atr_3m
            is_pullback = price <= pullback_threshold
            is_rebound = ret_3m > self.cfg.TREND_MIN_REBOUND_RET

            if not is_pullback:
                return CandidateSignal(
                    signal="HOLD",
                    score=0.0,
                    reason="TREND_UP_NO_PULLBACK",
                    params={"price": price, "threshold": pullback_threshold},
                )

            if not is_rebound:
                return CandidateSignal(
                    signal="HOLD",
                    score=0.0,
                    reason="TREND_UP_NO_REBOUND",
                    params={"ret_3m": ret_3m, "min_required": self.cfg.TREND_MIN_REBOUND_RET},
                )

        # Generate signal (adjusted score multiplier for lower slope values)
        score = min(1.0, 0.5 + abs(ema_slope_15m) * 50)
        params = default_params.copy()
        params["ema_slope_15m"] = ema_slope_15m
        params["ret_3m"] = ret_3m

        return CandidateSignal(
            signal="LONG_TREND_PULLBACK",
            score=score,
            reason="TREND_UP_PULLBACK_ENTRY",
            params=params,
        )

    def _generate_trend_down_signal(
        self,
        ctx: MarketContext,
        pos: PositionState,
        price: float,
        atr_3m: float,
        ema_fast: float,
        ema_slow: float,
        ret_3m: float,
        ema_slope_15m: float,
        default_params: Dict,
        fsm_state: Dict = None,
    ) -> CandidateSignal:
        """Generate TREND_DOWN regime signal (pullback only, no chase).

        Uses SHORT-specific parameters with STRONGER entry confirmation:
        1. Consecutive negative bars required
        2. Price below recent peak (bounce exhaustion)
        """
        # Check if SHORT trading is enabled
        if not self.cfg.SHORT_ENABLED:
            return CandidateSignal(
                signal="HOLD",
                score=0.0,
                reason="SHORT_DISABLED",
                params={},
            )

        # Already in position -> HOLD
        if pos.side != "FLAT":
            return CandidateSignal(
                signal="HOLD",
                score=0.0,
                reason="TREND_DOWN_IN_POSITION",
                params={},
            )

        # Check EMA slope (use SHORT-specific threshold)
        if abs(ema_slope_15m) < self.cfg.SHORT_MIN_EMA_SLOPE_15M:
            return CandidateSignal(
                signal="HOLD",
                score=0.0,
                reason="TREND_DOWN_WEAK_SLOPE",
                params={"ema_slope_15m": ema_slope_15m, "threshold": self.cfg.SHORT_MIN_EMA_SLOPE_15M},
            )

        # Check EMA alignment (fast < slow for downtrend)
        if self.cfg.SHORT_REQUIRE_EMA_ALIGNMENT and ema_fast >= ema_slow:
            return CandidateSignal(
                signal="HOLD",
                score=0.0,
                reason="TREND_DOWN_EMA_NOT_ALIGNED",
                params={"ema_fast": ema_fast, "ema_slow": ema_slow},
            )

        # Volatility filter - Added 2026-01-15 (loss analysis)
        # Losing trades had 12% higher volatility than winners
        if self.cfg.VOLATILITY_FILTER_ENABLED:
            volatility = ctx.row_3m.get("volatility", ctx.row_3m.get("vol", 0))
            if volatility > self.cfg.MAX_VOLATILITY_FOR_ENTRY:
                return CandidateSignal(
                    signal="HOLD",
                    score=0.0,
                    reason="TREND_DOWN_HIGH_VOLATILITY",
                    params={"volatility": volatility, "max": self.cfg.MAX_VOLATILITY_FOR_ENTRY},
                )

        # Check pullback condition (no chase) - use SHORT-specific params
        if self.cfg.TREND_CHASE_FORBIDDEN:
            pullback_threshold = ema_fast - self.cfg.SHORT_PULLBACK_TO_EMA_ATR * atr_3m
            is_pullback = price >= pullback_threshold
            is_rejection = ret_3m < -self.cfg.SHORT_MIN_REJECTION_RET

            if not is_pullback:
                return CandidateSignal(
                    signal="HOLD",
                    score=0.0,
                    reason="TREND_DOWN_NO_PULLBACK",
                    params={"price": price, "threshold": pullback_threshold},
                )

            if not is_rejection:
                return CandidateSignal(
                    signal="HOLD",
                    score=0.0,
                    reason="TREND_DOWN_NO_REJECTION",
                    params={"ret_3m": ret_3m, "min_required": -self.cfg.SHORT_MIN_REJECTION_RET},
                )

        # === EXTRA ENTRY CONFIRMATION (OPTIONAL) ===
        # These filters can be enabled for more conservative SHORT entries
        # But they create asymmetry with LONG (which has no equivalent filters)
        # Disabled by default (SHORT_EXTRA_FILTERS_ENABLED = False)

        if self.cfg.SHORT_EXTRA_FILTERS_ENABLED and fsm_state is not None:
            # 1. Check consecutive negative bars (bounce peak confirmation)
            n_consec_required = self.cfg.SHORT_CONSEC_NEG_BARS
            if n_consec_required > 0:
                recent_returns = fsm_state.get("recent_returns", [])

                if len(recent_returns) >= n_consec_required:
                    last_n = recent_returns[-n_consec_required:]
                    consec_neg = all(r < 0 for r in last_n)

                    if not consec_neg:
                        return CandidateSignal(
                            signal="HOLD",
                            score=0.0,
                            reason="TREND_DOWN_NO_CONSEC_NEG",
                            params={"required": n_consec_required, "recent": last_n},
                        )
                else:
                    # Not enough history yet
                    return CandidateSignal(
                        signal="HOLD",
                        score=0.0,
                        reason="TREND_DOWN_INSUFFICIENT_HISTORY",
                        params={},
                    )

            # 2. Check peak confirmation (price below recent high)
            peak_bars = self.cfg.SHORT_PEAK_CONFIRM_BARS
            if peak_bars > 0:
                recent_highs = fsm_state.get("recent_highs", [])

                if len(recent_highs) >= peak_bars:
                    recent_peak = max(recent_highs[-peak_bars:])
                    # Price should be below recent peak (bounce is exhausted)
                    if price >= recent_peak:
                        return CandidateSignal(
                            signal="HOLD",
                            score=0.0,
                            reason="TREND_DOWN_AT_PEAK",
                            params={"price": price, "recent_peak": recent_peak},
                        )

        # Generate signal (adjusted score multiplier for lower slope values)
        score = min(1.0, 0.5 + abs(ema_slope_15m) * 50)
        params = default_params.copy()
        params["ema_slope_15m"] = ema_slope_15m
        params["ret_3m"] = ret_3m

        return CandidateSignal(
            signal="SHORT_TREND_PULLBACK",
            score=score,
            reason="TREND_DOWN_PULLBACK_ENTRY",
            params=params,
        )

    def _generate_transition_signal(
        self,
        ctx: MarketContext,
        confirm: ConfirmContext,
        pos: PositionState,
        price: float,
        atr_3m: float,
        default_params: Dict,
    ) -> CandidateSignal:
        """Generate TRANSITION regime signal (breakout only or HOLD)."""
        # Already in position -> HOLD
        if pos.side != "FLAT":
            return CandidateSignal(
                signal="HOLD",
                score=0.0,
                reason="TRANSITION_IN_POSITION",
                params={},
            )

        if not self.cfg.TRANSITION_ALLOW_BREAKOUT_ONLY:
            return CandidateSignal(
                signal="HOLD",
                score=0.0,
                reason="TRANSITION_BLOCK",
                params={},
            )

        # Check breakout magnitude from confirm_metrics
        b_up = confirm.confirm_metrics.get("B_up", 0)
        b_dn = confirm.confirm_metrics.get("B_dn", 0)
        breakout_threshold = 1.0

        if b_up >= breakout_threshold:
            score = min(1.0, 0.5 + b_up * 0.2)
            params = default_params.copy()
            params["B_up"] = b_up
            params["size_mult"] = self.cfg.TRANSITION_SIZE_MULT

            return CandidateSignal(
                signal="LONG_BREAKOUT",
                score=score,
                reason="TRANSITION_BREAKOUT_ONLY_PASS",
                params=params,
            )

        if b_dn >= breakout_threshold:
            score = min(1.0, 0.5 + b_dn * 0.2)
            params = default_params.copy()
            params["B_dn"] = b_dn
            params["size_mult"] = self.cfg.TRANSITION_SIZE_MULT

            return CandidateSignal(
                signal="SHORT_BREAKOUT",
                score=score,
                reason="TRANSITION_BREAKOUT_ONLY_PASS",
                params=params,
            )

        return CandidateSignal(
            signal="HOLD",
            score=0.0,
            reason="TRANSITION_NO_BREAKOUT",
            params={"B_up": b_up, "B_dn": b_dn},
        )
