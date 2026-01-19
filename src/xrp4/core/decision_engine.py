"""DecisionEngine - Final Executor Policy.

Approves/rejects candidate signals from FSM with:
- Hard deny policy (HIGH_VOL, NO_TRADE)
- Cooldown policy
- Regime-based size scaling
- Position-aware actions
- Optional XGB approval hook

Pipeline: FSM(candidate signal) -> DecisionEngine(final action) -> Broker
"""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)
from typing import Optional, Tuple, Dict

from .types import (
    ConfirmContext,
    MarketContext,
    PositionState,
    CandidateSignal,
    Decision,
    ActionType,
)


@dataclass
class DecisionConfig:
    """DecisionEngine configuration parameters."""
    # Hard deny policy
    DENY_IN_HIGH_VOL: bool = True
    DENY_IN_NO_TRADE: bool = True
    DENY_BREAKOUT_IN_COOLDOWN: bool = True

    # Size scaling
    COOLDOWN_SIZE_MULT: float = 0.5
    TRANSITION_SIZE_MULT: float = 0.4

    # Base sizing
    BASE_SIZE_USDT: float = 100.0
    MAX_LEVERAGE: float = 3.0

    # XGB hook - Enabled 2026-01-15 (improves PnL from -$2.60 to +$32.08)
    XGB_ENABLED: bool = True
    XGB_PMIN_RANGE: float = 0.45        # Optimal threshold from training
    XGB_PMIN_TREND: float = 0.45        # Optimal threshold from training
    XGB_MODEL_PATH: str = "outputs/xgb_gate/xgb_model.json"


class DecisionEngine:
    """DecisionEngine - Final Executor Policy.

    Receives candidate signals from FSM and applies:
    1. Hard deny policy (HIGH_VOL, NO_TRADE regimes)
    2. Cooldown policy (reduced size, blocked breakouts)
    3. Regime-based size scaling
    4. Position-aware action conversion
    5. Optional XGB approval gate

    Pipeline: FSM(candidate signal) -> DecisionEngine(final action) -> Broker
    """

    def __init__(self, cfg: Optional[Dict] = None):
        """Initialize DecisionEngine with configuration.

        Args:
            cfg: Configuration dict or None for defaults.
                 Expected keys match DecisionConfig fields.
        """
        if cfg is None:
            self.cfg = DecisionConfig()
        elif isinstance(cfg, DecisionConfig):
            self.cfg = cfg
        else:
            # Extract DECISION and RISK config from dict
            decision_cfg = cfg.get("DECISION", {})
            risk_cfg = cfg.get("RISK", {})
            fsm_cfg = cfg.get("FSM", {})

            self.cfg = DecisionConfig(
                DENY_IN_HIGH_VOL=decision_cfg.get("DENY_IN_HIGH_VOL", True),
                DENY_IN_NO_TRADE=decision_cfg.get("DENY_IN_NO_TRADE", True),
                DENY_BREAKOUT_IN_COOLDOWN=decision_cfg.get("DENY_BREAKOUT_IN_COOLDOWN", True),
                COOLDOWN_SIZE_MULT=decision_cfg.get("COOLDOWN_SIZE_MULT", 0.5),
                TRANSITION_SIZE_MULT=fsm_cfg.get("TRANSITION_SIZE_MULT", 0.4),
                BASE_SIZE_USDT=risk_cfg.get("BASE_SIZE_USDT", 100.0),
                MAX_LEVERAGE=risk_cfg.get("MAX_LEVERAGE", 3.0),
                XGB_ENABLED=decision_cfg.get("XGB_ENABLED", False),
                XGB_PMIN_RANGE=decision_cfg.get("XGB_PMIN_RANGE", 0.55),
                XGB_PMIN_TREND=decision_cfg.get("XGB_PMIN_TREND", 0.60),
            )

        # XGB gate (lazy load)
        self._xgb_gate = None

    def decide(
        self,
        ctx: MarketContext,
        confirm: ConfirmContext,
        pos: PositionState,
        cand: CandidateSignal,
        engine_state: Optional[Dict] = None,
    ) -> Tuple[Decision, Dict]:
        """Make final decision on candidate signal.

        Args:
            ctx: Current market context
            confirm: Confirmation context from ConfirmLayer
            pos: Current position state
            cand: Candidate signal from FSM
            engine_state: Persistent engine state (JSON-serializable)

        Returns:
            Tuple of (Decision, updated_engine_state)
        """
        # Initialize state if needed
        if engine_state is None:
            engine_state = {
                "last_action": "NO_ACTION",
                "consecutive_denies": 0,
            }

        engine_state = engine_state.copy()

        regime = confirm.regime_confirmed
        signal = cand.signal

        # Priority 0: Hard deny policy
        deny_result = self._check_hard_deny(regime, pos, signal)
        if deny_result is not None:
            engine_state["last_action"] = deny_result.action
            engine_state["consecutive_denies"] = engine_state.get("consecutive_denies", 0) + 1
            return deny_result, engine_state

        # Priority 1: Cooldown policy
        in_cooldown = "COOLDOWN" in confirm.confirm_reason or regime == "NO_TRADE"
        if in_cooldown:
            cooldown_result = self._apply_cooldown_policy(signal)
            if cooldown_result is not None:
                engine_state["last_action"] = cooldown_result.action
                return cooldown_result, engine_state

        # Priority 2: XGB approval (if enabled)
        if self.cfg.XGB_ENABLED and self._should_check_xgb(signal):
            xgb_result = self._check_xgb_approval(ctx, regime, signal)
            if xgb_result is not None:
                engine_state["last_action"] = xgb_result.action
                return xgb_result, engine_state

        # Calculate size
        size = self._calculate_size(ctx, regime, cand, confirm, in_cooldown)

        # Convert signal to action
        decision = self._signal_to_action(pos, cand, size)

        engine_state["last_action"] = decision.action
        if decision.action != "NO_ACTION":
            engine_state["consecutive_denies"] = 0

        return decision, engine_state

    def _check_hard_deny(
        self,
        regime: str,
        pos: PositionState,
        signal: str,
    ) -> Optional[Decision]:
        """Check hard deny conditions for regime."""
        # HIGH_VOL deny
        if regime == "HIGH_VOL" and self.cfg.DENY_IN_HIGH_VOL:
            # Allow CLOSE only if in position
            if pos.side != "FLAT" and signal == "EXIT":
                return None  # Allow exit
            if signal in {"LONG_BOUNCE", "SHORT_BOUNCE", "LONG_TREND_PULLBACK",
                         "SHORT_TREND_PULLBACK", "LONG_BREAKOUT", "SHORT_BREAKOUT"}:
                return Decision(
                    action="NO_ACTION",
                    size=0.0,
                    reason="DE_DENY_HIGH_VOL",
                    meta={"blocked_signal": signal},
                )

        # NO_TRADE deny
        if regime == "NO_TRADE" and self.cfg.DENY_IN_NO_TRADE:
            # Allow CLOSE only if in position
            if pos.side != "FLAT" and signal == "EXIT":
                return None  # Allow exit
            if signal in {"LONG_BOUNCE", "SHORT_BOUNCE", "LONG_TREND_PULLBACK",
                         "SHORT_TREND_PULLBACK", "LONG_BREAKOUT", "SHORT_BREAKOUT"}:
                return Decision(
                    action="NO_ACTION",
                    size=0.0,
                    reason="DE_DENY_NO_TRADE",
                    meta={"blocked_signal": signal},
                )

        return None

    def _apply_cooldown_policy(self, signal: str) -> Optional[Decision]:
        """Apply cooldown policy restrictions."""
        # Block breakouts in cooldown
        if self.cfg.DENY_BREAKOUT_IN_COOLDOWN:
            if signal in {"LONG_BREAKOUT", "SHORT_BREAKOUT"}:
                return Decision(
                    action="NO_ACTION",
                    size=0.0,
                    reason="DE_COOLDOWN_RISK_OFF",
                    meta={"blocked_signal": signal, "blocked_type": "breakout"},
                )

        return None

    def _should_check_xgb(self, signal: str) -> bool:
        """Check if XGB approval should be checked for this signal."""
        return signal in {
            "LONG_BOUNCE", "SHORT_BOUNCE",
            "LONG_TREND_PULLBACK", "SHORT_TREND_PULLBACK",
            "LONG_BREAKOUT", "SHORT_BREAKOUT",
        }

    def _check_xgb_approval(
        self,
        ctx: MarketContext,
        regime: str,
        signal: str,
    ) -> Optional[Decision]:
        """Check XGB approval gate (if enabled)."""
        if self._xgb_gate is None:
            try:
                from .xgb_gate import XGBApprovalGate
                model_path = getattr(self.cfg, 'XGB_MODEL_PATH', 'outputs/xgb_gate/xgb_model.json')
                self._xgb_gate = XGBApprovalGate(model_path=model_path)
                logger.info(f"XGB gate loaded from {model_path}")
            except ImportError:
                logger.warning("XGB gate import failed")
                return None

        if not self._xgb_gate.is_loaded():
            return None

        # Get probability (pass signal and regime for feature extraction)
        p_win = self._xgb_gate.predict_proba(ctx.row_3m, signal, regime)

        # Determine threshold
        if regime == "RANGE":
            threshold = self.cfg.XGB_PMIN_RANGE
        else:
            threshold = self.cfg.XGB_PMIN_TREND

        if p_win < threshold:
            logger.info(f"[XGB_REJECT] {signal} p_win={p_win:.3f} < threshold={threshold:.2f}")
            return Decision(
                action="NO_ACTION",
                size=0.0,
                reason="DE_XGB_REJECT",
                meta={"p_win": p_win, "threshold": threshold, "signal": signal},
            )

        logger.debug(f"[XGB_APPROVE] {signal} p_win={p_win:.3f} >= threshold={threshold:.2f}")
        return None

    def _calculate_size(
        self,
        ctx: MarketContext,
        regime: str,
        cand: CandidateSignal,
        confirm: ConfirmContext,
        in_cooldown: bool,
    ) -> float:
        """Calculate position size based on regime and conditions."""
        base_size = self.cfg.BASE_SIZE_USDT

        # Regime-based scaling
        if regime == "TRANSITION":
            size = base_size * self.cfg.TRANSITION_SIZE_MULT
        elif regime in {"TREND_UP", "TREND_DOWN"}:
            size = base_size * 1.0
        elif regime == "RANGE":
            size = base_size * 1.0
        else:
            size = base_size * 0.5  # Unknown regime

        # Cooldown scaling
        if in_cooldown:
            size *= self.cfg.COOLDOWN_SIZE_MULT

        # Signal-specific size mult (from params)
        size_mult = cand.params.get("size_mult", 1.0)
        size *= size_mult

        # Score-based scaling (optional)
        score = cand.score
        if score > 0:
            size *= (0.5 + score * 0.5)

        return size

    def _signal_to_action(
        self,
        pos: PositionState,
        cand: CandidateSignal,
        size: float,
    ) -> Decision:
        """Convert FSM signal to DecisionEngine action."""
        signal = cand.signal

        # HOLD or no signal
        if signal == "HOLD":
            return Decision(
                action="NO_ACTION",
                size=0.0,
                reason="DE_HOLD",
                meta={},
            )

        # EXIT signal
        if signal == "EXIT":
            if pos.side == "FLAT":
                return Decision(
                    action="NO_ACTION",
                    size=0.0,
                    reason="DE_EXIT_NO_POSITION",
                    meta={},
                )
            return Decision(
                action="CLOSE",
                size=pos.size,
                reason="DE_CLOSE_FROM_EXIT_SIGNAL",
                meta={"cand_reason": cand.reason},
            )

        # LONG signals
        if signal in {"LONG_BOUNCE", "LONG_TREND_PULLBACK", "LONG_BREAKOUT"}:
            if pos.side == "FLAT":
                return Decision(
                    action="OPEN_LONG",
                    size=size,
                    reason="DE_OPEN_FROM_CANDIDATE",
                    meta={"cand_signal": signal, "cand_reason": cand.reason, "cand_score": cand.score},
                )
            elif pos.side == "SHORT":
                # Close short first (opposite signal)
                return Decision(
                    action="CLOSE",
                    size=pos.size,
                    reason="DE_CLOSE_FROM_OPPOSITE_SIGNAL",
                    meta={"cand_signal": signal},
                )
            else:
                # Already long -> HOLD
                return Decision(
                    action="NO_ACTION",
                    size=0.0,
                    reason="DE_ALREADY_LONG",
                    meta={},
                )

        # SHORT signals
        if signal in {"SHORT_BOUNCE", "SHORT_TREND_PULLBACK", "SHORT_BREAKOUT"}:
            if pos.side == "FLAT":
                return Decision(
                    action="OPEN_SHORT",
                    size=size,
                    reason="DE_OPEN_FROM_CANDIDATE",
                    meta={"cand_signal": signal, "cand_reason": cand.reason, "cand_score": cand.score},
                )
            elif pos.side == "LONG":
                # Close long first (opposite signal)
                return Decision(
                    action="CLOSE",
                    size=pos.size,
                    reason="DE_CLOSE_FROM_OPPOSITE_SIGNAL",
                    meta={"cand_signal": signal},
                )
            else:
                # Already short -> HOLD
                return Decision(
                    action="NO_ACTION",
                    size=0.0,
                    reason="DE_ALREADY_SHORT",
                    meta={},
                )

        # Unknown signal
        return Decision(
            action="NO_ACTION",
            size=0.0,
            reason="DE_UNKNOWN_SIGNAL",
            meta={"signal": signal},
        )
