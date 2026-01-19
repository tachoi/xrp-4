"""Unit tests for FSM and DecisionEngine.

Tests cover:
1. HIGH_VOL -> FSM emits HOLD and DecisionEngine denies opens
2. TRANSITION without breakout metrics -> HOLD
3. TREND_UP requires pullback (no chase)
4. RANGE near support with strong zone -> OPEN_LONG (bounce)
5. Cooldown blocks breakout but allows exit/hold
"""

import pytest
from xrp4.core.types import (
    ConfirmContext,
    MarketContext,
    PositionState,
    CandidateSignal,
    Decision,
)
from xrp4.core.fsm import TradingFSM, FSMConfig
from xrp4.core.decision_engine import DecisionEngine, DecisionConfig


@pytest.fixture
def fsm():
    """Create FSM with default config."""
    return TradingFSM()


@pytest.fixture
def decision_engine():
    """Create DecisionEngine with default config."""
    return DecisionEngine()


@pytest.fixture
def base_market_ctx():
    """Create base market context."""
    return MarketContext(
        symbol="XRPUSDT",
        ts=1700000000000,
        price=0.50,
        row_3m={
            "atr_3m": 0.01,
            "ema_fast_3m": 0.50,
            "ema_slow_3m": 0.49,
            "ret_3m": 0.001,
        },
        row_15m={
            "ema_slope_15m": 0.08,
        },
        zone={
            "support": 0.48,
            "resistance": 0.52,
            "strength": 0.0001,
            "dist_to_support": 2.0,
            "dist_to_resistance": 2.0,
        },
    )


@pytest.fixture
def flat_position():
    """Create flat position."""
    return PositionState(side="FLAT")


@pytest.fixture
def long_position():
    """Create long position."""
    return PositionState(
        side="LONG",
        entry_price=0.49,
        size=100.0,
        entry_ts=1699990000000,
        bars_held_3m=10,
        unrealized_pnl=1.0,
    )


class TestFSMHighVol:
    """Test FSM behavior in HIGH_VOL regime."""

    def test_high_vol_emits_hold(self, fsm, base_market_ctx, flat_position):
        """HIGH_VOL -> FSM emits HOLD."""
        confirm = ConfirmContext(
            regime_raw="HIGH_VOL",
            regime_confirmed="HIGH_VOL",
            confirm_reason="HIGH_VOL_ACTIVE",
            confirm_metrics={"V": 0.02},
        )

        signal, state = fsm.step(base_market_ctx, confirm, flat_position)

        assert signal.signal == "HOLD"
        assert signal.reason == "FSM_BLOCKED_REGIME"

    def test_decision_denies_open_in_high_vol(
        self, fsm, decision_engine, base_market_ctx, flat_position
    ):
        """DecisionEngine denies opens in HIGH_VOL."""
        confirm = ConfirmContext(
            regime_raw="HIGH_VOL",
            regime_confirmed="HIGH_VOL",
            confirm_reason="HIGH_VOL_ACTIVE",
            confirm_metrics={},
        )

        # Even if FSM somehow emits a signal
        fake_signal = CandidateSignal(
            signal="LONG_BOUNCE",
            score=0.7,
            reason="TEST",
            params={},
        )

        decision, state = decision_engine.decide(
            base_market_ctx, confirm, flat_position, fake_signal
        )

        assert decision.action == "NO_ACTION"
        assert decision.reason == "DE_DENY_HIGH_VOL"


class TestFSMNoTrade:
    """Test FSM behavior in NO_TRADE regime."""

    def test_no_trade_emits_hold(self, fsm, base_market_ctx, flat_position):
        """NO_TRADE -> FSM emits HOLD."""
        confirm = ConfirmContext(
            regime_raw="HIGH_VOL",
            regime_confirmed="NO_TRADE",
            confirm_reason="HIGH_VOL_COOLDOWN",
            confirm_metrics={},
        )

        signal, state = fsm.step(base_market_ctx, confirm, flat_position)

        assert signal.signal == "HOLD"
        assert signal.reason == "FSM_BLOCKED_REGIME"


class TestFSMTransition:
    """Test FSM behavior in TRANSITION regime."""

    def test_transition_without_breakout_holds(
        self, fsm, base_market_ctx, flat_position
    ):
        """TRANSITION without breakout metrics -> HOLD."""
        confirm = ConfirmContext(
            regime_raw="TRANSITION",
            regime_confirmed="TRANSITION",
            confirm_reason="TRANSITION_NOT_CONFIRMED",
            confirm_metrics={"B_up": 0.5, "B_dn": 0.3},  # Below threshold
        )

        signal, state = fsm.step(base_market_ctx, confirm, flat_position)

        assert signal.signal == "HOLD"
        assert "TRANSITION" in signal.reason

    def test_transition_with_breakout_signals(
        self, fsm, base_market_ctx, flat_position
    ):
        """TRANSITION with strong breakout -> LONG_BREAKOUT."""
        confirm = ConfirmContext(
            regime_raw="TRANSITION",
            regime_confirmed="TRANSITION",
            confirm_reason="TRANSITION_NOT_CONFIRMED",
            confirm_metrics={"B_up": 1.5, "B_dn": 0.3},  # Above threshold
        )

        signal, state = fsm.step(base_market_ctx, confirm, flat_position)

        assert signal.signal == "LONG_BREAKOUT"
        assert signal.reason == "TRANSITION_BREAKOUT_ONLY_PASS"


class TestFSMTrendUp:
    """Test FSM behavior in TREND_UP regime."""

    def test_trend_up_no_pullback_holds(self, fsm, flat_position):
        """TREND_UP without pullback -> HOLD (no chase)."""
        ctx = MarketContext(
            symbol="XRPUSDT",
            ts=1700000000000,
            price=0.55,  # Above EMA + threshold
            row_3m={
                "atr_3m": 0.01,
                "ema_fast_3m": 0.50,
                "ema_slow_3m": 0.49,
                "ret_3m": 0.002,
            },
            row_15m={"ema_slope_15m": 0.10},
            zone={},
        )

        confirm = ConfirmContext(
            regime_raw="TREND_UP",
            regime_confirmed="TREND_UP",
            confirm_reason="TREND_UP_PASSTHROUGH",
            confirm_metrics={},
        )

        signal, state = fsm.step(ctx, confirm, flat_position)

        assert signal.signal == "HOLD"
        assert "NO_PULLBACK" in signal.reason

    def test_trend_up_with_pullback_signals(self, fsm, flat_position):
        """TREND_UP with pullback + rebound -> LONG_TREND_PULLBACK."""
        ctx = MarketContext(
            symbol="XRPUSDT",
            ts=1700000000000,
            price=0.502,  # Near EMA (pullback)
            row_3m={
                "atr_3m": 0.01,
                "ema_fast_3m": 0.50,
                "ema_slow_3m": 0.49,
                "ret_3m": 0.001,  # Positive (rebound)
            },
            row_15m={"ema_slope_15m": 0.10},
            zone={},
        )

        confirm = ConfirmContext(
            regime_raw="TREND_UP",
            regime_confirmed="TREND_UP",
            confirm_reason="TREND_UP_PASSTHROUGH",
            confirm_metrics={},
        )

        signal, state = fsm.step(ctx, confirm, flat_position)

        assert signal.signal == "LONG_TREND_PULLBACK"
        assert signal.reason == "TREND_UP_PULLBACK_ENTRY"

    def test_trend_up_no_rebound_holds(self, fsm, flat_position):
        """TREND_UP pullback but no rebound -> HOLD."""
        ctx = MarketContext(
            symbol="XRPUSDT",
            ts=1700000000000,
            price=0.502,  # Near EMA (pullback)
            row_3m={
                "atr_3m": 0.01,
                "ema_fast_3m": 0.50,
                "ema_slow_3m": 0.49,
                "ret_3m": -0.001,  # Negative (no rebound)
            },
            row_15m={"ema_slope_15m": 0.10},
            zone={},
        )

        confirm = ConfirmContext(
            regime_raw="TREND_UP",
            regime_confirmed="TREND_UP",
            confirm_reason="TREND_UP_PASSTHROUGH",
            confirm_metrics={},
        )

        signal, state = fsm.step(ctx, confirm, flat_position)

        assert signal.signal == "HOLD"
        assert "NO_REBOUND" in signal.reason


class TestFSMRange:
    """Test FSM behavior in RANGE regime."""

    def test_range_near_support_signals_long(self, fsm, flat_position):
        """RANGE near support with strong zone -> LONG_BOUNCE."""
        ctx = MarketContext(
            symbol="XRPUSDT",
            ts=1700000000000,
            price=0.485,
            row_3m={"atr_3m": 0.01},
            row_15m={},
            zone={
                "support": 0.48,
                "resistance": 0.52,
                "strength": 0.0001,
                "dist_to_support": 0.5,  # Within threshold
                "dist_to_resistance": 3.5,
            },
        )

        confirm = ConfirmContext(
            regime_raw="RANGE",
            regime_confirmed="RANGE",
            confirm_reason="RANGE_PASSTHROUGH",
            confirm_metrics={},
        )

        signal, state = fsm.step(ctx, confirm, flat_position)

        assert signal.signal == "LONG_BOUNCE"
        assert signal.reason == "RANGE_LONG_BOUNCE_NEAR_SUPPORT"

    def test_range_near_resistance_signals_short(self, fsm, flat_position):
        """RANGE near resistance with strong zone -> SHORT_BOUNCE."""
        ctx = MarketContext(
            symbol="XRPUSDT",
            ts=1700000000000,
            price=0.515,
            row_3m={"atr_3m": 0.01},
            row_15m={},
            zone={
                "support": 0.48,
                "resistance": 0.52,
                "strength": 0.0001,
                "dist_to_support": 3.5,
                "dist_to_resistance": 0.5,  # Within threshold
            },
        )

        confirm = ConfirmContext(
            regime_raw="RANGE",
            regime_confirmed="RANGE",
            confirm_reason="RANGE_PASSTHROUGH",
            confirm_metrics={},
        )

        signal, state = fsm.step(ctx, confirm, flat_position)

        assert signal.signal == "SHORT_BOUNCE"
        assert signal.reason == "RANGE_SHORT_BOUNCE_NEAR_RESIST"

    def test_range_weak_zone_holds(self, fsm, flat_position):
        """RANGE with weak zone -> HOLD."""
        ctx = MarketContext(
            symbol="XRPUSDT",
            ts=1700000000000,
            price=0.485,
            row_3m={"atr_3m": 0.01},
            row_15m={},
            zone={
                "support": 0.48,
                "resistance": 0.52,
                "strength": 0.000001,  # Too weak
                "dist_to_support": 0.5,
                "dist_to_resistance": 3.5,
            },
        )

        confirm = ConfirmContext(
            regime_raw="RANGE",
            regime_confirmed="RANGE",
            confirm_reason="RANGE_PASSTHROUGH",
            confirm_metrics={},
        )

        signal, state = fsm.step(ctx, confirm, flat_position)

        assert signal.signal == "HOLD"
        assert "WEAK_ZONE" in signal.reason


class TestDecisionEngineCooldown:
    """Test DecisionEngine cooldown behavior."""

    def test_cooldown_blocks_breakout(self, decision_engine, base_market_ctx, flat_position):
        """Cooldown blocks breakout signals."""
        confirm = ConfirmContext(
            regime_raw="TRANSITION",
            regime_confirmed="NO_TRADE",
            confirm_reason="HIGH_VOL_COOLDOWN",
            confirm_metrics={},
        )

        breakout_signal = CandidateSignal(
            signal="LONG_BREAKOUT",
            score=0.8,
            reason="TRANSITION_BREAKOUT_ONLY_PASS",
            params={},
        )

        decision, state = decision_engine.decide(
            base_market_ctx, confirm, flat_position, breakout_signal
        )

        # First check deny for NO_TRADE
        assert decision.action == "NO_ACTION"
        assert "DE_DENY_NO_TRADE" in decision.reason or "COOLDOWN" in decision.reason

    def test_cooldown_allows_exit(self, decision_engine, base_market_ctx, long_position):
        """Cooldown allows exit signals."""
        confirm = ConfirmContext(
            regime_raw="HIGH_VOL",
            regime_confirmed="NO_TRADE",
            confirm_reason="HIGH_VOL_COOLDOWN",
            confirm_metrics={},
        )

        exit_signal = CandidateSignal(
            signal="EXIT",
            score=1.0,
            reason="TREND_EXIT_TIMEOUT",
            params={},
        )

        decision, state = decision_engine.decide(
            base_market_ctx, confirm, long_position, exit_signal
        )

        assert decision.action == "CLOSE"
        assert decision.reason == "DE_CLOSE_FROM_EXIT_SIGNAL"


class TestDecisionEnginePositionAware:
    """Test DecisionEngine position-aware behavior."""

    def test_open_long_from_flat(self, decision_engine, base_market_ctx, flat_position):
        """Open long from flat position."""
        confirm = ConfirmContext(
            regime_raw="TREND_UP",
            regime_confirmed="TREND_UP",
            confirm_reason="TREND_UP_PASSTHROUGH",
            confirm_metrics={},
        )

        long_signal = CandidateSignal(
            signal="LONG_TREND_PULLBACK",
            score=0.7,
            reason="TREND_UP_PULLBACK_ENTRY",
            params={},
        )

        decision, state = decision_engine.decide(
            base_market_ctx, confirm, flat_position, long_signal
        )

        assert decision.action == "OPEN_LONG"
        assert decision.size > 0

    def test_close_on_opposite_signal(self, decision_engine, base_market_ctx, long_position):
        """Close long on short signal."""
        confirm = ConfirmContext(
            regime_raw="TREND_DOWN",
            regime_confirmed="TREND_DOWN",
            confirm_reason="TREND_DOWN_PASSTHROUGH",
            confirm_metrics={},
        )

        short_signal = CandidateSignal(
            signal="SHORT_TREND_PULLBACK",
            score=0.7,
            reason="TREND_DOWN_PULLBACK_ENTRY",
            params={},
        )

        decision, state = decision_engine.decide(
            base_market_ctx, confirm, long_position, short_signal
        )

        assert decision.action == "CLOSE"
        assert decision.reason == "DE_CLOSE_FROM_OPPOSITE_SIGNAL"

    def test_hold_when_already_in_position(self, decision_engine, base_market_ctx, long_position):
        """HOLD when already long and receiving long signal."""
        confirm = ConfirmContext(
            regime_raw="TREND_UP",
            regime_confirmed="TREND_UP",
            confirm_reason="TREND_UP_PASSTHROUGH",
            confirm_metrics={},
        )

        long_signal = CandidateSignal(
            signal="LONG_TREND_PULLBACK",
            score=0.7,
            reason="TREND_UP_PULLBACK_ENTRY",
            params={},
        )

        decision, state = decision_engine.decide(
            base_market_ctx, confirm, long_position, long_signal
        )

        assert decision.action == "NO_ACTION"
        assert decision.reason == "DE_ALREADY_LONG"


class TestIntegration:
    """Integration tests for FSM + DecisionEngine pipeline."""

    def test_full_pipeline_trend_up(self, fsm, decision_engine, flat_position):
        """Test full pipeline: TREND_UP with pullback -> OPEN_LONG."""
        ctx = MarketContext(
            symbol="XRPUSDT",
            ts=1700000000000,
            price=0.502,
            row_3m={
                "atr_3m": 0.01,
                "ema_fast_3m": 0.50,
                "ema_slow_3m": 0.49,
                "ret_3m": 0.001,
            },
            row_15m={"ema_slope_15m": 0.10},
            zone={},
        )

        confirm = ConfirmContext(
            regime_raw="TREND_UP",
            regime_confirmed="TREND_UP",
            confirm_reason="TREND_UP_PASSTHROUGH",
            confirm_metrics={},
        )

        # FSM step
        signal, fsm_state = fsm.step(ctx, confirm, flat_position)
        assert signal.signal == "LONG_TREND_PULLBACK"

        # DecisionEngine step
        decision, de_state = decision_engine.decide(
            ctx, confirm, flat_position, signal
        )
        assert decision.action == "OPEN_LONG"
        assert decision.size > 0

    def test_full_pipeline_high_vol_blocked(self, fsm, decision_engine, flat_position):
        """Test full pipeline: HIGH_VOL -> blocked."""
        ctx = MarketContext(
            symbol="XRPUSDT",
            ts=1700000000000,
            price=0.50,
            row_3m={"atr_3m": 0.02},
            row_15m={},
            zone={},
        )

        confirm = ConfirmContext(
            regime_raw="HIGH_VOL",
            regime_confirmed="HIGH_VOL",
            confirm_reason="HIGH_VOL_ACTIVE",
            confirm_metrics={"V": 0.03},
        )

        # FSM step
        signal, fsm_state = fsm.step(ctx, confirm, flat_position)
        assert signal.signal == "HOLD"

        # DecisionEngine step
        decision, de_state = decision_engine.decide(
            ctx, confirm, flat_position, signal
        )
        assert decision.action == "NO_ACTION"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
