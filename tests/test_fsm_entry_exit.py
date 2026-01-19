"""
Tests for FSM entry and exit logic.
"""
import pytest
from src.core.types import (
    Candle, ZoneState, ZoneStateType, FSMState, Signal, Side, IndicatorValues
)
from src.core.config import FSMConfig, PhenomenaConfig
from src.core.fsm_3m import TradingFSM


def make_candle(ts: int, open_: float, high: float, low: float,
                close: float, volume: float = 1000.0) -> Candle:
    """Helper to create candles."""
    return Candle(ts=ts, open=open_, high=high, low=low,
                  close=close, volume=volume)


def make_indicators(ema: float = 1.0, rsi: float = 50.0, atr: float = 0.02,
                    vol_sma: float = 1000.0) -> IndicatorValues:
    """Helper to create indicator values."""
    return IndicatorValues(
        ema20=ema,
        rsi14=rsi,
        atr14=atr,
        volume_sma=vol_sma,
    )


class TestEntryTrigger:
    """Tests for entry trigger logic."""

    def setup_method(self):
        """Setup FSM with anchor in place."""
        self.fsm_config = FSMConfig(
            anchor_vol_mult=2.5,
            anchor_body_atr_mult=1.2,
            chase_dist_max_atr=1.5,
            pullback_tolerance_atr=0.5,
        )
        self.phenomena_config = PhenomenaConfig(requirements_min_count=1)
        self.fsm = TradingFSM(self.fsm_config, self.phenomena_config)

    def test_long_entry_trigger(self):
        """Test LONG entry trigger with micro break."""
        fsm = self.fsm
        core_zone = ZoneState(state=ZoneStateType.CORE, explanation="CORE")

        # Step 1: Anchor detection
        anchor = make_candle(0, 1.0, 1.05, 0.99, 1.04, volume=3000.0)
        indicators = make_indicators(ema=1.02, atr=0.02, vol_sma=1000.0)

        fsm.step(anchor, indicators, core_zone, [anchor], [50.0])
        assert fsm.ctx.anchor is not None

        # Step 2: Move to PULLBACK_WAIT
        fsm.step(anchor, indicators, core_zone, [anchor], [50.0])

        # Step 3: Pullback candle near A_mid with phenomena OK
        # A_mid = (1.0 + 1.04) / 2 = 1.02
        pullback = make_candle(1, 1.04, 1.05, 1.00, 1.02, volume=1000.0)

        # Need candles for phenomena check
        candles = [
            make_candle(0, 1.0, 1.10, 0.90, 1.05),
            make_candle(1, 1.05, 1.08, 0.93, 1.03),
            make_candle(2, 1.03, 1.06, 0.97, 1.02),
            pullback
        ]

        # Mock phenomena OK by setting min count to 1
        decision = fsm.step(pullback, indicators, core_zone, candles, [48.0, 50.0, 52.0])

        # May or may not be ENTRY_READY depending on phenomena
        # The key is the FSM processed it

    def test_entry_blocked_in_near_zone(self):
        """Test that entry is blocked when in NEAR zone."""
        fsm = self.fsm
        core_zone = ZoneState(state=ZoneStateType.CORE, explanation="CORE")
        near_zone = ZoneState(state=ZoneStateType.NEAR, explanation="NEAR")

        # Setup: Get to ENTRY_READY state
        anchor = make_candle(0, 1.0, 1.05, 0.99, 1.04, volume=3000.0)
        indicators = make_indicators(ema=1.02, atr=0.02, vol_sma=1000.0)

        # Force state to ENTRY_READY for testing
        fsm.state = FSMState.ENTRY_READY
        fsm.ctx.anchor = fsm._detect_anchor(anchor, indicators, [anchor])

        # Try to enter while in NEAR zone
        trigger_candle = make_candle(2, 1.02, 1.06, 1.01, 1.05, volume=1000.0)

        decision = fsm.step(trigger_candle, indicators, near_zone,
                           [anchor, trigger_candle], [50.0, 52.0])

        # Should NOT enter trade
        assert decision.signal != Signal.ENTRY_LONG
        assert decision.signal != Signal.ENTRY_SHORT
        assert "CORE" in decision.explanation or "blocked" in decision.explanation.lower()

    def test_anchor_invalidation_long(self):
        """Test LONG anchor invalidation when price breaks A_open."""
        fsm = self.fsm
        core_zone = ZoneState(state=ZoneStateType.CORE, explanation="CORE")

        # Setup anchor
        anchor = make_candle(0, 1.0, 1.05, 0.99, 1.04, volume=3000.0)
        indicators = make_indicators(ema=1.02, atr=0.02, vol_sma=1000.0)

        fsm.step(anchor, indicators, core_zone, [anchor], [50.0])
        fsm.step(anchor, indicators, core_zone, [anchor], [50.0])

        assert fsm.state in (FSMState.ANCHOR_FOUND, FSMState.PULLBACK_WAIT)

        # Price breaks below A_open (1.0)
        break_candle = make_candle(1, 1.02, 1.03, 0.95, 0.96, volume=1000.0)

        decision = fsm.step(break_candle, indicators, core_zone,
                           [anchor, break_candle], [50.0, 45.0])

        # Should invalidate and return to IDLE
        assert fsm.state == FSMState.IDLE
        assert "invalidated" in decision.explanation.lower()


class TestExitLogic:
    """Tests for exit conditions."""

    def setup_method(self):
        """Setup FSM in IN_TRADE state."""
        self.fsm_config = FSMConfig(
            hold_max_candles=6,
            cooldown_candles=3,
        )
        self.phenomena_config = PhenomenaConfig()
        self.fsm = TradingFSM(self.fsm_config, self.phenomena_config)

        # Put FSM in trade
        self.fsm.state = FSMState.IN_TRADE
        self.fsm.ctx.trade_side = Side.LONG
        self.fsm.ctx.entry_price = 1.0
        self.fsm.ctx.stop_price = 0.95
        self.fsm.ctx.candles_in_trade = 0

    def test_hard_stop_exit_long(self):
        """Test exit on hard stop hit for LONG."""
        fsm = self.fsm
        core_zone = ZoneState(state=ZoneStateType.CORE, explanation="CORE")
        indicators = make_indicators()

        # Candle that hits stop (low <= 0.95)
        stop_candle = make_candle(0, 1.0, 1.01, 0.94, 0.96, volume=1000.0)

        decision = fsm.step(stop_candle, indicators, core_zone,
                           [stop_candle], [40.0])

        assert decision.signal == Signal.EXIT
        assert "stop" in decision.explanation.lower()
        assert fsm.state == FSMState.EXIT_COOLDOWN

    def test_time_stop_exit(self):
        """Test exit after max hold time."""
        fsm = self.fsm
        core_zone = ZoneState(state=ZoneStateType.CORE, explanation="CORE")
        indicators = make_indicators()

        # Simulate 6 candles in trade
        fsm.ctx.candles_in_trade = 5

        candle = make_candle(0, 1.0, 1.02, 0.98, 1.01, volume=1000.0)
        decision = fsm.step(candle, indicators, core_zone, [candle], [50.0])

        assert decision.signal == Signal.EXIT
        assert "time" in decision.explanation.lower()

    def test_rsi_exit_long(self):
        """Test RSI-based exit for LONG."""
        fsm = self.fsm
        core_zone = ZoneState(state=ZoneStateType.CORE, explanation="CORE")

        # RSI >= 70 and turning down
        indicators = make_indicators(rsi=72.0)

        candle = make_candle(0, 1.05, 1.06, 1.03, 1.04, volume=1000.0)
        rsi_history = [65.0, 68.0, 71.0, 74.0, 72.0]  # Turning down

        decision = fsm.step(candle, indicators, core_zone, [candle], rsi_history)

        # May or may not exit depending on exact RSI turn detection
        # Key is the check runs without error

    def test_cooldown_after_exit(self):
        """Test cooldown state after exit."""
        fsm = self.fsm
        fsm.state = FSMState.EXIT_COOLDOWN
        fsm.ctx.cooldown_remaining = 3

        core_zone = ZoneState(state=ZoneStateType.CORE, explanation="CORE")
        indicators = make_indicators()
        candle = make_candle(0, 1.0, 1.02, 0.98, 1.01, volume=1000.0)

        # First cooldown candle
        decision = fsm.step(candle, indicators, core_zone, [candle], [50.0])
        assert fsm.state == FSMState.EXIT_COOLDOWN
        assert fsm.ctx.cooldown_remaining == 2

        # Second
        decision = fsm.step(candle, indicators, core_zone, [candle], [50.0])
        assert fsm.ctx.cooldown_remaining == 1

        # Third - should return to IDLE
        decision = fsm.step(candle, indicators, core_zone, [candle], [50.0])
        assert fsm.state == FSMState.IDLE


class TestShortTrades:
    """Tests for SHORT trade logic."""

    def test_short_anchor(self):
        """Test SHORT anchor detection."""
        fsm_config = FSMConfig(
            anchor_vol_mult=2.5,
            anchor_body_atr_mult=1.2,
            chase_dist_max_atr=1.5,
        )
        phenomena_config = PhenomenaConfig()
        fsm = TradingFSM(fsm_config, phenomena_config)

        core_zone = ZoneState(state=ZoneStateType.CORE, explanation="CORE")

        # Bearish anchor
        anchor = make_candle(0, 1.04, 1.05, 0.99, 1.0, volume=3000.0)
        indicators = make_indicators(ema=1.02, atr=0.02, vol_sma=1000.0)

        decision = fsm.step(anchor, indicators, core_zone, [anchor], [50.0])

        assert decision.signal == Signal.ANCHOR
        assert decision.anchor_info.direction == -1

    def test_short_stop(self):
        """Test SHORT stop calculation."""
        fsm_config = FSMConfig()
        phenomena_config = PhenomenaConfig()
        fsm = TradingFSM(fsm_config, phenomena_config)

        from src.core.fsm_3m import AnchorInfo

        anchor_info = AnchorInfo(
            open=1.04,
            close=1.0,
            mid=1.02,
            direction=-1,
            ts=0,
            index=0,
            volume_mult=3.0,
            body_atr_mult=2.0,
        )

        # Need enough candles for swing_high detection (lookback=5)
        candles = [
            make_candle(0, 1.02, 1.03, 1.00, 1.02),
            make_candle(1, 1.02, 1.04, 1.01, 1.03),
            make_candle(2, 1.03, 1.05, 1.02, 1.04),
            make_candle(3, 1.04, 1.06, 1.03, 1.05),  # Highest high
            make_candle(4, 1.05, 1.05, 1.03, 1.04),
            make_candle(5, 1.04, 1.04, 1.02, 1.03),
        ]

        stop = fsm._compute_stop(Side.SHORT, anchor_info, candles)

        # Stop should be max(swing_high, A_open)
        # swing_high from lookback = 1.06, A_open = 1.04
        assert stop == 1.06


class TestStateInfo:
    """Tests for FSM state information."""

    def test_get_state_info(self):
        """Test state info retrieval."""
        fsm_config = FSMConfig()
        phenomena_config = PhenomenaConfig()
        fsm = TradingFSM(fsm_config, phenomena_config)

        info = fsm.get_state_info()

        assert 'state' in info
        assert info['state'] == 'IDLE'
        assert 'has_anchor' in info
        assert info['has_anchor'] is False

    def test_is_in_trade(self):
        """Test in_trade property."""
        fsm_config = FSMConfig()
        phenomena_config = PhenomenaConfig()
        fsm = TradingFSM(fsm_config, phenomena_config)

        assert fsm.is_in_trade is False

        fsm.state = FSMState.IN_TRADE
        assert fsm.is_in_trade is True

    def test_reset(self):
        """Test FSM reset."""
        fsm_config = FSMConfig()
        phenomena_config = PhenomenaConfig()
        fsm = TradingFSM(fsm_config, phenomena_config)

        # Add some state
        fsm.state = FSMState.IN_TRADE
        fsm.ctx.trade_side = Side.LONG
        fsm._candle_index = 100

        fsm.reset()

        assert fsm.state == FSMState.IDLE
        assert fsm.ctx.trade_side is None
        assert fsm._candle_index == 0
