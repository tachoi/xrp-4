"""
Tests for FSM anchor detection.
"""
import pytest
from src.core.types import (
    Candle, ZoneState, ZoneStateType, FSMState, Signal, IndicatorValues
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


def make_core_zone_state() -> ZoneState:
    """Helper to create CORE zone state."""
    return ZoneState(
        state=ZoneStateType.CORE,
        explanation="CORE: test zone"
    )


def make_outside_zone_state() -> ZoneState:
    """Helper to create OUTSIDE zone state."""
    return ZoneState(
        state=ZoneStateType.OUTSIDE,
        explanation="OUTSIDE: no zone"
    )


class TestAnchorDetection:
    """Tests for anchor candle detection."""

    def test_anchor_detected_with_high_volume(self):
        """Test anchor detection with volume spike."""
        fsm_config = FSMConfig(
            anchor_vol_mult=2.5,
            anchor_body_atr_mult=1.2,
            chase_dist_max_atr=1.5,
        )
        phenomena_config = PhenomenaConfig()
        fsm = TradingFSM(fsm_config, phenomena_config)

        # Bullish candle with high volume
        candle = make_candle(
            ts=1000,
            open_=1.0,
            high=1.05,
            low=0.99,
            close=1.04,  # Bullish
            volume=3000.0  # 3x average
        )

        indicators = make_indicators(
            ema=1.02,  # Close to EMA
            atr=0.02,  # Body = 0.04, need 0.024 (1.2*0.02)
            vol_sma=1000.0
        )

        zone_state = make_core_zone_state()
        recent = [candle]

        decision = fsm.step(candle, indicators, zone_state, recent, [50.0])

        assert decision.signal == Signal.ANCHOR
        assert decision.anchor_info is not None
        assert decision.anchor_info.direction == 1  # Bullish

    def test_no_anchor_low_volume(self):
        """Test no anchor when volume is low."""
        fsm_config = FSMConfig(anchor_vol_mult=2.5)
        phenomena_config = PhenomenaConfig()
        fsm = TradingFSM(fsm_config, phenomena_config)

        candle = make_candle(
            ts=1000,
            open_=1.0,
            high=1.05,
            low=0.99,
            close=1.04,
            volume=1500.0  # Only 1.5x (need 2.5x)
        )

        indicators = make_indicators(vol_sma=1000.0, atr=0.02)
        zone_state = make_core_zone_state()

        decision = fsm.step(candle, indicators, zone_state, [candle], [50.0])

        assert decision.signal == Signal.NONE

    def test_no_anchor_small_body(self):
        """Test no anchor when body is too small."""
        fsm_config = FSMConfig(
            anchor_vol_mult=2.5,
            anchor_body_atr_mult=1.2,
        )
        phenomena_config = PhenomenaConfig()
        fsm = TradingFSM(fsm_config, phenomena_config)

        # Small body candle
        candle = make_candle(
            ts=1000,
            open_=1.0,
            high=1.03,
            low=0.97,
            close=1.01,  # Small body = 0.01
            volume=3000.0
        )

        indicators = make_indicators(
            vol_sma=1000.0,
            atr=0.02  # Need body >= 0.024
        )
        zone_state = make_core_zone_state()

        decision = fsm.step(candle, indicators, zone_state, [candle], [50.0])

        assert decision.signal == Signal.NONE

    def test_no_anchor_chase_filter_fail(self):
        """Test no anchor when chase filter fails."""
        fsm_config = FSMConfig(
            anchor_vol_mult=2.5,
            anchor_body_atr_mult=1.2,
            chase_dist_max_atr=1.5,
        )
        phenomena_config = PhenomenaConfig()
        fsm = TradingFSM(fsm_config, phenomena_config)

        candle = make_candle(
            ts=1000,
            open_=1.0,
            high=1.05,
            low=0.99,
            close=1.04,
            volume=3000.0
        )

        # EMA far from close (chase filter fail)
        indicators = make_indicators(
            ema=0.90,  # Distance = 0.14, ATR = 0.02, ratio = 7 > 1.5
            atr=0.02,
            vol_sma=1000.0
        )
        zone_state = make_core_zone_state()

        decision = fsm.step(candle, indicators, zone_state, [candle], [50.0])

        assert decision.signal == Signal.NONE

    def test_no_anchor_outside_zone(self):
        """Test no anchor detection when OUTSIDE zone."""
        fsm_config = FSMConfig()
        phenomena_config = PhenomenaConfig()
        fsm = TradingFSM(fsm_config, phenomena_config)

        # Valid anchor candle
        candle = make_candle(
            ts=1000, open_=1.0, high=1.05, low=0.99,
            close=1.04, volume=3000.0
        )

        indicators = make_indicators(ema=1.02, atr=0.02, vol_sma=1000.0)
        zone_state = make_outside_zone_state()  # OUTSIDE

        decision = fsm.step(candle, indicators, zone_state, [candle], [50.0])

        # Should force IDLE
        assert decision.new_state == FSMState.IDLE
        assert "Forced IDLE" in decision.explanation or "OUTSIDE" in decision.explanation

    def test_bearish_anchor(self):
        """Test bearish anchor detection."""
        fsm_config = FSMConfig(
            anchor_vol_mult=2.5,
            anchor_body_atr_mult=1.2,
            chase_dist_max_atr=1.5,
        )
        phenomena_config = PhenomenaConfig()
        fsm = TradingFSM(fsm_config, phenomena_config)

        # Bearish candle
        candle = make_candle(
            ts=1000,
            open_=1.04,  # Open high
            high=1.05,
            low=0.99,
            close=1.0,   # Close low (bearish)
            volume=3000.0
        )

        indicators = make_indicators(ema=1.02, atr=0.02, vol_sma=1000.0)
        zone_state = make_core_zone_state()

        decision = fsm.step(candle, indicators, zone_state, [candle], [50.0])

        assert decision.signal == Signal.ANCHOR
        assert decision.anchor_info.direction == -1  # Bearish


class TestAnchorExpiry:
    """Tests for anchor expiry logic."""

    def test_anchor_expires_after_n_candles(self):
        """Test anchor expiry after configured candles."""
        fsm_config = FSMConfig(
            anchor_expire_candles=4,
            anchor_vol_mult=2.5,
            anchor_body_atr_mult=1.2,
            chase_dist_max_atr=1.5,
        )
        phenomena_config = PhenomenaConfig()
        fsm = TradingFSM(fsm_config, phenomena_config)

        # First, detect anchor
        anchor_candle = make_candle(
            ts=1000, open_=1.0, high=1.05, low=0.99,
            close=1.04, volume=3000.0
        )
        indicators = make_indicators(ema=1.02, atr=0.02, vol_sma=1000.0)
        zone_state = make_core_zone_state()

        decision = fsm.step(anchor_candle, indicators, zone_state,
                           [anchor_candle], [50.0])
        assert decision.signal == Signal.ANCHOR

        # Feed 5 more candles (no pullback) to trigger expiry
        # After anchor: ANCHOR_FOUND -> PULLBACK_WAIT (1st call, no increment)
        # Then increments: 1, 2, 3, 4 -> expires at 4
        for i in range(5):
            candle = make_candle(
                ts=1000 + (i+1) * 180000,
                open_=1.05, high=1.06, low=1.04, close=1.055,
                volume=1000.0
            )
            decision = fsm.step(candle, indicators, zone_state,
                               [anchor_candle, candle], [50.0, 50.0])

        # Should have expired and returned to IDLE
        assert fsm.state == FSMState.IDLE
        assert "expired" in decision.explanation.lower()
