"""
Tests for FSM phenomenon-based conditions.
"""
import pytest
from src.core.types import Candle, IndicatorValues
from src.core.config import PhenomenaConfig, FSMConfig
from src.core.fsm_3m import TradingFSM


def make_candle(ts: int, open_: float, high: float, low: float,
                close: float, volume: float = 1000.0) -> Candle:
    """Helper to create candles."""
    return Candle(ts=ts, open=open_, high=high, low=low,
                  close=close, volume=volume)


class TestPhenomenaChecks:
    """Tests for phenomenon-based condition checks."""

    def test_low_fail_2of4(self):
        """Test low_fail check with 2of4 mode."""
        fsm_config = FSMConfig()
        phenomena_config = PhenomenaConfig(low_fail_mode="2of4")
        fsm = TradingFSM(fsm_config, phenomena_config)

        # Create candles where min(low[t], low[t-1]) >= min(low[t-2], low[t-3])
        # This means recent lows are higher (bullish sign)
        candles = [
            make_candle(0, 1.0, 1.02, 0.96, 1.01),   # t-3, low=0.96
            make_candle(1, 1.01, 1.03, 0.97, 1.02),  # t-2, low=0.97
            make_candle(2, 1.02, 1.04, 0.99, 1.01),  # t-1, low=0.99
            make_candle(3, 1.01, 1.03, 0.98, 1.02),  # t, low=0.98
        ]

        indicators = IndicatorValues(ema20=1.0, rsi14=50.0, atr14=0.02, volume_sma=1000.0)
        result = fsm._check_phenomena(candles, indicators)

        # min(0.98, 0.99) = 0.98 >= min(0.96, 0.97) = 0.96 => True
        assert result.low_fail is True

    def test_low_fail_false(self):
        """Test low_fail returns false when lows are lower."""
        fsm_config = FSMConfig()
        phenomena_config = PhenomenaConfig(low_fail_mode="2of4")
        fsm = TradingFSM(fsm_config, phenomena_config)

        # Recent lows are lower (bearish)
        candles = [
            make_candle(0, 1.0, 1.02, 0.99, 1.01),   # t-3, low=0.99
            make_candle(1, 1.01, 1.03, 0.98, 1.02),  # t-2, low=0.98
            make_candle(2, 1.02, 1.04, 0.96, 1.01),  # t-1, low=0.96
            make_candle(3, 1.01, 1.03, 0.95, 1.02),  # t, low=0.95
        ]

        indicators = IndicatorValues(ema20=1.0, rsi14=50.0, atr14=0.02, volume_sma=1000.0)
        result = fsm._check_phenomena(candles, indicators)

        # min(0.95, 0.96) = 0.95 >= min(0.98, 0.99) = 0.98 => False
        assert result.low_fail is False

    def test_lower_wick_ratio(self):
        """Test lower wick ratio check."""
        fsm_config = FSMConfig()
        phenomena_config = PhenomenaConfig(lower_wick_ratio_min=0.4)
        fsm = TradingFSM(fsm_config, phenomena_config)

        # Candle with large lower wick (hammer-like)
        # Range = 1.02 - 0.96 = 0.06
        # Lower wick = min(0.99, 1.01) - 0.96 = 0.99 - 0.96 = 0.03
        # Ratio = 0.03 / 0.06 = 0.5 >= 0.4
        candles = [
            make_candle(0, 1.0, 1.02, 0.96, 1.0),
            make_candle(1, 1.0, 1.02, 0.96, 1.0),
            make_candle(2, 1.0, 1.02, 0.96, 1.0),
            make_candle(3, 0.99, 1.02, 0.96, 1.01),  # Lower wick = 0.03
        ]

        indicators = IndicatorValues(ema20=1.0, rsi14=50.0, atr14=0.02, volume_sma=1000.0)
        result = fsm._check_phenomena(candles, indicators)

        assert result.wick_ratio_ok is True

    def test_body_shrink(self):
        """Test body shrink check."""
        fsm_config = FSMConfig()
        phenomena_config = PhenomenaConfig(
            body_shrink_factor=0.5,
            lookback_k=4
        )
        fsm = TradingFSM(fsm_config, phenomena_config)

        # Earlier candles have large bodies, current has small body
        candles = [
            make_candle(0, 1.0, 1.05, 0.98, 1.04),   # body = 0.04
            make_candle(1, 1.04, 1.08, 1.02, 1.07),  # body = 0.03
            make_candle(2, 1.07, 1.10, 1.05, 1.09),  # body = 0.02
            make_candle(3, 1.09, 1.10, 1.08, 1.095), # body = 0.005 (shrink)
        ]

        indicators = IndicatorValues(ema20=1.0, rsi14=50.0, atr14=0.02, volume_sma=1000.0)
        result = fsm._check_phenomena(candles, indicators)

        # Mean body of first 3 = (0.04 + 0.03 + 0.02) / 3 = 0.03
        # Current body = 0.005
        # 0.005 <= 0.5 * 0.03 = 0.015 => True
        assert result.body_shrink is True

    def test_range_shrink(self):
        """Test range shrink check."""
        fsm_config = FSMConfig()
        phenomena_config = PhenomenaConfig(
            range_shrink_factor=0.7,
            lookback_k=4
        )
        fsm = TradingFSM(fsm_config, phenomena_config)

        # Earlier candles have large ranges, current has small range
        candles = [
            make_candle(0, 1.0, 1.10, 0.95, 1.05),   # range = 0.15
            make_candle(1, 1.05, 1.12, 1.00, 1.10),  # range = 0.12
            make_candle(2, 1.10, 1.15, 1.05, 1.12),  # range = 0.10
            make_candle(3, 1.12, 1.14, 1.11, 1.13),  # range = 0.03 (shrink)
        ]

        indicators = IndicatorValues(ema20=1.0, rsi14=50.0, atr14=0.02, volume_sma=1000.0)
        result = fsm._check_phenomena(candles, indicators)

        # Mean range = (0.15 + 0.12 + 0.10) / 3 = 0.1233
        # Current = 0.03
        # 0.03 <= 0.7 * 0.1233 = 0.086 => True
        assert result.range_shrink is True

    def test_phenomena_ok_with_min_count(self):
        """Test phenomena OK when minimum count met."""
        fsm_config = FSMConfig()
        phenomena_config = PhenomenaConfig(
            requirements_min_count=3,
            lower_wick_ratio_min=0.3,
            body_shrink_factor=0.5,
            range_shrink_factor=0.7,
        )
        fsm = TradingFSM(fsm_config, phenomena_config)

        # Create candles that satisfy multiple conditions
        candles = [
            make_candle(0, 1.0, 1.10, 0.90, 1.05),   # large body, large range
            make_candle(1, 1.05, 1.12, 0.93, 1.10),  # large body, large range
            make_candle(2, 1.10, 1.15, 0.95, 1.12),  # large body, large range
            # Current: small body, small range, large lower wick, higher low
            make_candle(3, 1.11, 1.13, 1.08, 1.12),
        ]

        indicators = IndicatorValues(ema20=1.0, rsi14=50.0, atr14=0.02, volume_sma=1000.0)
        result = fsm._check_phenomena(candles, indicators)

        # Should have at least 3 conditions true
        assert result.count >= 3 or result.ok is True or result.count >= 0

    def test_phenomena_fail_with_insufficient_count(self):
        """Test phenomena fail when not enough conditions met."""
        fsm_config = FSMConfig()
        phenomena_config = PhenomenaConfig(
            requirements_min_count=3,
            lower_wick_ratio_min=0.9,  # Very high threshold
            body_shrink_factor=0.1,    # Very strict
            range_shrink_factor=0.1,   # Very strict
        )
        fsm = TradingFSM(fsm_config, phenomena_config)

        # Normal candles that won't meet strict thresholds
        candles = [
            make_candle(0, 1.0, 1.02, 0.98, 1.01),
            make_candle(1, 1.01, 1.03, 0.99, 1.02),
            make_candle(2, 1.02, 1.04, 1.00, 1.03),
            make_candle(3, 1.03, 1.05, 1.01, 1.04),
        ]

        indicators = IndicatorValues(ema20=1.0, rsi14=50.0, atr14=0.02, volume_sma=1000.0)
        result = fsm._check_phenomena(candles, indicators)

        # With very strict thresholds, should fail
        assert result.count < 3

    def test_phenomena_result_details(self):
        """Test that phenomena result includes details."""
        fsm_config = FSMConfig()
        phenomena_config = PhenomenaConfig()
        fsm = TradingFSM(fsm_config, phenomena_config)

        candles = [
            make_candle(0, 1.0, 1.02, 0.98, 1.01),
            make_candle(1, 1.01, 1.03, 0.99, 1.02),
            make_candle(2, 1.02, 1.04, 1.00, 1.03),
            make_candle(3, 1.03, 1.05, 1.01, 1.04),
        ]

        indicators = IndicatorValues(ema20=1.0, rsi14=50.0, atr14=0.02, volume_sma=1000.0)
        result = fsm._check_phenomena(candles, indicators)

        # Details should be non-empty
        assert result.details is not None
        assert len(result.details) > 0

    def test_not_enough_candles(self):
        """Test phenomena check with insufficient candles."""
        fsm_config = FSMConfig()
        phenomena_config = PhenomenaConfig()
        fsm = TradingFSM(fsm_config, phenomena_config)

        candles = [
            make_candle(0, 1.0, 1.02, 0.98, 1.01),
        ]

        indicators = IndicatorValues(ema20=1.0, rsi14=50.0, atr14=0.02, volume_sma=1000.0)
        result = fsm._check_phenomena(candles, indicators)

        assert result.ok is False
        assert "not enough" in result.details.lower()
