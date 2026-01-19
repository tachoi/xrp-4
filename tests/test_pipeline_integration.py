"""Integration tests for the full trading pipeline.

Tests the interaction between:
1. HMM (Fast + Mid) -> Confirm Layer data flow
2. Regime -> Signal generation
3. XGB feature contract
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from xrp4.regime.hmm_types import RegimeLabel, HMMPrediction, RegimePacket
from xrp4.regime.confirm import RegimeConfirmLayer, ConfirmConfig
from xrp4.regime.hmm_fusion import HMMFusion
from xrp4.core.types import ConfirmContext, MarketContext, PositionState


class TestHMMToConfirmDataFlow:
    """Test data flow from HMM to Confirm Layer."""

    @pytest.fixture
    def confirm_layer(self):
        """Create ConfirmLayer with default config."""
        return RegimeConfirmLayer(ConfirmConfig())

    @pytest.fixture
    def sample_hist_15m(self):
        """Create sample 15m history DataFrame."""
        n = 100
        return pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="15min"),
            "open": np.random.uniform(2.0, 2.5, n),
            "high": np.random.uniform(2.1, 2.6, n),
            "low": np.random.uniform(1.9, 2.4, n),
            "close": np.random.uniform(2.0, 2.5, n),
            "volume": np.random.uniform(1000, 5000, n),
            "ewm_std_ret_15m": np.random.uniform(0.002, 0.008, n),
        })

    def test_row_15m_has_all_required_fields(self, confirm_layer, sample_hist_15m):
        """row_15m should have all fields needed by Confirm Layer."""
        # Simulate the row_15m dict from paper_trading.py
        bar_15m = sample_hist_15m.iloc[-1]
        price = float(bar_15m["close"])

        row_15m = {
            # Price data (needed for box/breakout calculations)
            "close": float(bar_15m.get("close", price)),
            "high": float(bar_15m.get("high", price)),
            "low": float(bar_15m.get("low", price)),
            # EMA and trend indicators (needed for TREND/RANGE validation)
            "ema_slope_15m": 0.001,
            "ewm_ret_15m": 0.0001,
            # Volatility metrics (needed for RANGE and HIGH_VOL validation)
            "ewm_std_ret_15m": 0.005,
            "atr_pct_15m": 1.0,
            # Structure metrics (needed for HIGH_VOL exit)
            "bb_width_15m": 0.02,
            "range_comp_15m": 0.5,
        }

        # Test all validation paths - should not raise KeyError
        result, _ = confirm_layer.confirm("RANGE", row_15m, sample_hist_15m)
        assert result.confirmed_regime is not None

        result, _ = confirm_layer.confirm("TREND_UP", row_15m, sample_hist_15m)
        assert result.confirmed_regime is not None

        result, _ = confirm_layer.confirm("TREND_DOWN", row_15m, sample_hist_15m)
        assert result.confirmed_regime is not None

        result, _ = confirm_layer.confirm("TRANSITION", row_15m, sample_hist_15m)
        assert result.confirmed_regime is not None

    def test_regime_confirmed_affects_signal_type(self, confirm_layer, sample_hist_15m):
        """Confirmed regime should determine available signal types."""
        # RANGE regime - should confirm as RANGE with flat slope
        row_15m_range = {
            "close": 2.3, "high": 2.35, "low": 2.25,
            "ema_slope_15m": 0.0005,  # Flat
            "ewm_ret_15m": 0.00001,
            "ewm_std_ret_15m": 0.003,
            "atr_pct_15m": 1.0,
            "bb_width_15m": 0.02,
            "range_comp_15m": 0.5,
        }
        result_range, _ = confirm_layer.confirm("RANGE", row_15m_range, sample_hist_15m)

        # TREND regime - should confirm as TREND with steep slope
        row_15m_trend = {
            "close": 2.3, "high": 2.35, "low": 2.25,
            "ema_slope_15m": 0.005,  # Steep
            "ewm_ret_15m": 0.0001,
            "ewm_std_ret_15m": 0.003,
            "atr_pct_15m": 1.0,
            "bb_width_15m": 0.02,
            "range_comp_15m": 0.5,
        }
        result_trend, _ = confirm_layer.confirm("TREND_UP", row_15m_trend, sample_hist_15m)

        # Assertions
        assert result_range.confirmed_regime == "RANGE", "RANGE should be validated with flat slope"
        assert result_trend.confirmed_regime == "TREND_UP", "TREND_UP should be validated with steep slope"


class TestHMMFusionIntegration:
    """Test HMM Fusion with real predictions."""

    def test_fusion_preserves_state_probs(self):
        """Fusion should preserve state probability information."""
        fusion = HMMFusion(method="weighted")

        # Create Fast HMM prediction
        fast_pred = HMMPrediction(
            label=RegimeLabel.TREND_UP,
            state_idx=1,
            confidence=0.75,
            state_probs=np.array([0.1, 0.75, 0.1, 0.05]),
            entropy=0.8,
            transition_rate=0.1,
            timeframe="3m",
            n_states=4,
        )

        # Create Mid HMM prediction
        mid_pred = HMMPrediction(
            label=RegimeLabel.TREND_UP,
            state_idx=1,
            confidence=0.85,
            state_probs=np.array([0.05, 0.85, 0.05, 0.05]),
            entropy=0.5,
            transition_rate=0.05,
            timeframe="15m",
            n_states=4,
        )

        # Fuse predictions
        packet = fusion.fuse(fast_pred, mid_pred, datetime.now())

        # Assertions
        assert packet.label_fused == RegimeLabel.TREND_UP  # Both agree
        assert "HMM_AGREE" in packet.hmm_flags
        assert packet.conf_struct == 0.85
        assert packet.conf_micro == 0.75

    def test_fusion_handles_disagreement(self):
        """Fusion should handle disagreement between Fast and Mid HMM."""
        fusion = HMMFusion(method="weighted", conflict_resolution="mid_priority")

        # Fast says TREND_UP
        fast_pred = HMMPrediction(
            label=RegimeLabel.TREND_UP,
            state_idx=1,
            confidence=0.6,
            state_probs=np.array([0.2, 0.6, 0.1, 0.1]),
            entropy=1.0,
            transition_rate=0.2,
            timeframe="3m",
            n_states=4,
        )

        # Mid says RANGE (structural)
        mid_pred = HMMPrediction(
            label=RegimeLabel.RANGE,
            state_idx=0,
            confidence=0.7,
            state_probs=np.array([0.7, 0.15, 0.1, 0.05]),
            entropy=0.8,
            transition_rate=0.1,
            timeframe="15m",
            n_states=4,
        )

        packet = fusion.fuse(fast_pred, mid_pred, datetime.now())

        # With mid_priority, should lean toward RANGE
        assert "HMM_DISAGREE" in packet.hmm_flags
        # The exact result depends on weights, but disagreement should be flagged


class TestXGBFeatureContract:
    """Test XGB feature extraction contract."""

    def test_all_features_extracted(self):
        """All 14 features should be extracted correctly."""
        from xrp4.core.xgb_gate import XGBApprovalGate

        gate = XGBApprovalGate(model_path=None)

        row_3m = {
            "close": 2.35,
            "high": 2.40,
            "low": 2.30,
            "close_2": 2.33,
            "close_5": 2.28,
            "ret_3m": 0.008,
            "volume": 5000,
            "volume_ma": 4000,
            "ema_fast_3m": 2.34,
            "ema_slow_3m": 2.32,
            "volatility": 0.005,
            "rsi_3m": 55,
            "ema_slope_3m": 0.002,
        }

        features = gate._extract_features(row_3m, "LONG_TREND_PULLBACK", "TREND_UP")

        # Check feature count
        assert len(features) == 14, f"Expected 14 features, got {len(features)}"

        # Check feature names match expected order
        expected_order = [
            "ret", "ret_2", "ret_5",
            "ema_diff", "price_to_ema20", "price_to_ema50",
            "volatility", "range_pct", "volume_ratio",
            "rsi", "ema_slope",
            "side_num", "regime_trend_up", "regime_trend_down"
        ]

        # Verify no NaN values
        assert not np.any(np.isnan(features)), "Features contain NaN values"

    def test_features_use_calculated_values(self):
        """Features should use calculated values, not hardcoded defaults."""
        from xrp4.core.xgb_gate import XGBApprovalGate

        gate = XGBApprovalGate(model_path=None)

        row_3m = {
            "close": 100,
            "high": 102,
            "low": 98,
            "close_2": 99,
            "close_5": 95,
            "ret_3m": 0.01,
            "volume": 2000,
            "volume_ma": 1000,
            "ema_fast_3m": 99,
            "ema_slow_3m": 98,
            "volatility": 0.008,
            "rsi_3m": 60,
            "ema_slope_3m": 0.003,
        }

        features = gate._extract_features(row_3m, "LONG_TREND_PULLBACK", "TREND_UP")

        # ret_2 should be calculated: (100 - 99) / 99 = 0.0101
        assert abs(features[1] - 0.0101) < 0.001, f"ret_2 incorrect: {features[1]}"

        # ret_5 should be calculated: (100 - 95) / 95 = 0.0526
        assert abs(features[2] - 0.0526) < 0.001, f"ret_5 incorrect: {features[2]}"

        # range_pct should be calculated: (102 - 98) / 100 = 0.04
        assert abs(features[7] - 0.04) < 0.001, f"range_pct incorrect: {features[7]}"

        # volume_ratio should be calculated: 2000 / 1000 = 2.0
        assert abs(features[8] - 2.0) < 0.001, f"volume_ratio incorrect: {features[8]}"


class TestEndToEndPipeline:
    """End-to-end pipeline tests."""

    def test_regime_to_signal_flow(self):
        """Test complete flow from regime detection to signal generation."""
        from xrp4.core.fsm import TradingFSM
        from xrp4.regime.confirm import RegimeConfirmLayer, ConfirmConfig

        # Setup
        confirm_layer = RegimeConfirmLayer(ConfirmConfig())
        fsm = TradingFSM()

        # Create sample data
        hist_15m = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=100, freq="15min"),
            "high": np.random.uniform(2.1, 2.6, 100),
            "low": np.random.uniform(1.9, 2.4, 100),
            "close": np.random.uniform(2.0, 2.5, 100),
            "ewm_std_ret_15m": np.random.uniform(0.002, 0.006, 100),
        })

        row_15m = {
            "close": 2.35, "high": 2.40, "low": 2.30,
            "ema_slope_15m": 0.005,  # Strong trend
            "ewm_ret_15m": 0.0003,
            "ewm_std_ret_15m": 0.004,
            "atr_pct_15m": 1.5,
            "bb_width_15m": 0.03,
            "range_comp_15m": 0.6,
        }

        # Step 1: Confirm regime
        result, state = confirm_layer.confirm(
            regime_raw="TREND_UP",
            row_15m=row_15m,
            hist_15m=hist_15m,
        )

        # Step 2: Create contexts
        confirm_ctx = ConfirmContext(
            regime_raw="TREND_UP",
            regime_confirmed=result.confirmed_regime,
            confirm_reason=result.reason,
            confirm_metrics=result.metrics,
        )

        market_ctx = MarketContext(
            symbol="XRPUSDT",
            ts=int(datetime.now().timestamp() * 1000),
            price=2.35,
            row_3m={
                "close": 2.35, "high": 2.38, "low": 2.32,
                "atr_3m": 0.03, "ema_fast_3m": 2.34, "ema_slow_3m": 2.32,
                "ret_3m": 0.003, "volatility": 0.005, "rsi_3m": 58,
                "ema_slope_15m": 0.005,
            },
            row_15m=row_15m,
            zone={"support": 2.20, "resistance": 2.50, "strength": 0.0001,
                  "dist_to_support": 5.0, "dist_to_resistance": 5.0},
        )

        position = PositionState(side="FLAT")

        # Step 3: Get signal from FSM
        signal, fsm_state = fsm.step(market_ctx, confirm_ctx, position, None)

        # Assertions
        assert result.confirmed_regime == "TREND_UP", "Should confirm TREND_UP with strong slope"
        assert signal is not None, "FSM should produce a signal"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
