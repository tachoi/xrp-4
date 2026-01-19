"""Unit tests for HMM state labeling and Confirm Layer.

Tests the fixes for:
1. HMM composite scoring for state labeling
2. Confirm Layer RANGE validation
3. XGB feature extraction
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import Mock, MagicMock

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from xrp4.regime.hmm_types import RegimeLabel, HMMPrediction
from xrp4.regime.confirm import RegimeConfirmLayer, ConfirmConfig, ConfirmResult


class TestHMMStateLabeling:
    """Test HMM state labeling with composite scoring."""

    def test_fast_hmm_state_labels_unique(self):
        """All states should have unique labels."""
        from xrp4.regime.hmm_fast import FastHMM

        # Create and train HMM with synthetic data
        hmm = FastHMM(n_states=4)

        # Generate synthetic features with clear patterns
        np.random.seed(42)
        n_samples = 500
        features = np.random.randn(n_samples, 4) * 0.01
        features[:, 0] = np.random.randn(n_samples) * 0.005  # ret
        feature_names = ["ret_3m", "vol", "ema_slope", "bb_width"]

        hmm.train(features, feature_names)

        # Check unique labels
        labels = list(hmm.state_labels.values())
        assert len(labels) == len(set(labels)), f"Duplicate labels: {hmm.state_labels}"

    def test_mid_hmm_state_labels_unique(self):
        """All states should have unique labels."""
        from xrp4.regime.hmm_mid import MidHMM

        hmm = MidHMM(n_states=4)

        np.random.seed(42)
        n_samples = 500
        features = np.random.randn(n_samples, 4) * 0.01
        features[:, 0] = np.random.randn(n_samples) * 0.005
        feature_names = ["ret_15m", "vol", "ema_slope", "bb_width"]

        hmm.train(features, feature_names)

        labels = list(hmm.state_labels.values())
        assert len(labels) == len(set(labels)), f"Duplicate labels: {hmm.state_labels}"

    def test_find_primary_return_feature_exact_match(self):
        """Should prefer exact ret_3m/ret_15m matches."""
        from xrp4.regime.hmm_fast import FastHMM

        hmm = FastHMM(n_states=4)
        hmm.feature_names = ["vol", "ret_3m", "ema_slope", "other_ret"]

        idx = hmm._find_primary_return_feature()
        assert idx == 1, f"Expected idx 1 (ret_3m), got {idx}"

    def test_find_primary_return_feature_fallback(self):
        """Should fall back to first 'ret' prefix match."""
        from xrp4.regime.hmm_fast import FastHMM

        hmm = FastHMM(n_states=4)
        hmm.feature_names = ["vol", "ema_slope", "ret_custom", "bb_width"]

        idx = hmm._find_primary_return_feature()
        assert idx == 2, f"Expected idx 2 (ret_custom), got {idx}"


class TestConfirmLayerRangeValidation:
    """Test Confirm Layer RANGE validation."""

    @pytest.fixture
    def confirm_layer(self):
        """Create ConfirmLayer with default config."""
        return RegimeConfirmLayer(ConfirmConfig())

    @pytest.fixture
    def empty_hist(self):
        """Create history DataFrame with varying volatility to avoid HIGH_VOL triggering.

        Baseline ~0.004, MAD ~0.001, V_hi_on = 0.004 + 1.5*0.001 = 0.0055
        Test values should be < 0.0055 to not trigger HIGH_VOL
        """
        return pd.DataFrame({
            "ewm_std_ret_15m": np.concatenate([
                np.full(30, 0.003),
                np.full(40, 0.004),
                np.full(30, 0.005),
            ])
        })

    def test_range_validated_flat_slope(self, confirm_layer, empty_hist):
        """RANGE should be validated when EMA slope is flat."""
        row_15m = {
            "ema_slope_15m": 0.0005,  # Below threshold
            "ewm_std_ret_15m": 0.003,  # Below volatility threshold
            "ewm_ret_15m": 0.00005,    # Below drift threshold
        }

        result, _ = confirm_layer.confirm(
            regime_raw="RANGE",
            row_15m=row_15m,
            hist_15m=empty_hist,
        )

        assert result.confirmed_regime == "RANGE"
        assert result.reason == "RANGE_VALIDATED"

    def test_range_override_to_trend_up_on_slope(self, confirm_layer, empty_hist):
        """RANGE should be overridden to TREND_UP when EMA slope is steep."""
        row_15m = {
            "ema_slope_15m": 0.005,    # Above threshold (positive)
            "ewm_std_ret_15m": 0.003,
            "ewm_ret_15m": 0.00005,
        }

        result, _ = confirm_layer.confirm(
            regime_raw="RANGE",
            row_15m=row_15m,
            hist_15m=empty_hist,
        )

        assert result.confirmed_regime == "TREND_UP"
        assert result.reason == "RANGE_OVERRIDE_TREND_SLOPE"

    def test_range_override_to_trend_down_on_slope(self, confirm_layer, empty_hist):
        """RANGE should be overridden to TREND_DOWN when EMA slope is steep negative."""
        row_15m = {
            "ema_slope_15m": -0.005,   # Above threshold (negative)
            "ewm_std_ret_15m": 0.003,
            "ewm_ret_15m": 0.00005,
        }

        result, _ = confirm_layer.confirm(
            regime_raw="RANGE",
            row_15m=row_15m,
            hist_15m=empty_hist,
        )

        assert result.confirmed_regime == "TREND_DOWN"
        assert result.reason == "RANGE_OVERRIDE_TREND_SLOPE"

    def test_range_override_to_transition_on_high_vol(self, confirm_layer):
        """RANGE should be overridden to TRANSITION when volatility is high.

        Need higher baseline to avoid HIGH_VOL triggering before RANGE validation.
        """
        # Create hist_15m with higher baseline
        # Baseline ~0.008, MAD ~0.002, V_hi_on = 0.008 + 1.5*0.002 = 0.011
        hist_15m_high_baseline = pd.DataFrame({
            "ewm_std_ret_15m": np.concatenate([
                np.full(30, 0.006),
                np.full(40, 0.008),
                np.full(30, 0.010),
            ])
        })

        row_15m = {
            "ema_slope_15m": 0.0005,
            "ewm_std_ret_15m": 0.009,  # Above RANGE_MAX_VOLATILITY (0.008), below V_hi_on (0.011)
            "ewm_ret_15m": 0.00005,
        }

        result, _ = confirm_layer.confirm(
            regime_raw="RANGE",
            row_15m=row_15m,
            hist_15m=hist_15m_high_baseline,
        )

        assert result.confirmed_regime == "TRANSITION"
        assert result.reason == "RANGE_OVERRIDE_TRANSITION_HIGH_VOL"

    def test_range_override_to_trend_on_drift(self, confirm_layer, empty_hist):
        """RANGE should be overridden to TREND when drift is high."""
        row_15m = {
            "ema_slope_15m": 0.0005,
            "ewm_std_ret_15m": 0.003,
            "ewm_ret_15m": 0.001,     # Above drift threshold
        }

        result, _ = confirm_layer.confirm(
            regime_raw="RANGE",
            row_15m=row_15m,
            hist_15m=empty_hist,
        )

        assert result.confirmed_regime == "TREND_UP"
        assert result.reason == "RANGE_OVERRIDE_TREND_DRIFT"

    def test_trend_validated_strong_slope(self, confirm_layer, empty_hist):
        """TREND should be validated when EMA slope is strong."""
        row_15m = {
            "ema_slope_15m": 0.005,  # Above threshold
        }

        result, _ = confirm_layer.confirm(
            regime_raw="TREND_UP",
            row_15m=row_15m,
            hist_15m=empty_hist,
        )

        assert result.confirmed_regime == "TREND_UP"
        assert result.reason == "TREND_UP_VALIDATED"

    def test_trend_override_to_range_weak_slope(self, confirm_layer, empty_hist):
        """TREND should be overridden to RANGE when EMA slope is weak."""
        row_15m = {
            "ema_slope_15m": 0.0005,  # Below threshold
        }

        result, _ = confirm_layer.confirm(
            regime_raw="TREND_UP",
            row_15m=row_15m,
            hist_15m=empty_hist,
        )

        assert result.confirmed_regime == "RANGE"
        assert result.reason == "TREND_UP_OVERRIDE_RANGE_LOW_SLOPE"


class TestXGBFeatureExtraction:
    """Test XGB feature extraction fixes."""

    def test_ret_2_ret_5_calculated_from_close(self):
        """ret_2 and ret_5 should be calculated from historical close prices."""
        from xrp4.core.xgb_gate import XGBApprovalGate

        gate = XGBApprovalGate(model_path=None)  # Don't load model

        row_3m = {
            "close": 100,
            "close_2": 99,   # 2 bars ago
            "close_5": 97,   # 5 bars ago
            "ret_3m": 0.01,
            "high": 101,
            "low": 99,
            "volume": 1000,
            "volume_ma": 800,
            "ema_fast_3m": 99.5,
            "ema_slow_3m": 99,
            "rsi_3m": 55,
            "volatility": 0.005,
        }

        features = gate._extract_features(row_3m, "LONG_TREND_PULLBACK", "TREND_UP")

        ret = features[0]
        ret_2 = features[1]
        ret_5 = features[2]

        # ret_2 = (100 - 99) / 99 = 0.0101
        expected_ret_2 = (100 - 99) / 99
        # ret_5 = (100 - 97) / 97 = 0.0309
        expected_ret_5 = (100 - 97) / 97

        assert abs(ret_2 - expected_ret_2) < 0.001, f"ret_2: {ret_2}, expected: {expected_ret_2}"
        assert abs(ret_5 - expected_ret_5) < 0.001, f"ret_5: {ret_5}, expected: {expected_ret_5}"
        assert ret_2 != ret, "ret_2 should differ from ret"
        assert ret_5 != ret, "ret_5 should differ from ret"

    def test_range_pct_calculated_from_high_low(self):
        """range_pct should be calculated from actual high/low."""
        from xrp4.core.xgb_gate import XGBApprovalGate

        gate = XGBApprovalGate(model_path=None)

        row_3m = {
            "close": 100,
            "high": 102,
            "low": 98,
            "ret_3m": 0.01,
            "volume": 1000,
            "volume_ma": 800,
            "ema_fast_3m": 99.5,
            "ema_slow_3m": 99,
            "rsi_3m": 55,
            "volatility": 0.005,
        }

        features = gate._extract_features(row_3m, "LONG_TREND_PULLBACK", "TREND_UP")

        range_pct = features[7]  # Index 7 is range_pct
        expected = (102 - 98) / 100  # 0.04

        assert abs(range_pct - expected) < 0.001, f"range_pct: {range_pct}, expected: {expected}"

    def test_volume_ratio_calculated(self):
        """volume_ratio should be calculated from volume/volume_ma."""
        from xrp4.core.xgb_gate import XGBApprovalGate

        gate = XGBApprovalGate(model_path=None)

        row_3m = {
            "close": 100,
            "high": 101,
            "low": 99,
            "ret_3m": 0.01,
            "volume": 1000,
            "volume_ma": 800,
            "ema_fast_3m": 99.5,
            "ema_slow_3m": 99,
            "rsi_3m": 55,
            "volatility": 0.005,
        }

        features = gate._extract_features(row_3m, "LONG_TREND_PULLBACK", "TREND_UP")

        volume_ratio = features[8]  # Index 8 is volume_ratio
        expected = 1000 / 800  # 1.25

        assert abs(volume_ratio - expected) < 0.001, f"volume_ratio: {volume_ratio}, expected: {expected}"

    def test_ema_slope_prefers_3m(self):
        """ema_slope should prefer 3m over 15m."""
        from xrp4.core.xgb_gate import XGBApprovalGate

        gate = XGBApprovalGate(model_path=None)

        row_3m = {
            "close": 100,
            "high": 101,
            "low": 99,
            "ret_3m": 0.01,
            "volume": 1000,
            "volume_ma": 800,
            "ema_fast_3m": 99.5,
            "ema_slow_3m": 99,
            "rsi_3m": 55,
            "volatility": 0.005,
            "ema_slope_3m": 0.003,    # 3m slope
            "ema_slope_15m": 0.001,   # 15m slope (should be ignored)
        }

        features = gate._extract_features(row_3m, "LONG_TREND_PULLBACK", "TREND_UP")

        ema_slope = features[10]  # Index 10 is ema_slope
        expected = 0.003  # Should use 3m

        assert abs(ema_slope - expected) < 0.0001, f"ema_slope: {ema_slope}, expected: {expected}"


class TestHMMFusion:
    """Test HMM Fusion information preservation."""

    def test_state_probs_preserved_in_regime_probs(self):
        """State probs should influence regime probability distribution."""
        from xrp4.regime.hmm_fusion import HMMFusion

        fusion = HMMFusion()

        # Create prediction with high confidence (low entropy)
        pred = HMMPrediction(
            label=RegimeLabel.TREND_UP,
            state_idx=1,
            confidence=0.9,
            state_probs=np.array([0.05, 0.9, 0.03, 0.02]),
            entropy=0.5,
            transition_rate=0.1,
            timeframe="3m",
            n_states=4,
        )

        regime_probs = fusion._state_probs_to_regime_probs(pred)

        # TREND_UP should have highest probability
        assert regime_probs[RegimeLabel.TREND_UP] == 0.9

        # Remaining probability should be distributed
        total = sum(regime_probs.values())
        assert abs(total - 1.0) < 0.01, f"Total probability: {total}"

    def test_high_entropy_distributes_more_evenly(self):
        """High entropy should distribute remaining probability more evenly."""
        from xrp4.regime.hmm_fusion import HMMFusion

        fusion = HMMFusion()

        # High entropy prediction
        pred = HMMPrediction(
            label=RegimeLabel.RANGE,
            state_idx=0,
            confidence=0.5,
            state_probs=np.array([0.5, 0.2, 0.2, 0.1]),
            entropy=1.2,  # High entropy
            transition_rate=0.3,
            timeframe="3m",
            n_states=4,
        )

        regime_probs = fusion._state_probs_to_regime_probs(pred)

        # RANGE should have the confidence
        assert regime_probs[RegimeLabel.RANGE] == 0.5

        # Check that remaining is distributed (not all zero)
        # Note: SHOCK may not be in the distribution, so filter properly
        non_range_probs = [v for k, v in regime_probs.items()
                          if k != RegimeLabel.RANGE and k != RegimeLabel.UNKNOWN and k != RegimeLabel.SHOCK]
        assert len(non_range_probs) > 0, "Should have non-RANGE probabilities"
        assert all(p >= 0 for p in non_range_probs), "Probabilities should be non-negative"
        assert sum(non_range_probs) > 0, "Some probability should be distributed to other regimes"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
