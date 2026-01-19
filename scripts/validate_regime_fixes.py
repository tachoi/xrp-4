#!/usr/bin/env python
"""Validation script for regime detection fixes.

Validates the following fixes:
1. HMM state labeling (composite scoring)
2. Confirm Layer RANGE validation
3. XGB feature extraction
4. row_15m data passing

Usage:
    python scripts/validate_regime_fixes.py [--verbose]
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from xrp4.regime.hmm_fast import FastHMM
from xrp4.regime.hmm_mid import MidHMM
from xrp4.regime.hmm_types import RegimeLabel
from xrp4.regime.confirm import RegimeConfirmLayer, ConfirmConfig
from xrp4.regime.hmm_fusion import HMMFusion
from xrp4.core.xgb_gate import XGBApprovalGate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


class ValidationResult:
    """Container for validation results."""

    def __init__(self, name: str):
        self.name = name
        self.passed = 0
        self.failed = 0
        self.errors: List[str] = []

    def add_pass(self, msg: str):
        self.passed += 1
        logger.info(f"  [PASS] {msg}")

    def add_fail(self, msg: str):
        self.failed += 1
        self.errors.append(msg)
        logger.error(f"  [FAIL] {msg}")

    @property
    def success(self) -> bool:
        return self.failed == 0

    def summary(self) -> str:
        status = "PASSED" if self.success else "FAILED"
        return f"{self.name}: {status} ({self.passed} passed, {self.failed} failed)"


def validate_hmm_state_labeling() -> ValidationResult:
    """Validate HMM state labeling with composite scoring."""
    result = ValidationResult("HMM State Labeling")
    logger.info("=" * 60)
    logger.info("Validating HMM State Labeling...")
    logger.info("=" * 60)

    # Test 1: FastHMM unique labels
    try:
        hmm = FastHMM(n_states=4)
        np.random.seed(42)
        n_samples = 500
        features = np.random.randn(n_samples, 4) * 0.01
        features[:, 0] = np.random.randn(n_samples) * 0.005
        feature_names = ["ret_3m", "vol", "ema_slope", "bb_width"]

        hmm.train(features, feature_names)

        labels = list(hmm.state_labels.values())
        if len(labels) == len(set(labels)):
            result.add_pass("FastHMM: All states have unique labels")
        else:
            result.add_fail(f"FastHMM: Duplicate labels: {hmm.state_labels}")
    except Exception as e:
        result.add_fail(f"FastHMM training error: {e}")

    # Test 2: MidHMM unique labels
    try:
        hmm = MidHMM(n_states=4)
        np.random.seed(42)
        features = np.random.randn(500, 4) * 0.01
        feature_names = ["ret_15m", "vol", "ema_slope", "bb_width"]

        hmm.train(features, feature_names)

        labels = list(hmm.state_labels.values())
        if len(labels) == len(set(labels)):
            result.add_pass("MidHMM: All states have unique labels")
        else:
            result.add_fail(f"MidHMM: Duplicate labels: {hmm.state_labels}")
    except Exception as e:
        result.add_fail(f"MidHMM training error: {e}")

    # Test 3: Primary return feature detection
    try:
        hmm = FastHMM(n_states=4)
        hmm.feature_names = ["vol", "ret_3m", "ema_slope"]
        idx = hmm._find_primary_return_feature()
        if idx == 1:
            result.add_pass("Primary return feature detection works (ret_3m)")
        else:
            result.add_fail(f"Primary return feature incorrect: expected 1, got {idx}")
    except Exception as e:
        result.add_fail(f"Primary return feature detection error: {e}")

    return result


def validate_confirm_layer_range() -> ValidationResult:
    """Validate Confirm Layer RANGE validation."""
    result = ValidationResult("Confirm Layer RANGE Validation")
    logger.info("=" * 60)
    logger.info("Validating Confirm Layer RANGE Validation...")
    logger.info("=" * 60)

    confirm = RegimeConfirmLayer(ConfirmConfig())
    # Create hist_15m with varying volatility to establish proper baseline
    # The median will be ~0.004, MAD will be ~0.001
    # V_hi_on = 0.004 + 1.5 * 0.001 = 0.0055
    # So test values below 0.0055 won't trigger HIGH_VOL
    hist_15m = pd.DataFrame({
        "ewm_std_ret_15m": np.concatenate([
            np.full(30, 0.003),
            np.full(40, 0.004),
            np.full(30, 0.005),
        ]),
        "high": [2.35] * 100,
        "low": [2.30] * 100,
        "close": [2.32] * 100,
    })

    # Test 1: RANGE validated with flat slope
    # Use lower volatility than V_hi_on threshold
    row_15m = {
        "ema_slope_15m": 0.0005,
        "ewm_std_ret_15m": 0.003,  # Below V_hi_on threshold
        "ewm_ret_15m": 0.00005,
    }
    res, _ = confirm.confirm("RANGE", row_15m, hist_15m)
    if res.confirmed_regime == "RANGE" and res.reason == "RANGE_VALIDATED":
        result.add_pass("RANGE validated with flat slope")
    else:
        result.add_fail(f"RANGE not validated: {res.confirmed_regime}, {res.reason}")

    # Test 2: RANGE -> TREND_UP on steep positive slope
    row_15m = {
        "ema_slope_15m": 0.005,
        "ewm_std_ret_15m": 0.003,  # Below V_hi_on
        "ewm_ret_15m": 0.00005,
    }
    res, _ = confirm.confirm("RANGE", row_15m, hist_15m)
    if res.confirmed_regime == "TREND_UP":
        result.add_pass("RANGE overridden to TREND_UP on steep slope")
    else:
        result.add_fail(f"RANGE not overridden to TREND_UP: {res.confirmed_regime}, reason: {res.reason}")

    # Test 3: RANGE -> TREND_DOWN on steep negative slope
    row_15m = {
        "ema_slope_15m": -0.005,
        "ewm_std_ret_15m": 0.003,  # Below V_hi_on
        "ewm_ret_15m": 0.00005,
    }
    res, _ = confirm.confirm("RANGE", row_15m, hist_15m)
    if res.confirmed_regime == "TREND_DOWN":
        result.add_pass("RANGE overridden to TREND_DOWN on steep negative slope")
    else:
        result.add_fail(f"RANGE not overridden to TREND_DOWN: {res.confirmed_regime}, reason: {res.reason}")

    # Test 4: RANGE -> TRANSITION on high volatility
    # Note: This test needs high ewm_std_ret_15m that triggers RANGE validation
    # but not HIGH_VOL. RANGE_MAX_VOLATILITY = 0.008, V_hi_on = ~0.0055
    # So we use 0.009 which is > RANGE_MAX_VOLATILITY but < V_hi_on won't work
    # Actually, 0.009 > 0.0055 would trigger HIGH_VOL first
    # We need to test with a value that's > RANGE_MAX_VOLATILITY (0.008) but < V_hi_on
    # That's impossible since V_hi_on = 0.0055 < 0.008
    # So this test needs different hist_15m with higher baseline
    hist_15m_high_vol = pd.DataFrame({
        "ewm_std_ret_15m": np.concatenate([
            np.full(30, 0.006),
            np.full(40, 0.008),
            np.full(30, 0.010),
        ]),  # Baseline ~0.008, MAD ~0.002, V_hi_on = 0.008 + 1.5*0.002 = 0.011
        "high": [2.35] * 100,
        "low": [2.30] * 100,
        "close": [2.32] * 100,
    })
    row_15m = {
        "ema_slope_15m": 0.0005,  # Flat slope
        "ewm_std_ret_15m": 0.009,  # Above RANGE_MAX_VOLATILITY (0.008), below V_hi_on (0.011)
        "ewm_ret_15m": 0.00005,
    }
    res, _ = confirm.confirm("RANGE", row_15m, hist_15m_high_vol)
    if res.confirmed_regime == "TRANSITION":
        result.add_pass("RANGE overridden to TRANSITION on high volatility")
    else:
        result.add_fail(f"RANGE not overridden to TRANSITION: {res.confirmed_regime}, reason: {res.reason}")

    return result


def validate_xgb_feature_extraction() -> ValidationResult:
    """Validate XGB feature extraction fixes."""
    result = ValidationResult("XGB Feature Extraction")
    logger.info("=" * 60)
    logger.info("Validating XGB Feature Extraction...")
    logger.info("=" * 60)

    gate = XGBApprovalGate(model_path=None)

    # Test 1: ret_2 and ret_5 calculated correctly
    row_3m = {
        "close": 100, "close_2": 99, "close_5": 97,
        "high": 101, "low": 99,
        "ret_3m": 0.01,
        "volume": 1000, "volume_ma": 800,
        "ema_fast_3m": 99.5, "ema_slow_3m": 99,
        "rsi_3m": 55, "volatility": 0.005,
    }
    features = gate._extract_features(row_3m, "LONG_TREND_PULLBACK", "TREND_UP")

    ret_2 = features[1]
    ret_5 = features[2]
    expected_ret_2 = (100 - 99) / 99
    expected_ret_5 = (100 - 97) / 97

    if abs(ret_2 - expected_ret_2) < 0.001:
        result.add_pass(f"ret_2 calculated correctly: {ret_2:.4f}")
    else:
        result.add_fail(f"ret_2 incorrect: {ret_2:.4f}, expected {expected_ret_2:.4f}")

    if abs(ret_5 - expected_ret_5) < 0.001:
        result.add_pass(f"ret_5 calculated correctly: {ret_5:.4f}")
    else:
        result.add_fail(f"ret_5 incorrect: {ret_5:.4f}, expected {expected_ret_5:.4f}")

    # Test 2: range_pct calculated from high/low
    range_pct = features[7]
    expected_range_pct = (101 - 99) / 100
    if abs(range_pct - expected_range_pct) < 0.001:
        result.add_pass(f"range_pct calculated correctly: {range_pct:.4f}")
    else:
        result.add_fail(f"range_pct incorrect: {range_pct:.4f}, expected {expected_range_pct:.4f}")

    # Test 3: volume_ratio calculated
    volume_ratio = features[8]
    expected_volume_ratio = 1000 / 800
    if abs(volume_ratio - expected_volume_ratio) < 0.001:
        result.add_pass(f"volume_ratio calculated correctly: {volume_ratio:.4f}")
    else:
        result.add_fail(f"volume_ratio incorrect: {volume_ratio:.4f}, expected {expected_volume_ratio:.4f}")

    # Test 4: ema_slope prefers 3m
    row_3m["ema_slope_3m"] = 0.003
    row_3m["ema_slope_15m"] = 0.001
    features = gate._extract_features(row_3m, "LONG_TREND_PULLBACK", "TREND_UP")
    ema_slope = features[10]
    if abs(ema_slope - 0.003) < 0.0001:
        result.add_pass(f"ema_slope uses 3m value: {ema_slope}")
    else:
        result.add_fail(f"ema_slope should use 3m: {ema_slope}, expected 0.003")

    return result


def validate_hmm_fusion() -> ValidationResult:
    """Validate HMM Fusion information preservation."""
    result = ValidationResult("HMM Fusion")
    logger.info("=" * 60)
    logger.info("Validating HMM Fusion...")
    logger.info("=" * 60)

    from xrp4.regime.hmm_types import HMMPrediction

    fusion = HMMFusion()

    # Test 1: State probs influence regime probs
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
    if regime_probs[RegimeLabel.TREND_UP] == 0.9:
        result.add_pass("Confidence preserved in regime probs")
    else:
        result.add_fail(f"Confidence not preserved: {regime_probs[RegimeLabel.TREND_UP]}")

    # Test 2: Total probability sums to 1
    total = sum(regime_probs.values())
    if abs(total - 1.0) < 0.01:
        result.add_pass(f"Total probability sums to 1: {total:.4f}")
    else:
        result.add_fail(f"Total probability incorrect: {total:.4f}")

    return result


def generate_report(results: List[ValidationResult]) -> Dict:
    """Generate validation report."""
    total_passed = sum(r.passed for r in results)
    total_failed = sum(r.failed for r in results)
    all_passed = all(r.success for r in results)

    report = {
        "timestamp": datetime.now().isoformat(),
        "overall_status": "PASSED" if all_passed else "FAILED",
        "total_passed": total_passed,
        "total_failed": total_failed,
        "sections": [
            {
                "name": r.name,
                "status": "PASSED" if r.success else "FAILED",
                "passed": r.passed,
                "failed": r.failed,
                "errors": r.errors,
            }
            for r in results
        ]
    }

    return report


def main():
    parser = argparse.ArgumentParser(description="Validate regime detection fixes")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--output", "-o", type=str, help="Output file for report")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("=" * 60)
    logger.info("XRP-4 Regime Detection Fixes Validation")
    logger.info("=" * 60)

    results = []

    # Run validations
    results.append(validate_hmm_state_labeling())
    results.append(validate_confirm_layer_range())
    results.append(validate_xgb_feature_extraction())
    results.append(validate_hmm_fusion())

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 60)

    for r in results:
        status = "PASSED" if r.success else "FAILED"
        logger.info(f"  {r.name}: {status} ({r.passed}/{r.passed + r.failed})")

    all_passed = all(r.success for r in results)
    total_passed = sum(r.passed for r in results)
    total_failed = sum(r.failed for r in results)

    logger.info("")
    if all_passed:
        logger.info(f"OVERALL: PASSED ({total_passed} tests)")
    else:
        logger.error(f"OVERALL: FAILED ({total_passed} passed, {total_failed} failed)")

    # Generate report
    report = generate_report(results)

    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Report saved to: {output_path}")
    else:
        # Save to default location
        output_dir = Path(__file__).parent.parent / "outputs" / "validation"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Report saved to: {output_path}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
