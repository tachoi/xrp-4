"""HMM Fusion - Combines Fast HMM (3m) and Mid HMM (15m) predictions.

Implements fusion strategies for combining multi-timeframe HMM predictions
into a single coherent regime assessment (RegimePacket).
"""

import logging
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

from xrp4.regime.hmm_types import (
    HMMPrediction,
    RegimeLabel,
    RegimePacket,
)

logger = logging.getLogger(__name__)


class FusionMethod(Enum):
    """Available fusion methods for combining HMM predictions."""
    WEIGHTED = "weighted"
    VOTING = "voting"
    HIERARCHICAL = "hierarchical"


class ConflictResolution(Enum):
    """Conflict resolution strategies when HMMs disagree."""
    MID_PRIORITY = "mid_priority"
    FAST_PRIORITY = "fast_priority"
    CONSERVATIVE = "conservative"


class HMMFusion:
    """Fuses Fast HMM (3m) and Mid HMM (15m) predictions into RegimePacket.

    Supports multiple fusion strategies:
    - weighted: Weighted average of state probabilities
    - voting: Majority voting with confidence weighting
    - hierarchical: Mid HMM dominant unless micro signal is strong

    Attributes:
        method: Fusion method to use
        fast_weight: Weight for Fast HMM (3m) predictions
        mid_weight: Weight for Mid HMM (15m) predictions
        conflict_resolution: Strategy for handling disagreements
    """

    def __init__(
        self,
        method: str = "weighted",
        fast_weight: float = 0.4,
        mid_weight: float = 0.6,
        conflict_resolution: str = "mid_priority",
    ):
        """Initialize HMM Fusion.

        Args:
            method: Fusion method ("weighted", "voting", "hierarchical")
            fast_weight: Weight for Fast HMM predictions (default 0.4)
            mid_weight: Weight for Mid HMM predictions (default 0.6)
            conflict_resolution: Conflict resolution strategy
        """
        self.method = FusionMethod(method)
        self.fast_weight = fast_weight
        self.mid_weight = mid_weight
        self.conflict_resolution = ConflictResolution(conflict_resolution)

        # Normalize weights
        total = self.fast_weight + self.mid_weight
        if total != 1.0:
            self.fast_weight /= total
            self.mid_weight /= total

    def fuse(
        self,
        fast_pred: Optional[HMMPrediction],
        mid_pred: Optional[HMMPrediction],
        timestamp: Optional[datetime] = None,
    ) -> RegimePacket:
        """Fuse Fast HMM and Mid HMM predictions into RegimePacket.

        Args:
            fast_pred: Prediction from Fast HMM (3m), can be None
            mid_pred: Prediction from Mid HMM (15m), can be None
            timestamp: Timestamp for the fused result

        Returns:
            RegimePacket with fused regime label and metrics
        """
        # Handle missing predictions
        if fast_pred is None and mid_pred is None:
            return RegimePacket.unknown(timestamp)

        if fast_pred is None:
            return self._from_single_pred(mid_pred, timestamp)

        if mid_pred is None:
            return self._from_single_pred(fast_pred, timestamp)

        # Both predictions available - fuse them
        if self.method == FusionMethod.WEIGHTED:
            label_fused = self._weighted_fusion(fast_pred, mid_pred)
        elif self.method == FusionMethod.VOTING:
            label_fused = self._voting_fusion(fast_pred, mid_pred)
        elif self.method == FusionMethod.HIERARCHICAL:
            label_fused = self._hierarchical_fusion(fast_pred, mid_pred)
        else:
            label_fused = self._weighted_fusion(fast_pred, mid_pred)

        # Build flags
        hmm_flags = self._build_flags(fast_pred, mid_pred, label_fused)

        return RegimePacket(
            label_fused=label_fused,
            fast_pred=fast_pred,
            mid_pred=mid_pred,
            fusion_method=self.method.value,
            fusion_weight_fast=self.fast_weight,
            fusion_weight_mid=self.mid_weight,
            timestamp=timestamp,
            hmm_flags=hmm_flags,
        )

    def _weighted_fusion(
        self,
        fast_pred: HMMPrediction,
        mid_pred: HMMPrediction,
    ) -> RegimeLabel:
        """Fuse predictions using weighted probability averaging.

        Computes weighted average of state probabilities and selects
        the regime with highest combined probability.
        """
        # Map state probabilities to regime labels
        fast_regime_probs = self._state_probs_to_regime_probs(fast_pred)
        mid_regime_probs = self._state_probs_to_regime_probs(mid_pred)

        # Weighted average
        fused_probs = {}
        all_labels = set(fast_regime_probs.keys()) | set(mid_regime_probs.keys())

        for label in all_labels:
            fast_p = fast_regime_probs.get(label, 0.0)
            mid_p = mid_regime_probs.get(label, 0.0)
            fused_probs[label] = self.fast_weight * fast_p + self.mid_weight * mid_p

        # Select regime with highest fused probability
        best_label = max(fused_probs, key=fused_probs.get)
        return best_label

    def _voting_fusion(
        self,
        fast_pred: HMMPrediction,
        mid_pred: HMMPrediction,
    ) -> RegimeLabel:
        """Fuse predictions using confidence-weighted voting.

        Each HMM votes for its predicted regime, weighted by confidence.
        """
        fast_vote = fast_pred.confidence * self.fast_weight
        mid_vote = mid_pred.confidence * self.mid_weight

        # If same regime, return it
        if fast_pred.label == mid_pred.label:
            return fast_pred.label

        # Different regimes - use conflict resolution
        return self._resolve_conflict(fast_pred, mid_pred, fast_vote, mid_vote)

    def _hierarchical_fusion(
        self,
        fast_pred: HMMPrediction,
        mid_pred: HMMPrediction,
    ) -> RegimeLabel:
        """Fuse using hierarchical strategy (Mid HMM dominant).

        Mid HMM (structural) takes priority unless:
        - Mid HMM has low confidence
        - Fast HMM detects SHOCK with high confidence
        """
        # SHOCK always takes priority if confident
        if fast_pred.label == RegimeLabel.SHOCK and fast_pred.confidence > 0.7:
            return RegimeLabel.SHOCK

        # If mid is confident, use its prediction
        if mid_pred.confidence > 0.6:
            return mid_pred.label

        # Mid not confident - check if fast has strong signal
        if fast_pred.confidence > 0.75:
            return fast_pred.label

        # Both uncertain - use mid (structural) as default
        return mid_pred.label

    def _resolve_conflict(
        self,
        fast_pred: HMMPrediction,
        mid_pred: HMMPrediction,
        fast_vote: float,
        mid_vote: float,
    ) -> RegimeLabel:
        """Resolve conflict when HMMs disagree."""
        if self.conflict_resolution == ConflictResolution.MID_PRIORITY:
            # Mid wins unless fast has significantly higher confidence
            if fast_vote > mid_vote * 1.5:
                return fast_pred.label
            return mid_pred.label

        elif self.conflict_resolution == ConflictResolution.FAST_PRIORITY:
            # Fast wins unless mid has significantly higher confidence
            if mid_vote > fast_vote * 1.5:
                return mid_pred.label
            return fast_pred.label

        elif self.conflict_resolution == ConflictResolution.CONSERVATIVE:
            # Choose the more "cautious" regime
            priority = {
                RegimeLabel.SHOCK: 0,
                RegimeLabel.TRANSITION: 1,
                RegimeLabel.UNKNOWN: 2,
                RegimeLabel.RANGE: 3,
                RegimeLabel.TREND_DOWN: 4,
                RegimeLabel.TREND_UP: 4,
            }
            fast_priority = priority.get(fast_pred.label, 5)
            mid_priority = priority.get(mid_pred.label, 5)

            if fast_priority < mid_priority:
                return fast_pred.label
            return mid_pred.label

        # Default: higher vote wins
        if fast_vote > mid_vote:
            return fast_pred.label
        return mid_pred.label

    def _state_probs_to_regime_probs(
        self,
        pred: HMMPrediction,
    ) -> Dict[RegimeLabel, float]:
        """Convert state probabilities to regime label probabilities.

        Preserves information from state_probs instead of uniform distribution.
        Uses entropy-based weighting to reflect uncertainty more accurately.
        """
        regime_probs: Dict[RegimeLabel, float] = {label: 0.0 for label in RegimeLabel}

        # If we have state_probs, use them to distribute probability
        if pred.state_probs is not None and len(pred.state_probs) > 0:
            # The prediction already maps state_idx to a label
            # Use the state_probs to build regime probabilities
            #
            # Since we don't have access to the state->label mapping here,
            # we use a heuristic: the confidence is for the predicted label,
            # and remaining probability is distributed based on state_probs entropy

            regime_probs[pred.label] = pred.confidence

            # Calculate how to distribute remaining probability
            remaining = 1.0 - pred.confidence
            if remaining > 0:
                # Get sorted state probabilities (excluding the top one)
                sorted_probs = sorted(pred.state_probs, reverse=True)

                # Use the second-highest probability to weight the "next most likely" regime
                if len(sorted_probs) > 1 and sorted_probs[1] > 0.01:
                    # Weight based on entropy - higher entropy means more uncertainty
                    # If entropy is high, distribute more evenly
                    # If entropy is low (confident), give less to others
                    entropy_ratio = pred.entropy / np.log(pred.n_states)  # Normalized entropy (0-1)

                    # Determine likely alternative labels based on the main label
                    if pred.label == RegimeLabel.RANGE:
                        # RANGE alternatives: could be TRANSITION or weak TREND
                        alt_labels = [RegimeLabel.TRANSITION, RegimeLabel.TREND_UP, RegimeLabel.TREND_DOWN]
                    elif pred.label == RegimeLabel.TREND_UP:
                        # TREND_UP alternatives: could be RANGE or TRANSITION
                        alt_labels = [RegimeLabel.RANGE, RegimeLabel.TRANSITION, RegimeLabel.TREND_DOWN]
                    elif pred.label == RegimeLabel.TREND_DOWN:
                        # TREND_DOWN alternatives: could be RANGE or TRANSITION
                        alt_labels = [RegimeLabel.RANGE, RegimeLabel.TRANSITION, RegimeLabel.TREND_UP]
                    elif pred.label == RegimeLabel.TRANSITION:
                        # TRANSITION alternatives: could be any direction
                        alt_labels = [RegimeLabel.RANGE, RegimeLabel.TREND_UP, RegimeLabel.TREND_DOWN]
                    else:
                        alt_labels = [l for l in RegimeLabel if l != pred.label and l != RegimeLabel.UNKNOWN]

                    # Distribute remaining probability with entropy weighting
                    if alt_labels:
                        # Give more weight to the first alternative when entropy is low
                        weights = [1.0 - entropy_ratio * 0.5]  # First alt gets more
                        for _ in range(len(alt_labels) - 1):
                            weights.append(entropy_ratio / (len(alt_labels) - 1) if len(alt_labels) > 1 else 0)

                        # Normalize weights
                        weight_sum = sum(weights)
                        if weight_sum > 0:
                            for i, label in enumerate(alt_labels):
                                if i < len(weights):
                                    regime_probs[label] = remaining * weights[i] / weight_sum
                else:
                    # Very low entropy - distribute uniformly to others
                    other_labels = [l for l in RegimeLabel if l != pred.label and l != RegimeLabel.UNKNOWN]
                    if other_labels:
                        per_label = remaining / len(other_labels)
                        for label in other_labels:
                            regime_probs[label] = per_label
        else:
            # Fallback: use confidence and uniform distribution for rest
            regime_probs[pred.label] = pred.confidence
            remaining = 1.0 - pred.confidence
            other_labels = [l for l in RegimeLabel if l != pred.label and l != RegimeLabel.UNKNOWN]
            if other_labels:
                per_label = remaining / len(other_labels)
                for label in other_labels:
                    regime_probs[label] = per_label

        return regime_probs

    def _from_single_pred(
        self,
        pred: HMMPrediction,
        timestamp: Optional[datetime] = None,
    ) -> RegimePacket:
        """Create RegimePacket from single prediction when one HMM is missing."""
        is_fast = pred.timeframe == "3m"

        return RegimePacket(
            label_fused=pred.label,
            fast_pred=pred if is_fast else None,
            mid_pred=pred if not is_fast else None,
            fusion_method="single",
            fusion_weight_fast=1.0 if is_fast else 0.0,
            fusion_weight_mid=0.0 if is_fast else 1.0,
            timestamp=timestamp,
            hmm_flags=[f"SINGLE_{pred.timeframe.upper()}"],
        )

    def _build_flags(
        self,
        fast_pred: HMMPrediction,
        mid_pred: HMMPrediction,
        label_fused: RegimeLabel,
    ) -> List[str]:
        """Build HMM analysis flags for the fused result."""
        flags = []

        # Agreement flag
        if fast_pred.label == mid_pred.label:
            flags.append("HMM_AGREE")
        else:
            flags.append("HMM_DISAGREE")

        # Uncertainty flags
        if fast_pred.is_uncertain:
            flags.append("FAST_UNCERTAIN")
        if mid_pred.is_uncertain:
            flags.append("MID_UNCERTAIN")

        # Transition flags
        if fast_pred.is_transitioning:
            flags.append("FAST_TRANSITIONING")
        if mid_pred.is_transitioning:
            flags.append("MID_TRANSITIONING")

        # High entropy flags
        if fast_pred.entropy > 0.95:
            flags.append("FAST_HIGH_ENTROPY")
        if mid_pred.entropy > 1.10:
            flags.append("MID_HIGH_ENTROPY")

        # Regime-specific flags
        if label_fused == RegimeLabel.SHOCK:
            flags.append("REGIME_SHOCK")
        elif label_fused == RegimeLabel.TRANSITION:
            flags.append("REGIME_TRANSITION")
        elif label_fused == RegimeLabel.RANGE:
            flags.append("REGIME_RANGE")
        elif label_fused == RegimeLabel.TREND_UP:
            flags.append("REGIME_TREND_UP")
        elif label_fused == RegimeLabel.TREND_DOWN:
            flags.append("REGIME_TREND_DOWN")

        return flags

    def update_config(
        self,
        method: Optional[str] = None,
        fast_weight: Optional[float] = None,
        mid_weight: Optional[float] = None,
        conflict_resolution: Optional[str] = None,
    ) -> None:
        """Update fusion configuration.

        Args:
            method: New fusion method
            fast_weight: New fast HMM weight
            mid_weight: New mid HMM weight
            conflict_resolution: New conflict resolution strategy
        """
        if method is not None:
            self.method = FusionMethod(method)

        if fast_weight is not None:
            self.fast_weight = fast_weight

        if mid_weight is not None:
            self.mid_weight = mid_weight

        # Normalize weights
        total = self.fast_weight + self.mid_weight
        if total != 1.0:
            self.fast_weight /= total
            self.mid_weight /= total

        if conflict_resolution is not None:
            self.conflict_resolution = ConflictResolution(conflict_resolution)

        logger.info(
            f"HMM Fusion config updated: method={self.method.value}, "
            f"fast_weight={self.fast_weight:.2f}, mid_weight={self.mid_weight:.2f}, "
            f"conflict={self.conflict_resolution.value}"
        )


def create_fusion_from_config(config: Dict) -> HMMFusion:
    """Create HMMFusion from configuration dictionary.

    Args:
        config: Configuration dict with fusion settings

    Returns:
        Configured HMMFusion instance
    """
    fusion_cfg = config.get("fusion", {})

    return HMMFusion(
        method=fusion_cfg.get("method", "weighted"),
        fast_weight=fusion_cfg.get("fast_weight", 0.4),
        mid_weight=fusion_cfg.get("mid_weight", 0.6),
        conflict_resolution=fusion_cfg.get("conflict_resolution", "mid_priority"),
    )
