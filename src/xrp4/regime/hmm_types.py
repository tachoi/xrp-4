"""HMM Regime Layer Type Definitions.

Provides data structures for Multi-HMM regime detection including:
- RegimeLabel: HMM regime classification
- HMMPrediction: Single HMM output
- RegimePacket: Fused multi-HMM result
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

import numpy as np


class RegimeLabel(Enum):
    """HMM Regime classification labels.

    These are the regime states detected by HMM models.
    Maps to 4-state HMM: RANGE, TREND_UP, TREND_DOWN, SHOCK/TRANSITION
    """
    RANGE = "RANGE"
    TREND_UP = "TREND_UP"
    TREND_DOWN = "TREND_DOWN"
    TRANSITION = "TRANSITION"
    SHOCK = "SHOCK"
    UNKNOWN = "UNKNOWN"

    @classmethod
    def from_state_index(cls, idx: int, n_states: int = 4) -> "RegimeLabel":
        """Map HMM state index to regime label.

        Default mapping for 4-state HMM:
        - 0: RANGE
        - 1: TREND_UP
        - 2: TREND_DOWN
        - 3: TRANSITION (or SHOCK depending on volatility)

        Args:
            idx: State index from HMM
            n_states: Total number of states

        Returns:
            RegimeLabel enum value
        """
        if n_states == 4:
            mapping = {
                0: cls.RANGE,
                1: cls.TREND_UP,
                2: cls.TREND_DOWN,
                3: cls.TRANSITION,
            }
            return mapping.get(idx, cls.UNKNOWN)
        else:
            # For other state counts, use generic mapping
            return cls.UNKNOWN


@dataclass
class HMMPrediction:
    """Single HMM model prediction output.

    Contains the predicted state, probabilities, and uncertainty metrics.
    """
    # Primary prediction
    label: RegimeLabel
    state_idx: int
    confidence: float  # Max posterior probability

    # Full state distribution
    state_probs: np.ndarray  # Shape: (n_states,)

    # Uncertainty metrics
    entropy: float  # Shannon entropy of state distribution
    transition_rate: float  # Recent transition frequency

    # Metadata
    timeframe: str  # "3m" or "15m"
    timestamp: Optional[datetime] = None
    n_states: int = 4

    def __post_init__(self):
        """Validate prediction data."""
        if self.state_probs is not None:
            assert len(self.state_probs) == self.n_states, \
                f"state_probs length {len(self.state_probs)} != n_states {self.n_states}"

    @property
    def is_uncertain(self) -> bool:
        """Check if prediction is uncertain based on confidence."""
        return self.confidence < 0.55

    @property
    def is_transitioning(self) -> bool:
        """Check if in transition based on rate."""
        return self.transition_rate > 0.35

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "label": self.label.value,
            "state_idx": self.state_idx,
            "confidence": float(self.confidence),
            "state_probs": self.state_probs.tolist() if self.state_probs is not None else None,
            "entropy": float(self.entropy),
            "transition_rate": float(self.transition_rate),
            "timeframe": self.timeframe,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "n_states": self.n_states,
        }


@dataclass
class RegimePacket:
    """Fused Multi-HMM regime detection result.

    Combines Fast HMM (3m) and Mid HMM (15m) predictions into a single
    coherent regime assessment.
    """
    # Fused label (primary output)
    label_fused: RegimeLabel

    # Individual HMM predictions
    fast_pred: Optional[HMMPrediction] = None  # 3m HMM
    mid_pred: Optional[HMMPrediction] = None   # 15m HMM

    # Fused uncertainty metrics
    entropy_struct: float = 0.0  # Structural (15m) entropy
    entropy_micro: float = 0.0   # Micro (3m) entropy
    conf_struct: float = 1.0     # Structural confidence
    conf_micro: float = 1.0      # Micro confidence

    # Transition rates
    transition_rate_struct: float = 0.0
    transition_rate_micro: float = 0.0

    # Fusion metadata
    fusion_method: str = "weighted"
    fusion_weight_fast: float = 0.4
    fusion_weight_mid: float = 0.6
    agreement: bool = True  # Whether fast and mid agree

    # Timestamp
    timestamp: Optional[datetime] = None

    # Context flags from HMM analysis
    hmm_flags: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Extract metrics from individual predictions."""
        if self.fast_pred:
            self.entropy_micro = self.fast_pred.entropy
            self.conf_micro = self.fast_pred.confidence
            self.transition_rate_micro = self.fast_pred.transition_rate

        if self.mid_pred:
            self.entropy_struct = self.mid_pred.entropy
            self.conf_struct = self.mid_pred.confidence
            self.transition_rate_struct = self.mid_pred.transition_rate

        # Check agreement
        if self.fast_pred and self.mid_pred:
            self.agreement = (self.fast_pred.label == self.mid_pred.label)

    @property
    def is_shock(self) -> bool:
        """Check if regime is SHOCK."""
        return self.label_fused == RegimeLabel.SHOCK

    @property
    def is_transition(self) -> bool:
        """Check if regime is TRANSITION."""
        return self.label_fused == RegimeLabel.TRANSITION

    @property
    def is_range(self) -> bool:
        """Check if regime is RANGE."""
        return self.label_fused == RegimeLabel.RANGE

    @property
    def is_trend(self) -> bool:
        """Check if regime is TREND_UP or TREND_DOWN."""
        return self.label_fused in (RegimeLabel.TREND_UP, RegimeLabel.TREND_DOWN)

    @property
    def is_uncertain(self) -> bool:
        """Check if HMM prediction is uncertain."""
        return self.conf_struct < 0.55 or self.conf_micro < 0.55

    @property
    def is_high_entropy(self) -> bool:
        """Check if entropy is high (uncertain distribution)."""
        return self.entropy_struct > 1.1 or self.entropy_micro > 0.95

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "label_fused": self.label_fused.value,
            "fast_pred": self.fast_pred.to_dict() if self.fast_pred else None,
            "mid_pred": self.mid_pred.to_dict() if self.mid_pred else None,
            "entropy_struct": float(self.entropy_struct),
            "entropy_micro": float(self.entropy_micro),
            "conf_struct": float(self.conf_struct),
            "conf_micro": float(self.conf_micro),
            "transition_rate_struct": float(self.transition_rate_struct),
            "transition_rate_micro": float(self.transition_rate_micro),
            "fusion_method": self.fusion_method,
            "agreement": self.agreement,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "hmm_flags": self.hmm_flags,
        }

    @classmethod
    def unknown(cls, timestamp: Optional[datetime] = None) -> "RegimePacket":
        """Create an UNKNOWN regime packet."""
        return cls(
            label_fused=RegimeLabel.UNKNOWN,
            timestamp=timestamp,
            hmm_flags=["HMM_UNKNOWN"]
        )


@dataclass
class HMMCheckpoint:
    """HMM model checkpoint for persistence.

    Contains trained model parameters and metadata for saving/loading.
    """
    # Model identification
    run_id: str
    timeframe: str  # "3m" or "15m"
    n_states: int

    # Model parameters (serializable)
    means: np.ndarray          # Shape: (n_states, n_features)
    covars: np.ndarray         # Shape depends on covariance_type
    transmat: np.ndarray       # Shape: (n_states, n_states)
    startprob: np.ndarray      # Shape: (n_states,)

    # Feature configuration
    feature_names: List[str]
    covariance_type: str = "full"

    # Training metadata
    train_start: Optional[datetime] = None
    train_end: Optional[datetime] = None
    n_samples: int = 0
    log_likelihood: float = 0.0

    # State label mapping (learned from data)
    state_labels: Dict[int, str] = field(default_factory=dict)

    # Timestamp
    created_at: Optional[datetime] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        # Convert state_labels keys to regular Python int (JSON compatibility)
        state_labels_clean = {int(k): v for k, v in self.state_labels.items()}

        return {
            "run_id": self.run_id,
            "timeframe": self.timeframe,
            "n_states": int(self.n_states),
            "means": self.means.tolist(),
            "covars": self.covars.tolist(),
            "transmat": self.transmat.tolist(),
            "startprob": self.startprob.tolist(),
            "feature_names": self.feature_names,
            "covariance_type": self.covariance_type,
            "train_start": self.train_start.isoformat() if self.train_start else None,
            "train_end": self.train_end.isoformat() if self.train_end else None,
            "n_samples": int(self.n_samples),
            "log_likelihood": float(self.log_likelihood),
            "state_labels": state_labels_clean,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "HMMCheckpoint":
        """Create from dictionary."""
        return cls(
            run_id=data["run_id"],
            timeframe=data["timeframe"],
            n_states=data["n_states"],
            means=np.array(data["means"]),
            covars=np.array(data["covars"]),
            transmat=np.array(data["transmat"]),
            startprob=np.array(data["startprob"]),
            feature_names=data["feature_names"],
            covariance_type=data.get("covariance_type", "full"),
            train_start=datetime.fromisoformat(data["train_start"]) if data.get("train_start") else None,
            train_end=datetime.fromisoformat(data["train_end"]) if data.get("train_end") else None,
            n_samples=data.get("n_samples", 0),
            log_likelihood=data.get("log_likelihood", 0.0),
            state_labels=data.get("state_labels", {}),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else None,
        )
