"""Mid HMM (15m timeframe) for structural regime detection.

Implements a 4-state Gaussian HMM for structural regime classification
on 15-minute OHLCV data. Provides more stable, longer-term regime signals
compared to FastHMM.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from hmmlearn import hmm
from scipy.stats import entropy

from xrp4.regime.hmm_types import (
    HMMCheckpoint,
    HMMPrediction,
    RegimeLabel,
)

logger = logging.getLogger(__name__)


class MidHMM:
    """Mid HMM model for 15-minute timeframe structural regime detection.

    Uses a 4-state Gaussian HMM to classify market regimes with emphasis
    on structural, longer-term regime identification:
    - State 0: RANGE (consolidation, low directional bias)
    - State 1: TREND_UP (sustained bullish momentum)
    - State 2: TREND_DOWN (sustained bearish momentum)
    - State 3: TRANSITION (structural change / high uncertainty)

    The 15m HMM is designed to be more stable than the 3m HMM, capturing
    structural market conditions rather than micro-fluctuations.

    Attributes:
        n_states: Number of HMM states (default 4)
        timeframe: Timeframe identifier (always "15m")
        model: The underlying GaussianHMM model
        feature_names: List of feature names used for training
        state_labels: Mapping from state index to RegimeLabel
    """

    def __init__(
        self,
        n_states: int = 4,
        covariance_type: str = "full",
        n_iter: int = 150,  # More iterations for structural stability
        random_state: int = 42,
    ):
        """Initialize MidHMM.

        Args:
            n_states: Number of hidden states
            covariance_type: Type of covariance ("full", "diag", "tied", "spherical")
            n_iter: Maximum number of EM iterations
            random_state: Random seed for reproducibility
        """
        self.n_states = n_states
        self.timeframe = "15m"
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.random_state = random_state

        self.model: Optional[hmm.GaussianHMM] = None
        self.feature_names: List[str] = []
        self.state_labels: Dict[int, RegimeLabel] = {}
        self._is_trained = False

        # Training metadata
        self.train_start: Optional[datetime] = None
        self.train_end: Optional[datetime] = None
        self.n_samples: int = 0
        self.log_likelihood: float = 0.0

        # State history for transition rate calculation
        # Shorter window for 15m since each bar is 5x longer
        self._state_history: List[int] = []
        self._max_history_len = 20

        # Feature scaling parameters
        self._feature_mean: Optional[np.ndarray] = None
        self._feature_std: Optional[np.ndarray] = None

    def _create_model(self) -> hmm.GaussianHMM:
        """Create a new GaussianHMM model."""
        return hmm.GaussianHMM(
            n_components=self.n_states,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=self.random_state,
            verbose=False,
        )

    def train(
        self,
        features: np.ndarray,
        feature_names: List[str],
        timestamps: Optional[pd.DatetimeIndex] = None,
    ) -> float:
        """Train the HMM on feature data.

        Args:
            features: Feature matrix, shape (n_samples, n_features)
            feature_names: Names of the features
            timestamps: Optional timestamps for metadata

        Returns:
            Log-likelihood of the training data

        Raises:
            ValueError: If features are invalid or have insufficient samples
        """
        if features.ndim != 2:
            raise ValueError(f"Features must be 2D, got shape {features.shape}")

        n_samples, n_features = features.shape
        # 15m requires fewer samples since each represents more time
        min_samples = self.n_states * 8
        if n_samples < min_samples:
            raise ValueError(
                f"Insufficient samples: {n_samples} < {min_samples} minimum"
            )

        # Handle NaN values
        features = features.copy()
        if np.any(np.isnan(features)):
            logger.warning("NaN values in features, filling with column means")
            col_means = np.nanmean(features, axis=0)
            for i in range(n_features):
                mask = np.isnan(features[:, i])
                features[mask, i] = col_means[i]

        # Handle inf values
        if np.any(np.isinf(features)):
            logger.warning("Inf values in features, clipping")
            features = np.clip(features, -1e10, 1e10)

        # Standardize features
        self._feature_mean = np.mean(features, axis=0)
        self._feature_std = np.std(features, axis=0)
        self._feature_std[self._feature_std == 0] = 1.0
        features_scaled = (features - self._feature_mean) / self._feature_std

        # Create and fit model
        self.model = self._create_model()
        self.model.fit(features_scaled)

        # Store metadata
        self.feature_names = feature_names.copy()
        self.n_samples = n_samples
        self.log_likelihood = self.model.score(features_scaled)

        if timestamps is not None and len(timestamps) > 0:
            self.train_start = timestamps[0].to_pydatetime() if hasattr(timestamps[0], 'to_pydatetime') else timestamps[0]
            self.train_end = timestamps[-1].to_pydatetime() if hasattr(timestamps[-1], 'to_pydatetime') else timestamps[-1]

        # Map states to regime labels
        self._map_states_to_labels(features_scaled)

        self._is_trained = True
        logger.info(
            f"MidHMM trained: {n_samples} samples, {n_features} features, "
            f"log_likelihood={self.log_likelihood:.2f}"
        )

        return self.log_likelihood

    def _map_states_to_labels(self, features: np.ndarray) -> None:
        """Map HMM states to regime labels using composite scoring.

        For the structural 15m HMM, uses multiple metrics:
        - RANGE: high persistence (self-transition) + low directional drift
        - TRANSITION: low persistence + high variance
        - TREND_UP: positive drift + high persistence
        - TREND_DOWN: negative drift + high persistence

        This replaces the incorrect "lowest variance = RANGE" assumption.
        """
        if self.model is None:
            return

        means = self.model.means_
        transmat = self.model.transmat_

        # Get self-transition probabilities (persistence)
        self_trans = np.diag(transmat)

        # Get state variances
        covars = self.model.covars_
        if self.covariance_type == "full":
            variances = np.array([np.trace(c) for c in covars])
        elif self.covariance_type == "diag":
            if covars.ndim == 2:
                variances = np.sum(covars, axis=1)
            else:
                variances = np.ones(self.n_states)
        else:
            variances = np.ones(self.n_states)

        if len(variances) != self.n_states:
            variances = np.ones(self.n_states)

        # Find primary return feature
        return_idx = self._find_primary_return_feature()
        state_returns = means[:, return_idx]

        # Normalize metrics for scoring
        var_norm = (variances - variances.min()) / (variances.max() - variances.min() + 1e-10)
        trans_norm = self_trans  # Already 0-1

        # Composite scores for each regime type
        # RANGE: high persistence + low absolute drift (direction-neutral)
        range_score = trans_norm - np.abs(state_returns) * 10

        # TRANSITION: low persistence + high variance
        transition_score = var_norm - trans_norm

        # Greedy assignment (avoid duplicate labels)
        assigned = set()
        self.state_labels = {}

        # 1. TRANSITION first (usually most distinct)
        trans_score_copy = transition_score.copy()
        trans_state = int(np.argmax(trans_score_copy))
        self.state_labels[trans_state] = RegimeLabel.TRANSITION
        assigned.add(trans_state)

        # 2. RANGE (high persistence, low drift, excluding TRANSITION)
        range_score_copy = range_score.copy()
        range_score_copy[list(assigned)] = -np.inf
        range_state = int(np.argmax(range_score_copy))
        self.state_labels[range_state] = RegimeLabel.RANGE
        assigned.add(range_state)

        # 3. TREND_UP and TREND_DOWN from remaining states
        # Use ABSOLUTE SIGN of return mean, not just relative ordering
        # This ensures correct labeling regardless of market bias in training data
        remaining = [i for i in range(self.n_states) if i not in assigned]

        if len(remaining) >= 2:
            # Separate by return sign first
            positive_states = [s for s in remaining if state_returns[s] > 0]
            negative_states = [s for s in remaining if state_returns[s] <= 0]

            if positive_states and negative_states:
                # Ideal case: have both positive and negative return states
                # Pick the most extreme of each
                best_up = max(positive_states, key=lambda s: state_returns[s])
                best_down = min(negative_states, key=lambda s: state_returns[s])
                self.state_labels[best_up] = RegimeLabel.TREND_UP
                self.state_labels[best_down] = RegimeLabel.TREND_DOWN
            else:
                # Fallback: all same sign - use relative ordering but log warning
                remaining_sorted = sorted(remaining, key=lambda s: state_returns[s])
                self.state_labels[remaining_sorted[0]] = RegimeLabel.TREND_DOWN
                self.state_labels[remaining_sorted[-1]] = RegimeLabel.TREND_UP
                logger.warning(
                    f"MidHMM: All remaining states have same sign returns! "
                    f"Returns: {[state_returns[s] for s in remaining_sorted]}. "
                    f"Training data may be biased."
                )
        elif len(remaining) == 1:
            state = remaining[0]
            if state_returns[state] > 0:
                self.state_labels[state] = RegimeLabel.TREND_UP
            else:
                self.state_labels[state] = RegimeLabel.TREND_DOWN

        # Validate unique labels
        labels = list(self.state_labels.values())
        if len(labels) != len(set(labels)):
            logger.warning(f"MidHMM: Duplicate labels detected: {self.state_labels}")

        logger.debug(f"MidHMM state label mapping: {self.state_labels}")
        logger.debug(f"  self_trans: {self_trans}")
        logger.debug(f"  variances: {variances}")
        logger.debug(f"  state_returns: {state_returns}")

    def _find_primary_return_feature(self) -> int:
        """Find the index of the primary return feature.

        Prefers exact matches (ret_15m, ret_1h) over partial matches.
        """
        # Exact matches first
        for i, name in enumerate(self.feature_names):
            if name == "ret_15m" or name == "ret_1h":
                return i

        # Partial matches as fallback
        for i, name in enumerate(self.feature_names):
            if name.startswith("ret") or "return" in name.lower():
                return i

        # Default to first feature
        return 0

    def predict(
        self,
        features: np.ndarray,
        timestamp: Optional[datetime] = None,
    ) -> HMMPrediction:
        """Predict structural regime for given features.

        Args:
            features: Feature vector, shape (1, n_features) or (n_features,)
            timestamp: Optional timestamp for the prediction

        Returns:
            HMMPrediction with regime label and uncertainty metrics
        """
        if not self._is_trained or self.model is None:
            raise RuntimeError("Model must be trained before prediction")

        # Reshape if needed
        if features.ndim == 1:
            features = features.reshape(1, -1)

        if features.shape[1] != len(self.feature_names):
            raise ValueError(
                f"Feature count mismatch: got {features.shape[1]}, "
                f"expected {len(self.feature_names)}"
            )

        # Handle NaN/Inf values
        features = features.copy()
        if np.any(np.isnan(features)):
            for i in range(features.shape[1]):
                if np.isnan(features[0, i]):
                    features[0, i] = self._feature_mean[i]

        if np.any(np.isinf(features)):
            features = np.clip(features, -1e10, 1e10)

        # Standardize and clip to handle distribution shift
        features_scaled = (features - self._feature_mean) / self._feature_std
        features_scaled = np.clip(features_scaled, -4.0, 4.0)

        # Get state prediction
        log_prob, state_seq = self.model.decode(features_scaled, algorithm="viterbi")
        state_idx = state_seq[0]

        # Get posterior probabilities
        state_probs = self.model.predict_proba(features_scaled)[0]

        # Handle degenerate case where one state has all probability (numerical issue)
        if np.max(state_probs) > 0.9999:
            # Use forward algorithm to get better probability estimates
            log_proba = self.model.score_samples(features_scaled)[1]
            state_probs = np.exp(log_proba[0] - np.max(log_proba[0]))
            state_probs = state_probs / (state_probs.sum() + 1e-10)

        # Calculate entropy (structural entropy)
        state_entropy = entropy(state_probs + 1e-10)

        # Update state history
        self._state_history.append(state_idx)
        if len(self._state_history) > self._max_history_len:
            self._state_history.pop(0)

        transition_rate = self._calculate_transition_rate()

        label = self.state_labels.get(state_idx, RegimeLabel.UNKNOWN)

        return HMMPrediction(
            label=label,
            state_idx=state_idx,
            confidence=float(state_probs[state_idx]),
            state_probs=state_probs,
            entropy=float(state_entropy),
            transition_rate=transition_rate,
            timeframe=self.timeframe,
            timestamp=timestamp,
            n_states=self.n_states,
        )

    def predict_sequence(
        self,
        features: np.ndarray,
        timestamps: Optional[pd.DatetimeIndex] = None,
    ) -> List[HMMPrediction]:
        """Predict structural regime for a sequence of feature vectors.

        Args:
            features: Feature matrix, shape (n_samples, n_features)
            timestamps: Optional timestamps for each prediction

        Returns:
            List of HMMPrediction objects
        """
        if not self._is_trained or self.model is None:
            raise RuntimeError("Model must be trained before prediction")

        features = features.copy()

        # Handle NaN/Inf
        if np.any(np.isnan(features)):
            col_means = np.nanmean(features, axis=0)
            for i in range(features.shape[1]):
                mask = np.isnan(features[:, i])
                features[mask, i] = col_means[i]

        if np.any(np.isinf(features)):
            features = np.clip(features, -1e10, 1e10)

        # Standardize and clip to handle distribution shift
        features_scaled = (features - self._feature_mean) / self._feature_std
        features_scaled = np.clip(features_scaled, -4.0, 4.0)

        # Get full sequence decode
        log_prob, state_seq = self.model.decode(features_scaled, algorithm="viterbi")
        state_probs_seq = self.model.predict_proba(features_scaled)

        predictions = []
        for i in range(len(features)):
            state_idx = state_seq[i]
            state_probs = state_probs_seq[i]

            self._state_history.append(state_idx)
            if len(self._state_history) > self._max_history_len:
                self._state_history.pop(0)

            ts = timestamps[i] if timestamps is not None else None
            if hasattr(ts, 'to_pydatetime'):
                ts = ts.to_pydatetime()

            pred = HMMPrediction(
                label=self.state_labels.get(state_idx, RegimeLabel.UNKNOWN),
                state_idx=state_idx,
                confidence=float(state_probs[state_idx]),
                state_probs=state_probs,
                entropy=float(entropy(state_probs + 1e-10)),
                transition_rate=self._calculate_transition_rate(),
                timeframe=self.timeframe,
                timestamp=ts,
                n_states=self.n_states,
            )
            predictions.append(pred)

        return predictions

    def _calculate_transition_rate(self) -> float:
        """Calculate recent state transition rate."""
        if len(self._state_history) < 2:
            return 0.0

        transitions = sum(
            1 for i in range(1, len(self._state_history))
            if self._state_history[i] != self._state_history[i - 1]
        )
        return transitions / (len(self._state_history) - 1)

    def reset_history(self) -> None:
        """Reset state history for transition rate calculation."""
        self._state_history = []

    def save_checkpoint(self, run_id: str) -> HMMCheckpoint:
        """Save model to checkpoint."""
        if not self._is_trained or self.model is None:
            raise RuntimeError("Model must be trained before saving")

        # Handle covariance extraction based on type
        # hmmlearn stores diag covars as 3D internally but expects 2D on assignment
        covars = self.model.covars_.copy()
        if self.covariance_type == "diag" and covars.ndim == 3:
            # Extract diagonal elements: (n_components, n_features, n_features) -> (n_components, n_features)
            covars = np.array([np.diag(c) for c in covars])

        return HMMCheckpoint(
            run_id=run_id,
            timeframe=self.timeframe,
            n_states=self.n_states,
            means=self.model.means_.copy(),
            covars=covars,
            transmat=self.model.transmat_.copy(),
            startprob=self.model.startprob_.copy(),
            feature_names=self.feature_names.copy(),
            covariance_type=self.covariance_type,
            train_start=self.train_start,
            train_end=self.train_end,
            n_samples=self.n_samples,
            log_likelihood=self.log_likelihood,
            state_labels={k: v.value for k, v in self.state_labels.items()},
            created_at=datetime.utcnow(),
        )

    def load_checkpoint(self, checkpoint: HMMCheckpoint) -> None:
        """Load model from checkpoint."""
        if checkpoint.timeframe != self.timeframe:
            raise ValueError(
                f"Checkpoint timeframe {checkpoint.timeframe} != {self.timeframe}"
            )

        self.n_states = checkpoint.n_states
        self.covariance_type = checkpoint.covariance_type
        self.feature_names = checkpoint.feature_names.copy()

        self.model = self._create_model()
        self.model.means_ = checkpoint.means
        self.model.covars_ = checkpoint.covars
        self.model.transmat_ = checkpoint.transmat
        self.model.startprob_ = checkpoint.startprob

        self.state_labels = {
            int(k): RegimeLabel(v) for k, v in checkpoint.state_labels.items()
        }

        self.train_start = checkpoint.train_start
        self.train_end = checkpoint.train_end
        self.n_samples = checkpoint.n_samples
        self.log_likelihood = checkpoint.log_likelihood

        n_features = len(self.feature_names)
        self._feature_mean = np.zeros(n_features)
        self._feature_std = np.ones(n_features)

        # Force uniform start probabilities to avoid bias during inference
        # The trained startprob reflects training data sequence, not desired inference behavior
        self.model.startprob_ = np.ones(self.n_states) / self.n_states

        self._is_trained = True
        logger.info(f"Loaded MidHMM checkpoint: run_id={checkpoint.run_id}")

    def set_scaling_params(self, mean: np.ndarray, std: np.ndarray) -> None:
        """Set feature scaling parameters after loading checkpoint."""
        self._feature_mean = mean.copy()
        self._feature_std = std.copy()

    @property
    def is_trained(self) -> bool:
        """Check if model is trained."""
        return self._is_trained
