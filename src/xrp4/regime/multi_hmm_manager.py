"""Multi-HMM Manager - Orchestrates Fast HMM, Mid HMM, and Fusion.

Manages the complete Multi-HMM regime detection pipeline:
- Training both HMM models
- Loading/saving checkpoints
- Coordinating predictions across timeframes
- Producing fused RegimePacket outputs
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import yaml

from xrp4.regime.hmm_fast import FastHMM
from xrp4.regime.hmm_mid import MidHMM
from xrp4.regime.hmm_fusion import HMMFusion, create_fusion_from_config
from xrp4.regime.hmm_types import (
    HMMCheckpoint,
    HMMPrediction,
    RegimeLabel,
    RegimePacket,
)

logger = logging.getLogger(__name__)


class MultiHMMManager:
    """Manager for Multi-HMM regime detection system.

    Coordinates Fast HMM (3m) and Mid HMM (15m) models, handles
    checkpoint persistence, and produces fused regime predictions.

    Attributes:
        fast_hmm: Fast HMM model for 3m timeframe
        mid_hmm: Mid HMM model for 15m timeframe
        fusion: HMM Fusion component
        run_id: Unique identifier for current training run
        checkpoint_dir: Directory for saving/loading checkpoints
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        checkpoint_dir: Optional[Union[str, Path]] = None,
    ):
        """Initialize MultiHMMManager.

        Args:
            config: Configuration dict (from hmm_gate_policy.yaml)
            checkpoint_dir: Directory for checkpoint storage
        """
        self.config = config or {}
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else Path("checkpoints/hmm")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Extract HMM config
        hmm_config = self.config.get("hmm_config", {})

        # Initialize Fast HMM (3m)
        fast_cfg = hmm_config.get("fast", {})
        self.fast_hmm = FastHMM(
            n_states=fast_cfg.get("n_states", 4),
            covariance_type=fast_cfg.get("covariance_type", "full"),
        )

        # Initialize Mid HMM (15m)
        mid_cfg = hmm_config.get("mid", {})
        self.mid_hmm = MidHMM(
            n_states=mid_cfg.get("n_states", 4),
            covariance_type=mid_cfg.get("covariance_type", "full"),
        )

        # Initialize Fusion
        self.fusion = create_fusion_from_config(hmm_config)

        # Runtime state
        self.run_id: Optional[str] = None
        self._fast_features: List[str] = []
        self._mid_features: List[str] = []

        # Scaling parameters for each model
        self._fast_scaling: Optional[Dict[str, np.ndarray]] = None
        self._mid_scaling: Optional[Dict[str, np.ndarray]] = None

    @classmethod
    def from_config_file(
        cls,
        config_path: Union[str, Path],
        checkpoint_dir: Optional[Union[str, Path]] = None,
    ) -> "MultiHMMManager":
        """Create manager from config file.

        Args:
            config_path: Path to hmm_gate_policy.yaml
            checkpoint_dir: Optional checkpoint directory

        Returns:
            Configured MultiHMMManager instance
        """
        config_path = Path(config_path)
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        policy_config = config.get("hmm_gate_policy", config)

        return cls(
            config=policy_config,
            checkpoint_dir=checkpoint_dir,
        )

    def train(
        self,
        fast_features: np.ndarray,
        fast_feature_names: List[str],
        mid_features: np.ndarray,
        mid_feature_names: List[str],
        fast_timestamps: Optional[pd.DatetimeIndex] = None,
        mid_timestamps: Optional[pd.DatetimeIndex] = None,
        run_id: Optional[str] = None,
    ) -> Dict[str, float]:
        """Train both HMM models.

        Args:
            fast_features: Feature matrix for Fast HMM (3m data)
            fast_feature_names: Feature names for Fast HMM
            mid_features: Feature matrix for Mid HMM (15m data)
            mid_feature_names: Feature names for Mid HMM
            fast_timestamps: Timestamps for Fast HMM data
            mid_timestamps: Timestamps for Mid HMM data
            run_id: Unique identifier for this training run

        Returns:
            Dict with log-likelihoods for each model
        """
        self.run_id = run_id or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        self._fast_features = fast_feature_names
        self._mid_features = mid_feature_names

        logger.info(f"Training MultiHMM, run_id={self.run_id}")

        # Train Fast HMM
        logger.info(f"Training Fast HMM (3m): {fast_features.shape[0]} samples")
        fast_ll = self.fast_hmm.train(
            fast_features,
            fast_feature_names,
            fast_timestamps,
        )

        # Store scaling params
        self._fast_scaling = {
            "mean": self.fast_hmm._feature_mean.copy(),
            "std": self.fast_hmm._feature_std.copy(),
        }

        # Train Mid HMM
        logger.info(f"Training Mid HMM (15m): {mid_features.shape[0]} samples")
        mid_ll = self.mid_hmm.train(
            mid_features,
            mid_feature_names,
            mid_timestamps,
        )

        # Store scaling params
        self._mid_scaling = {
            "mean": self.mid_hmm._feature_mean.copy(),
            "std": self.mid_hmm._feature_std.copy(),
        }

        logger.info(
            f"MultiHMM training complete: fast_ll={fast_ll:.2f}, mid_ll={mid_ll:.2f}"
        )

        return {
            "fast_log_likelihood": fast_ll,
            "mid_log_likelihood": mid_ll,
        }

    def predict(
        self,
        fast_features: Optional[np.ndarray] = None,
        mid_features: Optional[np.ndarray] = None,
        timestamp: Optional[datetime] = None,
    ) -> RegimePacket:
        """Predict fused regime from feature inputs.

        Args:
            fast_features: Feature vector for Fast HMM (optional)
            mid_features: Feature vector for Mid HMM (optional)
            timestamp: Timestamp for prediction

        Returns:
            Fused RegimePacket

        Note:
            At least one of fast_features or mid_features must be provided.
        """
        fast_pred = None
        mid_pred = None

        if fast_features is not None and self.fast_hmm.is_trained:
            fast_pred = self.fast_hmm.predict(fast_features, timestamp)

        if mid_features is not None and self.mid_hmm.is_trained:
            mid_pred = self.mid_hmm.predict(mid_features, timestamp)

        return self.fusion.fuse(fast_pred, mid_pred, timestamp)

    def predict_sequence(
        self,
        fast_features: Optional[np.ndarray] = None,
        mid_features: Optional[np.ndarray] = None,
        fast_timestamps: Optional[pd.DatetimeIndex] = None,
        mid_timestamps: Optional[pd.DatetimeIndex] = None,
        align_to_fast: bool = True,
    ) -> List[RegimePacket]:
        """Predict fused regime for sequences of features.

        Args:
            fast_features: Feature matrix for Fast HMM
            mid_features: Feature matrix for Mid HMM
            fast_timestamps: Timestamps for Fast HMM predictions
            mid_timestamps: Timestamps for Mid HMM predictions
            align_to_fast: If True, output is aligned to 3m timestamps

        Returns:
            List of RegimePacket objects
        """
        # Get predictions from each model
        fast_preds = []
        mid_preds = []

        if fast_features is not None and self.fast_hmm.is_trained:
            fast_preds = self.fast_hmm.predict_sequence(fast_features, fast_timestamps)

        if mid_features is not None and self.mid_hmm.is_trained:
            mid_preds = self.mid_hmm.predict_sequence(mid_features, mid_timestamps)

        # Fuse predictions
        if align_to_fast and fast_preds:
            # Align mid predictions to fast timestamps
            return self._fuse_aligned_to_fast(
                fast_preds, mid_preds, fast_timestamps, mid_timestamps
            )
        elif mid_preds:
            # Align to mid timestamps
            return self._fuse_aligned_to_mid(
                fast_preds, mid_preds, fast_timestamps, mid_timestamps
            )
        else:
            # Only fast predictions
            return [
                self.fusion.fuse(fp, None, fp.timestamp if fp else None)
                for fp in fast_preds
            ]

    def _fuse_aligned_to_fast(
        self,
        fast_preds: List[HMMPrediction],
        mid_preds: List[HMMPrediction],
        fast_timestamps: Optional[pd.DatetimeIndex],
        mid_timestamps: Optional[pd.DatetimeIndex],
    ) -> List[RegimePacket]:
        """Fuse predictions aligned to 3m (fast) timestamps.

        For each 3m prediction, find the corresponding 15m prediction
        (15m prediction covers 5 x 3m bars).

        Optimized O(n) algorithm: floor 3m timestamp to 15m boundary for direct lookup.
        """
        results = []

        # Build mid prediction index by timestamp (O(m))
        mid_by_ts: Dict[datetime, HMMPrediction] = {}
        if mid_preds and mid_timestamps is not None:
            for i, pred in enumerate(mid_preds):
                ts = mid_timestamps[i]
                if hasattr(ts, 'to_pydatetime'):
                    ts = ts.to_pydatetime()
                mid_by_ts[ts] = pred

        # Process each 3m prediction (O(n))
        for i, fast_pred in enumerate(fast_preds):
            ts = fast_timestamps[i] if fast_timestamps is not None else None
            if hasattr(ts, 'to_pydatetime'):
                ts = ts.to_pydatetime()

            # Find corresponding mid prediction using O(1) lookup
            mid_pred = None
            if ts and mid_by_ts:
                # Floor timestamp to 15m boundary: e.g., 10:03 -> 10:00, 10:14 -> 10:00
                # This gives us the start of the 15m bar that contains this 3m bar
                ts_floor = ts.replace(minute=(ts.minute // 15) * 15, second=0, microsecond=0)
                mid_pred = mid_by_ts.get(ts_floor)

            packet = self.fusion.fuse(fast_pred, mid_pred, ts)
            results.append(packet)

        return results

    def _fuse_aligned_to_mid(
        self,
        fast_preds: List[HMMPrediction],
        mid_preds: List[HMMPrediction],
        fast_timestamps: Optional[pd.DatetimeIndex],
        mid_timestamps: Optional[pd.DatetimeIndex],
    ) -> List[RegimePacket]:
        """Fuse predictions aligned to 15m (mid) timestamps.

        For each 15m prediction, find the latest corresponding 3m prediction.

        Optimized O(n+m) algorithm: group fast preds by 15m boundary, then lookup.
        """
        results = []

        # Build fast prediction index grouped by 15m boundary (O(n))
        # Store the latest 3m prediction for each 15m window
        fast_by_15m: Dict[datetime, Tuple[datetime, HMMPrediction]] = {}
        if fast_preds and fast_timestamps is not None:
            for i, pred in enumerate(fast_preds):
                ts = fast_timestamps[i]
                if hasattr(ts, 'to_pydatetime'):
                    ts = ts.to_pydatetime()
                # Floor to 15m boundary
                ts_floor = ts.replace(minute=(ts.minute // 15) * 15, second=0, microsecond=0)
                # Keep only the latest 3m bar in each 15m window
                if ts_floor not in fast_by_15m or ts > fast_by_15m[ts_floor][0]:
                    fast_by_15m[ts_floor] = (ts, pred)

        # Process each 15m prediction (O(m))
        for i, mid_pred in enumerate(mid_preds):
            ts = mid_timestamps[i] if mid_timestamps is not None else None
            if hasattr(ts, 'to_pydatetime'):
                ts = ts.to_pydatetime()

            # Find latest 3m prediction using O(1) lookup
            fast_pred = None
            if ts and fast_by_15m and ts in fast_by_15m:
                fast_pred = fast_by_15m[ts][1]

            packet = self.fusion.fuse(fast_pred, mid_pred, ts)
            results.append(packet)

        return results

    def save_checkpoints(self, run_id: Optional[str] = None) -> Dict[str, Path]:
        """Save both HMM model checkpoints.

        Args:
            run_id: Optional run ID (uses training run_id if not provided)

        Returns:
            Dict mapping model name to checkpoint file path
        """
        run_id = run_id or self.run_id
        if not run_id:
            run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

        saved_paths = {}

        # Save Fast HMM
        if self.fast_hmm.is_trained:
            fast_ckpt = self.fast_hmm.save_checkpoint(run_id)
            fast_path = self.checkpoint_dir / f"fast_hmm_{run_id}.json"
            with open(fast_path, "w", encoding="utf-8") as f:
                json.dump(fast_ckpt.to_dict(), f, indent=2)
            saved_paths["fast"] = fast_path

            # Save scaling params
            fast_scale_path = self.checkpoint_dir / f"fast_hmm_{run_id}_scaling.npz"
            if self._fast_scaling:
                np.savez(
                    fast_scale_path,
                    mean=self._fast_scaling["mean"],
                    std=self._fast_scaling["std"],
                )
            saved_paths["fast_scaling"] = fast_scale_path

        # Save Mid HMM
        if self.mid_hmm.is_trained:
            mid_ckpt = self.mid_hmm.save_checkpoint(run_id)
            mid_path = self.checkpoint_dir / f"mid_hmm_{run_id}.json"
            with open(mid_path, "w", encoding="utf-8") as f:
                json.dump(mid_ckpt.to_dict(), f, indent=2)
            saved_paths["mid"] = mid_path

            # Save scaling params
            mid_scale_path = self.checkpoint_dir / f"mid_hmm_{run_id}_scaling.npz"
            if self._mid_scaling:
                np.savez(
                    mid_scale_path,
                    mean=self._mid_scaling["mean"],
                    std=self._mid_scaling["std"],
                )
            saved_paths["mid_scaling"] = mid_scale_path

        logger.info(f"Saved HMM checkpoints: {list(saved_paths.keys())}")
        return saved_paths

    def load_checkpoints(self, run_id: str) -> None:
        """Load both HMM model checkpoints.

        Args:
            run_id: Run ID of checkpoints to load
        """
        self.run_id = run_id

        # Load Fast HMM
        fast_path = self.checkpoint_dir / f"fast_hmm_{run_id}.json"
        if fast_path.exists():
            with open(fast_path, "r", encoding="utf-8") as f:
                fast_data = json.load(f)
            fast_ckpt = HMMCheckpoint.from_dict(fast_data)
            self.fast_hmm.load_checkpoint(fast_ckpt)
            self._fast_features = fast_ckpt.feature_names

            # Load scaling params
            fast_scale_path = self.checkpoint_dir / f"fast_hmm_{run_id}_scaling.npz"
            if fast_scale_path.exists():
                scale_data = np.load(fast_scale_path)
                self.fast_hmm.set_scaling_params(scale_data["mean"], scale_data["std"])
                self._fast_scaling = {
                    "mean": scale_data["mean"],
                    "std": scale_data["std"],
                }

            logger.info(f"Loaded Fast HMM checkpoint: {run_id}")
        else:
            logger.warning(f"Fast HMM checkpoint not found: {fast_path}")

        # Load Mid HMM
        mid_path = self.checkpoint_dir / f"mid_hmm_{run_id}.json"
        if mid_path.exists():
            with open(mid_path, "r", encoding="utf-8") as f:
                mid_data = json.load(f)
            mid_ckpt = HMMCheckpoint.from_dict(mid_data)
            self.mid_hmm.load_checkpoint(mid_ckpt)
            self._mid_features = mid_ckpt.feature_names

            # Load scaling params
            mid_scale_path = self.checkpoint_dir / f"mid_hmm_{run_id}_scaling.npz"
            if mid_scale_path.exists():
                scale_data = np.load(mid_scale_path)
                self.mid_hmm.set_scaling_params(scale_data["mean"], scale_data["std"])
                self._mid_scaling = {
                    "mean": scale_data["mean"],
                    "std": scale_data["std"],
                }

            logger.info(f"Loaded Mid HMM checkpoint: {run_id}")
        else:
            logger.warning(f"Mid HMM checkpoint not found: {mid_path}")

    def reset_history(self) -> None:
        """Reset state history for both HMM models."""
        self.fast_hmm.reset_history()
        self.mid_hmm.reset_history()

    def get_feature_names(self) -> Dict[str, List[str]]:
        """Get feature names for each model.

        Returns:
            Dict with 'fast' and 'mid' feature name lists
        """
        return {
            "fast": self._fast_features.copy(),
            "mid": self._mid_features.copy(),
        }

    def get_training_report(self) -> Dict[str, Any]:
        """Get training summary report.

        Returns:
            Dict with training statistics for both models
        """
        report = {
            "run_id": self.run_id,
            "fast_hmm": {
                "is_trained": self.fast_hmm.is_trained,
                "n_states": self.fast_hmm.n_states,
                "n_features": len(self._fast_features),
                "n_samples": self.fast_hmm.n_samples,
                "log_likelihood": self.fast_hmm.log_likelihood,
                "train_start": str(self.fast_hmm.train_start) if self.fast_hmm.train_start else None,
                "train_end": str(self.fast_hmm.train_end) if self.fast_hmm.train_end else None,
                "state_labels": {
                    k: v.value for k, v in self.fast_hmm.state_labels.items()
                } if self.fast_hmm.state_labels else {},
            },
            "mid_hmm": {
                "is_trained": self.mid_hmm.is_trained,
                "n_states": self.mid_hmm.n_states,
                "n_features": len(self._mid_features),
                "n_samples": self.mid_hmm.n_samples,
                "log_likelihood": self.mid_hmm.log_likelihood,
                "train_start": str(self.mid_hmm.train_start) if self.mid_hmm.train_start else None,
                "train_end": str(self.mid_hmm.train_end) if self.mid_hmm.train_end else None,
                "state_labels": {
                    k: v.value for k, v in self.mid_hmm.state_labels.items()
                } if self.mid_hmm.state_labels else {},
            },
            "fusion": {
                "method": self.fusion.method.value,
                "fast_weight": self.fusion.fast_weight,
                "mid_weight": self.fusion.mid_weight,
                "conflict_resolution": self.fusion.conflict_resolution.value,
            },
        }
        return report

    @property
    def is_trained(self) -> bool:
        """Check if both models are trained."""
        return self.fast_hmm.is_trained and self.mid_hmm.is_trained
