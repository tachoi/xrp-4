"""Feature Contract - Single source of truth for HMM feature lists.

Enforces that training and inference use IDENTICAL feature lists
loaded from configs/hmm_features.yaml.

Usage:
    from xrp3.regime.feature_contract import FeatureContract, get_contract

    # Load contract
    contract = get_contract()

    # Get feature lists
    fast_features = contract.get_fast_features()
    mid_features = contract.get_mid_features()

    # Validate features
    contract.validate_features(df, "fast_3m")  # Raises if mismatch
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import yaml


class FeatureContractError(Exception):
    """Error raised when feature contract is violated."""

    pass


@dataclass
class HMMFeatureSpec:
    """Specification for a single HMM model's features."""

    name: str
    timeframe: str
    features: List[str]
    forbidden_features: List[str]


@dataclass
class FeatureContract:
    """Feature contract loaded from YAML.

    Enforces identical feature lists between training and inference.
    """

    version: int
    notes: List[str]
    fast_3m: HMMFeatureSpec
    mid_15m: HMMFeatureSpec
    by_timeframe: Dict[str, List[str]]

    @classmethod
    def from_yaml(cls, path: Path) -> "FeatureContract":
        """Load feature contract from YAML file.

        Args:
            path: Path to hmm_features.yaml

        Returns:
            FeatureContract instance
        """
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        version = data.get("version", 1)
        notes = data.get("notes", [])

        hmm = data.get("hmm", {})

        fast_3m_data = hmm.get("fast_3m", {})
        fast_3m = HMMFeatureSpec(
            name=fast_3m_data.get("name", "fast_hmm_3m"),
            timeframe=fast_3m_data.get("timeframe", "3m"),
            features=fast_3m_data.get("features", []),
            forbidden_features=fast_3m_data.get("forbidden_features", []),
        )

        mid_15m_data = hmm.get("mid_15m", {})
        mid_15m = HMMFeatureSpec(
            name=mid_15m_data.get("name", "mid_hmm_15m"),
            timeframe=mid_15m_data.get("timeframe", "15m"),
            features=mid_15m_data.get("features", []),
            forbidden_features=mid_15m_data.get("forbidden_features", []),
        )

        by_timeframe = data.get("by_timeframe", {})

        return cls(
            version=version,
            notes=notes,
            fast_3m=fast_3m,
            mid_15m=mid_15m,
            by_timeframe=by_timeframe,
        )

    def get_fast_features(self) -> List[str]:
        """Get feature list for Fast HMM (3m)."""
        return list(self.fast_3m.features)

    def get_mid_features(self) -> List[str]:
        """Get feature list for Mid HMM (15m)."""
        return list(self.mid_15m.features)

    def get_forbidden_fast(self) -> List[str]:
        """Get forbidden feature list for Fast HMM."""
        return list(self.fast_3m.forbidden_features)

    def get_forbidden_mid(self) -> List[str]:
        """Get forbidden feature list for Mid HMM."""
        return list(self.mid_15m.forbidden_features)

    def validate_fast_features(
        self,
        features: List[str],
        strict: bool = True,
    ) -> None:
        """Validate features against Fast HMM contract.

        Args:
            features: List of feature names to validate
            strict: If True, features must match exactly. If False, subset allowed.

        Raises:
            FeatureContractError: If validation fails
        """
        self._validate_features(
            features=features,
            expected=self.fast_3m.features,
            forbidden=self.fast_3m.forbidden_features,
            model_name="fast_3m",
            strict=strict,
        )

    def validate_mid_features(
        self,
        features: List[str],
        strict: bool = True,
    ) -> None:
        """Validate features against Mid HMM contract.

        Args:
            features: List of feature names to validate
            strict: If True, features must match exactly. If False, subset allowed.

        Raises:
            FeatureContractError: If validation fails
        """
        self._validate_features(
            features=features,
            expected=self.mid_15m.features,
            forbidden=self.mid_15m.forbidden_features,
            model_name="mid_15m",
            strict=strict,
        )

    def _validate_features(
        self,
        features: List[str],
        expected: List[str],
        forbidden: List[str],
        model_name: str,
        strict: bool,
    ) -> None:
        """Internal validation logic.

        Args:
            features: Features to validate
            expected: Expected feature list from contract
            forbidden: Forbidden features
            model_name: Name for error messages
            strict: Whether to require exact match

        Raises:
            FeatureContractError: If validation fails
        """
        features_set = set(features)
        expected_set = set(expected)
        forbidden_set = set(forbidden)

        # Check for forbidden features
        forbidden_present = features_set & forbidden_set
        if forbidden_present:
            raise FeatureContractError(
                f"[{model_name}] FORBIDDEN features present: {sorted(forbidden_present)}. "
                f"These features are not allowed in HMM v2."
            )

        if strict:
            # Check for missing features
            missing = expected_set - features_set
            if missing:
                raise FeatureContractError(
                    f"[{model_name}] MISSING features: {sorted(missing)}. "
                    f"Contract requires: {expected}"
                )

            # Check for extra features
            extra = features_set - expected_set
            if extra:
                raise FeatureContractError(
                    f"[{model_name}] EXTRA features: {sorted(extra)}. "
                    f"Contract allows only: {expected}"
                )
        else:
            # Non-strict: just check that all features are in expected set
            invalid = features_set - expected_set
            if invalid:
                raise FeatureContractError(
                    f"[{model_name}] Invalid features: {sorted(invalid)}. "
                    f"Must be subset of: {expected}"
                )

    def print_summary(self) -> str:
        """Print human-readable summary of contract."""
        lines = [
            f"=== Feature Contract v{self.version} ===",
            "",
            f"Fast HMM (3m): {self.fast_3m.name}",
            f"  Features ({len(self.fast_3m.features)}):",
        ]
        for f in self.fast_3m.features:
            lines.append(f"    - {f}")
        lines.append(f"  Forbidden ({len(self.fast_3m.forbidden_features)}):")
        for f in self.fast_3m.forbidden_features:
            lines.append(f"    - {f}")

        lines.extend([
            "",
            f"Mid HMM (15m): {self.mid_15m.name}",
            f"  Features ({len(self.mid_15m.features)}):",
        ])
        for f in self.mid_15m.features:
            lines.append(f"    - {f}")
        lines.append(f"  Forbidden ({len(self.mid_15m.forbidden_features)}):")
        for f in self.mid_15m.forbidden_features:
            lines.append(f"    - {f}")

        return "\n".join(lines)


# Global contract instance (lazy-loaded)
_contract: Optional[FeatureContract] = None
_contract_path: Optional[Path] = None


def get_contract(
    config_path: Optional[Path] = None,
    force_reload: bool = False,
) -> FeatureContract:
    """Get the global feature contract instance.

    Args:
        config_path: Path to hmm_features.yaml (default: configs/hmm_features.yaml)
        force_reload: Force reload from disk

    Returns:
        FeatureContract instance
    """
    global _contract, _contract_path

    if config_path is None:
        # Default path relative to project root
        config_path = Path(__file__).resolve().parents[3] / "configs" / "hmm_features.yaml"

    if _contract is None or force_reload or _contract_path != config_path:
        if not config_path.exists():
            raise FileNotFoundError(f"Feature contract not found: {config_path}")
        _contract = FeatureContract.from_yaml(config_path)
        _contract_path = config_path

    return _contract


def validate_training_features(
    fast_features: List[str],
    mid_features: List[str],
    config_path: Optional[Path] = None,
) -> None:
    """Validate features before HMM training.

    Call this at the start of train_hmm.py to ensure
    feature lists match the contract.

    Args:
        fast_features: Features for Fast HMM
        mid_features: Features for Mid HMM
        config_path: Path to hmm_features.yaml

    Raises:
        FeatureContractError: If validation fails
    """
    contract = get_contract(config_path)
    contract.validate_fast_features(fast_features, strict=True)
    contract.validate_mid_features(mid_features, strict=True)


def validate_inference_features(
    fast_features: List[str],
    mid_features: List[str],
    config_path: Optional[Path] = None,
) -> None:
    """Validate features before HMM inference.

    Call this at the start of backtest/paper trading to ensure
    feature lists match the contract.

    Args:
        fast_features: Features for Fast HMM
        mid_features: Features for Mid HMM
        config_path: Path to hmm_features.yaml

    Raises:
        FeatureContractError: If validation fails
    """
    # Same validation as training - must be identical
    validate_training_features(fast_features, mid_features, config_path)
