"""Regime detection module using Hidden Markov Models."""

from xrp4.regime.hmm_types import (
    RegimeLabel,
    HMMPrediction,
    RegimePacket,
    HMMCheckpoint,
)
from xrp4.regime.hmm_fast import FastHMM
from xrp4.regime.hmm_mid import MidHMM
from xrp4.regime.hmm_fusion import HMMFusion, create_fusion_from_config
from xrp4.regime.multi_hmm_manager import MultiHMMManager
from xrp4.regime.feature_contract import (
    FeatureContract,
    FeatureContractError,
    get_contract,
    validate_training_features,
    validate_inference_features,
)

__all__ = [
    "RegimeLabel",
    "HMMPrediction",
    "RegimePacket",
    "HMMCheckpoint",
    "FastHMM",
    "MidHMM",
    "HMMFusion",
    "create_fusion_from_config",
    "MultiHMMManager",
    "FeatureContract",
    "FeatureContractError",
    "get_contract",
    "validate_training_features",
    "validate_inference_features",
]
