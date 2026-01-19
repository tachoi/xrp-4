"""XGB Approval Gate - ML filter for trade signals.

Filters low-probability trades using XGBoost model trained on historical trades.
Enabled by default (XGB_ENABLED: true, XGB_PMIN_TREND: 0.45).

Training results (2025 test):
- Without XGB: -$2.60 PnL, 54.2% WR
- With XGB (0.45): +$32.08 PnL, 57.8% WR (+34.68 improvement)
"""

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


class XGBApprovalGate:
    """XGBoost approval gate for trade signal filtering.

    Predicts win probability for trades and filters low-quality signals.

    Features used (must match training):
    - ret, ret_2, ret_5: Recent returns
    - ema_diff: EMA 20/50 difference
    - price_to_ema20, price_to_ema50: Price relative to EMAs
    - volatility, range_pct: Volatility measures
    - volume_ratio: Volume relative to MA
    - rsi, ema_slope: Technical indicators
    - side_num: Trade direction (1=LONG, -1=SHORT)
    - regime_trend_up, regime_trend_down: Regime indicators
    """

    # Feature names in order (must match training)
    FEATURE_NAMES = [
        'ret', 'ret_2', 'ret_5',
        'ema_diff', 'price_to_ema20', 'price_to_ema50',
        'volatility', 'range_pct', 'volume_ratio',
        'rsi', 'ema_slope',
        'side_num', 'regime_trend_up', 'regime_trend_down'
    ]

    def __init__(self, model_path: Optional[str] = None):
        """Initialize XGB gate.

        Args:
            model_path: Path to XGBoost model file. If None, uses default path.
        """
        self._model = None
        self._is_loaded = False
        self._model_path = model_path or "outputs/xgb_gate/xgb_model.json"

        # Try to load model on init
        self.load_model(Path(self._model_path))

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._is_loaded

    def load_model(self, path: Path) -> bool:
        """Load XGBoost model from file.

        Args:
            path: Path to model file (JSON format)

        Returns:
            True if loaded successfully
        """
        try:
            import xgboost as xgb

            if not path.exists():
                logger.warning(f"XGB model not found at {path}")
                self._is_loaded = False
                return False

            self._model = xgb.Booster()
            self._model.load_model(str(path))
            self._is_loaded = True
            logger.info(f"XGB model loaded from {path}")
            return True

        except ImportError:
            logger.warning("xgboost not installed, XGB gate disabled")
            self._is_loaded = False
            return False
        except Exception as e:
            logger.warning(f"Failed to load XGB model: {e}")
            self._is_loaded = False
            return False

    def predict_proba(self, row_3m: Dict, signal: str, regime: str) -> float:
        """Predict win probability for given trade context.

        Args:
            row_3m: Current 3m bar features (dict)
            signal: Trade signal (LONG_TREND_PULLBACK, SHORT_TREND_PULLBACK, etc.)
            regime: Current regime (TREND_UP, TREND_DOWN, RANGE)

        Returns:
            Probability of trade success (0.0 to 1.0)
            Returns 0.5 (neutral) if model not loaded
        """
        if not self._is_loaded or self._model is None:
            return 0.5  # Neutral probability when disabled

        try:
            import xgboost as xgb

            # Extract features from row_3m
            features = self._extract_features(row_3m, signal, regime)

            # Create DMatrix and predict
            dmat = xgb.DMatrix(features.reshape(1, -1))
            proba = self._model.predict(dmat)[0]

            return float(proba)

        except Exception as e:
            logger.debug(f"XGB prediction failed: {e}")
            return 0.5

    def _extract_features(self, row_3m: Dict, signal: str, regime: str) -> np.ndarray:
        """Extract feature vector from market context.

        Args:
            row_3m: Current 3m bar features
            signal: Trade signal
            regime: Current regime

        Returns:
            Feature vector as numpy array
        """
        # Get close price for calculations
        close = row_3m.get('close', row_3m.get('price', 1.0))
        high = row_3m.get('high', close)
        low = row_3m.get('low', close)

        # Calculate derived features if not present
        ema_20 = row_3m.get('ema_20', row_3m.get('ema_fast_3m', close))
        ema_50 = row_3m.get('ema_50', row_3m.get('ema_slow_3m', close))

        # Get returns - compute ret_2 and ret_5 from close prices if available
        ret = row_3m.get('ret', row_3m.get('ret_3m', 0))

        # ret_2: 2-bar return (if close_2 available, compute it)
        close_2 = row_3m.get('close_2', None)
        if close_2 is not None and close_2 > 0:
            ret_2 = (close - close_2) / close_2
        else:
            ret_2 = row_3m.get('ret_2', ret)  # Fallback to ret if not computed

        # ret_5: 5-bar return (if close_5 available, compute it)
        close_5 = row_3m.get('close_5', None)
        if close_5 is not None and close_5 > 0:
            ret_5 = (close - close_5) / close_5
        else:
            ret_5 = row_3m.get('ret_5', ret)  # Fallback to ret if not computed

        # Calculate range_pct from actual high/low (not hardcoded)
        if close > 0:
            range_pct = row_3m.get('range_pct', (high - low) / close)
        else:
            range_pct = 0.005

        # Calculate volume_ratio from actual volume (not hardcoded)
        volume = row_3m.get('volume', 1.0)
        volume_ma = row_3m.get('volume_ma', row_3m.get('volume_ma_20', volume))
        if volume_ma > 0:
            volume_ratio = row_3m.get('volume_ratio', volume / volume_ma)
        else:
            volume_ratio = 1.0

        # EMA slope - prefer 3m slope, fall back to 15m
        ema_slope = row_3m.get('ema_slope_3m', row_3m.get('ema_slope', row_3m.get('ema_slope_15m', 0)))

        features = np.array([
            # Returns
            ret,
            ret_2,
            ret_5,

            # EMA features
            row_3m.get('ema_diff', (ema_20 - ema_50) / close if close > 0 else 0),
            row_3m.get('price_to_ema20', (close - ema_20) / ema_20 if ema_20 > 0 else 0),
            row_3m.get('price_to_ema50', (close - ema_50) / ema_50 if ema_50 > 0 else 0),

            # Volatility
            row_3m.get('volatility', row_3m.get('vol', 0.005)),
            range_pct,

            # Volume
            volume_ratio,

            # Technical indicators
            row_3m.get('rsi', row_3m.get('rsi_3m', 50)),
            ema_slope,

            # Side (direction)
            1 if 'LONG' in signal else -1,

            # Regime indicators
            1 if regime == 'TREND_UP' else 0,
            1 if regime == 'TREND_DOWN' else 0,
        ], dtype=np.float32)

        return features

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from model.

        Returns:
            Dict of feature name -> importance score
        """
        if not self._is_loaded or self._model is None:
            return {}

        try:
            importance = self._model.get_score(importance_type='gain')
            # Map f0, f1, ... to actual feature names
            result = {}
            for key, value in importance.items():
                idx = int(key.replace('f', ''))
                if idx < len(self.FEATURE_NAMES):
                    result[self.FEATURE_NAMES[idx]] = value
            return result
        except Exception:
            return {}
