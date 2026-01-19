"""Data contracts for FSM and DecisionEngine.

Defines the standardized types used throughout the trading pipeline:
Features -> HMM(raw) -> ConfirmLayer(confirmed_regime) -> FSM(candidate signal) -> DecisionEngine(final action) -> Broker
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Literal


# Signal types emitted by FSM
SignalType = Literal[
    "HOLD",
    "LONG_BOUNCE",
    "SHORT_BOUNCE",
    "LONG_BREAKOUT",
    "SHORT_BREAKOUT",
    "LONG_TREND_PULLBACK",
    "SHORT_TREND_PULLBACK",
    "EXIT",
]

# Action types from DecisionEngine
ActionType = Literal[
    "NO_ACTION",
    "OPEN_LONG",
    "OPEN_SHORT",
    "CLOSE",
    "REDUCE",
    "INCREASE",
]


@dataclass
class ConfirmContext:
    """Context from ConfirmLayer output.

    Contains regime information and confirmation metrics.
    """
    regime_raw: str                        # Raw HMM regime
    regime_confirmed: str                  # RANGE/TREND_UP/TREND_DOWN/HIGH_VOL/TRANSITION/NO_TRADE
    confirm_reason: str                    # Short reason code from ConfirmLayer
    confirm_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class MarketContext:
    """Current market context for signal generation.

    Provides all necessary market data for FSM decision making.
    """
    symbol: str
    ts: int                                # Epoch milliseconds
    price: float                           # Current price
    row_3m: Dict[str, float] = field(default_factory=dict)   # Current 3m features (ema/rsi/atr etc.)
    row_15m: Dict[str, float] = field(default_factory=dict)  # Current 15m features
    zone: Dict[str, float] = field(default_factory=dict)     # Zone info: support, resistance, strength, dist_to_support, dist_to_resistance


@dataclass
class PositionState:
    """Current position state.

    Tracks open position details for FSM and DecisionEngine.
    """
    side: Literal["FLAT", "LONG", "SHORT"] = "FLAT"
    entry_price: float = 0.0
    size: float = 0.0
    entry_ts: int = 0
    bars_held_3m: int = 0
    unrealized_pnl: float = 0.0


@dataclass
class CandidateSignal:
    """Candidate signal from FSM.

    Contains signal type, confidence score, and parameters for execution.
    """
    signal: SignalType = "HOLD"
    score: float = 0.0                     # 0~1 confidence score
    reason: str = ""                       # Short reason code
    params: Dict[str, float] = field(default_factory=dict)   # stop/tp/trail targets etc.


@dataclass
class Decision:
    """Final decision from DecisionEngine.

    Contains action to execute with size and reasoning.
    """
    action: ActionType = "NO_ACTION"
    size: float = 0.0
    reason: str = ""
    meta: Dict[str, float] = field(default_factory=dict)
