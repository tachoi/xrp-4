"""
Core Engine for the XRP Trading System.
Wires together zones, gate, FSM, and risk management.
"""
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field

from .types import (
    Candle, Zone, ZoneSet, ZoneState, ZoneStateType,
    Decision, Signal, Side, EntryType, OrderIntent, IndicatorValues
)
from .config import CoreConfig
from .zones_15m import ZoneEngine
from .gate_15m import ZoneGate
from .fsm_3m import TradingFSM
from .risk import RiskManager, calculate_pnl_pct
from .indicators import ema, rsi, atr, sma, closes, volumes
from .explain import DecisionLogger


@dataclass
class EngineState:
    """Engine state tracking."""
    candles_15m: List[Candle] = field(default_factory=list)
    candles_3m: List[Candle] = field(default_factory=list)
    current_price: float = 0.0
    current_atr_15m: float = 0.0
    current_atr_3m: float = 0.0
    last_zone_update_ts: int = 0
    order_intents: List[OrderIntent] = field(default_factory=list)


@dataclass
class EngineStats:
    """Engine statistics."""
    total_candles_3m: int = 0
    total_candles_15m: int = 0
    anchors_detected: int = 0
    entries: int = 0
    exits: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl_pct: float = 0.0
    # Zone statistics
    zones_created: int = 0
    zones_current: int = 0
    max_zones: int = 0

    def summary(self) -> str:
        """Get summary string."""
        win_rate = self.wins / max(1, self.entries) * 100
        return (
            f"Candles: 15m={self.total_candles_15m}, 3m={self.total_candles_3m} | "
            f"Zones: {self.zones_current} (max={self.max_zones}, created={self.zones_created}) | "
            f"Anchors: {self.anchors_detected} | "
            f"Entries: {self.entries}, Exits: {self.exits} | "
            f"W/L: {self.wins}/{self.losses} ({win_rate:.1f}%) | "
            f"PnL: {self.total_pnl_pct:.2f}%"
        )


class CoreEngine:
    """
    Core Trading Engine.

    Orchestrates:
    - 15m zone detection and updates
    - 15m gate for zone state
    - 3m FSM for trade signals
    - Risk management for position sizing
    - Order intent generation

    Usage:
        engine = CoreEngine(config)
        engine.feed_15m(candle_15m)  # Update zones on new 15m candle
        decision = engine.feed_3m(candle_3m)  # Process 3m candle
        if decision.signal in (Signal.ENTRY_LONG, Signal.ENTRY_SHORT):
            order = engine.get_last_order_intent()
    """

    def __init__(self, config: CoreConfig, equity: float = 10000.0):
        self.config = config
        self.equity = equity

        # Components
        self.zone_engine = ZoneEngine(config.zone)
        self.gate = ZoneGate(config.zone)
        self.fsm = TradingFSM(config.fsm, config.phenomena)
        self.risk_manager = RiskManager(config.risk)
        self.logger = DecisionLogger()

        # State
        self.state = EngineState()
        self.stats = EngineStats()

        # Indicator caches
        self._ema_3m: List[float] = []
        self._rsi_3m: List[float] = []
        self._atr_3m: List[float] = []
        self._vol_sma_3m: List[float] = []
        self._atr_15m: List[float] = []

        # Trade tracking
        self._current_trade_entry: Optional[float] = None
        self._current_trade_side: Optional[Side] = None

    def reset(self) -> None:
        """Reset engine to initial state."""
        self.zone_engine.reset()
        self.fsm.reset()
        self.risk_manager.reset_daily()
        self.state = EngineState()
        self.stats = EngineStats()
        self._ema_3m = []
        self._rsi_3m = []
        self._atr_3m = []
        self._vol_sma_3m = []
        self._atr_15m = []
        self._current_trade_entry = None
        self._current_trade_side = None

    # Maximum candles to keep in memory (performance optimization)
    # 15m: 500 candles = ~5 days of data (enough for zone detection)
    # 3m: 1000 candles = ~50 hours
    MAX_CANDLES_15M = 500
    MAX_CANDLES_3M = 1000

    def feed_15m(self, candle: Candle) -> ZoneSet:
        """
        Feed a 15m candle and update zones.

        Args:
            candle: 15m candle

        Returns:
            Updated ZoneSet
        """
        self.state.candles_15m.append(candle)
        self.stats.total_candles_15m += 1

        # Limit candle history to save memory and improve performance
        if len(self.state.candles_15m) > self.MAX_CANDLES_15M:
            # Calculate how many candles are being removed
            removed_count = len(self.state.candles_15m) - self.MAX_CANDLES_15M
            self.state.candles_15m = self.state.candles_15m[-self.MAX_CANDLES_15M:]
            # Adjust zone engine's last processed index
            self.zone_engine._last_processed_index = max(0, self.zone_engine._last_processed_index - removed_count)
            # Adjust candidate indices and filter out invalid ones
            adjusted_candidates = []
            for c in self.zone_engine._candidates:
                c.index -= removed_count
                if c.index >= 0:  # Keep only valid candidates
                    adjusted_candidates.append(c)
            self.zone_engine._candidates = adjusted_candidates

        # Update zones
        zones = self.zone_engine.update(self.state.candles_15m)

        # Update zone statistics
        self.stats.zones_created = self.zone_engine._zone_counter
        current_zone_count = len(zones.all_zones)
        self.stats.zones_current = current_zone_count
        self.stats.max_zones = max(self.stats.max_zones, current_zone_count)

        # Use zone engine's cached ATR (avoid redundant calculation)
        self.state.current_atr_15m = self.zone_engine._current_atr

        self.state.last_zone_update_ts = candle.ts

        return zones

    def feed_3m(self, candle: Candle) -> Decision:
        """
        Feed a 3m candle and process through FSM.

        Args:
            candle: 3m candle

        Returns:
            Trading decision
        """
        self.state.candles_3m.append(candle)
        self.state.current_price = candle.close
        self.stats.total_candles_3m += 1

        # Limit candle history (performance optimization)
        if len(self.state.candles_3m) > self.MAX_CANDLES_3M:
            self.state.candles_3m = self.state.candles_3m[-self.MAX_CANDLES_3M:]

        # Update 3m indicators
        indicators = self._update_3m_indicators()

        # Get zone state
        zone_state = self.gate.check(
            candle.close,
            self.zone_engine.zones,
            self.state.current_atr_15m
        )

        # Get recent candles for FSM
        lookback = max(
            self.config.phenomena.lookback_k,
            self.config.fsm.anchor_vol_sma_n,
            20
        )
        recent = self.state.candles_3m[-lookback:]

        # Process through FSM
        decision = self.fsm.step(
            candle, indicators, zone_state, recent, self._rsi_3m[-20:]
        )

        # Log decision
        self.logger.log(decision)

        # Update stats
        self._update_stats(decision, candle)

        # Generate order intent if needed
        if decision.signal in (Signal.ENTRY_LONG, Signal.ENTRY_SHORT):
            self._generate_order_intent(decision, candle)

        # Handle exit for PnL tracking
        if decision.signal == Signal.EXIT:
            self._process_exit(decision, candle)

        return decision

    def _update_3m_indicators(self) -> IndicatorValues:
        """Update and return 3m indicators."""
        candles = self.state.candles_3m
        if not candles:
            return IndicatorValues.empty()

        close_prices = closes(candles)
        volume_values = volumes(candles)

        # Compute indicators
        self._ema_3m = ema(close_prices, self.config.fsm.ema_period)
        self._rsi_3m = rsi(close_prices, self.config.fsm.rsi_period)
        self._atr_3m = atr(candles, self.config.fsm.atr_period)
        self._vol_sma_3m = sma(volume_values, self.config.fsm.anchor_vol_sma_n)

        # Current values
        current_ema = self._ema_3m[-1] if self._ema_3m else 0
        current_rsi = self._rsi_3m[-1] if self._rsi_3m else 50
        current_atr = self._atr_3m[-1] if self._atr_3m else 0
        current_vol_sma = self._vol_sma_3m[-1] if self._vol_sma_3m else 0
        prev_rsi = self._rsi_3m[-2] if len(self._rsi_3m) > 1 else None

        self.state.current_atr_3m = current_atr

        # EMA distance
        ema_dist = None
        if current_ema > 0 and current_atr > 0:
            ema_dist = abs(candles[-1].close - current_ema) / current_atr

        return IndicatorValues(
            ema20=current_ema,
            rsi14=current_rsi,
            atr14=current_atr,
            volume_sma=current_vol_sma,
            rsi_prev=prev_rsi,
            ema_distance=ema_dist,
        )

    def _update_stats(self, decision: Decision, candle: Candle) -> None:
        """Update engine statistics."""
        if decision.signal == Signal.ANCHOR:
            self.stats.anchors_detected += 1

        if decision.signal in (Signal.ENTRY_LONG, Signal.ENTRY_SHORT):
            self.stats.entries += 1
            self._current_trade_entry = candle.close
            self._current_trade_side = decision.side

        if decision.signal == Signal.EXIT:
            self.stats.exits += 1

    def _generate_order_intent(self, decision: Decision,
                                candle: Candle) -> None:
        """Generate order intent for entry signal."""
        if decision.side is None or decision.stop_price is None:
            return

        # Get position size
        can_trade, reason = self.risk_manager.can_trade()
        if not can_trade:
            return

        size, size_explanation = self.risk_manager.compute_position_size(
            self.equity, candle.close, decision.stop_price
        )

        if size <= 0:
            return

        intent = OrderIntent(
            side=decision.side,
            entry_type=EntryType.MARKET,
            entry_price=candle.close,
            stop_price=decision.stop_price,
            size=size,
            explanation=f"{decision.explanation} | {size_explanation}",
            debug={
                'anchor_info': decision.anchor_info,
                'phenomena': decision.phenomena,
            }
        )

        self.state.order_intents.append(intent)

    def _process_exit(self, decision: Decision, candle: Candle) -> None:
        """Process exit for PnL tracking."""
        if self._current_trade_entry is None:
            return

        entry = self._current_trade_entry
        exit_price = candle.close
        is_long = self._current_trade_side == Side.LONG

        pnl_pct = calculate_pnl_pct(entry, exit_price, is_long)

        if pnl_pct > 0:
            self.stats.wins += 1
        else:
            self.stats.losses += 1

        self.stats.total_pnl_pct += pnl_pct
        self.risk_manager.update_pnl(pnl_pct)

        # Reset tracking
        self._current_trade_entry = None
        self._current_trade_side = None

    def get_last_order_intent(self) -> Optional[OrderIntent]:
        """Get most recent order intent."""
        if self.state.order_intents:
            return self.state.order_intents[-1]
        return None

    def get_zones(self) -> ZoneSet:
        """Get current zones."""
        return self.zone_engine.zones

    def get_fsm_state(self) -> Dict[str, Any]:
        """Get FSM state info."""
        return self.fsm.get_state_info()

    def get_summary(self) -> str:
        """Get engine summary."""
        return self.stats.summary()

    def get_risk_summary(self) -> str:
        """Get risk manager summary."""
        return self.risk_manager.get_summary()

    def is_in_trade(self) -> bool:
        """Check if currently in a trade."""
        return self.fsm.is_in_trade


def run_backtest(candles_15m: List[Candle], candles_3m: List[Candle],
                  config: CoreConfig, equity: float = 10000.0,
                  verbose: bool = False) -> Tuple[EngineStats, List[Decision]]:
    """
    Run backtest on historical candles.

    Assumes candles are synchronized by timestamp. For each 3m candle,
    checks if a new 15m candle should be processed.

    Args:
        candles_15m: 15m historical candles
        candles_3m: 3m historical candles
        config: Trading configuration
        equity: Starting equity
        verbose: Print decisions if True

    Returns:
        Tuple of (stats, decisions)
    """
    engine = CoreEngine(config, equity)
    decisions: List[Decision] = []

    # Index into 15m candles
    idx_15m = 0

    for candle_3m in candles_3m:
        # Check if we should process new 15m candle(s)
        while (idx_15m < len(candles_15m) and
               candles_15m[idx_15m].ts <= candle_3m.ts):
            engine.feed_15m(candles_15m[idx_15m])
            idx_15m += 1

        # Process 3m candle
        decision = engine.feed_3m(candle_3m)
        decisions.append(decision)

        if verbose and decision.signal != Signal.NONE:
            print(f"[{candle_3m.ts}] {decision.signal.value}: {decision.explanation}")

    return engine.stats, decisions
