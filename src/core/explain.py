"""
Explanation generation for the XRP Core Trading System.
Produces human-readable explanations of trading decisions.
"""
from typing import Dict, Any, Optional, List

from .types import (
    Decision, Signal, FSMState, ZoneState, ZoneStateType,
    Side, AnchorInfo, PhenomenaResult, Zone
)


class ExplanationBuilder:
    """Builds structured explanations for trading decisions."""

    def __init__(self):
        self._parts: List[str] = []

    def add(self, text: str) -> 'ExplanationBuilder':
        """Add text to explanation."""
        if text:
            self._parts.append(text)
        return self

    def add_zone_context(self, zone_state: ZoneState) -> 'ExplanationBuilder':
        """Add zone context."""
        self._parts.append(zone_state.explanation)
        return self

    def add_anchor_info(self, anchor: Optional[AnchorInfo]) -> 'ExplanationBuilder':
        """Add anchor information."""
        if anchor is None:
            return self

        direction = "bullish" if anchor.direction > 0 else "bearish"
        self._parts.append(
            f"Anchor ({direction}): vol {anchor.volume_mult:.1f}x, "
            f"body {anchor.body_atr_mult:.1f} ATR, "
            f"mid={anchor.mid:.4f}"
        )
        return self

    def add_phenomena(self, phenomena: Optional[PhenomenaResult]) -> 'ExplanationBuilder':
        """Add phenomena check results."""
        if phenomena is None:
            return self

        status = "OK" if phenomena.ok else "FAIL"
        self._parts.append(f"Phenomena {status}: {phenomena.details}")
        return self

    def add_state_transition(self, prev: FSMState, new: FSMState) -> 'ExplanationBuilder':
        """Add state transition info."""
        if prev != new:
            self._parts.append(f"State: {prev.value} -> {new.value}")
        return self

    def add_signal(self, signal: Signal) -> 'ExplanationBuilder':
        """Add signal info."""
        if signal != Signal.NONE:
            self._parts.append(f"Signal: {signal.value}")
        return self

    def build(self) -> str:
        """Build final explanation string."""
        return " | ".join(self._parts)

    def clear(self) -> 'ExplanationBuilder':
        """Clear builder."""
        self._parts = []
        return self


def format_decision(decision: Decision, zone_state: Optional[ZoneState] = None) -> str:
    """
    Format a Decision into a human-readable explanation.

    Args:
        decision: The trading decision
        zone_state: Optional zone state for context

    Returns:
        Formatted explanation string
    """
    builder = ExplanationBuilder()

    # State transition
    if decision.prev_state != decision.new_state:
        builder.add(f"[{decision.prev_state.value} -> {decision.new_state.value}]")

    # Signal
    if decision.signal != Signal.NONE:
        builder.add(f"Signal: {decision.signal.value}")

    # Zone context
    if zone_state:
        builder.add(zone_state.explanation)

    # Main explanation
    builder.add(decision.explanation)

    return builder.build()


def explain_anchor_detection(anchor: AnchorInfo, zone: Optional[Zone] = None) -> str:
    """
    Generate explanation for anchor detection.

    Args:
        anchor: Detected anchor info
        zone: Current zone if available

    Returns:
        Explanation string
    """
    direction = "Bullish" if anchor.direction > 0 else "Bearish"

    zone_context = ""
    if zone:
        zone_context = f" near {zone.kind.value} zone {zone.id}"

    return (
        f"{direction} anchor detected{zone_context}: "
        f"volume {anchor.volume_mult:.1f}x average, "
        f"body {anchor.body_atr_mult:.1f}x ATR. "
        f"A_mid={anchor.mid:.4f}, waiting for pullback."
    )


def explain_entry(decision: Decision, zone: Optional[Zone] = None) -> str:
    """
    Generate explanation for trade entry.

    Args:
        decision: Entry decision
        zone: Current zone if available

    Returns:
        Explanation string
    """
    if decision.side is None:
        return "Entry with unknown side"

    side = decision.side.value
    zone_context = ""
    if zone:
        zone_context = f" in {zone.kind.value} zone"

    entry_price = decision.entry_price or 0
    stop_price = decision.stop_price or 0

    parts = [f"ENTRY {side}{zone_context}"]

    if decision.anchor_info:
        anchor = decision.anchor_info
        parts.append(f"after pullback to A_mid ({anchor.mid:.4f})")

    if decision.phenomena and decision.phenomena.ok:
        parts.append(f"phenomena confirmed ({decision.phenomena.details})")

    parts.append(f"entry={entry_price:.4f}, stop={stop_price:.4f}")

    return " | ".join(parts)


def explain_exit(decision: Decision, exit_price: float = 0,
                 pnl_pct: float = 0) -> str:
    """
    Generate explanation for trade exit.

    Args:
        decision: Exit decision
        exit_price: Exit price
        pnl_pct: PnL percentage

    Returns:
        Explanation string
    """
    side = decision.side.value if decision.side else "?"
    entry = decision.entry_price or 0
    stop = decision.stop_price or 0

    result = "WIN" if pnl_pct > 0 else "LOSS" if pnl_pct < 0 else "FLAT"

    return (
        f"EXIT {side}: {decision.explanation} | "
        f"Entry={entry:.4f}, Exit={exit_price:.4f}, "
        f"PnL={pnl_pct:.2f}% ({result})"
    )


def explain_phenomena(result: PhenomenaResult) -> str:
    """
    Generate explanation for phenomena check.

    Args:
        result: Phenomena check result

    Returns:
        Explanation string
    """
    checks = []

    if result.low_fail:
        checks.append("low_fail=YES (higher lows)")
    else:
        checks.append("low_fail=NO")

    if result.wick_ratio_ok:
        checks.append("wick_ratio=OK (demand)")
    else:
        checks.append("wick_ratio=LOW")

    if result.body_shrink:
        checks.append("body_shrink=YES (momentum fading)")
    else:
        checks.append("body_shrink=NO")

    if result.range_shrink:
        checks.append("range_shrink=YES (volatility contracting)")
    else:
        checks.append("range_shrink=NO")

    status = "PASS" if result.ok else "FAIL"
    return f"Phenomena {status} [{result.count}/3]: {', '.join(checks)}"


def explain_zone_gate(zone_state: ZoneState, action: str) -> str:
    """
    Generate explanation for zone gate decision.

    Args:
        zone_state: Current zone state
        action: Attempted action (e.g., "anchor", "entry")

    Returns:
        Explanation string
    """
    state = zone_state.state

    if state == ZoneStateType.OUTSIDE:
        return f"Gate BLOCKED {action}: Price outside all zones. {zone_state.explanation}"

    if state == ZoneStateType.NEAR:
        if action == "entry":
            return f"Gate BLOCKED entry: NEAR zone only. {zone_state.explanation}"
        return f"Gate ALLOWED {action}: NEAR zone. {zone_state.explanation}"

    # CORE
    return f"Gate ALLOWED {action}: CORE zone. {zone_state.explanation}"


def summarize_trade(entry_decision: Decision, exit_decision: Decision,
                    exit_price: float, pnl_pct: float) -> str:
    """
    Generate summary of a completed trade.

    Args:
        entry_decision: Entry decision
        exit_decision: Exit decision
        exit_price: Exit price
        pnl_pct: PnL percentage

    Returns:
        Trade summary string
    """
    side = entry_decision.side.value if entry_decision.side else "?"
    entry = entry_decision.entry_price or 0
    stop = entry_decision.stop_price or 0

    anchor_info = ""
    if entry_decision.anchor_info:
        anchor = entry_decision.anchor_info
        anchor_info = f"Anchor vol={anchor.volume_mult:.1f}x, body={anchor.body_atr_mult:.1f} ATR. "

    result = "WIN" if pnl_pct > 0 else "LOSS" if pnl_pct < 0 else "FLAT"

    return (
        f"Trade {side} {result}: {anchor_info}"
        f"Entry={entry:.4f}, Stop={stop:.4f}, Exit={exit_price:.4f}, "
        f"PnL={pnl_pct:.2f}%"
    )


class DecisionLogger:
    """Collects and formats decision history."""

    def __init__(self):
        self._decisions: List[Decision] = []
        self._trades: List[Dict[str, Any]] = []

    def log(self, decision: Decision) -> None:
        """Log a decision."""
        self._decisions.append(decision)

        # Track trades
        if decision.signal in (Signal.ENTRY_LONG, Signal.ENTRY_SHORT):
            self._trades.append({
                'entry_decision': decision,
                'entry_time': len(self._decisions),
                'exit_decision': None,
                'exit_time': None,
            })
        elif decision.signal == Signal.EXIT and self._trades:
            # Update last trade
            last = self._trades[-1]
            if last['exit_decision'] is None:
                last['exit_decision'] = decision
                last['exit_time'] = len(self._decisions)

    def get_summary(self) -> str:
        """Get session summary."""
        total = len(self._decisions)
        signals = [d for d in self._decisions if d.signal != Signal.NONE]
        entries = [d for d in signals if d.signal in (Signal.ENTRY_LONG, Signal.ENTRY_SHORT)]
        exits = [d for d in signals if d.signal == Signal.EXIT]

        return (
            f"Decisions: {total}, Signals: {len(signals)}, "
            f"Entries: {len(entries)}, Exits: {len(exits)}"
        )

    def format_history(self, last_n: int = 10) -> str:
        """Format recent decision history."""
        recent = self._decisions[-last_n:]
        lines = []

        for i, d in enumerate(recent, 1):
            lines.append(f"{i}. [{d.signal.value}] {d.explanation}")

        return "\n".join(lines)
