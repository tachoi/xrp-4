"""
Tests for explanation generation.
"""
import pytest
from src.core.types import (
    Decision, Signal, FSMState, ZoneState, ZoneStateType, Side,
    Zone, ZoneKind, AnchorInfo, PhenomenaResult
)
from src.core.explain import (
    ExplanationBuilder, format_decision, explain_anchor_detection,
    explain_entry, explain_exit, explain_phenomena, explain_zone_gate,
    summarize_trade, DecisionLogger
)


class TestExplanationBuilder:
    """Tests for ExplanationBuilder."""

    def test_add_parts(self):
        """Test adding parts to explanation."""
        builder = ExplanationBuilder()
        builder.add("Part 1")
        builder.add("Part 2")

        result = builder.build()

        assert "Part 1" in result
        assert "Part 2" in result

    def test_empty_parts_ignored(self):
        """Test that empty parts are ignored."""
        builder = ExplanationBuilder()
        builder.add("Valid")
        builder.add("")
        builder.add("Also Valid")

        result = builder.build()

        assert result.count("|") == 1  # Only one separator

    def test_clear(self):
        """Test clearing builder."""
        builder = ExplanationBuilder()
        builder.add("Something")
        builder.clear()

        result = builder.build()

        assert result == ""

    def test_add_zone_context(self):
        """Test adding zone context."""
        builder = ExplanationBuilder()
        zone_state = ZoneState(
            state=ZoneStateType.CORE,
            explanation="CORE: inside support zone"
        )

        builder.add_zone_context(zone_state)
        result = builder.build()

        assert "CORE" in result

    def test_add_anchor_info(self):
        """Test adding anchor info."""
        builder = ExplanationBuilder()
        anchor = AnchorInfo(
            open=1.0,
            close=1.04,
            mid=1.02,
            direction=1,
            ts=0,
            index=0,
            volume_mult=3.0,
            body_atr_mult=1.5,
        )

        builder.add_anchor_info(anchor)
        result = builder.build()

        assert "bullish" in result.lower()
        assert "3.0x" in result
        assert "1.5" in result

    def test_add_phenomena(self):
        """Test adding phenomena result."""
        builder = ExplanationBuilder()
        phenomena = PhenomenaResult(
            ok=True,
            low_fail=True,
            wick_ratio_ok=True,
            body_shrink=False,
            range_shrink=False,
            count=2,
            details="low_fail, wick_ratio"
        )

        builder.add_phenomena(phenomena)
        result = builder.build()

        assert "OK" in result
        assert "low_fail" in result


class TestFormatDecision:
    """Tests for format_decision function."""

    def test_format_with_signal(self):
        """Test formatting decision with signal."""
        decision = Decision(
            signal=Signal.ANCHOR,
            prev_state=FSMState.IDLE,
            new_state=FSMState.ANCHOR_FOUND,
            explanation="Anchor detected with high volume"
        )

        result = format_decision(decision)

        assert "ANCHOR" in result
        assert "IDLE" in result
        assert "ANCHOR_FOUND" in result

    def test_format_no_state_change(self):
        """Test formatting when state doesn't change."""
        decision = Decision(
            signal=Signal.NONE,
            prev_state=FSMState.IDLE,
            new_state=FSMState.IDLE,
            explanation="Waiting for anchor"
        )

        result = format_decision(decision)

        # Should not have state transition
        assert "->" not in result or result.count("->") == 0

    def test_format_with_zone_state(self):
        """Test formatting with zone state context."""
        decision = Decision(
            signal=Signal.ENTRY_LONG,
            prev_state=FSMState.ENTRY_READY,
            new_state=FSMState.IN_TRADE,
            explanation="Entry confirmed"
        )

        zone_state = ZoneState(
            state=ZoneStateType.CORE,
            explanation="CORE: support zone"
        )

        result = format_decision(decision, zone_state)

        assert "ENTRY_LONG" in result
        assert "CORE" in result


class TestExplainFunctions:
    """Tests for individual explain functions."""

    def test_explain_anchor_detection(self):
        """Test anchor detection explanation."""
        anchor = AnchorInfo(
            open=1.0,
            close=1.04,
            mid=1.02,
            direction=1,
            ts=0,
            index=0,
            volume_mult=2.8,
            body_atr_mult=1.3,
        )

        result = explain_anchor_detection(anchor)

        assert "Bullish" in result
        assert "2.8x" in result
        assert "A_mid" in result

    def test_explain_anchor_with_zone(self):
        """Test anchor explanation with zone context."""
        anchor = AnchorInfo(
            open=1.0, close=1.04, mid=1.02, direction=1,
            ts=0, index=0, volume_mult=2.8, body_atr_mult=1.3,
        )

        zone = Zone(
            id="z1",
            center=1.0,
            radius=0.05,
            strength=5.0,
            kind=ZoneKind.SUPPORT,
            last_updated_ts=0,
        )

        result = explain_anchor_detection(anchor, zone)

        assert "support" in result.lower()

    def test_explain_entry(self):
        """Test entry explanation."""
        decision = Decision(
            signal=Signal.ENTRY_LONG,
            prev_state=FSMState.ENTRY_READY,
            new_state=FSMState.IN_TRADE,
            explanation="Entry triggered",
            side=Side.LONG,
            entry_price=1.02,
            stop_price=0.98,
        )

        result = explain_entry(decision)

        assert "LONG" in result
        assert "1.02" in result
        assert "0.98" in result

    def test_explain_exit(self):
        """Test exit explanation."""
        decision = Decision(
            signal=Signal.EXIT,
            prev_state=FSMState.IN_TRADE,
            new_state=FSMState.EXIT_COOLDOWN,
            explanation="Time stop hit",
            side=Side.LONG,
            entry_price=1.0,
            stop_price=0.95,
        )

        result = explain_exit(decision, exit_price=1.05, pnl_pct=5.0)

        assert "LONG" in result
        assert "WIN" in result
        assert "5.00" in result

    def test_explain_phenomena(self):
        """Test phenomena explanation."""
        result = PhenomenaResult(
            ok=True,
            low_fail=True,
            wick_ratio_ok=True,
            body_shrink=True,
            range_shrink=False,
            count=3,
            details="test"
        )

        explanation = explain_phenomena(result)

        assert "PASS" in explanation
        assert "low_fail=YES" in explanation
        assert "wick_ratio=OK" in explanation
        assert "body_shrink=YES" in explanation
        assert "range_shrink=NO" in explanation

    def test_explain_zone_gate_blocked(self):
        """Test zone gate blocked explanation."""
        zone_state = ZoneState(
            state=ZoneStateType.OUTSIDE,
            explanation="Price outside zones"
        )

        result = explain_zone_gate(zone_state, "anchor")

        assert "BLOCKED" in result
        assert "anchor" in result

    def test_explain_zone_gate_allowed(self):
        """Test zone gate allowed explanation."""
        zone_state = ZoneState(
            state=ZoneStateType.CORE,
            explanation="Inside zone"
        )

        result = explain_zone_gate(zone_state, "entry")

        assert "ALLOWED" in result
        assert "entry" in result


class TestSummarizeTrade:
    """Tests for trade summarization."""

    def test_summarize_winning_trade(self):
        """Test summarizing a winning trade."""
        entry = Decision(
            signal=Signal.ENTRY_LONG,
            prev_state=FSMState.ENTRY_READY,
            new_state=FSMState.IN_TRADE,
            explanation="Entry",
            side=Side.LONG,
            entry_price=1.0,
            stop_price=0.95,
            anchor_info=AnchorInfo(
                open=0.98, close=1.02, mid=1.0, direction=1,
                ts=0, index=0, volume_mult=3.0, body_atr_mult=1.5,
            ),
        )

        exit_ = Decision(
            signal=Signal.EXIT,
            prev_state=FSMState.IN_TRADE,
            new_state=FSMState.EXIT_COOLDOWN,
            explanation="RSI exit",
            side=Side.LONG,
            entry_price=1.0,
            stop_price=0.95,
        )

        result = summarize_trade(entry, exit_, exit_price=1.05, pnl_pct=5.0)

        assert "WIN" in result
        assert "5.00%" in result
        assert "LONG" in result

    def test_summarize_losing_trade(self):
        """Test summarizing a losing trade."""
        entry = Decision(
            signal=Signal.ENTRY_LONG,
            prev_state=FSMState.ENTRY_READY,
            new_state=FSMState.IN_TRADE,
            explanation="Entry",
            side=Side.LONG,
            entry_price=1.0,
            stop_price=0.95,
        )

        exit_ = Decision(
            signal=Signal.EXIT,
            prev_state=FSMState.IN_TRADE,
            new_state=FSMState.EXIT_COOLDOWN,
            explanation="Stop hit",
            side=Side.LONG,
            entry_price=1.0,
            stop_price=0.95,
        )

        result = summarize_trade(entry, exit_, exit_price=0.95, pnl_pct=-5.0)

        assert "LOSS" in result
        assert "-5.00%" in result


class TestDecisionLogger:
    """Tests for DecisionLogger."""

    def test_log_decisions(self):
        """Test logging decisions."""
        logger = DecisionLogger()

        d1 = Decision(Signal.NONE, FSMState.IDLE, FSMState.IDLE, "Waiting")
        d2 = Decision(Signal.ANCHOR, FSMState.IDLE, FSMState.ANCHOR_FOUND, "Anchor!")

        logger.log(d1)
        logger.log(d2)

        summary = logger.get_summary()

        assert "2" in summary  # 2 decisions

    def test_format_history(self):
        """Test formatting decision history."""
        logger = DecisionLogger()

        for i in range(5):
            d = Decision(
                signal=Signal.NONE,
                prev_state=FSMState.IDLE,
                new_state=FSMState.IDLE,
                explanation=f"Decision {i}"
            )
            logger.log(d)

        history = logger.format_history(last_n=3)

        assert "Decision 2" in history
        assert "Decision 3" in history
        assert "Decision 4" in history

    def test_track_trades(self):
        """Test trade tracking."""
        logger = DecisionLogger()

        entry = Decision(
            signal=Signal.ENTRY_LONG,
            prev_state=FSMState.ENTRY_READY,
            new_state=FSMState.IN_TRADE,
            explanation="Entry"
        )
        logger.log(entry)

        exit_ = Decision(
            signal=Signal.EXIT,
            prev_state=FSMState.IN_TRADE,
            new_state=FSMState.EXIT_COOLDOWN,
            explanation="Exit"
        )
        logger.log(exit_)

        assert len(logger._trades) == 1
        assert logger._trades[0]['exit_decision'] is not None


class TestExplanationNonEmpty:
    """Tests ensuring explanations are never empty."""

    def test_decision_explanation_non_empty(self):
        """Test that Decision explanation is never empty."""
        decision = Decision(
            signal=Signal.ANCHOR,
            prev_state=FSMState.IDLE,
            new_state=FSMState.ANCHOR_FOUND,
            explanation="Test explanation"
        )

        assert len(decision.explanation) > 0

    def test_format_decision_non_empty(self):
        """Test that formatted decision is non-empty."""
        decision = Decision(
            signal=Signal.NONE,
            prev_state=FSMState.IDLE,
            new_state=FSMState.IDLE,
            explanation="Waiting"
        )

        result = format_decision(decision)

        assert len(result) > 0

    def test_all_explain_functions_non_empty(self):
        """Test all explain functions return non-empty strings."""
        anchor = AnchorInfo(
            open=1.0, close=1.04, mid=1.02, direction=1,
            ts=0, index=0, volume_mult=3.0, body_atr_mult=1.5,
        )

        assert len(explain_anchor_detection(anchor)) > 0

        phenomena = PhenomenaResult(ok=True, count=3, details="test")
        assert len(explain_phenomena(phenomena)) > 0

        zone_state = ZoneState(state=ZoneStateType.CORE, explanation="test")
        assert len(explain_zone_gate(zone_state, "test")) > 0
