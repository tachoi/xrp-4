# Claude Code Prompt — Box Height ATR Threshold 결정 실험 (Step 0.1 Validation)

You are working in my local workspace.
`xrp-4` already implements:
- Step0 baseline (Zone + Bounce/Breakout + fixed risk, NO ML)
- Step0.1 Box Strength filter (box_height_atr based gating)

Your task is to DESIGN, IMPLEMENT, and ANALYZE a controlled experiment
to determine the **optimal box_height_atr threshold**.

ABSOLUTE RULES
- Still NO HMM / NO XGB / NO FSM.
- No discretionary tuning. Decisions must be based on reported statistics.
- Threshold selection must be JUSTIFIED, not optimized for max return.
- Code + reports must make the decision reproducible.

================================================================================
0) EXPERIMENT GOAL
================================================================================
Answer this precise question:

❓ "From which minimum box_height_atr does the strategy become structurally non-negative EV?"

We are NOT looking for:
- maximum profit
- best Sharpe
- curve-fit sweet spot

We ARE looking for:
- structural break from -EV → ~0 or +EV
- monotonic improvement as box_height_atr increases
- robustness across time

================================================================================
1) EXPERIMENT DESIGN
================================================================================

A) Independent Variable (Sweep Parameter)
----------------------------------------
Sweep `box_filter.min_height_atr` over fixed grid:

box_height_atr_grid = [
  1.5,
  2.0,
  2.5,
  3.0,
  3.5,
  4.0,
  5.0,
  6.0
]

Rules:
- Same data
- Same risk
- Same SL/TP
- Same fees/slippage
- Same random seed (if any)

ONLY difference is `min_height_atr`.

----------------------------------------
B) Fixed Strategy Mode
----------------------------------------
Run the experiment in two modes (separately):

Mode A:
- Bounce-only

Mode B:
- Breakout-only

(Do NOT mix signals in this experiment)

----------------------------------------
C) Time Structure
----------------------------------------
Run on FULL 3-year dataset first.

Optionally (if infra exists):
- Split into:
  - Period 1: Year 1
  - Period 2: Year 2
  - Period 3: Year 3

This is NOT walk-forward; it is robustness checking.

================================================================================
2) IMPLEMENTATION TASKS
================================================================================

Task 1) Batch Runner
--------------------
Implement a batch experiment runner:
- Iterates over `box_height_atr_grid`
- For each value:
  - override config at runtime
  - run full backtest
  - collect metrics

Prefer implementation:
- `xrp4/experiments/box_threshold_sweep.py`

CLI:
