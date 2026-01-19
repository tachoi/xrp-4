# Claude Code Prompt — Step 0.1: Box Strength 기반 필터 추가 (xrp-4 Step0 Baseline)

You are working in my workspace. `xrp-4` already has Step0 baseline backtest (Zone-based Bounce/Breakout + fixed risk, no ML).
Now implement **Step 0.1: Box Strength 기반 필터** to avoid trading low-quality RANGE/LOWVOL noise zones.

ABSOLUTE RULES
- Still NO HMM / NO XGB / NO FSM. (No ML)
- This is a structural market-quality filter only.
- Keep changes minimal and localized (prefer adding new module + small hook in signal pipeline).
- Must be fully configurable via `configs/step0.yaml`.
- Must output diagnostics to prove the filter is working (accept/reject counts, distributions, and performance split).

================================================================================
0) GOAL
================================================================================
Add a filter that blocks trades unless the current "box" (range structure) is strong enough:
- Box must be **wide enough relative to volatility**
- Box must show **reaction evidence** (optional toggle, Step 0.1 can start with width-only)
- Box must have **positive expected tradability** after costs

This should reduce trades and improve PF/MDD compared to Step0 baseline.

================================================================================
1) DEFINITIONS
================================================================================
We define "box" over a rolling window of recent candles (lookback L bars):
- box_high = rolling_max(high, L)
- box_low  = rolling_min(low,  L)
- box_height = box_high - box_low
- atr = ATR(tf=15m if available else 3m) aligned to 3m bars
- box_height_atr = box_height / atr

A) Box Strength (v1 = simplest)
- box_strength = box_height_atr
- Pass if: box_strength >= BOX_MIN_HEIGHT_ATR

B) Optional: Reaction Evidence (v1.1 toggle)
Measure if price "bounces" near edges historically:
- Define edge bands:
  support_band = [box_low, box_low + edge_band_atr_k*atr]
  resist_band  = [box_high - edge_band_atr_k*atr, box_high]
- For last N touches, compute success rate:
  - A "touch" at support: candle.low <= support_band_high
  - "success" if within next M bars, price moves upward by >= move_target_atr_k*atr without hitting stop
  - Similar for resistance.
- reaction_score = max(support_success_rate, resistance_success_rate)
- Pass if reaction_score >= REACTION_MIN_RATE

C) Cost-aware tradability (v1 optional)
- expected_move_atr = box_height_atr / 2  (rough half-box traversal)
- costs_atr = (fee_bps + slippage_bps)/10000 * close / atr
- Pass if expected_move_atr >= costs_atr * COST_SAFETY_MULT

================================================================================
2) CONFIG CHANGES (configs/step0.yaml)
================================================================================
Add a new section:

box_filter:
  enabled: true
  lookback_bars: 64                 # ~3h on 3m
  atr_period: 14
  atr_tf: "15m"                     # fallback to 3m if not available
  min_height_atr: 3.0               # ★ 핵심
  update_every_bars: 10             # recompute box/strength periodically

  # optional toggles
  use_reaction_filter: false
  edge_band_atr_k: 0.3
  reaction_lookback_touches: 20
  reaction_horizon_bars: 6
  move_target_atr_k: 1.0
  reaction_min_rate: 0.48

  use_cost_filter: false
  cost_safety_mult: 2.0

Also add reporting toggles:
report:
  dump_box_filter_diagnostics: true

================================================================================
3) IMPLEMENTATION TASKS
================================================================================
Task 1) Locate xrp-4 signal decision point
- Find where Step0 generates candidate signals and decides to place a trade.
- Insert the box filter gating BEFORE signal confirmation/entry.
- The filter should return PASS/FAIL with reason codes.

Task 2) Add module: `src/xrp4/filters/box_strength.py`
Implement:

- dataclass BoxFilterConfig
- class BoxStrengthFilter:
    - __init__(config, fee_bps, slippage_bps)
    - update(df, idx) -> BoxState  (compute box_high/low/atr/strength at idx)
    - allow_trade(df, idx, side, zone=None) -> (bool, reason:str, debug:dict)

BoxState fields:
- box_high, box_low, box_height
- atr, box_height_atr
- expected_move_atr, costs_atr
- reaction_score (if enabled)

Reason codes:
- "PASS"
- "FAIL_MIN_HEIGHT_ATR"
- "FAIL_REACTION"
- "FAIL_COST"

Task 3) ATR availability
- If xrp-4 already computes ATR, reuse it.
- Otherwise implement a simple ATR (Wilder) in `xrp4/ta/atr.py` and cache results to avoid per-step recomputation.

Task 4) Efficiency
- Do not recompute rolling max/min from scratch each bar.
- Use pandas rolling precomputation:
    - df["box_high_L"] = df["high"].rolling(L).max()
    - df["box_low_L"]  = df["low"].rolling(L).min()
    - df["atr"] = ATR(...)
- Box filter reads these columns at idx.
- Recompute columns only once at start; update_every_bars can be used to reduce reaction computations.

Task 5) Diagnostics (must have)
Add to backtest output:
- counts: total candidate entries, passed, failed_by_reason
- distribution: histogram-like bins of box_height_atr at entry candidates (passed vs failed)
- performance split:
    - trades where box_height_atr in [0-2), [2-3), [3-4), [4-6), [6+)
    - compute PF, winrate, avgR per bin
Write as CSV/JSON into outputs folder if `dump_box_filter_diagnostics=true`.

Task 6) Add CLI flag for quick sweep (optional, but recommended)
Allow overriding `box_filter.min_height_atr` from CLI:
- `python -m xrp4.run_step0 --config configs/step0.yaml --override box_filter.min_height_atr=3.5`
If xrp-4 has no override mechanism, implement a minimal one.

================================================================================
4) VALIDATION STEPS (YOU MUST RUN/PRINT)
================================================================================
After implementation:
1) Run baseline Step0 and Step0.1 on same period and print comparison:
   - trades, PF, PnL, MDD
2) Print box filter diagnostics summary:
   - pass rate
   - top fail reason
   - median box_height_atr of passed vs failed
3) Confirm no ML modules imported.

================================================================================
5) OUTPUT REQUIREMENTS
================================================================================
At the end, provide:
- list of files created/modified
- exact commands used
- where diagnostics files are saved
- short interpretation of whether filter improved results

Now proceed. Start by locating xrp-4 code paths for signal generation and backtest entry, then implement box strength filter as specified.
