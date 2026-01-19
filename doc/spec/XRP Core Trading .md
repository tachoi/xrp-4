# Claude Code Prompt — XRP Core Trading System (Zone + 3m FSM) Implementation Spec v1

You are to implement a rule-based automated trading "Core" system with:
- 15m zones (support/resistance as zones) used as a top-level gate (context)
- 3m execution via FSM using:
  - Anchor Candle ("기준봉") detection
  - EMA20 distance (chase block)
  - RSI14 + optional divergence (confirm/exit)
  - "phenomenon-based" conditions that replace W-bottom / inverted hammer / doji

IMPORTANT CONSTRAINTS
- No HMM / No XGB / No ML.
- No Satellite trend strategy.
- Everything must be explainable with logged sentences.
- 15m zone gate must strictly disable 3m signals outside zones.
- Use zones as *ranges*, not single lines.
- All thresholds must be configurable.

Deliverables must include:
1) Clean Python module structure
2) Core logic implementation
3) Backtest-friendly interfaces (no exchange dependency inside core)
4) Unit tests for core rules + deterministic behavior
5) Example config + example run script

---

## 0. Terminology / High-Level Behavior

### Timeframes
- 15m: Zone engine. Produces zones (upper and lower) with radius based on ATR_15m.
- 3m: FSM engine. Only active when price is in/near a zone.

### Zone states
- CORE: price inside zone range
- NEAR: price inside zone range + pad
- OUTSIDE: price not in any zone range + pad

Rule:
- If OUTSIDE: ignore ALL 3m signals (anchor detection, EMA/RSI, phenomenon checks).

### FSM states (3m)
- IDLE
- ANCHOR_FOUND
- PULLBACK_WAIT
- ENTRY_READY
- IN_TRADE
- EXIT_COOLDOWN

---

## 1. Project Layout

Create the following structure:

src/
  core/
    __init__.py
    config.py
    types.py
    indicators.py
    zones_15m.py
    gate_15m.py
    fsm_3m.py
    risk.py
    explain.py
    engine.py
  cli/
    run_backtest_demo.py
tests/
  test_zones_15m.py
  test_gate_15m.py
  test_fsm_anchor.py
  test_fsm_phenomena.py
  test_fsm_entry_exit.py
  test_explain.py
configs/
  core_default.yaml
README.md

---

## 2. Data Model

### Candle
Define a dataclass Candle with:
- ts: datetime or int (epoch ms)
- open, high, low, close: float
- volume: float

### CandleSeries
A list/array of Candle. Provide helper to convert to pandas DataFrame if desired, but core logic must work without pandas.

### Events / Outputs
- Zone: id, center Z, radius R, low, high, strength S, kind ("support"|"resistance"), last_updated_ts
- ZoneState: enum CORE/NEAR/OUTSIDE, matched_zone_id (optional)
- Signal: enum (NONE, ARM, ANCHOR, ENTRY_LONG, ENTRY_SHORT, EXIT, COOL)
- Decision: contains signal + explanation string + debug fields

---

## 3. Config (configs/core_default.yaml)

Include at least:

### 15m zone config
ZONE_ATR_PERIOD_15M: 14
ZONE_RADIUS_R: 0.6          # zone radius = r * ATR_15m
ZONE_PAD_P: 0.4             # pad = p * ATR_15m for NEAR band
PIVOT_L: 2                  # swing pivot left/right window
REACTION_LOOKAHEAD_M: 12    # 15m candles to confirm reaction
REACTION_K_ATR: 0.8         # rebound >= k*ATR_15m counts as reaction
ZONE_MAX_PER_SIDE: 6        # keep top zones by strength per side
ZONE_DECAY_HALFLIFE_HOURS: 12

### 3m FSM config
EMA_PERIOD_3M: 20
RSI_PERIOD_3M: 14
ATR_PERIOD_3M: 14
ANCHOR_VOL_MULT: 2.5        # volume >= mult * SMA(volume,N)
ANCHOR_VOL_SMA_N: 50
ANCHOR_BODY_ATR_MULT: 1.2   # |close-open| >= mult * ATR_3m
CHASE_DIST_MAX_ATR: 1.5     # abs(price-EMA)/ATR <= Dmax
ANCHOR_EXPIRE_CANDLES_3M: 4 # 12 minutes
HOLD_MAX_CANDLES_3M: 6      # 18 minutes
COOLDOWN_CANDLES_3M: 3

### Phenomenon conditions (3m)
PH_K: 8 # lookback for mean calculations
PH_LOW_FAIL_MODE: "2of4" # low fail detection mode
PH_LOWER_WICK_RATIO_MIN: 0.4
PH_BODY_SHRINK_FACTOR: 0.5
PH_RANGE_SHRINK_FACTOR: 0.7
PH_REQUIREMENTS_MIN_COUNT: 3 # number of phenomenon checks required to turn true

### Risk / Position sizing (backtest friendly)
RISK_PER_TRADE_PCT: 0.3
DAILY_MAX_LOSS_PCT: 2.0
DRAWDOWN_SIZE_REDUCE_STEPS: [0.5, 0.3]  # multipliers when approaching daily loss

---

## 4. Indicators (src/core/indicators.py)

Implement:
- EMA(series, period)
- RSI(series, period)
- ATR(high, low, close, period)
- SMA(series, period)
- Helper to compute candle anatomy:
  - body = abs(close-open)
  - range = high-low (handle zero)
  - lower_wick = min(open,close) - low
  - upper_wick = high - max(open,close)
  - lower_wick_ratio = lower_wick / range
  - body_ratio = body / range

No external TA libs. Use numpy only (optional). Must be deterministic.

---

## 5. 15m Zone Engine (src/core/zones_15m.py)

Goal: produce support/resistance zones as ranges with strength and decay.

### Steps
1) Compute ATR_15m.
2) Detect pivot highs/lows using PIVOT_L.
3) Candidate level price = pivot high or low.
4) Validate reaction:
   - Within next REACTION_LOOKAHEAD_M 15m candles, if price moved away from candidate by >= REACTION_K_ATR * ATR_15m (use ATR at pivot or current), then candidate gets a reaction count.
5) Merge candidates into zones using ATR-based radius:
   - radius = ZONE_RADIUS_R * ATR_15m(current)
   - group candidates whose distance <= radius
   - representative center = weighted median (weights based on reaction count and recency)
6) Maintain strength:
   - strength increases with reaction events
   - strength decreases on decisive breaks (close beyond zone boundary by > radius)
   - time decay: exponential decay with half-life ZONE_DECAY_HALFLIFE_HOURS
7) Keep only top ZONE_MAX_PER_SIDE zones for support and resistance.

Deliver:
- ZoneSet with list of zones and method `find_zone_state(price)` returning ZoneState CORE/NEAR/OUTSIDE.

---

## 6. 15m Gate (src/core/gate_15m.py)

Implement:
- `zone_state(price, zoneset) -> ZoneState`
- Rule:
  - If OUTSIDE: 3m FSM must not progress (force IDLE, no anchor detection).
  - If NEAR: FSM can ARM but cannot enter trade (no ENTRY).
  - If CORE: FSM can fully operate.

Include clear explanation strings:
- "OUTSIDE: price not within any zone+pad"
- "NEAR: price near zone X"
- "CORE: inside zone X"

---

## 7. 3m FSM (src/core/fsm_3m.py)

Implement an FSM class with:
- current_state
- context storage (anchor candle info, timers, last signals)
- method `step(candle_3m, indicators, zone_state, config) -> Decision`

### 7.1 Anchor Candle Detection
Anchor at time t if:
- volume[t] >= ANCHOR_VOL_MULT * SMA(volume, ANCHOR_VOL_SMA_N)
- body[t] >= ANCHOR_BODY_ATR_MULT * ATR_3m[t]
- AND chase filter passes at detection time: abs(close-EMA)/ATR <= CHASE_DIST_MAX_ATR
When detected:
- store A_open, A_close, A_mid, A_dir (+1/-1), anchor_ts, anchor_index

If zone_state != CORE or NEAR?:
- Only allow anchor detection in CORE or NEAR.
- If OUTSIDE: disable detection.

### 7.2 Transition rules
- OUTSIDE => force IDLE, clear anchor.
- NEAR => allow ANCHOR_FOUND, PULLBACK_WAIT, but prohibit ENTRY (do not transition to IN_TRADE).
- CORE => full transitions allowed.

State flow:
IDLE
  -> ANCHOR_FOUND (on anchor detected)
ANCHOR_FOUND
  -> PULLBACK_WAIT (immediate)
PULLBACK_WAIT
  -> ENTRY_READY if pullback+phenomena+RSI condition satisfied and not expired
ENTRY_READY
  -> IN_TRADE if trigger confirms re-advance (below)
  -> IDLE if expired or invalidated
IN_TRADE
  -> EXIT_COOLDOWN on exit condition or time stop or hard stop
EXIT_COOLDOWN
  -> IDLE after cooldown candles

### 7.3 Pullback / Retest logic around A_mid
For LONG (A_dir=+1):
- price returns near A_mid: abs(close - A_mid) <= 0.5*ATR_3m (configurable if you add)
- AND phenomena_ok == True (see below)
- AND RSI confirms turn: RSI rising vs previous (or RSI crosses up its short SMA)

For SHORT similarly around A_mid with RSI falling.

Invalidation:
- LONG invalid if close < A_open (or low < A_open depending config; use close by default)
- SHORT invalid if close > A_open
Expire:
- if candles since anchor >= ANCHOR_EXPIRE_CANDLES_3M: invalid -> IDLE

### 7.4 Phenomenon-based conditions (replace W/inverted hammer/doji)
Implement function `phenomena_ok(recent candles, indicators) -> (bool, details)`
Compute these boolean checks at current bar t:
A) low_fail:
  - Mode "2of4": min(low[t],low[t-1]) >= min(low[t-2],low[t-3])
B) lower_wick_ratio >= PH_LOWER_WICK_RATIO_MIN
C) body shrink: body[t] <= PH_BODY_SHRINK_FACTOR * mean(body over last PH_K)
D) range shrink: range[t] <= PH_RANGE_SHRINK_FACTOR * mean(range over last PH_K)

phenomena_ok true if at least PH_REQUIREMENTS_MIN_COUNT of (A,B,C,D) are true.
Return details for logging.

### 7.5 Entry trigger (re-advance confirmation)
For LONG:
- close > A_mid AND close > previous candle high (1-bar micro break) OR close > max(high[t-1], high[t-2])
For SHORT:
- close < A_mid AND close < previous candle low OR close < min(low[t-1], low[t-2])

Entry is prohibited if zone_state != CORE.

### 7.6 Exit logic
Hard stop:
- LONG stop = min(last_swing_low_3m, A_open) (implement a simple swing low finder) or simply A_open if swing not available.
- SHORT stop = max(last_swing_high_3m, A_open)
For backtest, simulate stop if low <= stop (LONG) or high >= stop (SHORT).

Time stop:
- if candles in trade >= HOLD_MAX_CANDLES_3M => EXIT

RSI exit:
- LONG: if RSI >= 70 and turns down OR bearish divergence detected (optional)
- SHORT: if RSI <= 30 and turns up OR bullish divergence detected (optional)

Cooldown:
- after exit, block new entries for COOLDOWN_CANDLES_3M

### 7.7 Explanation generation
Every Decision must include:
- state transition summary
- main reason in 1-2 sentences
Examples:
- "CORE zone X: Anchor found (vol 2.8x, body 1.3 ATR). Waiting pullback."
- "Phenomena OK (low_fail, wick_ratio). RSI turned up. ENTRY_READY."
- "ENTRY_LONG confirmed: close reclaimed A_mid and broke micro high."
- "EXIT: time stop (6 candles) and RSI rolled over."

---

## 8. Risk Module (src/core/risk.py)

Implement simple position sizing:
- position_size = equity * RISK_PER_TRADE_PCT / stop_distance
- apply size multipliers when daily loss approaches DAILY_MAX_LOSS_PCT
- if daily loss exceeded => block entries

This should be independent from exchange. Provide functions:
- `compute_position_size(equity, entry, stop, config, daily_pnl_pct)`

---

## 9. Engine Wiring (src/core/engine.py)

Implement `CoreEngine` that:
- accepts 15m candles and 3m candles (synchronized by timestamps)
- updates zones periodically (on each new 15m candle)
- on each new 3m candle:
  - compute required indicators
  - determine zone_state using latest zones
  - feed to FSM.step(...)
  - if Decision includes ENTRY/EXIT, call risk sizing and output an OrderIntent (not actual order)

OrderIntent fields:
- side (LONG/SHORT)
- entry_type (MARKET/LIMIT placeholder)
- stop_price
- take_profit optional (not required)
- size
- explanation

---

## 10. Demo CLI (src/cli/run_backtest_demo.py)

Provide a simple runnable demo that:
- loads sample OHLCV CSV (user will replace)
- builds 15m and 3m candles
- runs CoreEngine over time
- prints Decisions and final summary:
  - number of anchors detected
  - entries
  - exits
  - win/loss counts (if you implement simple PnL)

Keep this simple and deterministic.

---

## 11. Tests (pytest)

Write unit tests for:
- zones: pivot detection + merge with ATR radius
- gate: CORE/NEAR/OUTSIDE classification
- anchor: detection triggers only when volume+body thresholds met and chase filter passes
- phenomena: A/B/C/D checks and min-count logic
- entry: LONG/SHORT triggers and invalidations (A_open break, expiry)
- explain: Decision explanation contains key phrases and is non-empty

All tests must be deterministic and not require external data.

---

## 12. README.md

Explain:
- what the Core system is (15m zones gate 3m FSM)
- how to configure
- how to run demo
- what each module does
- emphasize: no ML, explainable, conservative by design

---

## Implementation Notes
- Use pure python + standard libs; numpy allowed.
- Avoid pandas dependency in core logic (ok in demo).
- Ensure clean type hints.
- Keep calculations safe (division by zero).
- Add logging-friendly debug dict fields in Decision.

Now implement everything.
