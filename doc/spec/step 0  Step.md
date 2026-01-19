# Claude Code Prompt — Build `xrp-4` Step0 baseline using xrp-3 DB (TimescaleDB OHLCV + MongoDB features)

You are in my workspace where `xrp-3` already exists and is working.
I want to create a new sibling project `xrp-4` that reuses xrp-3 code as much as possible.

CRITICAL FACTS (MUST FOLLOW):
- About ~3 years of OHLCV price data is stored in TimescaleDB.
- Feature data is stored in MongoDB.
- `xrp-3` already uses these databases. DO NOT reimplement DB connectivity if reusable modules exist.
- `xrp-4` Step0 baseline MUST use the same data sources (TimescaleDB for candles; MongoDB optional only if needed).
- Step0 is NO-ML: no HMM, no XGB, no FSM. Do not import or call those modules.

PRIMARY GOAL:
- Implement Step0 baseline backtest: Zone-based Bounce/Breakout + fixed risk, with fees+slippage, equity curve, trade list, summary metrics.

SECONDARY GOAL:
- Minimal changes to `xrp-3`. Prefer new code under `xrp-4` and reuse via imports.

================================================================================
0) FIRST ACTION: DISCOVER xrp-3 DB ACCESS LAYER
================================================================================
1) Print a focused tree of the relevant xrp-3 modules:
   - DB connectors / clients
   - TimescaleDB candle loader / query code
   - MongoDB feature loader code
   - Any existing backtest runner, metrics, reporting utils
2) Locate:
   - how DB URIs are configured (env vars? yaml? .env?)
   - which tables/collections contain candles/features
   - the canonical dataframe schema returned by the loader (columns names)
3) Identify EXACT import paths we can reuse in xrp-4.

IMPORTANT:
- If xrp-3 uses `src/xrp3/...` packages, preserve package layout in xrp-4 with `src/xrp4/...`.
- Do not break xrp-3 imports.

================================================================================
1) xrp-4 DIRECTORY STRUCTURE (CREATE)
================================================================================
Create a new folder `xrp-4/`:

xrp-4/
  configs/
    step0.yaml
  src/
    xrp4/
      __init__.py
      run_step0.py
      config.py
      data/
        candles.py        # wrapper over xrp-3 Timescale candle loader
        features.py       # wrapper over xrp-3 Mongo feature loader (optional)
      zones/
        zone_builder.py
        zone_models.py
      strategy/
        signals.py
        risk.py
        executor.py
      backtest/
        engine.py
        metrics.py
      report/
        writer.py
  README.md

Rule: xrp-4 should be runnable via:
- `python -m xrp4.run_step0 --config configs/step0.yaml`

================================================================================
2) CONFIG (step0.yaml)
================================================================================
Create `configs/step0.yaml` with (include defaults):
- symbol: "XRPUSDT"
- timeframe: "3m"
- start: "YYYY-MM-DD"
- end: "YYYY-MM-DD"
- initial_capital: 10000

# Costs
- fee_bps: 4.0               # configurable
- slippage_bps: 2.0          # configurable

# Risk
- risk_per_trade: 0.0025     # 0.25% equity
- max_position: 1
- cooldown_bars: 3

# Zones
- zone_method: "pivot_cluster"
- pivot_left: 3
- pivot_right: 3
- zone_lookback_bars: 500
- max_zones: 12
- atr_tf: "15m"              # if available; fallback "3m"
- zone_width_atr_k: 1.0

# Signal bands (ATR-multiples)
- touch_band_atr_k: 0.2
- break_band_atr_k: 0.2
- confirm_mode: "next_close" # "next_close" only for Step0 (simple)

# Exits
- sl_band_atr_k: 0.2
- rr_bounce: 1.2
- rr_breakout: 1.5

================================================================================
3) DATA LOADING (MUST REUSE xrp-3)
================================================================================
A) Candles (TimescaleDB)
- Implement `xrp4/data/candles.py` as a THIN wrapper over the existing xrp-3 candle loader.
- Do NOT write raw SQL unless xrp-3 lacks it.
- Output must be a pandas DataFrame sorted by timestamp, with at least:
  ["ts","open","high","low","close","volume"]
- Ensure timezone/naive datetime handling matches xrp-3.

B) Features (MongoDB)
- Step0 baseline should NOT depend on Mongo features.
- ONLY use Mongo features if ATR is already computed/stored there and is easiest to reuse.
- Otherwise compute ATR locally from candles (simple TA).

C) DB Config
- Reuse xrp-3’s mechanism (env vars / yaml). xrp-4 should read the same config keys.
- Do not hardcode secrets.

================================================================================
4) STRATEGY SPEC (NO-ML)
================================================================================
Zones:
- Build zones from pivot highs/lows within rolling lookback window.
- Cluster pivot prices: merge pivots within `zone_width` into a single zone (deterministic).
- zone_width = zone_width_atr_k * ATR(atr_tf)
- Keep top `max_zones` by touch count.

Signals:
1) Bounce:
- LONG when price touches support zone and confirms:
  - touch: candle.low <= zone.high AND candle.low >= zone.low - touch_band
  - confirm: next candle close > zone.low
- SHORT symmetric at resistance.

2) Breakout:
- LONG when close breaks above zone.high + break_band, confirm by next close staying above.
- SHORT symmetric.

Conflict:
- If multiple zones produce signals, pick the closest zone to price.
- If bounce & breakout both true => NO_TRADE.

Risk/Execution:
- Fixed risk per trade (fraction of equity).
- SL/TP:
  - Bounce: SL outside zone by sl_band, TP fixed RR (rr_bounce)
  - Breakout: SL around breakout level by sl_band, TP fixed RR (rr_breakout)
- Fees+slippage applied on entry and exit.
- Single position at a time.
- Cooldown bars after exit.

================================================================================
5) BACKTEST + REPORTING
================================================================================
Backtest engine requirements:
- Iterate candles
- Build/update zones on schedule:
  - simplest: rebuild zones every N bars (e.g., every 30 bars) using the last zone_lookback_bars
- Record:
  - trades list with entry/exit time, side, entry/exit px, size, pnl, R, reason
  - equity curve (per bar)
Metrics:
- PF, winrate, avg R, trade count, MDD
- Monthly PnL table

Outputs:
- Write to `xrp-4/outputs/step0/<run_id>/`
  - trades.csv
  - equity.csv
  - summary.json
  - monthly_pnl.csv

================================================================================
6) IMPLEMENTATION PLAN (STRICT ORDER)
================================================================================
Step 1) Inspect xrp-3, identify & reuse:
- Timescale candle loader module path + function signature
- Mongo feature loader (optional)
- Any metrics/report utils

Step 2) Implement xrp-4 skeleton & wrappers.

Step 3) Implement ATR computation locally if not reused:
- ATR(14) computed from high/low/close with Wilder smoothing.

Step 4) Implement ZoneBuilder.

Step 5) Implement signal generation + executor.

Step 6) Implement backtest engine + metrics.

Step 7) Implement CLI + README.

================================================================================
7) FINAL RESPONSE FORMAT
================================================================================
After completing:
1) List created/changed files
2) Show exact run command
3) Show sample stdout summary
4) Confirm we did NOT use HMM/XGB/FSM anywhere.

Now proceed.
