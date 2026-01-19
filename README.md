# XRP Core Trading System

A rule-based automated trading system with 15-minute zones gating 3-minute FSM execution.

## Overview

This system implements a conservative, explainable trading approach:

- **15m Zone Engine**: Detects support/resistance zones as ranges (not lines) with strength scoring and time decay
- **15m Gate**: Controls 3m FSM operation based on price proximity to zones
- **3m FSM (Finite State Machine)**: Handles anchor detection, pullback/retest, phenomenon conditions, and entry/exit logic
- **Risk Management**: Position sizing based on risk percentage with daily loss limits

### Key Design Principles

- **No ML/HMM/XGB**: Pure rule-based logic
- **Explainable**: Every decision includes human-readable explanations
- **Conservative**: 15m zones gate 3m signals - no trades outside zones
- **Configurable**: All thresholds are configurable via YAML
- **Backtest-friendly**: No exchange dependency in core logic

## Project Structure

```
src/
  core/
    __init__.py       # Package exports
    config.py         # Configuration management
    types.py          # Data types (Candle, Zone, Signal, etc.)
    indicators.py     # Technical indicators (EMA, RSI, ATR, SMA)
    zones_15m.py      # 15m zone detection engine
    gate_15m.py       # Zone state gate (CORE/NEAR/OUTSIDE)
    fsm_3m.py         # 3m trading FSM
    risk.py           # Position sizing and risk management
    explain.py        # Explanation generation
    engine.py         # Main engine wiring
  cli/
    run_backtest_demo.py  # Demo backtest script
tests/
  test_zones_15m.py       # Zone detection tests
  test_gate_15m.py        # Gate classification tests
  test_fsm_anchor.py      # Anchor detection tests
  test_fsm_phenomena.py   # Phenomenon condition tests
  test_fsm_entry_exit.py  # Entry/exit logic tests
  test_explain.py         # Explanation tests
configs/
  core_default.yaml       # Default configuration
```

## Installation

```bash
# Clone or create the project
cd xrp-4

# Install dependencies (optional - only PyYAML required for config)
pip install pyyaml pytest

# Optional: Install numpy for faster calculations
pip install numpy
```

## Configuration

Edit `configs/core_default.yaml` to adjust parameters:

### 15m Zone Settings
- `zone.radius_multiplier`: Zone radius as multiple of ATR (default: 0.6)
- `zone.pad_multiplier`: NEAR zone pad as multiple of ATR (default: 0.4)
- `zone.decay_halflife_hours`: Zone strength decay (default: 12 hours)

### 3m FSM Settings
- `fsm.anchor_vol_mult`: Volume spike threshold (default: 2.5x)
- `fsm.anchor_body_atr_mult`: Body size threshold (default: 1.2x ATR)
- `fsm.chase_dist_max_atr`: Max distance from EMA (default: 1.5x ATR)
- `fsm.anchor_expire_candles`: Anchor expiry (default: 4 candles = 12 min)

### Phenomenon Conditions
- `phenomena.requirements_min_count`: Minimum conditions to pass (default: 3 of 4)
- `phenomena.lower_wick_ratio_min`: Minimum lower wick ratio (default: 0.4)

### Risk Settings
- `risk.risk_per_trade_pct`: Risk per trade (default: 0.3%)
- `risk.daily_max_loss_pct`: Daily loss limit (default: 2%)

## Usage

### Run Demo Backtest

```bash
# Using sample generated data
python -m src.cli.run_backtest_demo --sample --verbose

# Using your own CSV data
python -m src.cli.run_backtest_demo --csv data/xrp_3m.csv --verbose

# With custom config
python -m src.cli.run_backtest_demo --config configs/core_default.yaml --csv data/xrp.csv
```

### Programmatic Usage

```python
from src.core import CoreEngine, CoreConfig
from src.core.types import Candle, Signal

# Load configuration
config = CoreConfig.from_yaml('configs/core_default.yaml')

# Create engine
engine = CoreEngine(config, equity=10000.0)

# Feed 15m candles to update zones
for candle_15m in candles_15m:
    engine.feed_15m(candle_15m)

# Feed 3m candles and get decisions
for candle_3m in candles_3m:
    decision = engine.feed_3m(candle_3m)

    if decision.signal in (Signal.ENTRY_LONG, Signal.ENTRY_SHORT):
        order = engine.get_last_order_intent()
        print(f"Entry: {order.side} @ {order.entry_price}, stop={order.stop_price}")

    elif decision.signal == Signal.EXIT:
        print(f"Exit: {decision.explanation}")

# Get summary
print(engine.get_summary())
```

## Trading Logic

### Zone States

| State | Description | FSM Behavior |
|-------|-------------|--------------|
| **CORE** | Price inside zone range | Full operation allowed |
| **NEAR** | Price within zone + pad | Anchor detection allowed, entry blocked |
| **OUTSIDE** | Price far from zones | FSM forced to IDLE |

### FSM States

```
IDLE -> ANCHOR_FOUND -> PULLBACK_WAIT -> ENTRY_READY -> IN_TRADE -> EXIT_COOLDOWN -> IDLE
```

1. **IDLE**: Waiting for anchor candle
2. **ANCHOR_FOUND**: High-volume, large-body candle detected
3. **PULLBACK_WAIT**: Waiting for price to return to A_mid
4. **ENTRY_READY**: Pullback + phenomena confirmed, waiting for trigger
5. **IN_TRADE**: Position open, monitoring for exit
6. **EXIT_COOLDOWN**: After exit, blocking new entries

### Anchor Detection (기준봉)

An anchor candle must satisfy:
- Volume >= 2.5x SMA(volume, 50)
- Body >= 1.2x ATR
- Chase filter: |close - EMA20| / ATR <= 1.5

### Phenomenon Conditions

For entry confirmation, at least 3 of 4 must be true:
- **Low Fail**: min(low[t], low[t-1]) >= min(low[t-2], low[t-3])
- **Wick Ratio**: Lower wick ratio >= 0.4
- **Body Shrink**: Body <= 0.5x mean(recent bodies)
- **Range Shrink**: Range <= 0.7x mean(recent ranges)

### Entry Trigger

For LONG after phenomena OK:
- Close > A_mid AND close > max(prev highs)

### Exit Conditions

- **Hard Stop**: Price hits stop loss
- **Time Stop**: Max hold time (18 minutes default)
- **RSI Exit**: RSI >= 70 turning down (LONG) or <= 30 turning up (SHORT)

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_fsm_anchor.py -v

# Run with coverage
pytest tests/ --cov=src/core
```

## Module Reference

### types.py
Core data types including `Candle`, `Zone`, `ZoneState`, `Signal`, `Decision`, `OrderIntent`.

### indicators.py
Technical indicators without external dependencies:
- `ema(values, period)` - Exponential Moving Average
- `rsi(values, period)` - Relative Strength Index
- `atr(candles, period)` - Average True Range
- `sma(values, period)` - Simple Moving Average

### zones_15m.py
`ZoneEngine` class for detecting and maintaining support/resistance zones:
- Pivot high/low detection
- Reaction validation
- Zone merging with ATR-based radius
- Strength decay over time

### gate_15m.py
`ZoneGate` class for controlling FSM based on zone proximity:
- CORE/NEAR/OUTSIDE classification
- Entry permission control
- Explanation generation

### fsm_3m.py
`TradingFSM` class implementing the state machine:
- Anchor candle detection
- Pullback/retest logic
- Phenomenon condition checking
- Entry trigger confirmation
- Exit condition monitoring

### risk.py
`RiskManager` class for position sizing:
- Risk-based position sizing
- Daily loss limits
- Progressive size reduction

### engine.py
`CoreEngine` class wiring everything together:
- Zone updates on 15m candles
- FSM processing on 3m candles
- Order intent generation
- Statistics tracking

## License

MIT License
