# Claude Code Prompt — FSM + DecisionEngine (Confirm Layer 연동) 구현
## Project: XRP Perp Auto-Trading System (xrp_4 / xrp_trading_system)
## Scope
- **FSM (Signal Generator)**: confirmed_regime 기반으로 “진입/청산 후보 신호” 생성
- **DecisionEngine (Executor Policy)**: 후보 신호를 승인/거절 + 사이징 + 주문/리스크 정책 적용
- **Confirm Layer 출력(confirmed_regime / reason / metrics)** 을 FSM/DecisionEngine이 반드시 사용

> NON-NEGOTIABLE
- 파일 경로/클래스/함수 이름을 아래에 지정한 대로 **그대로** 만들 것
- 기존 코드 리팩터링 금지 (배선만 추가)
- 모든 결과는 reason 코드와 함께 로깅 가능해야 함 (DuckDB)
- Backtest 모드와 Paper/Live 모드 모두에서 동일한 인터페이스 유지

---

## 0) 시스템 레이어 관계 (반드시 준수)
Pipeline must be:

Features → HMM(raw) → ConfirmLayer(confirmed_regime) → FSM(candidate signal) → DecisionEngine(final action) → Broker

- FSM 입력은 `confirmed_regime`만 사용 (raw regime 사용 금지)
- DecisionEngine은 confirmed_regime에 따라 “DENY_ALL / RISK_OFF / NORMAL” 우선 정책을 강제

---

## 1) Config 추가
### File
- `configs/base.yaml`

### Add keys (exact naming)
```yaml
FSM:
  TF_ENTRY: "3m"
  TF_CONTEXT: "15m"
  EMA_FAST_3M: 20
  EMA_SLOW_3M: 50
  RSI_LEN_3M: 14
  ATR_LEN_3M: 14

  # Range (bounce)
  RANGE_PULLBACK_ATR: 0.6
  RANGE_MIN_ZONE_STRENGTH: 0.000010   # 예: strength v2 band 이후 조정
  RANGE_MAX_SPREAD_ATR: 0.25          # (optional) 슬리피지/스프레드 제한

  # Trend (pullback)
  TREND_PULLBACK_TO_EMA_ATR: 0.5
  TREND_MIN_EMA_SLOPE_15M: 0.07
  TREND_CONFIRM_REQUIRED: true         # confirmed_regime == TREND_UP/DOWN
  TREND_CHASE_FORBIDDEN: true          # pullback 없으면 금지

  # Transition 제한
  TRANSITION_ALLOW_BREAKOUT_ONLY: true
  TRANSITION_SIZE_MULT: 0.4

RISK:
  BASE_SIZE_USDT: 100
  MAX_LEVERAGE: 3
  STOP_ATR_MULT: 1.2
  TP_ATR_MULT: 1.0
  TRAIL_ATR_MULT: 1.0
  MAX_HOLD_BARS_3M: 60

DECISION:
  DENY_IN_HIGH_VOL: true
  DENY_IN_NO_TRADE: true
  DENY_BREAKOUT_IN_COOLDOWN: true
  COOLDOWN_SIZE_MULT: 0.5

  # XGB optional hook (can be noop)
  XGB_ENABLED: false
  XGB_PMIN_RANGE: 0.55
  XGB_PMIN_TREND: 0.60

2) Data contracts (must standardize)
Create file

src/xrp3/core/types.py

Add these dataclasses (exact names):

from dataclasses import dataclass
from typing import Dict, Optional, Literal

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

ActionType = Literal["NO_ACTION", "OPEN_LONG", "OPEN_SHORT", "CLOSE", "REDUCE", "INCREASE"]

@dataclass
class ConfirmContext:
    regime_raw: str
    regime_confirmed: str      # RANGE/TREND_UP/TREND_DOWN/HIGH_VOL/TRANSITION/NO_TRADE
    confirm_reason: str
    confirm_metrics: Dict[str, float]

@dataclass
class MarketContext:
    symbol: str
    ts: int                   # epoch ms
    price: float
    row_3m: Dict[str, float]  # current 3m features (must include ema/rsi/atr etc.)
    row_15m: Dict[str, float] # current 15m features
    zone: Dict[str, float]    # {support, resistance, strength, dist_to_support, dist_to_resistance, ...}

@dataclass
class PositionState:
    side: Literal["FLAT","LONG","SHORT"]
    entry_price: float
    size: float
    entry_ts: int
    bars_held_3m: int
    unrealized_pnl: float

@dataclass
class CandidateSignal:
    signal: SignalType
    score: float                 # 0~1
    reason: str                  # short reason code
    params: Dict[str, float]     # stop/tp/trail targets etc.

@dataclass
class Decision:
    action: ActionType
    size: float
    reason: str
    meta: Dict[str, float]

3) Implement FSM (Candidate Signal Generator)
Create module

src/xrp3/core/fsm.py

Required class API (exact)
from typing import Optional
from .types import ConfirmContext, MarketContext, PositionState, CandidateSignal

class TradingFSM:
    def __init__(self, cfg: dict):
        ...

    def step(
        self,
        ctx: MarketContext,
        confirm: ConfirmContext,
        pos: PositionState,
        fsm_state: Optional[dict] = None,
    ) -> tuple[CandidateSignal, dict]:
        """
        Returns (candidate_signal, updated_fsm_state)
        updated_fsm_state must be JSON-serializable.
        """
        ...

FSM design rules (must follow)

FSM MUST be a regime-conditioned state machine:

Priority 0) hard states

if confirm.regime_confirmed in {"HIGH_VOL","NO_TRADE"}:

emit CandidateSignal(signal="HOLD", score=0, reason="FSM_BLOCKED_REGIME", params={})

do not mutate states except counters

1) RANGE regime → Bounce only (baseline)

Entry candidates:

LONG_BOUNCE if price near support within RANGE_PULLBACK_ATR * atr_3m AND zone.strength >= RANGE_MIN_ZONE_STRENGTH

SHORT_BOUNCE if price near resistance similarly

“near” uses ctx.zone.dist_to_support / atr_3m, dist_to_resistance / atr_3m

If both near (tight box), choose higher strength side (or HOLD)

Exit candidates (if pos not FLAT):

EXIT if price returns to midline/ema OR if opposite zone reached OR max_hold exceeded

Emit reason codes:

"RANGE_LONG_BOUNCE_NEAR_SUPPORT"

"RANGE_SHORT_BOUNCE_NEAR_RESIST"

"RANGE_EXIT_TARGET"

"RANGE_EXIT_TIMEOUT"

2) TREND_UP/DOWN regime → Pullback only (no chase)

If TREND_CHASE_FORBIDDEN:

Require pullback condition:

TREND_UP: price <= ema_fast_3m + TREND_PULLBACK_TO_EMA_ATR * atr_3m AND ret_3m turns positive (rebound)

TREND_DOWN: price >= ema_fast_3m - TREND_PULLBACK_TO_EMA_ATR * atr_3m AND ret_3m turns negative (rejection)

Entry candidates:

TREND_UP → LONG_TREND_PULLBACK

TREND_DOWN → SHORT_TREND_PULLBACK

Exit candidates:

EXIT if price crosses ema_slow_3m against direction OR time stop OR trailing stop hint

Reason codes:

"TREND_UP_PULLBACK_ENTRY"

"TREND_DOWN_PULLBACK_ENTRY"

"TREND_EXIT_EMA_SLOW_CROSS"

"TREND_EXIT_TIMEOUT"

3) TRANSITION regime → 제한 모드

If TRANSITION_ALLOW_BREAKOUT_ONLY:

only allow LONG_BREAKOUT/SHORT_BREAKOUT if confirm_metrics include breakout magnitude (B_up/B_dn) and exceeds 1.0 (or derive from 15m confirm metrics)

else HOLD

Otherwise HOLD

Reason codes:

"TRANSITION_BREAKOUT_ONLY_PASS"

"TRANSITION_BLOCK"

4) Score policy (simple, deterministic)

score in [0,1]:

base = 0.5

add based on zone strength (range) or ema_slope_15m (trend) or breakout magnitude (transition)

clamp

Return params (targets):

stop_atr_mult, tp_atr_mult, trail_atr_mult (from cfg.RISK)

plus any local fields needed

4) Implement DecisionEngine (Final Executor Policy)
Create module

src/xrp3/core/decision_engine.py

Required API (exact)
from typing import Optional
from .types import ConfirmContext, MarketContext, PositionState, CandidateSignal, Decision

class DecisionEngine:
    def __init__(self, cfg: dict):
        ...

    def decide(
        self,
        ctx: MarketContext,
        confirm: ConfirmContext,
        pos: PositionState,
        cand: CandidateSignal,
        engine_state: Optional[dict] = None,
    ) -> tuple[Decision, dict]:
        """
        Returns (decision, updated_engine_state)
        """
        ...

DecisionEngine rules (must follow)
0) Hard deny policy

if confirmed_regime == "HIGH_VOL" and DECISION.DENY_IN_HIGH_VOL: deny all opens, allow CLOSE only

if confirmed_regime == "NO_TRADE" and DECISION.DENY_IN_NO_TRADE: deny all opens, allow CLOSE only
Return:

action="NO_ACTION" for opens, or "CLOSE" if pos open and emergency exit conditions met

Reason:

"DE_DENY_HIGH_VOL"

"DE_DENY_NO_TRADE"

1) Cooldown policy (from ConfirmLayer)

If confirm.confirm_reason contains "COOLDOWN" or regime_confirmed == "NO_TRADE" because cooldown:

size multiplier = COOLDOWN_SIZE_MULT

if DENY_BREAKOUT_IN_COOLDOWN: block cand signals that are BREAKOUT types
Reason:

"DE_COOLDOWN_RISK_OFF"

2) Regime-based size scaling

base_size = RISK.BASE_SIZE_USDT

if regime_confirmed == "TRANSITION": size *= TRANSITION_SIZE_MULT

if regime_confirmed in {"TREND_UP","TREND_DOWN"}: size *= 1.0

if regime_confirmed == "RANGE": size *= 1.0

3) Position-aware actions

If pos.FLAT:

if cand.signal indicates LONG_*: action="OPEN_LONG" size=scaled size

if cand.signal indicates SHORT_*: action="OPEN_SHORT"

else NO_ACTION

If pos.LONG:

if cand.signal == "EXIT" or opposite entry signal:

action="CLOSE"

else NO_ACTION (or INCREASE if you want, but default is NO)

If pos.SHORT: symmetric

Reason codes:

"DE_OPEN_FROM_CANDIDATE"

"DE_CLOSE_FROM_EXIT_SIGNAL"

"DE_HOLD"

4) (Optional) XGB Approval Hook (noop by default)

If DECISION.XGB_ENABLED:

call xgb_gate.predict_proba(features) to get p_win

thresholds:

range signals: p >= XGB_PMIN_RANGE

trend signals: p >= XGB_PMIN_TREND

if fail: deny open, reason="DE_XGB_REJECT"

Implement stub interface:

src/xrp3/core/xgb_gate.py with class XGBApprovalGate that can be disabled cleanly.

5) Wiring in runner (Backtest + Paper)
Find existing runner/loop

Search for where per-bar processing occurs (likely in src/xrp3/backtest/ and src/xrp3/paper/ or similar).
You must:

call ConfirmLayer first -> ConfirmContext

then call FSM.step -> CandidateSignal

then DecisionEngine.decide -> Decision

then execute order simulation or broker call

Maintain persistent:

confirm_state per symbol

fsm_state per symbol

engine_state per symbol

6) Logging to DuckDB

Wherever trades/signals are logged, add:

regime_raw, regime_confirmed, confirm_reason

cand_signal, cand_score, cand_reason

decision_action, decision_reason, decision_size

metrics_json (confirm_metrics + cand.params + decision.meta) as json string

If no suitable table exists:

Create signal_events table in duckdb

(run_id, ts, symbol, regime_raw, regime_confirmed, confirm_reason,
cand_signal, cand_score, cand_reason,
decision_action, decision_size, decision_reason,
metrics_json)

7) Tests
Create

tests/test_fsm_and_decision_engine.py

Include minimal synthetic tests:

HIGH_VOL → FSM emits HOLD and DecisionEngine denies opens

TRANSITION without breakout metrics → HOLD

TREND_UP requires pullback (no chase) → without pullback HOLD, with pullback OPEN_LONG

RANGE near support with strong zone → OPEN_LONG (bounce)

Cooldown blocks breakout but allows exit/hold

Use simple dict rows for ctx.row_3m/15m and zone.

8) Deliverables

New files:

src/xrp3/core/types.py

src/xrp3/core/fsm.py

src/xrp3/core/decision_engine.py

src/xrp3/core/xgb_gate.py (stub)

tests/test_fsm_and_decision_engine.py

Updated files:

configs/base.yaml

runner wiring + duckdb logging

All tests must pass.

END