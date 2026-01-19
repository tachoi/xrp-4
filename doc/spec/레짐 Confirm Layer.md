# Claude Code Prompt — TRANSITION→TREND 확정 규칙 + HIGH_VOL 자동 해제 조건 구현
## Project: XRP Perp Auto-Trading System (xrp_4 / xrp_trading_system)
## Goal
HMM 레짐 출력(HIGH_VOL, TRANSITION 등)을 “바로 매매”에 쓰지 않고,
**확정 레이어(Confirm Layer)** 를 추가하여:

1) ✅ TRANSITION → TREND_UP / TREND_DOWN “확정” 규칙을 수식 기반으로 구현  
2) ✅ HIGH_VOL “자동 해제(Exit)” + 쿨다운(aftershock) + 재트리거 연장 규칙 구현  
3) ✅ 모든 결정은 “reason 코드”로 로깅되어 백테스트 리포트에서 집계 가능

> IMPORTANT
- 코드/클래스/파일 경로는 반드시 본 프롬프트를 그대로 따를 것 (임의 변경 금지)
- 신규 컬럼 추가는 하지 말고, **이미 존재하는 feature 컬럼을 우선 사용**
- 없는 컬럼이 필요한 경우에만 최소 추가하고, feature builder 쪽까지 함께 패치
- “15m 판정 + 3m 진입” 구조 유지

---

## 0) Assumptions (Existing)
- feature row에는 아래 컬럼이 존재한다고 가정(없으면 생성 로직도 추가):
  - ret_15m, ret_1h, ewm_ret_15m, ewm_ret_1h
  - atr_pct_15m, atr_pct_1h
  - ewm_std_ret_15m, ewm_std_ret_1h
  - bb_width_15m, bb_width_15m_pct
  - range_comp_15m
  - body_ratio, upper_wick_ratio, lower_wick_ratio
  - vol_z_15m
  - price_z_from_ema_1h
  - ema_slope_15m (없을 확률 높음 → 없으면 추가)

- zone 관련(rolling HH/LL) 계산 컬럼이 없으면 ConfirmLayer에서 직접 rolling로 계산해도 됨(duckdb/df에서).
  - 최근 n개 15m high/low → HH_n, LL_n (혹은 box_high_15m, box_low_15m)

- HMM inference 결과는 `regime` 문자열로 들어온다고 가정:
  - "RANGE", "TREND_UP", "TREND_DOWN", "HIGH_VOL", "TRANSITION"

---

## 1) Add Config Params
### File
- `configs/base.yaml`

### Add keys (must keep naming exactly)
```yaml
CONFIRM:
  TF_DECISION: "15m"
  BOX_LOOKBACK_15M: 32            # 6~8 hours (15m)
  TREND_CONFIRM_B_ATR: 0.8        # theta_B
  TREND_CONFIRM_S_ATR: 0.07       # theta_S (ema slope normalized)
  TREND_CONFIRM_EWM_RET_SIGMA: 0.20  # theta_mu as 0.2*sigma
  TREND_CONFIRM_CONSEC_BARS: 2

  VOL_BASE_LOOKBACK_15M: 96       # 24h baseline window
  HIGH_VOL_LAMBDA_ON: 1.50
  HIGH_VOL_LAMBDA_OFF: 0.50
  HIGH_VOL_STABLE_N: 6
  HIGH_VOL_STABLE_K: 4
  HIGH_VOL_COOLDOWN_BARS_15M: 4   # 60m
  HIGH_VOL_RETRIGGER_EXTEND: true

2) Implement Confirm Layer
Create new module

src/xrp3/regime/confirm.py

Required public API

Implement these EXACT functions/classes (names must match):

from dataclasses import dataclass
from typing import Dict, Tuple, Optional

@dataclass
class ConfirmResult:
    confirmed_regime: str          # one of: "RANGE","TREND_UP","TREND_DOWN","HIGH_VOL","TRANSITION","NO_TRADE"
    reason: str                    # short reason code
    metrics: Dict[str, float]      # key computed stats for logging

class RegimeConfirmLayer:
    def __init__(self, cfg: dict):
        ...

    def confirm(
        self,
        regime_raw: str,           # HMM raw regime
        row_15m: dict,             # feature row at t (15m)
        hist_15m: "pd.DataFrame",  # last N rows including t, must contain high/low/close + needed features
        state: Optional[dict] = None,  # persistent state for cooldown/retrigger
    ) -> Tuple[ConfirmResult, dict]:
        """
        Returns (result, updated_state)
        updated_state must be JSON-serializable
        """
        ...

Confirm Logic (must follow exactly)
2.1 TRANSITION → TREND confirm

When regime_raw == "TRANSITION", allow confirmation to TREND_UP or TREND_DOWN ONLY if:

Shared computations

ATR = atr_pct_15m * close OR if only atr_pct exists, use atr_pct directly for normalization consistently (choose one and document in code)

rolling HH_n, LL_n over last BOX_LOOKBACK_15M bars from hist_15m:

HH_n = max(high)

LL_n = min(low)

Breakout magnitude

B_up = (close - HH_n_prev) / ATR

B_dn = (LL_n_prev - close) / ATR
Where HH_n_prev/LL_n_prev are computed on window excluding current bar if possible (avoid using same close to set HH/LL)

Drift

mu = ewm_ret_15m

sigma = ewm_std_ret_15m

require mu >= theta_mu where theta_mu = TREND_CONFIRM_EWM_RET_SIGMA * sigma for UP

require mu <= -theta_mu for DOWN

Slope

S = ema_slope_15m normalized by ATR (if ema_slope_15m is already normalized, treat as is)

require S >= TREND_CONFIRM_S_ATR for UP

require S <= -TREND_CONFIRM_S_ATR for DOWN

Consecutive bars

require breakout magnitude condition satisfied in >= TREND_CONFIRM_CONSEC_BARS within last that many bars

e.g. last 2 bars both satisfy B_up>=theta_B

If UP rules pass → confirmed_regime="TREND_UP" reason="TRANSITION_CONFIRM_UP"
If DOWN rules pass → confirmed_regime="TREND_DOWN" reason="TRANSITION_CONFIRM_DOWN"
Else remain TRANSITION reason="TRANSITION_NOT_CONFIRMED"

Also include metrics:

B_up, B_dn, mu, sigma, S, HH_n, LL_n, ATR

2.2 HIGH_VOL exit rules (+ cooldown)

We maintain a state machine for HIGH_VOL:

State keys:

state["high_vol_active"]: bool

state["high_vol_cooldown_left"]: int

state["high_vol_last_v"]: float (last V value)

Define volatility measure V:

choose primary: ewm_std_ret_15m (or atr_pct_15m). Use ONE.

compute baseline: median over last VOL_BASE_LOOKBACK_15M bars

compute MAD similarly (median absolute deviation)

Entry threshold:

V_hi_on = base + HIGH_VOL_LAMBDA_ON * MAD

Exit threshold (hysteresis):

V_hi_off = base + HIGH_VOL_LAMBDA_OFF * MAD

If regime_raw == "HIGH_VOL" OR V >= V_hi_on:

set high_vol_active=True

set cooldown_left = HIGH_VOL_COOLDOWN_BARS_15M (reset)

confirmed_regime="HIGH_VOL" reason="HIGH_VOL_ACTIVE"

Else if high_vol_active == True:

Evaluate exit only if V <= V_hi_off AND stability condition holds:

Over last HIGH_VOL_STABLE_N bars, count decreases: (V_t - V_{t-1}) <= 0

require count >= HIGH_VOL_STABLE_K

And require structure restored:

Either RANGE restored: range_comp_15m >= some safe threshold (use 0.8 default inside code if not configured) AND bb_width_15m not expanding

Or TREND restored: pass TREND confirm (but WITHOUT bb_width expansion requirement; just breakout+drift+slope+consec)

If exit satisfied:

set high_vol_active=False

set cooldown_left = HIGH_VOL_COOLDOWN_BARS_15M

confirmed_regime="TRANSITION" reason="HIGH_VOL_EXIT_TO_TRANSITION"

During cooldown:

if cooldown_left > 0:

decrement by 1 per new 15m bar

confirmed_regime="NO_TRADE" reason="HIGH_VOL_COOLDOWN"

BUT if HIGH_VOL re-triggers (V >= V_hi_on) and HIGH_VOL_RETRIGGER_EXTEND true → reactivate HIGH_VOL and reset cooldown

Always return updated_state.

2.3 Priority rule

Confirm order must be:

HIGH_VOL handling (can override any raw regime)

TRANSITION→TREND confirm

otherwise pass-through raw regime

3) Wire into existing regime pipeline
Find existing regime router / gates (likely paths)

src/xrp3/regime/ (gates.py, hmm inference wrapper)

search for where HMM regime is produced and passed to DecisionEngine

Required changes

Instantiate RegimeConfirmLayer using loaded cfg

Maintain persistent confirm state per symbol (XRPUSDT.P)

In backtest: keep in memory

In paper/live: store in an in-memory dict keyed by symbol (no DB needed)

Ensure downstream receives confirmed_regime not raw.

Pass reason and metrics into trade logs.

4) Logging + DuckDB report hooks
Add/extend trade log schema (duckdb)

Wherever you store per-bar/per-signal logs, include:

regime_raw

regime_confirmed

confirm_reason

confirm_metrics_json (stringified json)

If you already have a per-bar table, append columns there; if not, create a small confirm_events table:

run_id, ts, regime_raw, regime_confirmed, reason, metrics_json

5) Unit tests
Create tests

tests/test_regime_confirm.py

Test cases required:

TRANSITION not confirmed when breakout magnitude < theta_B

TRANSITION confirmed to TREND_UP when all conditions satisfied for 2 consecutive bars

HIGH_VOL activates when V >= V_hi_on even if raw regime != HIGH_VOL

HIGH_VOL does NOT exit unless (V <= off) AND stability AND structure restored

cooldown blocks trading with NO_TRADE reason, and retrigger extends

Use small synthetic DataFrame fixtures.

6) Deliverable

Implement code exactly with the specified file paths + names

All tests passing

Add minimal docstring explaining formulas and normalization choices

Provide a short console print sample in tests or docs showing:

regime_raw -> confirmed_regime + reason + key metrics

Notes / Non-negotiables

DO NOT change unrelated strategy logic

DO NOT refactor other modules except to wire ConfirmLayer

Keep computation deterministic and reproducible

If a required feature column is missing, add it in the existing feature builder with the same column name

END