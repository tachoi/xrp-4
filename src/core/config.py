"""
Configuration management for the XRP Core Trading System.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import yaml
from pathlib import Path


@dataclass
class ZoneConfig:
    """15m zone engine configuration."""
    atr_period: int = 14
    radius_multiplier: float = 0.6  # zone radius = r * ATR_15m
    pad_multiplier: float = 0.4     # pad = p * ATR_15m for NEAR
    pivot_lookback: int = 2         # swing pivot left/right window
    reaction_lookahead: int = 12    # 15m candles to confirm reaction
    reaction_k_atr: float = 0.8     # rebound >= k*ATR counts as reaction
    max_per_side: int = 6           # keep top zones by strength
    decay_halflife_hours: float = 12.0
    # Zone expiration settings
    min_strength_threshold: float = 0.1   # remove zone if strength < threshold
    max_age_hours: float = 48.0           # remove zone if older than this
    price_distance_atr: float = 8.0       # remove zone if price moves > N*ATR away
    break_removal_enabled: bool = True    # remove zone on decisive break
    # Single zone mode
    single_zone_mode: bool = True         # only use nearest zone (ignore others)


@dataclass
class FSMConfig:
    """3m FSM engine configuration."""
    ema_period: int = 20
    rsi_period: int = 14
    atr_period: int = 14
    anchor_vol_mult: float = 2.5      # volume >= mult * SMA(volume)
    anchor_vol_sma_n: int = 50
    anchor_body_atr_mult: float = 1.2  # |close-open| >= mult * ATR
    chase_dist_max_atr: float = 1.5   # abs(price-EMA)/ATR <= Dmax
    anchor_expire_candles: int = 4    # 12 minutes at 3m
    hold_max_candles: int = 6         # 18 minutes
    cooldown_candles: int = 3
    pullback_tolerance_atr: float = 0.5  # pullback near A_mid tolerance
    # Pullback entry settings (avoid chasing)
    entry_pullback_enabled: bool = True      # Wait for pullback before entry
    entry_pullback_atr: float = 0.3          # Pullback depth in ATR units
    entry_pullback_max_candles: int = 3      # Max candles to wait for pullback
    # Exit improvements
    min_candles_for_exit: int = 5            # Min candles before tech exits
    min_mfe_for_tech_exit: float = 0.15      # Min MFE % before tech exits allowed
    take_profit_r: float = 2.0        # Take profit at 2R (2x risk)
    trailing_start_r: float = 1.0     # Start trailing after 1R profit
    trailing_step_r: float = 0.5      # Trail by 0.5R increments
    time_stop_ignore_if_profit: bool = True  # Don't time stop if in profit
    # SHORT-specific early exit (SHORT loses 77.8% of profits)
    short_early_exit_enabled: bool = True    # Enable early exit for SHORT
    short_take_profit_pct: float = 0.10      # Take profit at 0.10% for SHORT
    short_min_candles: int = 2               # Min candles before SHORT exit
    short_disable_ema_exit: bool = True      # Disable EMA cross exit for SHORT (0% WR)
    # LONG entry filter (avoid counter-trend)
    long_trend_filter_enabled: bool = True   # Check EMA slope before LONG entry
    long_ema_slope_threshold: float = -0.3   # Block LONG if EMA slope < threshold (downtrend)
    # SHORT entry filter (avoid counter-trend)
    short_trend_filter_enabled: bool = False # Check EMA slope before SHORT entry
    short_ema_slope_threshold: float = 0.3   # Block SHORT if EMA slope > threshold
    # Breakeven trailing stop
    breakeven_trail_enabled: bool = True     # Move stop to breakeven after profit
    breakeven_trigger_pct: float = 0.10      # Trigger breakeven at 0.10% profit
    breakeven_offset_pct: float = 0.02       # Offset from entry (small buffer)


@dataclass
class PhenomenaConfig:
    """Phenomenon-based condition configuration."""
    lookback_k: int = 8               # lookback for mean calculations
    low_fail_mode: str = "2of4"       # low fail detection mode
    lower_wick_ratio_min: float = 0.4
    body_shrink_factor: float = 0.5
    range_shrink_factor: float = 0.7
    requirements_min_count: int = 3   # min checks to pass


@dataclass
class RiskConfig:
    """Risk and position sizing configuration."""
    risk_per_trade_pct: float = 0.3
    daily_max_loss_pct: float = 2.0
    drawdown_size_reduce_steps: List[float] = field(
        default_factory=lambda: [0.5, 0.3]
    )


@dataclass
class CoreConfig:
    """Main configuration container."""
    zone: ZoneConfig = field(default_factory=ZoneConfig)
    fsm: FSMConfig = field(default_factory=FSMConfig)
    phenomena: PhenomenaConfig = field(default_factory=PhenomenaConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)

    @classmethod
    def from_yaml(cls, path: str) -> 'CoreConfig':
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CoreConfig':
        """Create config from dictionary."""
        zone_data = data.get('zone', {})
        fsm_data = data.get('fsm', {})
        phenomena_data = data.get('phenomena', {})
        risk_data = data.get('risk', {})

        zone_config = ZoneConfig(
            atr_period=zone_data.get('atr_period', 14),
            radius_multiplier=zone_data.get('radius_multiplier', 0.6),
            pad_multiplier=zone_data.get('pad_multiplier', 0.4),
            pivot_lookback=zone_data.get('pivot_lookback', 2),
            reaction_lookahead=zone_data.get('reaction_lookahead', 12),
            reaction_k_atr=zone_data.get('reaction_k_atr', 0.8),
            max_per_side=zone_data.get('max_per_side', 6),
            decay_halflife_hours=zone_data.get('decay_halflife_hours', 12.0),
            min_strength_threshold=zone_data.get('min_strength_threshold', 0.1),
            max_age_hours=zone_data.get('max_age_hours', 48.0),
            price_distance_atr=zone_data.get('price_distance_atr', 8.0),
            break_removal_enabled=zone_data.get('break_removal_enabled', True),
            single_zone_mode=zone_data.get('single_zone_mode', True),
        )

        fsm_config = FSMConfig(
            ema_period=fsm_data.get('ema_period', 20),
            rsi_period=fsm_data.get('rsi_period', 14),
            atr_period=fsm_data.get('atr_period', 14),
            anchor_vol_mult=fsm_data.get('anchor_vol_mult', 2.5),
            anchor_vol_sma_n=fsm_data.get('anchor_vol_sma_n', 50),
            anchor_body_atr_mult=fsm_data.get('anchor_body_atr_mult', 1.2),
            chase_dist_max_atr=fsm_data.get('chase_dist_max_atr', 1.5),
            anchor_expire_candles=fsm_data.get('anchor_expire_candles', 4),
            hold_max_candles=fsm_data.get('hold_max_candles', 6),
            cooldown_candles=fsm_data.get('cooldown_candles', 3),
            pullback_tolerance_atr=fsm_data.get('pullback_tolerance_atr', 0.5),
            entry_pullback_enabled=fsm_data.get('entry_pullback_enabled', True),
            entry_pullback_atr=fsm_data.get('entry_pullback_atr', 0.3),
            entry_pullback_max_candles=fsm_data.get('entry_pullback_max_candles', 3),
            min_candles_for_exit=fsm_data.get('min_candles_for_exit', 5),
            min_mfe_for_tech_exit=fsm_data.get('min_mfe_for_tech_exit', 0.15),
            take_profit_r=fsm_data.get('take_profit_r', 2.0),
            trailing_start_r=fsm_data.get('trailing_start_r', 1.0),
            trailing_step_r=fsm_data.get('trailing_step_r', 0.5),
            time_stop_ignore_if_profit=fsm_data.get('time_stop_ignore_if_profit', True),
            short_early_exit_enabled=fsm_data.get('short_early_exit_enabled', True),
            short_take_profit_pct=fsm_data.get('short_take_profit_pct', 0.10),
            short_min_candles=fsm_data.get('short_min_candles', 2),
            short_disable_ema_exit=fsm_data.get('short_disable_ema_exit', True),
            long_trend_filter_enabled=fsm_data.get('long_trend_filter_enabled', True),
            long_ema_slope_threshold=fsm_data.get('long_ema_slope_threshold', -0.3),
            short_trend_filter_enabled=fsm_data.get('short_trend_filter_enabled', False),
            short_ema_slope_threshold=fsm_data.get('short_ema_slope_threshold', 0.3),
            breakeven_trail_enabled=fsm_data.get('breakeven_trail_enabled', True),
            breakeven_trigger_pct=fsm_data.get('breakeven_trigger_pct', 0.10),
            breakeven_offset_pct=fsm_data.get('breakeven_offset_pct', 0.02),
        )

        phenomena_config = PhenomenaConfig(
            lookback_k=phenomena_data.get('lookback_k', 8),
            low_fail_mode=phenomena_data.get('low_fail_mode', '2of4'),
            lower_wick_ratio_min=phenomena_data.get('lower_wick_ratio_min', 0.4),
            body_shrink_factor=phenomena_data.get('body_shrink_factor', 0.5),
            range_shrink_factor=phenomena_data.get('range_shrink_factor', 0.7),
            requirements_min_count=phenomena_data.get('requirements_min_count', 3),
        )

        risk_config = RiskConfig(
            risk_per_trade_pct=risk_data.get('risk_per_trade_pct', 0.3),
            daily_max_loss_pct=risk_data.get('daily_max_loss_pct', 2.0),
            drawdown_size_reduce_steps=risk_data.get(
                'drawdown_size_reduce_steps', [0.5, 0.3]
            ),
        )

        return cls(
            zone=zone_config,
            fsm=fsm_config,
            phenomena=phenomena_config,
            risk=risk_config,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'zone': {
                'atr_period': self.zone.atr_period,
                'radius_multiplier': self.zone.radius_multiplier,
                'pad_multiplier': self.zone.pad_multiplier,
                'pivot_lookback': self.zone.pivot_lookback,
                'reaction_lookahead': self.zone.reaction_lookahead,
                'reaction_k_atr': self.zone.reaction_k_atr,
                'max_per_side': self.zone.max_per_side,
                'decay_halflife_hours': self.zone.decay_halflife_hours,
                'min_strength_threshold': self.zone.min_strength_threshold,
                'max_age_hours': self.zone.max_age_hours,
                'price_distance_atr': self.zone.price_distance_atr,
                'break_removal_enabled': self.zone.break_removal_enabled,
                'single_zone_mode': self.zone.single_zone_mode,
            },
            'fsm': {
                'ema_period': self.fsm.ema_period,
                'rsi_period': self.fsm.rsi_period,
                'atr_period': self.fsm.atr_period,
                'anchor_vol_mult': self.fsm.anchor_vol_mult,
                'anchor_vol_sma_n': self.fsm.anchor_vol_sma_n,
                'anchor_body_atr_mult': self.fsm.anchor_body_atr_mult,
                'chase_dist_max_atr': self.fsm.chase_dist_max_atr,
                'anchor_expire_candles': self.fsm.anchor_expire_candles,
                'hold_max_candles': self.fsm.hold_max_candles,
                'cooldown_candles': self.fsm.cooldown_candles,
                'pullback_tolerance_atr': self.fsm.pullback_tolerance_atr,
                'entry_pullback_enabled': self.fsm.entry_pullback_enabled,
                'entry_pullback_atr': self.fsm.entry_pullback_atr,
                'entry_pullback_max_candles': self.fsm.entry_pullback_max_candles,
                'min_candles_for_exit': self.fsm.min_candles_for_exit,
                'min_mfe_for_tech_exit': self.fsm.min_mfe_for_tech_exit,
                'take_profit_r': self.fsm.take_profit_r,
                'trailing_start_r': self.fsm.trailing_start_r,
                'trailing_step_r': self.fsm.trailing_step_r,
                'time_stop_ignore_if_profit': self.fsm.time_stop_ignore_if_profit,
                'short_early_exit_enabled': self.fsm.short_early_exit_enabled,
                'short_take_profit_pct': self.fsm.short_take_profit_pct,
                'short_min_candles': self.fsm.short_min_candles,
                'short_disable_ema_exit': self.fsm.short_disable_ema_exit,
                'long_trend_filter_enabled': self.fsm.long_trend_filter_enabled,
                'long_ema_slope_threshold': self.fsm.long_ema_slope_threshold,
                'short_trend_filter_enabled': self.fsm.short_trend_filter_enabled,
                'short_ema_slope_threshold': self.fsm.short_ema_slope_threshold,
                'breakeven_trail_enabled': self.fsm.breakeven_trail_enabled,
                'breakeven_trigger_pct': self.fsm.breakeven_trigger_pct,
                'breakeven_offset_pct': self.fsm.breakeven_offset_pct,
            },
            'phenomena': {
                'lookback_k': self.phenomena.lookback_k,
                'low_fail_mode': self.phenomena.low_fail_mode,
                'lower_wick_ratio_min': self.phenomena.lower_wick_ratio_min,
                'body_shrink_factor': self.phenomena.body_shrink_factor,
                'range_shrink_factor': self.phenomena.range_shrink_factor,
                'requirements_min_count': self.phenomena.requirements_min_count,
            },
            'risk': {
                'risk_per_trade_pct': self.risk.risk_per_trade_pct,
                'daily_max_loss_pct': self.risk.daily_max_loss_pct,
                'drawdown_size_reduce_steps': self.risk.drawdown_size_reduce_steps,
            },
        }

    def save_yaml(self, path: str) -> None:
        """Save configuration to YAML file."""
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
