"""Configuration loader for XRP-4 Step0 baseline."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class BoxFilterConfig:
    """Box filter configuration."""

    enabled: bool = True
    lookback_bars: int = 64
    atr_period: int = 14
    atr_tf: str = "15m"
    min_height_atr: float = 3.0
    update_every_bars: int = 10

    # Reaction filter (optional)
    use_reaction_filter: bool = False
    edge_band_atr_k: float = 0.3
    reaction_lookback_touches: int = 20
    reaction_horizon_bars: int = 6
    move_target_atr_k: float = 1.0
    reaction_min_rate: float = 0.48

    # Cost filter (optional)
    use_cost_filter: bool = False
    cost_safety_mult: float = 2.0


@dataclass
class ReportConfig:
    """Report configuration."""

    dump_box_filter_diagnostics: bool = True


@dataclass
class Step0Config:
    """Step0 baseline configuration."""

    # Data parameters
    symbol: str
    timeframe: str
    start: str
    end: str

    # Capital & Risk
    initial_capital: float
    risk_per_trade: float
    max_position: int
    cooldown_bars: int

    # Costs
    fee_bps: float
    slippage_bps: float

    # Zone parameters
    zone_method: str
    pivot_left: int
    pivot_right: int
    zone_lookback_bars: int
    zone_rebuild_freq: int
    max_zones: int
    atr_tf: str
    zone_width_atr_k: float

    # Signal bands
    touch_band_atr_k: float
    break_band_atr_k: float
    confirm_mode: str

    # Exit parameters
    sl_band_atr_k: float
    rr_bounce: float
    rr_breakout: float

    # Output
    output_dir: str

    # Fields with default values (must come last)
    signal_mode: str = "all"  # "all", "bounce_only", "breakout_only"
    run_id: Optional[str] = None

    # Box filter (Step 0.1)
    box_filter: BoxFilterConfig = field(default_factory=BoxFilterConfig)

    # Report options
    report: ReportConfig = field(default_factory=ReportConfig)

    @property
    def fee_rate(self) -> float:
        """Convert fee from bps to rate (e.g., 4 bps -> 0.0004)."""
        return self.fee_bps / 10000.0

    @property
    def slippage_rate(self) -> float:
        """Convert slippage from bps to rate."""
        return self.slippage_bps / 10000.0


def load_config(config_path: str | Path, overrides: Optional[Dict[str, Any]] = None) -> Step0Config:
    """Load configuration from YAML file.

    Args:
        config_path: Path to YAML config file
        overrides: Optional dict of config overrides (e.g., {"box_filter.min_height_atr": 3.5})

    Returns:
        Step0Config instance
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        data = yaml.safe_load(f)

    # Apply overrides
    if overrides:
        for key, value in overrides.items():
            _set_nested_value(data, key, value)

    # Auto-generate run_id if not provided
    if not data.get("run_id"):
        from datetime import datetime
        data["run_id"] = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Parse nested configs
    box_filter_data = data.pop("box_filter", {})
    report_data = data.pop("report", {})

    # Create nested config objects
    box_filter = BoxFilterConfig(**box_filter_data) if box_filter_data else BoxFilterConfig()
    report = ReportConfig(**report_data) if report_data else ReportConfig()

    return Step0Config(**data, box_filter=box_filter, report=report)


def _set_nested_value(data: dict, key: str, value: Any) -> None:
    """Set a nested value in a dictionary using dot notation.

    Args:
        data: Dictionary to update
        key: Dot-separated key path (e.g., "box_filter.min_height_atr")
        value: Value to set
    """
    keys = key.split(".")
    current = data

    for k in keys[:-1]:
        if k not in current:
            current[k] = {}
        current = current[k]

    # Convert value to appropriate type
    final_key = keys[-1]
    if isinstance(value, str):
        # Try to parse as number
        try:
            if "." in value:
                value = float(value)
            else:
                value = int(value)
        except ValueError:
            # Try to parse as boolean
            if value.lower() == "true":
                value = True
            elif value.lower() == "false":
                value = False

    current[final_key] = value


def parse_overrides(override_args: list) -> Dict[str, Any]:
    """Parse CLI override arguments.

    Args:
        override_args: List of "key=value" strings

    Returns:
        Dict of overrides
    """
    overrides = {}
    for arg in override_args:
        if "=" in arg:
            key, value = arg.split("=", 1)
            overrides[key] = value
    return overrides
