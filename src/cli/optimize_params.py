#!/usr/bin/env python3
"""
Parameter optimization for the XRP Core Trading System.

Usage:
    cd xrp-4
    python src/cli/optimize_params.py --start 2024-01-01 --end 2024-06-30
"""
import argparse
import sys
import json
import itertools
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, asdict
import time

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd

from src.core.types import Candle, Signal
from src.core.config import CoreConfig, ZoneConfig, FSMConfig, PhenomenaConfig, RiskConfig
from src.core.engine import run_backtest
from src.core.backtest_vectorized import run_vectorized_backtest, VectorizedResult
from src.xrp4.data.loader import load_ohlcv_sync


@dataclass
class OptimizationResult:
    """Result of one parameter combination test."""
    params: Dict[str, Any]
    total_pnl_pct: float
    win_rate: float
    total_trades: int
    wins: int
    losses: int
    anchors: int
    sharpe_approx: float  # Simplified Sharpe
    profit_factor: float
    max_consecutive_losses: int


# Parameter search space
PARAM_GRID = {
    # Zone parameters
    'zone_radius_mult': [0.4, 0.6, 0.8],
    'zone_pad_mult': [0.3, 0.4, 0.5],

    # Anchor detection
    'anchor_vol_mult': [2.0, 2.5, 3.0],
    'anchor_body_atr_mult': [1.0, 1.2, 1.5],
    'chase_dist_max_atr': [1.0, 1.5, 2.0],

    # FSM timing
    'anchor_expire_candles': [4, 6, 8],
    'hold_max_candles': [6, 10, 15],
    'pullback_tolerance_atr': [0.3, 0.5, 0.7],

    # Phenomena conditions
    'phenomena_min_count': [2, 3],
    'lower_wick_ratio_min': [0.3, 0.4, 0.5],
}

# Reduced grid for faster optimization
PARAM_GRID_FAST = {
    'anchor_vol_mult': [1.5, 2.0, 2.5],
    'anchor_body_atr_mult': [0.8, 1.0, 1.2],
    'phenomena_min_count': [1, 2],
    'hold_max_candles': [8, 12, 16],
    'pullback_tolerance_atr': [0.6, 0.8, 1.0],
}


def df_to_candles(df: pd.DataFrame) -> List[Candle]:
    """Convert pandas DataFrame to list of Candle objects."""
    candles = []
    for _, row in df.iterrows():
        ts = int(row['timestamp'].timestamp() * 1000)
        candles.append(Candle(
            ts=ts,
            open=float(row['open']),
            high=float(row['high']),
            low=float(row['low']),
            close=float(row['close']),
            volume=float(row['volume']),
        ))
    return candles


def resample_to_15m(candles_3m: List[Candle]) -> List[Candle]:
    """Resample 3m candles to 15m candles."""
    candles_15m = []
    for i in range(0, len(candles_3m), 5):
        group = candles_3m[i:i+5]
        if len(group) < 5:
            continue
        candle_15m = Candle(
            ts=group[0].ts,
            open=group[0].open,
            high=max(c.high for c in group),
            low=min(c.low for c in group),
            close=group[-1].close,
            volume=sum(c.volume for c in group),
        )
        candles_15m.append(candle_15m)
    return candles_15m


def create_config(params: Dict[str, Any]) -> CoreConfig:
    """Create CoreConfig from parameter dict."""
    zone_config = ZoneConfig(
        atr_period=14,
        radius_multiplier=params.get('zone_radius_mult', 0.6),
        pad_multiplier=params.get('zone_pad_mult', 0.4),
        pivot_lookback=2,
        reaction_lookahead=12,
        reaction_k_atr=0.8,
        max_per_side=6,
        decay_halflife_hours=12.0,
    )

    fsm_config = FSMConfig(
        ema_period=20,
        rsi_period=14,
        atr_period=14,
        anchor_vol_mult=params.get('anchor_vol_mult', 2.5),
        anchor_vol_sma_n=50,
        anchor_body_atr_mult=params.get('anchor_body_atr_mult', 1.2),
        chase_dist_max_atr=params.get('chase_dist_max_atr', 1.5),
        anchor_expire_candles=params.get('anchor_expire_candles', 4),
        hold_max_candles=params.get('hold_max_candles', 6),
        cooldown_candles=3,
        pullback_tolerance_atr=params.get('pullback_tolerance_atr', 0.5),
    )

    phenomena_config = PhenomenaConfig(
        lookback_k=8,
        low_fail_mode="2of4",
        lower_wick_ratio_min=params.get('lower_wick_ratio_min', 0.4),
        body_shrink_factor=0.5,
        range_shrink_factor=0.7,
        requirements_min_count=params.get('phenomena_min_count', 3),
    )

    risk_config = RiskConfig(
        risk_per_trade_pct=0.3,
        daily_max_loss_pct=2.0,
        drawdown_size_reduce_steps=[0.5, 0.3],
    )

    return CoreConfig(
        zone=zone_config,
        fsm=fsm_config,
        phenomena=phenomena_config,
        risk=risk_config,
    )


def calculate_metrics(decisions: List, stats) -> Tuple[float, float, int]:
    """Calculate additional metrics from backtest results."""
    # Extract trade results
    trades = []
    in_trade = False
    entry_price = 0
    is_long = True

    for d in decisions:
        if d.signal in (Signal.ENTRY_LONG, Signal.ENTRY_SHORT):
            in_trade = True
            entry_price = d.entry_price or 0
            is_long = d.signal == Signal.ENTRY_LONG
        elif d.signal == Signal.EXIT and in_trade:
            # Simplified - we don't have exit price in decision
            in_trade = False

    # Calculate profit factor
    if stats.losses > 0 and stats.total_pnl_pct < 0:
        # Rough estimate
        avg_loss = abs(stats.total_pnl_pct) / max(1, stats.losses)
        avg_win = (stats.total_pnl_pct + abs(stats.total_pnl_pct)) / max(1, stats.wins) if stats.wins > 0 else 0
        profit_factor = (stats.wins * avg_win) / (stats.losses * avg_loss) if avg_loss > 0 else 0
    elif stats.total_pnl_pct > 0:
        profit_factor = 2.0  # Winning overall
    else:
        profit_factor = 0.0

    # Simplified Sharpe (PnL / volatility proxy)
    if stats.entries > 0:
        avg_pnl = stats.total_pnl_pct / stats.entries
        sharpe = avg_pnl / max(0.1, abs(stats.total_pnl_pct / 10))  # Rough estimate
    else:
        sharpe = 0.0

    # Max consecutive losses
    max_consec = 0
    current_consec = 0
    for d in decisions:
        if d.signal == Signal.EXIT:
            # Check if this was a loss (simplified)
            if stats.losses > stats.wins:  # More losses overall
                current_consec += 1
                max_consec = max(max_consec, current_consec)
            else:
                current_consec = 0

    return sharpe, profit_factor, max_consec


def run_single_optimization(params: Dict[str, Any],
                            candles_15m: List[Candle],
                            candles_3m: List[Candle],
                            equity: float = 10000.0) -> OptimizationResult:
    """Run backtest with single parameter combination."""
    config = create_config(params)

    stats, decisions = run_backtest(
        candles_15m, candles_3m, config,
        equity=equity,
        verbose=False
    )

    win_rate = stats.wins / max(1, stats.entries) * 100
    sharpe, profit_factor, max_consec = calculate_metrics(decisions, stats)

    return OptimizationResult(
        params=params,
        total_pnl_pct=stats.total_pnl_pct,
        win_rate=win_rate,
        total_trades=stats.entries,
        wins=stats.wins,
        losses=stats.losses,
        anchors=stats.anchors_detected,
        sharpe_approx=sharpe,
        profit_factor=profit_factor,
        max_consecutive_losses=max_consec,
    )


def run_vectorized_optimization(params: Dict[str, Any],
                                 df_3m: pd.DataFrame,
                                 df_15m: pd.DataFrame) -> OptimizationResult:
    """Run vectorized backtest with single parameter combination (fast)."""
    config = create_config(params)

    result = run_vectorized_backtest(df_3m, df_15m, config)

    # Estimate sharpe from avg trade pnl
    sharpe = result.avg_trade_pnl / max(0.1, abs(result.max_drawdown_pct)) if result.max_drawdown_pct != 0 else 0

    return OptimizationResult(
        params=params,
        total_pnl_pct=result.total_pnl_pct,
        win_rate=result.win_rate,
        total_trades=result.total_trades,
        wins=result.wins,
        losses=result.losses,
        anchors=result.anchors_detected,
        sharpe_approx=sharpe,
        profit_factor=result.profit_factor,
        max_consecutive_losses=0,  # Not tracked in vectorized version
    )


def generate_param_combinations(grid: Dict[str, List]) -> List[Dict[str, Any]]:
    """Generate all parameter combinations from grid."""
    keys = list(grid.keys())
    values = list(grid.values())

    combinations = []
    for combo in itertools.product(*values):
        params = dict(zip(keys, combo))
        combinations.append(params)

    return combinations


def optimize(candles_15m: List[Candle],
             candles_3m: List[Candle],
             param_grid: Dict[str, List],
             equity: float = 10000.0,
             top_n: int = 10) -> List[OptimizationResult]:
    """Run optimization across all parameter combinations."""
    combinations = generate_param_combinations(param_grid)
    total = len(combinations)

    print(f"Testing {total} parameter combinations...")
    print("-" * 60)

    results = []
    start_time = time.time()

    for i, params in enumerate(combinations):
        result = run_single_optimization(params, candles_15m, candles_3m, equity)
        results.append(result)

        # Progress update
        if (i + 1) % 10 == 0 or i == total - 1:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            remaining = (total - i - 1) / rate if rate > 0 else 0
            print(f"Progress: {i+1}/{total} ({(i+1)/total*100:.1f}%) | "
                  f"ETA: {remaining:.0f}s | "
                  f"Best PnL so far: {max(r.total_pnl_pct for r in results):.2f}%")

    # Sort by PnL (primary) and win rate (secondary)
    results.sort(key=lambda r: (r.total_pnl_pct, r.win_rate), reverse=True)

    return results[:top_n]


def optimize_vectorized(df_3m: pd.DataFrame,
                        df_15m: pd.DataFrame,
                        param_grid: Dict[str, List],
                        top_n: int = 10) -> List[OptimizationResult]:
    """Run vectorized optimization (much faster)."""
    combinations = generate_param_combinations(param_grid)
    total = len(combinations)

    print(f"Testing {total} parameter combinations (vectorized)...")
    print("-" * 60)

    results = []
    start_time = time.time()

    for i, params in enumerate(combinations):
        result = run_vectorized_optimization(params, df_3m, df_15m)
        results.append(result)

        # Progress update
        if (i + 1) % 10 == 0 or i == total - 1:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            remaining = (total - i - 1) / rate if rate > 0 else 0
            best_pnl = max(r.total_pnl_pct for r in results)
            print(f"Progress: {i+1}/{total} ({(i+1)/total*100:.1f}%) | "
                  f"Rate: {rate:.1f}/s | ETA: {remaining:.0f}s | "
                  f"Best PnL: {best_pnl:+.2f}%")

    # Sort by PnL (primary) and win rate (secondary)
    results.sort(key=lambda r: (r.total_pnl_pct, r.win_rate), reverse=True)

    return results[:top_n]


def print_results(results: List[OptimizationResult]):
    """Print optimization results."""
    print("\n" + "=" * 80)
    print("TOP OPTIMIZATION RESULTS")
    print("=" * 80)

    for i, r in enumerate(results, 1):
        print(f"\n[Rank {i}] PnL: {r.total_pnl_pct:+.2f}% | "
              f"Win Rate: {r.win_rate:.1f}% | "
              f"Trades: {r.total_trades} | "
              f"W/L: {r.wins}/{r.losses}")
        print(f"         Anchors: {r.anchors} | "
              f"Profit Factor: {r.profit_factor:.2f}")
        print(f"         Params: {r.params}")


def save_results(results: List[OptimizationResult], output_path: str):
    """Save results to JSON file."""
    data = [asdict(r) for r in results]
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='XRP Core Trading System - Parameter Optimization'
    )
    parser.add_argument('--start', '-s', type=str, default='2024-01-01',
                        help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', '-e', type=str, default='2024-03-31',
                        help='End date (YYYY-MM-DD)')
    parser.add_argument('--symbol', type=str, default='XRPUSDT',
                        help='Trading symbol')
    parser.add_argument('--equity', type=float, default=10000.0,
                        help='Starting equity')
    parser.add_argument('--fast', action='store_true',
                        help='Use reduced parameter grid for faster optimization')
    parser.add_argument('--top', type=int, default=10,
                        help='Number of top results to show')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output JSON file path')
    parser.add_argument('--sequential', action='store_true',
                        help='Use sequential (slow) backtest instead of vectorized')

    args = parser.parse_args()

    print("=" * 60)
    print("XRP Core Trading System - Parameter Optimization")
    print("=" * 60)
    print(f"Symbol: {args.symbol}")
    print(f"Period: {args.start} to {args.end}")
    print(f"Mode: {'Fast (reduced grid)' if args.fast else 'Full grid'}")
    print(f"Engine: {'Sequential' if args.sequential else 'Vectorized (fast)'}")
    print()

    # Load data
    print(f"\nLoading data from TimescaleDB...")
    try:
        df = load_ohlcv_sync(
            symbol=args.symbol,
            start=args.start,
            end=args.end,
            validate=True,
            fix_issues=True
        )
        print(f"Loaded {len(df)} 3m candles")
    except Exception as e:
        print(f"ERROR: Failed to load data: {e}")
        sys.exit(1)

    if len(df) < 1000:
        print("WARNING: Limited data may affect optimization quality")

    # Select parameter grid
    param_grid = PARAM_GRID_FAST if args.fast else PARAM_GRID

    # Run optimization
    print()
    if args.sequential:
        # Convert to candles for sequential backtest
        candles_3m = df_to_candles(df)
        candles_15m = resample_to_15m(candles_3m)
        print(f"Created {len(candles_3m)} 3m candles, {len(candles_15m)} 15m candles")
        results = optimize(candles_15m, candles_3m, param_grid, args.equity, args.top)
    else:
        # Use vectorized backtest (much faster)
        # Create 15m DataFrame by resampling
        df_15m = df.set_index('timestamp').resample('15min').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna().reset_index()
        print(f"Created {len(df)} 3m rows, {len(df_15m)} 15m rows (DataFrames)")
        results = optimize_vectorized(df, df_15m, param_grid, args.top)

    # Print results
    print_results(results)

    # Save results
    if args.output:
        save_results(results, args.output)
    else:
        default_output = f"/mnt/c/Users/tae am choi/xrp-4/outputs/optimization_{args.start}_{args.end}.json"
        Path(default_output).parent.mkdir(parents=True, exist_ok=True)
        save_results(results, default_output)

    # Print best config
    if results:
        best = results[0]
        print("\n" + "=" * 60)
        print("BEST CONFIGURATION")
        print("=" * 60)
        print(f"PnL: {best.total_pnl_pct:+.2f}%")
        print(f"Win Rate: {best.win_rate:.1f}%")
        print(f"Trades: {best.total_trades}")
        print("\nOptimal Parameters:")
        for k, v in best.params.items():
            print(f"  {k}: {v}")


if __name__ == '__main__':
    main()
