#!/usr/bin/env python3
"""
Cross-validated parameter optimization to reduce overfitting.
Tests each parameter combination on both training and validation sets.
"""
import sys
from pathlib import Path
import itertools
import time

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
from src.xrp4.data.loader import load_ohlcv_sync
from src.core.config import CoreConfig, ZoneConfig, FSMConfig, PhenomenaConfig, RiskConfig
from src.core.backtest_vectorized import run_vectorized_backtest


def create_config(params):
    """Create CoreConfig from parameter dict."""
    zone_config = ZoneConfig(
        atr_period=14,
        radius_multiplier=params.get('zone_radius_mult', 0.6),
        pad_multiplier=0.4,
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
        anchor_vol_mult=params.get('anchor_vol_mult', 1.5),
        anchor_vol_sma_n=50,
        anchor_body_atr_mult=params.get('anchor_body_atr_mult', 1.2),
        chase_dist_max_atr=params.get('chase_dist_max_atr', 1.5),
        anchor_expire_candles=params.get('anchor_expire_candles', 4),
        hold_max_candles=params.get('hold_max_candles', 12),
        cooldown_candles=3,
        pullback_tolerance_atr=params.get('pullback_tolerance_atr', 0.6),
    )

    phenomena_config = PhenomenaConfig(
        lookback_k=8,
        low_fail_mode="2of4",
        lower_wick_ratio_min=params.get('lower_wick_ratio_min', 0.4),
        body_shrink_factor=0.5,
        range_shrink_factor=0.7,
        requirements_min_count=params.get('phenomena_min_count', 1),
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


def load_data(start, end):
    """Load and prepare data."""
    df = load_ohlcv_sync(symbol='XRPUSDT', start=start, end=end, validate=True, fix_issues=True)
    df_15m = df.set_index('timestamp').resample('15min').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
    }).dropna().reset_index()
    return df, df_15m


def run_cv_optimization():
    """Run cross-validated optimization."""
    print("=" * 70)
    print("XRP Core Trading System - Cross-Validated Optimization")
    print("=" * 70)

    # Load training and validation data
    print("\nLoading data...")
    train_df, train_15m = load_data('2024-07-01', '2025-07-01')
    val_df, val_15m = load_data('2025-07-01', '2026-01-01')
    print(f"Training: {len(train_df)} candles, Validation: {len(val_df)} candles")

    # Parameter grid - focus on robustness
    param_grid = {
        'anchor_vol_mult': [1.5, 2.0, 2.5],
        'anchor_body_atr_mult': [1.0, 1.2, 1.5],
        'phenomena_min_count': [1, 2, 3],
        'hold_max_candles': [8, 12, 16],
        'pullback_tolerance_atr': [0.5, 0.7, 0.9],
        'chase_dist_max_atr': [1.0, 1.5, 2.0],
    }

    # Generate combinations
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
    total = len(combinations)

    print(f"\nTesting {total} parameter combinations...")
    print("-" * 70)

    results = []
    start_time = time.time()

    for i, params in enumerate(combinations):
        config = create_config(params)

        # Run on training
        train_result = run_vectorized_backtest(train_df, train_15m, config)

        # Run on validation
        val_result = run_vectorized_backtest(val_df, val_15m, config)

        # Calculate combined score (weighted average favoring validation)
        # Also penalize large gaps between train/val performance
        train_pnl = train_result.total_pnl_pct
        val_pnl = val_result.total_pnl_pct
        gap = abs(train_pnl - val_pnl)

        # Score: 40% train + 60% val - gap penalty
        combined_score = 0.4 * train_pnl + 0.6 * val_pnl - 0.5 * gap

        # Minimum trades filter
        min_trades = min(train_result.total_trades, val_result.total_trades)
        if min_trades < 3:
            combined_score = -999  # Penalize too few trades

        results.append({
            'params': params,
            'train_pnl': train_pnl,
            'train_trades': train_result.total_trades,
            'train_wr': train_result.win_rate,
            'train_pf': train_result.profit_factor,
            'val_pnl': val_pnl,
            'val_trades': val_result.total_trades,
            'val_wr': val_result.win_rate,
            'val_pf': val_result.profit_factor,
            'gap': gap,
            'combined_score': combined_score,
        })

        # Progress
        if (i + 1) % 50 == 0 or i == total - 1:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            eta = (total - i - 1) / rate if rate > 0 else 0
            best_score = max(r['combined_score'] for r in results)
            print(f"Progress: {i+1}/{total} | Rate: {rate:.1f}/s | ETA: {eta:.0f}s | Best Score: {best_score:.2f}")

    # Sort by combined score
    results.sort(key=lambda r: r['combined_score'], reverse=True)

    # Print top results
    print("\n" + "=" * 70)
    print("TOP 10 RESULTS (Cross-Validated)")
    print("=" * 70)
    print(f"{'Rank':<5} | {'Train PnL':>10} | {'Val PnL':>10} | {'Gap':>6} | {'Score':>7} | {'T Trades':>8} | {'V Trades':>8}")
    print("-" * 70)

    for i, r in enumerate(results[:10], 1):
        print(f"{i:<5} | {r['train_pnl']:>+9.2f}% | {r['val_pnl']:>+9.2f}% | {r['gap']:>5.2f}% | {r['combined_score']:>7.2f} | {r['train_trades']:>8} | {r['val_trades']:>8}")

    # Print best config details
    best = results[0]
    print("\n" + "=" * 70)
    print("BEST CONFIGURATION (Cross-Validated)")
    print("=" * 70)
    print(f"Training:   PnL {best['train_pnl']:+.2f}%, {best['train_trades']} trades, {best['train_wr']:.1f}% WR, PF {best['train_pf']:.2f}")
    print(f"Validation: PnL {best['val_pnl']:+.2f}%, {best['val_trades']} trades, {best['val_wr']:.1f}% WR, PF {best['val_pf']:.2f}")
    print(f"Gap: {best['gap']:.2f}%, Combined Score: {best['combined_score']:.2f}")
    print("\nParameters:")
    for k, v in best['params'].items():
        print(f"  {k}: {v}")

    # Save best config
    config = create_config(best['params'])
    output_path = '/mnt/c/Users/tae am choi/xrp-4/configs/core_cv_optimized.yaml'
    config.to_yaml(output_path)
    print(f"\nSaved to: {output_path}")

    return results


if __name__ == '__main__':
    run_cv_optimization()
