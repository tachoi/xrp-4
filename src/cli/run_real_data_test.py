#!/usr/bin/env python3
"""
Test the XRP Core Trading System with real XRP data from TimescaleDB.

Usage:
    cd xrp-4
    python src/cli/run_real_data_test.py --start 2024-01-01 --end 2024-01-31

Requires:
    - TimescaleDB running with XRP data
    - Proper .env configuration in xrp-4
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd

from src.core.types import Candle, Signal
from src.core.config import CoreConfig
from src.core.engine import CoreEngine, run_backtest
from src.xrp4.data.loader import load_ohlcv_sync


def df_to_candles(df: pd.DataFrame) -> List[Candle]:
    """Convert pandas DataFrame to list of Candle objects (optimized)."""
    # Vectorized conversion - much faster than iterrows
    timestamps = (df['timestamp'].astype('int64') // 10**6).values
    opens = df['open'].astype(float).values
    highs = df['high'].astype(float).values
    lows = df['low'].astype(float).values
    closes = df['close'].astype(float).values
    volumes = df['volume'].astype(float).values

    candles = [
        Candle(ts=int(ts), open=o, high=h, low=l, close=c, volume=v)
        for ts, o, h, l, c, v in zip(timestamps, opens, highs, lows, closes, volumes)
    ]
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


def main():
    parser = argparse.ArgumentParser(
        description='XRP Core Trading System - Real Data Test'
    )
    parser.add_argument(
        '--start', '-s',
        type=str,
        default='2024-01-01',
        help='Start date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--end', '-e',
        type=str,
        default='2024-01-31',
        help='End date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--symbol',
        type=str,
        default='XRPUSDT',
        help='Trading symbol'
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        default=None,
        help='Path to config YAML file'
    )
    parser.add_argument(
        '--equity',
        type=float,
        default=10000.0,
        help='Starting equity'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Print all signals'
    )

    args = parser.parse_args()

    print("=" * 60)
    print("XRP Core Trading System - Real Data Test")
    print("=" * 60)
    print(f"Symbol: {args.symbol}")
    print(f"Period: {args.start} to {args.end}")
    print(f"Equity: ${args.equity:,.2f}")
    print()

    # Load data from TimescaleDB
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
        print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"Price range: ${df['low'].min():.4f} to ${df['high'].max():.4f}")
    except Exception as e:
        print(f"ERROR: Failed to load data: {e}")
        print("Make sure TimescaleDB is running and .env is configured")
        sys.exit(1)

    if len(df) < 100:
        print("ERROR: Not enough data (need at least 100 candles)")
        sys.exit(1)

    # Convert to Candle objects
    print("\nConverting to Candle objects...")
    candles_3m = df_to_candles(df)
    candles_15m = resample_to_15m(candles_3m)
    print(f"Created {len(candles_3m)} 3m candles, {len(candles_15m)} 15m candles")

    # Load config
    if args.config:
        print(f"\nLoading config from: {args.config}")
        config = CoreConfig.from_yaml(args.config)
    else:
        print("\nUsing default configuration")
        config = CoreConfig()

    # Run backtest
    print("\n" + "=" * 60)
    print("Running backtest...")
    print("=" * 60)

    stats, decisions = run_backtest(
        candles_15m, candles_3m, config,
        equity=args.equity,
        verbose=args.verbose
    )

    # Print results
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)
    print(stats.summary())

    # Signal breakdown
    signals = [d for d in decisions if d.signal != Signal.NONE]
    print(f"\nTotal signals: {len(signals)}")

    by_type = {}
    for d in signals:
        sig = d.signal.value
        by_type[sig] = by_type.get(sig, 0) + 1

    for sig, count in sorted(by_type.items()):
        print(f"  {sig}: {count}")

    # Win rate and PnL
    if stats.entries > 0:
        win_rate = stats.wins / stats.entries * 100
        print(f"\nPerformance:")
        print(f"  Win Rate: {win_rate:.1f}%")
        print(f"  Total PnL: {stats.total_pnl_pct:.2f}%")
        print(f"  Avg PnL/Trade: {stats.total_pnl_pct / stats.entries:.2f}%")

    # Extract and display trade details
    trades = []
    current_trade = {}
    for d in decisions:
        if d.signal in (Signal.ENTRY_LONG, Signal.ENTRY_SHORT):
            current_trade = {
                'side': d.side.value if d.side else '?',
                'entry': d.entry_price,
                'stop': d.stop_price,
            }
        elif d.signal == Signal.EXIT and current_trade:
            current_trade['exit_reason'] = d.explanation[:50]
            trades.append(current_trade)
            current_trade = {}

    if trades:
        print(f"\n" + "-" * 40)
        print("Trade Details:")
        print("-" * 40)
        for i, t in enumerate(trades, 1):
            print(f"  {i}. {t['side']} Entry={t.get('entry', '?'):.4f}, "
                  f"Stop={t.get('stop', '?'):.4f}")
            print(f"     Exit: {t.get('exit_reason', 'N/A')}")

    # Print sample signals
    if signals:
        print("\n" + "-" * 40)
        print("Sample Signals (first 10):")
        print("-" * 40)
        for d in signals[:10]:
            ts = datetime.fromtimestamp(candles_3m[0].ts / 1000)  # Approximate
            print(f"  [{d.signal.value:12}] {d.explanation[:55]}...")

        if len(signals) > 10:
            print("\n" + "-" * 40)
            print("Last 10 Signals:")
            print("-" * 40)
            for d in signals[-10:]:
                print(f"  [{d.signal.value:12}] {d.explanation[:55]}...")

    print("\n" + "=" * 60)
    print("Test complete.")


if __name__ == '__main__':
    main()
