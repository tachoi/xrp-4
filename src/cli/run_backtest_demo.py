#!/usr/bin/env python3
"""
Demo backtest script for the XRP Core Trading System.

Usage:
    python -m src.cli.run_backtest_demo [--config CONFIG_PATH] [--csv CSV_PATH]

The CSV should have columns: timestamp, open, high, low, close, volume
Timestamp can be epoch milliseconds or ISO format.
"""
import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.types import Candle, Signal
from src.core.config import CoreConfig
from src.core.engine import CoreEngine, run_backtest


def parse_timestamp(ts_str: str) -> int:
    """Parse timestamp string to epoch milliseconds."""
    # Try epoch milliseconds first
    try:
        ts = int(ts_str)
        if ts > 1e12:  # Already in milliseconds
            return ts
        return ts * 1000  # Convert seconds to milliseconds
    except ValueError:
        pass

    # Try ISO format
    try:
        dt = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
        return int(dt.timestamp() * 1000)
    except ValueError:
        pass

    # Try common formats
    for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S', '%Y/%m/%d %H:%M:%S']:
        try:
            dt = datetime.strptime(ts_str, fmt)
            return int(dt.timestamp() * 1000)
        except ValueError:
            continue

    raise ValueError(f"Cannot parse timestamp: {ts_str}")


def load_candles_from_csv(csv_path: str) -> List[Candle]:
    """
    Load candles from CSV file.

    Expected columns: timestamp, open, high, low, close, volume
    """
    candles = []

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)

        for row in reader:
            # Find timestamp column
            ts_key = None
            for key in ['timestamp', 'ts', 'time', 'date', 'datetime']:
                if key in row:
                    ts_key = key
                    break

            if ts_key is None:
                raise ValueError("No timestamp column found in CSV")

            ts = parse_timestamp(row[ts_key])

            candle = Candle(
                ts=ts,
                open=float(row.get('open', row.get('Open', 0))),
                high=float(row.get('high', row.get('High', 0))),
                low=float(row.get('low', row.get('Low', 0))),
                close=float(row.get('close', row.get('Close', 0))),
                volume=float(row.get('volume', row.get('Volume', 0))),
            )
            candles.append(candle)

    return candles


def resample_to_15m(candles_3m: List[Candle]) -> List[Candle]:
    """
    Resample 3m candles to 15m candles.

    Groups 5 consecutive 3m candles into one 15m candle.
    """
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


def generate_sample_data(num_candles: int = 500) -> Tuple[List[Candle], List[Candle]]:
    """
    Generate sample OHLCV data for testing.

    Creates a synthetic price series with trends and ranges.
    """
    import random
    random.seed(42)  # Deterministic

    candles_3m = []
    base_price = 0.50  # XRP around $0.50
    base_volume = 1000000

    ts = int(datetime(2024, 1, 1).timestamp() * 1000)
    price = base_price

    for i in range(num_candles):
        # Add some trend and noise
        trend = 0.0001 * (i % 100 - 50)  # Slight trend cycles
        noise = random.uniform(-0.005, 0.005)
        change = trend + noise

        open_price = price
        close_price = price * (1 + change)

        # Generate high/low
        volatility = abs(change) + random.uniform(0.001, 0.003)
        high = max(open_price, close_price) * (1 + volatility)
        low = min(open_price, close_price) * (1 - volatility)

        # Volume with occasional spikes
        vol_mult = 1.0
        if random.random() < 0.1:  # 10% chance of volume spike
            vol_mult = random.uniform(2.0, 4.0)

        volume = base_volume * vol_mult * random.uniform(0.5, 1.5)

        candle = Candle(
            ts=ts,
            open=round(open_price, 6),
            high=round(high, 6),
            low=round(low, 6),
            close=round(close_price, 6),
            volume=round(volume, 2),
        )
        candles_3m.append(candle)

        price = close_price
        ts += 3 * 60 * 1000  # 3 minutes in ms

    # Resample to 15m
    candles_15m = resample_to_15m(candles_3m)

    return candles_15m, candles_3m


def main():
    parser = argparse.ArgumentParser(
        description='XRP Core Trading System - Backtest Demo'
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        default=None,
        help='Path to config YAML file'
    )
    parser.add_argument(
        '--csv', '-f',
        type=str,
        default=None,
        help='Path to OHLCV CSV file (3m candles)'
    )
    parser.add_argument(
        '--equity', '-e',
        type=float,
        default=10000.0,
        help='Starting equity'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Print all signals'
    )
    parser.add_argument(
        '--sample', '-s',
        action='store_true',
        help='Use sample generated data'
    )

    args = parser.parse_args()

    # Load or generate data
    if args.csv:
        print(f"Loading candles from: {args.csv}")
        candles_3m = load_candles_from_csv(args.csv)
        candles_15m = resample_to_15m(candles_3m)
        print(f"Loaded {len(candles_3m)} 3m candles, {len(candles_15m)} 15m candles")
    else:
        print("Using sample generated data...")
        candles_15m, candles_3m = generate_sample_data(500)
        print(f"Generated {len(candles_3m)} 3m candles, {len(candles_15m)} 15m candles")

    # Load config
    if args.config:
        print(f"Loading config from: {args.config}")
        config = CoreConfig.from_yaml(args.config)
    else:
        print("Using default configuration")
        config = CoreConfig()

    # Run backtest
    print("\n" + "=" * 60)
    print("Running backtest...")
    print("=" * 60 + "\n")

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

    # Print signal summary
    signals = [d for d in decisions if d.signal != Signal.NONE]
    print(f"\nTotal signals: {len(signals)}")

    by_type = {}
    for d in signals:
        sig = d.signal.value
        by_type[sig] = by_type.get(sig, 0) + 1

    for sig, count in sorted(by_type.items()):
        print(f"  {sig}: {count}")

    # Print last 10 significant decisions
    print("\n" + "-" * 40)
    print("Last 10 Signals:")
    print("-" * 40)

    recent = [d for d in decisions if d.signal != Signal.NONE][-10:]
    for d in recent:
        print(f"  [{d.signal.value:12}] {d.explanation[:60]}...")

    print("\n" + "=" * 60)
    print("Backtest complete.")


if __name__ == '__main__':
    main()
