#!/usr/bin/env python3
"""Check EMA slope distribution."""
import sys
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

def load_and_check():
    import psycopg2

    conn = psycopg2.connect(
        host="localhost", port=5432, database="xrp_timeseries",
        user="xrp_user", password="xrp_password_change_me",
    )

    query = """
        SELECT time as timestamp, open, high, low, close, volume
        FROM ohlcv
        WHERE symbol = %s AND timeframe = %s
          AND time >= %s AND time < %s
        ORDER BY time ASC
    """

    start = datetime(2023, 1, 1)
    end = datetime(2024, 12, 31)

    df_15m = pd.read_sql(query, conn, params=("XRPUSDT", "15m", start, end))
    conn.close()

    df_15m["timestamp"] = pd.to_datetime(df_15m["timestamp"])
    df_15m.set_index("timestamp", inplace=True)

    # Calculate EMA slope
    df_15m["ema_20"] = df_15m["close"].ewm(span=20, adjust=False).mean()
    df_15m["ema_slope"] = df_15m["ema_20"].pct_change(5)

    slope = df_15m["ema_slope"].dropna()

    print("EMA Slope (15m) Distribution:")
    print(f"  Count: {len(slope)}")
    print(f"  Min: {slope.min():.6f}")
    print(f"  Max: {slope.max():.6f}")
    print(f"  Mean: {slope.mean():.6f}")
    print(f"  Std: {slope.std():.6f}")
    print(f"  Median: {slope.median():.6f}")
    print()
    print("Percentiles (abs value):")
    abs_slope = slope.abs()
    for p in [50, 75, 90, 95, 99]:
        print(f"  {p}th: {abs_slope.quantile(p/100):.6f}")
    print()
    print("Threshold analysis (abs(slope) > threshold):")
    for thresh in [0.001, 0.002, 0.003, 0.005, 0.007, 0.01, 0.02, 0.03, 0.05, 0.07]:
        pct = (abs_slope > thresh).mean() * 100
        print(f"  > {thresh:.3f}: {pct:.1f}%")


if __name__ == "__main__":
    load_and_check()
