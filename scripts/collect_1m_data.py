#!/usr/bin/env python3
"""
Collect 1-minute OHLCV data from Binance and store in TimescaleDB.
"""

import argparse
import time
from datetime import datetime, timedelta
import psycopg2
from psycopg2.extras import execute_values
import ccxt

# Database configuration
DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "xrp_timeseries",
    "user": "xrp_user",
    "password": "xrp_password_change_me",
}


def create_table_if_not_exists(conn):
    """Ensure the ohlcv table exists."""
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS ohlcv (
                time TIMESTAMPTZ NOT NULL,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                open DOUBLE PRECISION,
                high DOUBLE PRECISION,
                low DOUBLE PRECISION,
                close DOUBLE PRECISION,
                volume DOUBLE PRECISION,
                PRIMARY KEY (time, symbol, timeframe)
            );
        """)
        conn.commit()
        print("Table 'ohlcv' ensured.")


def fetch_ohlcv_binance(symbol: str, timeframe: str, since: int, limit: int = 1000):
    """
    Fetch OHLCV data from Binance.

    Args:
        symbol: Trading pair (e.g., 'XRP/USDT')
        timeframe: Timeframe (e.g., '1m')
        since: Start timestamp in milliseconds
        limit: Number of candles to fetch (max 1000)

    Returns:
        List of OHLCV data
    """
    exchange = ccxt.binance({
        'enableRateLimit': True,
        'options': {'defaultType': 'future'}  # Use futures for XRPUSDT
    })

    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
    return ohlcv


def insert_ohlcv(conn, data, symbol: str, timeframe: str):
    """
    Insert OHLCV data into TimescaleDB.

    Args:
        conn: Database connection
        data: List of [timestamp, open, high, low, close, volume]
        symbol: Symbol (e.g., 'XRPUSDT')
        timeframe: Timeframe (e.g., '1m')
    """
    if not data:
        return 0

    rows = []
    for candle in data:
        ts = datetime.utcfromtimestamp(candle[0] / 1000)
        rows.append((
            ts,
            symbol,
            timeframe,
            candle[1],  # open
            candle[2],  # high
            candle[3],  # low
            candle[4],  # close
            candle[5],  # volume
        ))

    with conn.cursor() as cur:
        execute_values(
            cur,
            """
            INSERT INTO ohlcv (time, symbol, timeframe, open, high, low, close, volume)
            VALUES %s
            ON CONFLICT (time, symbol, timeframe) DO UPDATE SET
                open = EXCLUDED.open,
                high = EXCLUDED.high,
                low = EXCLUDED.low,
                close = EXCLUDED.close,
                volume = EXCLUDED.volume
            """,
            rows
        )
        conn.commit()

    return len(rows)


def collect_historical_data(
    symbol: str = "XRPUSDT",
    timeframe: str = "1m",
    start_date: str = "2025-12-01",
    end_date: str = None,
):
    """
    Collect historical 1-minute data from Binance.

    Args:
        symbol: Trading pair symbol
        timeframe: Timeframe to collect
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD), defaults to now
    """
    # Parse dates
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    if end_date:
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    else:
        end_dt = datetime.utcnow()

    print(f"Collecting {timeframe} data for {symbol}")
    print(f"Period: {start_dt} to {end_dt}")

    # Connect to database
    conn = psycopg2.connect(**DB_CONFIG)
    create_table_if_not_exists(conn)

    # Convert symbol format for ccxt
    ccxt_symbol = symbol.replace("USDT", "/USDT")

    # Calculate timestamps
    current_ts = int(start_dt.timestamp() * 1000)
    end_ts = int(end_dt.timestamp() * 1000)

    # Timeframe to milliseconds
    tf_ms = {
        "1m": 60 * 1000,
        "3m": 3 * 60 * 1000,
        "5m": 5 * 60 * 1000,
        "15m": 15 * 60 * 1000,
        "1h": 60 * 60 * 1000,
    }
    candle_ms = tf_ms.get(timeframe, 60 * 1000)

    total_inserted = 0
    batch_count = 0

    try:
        while current_ts < end_ts:
            # Fetch data
            try:
                ohlcv = fetch_ohlcv_binance(ccxt_symbol, timeframe, current_ts, limit=1000)
            except Exception as e:
                print(f"Error fetching data: {e}")
                time.sleep(5)
                continue

            if not ohlcv:
                print("No more data available")
                break

            # Insert into database
            inserted = insert_ohlcv(conn, ohlcv, symbol, timeframe)
            total_inserted += inserted
            batch_count += 1

            # Update current timestamp
            last_ts = ohlcv[-1][0]
            current_ts = last_ts + candle_ms

            # Progress
            current_dt = datetime.utcfromtimestamp(last_ts / 1000)
            progress = (last_ts - int(start_dt.timestamp() * 1000)) / (end_ts - int(start_dt.timestamp() * 1000)) * 100
            print(f"Batch {batch_count}: {inserted} candles, up to {current_dt}, progress: {progress:.1f}%")

            # Rate limiting
            time.sleep(0.5)

    finally:
        conn.close()

    print(f"\nTotal inserted: {total_inserted} candles")
    return total_inserted


def main():
    parser = argparse.ArgumentParser(description="Collect OHLCV data from Binance")
    parser.add_argument("--symbol", default="XRPUSDT", help="Trading pair symbol")
    parser.add_argument("--timeframe", default="1m", help="Timeframe (1m, 3m, 5m, 15m, 1h)")
    parser.add_argument("--start", default="2025-12-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="End date (YYYY-MM-DD), default=now")

    args = parser.parse_args()

    collect_historical_data(
        symbol=args.symbol,
        timeframe=args.timeframe,
        start_date=args.start,
        end_date=args.end,
    )


if __name__ == "__main__":
    main()
