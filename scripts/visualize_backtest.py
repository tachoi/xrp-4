#!/usr/bin/env python3
"""
TradingView-style interactive backtest visualization.

Usage:
    python scripts/visualize_backtest.py --trades outputs/multi_hmm_backtest/trades_test.csv \
                                         --signals outputs/multi_hmm_backtest/signals_test.csv

    # Or use default (latest backtest)
    python scripts/visualize_backtest.py

Features:
    - Candlestick chart with zoom/pan
    - Entry/Exit markers with trade info
    - Regime background colors (TREND_UP=green, TREND_DOWN=red, RANGE=blue)
    - Cumulative PnL curve
    - Trade statistics panel
"""

import argparse
import os
import sys
import time
import requests
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    print("Installing plotly...")
    os.system("pip install plotly kaleido")
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots


class BinanceClient:
    """Binance REST API client for fetching historical klines."""

    BASE_URL = "https://api.binance.com"
    MAX_LIMIT_PER_REQUEST = 1000

    def __init__(self):
        self.session = requests.Session()

    def get_klines(
        self,
        symbol: str = "XRPUSDT",
        interval: str = "3m",
        limit: int = 500,
        start_time: int = None,
        end_time: int = None,
        show_progress: bool = False,
    ) -> pd.DataFrame:
        """Fetch klines with pagination support."""
        all_data = []
        remaining = limit
        current_end_time = end_time
        total_fetched = 0

        while remaining > 0:
            fetch_limit = min(remaining, self.MAX_LIMIT_PER_REQUEST)

            url = f"{self.BASE_URL}/api/v3/klines"
            params = {
                "symbol": symbol,
                "interval": interval,
                "limit": fetch_limit,
            }
            if start_time:
                params["startTime"] = start_time
            if current_end_time:
                params["endTime"] = current_end_time

            response = self.session.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            if not data:
                break

            all_data = data + all_data
            remaining -= len(data)
            total_fetched += len(data)

            if show_progress and limit > 1000:
                pct = (total_fetched / limit) * 100
                print(f"\r    Progress: {total_fetched:,}/{limit:,} ({pct:.0f}%)", end="", flush=True)

            if remaining > 0 and len(data) == fetch_limit:
                current_end_time = data[0][0] - 1
                time.sleep(0.05)
            else:
                break

        if show_progress and limit > 1000:
            print()

        if not all_data:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

        df = pd.DataFrame(all_data, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_base",
            "taker_buy_quote", "ignore"
        ])

        df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms")
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)

        df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        return df[["timestamp", "open", "high", "low", "close", "volume"]]


def load_ohlcv_data(start_date: str, end_date: str, timeframe: str = "3m", max_bars: int = 20000) -> pd.DataFrame:
    """Load OHLCV data from Binance.

    For long periods, use 15m timeframe instead of 3m to reduce data volume.
    """
    client = BinanceClient()

    # Calculate number of bars needed
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)

    # Convert to ms timestamp
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)

    # Estimate bars
    interval_ms = {"1m": 60000, "3m": 180000, "5m": 300000, "15m": 900000, "1h": 3600000}
    ms_per_bar = interval_ms.get(timeframe, 180000)
    estimated_bars = (end_ms - start_ms) // ms_per_bar + 100

    # If too many bars, switch to larger timeframe
    if estimated_bars > max_bars and timeframe == "3m":
        print(f"  Too many bars ({estimated_bars:,}), using 15m timeframe instead...")
        timeframe = "15m"
        ms_per_bar = interval_ms["15m"]
        estimated_bars = (end_ms - start_ms) // ms_per_bar + 100

    df = client.get_klines(
        symbol="XRPUSDT",
        interval=timeframe,
        limit=min(estimated_bars, max_bars),
        start_time=start_ms,
        end_time=end_ms,
        show_progress=True,
    )
    return df


def load_trades(trades_path: str) -> pd.DataFrame:
    """Load trades from CSV."""
    df = pd.read_csv(trades_path)
    return df


def load_signals(signals_path: str) -> pd.DataFrame:
    """Load signals from CSV."""
    df = pd.read_csv(signals_path)
    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)  # Remove timezone
    return df


def create_ohlcv_from_signals(signals_df: pd.DataFrame) -> pd.DataFrame:
    """Create OHLCV-like data from signals price data.

    Since signals only have close price, we'll create synthetic OHLC
    by using close as all prices (simplified view).
    """
    df = signals_df[['timestamp', 'price']].copy()
    df = df.rename(columns={'price': 'close'})
    df['open'] = df['close']
    df['high'] = df['close']
    df['low'] = df['close']
    df['volume'] = 1.0  # Placeholder

    # Calculate proper OHLC by grouping into larger windows
    # For visualization, we'll just use the close price as line chart
    return df


def create_regime_shapes(signals_df: pd.DataFrame) -> list:
    """Create background shapes for regime periods."""
    shapes = []

    # Color mapping
    regime_colors = {
        'TREND_UP': 'rgba(0, 255, 0, 0.1)',
        'TREND_DOWN': 'rgba(255, 0, 0, 0.1)',
        'RANGE': 'rgba(0, 0, 255, 0.08)',
        'TRANSITION': 'rgba(255, 255, 0, 0.1)',
        'HIGH_VOL': 'rgba(255, 165, 0, 0.15)',
        'NO_TRADE': 'rgba(128, 128, 128, 0.1)',
    }

    # Group consecutive same regimes
    signals_df = signals_df.copy()
    signals_df['regime_group'] = (signals_df['regime_confirmed'] != signals_df['regime_confirmed'].shift()).cumsum()

    for _, group in signals_df.groupby('regime_group'):
        regime = group['regime_confirmed'].iloc[0]
        start = group['timestamp'].iloc[0]
        end = group['timestamp'].iloc[-1]
        color = regime_colors.get(regime, 'rgba(200, 200, 200, 0.1)')

        shapes.append(dict(
            type="rect",
            xref="x",
            yref="paper",
            x0=start,
            x1=end,
            y0=0,
            y1=1,
            fillcolor=color,
            line=dict(width=0),
            layer="below"
        ))

    return shapes


def create_trade_markers(trades_df: pd.DataFrame, signals_df: pd.DataFrame) -> tuple:
    """Create entry and exit markers for trades."""
    entries = []
    exits = []

    for _, trade in trades_df.iterrows():
        exit_idx = trade['exit_idx']

        # Find entry and exit in signals
        if exit_idx < len(signals_df):
            exit_row = signals_df[signals_df['idx'] == exit_idx]
            if len(exit_row) > 0:
                exit_time = exit_row['timestamp'].iloc[0]
                exit_price = trade['exit_price']

                # Entry is bars_held before exit
                bars_held = int(trade['bars_held'])
                entry_idx = exit_idx - bars_held
                entry_row = signals_df[signals_df['idx'] == entry_idx]

                if len(entry_row) > 0:
                    entry_time = entry_row['timestamp'].iloc[0]
                else:
                    entry_time = exit_time - timedelta(minutes=3 * bars_held)

                entry_price = trade['entry_price']
                side = trade['side']
                pnl = trade['pnl']
                pnl_pct = trade['pnl_pct']

                # Entry marker
                entries.append({
                    'time': entry_time,
                    'price': entry_price,
                    'side': side,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'regime': trade.get('regime', 'N/A'),
                    'signal': trade.get('signal', 'N/A'),
                })

                # Exit marker
                exits.append({
                    'time': exit_time,
                    'price': exit_price,
                    'side': side,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'exit_reason': trade.get('exit_reason', 'N/A'),
                })

    return entries, exits


def create_visualization(
    ohlcv_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    signals_df: pd.DataFrame,
    title: str = "Backtest Results",
    use_candlestick: bool = True
) -> go.Figure:
    """Create the main visualization figure."""

    # Create subplots: price + cumulative PnL
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3],
        subplot_titles=('Price & Trades', 'Cumulative PnL')
    )

    # 1. Price chart
    if use_candlestick and 'open' in ohlcv_df.columns and ohlcv_df['open'].nunique() > 1:
        fig.add_trace(
            go.Candlestick(
                x=ohlcv_df['timestamp'],
                open=ohlcv_df['open'],
                high=ohlcv_df['high'],
                low=ohlcv_df['low'],
                close=ohlcv_df['close'],
                name='XRPUSDT',
                increasing_line_color='#26a69a',
                decreasing_line_color='#ef5350',
            ),
            row=1, col=1
        )
    else:
        # Line chart when using signals data
        fig.add_trace(
            go.Scatter(
                x=ohlcv_df['timestamp'],
                y=ohlcv_df['close'],
                mode='lines',
                name='XRPUSDT',
                line=dict(color='#1f77b4', width=1),
            ),
            row=1, col=1
        )

    # 2. Add regime background colors
    shapes = create_regime_shapes(signals_df)
    for shape in shapes:
        shape['yref'] = 'y'
        shape['y0'] = ohlcv_df['low'].min() * 0.99
        shape['y1'] = ohlcv_df['high'].max() * 1.01
        fig.add_shape(shape, row=1, col=1)

    # 3. Entry/Exit markers
    entries, exits = create_trade_markers(trades_df, signals_df)

    # Entry markers
    if entries:
        entry_df = pd.DataFrame(entries)

        # LONG entries (green triangle up)
        long_entries = entry_df[entry_df['side'] == 'LONG']
        if len(long_entries) > 0:
            fig.add_trace(
                go.Scatter(
                    x=long_entries['time'],
                    y=long_entries['price'],
                    mode='markers',
                    marker=dict(
                        symbol='triangle-up',
                        size=12,
                        color='green',
                        line=dict(width=1, color='darkgreen')
                    ),
                    name='LONG Entry',
                    hovertemplate=(
                        '<b>LONG Entry</b><br>'
                        'Price: $%{y:.4f}<br>'
                        'Regime: %{customdata[0]}<br>'
                        'Signal: %{customdata[1]}<br>'
                        '<extra></extra>'
                    ),
                    customdata=long_entries[['regime', 'signal']].values,
                ),
                row=1, col=1
            )

        # SHORT entries (red triangle down)
        short_entries = entry_df[entry_df['side'] == 'SHORT']
        if len(short_entries) > 0:
            fig.add_trace(
                go.Scatter(
                    x=short_entries['time'],
                    y=short_entries['price'],
                    mode='markers',
                    marker=dict(
                        symbol='triangle-down',
                        size=12,
                        color='red',
                        line=dict(width=1, color='darkred')
                    ),
                    name='SHORT Entry',
                    hovertemplate=(
                        '<b>SHORT Entry</b><br>'
                        'Price: $%{y:.4f}<br>'
                        'Regime: %{customdata[0]}<br>'
                        'Signal: %{customdata[1]}<br>'
                        '<extra></extra>'
                    ),
                    customdata=short_entries[['regime', 'signal']].values,
                ),
                row=1, col=1
            )

    # Exit markers
    if exits:
        exit_df = pd.DataFrame(exits)

        # Winning exits (green circle)
        win_exits = exit_df[exit_df['pnl'] > 0]
        if len(win_exits) > 0:
            fig.add_trace(
                go.Scatter(
                    x=win_exits['time'],
                    y=win_exits['price'],
                    mode='markers',
                    marker=dict(
                        symbol='circle',
                        size=10,
                        color='lime',
                        line=dict(width=1, color='green')
                    ),
                    name='Win Exit',
                    hovertemplate=(
                        '<b>WIN Exit</b><br>'
                        'Price: $%{y:.4f}<br>'
                        'PnL: $%{customdata[0]:.2f} (%{customdata[1]:.2f}%)<br>'
                        'Reason: %{customdata[2]}<br>'
                        '<extra></extra>'
                    ),
                    customdata=win_exits[['pnl', 'pnl_pct', 'exit_reason']].values,
                ),
                row=1, col=1
            )

        # Losing exits (red circle)
        loss_exits = exit_df[exit_df['pnl'] <= 0]
        if len(loss_exits) > 0:
            fig.add_trace(
                go.Scatter(
                    x=loss_exits['time'],
                    y=loss_exits['price'],
                    mode='markers',
                    marker=dict(
                        symbol='circle',
                        size=10,
                        color='salmon',
                        line=dict(width=1, color='red')
                    ),
                    name='Loss Exit',
                    hovertemplate=(
                        '<b>LOSS Exit</b><br>'
                        'Price: $%{y:.4f}<br>'
                        'PnL: $%{customdata[0]:.2f} (%{customdata[1]:.2f}%)<br>'
                        'Reason: %{customdata[2]}<br>'
                        '<extra></extra>'
                    ),
                    customdata=loss_exits[['pnl', 'pnl_pct', 'exit_reason']].values,
                ),
                row=1, col=1
            )

    # 4. Cumulative PnL
    trades_df_sorted = trades_df.copy()
    trades_df_sorted['cum_pnl'] = trades_df_sorted['pnl'].cumsum()

    # Map exit_idx to timestamp
    pnl_times = []
    pnl_values = []
    for _, trade in trades_df_sorted.iterrows():
        exit_idx = trade['exit_idx']
        exit_row = signals_df[signals_df['idx'] == exit_idx]
        if len(exit_row) > 0:
            pnl_times.append(exit_row['timestamp'].iloc[0])
            pnl_values.append(trade['cum_pnl'])

    if pnl_times:
        fig.add_trace(
            go.Scatter(
                x=pnl_times,
                y=pnl_values,
                mode='lines+markers',
                line=dict(color='blue', width=2),
                marker=dict(size=4),
                name='Cumulative PnL',
                hovertemplate='PnL: $%{y:.2f}<extra></extra>',
            ),
            row=2, col=1
        )

        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)

    # 6. Calculate statistics
    total_trades = len(trades_df)
    wins = len(trades_df[trades_df['pnl'] > 0])
    losses = len(trades_df[trades_df['pnl'] <= 0])
    win_rate = wins / total_trades * 100 if total_trades > 0 else 0
    total_pnl = trades_df['pnl'].sum()
    avg_pnl = trades_df['pnl'].mean() if total_trades > 0 else 0

    gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
    gross_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Stats annotation
    stats_text = (
        f"<b>Statistics</b><br>"
        f"Total Trades: {total_trades}<br>"
        f"Win Rate: {win_rate:.1f}%<br>"
        f"Profit Factor: {profit_factor:.2f}<br>"
        f"Total PnL: ${total_pnl:.2f}<br>"
        f"Avg PnL: ${avg_pnl:.2f}"
    )

    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.01, y=0.99,
        text=stats_text,
        showarrow=False,
        font=dict(size=11, family="monospace"),
        align="left",
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="gray",
        borderwidth=1,
    )

    # Regime legend
    regime_legend = (
        "<b>Regimes</b><br>"
        "<span style='color:green'>■</span> TREND_UP<br>"
        "<span style='color:red'>■</span> TREND_DOWN<br>"
        "<span style='color:blue'>■</span> RANGE<br>"
        "<span style='color:orange'>■</span> HIGH_VOL"
    )

    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.99, y=0.99,
        text=regime_legend,
        showarrow=False,
        font=dict(size=10),
        align="right",
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="gray",
        borderwidth=1,
    )

    # Layout
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            font=dict(size=18)
        ),
        xaxis_rangeslider_visible=False,
        height=900,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.75,
            xanchor="left",
            x=0.01
        ),
        hovermode='x unified',
    )

    # Update axes
    fig.update_xaxes(title_text="Time", row=2, col=1)
    fig.update_yaxes(title_text="Price (USDT)", row=1, col=1)
    fig.update_yaxes(title_text="PnL ($)", row=2, col=1)

    return fig


def find_latest_backtest() -> tuple:
    """Find the latest backtest results."""
    base_dirs = [
        Path("/mnt/c/Users/tae am choi/xrp-4/outputs/multi_hmm_backtest"),
        Path("/mnt/c/Users/tae am choi/xrp-4/outputs/fsm_with_xgb"),
        Path("/mnt/c/Users/tae am choi/xrp-4/outputs/fsm_improved"),
    ]

    for base_dir in base_dirs:
        trades_path = base_dir / "trades_test.csv"
        signals_path = base_dir / "signals_test.csv"
        if trades_path.exists() and signals_path.exists():
            return str(trades_path), str(signals_path)

    return None, None


def main():
    parser = argparse.ArgumentParser(description="Visualize backtest results")
    parser.add_argument("--trades", type=str, help="Path to trades CSV")
    parser.add_argument("--signals", type=str, help="Path to signals CSV")
    parser.add_argument("--output", type=str, default="backtest_chart.html",
                        help="Output HTML file")
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--show", action="store_true", help="Open in browser")
    parser.add_argument("--fetch-ohlcv", action="store_true",
                        help="Fetch OHLCV from Binance (slower, but shows candlesticks)")
    parser.add_argument("--sample", type=int, default=5000,
                        help="Sample size for signals (reduce for faster loading)")

    args = parser.parse_args()

    # Find trades and signals files
    trades_path = args.trades
    signals_path = args.signals

    if not trades_path or not signals_path:
        trades_path, signals_path = find_latest_backtest()
        if not trades_path:
            print("Error: Could not find backtest results. Please specify --trades and --signals")
            return
        print(f"Using latest backtest: {trades_path}")

    # Load data
    print("Loading trades and signals...")
    trades_df = load_trades(trades_path)
    signals_df = load_signals(signals_path)

    print(f"  Trades: {len(trades_df)}")
    print(f"  Signals: {len(signals_df)}")

    # Apply date filter if specified
    if args.start:
        start_dt = pd.to_datetime(args.start)
        signals_df = signals_df[signals_df['timestamp'] >= start_dt]
        trades_df = trades_df[trades_df['exit_idx'] >= signals_df['idx'].min()]
    if args.end:
        end_dt = pd.to_datetime(args.end) + timedelta(days=1)
        signals_df = signals_df[signals_df['timestamp'] <= end_dt]
        trades_df = trades_df[trades_df['exit_idx'] <= signals_df['idx'].max()]

    # Sample signals for faster rendering
    if len(signals_df) > args.sample:
        step = len(signals_df) // args.sample
        signals_df_plot = signals_df.iloc[::step].copy()
        print(f"  Sampled signals for plot: {len(signals_df_plot)}")
    else:
        signals_df_plot = signals_df

    # Determine date range
    start_date = signals_df['timestamp'].min().strftime('%Y-%m-%d')
    end_date = signals_df['timestamp'].max().strftime('%Y-%m-%d')

    # Get price data
    use_candlestick = False
    if args.fetch_ohlcv:
        print(f"Fetching OHLCV data: {start_date} to {end_date}...")
        ohlcv_df = load_ohlcv_data(start_date, end_date, timeframe="15m")
        print(f"  OHLCV bars: {len(ohlcv_df)}")
        use_candlestick = True
    else:
        print("Using signals price data (use --fetch-ohlcv for candlesticks)")
        ohlcv_df = create_ohlcv_from_signals(signals_df_plot)

    # Create visualization
    print("Creating visualization...")
    title = f"XRP-4 Backtest Results ({start_date} to {end_date})"
    fig = create_visualization(ohlcv_df, trades_df, signals_df, title, use_candlestick)

    # Save to HTML
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = Path("/mnt/c/Users/tae am choi/xrp-4/outputs") / output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path))
    print(f"\nChart saved to: {output_path}")

    if args.show:
        fig.show()
    else:
        print(f"\nOpen in browser: file://{output_path}")
        print("Or run with --show flag to open automatically")


if __name__ == "__main__":
    main()
