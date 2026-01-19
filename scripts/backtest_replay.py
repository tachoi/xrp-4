#!/usr/bin/env python
"""
Interactive Backtest Replay with 1-minute Candle Data.

Replays backtest like a live trading simulation with:
- 1-minute candlestick charts
- Real-time trade execution markers
- Regime detection visualization
- Play/Pause/Step controls

Usage:
    python scripts/backtest_replay.py --start 2025-12-01 --end 2025-12-15
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import psycopg2

from dash import Dash, html, dcc, callback, Input, Output, State, ctx
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# =============================================================================
# Data Loading
# =============================================================================

def load_ohlcv_from_db(symbol: str, start: datetime, end: datetime, timeframe: str = "1m") -> pd.DataFrame:
    """Load OHLCV data from TimescaleDB."""
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
    df = pd.read_sql(query, conn, params=(symbol, timeframe, start, end))
    conn.close()

    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
    return df


def load_backtest_results(trades_path: str, signals_path: str):
    """Load backtest trades and signals."""
    trades_df = pd.read_csv(trades_path)
    signals_df = pd.read_csv(signals_path)
    signals_df['timestamp'] = pd.to_datetime(signals_df['timestamp']).dt.tz_localize(None)
    return trades_df, signals_df


def find_latest_backtest():
    """Find the latest backtest results."""
    base_dirs = [
        Path("/mnt/c/Users/tae am choi/xrp-4/outputs/multi_hmm_backtest"),
        Path("/mnt/c/Users/tae am choi/xrp-4/outputs/fsm_with_xgb"),
    ]
    for base_dir in base_dirs:
        trades_path = base_dir / "trades_test.csv"
        signals_path = base_dir / "signals_test.csv"
        if trades_path.exists() and signals_path.exists():
            return str(trades_path), str(signals_path)
    return None, None


# =============================================================================
# Global State
# =============================================================================

class ReplayState:
    def __init__(self):
        self.df_1m = None  # 1-minute candle data
        self.df_3m = None  # 3-minute candle data (for signals)
        self.trades_df = None
        self.signals_df = None
        self.current_bar = 0
        self.max_bar = 0

state = ReplayState()


# =============================================================================
# Chart Creation
# =============================================================================

def create_replay_chart(current_bar: int, window_bars: int = 60) -> go.Figure:
    """Create candlestick chart for current replay position."""
    if state.df_1m is None or len(state.df_1m) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No data loaded", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig

    # Clamp current_bar
    current_bar = max(0, min(current_bar, len(state.df_1m) - 1))

    # Window: show bars up to current position
    start_bar = max(0, current_bar - window_bars)
    end_bar = current_bar + 1

    df = state.df_1m.iloc[start_bar:end_bar].copy()
    current_time = state.df_1m.iloc[current_bar]['timestamp']

    # Create figure
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=('Price (1m Candles)', 'Volume', 'Cumulative PnL')
    )

    # 1. Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df['timestamp'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='XRPUSDT',
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350',
        ),
        row=1, col=1
    )

    # Current bar marker (vertical line)
    fig.add_vline(
        x=current_time,
        line_width=2,
        line_dash="dash",
        line_color="yellow",
        row=1, col=1
    )

    # 2. Add regime background colors from signals
    if state.signals_df is not None and len(state.signals_df) > 0:
        regime_colors = {
            'TREND_UP': 'rgba(0, 200, 0, 0.15)',
            'TREND_DOWN': 'rgba(200, 0, 0, 0.15)',
            'RANGE': 'rgba(0, 0, 200, 0.1)',
            'TRANSITION': 'rgba(200, 200, 0, 0.1)',
            'HIGH_VOL': 'rgba(255, 165, 0, 0.2)',
        }

        # Find signals within the visible time range
        time_start = df['timestamp'].iloc[0]
        time_end = df['timestamp'].iloc[-1]
        visible_signals = state.signals_df[
            (state.signals_df['timestamp'] >= time_start) &
            (state.signals_df['timestamp'] <= time_end)
        ].copy()

        if len(visible_signals) > 0:
            visible_signals['regime_group'] = (
                visible_signals['regime_confirmed'] != visible_signals['regime_confirmed'].shift()
            ).cumsum()

            for _, group in visible_signals.groupby('regime_group'):
                regime = group['regime_confirmed'].iloc[0]
                color = regime_colors.get(regime, 'rgba(200, 200, 200, 0.05)')
                fig.add_vrect(
                    x0=group['timestamp'].iloc[0],
                    x1=group['timestamp'].iloc[-1],
                    fillcolor=color,
                    layer="below",
                    line_width=0,
                    row=1, col=1
                )

    # 3. Add trade markers (only completed trades)
    if state.trades_df is not None and len(state.trades_df) > 0 and state.signals_df is not None:
        # Find trades that have completed by current time
        for _, trade in state.trades_df.iterrows():
            exit_idx = trade['exit_idx']

            # Get exit time from signals
            exit_rows = state.signals_df[state.signals_df['idx'] == exit_idx]
            if len(exit_rows) == 0:
                continue

            exit_time = exit_rows.iloc[0]['timestamp']

            # Only show if trade exit is before or at current time
            if exit_time > current_time:
                continue

            # Get entry time
            bars_held = int(trade.get('bars_held', 1))
            entry_idx = exit_idx - bars_held
            entry_rows = state.signals_df[state.signals_df['idx'] == entry_idx]

            if len(entry_rows) > 0:
                entry_time = entry_rows.iloc[0]['timestamp']

                # Skip if entry is before visible window
                if entry_time < df['timestamp'].iloc[0]:
                    entry_time = None
            else:
                entry_time = None

            side = trade['side']
            pnl = trade['pnl']
            entry_price = trade['entry_price']
            exit_price = trade['exit_price']

            # 1분봉 활용 결과 추출
            exit_reason = trade.get('exit_reason', 'N/A')
            max_profit_pct = trade.get('max_profit_pct', 0)
            profit_given_back = trade.get('profit_given_back', 0)
            pnl_pct = trade.get('pnl_pct', 0)
            regime = trade.get('regime', 'N/A')
            signal_type = trade.get('signal', 'N/A')

            # Entry marker
            if entry_time is not None:
                fig.add_trace(
                    go.Scatter(
                        x=[entry_time],
                        y=[entry_price],
                        mode='markers',
                        marker=dict(
                            symbol='triangle-up' if side == 'LONG' else 'triangle-down',
                            size=14,
                            color='green' if side == 'LONG' else 'red',
                            line=dict(width=1, color='white')
                        ),
                        showlegend=False,
                        hovertemplate=(
                            f'<b>{side} Entry</b><br>'
                            f'Price: ${entry_price:.4f}<br>'
                            f'Signal: {signal_type}<br>'
                            f'Regime: {regime}'
                            '<extra></extra>'
                        ),
                    ),
                    row=1, col=1
                )

            # Exit marker - 1분봉 활용 결과 포함
            if exit_time >= df['timestamp'].iloc[0]:
                # 종료 이유에 따른 마커 스타일
                if 'TRAILING' in str(exit_reason):
                    marker_symbol = 'star'
                    marker_size = 14
                elif 'BREAKEVEN' in str(exit_reason):
                    marker_symbol = 'diamond'
                    marker_size = 12
                else:
                    marker_symbol = 'circle'
                    marker_size = 12

                fig.add_trace(
                    go.Scatter(
                        x=[exit_time],
                        y=[exit_price],
                        mode='markers',
                        marker=dict(
                            symbol=marker_symbol,
                            size=marker_size,
                            color='lime' if pnl > 0 else 'salmon',
                            line=dict(width=1, color='white')
                        ),
                        showlegend=False,
                        hovertemplate=(
                            f'<b>Exit: {exit_reason}</b><br>'
                            f'Price: ${exit_price:.4f}<br>'
                            f'PnL: ${pnl:.2f} ({pnl_pct:.2f}%)<br>'
                            f'───────────<br>'
                            f'Max Profit: {max_profit_pct:.2f}%<br>'
                            f'Given Back: {profit_given_back:.2f}%'
                            '<extra></extra>'
                        ),
                    ),
                    row=1, col=1
                )

    # 4. Volume bars
    colors = ['#26a69a' if c >= o else '#ef5350' for c, o in zip(df['close'], df['open'])]
    fig.add_trace(
        go.Bar(x=df['timestamp'], y=df['volume'], marker_color=colors, showlegend=False),
        row=2, col=1
    )

    # 5. Cumulative PnL
    if state.trades_df is not None and len(state.trades_df) > 0 and state.signals_df is not None:
        completed_trades = []
        for _, trade in state.trades_df.iterrows():
            exit_rows = state.signals_df[state.signals_df['idx'] == trade['exit_idx']]
            if len(exit_rows) > 0:
                exit_time = exit_rows.iloc[0]['timestamp']
                if exit_time <= current_time:
                    completed_trades.append({
                        'time': exit_time,
                        'pnl': trade['pnl']
                    })

        if completed_trades:
            pnl_df = pd.DataFrame(completed_trades).sort_values('time')
            pnl_df['cum_pnl'] = pnl_df['pnl'].cumsum()

            fig.add_trace(
                go.Scatter(
                    x=pnl_df['time'],
                    y=pnl_df['cum_pnl'],
                    mode='lines+markers',
                    line=dict(color='blue', width=2),
                    marker=dict(size=5),
                    showlegend=False,
                ),
                row=3, col=1
            )
            fig.add_hline(y=0, line_dash="dash", line_color="gray", row=3, col=1)

    # Layout
    fig.update_layout(
        height=700,
        showlegend=False,
        margin=dict(l=60, r=20, t=40, b=40),
        xaxis_rangeslider_visible=False,
    )
    fig.update_xaxes(tickformat="%m-%d %H:%M", row=1, col=1)
    fig.update_xaxes(tickformat="%H:%M", row=2, col=1)
    fig.update_xaxes(tickformat="%m-%d", title_text="Time", row=3, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Vol", row=2, col=1)
    fig.update_yaxes(title_text="PnL($)", row=3, col=1)

    return fig


def get_current_info(current_bar: int) -> dict:
    """Get info for current bar."""
    if state.df_1m is None or current_bar >= len(state.df_1m):
        return {}

    row = state.df_1m.iloc[current_bar]
    current_time = row['timestamp']

    # Find matching signal (closest 3m bar)
    regime = "N/A"
    signal = "N/A"
    action = "N/A"

    if state.signals_df is not None:
        # Find signal at or before current time
        matching = state.signals_df[state.signals_df['timestamp'] <= current_time]
        if len(matching) > 0:
            sig_row = matching.iloc[-1]
            regime = sig_row.get('regime_confirmed', 'N/A')
            signal = sig_row.get('signal', 'N/A')
            action = sig_row.get('action', 'N/A')

    # Count completed trades
    n_trades = 0
    total_pnl = 0
    win_rate = 0

    if state.trades_df is not None and state.signals_df is not None:
        for _, trade in state.trades_df.iterrows():
            exit_rows = state.signals_df[state.signals_df['idx'] == trade['exit_idx']]
            if len(exit_rows) > 0:
                if exit_rows.iloc[0]['timestamp'] <= current_time:
                    n_trades += 1
                    total_pnl += trade['pnl']

        if n_trades > 0:
            wins = sum(1 for _, t in state.trades_df.iterrows()
                      if t['pnl'] > 0 and
                      len(state.signals_df[state.signals_df['idx'] == t['exit_idx']]) > 0 and
                      state.signals_df[state.signals_df['idx'] == t['exit_idx']].iloc[0]['timestamp'] <= current_time)
            win_rate = wins / n_trades * 100

    return {
        'timestamp': str(current_time),
        'price': f"${row['close']:.4f}",
        'open': f"${row['open']:.4f}",
        'high': f"${row['high']:.4f}",
        'low': f"${row['low']:.4f}",
        'regime': regime,
        'signal': signal,
        'action': action,
        'n_trades': n_trades,
        'total_pnl': f"${total_pnl:.2f}",
        'win_rate': f"{win_rate:.1f}%",
    }


# =============================================================================
# Dash App
# =============================================================================

app = Dash(__name__)

app.layout = html.Div([
    html.H2("XRP-4 Backtest Replay (1m Candles)", style={'textAlign': 'center', 'marginBottom': '10px'}),

    # Controls
    html.Div([
        html.Button("⏮", id='btn-start', n_clicks=0, style={'fontSize': '20px', 'padding': '8px 15px', 'margin': '3px'}),
        html.Button("◀◀", id='btn-back10', n_clicks=0, style={'fontSize': '18px', 'padding': '8px 12px', 'margin': '3px'}),
        html.Button("◀", id='btn-back1', n_clicks=0, style={'fontSize': '18px', 'padding': '8px 12px', 'margin': '3px'}),
        html.Button("▶ Play", id='btn-play', n_clicks=0,
                   style={'fontSize': '18px', 'padding': '8px 20px', 'margin': '3px', 'backgroundColor': '#4CAF50', 'color': 'white'}),
        html.Button("⏸ Stop", id='btn-pause', n_clicks=0,
                   style={'fontSize': '18px', 'padding': '8px 20px', 'margin': '3px', 'backgroundColor': '#f44336', 'color': 'white'}),
        html.Button("▶", id='btn-fwd1', n_clicks=0, style={'fontSize': '18px', 'padding': '8px 12px', 'margin': '3px'}),
        html.Button("▶▶", id='btn-fwd10', n_clicks=0, style={'fontSize': '18px', 'padding': '8px 12px', 'margin': '3px'}),
        html.Button("⏭", id='btn-end', n_clicks=0, style={'fontSize': '20px', 'padding': '8px 15px', 'margin': '3px'}),

        html.Span(" | Speed: ", style={'marginLeft': '20px'}),
        dcc.Dropdown(
            id='speed-dropdown',
            options=[
                {'label': '1x', 'value': 1},
                {'label': '5x', 'value': 5},
                {'label': '10x', 'value': 10},
                {'label': '30x', 'value': 30},
                {'label': '60x', 'value': 60},
            ],
            value=5,
            style={'width': '80px', 'display': 'inline-block', 'verticalAlign': 'middle'},
            clearable=False,
        ),

        html.Span(" | Window: ", style={'marginLeft': '15px'}),
        dcc.Dropdown(
            id='window-dropdown',
            options=[
                {'label': '30 bars', 'value': 30},
                {'label': '60 bars', 'value': 60},
                {'label': '120 bars', 'value': 120},
                {'label': '240 bars', 'value': 240},
            ],
            value=60,
            style={'width': '100px', 'display': 'inline-block', 'verticalAlign': 'middle'},
            clearable=False,
        ),
    ], style={'textAlign': 'center', 'marginBottom': '10px', 'backgroundColor': '#f0f0f0', 'padding': '10px', 'borderRadius': '8px'}),

    # Progress
    html.Div([
        html.Span(id='progress-text', children='0 / 0', style={'fontWeight': 'bold'}),
        html.Span(" | ", style={'margin': '0 10px'}),
        html.Span(id='time-text', children='--'),
    ], style={'textAlign': 'center', 'marginBottom': '5px'}),

    dcc.Slider(id='progress-slider', min=0, max=100, step=1, value=0,
               tooltip={"placement": "bottom"}, marks={}),

    # Main content
    html.Div([
        # Chart
        html.Div([
            dcc.Graph(id='main-chart', style={'height': '700px'}),
        ], style={'flex': '1', 'minWidth': '0'}),

        # Info Panel
        html.Div([
            html.H4("Current Bar", style={'marginTop': '0', 'borderBottom': '2px solid #333', 'paddingBottom': '5px'}),
            html.Div(id='info-panel', style={'fontSize': '12px', 'lineHeight': '1.8'}),
            html.Hr(),
            html.H4("Legend", style={'marginBottom': '8px'}),
            html.Div([
                html.Div([html.Span("■ ", style={'color': '#00cc00'}), "TREND_UP"]),
                html.Div([html.Span("■ ", style={'color': '#cc0000'}), "TREND_DOWN"]),
                html.Div([html.Span("■ ", style={'color': '#0066cc'}), "RANGE"]),
                html.Div([html.Span("■ ", style={'color': '#ff9900'}), "HIGH_VOL"]),
                html.Hr(),
                html.Div([html.Span("▲ ", style={'color': 'green'}), "LONG Entry"]),
                html.Div([html.Span("▼ ", style={'color': 'red'}), "SHORT Entry"]),
                html.Hr(),
                html.Div([html.B("Exit Types (1m):")], style={'marginTop': '3px'}),
                html.Div([html.Span("★ ", style={'color': 'lime'}), "Trailing Stop"]),
                html.Div([html.Span("◆ ", style={'color': 'lime'}), "Break-even"]),
                html.Div([html.Span("● ", style={'color': 'lime'}), "Signal Exit"]),
                html.Div([html.Span("(salmon=loss)", style={'color': 'salmon', 'fontSize': '9px'})]),
            ], style={'fontSize': '11px'}),
            html.Hr(),
            html.H4("Exit Stats", style={'marginBottom': '5px'}),
            html.Div(id='exit-stats', style={'fontSize': '10px', 'lineHeight': '1.6'}),
        ], style={
            'width': '180px', 'flexShrink': '0', 'backgroundColor': '#f5f5f5',
            'padding': '12px', 'borderRadius': '8px', 'marginLeft': '10px',
        }),
    ], style={'display': 'flex', 'alignItems': 'flex-start'}),

    # Hidden state
    dcc.Store(id='current-bar', data=0),
    dcc.Store(id='max-bar', data=0),
    dcc.Store(id='is-playing', data=False),
    dcc.Interval(id='play-interval', interval=200, n_intervals=0, disabled=True),

], style={'padding': '15px', 'fontFamily': 'Arial'})


@callback(
    Output('max-bar', 'data'),
    Output('progress-slider', 'max'),
    Input('main-chart', 'id'),
)
def init_max(_):
    if state.df_1m is not None:
        max_bar = len(state.df_1m) - 1
        return max_bar, max_bar
    return 0, 0


@callback(
    Output('current-bar', 'data'),
    Output('is-playing', 'data'),
    Output('play-interval', 'disabled'),
    Input('btn-start', 'n_clicks'),
    Input('btn-back10', 'n_clicks'),
    Input('btn-back1', 'n_clicks'),
    Input('btn-play', 'n_clicks'),
    Input('btn-pause', 'n_clicks'),
    Input('btn-fwd1', 'n_clicks'),
    Input('btn-fwd10', 'n_clicks'),
    Input('btn-end', 'n_clicks'),
    Input('progress-slider', 'value'),
    Input('play-interval', 'n_intervals'),
    State('current-bar', 'data'),
    State('is-playing', 'data'),
    State('max-bar', 'data'),
    State('speed-dropdown', 'value'),
)
def update_position(start, back10, back1, play, pause, fwd1, fwd10, end, slider,
                   n_intervals, current, is_playing, max_bar, speed):
    triggered = ctx.triggered_id

    if triggered == 'btn-start':
        return 0, False, True
    elif triggered == 'btn-end':
        return max_bar, False, True
    elif triggered == 'btn-back10':
        return max(0, current - 10), False, True
    elif triggered == 'btn-back1':
        return max(0, current - 1), False, True
    elif triggered == 'btn-fwd1':
        return min(max_bar, current + 1), False, True
    elif triggered == 'btn-fwd10':
        return min(max_bar, current + 10), False, True
    elif triggered == 'btn-play':
        return current, True, False
    elif triggered == 'btn-pause':
        return current, False, True
    elif triggered == 'progress-slider':
        return slider, False, True
    elif triggered == 'play-interval' and is_playing:
        new_bar = min(max_bar, current + speed)
        if new_bar >= max_bar:
            return max_bar, False, True
        return new_bar, True, False

    return current, is_playing, not is_playing


@callback(
    Output('main-chart', 'figure'),
    Output('info-panel', 'children'),
    Output('progress-text', 'children'),
    Output('progress-slider', 'value'),
    Output('time-text', 'children'),
    Output('exit-stats', 'children'),
    Input('current-bar', 'data'),
    State('window-dropdown', 'value'),
    State('max-bar', 'data'),
)
def update_chart(current_bar, window, max_bar):
    fig = create_replay_chart(current_bar, window)
    info = get_current_info(current_bar)

    info_children = [
        html.Div(f"Time: {info.get('timestamp', '--')[:19]}"),
        html.Div(f"Close: {info.get('price', '--')}", style={'fontWeight': 'bold', 'fontSize': '14px'}),
        html.Div(f"O: {info.get('open', '--')} H: {info.get('high', '--')}"),
        html.Div(f"L: {info.get('low', '--')}"),
        html.Hr(),
        html.Div(f"Regime: {info.get('regime', '--')}",
                style={'fontWeight': 'bold',
                       'color': '#00aa00' if 'UP' in str(info.get('regime', '')) else
                               ('#aa0000' if 'DOWN' in str(info.get('regime', '')) else '#0066cc')}),
        html.Div(f"Signal: {info.get('signal', '--')}"),
        html.Div(f"Action: {info.get('action', '--')}"),
        html.Hr(),
        html.Div(f"Trades: {info.get('n_trades', 0)}"),
        html.Div(f"PnL: {info.get('total_pnl', '$0.00')}",
                style={'fontWeight': 'bold',
                       'color': 'green' if not info.get('total_pnl', '$0').startswith('$-') else 'red'}),
        html.Div(f"WR: {info.get('win_rate', '0.0%')}"),
    ]

    progress = f"{current_bar:,} / {max_bar:,}"
    time_text = info.get('timestamp', '--')[:19]

    # Exit reason 통계 계산
    exit_stats_children = get_exit_stats(current_bar)

    return fig, info_children, progress, current_bar, time_text, exit_stats_children


def get_exit_stats(current_bar: int) -> list:
    """Get exit reason statistics up to current bar."""
    if state.df_1m is None or current_bar >= len(state.df_1m):
        return [html.Div("No data")]

    current_time = state.df_1m.iloc[current_bar]['timestamp']

    if state.trades_df is None or state.signals_df is None:
        return [html.Div("No trades")]

    # 현재 시점까지 완료된 거래 필터링
    completed_trades = []
    for _, trade in state.trades_df.iterrows():
        exit_rows = state.signals_df[state.signals_df['idx'] == trade['exit_idx']]
        if len(exit_rows) > 0:
            if exit_rows.iloc[0]['timestamp'] <= current_time:
                completed_trades.append(trade)

    if not completed_trades:
        return [html.Div("No trades yet")]

    completed_df = pd.DataFrame(completed_trades)

    # Exit reason별 통계
    exit_stats = {}
    for _, trade in completed_df.iterrows():
        reason = trade.get('exit_reason', 'UNKNOWN')
        if reason not in exit_stats:
            exit_stats[reason] = {'count': 0, 'pnl': 0, 'wins': 0}
        exit_stats[reason]['count'] += 1
        exit_stats[reason]['pnl'] += trade.get('pnl', 0)
        if trade.get('pnl', 0) > 0:
            exit_stats[reason]['wins'] += 1

    # 표시용 children 생성
    children = []
    for reason, stats in sorted(exit_stats.items(), key=lambda x: -x[1]['count']):
        wr = (stats['wins'] / stats['count'] * 100) if stats['count'] > 0 else 0
        pnl_color = 'green' if stats['pnl'] >= 0 else 'red'

        # 종료 이유 축약
        short_reason = reason.replace('_1M', '').replace('DE_CLOSE_FROM_', '')

        children.append(html.Div([
            html.Span(f"{short_reason}: ", style={'fontWeight': 'bold'}),
            html.Span(f"{stats['count']}건 "),
            html.Span(f"WR:{wr:.0f}% ", style={'color': 'blue'}),
            html.Span(f"${stats['pnl']:.1f}", style={'color': pnl_color}),
        ]))

    # 전체 평균 Max Profit / Given Back
    if 'max_profit_pct' in completed_df.columns:
        avg_max_profit = completed_df['max_profit_pct'].mean()
        avg_given_back = completed_df.get('profit_given_back', pd.Series([0])).mean()
        children.append(html.Hr())
        children.append(html.Div(f"Avg Max: {avg_max_profit:.2f}%"))
        children.append(html.Div(f"Avg Given: {avg_given_back:.2f}%"))

    return children


def main():
    parser = argparse.ArgumentParser(description="Backtest Replay with 1m Candles")
    parser.add_argument("--start", type=str, default="2025-12-01")
    parser.add_argument("--end", type=str, default="2025-12-15")
    parser.add_argument("--symbol", type=str, default="XRPUSDT")
    parser.add_argument("--trades", type=str, help="Path to trades CSV")
    parser.add_argument("--signals", type=str, help="Path to signals CSV")
    parser.add_argument("--port", type=int, default=8050)
    args = parser.parse_args()

    # Load 1m candle data
    print(f"Loading 1m data: {args.start} to {args.end}")
    start_dt = datetime.fromisoformat(args.start)
    end_dt = datetime.fromisoformat(args.end)

    state.df_1m = load_ohlcv_from_db(args.symbol, start_dt, end_dt, "1m")
    print(f"  Loaded {len(state.df_1m)} 1m bars")

    # Load backtest results
    trades_path = args.trades
    signals_path = args.signals

    if not trades_path:
        trades_path, signals_path = find_latest_backtest()

    if trades_path:
        print(f"Loading backtest: {trades_path}")
        state.trades_df, state.signals_df = load_backtest_results(trades_path, signals_path)

        # Filter to date range
        state.signals_df = state.signals_df[
            (state.signals_df['timestamp'] >= start_dt) &
            (state.signals_df['timestamp'] <= end_dt)
        ]
        print(f"  {len(state.trades_df)} trades, {len(state.signals_df)} signals in range")

    state.max_bar = len(state.df_1m) - 1

    print(f"\nStarting server at http://localhost:{args.port}")
    app.run(debug=False, port=args.port)


if __name__ == "__main__":
    main()
