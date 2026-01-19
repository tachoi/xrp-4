#!/usr/bin/env python
"""
Interactive Backtest Viewer with Real-time Playback Controls.

Features:
- Play/Pause/Stop controls
- Speed adjustment (0.1x - 10x)
- Step forward/backward
- Real-time chart updates
- Trade markers and regime colors
- Statistics panel

Usage:
    python scripts/backtest_viewer.py
    python scripts/backtest_viewer.py --trades outputs/multi_hmm_backtest/trades_test.csv \
                                      --signals outputs/multi_hmm_backtest/signals_test.csv
"""

import argparse
import sys
from pathlib import Path
from datetime import timedelta

import pandas as pd
import numpy as np

from dash import Dash, html, dcc, callback, Input, Output, State, ctx
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Global state
class BacktestState:
    def __init__(self):
        self.signals_df = None
        self.trades_df = None
        self.current_idx = 0
        self.is_playing = False
        self.speed = 1.0  # bars per update
        self.window_size = 100  # bars to show

state = BacktestState()


def load_data(trades_path: str, signals_path: str):
    """Load backtest data."""
    state.trades_df = pd.read_csv(trades_path)
    state.signals_df = pd.read_csv(signals_path)
    state.signals_df['timestamp'] = pd.to_datetime(state.signals_df['timestamp']).dt.tz_localize(None)
    state.current_idx = state.window_size
    return len(state.signals_df)


def find_latest_backtest():
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


def create_chart(current_idx: int, window_size: int = 100) -> go.Figure:
    """Create chart for current position."""
    if state.signals_df is None or len(state.signals_df) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No data loaded", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig

    # Ensure valid index
    current_idx = max(0, min(current_idx, len(state.signals_df) - 1))

    # Get window of data - show from start, extend right as we progress
    start_idx = max(0, current_idx - window_size + 20)  # Keep some history
    end_idx = min(len(state.signals_df), current_idx + 20)  # Show some future bars

    df = state.signals_df.iloc[start_idx:end_idx].copy()

    if len(df) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No data in range", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig

    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.75, 0.25],
        subplot_titles=('Price & Trades', 'Cumulative PnL')
    )

    # Price line
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['price'],
            mode='lines',
            name='Price',
            line=dict(color='#1f77b4', width=1.5),
        ),
        row=1, col=1
    )

    # Current position marker
    if current_idx < len(state.signals_df):
        current_row = state.signals_df.iloc[current_idx]
        fig.add_trace(
            go.Scatter(
                x=[current_row['timestamp']],
                y=[current_row['price']],
                mode='markers',
                marker=dict(symbol='circle', size=12, color='yellow',
                           line=dict(width=2, color='black')),
                name='Current',
                hovertemplate=(
                    f"<b>Current Bar</b><br>"
                    f"Time: {current_row['timestamp']}<br>"
                    f"Price: ${current_row['price']:.4f}<br>"
                    f"Regime: {current_row.get('regime_confirmed', 'N/A')}<br>"
                    f"Signal: {current_row.get('signal', 'N/A')}<br>"
                    "<extra></extra>"
                ),
            ),
            row=1, col=1
        )

    # Regime background colors
    regime_colors = {
        'TREND_UP': 'rgba(0, 200, 0, 0.15)',
        'TREND_DOWN': 'rgba(200, 0, 0, 0.15)',
        'RANGE': 'rgba(0, 0, 200, 0.1)',
        'TRANSITION': 'rgba(200, 200, 0, 0.1)',
        'HIGH_VOL': 'rgba(255, 165, 0, 0.2)',
    }

    if 'regime_confirmed' in df.columns:
        df_regime = df.copy()
        df_regime['regime_group'] = (df_regime['regime_confirmed'] != df_regime['regime_confirmed'].shift()).cumsum()

        for _, group in df_regime.groupby('regime_group'):
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

    # Trade markers (only show trades within current view)
    if state.trades_df is not None and len(state.trades_df) > 0:
        signals_idx_map = {row['idx']: row['timestamp'] for _, row in df.iterrows() if 'idx' in df.columns}

        for _, trade in state.trades_df.iterrows():
            exit_idx = trade['exit_idx']
            bars_held = int(trade.get('bars_held', 1))
            entry_idx = exit_idx - bars_held

            # Only show if trade is visible and before current position
            if exit_idx > current_idx:
                continue
            if exit_idx < start_idx:
                continue

            # Get timestamps from signals
            if 'idx' in state.signals_df.columns:
                entry_rows = state.signals_df[state.signals_df['idx'] == entry_idx]
                exit_rows = state.signals_df[state.signals_df['idx'] == exit_idx]

                if len(entry_rows) > 0 and len(exit_rows) > 0:
                    entry_time = entry_rows.iloc[0]['timestamp']
                    exit_time = exit_rows.iloc[0]['timestamp']

                    side = trade['side']
                    pnl = trade['pnl']

                    # Entry marker
                    fig.add_trace(
                        go.Scatter(
                            x=[entry_time],
                            y=[trade['entry_price']],
                            mode='markers',
                            marker=dict(
                                symbol='triangle-up' if side == 'LONG' else 'triangle-down',
                                size=14,
                                color='green' if side == 'LONG' else 'red',
                                line=dict(width=1, color='black')
                            ),
                            name=f'{side}',
                            showlegend=False,
                            hovertemplate=f'{side} Entry @ ${trade["entry_price"]:.4f}<extra></extra>',
                        ),
                        row=1, col=1
                    )

                    # Exit marker
                    fig.add_trace(
                        go.Scatter(
                            x=[exit_time],
                            y=[trade['exit_price']],
                            mode='markers',
                            marker=dict(
                                symbol='circle',
                                size=10,
                                color='lime' if pnl > 0 else 'salmon',
                                line=dict(width=1, color='black')
                            ),
                            showlegend=False,
                            hovertemplate=f'Exit @ ${trade["exit_price"]:.4f}<br>PnL: ${pnl:.2f}<extra></extra>',
                        ),
                        row=1, col=1
                    )

    # Cumulative PnL up to current position
    if state.trades_df is not None and len(state.trades_df) > 0:
        completed_trades = state.trades_df[state.trades_df['exit_idx'] <= current_idx].copy()
        if len(completed_trades) > 0:
            completed_trades['cum_pnl'] = completed_trades['pnl'].cumsum()

            pnl_times = []
            pnl_values = []
            for _, trade in completed_trades.iterrows():
                exit_rows = state.signals_df[state.signals_df['idx'] == trade['exit_idx']]
                if len(exit_rows) > 0:
                    pnl_times.append(exit_rows.iloc[0]['timestamp'])
                    pnl_values.append(trade['cum_pnl'])

            if pnl_times:
                fig.add_trace(
                    go.Scatter(
                        x=pnl_times,
                        y=pnl_values,
                        mode='lines+markers',
                        line=dict(color='blue', width=2),
                        marker=dict(size=5),
                        name='Cumulative PnL',
                    ),
                    row=2, col=1
                )
                fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)

    # Layout
    fig.update_layout(
        height=550,
        showlegend=False,
        margin=dict(l=60, r=20, t=40, b=40),
        hovermode='x unified',
        xaxis_rangeslider_visible=False,
    )
    # Show time on x-axis
    fig.update_xaxes(
        title_text="",
        tickformat="%m-%d %H:%M",
        tickangle=-45,
        showticklabels=True,
        row=1, col=1
    )
    fig.update_xaxes(
        title_text="Time",
        tickformat="%m-%d %H:%M",
        tickangle=-45,
        row=2, col=1
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="PnL ($)", row=2, col=1)

    return fig


def get_current_info(idx: int) -> dict:
    """Get information for current bar."""
    if state.signals_df is None or idx >= len(state.signals_df):
        return {}

    row = state.signals_df.iloc[idx]

    # Count completed trades
    completed_trades = state.trades_df[state.trades_df['exit_idx'] <= idx] if state.trades_df is not None else pd.DataFrame()
    n_trades = len(completed_trades)
    total_pnl = completed_trades['pnl'].sum() if n_trades > 0 else 0
    wins = len(completed_trades[completed_trades['pnl'] > 0]) if n_trades > 0 else 0
    win_rate = wins / n_trades * 100 if n_trades > 0 else 0

    return {
        'timestamp': str(row['timestamp']),
        'price': f"${row['price']:.4f}",
        'regime': row.get('regime_confirmed', 'N/A'),
        'signal': row.get('signal', 'N/A'),
        'action': row.get('action', 'N/A'),
        'n_trades': n_trades,
        'total_pnl': f"${total_pnl:.2f}",
        'win_rate': f"{win_rate:.1f}%",
    }


# Create Dash app
app = Dash(__name__)

app.layout = html.Div([
    # Header
    html.H2("XRP-4 Backtest Viewer", style={'textAlign': 'center', 'marginBottom': '10px'}),

    # Control Panel
    html.Div([
        # Playback controls
        html.Div([
            html.Button("⏮ Start", id='btn-start', n_clicks=0,
                       style={'margin': '5px', 'fontSize': '16px', 'padding': '10px 15px'}),
            html.Button("◀ -10", id='btn-back10', n_clicks=0,
                       style={'margin': '5px', 'fontSize': '16px', 'padding': '10px 15px'}),
            html.Button("◀ -1", id='btn-back1', n_clicks=0,
                       style={'margin': '5px', 'fontSize': '16px', 'padding': '10px 15px'}),
            html.Button("▶ Play", id='btn-play', n_clicks=0,
                       style={'margin': '5px', 'fontSize': '18px', 'padding': '10px 20px', 'backgroundColor': '#4CAF50', 'color': 'white'}),
            html.Button("⏸ Pause", id='btn-pause', n_clicks=0,
                       style={'margin': '5px', 'fontSize': '18px', 'padding': '10px 20px', 'backgroundColor': '#f44336', 'color': 'white'}),
            html.Button("▶ +1", id='btn-fwd1', n_clicks=0,
                       style={'margin': '5px', 'fontSize': '16px', 'padding': '10px 15px'}),
            html.Button("▶ +10", id='btn-fwd10', n_clicks=0,
                       style={'margin': '5px', 'fontSize': '16px', 'padding': '10px 15px'}),
            html.Button("⏭ End", id='btn-end', n_clicks=0,
                       style={'margin': '5px', 'fontSize': '16px', 'padding': '10px 15px'}),
        ], style={'textAlign': 'center', 'marginBottom': '10px'}),

        # Speed control
        html.Div([
            html.Label("Speed: ", style={'fontSize': '14px'}),
            dcc.Slider(
                id='speed-slider',
                min=1,
                max=50,
                step=1,
                value=5,
                marks={1: '1x', 10: '10x', 25: '25x', 50: '50x'},
            ),
        ], style={'width': '300px', 'display': 'inline-block', 'marginLeft': '20px', 'verticalAlign': 'middle'}),

        # Window size
        html.Div([
            html.Label("Window: ", style={'fontSize': '14px'}),
            dcc.Slider(
                id='window-slider',
                min=50,
                max=500,
                step=50,
                value=100,
                marks={50: '50', 100: '100', 200: '200', 500: '500'},
            ),
        ], style={'width': '300px', 'display': 'inline-block', 'marginLeft': '20px', 'verticalAlign': 'middle'}),

    ], style={'backgroundColor': '#f0f0f0', 'padding': '15px', 'borderRadius': '10px', 'marginBottom': '10px'}),

    # Progress bar
    html.Div([
        html.Div([
            html.Span(id='progress-text', children='0 / 0'),
            html.Span(" | ", style={'margin': '0 10px'}),
            html.Span(id='position-text', children=''),
        ], style={'textAlign': 'center', 'marginBottom': '5px'}),
        dcc.Slider(
            id='progress-slider',
            min=0,
            max=100,
            step=1,
            value=0,
            marks={},
            tooltip={"placement": "bottom", "always_visible": False},
        ),
    ], style={'marginBottom': '10px'}),

    # Main content: Chart + Info Panel (using flexbox for better layout)
    html.Div([
        # Chart (left)
        html.Div([
            dcc.Graph(id='main-chart', style={'height': '550px'}),
        ], style={'flex': '1', 'minWidth': '0'}),

        # Info Panel (right)
        html.Div([
            html.H4("Current Bar Info", style={'marginTop': '0', 'marginBottom': '10px', 'borderBottom': '2px solid #333', 'paddingBottom': '5px'}),
            html.Div(id='info-panel', children=[
                html.P("Loading...", style={'color': '#666'}),
            ], style={'fontSize': '13px', 'lineHeight': '1.6'}),
            html.Hr(style={'margin': '15px 0'}),
            html.H4("Regime Legend", style={'marginBottom': '10px'}),
            html.Div([
                html.Div([html.Span("■ ", style={'color': '#00cc00', 'fontSize': '16px'}), "TREND_UP"], style={'marginBottom': '3px'}),
                html.Div([html.Span("■ ", style={'color': '#cc0000', 'fontSize': '16px'}), "TREND_DOWN"], style={'marginBottom': '3px'}),
                html.Div([html.Span("■ ", style={'color': '#0066cc', 'fontSize': '16px'}), "RANGE"], style={'marginBottom': '3px'}),
                html.Div([html.Span("■ ", style={'color': '#ff9900', 'fontSize': '16px'}), "HIGH_VOL"], style={'marginBottom': '3px'}),
            ], style={'fontSize': '12px'}),
        ], style={
            'width': '220px',
            'flexShrink': '0',
            'backgroundColor': '#f5f5f5',
            'padding': '15px',
            'borderRadius': '8px',
            'marginLeft': '15px',
            'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
        }),
    ], style={'display': 'flex', 'alignItems': 'flex-start'}),

    # Hidden components for state
    dcc.Store(id='current-idx', data=0),  # Start at beginning
    dcc.Store(id='is-playing', data=False),
    dcc.Store(id='max-idx', data=1000),
    dcc.Interval(id='play-interval', interval=200, n_intervals=0, disabled=True),

], style={'padding': '20px', 'fontFamily': 'Arial, sans-serif'})


@callback(
    Output('max-idx', 'data'),
    Output('progress-slider', 'max'),
    Input('main-chart', 'id'),  # Trigger on load
)
def init_max_idx(_):
    if state.signals_df is not None:
        max_idx = len(state.signals_df) - 1
        return max_idx, max_idx
    return 0, 0


@callback(
    Output('current-idx', 'data'),
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
    State('current-idx', 'data'),
    State('is-playing', 'data'),
    State('max-idx', 'data'),
    State('speed-slider', 'value'),
    State('window-slider', 'value'),
)
def update_position(start, back10, back1, play, pause, fwd1, fwd10, end, slider_val,
                   n_intervals, current_idx, is_playing, max_idx, speed, window):
    triggered = ctx.triggered_id

    # Determine interval disabled state based on is_playing
    interval_disabled = not is_playing

    if triggered == 'btn-start':
        return 0, False, True  # Stop and go to beginning
    elif triggered == 'btn-end':
        return max_idx, False, True  # Stop and go to end
    elif triggered == 'btn-back10':
        return max(0, current_idx - 10), False, True  # Stop and step back
    elif triggered == 'btn-back1':
        return max(0, current_idx - 1), False, True  # Stop and step back
    elif triggered == 'btn-fwd1':
        return min(max_idx, current_idx + 1), False, True  # Stop and step forward
    elif triggered == 'btn-fwd10':
        return min(max_idx, current_idx + 10), False, True  # Stop and step forward
    elif triggered == 'btn-play':
        return current_idx, True, False  # Start playing (interval enabled)
    elif triggered == 'btn-pause':
        return current_idx, False, True  # Stop playing (interval disabled)
    elif triggered == 'progress-slider':
        return max(0, slider_val), False, True  # Stop and jump to position
    elif triggered == 'play-interval' and is_playing:
        new_idx = min(max_idx, current_idx + speed)
        if new_idx >= max_idx:
            return max_idx, False, True  # Reached end, stop
        return new_idx, True, False  # Continue playing

    # Default: keep current state
    return current_idx, is_playing, interval_disabled


@callback(
    Output('main-chart', 'figure'),
    Output('info-panel', 'children'),
    Output('progress-text', 'children'),
    Output('progress-slider', 'value'),
    Output('position-text', 'children'),
    Input('current-idx', 'data'),
    State('window-slider', 'value'),
    State('max-idx', 'data'),
)
def update_chart(current_idx, window, max_idx):
    fig = create_chart(current_idx, window)
    info = get_current_info(current_idx)

    info_children = [
        html.P(f"Timestamp: {info.get('timestamp', '--')}"),
        html.P(f"Price: {info.get('price', '--')}"),
        html.P(f"Regime: {info.get('regime', '--')}",
               style={'fontWeight': 'bold',
                      'color': 'green' if 'UP' in info.get('regime', '') else
                              ('red' if 'DOWN' in info.get('regime', '') else 'blue')}),
        html.P(f"Signal: {info.get('signal', '--')}"),
        html.P(f"Action: {info.get('action', '--')}"),
        html.Hr(),
        html.P(f"Trades: {info.get('n_trades', '--')}"),
        html.P(f"Total PnL: {info.get('total_pnl', '--')}",
               style={'fontWeight': 'bold',
                      'color': 'green' if '$-' not in info.get('total_pnl', '$0') else 'red'}),
        html.P(f"Win Rate: {info.get('win_rate', '--')}"),
    ]

    progress_text = f"{current_idx} / {max_idx}"
    position_text = f"{info.get('timestamp', '')}"

    return fig, info_children, progress_text, current_idx, position_text


def main():
    parser = argparse.ArgumentParser(description="Interactive Backtest Viewer")
    parser.add_argument("--trades", type=str, help="Path to trades CSV")
    parser.add_argument("--signals", type=str, help="Path to signals CSV")
    parser.add_argument("--port", type=int, default=8050, help="Port to run on")
    args = parser.parse_args()

    # Find data files
    trades_path = args.trades
    signals_path = args.signals

    if not trades_path or not signals_path:
        trades_path, signals_path = find_latest_backtest()
        if not trades_path:
            print("Error: Could not find backtest results.")
            return

    print(f"Loading: {trades_path}")
    total_bars = load_data(trades_path, signals_path)
    print(f"Loaded {total_bars} bars, {len(state.trades_df)} trades")

    print(f"\nStarting server at http://localhost:{args.port}")
    print("Press Ctrl+C to stop\n")

    app.run(debug=False, port=args.port)


if __name__ == "__main__":
    main()
