"""Metrics calculation - wrapper over xrp-3's metrics module."""

import sys
from pathlib import Path
from typing import List

import pandas as pd

# Add xrp_3 to Python path for imports
XRP3_PATH = Path(__file__).parent.parent.parent.parent.parent / "xrp_3" / "src"
if str(XRP3_PATH) not in sys.path:
    sys.path.insert(0, str(XRP3_PATH))

from xrp3.backtest.metrics import BacktestMetrics, calculate_metrics, format_metrics_report

# Re-export for convenience
__all__ = ["BacktestMetrics", "calculate_metrics", "format_metrics_report"]


def compute_metrics(
    trades: List[dict],
    equity_curve: pd.Series,
    initial_capital: float,
) -> BacktestMetrics:
    """Compute backtest metrics from trade list and equity curve.

    This is a thin wrapper over xrp-3's calculate_metrics function.

    Args:
        trades: List of trade dictionaries
        equity_curve: Equity curve series
        initial_capital: Initial capital amount

    Returns:
        BacktestMetrics with calculated values
    """
    return calculate_metrics(
        trades=trades,
        equity_curve=equity_curve,
        initial_capital=initial_capital,
    )


def compute_monthly_pnl(trades: List[dict]) -> pd.DataFrame:
    """Compute monthly PnL breakdown.

    Args:
        trades: List of trade dictionaries with 'exit_time' and 'pnl'

    Returns:
        DataFrame with monthly PnL statistics
    """
    if not trades:
        return pd.DataFrame()

    df = pd.DataFrame(trades)

    if "exit_time" not in df.columns or "pnl" not in df.columns:
        return pd.DataFrame()

    # Ensure exit_time is datetime
    df["exit_time"] = pd.to_datetime(df["exit_time"])

    # Extract year-month
    df["year_month"] = df["exit_time"].dt.to_period("M")

    # Group by month
    monthly = df.groupby("year_month").agg({
        "pnl": ["sum", "count", "mean"],
    }).reset_index()

    # Flatten column names
    monthly.columns = ["year_month", "total_pnl", "trade_count", "avg_pnl"]

    # Calculate win rate
    df["is_win"] = df["pnl"] > 0
    monthly_winrate = df.groupby("year_month")["is_win"].mean().reset_index()
    monthly_winrate.columns = ["year_month", "win_rate"]

    # Merge
    monthly = monthly.merge(monthly_winrate, on="year_month", how="left")

    # Format year_month as string
    monthly["year_month"] = monthly["year_month"].astype(str)

    return monthly
