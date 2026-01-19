"""Box Height ATR Threshold Sweep Experiment.

Sweeps box_filter.min_height_atr to find optimal threshold.

Usage:
    python -m xrp4.experiments.box_threshold_sweep --config configs/step0.yaml
    python -m xrp4.experiments.box_threshold_sweep --config configs/step0.yaml --signal-mode bounce_only
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
import structlog

# Configure logging
structlog.configure(
    wrapper_class=structlog.make_filtering_bound_logger(min_level=20),
)

logger = structlog.get_logger(__name__)

# Default sweep grid
BOX_HEIGHT_ATR_GRID = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0]


def run_single_backtest(
    config_path: Path,
    min_height_atr: float,
    signal_mode: str = "all",
    run_id: str = None,
) -> Dict[str, Any]:
    """Run a single backtest with given parameters.

    Args:
        config_path: Path to base config file
        min_height_atr: Box filter min_height_atr threshold
        signal_mode: Signal mode ("all", "bounce_only", "breakout_only")
        run_id: Optional run identifier

    Returns:
        Dict with metrics and config
    """
    from xrp4.backtest.engine import BacktestEngine
    from xrp4.backtest.metrics import compute_metrics
    from xrp4.config import load_config
    from xrp4.data.candles import load_candles

    # Build overrides
    overrides = {
        "box_filter.min_height_atr": min_height_atr,
        "signal_mode": signal_mode,
    }

    if run_id:
        overrides["run_id"] = run_id

    # Load config with overrides
    config = load_config(config_path, overrides=overrides)

    # Load candles (reuse for all runs)
    df = load_candles(
        symbol=config.symbol,
        timeframe=config.timeframe,
        start=config.start,
        end=config.end,
    )

    if df.empty:
        raise RuntimeError("No candle data loaded")

    # Run backtest
    engine = BacktestEngine(config)
    trades, equity_curve = engine.run(df)

    # Compute metrics
    trade_dicts = [t.to_dict() for t in trades]
    metrics = compute_metrics(
        trades=trade_dicts,
        equity_curve=equity_curve,
        initial_capital=config.initial_capital,
    )

    # Get box filter diagnostics
    box_diag = engine.box_filter.diagnostics.to_dict()

    return {
        "min_height_atr": min_height_atr,
        "signal_mode": signal_mode,
        "total_trades": metrics.total_trades,
        "win_rate": metrics.win_rate,
        "total_pnl": metrics.total_pnl,
        "total_pnl_pct": metrics.total_pnl_pct,
        "profit_factor": metrics.profit_factor,
        "max_drawdown_pct": metrics.max_drawdown_pct,
        "sharpe_ratio": metrics.sharpe_ratio,
        "avg_pnl_per_trade": metrics.avg_pnl_per_trade,
        "avg_win": metrics.avg_win,
        "avg_loss": metrics.avg_loss,
        "box_filter_pass_rate": box_diag.get("pass_rate", 0),
        "box_filter_total_candidates": box_diag.get("total_candidates", 0),
        "box_filter_passed": box_diag.get("passed", 0),
    }


def run_sweep(
    config_path: Path,
    signal_mode: str = "all",
    grid: List[float] = None,
    output_dir: Path = None,
) -> pd.DataFrame:
    """Run full threshold sweep experiment.

    Args:
        config_path: Path to base config file
        signal_mode: Signal mode ("all", "bounce_only", "breakout_only")
        grid: List of min_height_atr values to test
        output_dir: Directory to save results

    Returns:
        DataFrame with all results
    """
    if grid is None:
        grid = BOX_HEIGHT_ATR_GRID

    if output_dir is None:
        output_dir = Path("outputs/experiments")
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []

    logger.info(
        "Starting box threshold sweep",
        signal_mode=signal_mode,
        grid=grid,
    )

    for min_height_atr in grid:
        logger.info(f"Running backtest with min_height_atr={min_height_atr}")

        try:
            result = run_single_backtest(
                config_path=config_path,
                min_height_atr=min_height_atr,
                signal_mode=signal_mode,
                run_id=f"sweep_{signal_mode}_{min_height_atr}",
            )
            results.append(result)

            logger.info(
                f"Completed min_height_atr={min_height_atr}",
                trades=result["total_trades"],
                pnl=f"{result['total_pnl']:.2f}",
                pf=f"{result['profit_factor']:.2f}",
            )

        except Exception as e:
            logger.error(f"Failed min_height_atr={min_height_atr}: {e}")
            results.append({
                "min_height_atr": min_height_atr,
                "signal_mode": signal_mode,
                "error": str(e),
            })

    # Create results DataFrame
    df_results = pd.DataFrame(results)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"box_threshold_sweep_{signal_mode}_{timestamp}.csv"
    df_results.to_csv(csv_path, index=False)
    logger.info(f"Results saved to {csv_path}")

    # Save as JSON too
    json_path = output_dir / f"box_threshold_sweep_{signal_mode}_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    return df_results


def print_sweep_summary(df: pd.DataFrame) -> None:
    """Print sweep results summary."""
    print("\n" + "=" * 80)
    print("BOX HEIGHT ATR THRESHOLD SWEEP RESULTS")
    print("=" * 80)

    print("\n" + "-" * 80)
    print(f"{'min_ATR':>8} {'Trades':>8} {'WinRate':>10} {'PnL':>12} {'PF':>8} {'MDD':>10} {'PassRate':>10}")
    print("-" * 80)

    for _, row in df.iterrows():
        if "error" in row and pd.notna(row.get("error")):
            print(f"{row['min_height_atr']:>8.1f}  ERROR: {row['error']}")
            continue

        print(
            f"{row['min_height_atr']:>8.1f} "
            f"{int(row['total_trades']):>8} "
            f"{row['win_rate']:>9.2f}% "
            f"{row['total_pnl']:>12.2f} "
            f"{row['profit_factor']:>8.2f} "
            f"{row['max_drawdown_pct']:>9.2f}% "
            f"{row['box_filter_pass_rate']:>10.2%}"
        )

    print("-" * 80)

    # Find structural break point
    print("\nANALYSIS:")
    print("-" * 80)

    # Filter valid rows
    valid = df[~df.get("error", pd.Series([None] * len(df))).notna()]

    if len(valid) > 0:
        # Find first positive PnL threshold
        positive_pnl = valid[valid["total_pnl"] > 0]
        if len(positive_pnl) > 0:
            first_positive = positive_pnl.iloc[0]["min_height_atr"]
            print(f"First positive PnL at min_height_atr >= {first_positive}")
        else:
            print("No positive PnL threshold found in sweep range")

        # Find first PF > 1.0
        profitable = valid[valid["profit_factor"] > 1.0]
        if len(profitable) > 0:
            first_profitable = profitable.iloc[0]["min_height_atr"]
            print(f"First PF > 1.0 at min_height_atr >= {first_profitable}")
        else:
            print("No profitable (PF > 1.0) threshold found in sweep range")

        # Check monotonicity
        pnl_series = valid["total_pnl"].values
        is_monotonic = all(pnl_series[i] <= pnl_series[i+1] for i in range(len(pnl_series)-1))
        print(f"PnL monotonically increasing: {is_monotonic}")

    print("=" * 80 + "\n")


def main():
    """Run box threshold sweep experiment."""
    parser = argparse.ArgumentParser(
        description="Box Height ATR Threshold Sweep Experiment"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to base config YAML file",
    )
    parser.add_argument(
        "--signal-mode",
        type=str,
        choices=["all", "bounce_only", "breakout_only"],
        default="all",
        help="Signal mode to test",
    )
    parser.add_argument(
        "--grid",
        type=float,
        nargs="+",
        default=None,
        help="Custom grid values (e.g., --grid 2.0 3.0 4.0)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/experiments",
        help="Output directory for results",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)

    output_dir = Path(args.output_dir)

    # Run sweep
    df_results = run_sweep(
        config_path=config_path,
        signal_mode=args.signal_mode,
        grid=args.grid,
        output_dir=output_dir,
    )

    # Print summary
    print_sweep_summary(df_results)

    # Confirmation
    print("\nCONFIRMATION: NO ML USED")
    print("- No HMM / No XGB / No FSM")
    print("- Pure structural box_height_atr threshold sweep")


if __name__ == "__main__":
    main()
