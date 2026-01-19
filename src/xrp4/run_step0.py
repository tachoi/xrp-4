"""Main runner for XRP-4 Step0 baseline backtest.

Usage:
    python -m xrp4.run_step0 --config configs/step0.yaml
    python -m xrp4.run_step0 --config configs/step0.yaml --override box_filter.min_height_atr=3.5
"""

import argparse
import sys
from pathlib import Path

import structlog

# Configure logging
structlog.configure(
    wrapper_class=structlog.make_filtering_bound_logger(min_level=20),  # INFO
)

logger = structlog.get_logger(__name__)


def main():
    """Run Step0 baseline backtest."""
    parser = argparse.ArgumentParser(description="XRP-4 Step0 Baseline Backtest")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--override",
        type=str,
        nargs="*",
        default=[],
        help="Config overrides in format key=value (e.g., box_filter.min_height_atr=3.5)",
    )
    args = parser.parse_args()

    # Import here to avoid circular imports
    from xrp4.backtest.engine import BacktestEngine
    from xrp4.backtest.metrics import compute_metrics
    from xrp4.config import load_config, parse_overrides
    from xrp4.data.candles import load_candles
    from xrp4.report.writer import ReportWriter

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)

    # Parse CLI overrides
    overrides = parse_overrides(args.override) if args.override else None
    config = load_config(config_path, overrides=overrides)

    if overrides:
        logger.info("Config overrides applied", overrides=overrides)

    logger.info(
        "Loaded configuration",
        symbol=config.symbol,
        timeframe=config.timeframe,
        start=config.start,
        end=config.end,
        initial_capital=config.initial_capital,
    )

    # Load candles from TimescaleDB (via xrp-3 loader)
    logger.info("Loading candles from TimescaleDB...")
    df = load_candles(
        symbol=config.symbol,
        timeframe=config.timeframe,
        start=config.start,
        end=config.end,
    )

    if df.empty:
        logger.error("No candle data loaded. Check DB connection and date range.")
        sys.exit(1)

    logger.info(
        "Loaded candles",
        rows=len(df),
        start=df["timestamp"].iloc[0],
        end=df["timestamp"].iloc[-1],
    )

    # Run backtest
    logger.info("Running backtest...")
    engine = BacktestEngine(config)
    trades, equity_curve = engine.run(df)

    logger.info(
        "Backtest completed",
        total_trades=len(trades),
        final_equity=engine.equity,
    )

    # Calculate metrics
    logger.info("Calculating metrics...")
    trade_dicts = [t.to_dict() for t in trades]
    metrics = compute_metrics(
        trades=trade_dicts,
        equity_curve=equity_curve,
        initial_capital=config.initial_capital,
    )

    # Write reports
    output_dir = Path(config.output_dir) / config.run_id
    logger.info(f"Writing reports to {output_dir}...")
    writer = ReportWriter(output_dir)
    writer.write_all(trade_dicts, equity_curve, metrics)

    # Write box filter diagnostics (Step 0.1)
    if config.box_filter.enabled and config.report.dump_box_filter_diagnostics:
        writer.write_box_filter_diagnostics(engine.box_filter.diagnostics)
        writer.print_box_filter_summary(engine.box_filter.diagnostics)

    # Print summary
    writer.print_summary(metrics)

    logger.info(
        "Backtest complete!",
        output_dir=str(output_dir),
    )

    # Confirmation: NO ML USED
    print("\n" + "=" * 60)
    print("CONFIRMATION: Step0 Baseline (NO-ML)")
    print("=" * 60)
    print("- No HMM (Hidden Markov Model) used")
    print("- No XGB (XGBoost) used")
    print("- No FSM (Finite State Machine) used")
    print("- Pure zone-based bounce/breakout strategy")
    if config.box_filter.enabled:
        print(f"- Box Filter enabled (min_height_atr={config.box_filter.min_height_atr})")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
