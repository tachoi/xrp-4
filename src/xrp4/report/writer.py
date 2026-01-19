"""Report writer for backtest results."""

import json
from pathlib import Path
from typing import List, Optional

import pandas as pd
import structlog

from xrp4.backtest.metrics import BacktestMetrics, compute_monthly_pnl
from xrp4.filters.box_strength import BoxFilterDiagnostics

logger = structlog.get_logger(__name__)


class ReportWriter:
    """Write backtest results to files."""

    def __init__(self, output_dir: Path):
        """Initialize report writer.

        Args:
            output_dir: Directory to write reports to
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_all(
        self,
        trades: List[dict],
        equity_curve: pd.Series,
        metrics: BacktestMetrics,
    ) -> None:
        """Write all reports.

        Args:
            trades: List of trade dictionaries
            equity_curve: Equity curve series
            metrics: Backtest metrics
        """
        self.write_trades(trades)
        self.write_equity_curve(equity_curve)
        self.write_summary(metrics)
        self.write_monthly_pnl(trades)

        logger.info(
            "Reports written",
            output_dir=str(self.output_dir),
        )

    def write_trades(self, trades: List[dict]) -> None:
        """Write trades to CSV.

        Args:
            trades: List of trade dictionaries
        """
        if not trades:
            logger.warning("No trades to write")
            return

        df = pd.DataFrame(trades)
        output_path = self.output_dir / "trades.csv"
        df.to_csv(output_path, index=False)

        logger.debug("Trades written", path=str(output_path), count=len(trades))

    def write_equity_curve(self, equity_curve: pd.Series) -> None:
        """Write equity curve to CSV.

        Args:
            equity_curve: Equity curve series
        """
        df = pd.DataFrame({
            "timestamp": equity_curve.index,
            "equity": equity_curve.values,
        })
        output_path = self.output_dir / "equity.csv"
        df.to_csv(output_path, index=False)

        logger.debug("Equity curve written", path=str(output_path), rows=len(df))

    def write_summary(self, metrics: BacktestMetrics) -> None:
        """Write summary metrics to JSON.

        Args:
            metrics: Backtest metrics
        """
        summary = metrics.to_dict()
        output_path = self.output_dir / "summary.json"

        # Convert numpy types to Python native types for JSON serialization
        def convert_types(obj):
            import numpy as np
            if isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, (np.integer,)):
                return int(obj)
            elif isinstance(obj, (np.floating,)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        summary = convert_types(summary)

        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)

        logger.debug("Summary written", path=str(output_path))

    def write_monthly_pnl(self, trades: List[dict]) -> None:
        """Write monthly PnL breakdown to CSV.

        Args:
            trades: List of trade dictionaries
        """
        monthly = compute_monthly_pnl(trades)

        if monthly.empty:
            logger.warning("No monthly PnL data to write")
            return

        output_path = self.output_dir / "monthly_pnl.csv"
        monthly.to_csv(output_path, index=False)

        logger.debug("Monthly PnL written", path=str(output_path), months=len(monthly))

    def write_box_filter_diagnostics(
        self,
        diagnostics: BoxFilterDiagnostics,
    ) -> None:
        """Write box filter diagnostics to JSON.

        Args:
            diagnostics: Box filter diagnostics
        """
        import numpy as np

        output_path = self.output_dir / "box_filter_diagnostics.json"

        def convert_types(obj):
            if isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(v) for v in obj]
            elif isinstance(obj, (np.integer,)):
                return int(obj)
            elif isinstance(obj, (np.floating,)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.Timestamp):
                return str(obj)
            return obj

        data = convert_types(diagnostics.to_dict())

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.debug("Box filter diagnostics written", path=str(output_path))

    def print_box_filter_summary(self, diagnostics: BoxFilterDiagnostics) -> None:
        """Print box filter summary to console.

        Args:
            diagnostics: Box filter diagnostics
        """
        print("\n" + "=" * 60)
        print("BOX FILTER DIAGNOSTICS (Step 0.1)")
        print("=" * 60)
        print(f"Total Candidates: {diagnostics.total_candidates}")
        print(f"Passed: {diagnostics.passed}")
        print(f"Pass Rate: {diagnostics.get_pass_rate():.2%}")
        print(f"Top Fail Reason: {diagnostics.get_top_fail_reason()}")
        print(f"  - Failed MIN_HEIGHT_ATR: {diagnostics.failed_min_height_atr}")
        print(f"  - Failed REACTION: {diagnostics.failed_reaction}")
        print(f"  - Failed COST: {diagnostics.failed_cost}")
        print(f"Median Box Height (Passed): {diagnostics.get_median_box_height(True):.2f} ATR")
        print(f"Median Box Height (Failed): {diagnostics.get_median_box_height(False):.2f} ATR")

        # Bin stats
        bin_stats = diagnostics.compute_bin_stats()
        if bin_stats:
            print("\nPERFORMANCE BY BOX HEIGHT ATR BIN:")
            print("-" * 60)
            print(f"{'Bin':<10} {'Count':>8} {'Win Rate':>10} {'PF':>8} {'Avg PnL':>10}")
            print("-" * 60)
            for bin_name in ["[0-2)", "[2-3)", "[3-4)", "[4-6)", "[6+)"]:
                if bin_name in bin_stats:
                    s = bin_stats[bin_name]
                    pf_str = f"{s['pf']:.2f}" if s['pf'] != float('inf') else "inf"
                    print(f"{bin_name:<10} {s['count']:>8} {s['win_rate']:>10.2%} {pf_str:>8} {s['avg_pnl']:>10.2f}")
        print("=" * 60 + "\n")

    def print_summary(self, metrics: BacktestMetrics) -> None:
        """Print summary to console.

        Args:
            metrics: Backtest metrics
        """
        from xrp4.backtest.metrics import format_metrics_report

        print("\n")
        print(format_metrics_report(metrics))
        print("\n")
