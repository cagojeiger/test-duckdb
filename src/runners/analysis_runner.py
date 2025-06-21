#!/usr/bin/env python3
"""
CLI Analysis Runner for DuckDB VSS Benchmarking Results
Analyzes experiment results and generates visualizations and reports
"""

import argparse
import sys
import time
from pathlib import Path
from typing import List, Any

from src.types.core import ExperimentResult
from src.pipelines.analysis.analysis_pipeline import (
    analysis_pipeline,
    quick_analysis_pipeline,
    load_experiment_results,
)


class AnalysisRunner:
    """Main analysis runner with CLI interface"""

    def __init__(
        self,
        checkpoint_dir: Path = Path("checkpoints"),
        output_dir: Path = Path("analysis"),
        interactive: bool = False,
        port: int = 8080,
    ):
        self.checkpoint_dir = checkpoint_dir
        self.output_dir = output_dir
        self.interactive = interactive
        self.port = port

        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "charts").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "reports").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "dashboard").mkdir(parents=True, exist_ok=True)

    def run_full_analysis(self) -> str:
        """Run complete analysis pipeline from checkpoint data to final report.

        Returns:
            Path to the generated comprehensive analysis report
        """
        print("Starting full analysis pipeline...")
        print(f"Checkpoint directory: {self.checkpoint_dir}")
        print(f"Output directory: {self.output_dir}")

        start_time = time.time()

        try:
            pipeline_io = analysis_pipeline(
                checkpoint_dir=str(self.checkpoint_dir), output_dir=str(self.output_dir)
            )

            report_path = pipeline_io.run()

            end_time = time.time()
            duration = end_time - start_time

            print(f"\n✅ Analysis completed successfully in {duration:.2f} seconds")
            print(f"📊 Comprehensive report generated: {report_path}")

            self._list_generated_files()

            return report_path

        except Exception as e:
            print(f"❌ Analysis failed: {str(e)}")
            sys.exit(1)

    def run_quick_analysis(self) -> None:
        """Run quick analysis on loaded results."""
        print("Starting quick analysis...")
        print(f"Checkpoint directory: {self.checkpoint_dir}")

        try:
            load_io = load_experiment_results(str(self.checkpoint_dir))
            results = load_io.run()

            if not results:
                print(f"❌ No experiment results found in {self.checkpoint_dir}")
                sys.exit(1)

            print(f"📊 Loaded {len(results)} experiment results")

            quick_io = quick_analysis_pipeline(results)
            analysis = quick_io.run()

            self._print_analysis_summary(analysis, results)

        except Exception as e:
            print(f"❌ Quick analysis failed: {str(e)}")
            sys.exit(1)

    def start_web_dashboard(self) -> None:
        """Start interactive web dashboard."""
        if not self.interactive:
            print("❌ Interactive mode not enabled. Use --interactive flag.")
            sys.exit(1)

        print(f"🚀 Starting web dashboard on port {self.port}...")

        try:
            from src.web.dashboard import create_dashboard_app

            load_io = load_experiment_results(str(self.checkpoint_dir))
            results = load_io.run()

            if not results:
                print(f"❌ No experiment results found in {self.checkpoint_dir}")
                sys.exit(1)

            create_dashboard_app(results)

            print(f"📊 Dashboard available at: http://localhost:{self.port}")
            print("Press Ctrl+C to stop the dashboard")

            print("🔧 Web dashboard implementation pending...")

        except ImportError:
            print("❌ Web dashboard dependencies not available")
            print(
                "💡 Run full analysis instead: python -m src.runners.analysis_runner --checkpoint-dir checkpoints/"
            )
            sys.exit(1)
        except Exception as e:
            print(f"❌ Dashboard startup failed: {str(e)}")
            sys.exit(1)

    def _print_analysis_summary(
        self, analysis: Any, results: List[ExperimentResult]
    ) -> None:
        """Print analysis summary to console."""
        print("\n" + "=" * 60)
        print("📊 ANALYSIS SUMMARY")
        print("=" * 60)

        print(f"\n📈 Experiment Results: {len(results)} total experiments")

        if analysis.dimension_performance:
            print("\n🔍 Vector Dimension Performance:")
            for dim, perf in sorted(analysis.dimension_performance.items()):
                print(f"  • {dim}D vectors: {perf:.2f}ms avg query time")

        if analysis.scale_performance:
            print("\n📏 Data Scale Performance:")
            for scale, perf in analysis.scale_performance.items():
                print(f"  • {scale}: {perf:.2f} QPS avg throughput")

        if analysis.search_type_comparison:
            print("\n🔎 Search Type Comparison:")
            for search_type, perf in analysis.search_type_comparison.items():
                print(f"  • {search_type}: {perf:.2f}ms avg query time")

        if analysis.filter_impact:
            print("\n🔧 Filter Impact:")
            for filter_key, impact in analysis.filter_impact.items():
                if isinstance(impact, (int, float)):
                    print(f"  • {filter_key}: {impact:.2f}ms")
                else:
                    print(f"  • {filter_key}: {impact}")

        print("\n" + "=" * 60)

    def _list_generated_files(self) -> None:
        """List all generated analysis files."""
        print("\n📁 Generated Files:")

        charts_dir = self.output_dir / "charts"
        if charts_dir.exists():
            chart_files = list(charts_dir.glob("*.png"))
            if chart_files:
                print(f"  📊 Charts ({len(chart_files)} files):")
                for chart_file in sorted(chart_files):
                    print(f"    • {chart_file}")

        reports_dir = self.output_dir / "reports"
        if reports_dir.exists():
            report_files = list(reports_dir.glob("*.md"))
            if report_files:
                print(f"  📄 Reports ({len(report_files)} files):")
                for report_file in sorted(report_files):
                    print(f"    • {report_file}")

        dashboard_dir = self.output_dir / "dashboard"
        if dashboard_dir.exists():
            dashboard_files = list(dashboard_dir.glob("*.html"))
            if dashboard_files:
                print(f"  🌐 Dashboard ({len(dashboard_files)} files):")
                for dashboard_file in sorted(dashboard_files):
                    print(f"    • {dashboard_file}")


def create_cli_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser"""
    parser = argparse.ArgumentParser(
        description="DuckDB VSS Benchmarking Analysis Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.runners.analysis_runner --checkpoint-dir checkpoints/ --output-dir analysis/

  python -m src.runners.analysis_runner --checkpoint-dir checkpoints/ --quick

  python -m src.runners.analysis_runner --checkpoint-dir checkpoints/ --interactive --port 8080

  python -m src.runners.analysis_runner --checkpoint-dir checkpoints/ --output-dir custom_analysis/
        """,
    )

    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints"),
        help="Directory containing experiment checkpoint files (default: checkpoints/)",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis"),
        help="Directory to save analysis outputs (default: analysis/)",
    )

    analysis_group = parser.add_mutually_exclusive_group()

    analysis_group.add_argument(
        "--quick",
        action="store_true",
        help="Run quick analysis and print summary to console",
    )

    analysis_group.add_argument(
        "--interactive",
        action="store_true",
        help="Start interactive web dashboard",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port for web dashboard (default: 8080)",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )

    return parser


def main() -> None:
    """Main entry point"""
    parser = create_cli_parser()
    args = parser.parse_args()

    if not args.checkpoint_dir.exists():
        print(f"❌ Checkpoint directory not found: {args.checkpoint_dir}")
        print("💡 Make sure to run experiments first using:")
        print("   python -m src.runners.experiment_runner --all")
        sys.exit(1)

    runner = AnalysisRunner(
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
        interactive=args.interactive,
        port=args.port,
    )

    if args.quick:
        runner.run_quick_analysis()
    elif args.interactive:
        runner.start_web_dashboard()
    else:
        report_path = runner.run_full_analysis()

        if args.verbose:
            print("\n📖 To view the comprehensive report:")
            print(f"   cat {report_path}")


if __name__ == "__main__":
    main()
