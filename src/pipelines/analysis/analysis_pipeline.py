"""Analysis pipeline composition for DuckDB VSS benchmarking results.

This module composes pure analysis functions and IO effects into complete
analysis pipelines following the functional programming architecture.
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional
from pathlib import Path
import json

from src.types.monads import IO
from src.types.core import ExperimentResult
from src.pure.analyzers.performance_analyzer import (
    PerformanceAnalysis,
    TrendAnalysis,
    analyze_dimension_performance,
    calculate_performance_trends,
)
from src.effects.visualization.chart_generator import (
    generate_performance_heatmap,
    create_trend_charts,
    export_interactive_dashboard,
    generate_summary_report,
)


def load_experiment_results(checkpoint_dir: str) -> IO[List[ExperimentResult]]:
    """Load experiment results from checkpoint directory.

    Args:
        checkpoint_dir: Directory containing checkpoint files

    Returns:
        IO[List[ExperimentResult]]: Loaded experiment results
    """

    def _load_results() -> List[ExperimentResult]:
        from src.types.core import (
            ExperimentConfig,
            DataScale,
            SearchType,
            FilterConfig,
            HNSWParams,
            Metrics,
            SearchResult,
            AccuracyMetrics,
            ExperimentResult,
            Dimension,
            Category,
            Distance,
        )
        from datetime import datetime

        checkpoint_path = Path(checkpoint_dir)
        if not checkpoint_path.exists():
            return []

        results = []

        for json_file in checkpoint_path.glob("*.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                if isinstance(data, list):
                    for item in data:
                        if "config" in item and "results" in item:
                            config_data = item["config"]
                            config = ExperimentConfig(
                                data_scale=DataScale(config_data["data_scale"]),
                                dimension=Dimension(config_data["dimension"]),
                                search_type=SearchType(config_data["search_type"]),
                                filter_config=FilterConfig(
                                    enabled=config_data.get("filter_enabled", False),
                                    category=Category(config_data["filter_category"])
                                    if config_data.get("filter_category")
                                    else None,
                                ),
                                hnsw_params=HNSWParams(
                                    ef_construction=config_data.get(
                                        "ef_construction", 128
                                    ),
                                    ef_search=config_data.get("ef_search", 64),
                                    M=config_data.get("M", 16),
                                    metric=config_data.get("metric", "cosine"),
                                ),
                                batch_size=config_data.get("batch_size", 1000),
                                num_queries=config_data.get("num_queries", 100),
                            )

                            metrics_data = item["results"]
                            insert_metrics = Metrics(
                                query_time_ms=metrics_data.get("insert_time_ms", 0.0),
                                throughput_qps=metrics_data.get(
                                    "insert_throughput", 0.0
                                ),
                                memory_usage_mb=metrics_data.get(
                                    "memory_usage_mb", 0.0
                                ),
                                index_size_mb=metrics_data.get("index_size_mb", 0.0),
                            )

                            index_metrics = Metrics(
                                query_time_ms=metrics_data.get(
                                    "index_build_time_ms", 0.0
                                ),
                                throughput_qps=0.0,
                                memory_usage_mb=metrics_data.get(
                                    "memory_usage_mb", 0.0
                                ),
                                index_size_mb=metrics_data.get("index_size_mb", 0.0),
                            )

                            search_results = []
                            search_data = metrics_data.get("search_results", [])
                            for search_item in search_data:
                                search_metrics = Metrics(
                                    query_time_ms=search_item.get("query_time_ms", 0.0),
                                    throughput_qps=search_item.get(
                                        "throughput_qps", 0.0
                                    ),
                                    memory_usage_mb=search_item.get(
                                        "memory_usage_mb", 0.0
                                    ),
                                    index_size_mb=search_item.get("index_size_mb", 0.0),
                                )

                                search_result = SearchResult(
                                    query_id=search_item.get("query_id", ""),
                                    retrieved_ids=search_item.get("retrieved_ids", []),
                                    distances=[
                                        Distance(d)
                                        for d in search_item.get("distances", [])
                                    ],
                                    metrics=search_metrics,
                                )
                                search_results.append(search_result)

                            accuracy_data = metrics_data.get("accuracy", {})
                            accuracy = AccuracyMetrics(
                                recall_at_1=accuracy_data.get("recall_at_1", 0.0),
                                recall_at_5=accuracy_data.get("recall_at_5", 0.0),
                                recall_at_10=accuracy_data.get("recall_at_10", 0.0),
                                mean_reciprocal_rank=accuracy_data.get("mrr", 0.0),
                            )

                            experiment_result = ExperimentResult(
                                config=config,
                                insert_metrics=insert_metrics,
                                index_metrics=index_metrics,
                                search_results=search_results,
                                accuracy=accuracy,
                                timestamp=datetime.fromisoformat(
                                    item.get("timestamp", datetime.now().isoformat())
                                ),
                            )

                            results.append(experiment_result)

            except (json.JSONDecodeError, KeyError, ValueError) as e:
                print(f"Warning: Could not parse {json_file}: {e}")
                continue

        return results

    class LoadResultsIO(IO[List[ExperimentResult]]):
        def run(self) -> List[ExperimentResult]:
            return _load_results()

    return LoadResultsIO()


def generate_visualizations(
    analysis: PerformanceAnalysis, trends: Optional[TrendAnalysis] = None
) -> IO[Dict[str, Any]]:
    """Generate all visualization outputs.

    Args:
        analysis: Performance analysis results
        trends: Optional trend analysis results

    Returns:
        IO[Dict[str, Any]]: Dictionary containing paths to generated files
    """

    def _generate_all() -> Dict[str, Any]:
        visualization_paths = {}

        heatmap_io = generate_performance_heatmap(analysis)
        heatmap_path = heatmap_io.run()
        visualization_paths["heatmap"] = heatmap_path

        if trends:
            trends_io = create_trend_charts(trends)
            trend_paths = trends_io.run()
            visualization_paths["trends"] = trend_paths

        dashboard_io = export_interactive_dashboard(analysis, trends)
        dashboard_path = dashboard_io.run()
        visualization_paths["dashboard"] = dashboard_path

        report_io = generate_summary_report(analysis, trends)
        report_path = report_io.run()
        visualization_paths["report"] = report_path

        return visualization_paths

    class GenerateVisualizationsIO(IO[Dict[str, Any]]):
        def run(self) -> Dict[str, Any]:
            return _generate_all()

    return GenerateVisualizationsIO()


def create_analysis_report(
    analysis: PerformanceAnalysis,
    trends: Optional[TrendAnalysis],
    visualization_paths: Dict[str, Any],
) -> IO[str]:
    """Create comprehensive analysis report.

    Args:
        analysis: Performance analysis results
        trends: Optional trend analysis results
        visualization_paths: Paths to generated visualization files

    Returns:
        IO[str]: Path to the generated analysis report
    """

    def _create_report() -> str:
        from datetime import datetime

        analysis_dir = Path("analysis/reports")
        analysis_dir.mkdir(parents=True, exist_ok=True)

        report_content = []
        report_content.append("# DuckDB VSS Performance Analysis Report\n\n")
        report_content.append(
            f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        )

        report_content.append("## Executive Summary\n\n")
        report_content.append(
            "This report presents a comprehensive analysis of DuckDB Vector Similarity Search (VSS) performance "
        )
        report_content.append(
            "across different configurations including vector dimensions, data scales, search types, and filter conditions.\n\n"
        )

        report_content.append("## Performance Analysis\n\n")

        if analysis.dimension_performance:
            report_content.append("### Vector Dimension Performance\n\n")
            report_content.append("| Dimension | Avg Query Time (ms) |\n")
            report_content.append("|-----------|--------------------|\n")
            for dim, perf in sorted(analysis.dimension_performance.items()):
                report_content.append(f"| {dim}D | {perf:.2f} |\n")
            report_content.append("\n")

        if analysis.scale_performance:
            report_content.append("### Data Scale Performance\n\n")
            report_content.append("| Scale | Avg Throughput (QPS) |\n")
            report_content.append("|-------|---------------------|\n")
            for scale, perf in analysis.scale_performance.items():
                report_content.append(f"| {scale} | {perf:.2f} |\n")
            report_content.append("\n")

        if analysis.search_type_comparison:
            report_content.append("### Search Type Comparison\n\n")
            report_content.append("| Search Type | Avg Query Time (ms) |\n")
            report_content.append("|-------------|--------------------|\n")
            for search_type, perf in analysis.search_type_comparison.items():
                report_content.append(f"| {search_type} | {perf:.2f} |\n")
            report_content.append("\n")

        if analysis.filter_impact:
            report_content.append("### Filter Impact Analysis\n\n")
            for filter_key, impact in analysis.filter_impact.items():
                if isinstance(impact, (int, float)):
                    report_content.append(f"- **{filter_key}**: {impact:.2f}ms\n")
                else:
                    report_content.append(f"- **{filter_key}**: {impact}\n")
            report_content.append("\n")

        if trends:
            report_content.append("## Trend Analysis\n\n")

            if trends.temporal_trends:
                report_content.append("### Temporal Performance Trends\n\n")
                for metric, data_points in trends.temporal_trends.items():
                    report_content.append(
                        f"- **{metric}**: {len(data_points)} measurements over time\n"
                    )
                report_content.append("\n")

            if trends.scale_trends:
                report_content.append("### Scale Performance Trends\n\n")
                for metric, data_points in trends.scale_trends.items():
                    scale_values = [dp[0] for dp in data_points]
                    report_content.append(
                        f"- **{metric}**: Performance scaling across {len(set(scale_values))} different data scales\n"
                    )
                report_content.append("\n")

            if trends.dimension_trends:
                report_content.append("### Dimension Performance Trends\n\n")
                for metric, data_points in trends.dimension_trends.items():
                    dimension_values = [dp[0] for dp in data_points]
                    report_content.append(
                        f"- **{metric}**: Performance across {len(set(dimension_values))} different vector dimensions\n"
                    )
                report_content.append("\n")

        report_content.append("## Generated Analysis Files\n\n")
        for file_type, file_path in visualization_paths.items():
            if isinstance(file_path, list):
                report_content.append(f"### {file_type.title()}\n")
                for path in file_path:
                    report_content.append(f"- `{path}`\n")
            else:
                report_content.append(f"- **{file_type.title()}**: `{file_path}`\n")
        report_content.append("\n")

        report_content.append("## Recommendations\n\n")

        if analysis.dimension_performance:
            best_dim = min(analysis.dimension_performance.items(), key=lambda x: x[1])
            worst_dim = max(analysis.dimension_performance.items(), key=lambda x: x[1])
            report_content.append(
                f"- **Optimal Vector Dimension**: {best_dim[0]}D shows best performance ({best_dim[1]:.2f}ms avg query time)\n"
            )
            report_content.append(
                f"- **Performance Impact**: {worst_dim[0]}D takes {worst_dim[1] / best_dim[1]:.1f}x longer than {best_dim[0]}D\n"
            )

        if analysis.search_type_comparison:
            best_search = min(
                analysis.search_type_comparison.items(), key=lambda x: x[1]
            )
            report_content.append(
                f"- **Recommended Search Type**: {best_search[0]} provides best performance ({best_search[1]:.2f}ms avg query time)\n"
            )

        if analysis.filter_impact and "impact_ratio" in analysis.filter_impact:
            impact_ratio = analysis.filter_impact["impact_ratio"]
            if impact_ratio > 1.5:
                report_content.append(
                    f"- **Filter Optimization Needed**: Filters add {impact_ratio:.1f}x overhead to query time\n"
                )
            else:
                report_content.append(
                    f"- **Filter Performance**: Acceptable {impact_ratio:.1f}x overhead from filtering\n"
                )

        report_content.append("\n")

        report_path = analysis_dir / "comprehensive_analysis_report.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.writelines(report_content)

        return str(report_path)

    class CreateReportIO(IO[str]):
        def run(self) -> str:
            return _create_report()

    return CreateReportIO()


def analysis_pipeline(checkpoint_dir: str, output_dir: str = "analysis") -> IO[str]:
    """Complete analysis pipeline from checkpoint data to final report.

    Args:
        checkpoint_dir: Directory containing experiment checkpoint files
        output_dir: Directory to save analysis outputs

    Returns:
        IO[str]: Path to the final comprehensive analysis report
    """

    def _run_pipeline() -> str:
        load_io = load_experiment_results(checkpoint_dir)
        results = load_io.run()

        if not results:
            raise ValueError(f"No experiment results found in {checkpoint_dir}")

        analysis = analyze_dimension_performance(results)
        trends = calculate_performance_trends(results)

        viz_io = generate_visualizations(analysis, trends)
        visualization_paths = viz_io.run()

        report_io = create_analysis_report(analysis, trends, visualization_paths)
        final_report_path = report_io.run()

        return final_report_path

    class AnalysisPipelineIO(IO[str]):
        def run(self) -> str:
            return _run_pipeline()

    return AnalysisPipelineIO()


def quick_analysis_pipeline(results: List[ExperimentResult]) -> IO[PerformanceAnalysis]:
    """Quick analysis pipeline for already loaded results.

    Args:
        results: List of experiment results to analyze

    Returns:
        IO[PerformanceAnalysis]: Performance analysis results
    """

    def _quick_analysis() -> PerformanceAnalysis:
        return analyze_dimension_performance(results)

    class QuickAnalysisIO(IO[PerformanceAnalysis]):
        def run(self) -> PerformanceAnalysis:
            return _quick_analysis()

    return QuickAnalysisIO()
