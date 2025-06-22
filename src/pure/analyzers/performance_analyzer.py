"""Performance analysis functions for DuckDB VSS benchmarking results.

This module contains pure functions for analyzing experiment results.
All functions are side-effect free and follow functional programming principles.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import statistics

from src.types.core import (
    ExperimentResult,
)


@dataclass(frozen=True)
class PerformanceAnalysis:
    """Comprehensive performance analysis results."""

    dimension_performance: Dict[int, float]
    scale_performance: Dict[str, float]
    search_type_comparison: Dict[str, float]
    filter_impact: Dict[str, float]


@dataclass(frozen=True)
class TrendAnalysis:
    """Performance trend analysis over time and scale."""

    temporal_trends: Dict[str, List[Tuple[datetime, float]]]
    scale_trends: Dict[str, List[Tuple[int, float]]]
    dimension_trends: Dict[str, List[Tuple[int, float]]]


@dataclass(frozen=True)
class StatisticalSummary:
    """Statistical summary of performance metrics."""

    mean: float
    median: float
    std_dev: float
    min_value: float
    max_value: float
    percentile_5: float
    percentile_10: float
    percentile_90: float
    percentile_95: float


def analyze_dimension_performance(
    results: List[ExperimentResult],
) -> PerformanceAnalysis:
    """Analyze performance characteristics by vector dimension.

    Args:
        results: List of experiment results to analyze

    Returns:
        PerformanceAnalysis containing dimension-based performance metrics
    """
    if not results:
        return PerformanceAnalysis(
            dimension_performance={},
            scale_performance={},
            search_type_comparison={},
            filter_impact={},
        )

    dimension_groups: Dict[int, List[ExperimentResult]] = {}
    for result in results:
        dim = int(result.config.dimension)
        if dim not in dimension_groups:
            dimension_groups[dim] = []
        dimension_groups[dim].append(result)

    dimension_performance = {}
    for dim, dim_results in dimension_groups.items():
        query_times = []
        for result in dim_results:
            for search_result in result.search_results:
                query_times.append(search_result.metrics.query_time_ms)

        if query_times:
            dimension_performance[dim] = statistics.mean(query_times)

    scale_groups: Dict[str, List[ExperimentResult]] = {}
    for result in results:
        scale_name = result.config.data_scale.name
        if scale_name not in scale_groups:
            scale_groups[scale_name] = []
        scale_groups[scale_name].append(result)

    scale_performance = {}
    for scale_name, scale_results in scale_groups.items():
        throughputs = []
        for result in scale_results:
            for search_result in result.search_results:
                throughputs.append(search_result.metrics.throughput_qps)

        if throughputs:
            scale_performance[scale_name] = statistics.mean(throughputs)

    search_type_groups: Dict[str, List[ExperimentResult]] = {}
    for result in results:
        search_type = result.config.search_type.value
        if search_type not in search_type_groups:
            search_type_groups[search_type] = []
        search_type_groups[search_type].append(result)

    search_type_comparison = {}
    for search_type, type_results in search_type_groups.items():
        query_times = []
        for result in type_results:
            for search_result in result.search_results:
                query_times.append(search_result.metrics.query_time_ms)

        if query_times:
            search_type_comparison[search_type] = statistics.mean(query_times)

    filtered_results = [r for r in results if r.config.filter_config.enabled]
    unfiltered_results = [r for r in results if not r.config.filter_config.enabled]

    filter_impact = {}
    if filtered_results and unfiltered_results:
        filtered_times = []
        for result in filtered_results:
            for search_result in result.search_results:
                filtered_times.append(search_result.metrics.query_time_ms)

        unfiltered_times = []
        for result in unfiltered_results:
            for search_result in result.search_results:
                unfiltered_times.append(search_result.metrics.query_time_ms)

        if filtered_times and unfiltered_times:
            filter_impact = {
                "with_filter": statistics.mean(filtered_times),
                "without_filter": statistics.mean(unfiltered_times),
                "impact_ratio": statistics.mean(filtered_times)
                / statistics.mean(unfiltered_times),
            }

    return PerformanceAnalysis(
        dimension_performance=dimension_performance,
        scale_performance=scale_performance,
        search_type_comparison=search_type_comparison,
        filter_impact=filter_impact,
    )


def analyze_search_type_performance(
    results: List[ExperimentResult],
) -> Dict[str, float]:
    """Analyze and compare performance between different search types.

    Args:
        results: List of experiment results to analyze

    Returns:
        Dictionary mapping search type to average query time
    """
    if not results:
        return {}

    search_type_groups: Dict[str, List[float]] = {}

    for result in results:
        search_type = result.config.search_type.value
        if search_type not in search_type_groups:
            search_type_groups[search_type] = []

        for search_result in result.search_results:
            search_type_groups[search_type].append(search_result.metrics.query_time_ms)

    performance_comparison = {}
    for search_type, query_times in search_type_groups.items():
        if query_times:
            performance_comparison[search_type] = statistics.mean(query_times)

    return performance_comparison


def calculate_performance_trends(results: List[ExperimentResult]) -> TrendAnalysis:
    """Calculate performance trends over time and scale.

    Args:
        results: List of experiment results to analyze

    Returns:
        TrendAnalysis containing temporal and scale-based trends
    """
    if not results:
        return TrendAnalysis(temporal_trends={}, scale_trends={}, dimension_trends={})

    sorted_results = sorted(results, key=lambda r: r.timestamp)

    temporal_trends = {}
    query_time_trend = []
    throughput_trend = []

    for result in sorted_results:
        timestamp = result.timestamp

        query_times = [sr.metrics.query_time_ms for sr in result.search_results]
        if query_times:
            avg_query_time = statistics.mean(query_times)
            query_time_trend.append((timestamp, avg_query_time))

        throughputs = [sr.metrics.throughput_qps for sr in result.search_results]
        if throughputs:
            avg_throughput = statistics.mean(throughputs)
            throughput_trend.append((timestamp, avg_throughput))

    temporal_trends["query_time_ms"] = query_time_trend
    temporal_trends["throughput_qps"] = throughput_trend

    scale_trends = {}
    scale_query_times = []
    scale_throughputs = []

    for result in sorted_results:
        scale_value = result.config.data_scale.value

        query_times = [sr.metrics.query_time_ms for sr in result.search_results]
        if query_times:
            avg_query_time = statistics.mean(query_times)
            scale_query_times.append((scale_value, avg_query_time))

        throughputs = [sr.metrics.throughput_qps for sr in result.search_results]
        if throughputs:
            avg_throughput = statistics.mean(throughputs)
            scale_throughputs.append((scale_value, avg_throughput))

    scale_trends["query_time_ms"] = scale_query_times
    scale_trends["throughput_qps"] = scale_throughputs

    dimension_trends = {}
    dim_query_times = []
    dim_throughputs = []

    for result in sorted_results:
        dimension = int(result.config.dimension)

        query_times = [sr.metrics.query_time_ms for sr in result.search_results]
        if query_times:
            avg_query_time = statistics.mean(query_times)
            dim_query_times.append((dimension, avg_query_time))

        throughputs = [sr.metrics.throughput_qps for sr in result.search_results]
        if throughputs:
            avg_throughput = statistics.mean(throughputs)
            dim_throughputs.append((dimension, avg_throughput))

    dimension_trends["query_time_ms"] = dim_query_times
    dimension_trends["throughput_qps"] = dim_throughputs

    return TrendAnalysis(
        temporal_trends=temporal_trends,
        scale_trends=scale_trends,
        dimension_trends=dimension_trends,
    )


def calculate_statistical_summary(values: List[float]) -> Optional[StatisticalSummary]:
    """Calculate statistical summary for a list of values.

    Args:
        values: List of numeric values to analyze

    Returns:
        StatisticalSummary or None if values is empty
    """
    if not values:
        return None

    sorted_values = sorted(values)

    return StatisticalSummary(
        mean=statistics.mean(values),
        median=statistics.median(values),
        std_dev=statistics.stdev(values) if len(values) > 1 else 0.0,
        min_value=min(values),
        max_value=max(values),
        percentile_5=sorted_values[int(0.05 * len(sorted_values))],
        percentile_10=sorted_values[int(0.10 * len(sorted_values))],
        percentile_90=sorted_values[int(0.90 * len(sorted_values))],
        percentile_95=sorted_values[int(0.95 * len(sorted_values))],
    )


def compare_accuracy_metrics(
    results: List[ExperimentResult],
) -> Dict[str, Dict[str, float]]:
    """Compare accuracy metrics across different experiment configurations.

    Args:
        results: List of experiment results to analyze

    Returns:
        Dictionary mapping configuration type to accuracy metrics
    """
    if not results:
        return {}

    accuracy_comparison = {}

    search_type_accuracy: Dict[str, Dict[str, List[float]]] = {}
    for result in results:
        search_type = result.config.search_type.value
        if search_type not in search_type_accuracy:
            search_type_accuracy[search_type] = {
                "recall_at_1": [],
                "recall_at_5": [],
                "recall_at_10": [],
                "mrr": [],
            }

        search_type_accuracy[search_type]["recall_at_1"].append(
            result.accuracy.recall_at_1
        )
        search_type_accuracy[search_type]["recall_at_5"].append(
            result.accuracy.recall_at_5
        )
        search_type_accuracy[search_type]["recall_at_10"].append(
            result.accuracy.recall_at_10
        )
        search_type_accuracy[search_type]["mrr"].append(
            result.accuracy.mean_reciprocal_rank
        )

    for search_type, metrics in search_type_accuracy.items():
        accuracy_comparison[search_type] = {
            "avg_recall_at_1": statistics.mean(metrics["recall_at_1"])
            if metrics["recall_at_1"]
            else 0.0,
            "avg_recall_at_5": statistics.mean(metrics["recall_at_5"])
            if metrics["recall_at_5"]
            else 0.0,
            "avg_recall_at_10": statistics.mean(metrics["recall_at_10"])
            if metrics["recall_at_10"]
            else 0.0,
            "avg_mrr": statistics.mean(metrics["mrr"]) if metrics["mrr"] else 0.0,
        }

    return accuracy_comparison
