"""Chart generation effects for DuckDB VSS benchmarking analysis.

This module contains IO-wrapped visualization functions that generate charts
and interactive dashboards from analysis results.
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional
import json
import os
from pathlib import Path

from src.types.monads import IO
from src.pure.analyzers.performance_analyzer import (
    PerformanceAnalysis,
    TrendAnalysis,
    StatisticalSummary
)


def generate_performance_heatmap(
    analysis: PerformanceAnalysis,
    output_dir: str = "analysis/charts"
) -> IO[str]:
    """Generate performance heatmap visualization.
    
    Args:
        analysis: Performance analysis results
        output_dir: Directory to save the chart
        
    Returns:
        IO[str]: Path to the generated heatmap file
    """
    def _generate_heatmap() -> str:
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            import pandas as pd
            import numpy as np
        except ImportError:
            raise ImportError("matplotlib, seaborn, and pandas are required for chart generation")
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        heatmap_data = []
        
        for dim, perf in analysis.dimension_performance.items():
            heatmap_data.append({
                'Metric': 'Dimension Performance',
                'Category': f'{dim}D',
                'Value': perf
            })
        
        for scale, perf in analysis.scale_performance.items():
            heatmap_data.append({
                'Metric': 'Scale Performance',
                'Category': scale,
                'Value': perf
            })
        
        for search_type, perf in analysis.search_type_comparison.items():
            heatmap_data.append({
                'Metric': 'Search Type',
                'Category': search_type,
                'Value': perf
            })
        
        if not heatmap_data:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'No data available for heatmap', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Performance Heatmap - No Data')
        else:
            df = pd.DataFrame(heatmap_data)
            pivot_df = df.pivot(index='Metric', columns='Category', values='Value')
            
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(pivot_df, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax)
            ax.set_title('DuckDB VSS Performance Heatmap')
            ax.set_xlabel('Configuration Category')
            ax.set_ylabel('Performance Metric')
        
        output_path = os.path.join(output_dir, 'performance_heatmap.png')
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    class GenerateHeatmapIO(IO[str]):
        def run(self) -> str:
            return _generate_heatmap()
    
    return GenerateHeatmapIO()


def create_trend_charts(
    trends: TrendAnalysis,
    output_dir: str = "analysis/charts"
) -> IO[List[str]]:
    """Create trend analysis charts.
    
    Args:
        trends: Trend analysis results
        output_dir: Directory to save the charts
        
    Returns:
        IO[List[str]]: List of paths to generated chart files
    """
    def _create_trends() -> List[str]:
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
        except ImportError:
            raise ImportError("matplotlib is required for chart generation")
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        chart_paths = []
        
        if trends.temporal_trends:
            fig, axes = plt.subplots(2, 1, figsize=(12, 10))
            
            if 'query_time_ms' in trends.temporal_trends:
                temporal_data = trends.temporal_trends['query_time_ms']
                if temporal_data:
                    timestamps, values = zip(*temporal_data)
                    axes[0].plot(timestamps, values, marker='o', linewidth=2)
                    axes[0].set_title('Query Time Trend Over Time')
                    axes[0].set_ylabel('Query Time (ms)')
                    axes[0].grid(True, alpha=0.3)
                    axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
                    plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45)
            
            if 'throughput_qps' in trends.temporal_trends:
                temporal_data = trends.temporal_trends['throughput_qps']
                if temporal_data:
                    timestamps, values = zip(*temporal_data)
                    axes[1].plot(timestamps, values, marker='s', linewidth=2, color='green')
                    axes[1].set_title('Throughput Trend Over Time')
                    axes[1].set_ylabel('Throughput (QPS)')
                    axes[1].set_xlabel('Time')
                    axes[1].grid(True, alpha=0.3)
                    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
                    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45)
            
            plt.tight_layout()
            temporal_path = os.path.join(output_dir, 'temporal_trends.png')
            plt.savefig(temporal_path, dpi=300, bbox_inches='tight')
            plt.close()
            chart_paths.append(temporal_path)
        
        if trends.scale_trends:
            fig, axes = plt.subplots(2, 1, figsize=(12, 10))
            
            if 'query_time_ms' in trends.scale_trends:
                scale_data = trends.scale_trends['query_time_ms']
                if scale_data:
                    scales, values = zip(*scale_data)
                    axes[0].scatter(scales, values, alpha=0.6, s=50)
                    axes[0].set_title('Query Time vs Data Scale')
                    axes[0].set_ylabel('Query Time (ms)')
                    axes[0].set_xscale('log')
                    axes[0].grid(True, alpha=0.3)
            
            if 'throughput_qps' in trends.scale_trends:
                scale_data = trends.scale_trends['throughput_qps']
                if scale_data:
                    scales, values = zip(*scale_data)
                    axes[1].scatter(scales, values, alpha=0.6, s=50, color='green')
                    axes[1].set_title('Throughput vs Data Scale')
                    axes[1].set_ylabel('Throughput (QPS)')
                    axes[1].set_xlabel('Data Scale (number of vectors)')
                    axes[1].set_xscale('log')
                    axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            scale_path = os.path.join(output_dir, 'scale_trends.png')
            plt.savefig(scale_path, dpi=300, bbox_inches='tight')
            plt.close()
            chart_paths.append(scale_path)
        
        if trends.dimension_trends:
            fig, axes = plt.subplots(2, 1, figsize=(12, 10))
            
            if 'query_time_ms' in trends.dimension_trends:
                dim_data = trends.dimension_trends['query_time_ms']
                if dim_data:
                    dimensions, values = zip(*dim_data)
                    axes[0].scatter(dimensions, values, alpha=0.6, s=50)
                    axes[0].set_title('Query Time vs Vector Dimension')
                    axes[0].set_ylabel('Query Time (ms)')
                    axes[0].grid(True, alpha=0.3)
            
            if 'throughput_qps' in trends.dimension_trends:
                dim_data = trends.dimension_trends['throughput_qps']
                if dim_data:
                    dimensions, values = zip(*dim_data)
                    axes[1].scatter(dimensions, values, alpha=0.6, s=50, color='green')
                    axes[1].set_title('Throughput vs Vector Dimension')
                    axes[1].set_ylabel('Throughput (QPS)')
                    axes[1].set_xlabel('Vector Dimension')
                    axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            dimension_path = os.path.join(output_dir, 'dimension_trends.png')
            plt.savefig(dimension_path, dpi=300, bbox_inches='tight')
            plt.close()
            chart_paths.append(dimension_path)
        
        return chart_paths
    
    class CreateTrendsIO(IO[List[str]]):
        def run(self) -> List[str]:
            return _create_trends()
    
    return CreateTrendsIO()


def export_interactive_dashboard(
    analysis: PerformanceAnalysis,
    trends: Optional[TrendAnalysis] = None,
    output_dir: str = "analysis/dashboard"
) -> IO[str]:
    """Export interactive Plotly dashboard.
    
    Args:
        analysis: Performance analysis results
        trends: Optional trend analysis results
        output_dir: Directory to save the dashboard
        
    Returns:
        IO[str]: Path to the generated HTML dashboard file
    """
    def _export_dashboard() -> str:
        try:
            import plotly.graph_objects as go
            import plotly.express as px
            from plotly.subplots import make_subplots
            import plotly.offline as pyo
        except ImportError:
            raise ImportError("plotly is required for interactive dashboard generation")
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Dimension Performance',
                'Search Type Comparison',
                'Scale Performance',
                'Filter Impact'
            ),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        if analysis.dimension_performance:
            dimensions = list(analysis.dimension_performance.keys())
            values = list(analysis.dimension_performance.values())
            fig.add_trace(
                go.Bar(x=[f"{d}D" for d in dimensions], y=values, name="Dimension"),
                row=1, col=1
            )
        
        if analysis.search_type_comparison:
            search_types = list(analysis.search_type_comparison.keys())
            values = list(analysis.search_type_comparison.values())
            fig.add_trace(
                go.Bar(x=search_types, y=values, name="Search Type"),
                row=1, col=2
            )
        
        if analysis.scale_performance:
            scales = list(analysis.scale_performance.keys())
            values = list(analysis.scale_performance.values())
            fig.add_trace(
                go.Bar(x=scales, y=values, name="Scale"),
                row=2, col=1
            )
        
        if analysis.filter_impact:
            filter_keys = list(analysis.filter_impact.keys())
            filter_values = list(analysis.filter_impact.values())
            fig.add_trace(
                go.Bar(x=filter_keys, y=filter_values, name="Filter"),
                row=2, col=2
            )
        
        fig.update_layout(
            title_text="DuckDB VSS Performance Analysis Dashboard",
            showlegend=False,
            height=800
        )
        
        if trends and trends.temporal_trends:
            trend_fig = go.Figure()
            
            if 'query_time_ms' in trends.temporal_trends:
                temporal_data = trends.temporal_trends['query_time_ms']
                if temporal_data:
                    timestamps, values = zip(*temporal_data)
                    trend_fig.add_trace(
                        go.Scatter(
                            x=timestamps, y=values,
                            mode='lines+markers',
                            name='Query Time Trend'
                        )
                    )
            
            trend_fig.update_layout(
                title="Performance Trends Over Time",
                xaxis_title="Time",
                yaxis_title="Query Time (ms)"
            )
        
        output_path = os.path.join(output_dir, 'interactive_dashboard.html')
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>DuckDB VSS Performance Dashboard</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .chart-container {{ margin: 20px 0; }}
                h1 {{ color: #333; text-align: center; }}
                h2 {{ color: #666; }}
            </style>
        </head>
        <body>
            <h1>DuckDB VSS Performance Analysis Dashboard</h1>
            
            <div class="chart-container">
                <h2>Performance Overview</h2>
                <div id="main-chart"></div>
            </div>
            
            {"<div class='chart-container'><h2>Temporal Trends</h2><div id='trend-chart'></div></div>" if trends and trends.temporal_trends else ""}
            
            <script>
                var mainChart = {fig.to_json()};
                Plotly.newPlot('main-chart', mainChart.data, mainChart.layout);
                
                {"var trendChart = " + trend_fig.to_json() + "; Plotly.newPlot('trend-chart', trendChart.data, trendChart.layout);" if trends and trends.temporal_trends else ""}
            </script>
        </body>
        </html>
        """
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return output_path
    
    class ExportDashboardIO(IO[str]):
        def run(self) -> str:
            return _export_dashboard()
    
    return ExportDashboardIO()


def generate_summary_report(
    analysis: PerformanceAnalysis,
    trends: Optional[TrendAnalysis] = None,
    output_dir: str = "analysis/reports"
) -> IO[str]:
    """Generate a comprehensive summary report.
    
    Args:
        analysis: Performance analysis results
        trends: Optional trend analysis results
        output_dir: Directory to save the report
        
    Returns:
        IO[str]: Path to the generated report file
    """
    def _generate_report() -> str:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        report_content = []
        report_content.append("# DuckDB VSS Performance Analysis Report\n")
        from datetime import datetime
        report_content.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        report_content.append("## Performance Analysis Summary\n\n")
        
        if analysis.dimension_performance:
            report_content.append("### Dimension Performance\n")
            for dim, perf in sorted(analysis.dimension_performance.items()):
                report_content.append(f"- {dim}D vectors: {perf:.2f}ms average query time\n")
            report_content.append("\n")
        
        if analysis.scale_performance:
            report_content.append("### Scale Performance\n")
            for scale, perf in analysis.scale_performance.items():
                report_content.append(f"- {scale}: {perf:.2f} QPS average throughput\n")
            report_content.append("\n")
        
        if analysis.search_type_comparison:
            report_content.append("### Search Type Comparison\n")
            for search_type, perf in analysis.search_type_comparison.items():
                report_content.append(f"- {search_type}: {perf:.2f}ms average query time\n")
            report_content.append("\n")
        
        if analysis.filter_impact:
            report_content.append("### Filter Impact Analysis\n")
            for filter_key, impact in analysis.filter_impact.items():
                if isinstance(impact, (int, float)):
                    report_content.append(f"- {filter_key}: {impact:.2f}ms\n")
                else:
                    report_content.append(f"- {filter_key}: {impact}\n")
            report_content.append("\n")
        
        if trends:
            report_content.append("## Trend Analysis\n\n")
            
            if trends.temporal_trends:
                report_content.append("### Temporal Trends\n")
                for metric, data_points in trends.temporal_trends.items():
                    report_content.append(f"- {metric}: {len(data_points)} data points collected\n")
                report_content.append("\n")
            
            if trends.scale_trends:
                report_content.append("### Scale Trends\n")
                for metric, data_points in trends.scale_trends.items():
                    report_content.append(f"- {metric}: {len(data_points)} scale measurements\n")
                report_content.append("\n")
        
        output_path = os.path.join(output_dir, 'performance_summary.md')
        with open(output_path, 'w', encoding='utf-8') as f:
            f.writelines(report_content)
        
        return output_path
    
    class GenerateReportIO(IO[str]):
        def run(self) -> str:
            return _generate_report()
    
    return GenerateReportIO()
