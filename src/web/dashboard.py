"""FastAPI + WebSocket based web dashboard for DuckDB VSS analysis.

This module provides a real-time web dashboard for monitoring and visualizing
DuckDB VSS benchmarking results with interactive charts and live updates.
"""

from __future__ import annotations
from typing import List, Dict, Any
import json
from datetime import datetime

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException

    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.middleware.cors import CORSMiddleware

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

from src.types.core import ExperimentResult
from src.types.monads import IO
from src.pure.analyzers.performance_analyzer import (
    analyze_dimension_performance,
    calculate_performance_trends,
    analyze_search_type_performance,
    compare_accuracy_metrics,
)


class ConnectionManager:
    """Manages WebSocket connections for real-time updates."""

    def __init__(self) -> None:
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket) -> None:
        await websocket.send_text(message)

    async def broadcast(self, message: str) -> None:
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception:
                self.disconnect(connection)


def create_dashboard_app(results: List[ExperimentResult]) -> FastAPI:
    """Create FastAPI application for the dashboard.

    Args:
        results: List of experiment results to display

    Returns:
        FastAPI application instance
    """
    if not FASTAPI_AVAILABLE:
        raise ImportError(
            "FastAPI is required for web dashboard. Install with: pip install fastapi uvicorn"
        )

    app = FastAPI(
        title="DuckDB VSS Performance Dashboard",
        description="Real-time dashboard for DuckDB Vector Similarity Search benchmarking",
        version="1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    manager = ConnectionManager()

    analysis_cache: Dict[str, Any] = {}

    def get_analysis_data() -> Dict[str, Any]:
        """Get cached analysis data or compute it."""
        if "performance_analysis" not in analysis_cache:
            performance_analysis = analyze_dimension_performance(results)
            trend_analysis = calculate_performance_trends(results)
            search_comparison = analyze_search_type_performance(results)
            accuracy_comparison = compare_accuracy_metrics(results)

            analysis_cache["performance_analysis"] = performance_analysis
            analysis_cache["trend_analysis"] = trend_analysis
            analysis_cache["search_comparison"] = search_comparison
            analysis_cache["accuracy_comparison"] = accuracy_comparison
            analysis_cache["last_updated"] = datetime.now().isoformat()

        return analysis_cache

    @app.get("/", response_class=HTMLResponse)
    async def dashboard_home() -> HTMLResponse:
        """Serve the main dashboard HTML page."""
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>DuckDB VSS Performance Dashboard</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            <style>
                body {
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background-color: #f5f5f5;
                }
                .header {
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 20px;
                    border-radius: 10px;
                    margin-bottom: 20px;
                    text-align: center;
                }
                .dashboard-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
                    gap: 20px;
                    margin-bottom: 20px;
                }
                .chart-container {
                    background: white;
                    padding: 20px;
                    border-radius: 10px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }
                .chart-title {
                    font-size: 18px;
                    font-weight: bold;
                    margin-bottom: 15px;
                    color: #333;
                }
                .status-indicator {
                    display: inline-block;
                    width: 10px;
                    height: 10px;
                    border-radius: 50%;
                    margin-right: 8px;
                }
                .status-live { background-color: #4CAF50; }
                .status-offline { background-color: #f44336; }
                .metrics-summary {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 15px;
                    margin-bottom: 20px;
                }
                .metric-card {
                    background: white;
                    padding: 15px;
                    border-radius: 8px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                    text-align: center;
                }
                .metric-value {
                    font-size: 24px;
                    font-weight: bold;
                    color: #667eea;
                }
                .metric-label {
                    font-size: 14px;
                    color: #666;
                    margin-top: 5px;
                }
                    position: fixed;
                    top: 20px;
                    right: 20px;
                    padding: 10px 15px;
                    border-radius: 5px;
                    color: white;
                    font-weight: bold;
                }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🚀 DuckDB VSS Performance Dashboard</h1>
                <p>Real-time monitoring and analysis of Vector Similarity Search benchmarks</p>
                <div id="connectionStatus">
                    <span class="status-indicator status-offline"></span>
                    Connecting...
                </div>
            </div>

            <div class="metrics-summary" id="metricsSummary">
                <!-- Metrics will be populated by JavaScript -->
            </div>

            <div class="dashboard-grid">
                <div class="chart-container">
                    <div class="chart-title">📊 Dimension Performance</div>
                    <div id="dimensionChart"></div>
                </div>

                <div class="chart-container">
                    <div class="chart-title">🔍 Search Type Comparison</div>
                    <div id="searchTypeChart"></div>
                </div>

                <div class="chart-container">
                    <div class="chart-title">📈 Performance Trends</div>
                    <div id="trendsChart"></div>
                </div>

                <div class="chart-container">
                    <div class="chart-title">🎯 Accuracy Metrics</div>
                    <div id="accuracyChart"></div>
                </div>
            </div>

            <script>
                // WebSocket connection for real-time updates
                const ws = new WebSocket(`ws://${window.location.host}/ws/live-analysis`);
                const connectionStatus = document.getElementById('connectionStatus');

                ws.onopen = function(event) {
                    connectionStatus.innerHTML = '<span class="status-indicator status-live"></span>Live';
                    connectionStatus.style.backgroundColor = '#4CAF50';
                    loadInitialData();
                };

                ws.onclose = function(event) {
                    connectionStatus.innerHTML = '<span class="status-indicator status-offline"></span>Offline';
                    connectionStatus.style.backgroundColor = '#f44336';
                };

                ws.onmessage = function(event) {
                    const data = JSON.parse(event.data);
                    updateDashboard(data);
                };

                async function loadInitialData() {
                    try {
                        const response = await fetch('/api/analysis/summary');
                        const data = await response.json();
                        updateDashboard(data);
                    } catch (error) {
                        console.error('Failed to load initial data:', error);
                    }
                }

                function updateDashboard(data) {
                    updateMetricsSummary(data);
                    updateCharts(data);
                }

                function updateMetricsSummary(data) {
                    const summary = document.getElementById('metricsSummary');
                    const totalExperiments = data.total_experiments || 0;
                    const avgQueryTime = data.avg_query_time || 0;
                    const avgThroughput = data.avg_throughput || 0;
                    const bestDimension = data.best_dimension || 'N/A';

                    summary.innerHTML = `
                        <div class="metric-card">
                            <div class="metric-value">${totalExperiments}</div>
                            <div class="metric-label">Total Experiments</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">${avgQueryTime.toFixed(2)}ms</div>
                            <div class="metric-label">Avg Query Time</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">${avgThroughput.toFixed(1)}</div>
                            <div class="metric-label">Avg Throughput (QPS)</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">${bestDimension}</div>
                            <div class="metric-label">Best Dimension</div>
                        </div>
                    `;
                }

                function updateCharts(data) {
                    // Dimension Performance Chart
                    if (data.dimension_performance) {
                        const dimensions = Object.keys(data.dimension_performance);
                        const values = Object.values(data.dimension_performance);

                        const dimensionTrace = {
                            x: dimensions.map(d => d + 'D'),
                            y: values,
                            type: 'bar',
                            marker: { color: '#667eea' }
                        };

                        Plotly.newPlot('dimensionChart', [dimensionTrace], {
                            title: 'Query Time by Vector Dimension',
                            xaxis: { title: 'Vector Dimension' },
                            yaxis: { title: 'Query Time (ms)' }
                        });
                    }

                    // Search Type Comparison Chart
                    if (data.search_comparison) {
                        const searchTypes = Object.keys(data.search_comparison);
                        const searchValues = Object.values(data.search_comparison);

                        const searchTrace = {
                            x: searchTypes,
                            y: searchValues,
                            type: 'bar',
                            marker: { color: '#764ba2' }
                        };

                        Plotly.newPlot('searchTypeChart', [searchTrace], {
                            title: 'Performance by Search Type',
                            xaxis: { title: 'Search Type' },
                            yaxis: { title: 'Query Time (ms)' }
                        });
                    }

                    // Accuracy Metrics Chart
                    if (data.accuracy_comparison) {
                        const accuracyData = [];
                        for (const [searchType, metrics] of Object.entries(data.accuracy_comparison)) {
                            accuracyData.push({
                                x: ['Recall@1', 'Recall@5', 'Recall@10', 'MRR'],
                                y: [
                                    metrics.avg_recall_at_1,
                                    metrics.avg_recall_at_5,
                                    metrics.avg_recall_at_10,
                                    metrics.avg_mrr
                                ],
                                name: searchType,
                                type: 'bar'
                            });
                        }

                        Plotly.newPlot('accuracyChart', accuracyData, {
                            title: 'Accuracy Metrics by Search Type',
                            xaxis: { title: 'Metric' },
                            yaxis: { title: 'Score' },
                            barmode: 'group'
                        });
                    }
                }

                // Auto-refresh every 30 seconds
                setInterval(() => {
                    if (ws.readyState === WebSocket.OPEN) {
                        ws.send(JSON.stringify({type: 'refresh_request'}));
                    }
                }, 30000);
            </script>
        </body>
        </html>
        """
        return HTMLResponse(content=html_content)

    @app.get("/api/analysis/summary")
    async def get_analysis_summary() -> JSONResponse:
        """Get analysis summary data."""
        try:
            analysis_data = get_analysis_data()

            total_experiments = len(results)

            all_query_times = []
            all_throughputs = []
            for result in results:
                for search_result in result.search_results:
                    all_query_times.append(search_result.metrics.query_time_ms)
                    all_throughputs.append(search_result.metrics.throughput_qps)

            avg_query_time = (
                sum(all_query_times) / len(all_query_times) if all_query_times else 0
            )
            avg_throughput = (
                sum(all_throughputs) / len(all_throughputs) if all_throughputs else 0
            )

            performance_analysis = analysis_data["performance_analysis"]
            best_dimension = None
            if performance_analysis.dimension_performance:
                best_dimension = min(
                    performance_analysis.dimension_performance.items(),
                    key=lambda x: x[1],
                )[0]
                best_dimension = f"{best_dimension}D"

            return JSONResponse(
                {
                    "total_experiments": total_experiments,
                    "avg_query_time": avg_query_time,
                    "avg_throughput": avg_throughput,
                    "best_dimension": best_dimension,
                    "dimension_performance": performance_analysis.dimension_performance,
                    "search_comparison": analysis_data["search_comparison"],
                    "accuracy_comparison": analysis_data["accuracy_comparison"],
                    "last_updated": analysis_data["last_updated"],
                }
            )

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/analysis/{experiment_id}")
    async def get_analysis(experiment_id: str) -> JSONResponse:
        """Get analysis results for a specific experiment."""
        try:
            experiment = None
            for result in results:
                if str(hash(str(result.config))) == experiment_id:
                    experiment = result
                    break

            if not experiment:
                raise HTTPException(status_code=404, detail="Experiment not found")

            return JSONResponse(
                {
                    "experiment_id": experiment_id,
                    "config": {
                        "data_scale": experiment.config.data_scale.name,
                        "dimension": int(experiment.config.dimension),
                        "search_type": experiment.config.search_type.value,
                        "filter_enabled": experiment.config.filter_config.enabled,
                    },
                    "metrics": {
                        "insert_time_ms": experiment.insert_metrics.query_time_ms,
                        "index_build_time_ms": experiment.index_metrics.query_time_ms,
                        "avg_query_time_ms": sum(
                            sr.metrics.query_time_ms for sr in experiment.search_results
                        )
                        / len(experiment.search_results),
                        "avg_throughput_qps": sum(
                            sr.metrics.throughput_qps
                            for sr in experiment.search_results
                        )
                        / len(experiment.search_results),
                        "memory_usage_mb": experiment.search_results[
                            0
                        ].metrics.memory_usage_mb
                        if experiment.search_results
                        else 0,
                    },
                    "accuracy": {
                        "recall_at_1": experiment.accuracy.recall_at_1,
                        "recall_at_5": experiment.accuracy.recall_at_5,
                        "recall_at_10": experiment.accuracy.recall_at_10,
                        "mrr": experiment.accuracy.mean_reciprocal_rank,
                    },
                    "timestamp": experiment.timestamp.isoformat(),
                }
            )

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.websocket("/ws/live-analysis")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        """WebSocket endpoint for real-time analysis updates."""
        await manager.connect(websocket)
        try:
            while True:
                data = await websocket.receive_text()
                message = json.loads(data)

                if message.get("type") == "refresh_request":
                    analysis_data = get_analysis_data()

                    response = {
                        "type": "analysis_update",
                        "data": {
                            "dimension_performance": analysis_data[
                                "performance_analysis"
                            ].dimension_performance,
                            "search_comparison": analysis_data["search_comparison"],
                            "accuracy_comparison": analysis_data["accuracy_comparison"],
                            "total_experiments": len(results),
                            "timestamp": datetime.now().isoformat(),
                        },
                    }

                    await manager.send_personal_message(json.dumps(response), websocket)

        except WebSocketDisconnect:
            manager.disconnect(websocket)
        except Exception as e:
            print(f"WebSocket error: {e}")
            manager.disconnect(websocket)

    @app.get("/health")
    async def health_check() -> JSONResponse:
        """Health check endpoint."""
        return JSONResponse(
            {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "experiments_loaded": len(results),
            }
        )

    return app


def start_dashboard_server(
    results: List[ExperimentResult], host: str = "0.0.0.0", port: int = 8080
) -> IO[None]:
    """Start the dashboard server.

    Args:
        results: List of experiment results to display
        host: Host to bind the server to
        port: Port to bind the server to

    Returns:
        IO[None]: IO action to start the server
    """

    def _start_server() -> None:
        if not FASTAPI_AVAILABLE:
            raise ImportError("FastAPI and uvicorn are required for web dashboard")

        try:
            import uvicorn
        except ImportError:
            raise ImportError(
                "uvicorn is required for web dashboard. Install with: pip install uvicorn"
            )

        app = create_dashboard_app(results)

        print("🚀 Starting DuckDB VSS Dashboard...")
        print(f"📊 Dashboard URL: http://{host}:{port}")
        print(f"🔗 API Documentation: http://{host}:{port}/docs")
        print(f"❤️  Health Check: http://{host}:{port}/health")
        print("Press Ctrl+C to stop the server")

        uvicorn.run(app, host=host, port=port, log_level="info")

    class StartServerIO(IO[None]):
        def run(self) -> None:
            return _start_server()

    return StartServerIO()
