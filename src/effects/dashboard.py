"""
Dashboard Effects for DuckDB VSS Benchmarking
IO operations for dashboard updates following functional programming patterns
"""

from typing import Callable, Optional

from ..types.monads import IO, PureIO
from ..types.dashboard import (
    DashboardState,
    DashboardUpdate,
    ExperimentProgress,
    ResourceMetrics,
    PerformanceCharts,
)
from ..types.experiment import ExperimentResult
from ..runners.monitoring import ResourceSnapshot


def update_dashboard_state(
    current_state: DashboardState, update: DashboardUpdate
) -> IO[DashboardState]:
    """Update dashboard state with new data"""

    def _update() -> DashboardState:
        if update.update_type == "progress":
            data = update.data
            new_progress = ExperimentProgress(
                total_experiments=data.get(
                    "total", current_state.experiment_progress.total_experiments
                ),
                completed_experiments=data.get(
                    "completed", current_state.experiment_progress.completed_experiments
                ),
                failed_experiments=current_state.experiment_progress.failed_experiments,
                current_experiment=data.get(
                    "current", current_state.experiment_progress.current_experiment
                ),
                current_batch=current_state.experiment_progress.current_batch,
                total_batches=current_state.experiment_progress.total_batches,
                estimated_time_remaining=current_state.experiment_progress.estimated_time_remaining,
            )

            return DashboardState(
                experiment_progress=new_progress,
                resource_metrics=current_state.resource_metrics,
                performance_charts=current_state.performance_charts,
                recent_results=current_state.recent_results,
                alerts=current_state.alerts,
                start_time=current_state.start_time,
            )

        elif update.update_type == "resource":
            snapshot = update.data["snapshot"]
            new_metrics = ResourceMetrics(
                latest_snapshot=snapshot,
                memory_usage_mb=snapshot.memory_usage_mb,
                memory_percent=snapshot.memory_percent,
                cpu_percent=snapshot.cpu_percent,
                available_memory_mb=snapshot.available_memory_mb,
                last_updated=snapshot.timestamp,
            )

            return DashboardState(
                experiment_progress=current_state.experiment_progress,
                resource_metrics=new_metrics,
                performance_charts=current_state.performance_charts,
                recent_results=current_state.recent_results,
                alerts=current_state.alerts,
                start_time=current_state.start_time,
            )

        elif update.update_type == "result":
            result = update.data["result"]
            new_results = list(current_state.recent_results)
            new_results.append(result)

            if len(new_results) > 10:
                new_results = new_results[-10:]

            new_charts = PerformanceCharts(
                latest_insert_time_ms=result.insert_metrics.query_time_ms,
                latest_insert_qps=result.insert_metrics.throughput_qps,
                latest_index_time_ms=result.index_metrics.query_time_ms,
                latest_recall_at_1=result.accuracy.recall_at_1,
                latest_recall_at_5=result.accuracy.recall_at_5,
                latest_recall_at_10=result.accuracy.recall_at_10,
                latest_mrr=result.accuracy.mean_reciprocal_rank,
            )

            return DashboardState(
                experiment_progress=current_state.experiment_progress,
                resource_metrics=current_state.resource_metrics,
                performance_charts=new_charts,
                recent_results=new_results,
                alerts=current_state.alerts,
                start_time=current_state.start_time,
            )

        elif update.update_type == "alert":
            message = update.data["message"]
            timestamp = update.timestamp.strftime("%H:%M:%S")
            alert_message = f"[{timestamp}] {message}"

            new_alerts = list(current_state.alerts)
            new_alerts.append(alert_message)

            if len(new_alerts) > 5:
                new_alerts = new_alerts[-5:]

            return DashboardState(
                experiment_progress=current_state.experiment_progress,
                resource_metrics=current_state.resource_metrics,
                performance_charts=current_state.performance_charts,
                recent_results=current_state.recent_results,
                alerts=new_alerts,
                start_time=current_state.start_time,
            )

        else:
            return current_state

    return PureIO(_update())


def get_dashboard_state() -> IO[DashboardState]:
    """Get current dashboard state from monitoring systems"""

    def _get_state() -> DashboardState:
        return DashboardState.create_initial()

    return PureIO(_get_state())


def apply_dashboard_update(
    state: DashboardState, update: DashboardUpdate
) -> IO[DashboardState]:
    """Apply a dashboard update to the current state"""
    return update_dashboard_state(state, update)


def create_progress_update_effect(
    completed: int, total: int, current: Optional[str] = None
) -> IO[DashboardUpdate]:
    """Create effect for progress update"""

    def _create_update() -> DashboardUpdate:
        return DashboardUpdate.progress_update(completed, total, current)

    return PureIO(_create_update())


def create_resource_update_effect(snapshot: ResourceSnapshot) -> IO[DashboardUpdate]:
    """Create effect for resource update"""

    def _create_update() -> DashboardUpdate:
        return DashboardUpdate.resource_update(snapshot)

    return PureIO(_create_update())


def create_result_update_effect(result: ExperimentResult) -> IO[DashboardUpdate]:
    """Create effect for result update"""

    def _create_update() -> DashboardUpdate:
        return DashboardUpdate.result_update(result)

    return PureIO(_create_update())


def create_alert_update_effect(message: str) -> IO[DashboardUpdate]:
    """Create effect for alert update"""

    def _create_update() -> DashboardUpdate:
        return DashboardUpdate.alert_update(message)

    return PureIO(_create_update())


def render_dashboard_effect(
    state: DashboardState, render_callback: Callable[[DashboardState], None]
) -> IO[None]:
    """Effect to render dashboard with current state"""

    def _render() -> None:
        render_callback(state)
        return None

    return PureIO(_render())
