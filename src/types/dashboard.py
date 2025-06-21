"""
Dashboard Data Types for DuckDB VSS Benchmarking
Immutable data structures for dashboard state management
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List, Any
from ..types.core import ExperimentResult
from ..runners.monitoring import ResourceSnapshot


@dataclass(frozen=True)
class ExperimentProgress:
    """Progress tracking for experiment execution"""

    total_experiments: int = 0
    completed_experiments: int = 0
    failed_experiments: int = 0
    current_experiment: Optional[str] = None
    current_batch: int = 0
    total_batches: int = 0
    estimated_time_remaining: Optional[float] = None


@dataclass(frozen=True)
class ResourceMetrics:
    """Resource usage metrics for dashboard display"""

    latest_snapshot: Optional[ResourceSnapshot] = None
    memory_usage_mb: float = 0.0
    memory_percent: float = 0.0
    cpu_percent: float = 0.0
    available_memory_mb: float = 0.0
    last_updated: Optional[datetime] = None


@dataclass(frozen=True)
class PerformanceCharts:
    """Performance metrics for dashboard charts"""

    latest_insert_time_ms: Optional[float] = None
    latest_insert_qps: Optional[float] = None
    latest_index_time_ms: Optional[float] = None
    latest_recall_at_1: Optional[float] = None
    latest_recall_at_5: Optional[float] = None
    latest_recall_at_10: Optional[float] = None
    latest_mrr: Optional[float] = None


@dataclass(frozen=True)
class DashboardState:
    """Complete dashboard state with all monitoring data"""

    experiment_progress: ExperimentProgress
    resource_metrics: ResourceMetrics
    performance_charts: PerformanceCharts
    recent_results: List[ExperimentResult]
    alerts: List[str]
    start_time: Optional[datetime] = None

    @classmethod
    def create_initial(cls) -> "DashboardState":
        """Create initial dashboard state"""
        return cls(
            experiment_progress=ExperimentProgress(),
            resource_metrics=ResourceMetrics(),
            performance_charts=PerformanceCharts(),
            recent_results=[],
            alerts=[],
            start_time=datetime.now(),
        )


@dataclass(frozen=True)
class DashboardUpdate:
    """Represents a dashboard update operation"""

    update_type: str
    data: dict[str, Any]
    timestamp: datetime

    @classmethod
    def progress_update(
        cls, completed: int, total: int, current: Optional[str] = None
    ) -> "DashboardUpdate":
        """Create progress update"""
        return cls(
            update_type="progress",
            data={"completed": completed, "total": total, "current": current},
            timestamp=datetime.now(),
        )

    @classmethod
    def resource_update(cls, snapshot: ResourceSnapshot) -> "DashboardUpdate":
        """Create resource update"""
        return cls(
            update_type="resource",
            data={"snapshot": snapshot},
            timestamp=datetime.now(),
        )

    @classmethod
    def result_update(cls, result: ExperimentResult) -> "DashboardUpdate":
        """Create result update"""
        return cls(
            update_type="result", data={"result": result}, timestamp=datetime.now()
        )

    @classmethod
    def alert_update(cls, message: str) -> "DashboardUpdate":
        """Create alert update"""
        return cls(
            update_type="alert", data={"message": message}, timestamp=datetime.now()
        )
