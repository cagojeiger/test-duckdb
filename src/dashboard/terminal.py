"""
Terminal Dashboard for DuckDB VSS Benchmarking
Real-time monitoring interface using Rich library
"""

from datetime import datetime
from typing import Optional, List
from dataclasses import dataclass

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.align import Align

from ..types.core import ExperimentResult
from ..runners.monitoring import ResourceSnapshot


@dataclass
class DashboardState:
    """Immutable state for dashboard display"""

    current_experiment: Optional[str] = None
    total_experiments: int = 0
    completed_experiments: int = 0
    failed_experiments: int = 0
    current_batch: int = 0
    total_batches: int = 0

    latest_resource_snapshot: Optional[ResourceSnapshot] = None
    recent_results: Optional[List[ExperimentResult]] = None
    alerts: Optional[List[str]] = None

    start_time: Optional[datetime] = None

    def __post_init__(self) -> None:
        if self.recent_results is None:
            object.__setattr__(self, "recent_results", [])
        if self.alerts is None:
            object.__setattr__(self, "alerts", [])


class TerminalDashboard:
    """Terminal-based dashboard for real-time experiment monitoring"""

    def __init__(self) -> None:
        self.console = Console()
        self.layout = Layout()
        self.state = DashboardState()
        self.is_running = False
        self.live: Optional[Live] = None

        self._setup_layout()

    def _setup_layout(self) -> None:
        """Configure the dashboard layout structure"""
        self.layout.split(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=7),
        )

        self.layout["main"].split_row(Layout(name="left"), Layout(name="right"))

        self.layout["left"].split(
            Layout(name="progress", size=8), Layout(name="resources", ratio=1)
        )

        self.layout["right"].split(
            Layout(name="metrics", ratio=1), Layout(name="results", ratio=1)
        )

    def start(self) -> None:
        """Start the dashboard display"""
        if self.is_running:
            return

        self.is_running = True
        self.state = DashboardState(start_time=datetime.now())

        self.live = Live(
            self.layout, console=self.console, screen=True, refresh_per_second=2
        )
        self.live.start()

        self._update_display()

    def stop(self) -> None:
        """Stop the dashboard display"""
        if not self.is_running:
            return

        self.is_running = False

        if self.live:
            self.live.stop()
            self.live = None

    def update_progress(
        self, completed: int, total: int, current_experiment: Optional[str] = None
    ) -> None:
        """Update experiment progress"""
        new_state = DashboardState(
            current_experiment=current_experiment or self.state.current_experiment,
            total_experiments=total,
            completed_experiments=completed,
            failed_experiments=self.state.failed_experiments,
            current_batch=self.state.current_batch,
            total_batches=self.state.total_batches,
            latest_resource_snapshot=self.state.latest_resource_snapshot,
            recent_results=self.state.recent_results,
            alerts=self.state.alerts,
            start_time=self.state.start_time,
        )

        self.state = new_state
        try:
            self._update_display()
        except Exception:
            pass

    def update_batch_progress(self, current_batch: int, total_batches: int) -> None:
        """Update batch progress"""
        new_state = DashboardState(
            current_experiment=self.state.current_experiment,
            total_experiments=self.state.total_experiments,
            completed_experiments=self.state.completed_experiments,
            failed_experiments=self.state.failed_experiments,
            current_batch=current_batch,
            total_batches=total_batches,
            latest_resource_snapshot=self.state.latest_resource_snapshot,
            recent_results=self.state.recent_results,
            alerts=self.state.alerts,
            start_time=self.state.start_time,
        )

        self.state = new_state
        try:
            self._update_display()
        except Exception:
            pass

    def update_resources(self, snapshot: ResourceSnapshot) -> None:
        """Update resource monitoring data"""
        new_state = DashboardState(
            current_experiment=self.state.current_experiment,
            total_experiments=self.state.total_experiments,
            completed_experiments=self.state.completed_experiments,
            failed_experiments=self.state.failed_experiments,
            current_batch=self.state.current_batch,
            total_batches=self.state.total_batches,
            latest_resource_snapshot=snapshot,
            recent_results=self.state.recent_results,
            alerts=self.state.alerts,
            start_time=self.state.start_time,
        )

        self.state = new_state
        try:
            self._update_display()
        except Exception:
            pass

    def add_result(self, result: ExperimentResult) -> None:
        """Add a completed experiment result"""
        recent_results = list(self.state.recent_results or [])
        recent_results.append(result)

        if len(recent_results) > 10:
            recent_results = recent_results[-10:]

        new_state = DashboardState(
            current_experiment=self.state.current_experiment,
            total_experiments=self.state.total_experiments,
            completed_experiments=self.state.completed_experiments,
            failed_experiments=self.state.failed_experiments,
            current_batch=self.state.current_batch,
            total_batches=self.state.total_batches,
            latest_resource_snapshot=self.state.latest_resource_snapshot,
            recent_results=recent_results,
            alerts=self.state.alerts,
            start_time=self.state.start_time,
        )

        self.state = new_state
        try:
            self._update_display()
        except Exception:
            pass

    def add_alert(self, message: str) -> None:
        """Add an alert message"""
        alerts = list(self.state.alerts or [])
        timestamp = datetime.now().strftime("%H:%M:%S")
        alerts.append(f"[{timestamp}] {message}")

        if len(alerts) > 5:
            alerts = alerts[-5:]

        new_state = DashboardState(
            current_experiment=self.state.current_experiment,
            total_experiments=self.state.total_experiments,
            completed_experiments=self.state.completed_experiments,
            failed_experiments=self.state.failed_experiments,
            current_batch=self.state.current_batch,
            total_batches=self.state.total_batches,
            latest_resource_snapshot=self.state.latest_resource_snapshot,
            recent_results=self.state.recent_results,
            alerts=alerts,
            start_time=self.state.start_time,
        )

        self.state = new_state
        try:
            self._update_display()
        except Exception:
            pass

    def _update_display(self) -> None:
        """Update all dashboard panels"""
        if not self.is_running:
            return

        try:
            self.layout["header"].update(self._create_header_panel())
            self.layout["progress"].update(self._create_progress_panel())
            self.layout["resources"].update(self._create_resources_panel())
            self.layout["metrics"].update(self._create_metrics_panel())
            self.layout["results"].update(self._create_results_panel())
            self.layout["footer"].update(self._create_footer_panel())
        except Exception:
            pass

    def _create_header_panel(self) -> Panel:
        """Create header panel with title and status"""
        title = Text("DuckDB VSS Benchmarking Dashboard", style="bold blue")

        if self.state.start_time:
            elapsed = datetime.now() - self.state.start_time
            elapsed_str = str(elapsed).split(".")[0]
            status = f"Running for {elapsed_str}"
        else:
            status = "Initializing..."

        content = Align.center(f"{title}\n{status}")
        return Panel(content, style="blue")

    def _create_progress_panel(self) -> Panel:
        """Create progress panel with experiment and batch progress"""
        content_lines = []

        if self.state.total_experiments > 0:
            progress_pct = (
                self.state.completed_experiments / self.state.total_experiments
            ) * 100
            content_lines.append(
                f"Experiments: {self.state.completed_experiments}/{self.state.total_experiments} ({progress_pct:.1f}%)"
            )
        else:
            content_lines.append("Experiments: 0/0 (0.0%)")

        if self.state.total_batches > 0:
            batch_pct = (self.state.current_batch / self.state.total_batches) * 100
            content_lines.append(
                f"Batches: {self.state.current_batch}/{self.state.total_batches} ({batch_pct:.1f}%)"
            )

        if self.state.current_experiment:
            content_lines.append(f"Current: {self.state.current_experiment}")

        if not content_lines:
            content_lines.append("No experiments running")

        content = "\n".join(content_lines)
        return Panel(content, title="Progress", style="green")

    def _create_resources_panel(self) -> Panel:
        """Create resource monitoring panel"""
        if not self.state.latest_resource_snapshot:
            return Panel(
                "No resource data available", title="Resources", style="yellow"
            )

        snapshot = self.state.latest_resource_snapshot

        table = Table(show_header=False, box=None)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")

        table.add_row("Memory Usage", f"{snapshot.memory_usage_mb:.1f} MB")
        table.add_row("Memory %", f"{snapshot.memory_percent:.1f}%")
        table.add_row("CPU %", f"{snapshot.cpu_percent:.1f}%")
        table.add_row("Available", f"{snapshot.available_memory_mb:.1f} MB")
        table.add_row("Updated", snapshot.timestamp.strftime("%H:%M:%S"))

        return Panel(table, title="Resources", style="yellow")

    def _create_metrics_panel(self) -> Panel:
        """Create performance metrics panel"""
        if not self.state.recent_results:
            return Panel("No metrics available", title="Performance", style="magenta")

        recent_result = self.state.recent_results[-1]

        table = Table(show_header=False, box=None)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")

        table.add_row(
            "Insert Time", f"{recent_result.insert_metrics.query_time_ms:.1f} ms"
        )
        table.add_row(
            "Insert QPS", f"{recent_result.insert_metrics.throughput_qps:.1f}"
        )
        table.add_row(
            "Index Time", f"{recent_result.index_metrics.query_time_ms:.1f} ms"
        )
        table.add_row("Recall@1", f"{recent_result.accuracy.recall_at_1:.3f}")
        table.add_row("Recall@5", f"{recent_result.accuracy.recall_at_5:.3f}")

        return Panel(table, title="Performance", style="magenta")

    def _create_results_panel(self) -> Panel:
        """Create recent results panel"""
        if not self.state.recent_results:
            return Panel("No results yet", title="Recent Results", style="cyan")

        table = Table(show_header=True, box=None)
        table.add_column("Config", style="white")
        table.add_column("Status", style="green")
        table.add_column("Time", style="yellow")

        for result in self.state.recent_results[-5:]:
            config_summary = f"{result.config.data_scale.name.lower()}-{int(result.config.dimension)}d"
            status = "✅ Success"
            time_str = result.timestamp.strftime("%H:%M:%S")

            table.add_row(config_summary, status, time_str)

        return Panel(table, title="Recent Results", style="cyan")

    def _create_footer_panel(self) -> Panel:
        """Create footer panel with alerts and controls"""
        content = []

        if self.state.alerts:
            content.append("🚨 Alerts:")
            for alert in self.state.alerts[-3:]:
                content.append(f"  {alert}")
        else:
            content.append("No alerts")

        content.append("")
        content.append("Press Ctrl+C to stop")

        return Panel("\n".join(content), title="Status", style="red")
