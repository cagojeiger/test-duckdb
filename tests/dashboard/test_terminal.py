"""
Unit tests for the Terminal Dashboard
Tests the TerminalDashboard class and dashboard functionality
"""

from unittest.mock import Mock, patch
from datetime import datetime

import pytest

from src.dashboard.terminal import TerminalDashboard, DashboardState
from src.types.core import ExperimentResult, ExperimentConfig
from src.types.core import (
    DataScale,
    Dimension,
    SearchType,
    FilterConfig,
    HNSWParams,
    Metrics,
    AccuracyMetrics,
)
from src.runners.monitoring import ResourceSnapshot


class TestDashboardState:
    """Test cases for DashboardState dataclass"""

    def test_dashboard_state_creation(self) -> None:
        """Test DashboardState creation with default values"""
        state = DashboardState()

        assert state.current_experiment is None
        assert state.total_experiments == 0
        assert state.completed_experiments == 0
        assert state.failed_experiments == 0
        assert state.current_batch == 0
        assert state.total_batches == 0
        assert state.latest_resource_snapshot is None
        assert state.recent_results == []
        assert state.alerts == []
        assert state.start_time is None

    def test_dashboard_state_post_init(self) -> None:
        """Test DashboardState __post_init__ method"""
        state = DashboardState(
            current_experiment="test-experiment",
            total_experiments=10,
            completed_experiments=5,
        )

        assert state.recent_results == []
        assert state.alerts == []
        assert state.current_experiment == "test-experiment"
        assert state.total_experiments == 10
        assert state.completed_experiments == 5


class TestTerminalDashboard:
    """Test cases for TerminalDashboard class"""

    def setup_method(self) -> None:
        """Set up test fixtures"""
        self.dashboard = TerminalDashboard()

    def test_dashboard_initialization(self) -> None:
        """Test TerminalDashboard initialization"""
        assert self.dashboard.console is not None
        assert self.dashboard.layout is not None
        assert isinstance(self.dashboard.state, DashboardState)
        assert self.dashboard.is_running is False
        assert self.dashboard.live is None

    @patch("src.dashboard.terminal.Live")
    def test_dashboard_start(self, mock_live_class: Mock) -> None:
        """Test dashboard start functionality"""
        mock_live = Mock()
        mock_live_class.return_value = mock_live

        self.dashboard.start()

        assert self.dashboard.is_running is True
        assert self.dashboard.state.start_time is not None
        mock_live_class.assert_called_once()
        mock_live.start.assert_called_once()

    @patch("src.dashboard.terminal.Live")
    def test_dashboard_start_already_running(self, mock_live_class: Mock) -> None:
        """Test dashboard start when already running"""
        self.dashboard.is_running = True

        self.dashboard.start()

        mock_live_class.assert_not_called()

    def test_dashboard_stop(self) -> None:
        """Test dashboard stop functionality"""
        mock_live = Mock()
        self.dashboard.live = mock_live
        self.dashboard.is_running = True

        self.dashboard.stop()

        assert self.dashboard.is_running is False
        mock_live.stop.assert_called_once()
        assert self.dashboard.live is None

    def test_dashboard_stop_not_running(self) -> None:
        """Test dashboard stop when not running"""
        self.dashboard.is_running = False
        self.dashboard.live = None

        self.dashboard.stop()

        assert self.dashboard.is_running is False
        assert self.dashboard.live is None

    @patch.object(TerminalDashboard, "_update_display")
    def test_update_progress(self, mock_update_display: Mock) -> None:
        """Test progress update functionality"""
        initial_state = self.dashboard.state

        self.dashboard.update_progress(5, 10, "test-experiment")

        assert self.dashboard.state.completed_experiments == 5
        assert self.dashboard.state.total_experiments == 10
        assert self.dashboard.state.current_experiment == "test-experiment"
        assert self.dashboard.state.start_time == initial_state.start_time
        mock_update_display.assert_called_once()

    @patch.object(TerminalDashboard, "_update_display")
    def test_update_batch_progress(self, mock_update_display: Mock) -> None:
        """Test batch progress update functionality"""
        self.dashboard.update_batch_progress(2, 5)

        assert self.dashboard.state.current_batch == 2
        assert self.dashboard.state.total_batches == 5
        mock_update_display.assert_called_once()

    @patch.object(TerminalDashboard, "_update_display")
    def test_update_resources(self, mock_update_display: Mock) -> None:
        """Test resource update functionality"""
        snapshot = ResourceSnapshot(
            timestamp=datetime.now(),
            memory_usage_mb=1024.0,
            memory_percent=75.0,
            cpu_percent=50.0,
            available_memory_mb=2048.0,
            process_memory_mb=1024.0,
        )

        self.dashboard.update_resources(snapshot)

        assert self.dashboard.state.latest_resource_snapshot == snapshot
        mock_update_display.assert_called_once()

    @patch.object(TerminalDashboard, "_update_display")
    def test_add_result(self, mock_update_display: Mock) -> None:
        """Test adding experiment result"""
        result = self._create_mock_result()

        self.dashboard.add_result(result)

        assert len(self.dashboard.state.recent_results) == 1
        assert self.dashboard.state.recent_results[0] == result
        mock_update_display.assert_called_once()

    @patch.object(TerminalDashboard, "_update_display")
    def test_add_result_max_limit(self, mock_update_display: Mock) -> None:
        """Test adding results with maximum limit"""
        results = [self._create_mock_result() for _ in range(12)]

        for result in results:
            self.dashboard.add_result(result)

        assert len(self.dashboard.state.recent_results) == 10
        assert self.dashboard.state.recent_results[-1] == results[-1]

    @patch.object(TerminalDashboard, "_update_display")
    def test_add_alert(self, mock_update_display: Mock) -> None:
        """Test adding alert message"""
        message = "Test alert message"

        self.dashboard.add_alert(message)

        assert len(self.dashboard.state.alerts) == 1
        assert message in self.dashboard.state.alerts[0]
        mock_update_display.assert_called_once()

    @patch.object(TerminalDashboard, "_update_display")
    def test_add_alert_max_limit(self, mock_update_display: Mock) -> None:
        """Test adding alerts with maximum limit"""
        messages = [f"Alert {i}" for i in range(7)]

        for message in messages:
            self.dashboard.add_alert(message)

        assert len(self.dashboard.state.alerts) == 5
        assert "Alert 6" in self.dashboard.state.alerts[-1]

    def test_update_display_not_running(self) -> None:
        """Test update display when dashboard is not running"""
        self.dashboard.is_running = False

        self.dashboard._update_display()

    @patch.object(TerminalDashboard, "_create_header_panel")
    @patch.object(TerminalDashboard, "_create_progress_panel")
    @patch.object(TerminalDashboard, "_create_resources_panel")
    @patch.object(TerminalDashboard, "_create_metrics_panel")
    @patch.object(TerminalDashboard, "_create_results_panel")
    @patch.object(TerminalDashboard, "_create_footer_panel")
    def test_update_display_running(
        self,
        mock_footer: Mock,
        mock_results: Mock,
        mock_metrics: Mock,
        mock_resources: Mock,
        mock_progress: Mock,
        mock_header: Mock,
    ) -> None:
        """Test update display when dashboard is running"""
        self.dashboard.is_running = True
        mock_layout = Mock()
        mock_layout.__getitem__ = Mock(return_value=Mock())
        self.dashboard.layout = mock_layout

        self.dashboard._update_display()

        mock_header.assert_called_once()
        mock_progress.assert_called_once()
        mock_resources.assert_called_once()
        mock_metrics.assert_called_once()
        mock_results.assert_called_once()
        mock_footer.assert_called_once()

    def test_create_header_panel(self) -> None:
        """Test header panel creation"""
        self.dashboard.state = DashboardState(start_time=datetime.now())

        panel = self.dashboard._create_header_panel()

        assert panel is not None

    def test_create_progress_panel(self) -> None:
        """Test progress panel creation"""
        self.dashboard.state = DashboardState(
            total_experiments=10,
            completed_experiments=5,
            current_experiment="test-experiment",
        )

        panel = self.dashboard._create_progress_panel()

        assert panel is not None

    def test_create_resources_panel_no_data(self) -> None:
        """Test resources panel creation with no data"""
        panel = self.dashboard._create_resources_panel()

        assert panel is not None

    def test_create_resources_panel_with_data(self) -> None:
        """Test resources panel creation with resource data"""
        snapshot = ResourceSnapshot(
            timestamp=datetime.now(),
            memory_usage_mb=1024.0,
            memory_percent=75.0,
            cpu_percent=50.0,
            available_memory_mb=2048.0,
            process_memory_mb=1024.0,
        )
        self.dashboard.state = DashboardState(latest_resource_snapshot=snapshot)

        panel = self.dashboard._create_resources_panel()

        assert panel is not None

    def test_create_metrics_panel_no_data(self) -> None:
        """Test metrics panel creation with no data"""
        panel = self.dashboard._create_metrics_panel()

        assert panel is not None

    def test_create_metrics_panel_with_data(self) -> None:
        """Test metrics panel creation with performance data"""
        result = self._create_mock_result()
        self.dashboard.state = DashboardState(recent_results=[result])

        panel = self.dashboard._create_metrics_panel()

        assert panel is not None

    def test_create_results_panel_no_data(self) -> None:
        """Test results panel creation with no data"""
        panel = self.dashboard._create_results_panel()

        assert panel is not None

    def test_create_results_panel_with_data(self) -> None:
        """Test results panel creation with results data"""
        results = [self._create_mock_result() for _ in range(3)]
        self.dashboard.state = DashboardState(recent_results=results)

        panel = self.dashboard._create_results_panel()

        assert panel is not None

    def test_create_footer_panel_no_alerts(self) -> None:
        """Test footer panel creation with no alerts"""
        panel = self.dashboard._create_footer_panel()

        assert panel is not None

    def test_create_footer_panel_with_alerts(self) -> None:
        """Test footer panel creation with alerts"""
        alerts = ["Alert 1", "Alert 2", "Alert 3"]
        self.dashboard.state = DashboardState(alerts=alerts)

        panel = self.dashboard._create_footer_panel()

        assert panel is not None

    def _create_mock_config(self) -> ExperimentConfig:
        """Create a mock experiment configuration"""
        return ExperimentConfig(
            data_scale=DataScale.SMALL,
            dimension=Dimension(128),
            search_type=SearchType.PURE_VECTOR,
            filter_config=FilterConfig(enabled=False),
            hnsw_params=HNSWParams(),
            batch_size=100,
            num_queries=10,
        )

    def _create_mock_result(self) -> ExperimentResult:
        """Create a mock experiment result"""
        return ExperimentResult(
            config=self._create_mock_config(),
            insert_metrics=Metrics(
                query_time_ms=100.0,
                throughput_qps=1000.0,
                memory_usage_mb=500.0,
                index_size_mb=100.0,
            ),
            index_metrics=Metrics(
                query_time_ms=50.0,
                throughput_qps=0.0,
                memory_usage_mb=600.0,
                index_size_mb=150.0,
            ),
            search_results=[],
            accuracy=AccuracyMetrics(
                recall_at_1=0.9,
                recall_at_5=0.95,
                recall_at_10=0.98,
                mean_reciprocal_rank=0.92,
            ),
            timestamp=datetime.now(),
        )


class TestDashboardIntegration:
    """Test cases for dashboard integration scenarios"""

    def setup_method(self) -> None:
        """Set up test fixtures"""
        self.dashboard = TerminalDashboard()

    @patch("src.dashboard.terminal.Live")
    def test_dashboard_lifecycle(self, mock_live_class: Mock) -> None:
        """Test complete dashboard lifecycle"""
        mock_live = Mock()
        mock_live_class.return_value = mock_live

        self.dashboard.start()
        assert self.dashboard.is_running is True

        self.dashboard.update_progress(5, 10, "test-experiment")
        assert self.dashboard.state.completed_experiments == 5

        snapshot = ResourceSnapshot(
            timestamp=datetime.now(),
            memory_usage_mb=1024.0,
            memory_percent=75.0,
            cpu_percent=50.0,
            available_memory_mb=2048.0,
            process_memory_mb=1024.0,
        )
        self.dashboard.update_resources(snapshot)
        assert self.dashboard.state.latest_resource_snapshot == snapshot

        self.dashboard.stop()
        assert self.dashboard.is_running is False

    @patch.object(TerminalDashboard, "_update_display")
    def test_dashboard_error_handling(self, mock_update_display: Mock) -> None:
        """Test dashboard error handling"""
        mock_update_display.side_effect = Exception("Display error")

        try:
            self.dashboard.update_progress(1, 10)
        except Exception:
            pytest.fail("Dashboard should handle display errors gracefully")

    def test_dashboard_state_immutability(self) -> None:
        """Test that dashboard state updates create new state objects"""
        initial_state = self.dashboard.state

        self.dashboard.update_progress(5, 10)

        assert self.dashboard.state is not initial_state
        assert initial_state.completed_experiments == 0
        assert self.dashboard.state.completed_experiments == 5
