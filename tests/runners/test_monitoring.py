"""
Unit tests for the Resource Monitoring System
Tests the ResourceMonitor class and ResourceSnapshot functionality
"""

from unittest.mock import Mock, patch
from datetime import datetime

from src.runners.monitoring import ResourceMonitor, ResourceSnapshot, ResourceError


class TestResourceSnapshot:
    """Test cases for ResourceSnapshot dataclass"""

    def test_resource_snapshot_creation(self) -> None:
        """Test ResourceSnapshot creation and attributes"""
        timestamp = datetime.now()
        snapshot = ResourceSnapshot(
            timestamp=timestamp,
            memory_usage_mb=1024.0,
            memory_percent=75.5,
            cpu_percent=45.2,
            available_memory_mb=2048.0,
            process_memory_mb=1024.0,
        )

        assert snapshot.timestamp == timestamp
        assert snapshot.memory_usage_mb == 1024.0
        assert snapshot.memory_percent == 75.5
        assert snapshot.cpu_percent == 45.2
        assert snapshot.available_memory_mb == 2048.0
        assert snapshot.process_memory_mb == 1024.0


class TestResourceMonitor:
    """Test cases for ResourceMonitor class"""

    def setup_method(self) -> None:
        """Set up test fixtures"""
        self.monitor = ResourceMonitor(
            memory_threshold_mb=1000, monitoring_interval=0.1
        )

    def teardown_method(self) -> None:
        """Clean up test fixtures"""
        if self.monitor.is_monitoring:
            self.monitor.stop_monitoring()

    def test_resource_monitor_initialization(self) -> None:
        """Test ResourceMonitor initialization"""
        assert self.monitor.memory_threshold_mb == 1000
        assert self.monitor.monitoring_interval == 0.1
        assert self.monitor.is_monitoring is False
        assert self.monitor.monitoring_thread is None
        assert len(self.monitor.resource_history) == 0
        assert len(self.monitor.memory_alerts) == 0
        assert self.monitor.process is not None

    @patch("psutil.Process")
    def test_get_memory_usage(self, mock_process_class: Mock) -> None:
        """Test getting current memory usage"""
        mock_process = Mock()
        mock_memory_info = Mock()
        mock_memory_info.rss = 1024 * 1024 * 512  # 512 MB in bytes
        mock_process.memory_info.return_value = mock_memory_info
        mock_process_class.return_value = mock_process

        monitor = ResourceMonitor()
        memory_usage = monitor.get_memory_usage()

        assert memory_usage == 512.0  # Should be 512 MB

    def test_check_memory_available(self) -> None:
        """Test checking if sufficient memory is available"""
        with patch.object(self.monitor, "get_system_memory_info") as mock_get_memory:
            mock_get_memory.return_value = {"available_mb": 4000.0}
            assert self.monitor.check_memory_available(2000) is True

            mock_get_memory.return_value = {"available_mb": 1000.0}
            assert self.monitor.check_memory_available(2000) is False


class TestResourceError:
    """Test cases for ResourceError exception"""

    def test_resource_error_creation(self) -> None:
        """Test ResourceError exception creation"""
        error = ResourceError("Test resource error")
        assert str(error) == "Test resource error"
        assert isinstance(error, Exception)
