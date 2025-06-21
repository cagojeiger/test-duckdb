"""
Unit tests for the Parallel Experiment Runner
Tests the ParallelExperimentRunner class and parallel execution functionality
"""

from unittest.mock import Mock, patch
from concurrent.futures import Future
from datetime import datetime

import pytest

from src.runners.parallel_runner import (
    ParallelExperimentRunner,
    ParallelConfig,
    ParallelResult,
    _run_single_experiment_worker,
    create_parallel_runner,
)
from src.types.core import (
    ExperimentConfig,
    ExperimentResult,
    DataScale,
    Dimension,
    SearchType,
    FilterConfig,
    HNSWParams,
    Metrics,
    AccuracyMetrics,
)


class TestParallelConfig:
    """Test cases for ParallelConfig dataclass"""

    def test_parallel_config_creation(self) -> None:
        """Test ParallelConfig creation with default values"""
        config = ParallelConfig()

        assert config.max_workers == 4
        assert config.memory_threshold_mb == 6000
        assert config.experiment_timeout_seconds == 300
        assert config.min_workers == 1
        assert config.max_workers_limit == 8

    def test_parallel_config_custom_values(self) -> None:
        """Test ParallelConfig creation with custom values"""
        config = ParallelConfig(
            max_workers=8,
            memory_threshold_mb=8000,
            experiment_timeout_seconds=600,
            min_workers=2,
            max_workers_limit=16,
        )

        assert config.max_workers == 8
        assert config.memory_threshold_mb == 8000
        assert config.experiment_timeout_seconds == 600
        assert config.min_workers == 2
        assert config.max_workers_limit == 16


class TestParallelResult:
    """Test cases for ParallelResult dataclass"""

    def test_parallel_result_creation(self) -> None:
        """Test ParallelResult creation"""
        mock_result = self._create_mock_result()
        mock_config = self._create_mock_config()

        parallel_result = ParallelResult(
            results=[mock_result],
            failed_configs=[mock_config],
            execution_time_seconds=120.5,
            peak_memory_mb=2048.0,
            worker_count_used=4,
        )

        assert len(parallel_result.results) == 1
        assert len(parallel_result.failed_configs) == 1
        assert parallel_result.execution_time_seconds == 120.5
        assert parallel_result.peak_memory_mb == 2048.0
        assert parallel_result.worker_count_used == 4

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


class TestWorkerFunction:
    """Test cases for the worker function"""

    @patch("src.runners.parallel_runner.single_experiment_pipeline")
    def test_run_single_experiment_worker_success(self, mock_pipeline: Mock) -> None:
        """Test successful experiment execution in worker"""
        mock_config = ExperimentConfig(
            data_scale=DataScale.SMALL,
            dimension=Dimension(128),
            search_type=SearchType.PURE_VECTOR,
            filter_config=FilterConfig(enabled=False),
            hnsw_params=HNSWParams(),
            batch_size=100,
            num_queries=10,
        )

        mock_result = ExperimentResult(
            config=mock_config,
            insert_metrics=Metrics(100.0, 1000.0, 500.0, 100.0),
            index_metrics=Metrics(50.0, 0.0, 600.0, 150.0),
            search_results=[],
            accuracy=AccuracyMetrics(0.9, 0.95, 0.98, 0.92),
            timestamp=datetime.now(),
        )

        mock_io = Mock()
        mock_io.run.return_value = mock_result
        mock_pipeline.return_value = mock_io

        result = _run_single_experiment_worker(mock_config)

        assert result == mock_result
        mock_pipeline.assert_called_once_with(mock_config)
        mock_io.run.assert_called_once()

    @patch("src.runners.parallel_runner.single_experiment_pipeline")
    def test_run_single_experiment_worker_failure(self, mock_pipeline: Mock) -> None:
        """Test experiment failure in worker"""
        mock_config = ExperimentConfig(
            data_scale=DataScale.SMALL,
            dimension=Dimension(128),
            search_type=SearchType.PURE_VECTOR,
            filter_config=FilterConfig(enabled=False),
            hnsw_params=HNSWParams(),
            batch_size=100,
            num_queries=10,
        )

        mock_io = Mock()
        mock_io.run.side_effect = Exception("Database connection failed")
        mock_pipeline.return_value = mock_io

        with pytest.raises(RuntimeError, match="Experiment failed for config"):
            _run_single_experiment_worker(mock_config)


class TestParallelExperimentRunner:
    """Test cases for ParallelExperimentRunner class"""

    def setup_method(self) -> None:
        """Set up test fixtures"""
        self.config = ParallelConfig(
            max_workers=2,
            memory_threshold_mb=4000,
            experiment_timeout_seconds=60,
        )
        self.runner = ParallelExperimentRunner(self.config)

    def test_runner_initialization(self) -> None:
        """Test ParallelExperimentRunner initialization"""
        assert self.runner.config == self.config
        assert self.runner.resource_monitor is not None

    @patch("src.runners.parallel_runner.psutil.virtual_memory")
    @patch("src.runners.parallel_runner.psutil.cpu_count")
    def test_calculate_optimal_workers(
        self, mock_cpu_count: Mock, mock_memory: Mock
    ) -> None:
        """Test optimal worker calculation based on system resources"""
        mock_memory_info = Mock()
        mock_memory_info.available = 8 * 1024 * 1024 * 1024  # 8GB available
        mock_memory.return_value = mock_memory_info
        mock_cpu_count.return_value = 8

        optimal_workers = self.runner._calculate_optimal_workers()

        assert optimal_workers == 2
        mock_memory.assert_called_once()
        mock_cpu_count.assert_called_once()

    @patch("src.runners.parallel_runner.psutil.virtual_memory")
    def test_check_memory_pressure(self, mock_memory: Mock) -> None:
        """Test memory pressure detection"""
        mock_memory_info = Mock()
        mock_memory_info.percent = 90.0
        mock_memory.return_value = mock_memory_info

        assert self.runner._check_memory_pressure() is True

        mock_memory_info.percent = 70.0
        assert self.runner._check_memory_pressure() is False

    @patch("concurrent.futures.ProcessPoolExecutor")
    @patch.object(ParallelExperimentRunner, "_calculate_optimal_workers")
    def test_run_experiments_parallel_success(
        self, mock_calc_workers: Mock, mock_executor_class: Mock
    ) -> None:
        """Test successful parallel experiment execution"""
        mock_calc_workers.return_value = 2

        mock_configs = [
            self._create_mock_config(DataScale.SMALL, Dimension(128)),
            self._create_mock_config(DataScale.SMALL, Dimension(256)),
        ]

        mock_results = [self._create_mock_result(config) for config in mock_configs]

        mock_executor = Mock()
        mock_executor_class.return_value.__enter__.return_value = mock_executor

        mock_futures = []
        for i, result in enumerate(mock_results):
            mock_future = Mock(spec=Future)
            mock_future.result.return_value = result
            mock_futures.append(mock_future)

        mock_executor.submit.side_effect = mock_futures

        with patch("concurrent.futures.as_completed", return_value=mock_futures):
            with patch.object(self.runner.resource_monitor, "start_monitoring"):
                with patch.object(self.runner.resource_monitor, "stop_monitoring"):
                    with patch.object(
                        self.runner.resource_monitor,
                        "get_memory_usage",
                        return_value=1000.0,
                    ):
                        parallel_io = self.runner.run_experiments_parallel(mock_configs)
                        result = parallel_io.run()

        assert len(result.results) == 2
        assert len(result.failed_configs) == 0
        assert result.worker_count_used == 2
        assert result.execution_time_seconds > 0
        assert result.peak_memory_mb >= 1000.0

    @patch("concurrent.futures.ProcessPoolExecutor")
    @patch.object(ParallelExperimentRunner, "_calculate_optimal_workers")
    def test_run_experiments_parallel_with_failures(
        self, mock_calc_workers: Mock, mock_executor_class: Mock
    ) -> None:
        """Test parallel execution with some experiment failures"""
        mock_calc_workers.return_value = 2

        mock_configs = [
            self._create_mock_config(DataScale.SMALL, Dimension(128)),
            self._create_mock_config(DataScale.SMALL, Dimension(256)),
        ]

        mock_executor = Mock()
        mock_executor_class.return_value.__enter__.return_value = mock_executor

        mock_future_success = Mock()
        mock_future_success.result.return_value = self._create_mock_result(
            mock_configs[0]
        )

        mock_future_failure = Mock()
        mock_future_failure.result.side_effect = Exception("Experiment failed")

        future_to_config = {
            mock_future_success: mock_configs[0],
            mock_future_failure: mock_configs[1],
        }

        mock_executor.submit.side_effect = [mock_future_success, mock_future_failure]

        with patch("concurrent.futures.as_completed") as mock_as_completed:
            mock_as_completed.return_value = [mock_future_success, mock_future_failure]

            with patch.object(self.runner.resource_monitor, "start_monitoring"):
                with patch.object(self.runner.resource_monitor, "stop_monitoring"):
                    with patch.object(
                        self.runner.resource_monitor,
                        "get_memory_usage",
                        return_value=1000.0,
                    ):
                        parallel_io = self.runner.run_experiments_parallel(mock_configs)
                        result = parallel_io.run()

        assert len(result.results) == 1  # One success
        assert len(result.failed_configs) == 1  # One failure
        assert result.worker_count_used == 2

    def test_run_experiments_batched_parallel(self) -> None:
        """Test batched parallel execution"""
        mock_configs = [
            self._create_mock_config(DataScale.SMALL, Dimension(128)),
            self._create_mock_config(DataScale.SMALL, Dimension(256)),
            self._create_mock_config(DataScale.MEDIUM, Dimension(128)),
            self._create_mock_config(DataScale.MEDIUM, Dimension(256)),
        ]

        with patch.object(self.runner, "run_experiments_parallel") as mock_run_parallel:
            mock_batch_result = ParallelResult(
                results=[
                    self._create_mock_result(config) for config in mock_configs[:2]
                ],
                failed_configs=[],
                execution_time_seconds=60.0,
                peak_memory_mb=2000.0,
                worker_count_used=2,
            )

            mock_io = Mock()
            mock_io.run.return_value = mock_batch_result
            mock_run_parallel.return_value = mock_io

            with patch.object(
                self.runner, "_calculate_optimal_workers", return_value=2
            ):
                with patch.object(
                    self.runner.resource_monitor, "cleanup_between_batches"
                ):
                    batched_io = self.runner.run_experiments_batched_parallel(
                        mock_configs, batch_size=2
                    )
                    result = batched_io.run()

        assert mock_run_parallel.call_count == 2
        assert len(result.results) == 4  # 2 results per batch * 2 batches
        assert len(result.failed_configs) == 0

    def _create_mock_config(
        self,
        data_scale: DataScale = DataScale.SMALL,
        dimension: Dimension = Dimension(128),
        search_type: SearchType = SearchType.PURE_VECTOR,
    ) -> ExperimentConfig:
        """Create a mock experiment configuration"""
        return ExperimentConfig(
            data_scale=data_scale,
            dimension=dimension,
            search_type=search_type,
            filter_config=FilterConfig(enabled=False),
            hnsw_params=HNSWParams(),
            batch_size=100,
            num_queries=10,
        )

    def _create_mock_result(self, config: ExperimentConfig) -> ExperimentResult:
        """Create a mock experiment result"""
        return ExperimentResult(
            config=config,
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


class TestFactoryFunction:
    """Test cases for the factory function"""

    def test_create_parallel_runner(self) -> None:
        """Test parallel runner factory function"""
        runner = create_parallel_runner(
            max_workers=6,
            memory_threshold_mb=8000,
            experiment_timeout_seconds=600,
        )

        assert isinstance(runner, ParallelExperimentRunner)
        assert runner.config.max_workers == 6
        assert runner.config.memory_threshold_mb == 8000
        assert runner.config.experiment_timeout_seconds == 600

    def test_create_parallel_runner_defaults(self) -> None:
        """Test parallel runner factory function with defaults"""
        runner = create_parallel_runner()

        assert isinstance(runner, ParallelExperimentRunner)
        assert runner.config.max_workers == 4
        assert runner.config.memory_threshold_mb == 6000
        assert runner.config.experiment_timeout_seconds == 300
