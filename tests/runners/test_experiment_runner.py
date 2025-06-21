"""
Unit tests for the CLI Experiment Runner
Tests the ExperimentRunner class and CLI functionality
"""

from unittest.mock import patch, Mock
from pathlib import Path
import tempfile
from datetime import datetime

from src.runners.experiment_runner import ExperimentRunner, create_cli_parser
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


class TestExperimentRunner:
    """Test cases for ExperimentRunner class"""

    def setup_method(self) -> None:
        """Set up test fixtures"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.output_dir = self.temp_dir / "results"
        self.checkpoint_dir = self.temp_dir / "checkpoints"

        self.runner = ExperimentRunner(
            output_dir=self.output_dir, checkpoint_dir=self.checkpoint_dir
        )

    def teardown_method(self) -> None:
        """Clean up test fixtures"""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_runner_initialization(self) -> None:
        """Test ExperimentRunner initialization"""
        assert self.runner.output_dir == self.output_dir
        assert self.runner.checkpoint_dir == self.checkpoint_dir
        assert self.output_dir.exists()
        assert self.checkpoint_dir.exists()
        assert self.runner.checkpoint_manager is not None
        assert self.runner.resource_monitor is not None

    @patch("src.runners.experiment_runner.generate_experiment_matrix")
    def test_run_all_experiments_fresh_start(self, mock_generate_matrix: Mock) -> None:
        """Test running all experiments from fresh start"""
        mock_configs = [
            self._create_mock_config(DataScale.SMALL, Dimension(128)),
            self._create_mock_config(DataScale.SMALL, Dimension(256)),
        ]
        mock_generate_matrix.return_value = mock_configs

        with (
            patch.object(self.runner.resource_monitor, "start_monitoring"),
            patch.object(self.runner.resource_monitor, "stop_monitoring"),
            patch.object(
                self.runner.resource_monitor,
                "check_memory_available",
                return_value=True,
            ),
            patch.object(self.runner.resource_monitor, "cleanup_between_batches"),
            patch.object(self.runner, "_run_experiment_batch") as mock_run_batch,
            patch.object(self.runner.checkpoint_manager, "save_batch_results"),
        ):
            mock_results = [self._create_mock_result(config) for config in mock_configs]
            mock_run_batch.return_value = mock_results

            results = self.runner.run_all_experiments(batch_size=2, resume=False)

            assert len(results) == 2
            mock_generate_matrix.assert_called_once()
            mock_run_batch.assert_called_once()

    def test_config_summary(self) -> None:
        """Test configuration summary generation"""
        config = self._create_mock_config(
            DataScale.SMALL, Dimension(256), SearchType.HYBRID
        )
        summary = self.runner._config_summary(config)

        assert "small" in summary
        assert "256d" in summary
        assert "hybrid" in summary
        assert "unfiltered" in summary  # Default filter config

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


class TestCLIParser:
    """Test cases for CLI argument parser"""

    def test_create_cli_parser(self) -> None:
        """Test CLI parser creation"""
        parser = create_cli_parser()
        assert parser is not None
        assert parser.description == "DuckDB VSS Benchmarking Experiment Runner"

    def test_parse_all_experiments(self) -> None:
        """Test parsing --all flag"""
        parser = create_cli_parser()
        args = parser.parse_args(["--all"])

        assert args.all is True
        assert args.batch_size == 4  # default
        assert args.resume is False  # default
