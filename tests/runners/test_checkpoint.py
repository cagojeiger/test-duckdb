"""
Unit tests for the Checkpoint Management System
Tests the CheckpointManager class functionality
"""

import json
import tempfile
from pathlib import Path
from datetime import datetime
from typing import List

from src.runners.checkpoint import CheckpointManager, CheckpointError
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
    SearchResult,
)


class TestCheckpointManager:
    """Test cases for CheckpointManager class"""

    def setup_method(self) -> None:
        """Set up test fixtures"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.checkpoint_manager = CheckpointManager(self.temp_dir)

    def teardown_method(self) -> None:
        """Clean up test fixtures"""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_checkpoint_manager_initialization(self) -> None:
        """Test CheckpointManager initialization"""
        assert self.checkpoint_manager.checkpoint_dir == self.temp_dir
        assert self.temp_dir.exists()

        assert (
            self.checkpoint_manager.progress_file
            == self.temp_dir / "experiment_progress.json"
        )
        assert (
            self.checkpoint_manager.results_file
            == self.temp_dir / "experiment_results.pkl"
        )
        assert (
            self.checkpoint_manager.metadata_file
            == self.temp_dir / "checkpoint_metadata.json"
        )

    def test_save_and_load_batch_results(self) -> None:
        """Test saving and loading batch results"""
        config1 = self._create_mock_config(DataScale.SMALL, Dimension(128))
        config2 = self._create_mock_config(DataScale.SMALL, Dimension(256))
        results = [self._create_mock_result(config1), self._create_mock_result(config2)]

        self.checkpoint_manager.save_batch_results(results)

        assert self.checkpoint_manager.results_file.exists()
        assert self.checkpoint_manager.progress_file.exists()
        assert self.checkpoint_manager.metadata_file.exists()

        loaded_results = self.checkpoint_manager.load_all_results()
        assert len(loaded_results) == 2
        assert loaded_results[0].config.data_scale == DataScale.SMALL
        assert loaded_results[1].config.dimension == Dimension(256)

    def test_load_all_results_no_file(self) -> None:
        """Test loading results when no file exists"""
        results = self.checkpoint_manager.load_all_results()
        assert results == []

    def test_load_all_results_corrupted_file(self) -> None:
        """Test loading results from corrupted file"""
        with open(self.checkpoint_manager.results_file, "wb") as f:
            f.write(b"corrupted data")

        results = self.checkpoint_manager.load_all_results()
        assert results == []

    def test_load_completed_experiments(self) -> None:
        """Test loading completed experiment configurations"""
        config = self._create_mock_config(DataScale.SMALL, Dimension(128))
        result = self._create_mock_result(config)
        self.checkpoint_manager.save_batch_results([result])

        completed = self.checkpoint_manager.load_completed_experiments()
        assert len(completed) == 1
        assert completed[0]["data_scale"] == "SMALL"
        assert completed[0]["dimension"] == 128
        assert completed[0]["search_type"] == "PURE_VECTOR"
        assert completed[0]["filter_enabled"] is False

    def test_load_completed_experiments_no_file(self) -> None:
        """Test loading completed experiments when no file exists"""
        completed = self.checkpoint_manager.load_completed_experiments()
        assert completed == []

    def test_get_checkpoint_status_no_metadata(self) -> None:
        """Test getting checkpoint status when no metadata exists"""
        status = self.checkpoint_manager.get_checkpoint_status()

        assert status["exists"] is False
        assert status["total_results"] == 0
        assert status["last_updated"] is None
        assert status["completion_percentage"] == 0.0

    def test_get_checkpoint_status_with_metadata(self) -> None:
        """Test getting checkpoint status with existing metadata"""
        config = self._create_mock_config(DataScale.SMALL, Dimension(128))
        results = [self._create_mock_result(config) for _ in range(5)]
        self.checkpoint_manager.save_batch_results(results)

        status = self.checkpoint_manager.get_checkpoint_status()

        assert status["exists"] is True
        assert status["total_results"] == 5
        assert status["last_updated"] is not None
        assert status["completion_percentage"] == (5 / 48.0) * 100

    def test_clear_checkpoints(self) -> None:
        """Test clearing all checkpoint data"""
        config = self._create_mock_config(DataScale.SMALL, Dimension(128))
        result = self._create_mock_result(config)
        self.checkpoint_manager.save_batch_results([result])

        assert self.checkpoint_manager.results_file.exists()
        assert self.checkpoint_manager.progress_file.exists()
        assert self.checkpoint_manager.metadata_file.exists()

        self.checkpoint_manager.clear_checkpoints()

        assert not self.checkpoint_manager.results_file.exists()
        assert not self.checkpoint_manager.progress_file.exists()
        assert not self.checkpoint_manager.metadata_file.exists()

    def test_export_results_to_json(self) -> None:
        """Test exporting results to JSON format"""
        config1 = self._create_mock_config(DataScale.SMALL, Dimension(128))
        config2 = self._create_mock_config(DataScale.MEDIUM, Dimension(256))
        results = [self._create_mock_result(config1), self._create_mock_result(config2)]

        self.checkpoint_manager.save_batch_results(results)

        output_file = self.temp_dir / "exported_results.json"
        self.checkpoint_manager.export_results_to_json(output_file)

        assert output_file.exists()

        with open(output_file, "r") as f:
            data = json.load(f)

        assert "export_metadata" in data
        assert "experiments" in data
        assert len(data["experiments"]) == 2
        assert data["export_metadata"]["total_experiments"] == 2
        assert data["export_metadata"]["experiment_matrix_size"] == 48

        experiment = data["experiments"][0]
        assert "experiment_id" in experiment
        assert "config" in experiment
        assert "performance_metrics" in experiment
        assert "accuracy_metrics" in experiment
        assert "execution_info" in experiment

    def test_export_results_no_data(self) -> None:
        """Test exporting results when no data exists"""
        output_file = self.temp_dir / "empty_export.json"
        self.checkpoint_manager.export_results_to_json(output_file)

        assert not output_file.exists()

    def test_generate_experiment_id(self) -> None:
        """Test experiment ID generation"""
        config1 = self._create_mock_config(
            DataScale.SMALL, Dimension(128), SearchType.PURE_VECTOR
        )
        config2 = self._create_mock_config(
            DataScale.MEDIUM, Dimension(256), SearchType.HYBRID
        )
        config3 = ExperimentConfig(
            data_scale=DataScale.LARGE,
            dimension=Dimension(512),
            search_type=SearchType.PURE_VECTOR,
            filter_config=FilterConfig(enabled=True),
            hnsw_params=HNSWParams(),
            batch_size=100,
            num_queries=10,
        )

        id1 = self.checkpoint_manager._generate_experiment_id(config1)
        id2 = self.checkpoint_manager._generate_experiment_id(config2)
        id3 = self.checkpoint_manager._generate_experiment_id(config3)

        assert id1 == "s128vn"  # small, 128d, vector, no filter
        assert id2 == "m256hn"  # medium, 256d, hybrid, no filter
        assert id3 == "l512vf"  # large, 512d, vector, filter

        assert id1 != id2 != id3

    def test_update_progress_tracking(self) -> None:
        """Test progress tracking updates"""
        config = self._create_mock_config(DataScale.SMALL, Dimension(128))
        results = [self._create_mock_result(config)]

        self.checkpoint_manager.save_batch_results(results)

        assert self.checkpoint_manager.progress_file.exists()

        with open(self.checkpoint_manager.progress_file, "r") as f:
            progress_data = json.load(f)

        assert "completed_experiments" in progress_data
        assert "start_time" in progress_data
        assert "last_updated" in progress_data
        assert len(progress_data["completed_experiments"]) == 1

        completed_exp = progress_data["completed_experiments"][0]
        assert completed_exp["data_scale"] == "SMALL"
        assert completed_exp["dimension"] == 128
        assert "experiment_id" in completed_exp
        assert "completed_at" in completed_exp

    def test_update_checkpoint_metadata(self) -> None:
        """Test checkpoint metadata updates"""
        config = self._create_mock_config(DataScale.SMALL, Dimension(128))
        results = [self._create_mock_result(config) for _ in range(3)]

        self.checkpoint_manager.save_batch_results(results)

        assert self.checkpoint_manager.metadata_file.exists()

        with open(self.checkpoint_manager.metadata_file, "r") as f:
            metadata = json.load(f)

        assert metadata["total_results"] == 3
        assert metadata["checkpoint_version"] == "1.0"
        assert metadata["completion_percentage"] == (3 / 48.0) * 100
        assert "last_updated" in metadata

    def test_estimate_experiment_duration(self) -> None:
        """Test experiment duration estimation"""
        config = self._create_mock_config(DataScale.SMALL, Dimension(128))

        insert_metrics = Metrics(
            query_time_ms=1000.0,
            throughput_qps=1000.0,
            memory_usage_mb=500.0,
            index_size_mb=100.0,
        )
        index_metrics = Metrics(
            query_time_ms=2000.0,
            throughput_qps=0.0,
            memory_usage_mb=600.0,
            index_size_mb=150.0,
        )

        result = ExperimentResult(
            config=config,
            insert_metrics=insert_metrics,
            index_metrics=index_metrics,
            search_results=[],
            accuracy=AccuracyMetrics(
                recall_at_1=0.9,
                recall_at_5=0.95,
                recall_at_10=0.98,
                mean_reciprocal_rank=0.92,
            ),
            timestamp=datetime.now(),
        )

        duration = self.checkpoint_manager._estimate_experiment_duration(result)
        assert duration == 3.0  # 3 seconds total (1000ms + 2000ms)

    def test_calculate_avg_query_time(self) -> None:
        """Test average query time calculation"""
        search_results: List[SearchResult] = []
        avg_time = self.checkpoint_manager._calculate_avg_query_time(search_results)
        assert avg_time == 0.0

        avg_time = self.checkpoint_manager._calculate_avg_query_time([])
        assert avg_time == 0.0

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


class TestCheckpointError:
    """Test cases for CheckpointError exception"""

    def test_checkpoint_error_creation(self) -> None:
        """Test CheckpointError exception creation"""
        error = CheckpointError("Test error message")
        assert str(error) == "Test error message"
        assert isinstance(error, Exception)
