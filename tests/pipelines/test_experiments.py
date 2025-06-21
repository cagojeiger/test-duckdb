"""Tests for experiment pipeline functionality."""

from unittest.mock import Mock, patch
from src.pipelines.experiments import (
    generate_experiment_matrix,
    generate_experiment_data,
    single_experiment_pipeline,
)
from src.types.core import DataScale, Dimension, SearchType


class TestExperimentPipelines:
    """실험 파이프라인 테스트"""

    def test_generate_experiment_matrix(self) -> None:
        """실험 매트릭스 생성 테스트"""
        configs = generate_experiment_matrix()

        assert len(configs) == 48

        config_hashes = [hash(str(config)) for config in configs]
        assert len(set(config_hashes)) == 48

        first_config = configs[0]
        assert first_config.data_scale == DataScale.SMALL
        assert first_config.dimension == Dimension(128)

    def test_generate_experiment_data(self) -> None:
        """실험 데이터 생성 테스트"""
        from src.types.core import ExperimentConfig

        from src.types.core import FilterConfig, HNSWParams

        config = ExperimentConfig(
            data_scale=DataScale.SMALL,
            dimension=Dimension(128),
            search_type=SearchType.PURE_VECTOR,
            filter_config=FilterConfig(enabled=False),
            hnsw_params=HNSWParams(),
            batch_size=1000,
            num_queries=100,
        )

        data_io = generate_experiment_data(config)
        data = data_io.run()

        assert len(data.documents) == 10_000
        assert len(data.query_vectors) == 100

        first_doc_vector = data.documents[0][2]
        assert first_doc_vector.dimension == Dimension(128)
        assert len(first_doc_vector.data) == 128

        first_query = data.query_vectors[0]
        assert first_query.dimension == Dimension(128)

    @patch("src.effects.db.connection.create_connection")
    def test_single_experiment_pipeline_structure(
        self, mock_create_connection: Mock
    ) -> None:
        """단일 실험 파이프라인 구조 테스트"""
        from src.types.core import ExperimentConfig

        mock_conn = Mock()
        mock_create_connection.return_value.run.return_value = mock_conn

        from src.types.core import FilterConfig, HNSWParams

        config = ExperimentConfig(
            data_scale=DataScale.SMALL,
            dimension=Dimension(128),
            search_type=SearchType.PURE_VECTOR,
            filter_config=FilterConfig(enabled=False),
            hnsw_params=HNSWParams(),
            batch_size=1000,
            num_queries=100,
        )

        pipeline_io = single_experiment_pipeline(config)

        assert hasattr(pipeline_io, "run")

    def test_experiment_config_combinations(self) -> None:
        """실험 설정 조합 검증"""
        configs = generate_experiment_matrix()

        small_configs = [c for c in configs if c.data_scale == DataScale.SMALL]
        medium_configs = [c for c in configs if c.data_scale == DataScale.MEDIUM]
        large_configs = [c for c in configs if c.data_scale == DataScale.LARGE]

        assert len(small_configs) == 16
        assert len(medium_configs) == 16
        assert len(large_configs) == 16

        dim_128_configs = [c for c in configs if c.dimension == Dimension(128)]
        dim_256_configs = [c for c in configs if c.dimension == Dimension(256)]
        dim_512_configs = [c for c in configs if c.dimension == Dimension(512)]
        dim_1024_configs = [c for c in configs if c.dimension == Dimension(1024)]

        assert len(dim_128_configs) == 12
        assert len(dim_256_configs) == 12
        assert len(dim_512_configs) == 12
        assert len(dim_1024_configs) == 12

        vector_configs = [c for c in configs if c.search_type == SearchType.PURE_VECTOR]
        hybrid_configs = [c for c in configs if c.search_type == SearchType.HYBRID]

        assert len(vector_configs) == 24
        assert len(hybrid_configs) == 24
