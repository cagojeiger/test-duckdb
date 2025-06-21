import pytest
from src.types.core import (
    DataScale,
    Dimension,
    SearchType,
    FilterConfig,
    HNSWParams,
    ExperimentConfig,
    Category,
)


@pytest.fixture(scope="session")
def experiment_configs() -> list[ExperimentConfig]:
    """48가지 실험 설정 생성"""
    configs = []

    data_scales = [DataScale.SMALL, DataScale.MEDIUM, DataScale.LARGE]
    dimensions = [Dimension(128), Dimension(256), Dimension(512), Dimension(1024)]
    search_types = [SearchType.PURE_VECTOR, SearchType.HYBRID]
    filter_configs = [
        FilterConfig(enabled=False),
        FilterConfig(enabled=True, category=Category.NEWS),
    ]

    for data_scale in data_scales:
        for dimension in dimensions:
            for search_type in search_types:
                for filter_config in filter_configs:
                    config = ExperimentConfig(
                        data_scale=data_scale,
                        dimension=dimension,
                        search_type=search_type,
                        filter_config=filter_config,
                        hnsw_params=HNSWParams(),
                        batch_size=1000,
                        num_queries=100,
                    )
                    configs.append(config)

    return configs


@pytest.fixture
def small_config() -> ExperimentConfig:
    """작은 규모 테스트용 설정"""
    return ExperimentConfig(
        data_scale=DataScale.SMALL,
        dimension=Dimension(128),
        search_type=SearchType.PURE_VECTOR,
        filter_config=FilterConfig(enabled=False),
        hnsw_params=HNSWParams(ef_construction=64, ef_search=32, M=8),
        batch_size=100,
        num_queries=10,
    )
