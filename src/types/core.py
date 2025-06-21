from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import List, NewType, Optional, Tuple
import math

Dimension = NewType("Dimension", int)
VectorData = NewType("VectorData", List[float])
Distance = NewType("Distance", float)


@dataclass(frozen=True)
class Vector:
    """불변 벡터 표현"""

    dimension: Dimension
    data: VectorData

    def __post_init__(self) -> None:
        if len(self.data) != self.dimension:
            raise ValueError(
                f"Vector dimension mismatch: expected {self.dimension}, got {len(self.data)}"
            )
        if any(math.isnan(x) for x in self.data):
            raise ValueError("Vector contains NaN values")


class Category(Enum):
    NEWS = "news"
    REVIEW = "review"
    DOCUMENT = "document"
    SOCIAL = "social"


@dataclass(frozen=True)
class TextContent:
    """한국어 텍스트 콘텐츠"""

    text: str
    title: str
    category: Category
    created_at: datetime


@dataclass(frozen=True)
class Document:
    """텍스트와 벡터가 결합된 문서"""

    id: str
    content: TextContent
    vector: Vector


class DataScale(Enum):
    SMALL = 10_000
    MEDIUM = 100_000
    LARGE = 250_000


class SearchType(Enum):
    PURE_VECTOR = "pure_vector"
    HYBRID = "hybrid"


@dataclass(frozen=True)
class FilterConfig:
    """필터 설정"""

    enabled: bool
    category: Optional[Category] = None
    date_range: Optional[Tuple[datetime, datetime]] = None


@dataclass(frozen=True)
class HNSWParams:
    """HNSW 인덱스 파라미터"""

    ef_construction: int = 128
    ef_search: int = 64
    M: int = 16
    metric: str = "cosine"


@dataclass(frozen=True)
class DatabaseConfig:
    """데이터베이스 연결 설정"""

    database_path: str = ":memory:"
    memory_limit_mb: Optional[int] = None
    threads: Optional[int] = None


@dataclass(frozen=True)
class ExperimentConfig:
    """단일 실험 설정"""

    data_scale: DataScale
    dimension: Dimension
    search_type: SearchType
    filter_config: FilterConfig
    hnsw_params: HNSWParams
    batch_size: int = 1000
    num_queries: int = 100


@dataclass(frozen=True)
class Metrics:
    """성능 측정 결과"""

    query_time_ms: float
    throughput_qps: float
    memory_usage_mb: float
    index_size_mb: float


@dataclass(frozen=True)
class SearchResult:
    """검색 결과"""

    query_id: str
    retrieved_ids: List[str]
    distances: List[Distance]
    metrics: Metrics


@dataclass(frozen=True)
class AccuracyMetrics:
    """정확도 메트릭"""

    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    mean_reciprocal_rank: float


@dataclass(frozen=True)
class ExperimentResult:
    """실험 결과"""

    config: ExperimentConfig
    insert_metrics: Metrics
    index_metrics: Metrics
    search_results: List[SearchResult]
    accuracy: AccuracyMetrics
    timestamp: datetime
