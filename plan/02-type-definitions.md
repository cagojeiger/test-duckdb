# 타입 정의 및 데이터 모델

## 개요

강타입 시스템을 통해 함수형 프로그래밍의 안정성을 높입니다. Python의 타입 힌팅과 dataclass를 활용하여 불변 데이터 구조를 정의합니다.

## 기본 타입

### 1. 벡터 관련 타입

```python
from typing import NewType, List, Tuple
from dataclasses import dataclass

# 기본 타입 별칭
Dimension = NewType('Dimension', int)  # 128, 256, 512, 1024
VectorData = NewType('VectorData', List[float])
Distance = NewType('Distance', float)

@dataclass(frozen=True)
class Vector:
    """불변 벡터 표현"""
    dimension: Dimension
    data: VectorData

    def __post_init__(self):
        assert len(self.data) == self.dimension
```

### 2. 텍스트 및 문서 타입

```python
from datetime import datetime
from enum import Enum

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
```

### 3. 실험 설정 타입

```python
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
    category: Optional[Category]
    date_range: Optional[Tuple[datetime, datetime]]

@dataclass(frozen=True)
class HNSWParams:
    """HNSW 인덱스 파라미터"""
    ef_construction: int = 128
    ef_search: int = 64
    M: int = 16
    metric: str = "cosine"

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
```

### 4. 메트릭 및 결과 타입

```python
@dataclass(frozen=True)
class Metrics:
    """성능 측정 결과"""
    elapsed_time: float  # seconds
    memory_used: float   # MB
    cpu_percent: float
    throughput: float    # items/sec

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
```

## Effect 타입

### 1. IO 모나드

```python
from typing import TypeVar, Generic, Callable
from abc import ABC, abstractmethod

T = TypeVar('T')
E = TypeVar('E')

class IO(Generic[T], ABC):
    """부수 효과를 캡슐화하는 IO 모나드"""
    @abstractmethod
    def run(self) -> T:
        """실제 효과 실행"""
        pass

    def map(self, f: Callable[[T], U]) -> 'IO[U]':
        """결과 변환"""
        return MappedIO(self, f)

    def flat_map(self, f: Callable[[T], 'IO[U]']) -> 'IO[U]':
        """효과 체이닝"""
        return FlatMappedIO(self, f)
```

### 2. Either 타입

```python
@dataclass(frozen=True)
class Left(Generic[E, T]):
    """실패를 나타내는 Either"""
    value: E

@dataclass(frozen=True)
class Right(Generic[E, T]):
    """성공을 나타내는 Either"""
    value: T

Either = Union[Left[E, T], Right[E, T]]

class EitherOps:
    """Either 연산 헬퍼"""
    @staticmethod
    def map(either: Either[E, T], f: Callable[[T], U]) -> Either[E, U]:
        if isinstance(either, Right):
            return Right(f(either.value))
        return either
```

### 3. Reader 모나드

```python
@dataclass(frozen=True)
class Reader(Generic[R, T]):
    """설정 의존성을 캡슐화하는 Reader 모나드"""
    run: Callable[[R], T]

    def map(self, f: Callable[[T], U]) -> 'Reader[R, U]':
        return Reader(lambda r: f(self.run(r)))

    def flat_map(self, f: Callable[[T], 'Reader[R, U]']) -> 'Reader[R, U]':
        return Reader(lambda r: f(self.run(r)).run(r))
```

## 복합 타입

### 1. 파이프라인 타입

```python
@dataclass(frozen=True)
class Pipeline(Generic[A, B]):
    """함수 파이프라인"""
    steps: List[Callable[[Any], Any]]

    def run(self, input: A) -> B:
        result = input
        for step in self.steps:
            result = step(result)
        return result
```

### 2. 배치 처리 타입

```python
@dataclass(frozen=True)
class Batch(Generic[T]):
    """배치 처리를 위한 컨테이너"""
    items: List[T]
    size: int

    def map(self, f: Callable[[T], U]) -> 'Batch[U]':
        return Batch([f(item) for item in self.items], self.size)
```

## 타입 별칭

```python
# 자주 사용되는 타입 조합
ConfigReader[T] = Reader[ExperimentConfig, T]
IOResult[T] = IO[Either[Exception, T]]
BatchResult = List[ExperimentResult]

# 함수 시그니처 타입
DataGenerator = Callable[[ExperimentConfig], IO[List[Document]]]
VectorSearch = Callable[[Vector, int], IO[List[SearchResult]]]
MetricsCollector = Callable[[IO[T]], IO[Tuple[T, Metrics]]]
```

## 타입 검증

```python
def validate_vector(vector: Vector) -> Either[str, Vector]:
    """벡터 유효성 검증"""
    if len(vector.data) != vector.dimension:
        return Left(f"Vector dimension mismatch")
    if any(math.isnan(x) for x in vector.data):
        return Left("Vector contains NaN values")
    return Right(vector)

def validate_config(config: ExperimentConfig) -> Either[str, ExperimentConfig]:
    """실험 설정 유효성 검증"""
    if config.hnsw_params.ef_search > config.hnsw_params.ef_construction:
        return Left("ef_search cannot exceed ef_construction")
    return Right(config)
```

이러한 타입 정의를 통해 컴파일 타임에 많은 오류를 잡을 수 있으며, 함수 시그니처가 명확해져 코드의 의도를 쉽게 파악할 수 있습니다.
