# 파이프라인 합성 설계

## 개요

이 문서는 순수 함수와 효과를 조합하여 완전한 실험 파이프라인을 구성하는 방법을 설명합니다. 함수 합성을 통해 복잡한 워크플로우를 간단하고 테스트 가능한 단위로 구성합니다.

## 1. 기본 합성 연산자

### 함수 합성 유틸리티

```python
from typing import TypeVar, Callable, List, Tuple
import functools

A = TypeVar('A')
B = TypeVar('B')
C = TypeVar('C')

def compose(f: Callable[[B], C], g: Callable[[A], B]) -> Callable[[A], C]:
    """두 함수를 합성 (g 다음 f 실행)"""
    return lambda x: f(g(x))

def pipe(*functions: Callable) -> Callable:
    """함수들을 순차적으로 연결 (왼쪽에서 오른쪽)"""
    return functools.reduce(compose, functions)

def identity(x: A) -> A:
    """항등 함수"""
    return x

def curry(f: Callable) -> Callable:
    """함수를 커링된 형태로 변환"""
    @functools.wraps(f)
    def curried(*args, **kwargs):
        if len(args) + len(kwargs) >= f.__code__.co_argcount:
            return f(*args, **kwargs)
        return lambda *more_args, **more_kwargs: curried(
            *args, *more_args, **{**kwargs, **more_kwargs}
        )
    return curried
```

### IO 합성 연산자

```python
def kleisli_compose(
    f: Callable[[B], IO[C]],
    g: Callable[[A], IO[B]]
) -> Callable[[A], IO[C]]:
    """Kleisli 화살표 합성 (IO를 반환하는 함수들의 합성)"""
    return lambda a: g(a).flat_map(f)

def io_pipe(*io_functions: Callable[..., IO]) -> Callable[..., IO]:
    """IO 함수들을 순차적으로 연결"""
    return functools.reduce(kleisli_compose, io_functions)

def lift_io(f: Callable[[A], B]) -> Callable[[A], IO[B]]:
    """순수 함수를 IO 함수로 변환"""
    return lambda a: io_pure(f(a))
```

## 2. 실험 파이프라인 구성

### 데이터 준비 파이프라인

```python
def data_preparation_pipeline(config: ExperimentConfig) -> IO[List[Document]]:
    """데이터 생성 및 준비 파이프라인"""

    # 순수 함수 단계
    def generate_documents(seed: int) -> List[Document]:
        documents = []

        for i in range(config.data_scale.value):
            # 텍스트 생성
            content = create_text_content(
                seed=seed + i,
                category=Category.NEWS,  # 또는 랜덤 선택
                timestamp=datetime.now()
            )

            # 벡터 생성
            vector = generate_vector(
                seed=seed + i + 1000000,
                dimension=config.dimension
            )

            # 문서 생성
            doc = create_document(
                doc_id=f"doc_{i}",
                content=content,
                vector=vector
            )

            documents.append(doc)

        return documents

    # IO로 리프팅
    return io_pure(42).map(generate_documents)  # 시드 42 사용

def data_insertion_pipeline(
    config: ExperimentConfig,
    documents: List[Document]
) -> IO[Tuple[str, Metrics]]:
    """데이터 삽입 파이프라인"""

    table_name = f"vectors_{config.data_scale.name}_{config.dimension}"

    # 테이블 생성 → 배치 삽입 → 메트릭 반환
    return (
        create_table(table_name, config.dimension)(config)
        .flat_map(lambda _:
            # 배치로 나누어 삽입
            io_traverse(
                lambda batch: insert_documents(table_name, batch.items)(config),
                batch_documents(documents, config.batch_size)
            )
        )
        .map(lambda metrics_list: (
            table_name,
            aggregate_metrics(metrics_list)
        ))
    )
```

### 인덱스 구축 파이프라인

```python
def index_building_pipeline(
    config: ExperimentConfig,
    table_name: str
) -> IO[Metrics]:
    """HNSW 인덱스 구축 파이프라인"""

    # 다양한 파라미터 조합 시도
    param_combinations = [
        HNSWParams(ef_construction=64, M=8),
        HNSWParams(ef_construction=128, M=16),
        HNSWParams(ef_construction=256, M=32)
    ]

    # 최적 파라미터 찾기 (선택적)
    if config.dimension <= 256:
        params = param_combinations[0]  # 작은 차원은 가벼운 파라미터
    elif config.dimension <= 512:
        params = param_combinations[1]
    else:
        params = param_combinations[2]  # 큰 차원은 무거운 파라미터

    return create_hnsw_index(table_name, params)(config)
```

### 검색 실행 파이프라인

```python
def search_execution_pipeline(
    config: ExperimentConfig,
    table_name: str,
    query_vectors: List[Vector]
) -> IO[List[SearchResult]]:
    """검색 실행 파이프라인"""

    # 필터 SQL 생성
    filter_sql = None
    if config.filter_config.enabled:
        if config.filter_config.category:
            filter_sql = f"category = '{config.filter_config.category.value}'"
        if config.filter_config.date_range:
            start, end = config.filter_config.date_range
            date_filter = f"created_at BETWEEN '{start}' AND '{end}'"
            filter_sql = f"{filter_sql} AND {date_filter}" if filter_sql else date_filter

    # 각 쿼리 벡터에 대해 검색 실행
    def search_single(query_vector: Vector) -> IO[SearchResult]:
        if config.search_type == SearchType.PURE_VECTOR:
            return vector_search(table_name, query_vector, 10, filter_sql)(config)
        else:
            # 하이브리드 검색 (벡터 + BM25)
            return hybrid_search(table_name, query_vector, 10, filter_sql)(config)

    # 병렬로 검색 실행
    return parallel_map_io(search_single, query_vectors, max_workers=4)
```

## 3. 전체 실험 파이프라인

### 단일 실험 파이프라인

```python
def single_experiment_pipeline(config: ExperimentConfig) -> IO[ExperimentResult]:
    """단일 실험 전체 파이프라인"""

    def run_experiment():
        # 1. 데이터 준비
        documents_io = data_preparation_pipeline(config)

        # 2. 데이터 삽입
        insertion_io = documents_io.flat_map(lambda docs:
            data_insertion_pipeline(config, docs)
        )

        # 3. 인덱스 구축
        index_io = insertion_io.flat_map(lambda result:
            index_building_pipeline(config, result[0])
                .map(lambda metrics: (result[0], result[1], metrics))
        )

        # 4. 쿼리 벡터 생성
        query_vectors = [
            generate_vector(seed=9999 + i, dimension=config.dimension)
            for i in range(config.num_queries)
        ]

        # 5. 검색 실행
        search_io = index_io.flat_map(lambda result:
            search_execution_pipeline(config, result[0], query_vectors)
                .map(lambda search_results: (*result, search_results))
        )

        # 6. 결과 집계
        return search_io.map(lambda result:
            ExperimentResult(
                config=config,
                insert_metrics=result[1],
                index_metrics=result[2],
                search_results=result[3],
                accuracy=calculate_accuracy(result[3]),  # Ground truth 필요
                timestamp=datetime.now()
            )
        )

    # 로깅과 에러 처리 추가
    return (
        with_logging(run_experiment(), f"Experiment {config}")
        .flat_map(lambda io: with_retry(io, max_attempts=3))
        .map(lambda either:
            either.value if isinstance(either, Right) else raise_error(either.value)
        )
    )
```

### 전체 실험 세트 파이프라인

```python
def full_experiment_pipeline(output_dir: Path) -> IO[List[ExperimentResult]]:
    """48개 실험 전체 실행 파이프라인"""

    # 모든 실험 설정 생성
    all_configs = generate_experiment_configs()

    # 실험을 배치로 나누어 실행 (메모리 관리)
    batches = partition_experiments(all_configs, num_partitions=4)

    def run_batch(batch: List[ExperimentConfig]) -> IO[List[ExperimentResult]]:
        """배치 단위 실험 실행"""
        return parallel_map_io(
            single_experiment_pipeline,
            batch,
            max_workers=2  # 동시 실행 제한
        )

    # 모든 배치 순차 실행
    results_io = io_traverse(run_batch, batches).map(
        lambda batch_results: [r for batch in batch_results for r in batch]
    )

    # 결과 저장
    return results_io.flat_map(lambda results:
        save_results_csv(output_dir / "results.csv", results)
            .flat_map(lambda _: save_json(
                output_dir / "results.json",
                [dataclasses.asdict(r) for r in results]
            ))
            .map(lambda _: results)
    )
```

## 4. 분석 파이프라인

### 결과 분석 파이프라인

```python
def analysis_pipeline(results: List[ExperimentResult]) -> IO[Dict[str, Any]]:
    """실험 결과 분석 파이프라인"""

    # 순수 함수로 분석
    def analyze_results():
        # 차원별 성능 분석
        by_dimension = group_by(results, lambda r: r.config.dimension)
        dimension_analysis = {
            dim: {
                "mean_search_time": mean([
                    sr.metrics.elapsed_time
                    for r in results
                    for sr in r.search_results
                ]),
                "mean_recall": mean([r.accuracy.recall_at_10 for r in results])
            }
            for dim, results in by_dimension.items()
        }

        # 검색 유형별 분석
        by_search_type = group_by(results, lambda r: r.config.search_type)
        search_type_analysis = {
            st.value: {
                "mean_time": mean([
                    sr.metrics.elapsed_time
                    for r in results
                    for sr in r.search_results
                ]),
                "throughput": mean([r.insert_metrics.throughput for r in results])
            }
            for st, results in by_search_type.items()
        }

        return {
            "total_experiments": len(results),
            "by_dimension": dimension_analysis,
            "by_search_type": search_type_analysis,
            "overall_metrics": {
                "total_time": sum(r.insert_metrics.elapsed_time for r in results),
                "mean_accuracy": mean([r.accuracy.recall_at_10 for r in results])
            }
        }

    # 분석 결과 저장
    analysis = analyze_results()
    return save_json(Path("analysis_report.json"), analysis).map(lambda _: analysis)
```

## 5. 고급 합성 패턴

### 조건부 파이프라인

```python
def conditional_pipeline(
    condition: Callable[[A], bool],
    if_true: Callable[[A], IO[B]],
    if_false: Callable[[A], IO[B]]
) -> Callable[[A], IO[B]]:
    """조건에 따라 다른 파이프라인 실행"""
    return lambda a: if_true(a) if condition(a) else if_false(a)

# 사용 예시
optimized_search = conditional_pipeline(
    lambda config: config.dimension <= 256,
    lambda config: fast_search_pipeline(config),
    lambda config: accurate_search_pipeline(config)
)
```

### 재시도 파이프라인

```python
def retry_pipeline(
    pipeline: Callable[[A], IO[B]],
    max_attempts: int = 3,
    backoff_factor: float = 2.0
) -> Callable[[A], IO[Either[Exception, B]]]:
    """지수 백오프를 사용한 재시도 파이프라인"""

    def with_retry(input: A) -> IO[Either[Exception, B]]:
        def attempt(n: int) -> IO[Either[Exception, B]]:
            if n >= max_attempts:
                return io_pure(Left(Exception("Max attempts exceeded")))

            return pipeline(input).map(Right).flat_map(
                lambda result: io_pure(result) if isinstance(result, Right)
                else io_effect(lambda: time.sleep(backoff_factor ** n))
                    .flat_map(lambda _: attempt(n + 1))
            )

        return attempt(0)

    return with_retry
```

### 캐싱 파이프라인

```python
from functools import lru_cache

def cached_pipeline(
    pipeline: Callable[[A], IO[B]],
    cache_key: Callable[[A], str]
) -> Callable[[A], IO[B]]:
    """결과를 캐싱하는 파이프라인"""

    @lru_cache(maxsize=128)
    def cached_run(key: str) -> B:
        # 주의: IO의 실행 결과를 캐싱 (순수성 깨짐)
        return None

    def with_cache(input: A) -> IO[B]:
        key = cache_key(input)

        def effect():
            cached = cached_run.__wrapped__(key)
            if cached is not None:
                return cached

            result = pipeline(input).unsafe_run()
            cached_run.cache_info()  # 캐시 통계
            return result

        return io_effect(effect)

    return with_cache
```

## 6. 파이프라인 테스트

```python
def test_pipeline(
    pipeline: Callable[[A], IO[B]],
    test_input: A,
    expected_output: B,
    mock_effects: Dict[str, Any]
) -> bool:
    """파이프라인 단위 테스트"""

    # Mock IO 구현
    class MockIO(IO[T]):
        def __init__(self, value: T):
            self.value = value

        def unsafe_run(self) -> T:
            return self.value

    # 효과를 Mock으로 대체
    with patch.multiple('effects', **mock_effects):
        result = pipeline(test_input).unsafe_run()
        return result == expected_output
```

이러한 파이프라인 합성 패턴을 통해 복잡한 실험 워크플로우를 관리 가능한 단위로 구성하고, 각 단계를 독립적으로 테스트할 수 있습니다.
