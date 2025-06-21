# 순수 함수 설계

## 개요

이 문서는 DuckDB 벡터 검색 벤치마킹에서 사용될 순수 함수들을 정의합니다. 모든 함수는 부수 효과가 없으며, 같은 입력에 대해 항상 같은 출력을 반환합니다.

## 1. 데이터 생성 함수

### 텍스트 생성

```python
def generate_korean_text(
    seed: int,
    category: Category,
    length_range: Tuple[int, int] = (100, 500)
) -> str:
    """시드 기반 한국어 텍스트 생성 (Faker 대신 결정적 생성)"""
    # 시드를 기반으로 템플릿과 단어 선택
    # 순수 함수로 구현하기 위해 내부에서 랜덤 상태 관리
    pass

def generate_title(seed: int, category: Category) -> str:
    """카테고리별 제목 생성"""
    templates = {
        Category.NEWS: ["{}에서 {} 발표", "{} 관련 {} 소식"],
        Category.REVIEW: ["{} 제품 {} 리뷰", "{} 사용 {} 후기"],
        # ...
    }
    # 시드 기반 템플릿 선택 및 채우기
    pass

def create_text_content(
    seed: int,
    category: Category,
    timestamp: datetime
) -> TextContent:
    """완전한 텍스트 콘텐츠 생성"""
    return TextContent(
        text=generate_korean_text(seed, category),
        title=generate_title(seed * 2, category),
        category=category,
        created_at=timestamp
    )
```

### 벡터 생성

```python
def generate_vector(
    seed: int,
    dimension: Dimension,
    distribution: str = "normal"
) -> Vector:
    """시드 기반 정규화된 벡터 생성"""
    # 시드로부터 결정적 난수 생성
    values = deterministic_random(seed, dimension, distribution)
    normalized = normalize_vector(values)
    return Vector(dimension, VectorData(normalized))

def normalize_vector(values: List[float]) -> List[float]:
    """벡터 정규화 (L2 norm = 1)"""
    norm = math.sqrt(sum(x * x for x in values))
    return [x / norm for x in values] if norm > 0 else values

def create_document(
    doc_id: str,
    content: TextContent,
    vector: Vector
) -> Document:
    """문서 생성"""
    return Document(
        id=doc_id,
        content=content,
        vector=vector
    )
```

## 2. 거리 계산 함수

```python
def cosine_distance(v1: Vector, v2: Vector) -> Distance:
    """코사인 거리 계산"""
    assert v1.dimension == v2.dimension
    dot_product = sum(a * b for a, b in zip(v1.data, v2.data))
    # 정규화된 벡터의 경우 norm은 1
    return Distance(1.0 - dot_product)

def euclidean_distance(v1: Vector, v2: Vector) -> Distance:
    """유클리드 거리 계산"""
    assert v1.dimension == v2.dimension
    sum_sq = sum((a - b) ** 2 for a, b in zip(v1.data, v2.data))
    return Distance(math.sqrt(sum_sq))

def inner_product_distance(v1: Vector, v2: Vector) -> Distance:
    """내적 기반 거리 (유사도의 역)"""
    assert v1.dimension == v2.dimension
    dot_product = sum(a * b for a, b in zip(v1.data, v2.data))
    return Distance(-dot_product)  # 음수로 변환하여 거리화

# 거리 함수 선택
def get_distance_function(metric: str) -> Callable[[Vector, Vector], Distance]:
    """메트릭 이름으로 거리 함수 반환"""
    functions = {
        "cosine": cosine_distance,
        "l2sq": euclidean_distance,
        "ip": inner_product_distance
    }
    return functions.get(metric, cosine_distance)
```

## 3. 검색 결과 처리 함수

```python
def calculate_recall_at_k(
    retrieved: List[str],
    relevant: List[str],
    k: int
) -> float:
    """Recall@K 계산"""
    if not relevant:
        return 0.0

    retrieved_k = set(retrieved[:k])
    relevant_set = set(relevant)
    intersection = retrieved_k & relevant_set

    return len(intersection) / len(relevant_set)

def calculate_mrr(
    retrieved: List[str],
    relevant: List[str]
) -> float:
    """Mean Reciprocal Rank 계산"""
    for i, doc_id in enumerate(retrieved):
        if doc_id in relevant:
            return 1.0 / (i + 1)
    return 0.0

def aggregate_search_metrics(
    results: List[SearchResult],
    ground_truth: Dict[str, List[str]]
) -> AccuracyMetrics:
    """검색 결과에서 정확도 메트릭 집계"""
    recalls_1 = []
    recalls_5 = []
    recalls_10 = []
    mrrs = []

    for result in results:
        relevant = ground_truth.get(result.query_id, [])
        retrieved = result.retrieved_ids

        recalls_1.append(calculate_recall_at_k(retrieved, relevant, 1))
        recalls_5.append(calculate_recall_at_k(retrieved, relevant, 5))
        recalls_10.append(calculate_recall_at_k(retrieved, relevant, 10))
        mrrs.append(calculate_mrr(retrieved, relevant))

    return AccuracyMetrics(
        recall_at_1=sum(recalls_1) / len(recalls_1),
        recall_at_5=sum(recalls_5) / len(recalls_5),
        recall_at_10=sum(recalls_10) / len(recalls_10),
        mean_reciprocal_rank=sum(mrrs) / len(mrrs)
    )
```

## 4. 실험 조합 생성 함수

```python
def generate_experiment_configs() -> List[ExperimentConfig]:
    """48가지 실험 조합 생성"""
    configs = []

    for data_scale in DataScale:
        for dimension in [128, 256, 512, 1024]:
            for search_type in SearchType:
                for use_filter in [True, False]:
                    filter_config = FilterConfig(
                        enabled=use_filter,
                        category=Category.NEWS if use_filter else None,
                        date_range=None
                    )

                    config = ExperimentConfig(
                        data_scale=data_scale,
                        dimension=Dimension(dimension),
                        search_type=search_type,
                        filter_config=filter_config,
                        hnsw_params=HNSWParams()
                    )
                    configs.append(config)

    return configs

def partition_experiments(
    configs: List[ExperimentConfig],
    num_partitions: int
) -> List[List[ExperimentConfig]]:
    """실험을 균등하게 분할 (병렬 처리용) - Phase 4B 구현 완료 ✅"""
    partition_size = len(configs) // num_partitions
    partitions = []

    for i in range(num_partitions):
        start = i * partition_size
        end = start + partition_size if i < num_partitions - 1 else len(configs)
        partitions.append(configs[start:end])

    return partitions

# Phase 4B: 병렬 실행 관련 순수 함수들 (구현 완료)
def calculate_optimal_workers(
    available_memory_mb: int,
    memory_threshold_mb: int,
    cpu_count: int
) -> int:
    """시스템 리소스 기반 최적 워커 수 계산"""
    memory_based_workers = available_memory_mb // memory_threshold_mb
    cpu_based_workers = max(1, cpu_count - 1)  # 하나는 메인 프로세스용 예약
    return min(memory_based_workers, cpu_based_workers, 8)  # 최대 8개 제한

def batch_configs_for_parallel(
    configs: List[ExperimentConfig],
    batch_size: int
) -> List[List[ExperimentConfig]]:
    """병렬 실행을 위한 설정 배치 분할"""
    batches = []
    for i in range(0, len(configs), batch_size):
        batch = configs[i:i + batch_size]
        batches.append(batch)
    return batches
```

## 5. 메트릭 계산 함수

```python
def calculate_throughput(
    num_items: int,
    elapsed_time: float
) -> float:
    """처리량 계산 (items/sec)"""
    return num_items / elapsed_time if elapsed_time > 0 else 0.0

def calculate_percentile(
    values: List[float],
    percentile: float
) -> float:
    """백분위수 계산"""
    if not values:
        return 0.0

    sorted_values = sorted(values)
    index = int(len(sorted_values) * percentile / 100)
    return sorted_values[min(index, len(sorted_values) - 1)]

def aggregate_metrics(metrics_list: List[Metrics]) -> Dict[str, float]:
    """메트릭 리스트 집계"""
    if not metrics_list:
        return {}

    times = [m.elapsed_time for m in metrics_list]
    memories = [m.memory_used for m in metrics_list]
    cpus = [m.cpu_percent for m in metrics_list]

    return {
        "mean_time": sum(times) / len(times),
        "p50_time": calculate_percentile(times, 50),
        "p95_time": calculate_percentile(times, 95),
        "p99_time": calculate_percentile(times, 99),
        "mean_memory": sum(memories) / len(memories),
        "max_memory": max(memories),
        "mean_cpu": sum(cpus) / len(cpus),
        "max_cpu": max(cpus)
    }
```

## 6. 데이터 변환 함수

```python
def batch_documents(
    documents: List[Document],
    batch_size: int
) -> List[Batch[Document]]:
    """문서를 배치로 분할"""
    batches = []
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        batches.append(Batch(batch, len(batch)))
    return batches

def documents_to_rows(documents: List[Document]) -> List[Dict[str, Any]]:
    """문서를 DB 행으로 변환"""
    return [
        {
            "id": doc.id,
            "text": doc.content.text,
            "title": doc.content.title,
            "category": doc.content.category.value,
            "created_at": doc.content.created_at.isoformat(),
            "vector": list(doc.vector.data)
        }
        for doc in documents
    ]

def results_to_dataframe(results: List[ExperimentResult]) -> Dict[str, List[Any]]:
    """실험 결과를 데이터프레임 형식으로 변환"""
    data = {
        "data_scale": [],
        "dimension": [],
        "search_type": [],
        "use_filter": [],
        "insert_time": [],
        "index_time": [],
        "mean_search_time": [],
        "recall_at_10": [],
        "throughput": []
    }

    for result in results:
        data["data_scale"].append(result.config.data_scale.value)
        data["dimension"].append(result.config.dimension)
        data["search_type"].append(result.config.search_type.value)
        data["use_filter"].append(result.config.filter_config.enabled)
        data["insert_time"].append(result.insert_metrics.elapsed_time)
        data["index_time"].append(result.index_metrics.elapsed_time)

        search_times = [sr.metrics.elapsed_time for sr in result.search_results]
        data["mean_search_time"].append(sum(search_times) / len(search_times))
        data["recall_at_10"].append(result.accuracy.recall_at_10)
        data["throughput"].append(result.insert_metrics.throughput)

    return data
```

## 7. 검증 함수

```python
def validate_vector_dimension(
    vector: Vector,
    expected: Dimension
) -> Either[str, Vector]:
    """벡터 차원 검증"""
    if vector.dimension != expected:
        return Left(f"Expected dimension {expected}, got {vector.dimension}")
    return Right(vector)

def validate_batch_size(
    batch: Batch[T],
    max_size: int
) -> Either[str, Batch[T]]:
    """배치 크기 검증"""
    if batch.size > max_size:
        return Left(f"Batch size {batch.size} exceeds maximum {max_size}")
    return Right(batch)

def validate_experiment_result(
    result: ExperimentResult
) -> Either[str, ExperimentResult]:
    """실험 결과 유효성 검증"""
    if result.accuracy.recall_at_1 > result.accuracy.recall_at_5:
        return Left("Recall@1 cannot exceed Recall@5")
    if result.accuracy.recall_at_5 > result.accuracy.recall_at_10:
        return Left("Recall@5 cannot exceed Recall@10")
    if not 0 <= result.accuracy.mean_reciprocal_rank <= 1:
        return Left("MRR must be between 0 and 1")
    return Right(result)
```

## 8. 병렬 처리 지원 함수 (Phase 4B 완료)

```python
def merge_parallel_results(
    results_list: List[List[ExperimentResult]]
) -> List[ExperimentResult]:
    """병렬 실행 결과 병합"""
    merged = []
    for results in results_list:
        merged.extend(results)
    return merged

def calculate_parallel_efficiency(
    sequential_time: float,
    parallel_time: float,
    worker_count: int
) -> Dict[str, float]:
    """병렬 실행 효율성 계산"""
    speedup = sequential_time / parallel_time if parallel_time > 0 else 0
    efficiency = speedup / worker_count if worker_count > 0 else 0
    
    return {
        "speedup": speedup,
        "efficiency": efficiency,
        "parallel_overhead": max(0, (worker_count * parallel_time) - sequential_time)
    }

def validate_parallel_config(config: ParallelConfig) -> Either[str, ParallelConfig]:
    """병렬 실행 설정 검증"""
    if config.max_workers <= 0:
        return Left("max_workers must be positive")
    if config.memory_threshold_mb <= 0:
        return Left("memory_threshold_mb must be positive")
    if config.experiment_timeout_seconds <= 0:
        return Left("experiment_timeout_seconds must be positive")
    return Right(config)
```

이러한 순수 함수들은 테스트가 쉽고, 병렬 처리가 안전하며, 재사용이 가능합니다. 각 함수는 명확한 입력과 출력을 가지며, 외부 상태에 의존하지 않습니다. Phase 4B에서 추가된 병렬 처리 함수들은 멀티프로세싱 환경에서 안전하게 동작하도록 설계되었습니다.
