# 실험 워크플로우 설계

## 개요

이 문서는 DuckDB 벡터 검색 벤치마킹의 전체 실험 워크플로우를 단계별로 설명합니다. 각 단계는 독립적인 함수로 설계되어 테스트와 재사용이 가능합니다.

## 1. 실험 워크플로우 개요

```
┌─────────────────┐
│ 실험 설정 생성  │ → 48가지 조합 생성
└────────┬────────┘
         │
┌────────▼────────┐
│ 데이터 생성     │ → 한국어 텍스트 + 벡터
└────────┬────────┘
         │
┌────────▼────────┐
│ DB 초기화       │ → 테이블 생성, VSS 로드
└────────┬────────┘
         │
┌────────▼────────┐
│ 데이터 삽입     │ → 배치 단위 삽입
└────────┬────────┘
         │
┌────────▼────────┐
│ 인덱스 구축     │ → HNSW 인덱스 생성
└────────┬────────┘
         │
┌────────▼────────┐
│ 검색 실행       │ → 쿼리 실행 및 측정
└────────┬────────┘
         │
┌────────▼────────┐
│ 결과 수집       │ → 메트릭 집계
└────────┬────────┘
         │
┌────────▼────────┐
│ 분석 및 리포트  │ → 시각화 및 보고서
└─────────────────┘
```

## 2. 단계별 상세 설계

### 2.1 실험 설정 생성

```python
def create_experiment_matrix() -> List[ExperimentConfig]:
    """48가지 실험 매트릭스 생성"""
    
    configs = []
    experiment_id = 0
    
    # 모든 조합 생성
    for data_scale in [DataScale.SMALL, DataScale.MEDIUM, DataScale.LARGE]:
        for dimension in [128, 256, 512, 1024]:
            for search_type in [SearchType.PURE_VECTOR, SearchType.HYBRID]:
                for use_filter in [False, True]:
                    
                    # 필터 설정
                    if use_filter:
                        # 30% 선택도를 가진 필터
                        filter_config = FilterConfig(
                            enabled=True,
                            category=Category.NEWS,
                            date_range=(
                                datetime.now() - timedelta(days=7),
                                datetime.now()
                            )
                        )
                    else:
                        filter_config = FilterConfig(enabled=False)
                    
                    # HNSW 파라미터 (차원에 따라 조정)
                    hnsw_params = optimize_hnsw_params(dimension)
                    
                    config = ExperimentConfig(
                        experiment_id=f"exp_{experiment_id:03d}",
                        data_scale=data_scale,
                        dimension=Dimension(dimension),
                        search_type=search_type,
                        filter_config=filter_config,
                        hnsw_params=hnsw_params,
                        batch_size=calculate_optimal_batch_size(data_scale),
                        num_queries=100
                    )
                    
                    configs.append(config)
                    experiment_id += 1
    
    return configs

def optimize_hnsw_params(dimension: int) -> HNSWParams:
    """차원에 따른 최적 HNSW 파라미터 결정"""
    if dimension <= 256:
        return HNSWParams(
            ef_construction=64,
            ef_search=32,
            M=8,
            metric="cosine"
        )
    elif dimension <= 512:
        return HNSWParams(
            ef_construction=128,
            ef_search=64,
            M=16,
            metric="cosine"
        )
    else:
        return HNSWParams(
            ef_construction=256,
            ef_search=128,
            M=32,
            metric="cosine"
        )

def calculate_optimal_batch_size(data_scale: DataScale) -> int:
    """데이터 규모에 따른 최적 배치 크기"""
    return {
        DataScale.SMALL: 1000,
        DataScale.MEDIUM: 5000,
        DataScale.LARGE: 10000
    }[data_scale]
```

### 2.2 데이터 생성 단계

```python
def generate_experiment_data(config: ExperimentConfig) -> IO[ExperimentData]:
    """실험용 데이터 생성"""
    
    def create_data():
        # 시드 설정 (재현 가능성)
        base_seed = hash(config.experiment_id) % 1000000
        
        # 문서 생성
        documents = []
        for i in range(config.data_scale.value):
            # 카테고리 분포 (실제 데이터 모방)
            category_weights = {
                Category.NEWS: 0.3,
                Category.REVIEW: 0.25,
                Category.DOCUMENT: 0.25,
                Category.SOCIAL: 0.2
            }
            category = weighted_choice(category_weights, seed=base_seed + i)
            
            # 텍스트 생성
            content = create_text_content(
                seed=base_seed + i,
                category=category,
                timestamp=generate_timestamp(base_seed + i)
            )
            
            # 벡터 생성 (텍스트 임베딩 시뮬레이션)
            vector = generate_text_embedding_vector(
                text=content.text,
                seed=base_seed + i + 1000000,
                dimension=config.dimension
            )
            
            doc = create_document(
                doc_id=f"{config.experiment_id}_doc_{i}",
                content=content,
                vector=vector
            )
            
            documents.append(doc)
        
        # 쿼리 생성 (실제 검색 시나리오)
        queries = []
        for i in range(config.num_queries):
            query_seed = base_seed + 2000000 + i
            
            # 쿼리 타입 (신규 쿼리 vs 기존 문서 변형)
            if i < config.num_queries // 2:
                # 새로운 쿼리
                query_text = generate_korean_text(
                    seed=query_seed,
                    category=Category.NEWS,
                    length_range=(50, 100)
                )
            else:
                # 기존 문서 변형 (유사 검색 테스트)
                source_doc = documents[i % len(documents)]
                query_text = modify_text(source_doc.content.text, query_seed)
            
            query_vector = generate_text_embedding_vector(
                text=query_text,
                seed=query_seed,
                dimension=config.dimension
            )
            
            queries.append(Query(
                query_id=f"{config.experiment_id}_query_{i}",
                text=query_text,
                vector=query_vector
            ))
        
        # Ground truth 생성 (평가용)
        ground_truth = generate_ground_truth(documents, queries)
        
        return ExperimentData(
            config=config,
            documents=documents,
            queries=queries,
            ground_truth=ground_truth
        )
    
    return io_effect(create_data)
```

### 2.3 데이터베이스 초기화

```python
def initialize_database(config: ExperimentConfig) -> IO[DBContext]:
    """데이터베이스 초기화 및 설정"""
    
    def init_db(conn: duckdb.DuckDBPyConnection) -> DBContext:
        # 메모리 설정 (데이터 크기에 따라)
        memory_limit = {
            DataScale.SMALL: "2GB",
            DataScale.MEDIUM: "4GB",
            DataScale.LARGE: "8GB"
        }[config.data_scale]
        
        # DuckDB 설정
        conn.execute(f"SET memory_limit='{memory_limit}'")
        conn.execute(f"SET threads={multiprocessing.cpu_count()}")
        
        # VSS 확장 로드
        conn.execute("INSTALL vss")
        conn.execute("LOAD vss")
        
        # 테이블 생성
        table_name = f"vectors_{config.experiment_id}"
        conn.execute(f"""
            CREATE TABLE {table_name} (
                id VARCHAR PRIMARY KEY,
                text TEXT,
                title VARCHAR,
                category VARCHAR,
                created_at TIMESTAMP,
                vector FLOAT[{config.dimension}]
            )
        """)
        
        # 텍스트 검색을 위한 추가 인덱스 (하이브리드 검색용)
        if config.search_type == SearchType.HYBRID:
            conn.execute(f"""
                CREATE INDEX idx_text_{table_name} 
                ON {table_name}(text)
            """)
        
        return DBContext(
            connection=conn,
            table_name=table_name,
            config=config
        )
    
    db_config = DBConfig(
        path=f"experiments/{config.experiment_id}.db",
        threads=multiprocessing.cpu_count(),
        memory_limit="8GB"
    )
    
    return with_db_connection(db_config, init_db)
```

### 2.4 데이터 삽입 단계

```python
def insert_data_workflow(
    db_context: DBContext,
    data: ExperimentData
) -> IO[InsertionMetrics]:
    """데이터 삽입 워크플로우"""
    
    def insert_batches(conn: duckdb.DuckDBPyConnection) -> InsertionMetrics:
        total_time = 0.0
        total_memory = 0.0
        batch_metrics = []
        
        # 배치 단위로 삽입
        batches = batch_documents(data.documents, data.config.batch_size)
        
        for i, batch in enumerate(batches):
            # 배치 삽입 전 메모리 측정
            start_memory = get_memory_usage()
            start_time = time.time()
            
            # 데이터 변환
            rows = documents_to_rows(batch.items)
            
            # 벌크 삽입
            conn.executemany(
                f"""
                INSERT INTO {db_context.table_name} 
                (id, text, title, category, created_at, vector)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                rows
            )
            
            # 메트릭 수집
            elapsed = time.time() - start_time
            memory_delta = get_memory_usage() - start_memory
            
            batch_metric = BatchMetrics(
                batch_id=i,
                size=batch.size,
                elapsed_time=elapsed,
                memory_used=memory_delta,
                throughput=batch.size / elapsed
            )
            
            batch_metrics.append(batch_metric)
            total_time += elapsed
            total_memory += memory_delta
            
            # 진행 상황 로깅
            if (i + 1) % 10 == 0:
                print(f"Inserted {(i + 1) * data.config.batch_size} documents...")
        
        return InsertionMetrics(
            total_documents=len(data.documents),
            total_time=total_time,
            total_memory=total_memory,
            average_throughput=len(data.documents) / total_time,
            batch_metrics=batch_metrics
        )
    
    return with_db_connection(db_context.config, insert_batches)
```

### 2.5 인덱스 구축 단계

```python
def build_index_workflow(
    db_context: DBContext
) -> IO[IndexMetrics]:
    """HNSW 인덱스 구축 워크플로우"""
    
    def build_index(conn: duckdb.DuckDBPyConnection) -> IndexMetrics:
        params = db_context.config.hnsw_params
        
        # 인덱스 구축 전 상태
        start_time = time.time()
        start_memory = get_memory_usage()
        
        # CPU 사용률 모니터링 시작
        cpu_monitor = CPUMonitor()
        cpu_monitor.start()
        
        try:
            # HNSW 인덱스 생성
            conn.execute(f"""
                CREATE INDEX idx_hnsw_{db_context.table_name}
                ON {db_context.table_name}
                USING HNSW(vector)
                WITH (
                    ef_construction = {params.ef_construction},
                    ef_search = {params.ef_search},
                    M = {params.M},
                    M0 = {params.M * 2},
                    metric = '{params.metric}'
                )
            """)
            
            # 인덱스 통계
            index_size = get_index_size(conn, db_context.table_name)
            
        finally:
            cpu_monitor.stop()
        
        elapsed = time.time() - start_time
        memory_delta = get_memory_usage() - start_memory
        
        return IndexMetrics(
            build_time=elapsed,
            memory_used=memory_delta,
            index_size_mb=index_size / 1024 / 1024,
            cpu_stats=cpu_monitor.get_stats(),
            parameters=params
        )
    
    return with_db_connection(db_context.config, build_index)
```

### 2.6 검색 실행 단계

```python
def search_execution_workflow(
    db_context: DBContext,
    queries: List[Query]
) -> IO[List[SearchResult]]:
    """검색 실행 워크플로우"""
    
    def execute_searches(conn: duckdb.DuckDBPyConnection) -> List[SearchResult]:
        results = []
        
        for query in queries:
            # 검색 전 캐시 클리어 (공정한 측정)
            conn.execute("PRAGMA clear_cache")
            
            start_time = time.time()
            
            if db_context.config.search_type == SearchType.PURE_VECTOR:
                # 순수 벡터 검색
                result = execute_vector_search(
                    conn, 
                    db_context.table_name,
                    query.vector,
                    k=10,
                    filter_config=db_context.config.filter_config
                )
            else:
                # 하이브리드 검색 (벡터 + BM25)
                result = execute_hybrid_search(
                    conn,
                    db_context.table_name,
                    query.text,
                    query.vector,
                    k=10,
                    filter_config=db_context.config.filter_config,
                    vector_weight=0.7,
                    text_weight=0.3
                )
            
            elapsed = time.time() - start_time
            
            search_result = SearchResult(
                query_id=query.query_id,
                retrieved_ids=[row[0] for row in result],
                distances=[Distance(row[1]) for row in result],
                metrics=Metrics(
                    elapsed_time=elapsed,
                    memory_used=0,  # 검색은 메모리 증가 미미
                    cpu_percent=get_cpu_percent(),
                    throughput=1.0 / elapsed
                )
            )
            
            results.append(search_result)
        
        return results
    
    # 검색을 배치로 실행 (병렬 처리)
    query_batches = [queries[i:i+10] for i in range(0, len(queries), 10)]
    
    def run_batch(batch: List[Query]) -> IO[List[SearchResult]]:
        return with_db_connection(db_context.config, 
                                 lambda conn: execute_searches(conn))
    
    return io_traverse(run_batch, query_batches).map(
        lambda results: [r for batch in results for r in batch]
    )
```

### 2.7 결과 수집 및 분석

```python
def collect_and_analyze_results(
    config: ExperimentConfig,
    data: ExperimentData,
    insertion_metrics: InsertionMetrics,
    index_metrics: IndexMetrics,
    search_results: List[SearchResult]
) -> IO[ExperimentResult]:
    """결과 수집 및 분석"""
    
    def analyze():
        # 정확도 계산
        accuracy = aggregate_search_metrics(
            search_results,
            data.ground_truth
        )
        
        # 검색 성능 통계
        search_times = [r.metrics.elapsed_time for r in search_results]
        search_stats = {
            "mean": statistics.mean(search_times),
            "median": statistics.median(search_times),
            "p95": calculate_percentile(search_times, 95),
            "p99": calculate_percentile(search_times, 99),
            "std": statistics.stdev(search_times) if len(search_times) > 1 else 0
        }
        
        # 전체 실험 결과
        return ExperimentResult(
            config=config,
            insertion_metrics=insertion_metrics,
            index_metrics=index_metrics,
            search_results=search_results,
            search_statistics=search_stats,
            accuracy=accuracy,
            timestamp=datetime.now(),
            total_experiment_time=sum([
                insertion_metrics.total_time,
                index_metrics.build_time,
                sum(search_times)
            ])
        )
    
    return io_effect(analyze)
```

## 3. 실패 처리 및 복구

### 3.1 체크포인트 시스템

```python
@dataclass(frozen=True)
class ExperimentCheckpoint:
    """실험 진행 상태 체크포인트"""
    experiment_id: str
    stage: str  # "data_generated", "data_inserted", "index_built", etc.
    timestamp: datetime
    partial_results: Optional[Any]

def save_checkpoint(
    checkpoint: ExperimentCheckpoint,
    checkpoint_dir: Path
) -> IO[None]:
    """체크포인트 저장"""
    path = checkpoint_dir / f"{checkpoint.experiment_id}_{checkpoint.stage}.json"
    return save_json(path, dataclasses.asdict(checkpoint))

def load_checkpoint(
    experiment_id: str,
    stage: str,
    checkpoint_dir: Path
) -> IO[Optional[ExperimentCheckpoint]]:
    """체크포인트 로드"""
    path = checkpoint_dir / f"{experiment_id}_{stage}.json"
    
    def load():
        if not path.exists():
            return None
        
        with open(path, 'r') as f:
            data = json.load(f)
            return ExperimentCheckpoint(**data)
    
    return io_effect(load)
```

### 3.2 재시작 가능한 워크플로우

```python
def resumable_experiment_workflow(
    config: ExperimentConfig,
    checkpoint_dir: Path
) -> IO[ExperimentResult]:
    """재시작 가능한 실험 워크플로우"""
    
    # 각 단계를 체크포인트와 함께 실행
    def run_with_checkpoint(
        stage: str,
        operation: IO[T]
    ) -> IO[T]:
        return (
            load_checkpoint(config.experiment_id, stage, checkpoint_dir)
            .flat_map(lambda checkpoint:
                io_pure(checkpoint.partial_results) if checkpoint
                else operation.flat_map(lambda result:
                    save_checkpoint(
                        ExperimentCheckpoint(
                            experiment_id=config.experiment_id,
                            stage=stage,
                            timestamp=datetime.now(),
                            partial_results=result
                        ),
                        checkpoint_dir
                    ).map(lambda _: result)
                )
            )
        )
    
    # 단계별 실행
    return (
        run_with_checkpoint("data_generated", 
                          generate_experiment_data(config))
        .flat_map(lambda data:
            run_with_checkpoint("db_initialized",
                              initialize_database(config))
            .map(lambda db: (data, db))
        )
        .flat_map(lambda data_db:
            run_with_checkpoint("data_inserted",
                              insert_data_workflow(data_db[1], data_db[0]))
            .map(lambda metrics: (*data_db, metrics))
        )
        # ... 나머지 단계들
    )
```

## 4. 모니터링 및 진행 상황

```python
class ExperimentMonitor:
    """실험 진행 상황 모니터"""
    
    def __init__(self, total_experiments: int):
        self.total = total_experiments
        self.completed = 0
        self.failed = 0
        self.start_time = time.time()
    
    def update(self, experiment_id: str, status: str, message: str = ""):
        """진행 상황 업데이트"""
        if status == "completed":
            self.completed += 1
        elif status == "failed":
            self.failed += 1
        
        elapsed = time.time() - self.start_time
        rate = self.completed / elapsed if elapsed > 0 else 0
        eta = (self.total - self.completed) / rate if rate > 0 else float('inf')
        
        print(f"[{datetime.now()}] {experiment_id}: {status}")
        print(f"Progress: {self.completed}/{self.total} "
              f"(Failed: {self.failed}) "
              f"Rate: {rate:.2f} exp/sec "
              f"ETA: {eta/60:.1f} min")
        
        if message:
            print(f"  {message}")
```

이러한 워크플로우 설계를 통해 각 실험 단계를 독립적으로 관리하고, 실패 시 복구가 가능하며, 진행 상황을 모니터링할 수 있습니다.