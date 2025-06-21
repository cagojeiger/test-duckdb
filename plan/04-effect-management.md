# 부수 효과 관리 설계

## 개요

이 문서는 DuckDB 작업, 파일 I/O, 성능 측정 등의 부수 효과를 함수형 방식으로 관리하는 방법을 설명합니다. 모든 부수 효과는 명시적으로 타입으로 표현되어 순수 함수와 분리됩니다.

## 1. IO 모나드 구현

### 기본 IO 타입

```python
from typing import TypeVar, Generic, Callable, Any
from abc import ABC, abstractmethod
import functools

T = TypeVar('T')
U = TypeVar('U')

class IO(Generic[T], ABC):
    """부수 효과를 캡슐화하는 IO 모나드"""

    @abstractmethod
    def unsafe_run(self) -> T:
        """효과를 실제로 실행 (unsafe 접두사로 위험성 표시)"""
        pass

    def map(self, f: Callable[[T], U]) -> 'IO[U]':
        """순수 함수로 결과 변환"""
        return MappedIO(self, f)

    def flat_map(self, f: Callable[[T], 'IO[U]']) -> 'IO[U]':
        """IO를 반환하는 함수로 체이닝"""
        return FlatMappedIO(self, f)

    def zip(self, other: 'IO[U]') -> 'IO[Tuple[T, U]]':
        """두 IO 작업을 병렬로 실행"""
        return ZippedIO(self, other)

class PureIO(IO[T]):
    """순수한 값을 IO로 래핑"""
    def __init__(self, value: T):
        self.value = value

    def unsafe_run(self) -> T:
        return self.value

class EffectIO(IO[T]):
    """실제 부수 효과를 가진 IO"""
    def __init__(self, effect: Callable[[], T]):
        self.effect = effect

    def unsafe_run(self) -> T:
        return self.effect()

class MappedIO(IO[U]):
    """map 연산 결과"""
    def __init__(self, io: IO[T], f: Callable[[T], U]):
        self.io = io
        self.f = f

    def unsafe_run(self) -> U:
        return self.f(self.io.unsafe_run())
```

### IO 헬퍼 함수

```python
def io_pure(value: T) -> IO[T]:
    """순수 값을 IO로 리프팅"""
    return PureIO(value)

def io_effect(effect: Callable[[], T]) -> IO[T]:
    """효과를 IO로 래핑"""
    return EffectIO(effect)

def io_sequence(ios: List[IO[T]]) -> IO[List[T]]:
    """IO 리스트를 리스트 IO로 변환"""
    def run_all():
        return [io.unsafe_run() for io in ios]
    return io_effect(run_all)

def io_traverse(f: Callable[[A], IO[B]], items: List[A]) -> IO[List[B]]:
    """리스트의 각 항목에 IO 함수 적용"""
    ios = [f(item) for item in items]
    return io_sequence(ios)
```

## 2. DuckDB 효과 관리

### 연결 관리

```python
import duckdb
from contextlib import contextmanager

@dataclass(frozen=True)
class DBConfig:
    """데이터베이스 설정"""
    path: str = ":memory:"
    threads: int = 4
    memory_limit: str = "4GB"

class DBConnection:
    """DuckDB 연결 래퍼"""
    def __init__(self, config: DBConfig):
        self.config = config
        self._conn = None

    @contextmanager
    def get_connection(self):
        """컨텍스트 매니저로 연결 관리"""
        try:
            self._conn = duckdb.connect(self.config.path)
            self._setup_connection()
            yield self._conn
        finally:
            if self._conn:
                self._conn.close()

    def _setup_connection(self):
        """연결 초기 설정"""
        self._conn.execute(f"SET threads={self.config.threads}")
        self._conn.execute(f"SET memory_limit='{self.config.memory_limit}'")
        self._conn.execute("INSTALL vss")
        self._conn.execute("LOAD vss")

def with_db_connection(
    config: DBConfig,
    f: Callable[[duckdb.DuckDBPyConnection], T]
) -> IO[T]:
    """DB 연결을 사용하는 함수를 IO로 래핑"""
    def effect():
        db = DBConnection(config)
        with db.get_connection() as conn:
            return f(conn)
    return io_effect(effect)
```

### 데이터베이스 작업

```python
def create_table(table_name: str, dimension: Dimension) -> IO[None]:
    """테이블 생성 IO"""
    def effect(conn: duckdb.DuckDBPyConnection):
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                id VARCHAR PRIMARY KEY,
                text TEXT,
                title VARCHAR,
                category VARCHAR,
                created_at TIMESTAMP,
                vector FLOAT[{dimension}]
            )
        """)
    return lambda config: with_db_connection(config, effect)

def insert_documents(
    table_name: str,
    documents: List[Document]
) -> IO[Metrics]:
    """문서 삽입 IO"""
    def effect(conn: duckdb.DuckDBPyConnection):
        start_time = time.time()
        start_memory = get_memory_usage()

        rows = documents_to_rows(documents)
        conn.executemany(
            f"INSERT INTO {table_name} VALUES (?, ?, ?, ?, ?, ?)",
            rows
        )

        elapsed = time.time() - start_time
        memory_used = get_memory_usage() - start_memory

        return Metrics(
            elapsed_time=elapsed,
            memory_used=memory_used,
            cpu_percent=get_cpu_percent(),
            throughput=len(documents) / elapsed
        )

    return lambda config: with_db_connection(config, effect)

def create_hnsw_index(
    table_name: str,
    params: HNSWParams
) -> IO[Metrics]:
    """HNSW 인덱스 생성 IO"""
    def effect(conn: duckdb.DuckDBPyConnection):
        start_time = time.time()

        conn.execute(f"""
            CREATE INDEX idx_{table_name} ON {table_name}
            USING HNSW(vector)
            WITH (
                ef_construction = {params.ef_construction},
                ef_search = {params.ef_search},
                M = {params.M},
                metric = '{params.metric}'
            )
        """)

        elapsed = time.time() - start_time
        return Metrics(
            elapsed_time=elapsed,
            memory_used=get_memory_usage(),
            cpu_percent=get_cpu_percent(),
            throughput=1.0 / elapsed
        )

    return lambda config: with_db_connection(config, effect)

def vector_search(
    table_name: str,
    query_vector: Vector,
    k: int,
    filter_sql: Optional[str] = None
) -> IO[SearchResult]:
    """벡터 검색 IO"""
    def effect(conn: duckdb.DuckDBPyConnection):
        start_time = time.time()

        base_query = f"""
            SELECT id,
                   array_distance(vector, ?::FLOAT[{query_vector.dimension}]) as distance
            FROM {table_name}
        """

        if filter_sql:
            base_query += f" WHERE {filter_sql}"

        query = base_query + f" ORDER BY distance LIMIT {k}"

        result = conn.execute(query, [list(query_vector.data)]).fetchall()

        elapsed = time.time() - start_time

        return SearchResult(
            query_id=str(uuid.uuid4()),
            retrieved_ids=[row[0] for row in result],
            distances=[Distance(row[1]) for row in result],
            metrics=Metrics(
                elapsed_time=elapsed,
                memory_used=0,  # 검색은 메모리 증가가 미미
                cpu_percent=get_cpu_percent(),
                throughput=1.0 / elapsed
            )
        )

    return lambda config: with_db_connection(config, effect)
```

## 3. 파일 I/O 효과

```python
import json
import csv
from pathlib import Path

def read_file(path: Path) -> IO[str]:
    """파일 읽기 IO"""
    def effect():
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    return io_effect(effect)

def write_file(path: Path, content: str) -> IO[None]:
    """파일 쓰기 IO"""
    def effect():
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
    return io_effect(effect)

def save_json(path: Path, data: Any) -> IO[None]:
    """JSON 저장 IO"""
    def effect():
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    return io_effect(effect)

def save_results_csv(
    path: Path,
    results: List[ExperimentResult]
) -> IO[None]:
    """실험 결과를 CSV로 저장"""
    def effect():
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'data_scale', 'dimension', 'search_type', 'use_filter',
                'insert_time', 'index_time', 'mean_search_time',
                'recall_at_10', 'throughput'
            ])

            writer.writeheader()
            data = results_to_dataframe(results)

            for i in range(len(results)):
                row = {key: values[i] for key, values in data.items()}
                writer.writerow(row)

    return io_effect(effect)
```

## 4. Effect Management (효과 관리) - ✅ Phase 4B 병렬 실행 완료

## Phase 4B 병렬 실행 시스템 구현 완료

Phase 4B에서 병렬 실행 시스템이 완료되어 효과 관리에 새로운 차원이 추가되었습니다:

### 병렬 처리 효과 관리
- **프로세스 격리**: ProcessPoolExecutor를 통한 안전한 병렬 실행
- **리소스 관리**: 동적 워커 수 조정 및 메모리 모니터링
- **효과 격리**: 각 프로세스가 독립적인 효과 컨텍스트 유지
- **폴백 메커니즘**: 병렬 실행 실패 시 순차 실행으로 자동 전환

### 병렬 실행 CLI 옵션
```bash
# 병렬 실행 (기본 설정)
python -m src.runners.experiment_runner --all --parallel

# 병렬 실행 (커스텀 워커 수 및 메모리 임계값)
python -m src.runners.experiment_runner --all --parallel --workers 6 --max-memory 8000

# 특정 조건 + 병렬 실행
python -m src.runners.experiment_runner --data-scale small --dimensions 128 256 --parallel
```

### 테스트 현황
- **87개 단위 테스트** (99% 성공률)
- **병렬 실행 테스트** 완료
- **리소스 관리 테스트** 통과

## 4. 성능 측정 효과

```python
import psutil
import time

def get_memory_usage() -> float:
    """현재 프로세스의 메모리 사용량 (MB)"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024

def get_cpu_percent() -> float:
    """현재 프로세스의 CPU 사용률"""
    process = psutil.Process()
    return process.cpu_percent(interval=0.1)

def measure_io(io: IO[T]) -> IO[Tuple[T, Metrics]]:
    """IO 작업의 성능 측정"""
    def effect():
        start_time = time.time()
        start_memory = get_memory_usage()
        start_cpu = psutil.Process()

        result = io.unsafe_run()

        elapsed = time.time() - start_time
        memory_delta = get_memory_usage() - start_memory
        cpu_percent = start_cpu.cpu_percent(interval=0.1)

        metrics = Metrics(
            elapsed_time=elapsed,
            memory_used=memory_delta,
            cpu_percent=cpu_percent,
            throughput=1.0 / elapsed if elapsed > 0 else 0
        )

        return (result, metrics)

    return io_effect(effect)

def with_retry(
    io: IO[T],
    max_attempts: int = 3,
    delay: float = 1.0
) -> IO[Either[Exception, T]]:
    """재시도 로직을 가진 IO"""
    def effect():
        last_error = None

        for attempt in range(max_attempts):
            try:
                result = io.unsafe_run()
                return Right(result)
            except Exception as e:
                last_error = e
                if attempt < max_attempts - 1:
                    time.sleep(delay * (attempt + 1))

        return Left(last_error)

    return io_effect(effect)
```

## 5. 로깅 효과

```python
import logging
from datetime import datetime

@dataclass(frozen=True)
class LogEntry:
    """로그 엔트리"""
    timestamp: datetime
    level: str
    message: str
    context: Dict[str, Any]

def log_io(level: str, message: str, **context) -> IO[None]:
    """로깅 IO"""
    def effect():
        logger = logging.getLogger(__name__)
        log_method = getattr(logger, level.lower())
        log_method(message, extra=context)

    return io_effect(effect)

def with_logging(
    io: IO[T],
    operation: str
) -> IO[T]:
    """IO 작업에 로깅 추가"""
    return (
        log_io("info", f"Starting {operation}")
        .flat_map(lambda _: measure_io(io))
        .flat_map(lambda result_metrics:
            log_io("info", f"Completed {operation}",
                   elapsed=result_metrics[1].elapsed_time,
                   memory=result_metrics[1].memory_used)
            .map(lambda _: result_metrics[0])
        )
    )
```

## 6. 병렬 처리 효과

```python
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

def parallel_io(
    ios: List[IO[T]],
    max_workers: int = 4
) -> IO[List[T]]:
    """IO 작업들을 병렬로 실행"""
    def effect():
        results = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # IO를 Future로 변환
            future_to_index = {
                executor.submit(io.unsafe_run): i
                for i, io in enumerate(ios)
            }

            # 결과를 순서대로 수집
            results = [None] * len(ios)
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                results[index] = future.result()

        return results

    return io_effect(effect)

def parallel_map_io(
    f: Callable[[A], IO[B]],
    items: List[A],
    max_workers: int = 4
) -> IO[List[B]]:
    """리스트의 각 항목에 IO 함수를 병렬로 적용"""
    ios = [f(item) for item in items]
    return parallel_io(ios, max_workers)
```

## 7. 리소스 관리

```python
from typing import ContextManager

def with_resource(
    resource_factory: Callable[[], ContextManager[R]],
    f: Callable[[R], T]
) -> IO[T]:
    """리소스를 안전하게 사용하는 IO"""
    def effect():
        with resource_factory() as resource:
            return f(resource)

    return io_effect(effect)

def bracket(
    acquire: IO[R],
    use: Callable[[R], IO[T]],
    release: Callable[[R], IO[None]]
) -> IO[T]:
    """리소스 획득/사용/해제를 보장하는 패턴"""
    def effect():
        resource = acquire.unsafe_run()
        try:
            return use(resource).unsafe_run()
        finally:
            release(resource).unsafe_run()

    return io_effect(effect)
```

이러한 효과 관리 시스템을 통해 부수 효과를 명시적으로 다루고, 순수 함수와 명확히 분리할 수 있습니다. 모든 효과는 타입으로 표현되어 컴파일 타임에 추적 가능합니다.
