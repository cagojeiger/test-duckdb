# 구현 가이드라인

## 개요

이 문서는 Python에서 함수형 프로그래밍 패러다임을 사용하여 DuckDB 벡터 검색 벤치마킹을 구현하는 실용적인 가이드를 제공합니다.

## 1. Python 함수형 프로그래밍 도구

### 1.1 내장 함수형 도구

```python
from functools import partial, reduce, wraps, lru_cache
from itertools import chain, groupby, starmap, tee
from operator import itemgetter, attrgetter
from typing import TypeVar, Callable, Optional, Union, List, Tuple

# 함수 합성
def compose(*functions):
    """오른쪽에서 왼쪽으로 함수 합성"""
    return reduce(lambda f, g: lambda x: f(g(x)), functions, lambda x: x)

# 파이프라인 (왼쪽에서 오른쪽)
def pipe(*functions):
    """왼쪽에서 오른쪽으로 함수 연결"""
    return reduce(lambda f, g: lambda x: g(f(x)), functions, lambda x: x)

# 커링
def curry(func):
    """함수를 커링된 형태로 변환"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        if len(args) + len(kwargs) >= func.__code__.co_argcount:
            return func(*args, **kwargs)
        return partial(func, *args, **kwargs)
    return wrapper

# 사용 예시
@curry
def add(x, y):
    return x + y

add_five = add(5)  # 부분 적용
result = add_five(3)  # 8
```

### 1.2 불변 데이터 구조

```python
from dataclasses import dataclass, field, replace
from typing import FrozenSet, Tuple
import collections

# 불변 데이터클래스
@dataclass(frozen=True)
class ImmutableConfig:
    """불변 설정 객체"""
    dimension: int
    batch_size: int
    parameters: Tuple[str, ...] = field(default_factory=tuple)

    def with_dimension(self, new_dimension: int) -> 'ImmutableConfig':
        """새로운 차원으로 복사본 생성"""
        return replace(self, dimension=new_dimension)

# 불변 컬렉션
ImmutableDict = collections.namedtuple('ImmutableDict', ['data'])

def immutable_dict(**kwargs):
    """불변 딕셔너리 생성"""
    return ImmutableDict(data=frozenset(kwargs.items()))

def get_value(idict: ImmutableDict, key: str, default=None):
    """불변 딕셔너리에서 값 조회"""
    for k, v in idict.data:
        if k == key:
            return v
    return default

# 영속 데이터 구조 (pyrsistent 라이브러리 사용 권장)
from pyrsistent import pmap, pvector, pset

# 영속 맵
config = pmap({'dimension': 128, 'batch_size': 1000})
new_config = config.set('dimension', 256)  # 새로운 맵 반환

# 영속 벡터
vectors = pvector([1, 2, 3])
new_vectors = vectors.append(4)  # 새로운 벡터 반환
```

### 1.3 타입 힌팅 활용

```python
from typing import TypeVar, Generic, Protocol, runtime_checkable

# 제네릭 타입
T = TypeVar('T')
U = TypeVar('U')
E = TypeVar('E')

# 프로토콜 정의
@runtime_checkable
class Mappable(Protocol[T, U]):
    """map 메서드를 가진 타입"""
    def map(self, f: Callable[[T], U]) -> 'Mappable[U]':
        ...

# 타입 별칭
Vector = List[float]
Distance = float
DocumentId = str
SearchResults = List[Tuple[DocumentId, Distance]]

# 함수 시그니처 타입
DataGenerator = Callable[[int], List[Document]]
DistanceFunction = Callable[[Vector, Vector], Distance]
```

## 2. 함수형 패턴 구현

### 2.1 모나드 패턴

```python
from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Callable

T = TypeVar('T')
U = TypeVar('U')

class Monad(Generic[T], ABC):
    """모나드 베이스 클래스"""

    @abstractmethod
    def bind(self, f: Callable[[T], 'Monad[U]']) -> 'Monad[U]':
        """모나드 바인드 (flatMap)"""
        pass

    @abstractmethod
    def map(self, f: Callable[[T], U]) -> 'Monad[U]':
        """펑터 맵"""
        pass

    @classmethod
    @abstractmethod
    def pure(cls, value: T) -> 'Monad[T]':
        """값을 모나드로 리프팅"""
        pass

# Maybe 모나드 구현
class Maybe(Monad[T]):
    """옵셔널 값을 다루는 Maybe 모나드"""

    def __init__(self, value: Optional[T]):
        self._value = value

    def bind(self, f: Callable[[T], 'Maybe[U]']) -> 'Maybe[U]':
        if self._value is None:
            return Maybe(None)
        return f(self._value)

    def map(self, f: Callable[[T], U]) -> 'Maybe[U]':
        if self._value is None:
            return Maybe(None)
        return Maybe(f(self._value))

    @classmethod
    def pure(cls, value: T) -> 'Maybe[T]':
        return cls(value)

    def get_or_else(self, default: T) -> T:
        return self._value if self._value is not None else default

# 사용 예시
def safe_divide(x: float, y: float) -> Maybe[float]:
    if y == 0:
        return Maybe(None)
    return Maybe.pure(x / y)

result = (Maybe.pure(10)
          .bind(lambda x: safe_divide(x, 2))
          .bind(lambda x: safe_divide(x, 5))
          .get_or_else(0))  # 1.0
```

### 2.2 함수형 에러 처리

```python
from typing import Union, TypeVar, Generic, Callable

E = TypeVar('E')  # Error type
T = TypeVar('T')  # Success type

class Result(Generic[E, T]):
    """성공 또는 실패를 나타내는 Result 타입"""

    def __init__(self, value: Union[E, T], is_error: bool):
        self._value = value
        self._is_error = is_error

    @classmethod
    def success(cls, value: T) -> 'Result[E, T]':
        return cls(value, False)

    @classmethod
    def error(cls, error: E) -> 'Result[E, T]':
        return cls(error, True)

    def map(self, f: Callable[[T], U]) -> 'Result[E, U]':
        if self._is_error:
            return Result.error(self._value)
        return Result.success(f(self._value))

    def flat_map(self, f: Callable[[T], 'Result[E, U]']) -> 'Result[E, U]':
        if self._is_error:
            return Result.error(self._value)
        return f(self._value)

    def map_error(self, f: Callable[[E], F]) -> 'Result[F, T]':
        if self._is_error:
            return Result.error(f(self._value))
        return Result.success(self._value)

    def unwrap_or(self, default: T) -> T:
        return default if self._is_error else self._value

# 사용 예시
def parse_int(s: str) -> Result[str, int]:
    try:
        return Result.success(int(s))
    except ValueError:
        return Result.error(f"Cannot parse '{s}' as integer")

result = (parse_int("42")
          .map(lambda x: x * 2)
          .flat_map(lambda x: parse_int(str(x)))
          .unwrap_or(0))  # 84
```

### 2.3 지연 평가

```python
from typing import Callable, TypeVar, Generic

T = TypeVar('T')

class Lazy(Generic[T]):
    """지연 평가를 위한 Lazy 컨테이너"""

    def __init__(self, computation: Callable[[], T]):
        self._computation = computation
        self._value = None
        self._computed = False

    def get(self) -> T:
        """값을 계산하고 캐싱"""
        if not self._computed:
            self._value = self._computation()
            self._computed = True
        return self._value

    def map(self, f: Callable[[T], U]) -> 'Lazy[U]':
        """지연 맵핑"""
        return Lazy(lambda: f(self.get()))

    def flat_map(self, f: Callable[[T], 'Lazy[U]']) -> 'Lazy[U]':
        """지연 바인드"""
        return Lazy(lambda: f(self.get()).get())

# 무한 시퀀스
def lazy_range(start: int = 0) -> Lazy[List[int]]:
    """무한 범위를 지연 평가"""
    def generate(n: int):
        current = start
        result = []
        for _ in range(n):
            result.append(current)
            current += 1
        return result

    return Lazy(lambda: generate)

# 사용 예시
expensive_computation = Lazy(lambda: sum(range(1000000)))
# 실제로 필요할 때까지 계산하지 않음
result = expensive_computation.map(lambda x: x * 2).get()
```

## 3. 실용적인 구현 팁

### 3.1 부분 적용과 파이프라인

```python
from functools import partial
from typing import List, Dict, Any

# 설정 가능한 함수들
def filter_by_category(category: str, documents: List[Document]) -> List[Document]:
    return [doc for doc in documents if doc.content.category.value == category]

def sort_by_date(reverse: bool, documents: List[Document]) -> List[Document]:
    return sorted(documents,
                  key=lambda d: d.content.created_at,
                  reverse=reverse)

def limit_results(n: int, documents: List[Document]) -> List[Document]:
    return documents[:n]

# 파이프라인 구성
news_pipeline = pipe(
    partial(filter_by_category, "news"),
    partial(sort_by_date, reverse=True),
    partial(limit_results, 10)
)

# 사용
recent_news = news_pipeline(all_documents)
```

### 3.2 메모이제이션

```python
from functools import lru_cache, wraps
import hashlib
import json

def memoize_to_disk(cache_dir: Path):
    """디스크 기반 메모이제이션 데코레이터"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 캐시 키 생성
            key_data = json.dumps((args, kwargs), sort_keys=True)
            key = hashlib.md5(key_data.encode()).hexdigest()
            cache_file = cache_dir / f"{func.__name__}_{key}.json"

            # 캐시 확인
            if cache_file.exists():
                with open(cache_file, 'r') as f:
                    return json.load(f)

            # 계산 실행
            result = func(*args, **kwargs)

            # 캐시 저장
            cache_dir.mkdir(parents=True, exist_ok=True)
            with open(cache_file, 'w') as f:
                json.dump(result, f)

            return result

        return wrapper
    return decorator

# 사용 예시
@memoize_to_disk(Path("cache"))
def expensive_analysis(config: Dict[str, Any]) -> Dict[str, float]:
    # 비용이 큰 분석 작업
    pass
```

### 3.3 병렬 처리

```python
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import List, Callable, TypeVar
import multiprocessing

T = TypeVar('T')
U = TypeVar('U')

def parallel_map(
    func: Callable[[T], U],
    items: List[T],
    max_workers: Optional[int] = None,
    use_threads: bool = False
) -> List[U]:
    """병렬 맵 구현"""
    if max_workers is None:
        max_workers = multiprocessing.cpu_count()

    Executor = ThreadPoolExecutor if use_threads else ProcessPoolExecutor

    with Executor(max_workers=max_workers) as executor:
        # 순서 보장
        return list(executor.map(func, items))

def parallel_filter(
    predicate: Callable[[T], bool],
    items: List[T],
    max_workers: Optional[int] = None
) -> List[T]:
    """병렬 필터 구현"""
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # (item, keep) 튜플 생성
        results = executor.map(lambda x: (x, predicate(x)), items)
        # True인 항목만 필터링
        return [item for item, keep in results if keep]

# 사용 예시
vectors = parallel_map(
    lambda i: generate_vector(seed=i, dimension=512),
    range(10000),
    max_workers=8
)
```

### 3.4 리소스 관리

```python
from contextlib import contextmanager
from typing import ContextManager, TypeVar, Callable

R = TypeVar('R')  # Resource type
T = TypeVar('T')  # Result type

@contextmanager
def managed_resource(
    acquire: Callable[[], R],
    release: Callable[[R], None]
) -> ContextManager[R]:
    """리소스 관리를 위한 컨텍스트 매니저"""
    resource = None
    try:
        resource = acquire()
        yield resource
    finally:
        if resource is not None:
            release(resource)

# 사용 예시
def with_temp_db(func: Callable[[duckdb.DuckDBPyConnection], T]) -> T:
    """임시 DB를 사용하는 함수 실행"""
    with managed_resource(
        acquire=lambda: duckdb.connect(":memory:"),
        release=lambda conn: conn.close()
    ) as conn:
        return func(conn)

result = with_temp_db(lambda conn:
    conn.execute("SELECT 1").fetchone()[0]
)
```

### 3.5 타입 안전성

```python
from typing import TypeVar, Protocol, runtime_checkable
from dataclasses import dataclass

# 타입 제약
T = TypeVar('T', bound='Comparable')

@runtime_checkable
class Comparable(Protocol):
    """비교 가능한 타입"""
    def __lt__(self, other: 'Comparable') -> bool: ...
    def __eq__(self, other: 'Comparable') -> bool: ...

def find_min(items: List[T]) -> Optional[T]:
    """타입 안전한 최소값 찾기"""
    if not items:
        return None
    return min(items)

# 제네릭 컨테이너
@dataclass(frozen=True)
class Box(Generic[T]):
    """제네릭 박스 컨테이너"""
    value: T

    def map(self, f: Callable[[T], U]) -> 'Box[U]':
        return Box(f(self.value))

    def flat_map(self, f: Callable[[T], 'Box[U]']) -> 'Box[U]':
        return f(self.value)
```

## 4. 성능 최적화

### 4.1 제너레이터 활용

```python
from typing import Generator, Iterator

def chunked_reader(
    file_path: Path,
    chunk_size: int = 1024 * 1024
) -> Generator[str, None, None]:
    """대용량 파일을 청크 단위로 읽기"""
    with open(file_path, 'r', encoding='utf-8') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            yield chunk

def process_large_dataset(
    file_path: Path,
    processor: Callable[[str], T]
) -> Iterator[T]:
    """대용량 데이터셋을 스트리밍 처리"""
    for chunk in chunked_reader(file_path):
        yield processor(chunk)
```

### 4.2 벡터화 연산

```python
import numpy as np
from typing import List

def vectorized_cosine_distance(
    query: np.ndarray,
    vectors: np.ndarray
) -> np.ndarray:
    """벡터화된 코사인 거리 계산"""
    # 정규화된 벡터 가정
    dots = np.dot(vectors, query)
    return 1.0 - dots

# 배치 처리
def batch_vector_search(
    queries: List[Vector],
    database: np.ndarray,
    k: int = 10
) -> List[List[int]]:
    """배치 벡터 검색"""
    query_matrix = np.array(queries)

    # 모든 쿼리에 대한 거리 계산
    distances = 1.0 - np.dot(database, query_matrix.T)

    # 각 쿼리별 top-k 인덱스
    results = []
    for i in range(len(queries)):
        top_k_indices = np.argpartition(distances[:, i], k)[:k]
        top_k_indices = top_k_indices[np.argsort(distances[top_k_indices, i])]
        results.append(top_k_indices.tolist())

    return results
```

이러한 구현 가이드라인을 따르면 함수형 프로그래밍의 장점을 활용하면서도 Python의 실용적인 기능들을 효과적으로 사용할 수 있습니다.
