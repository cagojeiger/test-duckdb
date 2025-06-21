from typing import Callable, List, Tuple, TypeVar
from .core import (
    ExperimentConfig,
    ExperimentResult,
    Document,
    Vector,
    SearchResult,
    Metrics,
)
from .monads import IO, Either, Reader

T = TypeVar("T")

ConfigReader = Reader[ExperimentConfig, T]
IOResult = IO[Either[Exception, T]]
BatchResult = List[ExperimentResult]

DataGenerator = Callable[[ExperimentConfig], IO[List[Document]]]
VectorSearch = Callable[[Vector, int], IO[List[SearchResult]]]
MetricsCollector = Callable[[IO[T]], IO[Tuple[T, Metrics]]]
