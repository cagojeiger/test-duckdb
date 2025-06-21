from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Generic, TypeVar, Union

T = TypeVar("T")
U = TypeVar("U")
E = TypeVar("E")
R = TypeVar("R")


class IO(Generic[T], ABC):
    """부수 효과를 캡슐화하는 IO 모나드"""

    @abstractmethod
    def run(self) -> T:
        """실제 효과 실행"""
        pass

    def map(self, f: Callable[[T], U]) -> IO[U]:
        """결과 변환"""
        return MappedIO(self, f)

    def flat_map(self, f: Callable[[T], IO[U]]) -> IO[U]:
        """효과 체이닝"""
        return FlatMappedIO(self, f)


@dataclass(frozen=True)
class PureIO(IO[T]):
    """순수 값을 IO로 래핑"""

    value: T

    def run(self) -> T:
        return self.value


@dataclass(frozen=True)
class MappedIO(IO[U], Generic[T, U]):
    """매핑된 IO"""

    io: IO[T]
    f: Callable[[T], U]

    def run(self) -> U:
        return self.f(self.io.run())


@dataclass(frozen=True)
class FlatMappedIO(IO[U], Generic[T, U]):
    """플랫 매핑된 IO"""

    io: IO[T]
    f: Callable[[T], IO[U]]

    def run(self) -> U:
        return self.f(self.io.run()).run()


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
        return Left(either.value)

    @staticmethod
    def flat_map(either: Either[E, T], f: Callable[[T], Either[E, U]]) -> Either[E, U]:
        if isinstance(either, Right):
            return f(either.value)
        return Left(either.value)

    @staticmethod
    def get_or_else(either: Either[E, T], default: T) -> T:
        if isinstance(either, Right):
            return either.value
        return default


@dataclass(frozen=True)
class Reader(Generic[R, T]):
    """설정 의존성을 캡슐화하는 Reader 모나드"""

    run: Callable[[R], T]

    def map(self, f: Callable[[T], U]) -> Reader[R, U]:
        return Reader(lambda r: f(self.run(r)))

    def flat_map(self, f: Callable[[T], Reader[R, U]]) -> Reader[R, U]:
        return Reader(lambda r: f(self.run(r)).run(r))


@dataclass(frozen=True)
class Batch(Generic[T]):
    """배치 처리를 위한 컨테이너"""

    items: list[T]
    size: int

    def map(self, f: Callable[[T], U]) -> Batch[U]:
        return Batch([f(item) for item in self.items], self.size)

    def flat_map(self, f: Callable[[T], list[U]]) -> Batch[U]:
        result = []
        for item in self.items:
            result.extend(f(item))
        return Batch(result, len(result))
