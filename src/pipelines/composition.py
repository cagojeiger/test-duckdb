"""Function composition utilities for building complex workflows."""

from __future__ import annotations
from typing import TypeVar, Callable, Any
from functools import reduce
from src.types.monads import IO, Either

A = TypeVar("A")
B = TypeVar("B")
C = TypeVar("C")
E = TypeVar("E")


def compose(f: Callable[[B], C], g: Callable[[A], B]) -> Callable[[A], C]:
    """Compose two functions: compose(f, g)(x) = f(g(x))"""
    return lambda x: f(g(x))


def pipe(value: A, *functions: Callable[[Any], Any]) -> Any:
    """Pipe a value through a sequence of functions."""
    return reduce(lambda acc, f: f(acc), functions, value)


def kleisli_compose(
    f: Callable[[B], IO[C]], g: Callable[[A], IO[B]]
) -> Callable[[A], IO[C]]:
    """Compose two IO-returning functions using Kleisli composition."""

    def composed(x: A) -> IO[C]:
        return g(x).flat_map(f)

    return composed


def io_pipe(io_value: IO[A], *io_functions: Callable[[A], IO[Any]]) -> IO[Any]:
    """Pipe an IO value through a sequence of IO-returning functions."""
    return reduce(lambda acc, f: acc.flat_map(f), io_functions, io_value)


def lift_io(pure_function: Callable[[A], B]) -> Callable[[A], IO[B]]:
    """Lift a pure function into the IO monad."""

    def lifted(x: A) -> IO[B]:
        class LiftedIO(IO[B]):
            def run(self) -> B:
                return pure_function(x)

        return LiftedIO()

    return lifted


def either_pipe(
    either_value: Either[E, A], *either_functions: Callable[[A], Either[E, Any]]
) -> Either[E, Any]:
    """Pipe an Either value through a sequence of Either-returning functions."""
    return reduce(lambda acc, f: acc.flat_map(f), either_functions, either_value)


def sequence_io(io_list: list[IO[A]]) -> IO[list[A]]:
    """Convert a list of IO actions into an IO action of a list."""

    def run_sequence() -> list[A]:
        return [io_action.run() for io_action in io_list]

    class SequenceIO(IO[list[A]]):
        def run(self) -> list[A]:
            return run_sequence()

    return SequenceIO()


def parallel_map(func: Callable[[A], B], items: list[A]) -> list[B]:
    """Apply a function to each item in a list (sequential for now, can be parallelized later)."""
    return [func(item) for item in items]


def io_parallel_map(func: Callable[[A], IO[B]], items: list[A]) -> IO[list[B]]:
    """Apply an IO function to each item in a list and collect results."""
    io_results = [func(item) for item in items]
    return sequence_io(io_results)


def retry_io(io_action: IO[A], max_attempts: int = 3) -> IO[A]:
    """Retry an IO action up to max_attempts times."""

    def retry_logic() -> A:
        last_exception = None
        for attempt in range(max_attempts):
            try:
                return io_action.run()
            except Exception as e:
                last_exception = e
                if attempt == max_attempts - 1:
                    raise
        raise last_exception or Exception("Retry failed")

    class RetryIO(IO[A]):
        def run(self) -> A:
            return retry_logic()

    return RetryIO()


def conditional_io(
    condition: Callable[[A], bool],
    then_action: Callable[[A], IO[B]],
    else_action: Callable[[A], IO[B]],
) -> Callable[[A], IO[B]]:
    """Conditional execution of IO actions."""

    def conditional(x: A) -> IO[B]:
        if condition(x):
            return then_action(x)
        else:
            return else_action(x)

    return conditional
