"""Tests for function composition utilities."""

from src.pipelines.composition import (
    compose,
    pipe,
    kleisli_compose,
    io_pipe,
    lift_io,
    parallel_map,
    sequence_io,
    retry_io,
)
from src.types.monads import IO


class TestFunctionComposition:
    """함수 합성 테스트"""

    def test_compose(self) -> None:
        """함수 합성 테스트"""

        def add_one(x: int) -> int:
            return x + 1

        def multiply_two(x: int) -> int:
            return x * 2

        composed = compose(multiply_two, add_one)
        result = composed(5)

        assert result == 12  # (5 + 1) * 2

    def test_pipe(self) -> None:
        """파이프 연산 테스트"""

        def add_one(x: int) -> int:
            return x + 1

        def multiply_two(x: int) -> int:
            return x * 2

        def subtract_three(x: int) -> int:
            return x - 3

        result = pipe(5, add_one, multiply_two, subtract_three)

        assert result == 9  # ((5 + 1) * 2) - 3

    def test_lift_io(self) -> None:
        """순수 함수를 IO로 리프트 테스트"""

        def add_ten(x: int) -> int:
            return x + 10

        lifted = lift_io(add_ten)

        io_result = lifted(5)
        result = io_result.run()

        assert result == 15

    def test_io_pipe(self) -> None:
        """IO 파이프 테스트"""

        def add_io(x: int) -> IO[int]:
            class AddIO(IO[int]):
                def run(self) -> int:
                    return x + 1

            return AddIO()

        def multiply_io(x: int) -> IO[int]:
            class MultiplyIO(IO[int]):
                def run(self) -> int:
                    return x * 2

            return MultiplyIO()

        initial_io = lift_io(lambda x: x)(5)
        result_io = io_pipe(initial_io, add_io, multiply_io)
        result = result_io.run()

        assert result == 12  # (5 + 1) * 2

    def test_parallel_map(self) -> None:
        """병렬 맵 테스트"""
        numbers = [1, 2, 3, 4, 5]

        def square(x: int) -> int:
            return x * x

        results = parallel_map(square, numbers)

        assert results == [1, 4, 9, 16, 25]

    def test_sequence_io(self) -> None:
        """IO 시퀀스 테스트"""

        def create_io(value: int) -> IO[int]:
            class ValueIO(IO[int]):
                def run(self) -> int:
                    return value

            return ValueIO()

        io_list = [create_io(1), create_io(2), create_io(3)]
        sequence = sequence_io(io_list)
        result = sequence.run()

        assert result == [1, 2, 3]

    def test_retry_io(self) -> None:
        """IO 재시도 테스트"""
        attempt_count = 0

        def failing_io() -> IO[str]:
            nonlocal attempt_count

            class FailingIO(IO[str]):
                def run(self) -> str:
                    nonlocal attempt_count
                    attempt_count += 1
                    if attempt_count < 3:
                        raise Exception("Simulated failure")
                    return "success"

            return FailingIO()

        retry_action = retry_io(failing_io(), max_attempts=3)
        result = retry_action.run()

        assert result == "success"
        assert attempt_count == 3

    def test_kleisli_compose(self) -> None:
        """클라이슬리 합성 테스트"""

        def add_io(x: int) -> IO[int]:
            class AddIO(IO[int]):
                def run(self) -> int:
                    return x + 1

            return AddIO()

        def multiply_io(x: int) -> IO[int]:
            class MultiplyIO(IO[int]):
                def run(self) -> int:
                    return x * 2

            return MultiplyIO()

        composed = kleisli_compose(multiply_io, add_io)
        result_io = composed(5)
        result = result_io.run()

        assert result == 12  # (5 + 1) * 2
