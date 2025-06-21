import math
import random
from typing import List
from ...types.core import Dimension, Vector, VectorData


def deterministic_random(
    seed: int, size: int, distribution: str = "normal"
) -> List[float]:
    """시드 기반 결정적 난수 생성"""
    rng = random.Random(seed)

    if distribution == "normal":
        values = []
        for i in range(size):
            if i % 2 == 0:
                u1 = rng.random()
                u2 = rng.random()
                z0 = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
                z1 = math.sqrt(-2 * math.log(u1)) * math.sin(2 * math.pi * u2)
                values.extend([z0, z1])

        return values[:size]

    elif distribution == "uniform":
        return [rng.uniform(-1, 1) for _ in range(size)]

    else:
        raise ValueError(f"Unsupported distribution: {distribution}")


def normalize_vector(values: List[float]) -> List[float]:
    """벡터 정규화 (L2 norm = 1)"""
    norm = math.sqrt(sum(x * x for x in values))
    return [x / norm for x in values] if norm > 0 else values


def generate_vector(
    seed: int, dimension: Dimension, distribution: str = "normal"
) -> Vector:
    """시드 기반 정규화된 벡터 생성"""
    values = deterministic_random(seed, dimension, distribution)
    normalized = normalize_vector(values)
    return Vector(dimension, VectorData(normalized))


def generate_random_vectors(
    count: int, dimension: Dimension, seed: int, distribution: str = "normal"
) -> List[Vector]:
    """다수의 랜덤 벡터 생성"""
    vectors = []
    for i in range(count):
        vector = generate_vector(seed + i, dimension, distribution)
        vectors.append(vector)
    return vectors


def generate_query_vectors(
    seed: int, dimension: Dimension, count: int, distribution: str = "normal"
) -> List[Vector]:
    """쿼리용 벡터들 생성"""
    vectors = []
    for i in range(count):
        vector = generate_vector(seed + i, dimension, distribution)
        vectors.append(vector)
    return vectors
