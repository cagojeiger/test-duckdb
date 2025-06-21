import math
from typing import List
from ...types.core import Vector, Distance


def cosine_distance(v1: Vector, v2: Vector) -> Distance:
    """코사인 거리 계산 (1 - cosine_similarity)"""
    if v1.dimension != v2.dimension:
        raise ValueError(
            f"Vector dimensions must match: {v1.dimension} != {v2.dimension}"
        )

    dot_product = sum(a * b for a, b in zip(v1.data, v2.data))
    norm_v1 = math.sqrt(sum(x * x for x in v1.data))
    norm_v2 = math.sqrt(sum(x * x for x in v2.data))

    if norm_v1 == 0 or norm_v2 == 0:
        return Distance(1.0)

    cosine_similarity = dot_product / (norm_v1 * norm_v2)
    return Distance(1.0 - cosine_similarity)


def euclidean_distance(v1: Vector, v2: Vector) -> Distance:
    """유클리드 거리 계산"""
    if v1.dimension != v2.dimension:
        raise ValueError(
            f"Vector dimensions must match: {v1.dimension} != {v2.dimension}"
        )

    squared_diff = sum((a - b) ** 2 for a, b in zip(v1.data, v2.data))
    return Distance(math.sqrt(squared_diff))


def inner_product_distance(v1: Vector, v2: Vector) -> Distance:
    """내적 거리 계산 (음의 내적)"""
    if v1.dimension != v2.dimension:
        raise ValueError(
            f"Vector dimensions must match: {v1.dimension} != {v2.dimension}"
        )

    dot_product = sum(a * b for a, b in zip(v1.data, v2.data))
    return Distance(-dot_product)


def batch_distances(
    query: Vector, vectors: List[Vector], metric: str = "cosine"
) -> List[Distance]:
    """배치 거리 계산"""
    distance_func = {
        "cosine": cosine_distance,
        "euclidean": euclidean_distance,
        "inner_product": inner_product_distance,
    }[metric]

    return [distance_func(query, vector) for vector in vectors]
