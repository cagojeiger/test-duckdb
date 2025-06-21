import pytest
from src.types.core import (
    Vector,
    Dimension,
    VectorData,
    SearchResult,
    Metrics,
    Distance,
)
from src.pure.calculators.distances import (
    cosine_distance,
    euclidean_distance,
    inner_product_distance,
    batch_distances,
)
from src.pure.calculators.metrics import (
    calculate_recall_at_k,
    calculate_mean_reciprocal_rank,
    calculate_accuracy_metrics,
)


class TestDistanceCalculations:
    """거리 계산 함수 테스트"""

    def test_cosine_distance_identical_vectors(self) -> None:
        """동일한 벡터의 코사인 거리는 0"""
        v1 = Vector(Dimension(3), VectorData([1.0, 0.0, 0.0]))
        v2 = Vector(Dimension(3), VectorData([1.0, 0.0, 0.0]))

        distance = cosine_distance(v1, v2)
        assert abs(distance) < 1e-10

    def test_cosine_distance_orthogonal_vectors(self) -> None:
        """직교 벡터의 코사인 거리는 1"""
        v1 = Vector(Dimension(2), VectorData([1.0, 0.0]))
        v2 = Vector(Dimension(2), VectorData([0.0, 1.0]))

        distance = cosine_distance(v1, v2)
        assert abs(distance - 1.0) < 1e-10

    def test_euclidean_distance_identical_vectors(self) -> None:
        """동일한 벡터의 유클리드 거리는 0"""
        v1 = Vector(Dimension(3), VectorData([1.0, 2.0, 3.0]))
        v2 = Vector(Dimension(3), VectorData([1.0, 2.0, 3.0]))

        distance = euclidean_distance(v1, v2)
        assert abs(distance) < 1e-10

    def test_euclidean_distance_known_case(self) -> None:
        """알려진 경우의 유클리드 거리"""
        v1 = Vector(Dimension(2), VectorData([0.0, 0.0]))
        v2 = Vector(Dimension(2), VectorData([3.0, 4.0]))

        distance = euclidean_distance(v1, v2)
        assert abs(distance - 5.0) < 1e-10

    def test_inner_product_distance(self) -> None:
        """내적 거리 계산"""
        v1 = Vector(Dimension(2), VectorData([1.0, 2.0]))
        v2 = Vector(Dimension(2), VectorData([3.0, 4.0]))

        distance = inner_product_distance(v1, v2)
        assert abs(distance - (-11.0)) < 1e-10

    def test_dimension_mismatch_error(self) -> None:
        """차원 불일치 시 에러 발생"""
        v1 = Vector(Dimension(2), VectorData([1.0, 2.0]))
        v2 = Vector(Dimension(3), VectorData([1.0, 2.0, 3.0]))

        with pytest.raises(ValueError):
            cosine_distance(v1, v2)

        with pytest.raises(ValueError):
            euclidean_distance(v1, v2)

        with pytest.raises(ValueError):
            inner_product_distance(v1, v2)

    def test_batch_distances(self) -> None:
        """배치 거리 계산"""
        query = Vector(Dimension(2), VectorData([1.0, 0.0]))
        vectors = [
            Vector(Dimension(2), VectorData([1.0, 0.0])),  # 동일
            Vector(Dimension(2), VectorData([0.0, 1.0])),  # 직교
            Vector(Dimension(2), VectorData([-1.0, 0.0])),  # 반대
        ]

        distances = batch_distances(query, vectors, "cosine")

        assert len(distances) == 3
        assert abs(distances[0]) < 1e-10  # 동일한 벡터
        assert abs(distances[1] - 1.0) < 1e-10  # 직교 벡터
        assert abs(distances[2] - 2.0) < 1e-10  # 반대 벡터


class TestAccuracyMetrics:
    """정확도 메트릭 테스트"""

    def test_calculate_recall_at_k_perfect(self) -> None:
        """완벽한 검색 결과의 Recall@K"""
        retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        relevant = ["doc1", "doc2", "doc3"]

        recall_1 = calculate_recall_at_k(retrieved, relevant, 1)
        recall_3 = calculate_recall_at_k(retrieved, relevant, 3)
        recall_5 = calculate_recall_at_k(retrieved, relevant, 5)

        assert abs(recall_1 - (1 / 3)) < 1e-10
        assert abs(recall_3 - 1.0) < 1e-10
        assert abs(recall_5 - 1.0) < 1e-10

    def test_calculate_recall_at_k_no_relevant(self) -> None:
        """관련 문서가 없는 경우"""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant: list[str] = []

        recall = calculate_recall_at_k(retrieved, relevant, 3)
        assert recall == 0.0

    def test_calculate_mean_reciprocal_rank(self) -> None:
        """MRR 계산 테스트"""
        retrieved = ["doc1", "doc2", "doc3", "doc4"]
        relevant = ["doc3", "doc5"]

        mrr = calculate_mean_reciprocal_rank(retrieved, relevant)
        assert abs(mrr - (1 / 3)) < 1e-10  # doc3이 3번째 위치

    def test_calculate_mean_reciprocal_rank_no_match(self) -> None:
        """매치되는 문서가 없는 경우"""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = ["doc4", "doc5"]

        mrr = calculate_mean_reciprocal_rank(retrieved, relevant)
        assert mrr == 0.0

    def test_calculate_accuracy_metrics(self) -> None:
        """전체 정확도 메트릭 계산"""
        search_results = [
            SearchResult(
                query_id="q1",
                retrieved_ids=["doc1", "doc2", "doc3"],
                distances=[Distance(0.1), Distance(0.2), Distance(0.3)],
                metrics=Metrics(1.0, 100.0, 50.0, 1000.0),
            ),
            SearchResult(
                query_id="q2",
                retrieved_ids=["doc4", "doc1", "doc5"],
                distances=[Distance(0.1), Distance(0.2), Distance(0.3)],
                metrics=Metrics(1.0, 100.0, 50.0, 1000.0),
            ),
        ]

        ground_truth = {"q1": ["doc1", "doc2"], "q2": ["doc1", "doc3"]}

        accuracy = calculate_accuracy_metrics(search_results, ground_truth)

        assert accuracy.recall_at_1 > 0.0
        assert accuracy.recall_at_5 > 0.0
        assert accuracy.recall_at_10 > 0.0
        assert accuracy.mean_reciprocal_rank > 0.0
