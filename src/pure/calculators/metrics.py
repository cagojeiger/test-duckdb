from typing import List
from ...types.core import SearchResult, AccuracyMetrics


def calculate_recall_at_k(
    retrieved_ids: List[str], relevant_ids: List[str], k: int
) -> float:
    """Recall@K 계산"""
    if not relevant_ids:
        return 0.0

    retrieved_k = retrieved_ids[:k]
    relevant_set = set(relevant_ids)
    retrieved_set = set(retrieved_k)

    intersection = len(relevant_set & retrieved_set)
    return intersection / len(relevant_set)


def calculate_mean_reciprocal_rank(
    retrieved_ids: List[str], relevant_ids: List[str]
) -> float:
    """Mean Reciprocal Rank 계산"""
    if not relevant_ids:
        return 0.0

    relevant_set = set(relevant_ids)

    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant_set:
            return 1.0 / (i + 1)

    return 0.0


def calculate_accuracy_metrics(
    search_results: List[SearchResult], ground_truth: dict[str, List[str]]
) -> AccuracyMetrics:
    """전체 정확도 메트릭 계산"""
    recall_1_scores = []
    recall_5_scores = []
    recall_10_scores = []
    mrr_scores = []

    for result in search_results:
        relevant_ids = ground_truth.get(result.query_id, [])

        recall_1 = calculate_recall_at_k(result.retrieved_ids, relevant_ids, 1)
        recall_5 = calculate_recall_at_k(result.retrieved_ids, relevant_ids, 5)
        recall_10 = calculate_recall_at_k(result.retrieved_ids, relevant_ids, 10)
        mrr = calculate_mean_reciprocal_rank(result.retrieved_ids, relevant_ids)

        recall_1_scores.append(recall_1)
        recall_5_scores.append(recall_5)
        recall_10_scores.append(recall_10)
        mrr_scores.append(mrr)

    return AccuracyMetrics(
        recall_at_1=sum(recall_1_scores) / len(recall_1_scores)
        if recall_1_scores
        else 0.0,
        recall_at_5=sum(recall_5_scores) / len(recall_5_scores)
        if recall_5_scores
        else 0.0,
        recall_at_10=sum(recall_10_scores) / len(recall_10_scores)
        if recall_10_scores
        else 0.0,
        mean_reciprocal_rank=sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0.0,
    )
