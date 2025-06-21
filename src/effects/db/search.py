"""Vector similarity search operations."""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
from src.types.monads import IO
from src.types.core import Vector, SearchResult, Distance, Metrics, Category
from src.effects.db.connection import DBConnection
import time


@dataclass(frozen=True)
class SearchQuery:
    """Vector search query configuration."""

    query_vector: Vector
    k: int = 10
    filter_category: Optional[Category] = None
    include_distances: bool = True


def vector_similarity_search(
    conn: DBConnection, table_name: str, query: SearchQuery
) -> IO[SearchResult]:
    """Perform vector similarity search using HNSW index."""

    def _search() -> SearchResult:
        start_time = time.time()

        base_sql = f"""
        SELECT id, title, content, category, created_at,
               array_distance(vector, ?::FLOAT[{query.query_vector.dimension}]) as distance
        FROM {table_name}
        """

        where_clause = ""
        params: list[str | list[float]] = [query.query_vector.data]

        if query.filter_category:
            where_clause = " WHERE category = ?"
            params.append(str(query.filter_category.value))

        full_sql = f"""
        {base_sql}{where_clause}
        ORDER BY distance ASC
        LIMIT {query.k}
        """

        results = conn.conn.execute(full_sql, params).fetchall()
        query_time = time.time() - start_time

        retrieved_ids = [row[0] for row in results]
        distances = [Distance(row[5]) for row in results]

        metrics = Metrics(
            query_time_ms=query_time * 1000,
            throughput_qps=1.0 / query_time if query_time > 0 else 0.0,
            memory_usage_mb=0.0,  # Will be calculated separately
            index_size_mb=0.0,  # Will be calculated separately
        )

        return SearchResult(
            query_id=f"query_{int(time.time() * 1000)}",
            retrieved_ids=retrieved_ids,
            distances=distances,
            metrics=metrics,
        )

    class VectorSearchIO(IO[SearchResult]):
        def run(self) -> SearchResult:
            return _search()

    return VectorSearchIO()


def hybrid_search(
    conn: DBConnection,
    table_name: str,
    query: SearchQuery,
    text_query: str,
    vector_weight: float = 0.7,
    text_weight: float = 0.3,
) -> IO[SearchResult]:
    """Perform hybrid search combining vector similarity and text search."""

    def _hybrid_search() -> SearchResult:
        start_time = time.time()

        base_sql = f"""
        WITH vector_scores AS (
            SELECT id, title, content, category, created_at,
                   array_distance(vector, ?::FLOAT[{query.query_vector.dimension}]) as vector_distance,
                   CASE WHEN content LIKE ? THEN 0.0 ELSE 1.0 END as text_distance
            FROM {table_name}
        ),
        combined_scores AS (
            SELECT *,
                   ({vector_weight} * vector_distance + {text_weight} * text_distance) as combined_score
            FROM vector_scores
        )
        """

        where_clause = ""
        params: list[str | list[float]] = [query.query_vector.data, f"%{text_query}%"]

        if query.filter_category:
            where_clause = " WHERE category = ?"
            params.append(str(query.filter_category.value))

        full_sql = f"""
        {base_sql}
        SELECT id, title, content, category, created_at, combined_score as distance
        FROM combined_scores
        {where_clause}
        ORDER BY combined_score ASC
        LIMIT {query.k}
        """

        results = conn.conn.execute(full_sql, params).fetchall()
        query_time = time.time() - start_time

        retrieved_ids = [row[0] for row in results]
        distances = [Distance(row[5]) for row in results]

        metrics = Metrics(
            query_time_ms=query_time * 1000,
            throughput_qps=1.0 / query_time if query_time > 0 else 0.0,
            memory_usage_mb=0.0,
            index_size_mb=0.0,
        )

        return SearchResult(
            query_id=f"hybrid_query_{int(time.time() * 1000)}",
            retrieved_ids=retrieved_ids,
            distances=distances,
            metrics=metrics,
        )

    class HybridSearchIO(IO[SearchResult]):
        def run(self) -> SearchResult:
            return _hybrid_search()

    return HybridSearchIO()


def batch_vector_search(
    conn: DBConnection, table_name: str, queries: List[SearchQuery]
) -> IO[List[SearchResult]]:
    """Perform multiple vector searches efficiently."""

    def _batch_search() -> List[SearchResult]:
        results = []
        for query in queries:
            search_io = vector_similarity_search(conn, table_name, query)
            result = search_io.run()
            results.append(result)
        return results

    class BatchSearchIO(IO[List[SearchResult]]):
        def run(self) -> List[SearchResult]:
            return _batch_search()

    return BatchSearchIO()


def get_database_stats(
    conn: DBConnection, table_name: str
) -> IO[dict[str, int | str | bool]]:
    """Get database and index statistics."""

    def _get_stats() -> dict[str, int | str | bool]:
        count_result = conn.conn.execute(
            f"SELECT COUNT(*) FROM {table_name}"
        ).fetchone()
        row_count = count_result[0] if count_result else 0

        stats = {
            "row_count": row_count,
            "table_name": table_name,
            "has_hnsw_index": True,  # Assume index exists for now
        }

        return stats

    class GetStatsIO(IO[dict[str, int | str | bool]]):
        def run(self) -> dict[str, int | str | bool]:
            return _get_stats()

    return GetStatsIO()
