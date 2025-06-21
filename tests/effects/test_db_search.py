"""Tests for database search operations."""

from unittest.mock import Mock, patch
from src.types.core import (
    DatabaseConfig,
    Dimension,
    Vector,
    VectorData,
    Category,
    Metrics,
)
from src.effects.db.connection import create_connection
from src.effects.db.search import SearchQuery, vector_similarity_search, hybrid_search


class TestVectorSearch:
    """벡터 검색 테스트"""

    def test_vector_similarity_search(self) -> None:
        """벡터 유사도 검색 테스트"""
        config = DatabaseConfig()
        query_vector = Vector(Dimension(128), VectorData([0.1] * 128))
        search_query = SearchQuery(query_vector, k=5)

        mock_results = [
            ("doc1", "제목1", "내용1", "news", "2024-01-01", 0.1),
            ("doc2", "제목2", "내용2", "review", "2024-01-02", 0.2),
        ]

        with patch("duckdb.connect") as mock_connect:
            mock_conn = Mock()
            mock_connect.return_value = mock_conn
            mock_conn.execute.return_value.fetchall.return_value = mock_results

            conn_io = create_connection(config)
            db_conn = conn_io.run()

            search_io = vector_similarity_search(db_conn, "test_table", search_query)
            result = search_io.run()

            assert len(result.retrieved_ids) == 2
            assert result.retrieved_ids == ["doc1", "doc2"]
            assert len(result.distances) == 2
            assert isinstance(result.metrics, Metrics)

    def test_vector_search_with_category_filter(self) -> None:
        """카테고리 필터가 있는 벡터 검색 테스트"""
        config = DatabaseConfig()
        query_vector = Vector(Dimension(128), VectorData([0.1] * 128))
        search_query = SearchQuery(query_vector, k=3, filter_category=Category.NEWS)

        mock_results = [("doc1", "제목1", "내용1", "news", "2024-01-01", 0.1)]

        with patch("duckdb.connect") as mock_connect:
            mock_conn = Mock()
            mock_connect.return_value = mock_conn
            mock_conn.execute.return_value.fetchall.return_value = mock_results

            conn_io = create_connection(config)
            db_conn = conn_io.run()

            search_io = vector_similarity_search(db_conn, "test_table", search_query)
            search_io.run()

            mock_conn.execute.assert_called()
            call_args = mock_conn.execute.call_args
            assert "WHERE category = ?" in call_args[0][0]

    def test_hybrid_search(self) -> None:
        """하이브리드 검색 테스트"""
        config = DatabaseConfig()
        query_vector = Vector(Dimension(128), VectorData([0.1] * 128))
        search_query = SearchQuery(query_vector, k=5)
        text_query = "테스트"

        mock_results = [
            ("doc1", "제목1", "내용1", "news", "2024-01-01", 0.15),
            ("doc2", "제목2", "내용2", "review", "2024-01-02", 0.25),
        ]

        with patch("duckdb.connect") as mock_connect:
            mock_conn = Mock()
            mock_connect.return_value = mock_conn
            mock_conn.execute.return_value.fetchall.return_value = mock_results

            conn_io = create_connection(config)
            db_conn = conn_io.run()

            search_io = hybrid_search(
                db_conn,
                "test_table",
                search_query,
                text_query,
                vector_weight=0.6,
                text_weight=0.4,
            )
            result = search_io.run()

            assert len(result.retrieved_ids) == 2
            assert result.query_id.startswith("hybrid_query_")
            assert isinstance(result.metrics, Metrics)

    def test_search_query_creation(self) -> None:
        """검색 쿼리 생성 테스트"""
        query_vector = Vector(Dimension(256), VectorData([0.5] * 256))

        basic_query = SearchQuery(query_vector)
        assert basic_query.k == 10
        assert basic_query.filter_category is None
        assert basic_query.include_distances is True

        filtered_query = SearchQuery(
            query_vector,
            k=20,
            filter_category=Category.DOCUMENT,
            include_distances=False,
        )
        assert filtered_query.k == 20
        assert filtered_query.filter_category == Category.DOCUMENT
        assert filtered_query.include_distances is False
