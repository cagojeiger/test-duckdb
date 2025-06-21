"""Tests for database table operations."""

from unittest.mock import Mock, patch
from datetime import datetime
from src.types.core import (
    DatabaseConfig,
    Dimension,
    Category,
    TextContent,
    Vector,
    VectorData,
)
from src.effects.db.connection import create_connection
from src.effects.db.tables import (
    TableSchema,
    create_documents_table,
    create_hnsw_index,
    insert_document,
    batch_insert_documents,
    drop_table,
)


class TestTableOperations:
    """테이블 작업 테스트"""

    def test_create_documents_table(self) -> None:
        """문서 테이블 생성 테스트"""
        config = DatabaseConfig()
        schema = TableSchema("test_docs", Dimension(128))

        with patch("duckdb.connect") as mock_connect:
            mock_conn = Mock()
            mock_connect.return_value = mock_conn

            conn_io = create_connection(config)
            db_conn = conn_io.run()

            table_io = create_documents_table(db_conn, schema)
            table_io.run()

            mock_conn.execute.assert_called()

    def test_create_hnsw_index(self) -> None:
        """HNSW 인덱스 생성 테스트"""
        config = DatabaseConfig()
        schema = TableSchema("test_docs", Dimension(256))

        with patch("duckdb.connect") as mock_connect:
            mock_conn = Mock()
            mock_connect.return_value = mock_conn

            conn_io = create_connection(config)
            db_conn = conn_io.run()

            index_io = create_hnsw_index(
                db_conn,
                schema,
                ef_construction=64,
                ef_search=32,
                M=8,
                metric="euclidean",
            )
            index_io.run()

            mock_conn.execute.assert_called()

    def test_insert_document(self) -> None:
        """단일 문서 삽입 테스트"""
        config = DatabaseConfig()
        schema = TableSchema("test_docs", Dimension(128))

        content = TextContent(
            text="테스트 내용",
            title="테스트 제목",
            category=Category.NEWS,
            created_at=datetime.now(),
        )
        vector = Vector(Dimension(128), VectorData([0.1] * 128))

        with patch("duckdb.connect") as mock_connect:
            mock_conn = Mock()
            mock_connect.return_value = mock_conn

            conn_io = create_connection(config)
            db_conn = conn_io.run()

            insert_io = insert_document(db_conn, schema, "doc1", content, vector)
            insert_io.run()

            mock_conn.execute.assert_called()

    def test_batch_insert_documents(self) -> None:
        """배치 문서 삽입 테스트"""
        config = DatabaseConfig()
        schema = TableSchema("test_docs", Dimension(128))

        documents = []
        for i in range(3):
            content = TextContent(
                text=f"테스트 내용 {i}",
                title=f"테스트 제목 {i}",
                category=Category.REVIEW,
                created_at=datetime.now(),
            )
            vector = Vector(Dimension(128), VectorData([0.1 * i] * 128))
            documents.append((f"doc{i}", content, vector))

        with patch("duckdb.connect") as mock_connect:
            mock_conn = Mock()
            mock_connect.return_value = mock_conn

            conn_io = create_connection(config)
            db_conn = conn_io.run()

            batch_io = batch_insert_documents(db_conn, schema, documents)
            batch_io.run()

            mock_conn.executemany.assert_called_once()

    def test_drop_table(self) -> None:
        """테이블 삭제 테스트"""
        config = DatabaseConfig()

        with patch("duckdb.connect") as mock_connect:
            mock_conn = Mock()
            mock_connect.return_value = mock_conn

            conn_io = create_connection(config)
            db_conn = conn_io.run()

            drop_io = drop_table(db_conn, "test_table")
            drop_io.run()

            mock_conn.execute.assert_called_with("DROP TABLE IF EXISTS test_table")
