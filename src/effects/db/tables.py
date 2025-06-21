"""Table creation and management operations."""

from __future__ import annotations
from dataclasses import dataclass
from typing import List
from src.types.monads import IO
from src.types.core import Dimension, TextContent, Vector
from src.effects.db.connection import DBConnection


@dataclass(frozen=True)
class TableSchema:
    """Database table schema definition."""

    table_name: str
    dimension: Dimension
    has_metadata: bool = True


def create_documents_table(conn: DBConnection, schema: TableSchema) -> IO[None]:
    """Create documents table with vector column."""

    def _create_table() -> None:
        create_sql = f"""
        CREATE TABLE IF NOT EXISTS {schema.table_name} (
            id VARCHAR PRIMARY KEY,
            title VARCHAR NOT NULL,
            content TEXT NOT NULL,
            category VARCHAR NOT NULL,
            created_at TIMESTAMP NOT NULL,
            vector FLOAT[{schema.dimension}] NOT NULL
        )
        """
        conn.conn.execute(create_sql)

    class CreateTableIO(IO[None]):
        def run(self) -> None:
            return _create_table()

    return CreateTableIO()


def create_hnsw_index(
    conn: DBConnection,
    schema: TableSchema,
    ef_construction: int = 128,
    ef_search: int = 64,
    M: int = 16,
    metric: str = "cosine",
) -> IO[None]:
    """Create HNSW index on vector column."""

    def _create_index() -> None:
        index_name = f"idx_{schema.table_name}_vector_hnsw"
        create_index_sql = f"""
        CREATE INDEX {index_name} ON {schema.table_name}
        USING HNSW(vector)
        WITH (ef_construction = {ef_construction}, ef_search = {ef_search}, M = {M}, metric = '{metric}')
        """
        conn.conn.execute(create_index_sql)

    class CreateIndexIO(IO[None]):
        def run(self) -> None:
            return _create_index()

    return CreateIndexIO()


def insert_document(
    conn: DBConnection,
    schema: TableSchema,
    doc_id: str,
    content: TextContent,
    vector: Vector,
) -> IO[None]:
    """Insert a single document with its vector."""

    def _insert() -> None:
        insert_sql = f"""
        INSERT INTO {schema.table_name} (id, title, content, category, created_at, vector)
        VALUES (?, ?, ?, ?, ?, ?)
        """
        params = [
            doc_id,
            content.title,
            content.text,
            content.category.value,
            content.created_at,
            vector.data,  # DuckDB expects list of floats for FLOAT[] type
        ]
        conn.conn.execute(insert_sql, params)

    class InsertDocumentIO(IO[None]):
        def run(self) -> None:
            return _insert()

    return InsertDocumentIO()


def batch_insert_documents(
    conn: DBConnection,
    schema: TableSchema,
    documents: List[tuple[str, TextContent, Vector]],
) -> IO[None]:
    """Insert multiple documents in a single transaction."""

    def _batch_insert() -> None:
        insert_sql = f"""
        INSERT INTO {schema.table_name} (id, title, content, category, created_at, vector)
        VALUES (?, ?, ?, ?, ?, ?)
        """

        batch_data = []
        for doc_id, content, vector in documents:
            batch_data.append(
                [
                    doc_id,
                    content.title,
                    content.text,
                    content.category.value,
                    content.created_at,
                    vector.data,
                ]
            )

        conn.conn.executemany(insert_sql, batch_data)

    class BatchInsertIO(IO[None]):
        def run(self) -> None:
            return _batch_insert()

    return BatchInsertIO()


def drop_table(conn: DBConnection, table_name: str) -> IO[None]:
    """Drop a table if it exists."""

    def _drop() -> None:
        conn.conn.execute(f"DROP TABLE IF EXISTS {table_name}")

    class DropTableIO(IO[None]):
        def run(self) -> None:
            return _drop()

    return DropTableIO()


def get_table_info(conn: DBConnection, table_name: str) -> IO[list[tuple[str, ...]]]:
    """Get table schema information."""

    def _get_info() -> list[tuple[str, ...]]:
        result = conn.conn.execute(f"DESCRIBE {table_name}").fetchall()
        return [tuple(str(item) for item in row) for row in result]

    class GetTableInfoIO(IO[list[tuple[str, ...]]]):
        def run(self) -> list[tuple[str, ...]]:
            return _get_info()

    return GetTableInfoIO()
