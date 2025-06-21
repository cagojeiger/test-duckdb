"""DuckDB connection management with VSS extension."""

from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Generator, Optional
import duckdb
from src.types.monads import IO
from src.types.core import DatabaseConfig


@dataclass(frozen=True)
class DBConnection:
    """DuckDB connection wrapper with VSS extension."""

    conn: duckdb.DuckDBPyConnection
    config: DatabaseConfig

    def close(self) -> None:
        """Close the database connection."""
        self.conn.close()


def create_connection(config: DatabaseConfig) -> IO[DBConnection]:
    """Create a new DuckDB connection with VSS extension loaded."""

    class CreateConnectionIO(IO[DBConnection]):
        def run(self) -> DBConnection:
            conn = duckdb.connect(config.database_path)

            conn.execute("INSTALL vss")
            conn.execute("LOAD vss")

            if config.memory_limit_mb:
                conn.execute(f"SET memory_limit='{config.memory_limit_mb}MB'")

            if config.threads:
                conn.execute(f"SET threads={config.threads}")

            return DBConnection(conn, config)

    return CreateConnectionIO()


@contextmanager
def connection_context(config: DatabaseConfig) -> Generator[DBConnection, None, None]:
    """Context manager for database connections."""
    conn_io = create_connection(config)
    conn = conn_io.run()
    try:
        yield conn
    finally:
        conn.close()


def execute_query(
    conn: DBConnection, query: str, params: Optional[list[str]] = None
) -> IO[list[tuple[str, ...]]]:
    """Execute a query and return results."""

    class ExecuteQueryIO(IO[list[tuple[str, ...]]]):
        def run(self) -> list[tuple[str, ...]]:
            if params:
                result = conn.conn.execute(query, params).fetchall()
                return [tuple(str(item) for item in row) for row in result]
            result = conn.conn.execute(query).fetchall()
            return [tuple(str(item) for item in row) for row in result]

    return ExecuteQueryIO()


def execute_command(
    conn: DBConnection, command: str, params: Optional[list[str]] = None
) -> IO[None]:
    """Execute a command without returning results."""

    class ExecuteCommandIO(IO[None]):
        def run(self) -> None:
            if params:
                conn.conn.execute(command, params)
            else:
                conn.conn.execute(command)

    return ExecuteCommandIO()
