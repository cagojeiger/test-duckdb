"""Tests for database connection management."""

from unittest.mock import Mock, patch
from src.types.core import DatabaseConfig
from src.effects.db.connection import create_connection, connection_context


class TestDatabaseConnection:
    """데이터베이스 연결 테스트"""

    def test_create_connection_default_config(self) -> None:
        """기본 설정으로 연결 생성"""
        config = DatabaseConfig()

        with patch("duckdb.connect") as mock_connect:
            mock_conn = Mock()
            mock_connect.return_value = mock_conn

            conn_io = create_connection(config)
            db_conn = conn_io.run()

            mock_conn.execute.assert_any_call("INSTALL vss")
            mock_conn.execute.assert_any_call("LOAD vss")

            assert db_conn.config == config

    def test_create_connection_with_memory_limit(self) -> None:
        """메모리 제한 설정으로 연결 생성"""
        config = DatabaseConfig(memory_limit_mb=1024)

        with patch("duckdb.connect") as mock_connect:
            mock_conn = Mock()
            mock_connect.return_value = mock_conn

            conn_io = create_connection(config)
            conn_io.run()

            mock_conn.execute.assert_any_call("SET memory_limit='1024MB'")

    def test_create_connection_with_threads(self) -> None:
        """스레드 수 설정으로 연결 생성"""
        config = DatabaseConfig(threads=4)

        with patch("duckdb.connect") as mock_connect:
            mock_conn = Mock()
            mock_connect.return_value = mock_conn

            conn_io = create_connection(config)
            conn_io.run()

            mock_conn.execute.assert_any_call("SET threads=4")

    def test_connection_context_manager(self) -> None:
        """연결 컨텍스트 매니저 테스트"""
        config = DatabaseConfig()

        with patch("duckdb.connect") as mock_connect:
            mock_conn = Mock()
            mock_connect.return_value = mock_conn

            with connection_context(config) as db_conn:
                assert db_conn.config == config

            mock_conn.close.assert_called_once()

    def test_execute_query_with_params(self) -> None:
        """파라미터가 있는 쿼리 실행"""
        config = DatabaseConfig()

        with patch("duckdb.connect") as mock_connect:
            mock_conn = Mock()
            mock_connect.return_value = mock_conn
            mock_conn.execute.return_value.fetchall.return_value = [("result",)]

            conn_io = create_connection(config)
            db_conn = conn_io.run()

            from src.effects.db.connection import execute_query

            query_io = execute_query(
                db_conn, "SELECT * FROM test WHERE id = ?", ["123"]
            )
            result = query_io.run()

            mock_conn.execute.assert_called_with(
                "SELECT * FROM test WHERE id = ?", ["123"]
            )
            assert result == [("result",)]

    def test_execute_command_without_params(self) -> None:
        """파라미터 없는 명령 실행"""
        config = DatabaseConfig()

        with patch("duckdb.connect") as mock_connect:
            mock_conn = Mock()
            mock_connect.return_value = mock_conn

            conn_io = create_connection(config)
            db_conn = conn_io.run()

            from src.effects.db.connection import execute_command

            command_io = execute_command(db_conn, "CREATE TABLE test (id INTEGER)")
            command_io.run()

            mock_conn.execute.assert_called_with("CREATE TABLE test (id INTEGER)")
