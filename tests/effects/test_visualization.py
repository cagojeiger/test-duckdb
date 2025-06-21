from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path
import tempfile
import os

from src.types.monads import IO
from src.pure.analyzers.performance_analyzer import PerformanceAnalysis, TrendAnalysis
from src.effects.visualization.chart_generator import (
    generate_performance_heatmap,
    create_trend_charts,
    export_interactive_dashboard,
    generate_summary_report,
)


class TestVisualizationEffects:
    """시각화 효과 함수 테스트"""

    def create_sample_performance_analysis(self) -> PerformanceAnalysis:
        """테스트용 성능 분석 결과 생성"""
        return PerformanceAnalysis(
            dimension_performance={128: 10.0, 256: 15.0, 512: 25.0, 1024: 40.0},
            scale_performance={"SMALL": 100.0, "MEDIUM": 80.0, "LARGE": 60.0},
            search_type_comparison={"pure_vector": 12.0, "hybrid": 18.0},
            filter_impact={
                "with_filter": 20.0,
                "without_filter": 15.0,
                "impact_ratio": 1.33,
            },
        )

    def create_sample_trend_analysis(self) -> TrendAnalysis:
        """테스트용 트렌드 분석 결과 생성"""
        from datetime import datetime

        return TrendAnalysis(
            temporal_trends={
                "query_time_ms": [
                    (datetime(2024, 1, 1, 10, 0), 10.0),
                    (datetime(2024, 1, 1, 11, 0), 12.0),
                    (datetime(2024, 1, 1, 12, 0), 8.0),
                ],
                "throughput_qps": [
                    (datetime(2024, 1, 1, 10, 0), 100.0),
                    (datetime(2024, 1, 1, 11, 0), 95.0),
                    (datetime(2024, 1, 1, 12, 0), 110.0),
                ],
            },
            scale_trends={
                "query_time_ms": [(10000, 8.0), (100000, 12.0), (250000, 18.0)],
                "throughput_qps": [(10000, 120.0), (100000, 100.0), (250000, 80.0)],
            },
            dimension_trends={
                "query_time_ms": [(128, 10.0), (256, 15.0), (512, 25.0), (1024, 40.0)],
                "throughput_qps": [
                    (128, 110.0),
                    (256, 95.0),
                    (512, 75.0),
                    (1024, 50.0),
                ],
            },
        )

    def test_generate_performance_heatmap_io_type(self) -> None:
        """성능 히트맵 생성 함수가 IO 타입을 반환하는지 테스트"""
        analysis = self.create_sample_performance_analysis()

        result = generate_performance_heatmap(analysis)

        assert isinstance(result, IO)

    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.close")
    @patch("seaborn.heatmap")
    @patch("matplotlib.pyplot.subplots")
    def test_generate_performance_heatmap_execution(
        self, mock_subplots, mock_heatmap, mock_close, mock_savefig
    ) -> None:
        """성능 히트맵 생성 실행 테스트 (Mock 사용)"""
        analysis = self.create_sample_performance_analysis()

        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        with tempfile.TemporaryDirectory() as temp_dir:
            heatmap_io = generate_performance_heatmap(analysis, temp_dir)
            result_path = heatmap_io.run()

            assert isinstance(result_path, str)
            assert result_path.endswith(".png")
            assert temp_dir in result_path

            mock_subplots.assert_called_once()
            mock_heatmap.assert_called_once()
            mock_savefig.assert_called_once()
            mock_close.assert_called_once()

    def test_create_trend_charts_io_type(self) -> None:
        """트렌드 차트 생성 함수가 IO 타입을 반환하는지 테스트"""
        trends = self.create_sample_trend_analysis()

        result = create_trend_charts(trends)

        assert isinstance(result, IO)

    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.close")
    @patch("matplotlib.pyplot.subplots")
    def test_create_trend_charts_execution(
        self, mock_subplots, mock_close, mock_savefig
    ) -> None:
        """트렌드 차트 생성 실행 테스트 (Mock 사용)"""
        trends = self.create_sample_trend_analysis()

        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        with tempfile.TemporaryDirectory() as temp_dir:
            charts_io = create_trend_charts(trends, temp_dir)
            result_paths = charts_io.run()

            assert isinstance(result_paths, list)
            assert len(result_paths) > 0

            for path in result_paths:
                assert isinstance(path, str)
                assert path.endswith(".png")
                assert temp_dir in path

            assert mock_subplots.call_count >= 3
            assert mock_savefig.call_count >= 3
            assert mock_close.call_count >= 3

    def test_export_interactive_dashboard_io_type(self) -> None:
        """인터랙티브 대시보드 내보내기 함수가 IO 타입을 반환하는지 테스트"""
        analysis = self.create_sample_performance_analysis()
        trends = self.create_sample_trend_analysis()

        result = export_interactive_dashboard(analysis, trends)

        assert isinstance(result, IO)

    @patch("builtins.open", new_callable=mock_open)
    def test_export_interactive_dashboard_execution(self, mock_file) -> None:
        """인터랙티브 대시보드 내보내기 실행 테스트 (Mock 사용)"""
        analysis = self.create_sample_performance_analysis()
        trends = self.create_sample_trend_analysis()

        with tempfile.TemporaryDirectory() as temp_dir:
            dashboard_io = export_interactive_dashboard(analysis, trends, temp_dir)
            result_path = dashboard_io.run()

            assert isinstance(result_path, str)
            assert result_path.endswith(".html")
            assert temp_dir in result_path

            mock_file.assert_called()

    def test_generate_summary_report_io_type(self) -> None:
        """요약 보고서 생성 함수가 IO 타입을 반환하는지 테스트"""
        analysis = self.create_sample_performance_analysis()
        trends = self.create_sample_trend_analysis()

        result = generate_summary_report(analysis, trends)

        assert isinstance(result, IO)

    def test_generate_summary_report_execution(self) -> None:
        """요약 보고서 생성 실행 테스트"""
        analysis = self.create_sample_performance_analysis()
        trends = self.create_sample_trend_analysis()

        with tempfile.TemporaryDirectory() as temp_dir:
            report_io = generate_summary_report(analysis, trends, temp_dir)
            result_path = report_io.run()

            assert isinstance(result_path, str)
            assert result_path.endswith(".md")
            assert temp_dir in result_path

            assert os.path.exists(result_path)

            with open(result_path, "r", encoding="utf-8") as f:
                content = f.read()
                assert "# DuckDB VSS Performance Analysis Report" in content
                assert "## Performance Analysis Summary" in content
                assert "128D" in content  # 차원 정보
                assert "SMALL" in content  # 스케일 정보

    def test_generate_performance_heatmap_empty_analysis(self) -> None:
        """빈 분석 결과에 대한 히트맵 생성 테스트"""
        empty_analysis = PerformanceAnalysis(
            dimension_performance={},
            scale_performance={},
            search_type_comparison={},
            filter_impact={},
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            heatmap_io = generate_performance_heatmap(empty_analysis, temp_dir)
            result_path = heatmap_io.run()

            assert isinstance(result_path, str)
            assert result_path.endswith(".png")

    def test_create_trend_charts_empty_trends(self) -> None:
        """빈 트렌드 데이터에 대한 차트 생성 테스트"""
        empty_trends = TrendAnalysis(
            temporal_trends={}, scale_trends={}, dimension_trends={}
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            charts_io = create_trend_charts(empty_trends, temp_dir)
            result_paths = charts_io.run()

            assert isinstance(result_paths, list)

    def test_io_monad_chaining(self) -> None:
        """IO 모나드 체이닝 테스트"""
        analysis = self.create_sample_performance_analysis()

        result_io = generate_performance_heatmap(analysis).map(
            lambda path: f"Generated: {path}"
        )

        result = result_io.run()
        assert result.startswith("Generated:")
        assert result.endswith(".png")

    def test_io_monad_flat_map(self) -> None:
        """IO 모나드 flat_map 테스트"""
        analysis = self.create_sample_performance_analysis()
        trends = self.create_sample_trend_analysis()

        def create_report_after_heatmap(heatmap_path: str) -> IO[str]:
            return generate_summary_report(analysis, trends)

        chained_io = generate_performance_heatmap(analysis).flat_map(
            create_report_after_heatmap
        )

        result = chained_io.run()
        assert isinstance(result, str)
        assert result.endswith(".md")

    @patch("matplotlib.pyplot.savefig")
    @patch("pathlib.Path.mkdir")
    def test_directory_creation(self, mock_mkdir, mock_savefig) -> None:
        """출력 디렉토리 생성 테스트"""
        analysis = self.create_sample_performance_analysis()

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = os.path.join(temp_dir, "nested", "output")
            heatmap_io = generate_performance_heatmap(analysis, output_dir)
            heatmap_io.run()

            mock_mkdir.assert_called()
            mock_savefig.assert_called()

    def test_file_path_validation(self) -> None:
        """파일 경로 검증 테스트"""
        analysis = self.create_sample_performance_analysis()

        with tempfile.TemporaryDirectory() as temp_dir:
            heatmap_io = generate_performance_heatmap(analysis, temp_dir)
            result_path = heatmap_io.run()

            path_obj = Path(result_path)
            assert path_obj.parent.exists()
            assert path_obj.suffix == ".png"
            assert "performance_heatmap" in path_obj.name

    def test_concurrent_io_execution(self) -> None:
        """동시 IO 실행 테스트"""
        analysis = self.create_sample_performance_analysis()
        trends = self.create_sample_trend_analysis()

        with tempfile.TemporaryDirectory() as temp_dir:
            heatmap_io = generate_performance_heatmap(analysis, temp_dir)
            charts_io = create_trend_charts(trends, temp_dir)
            report_io = generate_summary_report(analysis, trends, temp_dir)

            heatmap_path = heatmap_io.run()
            chart_paths = charts_io.run()
            report_path = report_io.run()

            assert isinstance(heatmap_path, str)
            assert isinstance(chart_paths, list)
            assert isinstance(report_path, str)

            assert temp_dir in heatmap_path
            assert temp_dir in report_path
            for chart_path in chart_paths:
                assert temp_dir in chart_path
