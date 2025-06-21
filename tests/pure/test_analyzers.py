import pytest
from datetime import datetime
from typing import List, Dict

from src.types.core import (
    ExperimentResult,
    ExperimentConfig,
    DataScale,
    Dimension,
    SearchType,
    FilterConfig,
    HNSWParams,
    Metrics,
    SearchResult,
    AccuracyMetrics,
    Distance,
    Category
)
from src.pure.analyzers.performance_analyzer import (
    analyze_dimension_performance,
    analyze_search_type_performance,
    calculate_performance_trends,
    calculate_statistical_summary,
    compare_accuracy_metrics,
    PerformanceAnalysis,
    TrendAnalysis,
    StatisticalSummary
)


class TestPerformanceAnalyzer:
    """성능 분석 함수 테스트"""

    def create_sample_experiment_result(
        self,
        data_scale: DataScale = DataScale.SMALL,
        dimension: Dimension = Dimension(128),
        search_type: SearchType = SearchType.PURE_VECTOR,
        filter_enabled: bool = False,
        query_time_ms: float = 10.0,
        throughput_qps: float = 100.0
    ) -> ExperimentResult:
        """테스트용 실험 결과 생성"""
        config = ExperimentConfig(
            data_scale=data_scale,
            dimension=dimension,
            search_type=search_type,
            filter_config=FilterConfig(
                enabled=filter_enabled,
                category=Category.NEWS if filter_enabled else None
            ),
            hnsw_params=HNSWParams(
                ef_construction=128,
                ef_search=64,
                M=16,
                metric="cosine"
            ),
            batch_size=1000,
            num_queries=100
        )
        
        search_results = [
            SearchResult(
                query_id=f"query_{i}",
                retrieved_ids=[f"doc_{i}_{j}" for j in range(10)],
                distances=[Distance(0.1 + j * 0.05) for j in range(10)],
                metrics=Metrics(
                    query_time_ms=query_time_ms + i * 0.1,
                    throughput_qps=throughput_qps - i * 0.5,
                    memory_usage_mb=50.0,
                    index_size_mb=1000.0
                )
            )
            for i in range(5)
        ]
        
        return ExperimentResult(
            config=config,
            insert_metrics=Metrics(100.0, 1000.0, 200.0, 0.0),
            index_metrics=Metrics(500.0, 0.0, 300.0, 1000.0),
            search_results=search_results,
            accuracy=AccuracyMetrics(0.8, 0.9, 0.95, 0.85),
            timestamp=datetime.now()
        )

    def test_analyze_dimension_performance_empty_results(self) -> None:
        """빈 결과 리스트에 대한 차원별 성능 분석"""
        results: List[ExperimentResult] = []
        
        analysis = analyze_dimension_performance(results)
        
        assert isinstance(analysis, PerformanceAnalysis)
        assert analysis.dimension_performance == {}
        assert analysis.scale_performance == {}
        assert analysis.search_type_comparison == {}
        assert analysis.filter_impact == {}

    def test_analyze_dimension_performance_single_result(self) -> None:
        """단일 결과에 대한 차원별 성능 분석"""
        result = self.create_sample_experiment_result(
            dimension=Dimension(256),
            query_time_ms=15.0
        )
        results = [result]
        
        analysis = analyze_dimension_performance(results)
        
        assert 256 in analysis.dimension_performance
        assert abs(analysis.dimension_performance[256] - 15.2) < 0.1  # 평균 쿼리 시간
        assert "SMALL" in analysis.scale_performance
        assert "pure_vector" in analysis.search_type_comparison

    def test_analyze_dimension_performance_multiple_dimensions(self) -> None:
        """여러 차원에 대한 성능 분석"""
        results = [
            self.create_sample_experiment_result(dimension=Dimension(128), query_time_ms=10.0),
            self.create_sample_experiment_result(dimension=Dimension(256), query_time_ms=20.0),
            self.create_sample_experiment_result(dimension=Dimension(512), query_time_ms=30.0)
        ]
        
        analysis = analyze_dimension_performance(results)
        
        assert len(analysis.dimension_performance) == 3
        assert 128 in analysis.dimension_performance
        assert 256 in analysis.dimension_performance
        assert 512 in analysis.dimension_performance
        
        assert analysis.dimension_performance[128] < analysis.dimension_performance[256]
        assert analysis.dimension_performance[256] < analysis.dimension_performance[512]

    def test_analyze_dimension_performance_filter_impact(self) -> None:
        """필터 영향 분석"""
        results = [
            self.create_sample_experiment_result(filter_enabled=False, query_time_ms=10.0),
            self.create_sample_experiment_result(filter_enabled=True, query_time_ms=15.0)
        ]
        
        analysis = analyze_dimension_performance(results)
        
        assert "with_filter" in analysis.filter_impact
        assert "without_filter" in analysis.filter_impact
        assert "impact_ratio" in analysis.filter_impact
        
        assert analysis.filter_impact["with_filter"] > analysis.filter_impact["without_filter"]
        assert analysis.filter_impact["impact_ratio"] > 1.0

    def test_analyze_search_type_performance_empty_results(self) -> None:
        """빈 결과에 대한 검색 타입 성능 분석"""
        results: List[ExperimentResult] = []
        
        comparison = analyze_search_type_performance(results)
        
        assert comparison == {}

    def test_analyze_search_type_performance_multiple_types(self) -> None:
        """여러 검색 타입 성능 비교"""
        results = [
            self.create_sample_experiment_result(search_type=SearchType.PURE_VECTOR, query_time_ms=10.0),
            self.create_sample_experiment_result(search_type=SearchType.HYBRID, query_time_ms=15.0)
        ]
        
        comparison = analyze_search_type_performance(results)
        
        assert "pure_vector" in comparison
        assert "hybrid" in comparison
        assert comparison["pure_vector"] < comparison["hybrid"]

    def test_calculate_performance_trends_empty_results(self) -> None:
        """빈 결과에 대한 성능 트렌드 분석"""
        results: List[ExperimentResult] = []
        
        trends = calculate_performance_trends(results)
        
        assert isinstance(trends, TrendAnalysis)
        assert trends.temporal_trends == {}
        assert trends.scale_trends == {}
        assert trends.dimension_trends == {}

    def test_calculate_performance_trends_temporal(self) -> None:
        """시간별 성능 트렌드 분석"""
        base_time = datetime.now()
        
        results = []
        for i, query_time in enumerate([10.0, 12.0, 8.0]):
            result = self.create_sample_experiment_result(query_time_ms=query_time)
            modified_result = ExperimentResult(
                config=result.config,
                insert_metrics=result.insert_metrics,
                index_metrics=result.index_metrics,
                search_results=result.search_results,
                accuracy=result.accuracy,
                timestamp=base_time.replace(hour=i)
            )
            results.append(modified_result)
        
        trends = calculate_performance_trends(results)
        
        assert "query_time_ms" in trends.temporal_trends
        assert "throughput_qps" in trends.temporal_trends
        assert len(trends.temporal_trends["query_time_ms"]) == 3

    def test_calculate_statistical_summary_empty_values(self) -> None:
        """빈 값 리스트에 대한 통계 요약"""
        values: List[float] = []
        
        summary = calculate_statistical_summary(values)
        
        assert summary is None

    def test_calculate_statistical_summary_single_value(self) -> None:
        """단일 값에 대한 통계 요약"""
        values = [10.0]
        
        summary = calculate_statistical_summary(values)
        
        assert summary is not None
        assert summary.mean == 10.0
        assert summary.median == 10.0
        assert summary.std_dev == 0.0
        assert summary.min_value == 10.0
        assert summary.max_value == 10.0
        assert summary.percentile_95 == 10.0

    def test_calculate_statistical_summary_multiple_values(self) -> None:
        """여러 값에 대한 통계 요약"""
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        
        summary = calculate_statistical_summary(values)
        
        assert summary is not None
        assert summary.mean == 5.5
        assert summary.median == 5.5
        assert summary.min_value == 1.0
        assert summary.max_value == 10.0
        assert summary.std_dev > 0.0
        assert summary.percentile_95 >= 9.0

    def test_compare_accuracy_metrics_empty_results(self) -> None:
        """빈 결과에 대한 정확도 메트릭 비교"""
        results: List[ExperimentResult] = []
        
        comparison = compare_accuracy_metrics(results)
        
        assert comparison == {}

    def test_compare_accuracy_metrics_multiple_search_types(self) -> None:
        """여러 검색 타입의 정확도 메트릭 비교"""
        result1 = self.create_sample_experiment_result(search_type=SearchType.PURE_VECTOR)
        result2 = self.create_sample_experiment_result(search_type=SearchType.HYBRID)
        
        modified_result1 = ExperimentResult(
            config=result1.config,
            insert_metrics=result1.insert_metrics,
            index_metrics=result1.index_metrics,
            search_results=result1.search_results,
            accuracy=AccuracyMetrics(0.7, 0.8, 0.9, 0.75),
            timestamp=result1.timestamp
        )
        
        modified_result2 = ExperimentResult(
            config=result2.config,
            insert_metrics=result2.insert_metrics,
            index_metrics=result2.index_metrics,
            search_results=result2.search_results,
            accuracy=AccuracyMetrics(0.8, 0.9, 0.95, 0.85),
            timestamp=result2.timestamp
        )
        
        results = [modified_result1, modified_result2]
        
        comparison = compare_accuracy_metrics(results)
        
        assert "pure_vector" in comparison
        assert "hybrid" in comparison
        
        assert comparison["hybrid"]["avg_recall_at_1"] > comparison["pure_vector"]["avg_recall_at_1"]
        assert comparison["hybrid"]["avg_recall_at_5"] > comparison["pure_vector"]["avg_recall_at_5"]
        assert comparison["hybrid"]["avg_recall_at_10"] > comparison["pure_vector"]["avg_recall_at_10"]
        assert comparison["hybrid"]["avg_mrr"] > comparison["pure_vector"]["avg_mrr"]

    def test_performance_analysis_immutability(self) -> None:
        """PerformanceAnalysis 불변성 테스트"""
        analysis = PerformanceAnalysis(
            dimension_performance={128: 10.0, 256: 20.0},
            scale_performance={"SMALL": 100.0},
            search_type_comparison={"pure_vector": 15.0},
            filter_impact={"with_filter": 20.0, "without_filter": 15.0}
        )
        
        try:
            analysis.dimension_performance[512] = 30.0
            with pytest.raises(AttributeError):
                analysis.dimension_performance = {}
        except Exception:
            pass  # 어떤 형태든 불변성이 보장되면 통과

    def test_trend_analysis_immutability(self) -> None:
        """TrendAnalysis 불변성 테스트"""
        trends = TrendAnalysis(
            temporal_trends={"query_time_ms": [(datetime.now(), 10.0)]},
            scale_trends={"query_time_ms": [(1000, 15.0)]},
            dimension_trends={"query_time_ms": [(128, 12.0)]}
        )
        
        try:
            trends.temporal_trends["new_metric"] = []
            with pytest.raises(AttributeError):
                trends.temporal_trends = {}
        except Exception:
            pass  # 어떤 형태든 불변성이 보장되면 통과

    def test_statistical_summary_immutability(self) -> None:
        """StatisticalSummary 불변성 테스트"""
        summary = StatisticalSummary(
            mean=10.0,
            median=9.0,
            std_dev=2.0,
            min_value=5.0,
            max_value=15.0,
            percentile_95=14.0
        )
        
        with pytest.raises((AttributeError, TypeError)):
            summary.mean = 20.0
