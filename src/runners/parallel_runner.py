#!/usr/bin/env python3
"""
Parallel Experiment Runner for DuckDB VSS Benchmarking
Implements concurrent experiment execution with process isolation and resource management
"""

import time
import psutil
from concurrent.futures import ProcessPoolExecutor, as_completed, Future
from typing import List, Optional, Dict, Any, Callable
from pathlib import Path
from dataclasses import dataclass

from src.types.core import ExperimentConfig, ExperimentResult
from src.types.monads import IO
from src.pipelines.experiments import single_experiment_pipeline
from src.runners.monitoring import ResourceMonitor


@dataclass(frozen=True)
class ParallelConfig:
    """Configuration for parallel execution"""

    max_workers: int = 4
    memory_threshold_mb: int = 6000
    experiment_timeout_seconds: int = 300
    min_workers: int = 1
    max_workers_limit: int = 8


@dataclass(frozen=True)
class ParallelResult:
    """Result of parallel experiment execution"""

    results: List[ExperimentResult]
    failed_configs: List[ExperimentConfig]
    execution_time_seconds: float
    peak_memory_mb: float
    worker_count_used: int


def _run_single_experiment_worker(config: ExperimentConfig) -> ExperimentResult:
    """Worker function to run a single experiment in a separate process"""
    try:
        experiment_io = single_experiment_pipeline(config)
        result = experiment_io.run()
        return result
    except Exception as e:
        raise RuntimeError(f"Experiment failed for config {config}: {str(e)}")


class ParallelExperimentRunner:
    """Parallel experiment runner with dynamic resource management"""

    def __init__(self, config: ParallelConfig = ParallelConfig()):
        self.config = config
        self.resource_monitor = ResourceMonitor()

    def run_experiments_parallel(
        self,
        configs: List[ExperimentConfig],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> IO[ParallelResult]:
        """Run experiments in parallel with dynamic worker adjustment"""

        def execute_parallel() -> ParallelResult:
            start_time = time.time()
            self.resource_monitor.start_monitoring()

            try:
                initial_workers = self._calculate_optimal_workers()
                print(f"🚀 Starting parallel execution with {initial_workers} workers")
                print(f"📊 Processing {len(configs)} experiments")

                results = []
                failed_configs = []
                completed_count = 0
                peak_memory = 0.0

                with ProcessPoolExecutor(max_workers=initial_workers) as executor:
                    future_to_config: Dict[Future, ExperimentConfig] = {}

                    for config in configs:
                        future = executor.submit(_run_single_experiment_worker, config)
                        future_to_config[future] = config

                    for future in as_completed(
                        future_to_config, timeout=self.config.experiment_timeout_seconds
                    ):
                        config = future_to_config[future]
                        completed_count += 1

                        try:
                            result = future.result()
                            results.append(result)

                            if progress_callback:
                                progress_callback(completed_count, len(configs))

                            current_memory = self.resource_monitor.get_memory_usage()
                            peak_memory = max(peak_memory, current_memory)

                            print(
                                f"  ✅ Experiment {completed_count}/{len(configs)} completed"
                            )
                            print(f"     Memory: {current_memory:.1f}MB")

                            if current_memory > self.config.memory_threshold_mb:
                                print(
                                    f"     ⚠️  High memory usage detected: {current_memory:.1f}MB"
                                )

                        except Exception as e:
                            failed_configs.append(config)
                            print(f"  ❌ Experiment failed: {str(e)}")

                execution_time = time.time() - start_time

                return ParallelResult(
                    results=results,
                    failed_configs=failed_configs,
                    execution_time_seconds=execution_time,
                    peak_memory_mb=peak_memory,
                    worker_count_used=initial_workers,
                )

            finally:
                self.resource_monitor.stop_monitoring()

        class ParallelExecutionIO(IO[ParallelResult]):
            def run(self) -> ParallelResult:
                return execute_parallel()

        return ParallelExecutionIO()

    def run_experiments_batched_parallel(
        self,
        configs: List[ExperimentConfig],
        batch_size: int = 8,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> IO[ParallelResult]:
        """Run experiments in parallel batches with dynamic resource management"""

        def execute_batched_parallel() -> ParallelResult:
            start_time = time.time()
            all_results = []
            all_failed_configs = []
            peak_memory = 0.0
            total_completed = 0

            batches = [
                configs[i : i + batch_size] for i in range(0, len(configs), batch_size)
            ]
            total_batches = len(batches)

            print(f"🚀 Starting batched parallel execution")
            print(
                f"📊 Processing {len(configs)} experiments in {total_batches} batches"
            )

            for batch_idx, batch_configs in enumerate(batches, 1):
                print(
                    f"\n🔄 Batch {batch_idx}/{total_batches} - {len(batch_configs)} experiments"
                )

                current_workers = self._calculate_optimal_workers()

                batch_io = self.run_experiments_parallel(
                    batch_configs,
                    lambda completed, total: progress_callback(
                        total_completed + completed, len(configs)
                    )
                    if progress_callback
                    else None,
                )
                batch_result = batch_io.run()

                all_results.extend(batch_result.results)
                all_failed_configs.extend(batch_result.failed_configs)
                peak_memory = max(peak_memory, batch_result.peak_memory_mb)
                total_completed += len(batch_result.results)

                print(f"✅ Batch {batch_idx} completed")
                print(f"   Results: {len(batch_result.results)}")
                print(f"   Failed: {len(batch_result.failed_configs)}")
                print(f"   Peak Memory: {batch_result.peak_memory_mb:.1f}MB")

                self.resource_monitor.cleanup_between_batches()

            execution_time = time.time() - start_time

            return ParallelResult(
                results=all_results,
                failed_configs=all_failed_configs,
                execution_time_seconds=execution_time,
                peak_memory_mb=peak_memory,
                worker_count_used=current_workers,
            )

        class BatchedParallelExecutionIO(IO[ParallelResult]):
            def run(self) -> ParallelResult:
                return execute_batched_parallel()

        return BatchedParallelExecutionIO()

    def _calculate_optimal_workers(self) -> int:
        """Calculate optimal number of workers based on available memory"""
        try:
            memory = psutil.virtual_memory()
            available_memory_mb = memory.available / (1024 * 1024)

            memory_per_worker_mb = 1500  # Each experiment uses ~1-1.5GB

            memory_based_workers = int(available_memory_mb / memory_per_worker_mb)

            cpu_count = psutil.cpu_count(logical=False) or 4
            cpu_based_workers = min(cpu_count, self.config.max_workers_limit)

            optimal_workers = min(
                memory_based_workers, cpu_based_workers, self.config.max_workers
            )

            optimal_workers = max(optimal_workers, self.config.min_workers)

            print(f"💡 Resource analysis:")
            print(f"   Available memory: {available_memory_mb:.0f}MB")
            print(f"   CPU cores: {cpu_count}")
            print(f"   Memory-based workers: {memory_based_workers}")
            print(f"   CPU-based workers: {cpu_based_workers}")
            print(f"   Optimal workers: {optimal_workers}")

            return optimal_workers

        except Exception as e:
            print(f"⚠️  Failed to calculate optimal workers: {e}")
            return self.config.min_workers

    def _check_memory_pressure(self) -> bool:
        """Check if system is under memory pressure"""
        try:
            memory = psutil.virtual_memory()
            return memory.percent > 85  # Consider 85%+ usage as pressure
        except Exception:
            return False


def create_parallel_runner(
    max_workers: int = 4,
    memory_threshold_mb: int = 6000,
    experiment_timeout_seconds: int = 300,
) -> ParallelExperimentRunner:
    """Factory function to create a parallel experiment runner"""
    config = ParallelConfig(
        max_workers=max_workers,
        memory_threshold_mb=memory_threshold_mb,
        experiment_timeout_seconds=experiment_timeout_seconds,
    )
    return ParallelExperimentRunner(config)
