#!/usr/bin/env python3
"""
CLI Experiment Runner for DuckDB VSS Benchmarking
Executes 48 experiment configurations with checkpoint and monitoring support
"""

import argparse
import sys
import json
import time
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

from src.types.core import (
    ExperimentConfig,
    ExperimentResult,
    DataScale,
    Dimension,
    SearchType,
)
from src.pipelines.experiments import (
    generate_experiment_matrix,
    single_experiment_pipeline,
)
from src.runners.checkpoint import CheckpointManager
from src.runners.monitoring import ResourceMonitor


class ExperimentRunner:
    """Main experiment runner with CLI interface"""

    def __init__(
        self,
        output_dir: Path = Path("results"),
        checkpoint_dir: Path = Path("checkpoints"),
    ):
        self.output_dir = output_dir
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_manager = CheckpointManager(checkpoint_dir)
        self.resource_monitor = ResourceMonitor()

        self.output_dir.mkdir(exist_ok=True)
        self.checkpoint_dir.mkdir(exist_ok=True)

    def run_all_experiments(
        self, batch_size: int = 4, resume: bool = False
    ) -> List[ExperimentResult]:
        """Run all 48 experiment configurations"""
        print("🚀 Starting DuckDB VSS Benchmarking - All 48 Experiments")
        print("=" * 60)

        all_configs = generate_experiment_matrix()
        print(f"📊 Generated {len(all_configs)} experiment configurations")

        if resume:
            completed_configs = self.checkpoint_manager.load_completed_experiments()
            remaining_configs = [
                c
                for c in all_configs
                if not self._is_config_completed(c, completed_configs)
            ]
            print(f"🔄 Resuming: {len(remaining_configs)} experiments remaining")
        else:
            remaining_configs = all_configs
            print(f"🆕 Starting fresh: {len(remaining_configs)} experiments to run")

        if not remaining_configs:
            print("✅ All experiments already completed!")
            return self.checkpoint_manager.load_all_results()

        self.resource_monitor.start_monitoring()

        try:
            all_results = []
            total_batches = (len(remaining_configs) + batch_size - 1) // batch_size

            for batch_idx in range(0, len(remaining_configs), batch_size):
                batch_configs = remaining_configs[batch_idx : batch_idx + batch_size]
                batch_num = (batch_idx // batch_size) + 1

                print(
                    f"\n🔄 Batch {batch_num}/{total_batches} - {len(batch_configs)} experiments"
                )
                print("-" * 40)

                if not self.resource_monitor.check_memory_available():
                    print("⚠️  Low memory detected, forcing garbage collection...")
                    self.resource_monitor.force_cleanup()

                batch_start = time.time()
                batch_results = self._run_experiment_batch(batch_configs)
                batch_time = time.time() - batch_start

                all_results.extend(batch_results)
                self.checkpoint_manager.save_batch_results(batch_results)

                print(f"✅ Batch {batch_num} completed in {batch_time:.1f}s")
                print(
                    f"   Memory usage: {self.resource_monitor.get_memory_usage():.1f}MB"
                )

                self.resource_monitor.cleanup_between_batches()

            print(f"\n🎉 All experiments completed! Total results: {len(all_results)}")
            return all_results

        finally:
            self.resource_monitor.stop_monitoring()

    def run_filtered_experiments(
        self,
        data_scales: Optional[List[str]] = None,
        dimensions: Optional[List[int]] = None,
        search_types: Optional[List[str]] = None,
        batch_size: int = 4,
    ) -> List[ExperimentResult]:
        """Run filtered subset of experiments"""
        print("🎯 Starting Filtered DuckDB VSS Benchmarking")
        print("=" * 50)

        all_configs = generate_experiment_matrix()
        filtered_configs = self._filter_configs(
            all_configs, data_scales, dimensions, search_types
        )

        print(f"📊 Filtered to {len(filtered_configs)} experiment configurations")

        if not filtered_configs:
            print("❌ No experiments match the specified filters")
            return []

        self.resource_monitor.start_monitoring()

        try:
            results = self._run_experiment_batch(filtered_configs)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = self.output_dir / f"filtered_results_{timestamp}.json"
            self._save_results_to_file(results, results_file)

            print(f"✅ Filtered experiments completed! Results saved to {results_file}")
            return results

        finally:
            self.resource_monitor.stop_monitoring()

    def _run_experiment_batch(
        self, configs: List[ExperimentConfig]
    ) -> List[ExperimentResult]:
        """Run a batch of experiments with monitoring"""
        results = []

        for i, config in enumerate(configs, 1):
            print(f"  🧪 Experiment {i}/{len(configs)}: {self._config_summary(config)}")

            try:
                experiment_start = time.time()
                result_io = single_experiment_pipeline(config)
                result = result_io.run()
                experiment_time = time.time() - experiment_start

                results.append(result)

                print(f"     ✅ Completed in {experiment_time:.1f}s")

                memory_mb = self.resource_monitor.get_memory_usage()
                if memory_mb > 6000:  # Warning at 6GB
                    print(f"     ⚠️  High memory usage: {memory_mb:.1f}MB")

            except Exception as e:
                print(f"     ❌ Failed: {str(e)}")
                continue

        return results

    def _filter_configs(
        self,
        configs: List[ExperimentConfig],
        data_scales: Optional[List[str]],
        dimensions: Optional[List[int]],
        search_types: Optional[List[str]],
    ) -> List[ExperimentConfig]:
        """Filter experiment configurations based on criteria"""
        filtered = configs

        if data_scales:
            scale_map = {
                "small": DataScale.SMALL,
                "medium": DataScale.MEDIUM,
                "large": DataScale.LARGE,
            }
            allowed_scales = [scale_map[s] for s in data_scales if s in scale_map]
            filtered = [c for c in filtered if c.data_scale in allowed_scales]

        if dimensions:
            allowed_dims = [Dimension(d) for d in dimensions]
            filtered = [c for c in filtered if c.dimension in allowed_dims]

        if search_types:
            type_map = {"vector": SearchType.PURE_VECTOR, "hybrid": SearchType.HYBRID}
            allowed_types = [type_map[t] for t in search_types if t in type_map]
            filtered = [c for c in filtered if c.search_type in allowed_types]

        return filtered

    def _config_summary(self, config: ExperimentConfig) -> str:
        """Generate human-readable config summary"""
        scale_name = config.data_scale.name.lower()
        dim = int(config.dimension)
        search_type = (
            "vector" if config.search_type == SearchType.PURE_VECTOR else "hybrid"
        )
        filter_status = "filtered" if config.filter_config.enabled else "unfiltered"

        return f"{scale_name}-{dim}d-{search_type}-{filter_status}"

    def _is_config_completed(
        self, config: ExperimentConfig, completed: List[Dict[str, Any]]
    ) -> bool:
        """Check if a configuration has already been completed"""
        config_dict = {
            "data_scale": config.data_scale.name,
            "dimension": int(config.dimension),
            "search_type": config.search_type.name,
            "filter_enabled": config.filter_config.enabled,
        }

        for completed_config in completed:
            if all(completed_config.get(k) == v for k, v in config_dict.items()):
                return True
        return False

    def _save_results_to_file(
        self, results: List[ExperimentResult], filepath: Path
    ) -> None:
        """Save experiment results to JSON file"""
        results_data = []

        for result in results:
            result_dict = {
                "config": {
                    "data_scale": result.config.data_scale.name,
                    "dimension": int(result.config.dimension),
                    "search_type": result.config.search_type.name,
                    "filter_enabled": result.config.filter_config.enabled,
                    "hnsw_params": {
                        "ef_construction": result.config.hnsw_params.ef_construction,
                        "ef_search": result.config.hnsw_params.ef_search,
                        "M": result.config.hnsw_params.M,
                    },
                },
                "metrics": {
                    "insert_time_ms": result.insert_metrics.query_time_ms,
                    "insert_throughput_qps": result.insert_metrics.throughput_qps,
                    "index_time_ms": result.index_metrics.query_time_ms,
                    "search_results_count": len(result.search_results),
                    "accuracy": {
                        "recall_at_1": result.accuracy.recall_at_1,
                        "recall_at_5": result.accuracy.recall_at_5,
                        "recall_at_10": result.accuracy.recall_at_10,
                        "mrr": result.accuracy.mean_reciprocal_rank,
                    },
                },
                "timestamp": result.timestamp.isoformat(),
            }
            results_data.append(result_dict)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)


def create_cli_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser"""
    parser = argparse.ArgumentParser(
        description="DuckDB VSS Benchmarking Experiment Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.runners.experiment_runner --all

  python -m src.runners.experiment_runner --data-scale small medium

  python -m src.runners.experiment_runner --dimensions 128 256

  python -m src.runners.experiment_runner --all --batch-size 2

  python -m src.runners.experiment_runner --all --resume
        """,
    )

    parser.add_argument(
        "--all", action="store_true", help="Run all 48 experiment configurations"
    )

    parser.add_argument(
        "--data-scale",
        nargs="+",
        choices=["small", "medium", "large"],
        help="Filter by data scale (small=10K, medium=100K, large=250K)",
    )

    parser.add_argument(
        "--dimensions",
        nargs="+",
        type=int,
        choices=[128, 256, 512, 1024],
        help="Filter by vector dimensions",
    )

    parser.add_argument(
        "--search-type",
        nargs="+",
        choices=["vector", "hybrid"],
        help="Filter by search type",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Number of experiments to run in parallel (default: 4)",
    )

    parser.add_argument(
        "--resume", action="store_true", help="Resume from previous checkpoint"
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Directory to save results (default: results/)",
    )

    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints"),
        help="Directory for checkpoints (default: checkpoints/)",
    )

    return parser


def main() -> int:
    """Main CLI entry point"""
    parser = create_cli_parser()
    args = parser.parse_args()

    if not args.all and not any([args.data_scale, args.dimensions, args.search_type]):
        parser.error("Must specify --all or at least one filter option")

    runner = ExperimentRunner(
        output_dir=args.output_dir, checkpoint_dir=args.checkpoint_dir
    )

    try:
        if args.all:
            results = runner.run_all_experiments(
                batch_size=args.batch_size, resume=args.resume
            )
        else:
            results = runner.run_filtered_experiments(
                data_scales=args.data_scale,
                dimensions=args.dimensions,
                search_types=args.search_type,
                batch_size=args.batch_size,
            )

        print("\n🎉 Experiment runner completed successfully!")
        print(f"📊 Total results: {len(results)}")
        return 0

    except KeyboardInterrupt:
        print("\n⚠️  Experiment runner interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Experiment runner failed: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
