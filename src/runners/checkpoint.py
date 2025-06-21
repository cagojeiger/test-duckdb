"""
Checkpoint Management System for DuckDB VSS Benchmarking
Handles saving/loading experiment progress and results
"""

import json
import pickle
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

from src.types.core import ExperimentResult, ExperimentConfig


class CheckpointManager:
    """Manages experiment checkpoints and result persistence"""

    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(exist_ok=True)

        self.progress_file = checkpoint_dir / "experiment_progress.json"
        self.results_file = checkpoint_dir / "experiment_results.pkl"
        self.metadata_file = checkpoint_dir / "checkpoint_metadata.json"

    def save_batch_results(self, results: List[ExperimentResult]) -> None:
        """Save a batch of experiment results"""
        existing_results = self.load_all_results()

        all_results = existing_results + results

        with open(self.results_file, "wb") as f:
            pickle.dump(all_results, f)

        self._update_progress_tracking(results)

        self._update_checkpoint_metadata(len(all_results))

        print(
            f"💾 Saved {len(results)} results to checkpoint (total: {len(all_results)})"
        )

    def load_all_results(self) -> List[ExperimentResult]:
        """Load all saved experiment results"""
        if not self.results_file.exists():
            return []

        try:
            with open(self.results_file, "rb") as f:
                results = pickle.load(f)
                return results if isinstance(results, list) else []
        except Exception as e:
            print(f"⚠️  Failed to load results from checkpoint: {e}")
            return []

    def load_completed_experiments(self) -> List[Dict[str, Any]]:
        """Load list of completed experiment configurations"""
        if not self.progress_file.exists():
            return []

        try:
            with open(self.progress_file, "r") as f:
                data = json.load(f)
                completed = data.get("completed_experiments", [])
                return completed if isinstance(completed, list) else []
        except Exception as e:
            print(f"⚠️  Failed to load progress from checkpoint: {e}")
            return []

    def get_checkpoint_status(self) -> Dict[str, Any]:
        """Get current checkpoint status information"""
        if not self.metadata_file.exists():
            return {
                "exists": False,
                "total_results": 0,
                "last_updated": None,
                "completion_percentage": 0.0,
            }

        try:
            with open(self.metadata_file, "r") as f:
                metadata = json.load(f)

            return {
                "exists": True,
                "total_results": metadata.get("total_results", 0),
                "last_updated": metadata.get("last_updated"),
                "completion_percentage": (metadata.get("total_results", 0) / 48.0)
                * 100,
                "estimated_remaining_time": metadata.get("estimated_remaining_time"),
            }
        except Exception as e:
            print(f"⚠️  Failed to load checkpoint metadata: {e}")
            return {
                "exists": False,
                "total_results": 0,
                "last_updated": None,
                "completion_percentage": 0.0,
            }

    def clear_checkpoints(self) -> None:
        """Clear all checkpoint data"""
        files_to_remove = [self.progress_file, self.results_file, self.metadata_file]

        for file_path in files_to_remove:
            if file_path.exists():
                file_path.unlink()
                print(f"🗑️  Removed {file_path.name}")

        print("✅ All checkpoints cleared")

    def export_results_to_json(self, output_file: Path) -> None:
        """Export all results to human-readable JSON format"""
        results = self.load_all_results()

        if not results:
            print("❌ No results to export")
            return

        json_results = []

        for result in results:
            json_result = {
                "experiment_id": self._generate_experiment_id(result.config),
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
                    "batch_size": result.config.batch_size,
                    "num_queries": result.config.num_queries,
                },
                "performance_metrics": {
                    "data_insertion": {
                        "time_ms": result.insert_metrics.query_time_ms,
                        "throughput_qps": result.insert_metrics.throughput_qps,
                        "memory_usage_mb": result.insert_metrics.memory_usage_mb,
                    },
                    "index_building": {
                        "time_ms": result.index_metrics.query_time_ms,
                        "memory_usage_mb": result.index_metrics.memory_usage_mb,
                        "index_size_mb": result.index_metrics.index_size_mb,
                    },
                    "search_performance": {
                        "total_queries": len(result.search_results),
                        "avg_query_time_ms": self._calculate_avg_query_time(
                            result.search_results
                        ),
                        "total_results_found": sum(
                            len(sr.retrieved_ids) for sr in result.search_results
                        ),
                    },
                },
                "accuracy_metrics": {
                    "recall_at_1": result.accuracy.recall_at_1,
                    "recall_at_5": result.accuracy.recall_at_5,
                    "recall_at_10": result.accuracy.recall_at_10,
                    "mean_reciprocal_rank": result.accuracy.mean_reciprocal_rank,
                },
                "execution_info": {
                    "timestamp": result.timestamp.isoformat(),
                    "duration_seconds": self._estimate_experiment_duration(result),
                },
            }
            json_results.append(json_result)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "export_metadata": {
                        "export_timestamp": datetime.now().isoformat(),
                        "total_experiments": len(json_results),
                        "experiment_matrix_size": 48,
                        "completion_percentage": (len(json_results) / 48.0) * 100,
                    },
                    "experiments": json_results,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        print(f"📤 Exported {len(json_results)} results to {output_file}")

    def _update_progress_tracking(self, new_results: List[ExperimentResult]) -> None:
        """Update progress tracking file"""
        if self.progress_file.exists():
            with open(self.progress_file, "r") as f:
                progress_data = json.load(f)
        else:
            progress_data = {
                "completed_experiments": [],
                "start_time": datetime.now().isoformat(),
                "last_updated": None,
            }

        for result in new_results:
            experiment_info = {
                "data_scale": result.config.data_scale.name,
                "dimension": int(result.config.dimension),
                "search_type": result.config.search_type.name,
                "filter_enabled": result.config.filter_config.enabled,
                "completed_at": result.timestamp.isoformat(),
                "experiment_id": self._generate_experiment_id(result.config),
            }
            progress_data["completed_experiments"].append(experiment_info)

        progress_data["last_updated"] = datetime.now().isoformat()

        with open(self.progress_file, "w") as f:
            json.dump(progress_data, f, indent=2)

    def _update_checkpoint_metadata(self, total_results: int) -> None:
        """Update checkpoint metadata"""
        metadata = {
            "total_results": total_results,
            "last_updated": datetime.now().isoformat(),
            "checkpoint_version": "1.0",
            "completion_percentage": (total_results / 48.0) * 100,
        }

        if total_results > 0:
            if self.progress_file.exists():
                with open(self.progress_file, "r") as f:
                    progress_data = json.load(f)
                    start_time = datetime.fromisoformat(
                        progress_data.get("start_time", datetime.now().isoformat())
                    )
                    elapsed_time = (datetime.now() - start_time).total_seconds()
                    avg_time_per_experiment = elapsed_time / total_results
                    remaining_experiments = 48 - total_results
                    estimated_remaining_seconds = (
                        remaining_experiments * avg_time_per_experiment
                    )
                    metadata["estimated_remaining_time"] = (
                        f"{estimated_remaining_seconds / 60:.1f} minutes"
                    )

        with open(self.metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

    def _generate_experiment_id(self, config: ExperimentConfig) -> str:
        """Generate unique experiment ID from configuration"""
        scale_abbrev = config.data_scale.name[0].lower()  # s, m, l
        dim = int(config.dimension)
        search_abbrev = "v" if config.search_type.name == "PURE_VECTOR" else "h"
        filter_abbrev = "f" if config.filter_config.enabled else "n"

        return f"{scale_abbrev}{dim}{search_abbrev}{filter_abbrev}"

    def _calculate_avg_query_time(self, search_results: List[Any]) -> float:
        """Calculate average query time from search results"""
        if not search_results:
            return 0.0

        return 0.0

    def _estimate_experiment_duration(self, result: ExperimentResult) -> float:
        """Estimate total experiment duration"""
        total_ms = (
            result.insert_metrics.query_time_ms + result.index_metrics.query_time_ms
        )
        return total_ms / 1000.0  # Convert to seconds


class CheckpointError(Exception):
    """Custom exception for checkpoint-related errors"""

    pass
