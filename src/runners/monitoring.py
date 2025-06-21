"""
Resource Monitoring System for DuckDB VSS Benchmarking
Monitors memory usage, system resources, and manages cleanup
"""

import gc
import time
import psutil
import threading
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
from dataclasses import dataclass


@dataclass
class ResourceSnapshot:
    """Snapshot of system resources at a point in time"""

    timestamp: datetime
    memory_usage_mb: float
    memory_percent: float
    cpu_percent: float
    available_memory_mb: float
    process_memory_mb: float


class ResourceMonitor:
    """Monitors system resources during experiment execution"""

    def __init__(
        self, 
        memory_threshold_mb: float = 7000, 
        monitoring_interval: float = 5.0,
        dashboard_callback: Optional[Callable[[ResourceSnapshot], None]] = None
    ):
        self.memory_threshold_mb = memory_threshold_mb
        self.monitoring_interval = monitoring_interval
        self.dashboard_callback = dashboard_callback

        self.is_monitoring = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.resource_history: List[ResourceSnapshot] = []

        self.process = psutil.Process()

        self.memory_alerts: List[Dict[str, Any]] = []

    def start_monitoring(self) -> None:
        """Start background resource monitoring"""
        if self.is_monitoring:
            return

        self.is_monitoring = True
        self.resource_history.clear()
        self.memory_alerts.clear()

        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop, daemon=True
        )
        self.monitoring_thread.start()

        print("📊 Resource monitoring started")

    def stop_monitoring(self) -> None:
        """Stop background resource monitoring"""
        if not self.is_monitoring:
            return

        self.is_monitoring = False

        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=1.0)

        print("📊 Resource monitoring stopped")
        self._print_monitoring_summary()

    def get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            memory_info = self.process.memory_info()
            return float(memory_info.rss / (1024 * 1024))  # Convert bytes to MB
        except Exception:
            return 0.0

    def get_system_memory_info(self) -> Dict[str, float]:
        """Get system memory information"""
        try:
            memory = psutil.virtual_memory()
            return {
                "total_mb": memory.total / (1024 * 1024),
                "available_mb": memory.available / (1024 * 1024),
                "used_mb": memory.used / (1024 * 1024),
                "percent": memory.percent,
            }
        except Exception:
            return {"total_mb": 0, "available_mb": 0, "used_mb": 0, "percent": 0}

    def check_memory_available(self, required_mb: float = 2000) -> bool:
        """Check if sufficient memory is available"""
        system_memory = self.get_system_memory_info()
        return system_memory["available_mb"] >= required_mb

    def force_cleanup(self) -> None:
        """Force garbage collection and memory cleanup"""
        print("🧹 Forcing memory cleanup...")

        for i in range(3):
            collected = gc.collect()
            if collected > 0:
                print(f"   GC pass {i + 1}: collected {collected} objects")

        try:
            import ctypes

            libc = ctypes.CDLL("libc.so.6")
            libc.malloc_trim(0)
        except Exception:
            pass  # Not available on all systems

        time.sleep(1)  # Allow system to process cleanup

        memory_after = self.get_memory_usage()
        print(f"   Memory after cleanup: {memory_after:.1f}MB")

    def cleanup_between_batches(self) -> None:
        """Cleanup resources between experiment batches"""
        print("🔄 Cleaning up between batches...")

        collected = gc.collect()
        if collected > 0:
            print(f"   Collected {collected} objects")

        time.sleep(0.5)

        memory_usage = self.get_memory_usage()
        system_memory = self.get_system_memory_info()

        print(f"   Process memory: {memory_usage:.1f}MB")
        print(f"   System available: {system_memory['available_mb']:.1f}MB")

        if memory_usage > self.memory_threshold_mb:
            self._add_memory_alert("High memory usage after cleanup", memory_usage)

    def get_resource_summary(self) -> Dict[str, Any]:
        """Get summary of resource usage during monitoring"""
        if not self.resource_history:
            return {"error": "No monitoring data available"}

        memory_values = [snapshot.memory_usage_mb for snapshot in self.resource_history]
        cpu_values = [snapshot.cpu_percent for snapshot in self.resource_history]

        return {
            "monitoring_duration_minutes": self._get_monitoring_duration(),
            "memory_usage": {
                "current_mb": self.get_memory_usage(),
                "peak_mb": max(memory_values) if memory_values else 0,
                "average_mb": sum(memory_values) / len(memory_values)
                if memory_values
                else 0,
                "min_mb": min(memory_values) if memory_values else 0,
            },
            "cpu_usage": {
                "peak_percent": max(cpu_values) if cpu_values else 0,
                "average_percent": sum(cpu_values) / len(cpu_values)
                if cpu_values
                else 0,
            },
            "system_memory": self.get_system_memory_info(),
            "alerts_count": len(self.memory_alerts),
            "snapshots_count": len(self.resource_history),
        }

    def _monitoring_loop(self) -> None:
        """Background monitoring loop"""
        while self.is_monitoring:
            try:
                snapshot = self._take_resource_snapshot()
                self.resource_history.append(snapshot)

                if self.dashboard_callback:
                    self.dashboard_callback(snapshot)

                if snapshot.memory_usage_mb > self.memory_threshold_mb:
                    self._add_memory_alert(
                        "Memory threshold exceeded", snapshot.memory_usage_mb
                    )

                if len(self.resource_history) > 1000:
                    self.resource_history = self.resource_history[
                        -500:
                    ]  # Keep last 500 snapshots

            except Exception as e:
                print(f"⚠️  Monitoring error: {e}")

            time.sleep(self.monitoring_interval)

    def _take_resource_snapshot(self) -> ResourceSnapshot:
        """Take a snapshot of current resource usage"""
        try:
            process_memory = self.process.memory_info()
            process_memory_mb = process_memory.rss / (1024 * 1024)

            system_memory = psutil.virtual_memory()

            cpu_percent = self.process.cpu_percent()

            return ResourceSnapshot(
                timestamp=datetime.now(),
                memory_usage_mb=process_memory_mb,
                memory_percent=system_memory.percent,
                cpu_percent=cpu_percent,
                available_memory_mb=system_memory.available / (1024 * 1024),
                process_memory_mb=process_memory_mb,
            )
        except Exception:
            return ResourceSnapshot(
                timestamp=datetime.now(),
                memory_usage_mb=0.0,
                memory_percent=0.0,
                cpu_percent=0.0,
                available_memory_mb=0.0,
                process_memory_mb=0.0,
            )

    def _add_memory_alert(self, message: str, memory_mb: float) -> None:
        """Add a memory usage alert"""
        alert = {
            "timestamp": datetime.now().isoformat(),
            "message": message,
            "memory_mb": memory_mb,
            "threshold_mb": self.memory_threshold_mb,
        }

        self.memory_alerts.append(alert)

        if len(self.memory_alerts) > 100:
            self.memory_alerts = self.memory_alerts[-50:]  # Keep last 50 alerts

    def _get_monitoring_duration(self) -> float:
        """Get total monitoring duration in minutes"""
        if len(self.resource_history) < 2:
            return 0.0

        start_time = self.resource_history[0].timestamp
        end_time = self.resource_history[-1].timestamp
        duration = end_time - start_time

        return duration.total_seconds() / 60.0

    def _print_monitoring_summary(self) -> None:
        """Print summary of monitoring session"""
        summary = self.get_resource_summary()

        if "error" in summary:
            print(f"⚠️  {summary['error']}")
            return

        print("\n📊 Resource Monitoring Summary")
        print("-" * 30)
        print(f"Duration: {summary['monitoring_duration_minutes']:.1f} minutes")
        print(f"Memory Peak: {summary['memory_usage']['peak_mb']:.1f}MB")
        print(f"Memory Average: {summary['memory_usage']['average_mb']:.1f}MB")
        print(f"CPU Peak: {summary['cpu_usage']['peak_percent']:.1f}%")
        print(f"CPU Average: {summary['cpu_usage']['average_percent']:.1f}%")

        if summary["alerts_count"] > 0:
            print(f"⚠️  Memory alerts: {summary['alerts_count']}")

        print(
            f"System memory available: {summary['system_memory']['available_mb']:.1f}MB"
        )


class ResourceError(Exception):
    """Custom exception for resource-related errors"""

    pass
