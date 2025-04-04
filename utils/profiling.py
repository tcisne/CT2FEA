# ct2fea/utils/profiling.py
import time
import cProfile
import pstats
from pathlib import Path
import logging
from typing import Optional, Dict, Any, Callable
from functools import wraps
import json
import psutil
import numpy as np


class PerformanceMonitor:
    """Monitor and profile pipeline performance"""

    def __init__(self, output_dir: Path):
        """Initialize performance monitor

        Args:
            output_dir: Directory to save profiling results
        """
        self.output_dir = output_dir
        self.profiler = cProfile.Profile()
        self.stage_times: Dict[str, float] = {}
        self.memory_usage: Dict[str, Dict[str, float]] = {}
        self.logger = logging.getLogger(__name__)

    def start_profiling(self) -> None:
        """Start profiling"""
        self.profiler.enable()

    def stop_profiling(self) -> None:
        """Stop profiling and save results"""
        self.profiler.disable()
        stats = pstats.Stats(self.profiler)
        stats.sort_stats("cumulative")

        # Save detailed profiling results
        stats_file = self.output_dir / "profile_stats.txt"
        stats.dump_stats(str(stats_file.with_suffix(".prof")))
        with open(stats_file, "w") as f:
            stats.stream = f
            stats.print_stats()

    def log_stage_time(self, stage: str, duration: float) -> None:
        """Log execution time for a pipeline stage

        Args:
            stage: Stage name
            duration: Execution time in seconds
        """
        self.stage_times[stage] = duration
        self.logger.info(f"Stage '{stage}' completed in {duration:.2f} seconds")

    def log_memory_usage(self, stage: str) -> None:
        """Log memory usage for current stage

        Args:
            stage: Stage name
        """
        process = psutil.Process()
        memory_info = process.memory_info()

        self.memory_usage[stage] = {
            "rss": memory_info.rss / (1024 * 1024),  # MB
            "vms": memory_info.vms / (1024 * 1024),  # MB
        }

        self.logger.info(
            f"Memory usage at '{stage}': "
            f"RSS={self.memory_usage[stage]['rss']:.1f}MB, "
            f"VMS={self.memory_usage[stage]['vms']:.1f}MB"
        )

    def generate_report(self) -> None:
        """Generate performance report"""
        report = {
            "stage_times": self.stage_times,
            "memory_usage": self.memory_usage,
            "total_time": sum(self.stage_times.values()),
            "peak_memory": max(stage["rss"] for stage in self.memory_usage.values()),
        }

        # Calculate statistics
        times = np.array(list(self.stage_times.values()))
        report["statistics"] = {
            "mean_stage_time": float(np.mean(times)),
            "std_stage_time": float(np.std(times)),
            "slowest_stage": max(self.stage_times.items(), key=lambda x: x[1])[0],
        }

        # Save report
        report_file = self.output_dir / "performance_report.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=4)

        self.logger.info(
            f"Pipeline completed - Total time: {report['total_time']:.2f}s, "
            f"Peak memory: {report['peak_memory']:.1f}MB"
        )


def profile_stage(monitor: Optional[PerformanceMonitor] = None):
    """Decorator for profiling pipeline stages

    Args:
        monitor: Optional performance monitor instance
    """

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()

            if monitor:
                monitor.log_memory_usage(func.__name__)

            result = func(*args, **kwargs)

            duration = time.time() - start_time
            if monitor:
                monitor.log_stage_time(func.__name__, duration)
                monitor.log_memory_usage(f"{func.__name__}_end")

            return result

        return wrapper

    return decorator


def estimate_memory_usage(shape: tuple, dtype: np.dtype) -> float:
    """Estimate memory usage for array operations

    Args:
        shape: Array shape
        dtype: Array data type

    Returns:
        Estimated memory usage in MB
    """
    element_size = np.dtype(dtype).itemsize
    total_elements = np.prod(shape)
    return (total_elements * element_size) / (1024 * 1024)  # Convert to MB


def check_memory_available(required_mb: float, threshold: float = 0.8) -> bool:
    """Check if sufficient memory is available

    Args:
        required_mb: Required memory in MB
        threshold: Memory usage threshold (0-1)

    Returns:
        True if sufficient memory is available
    """
    memory = psutil.virtual_memory()
    available_mb = memory.available / (1024 * 1024)
    return available_mb >= required_mb and memory.percent <= (threshold * 100)
