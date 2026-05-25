"""
Profiling utilities for SAM3 LoRA inference and training.
"""

import time
import psutil
import torch
from pathlib import Path
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass, field, asdict
from functools import wraps
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class TimingStats:
    """Statistics for timing measurements."""
    name: str
    count: int = 0
    total_time: float = 0.0
    min_time: float = float("inf")
    max_time: float = 0.0
    times: list = field(default_factory=list)

    def update(self, elapsed_time: float):
        """Update statistics with a new timing measurement."""
        self.count += 1
        self.total_time += elapsed_time
        self.min_time = min(self.min_time, elapsed_time)
        self.max_time = max(self.max_time, elapsed_time)
        self.times.append(elapsed_time)

    @property
    def mean_time(self) -> float:
        """Calculate mean time."""
        return self.total_time / self.count if self.count > 0 else 0.0

    @property
    def std_time(self) -> float:
        """Calculate standard deviation of times."""
        if self.count < 2:
            return 0.0
        mean = self.mean_time
        variance = sum((t - mean) ** 2 for t in self.times) / (self.count - 1)
        return variance ** 0.5

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "count": self.count,
            "total_time": round(self.total_time, 4),
            "mean_time": round(self.mean_time, 4),
            "std_time": round(self.std_time, 4),
            "min_time": round(self.min_time, 4),
            "max_time": round(self.max_time, 4),
        }


@dataclass
class MemoryStats:
    """Memory usage statistics."""
    peak_memory_mb: float = 0.0
    current_memory_mb: float = 0.0
    allocated_memory_mb: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "peak_memory_mb": round(self.peak_memory_mb, 2),
            "current_memory_mb": round(self.current_memory_mb, 2),
            "allocated_memory_mb": round(self.allocated_memory_mb, 2),
        }


class Profiler:
    """Profile timing and memory usage for model inference and training."""

    def __init__(self, name: str = "Profiler"):
        """
        Initialize profiler.

        Args:
            name: Name of the profiler instance
        """
        self.name = name
        self.timers: Dict[str, TimingStats] = {}
        self.memory_stats: Dict[str, MemoryStats] = {}
        self._current_timer: Optional[str] = None
        self._start_time: Optional[float] = None

    def start_timer(self, name: str):
        """
        Start a named timer.

        Args:
            name: Name of the timer
        """
        if self._current_timer is not None:
            logger.warning(
                f"Timer '{self._current_timer}' is already running. "
                f"Stopping it before starting '{name}'."
            )
            self.stop_timer()

        self._current_timer = name
        self._start_time = time.time()
        logger.debug(f"Started timer: {name}")

    def stop_timer(self):
        """Stop the current timer and record the measurement."""
        if self._start_time is None:
            logger.warning("No timer is currently running.")
            return

        elapsed = time.time() - self._start_time
        name = self._current_timer

        if name not in self.timers:
            self.timers[name] = TimingStats(name=name)

        self.timers[name].update(elapsed)
        logger.debug(f"Stopped timer: {name}, elapsed: {elapsed:.4f}s")

        self._current_timer = None
        self._start_time = None

    def context(self, name: str):
        """Context manager for timing a code block."""
        class TimerContext:
            def __init__(self, profiler, timer_name):
                self.profiler = profiler
                self.timer_name = timer_name

            def __enter__(self):
                self.profiler.start_timer(self.timer_name)
                return self

            def __exit__(self, *args):
                self.profiler.stop_timer()

        return TimerContext(self, name)

    def measure_memory(self, name: str = "peak") -> MemoryStats:
        """
        Measure current memory usage.

        Args:
            name: Name for this memory measurement

        Returns:
            MemoryStats instance
        """
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
            current_memory = torch.cuda.memory_allocated() / (1024 ** 2)
            allocated_memory = torch.cuda.memory_reserved() / (1024 ** 2)
        else:
            process = psutil.Process()
            mem_info = process.memory_info()
            peak_memory = mem_info.rss / (1024 ** 2)
            current_memory = mem_info.rss / (1024 ** 2)
            allocated_memory = mem_info.vms / (1024 ** 2)

        stats = MemoryStats(
            peak_memory_mb=peak_memory,
            current_memory_mb=current_memory,
            allocated_memory_mb=allocated_memory,
        )
        self.memory_stats[name] = stats
        logger.debug(f"Memory measurement '{name}': {stats}")

        return stats

    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all profiling data.

        Returns:
            Dictionary containing timing and memory statistics
        """
        summary = {
            "name": self.name,
            "timers": {name: stats.to_dict() for name, stats in self.timers.items()},
            "memory": {name: stats.to_dict() for name, stats in self.memory_stats.items()},
        }
        return summary

    def print_summary(self):
        """Print a formatted summary of profiling results."""
        summary = self.get_summary()
        print(f"\n{'='*60}")
        print(f"Profiling Summary: {summary['name']}")
        print(f"{'='*60}")

        if summary["timers"]:
            print("\n⏱️  Timing Statistics:")
            print(f"{'Measurement':<30} {'Count':<8} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
            print("-" * 86)
            for name, stats in summary["timers"].items():
                print(
                    f"{name:<30} {stats['count']:<8} "
                    f"{stats['mean_time']:<12.4f} {stats['std_time']:<12.4f} "
                    f"{stats['min_time']:<12.4f} {stats['max_time']:<12.4f}"
                )

        if summary["memory"]:
            print("\n💾 Memory Statistics:")
            for name, stats in summary["memory"].items():
                print(f"\n{name}:")
                print(f"  Peak Memory: {stats['peak_memory_mb']:.2f} MB")
                print(f"  Current Memory: {stats['current_memory_mb']:.2f} MB")
                print(f"  Allocated Memory: {stats['allocated_memory_mb']:.2f} MB")

        print(f"\n{'='*60}\n")

    def save_summary(self, output_path: str):
        """
        Save profiling summary to a JSON file.

        Args:
            output_path: Path to save the JSON summary
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        summary = self.get_summary()
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Profiling summary saved to {output_path}")


def profile_function(func: Callable) -> Callable:
    """
    Decorator to profile a function's execution time and memory.

    Args:
        func: Function to profile

    Returns:
        Wrapped function with profiling
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        profiler = Profiler(name=func.__name__)

        # Measure memory before
        profiler.measure_memory("before")

        # Time the function
        with profiler.context(func.__name__):
            result = func(*args, **kwargs)

        # Measure memory after
        profiler.measure_memory("after")

        # Log results
        logger.info(f"Function '{func.__name__}' profiling:")
        profiler.print_summary()

        return result

    return wrapper
