"""
Timing utilities for performance measurement.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator, Optional, List
import logging


@dataclass
class TimingResult:
    """Result of a timed operation."""
    name: str
    duration_sec: float
    success: bool = True
    error: Optional[str] = None
    
    @property
    def duration_ms(self) -> float:
        return self.duration_sec * 1000
    
    def __str__(self) -> str:
        if self.duration_sec < 1:
            return f"{self.name}: {self.duration_ms:.1f}ms"
        elif self.duration_sec < 60:
            return f"{self.name}: {self.duration_sec:.2f}s"
        else:
            minutes = int(self.duration_sec // 60)
            seconds = self.duration_sec % 60
            return f"{self.name}: {minutes}m {seconds:.1f}s"


@contextmanager
def timed_operation(
    name: str,
    logger: Optional[logging.Logger] = None,
    log_level: int = logging.DEBUG
) -> Iterator[TimingResult]:
    """
    Context manager for timing operations.
    
    Usage:
        with timed_operation("OCR processing", logger) as timing:
            # do work
        print(f"Took {timing.duration_sec:.2f}s")
    
    Args:
        name: Name of the operation (for logging)
        logger: Optional logger to log timing
        log_level: Log level for timing message
    
    Yields:
        TimingResult that will be populated on exit
    """
    result = TimingResult(name=name, duration_sec=0.0)
    start = time.perf_counter()
    
    try:
        yield result
        result.success = True
    except Exception as e:
        result.success = False
        result.error = str(e)
        raise
    finally:
        result.duration_sec = time.perf_counter() - start
        
        if logger:
            msg = str(result)
            if not result.success:
                msg += f" (failed: {result.error})"
            logger.log(log_level, msg)


class Timer:
    """
    Accumulating timer for tracking multiple operations.
    
    Usage:
        timer = Timer()
        
        timer.start("phase1")
        # do work
        timer.stop("phase1")
        
        timer.start("phase2")
        # do work
        timer.stop("phase2")
        
        print(timer.summary())
    """
    
    def __init__(self):
        self._starts: dict[str, float] = {}
        self._totals: dict[str, float] = {}
        self._counts: dict[str, int] = {}
        self._global_start: float = time.perf_counter()
    
    def start(self, name: str) -> None:
        """Start timing an operation."""
        self._starts[name] = time.perf_counter()
    
    def stop(self, name: str) -> float:
        """
        Stop timing an operation.
        
        Returns:
            Duration in seconds
        """
        if name not in self._starts:
            return 0.0
        
        duration = time.perf_counter() - self._starts[name]
        del self._starts[name]
        
        self._totals[name] = self._totals.get(name, 0.0) + duration
        self._counts[name] = self._counts.get(name, 0) + 1
        
        return duration
    
    def lap(self, name: str) -> float:
        """
        Stop current timing and start a new one (same name).
        
        Returns:
            Duration of completed lap
        """
        duration = self.stop(name)
        self.start(name)
        return duration
    
    def get_total(self, name: str) -> float:
        """Get total time for an operation."""
        return self._totals.get(name, 0.0)
    
    def get_count(self, name: str) -> int:
        """Get count of times an operation was timed."""
        return self._counts.get(name, 0)
    
    def get_average(self, name: str) -> float:
        """Get average time for an operation."""
        total = self._totals.get(name, 0.0)
        count = self._counts.get(name, 0)
        return total / count if count > 0 else 0.0
    
    @property
    def elapsed(self) -> float:
        """Get total elapsed time since timer creation."""
        return time.perf_counter() - self._global_start
    
    def summary(self) -> str:
        """Generate summary of all timed operations."""
        lines = ["Timing Summary:"]
        
        for name in sorted(self._totals.keys()):
            total = self._totals[name]
            count = self._counts[name]
            avg = total / count if count > 0 else 0.0
            
            if total < 1:
                total_str = f"{total * 1000:.1f}ms"
            else:
                total_str = f"{total:.2f}s"
            
            if count > 1:
                avg_str = f"{avg * 1000:.1f}ms" if avg < 1 else f"{avg:.2f}s"
                lines.append(f"  {name}: {total_str} total, {avg_str} avg ({count}x)")
            else:
                lines.append(f"  {name}: {total_str}")
        
        lines.append(f"  Total elapsed: {self.elapsed:.2f}s")
        
        return "\n".join(lines)
    
    def to_dict(self) -> dict[str, dict[str, float]]:
        """Convert timing data to dictionary."""
        return {
            name: {
                "total_sec": self._totals.get(name, 0.0),
                "count": self._counts.get(name, 0),
                "avg_sec": self.get_average(name),
            }
            for name in self._totals.keys()
        }


def format_duration(seconds: float) -> str:
    """
    Format duration in human-readable form.
    
    Args:
        seconds: Duration in seconds
    
    Returns:
        Formatted string
    """
    if seconds < 0.001:
        return f"{seconds * 1000000:.0f}μs"
    elif seconds < 1:
        return f"{seconds * 1000:.1f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.0f}s"
