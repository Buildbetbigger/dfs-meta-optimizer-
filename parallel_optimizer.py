"""
DFS Meta-Optimizer v8.0.0 - Performance Monitoring System
Real-time performance tracking, profiling, and optimization

NEW IN v8.0.0:
- Function-level timing decorators
- Performance metrics dashboard
- Bottleneck detection
- Memory profiling
- Real-time alerts for slow operations
- Historical performance tracking

TARGET: <1s for 20 lineups, <5s for 150 lineups
"""

import time
import functools
import logging
from typing import Callable, Dict, List, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
from datetime import datetime
import threading
import psutil
import os

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """Single performance measurement"""
    function_name: str
    duration_ms: float
    timestamp: datetime
    memory_delta_mb: float = 0.0
    cpu_percent: float = 0.0
    args_summary: str = ""
    
    
@dataclass
class FunctionStats:
    """Aggregated stats for a function"""
    name: str
    call_count: int = 0
    total_time_ms: float = 0.0
    avg_time_ms: float = 0.0
    min_time_ms: float = float('inf')
    max_time_ms: float = 0.0
    last_10_times: deque = field(default_factory=lambda: deque(maxlen=10))
    
    def update(self, duration_ms: float):
        """Update stats with new measurement"""
        self.call_count += 1
        self.total_time_ms += duration_ms
        self.avg_time_ms = self.total_time_ms / self.call_count
        self.min_time_ms = min(self.min_time_ms, duration_ms)
        self.max_time_ms = max(self.max_time_ms, duration_ms)
        self.last_10_times.append(duration_ms)


class PerformanceMonitor:
    """
    Central performance monitoring system.
    
    Thread-safe singleton for tracking all performance metrics.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self.metrics: List[PerformanceMetric] = []
        self.function_stats: Dict[str, FunctionStats] = defaultdict(
            lambda: FunctionStats(name="unknown")
        )
        self.alerts: List[str] = []
        
        # Performance thresholds (ms)
        self.thresholds = {
            'lineup_generation': 1000,  # 1s for 20 lineups
            'optimization': 5000,       # 5s for 150 lineups
            'api_call': 2000,           # 2s for Claude API
            'data_loading': 500,        # 500ms for data load
            'default': 1000
        }
        
        # System monitoring
        self.process = psutil.Process(os.getpid())
        self._initialized = True
        
        logger.info("PerformanceMonitor v8.0.0 initialized")
    
    def record_metric(self, metric: PerformanceMetric):
        """Record a performance metric"""
        with self._lock:
            self.metrics.append(metric)
            
            # Update function stats
            stats = self.function_stats[metric.function_name]
            stats.name = metric.function_name
            stats.update(metric.duration_ms)
            
            # Check for performance issues
            self._check_threshold(metric)
    
    def _check_threshold(self, metric: PerformanceMetric):
        """Check if metric exceeds threshold and create alert"""
        func_name = metric.function_name
        threshold = self.thresholds.get(func_name, self.thresholds['default'])
        
        if metric.duration_ms > threshold:
            alert = (
                f"⚠️ SLOW: {func_name} took {metric.duration_ms:.0f}ms "
                f"(threshold: {threshold}ms) - {metric.args_summary}"
            )
            self.alerts.append(alert)
            logger.warning(alert)
    
    def get_summary(self) -> Dict:
        """Get performance summary"""
        with self._lock:
            if not self.function_stats:
                return {}
            
            # Sort by total time
            sorted_funcs = sorted(
                self.function_stats.values(),
                key=lambda x: x.total_time_ms,
                reverse=True
            )
            
            return {
                'total_functions': len(self.function_stats),
                'total_calls': sum(s.call_count for s in self.function_stats.values()),
                'total_time_ms': sum(s.total_time_ms for s in self.function_stats.values()),
                'slowest_functions': [
                    {
                        'name': s.name,
                        'avg_time_ms': round(s.avg_time_ms, 2),
                        'max_time_ms': round(s.max_time_ms, 2),
                        'call_count': s.call_count,
                        'total_time_ms': round(s.total_time_ms, 2)
                    }
                    for s in sorted_funcs[:10]
                ],
                'recent_alerts': self.alerts[-10:] if self.alerts else []
            }
    
    def get_function_stats(self, func_name: str) -> Optional[FunctionStats]:
        """Get stats for specific function"""
        with self._lock:
            return self.function_stats.get(func_name)
    
    def reset(self):
        """Reset all metrics"""
        with self._lock:
            self.metrics.clear()
            self.function_stats.clear()
            self.alerts.clear()
            logger.info("Performance metrics reset")
    
    def export_metrics(self) -> List[Dict]:
        """Export all metrics as list of dicts"""
        with self._lock:
            return [
                {
                    'function': m.function_name,
                    'duration_ms': m.duration_ms,
                    'timestamp': m.timestamp.isoformat(),
                    'memory_delta_mb': m.memory_delta_mb,
                    'cpu_percent': m.cpu_percent
                }
                for m in self.metrics
            ]


# Global monitor instance
monitor = PerformanceMonitor()


def timed(category: str = "default", track_memory: bool = False):
    """
    Decorator to measure function execution time.
    
    Args:
        category: Category for threshold checking
        track_memory: Track memory usage (slower)
    
    Example:
        @timed(category='lineup_generation')
        def generate_lineups(num: int):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Get memory before
            mem_before = 0
            if track_memory:
                mem_before = monitor.process.memory_info().rss / 1024 / 1024
            
            # Time execution
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            
            duration_ms = (end_time - start_time) * 1000
            
            # Get memory after
            mem_delta = 0
            cpu_percent = 0
            if track_memory:
                mem_after = monitor.process.memory_info().rss / 1024 / 1024
                mem_delta = mem_after - mem_before
                cpu_percent = monitor.process.cpu_percent()
            
            # Create args summary
            args_summary = ""
            if args:
                if isinstance(args[0], int):
                    args_summary = f"n={args[0]}"
                elif hasattr(args[0], '__len__'):
                    args_summary = f"len={len(args[0])}"
            
            # Record metric
            metric = PerformanceMetric(
                function_name=func.__name__,
                duration_ms=duration_ms,
                timestamp=datetime.now(),
                memory_delta_mb=mem_delta,
                cpu_percent=cpu_percent,
                args_summary=args_summary
            )
            monitor.record_metric(metric)
            
            return result
        return wrapper
    return decorator


def time_section(name: str):
    """
    Context manager for timing code sections.
    
    Example:
        with time_section("data_loading"):
            df = load_data()
    """
    class TimerContext:
        def __init__(self, section_name: str):
            self.name = section_name
            self.start = None
            
        def __enter__(self):
            self.start = time.perf_counter()
            return self
            
        def __exit__(self, *args):
            duration_ms = (time.perf_counter() - self.start) * 1000
            metric = PerformanceMetric(
                function_name=self.name,
                duration_ms=duration_ms,
                timestamp=datetime.now()
            )
            monitor.record_metric(metric)
    
    return TimerContext(name)


class PerformanceReport:
    """Generate formatted performance reports"""
    
    @staticmethod
    def generate_text_report() -> str:
        """Generate text-based performance report"""
        summary = monitor.get_summary()
        
        if not summary:
            return "No performance data collected yet."
        
        report = "=" * 80 + "\n"
        report += "PERFORMANCE REPORT\n"
        report += "=" * 80 + "\n\n"
        
        report += f"Total Functions Tracked: {summary['total_functions']}\n"
        report += f"Total Function Calls: {summary['total_calls']}\n"
        report += f"Total Time: {summary['total_time_ms']:.0f}ms\n\n"
        
        report += "TOP 10 SLOWEST FUNCTIONS:\n"
        report += "-" * 80 + "\n"
        report += f"{'Function':<30} {'Avg (ms)':<12} {'Max (ms)':<12} {'Calls':<10}\n"
        report += "-" * 80 + "\n"
        
        for func in summary['slowest_functions']:
            report += f"{func['name']:<30} "
            report += f"{func['avg_time_ms']:<12.1f} "
            report += f"{func['max_time_ms']:<12.1f} "
            report += f"{func['call_count']:<10}\n"
        
        if summary['recent_alerts']:
            report += "\n\nRECENT ALERTS:\n"
            report += "-" * 80 + "\n"
            for alert in summary['recent_alerts'][-5:]:
                report += f"{alert}\n"
        
        report += "\n" + "=" * 80 + "\n"
        
        return report
    
    @staticmethod
    def get_dashboard_metrics() -> Dict:
        """Get metrics formatted for dashboard display"""
        summary = monitor.get_summary()
        
        if not summary:
            return {
                'status': 'No data',
                'total_time': 0,
                'avg_time': 0,
                'call_count': 0,
                'bottlenecks': []
            }
        
        total_time = summary['total_time_ms']
        total_calls = summary['total_calls']
        
        return {
            'status': 'Healthy' if not summary['recent_alerts'] else 'Issues Detected',
            'total_time_ms': round(total_time, 2),
            'avg_time_ms': round(total_time / total_calls if total_calls > 0 else 0, 2),
            'call_count': total_calls,
            'bottlenecks': summary['slowest_functions'][:5],
            'alerts': summary['recent_alerts'][-3:] if summary['recent_alerts'] else []
        }


# Convenience exports
__all__ = [
    'PerformanceMonitor',
    'PerformanceMetric',
    'FunctionStats',
    'timed',
    'time_section',
    'monitor',
    'PerformanceReport'
]
