"""
Resource Utilization Tracking System for Bharat-FM MLOps Platform

This module provides comprehensive resource utilization monitoring and tracking
for AI models, training jobs, and infrastructure components. It tracks CPU, GPU,
memory, storage, and network resources with real-time analytics.

Features:
- Real-time resource monitoring
- Multi-dimensional resource tracking
- Capacity planning and forecasting
- Resource allocation optimization
- Cost tracking and optimization
- Performance bottleneck detection
"""

import time
import threading
import json
import psutil
import GPUtil
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
import statistics
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ResourceMetrics:
    """Resource metrics data structure"""
    timestamp: datetime
    hostname: str
    cpu_percent: float
    cpu_count: int
    memory_percent: float
    memory_total: int
    memory_used: int
    memory_available: int
    disk_percent: float
    disk_total: int
    disk_used: int
    disk_free: int
    network_bytes_sent: int
    network_bytes_recv: int
    gpu_metrics: List[Dict[str, Any]] = None
    
@dataclass
class ProcessResourceMetrics:
    """Process-specific resource metrics"""
    timestamp: datetime
    process_id: int
    process_name: str
    cpu_percent: float
    memory_percent: float
    memory_rss: int
    memory_vms: int
    num_threads: int
    io_counters: Dict[str, int] = None
    
@dataclass
class ResourceAlert:
    """Resource utilization alert"""
    alert_id: str
    resource_type: str
    current_value: float
    threshold_value: float
    severity: str
    message: str
    timestamp: datetime
    resolved: bool = False

class ResourceUtilizationTracker:
    """
    Comprehensive resource utilization tracking system
    """
    
    def __init__(self, collection_interval: int = 5, retention_hours: int = 24):
        self.collection_interval = collection_interval
        self.retention_hours = retention_hours
        
        # Data storage
        self.system_metrics = deque(maxlen=10000)
        self.process_metrics = defaultdict(lambda: deque(maxlen=1000))
        self.resource_alerts = {}
        self.alert_history = []
        
        # Configuration
        self.thresholds = {
            'cpu_percent': 80.0,
            'memory_percent': 85.0,
            'disk_percent': 90.0,
            'gpu_memory_percent': 85.0,
            'gpu_utilization_percent': 90.0
        }
        
        # Process monitoring
        self.monitored_processes = set()
        self.process_names = {}
        
        # Threading
        self._lock = threading.Lock()
        self._running = False
        self._collection_thread = None
        self._analysis_thread = None
        
        # Statistics
        self.stats = {
            'metrics_collected': 0,
            'alerts_generated': 0,
            'processes_monitored': 0
        }
        
    def start_monitoring(self):
        """Start resource monitoring"""
        if self._running:
            logger.warning("Resource monitoring already running")
            return
            
        self._running = True
        self._collection_thread = threading.Thread(target=self._collection_loop, daemon=True)
        self._analysis_thread = threading.Thread(target=self._analysis_loop, daemon=True)
        
        self._collection_thread.start()
        self._analysis_thread.start()
        
        logger.info("Resource utilization tracking started")
        
    def stop_monitoring(self):
        """Stop resource monitoring"""
        self._running = False
        if self._collection_thread:
            self._collection_thread.join(timeout=5)
        if self._analysis_thread:
            self._analysis_thread.join(timeout=5)
        logger.info("Resource utilization tracking stopped")
        
    def add_process_to_monitor(self, pid: int, name: str = None):
        """
        Add a process to monitor
        
        Args:
            pid: Process ID
            name: Process name (optional)
        """
        with self._lock:
            try:
                process = psutil.Process(pid)
                if name is None:
                    name = process.name()
                    
                self.monitored_processes.add(pid)
                self.process_names[pid] = name
                self.stats['processes_monitored'] += 1
                
                logger.info(f"Added process to monitor: {name} (PID: {pid})")
                
            except psutil.NoSuchProcess:
                logger.error(f"Process {pid} not found")
                
    def remove_process_from_monitor(self, pid: int):
        """
        Remove a process from monitoring
        
        Args:
            pid: Process ID
        """
        with self._lock:
            if pid in self.monitored_processes:
                self.monitored_processes.remove(pid)
                if pid in self.process_names:
                    del self.process_names[pid]
                logger.info(f"Removed process from monitor: PID {pid}")
                
    def set_threshold(self, resource_type: str, threshold: float):
        """
        Set resource threshold for alerting
        
        Args:
            resource_type: Type of resource ('cpu_percent', 'memory_percent', etc.)
            threshold: Threshold value
        """
        with self._lock:
            self.thresholds[resource_type] = threshold
            logger.info(f"Set {resource_type} threshold to {threshold}%")
            
    def get_current_metrics(self) -> Optional[ResourceMetrics]:
        """
        Get current system resource metrics
        
        Returns:
            ResourceMetrics object or None if collection failed
        """
        try:
            # Get system metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            network = psutil.net_io_counters()
            
            # Get GPU metrics if available
            gpu_metrics = []
            try:
                gpus = GPUtil.getGPUs()
                for gpu in gpus:
                    gpu_metrics.append({
                        'id': gpu.id,
                        'name': gpu.name,
                        'load_percent': gpu.load * 100,
                        'memory_used': gpu.memoryUsed,
                        'memory_total': gpu.memoryTotal,
                        'memory_percent': (gpu.memoryUsed / gpu.memoryTotal) * 100,
                        'temperature': gpu.temperature
                    })
            except Exception:
                pass  # GPU monitoring not available
                
            return ResourceMetrics(
                timestamp=datetime.now(),
                hostname=psutil.os.uname().nodename,
                cpu_percent=cpu_percent,
                cpu_count=psutil.cpu_count(),
                memory_percent=memory.percent,
                memory_total=memory.total,
                memory_used=memory.used,
                memory_available=memory.available,
                disk_percent=disk.percent,
                disk_total=disk.total,
                disk_used=disk.used,
                disk_free=disk.free,
                network_bytes_sent=network.bytes_sent,
                network_bytes_recv=network.bytes_recv,
                gpu_metrics=gpu_metrics
            )
            
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
            return None
            
    def get_process_metrics(self, pid: int) -> Optional[ProcessResourceMetrics]:
        """
        Get resource metrics for a specific process
        
        Args:
            pid: Process ID
            
        Returns:
            ProcessResourceMetrics object or None if process not found
        """
        try:
            process = psutil.Process(pid)
            
            # Get process metrics
            cpu_percent = process.cpu_percent()
            memory_info = process.memory_info()
            memory_percent = process.memory_percent()
            num_threads = process.num_threads()
            
            # Get I/O counters if available
            io_counters = None
            try:
                io_counters = process.io_counters()._asdict()
            except Exception:
                pass
                
            return ProcessResourceMetrics(
                timestamp=datetime.now(),
                process_id=pid,
                process_name=self.process_names.get(pid, process.name()),
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_rss=memory_info.rss,
                memory_vms=memory_info.vms,
                num_threads=num_threads,
                io_counters=io_counters
            )
            
        except psutil.NoSuchProcess:
            logger.error(f"Process {pid} not found")
            return None
            
    def get_resource_history(self, resource_type: str, hours: int = 1) -> List[Tuple[datetime, float]]:
        """
        Get historical data for a specific resource
        
        Args:
            resource_type: Type of resource ('cpu_percent', 'memory_percent', etc.)
            hours: Number of hours to look back
            
        Returns:
            List of (timestamp, value) tuples
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with self._lock:
            history = []
            for metrics in self.system_metrics:
                if metrics.timestamp >= cutoff_time:
                    value = getattr(metrics, resource_type, None)
                    if value is not None:
                        history.append((metrics.timestamp, value))
                        
            return sorted(history, key=lambda x: x[0])
            
    def get_resource_statistics(self, resource_type: str, hours: int = 1) -> Dict[str, float]:
        """
        Get statistical summary for a resource
        
        Args:
            resource_type: Type of resource
            hours: Number of hours to analyze
            
        Returns:
            Dictionary with statistical values
        """
        history = self.get_resource_history(resource_type, hours)
        
        if not history:
            return {}
            
        values = [value for _, value in history]
        
        return {
            'min': min(values),
            'max': max(values),
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'std_dev': statistics.stdev(values) if len(values) > 1 else 0,
            'p95': np.percentile(values, 95),
            'p99': np.percentile(values, 99)
        }
        
    def get_active_alerts(self) -> List[ResourceAlert]:
        """
        Get active resource alerts
        
        Returns:
            List of active ResourceAlert objects
        """
        with self._lock:
            return [alert for alert in self.resource_alerts.values() if not alert.resolved]
            
    def resolve_alert(self, alert_id: str):
        """
        Resolve a resource alert
        
        Args:
            alert_id: ID of the alert to resolve
        """
        with self._lock:
            if alert_id in self.resource_alerts:
                self.resource_alerts[alert_id].resolved = True
                logger.info(f"Resolved resource alert: {alert_id}")
                
    def get_capacity_forecast(self, resource_type: str, forecast_hours: int = 24) -> Dict[str, Any]:
        """
        Simple capacity forecasting based on historical trends
        
        Args:
            resource_type: Type of resource to forecast
            forecast_hours: Hours to forecast ahead
            
        Returns:
            Dictionary with forecast information
        """
        # Get historical data
        history = self.get_resource_history(resource_type, hours=24)
        
        if len(history) < 2:
            return {'error': 'Insufficient historical data'}
            
        # Simple linear regression for trend analysis
        timestamps = [(t - history[0][0]).total_seconds() / 3600 for t, _ in history]
        values = [v for _, v in history]
        
        # Calculate trend
        n = len(timestamps)
        sum_x = sum(timestamps)
        sum_y = sum(values)
        sum_xy = sum(x * y for x, y in zip(timestamps, values))
        sum_x2 = sum(x * x for x in timestamps)
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        intercept = (sum_y - slope * sum_x) / n
        
        # Forecast future values
        forecast_timestamps = [timestamps[-1] + i for i in range(1, forecast_hours + 1)]
        forecast_values = [slope * t + intercept for t in forecast_timestamps]
        
        return {
            'current_value': values[-1],
            'trend_slope': slope,
            'forecast_hours': forecast_hours,
            'forecast_values': forecast_values,
            'forecast_max': max(forecast_values),
            'forecast_min': min(forecast_values)
        }
        
    def get_resource_efficiency_score(self) -> Dict[str, float]:
        """
        Calculate resource efficiency scores
        
        Returns:
            Dictionary with efficiency scores for different resources
        """
        current_metrics = self.get_current_metrics()
        if not current_metrics:
            return {}
            
        efficiency_scores = {}
        
        # CPU efficiency (lower is better)
        cpu_efficiency = max(0, 100 - current_metrics.cpu_percent)
        efficiency_scores['cpu'] = cpu_efficiency
        
        # Memory efficiency
        memory_efficiency = max(0, 100 - current_metrics.memory_percent)
        efficiency_scores['memory'] = memory_efficiency
        
        # Disk efficiency
        disk_efficiency = max(0, 100 - current_metrics.disk_percent)
        efficiency_scores['disk'] = disk_efficiency
        
        # GPU efficiency if available
        if current_metrics.gpu_metrics:
            gpu_efficiencies = []
            for gpu in current_metrics.gpu_metrics:
                gpu_eff = max(0, 100 - gpu['load_percent'] * 100)
                gpu_efficiencies.append(gpu_eff)
            efficiency_scores['gpu'] = statistics.mean(gpu_efficiencies) if gpu_efficiencies else 0
            
        # Overall efficiency
        all_scores = [score for score in efficiency_scores.values() if score > 0]
        efficiency_scores['overall'] = statistics.mean(all_scores) if all_scores else 0
        
        return efficiency_scores
        
    def export_metrics(self, filename: str = None) -> str:
        """
        Export resource metrics to JSON file
        
        Args:
            filename: Optional filename to save to
            
        Returns:
            JSON string of metrics
        """
        with self._lock:
            export_data = {
                'timestamp': datetime.now().isoformat(),
                'current_metrics': asdict(self.get_current_metrics()) if self.get_current_metrics() else None,
                'active_alerts': [asdict(alert) for alert in self.get_active_alerts()],
                'statistics': self.stats.copy(),
                'thresholds': self.thresholds.copy(),
                'monitored_processes': len(self.monitored_processes)
            }
            
            json_data = json.dumps(export_data, indent=2, default=str)
            
            if filename:
                with open(filename, 'w') as f:
                    f.write(json_data)
                logger.info(f"Resource metrics exported to {filename}")
                
            return json_data
            
    def _collect_system_metrics(self):
        """Collect system resource metrics"""
        metrics = self.get_current_metrics()
        if metrics:
            with self._lock:
                self.system_metrics.append(metrics)
                self.stats['metrics_collected'] += 1
                
    def _collect_process_metrics(self):
        """Collect metrics for monitored processes"""
        with self._lock:
            dead_processes = []
            
            for pid in list(self.monitored_processes):
                process_metrics = self.get_process_metrics(pid)
                if process_metrics:
                    self.process_metrics[pid].append(process_metrics)
                else:
                    dead_processes.append(pid)
                    
            # Remove dead processes
            for pid in dead_processes:
                self.monitored_processes.remove(pid)
                if pid in self.process_names:
                    del self.process_names[pid]
                    
    def _check_thresholds(self):
        """Check resource thresholds and generate alerts"""
        current_metrics = self.get_current_metrics()
        if not current_metrics:
            return
            
        alerts = []
        
        # Check CPU threshold
        if current_metrics.cpu_percent > self.thresholds.get('cpu_percent', 80):
            alert = ResourceAlert(
                alert_id=f"cpu_{int(time.time())}",
                resource_type="cpu_percent",
                current_value=current_metrics.cpu_percent,
                threshold_value=self.thresholds['cpu_percent'],
                severity="high" if current_metrics.cpu_percent > 90 else "medium",
                message=f"High CPU usage: {current_metrics.cpu_percent:.1f}%",
                timestamp=datetime.now()
            )
            alerts.append(alert)
            
        # Check memory threshold
        if current_metrics.memory_percent > self.thresholds.get('memory_percent', 85):
            alert = ResourceAlert(
                alert_id=f"memory_{int(time.time())}",
                resource_type="memory_percent",
                current_value=current_metrics.memory_percent,
                threshold_value=self.thresholds['memory_percent'],
                severity="high" if current_metrics.memory_percent > 95 else "medium",
                message=f"High memory usage: {current_metrics.memory_percent:.1f}%",
                timestamp=datetime.now()
            )
            alerts.append(alert)
            
        # Check disk threshold
        if current_metrics.disk_percent > self.thresholds.get('disk_percent', 90):
            alert = ResourceAlert(
                alert_id=f"disk_{int(time.time())}",
                resource_type="disk_percent",
                current_value=current_metrics.disk_percent,
                threshold_value=self.thresholds['disk_percent'],
                severity="critical" if current_metrics.disk_percent > 95 else "high",
                message=f"High disk usage: {current_metrics.disk_percent:.1f}%",
                timestamp=datetime.now()
            )
            alerts.append(alert)
            
        # Check GPU thresholds
        if current_metrics.gpu_metrics:
            for gpu in current_metrics.gpu_metrics:
                if gpu['memory_percent'] > self.thresholds.get('gpu_memory_percent', 85):
                    alert = ResourceAlert(
                        alert_id=f"gpu_memory_{gpu['id']}_{int(time.time())}",
                        resource_type="gpu_memory_percent",
                        current_value=gpu['memory_percent'],
                        threshold_value=self.thresholds['gpu_memory_percent'],
                        severity="high",
                        message=f"High GPU memory usage on GPU {gpu['id']}: {gpu['memory_percent']:.1f}%",
                        timestamp=datetime.now()
                    )
                    alerts.append(alert)
                    
                if gpu['load_percent'] > self.thresholds.get('gpu_utilization_percent', 90):
                    alert = ResourceAlert(
                        alert_id=f"gpu_util_{gpu['id']}_{int(time.time())}",
                        resource_type="gpu_utilization_percent",
                        current_value=gpu['load_percent'],
                        threshold_value=self.thresholds['gpu_utilization_percent'],
                        severity="high",
                        message=f"High GPU utilization on GPU {gpu['id']}: {gpu['load_percent']:.1f}%",
                        timestamp=datetime.now()
                    )
                    alerts.append(alert)
                    
        # Store alerts
        with self._lock:
            for alert in alerts:
                self.resource_alerts[alert.alert_id] = alert
                self.alert_history.append(alert)
                self.stats['alerts_generated'] += 1
                
                logger.warning(f"Resource alert: {alert.message}")
                
    def _cleanup_old_data(self):
        """Clean up old data based on retention policy"""
        cutoff_time = datetime.now() - timedelta(hours=self.retention_hours)
        
        with self._lock:
            # Clean up system metrics
            self.system_metrics = deque(
                [m for m in self.system_metrics if m.timestamp >= cutoff_time],
                maxlen=self.system_metrics.maxlen
            )
            
            # Clean up process metrics
            for pid in list(self.process_metrics.keys()):
                self.process_metrics[pid] = deque(
                    [m for m in self.process_metrics[pid] if m.timestamp >= cutoff_time],
                    maxlen=self.process_metrics[pid].maxlen
                )
                
    def _collection_loop(self):
        """Main collection loop"""
        while self._running:
            try:
                self._collect_system_metrics()
                self._collect_process_metrics()
                self._check_thresholds()
                
                time.sleep(self.collection_interval)
                
            except Exception as e:
                logger.error(f"Error in collection loop: {e}")
                time.sleep(5)
                
    def _analysis_loop(self):
        """Analysis loop for periodic tasks"""
        while self._running:
            try:
                # Clean up old data
                self._cleanup_old_data()
                
                # Log statistics
                if self.stats['metrics_collected'] > 0 and self.stats['metrics_collected'] % 100 == 0:
                    logger.info(f"Resource tracking stats: {self.stats}")
                    
                time.sleep(300)  # Run every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in analysis loop: {e}")
                time.sleep(60)

# Example usage and testing
def main():
    """Example usage of the resource utilization tracker"""
    tracker = ResourceUtilizationTracker(collection_interval=2)
    
    try:
        tracker.start_monitoring()
        
        # Monitor current process
        import os
        current_pid = os.getpid()
        tracker.add_process_to_monitor(current_pid, "resource_tracker")
        
        # Set custom thresholds
        tracker.set_threshold('cpu_percent', 70.0)
        tracker.set_threshold('memory_percent', 80.0)
        
        # Get current metrics
        current_metrics = tracker.get_current_metrics()
        if current_metrics:
            print(f"Current CPU: {current_metrics.cpu_percent:.1f}%")
            print(f"Current Memory: {current_metrics.memory_percent:.1f}%")
            print(f"Current Disk: {current_metrics.disk_percent:.1f}%")
            
        # Get resource statistics
        cpu_stats = tracker.get_resource_statistics('cpu_percent', hours=1)
        if cpu_stats:
            print(f"CPU Statistics: {cpu_stats}")
            
        # Get efficiency scores
        efficiency = tracker.get_resource_efficiency_score()
        print(f"Efficiency Scores: {efficiency}")
        
        # Get capacity forecast
        forecast = tracker.get_capacity_forecast('cpu_percent', forecast_hours=6)
        print(f"CPU Forecast: {forecast}")
        
        # Check for alerts
        alerts = tracker.get_active_alerts()
        print(f"Active alerts: {len(alerts)}")
        
        time.sleep(10)  # Let monitoring run for a bit
        
    finally:
        tracker.stop_monitoring()

if __name__ == "__main__":
    main()