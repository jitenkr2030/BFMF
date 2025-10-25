"""
Performance Tracker for Bharat-FM Inference Optimization
Comprehensive performance monitoring and analysis system
"""

import asyncio
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import logging
from collections import defaultdict, deque
import numpy as np
import psutil
import threading

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics"""
    timestamp: datetime
    request_id: str
    model_id: str
    latency: float
    throughput: float
    memory_usage_mb: float
    cpu_usage_percent: float
    gpu_usage_percent: float
    success: bool
    error_message: Optional[str] = None
    cost: float = 0.0
    tokens_processed: int = 0

@dataclass
class SystemMetrics:
    """System-level performance metrics"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_available_mb: float
    disk_usage_percent: float
    network_bytes_sent: int
    network_bytes_recv: int
    gpu_metrics: Dict[str, Any]

class PerformanceTracker:
    """Comprehensive performance tracking system"""
    
    def __init__(self, max_history_size: int = 10000):
        # Performance data storage
        self.request_metrics: deque = deque(maxlen=max_history_size)
        self.system_metrics: deque = deque(maxlen=max_history_size)
        self.model_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Real-time monitoring
        self.monitoring_active = False
        self.monitoring_thread = None
        self.system_stats = {}
        
        # Performance analysis
        self.performance_baselines = defaultdict(dict)
        self.anomaly_thresholds = {
            "latency": 2.0,  # 2x baseline
            "error_rate": 0.1,  # 10% error rate
            "memory_usage": 0.9,  # 90% memory usage
            "cpu_usage": 0.9  # 90% CPU usage
        }
        
        # Alerting
        self.performance_alerts = deque(maxlen=100)
        self.alert_callbacks = []
        
        # Background tasks
        self.analyzer_task = None
        self.running = False
        
    async def start(self):
        """Start performance tracking"""
        if self.running:
            return
        
        self.running = True
        self.monitoring_active = True
        
        # Start system monitoring
        self.monitoring_thread = threading.Thread(target=self._monitor_system, daemon=True)
        self.monitoring_thread.start()
        
        # Start performance analysis
        self.analyzer_task = asyncio.create_task(self._analyze_performance())
        
        logger.info("Performance tracker started")
    
    async def stop(self):
        """Stop performance tracking"""
        self.running = False
        self.monitoring_active = False
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        if self.analyzer_task:
            self.analyzer_task.cancel()
            try:
                await self.analyzer_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Performance tracker stopped")
    
    async def track_request(self, request_id: str, model_id: str, 
                          start_time: float, end_time: float,
                          success: bool, error_message: Optional[str] = None,
                          additional_metrics: Optional[Dict[str, Any]] = None):
        """Track individual request performance"""
        
        latency = end_time - start_time
        
        # Calculate throughput (requests per second)
        throughput = 1.0 / max(latency, 0.001)
        
        # Get system metrics
        system_stats = self._get_current_system_stats()
        
        # Create performance metrics
        metrics = PerformanceMetrics(
            timestamp=datetime.utcnow(),
            request_id=request_id,
            model_id=model_id,
            latency=latency,
            throughput=throughput,
            memory_usage_mb=system_stats.get("memory_usage_mb", 0),
            cpu_usage_percent=system_stats.get("cpu_percent", 0),
            gpu_usage_percent=system_stats.get("gpu_usage_percent", 0),
            success=success,
            error_message=error_message,
            cost=additional_metrics.get("cost", 0.0) if additional_metrics else 0.0,
            tokens_processed=additional_metrics.get("tokens_processed", 0) if additional_metrics else 0
        )
        
        # Store metrics
        self.request_metrics.append(metrics)
        self.model_metrics[model_id].append(metrics)
        
        # Check for anomalies
        await self._check_anomalies(metrics)
        
        # Update baselines
        await self._update_baselines(model_id, metrics)
    
    def _monitor_system(self):
        """Monitor system resources in background thread"""
        while self.monitoring_active:
            try:
                # Collect system metrics
                system_metrics = SystemMetrics(
                    timestamp=datetime.utcnow(),
                    cpu_percent=psutil.cpu_percent(interval=1),
                    memory_percent=psutil.virtual_memory().percent,
                    memory_available_mb=psutil.virtual_memory().available / (1024 * 1024),
                    disk_usage_percent=psutil.disk_usage('/').percent,
                    network_bytes_sent=psutil.net_io_counters().bytes_sent,
                    network_bytes_recv=psutil.net_io_counters().bytes_recv,
                    gpu_metrics=self._get_gpu_metrics()
                )
                
                self.system_metrics.append(system_metrics)
                
                # Sleep for monitoring interval
                time.sleep(5)
                
            except Exception as e:
                logger.error(f"Error in system monitoring: {e}")
                time.sleep(1)
    
    def _get_gpu_metrics(self) -> Dict[str, Any]:
        """Get GPU metrics if available"""
        try:
            # Try to import and use nvidia-ml-py if available
            import pynvml
            
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            
            gpu_metrics = {}
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                
                # Get GPU utilization
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                
                gpu_metrics[f"gpu_{i}"] = {
                    "gpu_utilization": utilization.gpu,
                    "memory_utilization": utilization.memory,
                    "memory_used_mb": memory_info.used / (1024 * 1024),
                    "memory_total_mb": memory_info.total / (1024 * 1024)
                }
            
            pynvml.nvmlShutdown()
            return gpu_metrics
            
        except ImportError:
            # nvidia-ml-py not available
            return {}
        except Exception as e:
            logger.debug(f"GPU metrics not available: {e}")
            return {}
    
    def _get_current_system_stats(self) -> Dict[str, Any]:
        """Get current system statistics"""
        if not self.system_metrics:
            return {}
        
        latest = self.system_metrics[-1]
        return {
            "memory_usage_mb": latest.memory_available_mb,
            "cpu_percent": latest.cpu_percent,
            "gpu_usage_percent": self._calculate_avg_gpu_usage(latest.gpu_metrics)
        }
    
    def _calculate_avg_gpu_usage(self, gpu_metrics: Dict[str, Any]) -> float:
        """Calculate average GPU usage"""
        if not gpu_metrics:
            return 0.0
        
        total_usage = sum(metrics.get("gpu_utilization", 0) for metrics in gpu_metrics.values())
        return total_usage / len(gpu_metrics)
    
    async def _check_anomalies(self, metrics: PerformanceMetrics):
        """Check for performance anomalies"""
        model_id = metrics.model_id
        
        # Check latency anomaly
        baseline_latency = self.performance_baselines[model_id].get("avg_latency", 0.1)
        if metrics.latency > baseline_latency * self.anomaly_thresholds["latency"]:
            alert = {
                "type": "high_latency",
                "model_id": model_id,
                "request_id": metrics.request_id,
                "latency": metrics.latency,
                "baseline": baseline_latency,
                "threshold": baseline_latency * self.anomaly_thresholds["latency"],
                "timestamp": metrics.timestamp.isoformat()
            }
            await self._trigger_alert(alert)
        
        # Check error rate
        if not metrics.success:
            alert = {
                "type": "request_error",
                "model_id": model_id,
                "request_id": metrics.request_id,
                "error": metrics.error_message,
                "timestamp": metrics.timestamp.isoformat()
            }
            await self._trigger_alert(alert)
        
        # Check resource usage
        if metrics.memory_usage_mb > 8000:  # 8GB
            alert = {
                "type": "high_memory_usage",
                "model_id": model_id,
                "request_id": metrics.request_id,
                "memory_usage_mb": metrics.memory_usage_mb,
                "timestamp": metrics.timestamp.isoformat()
            }
            await self._trigger_alert(alert)
        
        if metrics.cpu_usage_percent > 80:
            alert = {
                "type": "high_cpu_usage",
                "model_id": model_id,
                "request_id": metrics.request_id,
                "cpu_usage_percent": metrics.cpu_usage_percent,
                "timestamp": metrics.timestamp.isoformat()
            }
            await self._trigger_alert(alert)
    
    async def _trigger_alert(self, alert: Dict[str, Any]):
        """Trigger performance alert"""
        self.performance_alerts.append(alert)
        
        # Call alert callbacks
        for callback in self.alert_callbacks:
            try:
                await callback(alert)
            except Exception as e:
                logger.error(f"Error in performance alert callback: {e}")
        
        logger.warning(f"Performance alert triggered: {alert}")
    
    def add_alert_callback(self, callback):
        """Add callback for performance alerts"""
        self.alert_callbacks.append(callback)
    
    async def _update_baselines(self, model_id: str, metrics: PerformanceMetrics):
        """Update performance baselines"""
        model_history = self.model_metrics[model_id]
        
        if len(model_history) < 10:
            return
        
        # Calculate baseline metrics
        recent_metrics = list(model_history)[-100:]  # Last 100 requests
        
        baselines = self.performance_baselines[model_id]
        baselines["avg_latency"] = np.mean([m.latency for m in recent_metrics])
        baselines["avg_throughput"] = np.mean([m.throughput for m in recent_metrics])
        baselines["avg_memory_usage"] = np.mean([m.memory_usage_mb for m in recent_metrics])
        baselines["error_rate"] = 1 - (sum(1 for m in recent_metrics if m.success) / len(recent_metrics))
        baselines["p95_latency"] = np.percentile([m.latency for m in recent_metrics], 95)
        baselines["p99_latency"] = np.percentile([m.latency for m in recent_metrics], 99)
    
    async def _analyze_performance(self):
        """Background task for performance analysis"""
        while self.running:
            try:
                await asyncio.sleep(300)  # Analyze every 5 minutes
                
                # Generate performance report
                report = await self.generate_performance_report()
                
                # Log significant findings
                if report.get("anomalies_detected", 0) > 0:
                    logger.info(f"Performance analysis completed: {report['anomalies_detected']} anomalies detected")
                
                # Optimize baselines
                await self._optimize_baselines()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in performance analysis: {e}")
    
    async def _optimize_baselines(self):
        """Optimize performance baselines"""
        for model_id, baselines in self.performance_baselines.items():
            # Adjust baselines based on recent performance trends
            model_history = self.model_metrics[model_id]
            
            if len(model_history) < 50:
                continue
            
            # Calculate trend
            recent_latencies = [m.latency for m in list(model_history)[-20:]]
            older_latencies = [m.latency for m in list(model_history)[-50:-20]]
            
            if recent_latencies and older_latencies:
                recent_avg = np.mean(recent_latencies)
                older_avg = np.mean(older_latencies)
                
                # If performance is degrading, adjust baseline
                if recent_avg > older_avg * 1.2:  # 20% degradation
                    baselines["avg_latency"] = recent_avg
                    logger.info(f"Updated baseline for {model_id} due to performance degradation")
    
    async def generate_performance_report(self, 
                                       model_id: Optional[str] = None,
                                       hours: int = 24) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        now = datetime.utcnow()
        start_time = now - timedelta(hours=hours)
        
        # Filter metrics by time range
        time_filtered_metrics = [
            m for m in self.request_metrics
            if m.timestamp >= start_time and (model_id is None or m.model_id == model_id)
        ]
        
        if not time_filtered_metrics:
            return {"error": "No metrics available for specified period"}
        
        # Calculate summary statistics
        total_requests = len(time_filtered_metrics)
        successful_requests = sum(1 for m in time_filtered_metrics if m.success)
        error_rate = 1 - (successful_requests / total_requests)
        
        latencies = [m.latency for m in time_filtered_metrics]
        throughputs = [m.throughput for m in time_filtered_metrics]
        memory_usages = [m.memory_usage_mb for m in time_filtered_metrics]
        
        report = {
            "period": {
                "start": start_time.isoformat(),
                "end": now.isoformat(),
                "hours": hours
            },
            "summary": {
                "total_requests": total_requests,
                "successful_requests": successful_requests,
                "error_rate": error_rate,
                "avg_latency": np.mean(latencies),
                "p50_latency": np.percentile(latencies, 50),
                "p95_latency": np.percentile(latencies, 95),
                "p99_latency": np.percentile(latencies, 99),
                "avg_throughput": np.mean(throughputs),
                "avg_memory_usage_mb": np.mean(memory_usages),
                "peak_memory_usage_mb": max(memory_usages) if memory_usages else 0
            },
            "model_breakdown": {},
            "anomalies_detected": len([a for a in self.performance_alerts 
                                     if datetime.fromisoformat(a["timestamp"]) >= start_time]),
            "recommendations": await self._generate_performance_recommendations()
        }
        
        # Model breakdown
        model_stats = defaultdict(list)
        for m in time_filtered_metrics:
            model_stats[m.model_id].append(m)
        
        for mid, metrics in model_stats.items():
            model_latencies = [m.latency for m in metrics]
            model_throughputs = [m.throughput for m in metrics]
            model_successes = sum(1 for m in metrics if m.success)
            
            report["model_breakdown"][mid] = {
                "request_count": len(metrics),
                "success_rate": model_successes / len(metrics),
                "avg_latency": np.mean(model_latencies),
                "p95_latency": np.percentile(model_latencies, 95),
                "avg_throughput": np.mean(model_throughputs),
                "error_count": len(metrics) - model_successes
            }
        
        return report
    
    async def _generate_performance_recommendations(self) -> List[Dict[str, Any]]:
        """Generate performance optimization recommendations"""
        recommendations = []
        
        # Analyze overall performance
        if self.request_metrics:
            recent_metrics = list(self.request_metrics)[-100:]
            
            # High error rate
            error_rate = 1 - (sum(1 for m in recent_metrics if m.success) / len(recent_metrics))
            if error_rate > 0.05:  # 5% error rate
                recommendations.append({
                    "type": "high_error_rate",
                    "severity": "high",
                    "issue": f"High error rate: {error_rate:.1%}",
                    "recommendation": "Investigate error patterns and implement better error handling"
                })
            
            # High latency
            avg_latency = np.mean([m.latency for m in recent_metrics])
            if avg_latency > 1.0:  # 1 second
                recommendations.append({
                    "type": "high_latency",
                    "severity": "medium",
                    "issue": f"High average latency: {avg_latency:.2f}s",
                    "recommendation": "Consider model optimization, batching, or scaling"
                })
            
            # Memory usage
            avg_memory = np.mean([m.memory_usage_mb for m in recent_metrics])
            if avg_memory > 4000:  # 4GB
                recommendations.append({
                    "type": "high_memory_usage",
                    "severity": "medium",
                    "issue": f"High average memory usage: {avg_memory:.0f}MB",
                    "recommendation": "Consider model quantization or memory optimization"
                })
        
        return recommendations
    
    async def get_real_time_metrics(self) -> Dict[str, Any]:
        """Get real-time performance metrics"""
        if not self.system_metrics:
            return {"status": "no_data"}
        
        latest_system = self.system_metrics[-1]
        
        # Calculate request metrics from last minute
        now = datetime.utcnow()
        minute_ago = now - timedelta(minutes=1)
        
        recent_requests = [
            m for m in self.request_metrics
            if m.timestamp >= minute_ago
        ]
        
        request_rate = len(recent_requests) / 60.0  # requests per second
        
        return {
            "timestamp": now.isoformat(),
            "system": {
                "cpu_percent": latest_system.cpu_percent,
                "memory_percent": latest_system.memory_percent,
                "memory_available_mb": latest_system.memory_available_mb,
                "disk_usage_percent": latest_system.disk_usage_percent,
                "gpu_metrics": latest_system.gpu_metrics
            },
            "requests": {
                "rate_per_second": request_rate,
                "total_last_minute": len(recent_requests),
                "success_rate_last_minute": (
                    sum(1 for m in recent_requests if m.success) / max(1, len(recent_requests))
                )
            },
            "alerts_count": len(self.performance_alerts)
        }