"""
Error Rate Monitoring System for Bharat-FM MLOps Platform

This module provides comprehensive error rate monitoring across all AI models
and services in the Bharat-FM ecosystem. It tracks error patterns, calculates
metrics, and provides alerts for abnormal error conditions.

Features:
- Real-time error rate calculation
- Error pattern analysis
- Threshold-based alerting
- Historical error tracking
- Model-specific error monitoring
- Service health monitoring
"""

import time
import threading
import json
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
class ErrorMetrics:
    """Error metrics data structure"""
    timestamp: datetime
    model_id: str
    service_name: str
    error_count: int
    total_requests: int
    error_rate: float
    error_types: Dict[str, int]
    response_time_avg: float
    response_time_p95: float
    
@dataclass
class ErrorAlert:
    """Alert configuration and data"""
    alert_id: str
    model_id: str
    threshold: float
    current_rate: float
    severity: str  # 'low', 'medium', 'high', 'critical'
    message: str
    timestamp: datetime
    resolved: bool = False

class ErrorRateMonitor:
    """
    Comprehensive error rate monitoring system for AI models and services
    """
    
    def __init__(self, window_size: int = 1000, alert_thresholds: Dict[str, float] = None):
        self.window_size = window_size
        self.alert_thresholds = alert_thresholds or {
            'low': 0.01,      # 1%
            'medium': 0.05,   # 5%
            'high': 0.10,     # 10%
            'critical': 0.20  # 20%
        }
        
        # Data storage
        self.error_windows = defaultdict(lambda: deque(maxlen=window_size))
        self.error_metrics = defaultdict(list)
        self.active_alerts = {}
        self.alert_history = []
        
        # Threading
        self._lock = threading.Lock()
        self._running = False
        self._monitor_thread = None
        
        # Statistics
        self.total_models = 0
        self.total_services = 0
        
    def start_monitoring(self):
        """Start the monitoring system"""
        if self._running:
            logger.warning("Monitor already running")
            return
            
        self._running = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("Error rate monitoring started")
        
    def stop_monitoring(self):
        """Stop the monitoring system"""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        logger.info("Error rate monitoring stopped")
        
    def record_error(self, model_id: str, service_name: str, error_type: str, 
                    response_time: float = None):
        """
        Record an error occurrence
        
        Args:
            model_id: Identifier for the AI model
            service_name: Name of the service
            error_type: Type of error encountered
            response_time: Response time in milliseconds
        """
        with self._lock:
            key = f"{model_id}:{service_name}"
            timestamp = datetime.now()
            
            # Record error in sliding window
            self.error_windows[key].append({
                'timestamp': timestamp,
                'is_error': True,
                'error_type': error_type,
                'response_time': response_time
            })
            
            # Also record successful requests for rate calculation
            self.error_windows[key].append({
                'timestamp': timestamp,
                'is_error': False,
                'error_type': None,
                'response_time': response_time
            })
            
    def record_request(self, model_id: str, service_name: str, 
                      response_time: float = None):
        """
        Record a successful request
        
        Args:
            model_id: Identifier for the AI model
            service_name: Name of the service
            response_time: Response time in milliseconds
        """
        with self._lock:
            key = f"{model_id}:{service_name}"
            timestamp = datetime.now()
            
            # Record successful request
            self.error_windows[key].append({
                'timestamp': timestamp,
                'is_error': False,
                'error_type': None,
                'response_time': response_time
            })
            
    def calculate_error_rate(self, model_id: str, service_name: str, 
                           window_minutes: int = 5) -> Optional[float]:
        """
        Calculate error rate for a specific model and service
        
        Args:
            model_id: Identifier for the AI model
            service_name: Name of the service
            window_minutes: Time window in minutes for calculation
            
        Returns:
            Error rate as float (0.0 to 1.0) or None if no data
        """
        key = f"{model_id}:{service_name}"
        cutoff_time = datetime.now() - timedelta(minutes=window_minutes)
        
        with self._lock:
            if key not in self.error_windows:
                return None
                
            window_data = [
                entry for entry in self.error_windows[key]
                if entry['timestamp'] >= cutoff_time
            ]
            
            if not window_data:
                return None
                
            error_count = sum(1 for entry in window_data if entry['is_error'])
            total_requests = len(window_data)
            
            return error_count / total_requests if total_requests > 0 else 0.0
            
    def get_error_metrics(self, model_id: str, service_name: str, 
                         window_minutes: int = 5) -> Optional[ErrorMetrics]:
        """
        Get comprehensive error metrics for a model and service
        
        Args:
            model_id: Identifier for the AI model
            service_name: Name of the service
            window_minutes: Time window in minutes
            
        Returns:
            ErrorMetrics object or None if no data
        """
        key = f"{model_id}:{service_name}"
        cutoff_time = datetime.now() - timedelta(minutes=window_minutes)
        
        with self._lock:
            if key not in self.error_windows:
                return None
                
            window_data = [
                entry for entry in self.error_windows[key]
                if entry['timestamp'] >= cutoff_time
            ]
            
            if not window_data:
                return None
                
            error_count = sum(1 for entry in window_data if entry['is_error'])
            total_requests = len(window_data)
            error_rate = error_count / total_requests if total_requests > 0 else 0.0
            
            # Calculate error types distribution
            error_types = defaultdict(int)
            response_times = []
            
            for entry in window_data:
                if entry['is_error'] and entry['error_type']:
                    error_types[entry['error_type']] += 1
                if entry['response_time']:
                    response_times.append(entry['response_time'])
                    
            # Calculate response time statistics
            avg_response_time = statistics.mean(response_times) if response_times else 0
            p95_response_time = np.percentile(response_times, 95) if response_times else 0
            
            return ErrorMetrics(
                timestamp=datetime.now(),
                model_id=model_id,
                service_name=service_name,
                error_count=error_count,
                total_requests=total_requests,
                error_rate=error_rate,
                error_types=dict(error_types),
                response_time_avg=avg_response_time,
                response_time_p95=p95_response_time
            )
            
    def check_alerts(self) -> List[ErrorAlert]:
        """
        Check for error rate alerts across all models and services
        
        Returns:
            List of ErrorAlert objects for active alerts
        """
        alerts = []
        
        with self._lock:
            for key in self.error_windows:
                model_id, service_name = key.split(':', 1)
                metrics = self.get_error_metrics(model_id, service_name)
                
                if metrics and metrics.error_rate > 0:
                    # Determine alert severity
                    severity = None
                    for level, threshold in sorted(self.alert_thresholds.items(), 
                                                 key=lambda x: x[1], reverse=True):
                        if metrics.error_rate >= threshold:
                            severity = level
                            break
                            
                    if severity:
                        alert_id = f"{key}_{int(time.time())}"
                        alert = ErrorAlert(
                            alert_id=alert_id,
                            model_id=model_id,
                            threshold=self.alert_thresholds[severity],
                            current_rate=metrics.error_rate,
                            severity=severity,
                            message=f"High error rate detected: {metrics.error_rate:.2%} for {model_id}:{service_name}",
                            timestamp=datetime.now()
                        )
                        
                        # Check if this is a new alert or update existing
                        existing_key = f"{model_id}:{service_name}"
                        if existing_key not in self.active_alerts:
                            self.active_alerts[existing_key] = alert
                            alerts.append(alert)
                            self.alert_history.append(alert)
                            
        return alerts
        
    def get_system_health(self) -> Dict[str, Any]:
        """
        Get overall system health status
        
        Returns:
            Dictionary containing system health information
        """
        with self._lock:
            total_models = len(set(key.split(':')[0] for key in self.error_windows))
            total_services = len(set(key.split(':')[1] for key in self.error_windows))
            
            # Calculate overall error rate
            all_errors = 0
            all_requests = 0
            
            for key in self.error_windows:
                recent_data = [
                    entry for entry in self.error_windows[key]
                    if entry['timestamp'] >= datetime.now() - timedelta(minutes=5)
                ]
                
                if recent_data:
                    all_errors += sum(1 for entry in recent_data if entry['is_error'])
                    all_requests += len(recent_data)
                    
            overall_error_rate = all_errors / all_requests if all_requests > 0 else 0.0
            
            return {
                'timestamp': datetime.now().isoformat(),
                'total_models_monitored': total_models,
                'total_services_monitored': total_services,
                'overall_error_rate': overall_error_rate,
                'active_alerts_count': len(self.active_alerts),
                'alert_thresholds': self.alert_thresholds,
                'monitoring_status': 'active' if self._running else 'inactive'
            }
            
    def resolve_alert(self, alert_id: str):
        """
        Resolve an active alert
        
        Args:
            alert_id: ID of the alert to resolve
        """
        with self._lock:
            for key, alert in list(self.active_alerts.items()):
                if alert.alert_id == alert_id:
                    alert.resolved = True
                    del self.active_alerts[key]
                    logger.info(f"Alert {alert_id} resolved")
                    break
                    
    def get_alert_history(self, hours: int = 24) -> List[ErrorAlert]:
        """
        Get alert history for the specified time period
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            List of ErrorAlert objects
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with self._lock:
            return [
                alert for alert in self.alert_history
                if alert.timestamp >= cutoff_time
            ]
            
    def export_metrics(self, filename: str = None) -> str:
        """
        Export current metrics to JSON file
        
        Args:
            filename: Optional filename to save to
            
        Returns:
            JSON string of current metrics
        """
        with self._lock:
            export_data = {
                'timestamp': datetime.now().isoformat(),
                'system_health': self.get_system_health(),
                'active_alerts': [asdict(alert) for alert in self.active_alerts.values()],
                'model_metrics': {}
            }
            
            # Collect metrics for all models
            for key in self.error_windows:
                model_id, service_name = key.split(':', 1)
                metrics = self.get_error_metrics(model_id, service_name)
                if metrics:
                    if model_id not in export_data['model_metrics']:
                        export_data['model_metrics'][model_id] = {}
                    export_data['model_metrics'][model_id][service_name] = asdict(metrics)
                    
            json_data = json.dumps(export_data, indent=2, default=str)
            
            if filename:
                with open(filename, 'w') as f:
                    f.write(json_data)
                logger.info(f"Metrics exported to {filename}")
                
            return json_data
            
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self._running:
            try:
                # Check for alerts
                alerts = self.check_alerts()
                
                # Log new alerts
                for alert in alerts:
                    logger.warning(f"ALERT [{alert.severity.upper()}]: {alert.message}")
                    
                # Clean up old data periodically
                self._cleanup_old_data()
                
                # Sleep for monitoring interval
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(5)
                
    def _cleanup_old_data(self):
        """Clean up old data to prevent memory bloat"""
        cutoff_time = datetime.now() - timedelta(hours=24)
        
        with self._lock:
            for key in list(self.error_windows.keys()):
                # Keep only recent data in the window
                self.error_windows[key] = deque(
                    [entry for entry in self.error_windows[key] 
                     if entry['timestamp'] >= cutoff_time],
                    maxlen=self.window_size
                )

# Example usage and testing
def main():
    """Example usage of the error rate monitoring system"""
    monitor = ErrorRateMonitor(window_size=1000)
    
    try:
        monitor.start_monitoring()
        
        # Simulate some requests and errors
        model_id = "bharat-gpt-7b"
        service_name = "chat-service"
        
        # Record some successful requests
        for i in range(95):
            monitor.record_request(model_id, service_name, response_time=150 + i % 100)
            
        # Record some errors
        for i in range(5):
            monitor.record_error(model_id, service_name, "timeout", response_time=5000)
            
        # Check metrics
        metrics = monitor.get_error_metrics(model_id, service_name)
        if metrics:
            print(f"Error rate: {metrics.error_rate:.2%}")
            print(f"Error count: {metrics.error_count}")
            print(f"Total requests: {metrics.total_requests}")
            
        # Check for alerts
        alerts = monitor.check_alerts()
        for alert in alerts:
            print(f"ALERT: {alert.message}")
            
        # Get system health
        health = monitor.get_system_health()
        print(f"System health: {health}")
        
        # Export metrics
        monitor.export_metrics("error_metrics.json")
        
        time.sleep(2)  # Let monitoring run for a bit
        
    finally:
        monitor.stop_monitoring()

if __name__ == "__main__":
    main()