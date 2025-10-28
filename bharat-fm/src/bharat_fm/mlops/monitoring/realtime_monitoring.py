"""
Real-time Monitoring System for Bharat-FM MLOps Platform

This module provides comprehensive real-time monitoring capabilities for AI models,
services, and infrastructure. It offers live dashboards, real-time metrics,
and instant anomaly detection with automated alerting.

Features:
- Real-time metrics collection and visualization
- Live dashboard with WebSocket updates
- Anomaly detection and alerting
- Performance monitoring
- Health checks and status monitoring
- Event streaming and processing
"""

import time
import threading
import json
import asyncio
import websockets
import logging
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
import statistics
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RealtimeMetric:
    """Real-time metric data structure"""
    timestamp: datetime
    metric_name: str
    value: float
    unit: str
    source: str
    tags: Dict[str, str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = {}

@dataclass
class HealthStatus:
    """Health status data structure"""
    component: str
    status: str  # 'healthy', 'warning', 'critical', 'unknown'
    message: str
    timestamp: datetime
    metrics: Dict[str, float] = None
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {}

@dataclass
class AnomalyEvent:
    """Anomaly detection event"""
    event_id: str
    metric_name: str
    source: str
    anomaly_score: float
    threshold: float
    severity: str
    description: str
    timestamp: datetime
    resolved: bool = False

class RealtimeMonitor:
    """
    Real-time monitoring system with WebSocket support
    """
    
    def __init__(self, port: int = 8765, max_connections: int = 100):
        self.port = port
        self.max_connections = max_connections
        
        # Data storage
        self.metrics_buffer = defaultdict(lambda: deque(maxlen=1000))
        self.health_status = {}
        self.anomaly_events = {}
        self.websocket_clients = set()
        
        # Configuration
        self.anomaly_thresholds = {
            'default': 2.0,  # Standard deviations
            'error_rate': 3.0,
            'response_time': 2.5,
            'cpu_usage': 2.0,
            'memory_usage': 2.0
        }
        
        # Monitoring configuration
        self.monitored_metrics = set()
        self.health_checks = {}
        
        # Threading and async
        self._lock = threading.Lock()
        self._running = False
        self._monitor_thread = None
        self._websocket_server = None
        self._executor = ThreadPoolExecutor(max_workers=10)
        
        # Statistics
        self.stats = {
            'metrics_received': 0,
            'anomalies_detected': 0,
            'health_checks_performed': 0,
            'websocket_connections': 0
        }
        
    def start_monitoring(self):
        """Start the real-time monitoring system"""
        if self._running:
            logger.warning("Real-time monitoring already running")
            return
            
        self._running = True
        
        # Start monitoring thread
        self._monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitor_thread.start()
        
        # Start WebSocket server
        self._start_websocket_server()
        
        logger.info(f"Real-time monitoring started on port {self.port}")
        
    def stop_monitoring(self):
        """Stop the real-time monitoring system"""
        self._running = False
        
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
            
        if self._websocket_server:
            self._websocket_server.close()
            
        self._executor.shutdown(wait=True)
        
        logger.info("Real-time monitoring stopped")
        
    def add_metric(self, metric_name: str, value: float, unit: str, 
                   source: str, tags: Dict[str, str] = None):
        """
        Add a real-time metric
        
        Args:
            metric_name: Name of the metric
            value: Metric value
            unit: Unit of measurement
            source: Source of the metric
            tags: Additional tags for the metric
        """
        metric = RealtimeMetric(
            timestamp=datetime.now(),
            metric_name=metric_name,
            value=value,
            unit=unit,
            source=source,
            tags=tags or {}
        )
        
        with self._lock:
            # Store metric
            key = f"{source}:{metric_name}"
            self.metrics_buffer[key].append(metric)
            self.monitored_metrics.add(metric_name)
            self.stats['metrics_received'] += 1
            
            # Check for anomalies
            anomaly = self._detect_anomaly(metric)
            if anomaly:
                self.anomaly_events[anomaly.event_id] = anomaly
                self.stats['anomalies_detected'] += 1
                
                logger.warning(f"Anomaly detected: {anomaly.description}")
                
            # Broadcast to WebSocket clients
            self._broadcast_metric_update(metric, anomaly)
            
    def add_health_check(self, component: str, check_function: Callable, 
                        interval_seconds: int = 30):
        """
        Add a health check for a component
        
        Args:
            component: Name of the component
            check_function: Function that returns HealthStatus
            interval_seconds: Interval between checks
        """
        self.health_checks[component] = {
            'function': check_function,
            'interval': interval_seconds,
            'last_check': datetime.min
        }
        
    def get_current_metrics(self, source: str = None, 
                           metric_name: str = None) -> List[RealtimeMetric]:
        """
        Get current metrics, optionally filtered
        
        Args:
            source: Optional source filter
            metric_name: Optional metric name filter
            
        Returns:
            List of RealtimeMetric objects
        """
        with self._lock:
            metrics = []
            
            for key, metric_queue in self.metrics_buffer.items():
                if metric_queue:
                    latest_metric = metric_queue[-1]
                    
                    # Apply filters
                    if source and latest_metric.source != source:
                        continue
                    if metric_name and latest_metric.metric_name != metric_name:
                        continue
                        
                    metrics.append(latest_metric)
                    
            return metrics
            
    def get_metric_history(self, source: str, metric_name: str, 
                          minutes: int = 5) -> List[RealtimeMetric]:
        """
        Get historical data for a specific metric
        
        Args:
            source: Source of the metric
            metric_name: Name of the metric
            minutes: Number of minutes to look back
            
        Returns:
            List of RealtimeMetric objects
        """
        key = f"{source}:{metric_name}"
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        
        with self._lock:
            if key not in self.metrics_buffer:
                return []
                
            return [
                metric for metric in self.metrics_buffer[key]
                if metric.timestamp >= cutoff_time
            ]
            
    def get_health_status(self, component: str = None) -> Dict[str, HealthStatus]:
        """
        Get health status of components
        
        Args:
            component: Optional component filter
            
        Returns:
            Dictionary of HealthStatus objects
        """
        with self._lock:
            if component:
                return {component: self.health_status.get(component)}
            return self.health_status.copy()
            
    def get_active_anomalies(self) -> List[AnomalyEvent]:
        """
        Get active anomaly events
        
        Returns:
            List of AnomalyEvent objects
        """
        with self._lock:
            return [event for event in self.anomaly_events.values() if not event.resolved]
            
    def resolve_anomaly(self, event_id: str):
        """
        Resolve an anomaly event
        
        Args:
            event_id: ID of the anomaly to resolve
        """
        with self._lock:
            if event_id in self.anomaly_events:
                self.anomaly_events[event_id].resolved = True
                logger.info(f"Resolved anomaly: {event_id}")
                
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get monitoring statistics
        
        Returns:
            Dictionary containing statistics
        """
        with self._lock:
            return {
                'timestamp': datetime.now().isoformat(),
                'stats': self.stats.copy(),
                'monitored_metrics_count': len(self.monitored_metrics),
                'active_anomalies_count': len(self.get_active_anomalies()),
                'health_checks_count': len(self.health_checks),
                'websocket_clients_count': len(self.websocket_clients)
            }
            
    def create_dashboard_config(self) -> Dict[str, Any]:
        """
        Create dashboard configuration for frontend
        
        Returns:
            Dictionary with dashboard configuration
        """
        with self._lock:
            config = {
                'timestamp': datetime.now().isoformat(),
                'metrics': [],
                'health_components': list(self.health_status.keys()),
                'anomaly_thresholds': self.anomaly_thresholds.copy()
            }
            
            # Get unique metric names
            unique_metrics = set()
            for key in self.metrics_buffer.keys():
                source, metric_name = key.split(':', 1)
                unique_metrics.add(metric_name)
                
            config['metrics'] = list(unique_metrics)
            
            return config
            
    def export_realtime_data(self, filename: str = None) -> str:
        """
        Export real-time data to JSON file
        
        Args:
            filename: Optional filename to save to
            
        Returns:
            JSON string of real-time data
        """
        with self._lock:
            export_data = {
                'timestamp': datetime.now().isoformat(),
                'current_metrics': [asdict(metric) for metric in self.get_current_metrics()],
                'health_status': {k: asdict(v) for k, v in self.health_status.items()},
                'active_anomalies': [asdict(anomaly) for anomaly in self.get_active_anomalies()],
                'statistics': self.get_statistics(),
                'dashboard_config': self.create_dashboard_config()
            }
            
            json_data = json.dumps(export_data, indent=2, default=str)
            
            if filename:
                with open(filename, 'w') as f:
                    f.write(json_data)
                logger.info(f"Real-time data exported to {filename}")
                
            return json_data
            
    def _detect_anomaly(self, metric: RealtimeMetric) -> Optional[AnomalyEvent]:
        """
        Detect anomalies in metric data
        
        Args:
            metric: RealtimeMetric to check
            
        Returns:
            AnomalyEvent if anomaly detected, None otherwise
        """
        key = f"{metric.source}:{metric.metric_name}"
        
        with self._lock:
            if key not in self.metrics_buffer or len(self.metrics_buffer[key]) < 10:
                return None
                
            # Get recent values
            recent_values = [m.value for m in list(self.metrics_buffer[key])[-50:]]
            
            if len(recent_values) < 10:
                return None
                
            # Calculate statistics
            mean = statistics.mean(recent_values)
            std_dev = statistics.stdev(recent_values) if len(recent_values) > 1 else 0
            
            if std_dev == 0:
                return None
                
            # Calculate z-score
            z_score = abs(metric.value - mean) / std_dev
            
            # Get threshold for this metric type
            threshold = self.anomaly_thresholds.get(metric.metric_name, 
                                                   self.anomaly_thresholds['default'])
            
            # Check if anomaly
            if z_score > threshold:
                severity = 'low' if z_score < threshold * 1.5 else 'medium' if z_score < threshold * 2 else 'high'
                
                return AnomalyEvent(
                    event_id=f"anomaly_{int(time.time())}_{hash(key) % 1000}",
                    metric_name=metric.metric_name,
                    source=metric.source,
                    anomaly_score=z_score,
                    threshold=threshold,
                    severity=severity,
                    description=f"Anomaly detected in {metric.metric_name}: {metric.value:.2f} (z-score: {z_score:.2f})",
                    timestamp=datetime.now()
                )
                
        return None
        
    def _perform_health_checks(self):
        """Perform health checks for all components"""
        current_time = datetime.now()
        
        for component, check_config in self.health_checks.items():
            # Check if it's time to perform health check
            if (current_time - check_config['last_check']).total_seconds() < check_config['interval']:
                continue
                
            try:
                # Perform health check
                health_status = check_config['function']()
                
                with self._lock:
                    self.health_status[component] = health_status
                    check_config['last_check'] = current_time
                    self.stats['health_checks_performed'] += 1
                    
                # Broadcast health status update
                self._broadcast_health_update(component, health_status)
                
            except Exception as e:
                logger.error(f"Health check failed for {component}: {e}")
                
                # Create error health status
                error_status = HealthStatus(
                    component=component,
                    status='critical',
                    message=f"Health check failed: {str(e)}",
                    timestamp=current_time
                )
                
                with self._lock:
                    self.health_status[component] = error_status
                    self.stats['health_checks_performed'] += 1
                    
                self._broadcast_health_update(component, error_status)
                
    def _broadcast_metric_update(self, metric: RealtimeMetric, anomaly: AnomalyEvent = None):
        """Broadcast metric update to WebSocket clients"""
        if not self.websocket_clients:
            return
            
        message = {
            'type': 'metric_update',
            'metric': asdict(metric),
            'anomaly': asdict(anomaly) if anomaly else None
        }
        
        self._broadcast_to_clients(message)
        
    def _broadcast_health_update(self, component: str, health_status: HealthStatus):
        """Broadcast health status update to WebSocket clients"""
        if not self.websocket_clients:
            return
            
        message = {
            'type': 'health_update',
            'component': component,
            'health_status': asdict(health_status)
        }
        
        self._broadcast_to_clients(message)
        
    def _broadcast_to_clients(self, message: Dict[str, Any]):
        """Broadcast message to all WebSocket clients"""
        if not self.websocket_clients:
            return
            
        # Create message string
        message_str = json.dumps(message, default=str)
        
        # Send to all clients in separate threads
        for client in list(self.websocket_clients):
            try:
                self._executor.submit(self._send_to_client, client, message_str)
            except Exception as e:
                logger.error(f"Failed to schedule message to client: {e}")
                
    def _send_to_client(self, client, message: str):
        """Send message to a specific WebSocket client"""
        try:
            asyncio.run(client.send(message))
        except Exception as e:
            logger.error(f"Failed to send message to client: {e}")
            # Remove disconnected client
            with self._lock:
                if client in self.websocket_clients:
                    self.websocket_clients.remove(client)
                    
    async def _handle_websocket_connection(self, websocket, path):
        """Handle WebSocket connections"""
        with self._lock:
            self.websocket_clients.add(websocket)
            self.stats['websocket_connections'] += 1
            
        logger.info(f"WebSocket client connected: {websocket.remote_address}")
        
        try:
            # Send initial data
            initial_data = {
                'type': 'initial_data',
                'config': self.create_dashboard_config(),
                'current_metrics': [asdict(m) for m in self.get_current_metrics()],
                'health_status': {k: asdict(v) for k, v in self.get_health_status().items()},
                'active_anomalies': [asdict(a) for a in self.get_active_anomalies()]
            }
            
            await websocket.send(json.dumps(initial_data, default=str))
            
            # Keep connection alive and handle incoming messages
            async for message in websocket:
                try:
                    data = json.loads(message)
                    # Handle client requests if needed
                    if data.get('type') == 'get_history':
                        source = data.get('source')
                        metric_name = data.get('metric_name')
                        minutes = data.get('minutes', 5)
                        
                        history = self.get_metric_history(source, metric_name, minutes)
                        response = {
                            'type': 'metric_history',
                            'source': source,
                            'metric_name': metric_name,
                            'history': [asdict(m) for m in history]
                        }
                        
                        await websocket.send(json.dumps(response, default=str))
                        
                except Exception as e:
                    logger.error(f"Error handling WebSocket message: {e}")
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info("WebSocket client disconnected")
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
        finally:
            with self._lock:
                if websocket in self.websocket_clients:
                    self.websocket_clients.remove(websocket)
                    
    def _start_websocket_server(self):
        """Start the WebSocket server"""
        async def server():
            self._websocket_server = await websockets.serve(
                self._handle_websocket_connection,
                "0.0.0.0",
                self.port,
                max_size=1024*1024  # 1MB max message size
            )
            logger.info(f"WebSocket server started on port {self.port}")
            await self._websocket_server.wait_closed()
            
        # Run WebSocket server in a separate thread
        def run_server():
            asyncio.run(server())
            
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self._running:
            try:
                # Perform health checks
                self._perform_health_checks()
                
                # Clean up old data
                self._cleanup_old_data()
                
                # Log statistics periodically
                if self.stats['metrics_received'] > 0 and self.stats['metrics_received'] % 100 == 0:
                    logger.info(f"Real-time monitoring stats: {self.stats}")
                    
                time.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(5)
                
    def _cleanup_old_data(self):
        """Clean up old data to prevent memory bloat"""
        cutoff_time = datetime.now() - timedelta(hours=1)
        
        with self._lock:
            # Clean up old metrics
            for key in list(self.metrics_buffer.keys()):
                self.metrics_buffer[key] = deque(
                    [m for m in self.metrics_buffer[key] if m.timestamp >= cutoff_time],
                    maxlen=self.metrics_buffer[key].maxlen
                )
                
            # Clean up resolved anomalies older than 24 hours
            old_anomalies = [
                event_id for event_id, event in self.anomaly_events.items()
                if event.resolved and event.timestamp < datetime.now() - timedelta(hours=24)
            ]
            
            for event_id in old_anomalies:
                del self.anomaly_events[event_id]

# Example usage and testing
def main():
    """Example usage of the real-time monitoring system"""
    monitor = RealtimeMonitor(port=8765)
    
    try:
        monitor.start_monitoring()
        
        # Add some health checks
        def check_system_health():
            # Simulate health check
            import random
            status = random.choice(['healthy', 'warning', 'healthy', 'healthy'])
            return HealthStatus(
                component='system',
                status=status,
                message=f'System is {status}',
                timestamp=datetime.now(),
                metrics={'cpu': random.uniform(20, 80), 'memory': random.uniform(30, 70)}
            )
            
        def check_database_health():
            # Simulate database health check
            return HealthStatus(
                component='database',
                status='healthy',
                message='Database is healthy',
                timestamp=datetime.now(),
                metrics={'connections': 15, 'query_time': 25.5}
            )
            
        monitor.add_health_check('system', check_system_health, 30)
        monitor.add_health_check('database', check_database_health, 60)
        
        # Simulate adding metrics
        import random
        for i in range(20):
            monitor.add_metric(
                metric_name='cpu_usage',
                value=random.uniform(10, 90),
                unit='percent',
                source='server1',
                tags={'host': 'server1', 'datacenter': 'dc1'}
            )
            
            monitor.add_metric(
                metric_name='memory_usage',
                value=random.uniform(30, 85),
                unit='percent',
                source='server1',
                tags={'host': 'server1', 'datacenter': 'dc1'}
            )
            
            monitor.add_metric(
                metric_name='response_time',
                value=random.uniform(50, 500),
                unit='ms',
                source='api_service',
                tags={'service': 'api', 'version': 'v1'}
            )
            
            time.sleep(1)
            
        # Get current metrics
        current_metrics = monitor.get_current_metrics()
        print(f"Current metrics count: {len(current_metrics)}")
        
        # Get health status
        health_status = monitor.get_health_status()
        print(f"Health status: {health_status}")
        
        # Get active anomalies
        anomalies = monitor.get_active_anomalies()
        print(f"Active anomalies: {len(anomalies)}")
        
        # Get statistics
        stats = monitor.get_statistics()
        print(f"Statistics: {stats}")
        
        # Export data
        monitor.export_realtime_data("realtime_monitoring_data.json")
        
        print("Real-time monitoring system is running. Connect to ws://localhost:8765 for WebSocket updates.")
        
        # Keep running for demonstration
        time.sleep(30)
        
    finally:
        monitor.stop_monitoring()

if __name__ == "__main__":
    main()