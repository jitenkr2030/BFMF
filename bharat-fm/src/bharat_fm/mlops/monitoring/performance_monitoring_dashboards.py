"""
Performance Monitoring Dashboards for Bharat-FM MLOps Platform

This module provides comprehensive dashboard generation and management for
monitoring AI model performance, system health, and business metrics. It supports
real-time dashboards, custom visualizations, and automated reporting.

Features:
- Real-time dashboard generation
- Custom metric visualization
- Interactive charts and graphs
- Performance threshold monitoring
- Automated report generation
- Dashboard sharing and collaboration
- Multi-tenant support
"""

import time
import threading
import json
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque
import statistics
import numpy as np
import pandas as pd
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DashboardType(Enum):
    """Dashboard type enumeration"""
    SYSTEM_OVERVIEW = "system_overview"
    MODEL_PERFORMANCE = "model_performance"
    RESOURCE_UTILIZATION = "resource_utilization"
    BUSINESS_METRICS = "business_metrics"
    CUSTOM = "custom"

class ChartType(Enum):
    """Chart type enumeration"""
    LINE = "line"
    BAR = "bar"
    PIE = "pie"
    AREA = "area"
    SCATTER = "scatter"
    GAUGE = "gauge"
    HEATMAP = "heatmap"
    TABLE = "table"

class TimeRange(Enum):
    """Time range enumeration"""
    LAST_HOUR = "last_hour"
    LAST_24_HOURS = "last_24_hours"
    LAST_7_DAYS = "last_7_days"
    LAST_30_DAYS = "last_30_days"
    CUSTOM = "custom"

@dataclass
class MetricDefinition:
    """Metric definition for dashboards"""
    metric_id: str
    name: str
    description: str
    unit: str
    aggregation: str  # 'avg', 'sum', 'count', 'max', 'min'
    chart_type: ChartType
    color: str = "#3b82f6"
    threshold_value: float = None
    threshold_operator: str = None  # 'gt', 'lt', 'eq'
    
@dataclass
class DashboardWidget:
    """Dashboard widget definition"""
    widget_id: str
    title: str
    widget_type: str  # 'chart', 'metric', 'text', 'table'
    metrics: List[str]
    chart_type: ChartType = None
    time_range: TimeRange = TimeRange.LAST_24_HOURS
    position: Dict[str, int] = None  # {'x': 0, 'y': 0, 'width': 4, 'height': 3}
    refresh_interval_seconds: int = 60
    config: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.position is None:
            self.position = {'x': 0, 'y': 0, 'width': 4, 'height': 3}

@dataclass
class Dashboard:
    """Dashboard definition"""
    dashboard_id: str
    name: str
    description: str
    dashboard_type: DashboardType
    widgets: List[DashboardWidget]
    is_public: bool = False
    owner: str = "system"
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    tags: List[str] = field(default_factory=list)

@dataclass
class DashboardData:
    """Dashboard data point"""
    timestamp: datetime
    dashboard_id: str
    widget_id: str
    metric_id: str
    value: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DashboardSnapshot:
    """Dashboard snapshot for reporting"""
    snapshot_id: str
    dashboard_id: str
    name: str
    data: Dict[str, Any]
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: datetime = None

class PerformanceDashboardManager:
    """
    Comprehensive performance monitoring dashboard system
    """
    
    def __init__(self, storage_dir: str = "dashboard_storage"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        
        # Data storage
        self.dashboards = {}
        self.metric_definitions = {}
        self.dashboard_data = defaultdict(list)
        self.dashboard_snapshots = {}
        
        # Default metrics
        self.default_metrics = self._create_default_metrics()
        
        # Default dashboards
        self.default_dashboards = self._create_default_dashboards()
        
        # Configuration
        self.data_retention_days = 30
        self.snapshot_retention_days = 7
        
        # Threading
        self._lock = threading.Lock()
        self._running = False
        self._data_collection_thread = None
        self._cleanup_thread = None
        
        # Statistics
        self.stats = {
            'dashboards_created': 0,
            'data_points_collected': 0,
            'snapshots_created': 0,
            'api_requests_served': 0
        }
        
    def start_dashboard_manager(self):
        """Start the dashboard manager"""
        if self._running:
            logger.warning("Dashboard manager already running")
            return
            
        self._running = True
        self._data_collection_thread = threading.Thread(target=self._data_collection_loop, daemon=True)
        self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        
        self._data_collection_thread.start()
        self._cleanup_thread.start()
        
        logger.info("Performance dashboard manager started")
        
    def stop_dashboard_manager(self):
        """Stop the dashboard manager"""
        self._running = False
        if self._data_collection_thread:
            self._data_collection_thread.join(timeout=5)
        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=5)
        logger.info("Performance dashboard manager stopped")
        
    def create_dashboard(self, dashboard: Dashboard) -> str:
        """
        Create a new dashboard
        
        Args:
            dashboard: Dashboard object
            
        Returns:
            Dashboard ID
        """
        with self._lock:
            # Validate dashboard
            self._validate_dashboard(dashboard)
            
            # Store dashboard
            self.dashboards[dashboard.dashboard_id] = dashboard
            self.stats['dashboards_created'] += 1
            
            logger.info(f"Created dashboard: {dashboard.dashboard_id}")
            
            return dashboard.dashboard_id
            
    def update_dashboard(self, dashboard_id: str, dashboard: Dashboard):
        """
        Update an existing dashboard
        
        Args:
            dashboard_id: Dashboard identifier
            dashboard: Updated dashboard object
        """
        with self._lock:
            if dashboard_id not in self.dashboards:
                raise ValueError(f"Dashboard {dashboard_id} not found")
                
            # Validate dashboard
            self._validate_dashboard(dashboard)
            
            # Update dashboard
            self.dashboards[dashboard_id] = dashboard
            dashboard.updated_at = datetime.now()
            
            logger.info(f"Updated dashboard: {dashboard_id}")
            
    def delete_dashboard(self, dashboard_id: str):
        """
        Delete a dashboard
        
        Args:
            dashboard_id: Dashboard identifier
        """
        with self._lock:
            if dashboard_id not in self.dashboards:
                raise ValueError(f"Dashboard {dashboard_id} not found")
                
            # Delete dashboard
            del self.dashboards[dashboard_id]
            
            # Clean up associated data
            if dashboard_id in self.dashboard_data:
                del self.dashboard_data[dashboard_id]
                
            logger.info(f"Deleted dashboard: {dashboard_id}")
            
    def get_dashboard(self, dashboard_id: str) -> Optional[Dashboard]:
        """
        Get dashboard by ID
        
        Args:
            dashboard_id: Dashboard identifier
            
        Returns:
            Dashboard object or None
        """
        with self._lock:
            return self.dashboards.get(dashboard_id)
            
    def list_dashboards(self, dashboard_type: DashboardType = None, 
                      owner: str = None) -> List[Dashboard]:
        """
        List dashboards with optional filtering
        
        Args:
            dashboard_type: Optional dashboard type filter
            owner: Optional owner filter
            
        Returns:
            List of Dashboard objects
        """
        with self._lock:
            dashboards = list(self.dashboards.values())
            
            if dashboard_type:
                dashboards = [d for d in dashboards if d.dashboard_type == dashboard_type]
                
            if owner:
                dashboards = [d for d in dashboards if d.owner == owner]
                
            return sorted(dashboards, key=lambda x: x.updated_at, reverse=True)
            
    def add_metric_definition(self, metric: MetricDefinition):
        """
        Add a metric definition
        
        Args:
            metric: MetricDefinition object
        """
        with self._lock:
            self.metric_definitions[metric.metric_id] = metric
            logger.info(f"Added metric definition: {metric.metric_id}")
            
    def get_metric_definition(self, metric_id: str) -> Optional[MetricDefinition]:
        """
        Get metric definition by ID
        
        Args:
            metric_id: Metric identifier
            
        Returns:
            MetricDefinition object or None
        """
        with self._lock:
            return self.metric_definitions.get(metric_id)
            
    def record_metric(self, dashboard_id: str, widget_id: str, metric_id: str, 
                    value: float, metadata: Dict[str, Any] = None):
        """
        Record a metric data point
        
        Args:
            dashboard_id: Dashboard identifier
            widget_id: Widget identifier
            metric_id: Metric identifier
            value: Metric value
            metadata: Additional metadata
        """
        data_point = DashboardData(
            timestamp=datetime.now(),
            dashboard_id=dashboard_id,
            widget_id=widget_id,
            metric_id=metric_id,
            value=value,
            metadata=metadata or {}
        )
        
        with self._lock:
            self.dashboard_data[f"{dashboard_id}:{widget_id}:{metric_id}"].append(data_point)
            self.stats['data_points_collected'] += 1
            
    def get_widget_data(self, dashboard_id: str, widget_id: str, 
                      time_range: TimeRange = TimeRange.LAST_24_HOURS) -> Dict[str, List[DashboardData]]:
        """
        Get data for a specific widget
        
        Args:
            dashboard_id: Dashboard identifier
            widget_id: Widget identifier
            time_range: Time range for data
            
        Returns:
            Dictionary of metric data lists
        """
        dashboard = self.get_dashboard(dashboard_id)
        if not dashboard:
            return {}
            
        widget = next((w for w in dashboard.widgets if w.widget_id == widget_id), None)
        if not widget:
            return {}
            
        # Calculate time range
        cutoff_time = self._get_cutoff_time(time_range)
        
        # Get data for each metric
        metric_data = {}
        
        for metric_id in widget.metrics:
            key = f"{dashboard_id}:{widget_id}:{metric_id}"
            
            with self._lock:
                if key in self.dashboard_data:
                    data = [
                        dp for dp in self.dashboard_data[key]
                        if dp.timestamp >= cutoff_time
                    ]
                    metric_data[metric_id] = data
                    
        return metric_data
        
    def generate_dashboard_data(self, dashboard_id: str) -> Dict[str, Any]:
        """
        Generate complete dashboard data for rendering
        
        Args:
            dashboard_id: Dashboard identifier
            
        Returns:
            Dictionary with dashboard data
        """
        dashboard = self.get_dashboard(dashboard_id)
        if not dashboard:
            return {}
            
        dashboard_data = {
            'dashboard_id': dashboard_id,
            'name': dashboard.name,
            'description': dashboard.description,
            'type': dashboard.dashboard_type.value,
            'widgets': [],
            'generated_at': datetime.now().isoformat()
        }
        
        for widget in dashboard.widgets:
            widget_data = {
                'widget_id': widget.widget_id,
                'title': widget.title,
                'type': widget.widget_type,
                'position': widget.position,
                'data': self._process_widget_data(dashboard_id, widget)
            }
            
            dashboard_data['widgets'].append(widget_data)
            
        with self._lock:
            self.stats['api_requests_served'] += 1
            
        return dashboard_data
        
    def create_snapshot(self, dashboard_id: str, name: str = None) -> str:
        """
        Create a dashboard snapshot
        
        Args:
            dashboard_id: Dashboard identifier
            name: Optional snapshot name
            
        Returns:
            Snapshot ID
        """
        dashboard_data = self.generate_dashboard_data(dashboard_id)
        
        snapshot_id = f"snapshot_{int(time.time())}_{hash(dashboard_id) % 1000}"
        
        snapshot = DashboardSnapshot(
            snapshot_id=snapshot_id,
            dashboard_id=dashboard_id,
            name=name or f"Snapshot for {dashboard_id}",
            data=dashboard_data,
            expires_at=datetime.now() + timedelta(days=self.snapshot_retention_days)
        )
        
        with self._lock:
            self.dashboard_snapshots[snapshot_id] = snapshot
            self.stats['snapshots_created'] += 1
            
        logger.info(f"Created dashboard snapshot: {snapshot_id}")
        
        return snapshot_id
        
    def get_snapshot(self, snapshot_id: str) -> Optional[DashboardSnapshot]:
        """
        Get dashboard snapshot by ID
        
        Args:
            snapshot_id: Snapshot identifier
            
        Returns:
            DashboardSnapshot object or None
        """
        with self._lock:
            snapshot = self.dashboard_snapshots.get(snapshot_id)
            
            # Check if snapshot is expired
            if snapshot and snapshot.expires_at and snapshot.expires_at < datetime.now():
                del self.dashboard_snapshots[snapshot_id]
                return None
                
            return snapshot
            
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get dashboard manager statistics
        
        Returns:
            Dictionary with statistics
        """
        with self._lock:
            return {
                'timestamp': datetime.now().isoformat(),
                'stats': self.stats.copy(),
                'dashboards_count': len(self.dashboards),
                'metric_definitions_count': len(self.metric_definitions),
                'active_snapshots_count': len([
                    s for s in self.dashboard_snapshots.values()
                    if not s.expires_at or s.expires_at > datetime.now()
                ])
            }
            
    def export_dashboard_data(self, dashboard_id: str, format: str = 'json') -> str:
        """
        Export dashboard data
        
        Args:
            dashboard_id: Dashboard identifier
            format: Export format ('json', 'csv')
            
        Returns:
            Exported data string
        """
        dashboard_data = self.generate_dashboard_data(dashboard_id)
        
        if format.lower() == 'json':
            return json.dumps(dashboard_data, indent=2, default=str)
        elif format.lower() == 'csv':
            # Convert to CSV format
            csv_data = []
            csv_data.append("timestamp,widget_id,metric_id,value")
            
            for widget in dashboard_data.get('widgets', []):
                for metric_id, data_points in widget.get('data', {}).items():
                    for data_point in data_points:
                        csv_data.append(f"{data_point.timestamp},{widget['widget_id']},{metric_id},{data_point.value}")
                        
            return "\n".join(csv_data)
        else:
            raise ValueError(f"Unsupported export format: {format}")
            
    def _validate_dashboard(self, dashboard: Dashboard):
        """Validate dashboard definition"""
        if not dashboard.widgets:
            raise ValueError("Dashboard must have at least one widget")
            
        # Check for duplicate widget IDs
        widget_ids = [widget.widget_id for widget in dashboard.widgets]
        if len(widget_ids) != len(set(widget_ids)):
            raise ValueError("Duplicate widget IDs found in dashboard")
            
        # Validate each widget
        for widget in dashboard.widgets:
            if not widget.metrics:
                raise ValueError(f"Widget {widget.widget_id} must have at least one metric")
                
    def _process_widget_data(self, dashboard_id: str, widget: DashboardWidget) -> Dict[str, Any]:
        """Process widget data for rendering"""
        metric_data = self.get_widget_data(dashboard_id, widget.widget_id, widget.time_range)
        
        processed_data = {}
        
        for metric_id, data_points in metric_data.items():
            if not data_points:
                continue
                
            # Get metric definition
            metric_def = self.get_metric_definition(metric_id)
            
            # Aggregate data based on metric definition
            if metric_def and metric_def.aggregation:
                aggregated_value = self._aggregate_metric(data_points, metric_def.aggregation)
            else:
                aggregated_value = data_points[-1].value if data_points else 0
                
            # Prepare chart data
            chart_data = {
                'timestamps': [dp.timestamp.isoformat() for dp in data_points],
                'values': [dp.value for dp in data_points],
                'current_value': aggregated_value,
                'metric_definition': asdict(metric_def) if metric_def else None
            }
            
            # Add threshold information
            if metric_def and metric_def.threshold_value is not None:
                chart_data['threshold'] = {
                    'value': metric_def.threshold_value,
                    'operator': metric_def.threshold_operator
                }
                
            processed_data[metric_id] = chart_data
            
        return processed_data
        
    def _aggregate_metric(self, data_points: List[DashboardData], aggregation: str) -> float:
        """Aggregate metric data points"""
        values = [dp.value for dp in data_points]
        
        if not values:
            return 0
            
        if aggregation == 'avg':
            return statistics.mean(values)
        elif aggregation == 'sum':
            return sum(values)
        elif aggregation == 'count':
            return len(values)
        elif aggregation == 'max':
            return max(values)
        elif aggregation == 'min':
            return min(values)
        else:
            return statistics.mean(values)
            
    def _get_cutoff_time(self, time_range: TimeRange) -> datetime:
        """Get cutoff time for data retrieval"""
        now = datetime.now()
        
        if time_range == TimeRange.LAST_HOUR:
            return now - timedelta(hours=1)
        elif time_range == TimeRange.LAST_24_HOURS:
            return now - timedelta(hours=24)
        elif time_range == TimeRange.LAST_7_DAYS:
            return now - timedelta(days=7)
        elif time_range == TimeRange.LAST_30_DAYS:
            return now - timedelta(days=30)
        else:
            return now - timedelta(hours=24)  # Default to 24 hours
            
    def _create_default_metrics(self) -> Dict[str, MetricDefinition]:
        """Create default metric definitions"""
        metrics = {}
        
        # System metrics
        metrics['cpu_usage'] = MetricDefinition(
            metric_id="cpu_usage",
            name="CPU Usage",
            description="CPU utilization percentage",
            unit="%",
            aggregation="avg",
            chart_type=ChartType.LINE,
            color="#ef4444",
            threshold_value=80.0,
            threshold_operator="gt"
        )
        
        metrics['memory_usage'] = MetricDefinition(
            metric_id="memory_usage",
            name="Memory Usage",
            description="Memory utilization percentage",
            unit="%",
            aggregation="avg",
            chart_type=ChartType.LINE,
            color="#f59e0b",
            threshold_value=85.0,
            threshold_operator="gt"
        )
        
        metrics['disk_usage'] = MetricDefinition(
            metric_id="disk_usage",
            name="Disk Usage",
            description="Disk utilization percentage",
            unit="%",
            aggregation="avg",
            chart_type=ChartType.LINE,
            color="#10b981",
            threshold_value=90.0,
            threshold_operator="gt"
        )
        
        # Model performance metrics
        metrics['response_time'] = MetricDefinition(
            metric_id="response_time",
            name="Response Time",
            description="Average response time",
            unit="ms",
            aggregation="avg",
            chart_type=ChartType.LINE,
            color="#3b82f6",
            threshold_value=500.0,
            threshold_operator="gt"
        )
        
        metrics['error_rate'] = MetricDefinition(
            metric_id="error_rate",
            name="Error Rate",
            description="Request error rate",
            unit="%",
            aggregation="avg",
            chart_type=ChartType.LINE,
            color="#ef4444",
            threshold_value=5.0,
            threshold_operator="gt"
        )
        
        metrics['throughput'] = MetricDefinition(
            metric_id="throughput",
            name="Throughput",
            description="Requests per second",
            unit="rps",
            aggregation="avg",
            chart_type=ChartType.LINE,
            color="#8b5cf6"
        )
        
        # Business metrics
        metrics['conversion_rate'] = MetricDefinition(
            metric_id="conversion_rate",
            name="Conversion Rate",
            description="User conversion rate",
            unit="%",
            aggregation="avg",
            chart_type=ChartType.LINE,
            color="#10b981"
        )
        
        metrics['revenue'] = MetricDefinition(
            metric_id="revenue",
            name="Revenue",
            description="Total revenue",
            unit="$",
            aggregation="sum",
            chart_type=ChartType.LINE,
            color="#059669"
        )
        
        return metrics
        
    def _create_default_dashboards(self) -> Dict[str, Dashboard]:
        """Create default dashboards"""
        dashboards = {}
        
        # System Overview Dashboard
        system_widgets = [
            DashboardWidget(
                widget_id="cpu_widget",
                title="CPU Usage",
                widget_type="chart",
                metrics=["cpu_usage"],
                chart_type=ChartType.LINE,
                position={'x': 0, 'y': 0, 'width': 6, 'height': 4}
            ),
            DashboardWidget(
                widget_id="memory_widget",
                title="Memory Usage",
                widget_type="chart",
                metrics=["memory_usage"],
                chart_type=ChartType.LINE,
                position={'x': 6, 'y': 0, 'width': 6, 'height': 4}
            ),
            DashboardWidget(
                widget_id="disk_widget",
                title="Disk Usage",
                widget_type="chart",
                metrics=["disk_usage"],
                chart_type=ChartType.LINE,
                position={'x': 0, 'y': 4, 'width': 6, 'height': 4}
            ),
            DashboardWidget(
                widget_id="system_summary",
                title="System Summary",
                widget_type="metric",
                metrics=["cpu_usage", "memory_usage", "disk_usage"],
                position={'x': 6, 'y': 4, 'width': 6, 'height': 4}
            )
        ]
        
        system_dashboard = Dashboard(
            dashboard_id="system_overview",
            name="System Overview",
            description="System resource utilization and health",
            dashboard_type=DashboardType.SYSTEM_OVERVIEW,
            widgets=system_widgets,
            is_public=True
        )
        
        # Model Performance Dashboard
        model_widgets = [
            DashboardWidget(
                widget_id="response_time_widget",
                title="Response Time",
                widget_type="chart",
                metrics=["response_time"],
                chart_type=ChartType.LINE,
                position={'x': 0, 'y': 0, 'width': 8, 'height': 4}
            ),
            DashboardWidget(
                widget_id="error_rate_widget",
                title="Error Rate",
                widget_type="chart",
                metrics=["error_rate"],
                chart_type=ChartType.LINE,
                position={'x': 8, 'y': 0, 'width': 4, 'height': 4}
            ),
            DashboardWidget(
                widget_id="throughput_widget",
                title="Throughput",
                widget_type="chart",
                metrics=["throughput"],
                chart_type=ChartType.LINE,
                position={'x': 0, 'y': 4, 'width': 12, 'height': 4}
            )
        ]
        
        model_dashboard = Dashboard(
            dashboard_id="model_performance",
            name="Model Performance",
            description="AI model performance metrics",
            dashboard_type=DashboardType.MODEL_PERFORMANCE,
            widgets=model_widgets,
            is_public=True
        )
        
        dashboards[system_dashboard.dashboard_id] = system_dashboard
        dashboards[model_dashboard.dashboard_id] = model_dashboard
        
        # Add metric definitions
        for metric in self.default_metrics.values():
            self.add_metric_definition(metric)
            
        return dashboards
        
    def _collect_system_metrics(self):
        """Collect system metrics for default dashboards"""
        try:
            import psutil
            
            # Get system metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Record metrics for system overview dashboard
            self.record_metric("system_overview", "cpu_widget", "cpu_usage", cpu_percent)
            self.record_metric("system_overview", "memory_widget", "memory_usage", memory.percent)
            self.record_metric("system_overview", "disk_widget", "disk_usage", disk.percent)
            
        except ImportError:
            # psutil not available, use dummy data
            import random
            self.record_metric("system_overview", "cpu_widget", "cpu_usage", random.uniform(20, 80))
            self.record_metric("system_overview", "memory_widget", "memory_usage", random.uniform(30, 70))
            self.record_metric("system_overview", "disk_widget", "disk_usage", random.uniform(40, 60))
            
    def _collect_model_metrics(self):
        """Collect model metrics for default dashboards"""
        # Simulate model metrics
        import random
        
        response_time = random.uniform(50, 300)
        error_rate = random.uniform(0.1, 2.0)
        throughput = random.uniform(10, 100)
        
        # Record metrics for model performance dashboard
        self.record_metric("model_performance", "response_time_widget", "response_time", response_time)
        self.record_metric("model_performance", "error_rate_widget", "error_rate", error_rate)
        self.record_metric("model_performance", "throughput_widget", "throughput", throughput)
        
    def _data_collection_loop(self):
        """Main data collection loop"""
        while self._running:
            try:
                # Collect system metrics
                self._collect_system_metrics()
                
                # Collect model metrics
                self._collect_model_metrics()
                
                time.sleep(30)  # Collect every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in data collection loop: {e}")
                time.sleep(10)
                
    def _cleanup_loop(self):
        """Main cleanup loop"""
        while self._running:
            try:
                # Clean up old data
                self._cleanup_old_data()
                
                # Clean up expired snapshots
                self._cleanup_expired_snapshots()
                
                # Log statistics
                if self.stats['data_points_collected'] > 0 and self.stats['data_points_collected'] % 100 == 0:
                    logger.info(f"Dashboard manager stats: {self.stats}")
                    
                time.sleep(3600)  # Run every hour
                
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                time.sleep(300)
                
    def _cleanup_old_data(self):
        """Clean up old data based on retention policy"""
        cutoff_time = datetime.now() - timedelta(days=self.data_retention_days)
        
        with self._lock:
            # Clean up old dashboard data
            for key in list(self.dashboard_data.keys()):
                self.dashboard_data[key] = [
                    dp for dp in self.dashboard_data[key]
                    if dp.timestamp >= cutoff_time
                ]
                
    def _cleanup_expired_snapshots(self):
        """Clean up expired snapshots"""
        current_time = datetime.now()
        
        with self._lock:
            expired_snapshots = [
                snapshot_id for snapshot_id, snapshot in self.dashboard_snapshots.items()
                if snapshot.expires_at and snapshot.expires_at < current_time
            ]
            
            for snapshot_id in expired_snapshots:
                del self.dashboard_snapshots[snapshot_id]

# Example usage and testing
def main():
    """Example usage of the performance dashboard manager"""
    dashboard_manager = PerformanceDashboardManager()
    
    try:
        dashboard_manager.start_dashboard_manager()
        
        # List default dashboards
        dashboards = dashboard_manager.list_dashboards()
        print(f"Available dashboards: {len(dashboards)}")
        
        for dashboard in dashboards:
            print(f"  - {dashboard.name} ({dashboard.dashboard_id})")
            
        # Generate dashboard data
        if dashboards:
            dashboard_data = dashboard_manager.generate_dashboard_data(dashboards[0].dashboard_id)
            print(f"Generated dashboard data with {len(dashboard_data.get('widgets', []))} widgets")
            
        # Create a custom dashboard
        custom_widgets = [
            DashboardWidget(
                widget_id="custom_metric1",
                title="Custom Metric 1",
                widget_type="chart",
                metrics=["cpu_usage"],
                chart_type=ChartType.LINE,
                position={'x': 0, 'y': 0, 'width': 12, 'height': 6}
            )
        ]
        
        custom_dashboard = Dashboard(
            dashboard_id="custom_dashboard",
            name="Custom Dashboard",
            description="Custom dashboard example",
            dashboard_type=DashboardType.CUSTOM,
            widgets=custom_widgets
        )
        
        dashboard_id = dashboard_manager.create_dashboard(custom_dashboard)
        print(f"Created custom dashboard: {dashboard_id}")
        
        # Create a snapshot
        if dashboards:
            snapshot_id = dashboard_manager.create_snapshot(dashboards[0].dashboard_id)
            print(f"Created snapshot: {snapshot_id}")
            
        # Get statistics
        stats = dashboard_manager.get_statistics()
        print(f"Dashboard manager statistics: {stats}")
        
        # Export dashboard data
        if dashboards:
            json_data = dashboard_manager.export_dashboard_data(dashboards[0].dashboard_id, 'json')
            print(f"Exported JSON data (first 200 chars): {json_data[:200]}...")
            
        time.sleep(10)  # Let data collection run
        
    finally:
        dashboard_manager.stop_dashboard_manager()

if __name__ == "__main__":
    main()