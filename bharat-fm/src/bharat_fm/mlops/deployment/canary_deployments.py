"""
Canary Deployment System for Bharat-FM MLOps Platform

This module provides comprehensive canary deployment capabilities for AI models,
allowing gradual rollout with automated monitoring and rollback capabilities.
It supports traffic shifting, health checks, and performance validation.

Features:
- Gradual traffic shifting strategies
- Automated health checks and monitoring
- Performance validation and comparison
- Automated rollback on failure
- Multi-stage deployment pipelines
- Real-time metrics and alerting
"""

import time
import threading
import json
import logging
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque
import statistics
import requests
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DeploymentStatus(Enum):
    """Deployment status enumeration"""
    PENDING = "pending"
    DEPLOYING = "deploying"
    MONITORING = "monitoring"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"

class TrafficShiftStrategy(Enum):
    """Traffic shift strategies"""
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    MANUAL = "manual"
    IMMEDIATE = "immediate"

@dataclass
class CanaryDeployment:
    """Canary deployment configuration"""
    deployment_id: str
    model_id: str
    version: str
    baseline_version: str
    traffic_percentage: float
    max_traffic_percentage: float
    shift_strategy: TrafficShiftStrategy
    shift_duration_minutes: int
    health_check_interval_seconds: int
    success_criteria: Dict[str, Any]
    rollback_criteria: Dict[str, Any]
    status: DeploymentStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    current_metrics: Dict[str, float] = None
    baseline_metrics: Dict[str, float] = None
    
    def __post_init__(self):
        if self.current_metrics is None:
            self.current_metrics = {}
        if self.baseline_metrics is None:
            self.baseline_metrics = {}

@dataclass
class HealthCheck:
    """Health check configuration"""
    check_id: str
    name: str
    endpoint: str
    method: str = "GET"
    expected_status: int = 200
    timeout_seconds: int = 30
    headers: Dict[str, str] = None
    body: str = None
    
    def __post_init__(self):
        if self.headers is None:
            self.headers = {}

@dataclass
class DeploymentMetric:
    """Deployment metric data"""
    timestamp: datetime
    deployment_id: str
    metric_name: str
    value: float
    version: str  # 'baseline' or 'canary'
    tags: Dict[str, str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = {}

@dataclass
class DeploymentAlert:
    """Deployment alert data"""
    alert_id: str
    deployment_id: str
    alert_type: str
    severity: str
    message: str
    timestamp: datetime
    resolved: bool = False
    action_taken: str = None

class CanaryDeploymentManager:
    """
    Comprehensive canary deployment management system
    """
    
    def __init__(self, api_base_url: str = "http://localhost:8000"):
        self.api_base_url = api_base_url
        
        # Data storage
        self.deployments = {}
        self.health_checks = {}
        self.deployment_metrics = defaultdict(list)
        self.deployment_alerts = {}
        self.deployment_history = deque(maxlen=1000)
        
        # Configuration
        self.default_success_criteria = {
            'error_rate_threshold': 0.01,  # 1%
            'response_time_threshold': 100,  # ms
            'availability_threshold': 0.99,  # 99%
            'min_sample_size': 100
        }
        
        self.default_rollback_criteria = {
            'error_rate_threshold': 0.05,  # 5%
            'response_time_threshold': 500,  # ms
            'availability_threshold': 0.95,  # 95%
            'max_failures': 10
        }
        
        # Monitoring
        self.active_deployments = set()
        
        # Threading
        self._lock = threading.Lock()
        self._running = False
        self._monitoring_thread = None
        self._traffic_shift_thread = None
        
        # Statistics
        self.stats = {
            'deployments_created': 0,
            'deployments_completed': 0,
            'deployments_failed': 0,
            'rollbacks_triggered': 0,
            'health_checks_performed': 0
        }
        
    def start_deployment_manager(self):
        """Start the canary deployment manager"""
        if self._running:
            logger.warning("Canary deployment manager already running")
            return
            
        self._running = True
        self._monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._traffic_shift_thread = threading.Thread(target=self._traffic_shift_loop, daemon=True)
        
        self._monitoring_thread.start()
        self._traffic_shift_thread.start()
        
        logger.info("Canary deployment manager started")
        
    def stop_deployment_manager(self):
        """Stop the canary deployment manager"""
        self._running = False
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5)
        if self._traffic_shift_thread:
            self._traffic_shift_thread.join(timeout=5)
        logger.info("Canary deployment manager stopped")
        
    def create_deployment(self, model_id: str, version: str, baseline_version: str,
                          traffic_percentage: float = 10.0,
                          max_traffic_percentage: float = 100.0,
                          shift_strategy: TrafficShiftStrategy = TrafficShiftStrategy.LINEAR,
                          shift_duration_minutes: int = 60,
                          success_criteria: Dict[str, Any] = None,
                          rollback_criteria: Dict[str, Any] = None) -> CanaryDeployment:
        """
        Create a new canary deployment
        
        Args:
            model_id: Model identifier
            version: New version to deploy
            baseline_version: Current baseline version
            traffic_percentage: Initial traffic percentage
            max_traffic_percentage: Maximum traffic percentage
            shift_strategy: Traffic shift strategy
            shift_duration_minutes: Duration for traffic shift
            success_criteria: Success criteria for deployment
            rollback_criteria: Rollback criteria
            
        Returns:
            CanaryDeployment object
        """
        deployment_id = f"canary_{int(time.time())}_{hash(model_id) % 1000}"
        
        deployment = CanaryDeployment(
            deployment_id=deployment_id,
            model_id=model_id,
            version=version,
            baseline_version=baseline_version,
            traffic_percentage=traffic_percentage,
            max_traffic_percentage=max_traffic_percentage,
            shift_strategy=shift_strategy,
            shift_duration_minutes=shift_duration_minutes,
            health_check_interval_seconds=30,
            success_criteria=success_criteria or self.default_success_criteria,
            rollback_criteria=rollback_criteria or self.default_rollback_criteria,
            status=DeploymentStatus.PENDING,
            created_at=datetime.now()
        )
        
        with self._lock:
            self.deployments[deployment_id] = deployment
            self.stats['deployments_created'] += 1
            
        logger.info(f"Created canary deployment: {deployment_id}")
        
        return deployment
        
    def start_deployment(self, deployment_id: str):
        """
        Start a canary deployment
        
        Args:
            deployment_id: Deployment identifier
        """
        with self._lock:
            if deployment_id not in self.deployments:
                raise ValueError(f"Deployment {deployment_id} not found")
                
            deployment = self.deployments[deployment_id]
            
            if deployment.status != DeploymentStatus.PENDING:
                raise ValueError(f"Deployment {deployment_id} is not in PENDING status")
                
            deployment.status = DeploymentStatus.DEPLOYING
            deployment.started_at = datetime.now()
            self.active_deployments.add(deployment_id)
            
        logger.info(f"Started canary deployment: {deployment_id}")
        
    def stop_deployment(self, deployment_id: str, success: bool = True):
        """
        Stop a canary deployment
        
        Args:
            deployment_id: Deployment identifier
            success: Whether deployment was successful
        """
        with self._lock:
            if deployment_id not in self.deployments:
                raise ValueError(f"Deployment {deployment_id} not found")
                
            deployment = self.deployments[deployment_id]
            
            if deployment.status not in [DeploymentStatus.DEPLOYING, DeploymentStatus.MONITORING]:
                raise ValueError(f"Deployment {deployment_id} is not active")
                
            deployment.status = DeploymentStatus.COMPLETED if success else DeploymentStatus.FAILED
            deployment.completed_at = datetime.now()
            
            if deployment_id in self.active_deployments:
                self.active_deployments.remove(deployment_id)
                
            if success:
                self.stats['deployments_completed'] += 1
            else:
                self.stats['deployments_failed'] += 1
                
        logger.info(f"Stopped canary deployment: {deployment_id} (success: {success})")
        
    def rollback_deployment(self, deployment_id: str, reason: str = "Automatic rollback"):
        """
        Rollback a canary deployment
        
        Args:
            deployment_id: Deployment identifier
            reason: Reason for rollback
        """
        with self._lock:
            if deployment_id not in self.deployments:
                raise ValueError(f"Deployment {deployment_id} not found")
                
            deployment = self.deployments[deployment_id]
            
            if deployment.status not in [DeploymentStatus.DEPLOYING, DeploymentStatus.MONITORING]:
                raise ValueError(f"Deployment {deployment_id} is not active")
                
            deployment.status = DeploymentStatus.ROLLED_BACK
            deployment.completed_at = datetime.now()
            
            if deployment_id in self.active_deployments:
                self.active_deployments.remove(deployment_id)
                
            self.stats['rollbacks_triggered'] += 1
            
            # Create rollback alert
            alert = DeploymentAlert(
                alert_id=f"rollback_{int(time.time())}",
                deployment_id=deployment_id,
                alert_type="rollback",
                severity="high",
                message=f"Deployment rolled back: {reason}",
                timestamp=datetime.now(),
                action_taken="rollback"
            )
            
            self.deployment_alerts[alert.alert_id] = alert
            self.deployment_history.append(alert)
            
        logger.warning(f"Rolled back deployment {deployment_id}: {reason}")
        
    def add_health_check(self, deployment_id: str, health_check: HealthCheck):
        """
        Add a health check to a deployment
        
        Args:
            deployment_id: Deployment identifier
            health_check: HealthCheck object
        """
        with self._lock:
            if deployment_id not in self.deployments:
                raise ValueError(f"Deployment {deployment_id} not found")
                
            if deployment_id not in self.health_checks:
                self.health_checks[deployment_id] = []
                
            self.health_checks[deployment_id].append(health_check)
            
        logger.info(f"Added health check to deployment {deployment_id}")
        
    def record_metric(self, deployment_id: str, metric_name: str, value: float,
                      version: str, tags: Dict[str, str] = None):
        """
        Record a deployment metric
        
        Args:
            deployment_id: Deployment identifier
            metric_name: Name of the metric
            value: Metric value
            version: Version ('baseline' or 'canary')
            tags: Additional tags
        """
        metric = DeploymentMetric(
            timestamp=datetime.now(),
            deployment_id=deployment_id,
            metric_name=metric_name,
            value=value,
            version=version,
            tags=tags or {}
        )
        
        with self._lock:
            self.deployment_metrics[f"{deployment_id}:{metric_name}"].append(metric)
            
    def get_deployment(self, deployment_id: str) -> Optional[CanaryDeployment]:
        """
        Get deployment by ID
        
        Args:
            deployment_id: Deployment identifier
            
        Returns:
            CanaryDeployment object or None
        """
        with self._lock:
            return self.deployments.get(deployment_id)
            
    def get_active_deployments(self) -> List[CanaryDeployment]:
        """
        Get all active deployments
        
        Returns:
            List of active CanaryDeployment objects
        """
        with self._lock:
            return [self.deployments[dep_id] for dep_id in self.active_deployments]
            
    def get_deployment_metrics(self, deployment_id: str, metric_name: str = None,
                              minutes: int = 60) -> List[DeploymentMetric]:
        """
        Get deployment metrics
        
        Args:
            deployment_id: Deployment identifier
            metric_name: Optional metric name filter
            minutes: Number of minutes to look back
            
        Returns:
            List of DeploymentMetric objects
        """
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        
        with self._lock:
            metrics = []
            
            if metric_name:
                key = f"{deployment_id}:{metric_name}"
                if key in self.deployment_metrics:
                    metrics.extend([
                        m for m in self.deployment_metrics[key]
                        if m.timestamp >= cutoff_time
                    ])
            else:
                for key, metric_list in self.deployment_metrics.items():
                    if key.startswith(f"{deployment_id}:"):
                        metrics.extend([
                            m for m in metric_list
                            if m.timestamp >= cutoff_time
                        ])
                        
            return sorted(metrics, key=lambda x: x.timestamp)
            
    def get_deployment_alerts(self, deployment_id: str = None) -> List[DeploymentAlert]:
        """
        Get deployment alerts
        
        Args:
            deployment_id: Optional deployment filter
            
        Returns:
            List of DeploymentAlert objects
        """
        with self._lock:
            alerts = list(self.deployment_alerts.values())
            
            if deployment_id:
                alerts = [alert for alert in alerts if alert.deployment_id == deployment_id]
                
            return alerts
            
    def get_deployment_statistics(self, deployment_id: str) -> Dict[str, Any]:
        """
        Get deployment statistics
        
        Args:
            deployment_id: Deployment identifier
            
        Returns:
            Dictionary with deployment statistics
        """
        deployment = self.get_deployment(deployment_id)
        if not deployment:
            return {}
            
        # Get recent metrics
        recent_metrics = self.get_deployment_metrics(deployment_id, minutes=30)
        
        # Calculate statistics by version
        baseline_metrics = [m for m in recent_metrics if m.version == 'baseline']
        canary_metrics = [m for m in recent_metrics if m.version == 'canary']
        
        stats = {
            'deployment_id': deployment_id,
            'status': deployment.status.value,
            'traffic_percentage': deployment.traffic_percentage,
            'duration_minutes': (datetime.now() - deployment.started_at).total_seconds() / 60 if deployment.started_at else 0,
            'baseline_metrics_count': len(baseline_metrics),
            'canary_metrics_count': len(canary_metrics)
        }
        
        # Calculate metric comparisons
        metric_names = set(m.metric_name for m in recent_metrics)
        
        for metric_name in metric_names:
            baseline_values = [m.value for m in baseline_metrics if m.metric_name == metric_name]
            canary_values = [m.value for m in canary_metrics if m.metric_name == metric_name]
            
            if baseline_values and canary_values:
                baseline_mean = statistics.mean(baseline_values)
                canary_mean = statistics.mean(canary_values)
                
                stats[f"{metric_name}_baseline_mean"] = baseline_mean
                stats[f"{metric_name}_canary_mean"] = canary_mean
                stats[f"{metric_name}_difference"] = canary_mean - baseline_mean
                stats[f"{metric_name}_difference_percent"] = ((canary_mean - baseline_mean) / baseline_mean) * 100 if baseline_mean != 0 else 0
                
        return stats
        
    def update_traffic_percentage(self, deployment_id: str, new_percentage: float):
        """
        Update traffic percentage for a deployment
        
        Args:
            deployment_id: Deployment identifier
            new_percentage: New traffic percentage
        """
        with self._lock:
            if deployment_id not in self.deployments:
                raise ValueError(f"Deployment {deployment_id} not found")
                
            deployment = self.deployments[deployment_id]
            
            if deployment.status not in [DeploymentStatus.DEPLOYING, DeploymentStatus.MONITORING]:
                raise ValueError(f"Cannot update traffic for deployment {deployment_id} in status {deployment.status}")
                
            deployment.traffic_percentage = min(new_percentage, deployment.max_traffic_percentage)
            
        logger.info(f"Updated traffic percentage for {deployment_id} to {deployment.traffic_percentage}%")
        
    def export_deployment_data(self, filename: str = None) -> str:
        """
        Export deployment data to JSON file
        
        Args:
            filename: Optional filename to save to
            
        Returns:
            JSON string of deployment data
        """
        with self._lock:
            export_data = {
                'timestamp': datetime.now().isoformat(),
                'deployments': {k: asdict(v) for k, v in self.deployments.items()},
                'active_deployments': list(self.active_deployments),
                'health_checks': {k: [asdict(h) for h in v] for k, v in self.health_checks.items()},
                'statistics': self.stats.copy(),
                'deployment_alerts': [asdict(alert) for alert in self.get_deployment_alerts()]
            }
            
            json_data = json.dumps(export_data, indent=2, default=str)
            
            if filename:
                with open(filename, 'w') as f:
                    f.write(json_data)
                logger.info(f"Deployment data exported to {filename}")
                
            return json_data
            
    def _perform_health_checks(self, deployment: CanaryDeployment):
        """Perform health checks for a deployment"""
        if deployment.deployment_id not in self.health_checks:
            return
            
        health_check_results = []
        
        for health_check in self.health_checks[deployment.deployment_id]:
            try:
                # Perform HTTP health check
                response = requests.request(
                    method=health_check.method,
                    url=health_check.endpoint,
                    headers=health_check.headers,
                    data=health_check.body,
                    timeout=health_check.timeout_seconds
                )
                
                success = response.status_code == health_check.expected_status
                health_check_results.append({
                    'check_id': health_check.check_id,
                    'name': health_check.name,
                    'success': success,
                    'status_code': response.status_code,
                    'response_time': response.elapsed.total_seconds() * 1000  # Convert to ms
                })
                
                # Record response time metric
                self.record_metric(
                    deployment.deployment_id,
                    "health_check_response_time",
                    response.elapsed.total_seconds() * 1000,
                    "canary",
                    {"check_id": health_check.check_id}
                )
                
            except Exception as e:
                health_check_results.append({
                    'check_id': health_check.check_id,
                    'name': health_check.name,
                    'success': False,
                    'error': str(e)
                })
                
        with self._lock:
            self.stats['health_checks_performed'] += 1
            
        return health_check_results
        
    def _evaluate_deployment_health(self, deployment: CanaryDeployment) -> Dict[str, Any]:
        """Evaluate deployment health against criteria"""
        # Get recent metrics
        recent_metrics = self.get_deployment_metrics(deployment.deployment_id, minutes=10)
        
        canary_metrics = [m for m in recent_metrics if m.version == 'canary']
        
        if not canary_metrics:
            return {'healthy': False, 'reason': 'No metrics available'}
            
        # Calculate metrics by type
        error_rates = [m.value for m in canary_metrics if m.metric_name == 'error_rate']
        response_times = [m.value for m in canary_metrics if m.metric_name == 'response_time']
        availability_metrics = [m.value for m in canary_metrics if m.metric_name == 'availability']
        
        evaluation = {
            'healthy': True,
            'reason': '',
            'metrics': {
                'error_rate': statistics.mean(error_rates) if error_rates else 0,
                'response_time': statistics.mean(response_times) if response_times else 0,
                'availability': statistics.mean(availability_metrics) if availability_metrics else 1.0
            }
        }
        
        # Check rollback criteria
        rollback_criteria = deployment.rollback_criteria
        
        if (evaluation['metrics']['error_rate'] > rollback_criteria.get('error_rate_threshold', 0.05)):
            evaluation['healthy'] = False
            evaluation['reason'] = f"Error rate too high: {evaluation['metrics']['error_rate']:.3f}"
            
        elif (evaluation['metrics']['response_time'] > rollback_criteria.get('response_time_threshold', 500)):
            evaluation['healthy'] = False
            evaluation['reason'] = f"Response time too high: {evaluation['metrics']['response_time']:.1f}ms"
            
        elif (evaluation['metrics']['availability'] < rollback_criteria.get('availability_threshold', 0.95)):
            evaluation['healthy'] = False
            evaluation['reason'] = f"Availability too low: {evaluation['metrics']['availability']:.3f}"
            
        return evaluation
        
    def _calculate_traffic_shift(self, deployment: CanaryDeployment) -> float:
        """Calculate new traffic percentage based on shift strategy"""
        if deployment.shift_strategy == TrafficShiftStrategy.MANUAL:
            return deployment.traffic_percentage
            
        elapsed_minutes = (datetime.now() - deployment.started_at).total_seconds() / 60
        
        if deployment.shift_strategy == TrafficShiftStrategy.IMMEDIATE:
            return deployment.max_traffic_percentage
            
        elif deployment.shift_strategy == TrafficShiftStrategy.LINEAR:
            progress = min(elapsed_minutes / deployment.shift_duration_minutes, 1.0)
            new_percentage = deployment.traffic_percentage + (deployment.max_traffic_percentage - deployment.traffic_percentage) * progress
            
        elif deployment.shift_strategy == TrafficShiftStrategy.EXPONENTIAL:
            progress = min(elapsed_minutes / deployment.shift_duration_minutes, 1.0)
            # Exponential growth: faster at the beginning
            new_percentage = deployment.traffic_percentage + (deployment.max_traffic_percentage - deployment.traffic_percentage) * (1 - np.exp(-3 * progress))
            
        else:
            new_percentage = deployment.traffic_percentage
            
        return min(new_percentage, deployment.max_traffic_percentage)
        
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self._running:
            try:
                # Monitor active deployments
                for deployment_id in list(self.active_deployments):
                    deployment = self.get_deployment(deployment_id)
                    
                    if not deployment:
                        continue
                        
                    # Perform health checks
                    health_results = self._perform_health_checks(deployment)
                    
                    # Evaluate deployment health
                    health_evaluation = self._evaluate_deployment_health(deployment)
                    
                    # Take action based on health evaluation
                    if not health_evaluation['healthy']:
                        logger.warning(f"Deployment {deployment_id} unhealthy: {health_evaluation['reason']}")
                        
                        # Trigger rollback if criteria met
                        if deployment.status == DeploymentStatus.MONITORING:
                            self.rollback_deployment(deployment_id, health_evaluation['reason'])
                            
                    # Check if deployment should be completed
                    if (deployment.status == DeploymentStatus.MONITORING and 
                        deployment.traffic_percentage >= deployment.max_traffic_percentage):
                        
                        # Check success criteria
                        if self._check_success_criteria(deployment):
                            self.stop_deployment(deployment_id, success=True)
                            
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(10)
                
    def _traffic_shift_loop(self):
        """Traffic shift loop"""
        while self._running:
            try:
                # Process active deployments
                for deployment_id in list(self.active_deployments):
                    deployment = self.get_deployment(deployment_id)
                    
                    if not deployment or deployment.status != DeploymentStatus.DEPLOYING:
                        continue
                        
                    # Calculate new traffic percentage
                    new_percentage = self._calculate_traffic_shift(deployment)
                    
                    # Update traffic percentage
                    if new_percentage != deployment.traffic_percentage:
                        self.update_traffic_percentage(deployment_id, new_percentage)
                        
                        # Check if we should transition to monitoring
                        if new_percentage >= deployment.max_traffic_percentage:
                            deployment.status = DeploymentStatus.MONITORING
                            logger.info(f"Deployment {deployment_id} transitioned to monitoring phase")
                            
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in traffic shift loop: {e}")
                time.sleep(30)
                
    def _check_success_criteria(self, deployment: CanaryDeployment) -> bool:
        """Check if deployment meets success criteria"""
        recent_metrics = self.get_deployment_metrics(deployment.deployment_id, minutes=10)
        
        if len(recent_metrics) < deployment.success_criteria.get('min_sample_size', 100):
            return False
            
        canary_metrics = [m for m in recent_metrics if m.version == 'canary']
        
        if not canary_metrics:
            return False
            
        # Calculate metrics
        error_rates = [m.value for m in canary_metrics if m.metric_name == 'error_rate']
        response_times = [m.value for m in canary_metrics if m.metric_name == 'response_time']
        availability_metrics = [m.value for m in canary_metrics if m.metric_name == 'availability']
        
        success_criteria = deployment.success_criteria
        
        # Check each criterion
        if error_rates and statistics.mean(error_rates) > success_criteria.get('error_rate_threshold', 0.01):
            return False
            
        if response_times and statistics.mean(response_times) > success_criteria.get('response_time_threshold', 100):
            return False
            
        if availability_metrics and statistics.mean(availability_metrics) < success_criteria.get('availability_threshold', 0.99):
            return False
            
        return True
        
    def _cleanup_old_data(self):
        """Clean up old data based on retention policy"""
        cutoff_time = datetime.now() - timedelta(hours=24)
        
        with self._lock:
            # Clean up old metrics
            for key in list(self.deployment_metrics.keys()):
                self.deployment_metrics[key] = [
                    m for m in self.deployment_metrics[key]
                    if m.timestamp >= cutoff_time
                ]

# Example usage and testing
def main():
    """Example usage of the canary deployment system"""
    deployment_manager = CanaryDeploymentManager()
    
    try:
        deployment_manager.start_deployment_manager()
        
        # Create a canary deployment
        deployment = deployment_manager.create_deployment(
            model_id="bharat-gpt-7b",
            version="v2.0",
            baseline_version="v1.0",
            traffic_percentage=10.0,
            max_traffic_percentage=100.0,
            shift_strategy=TrafficShiftStrategy.LINEAR,
            shift_duration_minutes=120
        )
        
        # Add health checks
        health_check = HealthCheck(
            check_id="api_health",
            name="API Health Check",
            endpoint="http://localhost:8000/health",
            method="GET",
            expected_status=200,
            timeout_seconds=10
        )
        
        deployment_manager.add_health_check(deployment.deployment_id, health_check)
        
        # Start deployment
        deployment_manager.start_deployment(deployment.deployment_id)
        
        # Simulate some metrics
        import random
        for i in range(20):
            # Record baseline metrics
            deployment_manager.record_metric(
                deployment.deployment_id,
                "error_rate",
                random.uniform(0.001, 0.01),
                "baseline"
            )
            
            deployment_manager.record_metric(
                deployment.deployment_id,
                "response_time",
                random.uniform(50, 150),
                "baseline"
            )
            
            deployment_manager.record_metric(
                deployment.deployment_id,
                "availability",
                random.uniform(0.98, 1.0),
                "baseline"
            )
            
            # Record canary metrics
            deployment_manager.record_metric(
                deployment.deployment_id,
                "error_rate",
                random.uniform(0.001, 0.02),
                "canary"
            )
            
            deployment_manager.record_metric(
                deployment.deployment_id,
                "response_time",
                random.uniform(60, 180),
                "canary"
            )
            
            deployment_manager.record_metric(
                deployment.deployment_id,
                "availability",
                random.uniform(0.97, 1.0),
                "canary"
            )
            
            time.sleep(5)
            
        # Get deployment statistics
        stats = deployment_manager.get_deployment_statistics(deployment.deployment_id)
        print(f"Deployment statistics: {stats}")
        
        # Get active deployments
        active_deployments = deployment_manager.get_active_deployments()
        print(f"Active deployments: {len(active_deployments)}")
        
        # Export deployment data
        deployment_manager.export_deployment_data("canary_deployment_data.json")
        
        time.sleep(30)  # Let monitoring run for a bit
        
    finally:
        deployment_manager.stop_deployment_manager()

if __name__ == "__main__":
    main()