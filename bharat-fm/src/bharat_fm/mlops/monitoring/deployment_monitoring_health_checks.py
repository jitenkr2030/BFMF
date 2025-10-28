"""
Deployment Monitoring and Health Checks for Bharat-FM

This module provides comprehensive monitoring and health check capabilities for deployed models,
including real-time monitoring, automated health checks, performance metrics tracking,
and alerting systems.

Author: Bharat-FM Team
Version: 1.0.0
"""

import os
import sys
import json
import time
import logging
import asyncio
import threading
import requests
import psutil
import docker
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
from kubernetes import client, config, watch
from kubernetes.client.rest import ApiException
import prometheus_client as prom
from prometheus_client import CollectorRegistry, Gauge, Counter, Histogram
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import slack_sdk
from slack_sdk.webhook import WebhookClient

# Import from our MLOps deployment infrastructure
from mlops_deployment_infrastructure import (
    DeploymentPlatform, DeploymentStrategy, DeploymentStatus, DeploymentConfig
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('deployment_monitoring.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class HealthCheck:
    """Health check configuration"""
    name: str
    endpoint: str
    method: str = "GET"
    expected_status: int = 200
    timeout: int = 30
    interval: int = 60
    retry_count: int = 3
    headers: Dict[str, str] = field(default_factory=dict)
    body: Optional[str] = None


@dataclass
class HealthCheckResult:
    """Health check result"""
    check_name: str
    status: HealthStatus
    response_time: float
    status_code: Optional[int] = None
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DeploymentMetrics:
    """Deployment metrics"""
    deployment_id: str
    cpu_usage: float
    memory_usage: float
    request_count: int
    error_rate: float
    response_time_p50: float
    response_time_p95: float
    response_time_p99: float
    availability: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class Alert:
    """Alert configuration"""
    name: str
    condition: str  # e.g., "cpu_usage > 80"
    severity: AlertSeverity
    threshold: float
    duration: int  # seconds
    enabled: bool = True
    notification_channels: List[str] = field(default_factory=list)


@dataclass
class AlertEvent:
    """Alert event"""
    alert_name: str
    severity: AlertSeverity
    message: str
    deployment_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    resolved: bool = False
    resolved_at: Optional[datetime] = None


class PrometheusMetrics:
    """Prometheus metrics collector"""
    
    def __init__(self):
        """Initialize Prometheus metrics"""
        self.registry = CollectorRegistry()
        
        # Deployment metrics
        self.deployment_cpu_usage = Gauge(
            'deployment_cpu_usage_percent',
            'CPU usage percentage for deployment',
            ['deployment_id', 'model_name', 'platform'],
            registry=self.registry
        )
        
        self.deployment_memory_usage = Gauge(
            'deployment_memory_usage_percent',
            'Memory usage percentage for deployment',
            ['deployment_id', 'model_name', 'platform'],
            registry=self.registry
        )
        
        self.deployment_request_count = Counter(
            'deployment_request_count_total',
            'Total request count for deployment',
            ['deployment_id', 'model_name', 'platform', 'status_code'],
            registry=self.registry
        )
        
        self.deployment_response_time = Histogram(
            'deployment_response_time_seconds',
            'Response time for deployment',
            ['deployment_id', 'model_name', 'platform'],
            registry=self.registry,
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
        )
        
        self.deployment_availability = Gauge(
            'deployment_availability_percent',
            'Availability percentage for deployment',
            ['deployment_id', 'model_name', 'platform'],
            registry=self.registry
        )
        
        self.deployment_error_rate = Gauge(
            'deployment_error_rate_percent',
            'Error rate percentage for deployment',
            ['deployment_id', 'model_name', 'platform'],
            registry=self.registry
        )
        
        # Health check metrics
        self.health_check_status = Gauge(
            'health_check_status',
            'Health check status (1=healthy, 0=unhealthy)',
            ['check_name', 'deployment_id'],
            registry=self.registry
        )
        
        self.health_check_response_time = Gauge(
            'health_check_response_time_seconds',
            'Health check response time',
            ['check_name', 'deployment_id'],
            registry=self.registry
        )
        
        # Alert metrics
        self.alert_events_total = Counter(
            'alert_events_total',
            'Total alert events',
            ['alert_name', 'severity', 'deployment_id'],
            registry=self.registry
        )
        
        self.alert_active = Gauge(
            'alert_active',
            'Active alerts (1=active, 0=resolved)',
            ['alert_name', 'severity', 'deployment_id'],
            registry=self.registry
        )
    
    def update_deployment_metrics(self, metrics: DeploymentMetrics):
        """Update deployment metrics"""
        labels = {
            'deployment_id': metrics.deployment_id,
            'model_name': metrics.deployment_id.split('-')[0],  # Extract model name
            'platform': 'kubernetes'  # Default, can be extracted from deployment info
        }
        
        self.deployment_cpu_usage.labels(**labels).set(metrics.cpu_usage)
        self.deployment_memory_usage.labels(**labels).set(metrics.memory_usage)
        self.deployment_availability.labels(**labels).set(metrics.availability)
        self.deployment_error_rate.labels(**labels).set(metrics.error_rate)
        
        # Update response time histogram
        self.deployment_response_time.labels(**labels).observe(metrics.response_time_p50)
    
    def update_health_check(self, result: HealthCheckResult, deployment_id: str):
        """Update health check metrics"""
        status_value = 1 if result.status == HealthStatus.HEALTHY else 0
        
        self.health_check_status.labels(
            check_name=result.check_name,
            deployment_id=deployment_id
        ).set(status_value)
        
        self.health_check_response_time.labels(
            check_name=result.check_name,
            deployment_id=deployment_id
        ).set(result.response_time)
    
    def record_alert(self, alert: AlertEvent):
        """Record alert event"""
        self.alert_events_total.labels(
            alert_name=alert.alert_name,
            severity=alert.severity.value,
            deployment_id=alert.deployment_id
        ).inc()
        
        active_value = 0 if alert.resolved else 1
        self.alert_active.labels(
            alert_name=alert.alert_name,
            severity=alert.severity.value,
            deployment_id=alert.deployment_id
        ).set(active_value)


class NotificationManager:
    """Notification manager for alerts"""
    
    def __init__(self):
        """Initialize notification manager"""
        self.email_config = {
            'smtp_server': os.getenv('SMTP_SERVER', 'smtp.gmail.com'),
            'smtp_port': int(os.getenv('SMTP_PORT', '587')),
            'smtp_username': os.getenv('SMTP_USERNAME', ''),
            'smtp_password': os.getenv('SMTP_PASSWORD', ''),
            'from_email': os.getenv('FROM_EMAIL', '')
        }
        
        self.slack_webhook_url = os.getenv('SLACK_WEBHOOK_URL', '')
        self.slack_client = WebhookClient(self.slack_webhook_url) if self.slack_webhook_url else None
    
    def send_email_alert(self, alert: AlertEvent, recipients: List[str]):
        """Send email alert"""
        if not self.email_config['smtp_username']:
            logger.warning("Email configuration not set, skipping email alert")
            return
        
        try:
            msg = MIMEMultipart()
            msg['From'] = self.email_config['from_email']
            msg['To'] = ', '.join(recipients)
            msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.alert_name}"
            
            body = f"""
Alert: {alert.alert_name}
Severity: {alert.severity.value.upper()}
Deployment: {alert.deployment_id}
Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
Message: {alert.message}
Status: {'RESOLVED' if alert.resolved else 'ACTIVE'}
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port'])
            server.starttls()
            server.login(self.email_config['smtp_username'], self.email_config['smtp_password'])
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Email alert sent: {alert.alert_name}")
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
    
    def send_slack_alert(self, alert: AlertEvent):
        """Send Slack alert"""
        if not self.slack_client:
            logger.warning("Slack webhook not configured, skipping Slack alert")
            return
        
        try:
            color = {
                AlertSeverity.INFO: "#36a64f",
                AlertSeverity.WARNING: "#ff9500",
                AlertSeverity.ERROR: "#ff0000",
                AlertSeverity.CRITICAL: "#990000"
            }.get(alert.severity, "#36a64f")
            
            status_text = "RESOLVED" if alert.resolved else "ACTIVE"
            
            message = {
                "attachments": [
                    {
                        "color": color,
                        "title": f"{alert.severity.value.upper()}: {alert.alert_name}",
                        "text": alert.message,
                        "fields": [
                            {
                                "title": "Deployment",
                                "value": alert.deployment_id,
                                "short": True
                            },
                            {
                                "title": "Status",
                                "value": status_text,
                                "short": True
                            },
                            {
                                "title": "Time",
                                "value": alert.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                                "short": False
                            }
                        ]
                    }
                ]
            }
            
            response = self.slack_client.send(message=message)
            logger.info(f"Slack alert sent: {alert.alert_name}")
            
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
    
    def send_alert(self, alert: AlertEvent, channels: List[str]):
        """Send alert through specified channels"""
        for channel in channels:
            if channel == "email":
                recipients = os.getenv('ALERT_EMAIL_RECIPIENTS', '').split(',')
                if recipients:
                    self.send_email_alert(alert, recipients)
            elif channel == "slack":
                self.send_slack_alert(alert)


class KubernetesHealthChecker:
    """Kubernetes deployment health checker"""
    
    def __init__(self, kube_config_path: Optional[str] = None):
        """Initialize Kubernetes health checker"""
        try:
            if kube_config_path:
                config.load_kube_config(config_file=kube_config_path)
            else:
                config.load_kube_config()
            
            self.apps_v1 = client.AppsV1Api()
            self.core_v1 = client.CoreV1Api()
            self.metrics_v1beta1 = client.CustomObjectsApi()
            logger.info("Kubernetes health checker initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Kubernetes health checker: {e}")
            raise
    
    def check_deployment_health(self, deployment_id: str) -> HealthStatus:
        """Check deployment health"""
        try:
            deployment = self.apps_v1.read_namespaced_deployment(
                name=deployment_id,
                namespace="default"
            )
            
            # Check deployment status
            if deployment.status.unavailable_replicas > 0:
                return HealthStatus.DEGRADED
            
            if deployment.status.available_replicas == 0:
                return HealthStatus.UNHEALTHY
            
            # Check pod status
            pods = self.core_v1.list_namespaced_pod(
                namespace="default",
                label_selector=f"deployment-id={deployment_id}"
            )
            
            unhealthy_pods = 0
            for pod in pods.items:
                if pod.status.phase != "Running":
                    unhealthy_pods += 1
                else:
                    # Check container status
                    for container_status in pod.status.container_statuses:
                        if not container_status.ready:
                            unhealthy_pods += 1
                            break
            
            if unhealthy_pods == 0:
                return HealthStatus.HEALTHY
            elif unhealthy_pods < len(pods.items) / 2:
                return HealthStatus.DEGRADED
            else:
                return HealthStatus.UNHEALTHY
                
        except Exception as e:
            logger.error(f"Failed to check deployment health: {e}")
            return HealthStatus.UNKNOWN
    
    def get_deployment_metrics(self, deployment_id: str) -> Optional[DeploymentMetrics]:
        """Get deployment metrics"""
        try:
            # Get deployment pods
            pods = self.core_v1.list_namespaced_pod(
                namespace="default",
                label_selector=f"deployment-id={deployment_id}"
            )
            
            if not pods.items:
                return None
            
            # Calculate average CPU and memory usage
            total_cpu = 0
            total_memory = 0
            pod_count = 0
            
            for pod in pods.items:
                try:
                    # Get pod metrics
                    pod_metrics = self.metrics_v1beta1.get_namespaced_custom_object(
                        group="metrics.k8s.io",
                        version="v1beta1",
                        namespace="default",
                        plural="pods",
                        name=pod.metadata.name
                    )
                    
                    for container in pod_metrics.get('containers', []):
                        cpu_usage = container['usage']['cpu']
                        memory_usage = container['usage']['memory']
                        
                        # Convert to numeric values
                        cpu_cores = self._parse_cpu(cpu_usage)
                        memory_bytes = self._parse_memory(memory_usage)
                        
                        total_cpu += cpu_cores
                        total_memory += memory_bytes
                        pod_count += 1
                        
                except Exception as e:
                    logger.warning(f"Failed to get metrics for pod {pod.metadata.name}: {e}")
            
            if pod_count == 0:
                return None
            
            avg_cpu = (total_cpu / pod_count) * 100  # Convert to percentage
            avg_memory = (total_memory / pod_count) * 100  # Convert to percentage
            
            return DeploymentMetrics(
                deployment_id=deployment_id,
                cpu_usage=avg_cpu,
                memory_usage=avg_memory,
                request_count=0,  # Would need to get from actual metrics
                error_rate=0.0,
                response_time_p50=0.0,
                response_time_p95=0.0,
                response_time_p99=0.0,
                availability=100.0
            )
            
        except Exception as e:
            logger.error(f"Failed to get deployment metrics: {e}")
            return None
    
    def _parse_cpu(self, cpu_str: str) -> float:
        """Parse CPU usage string"""
        if cpu_str.endswith('n'):
            return float(cpu_str[:-1]) / 1e9
        elif cpu_str.endswith('u'):
            return float(cpu_str[:-1]) / 1e6
        elif cpu_str.endswith('m'):
            return float(cpu_str[:-1]) / 1000
        else:
            return float(cpu_str)
    
    def _parse_memory(self, memory_str: str) -> float:
        """Parse memory usage string"""
        if memory_str.endswith('Ki'):
            return float(memory_str[:-2]) * 1024
        elif memory_str.endswith('Mi'):
            return float(memory_str[:-2]) * 1024 * 1024
        elif memory_str.endswith('Gi'):
            return float(memory_str[:-2]) * 1024 * 1024 * 1024
        else:
            return float(memory_str)


class DockerHealthChecker:
    """Docker container health checker"""
    
    def __init__(self):
        """Initialize Docker health checker"""
        try:
            self.client = docker.from_env()
            logger.info("Docker health checker initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Docker health checker: {e}")
            raise
    
    def check_deployment_health(self, deployment_id: str) -> HealthStatus:
        """Check deployment health"""
        try:
            containers = self.client.containers.list(all=True, filters={"name": deployment_id})
            
            if not containers:
                return HealthStatus.UNHEALTHY
            
            healthy_count = 0
            for container in containers:
                container.reload()
                if container.status == "running":
                    # Check health status if available
                    if container.attrs.get('State', {}).get('Health', {}).get('Status') == 'healthy':
                        healthy_count += 1
                    elif container.attrs.get('State', {}).get('Health') is None:
                        # No health check defined, assume healthy if running
                        healthy_count += 1
            
            if healthy_count == len(containers):
                return HealthStatus.HEALTHY
            elif healthy_count > 0:
                return HealthStatus.DEGRADED
            else:
                return HealthStatus.UNHEALTHY
                
        except Exception as e:
            logger.error(f"Failed to check Docker deployment health: {e}")
            return HealthStatus.UNKNOWN
    
    def get_deployment_metrics(self, deployment_id: str) -> Optional[DeploymentMetrics]:
        """Get deployment metrics"""
        try:
            containers = self.client.containers.list(all=True, filters={"name": deployment_id})
            
            if not containers:
                return None
            
            total_cpu = 0
            total_memory = 0
            container_count = 0
            
            for container in containers:
                try:
                    stats = container.stats(stream=False)
                    
                    # Calculate CPU usage
                    cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - \
                               stats['precpu_stats']['cpu_usage']['total_usage']
                    system_delta = stats['cpu_stats']['system_cpu_usage'] - \
                                  stats['precpu_stats']['system_cpu_usage']
                    
                    if system_delta > 0:
                        cpu_percent = (cpu_delta / system_delta) * \
                                     len(stats['cpu_stats']['cpu_usage']['percpu_usage']) * 100
                    else:
                        cpu_percent = 0.0
                    
                    # Calculate memory usage
                    memory_usage = stats['memory_stats']['usage']
                    memory_limit = stats['memory_stats']['limit']
                    memory_percent = (memory_usage / memory_limit) * 100
                    
                    total_cpu += cpu_percent
                    total_memory += memory_percent
                    container_count += 1
                    
                except Exception as e:
                    logger.warning(f"Failed to get stats for container: {e}")
            
            if container_count == 0:
                return None
            
            avg_cpu = total_cpu / container_count
            avg_memory = total_memory / container_count
            
            return DeploymentMetrics(
                deployment_id=deployment_id,
                cpu_usage=avg_cpu,
                memory_usage=avg_memory,
                request_count=0,
                error_rate=0.0,
                response_time_p50=0.0,
                response_time_p95=0.0,
                response_time_p99=0.0,
                availability=100.0
            )
            
        except Exception as e:
            logger.error(f"Failed to get Docker deployment metrics: {e}")
            return None


class HealthCheckExecutor:
    """Health check executor"""
    
    def __init__(self):
        """Initialize health check executor"""
        self.session = requests.Session()
        self.session.timeout = 30
    
    def execute_health_check(self, health_check: HealthCheck) -> HealthCheckResult:
        """Execute a single health check"""
        start_time = time.time()
        
        for attempt in range(health_check.retry_count):
            try:
                response = self.session.request(
                    method=health_check.method,
                    url=health_check.endpoint,
                    headers=health_check.headers,
                    data=health_check.body,
                    timeout=health_check.timeout
                )
                
                response_time = time.time() - start_time
                
                if response.status_code == health_check.expected_status:
                    status = HealthStatus.HEALTHY
                else:
                    status = HealthStatus.UNHEALTHY
                
                return HealthCheckResult(
                    check_name=health_check.name,
                    status=status,
                    response_time=response_time,
                    status_code=response.status_code,
                    metrics={
                        'attempt': attempt + 1,
                        'response_size': len(response.content),
                        'headers': dict(response.headers)
                    }
                )
                
            except requests.exceptions.Timeout:
                if attempt == health_check.retry_count - 1:
                    return HealthCheckResult(
                        check_name=health_check.name,
                        status=HealthStatus.UNHEALTHY,
                        response_time=time.time() - start_time,
                        error_message="Request timeout"
                    )
            except requests.exceptions.ConnectionError:
                if attempt == health_check.retry_count - 1:
                    return HealthCheckResult(
                        check_name=health_check.name,
                        status=HealthStatus.UNHEALTHY,
                        response_time=time.time() - start_time,
                        error_message="Connection error"
                    )
            except Exception as e:
                if attempt == health_check.retry_count - 1:
                    return HealthCheckResult(
                        check_name=health_check.name,
                        status=HealthStatus.UNHEALTHY,
                        response_time=time.time() - start_time,
                        error_message=str(e)
                    )
            
            # Wait before retry
            time.sleep(1)
        
        return HealthCheckResult(
            check_name=health_check.name,
            status=HealthStatus.UNHEALTHY,
            response_time=time.time() - start_time,
            error_message="Max retries exceeded"
        )


class AlertManager:
    """Alert manager"""
    
    def __init__(self):
        """Initialize alert manager"""
        self.alerts: Dict[str, Alert] = {}
        self.active_alerts: Dict[str, AlertEvent] = {}
        self.alert_history: List[AlertEvent] = []
        self.notification_manager = NotificationManager()
        self.prometheus_metrics = PrometheusMetrics()
    
    def add_alert(self, alert: Alert):
        """Add an alert"""
        self.alerts[alert.name] = alert
        logger.info(f"Alert added: {alert.name}")
    
    def remove_alert(self, alert_name: str):
        """Remove an alert"""
        if alert_name in self.alerts:
            del self.alerts[alert_name]
            logger.info(f"Alert removed: {alert_name}")
    
    def evaluate_alerts(self, deployment_id: str, metrics: DeploymentMetrics):
        """Evaluate alerts against metrics"""
        metrics_dict = {
            'cpu_usage': metrics.cpu_usage,
            'memory_usage': metrics.memory_usage,
            'error_rate': metrics.error_rate,
            'availability': metrics.availability,
            'response_time_p50': metrics.response_time_p50,
            'response_time_p95': metrics.response_time_p95,
            'response_time_p99': metrics.response_time_p99
        }
        
        for alert_name, alert in self.alerts.items():
            if not alert.enabled:
                continue
            
            try:
                # Evaluate alert condition
                condition_met = self._evaluate_condition(alert.condition, metrics_dict)
                
                if condition_met:
                    # Check if alert is already active
                    if alert_name not in self.active_alerts:
                        # Create new alert event
                        alert_event = AlertEvent(
                            alert_name=alert_name,
                            severity=alert.severity,
                            message=f"Alert triggered: {alert.condition}",
                            deployment_id=deployment_id
                        )
                        
                        self.active_alerts[alert_name] = alert_event
                        self.alert_history.append(alert_event)
                        
                        # Send notifications
                        self.notification_manager.send_alert(alert_event, alert.notification_channels)
                        
                        # Record in Prometheus
                        self.prometheus_metrics.record_alert(alert_event)
                        
                        logger.warning(f"Alert triggered: {alert_name} for deployment {deployment_id}")
                
                else:
                    # Check if alert should be resolved
                    if alert_name in self.active_alerts:
                        alert_event = self.active_alerts[alert_name]
                        alert_event.resolved = True
                        alert_event.resolved_at = datetime.now()
                        
                        # Send resolution notification
                        resolution_event = AlertEvent(
                            alert_name=alert_name,
                            severity=alert.severity,
                            message=f"Alert resolved: {alert.condition}",
                            deployment_id=deployment_id,
                            resolved=True
                        )
                        
                        self.notification_manager.send_alert(resolution_event, alert.notification_channels)
                        
                        # Remove from active alerts
                        del self.active_alerts[alert_name]
                        
                        # Record in Prometheus
                        self.prometheus_metrics.record_alert(resolution_event)
                        
                        logger.info(f"Alert resolved: {alert_name} for deployment {deployment_id}")
                        
            except Exception as e:
                logger.error(f"Failed to evaluate alert {alert_name}: {e}")
    
    def _evaluate_condition(self, condition: str, metrics: Dict[str, float]) -> bool:
        """Evaluate alert condition"""
        try:
            # Simple condition evaluation (can be enhanced with proper expression parsing)
            for metric_name, value in metrics.items():
                if metric_name in condition:
                    # Replace metric name with value
                    condition = condition.replace(metric_name, str(value))
            
            # Evaluate the condition
            return eval(condition)
        except Exception as e:
            logger.error(f"Failed to evaluate condition '{condition}': {e}")
            return False
    
    def get_active_alerts(self) -> List[AlertEvent]:
        """Get active alerts"""
        return list(self.active_alerts.values())
    
    def get_alert_history(self, limit: int = 100) -> List[AlertEvent]:
        """Get alert history"""
        return self.alert_history[-limit:]


class DeploymentMonitoringService:
    """Main deployment monitoring service"""
    
    def __init__(self):
        """Initialize monitoring service"""
        self.kubernetes_checker = KubernetesHealthChecker()
        self.docker_checker = DockerHealthChecker()
        self.health_check_executor = HealthCheckExecutor()
        self.alert_manager = AlertManager()
        self.prometheus_metrics = PrometheusMetrics()
        
        self.deployments: Dict[str, Dict[str, Any]] = {}
        self.health_checks: Dict[str, List[HealthCheck]] = {}
        self.monitoring_threads: Dict[str, threading.Thread] = {}
        self.running = False
        
        logger.info("Deployment monitoring service initialized")
    
    def add_deployment(self, deployment_id: str, platform: DeploymentPlatform, 
                      health_checks: List[HealthCheck] = None):
        """Add deployment for monitoring"""
        self.deployments[deployment_id] = {
            'platform': platform,
            'health_checks': health_checks or [],
            'last_check': None,
            'status': HealthStatus.UNKNOWN
        }
        
        # Add default health checks if none provided
        if not health_checks:
            default_health_check = HealthCheck(
                name=f"{deployment_id}-default",
                endpoint="http://localhost:8000/health",
                method="GET",
                expected_status=200,
                timeout=30,
                interval=60
            )
            self.deployments[deployment_id]['health_checks'].append(default_health_check)
        
        # Start monitoring thread
        self._start_monitoring_thread(deployment_id)
        
        logger.info(f"Deployment added for monitoring: {deployment_id}")
    
    def remove_deployment(self, deployment_id: str):
        """Remove deployment from monitoring"""
        if deployment_id in self.deployments:
            # Stop monitoring thread
            if deployment_id in self.monitoring_threads:
                self.monitoring_threads[deployment_id].join(timeout=5)
                del self.monitoring_threads[deployment_id]
            
            del self.deployments[deployment_id]
            logger.info(f"Deployment removed from monitoring: {deployment_id}")
    
    def add_alert(self, alert: Alert):
        """Add alert"""
        self.alert_manager.add_alert(alert)
    
    def get_deployment_status(self, deployment_id: str) -> Optional[HealthStatus]:
        """Get deployment status"""
        deployment = self.deployments.get(deployment_id)
        return deployment['status'] if deployment else None
    
    def get_deployment_metrics(self, deployment_id: str) -> Optional[DeploymentMetrics]:
        """Get deployment metrics"""
        deployment = self.deployments.get(deployment_id)
        if not deployment:
            return None
        
        platform = deployment['platform']
        
        if platform == DeploymentPlatform.KUBERNETES:
            return self.kubernetes_checker.get_deployment_metrics(deployment_id)
        elif platform == DeploymentPlatform.DOCKER:
            return self.docker_checker.get_deployment_metrics(deployment_id)
        else:
            logger.warning(f"Unsupported platform for metrics: {platform}")
            return None
    
    def get_health_check_results(self, deployment_id: str) -> List[HealthCheckResult]:
        """Get health check results for deployment"""
        deployment = self.deployments.get(deployment_id)
        if not deployment:
            return []
        
        results = []
        for health_check in deployment['health_checks']:
            result = self.health_check_executor.execute_health_check(health_check)
            results.append(result)
        
        return results
    
    def get_active_alerts(self) -> List[AlertEvent]:
        """Get active alerts"""
        return self.alert_manager.get_active_alerts()
    
    def start_monitoring(self):
        """Start monitoring all deployments"""
        self.running = True
        logger.info("Deployment monitoring started")
        
        # Start monitoring threads for all deployments
        for deployment_id in self.deployments:
            self._start_monitoring_thread(deployment_id)
    
    def stop_monitoring(self):
        """Stop monitoring all deployments"""
        self.running = False
        logger.info("Deployment monitoring stopping...")
        
        # Stop all monitoring threads
        for deployment_id, thread in self.monitoring_threads.items():
            thread.join(timeout=5)
        
        self.monitoring_threads.clear()
        logger.info("Deployment monitoring stopped")
    
    def _start_monitoring_thread(self, deployment_id: str):
        """Start monitoring thread for deployment"""
        if deployment_id in self.monitoring_threads:
            return
        
        thread = threading.Thread(
            target=self._monitor_deployment,
            args=(deployment_id,),
            daemon=True
        )
        thread.start()
        self.monitoring_threads[deployment_id] = thread
    
    def _monitor_deployment(self, deployment_id: str):
        """Monitor deployment (runs in separate thread)"""
        deployment = self.deployments.get(deployment_id)
        if not deployment:
            return
        
        platform = deployment['platform']
        health_checks = deployment['health_checks']
        
        while self.running and deployment_id in self.deployments:
            try:
                # Check deployment health
                if platform == DeploymentPlatform.KUBERNETES:
                    health_status = self.kubernetes_checker.check_deployment_health(deployment_id)
                elif platform == DeploymentPlatform.DOCKER:
                    health_status = self.docker_checker.check_deployment_health(deployment_id)
                else:
                    health_status = HealthStatus.UNKNOWN
                
                deployment['status'] = health_status
                deployment['last_check'] = datetime.now()
                
                # Get deployment metrics
                metrics = self.get_deployment_metrics(deployment_id)
                if metrics:
                    # Update Prometheus metrics
                    self.prometheus_metrics.update_deployment_metrics(metrics)
                    
                    # Evaluate alerts
                    self.alert_manager.evaluate_alerts(deployment_id, metrics)
                
                # Execute health checks
                for health_check in health_checks:
                    result = self.health_check_executor.execute_health_check(health_check)
                    self.prometheus_metrics.update_health_check(result, deployment_id)
                
                # Sleep until next check
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error monitoring deployment {deployment_id}: {e}")
                time.sleep(60)
    
    def get_prometheus_metrics(self) -> str:
        """Get Prometheus metrics in text format"""
        from prometheus_client.exposition import generate_latest
        return generate_latest(self.prometheus_metrics.registry).decode('utf-8')


def main():
    """Main function for testing the monitoring service"""
    # Create monitoring service
    monitoring_service = DeploymentMonitoringService()
    
    # Add some alerts
    cpu_alert = Alert(
        name="high_cpu_usage",
        condition="cpu_usage > 80",
        severity=AlertSeverity.WARNING,
        threshold=80.0,
        duration=300,
        notification_channels=["email", "slack"]
    )
    
    memory_alert = Alert(
        name="high_memory_usage",
        condition="memory_usage > 90",
        severity=AlertSeverity.CRITICAL,
        threshold=90.0,
        duration=300,
        notification_channels=["email", "slack"]
    )
    
    monitoring_service.add_alert(cpu_alert)
    monitoring_service.add_alert(memory_alert)
    
    # Add a deployment for monitoring
    health_check = HealthCheck(
        name="model_health",
        endpoint="http://localhost:8000/health",
        method="GET",
        expected_status=200,
        timeout=30,
        interval=60
    )
    
    monitoring_service.add_deployment(
        deployment_id="test-deployment",
        platform=DeploymentPlatform.KUBERNETES,
        health_checks=[health_check]
    )
    
    # Start monitoring
    monitoring_service.start_monitoring()
    
    try:
        # Keep running
        while True:
            time.sleep(60)
            
            # Print status
            status = monitoring_service.get_deployment_status("test-deployment")
            print(f"Deployment status: {status}")
            
            # Print active alerts
            active_alerts = monitoring_service.get_active_alerts()
            if active_alerts:
                print(f"Active alerts: {len(active_alerts)}")
                for alert in active_alerts:
                    print(f"  - {alert.alert_name}: {alert.message}")
    
    except KeyboardInterrupt:
        print("Stopping monitoring...")
        monitoring_service.stop_monitoring()


if __name__ == "__main__":
    main()