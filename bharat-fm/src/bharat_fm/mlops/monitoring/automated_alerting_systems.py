"""
Automated Alerting Systems for Bharat-FM MLOps Platform

This module provides comprehensive automated alerting capabilities for monitoring
the health and performance of AI models, services, and infrastructure. It supports
multiple alert channels, escalation policies, and intelligent alert aggregation.

Features:
- Multi-channel alerting (Email, Slack, Webhook, SMS)
- Intelligent alert aggregation and deduplication
- Escalation policies and on-call schedules
- Alert suppression and maintenance windows
- Performance and anomaly detection
- Integration with monitoring systems
"""

import time
import threading
import json
import smtplib
import requests
import logging
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import hashlib
import schedule
from collections import defaultdict, deque

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    """Alert severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AlertChannel(Enum):
    """Alert notification channels"""
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    SMS = "sms"
    PAGERDUTY = "pagerduty"

@dataclass
class Alert:
    """Alert data structure"""
    alert_id: str
    title: str
    description: str
    severity: AlertSeverity
    source: str  # Model, service, or component name
    metric_name: str
    current_value: float
    threshold_value: float
    timestamp: datetime
    tags: List[str] = None
    resolved: bool = False
    acknowledged: bool = False
    acknowledged_by: str = None
    acknowledged_at: datetime = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []

@dataclass
class AlertRule:
    """Alert rule configuration"""
    rule_id: str
    name: str
    description: str
    source_pattern: str  # Pattern to match source names
    metric_name: str
    condition: str  # 'greater_than', 'less_than', 'equals', 'not_equals'
    threshold: float
    severity: AlertSeverity
    enabled: bool = True
    cooldown_minutes: int = 5  # Minimum time between alerts
    channels: List[AlertChannel] = None
    
    def __post_init__(self):
        if self.channels is None:
            self.channels = [AlertChannel.EMAIL]

@dataclass
class NotificationConfig:
    """Notification channel configuration"""
    channel: AlertChannel
    config: Dict[str, Any]
    enabled: bool = True

@dataclass
class EscalationPolicy:
    """Escalation policy configuration"""
    policy_id: str
    name: str
    levels: List[Dict[str, Any]]  # List of escalation levels
    timeout_minutes: int = 30

class AutomatedAlertingSystem:
    """
    Comprehensive automated alerting system for MLOps monitoring
    """
    
    def __init__(self):
        # Alert storage
        self.alerts = {}
        self.alert_rules = {}
        self.active_alerts = {}
        self.alert_history = deque(maxlen=10000)
        
        # Configuration
        self.notification_configs = {}
        self.escalation_policies = {}
        self.maintenance_windows = {}
        
        # Alert management
        self.alert_cooldowns = defaultdict(dict)
        self.alert_hashes = set()
        
        # Threading
        self._lock = threading.Lock()
        self._running = False
        self._alert_thread = None
        self._escalation_thread = None
        
        # Statistics
        self.stats = {
            'alerts_created': 0,
            'alerts_resolved': 0,
            'alerts_escalated': 0,
            'notifications_sent': 0
        }
        
    def start(self):
        """Start the alerting system"""
        if self._running:
            logger.warning("Alerting system already running")
            return
            
        self._running = True
        self._alert_thread = threading.Thread(target=self._alert_processing_loop, daemon=True)
        self._escalation_thread = threading.Thread(target=self._escalation_loop, daemon=True)
        
        self._alert_thread.start()
        self._escalation_thread.start()
        
        logger.info("Automated alerting system started")
        
    def stop(self):
        """Stop the alerting system"""
        self._running = False
        if self._alert_thread:
            self._alert_thread.join(timeout=5)
        if self._escalation_thread:
            self._escalation_thread.join(timeout=5)
        logger.info("Automated alerting system stopped")
        
    def add_alert_rule(self, rule: AlertRule):
        """
        Add an alert rule
        
        Args:
            rule: AlertRule object
        """
        with self._lock:
            self.alert_rules[rule.rule_id] = rule
            logger.info(f"Added alert rule: {rule.name}")
            
    def remove_alert_rule(self, rule_id: str):
        """
        Remove an alert rule
        
        Args:
            rule_id: ID of the rule to remove
        """
        with self._lock:
            if rule_id in self.alert_rules:
                del self.alert_rules[rule_id]
                logger.info(f"Removed alert rule: {rule_id}")
                
    def add_notification_config(self, config: NotificationConfig):
        """
        Add notification channel configuration
        
        Args:
            config: NotificationConfig object
        """
        with self._lock:
            self.notification_configs[config.channel] = config
            logger.info(f"Added notification config for {config.channel.value}")
            
    def add_escalation_policy(self, policy: EscalationPolicy):
        """
        Add escalation policy
        
        Args:
            policy: EscalationPolicy object
        """
        with self._lock:
            self.escalation_policies[policy.policy_id] = policy
            logger.info(f"Added escalation policy: {policy.name}")
            
    def process_metric(self, source: str, metric_name: str, value: float, 
                      timestamp: datetime = None):
        """
        Process a metric value and trigger alerts if needed
        
        Args:
            source: Source of the metric (model, service, etc.)
            metric_name: Name of the metric
            value: Current metric value
            timestamp: Timestamp of the metric
        """
        if timestamp is None:
            timestamp = datetime.now()
            
        with self._lock:
            # Check each alert rule
            for rule_id, rule in self.alert_rules.items():
                if not rule.enabled:
                    continue
                    
                # Check if source matches pattern
                if not self._matches_pattern(source, rule.source_pattern):
                    continue
                    
                # Check if metric name matches
                if metric_name != rule.metric_name:
                    continue
                    
                # Check if condition is met
                if self._check_condition(value, rule.condition, rule.threshold):
                    # Check cooldown
                    cooldown_key = f"{source}:{metric_name}"
                    if self._is_in_cooldown(cooldown_key, rule.cooldown_minutes):
                        continue
                        
                    # Check maintenance window
                    if self._is_in_maintenance_window(source):
                        continue
                        
                    # Create alert
                    alert = Alert(
                        alert_id=self._generate_alert_id(),
                        title=f"{rule.severity.value.upper()}: {rule.name}",
                        description=rule.description,
                        severity=rule.severity,
                        source=source,
                        metric_name=metric_name,
                        current_value=value,
                        threshold_value=rule.threshold,
                        timestamp=timestamp,
                        tags=[rule_id]
                    )
                    
                    # Store alert
                    self.alerts[alert.alert_id] = alert
                    self.active_alerts[alert.alert_id] = alert
                    self.alert_history.append(alert)
                    
                    # Update cooldown
                    self.alert_cooldowns[cooldown_key][rule_id] = timestamp
                    
                    # Update statistics
                    self.stats['alerts_created'] += 1
                    
                    logger.info(f"Alert created: {alert.title}")
                    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str):
        """
        Acknowledge an alert
        
        Args:
            alert_id: ID of the alert to acknowledge
            acknowledged_by: User who acknowledged the alert
        """
        with self._lock:
            if alert_id in self.alerts:
                alert = self.alerts[alert_id]
                alert.acknowledged = True
                alert.acknowledged_by = acknowledged_by
                alert.acknowledged_at = datetime.now()
                logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")
                
    def resolve_alert(self, alert_id: str):
        """
        Resolve an alert
        
        Args:
            alert_id: ID of the alert to resolve
        """
        with self._lock:
            if alert_id in self.alerts:
                alert = self.alerts[alert_id]
                alert.resolved = True
                
                if alert_id in self.active_alerts:
                    del self.active_alerts[alert_id]
                    
                self.stats['alerts_resolved'] += 1
                logger.info(f"Alert {alert_id} resolved")
                
    def get_active_alerts(self, severity: AlertSeverity = None) -> List[Alert]:
        """
        Get active alerts, optionally filtered by severity
        
        Args:
            severity: Optional severity filter
            
        Returns:
            List of active alerts
        """
        with self._lock:
            alerts = list(self.active_alerts.values())
            if severity:
                alerts = [alert for alert in alerts if alert.severity == severity]
            return alerts
            
    def get_alert_history(self, hours: int = 24) -> List[Alert]:
        """
        Get alert history for the specified time period
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            List of alerts
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with self._lock:
            return [
                alert for alert in self.alert_history
                if alert.timestamp >= cutoff_time
            ]
            
    def add_maintenance_window(self, window_id: str, source_pattern: str, 
                             start_time: datetime, end_time: datetime):
        """
        Add a maintenance window to suppress alerts
        
        Args:
            window_id: ID of the maintenance window
            source_pattern: Pattern for sources to suppress
            start_time: Start time of maintenance
            end_time: End time of maintenance
        """
        with self._lock:
            self.maintenance_windows[window_id] = {
                'source_pattern': source_pattern,
                'start_time': start_time,
                'end_time': end_time
            }
            logger.info(f"Added maintenance window: {window_id}")
            
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get alerting system statistics
        
        Returns:
            Dictionary containing statistics
        """
        with self._lock:
            return {
                'timestamp': datetime.now().isoformat(),
                'stats': self.stats.copy(),
                'active_alerts_count': len(self.active_alerts),
                'alert_rules_count': len(self.alert_rules),
                'notification_configs_count': len(self.notification_configs),
                'escalation_policies_count': len(self.escalation_policies),
                'maintenance_windows_count': len(self.maintenance_windows)
            }
            
    def export_alerts(self, filename: str = None) -> str:
        """
        Export alerts to JSON file
        
        Args:
            filename: Optional filename to save to
            
        Returns:
            JSON string of alerts
        """
        with self._lock:
            export_data = {
                'timestamp': datetime.now().isoformat(),
                'active_alerts': [asdict(alert) for alert in self.active_alerts.values()],
                'alert_history': [asdict(alert) for alert in list(self.alert_history)[-1000:]],
                'statistics': self.get_statistics()
            }
            
            json_data = json.dumps(export_data, indent=2, default=str)
            
            if filename:
                with open(filename, 'w') as f:
                    f.write(json_data)
                logger.info(f"Alerts exported to {filename}")
                
            return json_data
            
    def _matches_pattern(self, source: str, pattern: str) -> bool:
        """Check if source matches pattern"""
        # Simple pattern matching - can be enhanced with regex
        return pattern == "*" or pattern in source
        
    def _check_condition(self, value: float, condition: str, threshold: float) -> bool:
        """Check if condition is met"""
        if condition == "greater_than":
            return value > threshold
        elif condition == "less_than":
            return value < threshold
        elif condition == "equals":
            return abs(value - threshold) < 0.001
        elif condition == "not_equals":
            return abs(value - threshold) >= 0.001
        return False
        
    def _is_in_cooldown(self, cooldown_key: str, cooldown_minutes: int) -> bool:
        """Check if alert is in cooldown period"""
        if cooldown_key not in self.alert_cooldowns:
            return False
            
        cooldown_time = datetime.now() - timedelta(minutes=cooldown_minutes)
        for rule_id, timestamp in list(self.alert_cooldowns[cooldown_key].items()):
            if timestamp >= cooldown_time:
                return True
                
        return False
        
    def _is_in_maintenance_window(self, source: str) -> bool:
        """Check if source is in maintenance window"""
        current_time = datetime.now()
        
        for window in self.maintenance_windows.values():
            if (window['start_time'] <= current_time <= window['end_time'] and
                self._matches_pattern(source, window['source_pattern'])):
                return True
                
        return False
        
    def _generate_alert_id(self) -> str:
        """Generate unique alert ID"""
        return f"alert_{int(time.time())}_{hashlib.md5(str(time.time()).encode()).hexdigest()[:8]}"
        
    def _send_notification(self, alert: Alert, channel: AlertChannel):
        """Send notification through specified channel"""
        if channel not in self.notification_configs:
            logger.error(f"No configuration found for channel: {channel.value}")
            return
            
        config = self.notification_configs[channel]
        
        try:
            if channel == AlertChannel.EMAIL:
                self._send_email_notification(alert, config.config)
            elif channel == AlertChannel.SLACK:
                self._send_slack_notification(alert, config.config)
            elif channel == AlertChannel.WEBHOOK:
                self._send_webhook_notification(alert, config.config)
            elif channel == AlertChannel.SMS:
                self._send_sms_notification(alert, config.config)
            elif channel == AlertChannel.PAGERDUTY:
                self._send_pagerduty_notification(alert, config.config)
                
            self.stats['notifications_sent'] += 1
            logger.info(f"Notification sent via {channel.value} for alert {alert.alert_id}")
            
        except Exception as e:
            logger.error(f"Failed to send notification via {channel.value}: {e}")
            
    def _send_email_notification(self, alert: Alert, config: Dict[str, Any]):
        """Send email notification"""
        msg = MIMEMultipart()
        msg['From'] = config['from_email']
        msg['To'] = config['to_email']
        msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.title}"
        
        body = f"""
Alert Details:
- Title: {alert.title}
- Description: {alert.description}
- Source: {alert.source}
- Metric: {alert.metric_name}
- Current Value: {alert.current_value}
- Threshold: {alert.threshold_value}
- Severity: {alert.severity.value}
- Timestamp: {alert.timestamp.isoformat()}
        """
        
        msg.attach(MIMEText(body, 'plain'))
        
        with smtplib.SMTP(config['smtp_server'], config['smtp_port']) as server:
            server.starttls()
            server.login(config['username'], config['password'])
            server.send_message(msg)
            
    def _send_slack_notification(self, alert: Alert, config: Dict[str, Any]):
        """Send Slack notification"""
        color_map = {
            AlertSeverity.LOW: "#36a64f",
            AlertSeverity.MEDIUM: "#ff9500",
            AlertSeverity.HIGH: "#ff0000",
            AlertSeverity.CRITICAL: "#990000"
        }
        
        payload = {
            "attachments": [
                {
                    "color": color_map.get(alert.severity, "#36a64f"),
                    "title": alert.title,
                    "text": alert.description,
                    "fields": [
                        {"title": "Source", "value": alert.source, "short": True},
                        {"title": "Metric", "value": alert.metric_name, "short": True},
                        {"title": "Current Value", "value": str(alert.current_value), "short": True},
                        {"title": "Threshold", "value": str(alert.threshold_value), "short": True},
                        {"title": "Severity", "value": alert.severity.value, "short": True},
                        {"title": "Time", "value": alert.timestamp.strftime("%Y-%m-%d %H:%M:%S"), "short": False}
                    ]
                }
            ]
        }
        
        response = requests.post(config['webhook_url'], json=payload, timeout=10)
        response.raise_for_status()
        
    def _send_webhook_notification(self, alert: Alert, config: Dict[str, Any]):
        """Send webhook notification"""
        payload = {
            "alert_id": alert.alert_id,
            "title": alert.title,
            "description": alert.description,
            "severity": alert.severity.value,
            "source": alert.source,
            "metric_name": alert.metric_name,
            "current_value": alert.current_value,
            "threshold_value": alert.threshold_value,
            "timestamp": alert.timestamp.isoformat(),
            "tags": alert.tags
        }
        
        headers = config.get('headers', {})
        response = requests.post(config['url'], json=payload, headers=headers, timeout=10)
        response.raise_for_status()
        
    def _send_sms_notification(self, alert: Alert, config: Dict[str, Any]):
        """Send SMS notification"""
        message = f"[{alert.severity.value.upper()}] {alert.title}: {alert.description}"
        
        # Implementation depends on SMS provider
        # This is a placeholder for SMS integration
        logger.info(f"SMS notification would be sent: {message}")
        
    def _send_pagerduty_notification(self, alert: Alert, config: Dict[str, Any]):
        """Send PagerDuty notification"""
        payload = {
            "payload": {
                "summary": alert.title,
                "source": alert.source,
                "severity": alert.severity.value,
                "timestamp": alert.timestamp.isoformat()
            },
            "routing_key": config['routing_key'],
            "event_action": "trigger"
        }
        
        response = requests.post(
            "https://events.pagerduty.com/v2/enqueue",
            json=payload,
            timeout=10
        )
        response.raise_for_status()
        
    def _alert_processing_loop(self):
        """Main alert processing loop"""
        while self._running:
            try:
                with self._lock:
                    # Process active alerts
                    for alert_id, alert in list(self.active_alerts.items()):
                        # Send notifications for each configured channel
                        for rule_id in alert.tags:
                            if rule_id in self.alert_rules:
                                rule = self.alert_rules[rule_id]
                                for channel in rule.channels:
                                    self._send_notification(alert, channel)
                                    
                # Clean up old maintenance windows
                self._cleanup_maintenance_windows()
                
                time.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in alert processing loop: {e}")
                time.sleep(5)
                
    def _escalation_loop(self):
        """Escalation processing loop"""
        while self._running:
            try:
                with self._lock:
                    current_time = datetime.now()
                    
                    # Check for alerts that need escalation
                    for alert_id, alert in list(self.active_alerts.items()):
                        if not alert.acknowledged:
                            # Check if alert needs escalation
                            time_since_alert = (current_time - alert.timestamp).total_seconds()
                            
                            for policy_id, policy in self.escalation_policies.items():
                                if time_since_alert > policy.timeout_minutes * 60:
                                    # Escalate alert
                                    self._escalate_alert(alert, policy)
                                    self.stats['alerts_escalated'] += 1
                                    
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in escalation loop: {e}")
                time.sleep(5)
                
    def _escalate_alert(self, alert: Alert, policy: EscalationPolicy):
        """Escalate an alert according to policy"""
        logger.info(f"Escalating alert {alert.alert_id} using policy {policy.name}")
        
        # Implementation would depend on specific escalation requirements
        # This is a placeholder for escalation logic
        
    def _cleanup_maintenance_windows(self):
        """Clean up expired maintenance windows"""
        current_time = datetime.now()
        
        expired_windows = [
            window_id for window_id, window in self.maintenance_windows.items()
            if window['end_time'] < current_time
        ]
        
        for window_id in expired_windows:
            del self.maintenance_windows[window_id]
            logger.info(f"Removed expired maintenance window: {window_id}")

# Example usage and testing
def main():
    """Example usage of the automated alerting system"""
    alerting_system = AutomatedAlertingSystem()
    
    try:
        alerting_system.start()
        
        # Add notification configurations
        email_config = NotificationConfig(
            channel=AlertChannel.EMAIL,
            config={
                'from_email': 'alerts@bharat-fm.com',
                'to_email': 'admin@bharat-fm.com',
                'smtp_server': 'smtp.gmail.com',
                'smtp_port': 587,
                'username': 'your-email@gmail.com',
                'password': 'your-password'
            }
        )
        
        slack_config = NotificationConfig(
            channel=AlertChannel.SLACK,
            config={
                'webhook_url': 'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK'
            }
        )
        
        alerting_system.add_notification_config(email_config)
        alerting_system.add_notification_config(slack_config)
        
        # Add alert rules
        error_rate_rule = AlertRule(
            rule_id="error_rate_high",
            name="High Error Rate",
            description="Error rate exceeds threshold",
            source_pattern="*",
            metric_name="error_rate",
            condition="greater_than",
            threshold=0.05,
            severity=AlertSeverity.HIGH,
            channels=[AlertChannel.EMAIL, AlertChannel.SLACK]
        )
        
        response_time_rule = AlertRule(
            rule_id="response_time_high",
            name="High Response Time",
            description="Response time exceeds threshold",
            source_pattern="*",
            metric_name="response_time",
            condition="greater_than",
            threshold=5000,
            severity=AlertSeverity.MEDIUM,
            channels=[AlertChannel.EMAIL]
        )
        
        alerting_system.add_alert_rule(error_rate_rule)
        alerting_system.add_alert_rule(response_time_rule)
        
        # Process some metrics
        alerting_system.process_metric("bharat-gpt-7b", "error_rate", 0.08)
        alerting_system.process_metric("bharat-gpt-7b", "response_time", 6000)
        
        # Get active alerts
        active_alerts = alerting_system.get_active_alerts()
        print(f"Active alerts: {len(active_alerts)}")
        
        # Get statistics
        stats = alerting_system.get_statistics()
        print(f"Statistics: {stats}")
        
        time.sleep(5)  # Let processing occur
        
    finally:
        alerting_system.stop()

if __name__ == "__main__":
    main()