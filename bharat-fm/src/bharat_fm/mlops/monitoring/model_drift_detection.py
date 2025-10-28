"""
Model Drift Detection System for Bharat-FM MLOps Platform

This module provides comprehensive model drift detection capabilities to monitor
AI model performance and data distribution changes over time. It supports various
drift detection algorithms and provides automated alerting.

Features:
- Statistical drift detection (KS test, Chi-square, etc.)
- Performance-based drift detection
- Data distribution monitoring
- Concept drift detection
- Automated alerting and reporting
- Drift visualization and analysis
"""

import time
import threading
import json
import logging
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
import statistics
import numpy as np
from scipy import stats
from scipy.spatial.distance import jensenshannon
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DriftMetrics:
    """Drift metrics data structure"""
    timestamp: datetime
    model_id: str
    metric_name: str
    drift_score: float
    drift_threshold: float
    drift_detected: bool
    p_value: float = None
    confidence_interval: Tuple[float, float] = None
    feature_contributions: Dict[str, float] = None
    
    def __post_init__(self):
        if self.feature_contributions is None:
            self.feature_contributions = {}

@dataclass
class DriftAlert:
    """Drift alert data structure"""
    alert_id: str
    model_id: str
    drift_type: str
    severity: str
    drift_score: float
    threshold: float
    description: str
    timestamp: datetime
    features_affected: List[str] = None
    recommended_action: str = None
    resolved: bool = False
    
    def __post_init__(self):
        if self.features_affected is None:
            self.features_affected = []

@dataclass
class BaselineProfile:
    """Baseline profile for drift detection"""
    model_id: str
    profile_type: str  # 'data_distribution', 'performance', 'predictions'
    created_at: datetime
    data: Dict[str, Any]
    statistics: Dict[str, float]
    sample_size: int

class ModelDriftDetector:
    """
    Comprehensive model drift detection system
    """
    
    def __init__(self, detection_interval: int = 300, alert_threshold: float = 0.05):
        self.detection_interval = detection_interval
        self.alert_threshold = alert_threshold
        
        # Data storage
        self.baseline_profiles = {}
        self.current_data = defaultdict(lambda: deque(maxlen=1000))
        self.drift_metrics = defaultdict(list)
        self.drift_alerts = {}
        self.drift_history = deque(maxlen=10000)
        
        # Configuration
        self.detection_methods = {
            'kolmogorov_smirnov': self._ks_test_drift,
            'chi_square': self._chi_square_drift,
            'kl_divergence': self._kl_divergence_drift,
            'jensen_shannon': self._jensen_shannon_drift,
            'population_stability_index': self._psi_drift,
            'performance_drift': self._performance_drift
        }
        
        # Monitoring configuration
        self.monitored_models = set()
        self.model_configs = {}
        
        # Threading
        self._lock = threading.Lock()
        self._running = False
        self._detection_thread = None
        
        # Statistics
        self.stats = {
            'drift_detections': 0,
            'alerts_generated': 0,
            'baselines_created': 0,
            'models_monitored': 0
        }
        
    def start_monitoring(self):
        """Start drift detection monitoring"""
        if self._running:
            logger.warning("Drift detection already running")
            return
            
        self._running = True
        self._detection_thread = threading.Thread(target=self._detection_loop, daemon=True)
        self._detection_thread.start()
        
        logger.info("Model drift detection started")
        
    def stop_monitoring(self):
        """Stop drift detection monitoring"""
        self._running = False
        if self._detection_thread:
            self._detection_thread.join(timeout=5)
        logger.info("Model drift detection stopped")
        
    def add_model(self, model_id: str, config: Dict[str, Any] = None):
        """
        Add a model to monitor for drift
        
        Args:
            model_id: Unique identifier for the model
            config: Model configuration
        """
        with self._lock:
            self.monitored_models.add(model_id)
            self.model_configs[model_id] = config or {}
            self.stats['models_monitored'] += 1
            
            logger.info(f"Added model to drift monitoring: {model_id}")
            
    def remove_model(self, model_id: str):
        """
        Remove a model from drift monitoring
        
        Args:
            model_id: Model identifier
        """
        with self._lock:
            if model_id in self.monitored_models:
                self.monitored_models.remove(model_id)
                if model_id in self.model_configs:
                    del self.model_configs[model_id]
                logger.info(f"Removed model from drift monitoring: {model_id}")
                
    def create_baseline(self, model_id: str, profile_type: str, 
                      data: Dict[str, Any], sample_size: int = None) -> BaselineProfile:
        """
        Create a baseline profile for drift detection
        
        Args:
            model_id: Model identifier
            profile_type: Type of profile ('data_distribution', 'performance', 'predictions')
            data: Baseline data
            sample_size: Sample size for the baseline
            
        Returns:
            BaselineProfile object
        """
        # Calculate baseline statistics
        statistics = self._calculate_statistics(data)
        
        baseline = BaselineProfile(
            model_id=model_id,
            profile_type=profile_type,
            created_at=datetime.now(),
            data=data,
            statistics=statistics,
            sample_size=sample_size or len(data) if hasattr(data, '__len__') else 1000
        )
        
        with self._lock:
            key = f"{model_id}:{profile_type}"
            self.baseline_profiles[key] = baseline
            self.stats['baselines_created'] += 1
            
        logger.info(f"Created baseline profile for {model_id} ({profile_type})")
        
        return baseline
        
    def record_data(self, model_id: str, profile_type: str, data: Dict[str, Any]):
        """
        Record current data for drift detection
        
        Args:
            model_id: Model identifier
            profile_type: Type of profile
            data: Current data
        """
        with self._lock:
            key = f"{model_id}:{profile_type}"
            self.current_data[key].append({
                'timestamp': datetime.now(),
                'data': data
            })
            
    def detect_drift(self, model_id: str, profile_type: str, 
                    method: str = 'kolmogorov_smirnov') -> Optional[DriftMetrics]:
        """
        Detect drift for a specific model and profile type
        
        Args:
            model_id: Model identifier
            profile_type: Type of profile
            method: Drift detection method
            
        Returns:
            DriftMetrics object or None if no baseline exists
        """
        baseline_key = f"{model_id}:{profile_type}"
        current_key = f"{model_id}:{profile_type}"
        
        with self._lock:
            if baseline_key not in self.baseline_profiles:
                logger.warning(f"No baseline found for {model_id} ({profile_type})")
                return None
                
            if current_key not in self.current_data or not self.current_data[current_key]:
                logger.warning(f"No current data for {model_id} ({profile_type})")
                return None
                
            baseline = self.baseline_profiles[baseline_key]
            current_data_points = list(self.current_data[current_key])
            
        # Aggregate current data
        current_aggregated = self._aggregate_data(current_data_points)
        
        # Perform drift detection
        if method in self.detection_methods:
            drift_result = self.detection_methods[method](baseline.data, current_aggregated)
        else:
            logger.error(f"Unknown drift detection method: {method}")
            return None
            
        # Create drift metrics
        drift_metrics = DriftMetrics(
            timestamp=datetime.now(),
            model_id=model_id,
            metric_name=profile_type,
            drift_score=drift_result['score'],
            drift_threshold=self.alert_threshold,
            drift_detected=drift_result['score'] > self.alert_threshold,
            p_value=drift_result.get('p_value'),
            confidence_interval=drift_result.get('confidence_interval'),
            feature_contributions=drift_result.get('feature_contributions', {})
        )
        
        # Store drift metrics
        with self._lock:
            self.drift_metrics[baseline_key].append(drift_metrics)
            self.stats['drift_detections'] += 1
            
        # Generate alert if drift detected
        if drift_metrics.drift_detected:
            self._generate_drift_alert(drift_metrics, method)
            
        return drift_metrics
        
    def get_drift_history(self, model_id: str, profile_type: str, 
                          hours: int = 24) -> List[DriftMetrics]:
        """
        Get drift history for a model and profile type
        
        Args:
            model_id: Model identifier
            profile_type: Type of profile
            hours: Number of hours to look back
            
        Returns:
            List of DriftMetrics objects
        """
        key = f"{model_id}:{profile_type}"
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with self._lock:
            if key not in self.drift_metrics:
                return []
                
            return [
                metrics for metrics in self.drift_metrics[key]
                if metrics.timestamp >= cutoff_time
            ]
            
    def get_active_alerts(self, model_id: str = None) -> List[DriftAlert]:
        """
        Get active drift alerts
        
        Args:
            model_id: Optional model filter
            
        Returns:
            List of DriftAlert objects
        """
        with self._lock:
            alerts = [alert for alert in self.drift_alerts.values() if not alert.resolved]
            
            if model_id:
                alerts = [alert for alert in alerts if alert.model_id == model_id]
                
            return alerts
            
    def resolve_alert(self, alert_id: str):
        """
        Resolve a drift alert
        
        Args:
            alert_id: ID of the alert to resolve
        """
        with self._lock:
            if alert_id in self.drift_alerts:
                self.drift_alerts[alert_id].resolved = True
                logger.info(f"Resolved drift alert: {alert_id}")
                
    def get_drift_statistics(self, model_id: str, profile_type: str) -> Dict[str, float]:
        """
        Get statistical summary of drift for a model and profile type
        
        Args:
            model_id: Model identifier
            profile_type: Type of profile
            
        Returns:
            Dictionary with drift statistics
        """
        history = self.get_drift_history(model_id, profile_type, hours=168)  # 1 week
        
        if not history:
            return {}
            
        drift_scores = [metrics.drift_score for metrics in history]
        
        return {
            'mean_drift_score': statistics.mean(drift_scores),
            'max_drift_score': max(drift_scores),
            'min_drift_score': min(drift_scores),
            'std_dev_drift': statistics.stdev(drift_scores) if len(drift_scores) > 1 else 0,
            'drift_detections_count': len([m for m in history if m.drift_detected]),
            'total_measurements': len(history)
        }
        
    def set_drift_threshold(self, threshold: float):
        """
        Set the drift detection threshold
        
        Args:
            threshold: New threshold value
        """
        self.alert_threshold = threshold
        logger.info(f"Set drift threshold to {threshold}")
        
    def export_drift_data(self, filename: str = None) -> str:
        """
        Export drift data to JSON file
        
        Args:
            filename: Optional filename to save to
            
        Returns:
            JSON string of drift data
        """
        with self._lock:
            export_data = {
                'timestamp': datetime.now().isoformat(),
                'baseline_profiles': {
                    key: asdict(profile) for key, profile in self.baseline_profiles.items()
                },
                'active_alerts': [asdict(alert) for alert in self.get_active_alerts()],
                'statistics': self.stats.copy(),
                'drift_threshold': self.alert_threshold,
                'monitored_models': list(self.monitored_models)
            }
            
            json_data = json.dumps(export_data, indent=2, default=str)
            
            if filename:
                with open(filename, 'w') as f:
                    f.write(json_data)
                logger.info(f"Drift data exported to {filename}")
                
            return json_data
            
    def _ks_test_drift(self, baseline: Dict[str, Any], current: Dict[str, Any]) -> Dict[str, Any]:
        """Kolmogorov-Smirnov test for drift detection"""
        drift_scores = {}
        feature_contributions = {}
        
        for feature in baseline.keys():
            if feature in current:
                baseline_values = np.array(baseline[feature])
                current_values = np.array(current[feature])
                
                if len(baseline_values) > 0 and len(current_values) > 0:
                    # Perform KS test
                    statistic, p_value = stats.ks_2samp(baseline_values, current_values)
                    drift_scores[feature] = statistic
                    feature_contributions[feature] = statistic
                    
        # Overall drift score (maximum KS statistic)
        overall_score = max(drift_scores.values()) if drift_scores else 0
        
        return {
            'score': overall_score,
            'p_value': p_value if drift_scores else 1.0,
            'feature_contributions': feature_contributions
        }
        
    def _chi_square_drift(self, baseline: Dict[str, Any], current: Dict[str, Any]) -> Dict[str, Any]:
        """Chi-square test for categorical drift detection"""
        drift_scores = {}
        feature_contributions = {}
        
        for feature in baseline.keys():
            if feature in current:
                # Create contingency table
                baseline_counts = np.array(list(baseline[feature].values()))
                current_counts = np.array(list(current[feature].values()))
                
                if len(baseline_counts) > 0 and len(current_counts) > 0:
                    # Perform chi-square test
                    observed = np.array([baseline_counts, current_counts])
                    chi2, p_value, _, _ = stats.chi2_contingency(observed)
                    
                    # Normalize chi-square statistic
                    normalized_chi2 = chi2 / np.sum(observed)
                    drift_scores[feature] = normalized_chi2
                    feature_contributions[feature] = normalized_chi2
                    
        # Overall drift score (maximum normalized chi-square)
        overall_score = max(drift_scores.values()) if drift_scores else 0
        
        return {
            'score': overall_score,
            'p_value': p_value if drift_scores else 1.0,
            'feature_contributions': feature_contributions
        }
        
    def _kl_divergence_drift(self, baseline: Dict[str, Any], current: Dict[str, Any]) -> Dict[str, Any]:
        """Kullback-Leibler divergence for drift detection"""
        drift_scores = {}
        feature_contributions = {}
        
        for feature in baseline.keys():
            if feature in current:
                baseline_dist = np.array(baseline[feature])
                current_dist = np.array(current[feature])
                
                # Normalize to probability distributions
                baseline_dist = baseline_dist / np.sum(baseline_dist)
                current_dist = current_dist / np.sum(current_dist)
                
                # Add small epsilon to avoid log(0)
                epsilon = 1e-10
                baseline_dist = baseline_dist + epsilon
                current_dist = current_dist + epsilon
                
                # Calculate KL divergence
                kl_div = np.sum(baseline_dist * np.log(baseline_dist / current_dist))
                drift_scores[feature] = kl_div
                feature_contributions[feature] = kl_div
                
        # Overall drift score (maximum KL divergence)
        overall_score = max(drift_scores.values()) if drift_scores else 0
        
        return {
            'score': overall_score,
            'feature_contributions': feature_contributions
        }
        
    def _jensen_shannon_drift(self, baseline: Dict[str, Any], current: Dict[str, Any]) -> Dict[str, Any]:
        """Jensen-Shannon divergence for drift detection"""
        drift_scores = {}
        feature_contributions = {}
        
        for feature in baseline.keys():
            if feature in current:
                baseline_dist = np.array(baseline[feature])
                current_dist = np.array(current[feature])
                
                # Normalize to probability distributions
                baseline_dist = baseline_dist / np.sum(baseline_dist)
                current_dist = current_dist / np.sum(current_dist)
                
                # Calculate Jensen-Shannon divergence
                js_div = jensenshannon(baseline_dist, current_dist)
                drift_scores[feature] = js_div
                feature_contributions[feature] = js_div
                
        # Overall drift score (maximum JS divergence)
        overall_score = max(drift_scores.values()) if drift_scores else 0
        
        return {
            'score': overall_score,
            'feature_contributions': feature_contributions
        }
        
    def _psi_drift(self, baseline: Dict[str, Any], current: Dict[str, Any]) -> Dict[str, Any]:
        """Population Stability Index for drift detection"""
        drift_scores = {}
        feature_contributions = {}
        
        for feature in baseline.keys():
            if feature in current:
                baseline_dist = np.array(baseline[feature])
                current_dist = np.array(current[feature])
                
                # Normalize to probability distributions
                baseline_dist = baseline_dist / np.sum(baseline_dist)
                current_dist = current_dist / np.sum(current_dist)
                
                # Calculate PSI
                psi = np.sum((baseline_dist - current_dist) * np.log(baseline_dist / current_dist))
                drift_scores[feature] = psi
                feature_contributions[feature] = psi
                
        # Overall drift score (maximum PSI)
        overall_score = max(drift_scores.values()) if drift_scores else 0
        
        return {
            'score': overall_score,
            'feature_contributions': feature_contributions
        }
        
    def _performance_drift(self, baseline: Dict[str, Any], current: Dict[str, Any]) -> Dict[str, Any]:
        """Performance-based drift detection"""
        drift_scores = {}
        feature_contributions = {}
        
        # Compare performance metrics
        for metric in baseline.keys():
            if metric in current:
                baseline_value = baseline[metric]
                current_value = current[metric]
                
                # Calculate relative change
                if baseline_value != 0:
                    relative_change = abs(current_value - baseline_value) / abs(baseline_value)
                else:
                    relative_change = abs(current_value) if current_value != 0 else 0
                    
                drift_scores[metric] = relative_change
                feature_contributions[metric] = relative_change
                
        # Overall drift score (maximum relative change)
        overall_score = max(drift_scores.values()) if drift_scores else 0
        
        return {
            'score': overall_score,
            'feature_contributions': feature_contributions
        }
        
    def _calculate_statistics(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate statistics for baseline data"""
        statistics = {}
        
        for key, values in data.items():
            if isinstance(values, list) and len(values) > 0:
                try:
                    values_array = np.array(values)
                    statistics[f"{key}_mean"] = np.mean(values_array)
                    statistics[f"{key}_std"] = np.std(values_array)
                    statistics[f"{key}_min"] = np.min(values_array)
                    statistics[f"{key}_max"] = np.max(values_array)
                    statistics[f"{key}_median"] = np.median(values_array)
                except (TypeError, ValueError):
                    # Handle non-numeric data
                    statistics[f"{key}_count"] = len(values)
                    
        return statistics
        
    def _aggregate_data(self, data_points: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate current data points"""
        if not data_points:
            return {}
            
        # Simple aggregation - combine all data points
        aggregated = {}
        
        for data_point in data_points:
            data = data_point['data']
            
            for key, values in data.items():
                if key not in aggregated:
                    aggregated[key] = []
                    
                if isinstance(values, list):
                    aggregated[key].extend(values)
                else:
                    aggregated[key].append(values)
                    
        return aggregated
        
    def _generate_drift_alert(self, drift_metrics: DriftMetrics, method: str):
        """Generate drift alert"""
        alert_id = f"drift_{int(time.time())}_{hash(drift_metrics.model_id) % 1000}"
        
        # Determine severity
        if drift_metrics.drift_score > 0.2:
            severity = "critical"
        elif drift_metrics.drift_score > 0.1:
            severity = "high"
        elif drift_metrics.drift_score > 0.05:
            severity = "medium"
        else:
            severity = "low"
            
        # Get most affected features
        top_features = sorted(
            drift_metrics.feature_contributions.items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]
        
        alert = DriftAlert(
            alert_id=alert_id,
            model_id=drift_metrics.model_id,
            drift_type=method,
            severity=severity,
            drift_score=drift_metrics.drift_score,
            threshold=drift_metrics.drift_threshold,
            description=f"Drift detected in {drift_metrics.metric_name} using {method}",
            timestamp=datetime.now(),
            features_affected=[feature for feature, _ in top_features],
            recommended_action=self._get_recommended_action(severity, drift_metrics.metric_name)
        )
        
        with self._lock:
            self.drift_alerts[alert_id] = alert
            self.drift_history.append(alert)
            self.stats['alerts_generated'] += 1
            
        logger.warning(f"Drift alert generated: {alert.description}")
        
    def _get_recommended_action(self, severity: str, metric_type: str) -> str:
        """Get recommended action based on severity and metric type"""
        if severity == "critical":
            return "Immediate model retraining required"
        elif severity == "high":
            return "Schedule model retraining soon"
        elif severity == "medium":
            return "Monitor closely and consider retraining"
        else:
            return "Continue monitoring, no immediate action required"
            
    def _detection_loop(self):
        """Main drift detection loop"""
        while self._running:
            try:
                # Perform drift detection for all monitored models
                for model_id in list(self.monitored_models):
                    for profile_type in ['data_distribution', 'performance', 'predictions']:
                        baseline_key = f"{model_id}:{profile_type}"
                        
                        # Check if baseline exists and we have current data
                        if (baseline_key in self.baseline_profiles and 
                            f"{model_id}:{profile_type}" in self.current_data):
                            
                            # Detect drift using multiple methods
                            for method in ['kolmogorov_smirnov', 'jensen_shannon']:
                                try:
                                    self.detect_drift(model_id, profile_type, method)
                                except Exception as e:
                                    logger.error(f"Drift detection failed for {model_id} ({profile_type}) with {method}: {e}")
                                    
                # Clean up old data
                self._cleanup_old_data()
                
                time.sleep(self.detection_interval)
                
            except Exception as e:
                logger.error(f"Error in drift detection loop: {e}")
                time.sleep(60)
                
    def _cleanup_old_data(self):
        """Clean up old data based on retention policy"""
        cutoff_time = datetime.now() - timedelta(hours=48)  # Keep 48 hours of data
        
        with self._lock:
            # Clean up current data
            for key in list(self.current_data.keys()):
                self.current_data[key] = deque(
                    [dp for dp in self.current_data[key] if dp['timestamp'] >= cutoff_time],
                    maxlen=self.current_data[key].maxlen
                )
                
            # Clean up drift metrics
            for key in list(self.drift_metrics.keys()):
                self.drift_metrics[key] = [
                    metrics for metrics in self.drift_metrics[key]
                    if metrics.timestamp >= cutoff_time
                ]

# Example usage and testing
def main():
    """Example usage of the model drift detection system"""
    drift_detector = ModelDriftDetector(detection_interval=10, alert_threshold=0.05)
    
    try:
        drift_detector.start_monitoring()
        
        # Add a model to monitor
        drift_detector.add_model("bharat-gpt-7b")
        
        # Create baseline data
        baseline_data = {
            'feature1': np.random.normal(0, 1, 1000).tolist(),
            'feature2': np.random.normal(5, 2, 1000).tolist(),
            'feature3': np.random.exponential(1, 1000).tolist()
        }
        
        baseline = drift_detector.create_baseline(
            model_id="bharat-gpt-7b",
            profile_type="data_distribution",
            data=baseline_data,
            sample_size=1000
        )
        
        # Record some current data (similar to baseline)
        for i in range(5):
            current_data = {
                'feature1': np.random.normal(0, 1, 100).tolist(),
                'feature2': np.random.normal(5, 2, 100).tolist(),
                'feature3': np.random.exponential(1, 100).tolist()
            }
            
            drift_detector.record_data("bharat-gpt-7b", "data_distribution", current_data)
            time.sleep(1)
            
        # Record some drifted data
        for i in range(5):
            drifted_data = {
                'feature1': np.random.normal(0.5, 1, 100).tolist(),  # Shifted mean
                'feature2': np.random.normal(5, 2, 100).tolist(),
                'feature3': np.random.exponential(1.5, 100).tolist()  # Changed scale
            }
            
            drift_detector.record_data("bharat-gpt-7b", "data_distribution", drifted_data)
            time.sleep(1)
            
        # Detect drift
        drift_metrics = drift_detector.detect_drift("bharat-gpt-7b", "data_distribution", "kolmogorov_smirnov")
        
        if drift_metrics:
            print(f"Drift detected: {drift_metrics.drift_detected}")
            print(f"Drift score: {drift_metrics.drift_score:.4f}")
            print(f"P-value: {drift_metrics.p_value:.4f}")
            
        # Get active alerts
        alerts = drift_detector.get_active_alerts()
        print(f"Active alerts: {len(alerts)}")
        
        for alert in alerts:
            print(f"Alert: {alert.description} (Severity: {alert.severity})")
            
        # Get drift statistics
        stats = drift_detector.get_drift_statistics("bharat-gpt-7b", "data_distribution")
        print(f"Drift statistics: {stats}")
        
        # Export drift data
        drift_detector.export_drift_data("drift_detection_data.json")
        
        time.sleep(15)  # Let monitoring run for a bit
        
    finally:
        drift_detector.stop_monitoring()

if __name__ == "__main__":
    main()