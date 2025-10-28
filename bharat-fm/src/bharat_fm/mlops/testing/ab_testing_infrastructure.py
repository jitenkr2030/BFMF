"""
A/B Testing Infrastructure for Bharat-FM MLOps Platform

This module provides comprehensive A/B testing capabilities for AI models,
allowing controlled experiments with statistical analysis and automated
winner determination. It supports multiple variants, traffic allocation,
and performance monitoring.

Features:
- Multi-variant A/B testing
- Dynamic traffic allocation
- Statistical significance testing
- Performance metrics tracking
- Automated winner determination
- Real-time monitoring and alerting
- Experiment lifecycle management
"""

import time
import threading
import json
import logging
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque
import statistics
import numpy as np
from scipy import stats
import random
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExperimentStatus(Enum):
    """Experiment status enumeration"""
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    STOPPED = "stopped"

class VariantStatus(Enum):
    """Variant status enumeration"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    WINNER = "winner"
    LOSER = "loser"

class AllocationStrategy(Enum):
    """Traffic allocation strategies"""
    UNIFORM = "uniform"
    WEIGHTED = "weighted"
    THOMPSON_SAMPLING = "thompson_sampling"
    EPSILON_GREEDY = "epsilon_greedy"

@dataclass
class ExperimentVariant:
    """Experiment variant definition"""
    variant_id: str
    name: str
    model_id: str
    model_version: str
    traffic_percentage: float
    configuration: Dict[str, Any] = field(default_factory=dict)
    status: VariantStatus = VariantStatus.ACTIVE
    created_at: datetime = field(default_factory=datetime.now)
    
@dataclass
class ExperimentMetric:
    """Experiment metric definition"""
    metric_id: str
    name: str
    type: str  # 'conversion', 'revenue', 'engagement', 'performance'
    aggregation: str  # 'sum', 'mean', 'count', 'rate'
    primary: bool = False
    description: str = ""
    
@dataclass
class ExperimentResult:
    """Experiment result data"""
    timestamp: datetime
    experiment_id: str
    variant_id: str
    user_id: str
    session_id: str
    metrics: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
@dataclass
class ExperimentSummary:
    """Experiment summary statistics"""
    experiment_id: str
    variant_id: str
    sample_size: int
    metric_values: Dict[str, List[float]]
    conversion_rate: float = 0.0
    mean_value: float = 0.0
    confidence_interval: Dict[str, float] = None
    p_value: float = None
    improvement_percentage: float = 0.0
    
    def __post_init__(self):
        if self.confidence_interval is None:
            self.confidence_interval = {}

@dataclass
class Experiment:
    """A/B experiment definition"""
    experiment_id: str
    name: str
    description: str
    variants: List[ExperimentVariant]
    metrics: List[ExperimentMetric]
    allocation_strategy: AllocationStrategy
    status: ExperimentStatus
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    min_sample_size: int = 1000
    significance_level: float = 0.05
    min_detectable_effect: float = 0.05
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
class ABTestingInfrastructure:
    """
    Comprehensive A/B testing infrastructure
    """
    
    def __init__(self, storage_dir: str = "ab_testing_storage"):
        self.storage_dir = storage_dir
        
        # Data storage
        self.experiments = {}
        self.experiment_results = defaultdict(list)
        self.experiment_summaries = {}
        self.user_assignments = {}  # user_id -> (experiment_id, variant_id)
        
        # Configuration
        self.default_metrics = [
            ExperimentMetric(
                metric_id="conversion_rate",
                name="Conversion Rate",
                type="conversion",
                aggregation="rate",
                primary=True,
                description="Primary conversion metric"
            ),
            ExperimentMetric(
                metric_id="revenue",
                name="Revenue",
                type="revenue",
                aggregation="sum",
                primary=False,
                description="Revenue generated"
            )
        ]
        
        # Monitoring
        self.active_experiments = set()
        
        # Threading
        self._lock = threading.Lock()
        self._running = False
        self._analysis_thread = None
        
        # Statistics
        self.stats = {
            'experiments_created': 0,
            'experiments_started': 0,
            'experiments_completed': 0,
            'results_recorded': 0,
            'users_assigned': 0,
            'statistical_tests_performed': 0
        }
        
    def start_infrastructure(self):
        """Start the A/B testing infrastructure"""
        if self._running:
            logger.warning("A/B testing infrastructure already running")
            return
            
        self._running = True
        self._analysis_thread = threading.Thread(target=self._analysis_loop, daemon=True)
        self._analysis_thread.start()
        
        logger.info("A/B testing infrastructure started")
        
    def stop_infrastructure(self):
        """Stop the A/B testing infrastructure"""
        self._running = False
        if self._analysis_thread:
            self._analysis_thread.join(timeout=5)
        logger.info("A/B testing infrastructure stopped")
        
    def create_experiment(self, experiment: Experiment) -> str:
        """
        Create a new A/B experiment
        
        Args:
            experiment: Experiment object
            
        Returns:
            Experiment ID
        """
        with self._lock:
            # Validate experiment
            self._validate_experiment(experiment)
            
            # Store experiment
            self.experiments[experiment.experiment_id] = experiment
            self.stats['experiments_created'] += 1
            
            logger.info(f"Created A/B experiment: {experiment.experiment_id}")
            
            return experiment.experiment_id
            
    def start_experiment(self, experiment_id: str):
        """
        Start an A/B experiment
        
        Args:
            experiment_id: Experiment identifier
        """
        with self._lock:
            if experiment_id not in self.experiments:
                raise ValueError(f"Experiment {experiment_id} not found")
                
            experiment = self.experiments[experiment_id]
            
            if experiment.status != ExperimentStatus.DRAFT:
                raise ValueError(f"Experiment {experiment_id} is not in DRAFT status")
                
            experiment.status = ExperimentStatus.RUNNING
            experiment.start_time = datetime.now()
            experiment.updated_at = datetime.now()
            
            self.active_experiments.add(experiment_id)
            self.stats['experiments_started'] += 1
            
            logger.info(f"Started A/B experiment: {experiment_id}")
            
    def stop_experiment(self, experiment_id: str, declare_winner: bool = False):
        """
        Stop an A/B experiment
        
        Args:
            experiment_id: Experiment identifier
            declare_winner: Whether to declare a winner
        """
        with self._lock:
            if experiment_id not in self.experiments:
                raise ValueError(f"Experiment {experiment_id} not found")
                
            experiment = self.experiments[experiment_id]
            
            if experiment.status not in [ExperimentStatus.RUNNING, ExperimentStatus.PAUSED]:
                raise ValueError(f"Experiment {experiment_id} is not active")
                
            experiment.status = ExperimentStatus.STOPPED
            experiment.end_time = datetime.now()
            experiment.updated_at = datetime.now()
            
            if experiment_id in self.active_experiments:
                self.active_experiments.remove(experiment_id)
                
            if declare_winner:
                self._declare_winner(experiment_id)
                
            logger.info(f"Stopped A/B experiment: {experiment_id}")
            
    def pause_experiment(self, experiment_id: str):
        """
        Pause an A/B experiment
        
        Args:
            experiment_id: Experiment identifier
        """
        with self._lock:
            if experiment_id not in self.experiments:
                raise ValueError(f"Experiment {experiment_id} not found")
                
            experiment = self.experiments[experiment_id]
            
            if experiment.status != ExperimentStatus.RUNNING:
                raise ValueError(f"Experiment {experiment_id} is not running")
                
            experiment.status = ExperimentStatus.PAUSED
            experiment.updated_at = datetime.now()
            
            logger.info(f"Paused A/B experiment: {experiment_id}")
            
    def resume_experiment(self, experiment_id: str):
        """
        Resume a paused A/B experiment
        
        Args:
            experiment_id: Experiment identifier
        """
        with self._lock:
            if experiment_id not in self.experiments:
                raise ValueError(f"Experiment {experiment_id} not found")
                
            experiment = self.experiments[experiment_id]
            
            if experiment.status != ExperimentStatus.PAUSED:
                raise ValueError(f"Experiment {experiment_id} is not paused")
                
            experiment.status = ExperimentStatus.RUNNING
            experiment.updated_at = datetime.now()
            
            self.active_experiments.add(experiment_id)
            
            logger.info(f"Resumed A/B experiment: {experiment_id}")
            
    def assign_user(self, user_id: str, session_id: str, experiment_id: str = None) -> Optional[str]:
        """
        Assign a user to an experiment variant
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            experiment_id: Optional experiment identifier
            
        Returns:
            Variant ID or None if no assignment made
        """
        with self._lock:
            # If experiment_id is provided, assign to that specific experiment
            if experiment_id:
                return self._assign_to_experiment(user_id, session_id, experiment_id)
            else:
                # Assign to all active experiments
                assigned_variants = {}
                
                for exp_id in self.active_experiments:
                    variant_id = self._assign_to_experiment(user_id, session_id, exp_id)
                    if variant_id:
                        assigned_variants[exp_id] = variant_id
                        
                return assigned_variants.get(experiment_id) if experiment_id else assigned_variants
                    
    def record_result(self, experiment_id: str, variant_id: str, user_id: str,
                     session_id: str, metrics: Dict[str, float],
                     metadata: Dict[str, Any] = None):
        """
        Record experiment result
        
        Args:
            experiment_id: Experiment identifier
            variant_id: Variant identifier
            user_id: User identifier
            session_id: Session identifier
            metrics: Dictionary of metric values
            metadata: Additional metadata
        """
        result = ExperimentResult(
            timestamp=datetime.now(),
            experiment_id=experiment_id,
            variant_id=variant_id,
            user_id=user_id,
            session_id=session_id,
            metrics=metrics,
            metadata=metadata or {}
        )
        
        with self._lock:
            self.experiment_results[experiment_id].append(result)
            self.stats['results_recorded'] += 1
            
        logger.debug(f"Recorded result for experiment {experiment_id}, variant {variant_id}")
        
    def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        """
        Get experiment by ID
        
        Args:
            experiment_id: Experiment identifier
            
        Returns:
            Experiment object or None
        """
        with self._lock:
            return self.experiments.get(experiment_id)
            
    def get_user_assignment(self, user_id: str, experiment_id: str) -> Optional[str]:
        """
        Get user's variant assignment for an experiment
        
        Args:
            user_id: User identifier
            experiment_id: Experiment identifier
            
        Returns:
            Variant ID or None
        """
        key = f"{user_id}:{experiment_id}"
        return self.user_assignments.get(key)
        
    def get_experiment_results(self, experiment_id: str, 
                             hours: int = 24) -> List[ExperimentResult]:
        """
        Get experiment results for a time period
        
        Args:
            experiment_id: Experiment identifier
            hours: Number of hours to look back
            
        Returns:
            List of ExperimentResult objects
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with self._lock:
            if experiment_id not in self.experiment_results:
                return []
                
            return [
                result for result in self.experiment_results[experiment_id]
                if result.timestamp >= cutoff_time
            ]
            
    def get_experiment_summary(self, experiment_id: str) -> Dict[str, ExperimentSummary]:
        """
        Get experiment summary statistics
        
        Args:
            experiment_id: Experiment identifier
            
        Returns:
            Dictionary of variant summaries
        """
        experiment = self.get_experiment(experiment_id)
        if not experiment:
            return {}
            
        # Get recent results
        recent_results = self.get_experiment_results(experiment_id, hours=168)  # 1 week
        
        # Group results by variant
        variant_results = defaultdict(list)
        for result in recent_results:
            variant_results[result.variant_id].append(result)
            
        summaries = {}
        
        for variant in experiment.variants:
            variant_id = variant.variant_id
            results = variant_results.get(variant_id, [])
            
            if not results:
                continue
                
            # Calculate summary statistics
            summary = self._calculate_variant_summary(
                experiment_id, variant_id, results, experiment.metrics
            )
            
            summaries[variant_id] = summary
            
        with self._lock:
            self.experiment_summaries[experiment_id] = summaries
            
        return summaries
        
    def perform_statistical_test(self, experiment_id: str, 
                               metric_id: str = None) -> Dict[str, Any]:
        """
        Perform statistical analysis on experiment results
        
        Args:
            experiment_id: Experiment identifier
            metric_id: Optional metric identifier
            
        Returns:
            Dictionary with test results
        """
        experiment = self.get_experiment(experiment_id)
        if not experiment:
            return {}
            
        # Get experiment summary
        summaries = self.get_experiment_summary(experiment_id)
        
        if not summaries:
            return {}
            
        # Use primary metric if not specified
        if metric_id is None:
            primary_metric = next((m for m in experiment.metrics if m.primary), None)
            if not primary_metric:
                return {}
            metric_id = primary_metric.metric_id
            
        # Get variant data for comparison
        variant_data = {}
        for variant_id, summary in summaries.items():
            if metric_id in summary.metric_values:
                variant_data[variant_id] = summary.metric_values[metric_id]
                
        if len(variant_data) < 2:
            return {}
            
        # Perform statistical tests
        test_results = {}
        
        # Compare each variant against the first (control)
        control_variant_id = list(variant_data.keys())[0]
        control_data = variant_data[control_variant_id]
        
        for variant_id, data in list(variant_data.items())[1:]:
            # T-test
            t_stat, p_value = stats.ttest_ind(control_data, data)
            
            # Calculate effect size
            effect_size = (np.mean(data) - np.mean(control_data)) / np.std(control_data)
            
            # Calculate improvement percentage
            control_mean = np.mean(control_data)
            variant_mean = np.mean(data)
            improvement = ((variant_mean - control_mean) / control_mean * 100) if control_mean != 0 else 0
            
            test_results[variant_id] = {
                't_statistic': t_stat,
                'p_value': p_value,
                'effect_size': effect_size,
                'improvement_percentage': improvement,
                'significant': p_value < experiment.significance_level,
                'control_mean': control_mean,
                'variant_mean': variant_mean
            }
            
        with self._lock:
            self.stats['statistical_tests_performed'] += 1
            
        return test_results
        
    def declare_winner(self, experiment_id: str, variant_id: str = None):
        """
        Declare a winner for an experiment
        
        Args:
            experiment_id: Experiment identifier
            variant_id: Optional variant ID (if None, determine automatically)
        """
        experiment = self.get_experiment(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment {experiment_id} not found")
            
        if variant_id is None:
            # Determine winner automatically
            test_results = self.perform_statistical_test(experiment_id)
            
            if not test_results:
                raise ValueError("Cannot determine winner: insufficient data or no significant results")
                
            # Find variant with highest significant improvement
            best_variant = None
            best_improvement = -float('inf')
            
            for var_id, results in test_results.items():
                if results['significant'] and results['improvement_percentage'] > best_improvement:
                    best_improvement = results['improvement_percentage']
                    best_variant = var_id
                    
            if best_variant is None:
                raise ValueError("No significant winner found")
                
            variant_id = best_variant
            
        # Update variant statuses
        with self._lock:
            for variant in experiment.variants:
                if variant.variant_id == variant_id:
                    variant.status = VariantStatus.WINNER
                else:
                    variant.status = VariantStatus.LOSER
                    
            experiment.status = ExperimentStatus.COMPLETED
            experiment.end_time = datetime.now()
            experiment.updated_at = datetime.now()
            
            if experiment_id in self.active_experiments:
                self.active_experiments.remove(experiment_id)
                
            self.stats['experiments_completed'] += 1
            
        logger.info(f"Declared winner for experiment {experiment_id}: variant {variant_id}")
        
    def get_active_experiments(self) -> List[Experiment]:
        """
        Get all active experiments
        
        Returns:
            List of active Experiment objects
        """
        with self._lock:
            return [self.experiments[exp_id] for exp_id in self.active_experiments]
            
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get A/B testing statistics
        
        Returns:
            Dictionary with statistics
        """
        with self._lock:
            return {
                'timestamp': datetime.now().isoformat(),
                'stats': self.stats.copy(),
                'experiments_count': len(self.experiments),
                'active_experiments_count': len(self.active_experiments),
                'total_results_count': sum(len(results) for results in self.experiment_results.values())
            }
            
    def export_experiment_data(self, filename: str = None) -> str:
        """
        Export experiment data to JSON file
        
        Args:
            filename: Optional filename to save to
            
        Returns:
            JSON string of experiment data
        """
        with self._lock:
            export_data = {
                'timestamp': datetime.now().isoformat(),
                'experiments': {k: asdict(v) for k, v in self.experiments.items()},
                'active_experiments': list(self.active_experiments),
                'statistics': self.get_statistics(),
                'recent_results': {
                    exp_id: [asdict(r) for r in results[-100:]]  # Last 100 results per experiment
                    for exp_id, results in self.experiment_results.items()
                }
            }
            
            json_data = json.dumps(export_data, indent=2, default=str)
            
            if filename:
                with open(filename, 'w') as f:
                    f.write(json_data)
                logger.info(f"Experiment data exported to {filename}")
                
            return json_data
            
    def _validate_experiment(self, experiment: Experiment):
        """Validate experiment definition"""
        if not experiment.variants or len(experiment.variants) < 2:
            raise ValueError("Experiment must have at least 2 variants")
            
        # Check traffic allocation sums to 100%
        total_traffic = sum(variant.traffic_percentage for variant in experiment.variants)
        if not (99.0 <= total_traffic <= 101.0):  # Allow small rounding errors
            raise ValueError(f"Traffic allocation must sum to 100% (got {total_traffic}%)")
            
        # Check for duplicate variant IDs
        variant_ids = [variant.variant_id for variant in experiment.variants]
        if len(variant_ids) != len(set(variant_ids)):
            raise ValueError("Duplicate variant IDs found")
            
        # Check for primary metric
        primary_metrics = [m for m in experiment.metrics if m.primary]
        if len(primary_metrics) != 1:
            raise ValueError("Exactly one primary metric must be specified")
            
    def _assign_to_experiment(self, user_id: str, session_id: str, 
                            experiment_id: str) -> Optional[str]:
        """Assign user to a specific experiment variant"""
        experiment = self.get_experiment(experiment_id)
        if not experiment or experiment.status != ExperimentStatus.RUNNING:
            return None
            
        # Check if user is already assigned
        key = f"{user_id}:{experiment_id}"
        if key in self.user_assignments:
            return self.user_assignments[key]
            
        # Assign variant based on strategy
        variant_id = self._select_variant(experiment, user_id, session_id)
        
        if variant_id:
            self.user_assignments[key] = variant_id
            self.stats['users_assigned'] += 1
            
        return variant_id
        
    def _select_variant(self, experiment: Experiment, user_id: str, 
                       session_id: str) -> Optional[str]:
        """Select variant for user based on allocation strategy"""
        if experiment.allocation_strategy == AllocationStrategy.UNIFORM:
            return self._uniform_allocation(experiment, user_id)
        elif experiment.allocation_strategy == AllocationStrategy.WEIGHTED:
            return self._weighted_allocation(experiment, user_id)
        elif experiment.allocation_strategy == AllocationStrategy.THOMPSON_SAMPLING:
            return self._thompson_sampling_allocation(experiment, user_id)
        elif experiment.allocation_strategy == AllocationStrategy.EPSILON_GREEDY:
            return self._epsilon_greedy_allocation(experiment, user_id)
        else:
            return self._uniform_allocation(experiment, user_id)
            
    def _uniform_allocation(self, experiment: Experiment, user_id: str) -> str:
        """Uniform random allocation"""
        # Generate hash for consistent assignment
        hash_value = int(hashlib.md5(f"{user_id}:{experiment.experiment_id}".encode()).hexdigest(), 16)
        random.seed(hash_value)
        
        # Select variant based on traffic percentages
        rand_val = random.uniform(0, 100)
        cumulative = 0
        
        for variant in experiment.variants:
            cumulative += variant.traffic_percentage
            if rand_val <= cumulative:
                return variant.variant_id
                
        # Fallback to first variant
        return experiment.variants[0].variant_id
        
    def _weighted_allocation(self, experiment: Experiment, user_id: str) -> str:
        """Weighted random allocation based on performance"""
        # For now, fall back to uniform allocation
        # In a real implementation, this would use performance data
        return self._uniform_allocation(experiment, user_id)
        
    def _thompson_sampling_allocation(self, experiment: Experiment, user_id: str) -> str:
        """Thompson sampling allocation"""
        # For now, fall back to uniform allocation
        # In a real implementation, this would use Bayesian optimization
        return self._uniform_allocation(experiment, user_id)
        
    def _epsilon_greedy_allocation(self, experiment: Experiment, user_id: str) -> str:
        """Epsilon-greedy allocation"""
        # For now, fall back to uniform allocation
        # In a real implementation, this would use exploration/exploitation trade-off
        return self._uniform_allocation(experiment, user_id)
        
    def _calculate_variant_summary(self, experiment_id: str, variant_id: str,
                                 results: List[ExperimentResult], 
                                 metrics: List[ExperimentMetric]) -> ExperimentSummary:
        """Calculate summary statistics for a variant"""
        if not results:
            return ExperimentSummary(
                experiment_id=experiment_id,
                variant_id=variant_id,
                sample_size=0,
                metric_values={}
            )
            
        # Group metrics by type
        metric_values = {}
        for metric in metrics:
            values = []
            for result in results:
                if metric.metric_id in result.metrics:
                    values.append(result.metrics[metric.metric_id])
                    
            metric_values[metric.metric_id] = values
            
        # Calculate primary metric statistics
        primary_metric = next((m for m in metrics if m.primary), None)
        conversion_rate = 0.0
        mean_value = 0.0
        confidence_interval = {}
        p_value = None
        
        if primary_metric and primary_metric.metric_id in metric_values:
            values = metric_values[primary_metric.metric_id]
            
            if values:
                if primary_metric.aggregation == "rate":
                    conversion_rate = np.mean(values)
                    mean_value = conversion_rate
                    
                    # Calculate confidence interval for proportion
                    n = len(values)
                    p = conversion_rate
                    se = np.sqrt(p * (1 - p) / n)
                    z = stats.norm.ppf(0.975)  # 95% CI
                    confidence_interval = {
                        'lower': max(0, p - z * se),
                        'upper': min(1, p + z * se)
                    }
                else:
                    mean_value = np.mean(values)
                    se = stats.sem(values)
                    z = stats.norm.ppf(0.975)
                    confidence_interval = {
                        'lower': mean_value - z * se,
                        'upper': mean_value + z * se
                    }
                    
        return ExperimentSummary(
            experiment_id=experiment_id,
            variant_id=variant_id,
            sample_size=len(results),
            metric_values=metric_values,
            conversion_rate=conversion_rate,
            mean_value=mean_value,
            confidence_interval=confidence_interval,
            p_value=p_value
        )
        
    def _declare_winner(self, experiment_id: str):
        """Declare winner for experiment automatically"""
        try:
            self.declare_winner(experiment_id)
        except Exception as e:
            logger.warning(f"Could not declare winner for experiment {experiment_id}: {e}")
            
    def _analysis_loop(self):
        """Main analysis loop"""
        while self._running:
            try:
                # Analyze active experiments
                for experiment_id in list(self.active_experiments):
                    experiment = self.get_experiment(experiment_id)
                    
                    if not experiment:
                        continue
                        
                    # Check if experiment has enough data
                    summary = self.get_experiment_summary(experiment_id)
                    total_sample_size = sum(s.sample_size for s in summary.values())
                    
                    if total_sample_size >= experiment.min_sample_size:
                        # Perform statistical analysis
                        test_results = self.perform_statistical_test(experiment_id)
                        
                        # Check if we have a clear winner
                        significant_results = [
                            (variant_id, results) for variant_id, results in test_results.items()
                            if results['significant'] and results['improvement_percentage'] > experiment.min_detectable_effect * 100
                        ]
                        
                        if significant_results:
                            # Declare winner
                            best_variant_id = max(significant_results, key=lambda x: x[1]['improvement_percentage'])[0]
                            self.declare_winner(experiment_id, best_variant_id)
                            
                # Clean up old data
                self._cleanup_old_data()
                
                # Log statistics
                if self.stats['results_recorded'] > 0 and self.stats['results_recorded'] % 1000 == 0:
                    logger.info(f"A/B testing stats: {self.stats}")
                    
                time.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in analysis loop: {e}")
                time.sleep(60)
                
    def _cleanup_old_data(self):
        """Clean up old data based on retention policy"""
        cutoff_time = datetime.now() - timedelta(days=30)
        
        with self._lock:
            # Clean up old results
            for experiment_id in list(self.experiment_results.keys()):
                self.experiment_results[experiment_id] = [
                    result for result in self.experiment_results[experiment_id]
                    if result.timestamp >= cutoff_time
                ]

# Example usage and testing
def main():
    """Example usage of the A/B testing infrastructure"""
    ab_testing = ABTestingInfrastructure()
    
    try:
        ab_testing.start_infrastructure()
        
        # Create experiment variants
        variant1 = ExperimentVariant(
            variant_id="variant_a",
            name="Model A",
            model_id="bharat-gpt-7b",
            model_version="v1.0",
            traffic_percentage=50.0
        )
        
        variant2 = ExperimentVariant(
            variant_id="variant_b",
            name="Model B",
            model_id="bharat-gpt-7b",
            model_version="v2.0",
            traffic_percentage=50.0
        )
        
        # Create experiment
        experiment = Experiment(
            experiment_id="chat_model_comparison",
            name="Chat Model Comparison",
            description="Compare two versions of chat model",
            variants=[variant1, variant2],
            metrics=ab_testing.default_metrics,
            allocation_strategy=AllocationStrategy.UNIFORM,
            status=ExperimentStatus.DRAFT,
            min_sample_size=1000,
            significance_level=0.05
        )
        
        # Create experiment
        experiment_id = ab_testing.create_experiment(experiment)
        
        # Start experiment
        ab_testing.start_experiment(experiment_id)
        
        # Simulate user assignments and results
        import random
        for i in range(2000):
            user_id = f"user_{i}"
            session_id = f"session_{i}"
            
            # Assign user to variant
            variant_id = ab_testing.assign_user(user_id, session_id, experiment_id)
            
            if variant_id:
                # Simulate result
                conversion = random.random() < 0.1 if variant_id == "variant_a" else random.random() < 0.12
                revenue = random.uniform(0, 100) if conversion else 0
                
                ab_testing.record_result(
                    experiment_id=experiment_id,
                    variant_id=variant_id,
                    user_id=user_id,
                    session_id=session_id,
                    metrics={
                        "conversion_rate": 1.0 if conversion else 0.0,
                        "revenue": revenue
                    }
                )
                
            time.sleep(0.001)  # Small delay
            
        # Get experiment summary
        summary = ab_testing.get_experiment_summary(experiment_id)
        print(f"Experiment summary: {len(summary)} variants")
        
        for variant_id, variant_summary in summary.items():
            print(f"Variant {variant_id}: {variant_summary.sample_size} samples, "
                  f"conversion rate: {variant_summary.conversion_rate:.3f}")
            
        # Perform statistical test
        test_results = ab_testing.perform_statistical_test(experiment_id)
        print(f"Statistical test results: {test_results}")
        
        # Get statistics
        stats = ab_testing.get_statistics()
        print(f"A/B testing statistics: {stats}")
        
        # Export experiment data
        ab_testing.export_experiment_data("ab_testing_data.json")
        
        time.sleep(10)  # Let analysis run
        
    finally:
        ab_testing.stop_infrastructure()

if __name__ == "__main__":
    main()