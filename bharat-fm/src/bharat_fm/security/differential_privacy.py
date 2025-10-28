"""
Differential Privacy Module for Bharat-FM
Implements privacy-preserving data analysis and machine learning
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import math
import random
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)

@dataclass
class PrivacyConfig:
    """Configuration for differential privacy"""
    epsilon: float = 1.0  # Privacy budget
    delta: float = 1e-5   # Privacy failure probability
    sensitivity: float = 1.0  # Global sensitivity
    mechanism: str = "laplace"  # Noise mechanism
    
class PrivacyMechanism(ABC):
    """Abstract base class for privacy mechanisms"""
    
    @abstractmethod
    def add_noise(self, value: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Add privacy-preserving noise to value"""
        pass
    
    @abstractmethod
    def calibrate_sensitivity(self, sensitivity: float) -> float:
        """Calibrate noise scale based on sensitivity"""
        pass

class LaplaceMechanism(PrivacyMechanism):
    """Laplace mechanism for differential privacy"""
    
    def __init__(self, epsilon: float, sensitivity: float = 1.0):
        self.epsilon = epsilon
        self.sensitivity = sensitivity
        self.scale = sensitivity / epsilon
    
    def add_noise(self, value: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Add Laplace noise to value"""
        if isinstance(value, (int, float)):
            noise = np.random.laplace(0, self.scale)
            return value + noise
        elif isinstance(value, np.ndarray):
            noise = np.random.laplace(0, self.scale, size=value.shape)
            return value + noise
        else:
            raise ValueError("Unsupported value type")
    
    def calibrate_sensitivity(self, sensitivity: float) -> float:
        """Calibrate noise scale based on sensitivity"""
        self.sensitivity = sensitivity
        self.scale = sensitivity / self.epsilon
        return self.scale

class GaussianMechanism(PrivacyMechanism):
    """Gaussian mechanism for (ε, δ)-differential privacy"""
    
    def __init__(self, epsilon: float, delta: float, sensitivity: float = 1.0):
        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = sensitivity
        self.scale = self._calculate_scale()
    
    def _calculate_scale(self) -> float:
        """Calculate noise scale for Gaussian mechanism"""
        # σ = sqrt(2 * ln(1.25/δ)) * (sensitivity / ε)
        return math.sqrt(2 * math.log(1.25 / self.delta)) * (self.sensitivity / self.epsilon)
    
    def add_noise(self, value: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Add Gaussian noise to value"""
        if isinstance(value, (int, float)):
            noise = np.random.normal(0, self.scale)
            return value + noise
        elif isinstance(value, np.ndarray):
            noise = np.random.normal(0, self.scale, size=value.shape)
            return value + noise
        else:
            raise ValueError("Unsupported value type")
    
    def calibrate_sensitivity(self, sensitivity: float) -> float:
        """Calibrate noise scale based on sensitivity"""
        self.sensitivity = sensitivity
        self.scale = self._calculate_scale()
        return self.scale

class ExponentialMechanism(PrivacyMechanism):
    """Exponential mechanism for private selection"""
    
    def __init__(self, epsilon: float, sensitivity: float = 1.0):
        self.epsilon = epsilon
        self.sensitivity = sensitivity
    
    def select_private(self, candidates: List[Any], utility_function: Callable[[Any], float]) -> Any:
        """
        Select candidate privately using exponential mechanism
        """
        # Calculate utilities for all candidates
        utilities = [utility_function(candidate) for candidate in candidates]
        
        # Calculate probabilities
        max_utility = max(utilities)
        probabilities = []
        
        for utility in utilities:
            prob = math.exp((self.epsilon * utility) / (2 * self.sensitivity))
            probabilities.append(prob)
        
        # Normalize probabilities
        total_prob = sum(probabilities)
        probabilities = [p / total_prob for p in probabilities]
        
        # Select candidate based on probabilities
        return np.random.choice(candidates, p=probabilities)

class PrivateStatistics:
    """Private statistical computations"""
    
    def __init__(self, config: PrivacyConfig):
        self.config = config
        self.mechanism = self._create_mechanism()
    
    def _create_mechanism(self) -> PrivacyMechanism:
        """Create privacy mechanism based on configuration"""
        if self.config.mechanism == "laplace":
            return LaplaceMechanism(self.config.epsilon, self.config.sensitivity)
        elif self.config.mechanism == "gaussian":
            return GaussianMechanism(self.config.epsilon, self.config.delta, self.config.sensitivity)
        else:
            raise ValueError(f"Unknown mechanism: {self.config.mechanism}")
    
    def private_mean(self, data: np.ndarray) -> float:
        """Compute private mean"""
        if len(data) == 0:
            return 0.0
        
        # Calculate sensitivity for mean (assuming bounded data in [0,1])
        sensitivity = 1.0 / len(data)
        self.mechanism.calibrate_sensitivity(sensitivity)
        
        # Compute true mean and add noise
        true_mean = np.mean(data)
        private_mean = self.mechanism.add_noise(true_mean)
        
        return private_mean
    
    def private_sum(self, data: np.ndarray) -> float:
        """Compute private sum"""
        if len(data) == 0:
            return 0.0
        
        # Sensitivity for sum is the maximum possible change in one element
        sensitivity = 1.0  # Assuming bounded data
        self.mechanism.calibrate_sensitivity(sensitivity)
        
        # Compute true sum and add noise
        true_sum = np.sum(data)
        private_sum = self.mechanism.add_noise(true_sum)
        
        return private_sum
    
    def private_variance(self, data: np.ndarray) -> float:
        """Compute private variance"""
        if len(data) < 2:
            return 0.0
        
        # Use private mean
        private_mean = self.private_mean(data)
        
        # Compute variance with additional noise
        squared_diffs = [(x - private_mean)**2 for x in data]
        sensitivity = 4.0  # Sensitivity for variance calculation
        self.mechanism.calibrate_sensitivity(sensitivity)
        
        true_var = np.mean(squared_diffs)
        private_var = self.mechanism.add_noise(true_var)
        
        return max(0, private_var)  # Ensure non-negative
    
    def private_histogram(self, data: np.ndarray, bins: int = 10) -> np.ndarray:
        """Compute private histogram"""
        if len(data) == 0:
            return np.zeros(bins)
        
        # Compute true histogram
        hist, _ = np.histogram(data, bins=bins)
        
        # Add noise to each bin
        sensitivity = 1.0  # Each data point can change one bin by at most 1
        self.mechanism.calibrate_sensitivity(sensitivity)
        
        private_hist = self.mechanism.add_noise(hist)
        
        # Ensure non-negative counts
        private_hist = np.maximum(private_hist, 0)
        
        return private_hist

class PrivateML:
    """Private machine learning algorithms"""
    
    def __init__(self, config: PrivacyConfig):
        self.config = config
        self.mechanism = LaplaceMechanism(config.epsilon, config.sensitivity)
    
    def private_linear_regression(self, X: np.ndarray, y: np.ndarray, 
                                iterations: int = 1000, learning_rate: float = 0.01) -> np.ndarray:
        """
        Private linear regression using differentially private SGD
        """
        n_samples, n_features = X.shape
        
        # Initialize weights
        weights = np.random.normal(0, 0.1, n_features)
        
        # Calculate sensitivity for gradient
        max_norm = np.max(np.linalg.norm(X, axis=1))
        gradient_sensitivity = 2 * max_norm / n_samples
        
        for i in range(iterations):
            # Compute gradient
            predictions = X @ weights
            errors = predictions - y
            gradient = (X.T @ errors) / n_samples
            
            # Add noise to gradient
            noisy_gradient = self.mechanism.add_noise(gradient)
            
            # Update weights
            weights -= learning_rate * noisy_gradient
            
            # Project weights to feasible set (optional)
            weights = np.clip(weights, -10, 10)
        
        return weights
    
    def private_logistic_regression(self, X: np.ndarray, y: np.ndarray, 
                                  iterations: int = 1000, learning_rate: float = 0.01) -> np.ndarray:
        """
        Private logistic regression using differentially private SGD
        """
        n_samples, n_features = X.shape
        
        # Initialize weights
        weights = np.random.normal(0, 0.1, n_features)
        
        # Calculate sensitivity for gradient
        max_norm = np.max(np.linalg.norm(X, axis=1))
        gradient_sensitivity = 2 * max_norm / n_samples
        
        for i in range(iterations):
            # Compute gradient
            logits = X @ weights
            probs = 1 / (1 + np.exp(-logits))
            errors = probs - y
            gradient = (X.T @ errors) / n_samples
            
            # Add noise to gradient
            noisy_gradient = self.mechanism.add_noise(gradient)
            
            # Update weights
            weights -= learning_rate * noisy_gradient
            
            # Project weights to feasible set
            weights = np.clip(weights, -10, 10)
        
        return weights
    
    def private_kmeans(self, X: np.ndarray, n_clusters: int = 3, 
                      iterations: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Private k-means clustering
        """
        n_samples, n_features = X.shape
        
        # Initialize centroids randomly
        centroids = X[np.random.choice(n_samples, n_clusters, replace=False)]
        
        for i in range(iterations):
            # Assign points to clusters
            distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
            labels = np.argmin(distances, axis=1)
            
            # Update centroids with noise
            new_centroids = []
            for k in range(n_clusters):
                cluster_points = X[labels == k]
                if len(cluster_points) > 0:
                    # Add noise to centroid calculation
                    sensitivity = 2.0 / len(cluster_points)
                    self.mechanism.calibrate_sensitivity(sensitivity)
                    
                    noisy_centroid = self.mechanism.add_noise(np.mean(cluster_points, axis=0))
                    new_centroids.append(noisy_centroid)
                else:
                    new_centroids.append(centroids[k])
            
            centroids = np.array(new_centroids)
        
        return centroids, labels

class PrivacyAccountant:
    """Tracks privacy budget consumption"""
    
    def __init__(self, total_epsilon: float, total_delta: float = 1e-5):
        self.total_epsilon = total_epsilon
        self.total_delta = total_delta
        self.used_epsilon = 0.0
        self.used_delta = 0.0
        self.mechanism_usage = {}
    
    def spend_budget(self, epsilon: float, delta: float = 0.0, mechanism: str = "unknown"):
        """
        Spend privacy budget for a mechanism
        """
        if self.used_epsilon + epsilon > self.total_epsilon:
            raise ValueError(f"Insufficient privacy budget. Required: {epsilon}, Available: {self.total_epsilon - self.used_epsilon}")
        
        if self.used_delta + delta > self.total_delta:
            raise ValueError(f"Insufficient delta budget. Required: {delta}, Available: {self.total_delta - self.used_delta}")
        
        self.used_epsilon += epsilon
        self.used_delta += delta
        
        # Track mechanism usage
        if mechanism not in self.mechanism_usage:
            self.mechanism_usage[mechanism] = {"epsilon": 0.0, "delta": 0.0, "count": 0}
        
        self.mechanism_usage[mechanism]["epsilon"] += epsilon
        self.mechanism_usage[mechanism]["delta"] += delta
        self.mechanism_usage[mechanism]["count"] += 1
    
    def remaining_budget(self) -> Tuple[float, float]:
        """Return remaining privacy budget"""
        return self.total_epsilon - self.used_epsilon, self.total_delta - self.used_delta
    
    def get_usage_summary(self) -> Dict:
        """Get summary of privacy budget usage"""
        return {
            "total_epsilon": self.total_epsilon,
            "total_delta": self.total_delta,
            "used_epsilon": self.used_epsilon,
            "used_delta": self.used_delta,
            "remaining_epsilon": self.total_epsilon - self.used_epsilon,
            "remaining_delta": self.total_delta - self.used_delta,
            "mechanism_usage": self.mechanism_usage
        }

class PrivateDataRelease:
    """Privacy-preserving data release mechanisms"""
    
    def __init__(self, config: PrivacyConfig):
        self.config = config
        self.mechanism = LaplaceMechanism(config.epsilon, config.sensitivity)
    
    def release_private_count(self, data: List[Any]) -> int:
        """Release private count of data points"""
        true_count = len(data)
        sensitivity = 1.0
        self.mechanism.calibrate_sensitivity(sensitivity)
        
        private_count = self.mechanism.add_noise(true_count)
        return max(0, int(round(private_count)))
    
    def release_private_frequency(self, data: List[Any]) -> Dict[Any, int]:
        """Release private frequency counts"""
        from collections import Counter
        
        # Count frequencies
        counter = Counter(data)
        unique_items = list(counter.keys())
        
        # Add noise to each count
        sensitivity = 1.0
        self.mechanism.calibrate_sensitivity(sensitivity)
        
        private_counts = {}
        for item in unique_items:
            noisy_count = self.mechanism.add_noise(counter[item])
            private_counts[item] = max(0, int(round(noisy_count)))
        
        return private_counts
    
    def release_private_range_query(self, data: List[float], 
                                   min_val: float, max_val: float) -> int:
        """Release private range query result"""
        true_count = sum(1 for x in data if min_val <= x <= max_val)
        sensitivity = 1.0
        self.mechanism.calibrate_sensitivity(sensitivity)
        
        private_count = self.mechanism.add_noise(true_count)
        return max(0, int(round(private_count)))

# Factory functions
def create_privacy_mechanism(config: PrivacyConfig) -> PrivacyMechanism:
    """Create privacy mechanism based on configuration"""
    if config.mechanism == "laplace":
        return LaplaceMechanism(config.epsilon, config.sensitivity)
    elif config.mechanism == "gaussian":
        return GaussianMechanism(config.epsilon, config.delta, config.sensitivity)
    elif config.mechanism == "exponential":
        return ExponentialMechanism(config.epsilon, config.sensitivity)
    else:
        raise ValueError(f"Unknown mechanism: {config.mechanism}")

def create_private_statistics(epsilon: float, delta: float = 1e-5, 
                             sensitivity: float = 1.0, mechanism: str = "laplace") -> PrivateStatistics:
    """Create private statistics calculator"""
    config = PrivacyConfig(epsilon=epsilon, delta=delta, 
                          sensitivity=sensitivity, mechanism=mechanism)
    return PrivateStatistics(config)

# Example usage and testing
def test_differential_privacy():
    """Test differential privacy functionality"""
    print("Testing Differential Privacy...")
    
    # Create privacy configuration
    config = PrivacyConfig(epsilon=1.0, delta=1e-5, sensitivity=1.0)
    
    # Test privacy mechanisms
    print("Testing privacy mechanisms...")
    
    # Laplace mechanism
    laplace = LaplaceMechanism(epsilon=1.0, sensitivity=1.0)
    value = 10.0
    noisy_value = laplace.add_noise(value)
    print(f"Laplace: {value} -> {noisy_value}")
    
    # Gaussian mechanism
    gaussian = GaussianMechanism(epsilon=1.0, delta=1e-5, sensitivity=1.0)
    noisy_value = gaussian.add_noise(value)
    print(f"Gaussian: {value} -> {noisy_value}")
    
    # Test private statistics
    print("Testing private statistics...")
    private_stats = create_private_statistics(epsilon=0.5)
    
    # Generate test data
    np.random.seed(42)
    data = np.random.normal(5, 2, 1000)
    
    # Compute private statistics
    private_mean = private_stats.private_mean(data)
    private_var = private_stats.private_variance(data)
    private_sum = private_stats.private_sum(data)
    
    print(f"True mean: {np.mean(data):.3f}, Private mean: {private_mean:.3f}")
    print(f"True variance: {np.var(data):.3f}, Private variance: {private_var:.3f}")
    print(f"True sum: {np.sum(data):.3f}, Private sum: {private_sum:.3f}")
    
    # Test private machine learning
    print("Testing private machine learning...")
    private_ml = PrivateML(config)
    
    # Generate synthetic data
    X = np.random.randn(100, 3)
    true_weights = np.array([1.5, -2.0, 0.5])
    y = X @ true_weights + np.random.normal(0, 0.1, 100)
    
    # Private linear regression
    private_weights = private_ml.private_linear_regression(X, y, iterations=500)
    print(f"True weights: {true_weights}")
    print(f"Private weights: {private_weights}")
    
    # Test privacy accountant
    print("Testing privacy accountant...")
    accountant = PrivacyAccountant(total_epsilon=2.0, total_delta=1e-5)
    
    # Spend budget on different operations
    accountant.spend_budget(epsilon=0.5, mechanism="mean")
    accountant.spend_budget(epsilon=0.3, mechanism="variance")
    accountant.spend_budget(epsilon=0.7, mechanism="regression")
    
    summary = accountant.get_usage_summary()
    print(f"Budget usage: {summary['used_epsilon']:.2f}/{summary['total_epsilon']}")
    print(f"Remaining budget: {summary['remaining_epsilon']:.2f}")
    
    # Test private data release
    print("Testing private data release...")
    private_release = PrivateDataRelease(config)
    
    categorical_data = ['A', 'B', 'A', 'C', 'B', 'A', 'A', 'B', 'C', 'C']
    private_freq = private_release.release_private_frequency(categorical_data)
    print(f"Private frequencies: {private_freq}")
    
    numerical_data = [1.2, 3.4, 2.1, 4.5, 3.2, 1.8, 2.9, 3.7]
    private_range = private_release.release_private_range_query(numerical_data, 2.0, 4.0)
    print(f"Private range count (2.0-4.0): {private_range}")
    
    print("Differential privacy tests completed!")

if __name__ == "__main__":
    test_differential_privacy()