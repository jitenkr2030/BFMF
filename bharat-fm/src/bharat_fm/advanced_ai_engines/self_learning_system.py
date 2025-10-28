"""
Self-Learning and Adaptation System for Bharat-FM MLOps Platform

This module implements a sophisticated self-learning system that enables the AI to:
- Continuously learn from interactions and experiences
- Adapt behavior based on feedback and outcomes
- Optimize performance through self-improvement
- Develop personalized interaction strategies
- Evolve capabilities over time

Author: Advanced AI Team
Version: 2.0.0
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from collections import defaultdict, deque
import json
import time
from datetime import datetime, timedelta
import pickle
import threading
from concurrent.futures import ThreadPoolExecutor
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import networkx as nx
from pathlib import Path
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LearningType(Enum):
    """Types of learning mechanisms"""
    SUPERVISED = "supervised"  # Learning from labeled examples
    UNSUPERVISED = "unsupervised"  # Learning from unlabeled data
    REINFORCEMENT = "reinforcement"  # Learning from rewards/penalties
    TRANSFER = "transfer"  # Transfer learning between domains
    META = "meta"  # Learning to learn
    LIFELONG = "lifelong"  # Continuous learning over time
    ADAPTIVE = "adaptive"  # Adaptive learning strategies

class AdaptationStrategy(Enum):
    """Adaptation strategies for self-improvement"""
    GRADUAL = "gradual"  # Slow, steady adaptation
    RAPID = "rapid"  # Fast adaptation to changes
    CONSERVATIVE = "conservative"  # Cautious adaptation
    AGGRESSIVE = "aggressive"  # Bold adaptation
    BALANCED = "balanced"  # Balanced adaptation approach

@dataclass
class LearningExperience:
    """Represents a single learning experience"""
    experience_id: str
    timestamp: datetime
    input_data: Any
    output_data: Any
    feedback: Optional[float] = None
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    learning_type: LearningType = LearningType.UNSUPERVISED

@dataclass
class LearningPattern:
    """Represents a discovered learning pattern"""
    pattern_id: str
    pattern_type: LearningType
    features: List[str]
    pattern_data: Any
    confidence: float
    frequency: int
    last_observed: datetime
    utility_score: float = 0.0

@dataclass
class AdaptationEvent:
    """Represents an adaptation event"""
    event_id: str
    timestamp: datetime
    adaptation_type: str
    old_state: Dict[str, Any]
    new_state: Dict[str, Any]
    performance_change: float
    confidence: float
    rationale: str

class SelfLearningSystem:
    """
    Advanced self-learning system with continuous adaptation capabilities
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Learning parameters
        self.learning_rate = self.config.get('learning_rate', 0.01)
        self.adaptation_rate = self.config.get('adaptation_rate', 0.1)
        self.experience_buffer_size = self.config.get('experience_buffer_size', 10000)
        self.pattern_threshold = self.config.get('pattern_threshold', 0.7)
        self.adaptation_threshold = self.config.get('adaptation_threshold', 0.05)
        
        # Data structures
        self.experiences = deque(maxlen=self.experience_buffer_size)
        self.learning_patterns = {}
        self.adaptation_history = deque(maxlen=1000)
        self.performance_metrics = defaultdict(list)
        self.knowledge_base = {}
        
        # Learning models
        self.neural_network = self._create_neural_network()
        self.optimizer = optim.Adam(self.neural_network.parameters(), lr=self.learning_rate)
        self.clustering_model = KMeans(n_clusters=5, random_state=42)
        
        # Adaptation state
        self.current_strategy = AdaptationStrategy.BALANCED
        self.adaptation_counter = 0
        self.last_adaptation = datetime.now()
        self.learning_velocity = 0.0
        
        # Threading for continuous learning
        self.learning_thread = None
        self.is_learning = False
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Initialize learning components
        self._initialize_learning_components()
        
        logger.info("Self-Learning System initialized successfully")
    
    def _create_neural_network(self) -> nn.Module:
        """Create neural network for learning"""
        class LearningNetwork(nn.Module):
            def __init__(self, input_size=128, hidden_size=256, output_size=64):
                super().__init__()
                self.layers = nn.Sequential(
                    nn.Linear(input_size, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(hidden_size, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(hidden_size, output_size),
                    nn.ReLU()
                )
            
            def forward(self, x):
                return self.layers(x)
        
        return LearningNetwork()
    
    def _initialize_learning_components(self):
        """Initialize learning components and start background processes"""
        # Load previous learning state if available
        self._load_learning_state()
        
        # Start continuous learning thread
        self.start_continuous_learning()
        
        # Initialize performance tracking
        self._initialize_performance_tracking()
    
    def add_experience(self, experience: LearningExperience):
        """Add a new learning experience"""
        self.experiences.append(experience)
        
        # Trigger immediate learning if significant experience
        if experience.feedback and abs(experience.feedback) > 0.5:
            self._trigger_immediate_learning(experience)
        
        logger.debug(f"Added learning experience: {experience.experience_id}")
    
    def learn_from_interaction(self, input_data: Any, output_data: Any, 
                             feedback: Optional[float] = None, 
                             context: Optional[Dict[str, Any]] = None) -> str:
        """
        Learn from a single interaction
        
        Args:
            input_data: Input to the system
            output_data: Output generated by the system
            feedback: Feedback score (-1 to 1)
            context: Context information
            
        Returns:
            str: Experience ID
        """
        experience_id = f"exp_{int(time.time())}_{hashlib.md5(str(input_data).encode()).hexdigest()[:8]}"
        
        experience = LearningExperience(
            experience_id=experience_id,
            timestamp=datetime.now(),
            input_data=input_data,
            output_data=output_data,
            feedback=feedback,
            context=context or {},
            learning_type=LearningType.REINFORCEMENT if feedback else LearningType.UNSUPERVISED
        )
        
        self.add_experience(experience)
        return experience_id
    
    def discover_patterns(self) -> List[LearningPattern]:
        """Discover patterns in learning experiences"""
        if len(self.experiences) < 10:
            return []
        
        patterns = []
        
        try:
            # Extract features from experiences
            features = self._extract_experience_features()
            
            if len(features) > 0:
                # Cluster experiences to find patterns
                cluster_labels = self.clustering_model.fit_predict(features)
                
                # Analyze clusters for patterns
                for cluster_id in np.unique(cluster_labels):
                    cluster_experiences = [self.experiences[i] for i, label in enumerate(cluster_labels) if label == cluster_id]
                    
                    if len(cluster_experiences) >= 3:  # Minimum cluster size
                        pattern = self._analyze_cluster_pattern(cluster_id, cluster_experiences)
                        if pattern.confidence > self.pattern_threshold:
                            patterns.append(pattern)
                
                # Update patterns database
                self._update_patterns_database(patterns)
                
        except Exception as e:
            logger.error(f"Error in pattern discovery: {e}")
        
        return patterns
    
    def adapt_behavior(self, performance_change: float, context: Optional[Dict[str, Any]] = None):
        """
        Adapt system behavior based on performance feedback
        
        Args:
            performance_change: Change in performance (-1 to 1)
            context: Context for adaptation
        """
        if abs(performance_change) < self.adaptation_threshold:
            return
        
        # Determine adaptation strategy
        strategy = self._select_adaptation_strategy(performance_change, context)
        
        # Create adaptation event
        event_id = f"adapt_{int(time.time())}"
        old_state = self._get_current_state()
        
        # Apply adaptation
        new_state = self._apply_adaptation(strategy, performance_change, context)
        
        # Calculate actual performance change
        actual_change = self._calculate_performance_change(old_state, new_state)
        
        # Record adaptation event
        adaptation_event = AdaptationEvent(
            event_id=event_id,
            timestamp=datetime.now(),
            adaptation_type=strategy.value,
            old_state=old_state,
            new_state=new_state,
            performance_change=actual_change,
            confidence=min(abs(performance_change), 1.0),
            rationale=self._generate_adaptation_rationale(strategy, performance_change)
        )
        
        self.adaptation_history.append(adaptation_event)
        self.adaptation_counter += 1
        self.last_adaptation = datetime.now()
        
        logger.info(f"Applied adaptation: {strategy.value} with change {actual_change:.3f}")
    
    def optimize_performance(self) -> Dict[str, Any]:
        """
        Optimize system performance through self-improvement
        
        Returns:
            Dict: Optimization results
        """
        optimization_results = {
            'parameters_optimized': [],
            'performance_improvement': 0.0,
            'adaptations_made': [],
            'new_patterns_discovered': []
        }
        
        try:
            # Analyze current performance
            current_performance = self._analyze_current_performance()
            
            # Discover new patterns
            new_patterns = self.discover_patterns()
            optimization_results['new_patterns_discovered'] = [p.pattern_id for p in new_patterns]
            
            # Optimize neural network
            nn_improvement = self._optimize_neural_network()
            if nn_improvement > 0:
                optimization_results['performance_improvement'] += nn_improvement
                optimization_results['adaptations_made'].append('neural_network_optimization')
            
            # Adapt learning parameters
            param_adaptations = self._adapt_learning_parameters(current_performance)
            optimization_results['parameters_optimized'].extend(param_adaptations)
            
            # Update knowledge base
            kb_updates = self._update_knowledge_base()
            optimization_results['adaptations_made'].extend(kb_updates)
            
            logger.info(f"Performance optimization completed: {optimization_results['performance_improvement']:.3f} improvement")
            
        except Exception as e:
            logger.error(f"Error in performance optimization: {e}")
        
        return optimization_results
    
    def start_continuous_learning(self):
        """Start continuous learning in background thread"""
        if not self.is_learning:
            self.is_learning = True
            self.learning_thread = threading.Thread(target=self._continuous_learning_loop)
            self.learning_thread.daemon = True
            self.learning_thread.start()
            logger.info("Continuous learning started")
    
    def stop_continuous_learning(self):
        """Stop continuous learning"""
        self.is_learning = False
        if self.learning_thread:
            self.learning_thread.join()
        logger.info("Continuous learning stopped")
    
    def _continuous_learning_loop(self):
        """Background loop for continuous learning"""
        while self.is_learning:
            try:
                # Process experiences
                if len(self.experiences) > 0:
                    self._process_experiences()
                
                # Discover patterns periodically
                if self.adaptation_counter % 10 == 0:
                    self.discover_patterns()
                
                # Optimize performance periodically
                if self.adaptation_counter % 50 == 0:
                    self.optimize_performance()
                
                # Save learning state periodically
                if self.adaptation_counter % 100 == 0:
                    self._save_learning_state()
                
                # Sleep to prevent excessive CPU usage
                time.sleep(5)
                
            except Exception as e:
                logger.error(f"Error in continuous learning loop: {e}")
                time.sleep(10)
    
    def _process_experiences(self):
        """Process accumulated experiences for learning"""
        if len(self.experiences) < 5:
            return
        
        # Get recent experiences
        recent_experiences = list(self.experiences)[-100:]
        
        # Separate supervised and reinforcement learning experiences
        supervised_experiences = [exp for exp in recent_experiences if exp.feedback is not None]
        unsupervised_experiences = [exp for exp in recent_experiences if exp.feedback is None]
        
        # Process supervised learning
        if supervised_experiences:
            self._process_supervised_learning(supervised_experiences)
        
        # Process unsupervised learning
        if unsupervised_experiences:
            self._process_unsupervised_learning(unsupervised_experiences)
    
    def _process_supervised_learning(self, experiences: List[LearningExperience]):
        """Process supervised learning experiences"""
        try:
            # Prepare training data
            inputs = []
            targets = []
            
            for exp in experiences:
                if exp.feedback is not None:
                    # Convert experience to tensor
                    input_tensor = self._experience_to_tensor(exp)
                    target_tensor = torch.tensor([exp.feedback], dtype=torch.float32)
                    
                    inputs.append(input_tensor)
                    targets.append(target_tensor)
            
            if len(inputs) > 0:
                # Train neural network
                inputs = torch.stack(inputs)
                targets = torch.stack(targets)
                
                self.optimizer.zero_grad()
                outputs = self.neural_network(inputs)
                loss = nn.MSELoss()(outputs, targets)
                loss.backward()
                self.optimizer.step()
                
                logger.debug(f"Supervised learning completed with loss: {loss.item():.4f}")
                
        except Exception as e:
            logger.error(f"Error in supervised learning: {e}")
    
    def _process_unsupervised_learning(self, experiences: List[LearningExperience]):
        """Process unsupervised learning experiences"""
        try:
            # Extract features for clustering
            features = []
            for exp in experiences:
                feature_vector = self._extract_experience_features(exp)
                if len(feature_vector) > 0:
                    features.append(feature_vector)
            
            if len(features) > 0:
                features = np.array(features)
                
                # Update clustering model
                if len(features) >= self.clustering_model.n_clusters:
                    self.clustering_model.fit(features)
                    
                    # Calculate clustering quality
                    if len(features) > self.clustering_model.n_clusters:
                        silhouette_avg = silhouette_score(features, self.clustering_model.labels_)
                        logger.debug(f"Unsupervised learning completed with silhouette score: {silhouette_avg:.3f}")
                
        except Exception as e:
            logger.error(f"Error in unsupervised learning: {e}")
    
    def _extract_experience_features(self, experiences: Optional[List[LearningExperience]] = None) -> np.ndarray:
        """Extract features from experiences"""
        if experiences is None:
            experiences = list(self.experiences)
        
        features = []
        
        for exp in experiences:
            try:
                # Create feature vector from experience
                feature_vector = []
                
                # Time-based features
                time_diff = (datetime.now() - exp.timestamp).total_seconds()
                feature_vector.append(time_diff / 3600)  # Hours ago
                
                # Feedback features
                feature_vector.append(exp.feedback if exp.feedback is not None else 0.0)
                
                # Context features
                context_features = self._extract_context_features(exp.context)
                feature_vector.extend(context_features)
                
                # Input/output features
                input_features = self._extract_data_features(exp.input_data)
                output_features = self._extract_data_features(exp.output_data)
                feature_vector.extend(input_features)
                feature_vector.extend(output_features)
                
                # Pad to fixed length
                while len(feature_vector) < 128:
                    feature_vector.append(0.0)
                
                feature_vector = feature_vector[:128]  # Truncate to fixed length
                
                features.append(feature_vector)
                
            except Exception as e:
                logger.debug(f"Error extracting features from experience {exp.experience_id}: {e}")
        
        return np.array(features) if features else np.array([])
    
    def _extract_context_features(self, context: Dict[str, Any]) -> List[float]:
        """Extract features from context"""
        features = []
        
        # Simple context feature extraction
        features.append(len(context))
        features.append(float('user_id' in context))
        features.append(float('session_id' in context))
        features.append(float('task_type' in context))
        
        # Add numeric context values
        for key, value in context.items():
            if isinstance(value, (int, float)):
                features.append(float(value))
        
        # Pad to fixed length
        while len(features) < 10:
            features.append(0.0)
        
        return features[:10]
    
    def _extract_data_features(self, data: Any) -> List[float]:
        """Extract features from input/output data"""
        features = []
        
        try:
            if isinstance(data, str):
                # Text features
                features.append(len(data))
                features.append(len(data.split()))
                features.append(float('?' in data))
                features.append(float('!' in data))
            elif isinstance(data, (list, dict)):
                # Structure features
                features.append(len(data))
                features.append(float(bool(data)))
            elif isinstance(data, (int, float)):
                # Numeric features
                features.append(float(data))
                features.append(abs(data))
            else:
                # Generic features
                features.append(0.0)
                features.append(0.0)
                
        except Exception:
            features.extend([0.0, 0.0])
        
        return features
    
    def _experience_to_tensor(self, experience: LearningExperience) -> torch.Tensor:
        """Convert experience to tensor for neural network"""
        features = self._extract_experience_features([experience])[0]
        return torch.tensor(features, dtype=torch.float32)
    
    def _analyze_cluster_pattern(self, cluster_id: int, experiences: List[LearningExperience]) -> LearningPattern:
        """Analyze a cluster to identify patterns"""
        pattern_id = f"pattern_{cluster_id}_{int(time.time())}"
        
        # Calculate pattern statistics
        confidences = [exp.feedback for exp in experiences if exp.feedback is not None]
        avg_confidence = np.mean(confidences) if confidences else 0.5
        
        # Extract common features
        common_features = self._extract_common_features(experiences)
        
        # Calculate utility score
        utility_score = self._calculate_pattern_utility(experiences, avg_confidence)
        
        return LearningPattern(
            pattern_id=pattern_id,
            pattern_type=LearningType.UNSUPERVISED,
            features=common_features,
            pattern_data={'cluster_id': cluster_id, 'experience_count': len(experiences)},
            confidence=avg_confidence,
            frequency=len(experiences),
            last_observed=datetime.now(),
            utility_score=utility_score
        )
    
    def _extract_common_features(self, experiences: List[LearningExperience]) -> List[str]:
        """Extract common features from experiences"""
        common_features = []
        
        # Analyze common context keys
        context_keys = set()
        for exp in experiences:
            context_keys.update(exp.context.keys())
        
        for key in context_keys:
            # Check if key is present in most experiences
            presence_count = sum(1 for exp in experiences if key in exp.context)
            if presence_count / len(experiences) > 0.7:
                common_features.append(f"context_{key}")
        
        return common_features[:10]  # Limit to top 10 features
    
    def _calculate_pattern_utility(self, experiences: List[LearningExperience], confidence: float) -> float:
        """Calculate utility score for a pattern"""
        # Utility based on frequency, confidence, and recency
        frequency_score = min(len(experiences) / 10.0, 1.0)
        confidence_score = confidence
        
        # Recency score
        most_recent = max(exp.timestamp for exp in experiences)
        recency_score = max(0, 1 - (datetime.now() - most_recent).total_seconds() / (30 * 24 * 3600))
        
        # Combined utility
        utility = (frequency_score * 0.4 + confidence_score * 0.4 + recency_score * 0.2)
        
        return utility
    
    def _update_patterns_database(self, new_patterns: List[LearningPattern]):
        """Update patterns database with new discoveries"""
        for pattern in new_patterns:
            # Check if similar pattern exists
            existing_pattern = self._find_similar_pattern(pattern)
            
            if existing_pattern:
                # Update existing pattern
                existing_pattern.frequency += pattern.frequency
                existing_pattern.confidence = (existing_pattern.confidence + pattern.confidence) / 2
                existing_pattern.last_observed = pattern.last_observed
                existing_pattern.utility_score = (existing_pattern.utility_score + pattern.utility_score) / 2
            else:
                # Add new pattern
                self.learning_patterns[pattern.pattern_id] = pattern
        
        # Remove old patterns
        self._cleanup_old_patterns()
    
    def _find_similar_pattern(self, new_pattern: LearningPattern) -> Optional[LearningPattern]:
        """Find similar pattern in database"""
        for pattern in self.learning_patterns.values():
            if (pattern.pattern_type == new_pattern.pattern_type and 
                set(pattern.features) == set(new_pattern.features)):
                return pattern
        return None
    
    def _cleanup_old_patterns(self):
        """Remove old or low-utility patterns"""
        current_time = datetime.now()
        patterns_to_remove = []
        
        for pattern_id, pattern in self.learning_patterns.items():
            # Remove patterns older than 30 days with low utility
            age_days = (current_time - pattern.last_observed).total_seconds() / (24 * 3600)
            if age_days > 30 and pattern.utility_score < 0.3:
                patterns_to_remove.append(pattern_id)
        
        for pattern_id in patterns_to_remove:
            del self.learning_patterns[pattern_id]
        
        logger.debug(f"Cleaned up {len(patterns_to_remove)} old patterns")
    
    def _select_adaptation_strategy(self, performance_change: float, 
                                   context: Optional[Dict[str, Any]] = None) -> AdaptationStrategy:
        """Select adaptation strategy based on performance change and context"""
        if context and context.get('urgent', False):
            return AdaptationStrategy.RAPID
        
        if performance_change > 0.5:
            return AdaptationStrategy.AGGRESSIVE
        elif performance_change < -0.5:
            return AdaptationStrategy.CONSERVATIVE
        elif abs(performance_change) < 0.2:
            return AdaptationStrategy.GRADUAL
        else:
            return AdaptationStrategy.BALANCED
    
    def _get_current_state(self) -> Dict[str, Any]:
        """Get current system state"""
        return {
            'learning_rate': self.learning_rate,
            'adaptation_rate': self.adaptation_rate,
            'current_strategy': self.current_strategy.value,
            'adaptation_counter': self.adaptation_counter,
            'learning_velocity': self.learning_velocity,
            'pattern_count': len(self.learning_patterns),
            'experience_count': len(self.experiences)
        }
    
    def _apply_adaptation(self, strategy: AdaptationStrategy, performance_change: float,
                         context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Apply adaptation strategy"""
        new_state = self._get_current_state()
        
        if strategy == AdaptationStrategy.GRADUAL:
            self.learning_rate *= (1 + 0.01 * np.sign(performance_change))
            self.adaptation_rate *= (1 + 0.005 * np.sign(performance_change))
        
        elif strategy == AdaptationStrategy.RAPID:
            self.learning_rate *= (1 + 0.1 * np.sign(performance_change))
            self.adaptation_rate *= (1 + 0.05 * np.sign(performance_change))
        
        elif strategy == AdaptationStrategy.CONSERVATIVE:
            self.learning_rate *= 0.95
            self.adaptation_rate *= 0.98
        
        elif strategy == AdaptationStrategy.AGGRESSIVE:
            self.learning_rate *= 1.2
            self.adaptation_rate *= 1.1
        
        elif strategy == AdaptationStrategy.BALANCED:
            self.learning_rate *= (1 + 0.05 * np.sign(performance_change))
            self.adaptation_rate *= (1 + 0.025 * np.sign(performance_change))
        
        # Keep parameters in reasonable bounds
        self.learning_rate = np.clip(self.learning_rate, 0.001, 0.1)
        self.adaptation_rate = np.clip(self.adaptation_rate, 0.01, 0.5)
        
        self.current_strategy = strategy
        
        return self._get_current_state()
    
    def _calculate_performance_change(self, old_state: Dict[str, Any], 
                                    new_state: Dict[str, Any]) -> float:
        """Calculate actual performance change"""
        # Simplified performance calculation
        old_score = old_state.get('learning_velocity', 0.0)
        new_score = new_state.get('learning_velocity', 0.0)
        
        return new_score - old_score
    
    def _generate_adaptation_rationale(self, strategy: AdaptationStrategy, 
                                     performance_change: float) -> str:
        """Generate rationale for adaptation"""
        rationales = {
            AdaptationStrategy.GRADUAL: f"Gradual adaptation due to moderate performance change ({performance_change:.3f})",
            AdaptationStrategy.RAPID: f"Rapid adaptation required for significant performance change ({performance_change:.3f})",
            AdaptationStrategy.CONSERVATIVE: f"Conservative approach due to negative performance change ({performance_change:.3f})",
            AdaptationStrategy.AGGRESSIVE: f"Aggressive adaptation to capitalize on positive performance change ({performance_change:.3f})",
            AdaptationStrategy.BALANCED: f"Balanced adaptation for steady performance change ({performance_change:.3f})"
        }
        
        return rationales.get(strategy, "Adaptation based on performance feedback")
    
    def _trigger_immediate_learning(self, experience: LearningExperience):
        """Trigger immediate learning for significant experiences"""
        try:
            # Process this experience immediately
            if experience.feedback is not None:
                self._process_supervised_learning([experience])
            else:
                self._process_unsupervised_learning([experience])
            
            logger.debug(f"Immediate learning triggered for experience {experience.experience_id}")
            
        except Exception as e:
            logger.error(f"Error in immediate learning: {e}")
    
    def _optimize_neural_network(self) -> float:
        """Optimize neural network parameters"""
        try:
            if len(self.experiences) < 10:
                return 0.0
            
            # Prepare training data
            features = self._extract_experience_features()
            if len(features) < 10:
                return 0.0
            
            # Train for a few epochs
            inputs = torch.tensor(features, dtype=torch.float32)
            
            # Simple autoencoder training for representation learning
            self.optimizer.zero_grad()
            outputs = self.neural_network(inputs)
            loss = nn.MSELoss()(outputs, inputs)  # Reconstruction loss
            loss.backward()
            self.optimizer.step()
            
            # Return improvement (negative loss is improvement)
            return -loss.item()
            
        except Exception as e:
            logger.error(f"Error in neural network optimization: {e}")
            return 0.0
    
    def _adapt_learning_parameters(self, performance_data: Dict[str, Any]) -> List[str]:
        """Adapt learning parameters based on performance"""
        adaptations = []
        
        # Adapt learning rate based on performance
        if performance_data.get('recent_performance', 0) < 0.5:
            self.learning_rate *= 0.9
            adaptations.append('learning_rate_decreased')
        elif performance_data.get('recent_performance', 0) > 0.8:
            self.learning_rate *= 1.1
            adaptations.append('learning_rate_increased')
        
        # Adapt adaptation rate
        if len(self.adaptation_history) > 10:
            recent_adaptations = list(self.adaptation_history)[-10:]
            avg_performance_change = np.mean([adapt.performance_change for adapt in recent_adaptations])
            
            if avg_performance_change < 0:
                self.adaptation_rate *= 0.95
                adaptations.append('adaptation_rate_decreased')
            else:
                self.adaptation_rate *= 1.05
                adaptations.append('adaptation_rate_increased')
        
        return adaptations
    
    def _update_knowledge_base(self) -> List[str]:
        """Update knowledge base with new insights"""
        updates = []
        
        # Add new patterns to knowledge base
        for pattern in self.learning_patterns.values():
            if pattern.utility_score > 0.7 and pattern.pattern_id not in self.knowledge_base:
                self.knowledge_base[pattern.pattern_id] = {
                    'features': pattern.features,
                    'confidence': pattern.confidence,
                    'utility': pattern.utility_score,
                    'last_updated': datetime.now().isoformat()
                }
                updates.append(f"knowledge_added_{pattern.pattern_id}")
        
        # Remove outdated knowledge
        knowledge_to_remove = []
        for kb_id, kb_data in self.knowledge_base.items():
            last_updated = datetime.fromisoformat(kb_data['last_updated'])
            age_days = (datetime.now() - last_updated).total_seconds() / (24 * 3600)
            
            if age_days > 60 and kb_data['utility'] < 0.3:
                knowledge_to_remove.append(kb_id)
        
        for kb_id in knowledge_to_remove:
            del self.knowledge_base[kb_id]
            updates.append(f"knowledge_removed_{kb_id}")
        
        return updates
    
    def _analyze_current_performance(self) -> Dict[str, Any]:
        """Analyze current system performance"""
        performance = {
            'recent_performance': 0.0,
            'learning_velocity': 0.0,
            'adaptation_frequency': 0.0,
            'pattern_discovery_rate': 0.0
        }
        
        try:
            # Calculate recent performance from adaptation history
            if len(self.adaptation_history) > 0:
                recent_adaptations = list(self.adaptation_history)[-10:]
                performance['recent_performance'] = np.mean([adapt.performance_change for adapt in recent_adaptations])
            
            # Calculate learning velocity
            if len(self.experiences) > 1:
                recent_experiences = list(self.experiences)[-100:]
                feedback_values = [exp.feedback for exp in recent_experiences if exp.feedback is not None]
                if feedback_values:
                    performance['learning_velocity'] = np.mean(feedback_values)
            
            # Calculate adaptation frequency
            if len(self.adaptation_history) > 0:
                time_span = (datetime.now() - self.adaptation_history[0].timestamp).total_seconds()
                performance['adaptation_frequency'] = len(self.adaptation_history) / max(time_span, 1)
            
            # Calculate pattern discovery rate
            recent_patterns = [p for p in self.learning_patterns.values() 
                             if (datetime.now() - p.last_observed).total_seconds() < 24 * 3600]
            performance['pattern_discovery_rate'] = len(recent_patterns)
            
        except Exception as e:
            logger.error(f"Error analyzing performance: {e}")
        
        return performance
    
    def _initialize_performance_tracking(self):
        """Initialize performance tracking"""
        self.performance_metrics = defaultdict(list)
        
        # Track key metrics
        metrics_to_track = [
            'learning_rate',
            'adaptation_rate',
            'experience_count',
            'pattern_count',
            'adaptation_count',
            'performance_score'
        ]
        
        for metric in metrics_to_track:
            self.performance_metrics[metric] = deque(maxlen=1000)
    
    def _save_learning_state(self):
        """Save learning state to file"""
        try:
            state = {
                'learning_rate': self.learning_rate,
                'adaptation_rate': self.adaptation_rate,
                'current_strategy': self.current_strategy.value,
                'adaptation_counter': self.adaptation_counter,
                'learning_velocity': self.learning_velocity,
                'learning_patterns': {
                    pid: {
                        'pattern_type': p.pattern_type.value,
                        'features': p.features,
                        'confidence': p.confidence,
                        'frequency': p.frequency,
                        'last_observed': p.last_observed.isoformat(),
                        'utility_score': p.utility_score
                    }
                    for pid, p in self.learning_patterns.items()
                },
                'knowledge_base': self.knowledge_base,
                'performance_metrics': {
                    key: list(values) for key, values in self.performance_metrics.items()
                }
            }
            
            # Save neural network state
            torch.save(self.neural_network.state_dict(), 'neural_network_state.pth')
            
            # Save main state
            with open('learning_system_state.json', 'w') as f:
                json.dump(state, f, indent=2)
            
            logger.info("Learning state saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving learning state: {e}")
    
    def _load_learning_state(self):
        """Load learning state from file"""
        try:
            # Check if state files exist
            if Path('learning_system_state.json').exists():
                with open('learning_system_state.json', 'r') as f:
                    state = json.load(f)
                
                # Load parameters
                self.learning_rate = state.get('learning_rate', 0.01)
                self.adaptation_rate = state.get('adaptation_rate', 0.1)
                self.current_strategy = AdaptationStrategy(state.get('current_strategy', 'balanced'))
                self.adaptation_counter = state.get('adaptation_counter', 0)
                self.learning_velocity = state.get('learning_velocity', 0.0)
                
                # Load patterns
                self.learning_patterns = {}
                for pid, p_data in state.get('learning_patterns', {}).items():
                    pattern = LearningPattern(
                        pattern_id=pid,
                        pattern_type=LearningType(p_data['pattern_type']),
                        features=p_data['features'],
                        pattern_data={},
                        confidence=p_data['confidence'],
                        frequency=p_data['frequency'],
                        last_observed=datetime.fromisoformat(p_data['last_observed']),
                        utility_score=p_data['utility_score']
                    )
                    self.learning_patterns[pid] = pattern
                
                # Load knowledge base
                self.knowledge_base = state.get('knowledge_base', {})
                
                # Load performance metrics
                self.performance_metrics = defaultdict(list)
                for key, values in state.get('performance_metrics', {}).items():
                    self.performance_metrics[key] = deque(values, maxlen=1000)
                
                logger.info("Learning state loaded successfully")
            
            # Load neural network state
            if Path('neural_network_state.pth').exists():
                self.neural_network.load_state_dict(torch.load('neural_network_state.pth'))
                logger.info("Neural network state loaded successfully")
                
        except Exception as e:
            logger.error(f"Error loading learning state: {e}")
    
    def get_learning_statistics(self) -> Dict[str, Any]:
        """Get comprehensive learning statistics"""
        return {
            'experiences_processed': len(self.experiences),
            'patterns_discovered': len(self.learning_patterns),
            'adaptations_made': len(self.adaptation_history),
            'knowledge_base_size': len(self.knowledge_base),
            'current_learning_rate': self.learning_rate,
            'current_adaptation_rate': self.adaptation_rate,
            'current_strategy': self.current_strategy.value,
            'learning_velocity': self.learning_velocity,
            'adaptation_counter': self.adaptation_counter,
            'is_learning': self.is_learning,
            'performance_metrics': dict(self.performance_metrics)
        }
    
    def shutdown(self):
        """Shutdown the learning system"""
        self.stop_continuous_learning()
        self.executor.shutdown(wait=True)
        self._save_learning_state()
        logger.info("Self-Learning System shutdown completed")

# Example usage and demonstration
def demonstrate_self_learning():
    """Demonstrate the self-learning system capabilities"""
    print("=== Self-Learning System Demonstration ===")
    
    # Initialize self-learning system
    learning_system = SelfLearningSystem()
    
    # Simulate some interactions
    print("\nSimulating learning interactions...")
    
    for i in range(20):
        # Generate random interaction data
        input_data = f"Input_{i}"
        output_data = f"Output_{i}"
        feedback = np.random.normal(0, 0.3)  # Random feedback with noise
        
        # Learn from interaction
        experience_id = learning_system.learn_from_interaction(
            input_data=input_data,
            output_data=output_data,
            feedback=feedback,
            context={'session_id': f'session_{i%5}', 'user_id': f'user_{i%3}'}
        )
        
        if i % 5 == 0:
            print(f"Processed interaction {i+1}, feedback: {feedback:.3f}")
    
    # Discover patterns
    print("\nDiscovering patterns...")
    patterns = learning_system.discover_patterns()
    print(f"Discovered {len(patterns)} patterns:")
    
    for pattern in patterns[:3]:  # Show top 3 patterns
        print(f"  Pattern {pattern.pattern_id}: {pattern.pattern_type.value}")
        print(f"    Confidence: {pattern.confidence:.3f}")
        print(f"    Frequency: {pattern.frequency}")
        print(f"    Utility: {pattern.utility_score:.3f}")
    
    # Adapt behavior
    print("\nAdapting behavior...")
    learning_system.adapt_behavior(0.3)  # Positive performance change
    learning_system.adapt_behavior(-0.2)  # Negative performance change
    
    # Optimize performance
    print("\nOptimizing performance...")
    optimization_results = learning_system.optimize_performance()
    print(f"Optimization results: {optimization_results}")
    
    # Show statistics
    print("\n=== Learning Statistics ===")
    stats = learning_system.get_learning_statistics()
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"{key}: {len(value)} items")
        else:
            print(f"{key}: {value}")
    
    # Shutdown
    learning_system.shutdown()
    print("\nSelf-Learning System demonstration completed!")

if __name__ == "__main__":
    demonstrate_self_learning()