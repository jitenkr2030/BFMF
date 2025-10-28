"""
Causal Reasoning and Decision Making System for Bharat-FM MLOps Platform

This module implements sophisticated causal reasoning and decision-making capabilities that enable the AI to:
- Understand cause-and-effect relationships
- Make informed decisions based on causal models
- Predict outcomes of different actions
- Evaluate decision quality and trade-offs
- Optimize decisions for multiple objectives

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
import hashlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
import networkx as nx
from pathlib import Path
import itertools

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CausalRelationType(Enum):
    """Types of causal relationships"""
    DIRECT = "direct"  # Direct cause-effect relationship
    INDIRECT = "indirect"  # Mediated cause-effect relationship
    CONFOUNDING = "confounding"  # Common cause relationship
    MEDIATING = "mediating"  # Mediator relationship
    MODERATING = "moderating"  # Moderator relationship
    FEEDBACK = "feedback"  # Feedback loop relationship
    BIDIRECTIONAL = "bidirectional"  # Mutual causation

class DecisionType(Enum):
    """Types of decisions"""
    BINARY = "binary"  # Yes/no decisions
    CATEGORICAL = "categorical"  # Multi-choice decisions
    CONTINUOUS = "continuous"  # Continuous value decisions
    SEQUENTIAL = "sequential"  # Multi-step decisions
    STRATEGIC = "strategic"  # Long-term strategic decisions
    TACTICAL = "tactical"  # Short-term operational decisions
    OPTIMIZATION = "optimization"  # Optimization problems

class UncertaintyLevel(Enum):
    """Levels of uncertainty in reasoning"""
    CERTAIN = "certain"  # Known with certainty
    PROBABLE = "probable"  # High probability
    POSSIBLE = "possible"  # Moderate probability
    UNCERTAIN = "uncertain"  # Low probability
    UNKNOWN = "unknown"  # No information available

@dataclass
class CausalRelation:
    """Represents a causal relationship between variables"""
    relation_id: str
    cause: str
    effect: str
    relation_type: CausalRelationType
    strength: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    direction: str  # "positive" or "negative"
    evidence: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class CausalModel:
    """Represents a causal model of a system"""
    model_id: str
    variables: List[str]
    relations: List[CausalRelation]
    assumptions: List[str]
    scope: str
    confidence: float
    validation_metrics: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DecisionAlternative:
    """Represents a decision alternative"""
    alternative_id: str
    name: str
    description: str
    actions: List[str]
    expected_outcomes: Dict[str, float]
    costs: Dict[str, float]
    benefits: Dict[str, float]
    risks: List[str]
    probability_of_success: float
    implementation_time: timedelta

@dataclass
class DecisionCriteria:
    """Represents decision evaluation criteria"""
    criteria_id: str
    name: str
    description: str
    weight: float  # Importance weight
    min_value: float
    max_value: float
    is_better_higher: bool  # True if higher values are better

@dataclass
class DecisionResult:
    """Represents the result of a decision"""
    decision_id: str
    problem: str
    alternatives: List[DecisionAlternative]
    criteria: List[DecisionCriteria]
    selected_alternative: Optional[DecisionAlternative]
    scores: Dict[str, float]  # alternative_id -> score
    rationale: str
    confidence: float
    uncertainty_level: UncertaintyLevel
    timestamp: datetime = field(default_factory=datetime.now)

class CausalReasoningEngine:
    """
    Advanced causal reasoning and decision-making engine
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Causal reasoning parameters
        self.causality_threshold = self.config.get('causality_threshold', 0.7)
        self.confidence_threshold = self.config.get('confidence_threshold', 0.8)
        self.max_variables = self.config.get('max_variables', 100)
        
        # Decision making parameters
        self.optimization_objective = self.config.get('optimization_objective', 'maximize_utility')
        self.risk_tolerance = self.config.get('risk_tolerance', 0.5)
        self.time_horizon = self.config.get('time_horizon', 'medium_term')
        
        # Data structures
        self.causal_models = {}  # model_id -> CausalModel
        self.causal_relations = defaultdict(list)  # variable -> list of relations
        self.decision_history = deque(maxlen=1000)  # Decision history
        self.intervention_results = {}  # intervention_id -> results
        
        # Machine learning models
        self.causal_discovery_model = None
        self.prediction_model = None
        self.outcome_model = None
        
        # Initialize components
        self._initialize_causal_models()
        self._build_decision_criteria()
        self._initialize_ml_models()
        
        logger.info("Causal Reasoning Engine initialized successfully")
    
    def _initialize_causal_models(self):
        """Initialize predefined causal models"""
        # Create example causal models for different domains
        
        # Business performance causal model
        business_relations = [
            CausalRelation(
                relation_id="rel_1",
                cause="marketing_investment",
                effect="brand_awareness",
                relation_type=CausalRelationType.DIRECT,
                strength=0.8,
                confidence=0.9,
                direction="positive",
                evidence=["market_research", "case_studies"]
            ),
            CausalRelation(
                relation_id="rel_2",
                cause="brand_awareness",
                effect="customer_acquisition",
                relation_type=CausalRelationType.DIRECT,
                strength=0.7,
                confidence=0.85,
                direction="positive",
                evidence=["sales_data", "analytics"]
            ),
            CausalRelation(
                relation_id="rel_3",
                cause="customer_acquisition",
                effect="revenue",
                relation_type=CausalRelationType.DIRECT,
                strength=0.9,
                confidence=0.95,
                direction="positive",
                evidence=["financial_reports"]
            ),
            CausalRelation(
                relation_id="rel_4",
                cause="operational_efficiency",
                effect="costs",
                relation_type=CausalRelationType.DIRECT,
                strength=0.8,
                confidence=0.9,
                direction="negative",
                evidence=["operational_data"]
            ),
            CausalRelation(
                relation_id="rel_5",
                cause="revenue",
                effect="profit",
                relation_type=CausalRelationType.DIRECT,
                strength=0.9,
                confidence=0.95,
                direction="positive",
                evidence=["financial_statements"]
            ),
            CausalRelation(
                relation_id="rel_6",
                cause="costs",
                effect="profit",
                relation_type=CausalRelationType.DIRECT,
                strength=0.9,
                confidence=0.95,
                direction="negative",
                evidence=["financial_statements"]
            )
        ]
        
        business_model = CausalModel(
            model_id="business_performance",
            variables=["marketing_investment", "brand_awareness", "customer_acquisition", 
                      "revenue", "operational_efficiency", "costs", "profit"],
            relations=business_relations,
            assumptions=["linear_relationships", "no_external_shocks", "stable_market"],
            scope="business_performance",
            confidence=0.85,
            validation_metrics={"r_squared": 0.82, "mae": 0.15}
        )
        
        # Health intervention causal model
        health_relations = [
            CausalRelation(
                relation_id="rel_7",
                cause="exercise",
                effect="physical_health",
                relation_type=CausalRelationType.DIRECT,
                strength=0.7,
                confidence=0.8,
                direction="positive",
                evidence["medical_studies"]
            ),
            CausalRelation(
                relation_id="rel_8",
                cause="diet",
                effect="physical_health",
                relation_type=CausalRelationType.DIRECT,
                strength=0.6,
                confidence=0.75,
                direction="positive",
                evidence=["nutrition_research"]
            ),
            CausalRelation(
                relation_id="rel_9",
                cause="sleep",
                effect="physical_health",
                relation_type=CausalRelationType.DIRECT,
                strength=0.5,
                confidence=0.7,
                direction="positive",
                evidence=["sleep_studies"]
            ),
            CausalRelation(
                relation_id="rel_10",
                cause="physical_health",
                effect="mental_health",
                relation_type=CausalRelationType.DIRECT,
                strength=0.6,
                confidence=0.75,
                direction="positive",
                evidence=["health_research"]
            ),
            CausalRelation(
                relation_id="rel_11",
                cause="stress",
                effect="mental_health",
                relation_type=CausalRelationType.DIRECT,
                strength=0.7,
                confidence=0.8,
                direction="negative",
                evidence=["psychological_studies"]
            )
        ]
        
        health_model = CausalModel(
            model_id="health_intervention",
            variables=["exercise", "diet", "sleep", "physical_health", "mental_health", "stress"],
            relations=health_relations,
            assumptions=["individual_variability", "no_genetic_factors", "lifestyle_focus"],
            scope="health_outcomes",
            confidence=0.8,
            validation_metrics={"accuracy": 0.78, "precision": 0.75}
        )
        
        # Store models
        self.causal_models[business_model.model_id] = business_model
        self.causal_models[health_model.model_id] = health_model
        
        # Build causal relation index
        for model in self.causal_models.values():
            for relation in model.relations:
                self.causal_relations[relation.cause].append(relation)
                self.causal_relations[relation.effect].append(relation)
        
        logger.info(f"Initialized {len(self.causal_models)} causal models")
    
    def _build_decision_criteria(self):
        """Build standard decision evaluation criteria"""
        self.standard_criteria = [
            DecisionCriteria(
                criteria_id="cost",
                name="Cost",
                description="Financial cost of implementation",
                weight=0.2,
                min_value=0,
                max_value=1000000,
                is_better_higher=False
            ),
            DecisionCriteria(
                criteria_id="benefit",
                name="Benefit",
                description="Expected benefit or value",
                weight=0.3,
                min_value=0,
                max_value=1000000,
                is_better_higher=True
            ),
            DecisionCriteria(
                criteria_id="risk",
                name="Risk",
                description="Level of risk involved",
                weight=0.15,
                min_value=0,
                max_value=1,
                is_better_higher=False
            ),
            DecisionCriteria(
                criteria_id="time",
                name="Time",
                description="Time required for implementation",
                weight=0.1,
                min_value=0,
                max_value=365,
                is_better_higher=False
            ),
            DecisionCriteria(
                criteria_id="feasibility",
                name="Feasibility",
                description="Ease of implementation",
                weight=0.15,
                min_value=0,
                max_value=1,
                is_better_higher=True
            ),
            DecisionCriteria(
                criteria_id="scalability",
                name="Scalability",
                description="Ability to scale the solution",
                weight=0.1,
                min_value=0,
                max_value=1,
                is_better_higher=True
            )
        ]
        
        logger.info("Built standard decision criteria")
    
    def _initialize_ml_models(self):
        """Initialize machine learning models for causal reasoning"""
        # Initialize causal discovery model (simplified)
        self.causal_discovery_model = {
            'algorithm': 'pc_algorithm',  # Placeholder for PC algorithm
            'alpha': 0.05,  # Significance level
            'max_cond_set_size': 3
        }
        
        # Initialize prediction model
        self.prediction_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        # Initialize outcome model
        self.outcome_model = LogisticRegression(
            random_state=42,
            max_iter=1000
        )
        
        logger.info("Initialized ML models for causal reasoning")
    
    def discover_causal_relations(self, data: Dict[str, List[float]], 
                                variables: List[str]) -> List[CausalRelation]:
        """
        Discover causal relations from observational data
        
        Args:
            data: Dictionary mapping variable names to values
            variables: List of variable names
            
        Returns:
            List[CausalRelation]: Discovered causal relations
        """
        if len(variables) > self.max_variables:
            logger.warning(f"Too many variables ({len(variables)}), limiting to {self.max_variables}")
            variables = variables[:self.max_variables]
        
        discovered_relations = []
        
        # Prepare data for analysis
        X = np.array([data[var] for var in variables]).T
        
        # Simple causal discovery using correlation and conditional independence
        for i, cause in enumerate(variables):
            for j, effect in enumerate(variables):
                if i != j:
                    # Calculate correlation
                    correlation = np.corrcoef(X[:, i], X[:, j])[0, 1]
                    
                    # Test for conditional independence (simplified)
                    independence_score = self._test_conditional_independence(X[:, i], X[:, j], X)
                    
                    # Determine causal direction based on temporal or domain knowledge
                    direction, strength, confidence = self._determine_causal_direction(
                        cause, effect, correlation, independence_score
                    )
                    
                    if strength > self.causality_threshold and confidence > self.confidence_threshold:
                        relation = CausalRelation(
                            relation_id=f"discovered_{int(time.time())}_{i}_{j}",
                            cause=cause,
                            effect=effect,
                            relation_type=CausalRelationType.DIRECT,
                            strength=strength,
                            confidence=confidence,
                            direction=direction,
                            evidence=["statistical_analysis", "conditional_independence_test"]
                        )
                        discovered_relations.append(relation)
        
        logger.info(f"Discovered {len(discovered_relations)} causal relations")
        
        return discovered_relations
    
    def _test_conditional_independence(self, x: np.ndarray, y: np.ndarray, 
                                     data: np.ndarray) -> float:
        """Test conditional independence between variables"""
        # Simplified conditional independence test
        # In production, this would use more sophisticated methods
        
        # Calculate partial correlation
        try:
            from sklearn.linear_model import LinearRegression
            
            # Regress x on all other variables
            other_vars = [col for col in range(data.shape[1]) 
                         if not np.array_equal(data[:, col], x) and not np.array_equal(data[:, col], y)]
            
            if other_vars:
                X_other = data[:, other_vars]
                reg_x = LinearRegression().fit(X_other, x)
                x_residual = x - reg_x.predict(X_other)
                
                reg_y = LinearRegression().fit(X_other, y)
                y_residual = y - reg_y.predict(X_other)
                
                # Calculate correlation of residuals
                partial_corr = np.corrcoef(x_residual, y_residual)[0, 1]
                return abs(partial_corr)
            else:
                return abs(np.corrcoef(x, y)[0, 1])
                
        except Exception as e:
            logger.error(f"Error in conditional independence test: {e}")
            return 0.5
    
    def _determine_causal_direction(self, cause: str, effect: str, 
                                  correlation: float, independence_score: float) -> Tuple[str, float, float]:
        """Determine causal direction and strength"""
        # Simplified direction determination
        # In production, this would use temporal information, domain knowledge, etc.
        
        strength = abs(correlation)
        confidence = 1.0 - independence_score
        
        # Determine direction based on correlation sign
        direction = "positive" if correlation > 0 else "negative"
        
        return direction, strength, confidence
    
    def build_causal_model(self, relations: List[CausalRelation], 
                          variables: List[str], scope: str) -> CausalModel:
        """
        Build a causal model from discovered relations
        
        Args:
            relations: List of causal relations
            variables: List of variables in the model
            scope: Scope of the model
            
        Returns:
            CausalModel: Built causal model
        """
        model_id = f"model_{int(time.time())}_{hashlib.md5(scope.encode()).hexdigest()[:8]}"
        
        # Generate assumptions
        assumptions = [
            "causal_sufficiency",
            "no_unmeasured_confounding",
            "faithfulness",
            "acyclic"
        ]
        
        # Calculate model confidence
        avg_relation_confidence = np.mean([rel.confidence for rel in relations])
        model_confidence = avg_relation_confidence * 0.9  # Slight discount for model complexity
        
        # Calculate validation metrics (simplified)
        validation_metrics = {
            "relation_count": len(relations),
            "variable_count": len(variables),
            "avg_strength": np.mean([rel.strength for rel in relations]),
            "avg_confidence": avg_relation_confidence
        }
        
        model = CausalModel(
            model_id=model_id,
            variables=variables,
            relations=relations,
            assumptions=assumptions,
            scope=scope,
            confidence=model_confidence,
            validation_metrics=validation_metrics
        )
        
        # Store model
        self.causal_models[model_id] = model
        
        # Update causal relation index
        for relation in relations:
            self.causal_relations[relation.cause].append(relation)
            self.causal_relations[relation.effect].append(relation)
        
        logger.info(f"Built causal model {model_id} with {len(relations)} relations")
        
        return model
    
    def predict_intervention_outcome(self, model_id: str, intervention: Dict[str, float],
                                   target_variable: str) -> Dict[str, Any]:
        """
        Predict outcome of intervention on target variable
        
        Args:
            model_id: ID of causal model to use
            intervention: Dictionary mapping variables to intervention values
            target_variable: Variable to predict outcome for
            
        Returns:
            Dict: Prediction results
        """
        if model_id not in self.causal_models:
            raise ValueError(f"Causal model {model_id} not found")
        
        model = self.causal_models[model_id]
        
        # Build causal graph
        causal_graph = nx.DiGraph()
        
        # Add variables as nodes
        for var in model.variables:
            causal_graph.add_node(var)
        
        # Add relations as edges
        for relation in model.relations:
            causal_graph.add_edge(
                relation.cause, 
                relation.effect,
                weight=relation.strength,
                direction=relation.direction
            )
        
        # Calculate intervention effects
        intervention_effects = self._calculate_intervention_effects(
            causal_graph, intervention, target_variable
        )
        
        # Generate prediction
        prediction = {
            "target_variable": target_variable,
            "predicted_outcome": intervention_effects["predicted_value"],
            "confidence_interval": intervention_effects["confidence_interval"],
            "effect_size": intervention_effects["effect_size"],
            "causal_path": intervention_effects["causal_path"],
            "assumptions": model.assumptions,
            "confidence": model.confidence
        }
        
        logger.info(f"Predicted intervention outcome for {target_variable}")
        
        return prediction
    
    def _calculate_intervention_effects(self, causal_graph: nx.DiGraph, 
                                      intervention: Dict[str, float],
                                      target_variable: str) -> Dict[str, Any]:
        """Calculate effects of intervention on target variable"""
        # Simplified intervention effect calculation
        # In production, this would use do-calculus or causal inference methods
        
        # Find all paths from intervention variables to target
        all_paths = []
        for var, value in intervention.items():
            if var in causal_graph.nodes():
                try:
                    paths = nx.all_simple_paths(causal_graph, var, target_variable, cutoff=5)
                    all_paths.extend(list(paths))
                except nx.NetworkXNoPath:
                    continue
        
        # Calculate effect size
        total_effect = 0.0
        causal_path = []
        
        for path in all_paths:
            path_effect = 1.0
            path_edges = []
            
            for i in range(len(path) - 1):
                edge_data = causal_graph.edges[path[i], path[i + 1]]
                path_effect *= edge_data["weight"]
                path_edges.append((path[i], path[i + 1]))
            
            total_effect += path_effect
            if path_effect > 0:
                causal_path = path_edges
        
        # Calculate predicted value (simplified)
        base_value = 0.5  # Assumed baseline
        intervention_effect = sum(intervention.values()) * total_effect
        predicted_value = base_value + intervention_effect
        
        # Calculate confidence interval (simplified)
        margin_of_error = 0.1 * abs(intervention_effect)
        confidence_interval = (predicted_value - margin_of_error, predicted_value + margin_of_error)
        
        return {
            "predicted_value": predicted_value,
            "confidence_interval": confidence_interval,
            "effect_size": total_effect,
            "causal_path": causal_path
        }
    
    def make_decision(self, problem: str, alternatives: List[DecisionAlternative],
                     criteria: Optional[List[DecisionCriteria]] = None,
                     decision_type: DecisionType = DecisionType.CATEGORICAL) -> DecisionResult:
        """
        Make a decision using causal reasoning and multi-criteria analysis
        
        Args:
            problem: Decision problem description
            alternatives: List of decision alternatives
            criteria: List of evaluation criteria (optional)
            decision_type: Type of decision
            
        Returns:
            DecisionResult: Decision result
        """
        decision_id = f"decision_{int(time.time())}_{hashlib.md5(problem.encode()).hexdigest()[:8]}"
        
        # Use standard criteria if none provided
        if criteria is None:
            criteria = self.standard_criteria.copy()
        
        # Normalize criteria weights
        total_weight = sum(c.weight for c in criteria)
        for criterion in criteria:
            criterion.weight /= total_weight
        
        # Score each alternative
        scores = {}
        for alternative in alternatives:
            score = self._evaluate_alternative(alternative, criteria)
            scores[alternative.alternative_id] = score
        
        # Select best alternative
        best_alternative_id = max(scores.items(), key=lambda x: x[1])[0]
        best_alternative = next(alt for alt in alternatives if alt.alternative_id == best_alternative_id)
        
        # Generate rationale
        rationale = self._generate_decision_rationale(
            problem, alternatives, criteria, scores, best_alternative
        )
        
        # Calculate confidence and uncertainty
        confidence = self._calculate_decision_confidence(scores, criteria)
        uncertainty_level = self._determine_uncertainty_level(alternatives, scores)
        
        # Create decision result
        decision_result = DecisionResult(
            decision_id=decision_id,
            problem=problem,
            alternatives=alternatives,
            criteria=criteria,
            selected_alternative=best_alternative,
            scores=scores,
            rationale=rationale,
            confidence=confidence,
            uncertainty_level=uncertainty_level
        )
        
        # Store in history
        self.decision_history.append(decision_result)
        
        logger.info(f"Made decision {decision_id} with confidence {confidence:.3f}")
        
        return decision_result
    
    def _evaluate_alternative(self, alternative: DecisionAlternative, 
                            criteria: List[DecisionCriteria]) -> float:
        """Evaluate a decision alternative against criteria"""
        total_score = 0.0
        
        # Normalize alternative values
        normalized_values = self._normalize_alternative_values(alternative, criteria)
        
        # Calculate weighted score
        for criterion in criteria:
            value = normalized_values.get(criterion.name, 0.5)
            
            # Normalize to [0, 1] range
            normalized_value = (value - criterion.min_value) / (criterion.max_value - criterion.min_value)
            normalized_value = np.clip(normalized_value, 0, 1)
            
            # Invert if lower values are better
            if not criterion.is_better_higher:
                normalized_value = 1 - normalized_value
            
            # Add weighted score
            total_score += normalized_value * criterion.weight
        
        return total_score
    
    def _normalize_alternative_values(self, alternative: DecisionAlternative,
                                    criteria: List[DecisionCriteria]) -> Dict[str, float]:
        """Normalize alternative values for criteria evaluation"""
        normalized_values = {}
        
        # Cost
        total_cost = sum(alternative.costs.values())
        normalized_values["Cost"] = total_cost
        
        # Benefit
        total_benefit = sum(alternative.benefits.values())
        normalized_values["Benefit"] = total_benefit
        
        # Risk (inverse of success probability)
        normalized_values["Risk"] = 1 - alternative.probability_of_success
        
        # Time (days)
        normalized_values["Time"] = alternative.implementation_time.total_seconds() / (24 * 3600)
        
        # Feasibility (based on risks and complexity)
        feasibility = 1 - (len(alternative.risks) * 0.1)
        feasibility = max(0, min(1, feasibility))
        normalized_values["Feasibility"] = feasibility
        
        # Scalability (assumed based on description)
        scalability = 0.7  # Default value
        if "scalable" in alternative.description.lower():
            scalability = 0.9
        normalized_values["Scalability"] = scalability
        
        return normalized_values
    
    def _generate_decision_rationale(self, problem: str, alternatives: List[DecisionAlternative],
                                   criteria: List[DecisionCriteria], scores: Dict[str, float],
                                   selected_alternative: DecisionAlternative) -> str:
        """Generate rationale for decision"""
        rationale_parts = [
            f"Decision Problem: {problem}",
            f"Evaluated {len(alternatives)} alternatives against {len(criteria)} criteria.",
            f"Selected alternative '{selected_alternative.name}' with score {scores[selected_alternative.alternative_id]:.3f}."
        ]
        
        # Add key strengths of selected alternative
        strengths = []
        if selected_alternative.probability_of_success > 0.8:
            strengths.append("high success probability")
        if sum(selected_alternative.benefits.values()) > sum(selected_alternative.costs.values()):
            strengths.append("positive cost-benefit ratio")
        if len(selected_alternative.risks) < 2:
            strengths.append("low risk")
        
        if strengths:
            rationale_parts.append(f"Key strengths: {', '.join(strengths)}.")
        
        # Add comparison with other alternatives
        if len(alternatives) > 1:
            other_scores = [scores[alt.alternative_id] for alt in alternatives if alt.alternative_id != selected_alternative.alternative_id]
            avg_other_score = np.mean(other_scores)
            if scores[selected_alternative.alternative_id] > avg_other_score:
                rationale_parts.append(f"This alternative outperformed others by {scores[selected_alternative.alternative_id] - avg_other_score:.3f} points on average.")
        
        return " ".join(rationale_parts)
    
    def _calculate_decision_confidence(self, scores: Dict[str, float],
                                     criteria: List[DecisionCriteria]) -> float:
        """Calculate confidence in decision"""
        if not scores:
            return 0.5
        
        # Calculate score variance (lower variance = higher confidence)
        score_values = list(scores.values())
        score_variance = np.var(score_values)
        
        # Normalize variance to confidence
        max_variance = 1.0  # Maximum possible variance
        confidence = 1.0 - (score_variance / max_variance)
        
        # Adjust based on criteria quality
        criteria_quality = np.mean([c.weight for c in criteria])
        confidence *= criteria_quality
        
        return max(0, min(1, confidence))
    
    def _determine_uncertainty_level(self, alternatives: List[DecisionAlternative],
                                  scores: Dict[str, float]) -> UncertaintyLevel:
        """Determine uncertainty level of decision"""
        if not alternatives:
            return UncertaintyLevel.UNKNOWN
        
        # Calculate score differences
        if len(scores) < 2:
            return UncertaintyLevel.CERTAIN
        
        score_values = list(scores.values())
        max_score = max(score_values)
        min_score = min(score_values)
        score_range = max_score - min_score
        
        # Calculate average risk
        avg_risk = np.mean([len(alt.risks) for alt in alternatives])
        
        # Determine uncertainty based on score range and risk
        if score_range > 0.3 and avg_risk < 1:
            return UncertaintyLevel.CERTAIN
        elif score_range > 0.2 and avg_risk < 2:
            return UncertaintyLevel.PROBABLE
        elif score_range > 0.1 and avg_risk < 3:
            return UncertaintyLevel.POSSIBLE
        elif score_range > 0.05:
            return UncertaintyLevel.UNCERTAIN
        else:
            return UncertaintyLevel.UNKNOWN
    
    def optimize_decision(self, problem: str, decision_space: Dict[str, Any],
                        objectives: List[str], constraints: List[str]) -> DecisionResult:
        """
        Optimize decision using multi-objective optimization
        
        Args:
            problem: Decision problem description
            decision_space: Dictionary defining decision space
            objectives: List of optimization objectives
            constraints: List of constraints
            
        Returns:
            DecisionResult: Optimized decision result
        """
        # Generate decision alternatives
        alternatives = self._generate_decision_alternatives(decision_space, objectives, constraints)
        
        # Create optimization-focused criteria
        criteria = [
            DecisionCriteria(
                criteria_id="objective_1",
                name=objectives[0],
                description=f"Optimization objective: {objectives[0]}",
                weight=0.5,
                min_value=0,
                max_value=1,
                is_better_higher=True
            )
        ]
        
        if len(objectives) > 1:
            criteria.append(DecisionCriteria(
                criteria_id="objective_2",
                name=objectives[1],
                description=f"Optimization objective: {objectives[1]}",
                weight=0.3,
                min_value=0,
                max_value=1,
                is_better_higher=True
            ))
        
        criteria.append(DecisionCriteria(
            criteria_id="constraint_satisfaction",
            name="Constraint Satisfaction",
            description="Degree of constraint satisfaction",
            weight=0.2,
            min_value=0,
            max_value=1,
            is_better_higher=True
        ))
        
        # Make optimized decision
        decision_result = self.make_decision(
            problem=problem,
            alternatives=alternatives,
            criteria=criteria,
            decision_type=DecisionType.OPTIMIZATION
        )
        
        logger.info(f"Optimized decision with {len(alternatives)} alternatives")
        
        return decision_result
    
    def _generate_decision_alternatives(self, decision_space: Dict[str, Any],
                                      objectives: List[str], constraints: List[str]) -> List[DecisionAlternative]:
        """Generate decision alternatives for optimization"""
        alternatives = []
        
        # Simple alternative generation based on decision space
        # In production, this would use more sophisticated methods
        
        # Generate a few alternatives with different characteristics
        alternative_configs = [
            {"name": "Conservative Approach", "risk_level": "low", "investment": "low"},
            {"name": "Balanced Approach", "risk_level": "medium", "investment": "medium"},
            {"name": "Aggressive Approach", "risk_level": "high", "investment": "high"}
        ]
        
        for i, config in enumerate(alternative_configs):
            alternative = DecisionAlternative(
                alternative_id=f"alt_{i}",
                name=config["name"],
                description=f"{config['name']} for {objectives[0]} optimization",
                actions=[f"implement_{config['name'].lower().replace(' ', '_')}"],
                expected_outcomes={
                    objectives[0]: 0.5 + i * 0.2,  # Increasing benefit
                    objectives[1] if len(objectives) > 1 else "secondary": 0.4 + i * 0.15
                },
                costs={
                    "financial": 10000 * (i + 1),
                    "time": 30 * (i + 1)
                },
                benefits={
                    objectives[0]: 50000 * (i + 1),
                    objectives[1] if len(objectives) > 1 else "secondary": 30000 * (i + 1)
                },
                risks=[f"{config['risk_level']}_risk"],
                probability_of_success=0.9 - i * 0.1,  # Decreasing success with risk
                implementation_time=timedelta(days=30 * (i + 1))
            )
            alternatives.append(alternative)
        
        return alternatives
    
    def evaluate_decision_quality(self, decision_result: DecisionResult) -> Dict[str, Any]:
        """
        Evaluate the quality of a decision
        
        Args:
            decision_result: Decision result to evaluate
            
        Returns:
            Dict: Quality evaluation results
        """
        evaluation = {
            "decision_id": decision_result.decision_id,
            "confidence_score": decision_result.confidence,
            "uncertainty_assessment": decision_result.uncertainty_level.value,
            "alternative_analysis": {},
            "criteria_analysis": {},
            "overall_quality": 0.0,
            "recommendations": []
        }
        
        # Analyze alternatives
        scores = decision_result.scores
        if scores:
            score_values = list(scores.values())
            evaluation["alternative_analysis"] = {
                "score_range": max(score_values) - min(score_values),
                "score_variance": np.var(score_values),
                "clear_winner": max(score_values) - min(score_values) > 0.2
            }
        
        # Analyze criteria
        criteria_weights = [c.weight for c in decision_result.criteria]
        evaluation["criteria_analysis"] = {
            "weight_distribution": {
                "min_weight": min(criteria_weights),
                "max_weight": max(criteria_weights),
                "weight_variance": np.var(criteria_weights)
            },
            "criteria_count": len(decision_result.criteria)
        }
        
        # Calculate overall quality
        quality_components = [
            decision_result.confidence,
            1.0 - len(decision_result.selected_alternative.risks) * 0.1,
            decision_result.selected_alternative.probability_of_success
        ]
        evaluation["overall_quality"] = np.mean(quality_components)
        
        # Generate recommendations
        if evaluation["overall_quality"] < 0.6:
            evaluation["recommendations"].append("Consider gathering more information")
        if decision_result.uncertainty_level in [UncertaintyLevel.UNCERTAIN, UncertaintyLevel.UNKNOWN]:
            evaluation["recommendations"].append("Reduce uncertainty through additional analysis")
        if len(decision_result.selected_alternative.risks) > 3:
            evaluation["recommendations"].append("Develop risk mitigation strategies")
        
        return evaluation
    
    def learn_from_decision_outcome(self, decision_id: str, actual_outcome: Dict[str, Any],
                                 feedback: float) -> bool:
        """
        Learn from decision outcomes to improve future decisions
        
        Args:
            decision_id: ID of decision to learn from
            actual_outcome: Actual outcome of decision
            feedback: Feedback score (-1 to 1)
            
        Returns:
            bool: Success status
        """
        # Find decision in history
        decision = None
        for d in self.decision_history:
            if d.decision_id == decision_id:
                decision = d
                break
        
        if decision is None:
            logger.error(f"Decision {decision_id} not found in history")
            return False
        
        # Store intervention result
        self.intervention_results[decision_id] = {
            "decision": decision,
            "actual_outcome": actual_outcome,
            "feedback": feedback,
            "timestamp": datetime.now()
        }
        
        # Update causal models based on feedback
        if feedback > 0.5:
            # Positive feedback - strengthen causal relations
            self._update_causal_relations(decision, True)
        elif feedback < -0.5:
            # Negative feedback - weaken causal relations
            self._update_causal_relations(decision, False)
        
        logger.info(f"Learned from decision {decision_id} with feedback {feedback:.2f}")
        
        return True
    
    def _update_causal_relations(self, decision: DecisionResult, strengthen: bool):
        """Update causal relations based on decision feedback"""
        # Simplified relation update
        # In production, this would use more sophisticated learning algorithms
        
        # Find relevant causal relations
        relevant_relations = []
        for model in self.causal_models.values():
            for relation in model.relations:
                # Check if relation is relevant to decision
                if (relation.cause in decision.problem.lower() or 
                    relation.effect in decision.problem.lower()):
                    relevant_relations.append(relation)
        
        # Update relation strengths
        for relation in relevant_relations:
            if strengthen:
                relation.strength = min(relation.strength * 1.05, 1.0)
                relation.confidence = min(relation.confidence * 1.02, 1.0)
            else:
                relation.strength = max(relation.strength * 0.95, 0.1)
                relation.confidence = max(relation.confidence * 0.98, 0.1)
        
        logger.info(f"Updated {len(relevant_relations)} causal relations")
    
    def get_causal_statistics(self) -> Dict[str, Any]:
        """Get comprehensive causal reasoning statistics"""
        return {
            "total_causal_models": len(self.causal_models),
            "total_causal_relations": sum(len(model.relations) for model in self.causal_models.values()),
            "total_decisions": len(self.decision_history),
            "intervention_results": len(self.intervention_results),
            "relation_types": {
                rel_type.value: sum(1 for model in self.causal_models.values() 
                                 for rel in model.relations if rel.relation_type == rel_type)
                for rel_type in CausalRelationType
            },
            "decision_types": {
                dec_type.value: sum(1 for dec in self.decision_history 
                                 if dec.selected_alternative and 
                                 self._classify_decision_type(dec) == dec_type)
                for dec_type in DecisionType
            },
            "average_decision_confidence": np.mean([dec.confidence for dec in self.decision_history]) if self.decision_history else 0,
            "learning_iterations": len(self.intervention_results),
            "model_accuracy": np.mean([model.confidence for model in self.causal_models.values()]) if self.causal_models else 0
        }
    
    def _classify_decision_type(self, decision: DecisionResult) -> DecisionType:
        """Classify decision type based on decision characteristics"""
        if len(decision.alternatives) == 2:
            return DecisionType.BINARY
        elif decision.selected_alternative.implementation_time > timedelta(days=365):
            return DecisionType.STRATEGIC
        elif decision.selected_alternative.implementation_time < timedelta(days=30):
            return DecisionType.TACTICAL
        else:
            return DecisionType.CATEGORICAL
    
    def save_causal_data(self, filepath: str):
        """Save causal reasoning data to file"""
        try:
            data = {
                "causal_models": {
                    model_id: {
                        "variables": model.variables,
                        "relations": [
                            {
                                "cause": rel.cause,
                                "effect": rel.effect,
                                "relation_type": rel.relation_type.value,
                                "strength": rel.strength,
                                "confidence": rel.confidence,
                                "direction": rel.direction
                            }
                            for rel in model.relations
                        ],
                        "assumptions": model.assumptions,
                        "scope": model.scope,
                        "confidence": model.confidence,
                        "validation_metrics": model.validation_metrics
                    }
                    for model_id, model in self.causal_models.items()
                },
                "standard_criteria": [
                    {
                        "name": criterion.name,
                        "description": criterion.description,
                        "weight": criterion.weight,
                        "min_value": criterion.min_value,
                        "max_value": criterion.max_value,
                        "is_better_higher": criterion.is_better_higher
                    }
                    for criterion in self.standard_criteria
                ],
                "intervention_results": {
                    decision_id: {
                        "feedback": result["feedback"],
                        "timestamp": result["timestamp"].isoformat()
                    }
                    for decision_id, result in self.intervention_results.items()
                }
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Causal reasoning data saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving causal data: {e}")
    
    def load_causal_data(self, filepath: str):
        """Load causal reasoning data from file"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Load causal models
            self.causal_models = {}
            for model_id, model_data in data["causal_models"].items():
                relations = [
                    CausalRelation(
                        relation_id=f"loaded_{i}",
                        cause=rel["cause"],
                        effect=rel["effect"],
                        relation_type=CausalRelationType(rel["relation_type"]),
                        strength=rel["strength"],
                        confidence=rel["confidence"],
                        direction=rel["direction"]
                    )
                    for i, rel in enumerate(model_data["relations"])
                ]
                
                model = CausalModel(
                    model_id=model_id,
                    variables=model_data["variables"],
                    relations=relations,
                    assumptions=model_data["assumptions"],
                    scope=model_data["scope"],
                    confidence=model_data["confidence"],
                    validation_metrics=model_data["validation_metrics"]
                )
                
                self.causal_models[model_id] = model
            
            # Load standard criteria
            self.standard_criteria = [
                DecisionCriteria(
                    criteria_id=f"loaded_{i}",
                    name=criterion["name"],
                    description=criterion["description"],
                    weight=criterion["weight"],
                    min_value=criterion["min_value"],
                    max_value=criterion["max_value"],
                    is_better_higher=criterion["is_better_higher"]
                )
                for i, criterion in enumerate(data["standard_criteria"])
            ]
            
            # Load intervention results
            self.intervention_results = {
                decision_id: {
                    "feedback": result["feedback"],
                    "timestamp": datetime.fromisoformat(result["timestamp"])
                }
                for decision_id, result in data["intervention_results"].items()
            }
            
            # Rebuild causal relation index
            self.causal_relations = defaultdict(list)
            for model in self.causal_models.values():
                for relation in model.relations:
                    self.causal_relations[relation.cause].append(relation)
                    self.causal_relations[relation.effect].append(relation)
            
            logger.info(f"Causal reasoning data loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading causal data: {e}")

# Example usage and demonstration
def demonstrate_causal_reasoning():
    """Demonstrate the causal reasoning capabilities"""
    print("=== Causal Reasoning Engine Demonstration ===")
    
    # Initialize causal reasoning engine
    causal_engine = CausalReasoningEngine()
    
    # Test causal discovery
    print("\n=== Causal Discovery ===")
    
    # Generate sample data
    np.random.seed(42)
    sample_data = {
        "marketing_investment": np.random.normal(100, 20, 100),
        "brand_awareness": np.random.normal(60, 15, 100),
        "customer_acquisition": np.random.normal(30, 10, 100),
        "revenue": np.random.normal(500, 100, 100),
        "costs": np.random.normal(300, 50, 100),
        "profit": np.random.normal(200, 80, 100)
    }
    
    # Add causal relationships
    for i in range(100):
        sample_data["brand_awareness"][i] += 0.5 * sample_data["marketing_investment"][i] + np.random.normal(0, 5)
        sample_data["customer_acquisition"][i] += 0.3 * sample_data["brand_awareness"][i] + np.random.normal(0, 3)
        sample_data["revenue"][i] += 2.0 * sample_data["customer_acquisition"][i] + np.random.normal(0, 20)
        sample_data["profit"][i] += sample_data["revenue"][i] - sample_data["costs"][i] + np.random.normal(0, 10)
    
    variables = list(sample_data.keys())
    discovered_relations = causal_engine.discover_causal_relations(sample_data, variables)
    
    print(f"Discovered {len(discovered_relations)} causal relations:")
    for relation in discovered_relations[:3]:  # Show top 3
        print(f"  {relation.cause} -> {relation.effect} (strength: {relation.strength:.3f}, confidence: {relation.confidence:.3f})")
    
    # Build causal model
    causal_model = causal_engine.build_causal_model(
        discovered_relations, variables, "business_performance"
    )
    print(f"Built causal model: {causal_model.model_id}")
    
    # Test intervention prediction
    print("\n=== Intervention Prediction ===")
    intervention = {"marketing_investment": 150}  # Increase marketing investment
    prediction = causal_engine.predict_intervention_outcome(
        causal_model.model_id, intervention, "profit"
    )
    
    print(f"Intervention: {intervention}")
    print(f"Predicted profit outcome: {prediction['predicted_outcome']:.2f}")
    print(f"Effect size: {prediction['effect_size']:.3f}")
    print(f"Confidence interval: {prediction['confidence_interval']}")
    
    # Test decision making
    print("\n=== Decision Making ===")
    
    # Create decision alternatives
    alternatives = [
        DecisionAlternative(
            alternative_id="alt_1",
            name="Conservative Marketing",
            description="Increase marketing budget by 10%",
            actions=["increase_marketing_10_percent"],
            expected_outcomes={"brand_awareness": 0.1, "customer_acquisition": 0.05},
            costs={"financial": 50000, "time": 30},
            benefits={"revenue": 100000, "profit": 50000},
            risks=["market_saturation"],
            probability_of_success=0.8,
            implementation_time=timedelta(days=30)
        ),
        DecisionAlternative(
            alternative_id="alt_2",
            name="Aggressive Marketing",
            description="Increase marketing budget by 50%",
            actions=["increase_marketing_50_percent"],
            expected_outcomes={"brand_awareness": 0.3, "customer_acquisition": 0.15},
            costs={"financial": 150000, "time": 60},
            benefits={"revenue": 300000, "profit": 150000},
            risks=["high_investment", "market_risk"],
            probability_of_success=0.6,
            implementation_time=timedelta(days=60)
        ),
        DecisionAlternative(
            alternative_id="alt_3",
            name="Digital Transformation",
            description="Invest in digital marketing channels",
            actions=["digital_transformation"],
            expected_outcomes={"brand_awareness": 0.2, "customer_acquisition": 0.1},
            costs={"financial": 100000, "time": 90},
            benefits={"revenue": 200000, "profit": 100000},
            risks=["technical_challenges", "adoption_risk"],
            probability_of_success=0.7,
            implementation_time=timedelta(days=90)
        )
    ]
    
    # Make decision
    decision_result = causal_engine.make_decision(
        problem="Choose marketing strategy for next quarter",
        alternatives=alternatives,
        decision_type=DecisionType.CATEGORICAL
    )
    
    print(f"Decision Problem: {decision_result.problem}")
    print(f"Selected Alternative: {decision_result.selected_alternative.name}")
    print(f"Decision Score: {decision_result.scores[decision_result.selected_alternative.alternative_id]:.3f}")
    print(f"Confidence: {decision_result.confidence:.3f}")
    print(f"Uncertainty Level: {decision_result.uncertainty_level.value}")
    print(f"Rationale: {decision_result.rationale}")
    
    # Test decision optimization
    print("\n=== Decision Optimization ===")
    
    decision_space = {
        "budget_range": [50000, 200000],
        "time_horizon": [30, 90],
        "risk_tolerance": [0.1, 0.9]
    }
    
    objectives = ["profit_maximization", "brand_awareness"]
    constraints = ["budget_limit", "time_constraint"]
    
    optimized_decision = causal_engine.optimize_decision(
        problem="Optimize marketing investment strategy",
        decision_space=decision_space,
        objectives=objectives,
        constraints=constraints
    )
    
    print(f"Optimization Problem: {optimized_decision.problem}")
    print(f"Selected Alternative: {optimized_decision.selected_alternative.name}")
    print(f"Optimization Score: {optimized_decision.scores[optimized_decision.selected_alternative.alternative_id]:.3f}")
    
    # Test decision quality evaluation
    print("\n=== Decision Quality Evaluation ===")
    quality_evaluation = causal_engine.evaluate_decision_quality(decision_result)
    
    print(f"Overall Quality: {quality_evaluation['overall_quality']:.3f}")
    print(f"Confidence Score: {quality_evaluation['confidence_score']:.3f}")
    print(f"Clear Winner: {quality_evaluation['alternative_analysis']['clear_winner']}")
    print(f"Recommendations: {', '.join(quality_evaluation['recommendations'])}")
    
    # Test learning from outcomes
    print("\n=== Learning from Outcomes ===")
    success = causal_engine.learn_from_decision_outcome(
        decision_result.decision_id,
        actual_outcome={"profit_increase": 60000, "brand_awareness_increase": 0.15},
        feedback=0.8  # Positive feedback
    )
    
    if success:
        print("Successfully learned from decision outcome")
    
    # Show statistics
    print("\n=== Causal Reasoning Statistics ===")
    stats = causal_engine.get_causal_statistics()
    
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"{key}: {dict(value)}")
        elif isinstance(value, float):
            print(f"{key}: {value:.3f}")
        else:
            print(f"{key}: {value}")
    
    # Save causal data
    causal_engine.save_causal_data('causal_data.json')
    print("\nCausal reasoning data saved successfully!")

if __name__ == "__main__":
    demonstrate_causal_reasoning()