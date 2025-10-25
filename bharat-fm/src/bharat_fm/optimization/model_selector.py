"""
Model Selector for Bharat-FM Inference Optimization
Intelligently selects optimal models based on requirements and performance history
"""

import asyncio
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
import logging
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

@dataclass
class ModelPerformance:
    """Performance metrics for a model"""
    model_id: str
    avg_latency: float = 0.0
    avg_throughput: float = 0.0
    success_rate: float = 1.0
    error_rate: float = 0.0
    cost_per_request: float = 0.0
    quality_score: float = 0.0
    last_updated: datetime = None
    
    def __post_init__(self):
        if self.last_updated is None:
            self.last_updated = datetime.utcnow()

@dataclass
class SelectionCriteria:
    """Criteria for model selection"""
    max_latency: float = 1.0  # Maximum acceptable latency in seconds
    min_accuracy: float = 0.8  # Minimum acceptable accuracy
    max_cost: float = 0.01  # Maximum cost per request
    preferred_language: str = "en"  # Preferred language
    domain: str = "general"  # Target domain
    priority_weights: Dict[str, float] = None
    
    def __post_init__(self):
        if self.priority_weights is None:
            self.priority_weights = {
                "latency": 0.3,
                "accuracy": 0.3,
                "cost": 0.2,
                "throughput": 0.2
            }

class ModelSelector:
    """Intelligent model selection system"""
    
    def __init__(self):
        # Model registry
        self.available_models: Dict[str, Dict[str, Any]] = {}
        self.model_performance: Dict[str, ModelPerformance] = defaultdict(ModelPerformance)
        
        # Selection history
        self.selection_history = deque(maxlen=1000)
        self.performance_history = defaultdict(lambda: deque(maxlen=100))
        
        # Learning and adaptation
        self.selection_weights = defaultdict(lambda: {
            "latency": 0.3,
            "accuracy": 0.3,
            "cost": 0.2,
            "throughput": 0.2
        })
        
        # Background tasks
        self.optimizer_task = None
        self.running = False
        
    async def start(self):
        """Start model selector"""
        if self.running:
            return
        
        self.running = True
        self.optimizer_task = asyncio.create_task(self._optimize_selection())
        
        logger.info("Model selector started")
    
    async def stop(self):
        """Stop model selector"""
        self.running = False
        
        if self.optimizer_task:
            self.optimizer_task.cancel()
            try:
                await self.optimizer_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Model selector stopped")
    
    def register_model(self, model_id: str, model_config: Dict[str, Any]):
        """Register a new model"""
        self.available_models[model_id] = {
            **model_config,
            "registered_at": datetime.utcnow()
        }
        
        # Initialize performance tracking
        if model_id not in self.model_performance:
            self.model_performance[model_id] = ModelPerformance(model_id=model_id)
        
        logger.info(f"Model registered: {model_id}")
    
    async def select_model(self, request_model_id: str, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Select optimal model for given requirements"""
        # Parse requirements
        criteria = SelectionCriteria(
            max_latency=requirements.get("max_latency", 1.0),
            min_accuracy=requirements.get("min_accuracy", 0.8),
            max_cost=requirements.get("max_cost", 0.01),
            preferred_language=requirements.get("language", "en"),
            domain=requirements.get("domain", "general")
        )
        
        # Get candidate models
        candidates = await self._get_candidate_models(request_model_id, criteria)
        
        if not candidates:
            logger.warning(f"No suitable models found for requirements: {requirements}")
            return {"model_id": request_model_id, "reason": "no_candidates"}
        
        # Score candidates
        scored_candidates = await self._score_candidates(candidates, criteria)
        
        # Select best model
        best_model = max(scored_candidates, key=lambda x: x["score"])
        
        # Record selection
        await self._record_selection(best_model, criteria, scored_candidates)
        
        logger.info(f"Selected model: {best_model['model_id']} with score: {best_model['score']:.3f}")
        
        return best_model
    
    async def _get_candidate_models(self, request_model_id: str, criteria: SelectionCriteria) -> List[Dict[str, Any]]:
        """Get list of candidate models that meet basic requirements"""
        candidates = []
        
        # Start with requested model
        if request_model_id in self.available_models:
            candidates.append(request_model_id)
        
        # Add alternative models based on compatibility
        for model_id, model_config in self.available_models.items():
            if model_id == request_model_id:
                continue
            
            # Check basic compatibility
            if await self._is_model_compatible(model_config, criteria):
                candidates.append(model_id)
        
        # Filter by performance requirements
        compatible_candidates = []
        for model_id in candidates:
            performance = self.model_performance[model_id]
            
            # Check if model meets performance criteria
            if (performance.avg_latency <= criteria.max_latency and
                performance.success_rate >= criteria.min_accuracy and
                performance.cost_per_request <= criteria.max_cost):
                compatible_candidates.append(model_id)
        
        return compatible_candidates
    
    async def _is_model_compatible(self, model_config: Dict[str, Any], criteria: SelectionCriteria) -> bool:
        """Check if model is compatible with requirements"""
        # Check language support
        supported_languages = model_config.get("languages", ["en"])
        if criteria.preferred_language not in supported_languages:
            return False
        
        # Check domain compatibility
        model_domain = model_config.get("domain", "general")
        if criteria.domain != "general" and model_domain != criteria.domain:
            return False
        
        # Check capability requirements
        required_capabilities = criteria.__dict__.get("required_capabilities", [])
        model_capabilities = model_config.get("capabilities", [])
        
        for capability in required_capabilities:
            if capability not in model_capabilities:
                return False
        
        return True
    
    async def _score_candidates(self, candidate_ids: List[str], criteria: SelectionCriteria) -> List[Dict[str, Any]]:
        """Score candidate models based on multiple criteria"""
        scored_candidates = []
        
        for model_id in candidate_ids:
            model_config = self.available_models[model_id]
            performance = self.model_performance[model_id]
            
            # Calculate individual scores
            latency_score = self._calculate_latency_score(performance.avg_latency, criteria.max_latency)
            accuracy_score = performance.success_rate  # Already normalized
            cost_score = self._calculate_cost_score(performance.cost_per_request, criteria.max_cost)
            throughput_score = self._calculate_throughput_score(performance.avg_throughput)
            
            # Get adaptive weights for this model type
            weights = self.selection_weights[model_id]
            
            # Calculate weighted score
            total_score = (
                latency_score * weights["latency"] +
                accuracy_score * weights["accuracy"] +
                cost_score * weights["cost"] +
                throughput_score * weights["throughput"]
            )
            
            # Apply domain-specific adjustments
            if criteria.domain == model_config.get("domain", "general"):
                total_score *= 1.1  # 10% bonus for domain match
            
            scored_candidates.append({
                "model_id": model_id,
                "score": total_score,
                "latency_score": latency_score,
                "accuracy_score": accuracy_score,
                "cost_score": cost_score,
                "throughput_score": throughput_score,
                "performance": performance,
                "config": model_config
            })
        
        return scored_candidates
    
    def _calculate_latency_score(self, latency: float, max_latency: float) -> float:
        """Calculate latency score (lower is better)"""
        if latency <= 0:
            return 1.0
        
        # Normalize to 0-1 scale
        score = max(0, 1 - (latency / max_latency))
        return score
    
    def _calculate_cost_score(self, cost: float, max_cost: float) -> float:
        """Calculate cost score (lower is better)"""
        if cost <= 0:
            return 1.0
        
        # Normalize to 0-1 scale
        score = max(0, 1 - (cost / max_cost))
        return score
    
    def _calculate_throughput_score(self, throughput: float) -> float:
        """Calculate throughput score (higher is better)"""
        if throughput <= 0:
            return 0.0
        
        # Normalize using sigmoid function
        score = 1 / (1 + np.exp(-throughput / 10))  # Normalize around 10 req/s
        return score
    
    async def _record_selection(self, selected_model: Dict[str, Any], criteria: SelectionCriteria, 
                              all_candidates: List[Dict[str, Any]]):
        """Record model selection for learning"""
        selection_record = {
            "timestamp": datetime.utcnow(),
            "selected_model": selected_model["model_id"],
            "criteria": criteria.__dict__,
            "all_candidates": [c["model_id"] for c in all_candidates],
            "scores": {c["model_id"]: c["score"] for c in all_candidates}
        }
        
        self.selection_history.append(selection_record)
    
    async def update_model_performance(self, model_id: str, performance_data: Dict[str, Any]):
        """Update model performance metrics"""
        if model_id not in self.model_performance:
            self.model_performance[model_id] = ModelPerformance(model_id=model_id)
        
        performance = self.model_performance[model_id]
        
        # Update metrics with exponential smoothing
        alpha = 0.1  # Smoothing factor
        
        if "latency" in performance_data:
            performance.avg_latency = (
                alpha * performance_data["latency"] + 
                (1 - alpha) * performance.avg_latency
            )
        
        if "throughput" in performance_data:
            performance.avg_throughput = (
                alpha * performance_data["throughput"] + 
                (1 - alpha) * performance.avg_throughput
            )
        
        if "success" in performance_data:
            success = 1 if performance_data["success"] else 0
            performance.success_rate = (
                alpha * success + 
                (1 - alpha) * performance.success_rate
            )
            performance.error_rate = 1 - performance.success_rate
        
        if "cost" in performance_data:
            performance.cost_per_request = (
                alpha * performance_data["cost"] + 
                (1 - alpha) * performance.cost_per_request
            )
        
        if "quality_score" in performance_data:
            performance.quality_score = (
                alpha * performance_data["quality_score"] + 
                (1 - alpha) * performance.quality_score
            )
        
        performance.last_updated = datetime.utcnow()
        
        # Record in performance history
        self.performance_history[model_id].append({
            "timestamp": datetime.utcnow(),
            **performance_data
        })
    
    async def _optimize_selection(self):
        """Background task for optimizing model selection"""
        while self.running:
            try:
                await asyncio.sleep(3600)  # Optimize every hour
                
                # Analyze selection patterns
                await self._analyze_selection_patterns()
                
                # Update selection weights based on performance
                await self._update_selection_weights()
                
                logger.info("Model selection optimization completed")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in model selection optimization: {e}")
    
    async def _analyze_selection_patterns(self):
        """Analyze historical selection patterns"""
        if len(self.selection_history) < 10:
            return
        
        # Analyze success rates of selections
        model_success_rates = defaultdict(lambda: {"successes": 0, "total": 0})
        
        for record in self.selection_history:
            model_id = record["selected_model"]
            # In a real implementation, we would track actual success/failure
            # For now, assume 80% success rate
            success = np.random.random() < 0.8
            
            model_success_rates[model_id]["total"] += 1
            if success:
                model_success_rates[model_id]["successes"] += 1
        
        # Log insights
        for model_id, stats in model_success_rates.items():
            if stats["total"] >= 5:
                success_rate = stats["successes"] / stats["total"]
                if success_rate < 0.7:
                    logger.warning(f"Model {model_id} has low success rate: {success_rate:.2%}")
    
    async def _update_selection_weights(self):
        """Update selection weights based on performance feedback"""
        for model_id, performance in self.model_performance.items():
            # Adjust weights based on performance characteristics
            weights = self.selection_weights[model_id]
            
            # If latency is consistently high, increase latency weight
            if performance.avg_latency > 0.5:
                weights["latency"] = min(0.5, weights["latency"] + 0.05)
            
            # If cost is consistently high, increase cost weight
            if performance.cost_per_request > 0.005:
                weights["cost"] = min(0.4, weights["cost"] + 0.05)
            
            # Normalize weights
            total_weight = sum(weights.values())
            for key in weights:
                weights[key] /= total_weight
    
    async def get_selection_stats(self) -> Dict[str, Any]:
        """Get model selection statistics"""
        return {
            "available_models": len(self.available_models),
            "total_selections": len(self.selection_history),
            "model_performance": {
                model_id: {
                    "avg_latency": perf.avg_latency,
                    "avg_throughput": perf.avg_throughput,
                    "success_rate": perf.success_rate,
                    "cost_per_request": perf.cost_per_request,
                    "quality_score": perf.quality_score,
                    "last_updated": perf.last_updated.isoformat()
                }
                for model_id, perf in self.model_performance.items()
            },
            "selection_weights": dict(self.selection_weights),
            "recent_selections": [
                {
                    "timestamp": record["timestamp"].isoformat(),
                    "selected_model": record["selected_model"],
                    "score": record["scores"].get(record["selected_model"], 0)
                }
                for record in list(self.selection_history)[-10:]
            ]
        }