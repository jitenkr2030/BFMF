"""
Inference Optimizer for Bharat-FM
Main orchestrator for all inference optimization components
"""

import asyncio
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import logging

from .semantic_cache import SemanticCache
from .dynamic_batcher import DynamicBatcher
from .cost_monitor import CostMonitor
from .model_selector import ModelSelector
from .performance_tracker import PerformanceTracker

logger = logging.getLogger(__name__)

@dataclass
class InferenceRequest:
    """Inference request with optimization metadata"""
    id: str
    input_data: Any
    model_id: str
    requirements: Dict[str, Any]
    priority: int = 0
    user_id: Optional[str] = None
    session_id: Optional[str] = None

@dataclass
class InferenceResponse:
    """Optimized inference response"""
    request_id: str
    output: Any
    model_used: str
    latency: float
    cost: float
    cache_hit: bool
    batched: bool
    optimization_metrics: Dict[str, Any]
    error: Optional[str] = None

class InferenceOptimizer:
    """Main inference optimization orchestrator"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Initialize optimization components
        self.semantic_cache = SemanticCache(
            cache_dir=self.config.get("cache_dir", "./cache"),
            max_entries=self.config.get("max_cache_entries", 10000),
            default_ttl=self.config.get("cache_ttl", 3600),
            similarity_threshold=self.config.get("similarity_threshold", 0.95)
        )
        
        self.dynamic_batcher = DynamicBatcher(
            max_batch_size=self.config.get("max_batch_size", 8),
            max_wait_time=self.config.get("max_wait_time", 0.1),
            strategy=self.config.get("batching_strategy", "adaptive")
        )
        
        self.cost_monitor = CostMonitor()
        self.model_selector = ModelSelector()
        self.performance_tracker = PerformanceTracker()
        
        # Optimization state
        self.optimization_enabled = self.config.get("optimization_enabled", True)
        self.caching_enabled = self.config.get("caching_enabled", True)
        self.batching_enabled = self.config.get("batching_enabled", False)  # Disable batching by default
        self.cost_monitoring_enabled = self.config.get("cost_monitoring_enabled", True)
        
        # Statistics
        self.total_requests = 0
        self.cache_hits = 0
        self.batched_requests = 0
        self.total_cost_saved = 0.0
        self.total_latency_saved = 0.0
        
        # Background tasks
        self.optimizer_task = None
        self.running = False
        
    async def start(self):
        """Start inference optimizer"""
        if self.running:
            return
        
        self.running = True
        
        # Register default models
        await self._register_default_models()
        
        # Start all components (disable background tasks for debugging)
        await self.semantic_cache.clear()  # Clear old cache
        # await self.dynamic_batcher.start()  # Disable for debugging
        # await self.cost_monitor.start()  # Disable for debugging
        # await self.model_selector.start()  # Disable for debugging
        # await self.performance_tracker.start()  # Disable for debugging
        
        # Start optimization task
        # self.optimizer_task = asyncio.create_task(self._optimize_continuously())  # Disable for debugging
        
        logger.info("Inference optimizer started")
    
    async def stop(self):
        """Stop inference optimizer"""
        self.running = False
        
        # Stop all components
        # await self.dynamic_batcher.stop()  # Disable for debugging
        # await self.cost_monitor.stop()  # Disable for debugging
        # await self.model_selector.stop()  # Disable for debugging
        # await self.performance_tracker.stop()  # Disable for debugging
        
        if self.optimizer_task:
            self.optimizer_task.cancel()
            try:
                await self.optimizer_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Inference optimizer stopped")
    
    async def optimize_inference(self, request: InferenceRequest) -> InferenceResponse:
        """Main entry point for optimized inference"""
        start_time = time.time()
        self.total_requests += 1
        
        optimization_metrics = {
            "cache_checked": False,
            "cache_hit": False,
            "batched": False,
            "model_switched": False,
            "cost_optimized": False
        }
        
        try:
            # Step 1: Check semantic cache
            cached_result = None
            if self.caching_enabled:
                optimization_metrics["cache_checked"] = True
                cached_result = await self.semantic_cache.get({
                    "id": request.id,
                    "input": request.input_data,
                    "model_id": request.model_id
                })
                
                if cached_result:
                    optimization_metrics["cache_hit"] = True
                    self.cache_hits += 1
                    
                    # Calculate savings
                    latency_saved = 0.1  # Estimated cache latency saving
                    cost_saved = cached_result.cost * 0.8  # 80% cost saving
                    self.total_cost_saved += cost_saved
                    self.total_latency_saved += latency_saved
                    
                    logger.info(f"Cache hit for request {request.id}")
                    
                    return InferenceResponse(
                        request_id=request.id,
                        output=cached_result.response,
                        model_used=cached_result.model_used,
                        latency=time.time() - start_time,
                        cost=0.0,  # Cache hits are free
                        cache_hit=True,
                        batched=False,
                        optimization_metrics=optimization_metrics
                    )
            
            # Step 2: Model selection
            optimal_model = None
            if self.optimization_enabled:
                model_selection = await self.model_selector.select_model(
                    request.model_id, request.requirements
                )
                
                if model_selection.get("model_id") != request.model_id:
                    optimization_metrics["model_switched"] = True
                    optimal_model = model_selection
                    logger.info(f"Model switched for request {request.id}: {request.model_id} -> {model_selection['model_id']}")
            
            # Step 3: Prepare for inference
            model_info = optimal_model or {"id": request.model_id}
            
            # Step 4: Add to dynamic batcher
            batch_future = None
            if self.batching_enabled:
                batch_future = await self.dynamic_batcher.add_request(
                    {
                        "id": request.id,
                        "input": request.input_data,
                        "parameters": request.requirements.get("parameters", {})
                    },
                    model_info
                )
                optimization_metrics["batched"] = True
                self.batched_requests += 1
            
            # Step 5: Execute inference
            if batch_future:
                # Wait for batched result
                batch_result = await batch_future
                result = batch_result
                optimization_metrics["batched"] = True
            else:
                # Execute single inference
                result = await self._execute_single_inference(request, model_info)
            
            # Step 6: Calculate cost
            cost = 0.0
            if self.cost_monitoring_enabled:
                cost = await self.cost_monitor.calculate_cost(model_info, result)
                optimization_metrics["cost_optimized"] = True
            
            # Step 7: Cache result
            if self.caching_enabled and not optimization_metrics["cache_hit"]:
                await self.semantic_cache.store(
                    {
                        "id": request.id,
                        "input": request.input_data,
                        "model_id": request.model_id
                    },
                    result,
                    model_info["id"]
                )
            
            # Step 8: Track performance
            await self.performance_tracker.track_request(
                request.id,
                model_info["id"],
                start_time,
                time.time(),
                success=True,
                additional_metrics={
                    "cost": cost,
                    "tokens_processed": result.get("usage", {}).get("total_tokens", 0)
                }
            )
            
            # Step 9: Update model performance
            await self.model_selector.update_model_performance(
                model_info["id"],
                {
                    "latency": time.time() - start_time,
                    "throughput": 1.0 / max(time.time() - start_time, 0.001),
                    "success": True,
                    "cost": cost
                }
            )
            
            total_latency = time.time() - start_time
            
            return InferenceResponse(
                request_id=request.id,
                output=result,
                model_used=model_info["id"],
                latency=total_latency,
                cost=cost,
                cache_hit=False,
                batched=optimization_metrics["batched"],
                optimization_metrics=optimization_metrics
            )
            
        except Exception as e:
            # Track failed request
            await self.performance_tracker.track_request(
                request.id,
                request.model_id,
                start_time,
                time.time(),
                success=False,
                error_message=str(e)
            )
            
            logger.error(f"Error in optimized inference for request {request.id}: {e}")
            raise
    
    async def _execute_single_inference(self, request: InferenceRequest, 
                                      model_info: Dict[str, Any]) -> Dict[str, Any]:
        """Execute single inference request"""
        # This is a placeholder - in production, this would call the actual model
        await asyncio.sleep(0.05)  # Simulate 50ms inference time
        
        # Generate mock response
        input_text = request.input_data
        if isinstance(input_text, dict):
            input_text = input_text.get("prompt", "")
        
        return {
            "generated_text": f"Optimized response to: {input_text[:100]}...",
            "model_id": model_info["id"],
            "timestamp": datetime.utcnow().isoformat(),
            "usage": {
                "input_tokens": len(input_text.split()),
                "output_tokens": 50,  # Mock token count
                "total_tokens": len(input_text.split()) + 50
            },
            "compute_time": 0.05,
            "memory_usage_gb": 2.0
        }
    
    async def _optimize_continuously(self):
        """Background task for continuous optimization"""
        while self.running:
            try:
                await asyncio.sleep(300)  # Optimize every 5 minutes
                
                # Analyze optimization performance
                await self._analyze_optimization_performance()
                
                # Optimize component parameters
                await self._optimize_component_parameters()
                
                # Generate optimization report
                report = await self.generate_optimization_report()
                
                logger.info(f"Optimization cycle completed: {report}")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in continuous optimization: {e}")
    
    async def _analyze_optimization_performance(self):
        """Analyze optimization performance"""
        # Calculate cache hit rate
        cache_hit_rate = self.cache_hits / max(1, self.total_requests)
        
        # Calculate batching efficiency
        batching_rate = self.batched_requests / max(1, self.total_requests)
        
        logger.info(f"Optimization performance - Cache hit rate: {cache_hit_rate:.2%}, "
                   f"Batching rate: {batching_rate:.2%}, "
                   f"Total cost saved: ${self.total_cost_saved:.4f}, "
                   f"Total latency saved: {self.total_latency_saved:.2f}s")
    
    async def _optimize_component_parameters(self):
        """Optimize component parameters based on performance"""
        # Get performance insights
        cache_stats = await self.semantic_cache.get_stats()
        batcher_stats = await self.dynamic_batcher.get_stats()
        cost_patterns = await self.cost_monitor.get_patterns()
        
        # Optimize cache parameters
        if cache_stats["hit_rate"] < 0.5:  # Low hit rate
            logger.info("Cache hit rate low, considering TTL adjustment")
        
        # Optimize batching parameters
        if batcher_stats["avg_batch_size"] < 2:  # Poor batching
            logger.info("Batching efficiency low, considering wait time adjustment")
        
        # Analyze cost patterns
        if cost_patterns.get("optimization_suggestions"):
            for suggestion in cost_patterns["optimization_suggestions"]:
                logger.info(f"Cost optimization suggestion: {suggestion}")
    
    async def generate_optimization_report(self) -> Dict[str, Any]:
        """Generate comprehensive optimization report"""
        cache_stats = await self.semantic_cache.get_stats()
        batcher_stats = await self.dynamic_batcher.get_stats()
        cost_report = await self.cost_monitor.get_cost_report()
        performance_report = await self.performance_tracker.generate_performance_report()
        
        # Calculate overall optimization metrics
        cache_hit_rate = self.cache_hits / max(1, self.total_requests)
        batching_rate = self.batched_requests / max(1, self.total_requests)
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "overall_metrics": {
                "total_requests": self.total_requests,
                "cache_hits": self.cache_hits,
                "cache_hit_rate": cache_hit_rate,
                "batched_requests": self.batched_requests,
                "batching_rate": batching_rate,
                "total_cost_saved": self.total_cost_saved,
                "total_latency_saved": self.total_latency_saved
            },
            "cache_stats": cache_stats,
            "batcher_stats": batcher_stats,
            "cost_report": cost_report,
            "performance_report": performance_report,
            "optimization_enabled": self.optimization_enabled,
            "caching_enabled": self.caching_enabled,
            "batching_enabled": self.batching_enabled,
            "cost_monitoring_enabled": self.cost_monitoring_enabled
        }
    
    async def get_real_time_stats(self) -> Dict[str, Any]:
        """Get real-time optimization statistics"""
        return {
            "total_requests": self.total_requests,
            "cache_hits": self.cache_hits,
            "batched_requests": self.batched_requests,
            "total_cost_saved": self.total_cost_saved,
            "total_latency_saved": self.total_latency_saved,
            "cache_hit_rate": self.cache_hits / max(1, self.total_requests),
            "batching_rate": self.batched_requests / max(1, self.total_requests),
            "optimization_status": {
                "running": self.running,
                "optimization_enabled": self.optimization_enabled,
                "caching_enabled": self.caching_enabled,
                "batching_enabled": self.batching_enabled,
                "cost_monitoring_enabled": self.cost_monitoring_enabled
            },
            "real_time_metrics": await self.performance_tracker.get_real_time_metrics()
        }
    
    async def reset_stats(self):
        """Reset optimization statistics"""
        self.total_requests = 0
        self.cache_hits = 0
        self.batched_requests = 0
        self.total_cost_saved = 0.0
        self.total_latency_saved = 0.0
        
        await self.semantic_cache.clear()
        logger.info("Optimization statistics reset")
    
    async def _register_default_models(self):
        """Register default models for demo purposes"""
        default_models = {
            "default": {
                "name": "Default Model",
                "latency_ms": 200,
                "cost_per_1k_tokens": 0.002,
                "accuracy": 0.85,
                "max_tokens": 2048,
                "languages": ["en"],
                "domain": "general",
                "capabilities": ["text_generation"]
            },
            "fast": {
                "name": "Fast Model", 
                "latency_ms": 100,
                "cost_per_1k_tokens": 0.001,
                "accuracy": 0.75,
                "max_tokens": 1024,
                "languages": ["en"],
                "domain": "general",
                "capabilities": ["text_generation"]
            },
            "accurate": {
                "name": "Accurate Model",
                "latency_ms": 500,
                "cost_per_1k_tokens": 0.005,
                "accuracy": 0.95,
                "max_tokens": 4096,
                "languages": ["en"],
                "domain": "general", 
                "capabilities": ["text_generation"]
            }
        }
        
        for model_id, model_config in default_models.items():
            self.model_selector.register_model(model_id, model_config)
        
        logger.info(f"Registered {len(default_models)} default models")
    
    async def configure_optimization(self, config: Dict[str, Any]):
        """Configure optimization parameters"""
        self.optimization_enabled = config.get("optimization_enabled", self.optimization_enabled)
        self.caching_enabled = config.get("caching_enabled", self.caching_enabled)
        self.batching_enabled = config.get("batching_enabled", self.batching_enabled)
        self.cost_monitoring_enabled = config.get("cost_monitoring_enabled", self.cost_monitoring_enabled)
        
        logger.info(f"Optimization configuration updated: {config}")
    
    def add_performance_alert_callback(self, callback):
        """Add callback for performance alerts"""
        self.performance_tracker.add_alert_callback(callback)
    
    def add_cost_alert_callback(self, callback):
        """Add callback for cost alerts"""
        self.cost_monitor.add_alert_callback(callback)
    
    async def _execute_single_inference(self, request: InferenceRequest, model_info: Dict[str, Any]) -> Dict[str, Any]:
        """Execute single inference request (mock implementation)"""
        # This is a mock implementation for demo purposes
        # In production, this would call actual model inference
        
        await asyncio.sleep(0.1)  # Simulate inference latency
        
        # Generate mock response based on input
        input_text = request.input_data
        if isinstance(input_text, dict):
            input_text = input_text.get("prompt", str(input_text))
        
        # Simple response generation
        response_text = f"This is a mock response to: {input_text[:100]}..."
        
        return {
            "generated_text": response_text,
            "usage": {
                "total_tokens": len(response_text.split()),
                "prompt_tokens": len(input_text.split()),
                "completion_tokens": len(response_text.split())
            },
            "model": model_info["id"],
            "timestamp": datetime.utcnow().isoformat()
        }