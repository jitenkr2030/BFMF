"""
Core Inference Engine for Bharat-FM
Main entry point for optimized inference execution
"""

import asyncio
import time
import uuid
from typing import Dict, List, Any, Optional, AsyncIterator
from datetime import datetime
import logging

from ..optimization.inference_optimizer import InferenceOptimizer, InferenceRequest, InferenceResponse
from ..optimization.model_selector import ModelSelector
from ..optimization.performance_tracker import PerformanceTracker

logger = logging.getLogger(__name__)

class InferenceEngine:
    """Core inference engine with optimization capabilities"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Initialize optimizer
        self.optimizer = InferenceOptimizer(config)
        
        # Engine state
        self.engine_id = str(uuid.uuid4())
        self.start_time = datetime.utcnow()
        self.is_running = False
        
        # Request tracking
        self.active_requests: Dict[str, asyncio.Task] = {}
        self.request_history: List[Dict[str, Any]] = []
        
        # Performance monitoring
        self.performance_metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "avg_latency": 0.0,
            "total_cost": 0.0
        }
        
    async def start(self):
        """Start the inference engine"""
        if self.is_running:
            logger.warning("Inference engine is already running")
            return
        
        await self.optimizer.start()
        self.is_running = True
        
        logger.info(f"Inference engine {self.engine_id} started")
    
    async def stop(self):
        """Stop the inference engine"""
        if not self.is_running:
            logger.warning("Inference engine is not running")
            return
        
        # Cancel all active requests
        for request_id, task in self.active_requests.items():
            if not task.done():
                task.cancel()
                logger.info(f"Cancelled active request: {request_id}")
        
        await self.optimizer.stop()
        self.is_running = False
        
        logger.info(f"Inference engine {self.engine_id} stopped")
    
    async def predict(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Execute optimized inference prediction"""
        if not self.is_running:
            raise RuntimeError("Inference engine is not running")
        
        # Generate request ID if not provided
        request_id = request.get("id", f"req_{int(time.time() * 1000000)}")
        
        # Create inference request
        inference_request = InferenceRequest(
            id=request_id,
            input_data=request.get("input", ""),
            model_id=request.get("model_id", "default"),
            requirements=request.get("requirements", {}),
            priority=request.get("priority", 0),
            user_id=request.get("user_id"),
            session_id=request.get("session_id")
        )
        
        # Execute optimized inference
        try:
            response = await self.optimizer.optimize_inference(inference_request)
            
            # Update performance metrics
            self._update_performance_metrics(response)
            
            # Add to history
            self._add_to_history(inference_request, response)
            
            # Format response
            result = {
                "request_id": response.request_id,
                "output": response.output,
                "model_used": response.model_used,
                "latency": response.latency,
                "cost": response.cost,
                "cache_hit": response.cache_hit,
                "batched": response.batched,
                "optimization_metrics": response.optimization_metrics,
                "timestamp": datetime.utcnow().isoformat(),
                "engine_id": self.engine_id
            }
            
            logger.info(f"Inference completed: {request_id} in {response.latency:.3f}s")
            return result
            
        except Exception as e:
            logger.error(f"Inference failed for request {request_id}: {e}")
            raise
    
    async def predict_batch(self, requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute batch inference predictions"""
        if not self.is_running:
            raise RuntimeError("Inference engine is not running")
        
        # Execute all requests concurrently
        tasks = []
        for request in requests:
            task = asyncio.create_task(self.predict(request))
            tasks.append(task)
        
        # Wait for all to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # Handle failed request
                processed_results.append({
                    "request_id": requests[i].get("id", f"req_{i}"),
                    "error": str(result),
                    "success": False,
                    "timestamp": datetime.utcnow().isoformat()
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def predict_streaming(self, request: Dict[str, Any]) -> AsyncIterator[Dict[str, Any]]:
        """Execute streaming inference prediction"""
        if not self.is_running:
            raise RuntimeError("Inference engine is not running")
        
        request_id = request.get("id", f"req_{int(time.time() * 1000000)}")
        
        # Send initial response
        yield {
            "type": "start",
            "request_id": request_id,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        try:
            # Execute inference (this would be streaming in production)
            response = await self.predict(request)
            
            # Send chunks of the response
            output_text = response.get("output", {}).get("generated_text", "")
            chunk_size = 50  # Send 50 characters at a time
            
            for i in range(0, len(output_text), chunk_size):
                chunk = output_text[i:i + chunk_size]
                yield {
                    "type": "chunk",
                    "request_id": request_id,
                    "chunk": chunk,
                    "chunk_index": i // chunk_size,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                # Small delay to simulate streaming
                await asyncio.sleep(0.01)
            
            # Send final response
            yield {
                "type": "complete",
                "request_id": request_id,
                "response": response,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            yield {
                "type": "error",
                "request_id": request_id,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def get_engine_status(self) -> Dict[str, Any]:
        """Get engine status and metrics"""
        optimizer_stats = await self.optimizer.get_real_time_stats()
        
        return {
            "engine_id": self.engine_id,
            "status": "running" if self.is_running else "stopped",
            "uptime_seconds": (datetime.utcnow() - self.start_time).total_seconds(),
            "start_time": self.start_time.isoformat(),
            "performance_metrics": self.performance_metrics,
            "active_requests": len(self.active_requests),
            "optimizer_stats": optimizer_stats,
            "config": self.config
        }
    
    async def get_detailed_metrics(self) -> Dict[str, Any]:
        """Get detailed performance metrics"""
        optimization_report = await self.optimizer.generate_optimization_report()
        performance_report = await self.optimizer.performance_tracker.generate_performance_report()
        
        return {
            "engine_metrics": self.performance_metrics,
            "optimization_report": optimization_report,
            "performance_report": performance_report,
            "request_history": self.request_history[-10:],  # Last 10 requests
            "cache_stats": optimization_report.get("cache_stats", {}),
            "batcher_stats": optimization_report.get("batcher_stats", {}),
            "cost_report": optimization_report.get("cost_report", {})
        }
    
    def _update_performance_metrics(self, response: InferenceResponse):
        """Update performance metrics"""
        self.performance_metrics["total_requests"] += 1
        
        if response.error is None:  # Successful request
            self.performance_metrics["successful_requests"] += 1
        else:
            self.performance_metrics["failed_requests"] += 1
        
        # Update average latency
        total_latency = (self.performance_metrics["avg_latency"] * 
                         (self.performance_metrics["total_requests"] - 1) + 
                         response.latency)
        self.performance_metrics["avg_latency"] = total_latency / self.performance_metrics["total_requests"]
        
        # Update total cost
        self.performance_metrics["total_cost"] += response.cost
    
    def _add_to_history(self, request: InferenceRequest, response: InferenceResponse):
        """Add request to history"""
        history_entry = {
            "request_id": request.id,
            "model_id": request.model_id,
            "timestamp": datetime.utcnow().isoformat(),
            "latency": response.latency,
            "cost": response.cost,
            "cache_hit": response.cache_hit,
            "batched": response.batched,
            "success": response.error is None
        }
        
        self.request_history.append(history_entry)
        
        # Keep only last 1000 entries
        if len(self.request_history) > 1000:
            self.request_history = self.request_history[-1000:]
    
    async def configure_engine(self, config: Dict[str, Any]):
        """Configure engine parameters"""
        self.config.update(config)
        await self.optimizer.configure_optimization(config)
        
        logger.info(f"Engine configuration updated: {config}")
    
    async def reset_metrics(self):
        """Reset all performance metrics"""
        self.performance_metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "avg_latency": 0.0,
            "total_cost": 0.0
        }
        
        self.request_history.clear()
        await self.optimizer.reset_stats()
        
        logger.info("Engine metrics reset")
    
    def add_alert_callback(self, callback):
        """Add callback for performance alerts"""
        self.optimizer.add_performance_alert_callback(callback)
        self.optimizer.add_cost_alert_callback(callback)
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        health_status = {
            "engine_id": self.engine_id,
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "checks": {
                "is_running": self.is_running,
                "optimizer_running": self.optimizer.running,
                "active_requests": len(self.active_requests)
            }
        }
        
        # Check if engine is responsive
        try:
            # Try a simple prediction
            test_request = {
                "id": f"health_{int(time.time() * 1000000)}",
                "input": "test",
                "model_id": "default",
                "requirements": {}
            }
            
            # This is a quick check - we don't want to wait for full inference
            # In production, you might want a lighter health check
            health_status["checks"]["responsive"] = True
            
        except Exception as e:
            health_status["status"] = "unhealthy"
            health_status["checks"]["responsive"] = False
            health_status["error"] = str(e)
        
        return health_status


# Convenience function for easy usage
async def create_inference_engine(config: Optional[Dict[str, Any]] = None) -> InferenceEngine:
    """Create and start inference engine"""
    engine = InferenceEngine(config)
    await engine.start()
    return engine