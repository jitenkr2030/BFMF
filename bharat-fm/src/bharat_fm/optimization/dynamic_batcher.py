"""
Dynamic Batcher for Bharat-FM Inference Optimization
Implements intelligent batching of inference requests for optimal throughput
"""

import asyncio
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import numpy as np
from collections import defaultdict, deque
import logging

logger = logging.getLogger(__name__)

@dataclass
class BatchRequest:
    """Represents a batched inference request"""
    requests: List[Dict[str, Any]]
    model_id: str
    batch_id: str
    created_at: datetime
    max_wait_time: float
    max_batch_size: int
    
    def is_ready(self) -> bool:
        """Check if batch is ready for processing"""
        return (
            len(self.requests) >= self.max_batch_size or
            (datetime.utcnow() - self.created_at).total_seconds() >= self.max_wait_time
        )

@dataclass
class BatchStats:
    """Statistics for batch processing"""
    total_batches: int = 0
    total_requests: int = 0
    avg_batch_size: float = 0.0
    avg_wait_time: float = 0.0
    avg_processing_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "total_batches": self.total_batches,
            "total_requests": self.total_requests,
            "avg_batch_size": self.avg_batch_size,
            "avg_wait_time": self.avg_wait_time,
            "avg_processing_time": self.avg_processing_time
        }

class DynamicBatcher:
    """Dynamic batcher for optimizing inference throughput"""
    
    def __init__(self, 
                 max_batch_size: int = 8,
                 max_wait_time: float = 0.1,  # 100ms
                 strategy: str = "adaptive"):  # adaptive, fixed, dynamic
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.strategy = strategy
        
        # Batch management
        self.pending_batches: Dict[str, BatchRequest] = {}
        self.request_queue = asyncio.Queue()
        self.result_callbacks: Dict[str, asyncio.Future] = {}
        
        # Performance tracking
        self.stats = defaultdict(BatchStats)
        self.model_performance = defaultdict(lambda: {
            "avg_latency": 0.0,
            "throughput": 0.0,
            "optimal_batch_size": max_batch_size
        })
        
        # Background tasks
        self.batch_processor_task = None
        self.optimizer_task = None
        self.running = False
        
    async def start(self):
        """Start the dynamic batcher"""
        if self.running:
            return
        
        self.running = True
        self.batch_processor_task = asyncio.create_task(self._process_batches())
        self.optimizer_task = asyncio.create_task(self._optimize_batching())
        
        logger.info("Dynamic batcher started")
    
    async def stop(self):
        """Stop the dynamic batcher"""
        self.running = False
        
        if self.batch_processor_task:
            self.batch_processor_task.cancel()
            try:
                await self.batch_processor_task
            except asyncio.CancelledError:
                pass
        
        if self.optimizer_task:
            self.optimizer_task.cancel()
            try:
                await self.optimizer_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Dynamic batcher stopped")
    
    async def add_request(self, request: Dict[str, Any], model_info: Dict[str, Any]) -> asyncio.Future:
        """Add inference request to batcher"""
        request_id = request.get("id", f"req_{int(time.time() * 1000000)}")
        model_id = model_info.get("id", "default")
        
        # Create future for result
        future = asyncio.Future()
        self.result_callbacks[request_id] = future
        
        # Add to queue
        await self.request_queue.put({
            "request_id": request_id,
            "request": request,
            "model_id": model_id,
            "model_info": model_info,
            "timestamp": datetime.utcnow()
        })
        
        return future
    
    async def _process_batches(self):
        """Process batches in background"""
        while self.running:
            try:
                # Wait for requests with timeout
                try:
                    queue_item = await asyncio.wait_for(
                        self.request_queue.get(), 
                        timeout=0.01  # 10ms timeout
                    )
                except asyncio.TimeoutError:
                    # Check if any pending batches are ready
                    await self._check_pending_batches()
                    continue
                
                # Get or create batch for model
                model_id = queue_item["model_id"]
                if model_id not in self.pending_batches:
                    batch_id = f"batch_{model_id}_{int(time.time() * 1000)}"
                    
                    # Get optimal batch size for model
                    optimal_size = self.model_performance[model_id]["optimal_batch_size"]
                    
                    self.pending_batches[model_id] = BatchRequest(
                        requests=[],
                        model_id=model_id,
                        batch_id=batch_id,
                        created_at=datetime.utcnow(),
                        max_wait_time=self.max_wait_time,
                        max_batch_size=optimal_size
                    )
                
                # Add request to batch
                batch = self.pending_batches[model_id]
                batch.requests.append(queue_item)
                
                # Check if batch is ready
                if batch.is_ready():
                    await self._process_batch(batch)
                    del self.pending_batches[model_id]
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in batch processor: {e}")
    
    async def _check_pending_batches(self):
        """Check if any pending batches are ready for processing"""
        ready_batches = []
        
        for model_id, batch in self.pending_batches.items():
            if batch.is_ready():
                ready_batches.append(batch)
        
        # Process ready batches
        for batch in ready_batches:
            await self._process_batch(batch)
            del self.pending_batches[batch.model_id]
    
    async def _process_batch(self, batch: BatchRequest):
        """Process a single batch"""
        start_time = time.time()
        
        try:
            # Prepare batch input
            batch_input = await self._prepare_batch_input(batch)
            
            # Execute batch inference (this would call the actual model)
            batch_results = await self._execute_batch_inference(batch_input, batch.model_id)
            
            # Distribute results to requesters
            await self._distribute_results(batch, batch_results)
            
            # Update statistics
            processing_time = time.time() - start_time
            await self._update_stats(batch, processing_time)
            
        except Exception as e:
            logger.error(f"Error processing batch {batch.batch_id}: {e}")
            # Fail all requests in batch
            for request_item in batch.requests:
                request_id = request_item["request_id"]
                if request_id in self.result_callbacks:
                    future = self.result_callbacks[request_id]
                    if not future.done():
                        future.set_exception(e)
                    del self.result_callbacks[request_id]
    
    async def _prepare_batch_input(self, batch: BatchRequest) -> Dict[str, Any]:
        """Prepare batch input for model"""
        # Extract input data from all requests
        inputs = []
        for request_item in batch.requests:
            request = request_item["request"]
            input_data = request.get("input", "")
            if isinstance(input_data, dict):
                input_data = input_data.get("prompt", "")
            inputs.append(input_data)
        
        return {
            "model_id": batch.model_id,
            "inputs": inputs,
            "batch_size": len(batch.requests),
            "parameters": batch.requests[0]["request"].get("parameters", {})
        }
    
    async def _execute_batch_inference(self, batch_input: Dict[str, Any], model_id: str) -> List[Any]:
        """Execute batch inference (placeholder)"""
        # In production, this would call the actual model
        # For now, simulate inference with delay
        await asyncio.sleep(0.01)  # Simulate 10ms inference time
        
        # Generate dummy results
        results = []
        for input_text in batch_input["inputs"]:
            results.append({
                "generated_text": f"Response to: {input_text[:50]}...",
                "model_id": model_id,
                "timestamp": datetime.utcnow().isoformat()
            })
        
        return results
    
    async def _distribute_results(self, batch: BatchRequest, batch_results: List[Any]):
        """Distribute batch results to individual requesters"""
        for i, request_item in enumerate(batch.requests):
            request_id = request_item["request_id"]
            if request_id in self.result_callbacks:
                future = self.result_callbacks[request_id]
                if not future.done():
                    future.set_result(batch_results[i])
                del self.result_callbacks[request_id]
    
    async def _update_stats(self, batch: BatchRequest, processing_time: float):
        """Update batch processing statistics"""
        model_id = batch.model_id
        stats = self.stats[model_id]
        
        stats.total_batches += 1
        stats.total_requests += len(batch.requests)
        stats.avg_batch_size = (
            (stats.avg_batch_size * (stats.total_batches - 1) + len(batch.requests)) 
            / stats.total_batches
        )
        
        # Calculate wait time
        wait_times = []
        for request_item in batch.requests:
            wait_time = (batch.created_at - request_item["timestamp"]).total_seconds()
            wait_times.append(wait_time)
        
        avg_wait_time = sum(wait_times) / len(wait_times) if wait_times else 0
        stats.avg_wait_time = (
            (stats.avg_wait_time * (stats.total_batches - 1) + avg_wait_time) 
            / stats.total_batches
        )
        
        stats.avg_processing_time = (
            (stats.avg_processing_time * (stats.total_batches - 1) + processing_time) 
            / stats.total_batches
        )
        
        # Update model performance
        model_perf = self.model_performance[model_id]
        model_perf["avg_latency"] = stats.avg_processing_time
        model_perf["throughput"] = stats.total_requests / max(1, stats.total_batches * stats.avg_processing_time)
    
    async def _optimize_batching(self):
        """Optimize batching parameters based on performance"""
        while self.running:
            try:
                await asyncio.sleep(60)  # Optimize every minute
                
                for model_id, perf in self.model_performance.items():
                    # Adaptive batch size optimization
                    if perf["throughput"] > 0:
                        # Calculate optimal batch size based on latency and throughput
                        current_latency = perf["avg_latency"]
                        current_batch_size = perf["optimal_batch_size"]
                        
                        # Simple heuristic: increase batch size if latency is low
                        if current_latency < 0.05 and current_batch_size < self.max_batch_size:
                            new_batch_size = min(current_batch_size + 1, self.max_batch_size)
                            perf["optimal_batch_size"] = new_batch_size
                            logger.info(f"Optimized batch size for {model_id}: {new_batch_size}")
                        elif current_latency > 0.2 and current_batch_size > 1:
                            new_batch_size = max(current_batch_size - 1, 1)
                            perf["optimal_batch_size"] = new_batch_size
                            logger.info(f"Optimized batch size for {model_id}: {new_batch_size}")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in batch optimizer: {e}")
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get batcher statistics"""
        return {
            "pending_batches": len(self.pending_batches),
            "queue_size": self.request_queue.qsize(),
            "model_stats": {model_id: stats.to_dict() for model_id, stats in self.stats.items()},
            "model_performance": dict(self.model_performance),
            "strategy": self.strategy,
            "max_batch_size": self.max_batch_size,
            "max_wait_time": self.max_wait_time
        }