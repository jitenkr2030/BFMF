"""
Real Performance Testing System for Bharat-FM
Honest performance evaluation with real metrics and proper methodology
"""

import time
import asyncio
import statistics
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import threading
import queue
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

logger = logging.getLogger(__name__)

@dataclass
class PerformanceTestResult:
    """Result of a performance test"""
    test_name: str
    operation: str
    success_count: int
    failure_count: int
    total_requests: int
    response_times: List[float]
    throughput: float
    average_latency: float
    median_latency: float
    p95_latency: float
    p99_latency: float
    error_rate: float
    timestamp: datetime
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.response_times:
            self.average_latency = statistics.mean(self.response_times)
            self.median_latency = statistics.median(self.response_times)
            self.p95_latency = statistics.quantiles(self.response_times, n=20)[18]  # 95th percentile
            self.p99_latency = statistics.quantiles(self.response_times, n=100)[98]  # 99th percentile
        self.error_rate = (self.failure_count / self.total_requests * 100) if self.total_requests > 0 else 0

class RealPerformanceTester:
    """Real performance testing with honest metrics"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Test configuration
        self.max_concurrent_requests = self.config.get("max_concurrent_requests", 10)
        self.warmup_requests = self.config.get("warmup_requests", 10)
        self.test_duration = self.config.get("test_duration", 30)  # seconds
        self.request_timeout = self.config.get("request_timeout", 30)  # seconds
        
        # Results storage
        self.test_results: List[PerformanceTestResult] = []
        self.current_test = None
        
        # Performance monitoring
        self.monitoring_enabled = self.config.get("monitoring_enabled", True)
        self.system_stats = {}
    
    async def test_memory_system_performance(self, memory_system) -> PerformanceTestResult:
        """Test memory system performance with real metrics"""
        logger.info("Starting memory system performance test...")
        
        test_name = "Memory System Performance"
        operation = "Memory Operations"
        
        # Warmup
        await self._warmup_memory_system(memory_system)
        
        # Test data
        session_id = "perf_test_session"
        user_id = "perf_test_user"
        test_memories = [
            f"Test memory entry {i} for performance testing."
            for i in range(100)
        ]
        
        response_times = []
        success_count = 0
        failure_count = 0
        
        start_time = time.time()
        end_time = start_time + self.test_duration
        
        async def memory_operation(memory_text: str, op_type: str):
            try:
                op_start = time.time()
                
                if op_type == "add":
                    await memory_system.add_memory(
                        content=memory_text,
                        role="user",
                        session_id=session_id,
                        user_id=user_id,
                        tags=["performance", "test"]
                    )
                elif op_type == "context":
                    await memory_system.get_context(session_id, max_tokens=1000)
                elif op_type == "search":
                    await memory_system.search_memories(
                        query="performance test",
                        user_id=user_id,
                        limit=10
                    )
                
                op_end = time.time()
                return op_end - op_start, True
                
            except Exception as e:
                logger.error(f"Memory operation failed: {e}")
                return 0, False
        
        # Run concurrent memory operations
        tasks = []
        request_count = 0
        
        while time.time() < end_time:
            # Create batch of concurrent requests
            batch_tasks = []
            for i in range(min(self.max_concurrent_requests, 5)):  # Limit concurrency for memory ops
                if time.time() >= end_time:
                    break
                
                memory_text = test_memories[request_count % len(test_memories)]
                op_type = ["add", "context", "search"][request_count % 3]
                
                task = memory_operation(memory_text, op_type)
                batch_tasks.append(task)
                request_count += 1
            
            if batch_tasks:
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                for result in batch_results:
                    if isinstance(result, Exception):
                        failure_count += 1
                    else:
                        latency, success = result
                        response_times.append(latency)
                        if success:
                            success_count += 1
                        else:
                            failure_count += 1
            
            # Small delay to prevent overwhelming the system
            await asyncio.sleep(0.01)
        
        total_time = time.time() - start_time
        throughput = success_count / total_time if total_time > 0 else 0
        
        result = PerformanceTestResult(
            test_name=test_name,
            operation=operation,
            success_count=success_count,
            failure_count=failure_count,
            total_requests=success_count + failure_count,
            response_times=response_times,
            throughput=throughput,
            average_latency=0,
            median_latency=0,
            p95_latency=0,
            p99_latency=0,
            error_rate=0,
            timestamp=datetime.utcnow(),
            metadata={
                "test_duration": total_time,
                "concurrent_requests": self.max_concurrent_requests,
                "operation_types": ["add", "context", "search"],
                "memory_entries_created": success_count // 3  # Rough estimate
            }
        )
        
        self.test_results.append(result)
        
        logger.info(f"Memory System Test: {success_count} successes, {failure_count} failures, "
                   f"throughput: {throughput:.2f} ops/sec, avg latency: {result.average_latency:.3f}s")
        
        return result
    
    async def test_text_processing_performance(self, text_processor) -> PerformanceTestResult:
        """Test text processing performance with real metrics"""
        logger.info("Starting text processing performance test...")
        
        test_name = "Text Processing Performance"
        operation = "Text Processing Operations"
        
        # Warmup
        await self._warmup_text_processing(text_processor)
        
        # Test data
        test_texts = [
            "Hello, world! This is a test text for performance evaluation.",
            "नमस्ते दुनिया! यह प्रदर्शन मूल्यांकन के लिए एक परीक्षण पाठ है।",
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning and artificial intelligence are fascinating fields.",
            "Performance testing requires careful methodology and honest reporting."
        ] * 20  # Repeat for more test data
        
        response_times = []
        success_count = 0
        failure_count = 0
        
        start_time = time.time()
        end_time = start_time + self.test_duration
        
        async def text_operation(text: str, op_type: str):
            try:
                op_start = time.time()
                
                if op_type == "tokenize":
                    text_processor.tokenize(text, tokenizer_type="indian")
                elif op_type == "preprocess":
                    text_processor.preprocess_text(text)
                elif op_type == "detect_language":
                    text_processor.detect_language(text)
                elif op_type == "batch":
                    text_processor.tokenize_batch([text] * 5, tokenizer_type="indian")
                
                op_end = time.time()
                return op_end - op_start, True
                
            except Exception as e:
                logger.error(f"Text processing operation failed: {e}")
                return 0, False
        
        # Run concurrent text processing operations
        request_count = 0
        
        while time.time() < end_time:
            # Create batch of concurrent requests
            batch_tasks = []
            for i in range(min(self.max_concurrent_requests, 10)):
                if time.time() >= end_time:
                    break
                
                text = test_texts[request_count % len(test_texts)]
                op_type = ["tokenize", "preprocess", "detect_language", "batch"][request_count % 4]
                
                task = text_operation(text, op_type)
                batch_tasks.append(task)
                request_count += 1
            
            if batch_tasks:
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                for result in batch_results:
                    if isinstance(result, Exception):
                        failure_count += 1
                    else:
                        latency, success = result
                        response_times.append(latency)
                        if success:
                            success_count += 1
                        else:
                            failure_count += 1
            
            # Small delay to prevent overwhelming the system
            await asyncio.sleep(0.001)
        
        total_time = time.time() - start_time
        throughput = success_count / total_time if total_time > 0 else 0
        
        result = PerformanceTestResult(
            test_name=test_name,
            operation=operation,
            success_count=success_count,
            failure_count=failure_count,
            total_requests=success_count + failure_count,
            response_times=response_times,
            throughput=throughput,
            average_latency=0,
            median_latency=0,
            p95_latency=0,
            p99_latency=0,
            error_rate=0,
            timestamp=datetime.utcnow(),
            metadata={
                "test_duration": total_time,
                "concurrent_requests": self.max_concurrent_requests,
                "operation_types": ["tokenize", "preprocess", "detect_language", "batch"],
                "unique_texts": len(test_texts)
            }
        )
        
        self.test_results.append(result)
        
        logger.info(f"Text Processing Test: {success_count} successes, {failure_count} failures, "
                   f"throughput: {throughput:.2f} ops/sec, avg latency: {result.average_latency:.3f}s")
        
        return result
    
    def test_neural_network_performance(self, model) -> PerformanceTestResult:
        """Test neural network performance with real metrics"""
        logger.info("Starting neural network performance test...")
        
        test_name = "Neural Network Performance"
        operation = "Model Inference"
        
        # Warmup
        self._warmup_neural_network(model)
        
        # Create test data
        import torch
        batch_size = 4
        seq_length = 64
        d_model = 64  # Small model for testing
        
        test_inputs = []
        for _ in range(50):  # Create 50 test inputs
            input_tensor = torch.randn(batch_size, seq_length, d_model)
            test_inputs.append(input_tensor)
        
        response_times = []
        success_count = 0
        failure_count = 0
        
        start_time = time.time()
        end_time = start_time + self.test_duration
        
        def model_inference(input_tensor):
            try:
                op_start = time.time()
                
                with torch.no_grad():
                    output = model(input_tensor)
                
                op_end = time.time()
                return op_end - op_start, True, output.shape
                
            except Exception as e:
                logger.error(f"Model inference failed: {e}")
                return 0, False, None
        
        # Run inference operations
        request_count = 0
        
        with ThreadPoolExecutor(max_workers=self.max_concurrent_requests) as executor:
            futures = []
            
            while time.time() < end_time:
                if len(futures) >= self.max_concurrent_requests:
                    # Wait for some futures to complete
                    completed_futures = []
                    for future in futures:
                        if future.done():
                            completed_futures.append(future)
                    
                    for future in completed_futures:
                        latency, success, output_shape = future.result()
                        response_times.append(latency)
                        if success:
                            success_count += 1
                        else:
                            failure_count += 1
                        futures.remove(future)
                
                # Submit new requests
                if time.time() < end_time and len(futures) < self.max_concurrent_requests:
                    input_tensor = test_inputs[request_count % len(test_inputs)]
                    future = executor.submit(model_inference, input_tensor)
                    futures.append(future)
                    request_count += 1
                else:
                    # Small delay
                    time.sleep(0.001)
            
            # Wait for remaining futures
            for future in futures:
                latency, success, output_shape = future.result()
                response_times.append(latency)
                if success:
                    success_count += 1
                else:
                    failure_count += 1
        
        total_time = time.time() - start_time
        throughput = success_count / total_time if total_time > 0 else 0
        
        result = PerformanceTestResult(
            test_name=test_name,
            operation=operation,
            success_count=success_count,
            failure_count=failure_count,
            total_requests=success_count + failure_count,
            response_times=response_times,
            throughput=throughput,
            average_latency=0,
            median_latency=0,
            p95_latency=0,
            p99_latency=0,
            error_rate=0,
            timestamp=datetime.utcnow(),
            metadata={
                "test_duration": total_time,
                "concurrent_requests": self.max_concurrent_requests,
                "batch_size": batch_size,
                "seq_length": seq_length,
                "d_model": d_model,
                "model_parameters": sum(p.numel() for p in model.parameters()) if hasattr(model, 'parameters') else 0
            }
        )
        
        self.test_results.append(result)
        
        logger.info(f"Neural Network Test: {success_count} successes, {failure_count} failures, "
                   f"throughput: {throughput:.2f} inferences/sec, avg latency: {result.average_latency:.3f}s")
        
        return result
    
    async def test_integration_performance(self, components: Dict[str, Any]) -> PerformanceTestResult:
        """Test integration performance with real metrics"""
        logger.info("Starting integration performance test...")
        
        test_name = "Integration Performance"
        operation = "End-to-End Pipeline"
        
        # Warmup
        await self._warmup_integration(components)
        
        response_times = []
        success_count = 0
        failure_count = 0
        
        start_time = time.time()
        end_time = start_time + self.test_duration
        
        async def integration_pipeline(test_text: str):
            try:
                op_start = time.time()
                
                # Step 1: Text processing
                if "text_processor" in components:
                    processed = components["text_processor"].preprocess_text(test_text)
                    tokenized = components["text_processor"].tokenize(processed, tokenizer_type="indian")
                
                # Step 2: Memory operations
                if "memory_system" in components:
                    session_id = f"integration_test_{int(time.time())}"
                    user_id = "integration_test_user"
                    
                    await components["memory_system"].add_memory(
                        content=test_text,
                        role="user",
                        session_id=session_id,
                        user_id=user_id,
                        tags=["integration", "performance"]
                    )
                    
                    context = await components["memory_system"].get_context(session_id, max_tokens=500)
                
                # Step 3: Neural network inference (if available)
                if "model" in components:
                    import torch
                    # Create dummy input for model
                    dummy_input = torch.randn(1, 32, 64)  # batch_size=1, seq_len=32, d_model=64
                    with torch.no_grad():
                        output = components["model"](dummy_input)
                
                op_end = time.time()
                return op_end - op_start, True
                
            except Exception as e:
                logger.error(f"Integration pipeline failed: {e}")
                return 0, False
        
        # Test data
        test_texts = [
            "The weather is beautiful today.",
            "I love learning about artificial intelligence.",
            "Performance testing is important for honest software development.",
            "Integration testing ensures components work together properly."
        ]
        
        # Run integration pipelines
        request_count = 0
        
        while time.time() < end_time:
            # Create batch of concurrent requests
            batch_tasks = []
            for i in range(min(self.max_concurrent_requests, 5)):  # Limit concurrency for integration
                if time.time() >= end_time:
                    break
                
                test_text = test_texts[request_count % len(test_texts)]
                task = integration_pipeline(test_text)
                batch_tasks.append(task)
                request_count += 1
            
            if batch_tasks:
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                for result in batch_results:
                    if isinstance(result, Exception):
                        failure_count += 1
                    else:
                        latency, success = result
                        response_times.append(latency)
                        if success:
                            success_count += 1
                        else:
                            failure_count += 1
            
            # Small delay to prevent overwhelming the system
            await asyncio.sleep(0.01)
        
        total_time = time.time() - start_time
        throughput = success_count / total_time if total_time > 0 else 0
        
        result = PerformanceTestResult(
            test_name=test_name,
            operation=operation,
            success_count=success_count,
            failure_count=failure_count,
            total_requests=success_count + failure_count,
            response_times=response_times,
            throughput=throughput,
            average_latency=0,
            median_latency=0,
            p95_latency=0,
            p99_latency=0,
            error_rate=0,
            timestamp=datetime.utcnow(),
            metadata={
                "test_duration": total_time,
                "concurrent_requests": self.max_concurrent_requests,
                "components_tested": list(components.keys()),
                "pipeline_steps": ["text_processing", "memory_operations", "model_inference"]
            }
        )
        
        self.test_results.append(result)
        
        logger.info(f"Integration Test: {success_count} successes, {failure_count} failures, "
                   f"throughput: {throughput:.2f} pipelines/sec, avg latency: {result.average_latency:.3f}s")
        
        return result
    
    async def _warmup_memory_system(self, memory_system):
        """Warmup memory system"""
        logger.info("Warming up memory system...")
        
        for i in range(self.warmup_requests):
            try:
                await memory_system.add_memory(
                    content=f"Warmup memory {i}",
                    role="user",
                    session_id="warmup_session",
                    user_id="warmup_user",
                    tags=["warmup"]
                )
                await asyncio.sleep(0.001)  # Small delay
            except Exception as e:
                logger.warning(f"Warmup memory operation failed: {e}")
    
    async def _warmup_text_processing(self, text_processor):
        """Warmup text processing"""
        logger.info("Warming up text processing...")
        
        test_texts = ["Warmup text 1", "Warmup text 2", "Warmup text 3"]
        
        for i in range(self.warmup_requests):
            try:
                text = test_texts[i % len(test_texts)]
                text_processor.tokenize(text, tokenizer_type="indian")
                text_processor.preprocess_text(text)
                text_processor.detect_language(text)
                await asyncio.sleep(0.001)  # Small delay
            except Exception as e:
                logger.warning(f"Warmup text processing failed: {e}")
    
    def _warmup_neural_network(self, model):
        """Warmup neural network"""
        logger.info("Warming up neural network...")
        
        try:
            import torch
            dummy_input = torch.randn(1, 32, 64)
            
            with torch.no_grad():
                for _ in range(self.warmup_requests):
                    output = model(dummy_input)
        except Exception as e:
            logger.warning(f"Warmup neural network failed: {e}")
    
    async def _warmup_integration(self, components: Dict[str, Any]):
        """Warmup integration pipeline"""
        logger.info("Warming up integration pipeline...")
        
        for i in range(self.warmup_requests):
            try:
                test_text = f"Warmup integration text {i}"
                
                if "text_processor" in components:
                    components["text_processor"].preprocess_text(test_text)
                
                if "memory_system" in components:
                    await components["memory_system"].add_memory(
                        content=test_text,
                        role="user",
                        session_id="warmup_integration",
                        user_id="warmup_user",
                        tags=["warmup"]
                    )
                
                await asyncio.sleep(0.001)
            except Exception as e:
                logger.warning(f"Warmup integration failed: {e}")
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        if not self.test_results:
            return {"error": "No test results available"}
        
        report = {
            "test_summary": {
                "total_tests": len(self.test_results),
                "total_requests": sum(r.total_requests for r in self.test_results),
                "total_successes": sum(r.success_count for r in self.test_results),
                "total_failures": sum(r.failure_count for r in self.test_results),
                "overall_success_rate": (sum(r.success_count for r in self.test_results) / 
                                        max(sum(r.total_requests for r in self.test_results), 1)) * 100,
                "average_throughput": statistics.mean([r.throughput for r in self.test_results]),
                "average_latency": statistics.mean([r.average_latency for r in self.test_results])
            },
            "test_details": [],
            "performance_analysis": {},
            "recommendations": []
        }
        
        # Add test details
        for result in self.test_results:
            test_detail = {
                "test_name": result.test_name,
                "operation": result.operation,
                "success_count": result.success_count,
                "failure_count": result.failure_count,
                "total_requests": result.total_requests,
                "throughput": result.throughput,
                "average_latency": result.average_latency,
                "median_latency": result.median_latency,
                "p95_latency": result.p95_latency,
                "p99_latency": result.p99_latency,
                "error_rate": result.error_rate,
                "timestamp": result.timestamp.isoformat(),
                "metadata": result.metadata
            }
            report["test_details"].append(test_detail)
        
        # Performance analysis
        throughput_values = [r.throughput for r in self.test_results]
        latency_values = [r.average_latency for r in self.test_results]
        error_rates = [r.error_rate for r in self.test_results]
        
        report["performance_analysis"] = {
            "throughput_analysis": {
                "min_throughput": min(throughput_values),
                "max_throughput": max(throughput_values),
                "avg_throughput": statistics.mean(throughput_values),
                "throughput_std": statistics.stdev(throughput_values) if len(throughput_values) > 1 else 0
            },
            "latency_analysis": {
                "min_latency": min(latency_values),
                "max_latency": max(latency_values),
                "avg_latency": statistics.mean(latency_values),
                "latency_std": statistics.stdev(latency_values) if len(latency_values) > 1 else 0
            },
            "reliability_analysis": {
                "min_error_rate": min(error_rates),
                "max_error_rate": max(error_rates),
                "avg_error_rate": statistics.mean(error_rates),
                "tests_with_zero_errors": sum(1 for r in self.test_results if r.error_rate == 0)
            }
        }
        
        # Generate recommendations
        report["recommendations"] = self._generate_recommendations(report["performance_analysis"])
        
        return report
    
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate performance recommendations"""
        recommendations = []
        
        # Throughput recommendations
        avg_throughput = analysis["throughput_analysis"]["avg_throughput"]
        if avg_throughput < 10:
            recommendations.append("Consider optimizing algorithms for better throughput")
        elif avg_throughput > 100:
            recommendations.append("Excellent throughput performance achieved")
        
        # Latency recommendations
        avg_latency = analysis["latency_analysis"]["avg_latency"]
        if avg_latency > 1.0:
            recommendations.append("High latency detected - consider optimization strategies")
        elif avg_latency < 0.1:
            recommendations.append("Excellent low latency performance")
        
        # Reliability recommendations
        avg_error_rate = analysis["reliability_analysis"]["avg_error_rate"]
        if avg_error_rate > 5:
            recommendations.append("High error rate - improve error handling and stability")
        elif avg_error_rate < 1:
            recommendations.append("Excellent reliability with low error rate")
        
        # General recommendations
        recommendations.append("Continue monitoring performance trends")
        recommendations.append("Consider load testing for production scenarios")
        recommendations.append("Implement performance regression testing")
        
        return recommendations
    
    def save_results(self, filename: str = "real_performance_test_results.json"):
        """Save test results to file"""
        report = self.generate_performance_report()
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Performance test results saved to {filename}")
    
    def print_summary(self):
        """Print performance test summary"""
        if not self.test_results:
            logger.info("No test results available")
            return
        
        logger.info("\n" + "="*60)
        logger.info("REAL PERFORMANCE TEST SUMMARY")
        logger.info("="*60)
        
        total_requests = sum(r.total_requests for r in self.test_results)
        total_successes = sum(r.success_count for r in self.test_results)
        total_failures = sum(r.failure_count for r in self.test_results)
        overall_success_rate = (total_successes / max(total_requests, 1)) * 100
        
        logger.info(f"Total Requests: {total_requests}")
        logger.info(f"Total Successes: {total_successes}")
        logger.info(f"Total Failures: {total_failures}")
        logger.info(f"Overall Success Rate: {overall_success_rate:.1f}%")
        
        logger.info(f"\nTest Results:")
        for result in self.test_results:
            logger.info(f"  {result.test_name}:")
            logger.info(f"    Throughput: {result.throughput:.2f} ops/sec")
            logger.info(f"    Avg Latency: {result.average_latency:.3f}s")
            logger.info(f"    Error Rate: {result.error_rate:.1f}%")
            logger.info(f"    Success Rate: {result.success_count}/{result.total_requests} "
                       f"({result.success_count/max(result.total_requests, 1)*100:.1f}%)")
        
        logger.info("="*60)


# Factory function for easy usage
def create_real_performance_tester(config: Optional[Dict[str, Any]] = None) -> RealPerformanceTester:
    """Create a real performance tester"""
    return RealPerformanceTester(config)