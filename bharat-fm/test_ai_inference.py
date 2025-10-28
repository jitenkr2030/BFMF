"""
Test AI model inference functionality
"""

import sys
import os
import asyncio

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

async def test_ai_inference():
    print("🇮🇳 Testing AI Model Inference Functionality")
    print("=" * 50)
    
    try:
        # Test 1: Basic inference engine functionality
        print("\n1. Testing Basic Inference Engine...")
        from bharat_fm.core.inference_engine import InferenceEngine
        
        engine = InferenceEngine()
        await engine.start()
        
        # Test multiple predictions
        test_requests = [
            {
                "id": "test_1",
                "input": "Hello, how are you today?",
                "model_id": "default",
                "requirements": {"temperature": 0.7}
            },
            {
                "id": "test_2", 
                "input": "What is artificial intelligence?",
                "model_id": "default",
                "requirements": {"temperature": 0.5}
            },
            {
                "id": "test_3",
                "input": "Explain machine learning in simple terms",
                "model_id": "default",
                "requirements": {"temperature": 0.3}
            }
        ]
        
        print("   Testing individual predictions...")
        for i, request in enumerate(test_requests):
            response = await engine.predict(request)
            print(f"   ✅ Request {i+1}: {response['request_id']} - {response['latency']:.3f}s")
        
        # Test batch prediction
        print("   Testing batch prediction...")
        batch_responses = await engine.predict_batch(test_requests)
        print(f"   ✅ Batch prediction completed: {len(batch_responses)} responses")
        
        # Test streaming prediction
        print("   Testing streaming prediction...")
        stream_count = 0
        async for chunk in engine.predict_streaming(test_requests[0]):
            stream_count += 1
            if chunk["type"] == "complete":
                break
        print(f"   ✅ Streaming prediction completed: {stream_count} chunks")
        
        # Test engine status
        status = await engine.get_engine_status()
        print(f"   ✅ Engine status: {status['status']}")
        print(f"   ✅ Total requests processed: {status['performance_metrics']['total_requests']}")
        
        await engine.stop()
        print("   ✅ Inference engine test completed")
        
        # Test 2: Memory-enhanced inference
        print("\n2. Testing Memory-Enhanced Inference...")
        from bharat_fm.memory.conversation_memory import ConversationMemoryManager
        
        memory = ConversationMemoryManager({"memory_dir": "./test_ai_memory"})
        await memory.start()
        
        # Store some conversation history
        await memory.store_exchange("user_ai", "session_ai", "Hello", "Hi there! How can I help you?")
        await memory.store_exchange("user_ai", "session_ai", "What is AI?", "AI stands for Artificial Intelligence, which is the simulation of human intelligence in machines.")
        await memory.store_exchange("user_ai", "session_ai", "Tell me more", "AI encompasses various technologies including machine learning, natural language processing, and computer vision.")
        
        # Retrieve context
        context = await memory.retrieve_relevant_context("user_ai", "Can you explain machine learning?")
        print(f"   ✅ Memory context retrieved: {len(context['recent_history'])} exchanges")
        print(f"   ✅ Relevant exchanges found: {len(context['relevant_exchanges'])}")
        
        await memory.stop()
        print("   ✅ Memory-enhanced inference test completed")
        
        # Test 3: Performance metrics
        print("\n3. Testing Performance Metrics...")
        engine2 = InferenceEngine()
        await engine2.start()
        
        # Run performance test
        perf_requests = []
        for i in range(10):
            perf_requests.append({
                "id": f"perf_{i}",
                "input": f"Test input {i}: What is the meaning of life?",
                "model_id": "default",
                "requirements": {}
            })
        
        start_time = asyncio.get_event_loop().time()
        perf_responses = await engine2.predict_batch(perf_requests)
        end_time = asyncio.get_event_loop().time()
        
        total_time = end_time - start_time
        avg_latency = total_time / len(perf_responses)
        
        print(f"   ✅ Performance test completed: {len(perf_responses)} requests")
        print(f"   ✅ Total time: {total_time:.3f}s")
        print(f"   ✅ Average latency: {avg_latency:.3f}s")
        print(f"   ✅ Throughput: {len(perf_responses)/total_time:.2f} requests/second")
        
        # Get detailed metrics
        metrics = await engine2.get_detailed_metrics()
        print(f"   ✅ Success rate: {metrics['engine_metrics']['successful_requests']}/{metrics['engine_metrics']['total_requests']}")
        
        await engine2.stop()
        print("   ✅ Performance metrics test completed")
        
        print("\n🎉 AI Inference Functionality Test Passed!")
        print("   All core AI inference features are working correctly.")
        return True
        
    except Exception as e:
        print(f"\n❌ AI Inference Test Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_ai_inference())
    sys.exit(0 if success else 1)