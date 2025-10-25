"""
Minimal test for Bharat-FM components
"""

import asyncio
import sys
import os
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

async def minimal_test():
    print("🇮🇳 Minimal Bharat-FM Test")
    print("=" * 40)
    
    try:
        # Test basic imports
        print("Testing imports...")
        from bharat_fm.core.inference_engine import InferenceEngine
        from bharat_fm.memory.conversation_memory import ConversationMemoryManager
        print("✅ Imports successful")
        
        # Test memory manager
        print("\nTesting memory manager...")
        memory = ConversationMemoryManager({"memory_dir": "./test_memory"})
        await memory.start()
        print("✅ Memory manager started")
        
        # Test basic memory operations
        print("Testing memory operations...")
        await memory.store_exchange("user1", "session1", "Hello", "Hi there!")
        context = await memory.retrieve_relevant_context("user1", "How are you?")
        print(f"✅ Memory operations successful, found {len(context['recent_history'])} exchanges")
        
        await memory.stop()
        print("✅ Memory manager stopped")
        
        # Test inference engine with minimal config
        print("\nTesting inference engine...")
        engine = InferenceEngine({
            "optimization_enabled": False,  # Disable optimization
            "caching_enabled": False,
            "batching_enabled": False,
            "cost_monitoring_enabled": False
        })
        
        await engine.start()
        print("✅ Inference engine started")
        
        # Test simple inference
        print("Testing simple inference...")
        request = {
            "id": "test_1",
            "input": "What is AI?",
            "model_id": "default",
            "requirements": {}
        }
        
        result = await engine.predict(request)
        print(f"✅ Inference successful, latency: {result['latency']:.3f}s")
        
        await engine.stop()
        print("✅ Inference engine stopped")
        
        print("\n🎉 All minimal tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(minimal_test())
    sys.exit(0 if success else 1)