"""
Debug version of the demo to see where it hangs
"""

import asyncio
import time
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from bharat_fm.core.inference_engine import InferenceEngine

async def debug_inference():
    print("🚀 Debugging Inference Engine")
    print("=" * 40)
    
    try:
        # Initialize inference engine
        print("Creating inference engine...")
        engine = InferenceEngine({"optimization_enabled": True})
        
        print("Starting inference engine...")
        await engine.start()
        print("✅ Inference engine started")
        
        # Simple test request
        print("\nCreating test request...")
        request = {
            "id": "test_1",
            "input": "What is artificial intelligence?",
            "model_id": "default",
            "requirements": {"max_latency": 1.0}
        }
        print("✅ Test request created")
        
        print("\nExecuting prediction...")
        start_time = time.time()
        
        result = await engine.predict(request)
        
        end_time = time.time()
        print(f"✅ Prediction completed in {end_time - start_time:.3f}s")
        print(f"Result: {result}")
        
        # Clean up
        print("\nStopping engine...")
        await engine.stop()
        print("✅ Engine stopped")
        
        print("\n🎉 Debug test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Debug test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_inference())