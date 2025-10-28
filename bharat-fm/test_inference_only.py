"""
Test inference engine import
"""

import sys
import os
import asyncio

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

async def test_inference_import():
    print("🇮🇳 Testing Inference Engine Import")
    print("=" * 40)
    
    try:
        print("Testing inference engine import...")
        
        # Try to import the inference engine
        from bharat_fm.core.inference_engine import InferenceEngine
        print("✅ InferenceEngine import successful")
        
        # Test creating inference engine
        print("\nTesting inference engine creation...")
        engine = InferenceEngine()
        print("✅ Inference engine created successfully")
        
        # Test starting inference engine
        print("\nTesting inference engine start...")
        await engine.start()
        print("✅ Inference engine started successfully")
        
        # Test basic prediction
        print("\nTesting basic prediction...")
        request = {
            "id": "test_req_1",
            "input": "Hello, how are you?",
            "model_id": "default",
            "requirements": {}
        }
        
        response = await engine.predict(request)
        print("✅ Basic prediction successful")
        print(f"   Response ID: {response['request_id']}")
        print(f"   Model used: {response['model_used']}")
        print(f"   Latency: {response['latency']:.3f}s")
        
        # Test stopping inference engine
        print("\nTesting inference engine stop...")
        await engine.stop()
        print("✅ Inference engine stopped successfully")
        
        print("\n🎉 Inference engine test passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Inference engine test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_inference_import())
    sys.exit(0 if success else 1)