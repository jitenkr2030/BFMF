"""
Synchronous test for Bharat-FM
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def sync_test():
    print("🇮🇳 Synchronous Bharat-FM Test")
    print("=" * 40)
    
    try:
        # Test basic imports
        print("Testing imports...")
        from bharat_fm.memory.conversation_memory import ConversationMemoryManager
        from bharat_fm.core.inference_engine import InferenceEngine
        print("✅ Imports successful")
        
        # Test memory manager creation (without starting)
        print("\nTesting memory manager creation...")
        memory = ConversationMemoryManager({"memory_dir": "./test_memory"})
        print("✅ Memory manager created successfully")
        
        # Test inference engine creation (without starting)
        print("\nTesting inference engine creation...")
        engine = InferenceEngine({"optimization_enabled": False})
        print("✅ Inference engine created successfully")
        
        print("\n🎉 Synchronous test passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = sync_test()
    sys.exit(0 if success else 1)