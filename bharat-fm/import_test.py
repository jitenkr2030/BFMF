"""
Import test for Bharat-FM
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def import_test():
    print("🇮🇳 Bharat-FM Import Test")
    print("=" * 40)
    
    try:
        print("Testing basic imports...")
        
        # Test numpy import
        print("Importing numpy...")
        import numpy as np
        print("✅ numpy imported")
        
        # Test sklearn import
        print("Importing sklearn...")
        from sklearn.metrics.pairwise import cosine_similarity
        print("✅ sklearn imported")
        
        # Test internal imports
        print("Importing internal modules...")
        from bharat_fm.memory.conversation_memory import ConversationMemoryManager
        print("✅ ConversationMemoryManager imported")
        
        from bharat_fm.core.inference_engine import InferenceEngine
        print("✅ InferenceEngine imported")
        
        print("\n🎉 All imports successful!")
        return True
        
    except Exception as e:
        print(f"\n❌ Import test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = import_test()
    sys.exit(0 if success else 1)