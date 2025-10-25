"""
Simple test for Bharat-FM components
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    print("Testing imports...")
    
    from bharat_fm.core.inference_engine import InferenceEngine
    print("✅ InferenceEngine imported")
    
    from bharat_fm.memory.conversation_memory import ConversationMemoryManager
    print("✅ ConversationMemoryManager imported")
    
    from bharat_fm.core.chat_engine import ChatEngine
    print("✅ ChatEngine imported")
    
    print("\n🎉 All imports successful!")
    
except Exception as e:
    print(f"❌ Import failed: {e}")
    import traceback
    traceback.print_exc()