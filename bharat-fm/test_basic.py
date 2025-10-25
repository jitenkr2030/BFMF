"""
Basic test for Bharat-FM Phase 1
"""

import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from bharat_fm.core.inference_engine import InferenceEngine
from bharat_fm.memory.conversation_memory import ConversationMemoryManager
from bharat_fm.core.chat_engine import ChatEngine

async def test_basic():
    print("🇮🇳 Testing Bharat-FM Phase 1 Basic Functionality")
    print("=" * 60)
    
    try:
        # Test inference engine
        print("\n🚀 Testing Inference Engine...")
        engine = InferenceEngine({"optimization_enabled": True})
        await engine.start()
        print("✅ Inference engine started successfully")
        
        # Test memory manager
        print("\n🧠 Testing Memory Manager...")
        memory = ConversationMemoryManager({"memory_dir": "./test_memory"})
        await memory.start()
        print("✅ Memory manager started successfully")
        
        # Test chat engine
        print("\n🤖 Testing Chat Engine...")
        chat = ChatEngine({
            "inference": {"optimization_enabled": True},
            "memory": {"memory_dir": "./test_chat_memory"}
        })
        await chat.start()
        print("✅ Chat engine started successfully")
        
        # Clean up
        await engine.stop()
        await memory.stop()
        await chat.stop()
        
        print("\n🎉 All basic tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_basic())
    sys.exit(0 if success else 1)