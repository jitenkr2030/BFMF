"""
Basic test without optimization components
"""

import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

async def basic_test():
    print("🇮🇳 Basic Bharat-FM Test")
    print("=" * 40)
    
    try:
        # Test basic imports
        print("Testing imports...")
        from bharat_fm.memory.conversation_memory import ConversationMemoryManager
        print("✅ Memory manager import successful")
        
        # Test memory manager
        print("\nTesting memory manager...")
        memory = ConversationMemoryManager({"memory_dir": "./test_memory"})
        print("✅ Memory manager created")
        
        await memory.start()
        print("✅ Memory manager started")
        
        # Test basic memory operations
        print("Testing memory operations...")
        await memory.store_exchange("user1", "session1", "Hello", "Hi there!")
        print("✅ Memory store successful")
        
        context = await memory.retrieve_relevant_context("user1", "How are you?")
        print(f"✅ Memory retrieve successful, found {len(context['recent_history'])} exchanges")
        
        await memory.stop()
        print("✅ Memory manager stopped")
        
        print("\n🎉 Basic test passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(basic_test())
    sys.exit(0 if success else 1)