"""
Test conversation memory module import
"""

import sys
import os
import asyncio

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

async def test_memory_import():
    print("🇮🇳 Testing Conversation Memory Import")
    print("=" * 40)
    
    try:
        print("Testing conversation memory import...")
        
        # Try to import just the conversation memory module
        from bharat_fm.memory.conversation_memory import ConversationMemoryManager
        print("✅ ConversationMemoryManager import successful")
        
        # Test creating memory manager
        print("\nTesting memory manager creation...")
        memory = ConversationMemoryManager({"memory_dir": "./test_memory"})
        print("✅ Memory manager created successfully")
        
        # Test starting memory manager
        print("\nTesting memory manager start...")
        await memory.start()
        print("✅ Memory manager started successfully")
        
        # Test basic memory operations
        print("\nTesting memory operations...")
        await memory.store_exchange("user1", "session1", "Hello", "Hi there!")
        print("✅ Memory store successful")
        
        context = await memory.retrieve_relevant_context("user1", "How are you?")
        print(f"✅ Memory retrieve successful, found {len(context['recent_history'])} exchanges")
        
        # Test stopping memory manager
        print("\nTesting memory manager stop...")
        await memory.stop()
        print("✅ Memory manager stopped successfully")
        
        print("\n🎉 Memory import test passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Memory import test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_memory_import())
    sys.exit(0 if success else 1)