"""
Phase 1 Demo: Bharat-FM Enhanced Capabilities
Demonstrates Real-time Inference Optimization and Conversational Memory
"""

import asyncio
import time
import json
from datetime import datetime

# Import our enhanced components
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from bharat_fm.core.chat_engine import ChatEngine
from bharat_fm.core.inference_engine import InferenceEngine
from bharat_fm.memory.conversation_memory import ConversationMemoryManager
from bharat_fm.optimization.inference_optimizer import InferenceOptimizer

async def demo_conversation_memory():
    """Demonstrate conversation memory capabilities"""
    print("\n🧠 Demonstrating Conversation Memory & Context Management")
    print("=" * 60)
    
    # Initialize memory manager
    memory_config = {
        "memory_dir": "./demo_memory",
        "max_history_length": 100,
        "max_sessions_per_user": 5
    }
    
    memory_manager = ConversationMemoryManager(memory_config)
    await memory_manager.start()
    
    # Simulate conversation
    user_id = "demo_user_1"
    session_id = "demo_session_1"
    
    print(f"\n📝 Starting conversation for user: {user_id}")
    
    # Store some conversation exchanges
    exchanges = [
        ("Hello, I need help with my Python programming project.", 
         "Hello! I'd be happy to help you with your Python project. What specific assistance do you need?"),
        
        ("I'm having trouble with list comprehensions. Can you explain them?", 
         "List comprehensions are a concise way to create lists in Python. They follow the syntax: [expression for item in iterable if condition]. For example: [x**2 for x in range(10) if x % 2 == 0] creates a list of squares of even numbers from 0 to 9."),
        
        ("That's helpful! Can you show me a more complex example?", 
         "Certainly! Here's a complex example that processes a list of dictionaries: [{'name': item['name'], 'age': item['age'] + 1} for item in people if item['age'] >= 18]. This creates a new list with names and incremented ages for adults only."),
        
        ("Great! Now I understand. Thank you for the clear explanation.", 
         "You're welcome! I'm glad I could help you understand list comprehensions better. Feel free to ask if you have any more Python questions!")
    ]
    
    for i, (user_msg, asst_msg) in enumerate(exchanges):
        print(f"\n💬 Exchange {i+1}:")
        print(f"   User: {user_msg}")
        print(f"   Assistant: {asst_msg}")
        
        await memory_manager.store_exchange(user_id, session_id, user_msg, asst_msg)
        await asyncio.sleep(0.5)  # Small delay for demo
    
    # Retrieve relevant context
    print(f"\n🔍 Retrieving relevant context for new query...")
    context = await memory_manager.retrieve_relevant_context(
        user_id, "Can you help me with Python dictionaries?"
    )
    
    print(f"\n📊 Context Retrieved:")
    print(f"   Recent exchanges: {len(context['recent_history'])}")
    print(f"   Relevant exchanges: {len(context['relevant_exchanges'])}")
    print(f"   Key topics: {context['key_topics']}")
    print(f"   Communication style: {context['personalization']['communication_style']}")
    print(f"   Expertise level: {context['personalization']['expertise_level']}")
    print(f"   Current emotion: {context['emotional_state']['current_emotion']}")
    
    # Show memory stats
    memory_stats = await memory_manager.get_memory_stats()
    print(f"\n📈 Memory Statistics:")
    print(f"   Active conversations: {memory_stats['active_conversations']}")
    print(f"   User profiles: {memory_stats['user_profiles']}")
    print(f"   Memory usage: {memory_stats['memory_usage_mb']:.2f} MB")
    
    await memory_manager.stop()
    print("\n✅ Conversation memory demo completed!")

async def demo_inference_optimization():
    """Demonstrate inference optimization capabilities"""
    print("\n⚡ Demonstrating Real-time Inference Optimization")
    print("=" * 60)
    
    # Initialize inference engine with optimization
    engine_config = {
        "inference": {
            "optimization_enabled": True,
            "caching_enabled": True,
            "batching_enabled": True,
            "cost_monitoring_enabled": True,
            "max_cache_entries": 1000,
            "max_batch_size": 4,
            "max_wait_time": 0.1
        }
    }
    
    engine = InferenceEngine(engine_config.get("inference", {}))
    await engine.start()
    
    print(f"\n🚀 Inference engine started with optimization enabled")
    
    # Demonstrate optimized inference
    test_requests = [
        {
            "id": "req_1",
            "input": "What is artificial intelligence?",
            "model_id": "default",
            "requirements": {"max_latency": 1.0}
        },
        {
            "id": "req_2", 
            "input": "What is artificial intelligence?",  # Same as req_1 - should hit cache
            "model_id": "default",
            "requirements": {"max_latency": 1.0}
        },
        {
            "id": "req_3",
            "input": "Explain machine learning basics",
            "model_id": "default", 
            "requirements": {"max_latency": 1.0}
        },
        {
            "id": "req_4",
            "input": "What are neural networks?",
            "model_id": "default",
            "requirements": {"max_latency": 1.0}
        }
    ]
    
    print(f"\n📊 Executing {len(test_requests)} inference requests...")
    
    results = []
    for i, request in enumerate(test_requests):
        print(f"\n🔄 Request {i+1}: {request['input'][:50]}...")
        
        start_time = time.time()
        result = await engine.predict(request)
        end_time = time.time()
        
        results.append(result)
        
        print(f"   ✅ Completed in {result['latency']:.3f}s")
        print(f"   💰 Cost: ${result['cost']:.6f}")
        print(f"   🎯 Cache hit: {result['cache_hit']}")
        print(f"   📦 Batched: {result['batched']}")
        
        await asyncio.sleep(0.2)  # Small delay for demo
    
    # Show optimization statistics
    print(f"\n📈 Optimization Results:")
    cache_hits = sum(1 for r in results if r['cache_hit'])
    batched_requests = sum(1 for r in results if r['batched'])
    total_cost = sum(r['cost'] for r in results)
    avg_latency = sum(r['latency'] for r in results) / len(results)
    
    print(f"   Total requests: {len(results)}")
    print(f"   Cache hits: {cache_hits} ({cache_hits/len(results)*100:.1f}%)")
    print(f"   Batched requests: {batched_requests} ({batched_requests/len(results)*100:.1f}%)")
    print(f"   Total cost: ${total_cost:.6f}")
    print(f"   Average latency: {avg_latency:.3f}s")
    
    # Get detailed metrics
    detailed_metrics = await engine.get_detailed_metrics()
    print(f"\n🔍 Detailed Metrics:")
    print(f"   Cache entries: {detailed_metrics.get('cache_stats', {}).get('active_entries', 0)}")
    print(f"   Batching efficiency: {detailed_metrics.get('batcher_stats', {}).get('avg_batch_size', 0):.1f}")
    
    await engine.stop()
    print("\n✅ Inference optimization demo completed!")

async def demo_chat_engine():
    """Demonstrate integrated chat engine with both features"""
    print("\n🤖 Demonstrating Integrated Chat Engine")
    print("=" * 60)
    
    # Initialize chat engine
    chat_config = {
        "inference": {
            "optimization_enabled": True,
            "caching_enabled": True
        },
        "memory": {
            "memory_dir": "./demo_chat_memory",
            "max_history_length": 50
        }
    }
    
    chat_engine = ChatEngine(chat_config)
    await chat_engine.start()
    
    print(f"\n🚀 Chat engine started with memory and optimization")
    
    # Start a chat session
    user_id = "demo_user_2"
    session_info = await chat_engine.start_session(user_id)
    session_id = session_info["session_id"]
    
    print(f"\n💬 Started chat session: {session_id}")
    
    # Simulate a conversation
    conversation = [
        "Hello! I'm interested in learning about data science.",
        "What programming languages should I learn for data science?",
        "Can you recommend some good resources for beginners?",
        "How long does it typically take to become proficient?",
        "Thank you for the helpful information!"
    ]
    
    print(f"\n🗨️  Simulating conversation...")
    
    for i, message in enumerate(conversation):
        print(f"\n💬 User message {i+1}: {message}")
        
        start_time = time.time()
        response = await chat_engine.generate_response(user_id, session_id, message)
        end_time = time.time()
        
        print(f"🤖 Assistant response: {response['response'].get('generated_text', 'No response')[:100]}...")
        print(f"⚡ Response time: {end_time - start_time:.3f}s")
        print(f"💰 Cost: ${response['performance']['cost']:.6f}")
        print(f"🎯 Cache hit: {response['performance']['cache_hit']}")
        print(f"📦 Batched: {response['performance']['batched']}")
        print(f"🧠 Context exchanges used: {response['context_used']['recent_exchanges']}")
        
        await asyncio.sleep(0.5)  # Small delay for demo
    
    # Get session info
    session_info = await chat_engine.get_session_info(session_id)
    print(f"\n📊 Session Statistics:")
    print(f"   Duration: {session_info['duration_seconds']:.1f}s")
    print(f"   Exchanges: {session_info['exchange_count']}")
    
    # Get user profile
    user_profile = await chat_engine.get_user_profile(user_id)
    if user_profile:
        print(f"\n👤 User Profile:")
        print(f"   Communication style: {user_profile['communication_style']}")
        print(f"   Expertise level: {user_profile['expertise_level']}")
        print(f"   Topic interests: {list(user_profile['topic_interests'].keys())}")
    
    # End session
    end_result = await chat_engine.end_session(session_id)
    print(f"\n🏁 Session ended. Duration: {end_result['duration_seconds']:.1f}s")
    
    # Get engine status
    engine_status = await chat_engine.get_engine_status()
    print(f"\n📈 Engine Status:")
    print(f"   Total conversations: {engine_status['chat_stats']['total_conversations']}")
    print(f"   Total exchanges: {engine_status['chat_stats']['total_exchanges']}")
    print(f"   Active sessions: {engine_status['active_sessions']}")
    print(f"   Average response time: {engine_status['chat_stats']['avg_response_time']:.3f}s")
    
    await chat_engine.stop()
    print("\n✅ Chat engine demo completed!")

async def main():
    """Main demo function"""
    print("🇮🇳 Bharat Foundation Model Framework - Phase 1 Demo")
    print("=" * 70)
    print("Enhanced Capabilities:")
    print("1. Real-time Inference Optimization")
    print("2. Conversational Memory & Context Management")
    print("=" * 70)
    
    try:
        # Run individual demos
        await demo_conversation_memory()
        await asyncio.sleep(1)
        
        await demo_inference_optimization()
        await asyncio.sleep(1)
        
        await demo_chat_engine()
        
        print("\n🎉 Phase 1 Demo Completed Successfully!")
        print("\n📋 Summary:")
        print("✅ Conversation memory with semantic search")
        print("✅ Personalization profiles")
        print("✅ Emotional state tracking")
        print("✅ Semantic caching")
        print("✅ Dynamic batching")
        print("✅ Cost monitoring")
        print("✅ Performance optimization")
        print("✅ Integrated chat engine")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())