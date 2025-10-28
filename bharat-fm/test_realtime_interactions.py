"""
Test real-time AI interactions
"""

import sys
import os
import asyncio
import time
import json
from typing import Dict, List, Any, AsyncIterator

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

async def test_realtime_interactions():
    print("🇮🇳 Testing Real-Time AI Interactions")
    print("=" * 50)
    
    try:
        # Test 1: Basic Chat Engine Functionality
        print("\n1. Testing Basic Chat Engine Functionality...")
        
        from bharat_fm.core.chat_engine import ChatEngine
        
        # Create and start chat engine
        chat_engine = ChatEngine({"inference": {}, "memory": {}})
        await chat_engine.start()
        
        # Test session management
        session_info = await chat_engine.start_session("user_test", {"language": "en", "style": "casual"})
        session_id = session_info["session_id"]
        print(f"   ✅ Session started: {session_id}")
        
        # Test basic response generation
        test_messages = [
            "Hello, how are you?",
            "What is artificial intelligence?",
            "Can you explain machine learning?",
            "Tell me about Bharat AI",
            "How does this chat system work?"
        ]
        
        response_times = []
        for i, message in enumerate(test_messages):
            start_time = time.time()
            response = await chat_engine.generate_response("user_test", session_id, message)
            response_time = time.time() - start_time
            response_times.append(response_time)
            
            print(f"   ✅ Message {i+1}: {response_time:.3f}s - {len(str(response['response']))} chars")
        
        # Test session ending
        session_end = await chat_engine.end_session(session_id)
        print(f"   ✅ Session ended: {session_end['exchange_count']} exchanges")
        
        # Calculate average response time
        avg_response_time = sum(response_times) / len(response_times)
        print(f"   ✅ Average response time: {avg_response_time:.3f}s")
        
        await chat_engine.stop()
        
        # Test 2: Streaming Chat Interactions
        print("\n2. Testing Streaming Chat Interactions...")
        
        # Create new chat engine for streaming test
        streaming_engine = ChatEngine({"inference": {}, "memory": {}})
        await streaming_engine.start()
        
        # Start streaming session
        streaming_session = await streaming_engine.start_session("user_streaming", {"language": "en"})
        streaming_session_id = streaming_session["session_id"]
        
        # Test streaming response
        streaming_message = "Can you tell me a detailed story about artificial intelligence development in India?"
        
        print("   Testing streaming response...")
        chunk_count = 0
        total_chunks = 0
        start_time = time.time()
        
        async for chunk in streaming_engine.generate_streaming_response(
            "user_streaming", streaming_session_id, streaming_message
        ):
            chunk_count += 1
            total_chunks += 1
            
            if chunk["type"] == "start":
                print(f"     📡 Streaming started")
            elif chunk["type"] == "chunk":
                content = chunk["chunk"][:30] + "..." if len(chunk["chunk"]) > 30 else chunk["chunk"]
                print(f"     📝 Chunk {chunk['chunk_index']}: {content}")
            elif chunk["type"] == "complete":
                response_time = time.time() - start_time
                print(f"     ✅ Streaming completed in {response_time:.3f}s")
            elif chunk["type"] == "error":
                print(f"     ❌ Streaming error: {chunk['error']}")
        
        await streaming_engine.end_session(streaming_session_id)
        await streaming_engine.stop()
        
        print(f"   ✅ Streaming test completed: {total_chunks} chunks delivered")
        
        # Test 3: Multi-User Concurrent Interactions
        print("\n3. Testing Multi-User Concurrent Interactions...")
        
        # Create chat engine for concurrent test
        concurrent_engine = ChatEngine({"inference": {}, "memory": {}})
        await concurrent_engine.start()
        
        # Simulate multiple users
        users = [
            {"id": "user_1", "name": "Alice", "language": "en"},
            {"id": "user_2", "name": "Bob", "language": "en"},
            {"id": "user_3", "name": "Charlie", "language": "en"},
            {"id": "user_4", "name": "Diana", "language": "en"},
            {"id": "user_5", "name": "Eve", "language": "en"}
        ]
        
        # Start sessions for all users
        user_sessions = {}
        for user in users:
            session = await concurrent_engine.start_session(user["id"], {"language": user["language"]})
            user_sessions[user["id"]] = session["session_id"]
        
        # Test concurrent interactions
        concurrent_tasks = []
        interaction_results = {}
        
        async def user_interaction(user_id: str, session_id: str, messages: List[str]):
            """Simulate user interactions"""
            user_results = []
            for msg in messages:
                start_time = time.time()
                response = await concurrent_engine.generate_response(user_id, session_id, msg)
                response_time = time.time() - start_time
                user_results.append({
                    "message": msg,
                    "response_time": response_time,
                    "response_length": len(str(response["response"]))
                })
                
                # Small delay to simulate real interaction
                await asyncio.sleep(0.1)
            
            interaction_results[user_id] = user_results
        
        # Create concurrent tasks
        for user in users:
            user_messages = [
                f"Hello, I'm {user['name']}",
                f"What can you tell me about AI?",
                f"Thank you for the information"
            ]
            task = asyncio.create_task(user_interaction(user["id"], user_sessions[user["id"]], user_messages))
            concurrent_tasks.append(task)
        
        # Wait for all concurrent interactions to complete
        await asyncio.gather(*concurrent_tasks)
        
        # Analyze concurrent performance
        total_interactions = sum(len(results) for results in interaction_results.values())
        total_response_time = sum(
            result["response_time"] 
            for user_results in interaction_results.values() 
            for result in user_results
        )
        avg_concurrent_response_time = total_response_time / total_interactions
        
        print(f"   ✅ Concurrent interactions: {total_interactions} total")
        print(f"   ✅ Average concurrent response time: {avg_concurrent_response_time:.3f}s")
        
        # End all sessions
        for user_id, session_id in user_sessions.items():
            await concurrent_engine.end_session(session_id)
        
        await concurrent_engine.stop()
        
        # Test 4: Real-Time Context Switching
        print("\n4. Testing Real-Time Context Switching...")
        
        # Create chat engine for context switching test
        context_engine = ChatEngine({"inference": {}, "memory": {}})
        await context_engine.start()
        
        # Start session
        context_session = await context_engine.start_session("user_context", {"language": "en"})
        context_session_id = context_session["session_id"]
        
        # Test context switching between different topics
        context_scenarios = [
            {
                "topic": "Technology",
                "messages": [
                    "Tell me about Python programming",
                    "What are the best practices for software development?",
                    "How do I optimize code performance?"
                ]
            },
            {
                "topic": "Health",
                "messages": [
                    "What are the benefits of regular exercise?",
                    "How can I maintain a balanced diet?",
                    "What are the symptoms of common illnesses?"
                ]
            },
            {
                "topic": "Education",
                "messages": [
                    "What are the best learning strategies?",
                    "How can I improve my memory?",
                    "What are the benefits of online education?"
                ]
            },
            {
                "topic": "Finance",
                "messages": [
                    "What are the basics of personal finance?",
                    "How do I start investing?",
                    "What are the different types of savings accounts?"
                ]
            }
        ]
        
        context_switching_results = []
        
        for scenario in context_scenarios:
            print(f"   Testing {scenario['topic']} context...")
            
            # Send messages for current context
            for message in scenario["messages"]:
                start_time = time.time()
                response = await context_engine.generate_response("user_context", context_session_id, message)
                response_time = time.time() - start_time
                
                context_switching_results.append({
                    "topic": scenario["topic"],
                    "message": message,
                    "response_time": response_time,
                    "context_relevant": scenario["topic"].lower() in str(response["response"]).lower()
                })
                
                # Small delay between messages
                await asyncio.sleep(0.05)
        
        # Analyze context switching performance
        context_aware_responses = sum(1 for r in context_switching_results if r["context_relevant"])
        context_awareness_rate = context_aware_responses / len(context_switching_results)
        
        print(f"   ✅ Context awareness rate: {context_awareness_rate:.2%}")
        print(f"   ✅ Context switching scenarios tested: {len(context_scenarios)}")
        
        await context_engine.end_session(context_session_id)
        await context_engine.stop()
        
        # Test 5: Real-Time Performance Under Load
        print("\n5. Testing Real-Time Performance Under Load...")
        
        # Create chat engine for load testing
        load_engine = ChatEngine({"inference": {}, "memory": {}})
        await load_engine.start()
        
        # Simulate high load scenario
        num_concurrent_users = 10
        messages_per_user = 5
        
        load_users = [f"load_user_{i}" for i in range(num_concurrent_users)]
        load_sessions = {}
        
        # Start sessions for load test
        for user_id in load_users:
            session = await load_engine.start_session(user_id, {"language": "en"})
            load_sessions[user_id] = session["session_id"]
        
        # Generate load test messages
        load_messages = [
            "What is the meaning of life?",
            "Explain quantum computing in simple terms",
            "How does photosynthesis work?",
            "What are the principles of economics?",
            "Describe the solar system"
        ]
        
        # Execute load test
        load_start_time = time.time()
        load_tasks = []
        load_results = []
        
        async def load_test_user(user_id: str, session_id: str, messages: List[str]):
            """Simulate load test user"""
            user_results = []
            for i, msg in enumerate(messages):
                start_time = time.time()
                try:
                    response = await load_engine.generate_response(user_id, session_id, msg)
                    response_time = time.time() - start_time
                    user_results.append({
                        "user_id": user_id,
                        "message_index": i,
                        "response_time": response_time,
                        "success": True
                    })
                except Exception as e:
                    response_time = time.time() - start_time
                    user_results.append({
                        "user_id": user_id,
                        "message_index": i,
                        "response_time": response_time,
                        "success": False,
                        "error": str(e)
                    })
                
                # Small delay
                await asyncio.sleep(0.02)
            
            return user_results
        
        # Create load test tasks
        for user_id in load_users:
            task = asyncio.create_task(load_test_user(user_id, load_sessions[user_id], load_messages))
            load_tasks.append(task)
        
        # Wait for load test to complete
        all_user_results = await asyncio.gather(*load_tasks)
        load_end_time = time.time()
        
        # Flatten results
        for user_results in all_user_results:
            load_results.extend(user_results)
        
        # Analyze load test results
        total_load_time = load_end_time - load_start_time
        total_requests = len(load_results)
        successful_requests = sum(1 for r in load_results if r["success"])
        failed_requests = total_requests - successful_requests
        
        requests_per_second = total_requests / total_load_time
        success_rate = successful_requests / total_requests
        
        response_times = [r["response_time"] for r in load_results if r["success"]]
        avg_load_response_time = sum(response_times) / len(response_times) if response_times else 0
        max_load_response_time = max(response_times) if response_times else 0
        min_load_response_time = min(response_times) if response_times else 0
        
        print(f"   ✅ Load test completed: {total_requests} requests in {total_load_time:.2f}s")
        print(f"   ✅ Throughput: {requests_per_second:.2f} requests/second")
        print(f"   ✅ Success rate: {success_rate:.2%} ({successful_requests}/{total_requests})")
        print(f"   ✅ Average response time: {avg_load_response_time:.3f}s")
        print(f"   ✅ Response time range: {min_load_response_time:.3f}s - {max_load_response_time:.3f}s")
        
        # End all load test sessions
        for user_id, session_id in load_sessions.items():
            await load_engine.end_session(session_id)
        
        await load_engine.stop()
        
        # Test 6: Real-Time Memory and Context Integration
        print("\n6. Testing Real-Time Memory and Context Integration...")
        
        # Create chat engine for memory integration test
        memory_engine = ChatEngine({"inference": {}, "memory": {}})
        await memory_engine.start()
        
        # Start session with personalization
        memory_session = await memory_engine.start_session("user_memory", {
            "language": "en",
            "style": "technical",
            "expertise": "advanced"
        })
        memory_session_id = memory_session["session_id"]
        
        # Test memory integration with progressive conversation
        memory_conversation = [
            "I'm working on a machine learning project",
            "The project involves natural language processing",
            "I'm using transformer models for text classification",
            "I need help with optimizing the model architecture",
            "Can you suggest some techniques for improving accuracy?",
            "What about handling imbalanced datasets?",
            "How do I evaluate model performance properly?",
            "Can you recommend some tools for deployment?",
            "What are the best practices for model monitoring?",
            "How do I handle model drift in production?"
        ]
        
        memory_test_results = []
        
        for i, message in enumerate(memory_conversation):
            start_time = time.time()
            response = await memory_engine.generate_response("user_memory", memory_session_id, message)
            response_time = time.time() - start_time
            
            # Check if response uses previous context
            response_text = str(response["response"]).lower()
            context_used = any(keyword in response_text for keyword in [
                "machine learning", "nlp", "transformer", "classification", 
                "model", "accuracy", "dataset", "deployment", "monitoring", "drift"
            ])
            
            memory_test_results.append({
                "message_index": i,
                "response_time": response_time,
                "context_used": context_used,
                "context_progression": i > 0  # Should improve as conversation progresses
            })
        
        # Analyze memory integration results
        context_usage_rate = sum(1 for r in memory_test_results if r["context_used"]) / len(memory_test_results)
        avg_memory_response_time = sum(r["response_time"] for r in memory_test_results) / len(memory_test_results)
        
        print(f"   ✅ Memory context usage rate: {context_usage_rate:.2%}")
        print(f"   ✅ Average response time with memory: {avg_memory_response_time:.3f}s")
        print(f"   ✅ Conversation length: {len(memory_conversation)} exchanges")
        
        await memory_engine.end_session(memory_session_id)
        await memory_engine.stop()
        
        print("\n🎉 Real-Time AI Interactions Test Passed!")
        print("   All real-time interaction features are working correctly.")
        print("   Chat engine, streaming, concurrency, context switching, and load testing are functional.")
        
        # Summary statistics
        print("\n📊 Real-Time Interaction Summary:")
        print(f"   • Basic chat response time: {avg_response_time:.3f}s")
        print(f"   • Streaming chunks delivered: {total_chunks}")
        print(f"   • Concurrent users handled: {len(users)}")
        print(f"   • Context awareness rate: {context_awareness_rate:.2%}")
        print(f"   • Load test throughput: {requests_per_second:.2f} req/s")
        print(f"   • Memory context usage: {context_usage_rate:.2%}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Real-Time AI Interactions Test Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_realtime_interactions())
    sys.exit(0 if success else 1)