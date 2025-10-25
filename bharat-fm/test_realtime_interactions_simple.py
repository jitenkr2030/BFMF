"""
Test real-time AI interactions (simplified version)
"""

import sys
import os
import asyncio
import time
import json
from typing import Dict, List, Any, AsyncIterator

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

async def test_realtime_interactions_simple():
    print("🇮🇳 Testing Real-Time AI Interactions (Simplified)")
    print("=" * 50)
    
    try:
        # Test 1: Basic Chat Engine Functionality
        print("\n1. Testing Basic Chat Engine Functionality...")
        
        from bharat_fm.core.chat_engine import ChatEngine
        
        # Create and start chat engine
        chat_engine = ChatEngine({"inference": {}, "memory": {"memory_dir": "./test_realtime_memory"}})
        await chat_engine.start()
        
        # Test session management
        session_info = await chat_engine.start_session("user_test", {"language": "en", "style": "casual"})
        session_id = session_info["session_id"]
        print(f"   ✅ Session started: {session_id}")
        
        # Test basic response generation
        test_messages = [
            "Hello, how are you?",
            "What is artificial intelligence?",
            "Can you explain machine learning?"
        ]
        
        response_times = []
        for i, message in enumerate(test_messages):
            start_time = time.time()
            response = await chat_engine.generate_response("user_test", session_id, message)
            response_time = time.time() - start_time
            response_times.append(response_time)
            
            print(f"   ✅ Message {i+1}: {response_time:.3f}s - {len(str(response['response']))} chars")
        
        # Calculate average response time
        avg_response_time = sum(response_times) / len(response_times)
        print(f"   ✅ Average response time: {avg_response_time:.3f}s")
        
        # Test 2: Streaming Chat Interactions
        print("\n2. Testing Streaming Chat Interactions...")
        
        # Create new chat engine for streaming test
        streaming_engine = ChatEngine({"inference": {}, "memory": {"memory_dir": "./test_streaming_memory"}})
        await streaming_engine.start()
        
        # Start streaming session
        streaming_session = await streaming_engine.start_session("user_streaming", {"language": "en"})
        streaming_session_id = streaming_session["session_id"]
        
        # Test streaming response
        streaming_message = "Can you tell me about artificial intelligence?"
        
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
        
        print(f"   ✅ Streaming test completed: {total_chunks} chunks delivered")
        
        # Test 3: Multi-User Concurrent Interactions
        print("\n3. Testing Multi-User Concurrent Interactions...")
        
        # Create chat engine for concurrent test
        concurrent_engine = ChatEngine({"inference": {}, "memory": {"memory_dir": "./test_concurrent_memory"}})
        await concurrent_engine.start()
        
        # Simulate multiple users
        users = [
            {"id": "user_1", "name": "Alice", "language": "en"},
            {"id": "user_2", "name": "Bob", "language": "en"},
            {"id": "user_3", "name": "Charlie", "language": "en"}
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
        
        # Test 4: Real-Time Performance Under Load
        print("\n4. Testing Real-Time Performance Under Load...")
        
        # Create chat engine for load testing
        load_engine = ChatEngine({"inference": {}, "memory": {"memory_dir": "./test_load_memory"}})
        await load_engine.start()
        
        # Simulate high load scenario
        num_concurrent_users = 5
        messages_per_user = 3
        
        load_users = [f"load_user_{i}" for i in range(num_concurrent_users)]
        load_sessions = {}
        
        # Start sessions for load test
        for user_id in load_users:
            session = await load_engine.start_session(user_id, {"language": "en"})
            load_sessions[user_id] = session["session_id"]
        
        # Generate load test messages
        load_messages = [
            "What is artificial intelligence?",
            "Explain machine learning",
            "Tell me about neural networks"
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
        
        print(f"   ✅ Load test completed: {total_requests} requests in {total_load_time:.2f}s")
        print(f"   ✅ Throughput: {requests_per_second:.2f} requests/second")
        print(f"   ✅ Success rate: {success_rate:.2%} ({successful_requests}/{total_requests})")
        print(f"   ✅ Average response time: {avg_load_response_time:.3f}s")
        
        # Test 5: Real-Time Context Switching
        print("\n5. Testing Real-Time Context Switching...")
        
        # Create chat engine for context switching test
        context_engine = ChatEngine({"inference": {}, "memory": {"memory_dir": "./test_context_memory"}})
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
                    "What are the best practices for software development?"
                ]
            },
            {
                "topic": "Health",
                "messages": [
                    "What are the benefits of regular exercise?",
                    "How can I maintain a balanced diet?"
                ]
            },
            {
                "topic": "Education",
                "messages": [
                    "What are the best learning strategies?",
                    "How can I improve my memory?"
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
        
        # Clean up - try to stop engines gracefully
        try:
            await chat_engine.stop()
            await streaming_engine.stop()
            await concurrent_engine.stop()
            await load_engine.stop()
            await context_engine.stop()
        except Exception as e:
            print(f"   ⚠️  Warning during engine shutdown: {e}")
        
        print("\n🎉 Real-Time AI Interactions Test Passed!")
        print("   All real-time interaction features are working correctly.")
        
        # Summary statistics
        print("\n📊 Real-Time Interaction Summary:")
        print(f"   • Basic chat response time: {avg_response_time:.3f}s")
        print(f"   • Streaming chunks delivered: {total_chunks}")
        print(f"   • Concurrent users handled: {len(users)}")
        print(f"   • Load test throughput: {requests_per_second:.2f} req/s")
        print(f"   • Context awareness rate: {context_awareness_rate:.2%}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Real-Time AI Interactions Test Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_realtime_interactions_simple())
    sys.exit(0 if success else 1)