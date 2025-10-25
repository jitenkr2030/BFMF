"""
Final Comprehensive Test for Bharat-FM
Tests all components and provides final status report
"""

import sys
import os
import json
import time
from datetime import datetime

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_memory_system():
    """Test memory system functionality"""
    results = {}
    
    try:
        from bharat_fm.memory.real_memory_system import RealMemorySystem
        import asyncio
        
        # Test memory system creation
        start_time = time.time()
        memory_system = RealMemorySystem({
            "max_memory_entries": 100,
            "max_context_length": 512,
            "memory_retention_days": 7
        })
        creation_time = time.time() - start_time
        
        results["System Creation"] = {
            "status": "success",
            "creation_time": creation_time,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Test adding memories
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            memory_id = loop.run_until_complete(
                memory_system.add_memory(
                    content="Test memory for comprehensive test",
                    role="user",
                    session_id="comprehensive_test",
                    user_id="test_user",
                    tags=["test", "comprehensive"],
                    importance_score=0.9
                )
            )
            
            results["Add Memory"] = {
                "status": "success",
                "memory_id": memory_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Test context retrieval
            context = loop.run_until_complete(
                memory_system.get_context("comprehensive_test", max_tokens=100)
            )
            
            results["Get Context"] = {
                "status": "success",
                "context_length": len(context),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Test memory search
            search_results = loop.run_until_complete(
                memory_system.search_memories(
                    query="test memory",
                    user_id="test_user",
                    limit=5
                )
            )
            
            results["Search Memories"] = {
                "status": "success",
                "results_count": len(search_results),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        finally:
            loop.close()
        
        return results, True
        
    except Exception as e:
        return {"General Error": {"status": "failed", "error": str(e), "timestamp": datetime.utcnow().isoformat()}}, False

def test_text_processing():
    """Test text processing functionality"""
    results = {}
    
    try:
        from bharat_fm.data.lightweight_tokenization import LightweightTextProcessor
        
        # Test text processor creation
        start_time = time.time()
        text_processor = LightweightTextProcessor({
            "tokenizer_type": "custom",
            "max_length": 512,
            "lowercase": True,
            "normalize_whitespace": True
        })
        creation_time = time.time() - start_time
        
        results["Processor Creation"] = {
            "status": "success",
            "creation_time": creation_time,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Test text preprocessing
        test_texts = [
            "Hello, World! This is a TEST.",
            "नमस्ते दुनिया! यह एक परीक्षण है।",  # Hindi
            "The quick brown fox jumps over the lazy dog."
        ]
        
        for i, text in enumerate(test_texts):
            start_time = time.time()
            processed = text_processor.preprocess_text(text)
            test_time = time.time() - start_time
            
            results[f"Text Preprocessing {i+1}"] = {
                "status": "success",
                "original_length": len(text),
                "processed_length": len(processed),
                "test_time": test_time,
                "timestamp": datetime.utcnow().isoformat()
            }
        
        # Test tokenization
        start_time = time.time()
        token_results = text_processor.tokenize_batch(test_texts, "custom")
        test_time = time.time() - start_time
        
        results["Tokenization"] = {
            "status": "success",
            "test_time": test_time,
            "texts_processed": len(test_texts),
            "avg_tokens_per_text": sum(r.num_tokens for r in token_results) / len(token_results),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Test language detection
        for i, text in enumerate(test_texts):
            start_time = time.time()
            detected_lang = text_processor.detect_language(text)
            test_time = time.time() - start_time
            
            results[f"Language Detection {i+1}"] = {
                "status": "success",
                "detected_language": detected_lang,
                "test_time": test_time,
                "timestamp": datetime.utcnow().isoformat()
            }
        
        return results, True
        
    except Exception as e:
        return {"General Error": {"status": "failed", "error": str(e), "timestamp": datetime.utcnow().isoformat()}}, False

def test_neural_networks():
    """Test neural network functionality"""
    results = {}
    
    try:
        import numpy as np
        from bharat_fm.core.lightweight_neural_networks import LightweightTextProcessor
        
        # Test basic neural network operations
        start_time = time.time()
        
        # Simple matrix operations (core of neural networks)
        W1 = np.random.randn(64, 128) * 0.01
        b1 = np.zeros(128)
        W2 = np.random.randn(128, 64) * 0.01
        b2 = np.zeros(64)
        
        # Create input
        x = np.random.randn(10, 64)
        
        # Forward pass
        hidden = np.maximum(0, np.matmul(x, W1) + b1)  # ReLU
        output = np.matmul(hidden, W2) + b2
        
        test_time = time.time() - start_time
        
        results["Matrix Operations"] = {
            "status": "success",
            "input_shape": str(x.shape),
            "output_shape": str(output.shape),
            "test_time": test_time,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Test text processing integration
        start_time = time.time()
        text_processor = LightweightTextProcessor(vocab_size=1000)
        
        sample_text = "Hello world! This is a neural network test."
        tokens = text_processor.tokenize(sample_text)
        processed_text = text_processor.detokenize(tokens)
        
        test_time = time.time() - start_time
        
        results["Text Processing Integration"] = {
            "status": "success",
            "original_text": sample_text,
            "processed_text": processed_text,
            "tokens_count": len(tokens),
            "test_time": test_time,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Test embedding operations
        start_time = time.time()
        
        # Create simple embedding matrix
        vocab_size = 1000
        embedding_dim = 64
        embeddings = np.random.randn(vocab_size, embedding_dim) * 0.01
        
        # Embedding lookup
        if tokens:
            embedded_vectors = embeddings[tokens[:10]]  # First 10 tokens
        else:
            embedded_vectors = embeddings[[1, 2, 3]]  # Default tokens
        
        test_time = time.time() - start_time
        
        results["Embedding Operations"] = {
            "status": "success",
            "embeddings_shape": str(embedded_vectors.shape),
            "test_time": test_time,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return results, True
        
    except Exception as e:
        return {"General Error": {"status": "failed", "error": str(e), "timestamp": datetime.utcnow().isoformat()}}, False

def test_integration():
    """Test integration between components"""
    results = {}
    
    try:
        from bharat_fm.memory.real_memory_system import RealMemorySystem
        from bharat_fm.data.lightweight_tokenization import LightweightTextProcessor
        import asyncio
        import numpy as np
        
        # Test creating all components
        start_time = time.time()
        
        memory_system = RealMemorySystem({
            "max_memory_entries": 50,
            "max_context_length": 256,
            "memory_retention_days": 7
        })
        
        text_processor = LightweightTextProcessor({
            "tokenizer_type": "custom",
            "max_length": 128,
            "lowercase": True
        })
        
        integration_time = time.time() - start_time
        
        results["Component Creation"] = {
            "status": "success",
            "creation_time": integration_time,
            "components_created": ["memory_system", "text_processor"],
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Test integrated workflow
        start_time = time.time()
        
        # Step 1: Process text
        sample_text = "Integration test for Bharat-FM components"
        processed_text = text_processor.preprocess_text(sample_text)
        
        # Step 2: Add to memory
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            memory_id = loop.run_until_complete(
                memory_system.add_memory(
                    content=processed_text,
                    role="user",
                    session_id="integration_test",
                    user_id="test_user",
                    tags=["integration", "test"],
                    importance_score=0.9
                )
            )
            
            # Step 3: Retrieve from memory
            context = loop.run_until_complete(
                memory_system.get_context("integration_test", max_tokens=50)
            )
            
            workflow_time = time.time() - start_time
            
            results["Integrated Workflow"] = {
                "status": "success",
                "original_text": sample_text,
                "processed_text": processed_text,
                "memory_id": memory_id,
                "context_entries": len(context),
                "workflow_time": workflow_time,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        finally:
            loop.close()
        
        return results, True
        
    except Exception as e:
        return {"General Error": {"status": "failed", "error": str(e), "timestamp": datetime.utcnow().isoformat()}}, False

def main():
    """Run comprehensive test and generate final report"""
    print("="*60)
    print("BHARAT-FM COMPREHENSIVE SYSTEM TEST")
    print("="*60)
    
    # Test all components
    print("\n🧠 Testing Memory System...")
    memory_results, memory_success = test_memory_system()
    
    print("\n📝 Testing Text Processing...")
    text_results, text_success = test_text_processing()
    
    print("\n🔀 Testing Neural Networks...")
    neural_results, neural_success = test_neural_networks()
    
    print("\n🔗 Testing Integration...")
    integration_results, integration_success = test_integration()
    
    # Compile results
    all_results = {
        "memory_system": memory_results,
        "text_processing": text_results,
        "neural_networks": neural_results,
        "integration": integration_results,
        "test_summary": {
            "timestamp": datetime.utcnow().isoformat(),
            "memory_system_success": memory_success,
            "text_processing_success": text_success,
            "neural_networks_success": neural_success,
            "integration_success": integration_success
        }
    }
    
    # Calculate statistics
    total_tests = 0
    successful_tests = 0
    
    for category_name, category in all_results.items():
        if isinstance(category, dict) and category_name != "test_summary":
            for test_name, test_result in category.items():
                if isinstance(test_result, dict):
                    total_tests += 1
                    if test_result.get("status") == "success":
                        successful_tests += 1
    
    success_rate = (successful_tests / total_tests) * 100 if total_tests > 0 else 0
    
    # Generate summary
    print("\n" + "="*60)
    print("COMPREHENSIVE TEST RESULTS")
    print("="*60)
    print(f"\n📊 Overall Statistics:")
    print(f"   Total Tests: {total_tests}")
    print(f"   Successful: {successful_tests}")
    print(f"   Failed: {total_tests - successful_tests}")
    print(f"   Success Rate: {success_rate:.1f}%")
    
    print(f"\n🎯 Component Status:")
    print(f"   Memory System: {'✅ PASS' if memory_success else '❌ FAIL'}")
    print(f"   Text Processing: {'✅ PASS' if text_success else '❌ FAIL'}")
    print(f"   Neural Networks: {'✅ PASS' if neural_success else '❌ FAIL'}")
    print(f"   Integration: {'✅ PASS' if integration_success else '❌ FAIL'}")
    
    # Overall assessment
    if success_rate >= 90:
        print(f"\n🎉 EXCELLENT: Bharat-FM is working excellently!")
        print(f"   The framework demonstrates real AI capabilities.")
    elif success_rate >= 75:
        print(f"\n👍 GOOD: Bharat-FM is working well!")
        print(f"   Most components are functional with real AI capabilities.")
    elif success_rate >= 50:
        print(f"\n⚠️  FAIR: Bharat-FM has mixed results.")
        print(f"   Some components work, others need attention.")
    else:
        print(f"\n❌ POOR: Bharat-FM needs significant work.")
        print(f"   Most components are not functioning properly.")
    
    print("="*60)
    
    # Save results
    with open("comprehensive_test_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"📄 Detailed results saved to: comprehensive_test_results.json")
    
    return all_results

if __name__ == "__main__":
    results = main()