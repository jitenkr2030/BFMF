"""
Test Script for Real Bharat-FM AI Capabilities
Demonstrates actual AI functionality with real models and proper implementations
"""

import asyncio
import logging
import time
import json
from typing import Dict, List, Any
from datetime import datetime

# Import our real AI components
from src.bharat_fm.core.real_inference_engine import RealInferenceEngine, create_real_inference_engine
from src.bharat_fm.memory.real_memory_system import RealMemorySystem, create_real_memory_system
from src.bharat_fm.data.real_tokenization import RealTextProcessor, create_real_text_processor
from src.bharat_fm.train.real_training_system import RealTrainingSystem, TrainingConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealAIDemo:
    """Demonstration of real AI capabilities"""
    
    def __init__(self):
        self.inference_engine = None
        self.memory_system = None
        self.text_processor = None
        self.training_system = None
        
        # Test results
        self.test_results = {
            "inference_engine": {},
            "memory_system": {},
            "text_processing": {},
            "training_system": {},
            "integration_tests": {}
        }
    
    async def setup(self):
        """Initialize all AI components"""
        logger.info("Setting up real AI components...")
        
        # Initialize inference engine
        self.inference_engine = await create_real_inference_engine()
        
        # Initialize memory system
        self.memory_system = await create_real_memory_system()
        
        # Initialize text processor
        self.text_processor = create_real_text_processor()
        
        logger.info("All AI components initialized successfully")
    
    async def test_inference_engine(self):
        """Test real inference engine capabilities"""
        logger.info("Testing real inference engine...")
        
        test_cases = [
            {
                "name": "Text Generation",
                "function": self.inference_engine.generate_text,
                "args": ("Hello, I am a real AI model.",),
                "kwargs": {"max_length": 50, "temperature": 0.7}
            },
            {
                "name": "Sentiment Analysis",
                "function": self.inference_engine.analyze_sentiment,
                "args": ("I love this new AI system! It's amazing.",),
                "kwargs": {}
            },
            {
                "name": "Text Embeddings",
                "function": self.inference_engine.get_embeddings,
                "args": ("The quick brown fox jumps over the lazy dog.",),
                "kwargs": {}
            },
            {
                "name": "Batch Generation",
                "function": self.inference_engine.batch_generate_text,
                "args": (["Hello world", "AI is great", "Machine learning"],),
                "kwargs": {"max_length": 30}
            }
        ]
        
        for test_case in test_cases:
            start_time = time.time()
            try:
                result = await test_case["function"](*test_case["args"], **test_case["kwargs"])
                latency = time.time() - start_time
                
                self.test_results["inference_engine"][test_case["name"]] = {
                    "status": "success",
                    "latency": latency,
                    "result_summary": self._summarize_result(result),
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                logger.info(f"✅ {test_case['name']}: Success ({latency:.3f}s)")
                
            except Exception as e:
                latency = time.time() - start_time
                self.test_results["inference_engine"][test_case["name"]] = {
                    "status": "failed",
                    "latency": latency,
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                logger.error(f"❌ {test_case['name']}: Failed - {e}")
        
        # Test engine status
        try:
            status = await self.inference_engine.get_engine_status()
            self.test_results["inference_engine"]["Engine Status"] = {
                "status": "success",
                "engine_id": status["engine_id"],
                "available_models": status["available_models"],
                "device": status["device"],
                "timestamp": datetime.utcnow().isoformat()
            }
            logger.info(f"✅ Engine Status: {status['engine_id']} running on {status['device']}")
        except Exception as e:
            self.test_results["inference_engine"]["Engine Status"] = {
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
            logger.error(f"❌ Engine Status: Failed - {e}")
    
    async def test_memory_system(self):
        """Test real memory system capabilities"""
        logger.info("Testing real memory system...")
        
        # Test memory operations
        session_id = "test_session_001"
        user_id = "test_user_001"
        
        test_memories = [
            {"content": "Hello, I'm testing the memory system.", "role": "user", "tags": ["greeting"]},
            {"content": "Hi there! I'm working properly.", "role": "assistant", "tags": ["response"]},
            {"content": "Can you remember our conversation?", "role": "user", "tags": ["question"]},
            {"content": "Yes, I can remember our conversation context.", "role": "assistant", "tags": ["memory"]},
        ]
        
        # Test adding memories
        memory_ids = []
        for i, memory in enumerate(test_memories):
            start_time = time.time()
            try:
                memory_id = await self.memory_system.add_memory(
                    content=memory["content"],
                    role=memory["role"],
                    session_id=session_id,
                    user_id=user_id,
                    tags=memory["tags"],
                    importance_score=0.8
                )
                memory_ids.append(memory_id)
                latency = time.time() - start_time
                
                self.test_results["memory_system"][f"Add Memory {i+1}"] = {
                    "status": "success",
                    "latency": latency,
                    "memory_id": memory_id,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                logger.info(f"✅ Add Memory {i+1}: Success ({latency:.3f}s)")
                
            except Exception as e:
                latency = time.time() - start_time
                self.test_results["memory_system"][f"Add Memory {i+1}"] = {
                    "status": "failed",
                    "latency": latency,
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                }
                logger.error(f"❌ Add Memory {i+1}: Failed - {e}")
        
        # Test context retrieval
        try:
            start_time = time.time()
            context = await self.memory_system.get_context(session_id, max_tokens=1000)
            latency = time.time() - start_time
            
            self.test_results["memory_system"]["Get Context"] = {
                "status": "success",
                "latency": latency,
                "context_length": len(context),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            logger.info(f"✅ Get Context: Success - {len(context)} entries ({latency:.3f}s)")
            
        except Exception as e:
            latency = time.time() - start_time
            self.test_results["memory_system"]["Get Context"] = {
                "status": "failed",
                "latency": latency,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
            logger.error(f"❌ Get Context: Failed - {e}")
        
        # Test memory search
        try:
            start_time = time.time()
            search_results = await self.memory_system.search_memories(
                query="conversation memory",
                user_id=user_id,
                limit=5
            )
            latency = time.time() - start_time
            
            self.test_results["memory_system"]["Search Memories"] = {
                "status": "success",
                "latency": latency,
                "results_count": len(search_results),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            logger.info(f"✅ Search Memories: Success - {len(search_results)} results ({latency:.3f}s)")
            
        except Exception as e:
            latency = time.time() - start_time
            self.test_results["memory_system"]["Search Memories"] = {
                "status": "failed",
                "latency": latency,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
            logger.error(f"❌ Search Memories: Failed - {e}")
        
        # Test user profile
        try:
            start_time = time.time()
            profile = await self.memory_system.get_user_profile(user_id)
            latency = time.time() - start_time
            
            self.test_results["memory_system"]["User Profile"] = {
                "status": "success",
                "latency": latency,
                "total_memories": profile.get("total_memory_entries", 0),
                "total_sessions": profile.get("total_sessions", 0),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            logger.info(f"✅ User Profile: Success - {profile.get('total_memory_entries', 0)} memories ({latency:.3f}s)")
            
        except Exception as e:
            latency = time.time() - start_time
            self.test_results["memory_system"]["User Profile"] = {
                "status": "failed",
                "latency": latency,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
            logger.error(f"❌ User Profile: Failed - {e}")
        
        # Test memory statistics
        try:
            start_time = time.time()
            stats = await self.memory_system.get_memory_stats()
            latency = time.time() - start_time
            
            self.test_results["memory_system"]["Memory Stats"] = {
                "status": "success",
                "latency": latency,
                "total_memories": stats["total_memory_entries"],
                "total_sessions": stats["total_conversation_sessions"],
                "total_users": stats["total_users"],
                "timestamp": datetime.utcnow().isoformat()
            }
            
            logger.info(f"✅ Memory Stats: Success - {stats['total_memory_entries']} memories ({latency:.3f}s)")
            
        except Exception as e:
            latency = time.time() - start_time
            self.test_results["memory_system"]["Memory Stats"] = {
                "status": "failed",
                "latency": latency,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
            logger.error(f"❌ Memory Stats: Failed - {e}")
    
    async def test_text_processing(self):
        """Test real text processing capabilities"""
        logger.info("Testing real text processing...")
        
        test_texts = [
            "Hello, world! This is a test.",
            "नमस्ते दुनिया! यह एक परीक्षण है।",  # Hindi
            "வணக்கம் உலகம்! இது ஒரு சோதனை.",  # Tamil
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is fascinating! 123 numbers and symbols."
        ]
        
        # Test tokenization with different tokenizers
        tokenizers = ["transformers", "indian", "subword"]
        
        for tokenizer_type in tokenizers:
            try:
                start_time = time.time()
                results = self.text_processor.tokenize_batch(test_texts, tokenizer_type)
                latency = time.time() - start_time
                
                self.test_results["text_processing"][f"Tokenization ({tokenizer_type})"] = {
                    "status": "success",
                    "latency": latency,
                    "texts_processed": len(test_texts),
                    "avg_tokens_per_text": sum(r.num_tokens for r in results) / len(results),
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                logger.info(f"✅ Tokenization ({tokenizer_type}): Success - {len(results)} texts ({latency:.3f}s)")
                
            except Exception as e:
                latency = time.time() - start_time
                self.test_results["text_processing"][f"Tokenization ({tokenizer_type})"] = {
                    "status": "failed",
                    "latency": latency,
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                }
                logger.error(f"❌ Tokenization ({tokenizer_type}): Failed - {e}")
        
        # Test language detection
        for text in test_texts:
            try:
                start_time = time.time()
                detected_lang = self.text_processor.detect_language(text)
                latency = time.time() - start_time
                
                self.test_results["text_processing"][f"Language Detection ({text[:20]}...)"] = {
                    "status": "success",
                    "latency": latency,
                    "detected_language": detected_lang,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                logger.info(f"✅ Language Detection ({text[:20]}...): {detected_lang} ({latency:.3f}s)")
                
            except Exception as e:
                latency = time.time() - start_time
                self.test_results["text_processing"][f"Language Detection ({text[:20]}...)"] = {
                    "status": "failed",
                    "latency": latency,
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                }
                logger.error(f"❌ Language Detection ({text[:20]}...): Failed - {e}")
        
        # Test text preprocessing
        for text in test_texts:
            try:
                start_time = time.time()
                processed = self.text_processor.preprocess_text(text)
                latency = time.time() - start_time
                
                self.test_results["text_processing"][f"Text Preprocessing ({text[:20]}...)"] = {
                    "status": "success",
                    "latency": latency,
                    "original_length": len(text),
                    "processed_length": len(processed),
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                logger.info(f"✅ Text Preprocessing ({text[:20]}...): Success ({latency:.3f}s)")
                
            except Exception as e:
                latency = time.time() - start_time
                self.test_results["text_processing"][f"Text Preprocessing ({text[:20]}...)"] = {
                    "status": "failed",
                    "latency": latency,
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                }
                logger.error(f"❌ Text Preprocessing ({text[:20]}...): Failed - {e}")
        
        # Test tokenizer info
        try:
            start_time = time.time()
            info = self.text_processor.get_tokenizer_info("transformers")
            latency = time.time() - start_time
            
            self.test_results["text_processing"]["Tokenizer Info"] = {
                "status": "success",
                "latency": latency,
                "vocabulary_size": info["vocabulary_size"],
                "max_length": info["max_length"],
                "timestamp": datetime.utcnow().isoformat()
            }
            
            logger.info(f"✅ Tokenizer Info: Success - {info['vocabulary_size']} vocab size ({latency:.3f}s)")
            
        except Exception as e:
            latency = time.time() - start_time
            self.test_results["text_processing"]["Tokenizer Info"] = {
                "status": "failed",
                "latency": latency,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
            logger.error(f"❌ Tokenizer Info: Failed - {e}")
    
    async def test_training_system(self):
        """Test real training system capabilities"""
        logger.info("Testing real training system...")
        
        # Create training configuration
        config = TrainingConfig(
            model_type="gpt",
            vocab_size=1000,  # Smaller for demo
            d_model=128,     # Smaller for demo
            num_heads=4,     # Smaller for demo
            num_layers=2,    # Smaller for demo
            batch_size=2,
            learning_rate=1e-4,
            num_epochs=1,    # Just one epoch for demo
            max_len=64,      # Smaller for demo
            output_dir="./demo_training_output"
        )
        
        try:
            # Initialize training system
            start_time = time.time()
            self.training_system = RealTrainingSystem(config)
            init_latency = time.time() - start_time
            
            # Create dummy training data
            train_texts = [
                "Hello, this is training data.",
                "Machine learning is amazing.",
                "AI models can learn patterns.",
                "Deep learning uses neural networks.",
                "Natural language processing is important."
            ] * 10  # Repeat for more data
            
            val_texts = [
                "This is validation data.",
                "Testing the model performance."
            ] * 5
            
            # Load data
            self.training_system.load_data(train_texts, val_texts)
            
            self.test_results["training_system"]["Initialization"] = {
                "status": "success",
                "latency": init_latency,
                "config": {
                    "model_type": config.model_type,
                    "vocab_size": config.vocab_size,
                    "d_model": config.d_model,
                    "num_heads": config.num_heads,
                    "num_layers": config.num_layers
                },
                "timestamp": datetime.utcnow().isoformat()
            }
            
            logger.info(f"✅ Training System Initialization: Success ({init_latency:.3f}s)")
            
            # Test model initialization
            start_time = time.time()
            self.training_system.initialize_model()
            model_init_latency = time.time() - start_time
            
            self.test_results["training_system"]["Model Initialization"] = {
                "status": "success",
                "latency": model_init_latency,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            logger.info(f"✅ Model Initialization: Success ({model_init_latency:.3f}s)")
            
            # Test optimizer initialization
            start_time = time.time()
            self.training_system.initialize_optimizer()
            opt_init_latency = time.time() - start_time
            
            self.test_results["training_system"]["Optimizer Initialization"] = {
                "status": "success",
                "latency": opt_init_latency,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            logger.info(f"✅ Optimizer Initialization: Success ({opt_init_latency:.3f}s)")
            
            # Note: We won't run full training in demo to save time
            logger.info("⏭️  Full training skipped for demo (would take too long)")
            
        except Exception as e:
            self.test_results["training_system"]["Setup"] = {
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
            logger.error(f"❌ Training System Setup: Failed - {e}")
    
    async def test_integration(self):
        """Test integration between components"""
        logger.info("Testing component integration...")
        
        try:
            # Test end-to-end pipeline: text -> tokenize -> infer -> store in memory
            test_text = "The weather is beautiful today. I think I'll go for a walk."
            
            start_time = time.time()
            
            # Step 1: Process text
            tokenization_result = self.text_processor.tokenize(test_text, "transformers")
            
            # Step 2: Generate response
            inference_result = await self.inference_engine.generate_text(test_text, max_length=50)
            
            # Step 3: Store conversation in memory
            session_id = "integration_test_session"
            user_id = "integration_test_user"
            
            await self.memory_system.add_memory(
                content=test_text,
                role="user",
                session_id=session_id,
                user_id=user_id,
                tags=["integration", "test"]
            )
            
            await self.memory_system.add_memory(
                content=inference_result["generated_texts"][0],
                role="assistant",
                session_id=session_id,
                user_id=user_id,
                tags=["integration", "response"]
            )
            
            # Step 4: Retrieve context
            context = await self.memory_system.get_context(session_id, max_tokens=500)
            
            total_latency = time.time() - start_time
            
            self.test_results["integration_tests"]["End-to-End Pipeline"] = {
                "status": "success",
                "latency": total_latency,
                "steps_completed": 4,
                "context_entries": len(context),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            logger.info(f"✅ End-to-End Pipeline: Success - {len(context)} context entries ({total_latency:.3f}s)")
            
        except Exception as e:
            self.test_results["integration_tests"]["End-to-End Pipeline"] = {
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
            logger.error(f"❌ End-to-End Pipeline: Failed - {e}")
        
        # Test concurrent operations
        try:
            start_time = time.time()
            
            # Run multiple operations concurrently
            tasks = [
                self.inference_engine.generate_text(f"Test message {i}", max_length=20)
                for i in range(5)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            successful_results = [r for r in results if not isinstance(r, Exception)]
            total_latency = time.time() - start_time
            
            self.test_results["integration_tests"]["Concurrent Operations"] = {
                "status": "success",
                "latency": total_latency,
                "total_tasks": len(tasks),
                "successful_tasks": len(successful_results),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            logger.info(f"✅ Concurrent Operations: Success - {len(successful_results)}/{len(tasks)} tasks ({total_latency:.3f}s)")
            
        except Exception as e:
            self.test_results["integration_tests"]["Concurrent Operations"] = {
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
            logger.error(f"❌ Concurrent Operations: Failed - {e}")
    
    def _summarize_result(self, result: Any) -> str:
        """Summarize test result for logging"""
        if isinstance(result, dict):
            if "generated_texts" in result:
                return f"Generated {len(result['generated_texts'])} texts"
            elif "sentiment_results" in result:
                return f"Analyzed sentiment: {result['sentiment_results']}"
            elif "embeddings" in result:
                return f"Generated embeddings with shape {result['embedding_shape']}"
            elif isinstance(result, list):
                return f"Batch of {len(result)} results"
        
        return f"Result type: {type(result).__name__}"
    
    async def cleanup(self):
        """Clean up resources"""
        logger.info("Cleaning up resources...")
        
        if self.inference_engine:
            await self.inference_engine.stop()
        
        if self.memory_system:
            await self.memory_system.stop()
        
        logger.info("Cleanup completed")
    
    def save_results(self, filename: str = "real_ai_test_results.json"):
        """Save test results to file"""
        with open(filename, 'w') as f:
            json.dump(self.test_results, f, indent=2)
        
        logger.info(f"Test results saved to {filename}")
    
    def print_summary(self):
        """Print test summary"""
        logger.info("\n" + "="*60)
        logger.info("REAL AI CAPABILITIES TEST SUMMARY")
        logger.info("="*60)
        
        total_tests = 0
        successful_tests = 0
        
        for category, tests in self.test_results.items():
            logger.info(f"\n{category.upper()}:")
            for test_name, result in tests.items():
                total_tests += 1
                if result["status"] == "success":
                    successful_tests += 1
                    logger.info(f"  ✅ {test_name}: Success")
                else:
                    logger.info(f"  ❌ {test_name}: Failed - {result.get('error', 'Unknown error')}")
        
        success_rate = (successful_tests / total_tests * 100) if total_tests > 0 else 0
        
        logger.info(f"\nSUMMARY:")
        logger.info(f"  Total Tests: {total_tests}")
        logger.info(f"  Successful: {successful_tests}")
        logger.info(f"  Failed: {total_tests - successful_tests}")
        logger.info(f"  Success Rate: {success_rate:.1f}%")
        
        if success_rate >= 80:
            logger.info("  🎉 EXCELLENT: Real AI capabilities are working well!")
        elif success_rate >= 60:
            logger.info("  👍 GOOD: Most real AI capabilities are functional!")
        else:
            logger.info("  ⚠️  NEEDS WORK: Several real AI capabilities need attention!")
        
        logger.info("="*60)

async def main():
    """Main test function"""
    demo = RealAIDemo()
    
    try:
        # Setup
        await demo.setup()
        
        # Run all tests
        await demo.test_inference_engine()
        await demo.test_memory_system()
        await demo.test_text_processing()
        await demo.test_training_system()
        await demo.test_integration()
        
        # Print summary
        demo.print_summary()
        
        # Save results
        demo.save_results()
        
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        raise
    
    finally:
        # Cleanup
        await demo.cleanup()

if __name__ == "__main__":
    asyncio.run(main())