"""
Lightweight Demo of Bharat-FM AI Capabilities
Demonstrates real implementations without heavy dependencies
"""

import asyncio
import time
import json
import logging
from typing import Dict, List, Any
from datetime import datetime
import os
import sys

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LightweightAIDemo:
    """Demonstration of lightweight AI capabilities without heavy dependencies"""
    
    def __init__(self):
        self.results = {
            "neural_networks": {},
            "memory_system": {},
            "text_processing": {},
            "integration": {}
        }
    
    def test_neural_network_implementations(self):
        """Test our lightweight neural network implementations"""
        logger.info("Testing lightweight neural network implementations...")
        
        try:
            # Test basic neural network components
            from bharat_fm.core.lightweight_neural_networks import (
                LightweightMultiHeadAttention, 
                LightweightPositionalEncoding, 
                LightweightTransformerBlock,
                create_lightweight_gpt_model,
                create_lightweight_text_processor
            )
            
            # Test multi-head attention
            start_time = time.time()
            attention = LightweightMultiHeadAttention(d_model=64, num_heads=4, dropout=0.1)
            
            # Create dummy input
            import numpy as np
            np.random.seed(42)  # For reproducible results
            dummy_input = np.random.randn(2, 10, 64)  # batch_size=2, seq_len=10, d_model=64
            
            # Test forward pass
            output = attention.forward(dummy_input)
            test_time = time.time() - start_time
            
            self.results["neural_networks"]["Multi-Head Attention"] = {
                "status": "success",
                "input_shape": str(dummy_input.shape),
                "output_shape": str(output.shape),
                "test_time": test_time,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            logger.info(f"✅ Multi-Head Attention: Success - Input {dummy_input.shape} -> Output {output.shape} ({test_time:.3f}s)")
            
            # Test positional encoding
            start_time = time.time()
            pos_encoding = LightweightPositionalEncoding(d_model=64, max_len=100)
            pos_output = pos_encoding.forward(dummy_input.transpose(0, 1))  # pos_encoding expects (seq_len, batch_size, d_model)
            test_time = time.time() - start_time
            
            self.results["neural_networks"]["Positional Encoding"] = {
                "status": "success",
                "input_shape": str(dummy_input.shape),
                "output_shape": str(pos_output.shape),
                "test_time": test_time,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            logger.info(f"✅ Positional Encoding: Success - Input {dummy_input.shape} -> Output {pos_output.shape} ({test_time:.3f}s)")
            
            # Test transformer block
            start_time = time.time()
            transformer_block = LightweightTransformerBlock(d_model=64, num_heads=4, d_ff=256, dropout=0.1)
            transformer_output = transformer_block.forward(dummy_input)
            test_time = time.time() - start_time
            
            self.results["neural_networks"]["Transformer Block"] = {
                "status": "success",
                "input_shape": str(dummy_input.shape),
                "output_shape": str(transformer_output.shape),
                "test_time": test_time,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            logger.info(f"✅ Transformer Block: Success - Input {dummy_input.shape} -> Output {transformer_output.shape} ({test_time:.3f}s)")
            
            # Test model creation
            start_time = time.time()
            gpt_model = create_lightweight_gpt_model(vocab_size=1000, d_model=64, num_heads=4, num_layers=2)
            model_creation_time = time.time() - start_time
            
            # Count parameters
            param_count = (
                gpt_model.transformer.embedding.size +
                gpt_model.lm_head.size +
                sum(layer.attention.W_q.size + layer.attention.W_k.size + 
                    layer.attention.W_v.size + layer.attention.W_o.size +
                    layer.W1.size + layer.W2.size
                    for layer in gpt_model.transformer.layers)
            )
            
            self.results["neural_networks"]["GPT Model Creation"] = {
                "status": "success",
                "model_type": "Lightweight GPT",
                "vocab_size": 1000,
                "d_model": 64,
                "num_heads": 4,
                "num_layers": 2,
                "creation_time": model_creation_time,
                "total_parameters": param_count,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            logger.info(f"✅ GPT Model Creation: Success - {param_count} parameters ({model_creation_time:.3f}s)")
            
            # Test text generation
            start_time = time.time()
            # Create simple input
            text_processor = create_lightweight_text_processor(vocab_size=1000)
            sample_text = "hello world"
            input_ids = np.array([text_processor.tokenize(sample_text)])
            
            # Generate text
            generated_ids = gpt_model.generate(input_ids, max_length=20, temperature=0.7)
            generated_text = text_processor.detokenize(generated_ids[0].tolist())
            generation_time = time.time() - start_time
            
            self.results["neural_networks"]["Text Generation"] = {
                "status": "success",
                "input_text": sample_text,
                "generated_text": generated_text,
                "generation_time": generation_time,
                "output_length": len(generated_ids[0]),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            logger.info(f"✅ Text Generation: Success - '{sample_text}' -> '{generated_text[:50]}...' ({generation_time:.3f}s)")
            
        except Exception as e:
            self.results["neural_networks"]["General Error"] = {
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
            logger.error(f"❌ Neural Networks: Failed - {e}")
    
    def test_memory_system(self):
        """Test our real memory system"""
        logger.info("Testing real memory system...")
        
        try:
            from bharat_fm.memory.real_memory_system import RealMemorySystem
            
            # Test memory system creation
            start_time = time.time()
            memory_system = RealMemorySystem({
                "max_memory_entries": 1000,
                "max_context_length": 2048,
                "memory_retention_days": 30
            })
            creation_time = time.time() - start_time
            
            self.results["memory_system"]["System Creation"] = {
                "status": "success",
                "creation_time": creation_time,
                "config": {
                    "max_memory_entries": 1000,
                    "max_context_length": 2048,
                    "memory_retention_days": 30
                },
                "timestamp": datetime.utcnow().isoformat()
            }
            
            logger.info(f"✅ Memory System Creation: Success ({creation_time:.3f}s)")
            
            # Test adding memories
            session_id = "test_session_001"
            user_id = "test_user_001"
            
            test_memories = [
                {"content": "Hello, I'm testing the lightweight memory system.", "role": "user"},
                {"content": "Great! I'm working properly with lightweight implementations.", "role": "assistant"},
                {"content": "Can you remember our conversation?", "role": "user"},
                {"content": "Yes, I can remember our conversation with real memory management.", "role": "assistant"},
            ]
            
            memory_ids = []
            for i, memory in enumerate(test_memories):
                start_time = time.time()
                
                # For this demo, we'll simulate the async call
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                try:
                    memory_id = loop.run_until_complete(
                        memory_system.add_memory(
                            content=memory["content"],
                            role=memory["role"],
                            session_id=session_id,
                            user_id=user_id,
                            tags=["test", "demo", "lightweight"],
                            importance_score=0.8
                        )
                    )
                    memory_ids.append(memory_id)
                    test_time = time.time() - start_time
                    
                    self.results["memory_system"][f"Add Memory {i+1}"] = {
                        "status": "success",
                        "memory_id": memory_id,
                        "test_time": test_time,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                    
                    logger.info(f"✅ Add Memory {i+1}: Success - {memory_id} ({test_time:.3f}s)")
                    
                finally:
                    loop.close()
            
            # Test context retrieval
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                start_time = time.time()
                context = loop.run_until_complete(
                    memory_system.get_context(session_id, max_tokens=1000)
                )
                test_time = time.time() - start_time
                
                self.results["memory_system"]["Get Context"] = {
                    "status": "success",
                    "context_length": len(context),
                    "test_time": test_time,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                logger.info(f"✅ Get Context: Success - {len(context)} entries ({test_time:.3f}s)")
                
            finally:
                loop.close()
            
            # Test memory search
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                start_time = time.time()
                search_results = loop.run_until_complete(
                    memory_system.search_memories(
                        query="conversation memory",
                        user_id=user_id,
                        limit=5
                    )
                )
                test_time = time.time() - start_time
                
                self.results["memory_system"]["Search Memories"] = {
                    "status": "success",
                    "results_count": len(search_results),
                    "test_time": test_time,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                logger.info(f"✅ Search Memories: Success - {len(search_results)} results ({test_time:.3f}s)")
                
            finally:
                loop.close()
            
            # Test user profile
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                start_time = time.time()
                profile = loop.run_until_complete(
                    memory_system.get_user_profile(user_id)
                )
                test_time = time.time() - start_time
                
                self.results["memory_system"]["User Profile"] = {
                    "status": "success",
                    "total_memories": profile.get("total_memory_entries", 0),
                    "total_sessions": profile.get("total_sessions", 0),
                    "test_time": test_time,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                logger.info(f"✅ User Profile: Success - {profile.get('total_memory_entries', 0)} memories ({test_time:.3f}s)")
                
            finally:
                loop.close()
            
            # Test memory statistics
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                start_time = time.time()
                stats = loop.run_until_complete(
                    memory_system.get_memory_stats()
                )
                test_time = time.time() - start_time
                
                self.results["memory_system"]["Memory Stats"] = {
                    "status": "success",
                    "total_memories": stats["total_memory_entries"],
                    "total_sessions": stats["total_conversation_sessions"],
                    "total_users": stats["total_users"],
                    "test_time": test_time,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                logger.info(f"✅ Memory Stats: Success - {stats['total_memory_entries']} memories ({test_time:.3f}s)")
                
            finally:
                loop.close()
            
        except Exception as e:
            self.results["memory_system"]["General Error"] = {
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
            logger.error(f"❌ Memory System: Failed - {e}")
    
    def test_text_processing(self):
        """Test our lightweight text processing"""
        logger.info("Testing lightweight text processing...")
        
        try:
            from bharat_fm.data.lightweight_tokenization import LightweightTextProcessor, create_lightweight_text_processor
            
            # Test text processor creation
            start_time = time.time()
            text_processor = LightweightTextProcessor({
                "tokenizer_type": "custom",
                "max_length": 512,
                "lowercase": True,
                "normalize_whitespace": True
            })
            creation_time = time.time() - start_time
            
            self.results["text_processing"]["Processor Creation"] = {
                "status": "success",
                "creation_time": creation_time,
                "config": {
                    "tokenizer_type": "custom",
                    "max_length": 512,
                    "lowercase": True,
                    "normalize_whitespace": True
                },
                "timestamp": datetime.utcnow().isoformat()
            }
            
            logger.info(f"✅ Text Processor Creation: Success ({creation_time:.3f}s)")
            
            # Test text preprocessing
            test_texts = [
                "Hello, World! This is a TEST.",
                "नमस्ते दुनिया! यह एक परीक्षण है।",  # Hindi
                "The quick brown fox jumps over the lazy dog.",
                "  Extra   spaces   here  ",
                "Numbers 123 and symbols !@#"
            ]
            
            for i, text in enumerate(test_texts):
                start_time = time.time()
                processed = text_processor.preprocess_text(text)
                test_time = time.time() - start_time
                
                self.results["text_processing"][f"Text Preprocessing {i+1}"] = {
                    "status": "success",
                    "original_length": len(text),
                    "processed_length": len(processed),
                    "test_time": test_time,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                logger.info(f"✅ Text Preprocessing {i+1}: Success ({test_time:.3f}s)")
            
            # Test tokenization with different tokenizers
            tokenizers = ["custom", "indian", "subword"]
            
            for tokenizer_type in tokenizers:
                try:
                    start_time = time.time()
                    results = text_processor.tokenize_batch(test_texts, tokenizer_type)
                    test_time = time.time() - start_time
                    
                    self.results["text_processing"][f"Tokenization ({tokenizer_type})"] = {
                        "status": "success",
                        "test_time": test_time,
                        "texts_processed": len(test_texts),
                        "avg_tokens_per_text": sum(r.num_tokens for r in results) / len(results),
                        "timestamp": datetime.utcnow().isoformat()
                    }
                    
                    logger.info(f"✅ Tokenization ({tokenizer_type}): Success - {len(results)} texts ({test_time:.3f}s)")
                    
                except Exception as e:
                    self.results["text_processing"][f"Tokenization ({tokenizer_type})"] = {
                        "status": "failed",
                        "error": str(e),
                        "timestamp": datetime.utcnow().isoformat()
                    }
                    logger.error(f"❌ Tokenization ({tokenizer_type}): Failed - {e}")
            
            # Test language detection
            for i, text in enumerate(test_texts):
                try:
                    start_time = time.time()
                    detected_lang = text_processor.detect_language(text)
                    test_time = time.time() - start_time
                    
                    self.results["text_processing"][f"Language Detection {i+1}"] = {
                        "status": "success",
                        "detected_language": detected_lang,
                        "test_time": test_time,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                    
                    logger.info(f"✅ Language Detection {i+1}: {detected_lang} ({test_time:.3f}s)")
                    
                except Exception as e:
                    self.results["text_processing"][f"Language Detection {i+1}"] = {
                        "status": "failed",
                        "error": str(e),
                        "timestamp": datetime.utcnow().isoformat()
                    }
                    logger.error(f"❌ Language Detection {i+1}: Failed - {e}")
            
            # Test tokenizer info
            try:
                start_time = time.time()
                info = text_processor.get_tokenizer_info("custom")
                test_time = time.time() - start_time
                
                self.results["text_processing"]["Tokenizer Info"] = {
                    "status": "success",
                    "vocabulary_size": info["vocabulary_size"],
                    "special_tokens_count": len(info["special_tokens"]),
                    "test_time": test_time,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                logger.info(f"✅ Tokenizer Info: Success - {info['vocabulary_size']} vocab size ({test_time:.3f}s)")
                
            except Exception as e:
                self.results["text_processing"]["Tokenizer Info"] = {
                    "status": "failed",
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                }
                logger.error(f"❌ Tokenizer Info: Failed - {e}")
            
        except Exception as e:
            self.results["text_processing"]["General Error"] = {
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
            logger.error(f"❌ Text Processing: Failed - {e}")
    
    def test_integration(self):
        """Test integration between components"""
        logger.info("Testing component integration...")
        
        try:
            from bharat_fm.memory.real_memory_system import RealMemorySystem
            from bharat_fm.data.lightweight_tokenization import LightweightTextProcessor
            from bharat_fm.core.lightweight_neural_networks import create_lightweight_gpt_model, create_lightweight_text_processor
            
            # Test creating all components
            start_time = time.time()
            
            memory_system = RealMemorySystem({
                "max_memory_entries": 100,
                "max_context_length": 512,
                "memory_retention_days": 7
            })
            
            text_processor = LightweightTextProcessor({
                "tokenizer_type": "custom",
                "max_length": 256,
                "lowercase": True
            })
            
            neural_text_processor = create_lightweight_text_processor(vocab_size=1000)
            gpt_model = create_lightweight_gpt_model(vocab_size=1000, d_model=32, num_heads=2, num_layers=1)
            
            integration_time = time.time() - start_time
            
            self.results["integration"]["Component Creation"] = {
                "status": "success",
                "creation_time": integration_time,
                "components_created": ["memory_system", "text_processor", "neural_text_processor", "gpt_model"],
                "timestamp": datetime.utcnow().isoformat()
            }
            
            logger.info(f"✅ Component Creation: Success - All 4 components created ({integration_time:.3f}s)")
            
            # Test integrated workflow
            start_time = time.time()
            
            # Step 1: Process text
            sample_text = "Hello, this is an integration test!"
            processed_text = text_processor.preprocess_text(sample_text)
            
            # Step 2: Tokenize for neural network
            tokens = neural_text_processor.tokenize(processed_text)
            
            # Step 3: Add to memory
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
                
                # Step 4: Retrieve from memory
                context = loop.run_until_complete(
                    memory_system.get_context("integration_test", max_tokens=100)
                )
                
                # Step 5: Generate response with neural network
                import numpy as np
                input_ids = np.array([tokens[:10]])  # Limit sequence length
                generated_ids = gpt_model.generate(input_ids, max_length=15, temperature=0.8)
                generated_text = neural_text_processor.detokenize(generated_ids[0].tolist())
                
                workflow_time = time.time() - start_time
                
                self.results["integration"]["Integrated Workflow"] = {
                    "status": "success",
                    "original_text": sample_text,
                    "processed_text": processed_text,
                    "memory_id": memory_id,
                    "context_entries": len(context),
                    "generated_response": generated_text,
                    "workflow_time": workflow_time,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                logger.info(f"✅ Integrated Workflow: Success - Full pipeline completed ({workflow_time:.3f}s)")
                
            finally:
                loop.close()
            
        except Exception as e:
            self.results["integration"]["General Error"] = {
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
            logger.error(f"❌ Integration: Failed - {e}")
    
    def run_all_tests(self):
        """Run all tests and generate report"""
        logger.info("Starting comprehensive lightweight AI capabilities test...")
        
        # Run all test suites
        self.test_neural_network_implementations()
        self.test_memory_system()
        self.test_text_processing()
        self.test_integration()
        
        # Generate summary
        self.generate_summary()
        
        # Save results
        self.save_results()
        
        logger.info("✅ All lightweight AI tests completed!")
    
    def generate_summary(self):
        """Generate test summary"""
        total_tests = 0
        successful_tests = 0
        
        for category in self.results.values():
            for test_name, test_result in category.items():
                total_tests += 1
                if test_result.get("status") == "success":
                    successful_tests += 1
        
        success_rate = (successful_tests / total_tests) * 100 if total_tests > 0 else 0
        
        logger.info("\n" + "="*60)
        logger.info("LIGHTWEIGHT AI CAPABILITIES TEST SUMMARY")
        logger.info("="*60)
        logger.info(f"\nTotal Tests: {total_tests}")
        logger.info(f"Successful: {successful_tests}")
        logger.info(f"Failed: {total_tests - successful_tests}")
        logger.info(f"Success Rate: {success_rate:.1f}%")
        
        if success_rate >= 80:
            logger.info("   🎉 EXCELLENT: Most lightweight AI capabilities are working!")
        elif success_rate >= 60:
            logger.info("   👍 GOOD: Most lightweight AI capabilities are functional!")
        else:
            logger.info("   ⚠️  NEEDS WORK: Many lightweight AI capabilities need attention")
        
        logger.info("="*60)
    
    def save_results(self):
        """Save test results to file"""
        results_file = "lightweight_ai_test_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        logger.info(f"Test results saved to {results_file}")

def main():
    """Main function to run the demo"""
    demo = LightweightAIDemo()
    demo.run_all_tests()
    
    # Return results for potential programmatic use
    return demo.results

if __name__ == "__main__":
    main()