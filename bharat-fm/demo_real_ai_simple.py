"""
Simple Demo of Real Bharat-FM AI Capabilities
Demonstrates the real implementations without external dependencies
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

class SimpleRealAIDemo:
    """Simple demonstration of real AI capabilities without external dependencies"""
    
    def __init__(self):
        self.results = {
            "neural_networks": {},
            "memory_system": {},
            "text_processing": {},
            "integration": {}
        }
    
    def test_neural_network_implementations(self):
        """Test our real neural network implementations"""
        logger.info("Testing real neural network implementations...")
        
        try:
            # Test basic neural network components
            from bharat_fm.core.real_neural_networks import (
                RealMultiHeadAttention, 
                RealPositionalEncoding, 
                RealTransformerBlock,
                create_gpt_model,
                create_bert_model
            )
            
            # Test multi-head attention
            start_time = time.time()
            attention = RealMultiHeadAttention(d_model=64, num_heads=4, dropout=0.1)
            
            # Create dummy input
            import torch
            dummy_input = torch.randn(2, 10, 64)  # batch_size=2, seq_len=10, d_model=64
            
            # Test forward pass
            output = attention(dummy_input)
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
            pos_encoding = RealPositionalEncoding(d_model=64, max_len=100)
            pos_output = pos_encoding(dummy_input.transpose(0, 1))  # pos_encoding expects (seq_len, batch_size, d_model)
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
            transformer_block = RealTransformerBlock(d_model=64, num_heads=4, d_ff=256, dropout=0.1)
            transformer_output = transformer_block(dummy_input)
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
            gpt_model = create_gpt_model(vocab_size=1000, d_model=64, num_heads=4, num_layers=2)
            model_creation_time = time.time() - start_time
            
            self.results["neural_networks"]["GPT Model Creation"] = {
                "status": "success",
                "model_type": "GPT",
                "vocab_size": 1000,
                "d_model": 64,
                "num_heads": 4,
                "num_layers": 2,
                "creation_time": model_creation_time,
                "total_parameters": sum(p.numel() for p in gpt_model.parameters()),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            logger.info(f"✅ GPT Model Creation: Success - {sum(p.numel() for p in gpt_model.parameters())} parameters ({model_creation_time:.3f}s)")
            
            # Test BERT model creation
            start_time = time.time()
            bert_model = create_bert_model(vocab_size=1000, d_model=64, num_heads=4, num_layers=2)
            model_creation_time = time.time() - start_time
            
            self.results["neural_networks"]["BERT Model Creation"] = {
                "status": "success",
                "model_type": "BERT",
                "vocab_size": 1000,
                "d_model": 64,
                "num_heads": 4,
                "num_layers": 2,
                "creation_time": model_creation_time,
                "total_parameters": sum(p.numel() for p in bert_model.parameters()),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            logger.info(f"✅ BERT Model Creation: Success - {sum(p.numel() for p in bert_model.parameters())} parameters ({model_creation_time:.3f}s)")
            
        except ImportError as e:
            self.results["neural_networks"]["Import Error"] = {
                "status": "failed",
                "error": f"Missing dependency: {e}",
                "timestamp": datetime.utcnow().isoformat()
            }
            logger.error(f"❌ Neural Networks: Failed - Missing dependency: {e}")
            
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
            from bharat_fm.memory.real_memory_system import RealMemorySystem, create_real_memory_system
            
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
                {"content": "Hello, I'm testing the real memory system.", "role": "user"},
                {"content": "Great! I'm working properly with real implementations.", "role": "assistant"},
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
                            tags=["test", "demo"],
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
            
        except ImportError as e:
            self.results["memory_system"]["Import Error"] = {
                "status": "failed",
                "error": f"Missing dependency: {e}",
                "timestamp": datetime.utcnow().isoformat()
            }
            logger.error(f"❌ Memory System: Failed - Missing dependency: {e}")
            
        except Exception as e:
            self.results["memory_system"]["General Error"] = {
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
            logger.error(f"❌ Memory System: Failed - {e}")
    
    def test_text_processing(self):
        """Test our real text processing"""
        logger.info("Testing real text processing...")
        
        try:
            from bharat_fm.data.real_tokenization import RealTextProcessor, create_real_text_processor
            
            # Test text processor creation
            start_time = time.time()
            text_processor = RealTextProcessor({
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
                
                self.results["text_processing"][f"Preprocess Text {i+1}"] = {
                    "status": "success",
                    "original_length": len(text),
                    "processed_length": len(processed),
                    "test_time": test_time,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                logger.info(f"✅ Preprocess Text {i+1}: Success - {len(text)} -> {len(processed)} chars ({test_time:.3f}s)")
            
            # Test tokenization
            for i, text in enumerate(test_texts[:3]):  # Test first 3 texts
                start_time = time.time()
                result = text_processor.tokenize(text, tokenizer_type="indian")
                test_time = time.time() - start_time
                
                self.results["text_processing"][f"Tokenize Text {i+1}"] = {
                    "status": "success",
                    "original_text": text[:30] + "...",
                    "tokens_count": result.num_tokens,
                    "test_time": test_time,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                logger.info(f"✅ Tokenize Text {i+1}: Success - {result.num_tokens} tokens ({test_time:.3f}s)")
            
            # Test language detection
            for i, text in enumerate(test_texts):
                start_time = time.time()
                detected_lang = text_processor.detect_language(text)
                test_time = time.time() - start_time
                
                self.results["text_processing"][f"Language Detection {i+1}"] = {
                    "status": "success",
                    "detected_language": detected_lang,
                    "text_preview": text[:20] + "...",
                    "test_time": test_time,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                logger.info(f"✅ Language Detection {i+1}: Success - {detected_lang} ({test_time:.3f}s)")
            
            # Test batch processing
            start_time = time.time()
            batch_results = text_processor.tokenize_batch(test_texts, tokenizer_type="indian")
            test_time = time.time() - start_time
            
            self.results["text_processing"]["Batch Processing"] = {
                "status": "success",
                "texts_processed": len(test_texts),
                "avg_tokens_per_text": sum(r.num_tokens for r in batch_results) / len(batch_results),
                "test_time": test_time,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            logger.info(f"✅ Batch Processing: Success - {len(test_texts)} texts ({test_time:.3f}s)")
            
            # Test tokenizer info
            start_time = time.time()
            info = text_processor.get_tokenizer_info("indian")
            test_time = time.time() - start_time
            
            self.results["text_processing"]["Tokenizer Info"] = {
                "status": "success",
                "vocabulary_size": info["vocabulary_size"],
                "max_length": info["max_length"],
                "special_tokens_count": len(info["special_tokens"]),
                "test_time": test_time,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            logger.info(f"✅ Tokenizer Info: Success - {info['vocabulary_size']} vocab size ({test_time:.3f}s)")
            
        except ImportError as e:
            self.results["text_processing"]["Import Error"] = {
                "status": "failed",
                "error": f"Missing dependency: {e}",
                "timestamp": datetime.utcnow().isoformat()
            }
            logger.error(f"❌ Text Processing: Failed - Missing dependency: {e}")
            
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
            # Test that all components can be imported
            start_time = time.time()
            
            from bharat_fm.core.real_neural_networks import create_gpt_model
            from bharat_fm.memory.real_memory_system import RealMemorySystem
            from bharat_fm.data.real_tokenization import RealTextProcessor
            
            # Create instances
            gpt_model = create_gpt_model(vocab_size=1000, d_model=64, num_heads=4, num_layers=2)
            memory_system = RealMemorySystem({"max_memory_entries": 100})
            text_processor = RealTextProcessor({"max_length": 256})
            
            integration_time = time.time() - start_time
            
            self.results["integration"]["Component Integration"] = {
                "status": "success",
                "components_loaded": 3,
                "model_parameters": sum(p.numel() for p in gpt_model.parameters()),
                "integration_time": integration_time,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            logger.info(f"✅ Component Integration: Success - 3 components loaded ({integration_time:.3f}s)")
            
            # Test simple workflow: text -> process -> store
            start_time = time.time()
            
            # Process text
            test_text = "The weather is beautiful today."
            processed = text_processor.preprocess_text(test_text)
            tokenized = text_processor.tokenize(test_text, tokenizer_type="indian")
            
            # Store in memory (simulate async)
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                memory_id = loop.run_until_complete(
                    memory_system.add_memory(
                        content=test_text,
                        role="user",
                        session_id="integration_test",
                        user_id="test_user",
                        tags=["integration", "demo"]
                    )
                )
                
                workflow_time = time.time() - start_time
                
                self.results["integration"]["Text Processing Workflow"] = {
                    "status": "success",
                    "steps_completed": 3,
                    "original_text": test_text,
                    "processed_text": processed,
                    "tokens_count": tokenized.num_tokens,
                    "memory_id": memory_id,
                    "workflow_time": workflow_time,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                logger.info(f"✅ Text Processing Workflow: Success - 3 steps completed ({workflow_time:.3f}s)")
                
            finally:
                loop.close()
            
        except ImportError as e:
            self.results["integration"]["Import Error"] = {
                "status": "failed",
                "error": f"Missing dependency: {e}",
                "timestamp": datetime.utcnow().isoformat()
            }
            logger.error(f"❌ Integration: Failed - Missing dependency: {e}")
            
        except Exception as e:
            self.results["integration"]["General Error"] = {
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
            logger.error(f"❌ Integration: Failed - {e}")
    
    def save_results(self, filename: str = "simple_real_ai_test_results.json"):
        """Save test results to file"""
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"Test results saved to {filename}")
    
    def print_summary(self):
        """Print test summary"""
        logger.info("\n" + "="*60)
        logger.info("SIMPLE REAL AI CAPABILITIES TEST SUMMARY")
        logger.info("="*60)
        
        total_tests = 0
        successful_tests = 0
        
        for category, tests in self.results.items():
            logger.info(f"\n{category.upper().replace('_', ' ')}:")
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

def main():
    """Main test function"""
    demo = SimpleRealAIDemo()
    
    try:
        # Run all tests
        demo.test_neural_network_implementations()
        demo.test_memory_system()
        demo.test_text_processing()
        demo.test_integration()
        
        # Print summary
        demo.print_summary()
        
        # Save results
        demo.save_results()
        
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        raise

if __name__ == "__main__":
    main()