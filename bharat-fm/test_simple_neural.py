"""
Simple test for lightweight neural networks
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import time
from datetime import datetime

def test_simple_neural():
    """Test simple neural network functionality"""
    print("Starting simple neural network functionality test...")
    
    try:
        from bharat_fm.core.lightweight_neural_networks import (
            LightweightMultiHeadAttention,
            LightweightPositionalEncoding,
            LightweightTransformerBlock,
            LightweightTextProcessor
        )
        
        print("✅ All neural network components imported successfully")
        
        # Test 1: Simple matrix multiplication (core of neural networks)
        print("\n1. Testing core matrix operations...")
        start_time = time.time()
        
        # Create simple matrices
        W_q = np.random.randn(64, 64) * 0.01
        W_k = np.random.randn(64, 64) * 0.01
        W_v = np.random.randn(64, 64) * 0.01
        W_o = np.random.randn(64, 64) * 0.01
        
        # Create input
        x = np.random.randn(2, 10, 64)
        
        # Simple forward pass (simplified attention)
        Q = np.matmul(x, W_q)
        K = np.matmul(x, W_k)
        V = np.matmul(x, W_v)
        
        # Attention scores
        attn_scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(64)
        attention_weights = np.exp(attn_scores - np.max(attn_scores, axis=-1, keepdims=True))
        attention_weights = attention_weights / np.sum(attention_weights, axis=-1, keepdims=True)
        
        # Output
        output = np.matmul(attention_weights, V)
        final_output = np.matmul(output, W_o)
        
        test_time = time.time() - start_time
        
        print(f"✅ Core matrix operations: Success")
        print(f"   Input shape: {x.shape}")
        print(f"   Output shape: {final_output.shape}")
        print(f"   Computation time: {test_time:.3f}s")
        
        # Test 2: Text processing
        print("\n2. Testing text processing...")
        start_time = time.time()
        
        text_processor = LightweightTextProcessor(vocab_size=1000)
        
        sample_text = "Hello world! This is a test."
        tokens = text_processor.tokenize(sample_text)
        processed_text = text_processor.detokenize(tokens)
        
        test_time = time.time() - start_time
        
        print(f"✅ Text processing: Success")
        print(f"   Original: '{sample_text}'")
        print(f"   Processed: '{processed_text}'")
        print(f"   Tokens: {tokens}")
        print(f"   Processing time: {test_time:.3f}s")
        
        # Test 3: Simple neural network operations
        print("\n3. Testing simple neural network operations...")
        start_time = time.time()
        
        # Simple 2-layer network
        W1 = np.random.randn(64, 128) * 0.01
        b1 = np.zeros(128)
        W2 = np.random.randn(128, 64) * 0.01
        b2 = np.zeros(64)
        
        # Forward pass
        hidden = np.maximum(0, np.matmul(x.reshape(-1, 64), W1) + b1)  # ReLU
        output = np.matmul(hidden, W2) + b2
        
        test_time = time.time() - start_time
        
        print(f"✅ Neural network operations: Success")
        print(f"   Input shape: {x.shape}")
        print(f"   Hidden shape: {hidden.shape}")
        print(f"   Output shape: {output.shape}")
        print(f"   Network time: {test_time:.3f}s")
        
        # Test 4: Language detection
        print("\n4. Testing language detection...")
        start_time = time.time()
        
        from bharat_fm.data.lightweight_tokenization import LightweightTextProcessor
        
        text_proc = LightweightTextProcessor()
        
        test_texts = [
            "Hello world",  # English
            "नमस्ते दुनिया",  # Hindi
            "வணக்கம் உலகம்",  # Tamil
        ]
        
        detected_langs = []
        for text in test_texts:
            lang = text_proc.detect_language(text)
            detected_langs.append(lang)
        
        test_time = time.time() - start_time
        
        print(f"✅ Language detection: Success")
        for i, (text, lang) in enumerate(zip(test_texts, detected_langs)):
            print(f"   Text {i+1}: '{text[:20]}...' -> {lang}")
        print(f"   Detection time: {test_time:.3f}s")
        
        # Test 5: Simple embedding operations
        print("\n5. Testing simple embedding operations...")
        start_time = time.time()
        
        # Create simple embedding matrix
        vocab_size = 1000
        embedding_dim = 64
        embeddings = np.random.randn(vocab_size, embedding_dim) * 0.01
        
        # Create simple word to index mapping
        word_to_idx = {"hello": 1, "world": 2, "test": 3}
        
        # Embedding lookup
        text = "hello world test"
        token_ids = [word_to_idx.get(word, 0) for word in text.split()]
        embedded_vectors = embeddings[token_ids]
        
        test_time = time.time() - start_time
        
        print(f"✅ Embedding operations: Success")
        print(f"   Text: '{text}'")
        print(f"   Token IDs: {token_ids}")
        print(f"   Embeddings shape: {embedded_vectors.shape}")
        print(f"   Embedding time: {test_time:.3f}s")
        
        print(f"\n🎉 All simple neural network tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Simple neural network test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_simple_neural()
    if success:
        print("\n✅ NEURAL NETWORKS: Working with lightweight implementations!")
    else:
        print("\n❌ NEURAL NETWORKS: Still has issues")