"""
Lightweight Neural Network Implementations for Bharat-FM
Real implementations without heavy PyTorch dependencies
Uses numpy for computations instead of PyTorch
"""

import numpy as np
import math
from typing import Optional, Tuple, Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class LightweightMultiHeadAttention:
    """Lightweight multi-head attention implementation using numpy"""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Initialize weight matrices
        self.W_q = np.random.randn(d_model, d_model) * 0.01
        self.W_k = np.random.randn(d_model, d_model) * 0.01
        self.W_v = np.random.randn(d_model, d_model) * 0.01
        self.W_o = np.random.randn(d_model, d_model) * 0.01
        
        self.dropout = dropout
        self.attention_weights = None
    
    def scaled_dot_product_attention(self, 
                                   Q: np.ndarray, 
                                   K: np.ndarray, 
                                   V: np.ndarray,
                                   mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Compute scaled dot-product attention"""
        # Calculate attention scores
        attn_scores = np.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply mask if provided
        if mask is not None:
            attn_scores = np.where(mask == 0, -1e9, attn_scores)
        
        # Apply softmax to get attention weights
        attention_weights = self._softmax(attn_scores, axis=-1)
        
        # Apply dropout
        if self.dropout > 0:
            dropout_mask = np.random.binomial(1, 1 - self.dropout, size=attention_weights.shape)
            attention_weights = attention_weights * dropout_mask / (1 - self.dropout)
        
        # Compute output
        output = np.matmul(attention_weights, V)
        
        return output, attention_weights
    
    def _softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Softmax implementation"""
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    
    def split_heads(self, x: np.ndarray) -> np.ndarray:
        """Split tensor into multiple heads"""
        batch_size, seq_len, d_model = x.shape
        return x.reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
    
    def combine_heads(self, x: np.ndarray) -> np.ndarray:
        """Combine multiple heads"""
        batch_size, num_heads, seq_len, d_k = x.shape
        return x.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)
    
    def forward(self, 
                x: np.ndarray, 
                mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Forward pass"""
        batch_size, seq_len, d_model = x.shape
        
        # Linear projections and split heads
        Q = np.matmul(x, self.W_q)
        K = np.matmul(x, self.W_k)
        V = np.matmul(x, self.W_v)
        
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)
        
        # Compute attention
        attn_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        self.attention_weights = attention_weights
        
        # Combine heads and final linear projection
        output = self.combine_heads(attn_output)
        output = np.matmul(output, self.W_o)
        
        return output

class LightweightPositionalEncoding:
    """Lightweight positional encoding for transformer models"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        # Create positional encoding
        pe = np.zeros((max_len, d_model))
        position = np.arange(0, max_len, dtype=np.float32).reshape(-1, 1)
        div_term = np.exp(np.arange(0, d_model, 2) * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        self.pe = pe
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Add positional encoding to input"""
        seq_len = x.shape[1]
        return x + self.pe[:seq_len, :]

class LightweightTransformerBlock:
    """Lightweight transformer block with multi-head attention and feed-forward network"""
    
    def __init__(self, 
                 d_model: int, 
                 num_heads: int, 
                 d_ff: int, 
                 dropout: float = 0.1):
        super().__init__()
        
        # Multi-head attention
        self.attention = LightweightMultiHeadAttention(d_model, num_heads, dropout)
        
        # Layer normalization parameters
        self.gamma1 = np.ones(d_model)
        self.beta1 = np.zeros(d_model)
        self.gamma2 = np.ones(d_model)
        self.beta2 = np.zeros(d_model)
        
        # Feed-forward network parameters
        self.W1 = np.random.randn(d_model, d_ff) * 0.01
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) * 0.01
        self.b2 = np.zeros(d_model)
        
        self.dropout = dropout
    
    def _layer_norm(self, x: np.ndarray, gamma: np.ndarray, beta: np.ndarray) -> np.ndarray:
        """Layer normalization"""
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        return gamma * (x - mean) / np.sqrt(var + 1e-6) + beta
    
    def _relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU activation"""
        return np.maximum(0, x)
    
    def _dropout(self, x: np.ndarray) -> np.ndarray:
        """Apply dropout"""
        if self.dropout > 0:
            mask = np.random.binomial(1, 1 - self.dropout, size=x.shape)
            return x * mask / (1 - self.dropout)
        return x
    
    def forward(self, 
                x: np.ndarray, 
                mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Forward pass"""
        # Self-attention with residual connection
        attn_output = self.attention.forward(x, mask)
        x = self._layer_norm(x + self._dropout(attn_output), self.gamma1, self.beta1)
        
        # Feed-forward with residual connection
        ff_output = self._relu(np.matmul(x, self.W1) + self.b1)
        ff_output = np.matmul(ff_output, self.W2) + self.b2
        x = self._layer_norm(x + self._dropout(ff_output), self.gamma2, self.beta2)
        
        return x

class LightweightTransformerEncoder:
    """Lightweight transformer encoder"""
    
    def __init__(self, 
                 vocab_size: int, 
                 d_model: int, 
                 num_heads: int, 
                 num_layers: int, 
                 d_ff: int, 
                 max_len: int = 5000,
                 dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Embedding matrix
        self.embedding = np.random.randn(vocab_size, d_model) * 0.01
        self.pos_encoding = LightweightPositionalEncoding(d_model, max_len)
        
        # Transformer layers
        self.layers = [
            LightweightTransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ]
        
        self.dropout = dropout
    
    def _dropout(self, x: np.ndarray) -> np.ndarray:
        """Apply dropout"""
        if self.dropout > 0:
            mask = np.random.binomial(1, 1 - self.dropout, size=x.shape)
            return x * mask / (1 - self.dropout)
        return x
    
    def forward(self, 
                input_ids: np.ndarray, 
                mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Forward pass"""
        batch_size, seq_len = input_ids.shape
        
        # Embedding and positional encoding
        x = self.embedding[input_ids] * math.sqrt(self.d_model)
        x = self.pos_encoding.forward(x)
        x = self._dropout(x)
        
        # Transformer layers
        for layer in self.layers:
            x = layer.forward(x, mask)
        
        return x

class LightweightGPTStyleModel:
    """Lightweight GPT-style language model"""
    
    def __init__(self, 
                 vocab_size: int, 
                 d_model: int = 768, 
                 num_heads: int = 12, 
                 num_layers: int = 12, 
                 d_ff: int = 3072,
                 max_len: int = 1024,
                 dropout: float = 0.1):
        super().__init__()
        
        self.transformer = LightweightTransformerEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_ff=d_ff,
            max_len=max_len,
            dropout=dropout
        )
        
        # Language model head
        self.lm_head = np.random.randn(d_model, vocab_size) * 0.01
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        # Small random initialization for all parameters
        scale = 0.02
        for layer in self.transformer.layers:
            layer.attention.W_q *= scale
            layer.attention.W_k *= scale
            layer.attention.W_v *= scale
            layer.attention.W_o *= scale
            layer.W1 *= scale
            layer.W2 *= scale
        
        self.transformer.embedding *= scale
        self.lm_head *= scale
    
    def forward(self, 
                input_ids: np.ndarray, 
                attention_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Forward pass"""
        transformer_output = self.transformer.forward(input_ids, attention_mask)
        logits = np.matmul(transformer_output, self.lm_head)
        return logits
    
    def generate(self, 
                 input_ids: np.ndarray, 
                 max_length: int = 100,
                 temperature: float = 1.0,
                 do_sample: bool = True,
                 top_k: Optional[int] = None,
                 top_p: Optional[float] = None) -> np.ndarray:
        """Generate text autoregressively"""
        batch_size, seq_len = input_ids.shape
        
        for _ in range(max_length):
            # Get model predictions
            logits = self.forward(input_ids)
            
            # Focus on last token
            next_token_logits = logits[:, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k is not None:
                indices = np.argpartition(next_token_logits, -top_k, axis=-1)
                mask = np.zeros_like(next_token_logits)
                mask[np.arange(batch_size)[:, None], indices[:, -top_k:]] = 1
                next_token_logits = np.where(mask == 0, -1e9, next_token_logits)
            
            # Apply top-p filtering
            if top_p is not None:
                sorted_logits = np.sort(next_token_logits, axis=-1)[:, ::-1]
                sorted_indices = np.argsort(next_token_logits, axis=-1)[:, ::-1]
                cumulative_probs = np.cumsum(self._softmax(sorted_logits, axis=-1), axis=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].copy()
                sorted_indices_to_remove[:, 0] = 0
                
                indices_to_remove = np.zeros_like(next_token_logits, dtype=bool)
                np.put_along_axis(indices_to_remove, sorted_indices, sorted_indices_to_remove, axis=-1)
                next_token_logits = np.where(indices_to_remove, -1e9, next_token_logits)
            
            # Sample next token
            probs = self._softmax(next_token_logits, axis=-1)
            if do_sample:
                next_token = np.array([
                    np.random.choice(len(probs[i]), p=probs[i])
                    for i in range(batch_size)
                ])
            else:
                next_token = np.argmax(probs, axis=-1)
            
            # Append to input
            input_ids = np.concatenate([input_ids, next_token.reshape(-1, 1)], axis=1)
        
        return input_ids
    
    def _softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Softmax implementation"""
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

class LightweightTextProcessor:
    """Lightweight text processing for neural networks"""
    
    def __init__(self, vocab_size: int = 30000):
        self.vocab_size = vocab_size
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.special_tokens = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<BOS>': 2,
            '<EOS>': 3,
            '<MASK>': 4
        }
        
        # Initialize vocabulary
        self._initialize_vocab()
    
    def _initialize_vocab(self):
        """Initialize basic vocabulary"""
        self.word_to_idx.update(self.special_tokens)
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
    
    def tokenize(self, text: str) -> list:
        """Simple tokenization"""
        # Simple whitespace-based tokenization
        tokens = text.lower().split()
        return [self.word_to_idx.get(token, self.word_to_idx['<UNK>']) for token in tokens]
    
    def detokenize(self, tokens: list) -> str:
        """Convert tokens back to text"""
        words = [self.idx_to_word.get(token, '<UNK>') for token in tokens]
        return ' '.join(words)
    
    def add_tokens(self, tokens: list):
        """Add new tokens to vocabulary"""
        for token in tokens:
            if token not in self.word_to_idx:
                if len(self.word_to_idx) < self.vocab_size:
                    new_idx = len(self.word_to_idx)
                    self.word_to_idx[token] = new_idx
                    self.idx_to_word[new_idx] = token
    
    def create_attention_mask(self, input_ids: np.ndarray) -> np.ndarray:
        """Create attention mask"""
        return (input_ids != self.word_to_idx['<PAD>']).astype(np.int32)

# Factory functions for creating models
def create_lightweight_gpt_model(vocab_size: int = 30000, 
                                d_model: int = 768, 
                                num_heads: int = 12, 
                                num_layers: int = 12) -> LightweightGPTStyleModel:
    """Create a lightweight GPT-style model"""
    return LightweightGPTStyleModel(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers
    )

def create_lightweight_text_processor(vocab_size: int = 30000) -> LightweightTextProcessor:
    """Create a lightweight text processor"""
    return LightweightTextProcessor(vocab_size)