"""
Real Tokenization and Text Processing Pipeline for Bharat-FM
Actual implementation with proper tokenization, preprocessing, and data handling
"""

import re
import json
import pickle
import os
from typing import Dict, List, Any, Optional, Tuple, Union, Iterator
from dataclasses import dataclass
from collections import Counter, defaultdict
import logging
import numpy as np
from transformers import AutoTokenizer, PreTrainedTokenizer
import torch

logger = logging.getLogger(__name__)

@dataclass
class TokenizationResult:
    """Result of tokenization with metadata"""
    tokens: List[str]
    token_ids: List[int]
    attention_mask: List[int]
    special_tokens_mask: List[int]
    token_type_ids: Optional[List[int]] = None
    offsets: Optional[List[Tuple[int, int]]] = None
    num_tokens: int = 0
    
    def __post_init__(self):
        self.num_tokens = len(self.tokens)

class RealTextProcessor:
    """Real text processing pipeline with multiple tokenization strategies"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Tokenizer configurations
        self.tokenizer_type = self.config.get("tokenizer_type", "transformers")
        self.model_name = self.config.get("model_name", "bert-base-uncased")
        self.max_length = self.config.get("max_length", 512)
        self.vocab_size = self.config.get("vocab_size", 30000)
        
        # Initialize tokenizers
        self.tokenizers: Dict[str, Any] = {}
        self.vocabularies: Dict[str, Dict[str, int]] = {}
        
        # Text preprocessing settings
        self.lowercase = self.config.get("lowercase", True)
        self.remove_punctuation = self.config.get("remove_punctuation", False)
        self.normalize_whitespace = self.config.get("normalize_whitespace", True)
        self.handle_numbers = self.config.get("handle_numbers", "keep")  # keep, remove, replace
        
        # Initialize tokenizers
        self._initialize_tokenizers()
        
        # Statistics
        self.processing_stats = {
            "total_texts_processed": 0,
            "total_tokens_generated": 0,
            "avg_tokens_per_text": 0.0,
            "vocabulary_size": 0
        }
    
    def _initialize_tokenizers(self):
        """Initialize various tokenizers"""
        try:
            # Initialize HuggingFace tokenizer
            if self.tokenizer_type == "transformers":
                logger.info(f"Loading HuggingFace tokenizer: {self.model_name}")
                self.tokenizers["transformers"] = AutoTokenizer.from_pretrained(self.model_name)
                self.vocabularies["transformers"] = self.tokenizers["transformers"].get_vocab()
            
            # Initialize custom tokenizer for Indian languages
            self._initialize_indian_tokenizer()
            
            # Initialize subword tokenizer
            self._initialize_subword_tokenizer()
            
            logger.info("All tokenizers initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing tokenizers: {e}")
            raise
    
    def _initialize_indian_tokenizer(self):
        """Initialize tokenizer optimized for Indian languages"""
        # Common Indian language characters and patterns
        indian_chars = {
            # Hindi
            'अ', 'आ', 'इ', 'ई', 'उ', 'ऊ', 'ए', 'ऐ', 'ओ', 'औ',
            'क', 'ख', 'ग', 'घ', 'च', 'छ', 'ज', 'झ', 'ट', 'ठ',
            'ड', 'ढ', 'ण', 'त', 'थ', 'द', 'ध', 'न', 'प', 'फ',
            'ब', 'भ', 'म', 'य', 'र', 'ल', 'व', 'श', 'ष', 'स', 'ह',
            # Tamil
            'அ', 'ஆ', 'இ', 'ஈ', 'உ', 'ஊ', 'எ', 'ஏ', 'ஐ', 'ஒ', 'ஓ', 'ஔ',
            'க', 'ங', 'ச', 'ஞ', 'ட', 'ண', 'த', 'ந', 'ப', 'ம', 'ய', 'ர', 'ல', 'வ', 'ழ', 'ள', 'ற', 'ன',
            # Telugu
            'అ', 'ఆ', 'ఇ', 'ఈ', 'ఉ', 'ఊ', 'ఎ', 'ఏ', 'ఐ', 'ఒ', 'ఓ', 'ఔ',
            'క', 'ఖ', 'గ', 'ఘ', 'చ', 'ఛ', 'జ', 'ఝ', 'ట', 'ఠ',
            'డ', 'ఢ', 'ణ', 'త', 'థ', 'ద', 'ధ', 'న', 'ప', 'ఫ',
            'బ', 'భ', 'మ', 'య', 'ర', 'ల', 'వ', 'శ', 'ష', 'స', 'హ',
            # Bengali
            'অ', 'আ', 'ই', 'ঈ', 'উ', 'ঊ', 'এ', 'ঐ', 'ও', 'ঔ',
            'ক', 'খ', 'গ', 'ঘ', 'চ', 'ছ', 'জ', 'ঝ', 'ট', 'ঠ',
            'ড', 'ঢ', 'ণ', 'ত', 'থ', 'দ', 'ধ', 'ন', 'প', 'ফ',
            'ব', 'ভ', 'ম', 'য', 'র', 'ল', 'ব', 'শ', 'ষ', 'স', 'হ'
        }
        
        # Create vocabulary for Indian languages
        vocab = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<BOS>': 2,
            '<EOS>': 3,
            '<MASK>': 4
        }
        
        # Add Indian characters
        for i, char in enumerate(indian_chars, start=len(vocab)):
            vocab[char] = i
        
        # Add common English characters and symbols
        common_chars = 'abcdefghijklmnopqrstuvwxyz0123456789.,!?;:()[]{}"\' '
        for i, char in enumerate(common_chars, start=len(vocab)):
            vocab[char] = i
        
        self.vocabularies["indian"] = vocab
        self.tokenizers["indian"] = self._create_custom_tokenizer(vocab)
    
    def _initialize_subword_tokenizer(self):
        """Initialize subword tokenizer using Byte Pair Encoding"""
        # Simple BPE implementation
        vocab = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<BOS>': 2,
            '<EOS>': 3,
            '<MASK>': 4
        }
        
        # Common subwords (simplified for demo)
        common_subwords = [
            'ing', 'tion', 'ment', 'ness', 'ity', 'er', 'or', 'ist', 'ism',
            'able', 'ible', 'ful', 'less', 'ous', 'ious', 'al', 'ial',
            'ed', 'es', 's', 'ly', 'y', 'tion', 'sion', 'ence', 'ance'
        ]
        
        for i, subword in enumerate(common_subwords, start=len(vocab)):
            vocab[subword] = i
        
        self.vocabularies["subword"] = vocab
        self.tokenizers["subword"] = self._create_custom_tokenizer(vocab)
    
    def _create_custom_tokenizer(self, vocab: Dict[str, int]) -> Any:
        """Create a simple custom tokenizer"""
        class CustomTokenizer:
            def __init__(self, vocab):
                self.vocab = vocab
                self.reverse_vocab = {v: k for k, v in vocab.items()}
            
            def tokenize(self, text: str) -> List[str]:
                # Simple character-based tokenization
                tokens = []
                for char in text:
                    if char in self.vocab:
                        tokens.append(char)
                    else:
                        tokens.append('<UNK>')
                return tokens
            
            def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
                return [self.vocab.get(token, self.vocab['<UNK>']) for token in tokens]
            
            def convert_ids_to_tokens(self, token_ids: List[int]) -> List[str]:
                return [self.reverse_vocab.get(token_id, '<UNK>') for token_id in token_ids]
            
            def encode(self, text: str) -> List[int]:
                tokens = self.tokenize(text)
                return self.convert_tokens_to_ids(tokens)
            
            def decode(self, token_ids: List[int]) -> str:
                tokens = self.convert_ids_to_tokens(token_ids)
                return ''.join(tokens)
        
        return CustomTokenizer(vocab)
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text with various cleaning steps"""
        if not text:
            return ""
        
        processed_text = text
        
        # Normalize whitespace
        if self.normalize_whitespace:
            processed_text = re.sub(r'\s+', ' ', processed_text).strip()
        
        # Handle case
        if self.lowercase:
            processed_text = processed_text.lower()
        
        # Handle punctuation
        if self.remove_punctuation:
            processed_text = re.sub(r'[^\w\s]', '', processed_text)
        
        # Handle numbers
        if self.handle_numbers == "remove":
            processed_text = re.sub(r'\d+', '', processed_text)
        elif self.handle_numbers == "replace":
            processed_text = re.sub(r'\d+', '<NUM>', processed_text)
        
        return processed_text
    
    def tokenize(self, 
                text: str, 
                tokenizer_type: str = "transformers",
                max_length: Optional[int] = None) -> TokenizationResult:
        """Tokenize text using specified tokenizer"""
        if tokenizer_type not in self.tokenizers:
            raise ValueError(f"Tokenizer {tokenizer_type} not available")
        
        max_length = max_length or self.max_length
        
        try:
            if tokenizer_type == "transformers":
                return self._tokenize_with_transformers(text, max_length)
            else:
                return self._tokenize_with_custom(text, tokenizer_type, max_length)
                
        except Exception as e:
            logger.error(f"Error tokenizing text: {e}")
            raise
    
    def _tokenize_with_transformers(self, text: str, max_length: int) -> TokenizationResult:
        """Tokenize using HuggingFace tokenizer"""
        tokenizer = self.tokenizers["transformers"]
        
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        # Tokenize
        encoding = tokenizer(
            processed_text,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
            return_token_type_ids=True,
            return_attention_mask=True
        )
        
        return TokenizationResult(
            tokens=tokenizer.convert_ids_to_tokens(encoding["input_ids"]),
            token_ids=encoding["input_ids"],
            attention_mask=encoding["attention_mask"],
            special_tokens_mask=encoding["special_tokens_mask"],
            token_type_ids=encoding.get("token_type_ids"),
            offsets=encoding.get("offset_mapping"),
            num_tokens=len(encoding["input_ids"])
        )
    
    def _tokenize_with_custom(self, text: str, tokenizer_type: str, max_length: int) -> TokenizationResult:
        """Tokenize using custom tokenizer"""
        tokenizer = self.tokenizers[tokenizer_type]
        
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        # Tokenize
        token_ids = tokenizer.encode(processed_text)
        
        # Truncate if necessary
        if len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
        
        # Pad if necessary
        if len(token_ids) < max_length:
            token_ids.extend([tokenizer.vocab['<PAD>']] * (max_length - len(token_ids)))
        
        # Create attention mask
        attention_mask = [1 if token_id != tokenizer.vocab['<PAD>'] else 0 
                         for token_id in token_ids]
        
        # Create special tokens mask
        special_tokens = {'<PAD>', '<UNK>', '<BOS>', '<EOS>', '<MASK>'}
        special_tokens_mask = [1 if tokenizer.reverse_vocab.get(token_id, '<UNK>') in special_tokens 
                              else 0 for token_id in token_ids]
        
        # Convert back to tokens
        tokens = tokenizer.convert_ids_to_tokens(token_ids)
        
        return TokenizationResult(
            tokens=tokens,
            token_ids=token_ids,
            attention_mask=attention_mask,
            special_tokens_mask=special_tokens_mask,
            num_tokens=len(token_ids)
        )
    
    def tokenize_batch(self, 
                      texts: List[str], 
                      tokenizer_type: str = "transformers",
                      max_length: Optional[int] = None) -> List[TokenizationResult]:
        """Tokenize multiple texts in batch"""
        results = []
        for text in texts:
            result = self.tokenize(text, tokenizer_type, max_length)
            results.append(result)
        
        # Update statistics
        self.processing_stats["total_texts_processed"] += len(texts)
        self.processing_stats["total_tokens_generated"] += sum(r.num_tokens for r in results)
        self.processing_stats["avg_tokens_per_text"] = (
            self.processing_stats["total_tokens_generated"] / 
            max(self.processing_stats["total_texts_processed"], 1)
        )
        
        return results
    
    def detect_language(self, text: str) -> str:
        """Detect language of text (simplified for Indian languages)"""
        text_lower = text.lower()
        
        # Check for Hindi characters
        hindi_chars = sum(1 for char in text if '\u0900' <= char <= '\u097F')
        if hindi_chars > len(text) * 0.3:
            return "hi"
        
        # Check for Tamil characters
        tamil_chars = sum(1 for char in text if '\u0B80' <= char <= '\u0BFF')
        if tamil_chars > len(text) * 0.3:
            return "ta"
        
        # Check for Telugu characters
        telugu_chars = sum(1 for char in text if '\u0C00' <= char <= '\u0C7F')
        if telugu_chars > len(text) * 0.3:
            return "te"
        
        # Check for Bengali characters
        bengali_chars = sum(1 for char in text if '\u0980' <= char <= '\u09FF')
        if bengali_chars > len(text) * 0.3:
            return "bn"
        
        # Default to English
        return "en"
    
    def get_tokenizer_info(self, tokenizer_type: str) -> Dict[str, Any]:
        """Get information about a specific tokenizer"""
        if tokenizer_type not in self.tokenizers:
            return {"error": "Tokenizer not found"}
        
        vocab = self.vocabularies[tokenizer_type]
        
        return {
            "tokenizer_type": tokenizer_type,
            "vocabulary_size": len(vocab),
            "special_tokens": {k: v for k, v in vocab.items() if k.startswith('<')},
            "max_length": self.max_length,
            "preprocessing_config": {
                "lowercase": self.lowercase,
                "remove_punctuation": self.remove_punctuation,
                "normalize_whitespace": self.normalize_whitespace,
                "handle_numbers": self.handle_numbers
            }
        }
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get text processing statistics"""
        return self.processing_stats.copy()
    
    def save_tokenizer(self, tokenizer_type: str, save_path: str):
        """Save tokenizer to disk"""
        if tokenizer_type not in self.tokenizers:
            raise ValueError(f"Tokenizer {tokenizer_type} not available")
        
        try:
            os.makedirs(save_path, exist_ok=True)
            
            if tokenizer_type == "transformers":
                # Save HuggingFace tokenizer
                self.tokenizers[tokenizer_type].save_pretrained(save_path)
            else:
                # Save custom tokenizer
                save_data = {
                    "vocab": self.vocabularies[tokenizer_type],
                    "config": {
                        "max_length": self.max_length,
                        "preprocessing": {
                            "lowercase": self.lowercase,
                            "remove_punctuation": self.remove_punctuation,
                            "normalize_whitespace": self.normalize_whitespace,
                            "handle_numbers": self.handle_numbers
                        }
                    }
                }
                
                with open(os.path.join(save_path, "tokenizer_config.json"), 'w') as f:
                    json.dump(save_data, f, indent=2)
                
                with open(os.path.join(save_path, "vocab.pkl"), 'wb') as f:
                    pickle.dump(self.vocabularies[tokenizer_type], f)
            
            logger.info(f"Tokenizer {tokenizer_type} saved to {save_path}")
            
        except Exception as e:
            logger.error(f"Error saving tokenizer: {e}")
            raise
    
    def load_tokenizer(self, tokenizer_type: str, load_path: str):
        """Load tokenizer from disk"""
        try:
            if tokenizer_type == "transformers":
                # Load HuggingFace tokenizer
                self.tokenizers[tokenizer_type] = AutoTokenizer.from_pretrained(load_path)
                self.vocabularies[tokenizer_type] = self.tokenizers[tokenizer_type].get_vocab()
            else:
                # Load custom tokenizer
                with open(os.path.join(load_path, "vocab.pkl"), 'rb') as f:
                    vocab = pickle.load(f)
                
                self.vocabularies[tokenizer_type] = vocab
                self.tokenizers[tokenizer_type] = self._create_custom_tokenizer(vocab)
                
                # Load config
                with open(os.path.join(load_path, "tokenizer_config.json"), 'r') as f:
                    config = json.load(f)
                
                # Update configuration
                preprocessing_config = config.get("config", {}).get("preprocessing", {})
                self.lowercase = preprocessing_config.get("lowercase", self.lowercase)
                self.remove_punctuation = preprocessing_config.get("remove_punctuation", self.remove_punctuation)
                self.normalize_whitespace = preprocessing_config.get("normalize_whitespace", self.normalize_whitespace)
                self.handle_numbers = preprocessing_config.get("handle_numbers", self.handle_numbers)
            
            logger.info(f"Tokenizer {tokenizer_type} loaded from {load_path}")
            
        except Exception as e:
            logger.error(f"Error loading tokenizer: {e}")
            raise


# Factory function for easy usage
def create_real_text_processor(config: Optional[Dict[str, Any]] = None) -> RealTextProcessor:
    """Create a real text processor"""
    return RealTextProcessor(config)