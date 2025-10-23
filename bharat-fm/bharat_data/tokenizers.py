"""
Tokenizers specifically designed for Indian languages
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union
from transformers import AutoTokenizer, PreTrainedTokenizer


class IndicTokenizer:
    """
    Custom tokenizer for Indian languages with support for
    various scripts and transliteration.
    """
    
    def __init__(
        self,
        base_model: str = "ai4bharat/indic-bert",
        languages: List[str] = None,
        max_length: int = 512
    ):
        self.base_model = base_model
        self.languages = languages or ["hi", "en", "bn", "ta", "te", "mr", "gu", "kn", "ml", "pa"]
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        
    def tokenize(self, text: str, language: str = None) -> Dict:
        """Tokenize text with optional language specification"""
        if language and language not in self.languages:
            raise ValueError(f"Language {language} not supported. Supported: {self.languages}")
            
        return self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
    def tokenize_batch(self, texts: List[str], languages: List[str] = None) -> Dict:
        """Tokenize batch of texts"""
        return self.tokenizer(
            texts,
            max_length=self.max_length,
            truncation=True,
            padding=True,
            return_tensors="pt"
        )
        
    def decode(self, token_ids: Union[List[int], List[List[int]]]) -> str:
        """Decode token IDs back to text"""
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)
        
    def get_vocab_size(self) -> int:
        """Get vocabulary size"""
        return self.tokenizer.vocab_size
        
    def get_special_tokens(self) -> Dict[str, int]:
        """Get special token IDs"""
        return {
            "pad_token": self.tokenizer.pad_token_id,
            "unk_token": self.tokenizer.unk_token_id,
            "cls_token": self.tokenizer.cls_token_id,
            "sep_token": self.tokenizer.sep_token_id,
            "mask_token": self.tokenizer.mask_token_id if hasattr(self.tokenizer, 'mask_token_id') else None
        }
        
    def save_pretrained(self, path: str):
        """Save tokenizer to disk"""
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save base tokenizer
        self.tokenizer.save_pretrained(save_path)
        
        # Save custom config
        config = {
            "base_model": self.base_model,
            "languages": self.languages,
            "max_length": self.max_length
        }
        
        with open(save_path / "indic_tokenizer_config.json", 'w') as f:
            json.dump(config, f, indent=2)
            
    @classmethod
    def from_pretrained(cls, path: str):
        """Load tokenizer from disk"""
        load_path = Path(path)
        
        # Load custom config
        with open(load_path / "indic_tokenizer_config.json", 'r') as f:
            config = json.load(f)
            
        # Create instance
        instance = cls(
            base_model=config["base_model"],
            languages=config["languages"],
            max_length=config["max_length"]
        )
        
        # Load tokenizer
        instance.tokenizer = AutoTokenizer.from_pretrained(load_path)
        
        return instance


class BharatTokenizer:
    """
    Advanced tokenizer for BharatFM with support for:
    - Multiple Indian scripts
    - Code-switching
    - Transliteration
    - Domain-specific tokens
    """
    
    def __init__(
        self,
        vocab_size: int = 50000,
        languages: List[str] = None,
        domain_tokens: List[str] = None
    ):
        self.vocab_size = vocab_size
        self.languages = languages or ["hi", "en", "bn", "ta", "te", "mr", "gu", "kn", "ml", "pa"]
        self.domain_tokens = domain_tokens or ["[GOV]", "[EDU]", "[MED]", "[TECH]", "[LEGAL]"]
        self.vocab = {}
        self.reverse_vocab = {}
        self.special_tokens = {
            "<pad>": 0,
            "<unk>": 1,
            "<bos>": 2,
            "<eos>": 3,
            "<mask>": 4
        }
        
        self._initialize_vocab()
        
    def _initialize_vocab(self):
        """Initialize vocabulary with special tokens and domain tokens"""
        # Add special tokens
        for token, idx in self.special_tokens.items():
            self.vocab[token] = idx
            self.reverse_vocab[idx] = token
            
        # Add domain tokens
        for token in self.domain_tokens:
            idx = len(self.vocab)
            self.vocab[token] = idx
            self.reverse_vocab[idx] = token
            
    def train(self, texts: List[str]):
        """Train tokenizer on corpus of texts"""
        # Simple word-level tokenization for demonstration
        # In practice, you'd use SentencePiece or BPE
        word_counts = {}
        
        for text in texts:
            words = text.lower().split()
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
                
        # Add most frequent words to vocabulary
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        
        for word, count in sorted_words:
            if len(self.vocab) >= self.vocab_size:
                break
            if word not in self.vocab:
                idx = len(self.vocab)
                self.vocab[word] = idx
                self.reverse_vocab[idx] = word
                
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs"""
        tokens = text.lower().split()
        token_ids = []
        
        for token in tokens:
            if token in self.vocab:
                token_ids.append(self.vocab[token])
            else:
                token_ids.append(self.special_tokens["<unk>"])
                
        return token_ids
        
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text"""
        tokens = []
        for token_id in token_ids:
            if token_id in self.reverse_vocab:
                tokens.append(self.reverse_vocab[token_id])
            else:
                tokens.append("<unk>")
        return " ".join(tokens)
        
    def get_vocab_size(self) -> int:
        """Get vocabulary size"""
        return len(self.vocab)
        
    def save_vocab(self, path: str):
        """Save vocabulary to disk"""
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        with open(save_path / "vocab.json", 'w', encoding='utf-8') as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)
            
        config = {
            "vocab_size": self.vocab_size,
            "languages": self.languages,
            "domain_tokens": self.domain_tokens
        }
        
        with open(save_path / "tokenizer_config.json", 'w') as f:
            json.dump(config, f, indent=2)
            
    @classmethod
    def load_vocab(cls, path: str):
        """Load vocabulary from disk"""
        load_path = Path(path)
        
        # Load config
        with open(load_path / "tokenizer_config.json", 'r') as f:
            config = json.load(f)
            
        # Create instance
        instance = cls(
            vocab_size=config["vocab_size"],
            languages=config["languages"],
            domain_tokens=config["domain_tokens"]
        )
        
        # Load vocabulary
        with open(load_path / "vocab.json", 'r', encoding='utf-8') as f:
            vocab = json.load(f)
            
        instance.vocab = vocab
        instance.reverse_vocab = {v: k for k, v in vocab.items()}
        
        return instance