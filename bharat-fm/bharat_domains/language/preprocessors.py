"""
Preprocessing utilities for Language AI use cases
"""

import re
import unicodedata
from typing import Dict, List, Optional, Tuple, Union
from transformers import AutoTokenizer
from bharat_data.tokenizers import BharatTokenizer


class IndicTokenizer(BharatTokenizer):
    """
    Enhanced tokenizer for Indian languages with special handling for:
    - Complex scripts (Devanagari, Bengali, Tamil, etc.)
    - Code-switching (Hinglish, Tanglish, etc.)
    - Transliteration variations
    - Regional dialects
    """
    
    def __init__(self, base_model: str = "bert-base-multilingual-cased"):
        self.base_tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.indic_scripts = {
            'devanagari': r'[\u0900-\u097F]',
            'bengali': r'[\u0980-\u09FF]',
            'tamil': r'[\u0B80-\u0BFF]',
            'telugu': r'[\u0C00-\u0C7F]',
            'kannada': r'[\u0C80-\u0CFF]',
            'malayalam': r'[\u0D00-\u0D7F]',
            'gujarati': r'[\u0A80-\u0AFF]',
            'oriya': r'[\u0B00-\u0B7F]',
            'punjabi': r'[\u0A00-\u0A7F]',
            'assamese': r'[\u0980-\u09FF]'
        }
        
        # Special tokens for code-switching
        self.special_tokens = {
            'lang_switch': '⟨lang⟩',
            'code_switch': '⟨code⟩',
            'translit': '⟨translit⟩'
        }
        
        # Add special tokens to tokenizer
        self.base_tokenizer.add_special_tokens({
            'additional_special_tokens': list(self.special_tokens.values())
        })
    
    def tokenize(self, text: str, language: str = None) -> List[str]:
        """
        Tokenize text with language-specific preprocessing
        
        Args:
            text: Input text
            language: Language code for language-specific processing
        """
        # Preprocess text
        processed_text = self.preprocess_text(text, language)
        
        # Tokenize
        tokens = self.base_tokenizer.tokenize(processed_text)
        
        # Add language markers if specified
        if language:
            tokens = [f"[{language.upper()}]"] + tokens
        
        return tokens
    
    def preprocess_text(self, text: str, language: str = None) -> str:
        """
        Preprocess text for Indian languages
        """
        # Normalize Unicode
        text = unicodedata.normalize('NFC', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Language-specific preprocessing
        if language == 'hi':
            text = self._preprocess_hindi(text)
        elif language == 'bn':
            text = self._preprocess_bengali(text)
        elif language == 'ta':
            text = self._preprocess_tamil(text)
        elif language == 'te':
            text = self._preprocess_telugu(text)
        
        return text
    
    def _preprocess_hindi(self, text: str) -> str:
        """
        Hindi-specific preprocessing
        """
        # Handle half-letters (halant)
        text = re.sub(r'(\u094d)([^\u094d])', r'\1 \2', text)
        
        # Handle vowel combinations
        text = re.sub(r'(\u093e|\u093f|\u0940|\u0941|\u0942|\u0943|\u0946|\u0947|\u0948|\u094a|\u094b|\u094c|\u0962|\u0963)', 
                     r' \1 ', text)
        
        return text
    
    def _preprocess_bengali(self, text: str) -> str:
        """
        Bengali-specific preprocessing
        """
        # Handle kar (vowel signs)
        text = re.sub(r'(\u09be|\u09bf|\u09c0|\u09c1|\u09c2|\u09c3|\u09c7|\u09c8|\u09cb|\u09cc|\u09d7)', 
                     r' \1 ', text)
        
        return text
    
    def _preprocess_tamil(self, text: str) -> str:
        """
        Tamil-specific preprocessing
        """
        # Handle vowel signs and special characters
        text = re.sub(r'(\u0bbe|\u0bbf|\u0bc0|\u0bc1|\u0bc2|\u0bc6|\u0bc7|\u0bc8|\u0bca|\u0bcb|\u0bcc)', 
                     r' \1 ', text)
        
        return text
    
    def _preprocess_telugu(self, text: str) -> str:
        """
        Telugu-specific preprocessing
        """
        # Handle vowel signs and gunintamulu
        text = re.sub(r'(\u0c3e|\u0c3f|\u0c40|\u0c41|\u0c42|\u0c46|\u0c47|\u0c48|\u0c4a|\u0c4b|\u0c4c)', 
                     r' \1 ', text)
        
        return text
    
    def detect_code_switching(self, text: str) -> List[Tuple[str, str]]:
        """
        Detect code-switching points in text
        
        Returns:
            List of (segment, language) tuples
        """
        segments = []
        current_segment = ""
        current_lang = None
        
        # Simple heuristic: check for script changes
        for char in text:
            char_lang = self._detect_char_language(char)
            
            if current_lang is None:
                current_lang = char_lang
                current_segment = char
            elif char_lang == current_lang:
                current_segment += char
            else:
                if current_segment.strip():
                    segments.append((current_segment.strip(), current_lang))
                current_lang = char_lang
                current_segment = char
        
        if current_segment.strip():
            segments.append((current_segment.strip(), current_lang))
        
        return segments
    
    def _detect_char_language(self, char: str) -> str:
        """
        Detect language of a single character based on Unicode script
        """
        if char.isascii():
            return 'en'
        
        for script, pattern in self.indic_scripts.items():
            if re.match(pattern, char):
                return script
        
        return 'unknown'
    
    def add_code_switching_markers(self, text: str) -> str:
        """
        Add markers for code-switching segments
        """
        segments = self.detect_code_switching(text)
        
        marked_text = ""
        for segment, lang in segments:
            if lang != 'en':
                marked_text += f"[{lang.upper()}] {segment} "
            else:
                marked_text += f"[EN] {segment} "
        
        return marked_text.strip()


class CodeSwitchingProcessor:
    """
    Processor for handling code-switching in Indian languages
    """
    
    def __init__(self):
        self.language_patterns = {
            'hi': r'[\u0900-\u097F]',
            'bn': r'[\u0980-\u09FF]',
            'ta': r'[\u0B80-\u0BFF]',
            'te': r'[\u0C00-\u0C7F]',
            'en': r'[a-zA-Z]'
        }
    
    def identify_switch_points(self, text: str) -> List[Dict]:
        """
        Identify code-switching points in text
        
        Returns:
            List of dictionaries with switch point information
        """
        switch_points = []
        current_lang = None
        start_pos = 0
        
        for i, char in enumerate(text):
            char_lang = self._detect_char_language(char)
            
            if current_lang is None:
                current_lang = char_lang
                start_pos = i
            elif char_lang != current_lang and char_lang != 'unknown':
                if i - start_pos > 0:  # Only consider segments with actual content
                    switch_points.append({
                        'start': start_pos,
                        'end': i,
                        'language': current_lang,
                        'text': text[start_pos:i]
                    })
                current_lang = char_lang
                start_pos = i
        
        # Add final segment
        if start_pos < len(text):
            switch_points.append({
                'start': start_pos,
                'end': len(text),
                'language': current_lang,
                'text': text[start_pos:]
            })
        
        return switch_points
    
    def normalize_code_switching(self, text: str) -> str:
        """
        Normalize code-switched text for consistent processing
        """
        switch_points = self.identify_switch_points(text)
        
        normalized = ""
        for segment in switch_points:
            if segment['language'] != 'en':
                normalized += f"[{segment['language'].upper()}] {segment['text']} "
            else:
                normalized += f"[EN] {segment['text']} "
        
        return normalized.strip()
    
    def extract_code_switching_patterns(self, texts: List[str]) -> Dict[str, List[str]]:
        """
        Extract common code-switching patterns from a corpus
        
        Returns:
            Dictionary mapping language pairs to common patterns
        """
        patterns = {}
        
        for text in texts:
            switch_points = self.identify_switch_points(text)
            
            for i in range(len(switch_points) - 1):
                current_seg = switch_points[i]
                next_seg = switch_points[i + 1]
                
                lang_pair = f"{current_seg['language']}-{next_seg['language']}"
                
                if lang_pair not in patterns:
                    patterns[lang_pair] = []
                
                pattern = {
                    'before': current_seg['text'][-20:] if len(current_seg['text']) > 20 else current_seg['text'],
                    'after': next_seg['text'][:20] if len(next_seg['text']) > 20 else next_seg['text'],
                    'context': text[max(0, current_seg['start']-10):min(len(text), next_seg['end']+10)]
                }
                
                patterns[lang_pair].append(pattern)
        
        return patterns
    
    def _detect_char_language(self, char: str) -> str:
        """
        Detect language of a single character
        """
        if char.isascii() and char.isalpha():
            return 'en'
        
        for lang, pattern in self.language_patterns.items():
            if re.match(pattern, char):
                return lang
        
        return 'unknown'


class TransliterationNormalizer:
    """
    Normalizer for handling transliteration variations in Indian languages
    """
    
    def __init__(self):
        # Common transliteration mappings
        self.translit_maps = {
            'hi': {
                'namaste': ['नमस्ते', 'नमस्ते', 'नमस्ते'],
                'dhanyawad': ['धन्यवाद', 'धन्यवाद'],
                'kya': ['क्या', 'क्या'],
                'hai': ['है', 'है'],
                'nahin': ['नहीं', 'नहीं']
            },
            'ta': {
                'vanakkam': ['வணக்கம்', 'வணக்கம்'],
                'nandri': ['நன்றி', 'நன்றி'],
                'enna': ['என்ன', 'என்ன'],
                'illai': ['இல்லை', 'இல்லை']
            }
        }
    
    def normalize_transliteration(self, text: str, target_lang: str) -> str:
        """
        Normalize transliterated text to proper script
        """
        words = text.lower().split()
        normalized_words = []
        
        for word in words:
            normalized = self._normalize_word(word, target_lang)
            normalized_words.append(normalized)
        
        return ' '.join(normalized_words)
    
    def _normalize_word(self, word: str, target_lang: str) -> str:
        """
        Normalize a single word
        """
        if target_lang in self.translit_maps:
            for translit, variants in self.translit_maps[target_lang].items():
                if word.startswith(translit) or translit.startswith(word):
                    # Return the first proper script variant
                    return variants[0]
        
        return word
    
    def detect_transliteration(self, text: str) -> Dict[str, float]:
        """
        Detect if text contains transliterated content and score confidence
        
        Returns:
            Dictionary mapping languages to confidence scores
        """
        scores = {}
        words = text.lower().split()
        
        for lang, translit_map in self.translit_maps.items():
            score = 0
            for word in words:
                for translit in translit_map.keys():
                    if word.startswith(translit) or translit.startswith(word):
                        score += 1
                        break
            
            scores[lang] = score / len(words) if words else 0
        
        return scores