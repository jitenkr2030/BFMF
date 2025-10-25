"""
Bharat Language AI Module

Support for National & Regional Language AI use cases including:
- Multilingual models for 22+ scheduled Indian languages
- AI chatbots for government schemes in local languages
- Translation engines for official documents
- Voice-to-text transcription for rural services
- AI news summarization across Indian states
"""

from .models import BharatLang, BharatSpeech
from .datasets import IndicCorp, IndianWikipedia, LanguageBenchmarks
from .preprocessors import IndicTokenizer, CodeSwitchingProcessor
from .evaluators import MultilingualEvaluator, TranslationEvaluator

__all__ = [
    'BharatLang',
    'BharatSpeech', 
    'IndicCorp',
    'IndianWikipedia',
    'LanguageBenchmarks',
    'IndicTokenizer',
    'CodeSwitchingProcessor',
    'MultilingualEvaluator',
    'TranslationEvaluator'
]