"""
Bharat Data Module
==================

Module for preparing multilingual datasets, tokenizers, and cleaning pipelines
specifically designed for Indian languages and regional diversity.
"""

from .datasets import IndicDataset, MultilingualCorpus
from .tokenizers import IndicTokenizer, BharatTokenizer
from .preprocess import DataCleaner, TextNormalizer

__version__ = "0.1.0"
__all__ = [
    "IndicDataset",
    "MultilingualCorpus", 
    "IndicTokenizer",
    "BharatTokenizer",
    "DataCleaner",
    "TextNormalizer"
]