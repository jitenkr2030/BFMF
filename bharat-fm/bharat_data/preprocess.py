"""
Data preprocessing utilities for Indian languages
"""

import re
import unicodedata
from typing import List, Dict, Optional, Union
import pandas as pd


class DataCleaner:
    """
    Data cleaning utilities specifically for Indian language texts
    """
    
    def __init__(self, languages: List[str] = None):
        self.languages = languages or ["hi", "en", "bn", "ta", "te", "mr", "gu", "kn", "ml", "pa"]
        
        # Common patterns for Indian languages
        self.patterns = {
            "whitespace": re.compile(r'\s+'),
            "urls": re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'),
            "emails": re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            "html_tags": re.compile(r'<[^>]+>'),
            "special_chars": re.compile(r'[^\w\s\u0900-\u097F\u0B80-\u0BFF\u0C00-\u0C7F\u0C80-\u0CFF\u0D00-\u0D7F\u0A80-\u0AFF\u0B00-\u0B7F\u0AE0-\u0AE3\u0AE5\u0AE6\u0AE8-\u0AED\u0AF0-\u0AF1\u0B01-\u0B03\u0B05-\u0B0C\u0B0F-\u0B10\u0B13-\u0B28\u0B2A-\u0B30\u0B32-\u0B33\u0B35-\u0B39\u0B3C-\u0B44\u0B47-\u0B48\u0B4B-\u0B4D\u0B56-\u0B57\u0B5C-\u0B5D\u0B5F-\u0B61\u0B66-\u0B6F\u0B71-\u0B71\u0B83-\u0B83\u0B85-\u0B8A\u0B8E-\u0B90\u0B92-\u0B95\u0B99-\u0B9A\u0B9C\u0B9E-\u0B9F\u0BA3-\u0BA4\u0BA8-\u0BAA\u0BAE-\u0BB9\u0BBE-\u0BBF\u0BC1-\u0BC2\u0BC6-\u0BC8\u0BCA-\u0BCD\u0BD0\u0BD7\u0BE6-\u0BEF\u0C01-\u0C03\u0C05-\u0C0C\u0C0E-\u0C10\u0C12-\u0C28\u0C2A-\u0C33\u0C35-\u0C39\u0C3D-\u0C44\u0C46-\u0C48\u0C4A-\u0C4D\u0C55-\u0C56\u0C58-\u0C5A\u0C60-\u0C61\u0C66-\u0C6F\u0C82-\u0C83\u0C85-\u0C8C\u0C8E-\u0C90\u0C92-\u0CA8\u0CAA-\u0CB3\u0CB5-\u0CB9\u0CBD-\u0CC4\u0CC6-\u0CC8\u0CCA-\u0CCD\u0CD5-\u0CD6\u0CDE\u0CE0-\u0CE1\u0CE6-\u0CEF\u0D02-\u0D03\u0D05-\u0D0C\u0D0E-\u0D10\u0D12-\u0D28\u0D2A-\u0D39\u0D3D-\u0D44\u0D46-\u0D48\u0D4A-\u0D4D\u0D57\u0D60-\u0D61\u0D66-\u0D6F\u0D82-\u0D83\u0D85-\u0D8E\u0D91-\u0D96\u0D9A-\u0DA8\u0DAA-\u0DB1\u0DB3-\u0DBB\u0DBD\u0DC0-\u0DC6\u0DCF-\u0DD4\u0DD6\u0DD8-\u0DDF\u0DF2-\u0DF3\u0E01-\u0E3A\u0E40-\u0E4E\u0E50-\u0E59\u0E81-\u0E82\u0E84\u0E87-\u0E88\u0E8A\u0E8D\u0E94-\u0E97\u0E99-\u0E9F\u0EA1-\u0EA3\u0EA5\u0EA7\u0EAA-\u0EAB\u0EAD-\u0EB0\u0EB2-\u0EB3\u0EBD\u0EC0-\u0EC4\u0EC6\u0EC8-\u0ECD\u0ED0-\u0ED9\u0EDC-\u0EDD\u0F00\u0F18-\u0F19\u0F20-\u0F29\u0F35\u0F37\u0F39\u0F3E-\u0F47\u0F49-\u0F6C\u0F71-\u0F84\u0F86-\u0F97\u0F99-\u0FBC\u0FC6]'),
        }
        
    def clean_text(self, text: str) -> str:
        """Clean text by removing unwanted patterns"""
        if not text:
            return ""
            
        # Remove URLs
        text = self.patterns["urls"].sub('', text)
        
        # Remove email addresses
        text = self.patterns["emails"].sub('', text)
        
        # Remove HTML tags
        text = self.patterns["html_tags"].sub('', text)
        
        # Normalize whitespace
        text = self.patterns["whitespace"].sub(' ', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
        
    def remove_special_characters(self, text: str, keep_punctuation: bool = True) -> str:
        """Remove special characters while preserving Indian scripts"""
        if keep_punctuation:
            # Keep basic punctuation
            pattern = re.compile(r'[^\w\s\u0900-\u097F\u0B80-\u0BFF\u0C00-\u0C7F\u0C80-\u0CFF\u0D00-\u0D7F\u0A80-\u0AFF\u0B00-\u0B7F\u0AE0-\u0AE3\u0AE5\u0AE6\u0AE8-\u0AED\u0AF0-\u0AF1\u0B01-\u0B03\u0B05-\u0B0C\u0B0F-\u0B10\u0B13-\u0B28\u0B2A-\u0B30\u0B32-\u0B33\u0B35-\u0B39\u0B3C-\u0B44\u0B47-\u0B48\u0B4B-\u0B4D\u0B56-\u0B57\u0B5C-\u0B5D\u0B5F-\u0B61\u0B66-\u0B6F\u0B71-\u0B71\u0B83-\u0B83\u0B85-\u0B8A\u0B8E-\u0B90\u0B92-\u0B95\u0B99-\u0B9A\u0B9C\u0B9E-\u0B9F\u0BA3-\u0BA4\u0BA8-\u0BAA\u0BAE-\u0BB9\u0BBE-\u0BBF\u0BC1-\u0BC2\u0BC6-\u0BC8\u0BCA-\u0BCD\u0BD0\u0BD7\u0BE6-\u0BEF\u0C01-\u0C03\u0C05-\u0C0C\u0C0E-\u0C10\u0C12-\u0C28\u0C2A-\u0C33\u0C35-\u0C39\u0C3D-\u0C44\u0C46-\u0C48\u0C4A-\u0C4D\u0C55-\u0C56\u0C58-\u0C5A\u0C60-\u0C61\u0C66-\u0C6F\u0C82-\u0C83\u0C85-\u0C8C\u0C8E-\u0C90\u0C92-\u0CA8\u0CAA-\u0CB3\u0CB5-\u0CB9\u0CBD-\u0CC4\u0CC6-\u0CC8\u0CCA-\u0CCD\u0CD5-\u0CD6\u0CDE\u0CE0-\u0CE1\u0CE6-\u0CEF\u0D02-\u0D03\u0D05-\u0D0C\u0D0E-\u0D10\u0D12-\u0D28\u0D2A-\u0D39\u0D3D-\u0D44\u0D46-\u0D48\u0D4A-\u0D4D\u0D57\u0D60-\u0D61\u0D66-\u0D6F\u0D82-\u0D83\u0D85-\u0D8E\u0D91-\u0D96\u0D9A-\u0DA8\u0DAA-\u0DB1\u0DB3-\u0DBB\u0DBD\u0DC0-\u0DC6\u0DCF-\u0DD4\u0DD6\u0DD8-\u0DDF\u0DF2-\u0DF3\u0E01-\u0E3A\u0E40-\u0E4E\u0E50-\u0E59\u0E81-\u0E82\u0E84\u0E87-\u0E88\u0E8A\u0E8D\u0E94-\u0E97\u0E99-\u0E9F\u0EA1-\u0EA3\u0EA5\u0EA7\u0EAA-\u0EAB\u0EAD-\u0EB0\u0EB2-\u0EB3\u0EBD\u0EC0-\u0EC4\u0EC6\u0EC8-\u0ECD\u0ED0-\u0ED9\u0EDC-\u0EDD\u0F00\u0F18-\u0F19\u0F20-\u0F29\u0F35\u0F37\u0F39\u0F3E-\u0F47\u0F49-\u0F6C\u0F71-\u0F84\u0F86-\u0F97\u0F99-\u0FBC\u0FC6.,!?;:()\[\]{}"\'`]')
        else:
            pattern = self.patterns["special_chars"]
            
        return pattern.sub('', text)
        
    def normalize_unicode(self, text: str) -> str:
        """Normalize Unicode text"""
        return unicodedata.normalize('NFC', text)
        
    def clean_dataset(self, df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
        """Clean entire dataset"""
        df_clean = df.copy()
        df_clean[text_column] = df_clean[text_column].apply(self.clean_text)
        df_clean[text_column] = df_clean[text_column].apply(self.normalize_unicode)
        df_clean[text_column] = df_clean[text_column].apply(lambda x: self.remove_special_characters(x, keep_punctuation=True))
        
        # Remove empty rows
        df_clean = df_clean[df_clean[text_column].str.len() > 0]
        
        return df_clean


class TextNormalizer:
    """
    Text normalization utilities for Indian languages
    """
    
    def __init__(self):
        # Common normalization mappings for Indian languages
        self.normalization_maps = {
            # Hindi normalization
            "hi": {
                "।": ".",  # Hindi danda to period
                "॥": "..",  # Double danda
                "ँ": "ं",   # Chandrabindu to Anusvara
            },
            # Bengali normalization
            "bn": {
                "।": ".",
                "॥": "..",
                "ঁ": "ং",
            },
            # Tamil normalization
            "ta": {
                "।": ".",
                "॥": "..",
            }
        }
        
    def normalize_text(self, text: str, language: str = None) -> str:
        """Normalize text according to language-specific rules"""
        if language and language in self.normalization_maps:
            for old_char, new_char in self.normalization_maps[language].items():
                text = text.replace(old_char, new_char)
                
        # Common normalizations
        text = text.replace("''", '"')
        text = text.replace("``", '"')
        text = text.replace("´", "'")
        text = text.replace("`", "'")
        
        return text
        
    def normalize_numbers(self, text: str) -> str:
        """Convert Indian numerals to Western numerals"""
        # Devanagari numerals
        devanagari_map = {
            '०': '0', '१': '1', '२': '2', '३': '3', '४': '4',
            '५': '5', '६': '6', '७': '7', '८': '8', '९': '9'
        }
        
        # Bengali numerals
        bengali_map = {
            '০': '0', '১': '1', '২': '2', '৩': '3', '৪': '4',
            '৫': '5', '৬': '6', '৭': '7', '৮': '8', '৯': '9'
        }
        
        # Apply all mappings
        for old, new in {**devanagari_map, **bengali_map}.items():
            text = text.replace(old, new)
            
        return text
        
    def normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace characters"""
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        # Remove leading/trailing whitespace
        text = text.strip()
        return text
        
    def normalize_dataset(self, df: pd.DataFrame, text_column: str = 'text', language_column: str = 'language') -> pd.DataFrame:
        """Normalize entire dataset"""
        df_norm = df.copy()
        
        if language_column in df.columns:
            # Language-specific normalization
            for idx, row in df.iterrows():
                language = row[language_column]
                text = row[text_column]
                df_norm.at[idx, text_column] = self.normalize_text(text, language)
        else:
            # Generic normalization
            df_norm[text_column] = df_norm[text_column].apply(self.normalize_text)
            
        df_norm[text_column] = df_norm[text_column].apply(self.normalize_numbers)
        df_norm[text_column] = df_norm[text_column].apply(self.normalize_whitespace)
        
        return df_norm