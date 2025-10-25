"""
Dataset handling for Indian languages and multilingual corpora
"""

import json
import pandas as pd
from typing import Dict, List, Optional, Union
from pathlib import Path
import datasets
from transformers import AutoTokenizer


class IndicDataset:
    """
    Dataset class for Indian language corpora with support for
    multiple languages and preprocessing pipelines.
    """
    
    def __init__(
        self,
        name: str,
        languages: List[str] = None,
        data_path: Optional[str] = None
    ):
        self.name = name
        self.languages = languages or ["hi", "en", "bn", "ta", "te", "mr", "gu", "kn", "ml", "pa"]
        self.data_path = Path(data_path) if data_path else Path("./datasets")
        self.dataset = None
        
    def load_from_huggingface(self, dataset_name: str, split: str = "train"):
        """Load dataset from Hugging Face datasets hub"""
        self.dataset = datasets.load_dataset(dataset_name, split=split)
        return self
        
    def load_from_json(self, file_path: str):
        """Load dataset from JSON file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.dataset = datasets.Dataset.from_list(data)
        return self
        
    def load_from_csv(self, file_path: str, text_column: str = "text"):
        """Load dataset from CSV file"""
        df = pd.read_csv(file_path)
        self.dataset = datasets.Dataset.from_pandas(df)
        return self
        
    def filter_by_language(self, language: str):
        """Filter dataset by specific language"""
        if self.dataset is None:
            raise ValueError("Dataset not loaded")
        
        # Assuming dataset has a 'language' column
        filtered = self.dataset.filter(lambda x: x['language'] == language)
        return IndicDataset(f"{self.name}_{language}", [language], data_path=self.data_path).load_from_dict(filtered)
        
    def get_stats(self) -> Dict:
        """Get dataset statistics"""
        if self.dataset is None:
            return {}
            
        stats = {
            "total_samples": len(self.dataset),
            "languages": self.languages,
            "columns": list(self.dataset.column_names),
        }
        
        # Add language distribution if available
        if 'language' in self.dataset.column_names:
            lang_dist = {}
            for lang in self.languages:
                count = len(self.dataset.filter(lambda x: x['language'] == lang))
                lang_dist[lang] = count
            stats["language_distribution"] = lang_dist
            
        return stats


class MultilingualCorpus:
    """
    Corpus class for handling multilingual text collections
    with support for parallel and comparable corpora.
    """
    
    def __init__(self, name: str):
        self.name = name
        self.corpora = {}
        
    def add_corpus(self, language: str, texts: List[str]):
        """Add corpus for a specific language"""
        self.corpora[language] = texts
        
    def add_parallel_corpus(self, source_lang: str, target_lang: str, pairs: List[tuple]):
        """Add parallel corpus (translation pairs)"""
        key = f"{source_lang}-{target_lang}"
        self.corpora[key] = {
            "source_lang": source_lang,
            "target_lang": target_lang,
            "pairs": pairs
        }
        
    def get_corpus(self, language: str) -> List[str]:
        """Get corpus for a specific language"""
        return self.corpora.get(language, [])
        
    def get_parallel_corpus(self, source_lang: str, target_lang: str) -> List[tuple]:
        """Get parallel corpus for language pair"""
        key = f"{source_lang}-{target_lang}"
        return self.corpora.get(key, {}).get("pairs", [])
        
    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages"""
        languages = set()
        for key in self.corpora.keys():
            if "-" in key:  # Parallel corpus
                source, target = key.split("-")
                languages.add(source)
                languages.add(target)
            else:  # Monolingual corpus
                languages.add(key)
        return sorted(list(languages))
        
    def save_to_disk(self, path: str):
        """Save corpus to disk"""
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        with open(save_path / f"{self.name}.json", 'w', encoding='utf-8') as f:
            json.dump(self.corpora, f, ensure_ascii=False, indent=2)
            
    @classmethod
    def load_from_disk(cls, name: str, path: str):
        """Load corpus from disk"""
        load_path = Path(path) / f"{name}.json"
        with open(load_path, 'r', encoding='utf-8') as f:
            corpora = json.load(f)
            
        instance = cls(name)
        instance.corpora = corpora
        return instance