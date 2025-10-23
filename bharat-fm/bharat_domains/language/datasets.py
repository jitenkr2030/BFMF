"""
Domain-specific datasets for Language AI use cases
"""

from typing import Dict, List, Optional, Union, Iterator
import json
import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
from bharat_data.datasets import BharatDataset


class IndicCorp(BharatDataset):
    """
    IndicCorp: Large-scale corpus for Indian languages
    
    Contains text data from:
    - Wikipedia in 11 Indian languages
    - News articles from regional sources
    - Government documents
    - Social media content
    - Literary works
    """
    
    def __init__(self, languages: List[str] = None, split: str = "train"):
        """
        Initialize IndicCorp dataset
        
        Args:
            languages: List of language codes to include
            split: Dataset split (train, validation, test)
        """
        self.languages = languages or ['hi', 'bn', 'ta', 'te', 'mr', 'gu', 'kn', 'ml', 'pa', 'or', 'as']
        self.split = split
        self.data = self._load_data()
    
    def _load_data(self) -> DatasetDict:
        """
        Load IndicCorp data
        """
        # This would load from actual IndicCorp source
        # For now, we'll create a placeholder structure
        data_dict = {}
        
        for lang in self.languages:
            # Placeholder data structure
            data_dict[lang] = Dataset.from_dict({
                'text': [f"Sample text in {lang}"] * 1000,
                'language': [lang] * 1000,
                'source': ['wikipedia'] * 1000,
                'word_count': [10] * 1000
            })
        
        return DatasetDict(data_dict)
    
    def get_language_split(self, language: str) -> Dataset:
        """
        Get dataset for specific language
        """
        return self.data[language]
    
    def get_code_switching_data(self) -> Dataset:
        """
        Get code-switching (Hinglish, Tanglish, etc.) data
        """
        code_switching_examples = []
        
        for lang in ['hi', 'ta', 'bn']:  # Languages with common code-switching
            lang_data = self.data[lang]
            for i in range(min(100, len(lang_data))):
                # Create code-switching examples
                original = lang_data[i]['text']
                code_switched = self._create_code_switching_example(original, lang)
                code_switching_examples.append({
                    'original': original,
                    'code_switched': code_switched,
                    'primary_language': lang,
                    'secondary_language': 'en'
                })
        
        return Dataset.from_list(code_switching_examples)
    
    def _create_code_switching_example(self, text: str, lang: str) -> str:
        """
        Create code-switching example by mixing languages
        """
        # Simple implementation - replace some words with English
        words = text.split()
        for i in range(len(words)):
            if i % 5 == 0:  # Replace every 5th word
                words[i] = f"[EN:{words[i]}]"
        return ' '.join(words)


class IndianWikipedia(BharatDataset):
    """
    Indian Wikipedia dataset for knowledge and factual content
    """
    
    def __init__(self, languages: List[str] = None):
        self.languages = languages or ['hi', 'bn', 'ta', 'te', 'mr', 'gu', 'kn', 'ml', 'pa', 'or', 'as']
        self.data = self._load_wikipedia_data()
    
    def _load_wikipedia_data(self) -> DatasetDict:
        """
        Load Wikipedia dumps for Indian languages
        """
        data_dict = {}
        
        for lang in self.languages:
            # Load from Hugging Face datasets or local Wikipedia dumps
            try:
                wiki_data = load_dataset(f"wiki_{lang}", split="train")
                data_dict[lang] = wiki_data
            except:
                # Fallback to synthetic data
                data_dict[lang] = Dataset.from_dict({
                    'title': [f"Article in {lang}"] * 500,
                    'text': [f"Wikipedia article content in {lang}"] * 500,
                    'language': [lang] * 500,
                    'category': ['general'] * 500
                })
        
        return DatasetDict(data_dict)
    
    def get_knowledge_base(self, domain: str = None) -> Dataset:
        """
        Get knowledge base for specific domain
        """
        all_data = []
        
        for lang in self.languages:
            lang_data = self.data[lang]
            for item in lang_data:
                if domain is None or domain.lower() in item.get('category', '').lower():
                    all_data.append({
                        'language': lang,
                        'title': item['title'],
                        'text': item['text'],
                        'domain': item.get('category', 'general')
                    })
        
        return Dataset.from_list(all_data)


class LanguageBenchmarks(BharatDataset):
    """
    Evaluation benchmarks for Indian language models
    """
    
    def __init__(self):
        self.benchmarks = {
            'translation': self._load_translation_benchmarks(),
            'summarization': self._load_summarization_benchmarks(),
            'question_answering': self._load_qa_benchmarks(),
            'sentiment': self._load_sentiment_benchmarks()
        }
    
    def _load_translation_benchmarks(self) -> Dict[str, Dataset]:
        """
        Load translation evaluation datasets
        """
        benchmarks = {}
        
        # Example: Hindi-English translation
        benchmarks['hi-en'] = Dataset.from_dict({
            'source': ["नमस्ते दुनिया", "यह एक परीक्षण है"],
            'target': ["Hello world", "This is a test"],
            'source_lang': ['hi', 'hi'],
            'target_lang': ['en', 'en']
        })
        
        # Add more language pairs
        language_pairs = [
            ('bn', 'en'), ('ta', 'en'), ('te', 'en'), 
            ('mr', 'en'), ('gu', 'en'), ('kn', 'en')
        ]
        
        for src_lang, tgt_lang in language_pairs:
            benchmarks[f'{src_lang}-{tgt_lang}'] = Dataset.from_dict({
                'source': [f"Sample text in {src_lang}"] * 50,
                'target': [f"Sample text in {tgt_lang}"] * 50,
                'source_lang': [src_lang] * 50,
                'target_lang': [tgt_lang] * 50
            })
        
        return benchmarks
    
    def _load_summarization_benchmarks(self) -> Dict[str, Dataset]:
        """
        Load summarization evaluation datasets
        """
        benchmarks = {}
        
        for lang in ['hi', 'bn', 'ta', 'te']:
            benchmarks[lang] = Dataset.from_dict({
                'document': [f"Long document in {lang} " * 100] * 30,
                'summary': [f"Short summary in {lang}"] * 30,
                'language': [lang] * 30
            })
        
        return benchmarks
    
    def _load_qa_benchmarks(self) -> Dict[str, Dataset]:
        """
        Load question answering evaluation datasets
        """
        benchmarks = {}
        
        for lang in ['hi', 'bn', 'ta']:
            benchmarks[lang] = Dataset.from_dict({
                'question': [f"Question in {lang}?"] * 40,
                'context': [f"Context for question in {lang} " * 50] * 40,
                'answer': [f"Answer in {lang}"] * 40,
                'language': [lang] * 40
            })
        
        return benchmarks
    
    def _load_sentiment_benchmarks(self) -> Dict[str, Dataset]:
        """
        Load sentiment analysis evaluation datasets
        """
        benchmarks = {}
        
        for lang in ['hi', 'bn', 'ta']:
            benchmarks[lang] = Dataset.from_dict({
                'text': [f"Positive text in {lang}"] * 25 + [f"Negative text in {lang}"] * 25,
                'label': [1] * 25 + [0] * 25,
                'language': [lang] * 50
            })
        
        return benchmarks
    
    def get_benchmark(self, task: str, language: str = None) -> Dataset:
        """
        Get benchmark dataset for specific task and language
        """
        if task not in self.benchmarks:
            raise ValueError(f"Unknown benchmark task: {task}")
        
        task_benchmarks = self.benchmarks[task]
        
        if language is None:
            # Return combined dataset for all languages
            all_data = []
            for lang, dataset in task_benchmarks.items():
                for item in dataset:
                    all_data.append(item)
            return Dataset.from_list(all_data)
        
        # Return dataset for specific language
        if language in task_benchmarks:
            return task_benchmarks[language]
        else:
            raise ValueError(f"No benchmark available for {task} in {language}")


class GovernmentSchemesDataset(BharatDataset):
    """
    Dataset for government schemes and policies in Indian languages
    """
    
    def __init__(self):
        self.data = self._load_schemes_data()
    
    def _load_schemes_data(self) -> Dataset:
        """
        Load government schemes data
        """
        schemes_data = {
            'scheme_name': [
                'प्रधानमंत्री आवास योजना',
                'प्रधानमंत्री किसान सम्मान निधि',
                'उज्ज्वला योजना',
                'स्वच्छ भारत अभियान',
                'डिजिटल इंडिया'
            ],
            'description': [
                'हर गरीब को अपना घर देने की योजना',
                'किसानों को वित्तीय सहायता प्रदान करना',
                'गरीब महिलाओं को मुफ्त गैस कनेक्शन',
                'स्वच्छता और स्वच्छ जीवन को बढ़ावा देना',
                'डिजिटल भारत बनाने की पहल'
            ],
            'eligibility': [
                'गरीब परिवार, झुग्गी निवासी',
                'छोटे और सीमांत किसान',
                'बीपीएल परिवार की महिलाएं',
                'सभी नागरिक',
                'सभी नागरिक'
            ],
            'benefits': [
                'घर निर्माण के लिए वित्तीय सहायता',
                'प्रति वर्ष ₹6000 की सहायता',
                'मुफ्त LPG कनेक्शन',
                'स्वच्छ शौचालय और स्वच्छता',
                'डिजिटल सेवाएं और सुविधाएं'
            ],
            'language': ['hi'] * 5,
            'category': ['housing', 'agriculture', 'energy', 'sanitation', 'digital']
        }
        
        return Dataset.from_dict(schemes_data)
    
    def get_scheme_chatbot_data(self) -> Dataset:
        """
        Get data formatted for chatbot training
        """
        chat_data = []
        
        for i in range(len(self.data)):
            scheme = self.data[i]
            
            # Create question-answer pairs
            chat_data.extend([
                {
                    'question': f"{scheme['scheme_name']} क्या है?",
                    'answer': scheme['description'],
                    'scheme': scheme['scheme_name'],
                    'language': 'hi'
                },
                {
                    'question': f"{scheme['scheme_name']} के लिए पात्रता क्या है?",
                    'answer': scheme['eligibility'],
                    'scheme': scheme['scheme_name'],
                    'language': 'hi'
                },
                {
                    'question': f"{scheme['scheme_name']} के क्या लाभ हैं?",
                    'answer': scheme['benefits'],
                    'scheme': scheme['scheme_name'],
                    'language': 'hi'
                }
            ])
        
        return Dataset.from_list(chat_data)