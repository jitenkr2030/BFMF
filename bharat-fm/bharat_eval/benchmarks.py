"""
Benchmark registry and utilities for BharatFM evaluation
"""

import json
from typing import Dict, List, Optional, Any, Type
from abc import ABC, abstractmethod
from pathlib import Path
import logging
import numpy as np


class Benchmark(ABC):
    """Abstract base class for benchmarks"""
    
    def __init__(self, name: str, description: str, languages: List[str] = None):
        self.name = name
        self.description = description
        self.languages = languages or ["hi", "en", "bn", "ta", "te", "mr", "gu", "kn", "ml", "pa"]
        
    @abstractmethod
    def load_data(self, language: str = None) -> List[Dict[str, Any]]:
        """Load benchmark data"""
        pass
        
    @abstractmethod
    def evaluate(self, model, tokenizer, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate model on benchmark data"""
        pass
        
    @abstractmethod
    def compute_metrics(self, predictions: List[Any], references: List[Any]) -> Dict[str, float]:
        """Compute evaluation metrics"""
        pass


class PerplexityBenchmark(Benchmark):
    """Perplexity evaluation benchmark"""
    
    def __init__(self):
        super().__init__(
            name="perplexity",
            description="Evaluate model perplexity on test corpus"
        )
        
    def load_data(self, language: str = None) -> List[str]:
        """Load text data for perplexity evaluation"""
        # In practice, load from actual datasets
        return [
            "भारत एक महान देश है जिसकी समृद्ध संस्कृति और इतिहास है।",
            "India is a great country with rich culture and history.",
            "ভারত একটি মহান দেশ যার সমৃদ্ধ সংস্কৃতি এবং ইতিহাস রয়েছে।",
        ]
        
    def evaluate(self, model, tokenizer, data: List[str]) -> Dict[str, Any]:
        """Evaluate perplexity"""
        import torch
        
        perplexities = []
        
        for text in data:
            inputs = tokenizer.tokenize(text)
            input_ids = inputs["input_ids"]
            
            with torch.no_grad():
                outputs = model(input_ids=input_ids, labels=input_ids)
                loss = outputs["loss"]
                perplexity = torch.exp(loss).item()
                perplexities.append(perplexity)
                
        return {
            "perplexities": perplexities,
            "average_perplexity": np.mean(perplexities),
            "std_perplexity": np.std(perplexities)
        }
        
    def compute_metrics(self, predictions: List[float], references: List[float]) -> Dict[str, float]:
        """Compute perplexity metrics"""
        return {
            "mean_perplexity": np.mean(predictions),
            "std_perplexity": np.std(predictions),
            "min_perplexity": np.min(predictions),
            "max_perplexity": np.max(predictions)
        }


class GenerationQualityBenchmark(Benchmark):
    """Text generation quality benchmark"""
    
    def __init__(self):
        super().__init__(
            name="generation_quality",
            description="Evaluate text generation quality using BLEU, ROUGE, and BERTScore"
        )
        
    def load_data(self, language: str = None) -> List[Dict[str, str]]:
        """Load generation test data"""
        return [
            {"prompt": "भारत की राजधानी क्या है?", "reference": "भारत की राजधानी नई दिल्ली है।"},
            {"prompt": "What is the capital of India?", "reference": "The capital of India is New Delhi."},
            {"prompt": "ভারতের রাজধানী কি?", "reference": "ভারতের রাজধানী নতুন দিল্লী।"},
        ]
        
    def evaluate(self, model, tokenizer, data: List[Dict[str, str]]) -> Dict[str, Any]:
        """Evaluate generation quality"""
        generated_texts = []
        
        for item in data:
            prompt = item["prompt"]
            generated = self.generate_text(model, tokenizer, prompt)
            generated_texts.append(generated)
            
        references = [item["reference"] for item in data]
        
        return {
            "generated_texts": generated_texts,
            "reference_texts": references
        }
        
    def generate_text(self, model, tokenizer, prompt: str) -> str:
        """Generate text from prompt"""
        inputs = tokenizer.tokenize(prompt)
        input_ids = inputs["input_ids"]
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                max_new_tokens=100,
                num_beams=1,
                do_sample=False,
                pad_token_id=tokenizer.tokenizer.pad_token_id if hasattr(tokenizer, 'tokenizer') else 0,
                eos_token_id=tokenizer.tokenizer.eos_token_id if hasattr(tokenizer, 'tokenizer') else 2,
            )
            
        generated_ids = outputs[0][input_ids.shape[1]:]
        generated_text = tokenizer.decode(generated_ids)
        
        return generated_text
        
    def compute_metrics(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Compute generation quality metrics"""
        bleu_scores = self.compute_bleu(predictions, references)
        rouge_scores = self.compute_rouge(predictions, references)
        bert_scores = self.compute_bertscore(predictions, references)
        
        return {
            "bleu_score": np.mean(bleu_scores),
            "rouge_1": np.mean([score["rouge1"] for score in rouge_scores]),
            "rouge_2": np.mean([score["rouge2"] for score in rouge_scores]),
            "rouge_l": np.mean([score["rougeL"] for score in rouge_scores]),
            "bertscore_f1": np.mean(bert_scores["f1"]),
            "bertscore_precision": np.mean(bert_scores["precision"]),
            "bertscore_recall": np.mean(bert_scores["recall"]),
        }
        
    def compute_bleu(self, generated: List[str], references: List[str]) -> List[float]:
        """Compute BLEU scores (simplified implementation)"""
        bleu_scores = []
        
        for gen, ref in zip(generated, references):
            gen_tokens = gen.lower().split()
            ref_tokens = ref.lower().split()
            
            if len(gen_tokens) == 0:
                bleu_scores.append(0.0)
                continue
                
            matches = len(set(gen_tokens) & set(ref_tokens))
            precision = matches / len(gen_tokens)
            
            bp = 1.0 if len(gen_tokens) >= len(ref_tokens) else np.exp(1 - len(ref_tokens) / len(gen_tokens))
            
            bleu = bp * precision
            bleu_scores.append(bleu)
            
        return bleu_scores
        
    def compute_rouge(self, generated: List[str], references: List[str]) -> List[Dict]:
        """Compute ROUGE scores (simplified implementation)"""
        rouge_scores = []
        
        for gen, ref in zip(generated, references):
            gen_tokens = gen.lower().split()
            ref_tokens = ref.lower().split()
            
            if len(gen_tokens) == 0 or len(ref_tokens) == 0:
                rouge_scores.append({"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0})
                continue
                
            gen_unigrams = set(gen_tokens)
            ref_unigrams = set(ref_tokens)
            rouge1 = len(gen_unigrams & ref_unigrams) / len(ref_unigrams)
            
            gen_bigrams = set(zip(gen_tokens[:-1], gen_tokens[1:]))
            ref_bigrams = set(zip(ref_tokens[:-1], ref_tokens[1:]))
            rouge2 = len(gen_bigrams & ref_bigrams) / len(ref_bigrams) if ref_bigrams else 0
            
            rougeL = len(set(gen_tokens) & set(ref_tokens)) / max(len(gen_tokens), len(ref_tokens))
            
            rouge_scores.append({"rouge1": rouge1, "rouge2": rouge2, "rougeL": rougeL})
            
        return rouge_scores
        
    def compute_bertscore(self, generated: List[str], references: List[str]) -> Dict[str, List[float]]:
        """Compute BERTScore (simplified implementation)"""
        bert_scores = {"precision": [], "recall": [], "f1": []}
        
        for gen, ref in zip(generated, references):
            gen_tokens = gen.lower().split()
            ref_tokens = ref.lower().split()
            
            if len(gen_tokens) == 0 or len(ref_tokens) == 0:
                bert_scores["precision"].append(0.0)
                bert_scores["recall"].append(0.0)
                bert_scores["f1"].append(0.0)
                continue
                
            overlap = len(set(gen_tokens) & set(ref_tokens))
            precision = overlap / len(gen_tokens)
            recall = overlap / len(ref_tokens)
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            bert_scores["precision"].append(precision)
            bert_scores["recall"].append(recall)
            bert_scores["f1"].append(f1)
            
        return bert_scores


class MultilingualAccuracyBenchmark(Benchmark):
    """Multilingual accuracy benchmark"""
    
    def __init__(self):
        super().__init__(
            name="multilingual_accuracy",
            description="Evaluate model accuracy across multiple languages"
        )
        
    def load_data(self, language: str = None) -> List[Dict[str, str]]:
        """Load multilingual test data"""
        if language:
            return [
                {"question": f"भारत की राजधानी क्या है? ({language})", "answer": "नई दिल्ली"},
                {"question": f"What is the capital of India? ({language})", "answer": "New Delhi"},
                {"question": f"ভারতের রাজধানী কি? ({language})", "answer": "নতুন দিল্লী"},
            ]
        else:
            return [
                {"question": "भारत की राजधानी क्या है?", "answer": "नई दिल्ली", "language": "hi"},
                {"question": "What is the capital of India?", "answer": "New Delhi", "language": "en"},
                {"question": "ভারতের রাজধানী কি?", "answer": "নতুন দিল্লী", "language": "bn"},
            ]
        
    def evaluate(self, model, tokenizer, data: List[Dict[str, str]]) -> Dict[str, Any]:
        """Evaluate multilingual accuracy"""
        generated_answers = []
        
        for item in data:
            question = item["question"]
            generated = self.generate_text(model, tokenizer, question)
            generated_answers.append(generated)
            
        expected_answers = [item["answer"] for item in data]
        
        return {
            "generated_answers": generated_answers,
            "expected_answers": expected_answers
        }
        
    def generate_text(self, model, tokenizer, prompt: str) -> str:
        """Generate text from prompt"""
        inputs = tokenizer.tokenize(prompt)
        input_ids = inputs["input_ids"]
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                max_new_tokens=50,
                num_beams=1,
                do_sample=False,
                pad_token_id=tokenizer.tokenizer.pad_token_id if hasattr(tokenizer, 'tokenizer') else 0,
                eos_token_id=tokenizer.tokenizer.eos_token_id if hasattr(tokenizer, 'tokenizer') else 2,
            )
            
        generated_ids = outputs[0][input_ids.shape[1]:]
        generated_text = tokenizer.decode(generated_ids)
        
        return generated_text
        
    def compute_metrics(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Compute accuracy metrics"""
        correct = 0
        total = len(predictions)
        
        for pred, ref in zip(predictions, references):
            if self.check_answer_correctness(pred, ref):
                correct += 1
                
        accuracy = correct / total if total > 0 else 0
        
        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total
        }
        
    def check_answer_correctness(self, generated: str, expected: str) -> bool:
        """Check if generated answer is correct"""
        generated_clean = generated.lower().strip()
        expected_clean = expected.lower().strip()
        
        return expected_clean in generated_clean or generated_clean == expected_clean


class BenchmarkRegistry:
    """Registry for managing benchmarks"""
    
    def __init__(self):
        self.benchmarks = {}
        self.logger = logging.getLogger(__name__)
        
    def register(self, benchmark: Benchmark):
        """Register a benchmark"""
        self.benchmarks[benchmark.name] = benchmark
        self.logger.info(f"Registered benchmark: {benchmark.name}")
        
    def get(self, name: str) -> Optional[Benchmark]:
        """Get benchmark by name"""
        return self.benchmarks.get(name)
        
    def list_benchmarks(self) -> List[str]:
        """List all registered benchmarks"""
        return list(self.benchmarks.keys())
        
    def get_benchmarks_by_category(self, category: str) -> List[Benchmark]:
        """Get benchmarks by category"""
        category_benchmarks = []
        for benchmark in self.benchmarks.values():
            if category in benchmark.name.lower():
                category_benchmarks.append(benchmark)
        return category_benchmarks
        
    def run_benchmark(self, name: str, model, tokenizer, language: str = None) -> Dict[str, Any]:
        """Run a specific benchmark"""
        benchmark = self.get(name)
        if not benchmark:
            raise ValueError(f"Benchmark '{name}' not found")
            
        self.logger.info(f"Running benchmark: {name}")
        
        # Load data
        data = benchmark.load_data(language)
        
        # Evaluate
        results = benchmark.evaluate(model, tokenizer, data)
        
        # Compute metrics
        if "generated_texts" in results:
            metrics = benchmark.compute_metrics(
                results["generated_texts"], 
                results.get("reference_texts", [])
            )
        elif "generated_answers" in results:
            metrics = benchmark.compute_metrics(
                results["generated_answers"],
                results.get("expected_answers", [])
            )
        elif "perplexities" in results:
            metrics = benchmark.compute_metrics(
                results["perplexities"],
                []
            )
        else:
            metrics = {}
            
        return {
            "benchmark_name": name,
            "results": results,
            "metrics": metrics
        }
        
    def run_all_benchmarks(self, model, tokenizer, language: str = None) -> Dict[str, Dict[str, Any]]:
        """Run all registered benchmarks"""
        results = {}
        
        for name in self.benchmarks.keys():
            try:
                result = self.run_benchmark(name, model, tokenizer, language)
                results[name] = result
            except Exception as e:
                self.logger.error(f"Error running benchmark {name}: {e}")
                results[name] = {"error": str(e)}
                
        return results
        
    def save_results(self, results: Dict[str, Dict[str, Any]], output_path: str):
        """Save benchmark results to file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
            
        self.logger.info(f"Results saved to {output_path}")


# Global benchmark registry
benchmark_registry = BenchmarkRegistry()


def register_default_benchmarks():
    """Register default benchmarks"""
    benchmark_registry.register(PerplexityBenchmark())
    benchmark_registry.register(GenerationQualityBenchmark())
    benchmark_registry.register(MultilingualAccuracyBenchmark())


def get_benchmark(name: str) -> Optional[Benchmark]:
    """Get benchmark by name"""
    return benchmark_registry.get(name)


def get_benchmark_registry() -> BenchmarkRegistry:
    """Get the global benchmark registry"""
    return benchmark_registry


# Register default benchmarks on import
register_default_benchmarks()