"""
Evaluation utilities for Language AI use cases
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import sacrebleu
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from bharat_eval.evaluator import BharatEvaluator


class MultilingualEvaluator(BharatEvaluator):
    """
    Comprehensive evaluator for multilingual models
    """
    
    def __init__(self, model, tokenizer):
        super().__init__(model, tokenizer)
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.supported_languages = ['hi', 'bn', 'ta', 'te', 'mr', 'gu', 'kn', 'ml', 'pa', 'or', 'as', 'en']
    
    def evaluate_multilingual_performance(
        self,
        test_dataset: Dataset,
        languages: List[str] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate model performance across multiple languages
        
        Args:
            test_dataset: Dataset with multilingual examples
            languages: List of languages to evaluate (default: all supported)
        """
        languages = languages or self.supported_languages
        results = {}
        
        for lang in languages:
            lang_data = test_dataset.filter(lambda x: x.get('language', 'en') == lang)
            
            if len(lang_data) > 0:
                lang_results = self.evaluate_language_specific(lang_data, lang)
                results[lang] = lang_results
        
        # Calculate overall metrics
        overall_results = self._calculate_overall_metrics(results)
        results['overall'] = overall_results
        
        return results
    
    def evaluate_language_specific(self, dataset: Dataset, language: str) -> Dict[str, float]:
        """
        Evaluate performance for a specific language
        """
        results = {}
        
        # Perplexity evaluation
        perplexity = self._calculate_perplexity(dataset, language)
        results['perplexity'] = perplexity
        
        # Text generation quality
        if 'reference' in dataset.column_names:
            generation_scores = self._evaluate_generation_quality(dataset, language)
            results.update(generation_scores)
        
        # Classification accuracy (if labels available)
        if 'label' in dataset.column_names:
            accuracy = self._evaluate_classification_accuracy(dataset, language)
            results['accuracy'] = accuracy
        
        return results
    
    def _calculate_perplexity(self, dataset: Dataset, language: str) -> float:
        """
        Calculate perplexity for the given language
        """
        total_loss = 0
        total_tokens = 0
        
        for batch in dataset:
            inputs = self.tokenizer(
                batch['text'], 
                return_tensors='pt', 
                padding=True, 
                truncation=True
            )
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                loss = outputs.loss
                
            total_loss += loss.item() * inputs['input_ids'].size(1)
            total_tokens += inputs['input_ids'].size(1)
        
        avg_loss = total_loss / total_tokens
        perplexity = np.exp(avg_loss)
        
        return perplexity
    
    def _evaluate_generation_quality(self, dataset: Dataset, language: str) -> Dict[str, float]:
        """
        Evaluate text generation quality using various metrics
        """
        predictions = []
        references = []
        
        for item in dataset:
            # Generate prediction
            prompt = item.get('prompt', item['text'][:100])
            prediction = self._generate_text(prompt, language)
            predictions.append(prediction)
            references.append(item['reference'])
        
        # ROUGE scores
        rouge_scores = self._calculate_rouge_scores(predictions, references)
        
        # BERTScore
        bert_scores = self._calculate_bert_scores(predictions, references)
        
        return {**rouge_scores, **bert_scores}
    
    def _calculate_rouge_scores(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """
        Calculate ROUGE scores
        """
        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []
        
        for pred, ref in zip(predictions, references):
            scores = self.rouge_scorer.score(ref, pred)
            rouge1_scores.append(scores['rouge1'].fmeasure)
            rouge2_scores.append(scores['rouge2'].fmeasure)
            rougeL_scores.append(scores['rougeL'].fmeasure)
        
        return {
            'rouge1': np.mean(rouge1_scores),
            'rouge2': np.mean(rouge2_scores),
            'rougeL': np.mean(rougeL_scores)
        }
    
    def _calculate_bert_scores(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """
        Calculate BERTScore
        """
        P, R, F1 = bert_score(predictions, references, lang='en', verbose=False)
        
        return {
            'bert_precision': P.mean().item(),
            'bert_recall': R.mean().item(),
            'bert_f1': F1.mean().item()
        }
    
    def _evaluate_classification_accuracy(self, dataset: Dataset, language: str) -> float:
        """
        Evaluate classification accuracy
        """
        correct = 0
        total = 0
        
        for item in dataset:
            text = item['text']
            true_label = item['label']
            
            # Get model prediction
            inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
            with torch.no_grad():
                outputs = self.model(**inputs)
                predicted_label = torch.argmax(outputs.logits, dim=-1).item()
            
            if predicted_label == true_label:
                correct += 1
            total += 1
        
        return correct / total if total > 0 else 0
    
    def _generate_text(self, prompt: str, language: str) -> str:
        """
        Generate text in specified language
        """
        lang_prompt = f"[{language.upper()}] {prompt}"
        
        inputs = self.tokenizer(lang_prompt, return_tensors='pt', padding=True, truncation=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        outputs = self.model.generate(
            **inputs,
            max_length=200,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True
        )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def _calculate_overall_metrics(self, language_results: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """
        Calculate overall metrics across all languages
        """
        overall = {}
        
        # Get all metric keys
        all_metrics = set()
        for lang_results in language_results.values():
            all_metrics.update(lang_results.keys())
        
        # Calculate average for each metric
        for metric in all_metrics:
            values = []
            for lang_results in language_results.values():
                if metric in lang_results:
                    values.append(lang_results[metric])
            
            if values:
                overall[metric] = np.mean(values)
        
        return overall


class TranslationEvaluator(BharatEvaluator):
    """
    Evaluator for translation tasks
    """
    
    def __init__(self, model, tokenizer):
        super().__init__(model, tokenizer)
        self.supported_pairs = [
            ('hi', 'en'), ('en', 'hi'),
            ('bn', 'en'), ('en', 'bn'),
            ('ta', 'en'), ('en', 'ta'),
            ('te', 'en'), ('en', 'te'),
            ('mr', 'en'), ('en', 'mr'),
            ('gu', 'en'), ('en', 'gu')
        ]
    
    def evaluate_translation(
        self,
        test_dataset: Dataset,
        source_lang: str,
        target_lang: str
    ) -> Dict[str, float]:
        """
        Evaluate translation quality for a language pair
        """
        if (source_lang, target_lang) not in self.supported_pairs:
            raise ValueError(f"Unsupported language pair: {source_lang} -> {target_lang}")
        
        predictions = []
        references = []
        
        for item in test_dataset:
            source_text = item['source']
            target_text = item['target']
            
            # Translate
            translation = self._translate_text(source_text, source_lang, target_lang)
            predictions.append(translation)
            references.append(target_text)
        
        # Calculate BLEU score
        bleu_score = self._calculate_bleu_score(predictions, references)
        
        # Calculate ROUGE scores
        rouge_scores = self._calculate_rouge_scores(predictions, references)
        
        # Calculate BERTScore
        bert_scores = self._calculate_bert_scores(predictions, references)
        
        return {
            'bleu': bleu_score,
            **rouge_scores,
            **bert_scores
        }
    
    def _translate_text(self, text: str, source_lang: str, target_lang: str) -> str:
        """
        Translate text using the model
        """
        prompt = f"Translate from {source_lang} to {target_lang}: {text}"
        
        inputs = self.tokenizer(prompt, return_tensors='pt', padding=True, truncation=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        outputs = self.model.generate(
            **inputs,
            max_length=512,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True
        )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def _calculate_bleu_score(self, predictions: List[str], references: List[str]) -> float:
        """
        Calculate BLEU score
        """
        bleu = sacrebleu.corpus_bleu(predictions, [references])
        return bleu.score
    
    def _calculate_rouge_scores(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """
        Calculate ROUGE scores for translation
        """
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []
        
        for pred, ref in zip(predictions, references):
            scores = scorer.score(ref, pred)
            rouge1_scores.append(scores['rouge1'].fmeasure)
            rouge2_scores.append(scores['rouge2'].fmeasure)
            rougeL_scores.append(scores['rougeL'].fmeasure)
        
        return {
            'rouge1': np.mean(rouge1_scores),
            'rouge2': np.mean(rouge2_scores),
            'rougeL': np.mean(rougeL_scores)
        }
    
    def _calculate_bert_scores(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """
        Calculate BERTScore for translation
        """
        P, R, F1 = bert_score(predictions, references, lang='en', verbose=False)
        
        return {
            'bert_precision': P.mean().item(),
            'bert_recall': R.mean().item(),
            'bert_f1': F1.mean().item()
        }


class CodeSwitchingEvaluator(BharatEvaluator):
    """
    Evaluator for code-switching handling
    """
    
    def __init__(self, model, tokenizer):
        super().__init__(model, tokenizer)
    
    def evaluate_code_switching(
        self,
        test_dataset: Dataset
    ) -> Dict[str, float]:
        """
        Evaluate model's ability to handle code-switching
        """
        results = {}
        
        # Code-switching detection accuracy
        detection_accuracy = self._evaluate_detection_accuracy(test_dataset)
        results['detection_accuracy'] = detection_accuracy
        
        # Generation quality for code-switched input
        generation_quality = self._evaluate_code_switched_generation(test_dataset)
        results.update(generation_quality)
        
        # Language consistency
        language_consistency = self._evaluate_language_consistency(test_dataset)
        results['language_consistency'] = language_consistency
        
        return results
    
    def _evaluate_detection_accuracy(self, dataset: Dataset) -> float:
        """
        Evaluate accuracy of code-switching detection
        """
        correct = 0
        total = 0
        
        for item in dataset:
            text = item['text']
            true_segments = item.get('segments', [])
            
            # Detect code-switching
            detected_segments = self._detect_code_switching(text)
            
            # Simple accuracy check (can be made more sophisticated)
            if len(detected_segments) == len(true_segments):
                correct += 1
            total += 1
        
        return correct / total if total > 0 else 0
    
    def _evaluate_code_switched_generation(self, dataset: Dataset) -> Dict[str, float]:
        """
        Evaluate generation quality for code-switched input
        """
        predictions = []
        references = []
        
        for item in dataset:
            if 'prompt' in item and 'reference' in item:
                prompt = item['prompt']
                reference = item['reference']
                
                # Generate response
                response = self._generate_text(prompt)
                predictions.append(response)
                references.append(reference)
        
        if predictions:
            rouge_scores = self._calculate_rouge_scores(predictions, references)
            return rouge_scores
        
        return {}
    
    def _evaluate_language_consistency(self, dataset: Dataset) -> float:
        """
        Evaluate language consistency in generated text
        """
        consistency_scores = []
        
        for item in dataset:
            if 'prompt' in item:
                prompt = item['prompt']
                response = self._generate_text(prompt)
                
                # Check if response maintains language consistency
                consistency = self._check_language_consistency(response)
                consistency_scores.append(consistency)
        
        return np.mean(consistency_scores) if consistency_scores else 0
    
    def _detect_code_switching(self, text: str) -> List[Dict]:
        """
        Detect code-switching in text
        """
        # Simple implementation - can be enhanced
        segments = []
        current_lang = 'en'
        start_pos = 0
        
        for i, char in enumerate(text):
            if char.isascii() and current_lang != 'en':
                segments.append({
                    'start': start_pos,
                    'end': i,
                    'language': current_lang
                })
                current_lang = 'en'
                start_pos = i
            elif not char.isascii() and current_lang == 'en':
                segments.append({
                    'start': start_pos,
                    'end': i,
                    'language': current_lang
                })
                current_lang = 'hi'  # Default to Hindi for non-ASCII
                start_pos = i
        
        # Add final segment
        if start_pos < len(text):
            segments.append({
                'start': start_pos,
                'end': len(text),
                'language': current_lang
            })
        
        return segments
    
    def _generate_text(self, prompt: str) -> str:
        """
        Generate text from prompt
        """
        inputs = self.tokenizer(prompt, return_tensors='pt', padding=True, truncation=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        outputs = self.model.generate(
            **inputs,
            max_length=200,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True
        )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def _calculate_rouge_scores(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """
        Calculate ROUGE scores
        """
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []
        
        for pred, ref in zip(predictions, references):
            scores = scorer.score(ref, pred)
            rouge1_scores.append(scores['rouge1'].fmeasure)
            rouge2_scores.append(scores['rouge2'].fmeasure)
            rougeL_scores.append(scores['rougeL'].fmeasure)
        
        return {
            'rouge1': np.mean(rouge1_scores),
            'rouge2': np.mean(rouge2_scores),
            'rougeL': np.mean(rougeL_scores)
        }
    
    def _check_language_consistency(self, text: str) -> float:
        """
        Check language consistency in generated text
        """
        # Simple heuristic: check if language switches are appropriate
        segments = self._detect_code_switching(text)
        
        if len(segments) <= 1:
            return 1.0  # Single language is consistent
        
        # Check if switches make sense (simplified)
        consistency_score = 1.0
        for i in range(len(segments) - 1):
            if segments[i]['language'] == segments[i + 1]['language']:
                consistency_score -= 0.1  # Penalty for unnecessary switches
        
        return max(0, consistency_score)