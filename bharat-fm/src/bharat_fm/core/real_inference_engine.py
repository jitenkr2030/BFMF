"""
Real Inference Engine for Bharat-FM
Actual implementation using transformer models and real AI capabilities
"""

import asyncio
import time
import uuid
from typing import Dict, List, Any, Optional, AsyncIterator, Union
from datetime import datetime
import logging
import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    pipeline,
    TextGenerationPipeline,
    TextClassificationPipeline
)
from transformers import GPT2LMHeadModel, GPT2Tokenizer, BertModel, BertTokenizer
import numpy as np

logger = logging.getLogger(__name__)

class RealInferenceEngine:
    """Real inference engine with actual transformer models"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Engine state
        self.engine_id = str(uuid.uuid4())
        self.start_time = datetime.utcnow()
        self.is_running = False
        
        # Model registry - actual models loaded here
        self.models: Dict[str, Dict[str, Any]] = {}
        self.tokenizers: Dict[str, Any] = {}
        self.pipelines: Dict[str, Any] = {}
        
        # Performance tracking
        self.performance_metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "avg_latency": 0.0,
            "total_tokens_processed": 0,
            "model_usage_stats": {}
        }
        
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize actual transformer models"""
        try:
            # Initialize GPT-2 model for text generation
            logger.info("Loading GPT-2 model...")
            gpt2_model_name = "gpt2"  # Using smaller model for demo, can be upgraded
            gpt2_tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_name)
            gpt2_model = GPT2LMHeadModel.from_pretrained(gpt2_model_name)
            gpt2_model.to(self.device)
            
            self.models["gpt2"] = {
                "model": gpt2_model,
                "type": "text_generation",
                "max_length": 512,
                "description": "GPT-2 for text generation"
            }
            self.tokenizers["gpt2"] = gpt2_tokenizer
            
            # Initialize BERT model for text classification
            logger.info("Loading BERT model...")
            bert_model_name = "bert-base-uncased"
            bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
            bert_model = BertModel.from_pretrained(bert_model_name)
            bert_model.to(self.device)
            
            self.models["bert"] = {
                "model": bert_model,
                "type": "embedding",
                "max_length": 512,
                "description": "BERT for text embeddings"
            }
            self.tokenizers["bert"] = bert_tokenizer
            
            # Initialize sentiment analysis pipeline
            logger.info("Loading sentiment analysis pipeline...")
            sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device=0 if torch.cuda.is_available() else -1
            )
            
            self.pipelines["sentiment"] = sentiment_pipeline
            
            # Initialize text generation pipeline
            logger.info("Loading text generation pipeline...")
            text_gen_pipeline = pipeline(
                "text-generation",
                model=gpt2_model_name,
                tokenizer=gpt2_tokenizer,
                device=0 if torch.cuda.is_available() else -1
            )
            
            self.pipelines["text_generation"] = text_gen_pipeline
            
            logger.info("All models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            raise
    
    async def start(self):
        """Start the inference engine"""
        if self.is_running:
            logger.warning("Real inference engine is already running")
            return
        
        self.is_running = True
        logger.info(f"Real inference engine {self.engine_id} started")
    
    async def stop(self):
        """Stop the inference engine"""
        if not self.is_running:
            logger.warning("Real inference engine is not running")
            return
        
        # Cleanup models
        for model_name, model_info in self.models.items():
            if "model" in model_info:
                del model_info["model"]
        
        self.models.clear()
        self.tokenizers.clear()
        self.pipelines.clear()
        self.is_running = False
        
        logger.info(f"Real inference engine {self.engine_id} stopped")
    
    async def generate_text(self, 
                           prompt: str, 
                           model_id: str = "gpt2",
                           max_length: int = 100,
                           temperature: float = 0.7,
                           do_sample: bool = True,
                           num_return_sequences: int = 1) -> Dict[str, Any]:
        """Generate text using actual transformer model"""
        if not self.is_running:
            raise RuntimeError("Real inference engine is not running")
        
        start_time = time.time()
        request_id = f"gen_{int(time.time() * 1000000)}"
        
        try:
            if model_id not in self.models:
                raise ValueError(f"Model {model_id} not available")
            
            model_info = self.models[model_id]
            tokenizer = self.tokenizers[model_id]
            model = model_info["model"]
            
            # Tokenize input
            inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate text
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=max_length,
                    temperature=temperature,
                    do_sample=do_sample,
                    num_return_sequences=num_return_sequences,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # Decode generated text
            generated_texts = []
            for output in outputs:
                generated_text = tokenizer.decode(output, skip_special_tokens=True)
                generated_texts.append(generated_text)
            
            # Calculate metrics
            latency = time.time() - start_time
            input_tokens = inputs["input_ids"].shape[1]
            output_tokens = outputs.shape[1] - input_tokens
            total_tokens = input_tokens + output_tokens
            
            # Update performance metrics
            self._update_performance_metrics(model_id, latency, total_tokens, success=True)
            
            result = {
                "request_id": request_id,
                "generated_texts": generated_texts,
                "model_used": model_id,
                "latency": latency,
                "tokens": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": total_tokens
                },
                "parameters": {
                    "max_length": max_length,
                    "temperature": temperature,
                    "do_sample": do_sample,
                    "num_return_sequences": num_return_sequences
                },
                "timestamp": datetime.utcnow().isoformat(),
                "engine_id": self.engine_id
            }
            
            logger.info(f"Text generation completed: {request_id} in {latency:.3f}s")
            return result
            
        except Exception as e:
            logger.error(f"Text generation failed for request {request_id}: {e}")
            self._update_performance_metrics(model_id, time.time() - start_time, 0, success=False)
            raise
    
    async def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment using actual sentiment analysis model"""
        if not self.is_running:
            raise RuntimeError("Real inference engine is not running")
        
        start_time = time.time()
        request_id = f"sent_{int(time.time() * 1000000)}"
        
        try:
            if "sentiment" not in self.pipelines:
                raise ValueError("Sentiment analysis pipeline not available")
            
            pipeline = self.pipelines["sentiment"]
            
            # Perform sentiment analysis
            results = pipeline(text)
            
            # Calculate metrics
            latency = time.time() - start_time
            tokens = len(text.split())
            
            # Update performance metrics
            self._update_performance_metrics("sentiment", latency, tokens, success=True)
            
            result = {
                "request_id": request_id,
                "text": text,
                "sentiment_results": results,
                "model_used": "distilbert-sentiment",
                "latency": latency,
                "tokens_processed": tokens,
                "timestamp": datetime.utcnow().isoformat(),
                "engine_id": self.engine_id
            }
            
            logger.info(f"Sentiment analysis completed: {request_id} in {latency:.3f}s")
            return result
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed for request {request_id}: {e}")
            self._update_performance_metrics("sentiment", time.time() - start_time, 0, success=False)
            raise
    
    async def get_embeddings(self, text: str, model_id: str = "bert") -> Dict[str, Any]:
        """Get text embeddings using actual transformer model"""
        if not self.is_running:
            raise RuntimeError("Real inference engine is not running")
        
        start_time = time.time()
        request_id = f"emb_{int(time.time() * 1000000)}"
        
        try:
            if model_id not in self.models:
                raise ValueError(f"Model {model_id} not available")
            
            model_info = self.models[model_id]
            tokenizer = self.tokenizers[model_id]
            model = model_info["model"]
            
            # Tokenize input
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get embeddings
            with torch.no_grad():
                outputs = model(**inputs)
                
                # Use mean pooling of last hidden state
                embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            
            # Calculate metrics
            latency = time.time() - start_time
            tokens = inputs["input_ids"].shape[1]
            
            # Update performance metrics
            self._update_performance_metrics(model_id, latency, tokens, success=True)
            
            result = {
                "request_id": request_id,
                "text": text,
                "embeddings": embeddings.tolist(),
                "embedding_shape": embeddings.shape,
                "model_used": model_id,
                "latency": latency,
                "tokens_processed": tokens,
                "timestamp": datetime.utcnow().isoformat(),
                "engine_id": self.engine_id
            }
            
            logger.info(f"Embedding generation completed: {request_id} in {latency:.3f}s")
            return result
            
        except Exception as e:
            logger.error(f"Embedding generation failed for request {request_id}: {e}")
            self._update_performance_metrics(model_id, time.time() - start_time, 0, success=False)
            raise
    
    async def batch_generate_text(self, 
                                 prompts: List[str],
                                 model_id: str = "gpt2",
                                 max_length: int = 100,
                                 temperature: float = 0.7) -> List[Dict[str, Any]]:
        """Generate text for multiple prompts in batch"""
        if not self.is_running:
            raise RuntimeError("Real inference engine is not running")
        
        # Execute all requests concurrently
        tasks = []
        for i, prompt in enumerate(prompts):
            task = asyncio.create_task(
                self.generate_text(
                    prompt=prompt,
                    model_id=model_id,
                    max_length=max_length,
                    temperature=temperature
                )
            )
            tasks.append(task)
        
        # Wait for all to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    "prompt": prompts[i],
                    "error": str(result),
                    "success": False,
                    "timestamp": datetime.utcnow().isoformat()
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def get_engine_status(self) -> Dict[str, Any]:
        """Get engine status and real metrics"""
        return {
            "engine_id": self.engine_id,
            "status": "running" if self.is_running else "stopped",
            "uptime_seconds": (datetime.utcnow() - self.start_time).total_seconds(),
            "start_time": self.start_time.isoformat(),
            "device": str(self.device),
            "available_models": list(self.models.keys()),
            "available_pipelines": list(self.pipelines.keys()),
            "performance_metrics": self.performance_metrics,
            "model_details": {
                name: {
                    "type": info["type"],
                    "description": info["description"],
                    "max_length": info["max_length"]
                }
                for name, info in self.models.items()
            }
        }
    
    def _update_performance_metrics(self, model_id: str, latency: float, tokens: int, success: bool):
        """Update performance metrics with real data"""
        self.performance_metrics["total_requests"] += 1
        
        if success:
            self.performance_metrics["successful_requests"] += 1
        else:
            self.performance_metrics["failed_requests"] += 1
        
        # Update average latency
        total_latency = (self.performance_metrics["avg_latency"] * 
                         (self.performance_metrics["total_requests"] - 1) + 
                         latency)
        self.performance_metrics["avg_latency"] = total_latency / self.performance_metrics["total_requests"]
        
        # Update token count
        self.performance_metrics["total_tokens_processed"] += tokens
        
        # Update model usage stats
        if model_id not in self.performance_metrics["model_usage_stats"]:
            self.performance_metrics["model_usage_stats"][model_id] = {
                "usage_count": 0,
                "total_tokens": 0,
                "total_latency": 0.0
            }
        
        stats = self.performance_metrics["model_usage_stats"][model_id]
        stats["usage_count"] += 1
        stats["total_tokens"] += tokens
        stats["total_latency"] += latency
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform real health check"""
        health_status = {
            "engine_id": self.engine_id,
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "checks": {
                "is_running": self.is_running,
                "models_loaded": len(self.models) > 0,
                "pipelines_loaded": len(self.pipelines) > 0,
                "device_available": torch.cuda.is_available()
            }
        }
        
        # Check if engine is responsive with a real test
        try:
            # Test with a simple generation
            test_result = await self.generate_text("Hello", max_length=10)
            health_status["checks"]["responsive"] = True
            health_status["checks"]["test_latency"] = test_result["latency"]
            
        except Exception as e:
            health_status["status"] = "unhealthy"
            health_status["checks"]["responsive"] = False
            health_status["error"] = str(e)
        
        return health_status


# Convenience function for easy usage
async def create_real_inference_engine(config: Optional[Dict[str, Any]] = None) -> RealInferenceEngine:
    """Create and start real inference engine"""
    engine = RealInferenceEngine(config)
    await engine.start()
    return engine