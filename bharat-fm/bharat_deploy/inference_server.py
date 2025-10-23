"""
Inference server for BharatFM models with vLLM and Triton support
"""

import os
import json
import torch
import asyncio
from typing import Dict, List, Optional, Union, Any, AsyncGenerator
from concurrent.futures import ThreadPoolExecutor
import logging
from pathlib import Path
from datetime import datetime
import time
import numpy as np

from ..model import get_model_config, GLMForCausalLM, LlamaForCausalLM, MoEForCausalLM
from ..data import IndicTokenizer, BharatTokenizer


try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

try:
    import tritonclient.grpc as triton_grpc
    import tritonclient.http as triton_http
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


class InferenceConfig:
    """Configuration for inference server"""
    
    def __init__(
        self,
        # Model configuration
        model_path: str,
        model_type: str = "glm",
        
        # Server configuration
        host: str = "0.0.0.0",
        port: int = 8001,
        
        # Inference engine
        engine: str = "vllm",  # "vllm", "triton", "native"
        
        # vLLM configuration
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_num_batched_tokens: int = 8192,
        
        # Performance configuration
        max_batch_size: int = 32,
        max_context_length: int = 2048,
        
        # Bharat-specific settings
        enable_multilingual: bool = True,
        supported_languages: List[str] = None,
        
        # Monitoring
        enable_metrics: bool = True,
        metrics_interval: int = 60,
    ):
        self.model_path = model_path
        self.model_type = model_type
        self.host = host
        self.port = port
        self.engine = engine
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_batch_size = max_batch_size
        self.max_context_length = max_context_length
        self.enable_multilingual = enable_multilingual
        self.supported_languages = supported_languages or ["hi", "en", "bn", "ta", "te", "mr", "gu", "kn", "ml", "pa"]
        self.enable_metrics = enable_metrics
        self.metrics_interval = metrics_interval
        
    def to_dict(self) -> Dict:
        """Convert config to dictionary"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
        
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'InferenceConfig':
        """Create config from dictionary"""
        return cls(**config_dict)


class InferenceMetrics:
    """Metrics collection for inference server"""
    
    def __init__(self):
        self.total_requests = 0
        self.total_tokens_generated = 0
        self.total_generation_time = 0.0
        self.request_times = []
        self.error_count = 0
        self.language_counts = {}
        
    def record_request(self, tokens_generated: int, generation_time: float, language: str = None):
        """Record a completed request"""
        self.total_requests += 1
        self.total_tokens_generated += tokens_generated
        self.total_generation_time += generation_time
        self.request_times.append(generation_time)
        
        if language:
            self.language_counts[language] = self.language_counts.get(language, 0) + 1
            
    def record_error(self):
        """Record an error"""
        self.error_count += 1
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        avg_generation_time = (
            self.total_generation_time / self.total_requests
            if self.total_requests > 0 else 0.0
        )
        
        avg_tokens_per_second = (
            self.total_tokens_generated / self.total_generation_time
            if self.total_generation_time > 0 else 0.0
        )
        
        return {
            "total_requests": self.total_requests,
            "total_tokens_generated": self.total_tokens_generated,
            "total_generation_time": self.total_generation_time,
            "avg_generation_time": avg_generation_time,
            "avg_tokens_per_second": avg_tokens_per_second,
            "error_count": self.error_count,
            "error_rate": self.error_count / max(self.total_requests, 1),
            "language_distribution": self.language_counts,
            "timestamp": datetime.now().isoformat()
        }


class ModelServer:
    """Base class for model servers"""
    
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.metrics = InferenceMetrics()
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    async def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 50,
        num_beams: int = 1,
        do_sample: bool = True,
        language: str = None
    ) -> Dict[str, Any]:
        """Generate text from prompt"""
        raise NotImplementedError
        
    async def generate_batch(
        self,
        prompts: List[str],
        max_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 50,
        num_beams: int = 1,
        do_sample: bool = True,
        languages: List[str] = None
    ) -> List[Dict[str, Any]]:
        """Generate text for multiple prompts"""
        raise NotImplementedError
        
    async def get_embeddings(self, text: str) -> List[float]:
        """Get text embeddings"""
        raise NotImplementedError
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get server metrics"""
        return self.metrics.get_metrics()


class VLLMModelServer(ModelServer):
    """vLLM-based model server"""
    
    def __init__(self, config: InferenceConfig):
        super().__init__(config)
        
        if not VLLM_AVAILABLE:
            raise ImportError("vLLM is not available. Install with: pip install vllm")
            
        self.llm = None
        self.tokenizer = None
        
        # Initialize vLLM
        self._initialize_vllm()
        
    def _initialize_vllm(self):
        """Initialize vLLM engine"""
        self.logger.info("Initializing vLLM engine...")
        
        try:
            self.llm = LLM(
                model=self.config.model_path,
                tensor_parallel_size=self.config.tensor_parallel_size,
                gpu_memory_utilization=self.config.gpu_memory_utilization,
                max_num_batched_tokens=self.config.max_num_batched_tokens,
            )
            
            self.logger.info("vLLM engine initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing vLLM: {e}")
            raise
            
    async def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 50,
        num_beams: int = 1,
        do_sample: bool = True,
        language: str = None
    ) -> Dict[str, Any]:
        """Generate text using vLLM"""
        
        start_time = time.time()
        
        try:
            # Set up sampling parameters
            sampling_params = SamplingParams(
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                use_beam_search=num_beams > 1,
                best_of=num_beams,
            )
            
            # Generate
            outputs = self.llm.generate([prompt], sampling_params)
            
            # Extract generated text
            generated_text = outputs[0].outputs[0].text
            tokens_generated = len(outputs[0].outputs[0].token_ids)
            
            # Calculate generation time
            generation_time = time.time() - start_time
            
            # Record metrics
            self.metrics.record_request(tokens_generated, generation_time, language)
            
            return {
                "generated_text": generated_text,
                "prompt": prompt,
                "tokens_generated": tokens_generated,
                "generation_time": generation_time,
                "language": language
            }
            
        except Exception as e:
            self.logger.error(f"Generation error: {e}")
            self.metrics.record_error()
            raise
            
    async def generate_batch(
        self,
        prompts: List[str],
        max_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 50,
        num_beams: int = 1,
        do_sample: bool = True,
        languages: List[str] = None
    ) -> List[Dict[str, Any]]:
        """Generate text for multiple prompts using vLLM"""
        
        start_time = time.time()
        
        try:
            # Set up sampling parameters
            sampling_params = SamplingParams(
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                use_beam_search=num_beams > 1,
                best_of=num_beams,
            )
            
            # Generate
            outputs = self.llm.generate(prompts, sampling_params)
            
            # Process results
            results = []
            total_tokens = 0
            
            for i, output in enumerate(outputs):
                generated_text = output.outputs[0].text
                tokens_generated = len(output.outputs[0].token_ids)
                total_tokens += tokens_generated
                
                language = languages[i] if languages and i < len(languages) else None
                
                results.append({
                    "generated_text": generated_text,
                    "prompt": prompts[i],
                    "tokens_generated": tokens_generated,
                    "language": language
                })
                
            # Calculate generation time
            generation_time = time.time() - start_time
            
            # Record metrics
            for i, result in enumerate(results):
                language = languages[i] if languages and i < len(languages) else None
                self.metrics.record_request(result["tokens_generated"], generation_time, language)
                
            return results
            
        except Exception as e:
            self.logger.error(f"Batch generation error: {e}")
            self.metrics.record_error()
            raise


class NativeModelServer(ModelServer):
    """Native PyTorch model server"""
    
    def __init__(self, config: InferenceConfig):
        super().__init__(config)
        
        self.model = None
        self.tokenizer = None
        self.device = None
        
        # Initialize model
        self._initialize_model()
        
    def _initialize_model(self):
        """Initialize native PyTorch model"""
        self.logger.info("Initializing native PyTorch model...")
        
        try:
            # Setup device
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Load model configuration
            if os.path.exists(os.path.join(self.config.model_path, "config.json")):
                model_config = get_model_config("bharat-base")
                model_config = model_config.from_pretrained(self.config.model_path)
            else:
                model_config = get_model_config("bharat-base")
                
            # Load model based on type
            if self.config.model_type == "glm":
                self.model = GLMForCausalLM.from_pretrained(self.config.model_path)
            elif self.config.model_type == "llama":
                self.model = LlamaForCausalLM.from_pretrained(self.config.model_path)
            elif self.config.model_type == "moe":
                self.model = MoEForCausalLM.from_pretrained(self.config.model_path)
            else:
                raise ValueError(f"Unknown model type: {self.config.model_type}")
                
            # Move model to device
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Load tokenizer
            try:
                self.tokenizer = IndicTokenizer.from_pretrained(self.config.model_path)
            except:
                self.tokenizer = BharatTokenizer.load_vocab(self.config.model_path)
                
            self.logger.info("Native PyTorch model initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing native model: {e}")
            raise
            
    async def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 50,
        num_beams: int = 1,
        do_sample: bool = True,
        language: str = None
    ) -> Dict[str, Any]:
        """Generate text using native PyTorch"""
        
        start_time = time.time()
        
        try:
            # Tokenize
            inputs = self.tokenizer.tokenize(prompt)
            input_ids = inputs["input_ids"].to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=input_ids,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    num_beams=num_beams,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.tokenizer.pad_token_id if hasattr(self.tokenizer, 'tokenizer') else 0,
                    eos_token_id=self.tokenizer.tokenizer.eos_token_id if hasattr(self.tokenizer, 'tokenizer') else 2,
                )
                
            # Decode
            generated_ids = outputs[0][input_ids.shape[1]:]
            generated_text = self.tokenizer.decode(generated_ids)
            tokens_generated = len(generated_ids)
            
            # Calculate generation time
            generation_time = time.time() - start_time
            
            # Record metrics
            self.metrics.record_request(tokens_generated, generation_time, language)
            
            return {
                "generated_text": generated_text,
                "prompt": prompt,
                "tokens_generated": tokens_generated,
                "generation_time": generation_time,
                "language": language
            }
            
        except Exception as e:
            self.logger.error(f"Generation error: {e}")
            self.metrics.record_error()
            raise
            
    async def generate_batch(
        self,
        prompts: List[str],
        max_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 50,
        num_beams: int = 1,
        do_sample: bool = True,
        languages: List[str] = None
    ) -> List[Dict[str, Any]]:
        """Generate text for multiple prompts using native PyTorch"""
        
        results = []
        
        for i, prompt in enumerate(prompts):
            language = languages[i] if languages and i < len(languages) else None
            
            result = await self.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                num_beams=num_beams,
                do_sample=do_sample,
                language=language
            )
            
            results.append(result)
            
        return results
        
    async def get_embeddings(self, text: str) -> List[float]:
        """Get text embeddings"""
        try:
            # Tokenize
            inputs = self.tokenizer.tokenize(text)
            input_ids = inputs["input_ids"].to(self.device)
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.model.model(input_ids=input_ids)
                embeddings = outputs["last_hidden_state"]
                
            # Pool embeddings (mean pooling)
            embeddings = embeddings.mean(dim=1)
            
            return embeddings.squeeze().cpu().tolist()
            
        except Exception as e:
            self.logger.error(f"Embedding error: {e}")
            raise


class InferenceServer:
    """Main inference server class"""
    
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.server = None
        
        # Setup logging
        self.setup_logging()
        
        # Initialize server based on engine
        self._initialize_server()
        
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
    def _initialize_server(self):
        """Initialize the appropriate server based on engine"""
        if self.config.engine == "vllm":
            self.server = VLLMModelServer(self.config)
        elif self.config.engine == "triton":
            if not TRITON_AVAILABLE:
                raise ImportError("Triton is not available. Install with: pip install tritonclient")
            # Triton server would be implemented here
            raise NotImplementedError("Triton server not yet implemented")
        elif self.config.engine == "native":
            self.server = NativeModelServer(self.config)
        else:
            raise ValueError(f"Unknown inference engine: {self.config.engine}")
            
        self.logger.info(f"Initialized {self.config.engine} inference server")
        
    async def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 50,
        num_beams: int = 1,
        do_sample: bool = True,
        language: str = None
    ) -> Dict[str, Any]:
        """Generate text from prompt"""
        return await self.server.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            do_sample=do_sample,
            language=language
        )
        
    async def generate_batch(
        self,
        prompts: List[str],
        max_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 50,
        num_beams: int = 1,
        do_sample: bool = True,
        languages: List[str] = None
    ) -> List[Dict[str, Any]]:
        """Generate text for multiple prompts"""
        return await self.server.generate_batch(
            prompts=prompts,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            do_sample=do_sample,
            languages=languages
        )
        
    async def get_embeddings(self, text: str) -> List[float]:
        """Get text embeddings"""
        return await self.server.get_embeddings(text)
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get server metrics"""
        return self.server.get_metrics()
        
    async def stream_generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 50,
        do_sample: bool = True
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Generate text with streaming"""
        # For now, implement simple streaming
        # In practice, this would use proper streaming capabilities
        result = await self.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=do_sample
        )
        
        # Yield result in chunks
        text = result["generated_text"]
        chunk_size = max(1, len(text) // 10)
        
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i + chunk_size]
            yield {
                "chunk": chunk,
                "done": i + chunk_size >= len(text),
                "progress": min(100, (i + chunk_size) / len(text) * 100)
            }
            
            # Small delay for streaming effect
            await asyncio.sleep(0.01)