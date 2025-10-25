"""
FastAPI-based deployment API for BharatFM models
"""

import os
import json
import torch
from typing import Dict, List, Optional, Union, Any
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
import logging
from pathlib import Path
import uvicorn
from datetime import datetime

from ..model import get_model_config, GLMForCausalLM, LlamaForCausalLM, MoEForCausalLM
from ..data import IndicTokenizer, BharatTokenizer


class GenerationRequest(BaseModel):
    """Request model for text generation"""
    prompt: str = Field(..., description="Input prompt for text generation")
    max_tokens: int = Field(100, description="Maximum number of tokens to generate")
    temperature: float = Field(1.0, description="Sampling temperature")
    top_p: float = Field(1.0, description="Top-p sampling parameter")
    top_k: int = Field(50, description="Top-k sampling parameter")
    num_beams: int = Field(1, description="Number of beams for beam search")
    do_sample: bool = Field(True, description="Whether to use sampling")
    language: Optional[str] = Field(None, description="Language hint for generation")


class GenerationResponse(BaseModel):
    """Response model for text generation"""
    generated_text: str = Field(..., description="Generated text")
    prompt: str = Field(..., description="Original prompt")
    tokens_generated: int = Field(..., description="Number of tokens generated")
    generation_time: float = Field(..., description="Time taken for generation in seconds")
    language_detected: Optional[str] = Field(None, description="Detected language")


class ModelInfo(BaseModel):
    """Model information response"""
    model_name: str = Field(..., description="Name of the model")
    model_type: str = Field(..., description="Type of the model")
    model_size: str = Field(..., description="Size of the model")
    supported_languages: List[str] = Field(..., description="List of supported languages")
    max_context_length: int = Field(..., description="Maximum context length")
    version: str = Field(..., description="Model version")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Health status")
    timestamp: str = Field(..., description="Current timestamp")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    device: str = Field(..., description="Device being used")


class DeploymentConfig:
    """Configuration for model deployment"""
    
    def __init__(
        self,
        # Model configuration
        model_path: str,
        model_type: str = "glm",
        
        # Server configuration
        host: str = "0.0.0.0",
        port: int = 8000,
        workers: int = 1,
        
        # Model loading configuration
        device: str = "auto",
        dtype: str = "auto",
        
        # API configuration
        api_key: Optional[str] = None,
        rate_limit: int = 100,
        
        # Bharat-specific settings
        enable_multilingual: bool = True,
        supported_languages: List[str] = None,
        
        # Logging and monitoring
        log_level: str = "INFO",
        enable_metrics: bool = True,
    ):
        self.model_path = model_path
        self.model_type = model_type
        self.host = host
        self.port = port
        self.workers = workers
        self.device = device
        self.dtype = dtype
        self.api_key = api_key
        self.rate_limit = rate_limit
        self.enable_multilingual = enable_multilingual
        self.supported_languages = supported_languages or ["hi", "en", "bn", "ta", "te", "mr", "gu", "kn", "ml", "pa"]
        self.log_level = log_level
        self.enable_metrics = enable_metrics
        
    def to_dict(self) -> Dict:
        """Convert config to dictionary"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
        
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'DeploymentConfig':
        """Create config from dictionary"""
        return cls(**config_dict)
        
    def save(self, path: str):
        """Save config to file"""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


class BharatAPI:
    """FastAPI-based API for BharatFM models"""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = None
        self.app = None
        self.logger = None
        
        # Setup logging
        self.setup_logging()
        
        # Setup device
        self.setup_device()
        
        # Initialize FastAPI app
        self.setup_app()
        
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def setup_device(self):
        """Setup device for inference"""
        if self.config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config.device)
            
        self.logger.info(f"Using device: {self.device}")
        
    def setup_app(self):
        """Setup FastAPI application"""
        
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # Startup
            self.load_model()
            yield
            # Shutdown
            self.cleanup()
            
        self.app = FastAPI(
            title="BharatFM API",
            description="API for Bharat Foundation Model Framework",
            version="1.0.0",
            lifespan=lifespan
        )
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Setup routes
        self.setup_routes()
        
    def setup_routes(self):
        """Setup API routes"""
        
        @self.app.get("/", response_model=Dict[str, str])
        async def root():
            """Root endpoint"""
            return {
                "message": "BharatFM API",
                "version": "1.0.0",
                "docs": "/docs"
            }
            
        @self.app.get("/health", response_model=HealthResponse)
        async def health_check():
            """Health check endpoint"""
            return HealthResponse(
                status="healthy",
                timestamp=datetime.now().isoformat(),
                model_loaded=self.model is not None,
                device=str(self.device)
            )
            
        @self.app.get("/model/info", response_model=ModelInfo)
        async def model_info():
            """Get model information"""
            if self.model is None:
                raise HTTPException(status_code=503, detail="Model not loaded")
                
            return ModelInfo(
                model_name=self.config.model_path,
                model_type=self.config.model_type,
                model_size="1.3B",  # This should be determined from model config
                supported_languages=self.config.supported_languages,
                max_context_length=2048,  # This should be determined from model config
                version="1.0.0"
            )
            
        @self.app.post("/generate", response_model=GenerationResponse)
        async def generate_text(request: GenerationRequest):
            """Generate text from prompt"""
            if self.model is None:
                raise HTTPException(status_code=503, detail="Model not loaded")
                
            try:
                start_time = datetime.now()
                
                # Generate text
                generated_text, tokens_generated = self._generate_text(
                    prompt=request.prompt,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    top_k=request.top_k,
                    num_beams=request.num_beams,
                    do_sample=request.do_sample,
                    language=request.language
                )
                
                end_time = datetime.now()
                generation_time = (end_time - start_time).total_seconds()
                
                # Detect language (simplified)
                language_detected = self._detect_language(generated_text)
                
                return GenerationResponse(
                    generated_text=generated_text,
                    prompt=request.prompt,
                    tokens_generated=tokens_generated,
                    generation_time=generation_time,
                    language_detected=language_detected
                )
                
            except Exception as e:
                self.logger.error(f"Generation error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
                
        @self.app.post("/batch_generate")
        async def batch_generate(requests: List[GenerationRequest]):
            """Generate text for multiple prompts"""
            if self.model is None:
                raise HTTPException(status_code=503, detail="Model not loaded")
                
            try:
                responses = []
                
                for request in requests:
                    generated_text, tokens_generated = self._generate_text(
                        prompt=request.prompt,
                        max_tokens=request.max_tokens,
                        temperature=request.temperature,
                        top_p=request.top_p,
                        top_k=request.top_k,
                        num_beams=request.num_beams,
                        do_sample=request.do_sample,
                        language=request.language
                    )
                    
                    responses.append({
                        "generated_text": generated_text,
                        "prompt": request.prompt,
                        "tokens_generated": tokens_generated
                    })
                    
                return {"responses": responses}
                
            except Exception as e:
                self.logger.error(f"Batch generation error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
                
        @self.app.post("/embeddings")
        async def get_embeddings(text: str):
            """Get text embeddings"""
            if self.model is None:
                raise HTTPException(status_code=503, detail="Model not loaded")
                
            try:
                # Get embeddings from model
                embeddings = self._get_embeddings(text)
                
                return {
                    "embeddings": embeddings.tolist(),
                    "text": text,
                    "embedding_dim": len(embeddings)
                }
                
            except Exception as e:
                self.logger.error(f"Embedding error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
                
        @self.app.get("/languages")
        async def get_supported_languages():
            """Get list of supported languages"""
            return {
                "supported_languages": self.config.supported_languages,
                "multilingual_enabled": self.config.enable_multilingual
            }
            
    def load_model(self):
        """Load model and tokenizer"""
        self.logger.info(f"Loading model from {self.config.model_path}")
        
        try:
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
                
            self.logger.info("Model and tokenizer loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise
            
    def _generate_text(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 50,
        num_beams: int = 1,
        do_sample: bool = True,
        language: str = None
    ) -> tuple[str, int]:
        """Generate text from prompt"""
        
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
        
        return generated_text, tokens_generated
        
    def _get_embeddings(self, text: str) -> torch.Tensor:
        """Get text embeddings"""
        # Tokenize
        inputs = self.tokenizer.tokenize(text)
        input_ids = inputs["input_ids"].to(self.device)
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.model.model(input_ids=input_ids)
            embeddings = outputs["last_hidden_state"]
            
        # Pool embeddings (mean pooling)
        embeddings = embeddings.mean(dim=1)
        
        return embeddings.squeeze()
        
    def _detect_language(self, text: str) -> Optional[str]:
        """Detect language of text (simplified)"""
        # Simple language detection based on script
        if any(char in text for char in ["ँ", "ं", "ः", "अ", "आ", "इ", "ई"]):
            return "hi"
        elif any(char in text for char in ["ঁ", "ং", "ঃ", "অ", "আ", "ই", "ঈ"]):
            return "bn"
        elif any(char in text for char in ["ஂ", "ஃ", "அ", "ஆ", "இ", "ஈ"]):
            return "ta"
        elif any(char in text for char in ["ఁ", "ం", "ః", "అ", "ఆ", "ఇ", "ఈ"]):
            return "te"
        elif text.isascii():
            return "en"
        else:
            return None
            
    def cleanup(self):
        """Cleanup resources"""
        if self.model is not None:
            del self.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        self.logger.info("Cleanup complete")
        
    def run(self):
        """Run the API server"""
        self.logger.info(f"Starting BharatFM API server on {self.config.host}:{self.config.port}")
        
        uvicorn.run(
            self.app,
            host=self.config.host,
            port=self.config.port,
            workers=self.config.workers,
            log_level=self.config.log_level.lower()
        )
        
    def get_app(self) -> FastAPI:
        """Get FastAPI app instance"""
        return self.app