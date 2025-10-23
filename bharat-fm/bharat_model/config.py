"""
Configuration management for BharatFM models
"""

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union
from pathlib import Path


@dataclass
class ModelConfig:
    """Base configuration class for BharatFM models"""
    
    # Model architecture
    model_type: str = "glm"
    vocab_size: int = 50000
    hidden_size: int = 4096
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    intermediate_size: int = 16384
    
    # Indian language support
    supported_languages: List[str] = field(default_factory=lambda: ["hi", "en", "bn", "ta", "te", "mr", "gu", "kn", "ml", "pa"])
    primary_language: str = "hi"
    
    # Training configuration
    max_position_embeddings: int = 2048
    layer_norm_eps: float = 1e-5
    hidden_act: str = "silu"
    attention_dropout: float = 0.1
    hidden_dropout: float = 0.1
    initializer_range: float = 0.02
    
    # Optimization
    use_cache: bool = True
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    tie_word_embeddings: bool = False
    
    # Bharat-specific settings
    enable_indic_attention: bool = True
    multilingual_head: bool = True
    domain_adapters: List[str] = field(default_factory=lambda: ["general", "gov", "edu", "tech"])
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary"""
        return {
            k: v for k, v in self.__dict__.items() 
            if not k.startswith('_')
        }
        
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'ModelConfig':
        """Create config from dictionary"""
        return cls(**config_dict)
        
    def save_pretrained(self, save_directory: str):
        """Save config to directory"""
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)
        
        config_path = save_path / "config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2)
            
    @classmethod
    def load_pretrained(cls, load_directory: str) -> 'ModelConfig':
        """Load config from directory"""
        load_path = Path(load_directory)
        config_path = load_path / "config.json"
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
            
        return cls.from_dict(config_dict)


@dataclass
class GLMConfig(ModelConfig):
    """Configuration for GLM (General Language Model) architecture"""
    
    model_type: str = "glm"
    
    # GLM-specific parameters
    use_memory_attention: bool = True
    memory_length: int = 512
    position_encoding_2d: bool = True
    
    # Multi-query attention
    multi_query_attention: bool = True
    num_key_value_heads: int = 8
    
    # Rotary embeddings
    use_rotary_embeddings: bool = True
    rotary_dim: int = 64
    rope_scaling: Optional[Dict] = None


@dataclass
class LlamaConfig(ModelConfig):
    """Configuration for LLaMA architecture"""
    
    model_type: str = "llama"
    
    # LLaMA-specific parameters
    rms_norm_eps: float = 1e-6
    pretraining_tp: int = 1
    use_flash_attention: bool = True
    
    # Rotary embeddings
    rope_theta: float = 10000.0
    rope_scaling: Optional[Dict] = None
    
    # SwiGLU activation
    use_swiglu: bool = True
    swiglu_multiple: int = 2


@dataclass
class MoEConfig(ModelConfig):
    """Configuration for Mixture of Experts architecture"""
    
    model_type: str = "moe"
    
    # MoE-specific parameters
    num_experts: int = 8
    num_experts_per_token: int = 2
    expert_capacity: int = 128
    expert_output_size: Optional[int] = None
    
    # Routing
    router_aux_loss_coef: float = 0.01
    router_z_loss_coef: float = 0.001
    router_dtype: str = "float32"
    
    # Load balancing
    use_load_balancing: bool = True
    load_balancing_loss_coef: float = 0.01
    
    # Expert configuration
    expert_intermediate_size: int = 4096
    expert_hidden_size: Optional[int] = None


def get_model_config(model_name: str) -> ModelConfig:
    """
    Get predefined configuration for BharatFM models
    
    Args:
        model_name: Name of the model (e.g., "bharat-base", "bharat-lite", "bharat-moe")
    
    Returns:
        ModelConfig instance
    """
    configs = {
        "bharat-base": GLMConfig(
            hidden_size=4096,
            num_hidden_layers=32,
            num_attention_heads=32,
            intermediate_size=16384,
            vocab_size=50000,
            max_position_embeddings=2048,
        ),
        "bharat-lite": GLMConfig(
            hidden_size=2048,
            num_hidden_layers=24,
            num_attention_heads=16,
            intermediate_size=8192,
            vocab_size=32000,
            max_position_embeddings=2048,
        ),
        "bharat-moe": MoEConfig(
            hidden_size=4096,
            num_hidden_layers=32,
            num_attention_heads=32,
            intermediate_size=16384,
            vocab_size=50000,
            max_position_embeddings=2048,
            num_experts=8,
            num_experts_per_token=2,
        ),
        "bharat-gov": GLMConfig(
            hidden_size=4096,
            num_hidden_layers=32,
            num_attention_heads=32,
            intermediate_size=16384,
            vocab_size=50000,
            max_position_embeddings=2048,
            domain_adapters=["general", "gov", "legal", "policy"],
        ),
        "bharat-edu": GLMConfig(
            hidden_size=4096,
            num_hidden_layers=32,
            num_attention_heads=32,
            intermediate_size=16384,
            vocab_size=50000,
            max_position_embeddings=2048,
            domain_adapters=["general", "edu", "science", "mathematics"],
        ),
        "bharat-lang": GLMConfig(
            hidden_size=4096,
            num_hidden_layers=32,
            num_attention_heads=32,
            intermediate_size=16384,
            vocab_size=50000,
            max_position_embeddings=2048,
            multilingual_head=True,
            enable_indic_attention=True,
        ),
    }
    
    if model_name not in configs:
        raise ValueError(f"Unknown model name: {model_name}. Available: {list(configs.keys())}")
        
    return configs[model_name]


def create_config_from_args(args: Dict) -> ModelConfig:
    """
    Create model configuration from command line arguments
    
    Args:
        args: Dictionary of configuration arguments
    
    Returns:
        ModelConfig instance
    """
    model_type = args.get("model_type", "glm")
    
    if model_type == "glm":
        config = GLMConfig()
    elif model_type == "llama":
        config = LlamaConfig()
    elif model_type == "moe":
        config = MoEConfig()
    else:
        config = ModelConfig()
        
    # Override with provided arguments
    for key, value in args.items():
        if hasattr(config, key):
            setattr(config, key, value)
            
    return config