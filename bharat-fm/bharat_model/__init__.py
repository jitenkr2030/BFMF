"""
Bharat Model Module
===================

Module for defining base model architectures including decoder-only, 
encoder-decoder, and mixture-of-experts models specifically designed 
for Indian languages and regional diversity.
"""

from .modeling_glm import GLMForCausalLM, GLMConfig
from .modeling_llama import LlamaForCausalLM, LlamaConfig
from .modeling_moe import MoEForCausalLM, MoEConfig
from .config import ModelConfig, get_model_config

__version__ = "0.1.0"
__all__ = [
    "GLMForCausalLM",
    "GLMConfig", 
    "LlamaForCausalLM",
    "LlamaConfig",
    "MoEForCausalLM",
    "MoEConfig",
    "ModelConfig",
    "get_model_config"
]