"""
Bharat Training Module
======================

Module for distributed pretraining and fine-tuning pipeline
with support for Deepspeed, Axolotl, and FSDP.
"""

from .trainer import BharatTrainer, TrainingConfig
from .finetune import FineTuner, LoRAConfig
from .deepspeed_config import get_deepspeed_config

__version__ = "0.1.0"
__all__ = [
    "BharatTrainer",
    "TrainingConfig",
    "FineTuner", 
    "LoRAConfig",
    "get_deepspeed_config"
]