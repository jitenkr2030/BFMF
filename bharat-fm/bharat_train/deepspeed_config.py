"""
DeepSpeed configuration utilities for BharatFM
"""

import json
from typing import Dict, Optional, Any
from pathlib import Path


def get_deepspeed_config(
    stage: int = 2,
    batch_size: int = 32,
    micro_batch_size: int = 4,
    fp16: bool = True,
    bf16: bool = False,
    offload_optimizer: bool = True,
    offload_param: bool = False,
    zero_init: bool = True,
) -> Dict[str, Any]:
    """
    Get DeepSpeed configuration for BharatFM training
    
    Args:
        stage: ZeRO optimization stage (0, 1, 2, or 3)
        batch_size: Total batch size
        micro_batch_size: Micro batch size per GPU
        fp16: Enable FP16 mixed precision
        bf16: Enable BF16 mixed precision
        offload_optimizer: Offload optimizer states to CPU
        offload_param: Offload parameters to CPU
        zero_init: Initialize ZeRO states
        
    Returns:
        DeepSpeed configuration dictionary
    """
    
    # Calculate gradient accumulation steps
    gradient_accumulation_steps = batch_size // micro_batch_size
    
    config = {
        "train_batch_size": batch_size,
        "train_micro_batch_size_per_gpu": micro_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "steps_per_print": 100,
        "wall_clock_breakdown": False,
    }
    
    # Mixed precision configuration
    if fp16:
        config["fp16"] = {
            "enabled": True,
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "initial_scale_power": 16,
            "hysteresis": 2,
            "min_loss_scale": 1
        }
    elif bf16:
        config["bf16"] = {
            "enabled": True
        }
        
    # Optimizer configuration
    config["optimizer"] = {
        "type": "AdamW",
        "params": {
            "lr": 2e-5,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01
        }
    }
    
    # Scheduler configuration
    config["scheduler"] = {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": 0,
            "warmup_max_lr": 2e-5,
            "warmup_num_steps": 1000
        }
    }
    
    # ZeRO optimization configuration
    if stage > 0:
        zero_config = {
            "stage": stage,
            "contiguous_gradients": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 2e8,
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
            "overlap_comm": True,
        }
        
        if stage >= 2:
            zero_config["offload_optimizer"] = {
                "device": "cpu",
                "pin_memory": offload_optimizer
            }
            
        if stage >= 3:
            zero_config["offload_param"] = {
                "device": "cpu",
                "pin_memory": offload_param
            }
            zero_config["zero_init"] = zero_init
            
        config["zero_optimization"] = zero_config
        
    # Gradient clipping
    config["gradient_clipping"] = 1.0
    
    return config


def get_deepspeed_config_from_file(config_path: str) -> Dict[str, Any]:
    """
    Load DeepSpeed configuration from file
    
    Args:
        config_path: Path to DeepSpeed configuration file
        
    Returns:
        DeepSpeed configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def save_deepspeed_config(config: Dict[str, Any], save_path: str):
    """
    Save DeepSpeed configuration to file
    
    Args:
        config: DeepSpeed configuration dictionary
        save_path: Path to save configuration file
    """
    with open(save_path, 'w') as f:
        json.dump(config, f, indent=2)


def get_deepspeed_config_for_model_size(
    model_size: str,
    num_gpus: int = 1,
) -> Dict[str, Any]:
    """
    Get recommended DeepSpeed configuration based on model size
    
    Args:
        model_size: Model size ("small", "medium", "large", "xlarge")
        num_gpus: Number of available GPUs
        
    Returns:
        DeepSpeed configuration dictionary
    """
    
    # Model size configurations
    model_configs = {
        "small": {
            "batch_size": 32,
            "micro_batch_size": 8,
            "stage": 1,
            "offload_optimizer": False,
            "offload_param": False,
        },
        "medium": {
            "batch_size": 64,
            "micro_batch_size": 4,
            "stage": 2,
            "offload_optimizer": True,
            "offload_param": False,
        },
        "large": {
            "batch_size": 128,
            "micro_batch_size": 2,
            "stage": 2,
            "offload_optimizer": True,
            "offload_param": False,
        },
        "xlarge": {
            "batch_size": 256,
            "micro_batch_size": 1,
            "stage": 3,
            "offload_optimizer": True,
            "offload_param": True,
        }
    }
    
    if model_size not in model_configs:
        raise ValueError(f"Unknown model size: {model_size}. Available: {list(model_configs.keys())}")
        
    config = model_configs[model_size]
    
    # Adjust batch size based on number of GPUs
    config["batch_size"] = config["batch_size"] * num_gpus
    
    return get_deepspeed_config(
        stage=config["stage"],
        batch_size=config["batch_size"],
        micro_batch_size=config["micro_batch_size"],
        offload_optimizer=config["offload_optimizer"],
        offload_param=config["offload_param"],
    )


def get_memory_optimized_config(
    available_memory_gb: float,
    model_size_gb: float,
) -> Dict[str, Any]:
    """
    Get memory-optimized DeepSpeed configuration
    
    Args:
        available_memory_gb: Available GPU memory in GB
        model_size_gb: Model size in GB
        
    Returns:
        DeepSpeed configuration dictionary
    """
    
    # Calculate memory ratio
    memory_ratio = available_memory_gb / model_size_gb
    
    if memory_ratio >= 2.0:
        # Plenty of memory, use stage 1
        stage = 1
        offload_optimizer = False
        offload_param = False
    elif memory_ratio >= 1.5:
        # Moderate memory, use stage 2 with optimizer offload
        stage = 2
        offload_optimizer = True
        offload_param = False
    elif memory_ratio >= 1.0:
        # Limited memory, use stage 2 with full offload
        stage = 2
        offload_optimizer = True
        offload_param = True
    else:
        # Very limited memory, use stage 3
        stage = 3
        offload_optimizer = True
        offload_param = True
        
    return get_deepspeed_config(
        stage=stage,
        offload_optimizer=offload_optimizer,
        offload_param=offload_param,
    )


def validate_deepspeed_config(config: Dict[str, Any]) -> bool:
    """
    Validate DeepSpeed configuration
    
    Args:
        config: DeepSpeed configuration dictionary
        
    Returns:
        True if configuration is valid, False otherwise
    """
    
    required_keys = [
        "train_batch_size",
        "train_micro_batch_size_per_gpu",
        "gradient_accumulation_steps",
    ]
    
    # Check required keys
    for key in required_keys:
        if key not in config:
            return False
            
    # Validate batch size calculation
    batch_size = config["train_batch_size"]
    micro_batch_size = config["train_micro_batch_size_per_gpu"]
    gradient_accumulation_steps = config["gradient_accumulation_steps"]
    
    if batch_size != micro_batch_size * gradient_accumulation_steps:
        return False
        
    # Validate mixed precision
    if "fp16" in config and "bf16" in config:
        if config["fp16"]["enabled"] and config["bf16"]["enabled"]:
            return False
            
    # Validate ZeRO configuration
    if "zero_optimization" in config:
        stage = config["zero_optimization"]["stage"]
        if stage not in [0, 1, 2, 3]:
            return False
            
        if stage >= 2 and "offload_optimizer" not in config["zero_optimization"]:
            return False
            
        if stage >= 3 and "offload_param" not in config["zero_optimization"]:
            return False
            
    return True


def print_deepspeed_config_summary(config: Dict[str, Any]):
    """
    Print summary of DeepSpeed configuration
    
    Args:
        config: DeepSpeed configuration dictionary
    """
    
    print("=== DeepSpeed Configuration Summary ===")
    print(f"Batch Size: {config['train_batch_size']}")
    print(f"Micro Batch Size: {config['train_micro_batch_size_per_gpu']}")
    print(f"Gradient Accumulation Steps: {config['gradient_accumulation_steps']}")
    
    if "fp16" in config and config["fp16"]["enabled"]:
        print("Mixed Precision: FP16")
    elif "bf16" in config and config["bf16"]["enabled"]:
        print("Mixed Precision: BF16")
    else:
        print("Mixed Precision: None")
        
    if "zero_optimization" in config:
        stage = config["zero_optimization"]["stage"]
        print(f"ZeRO Stage: {stage}")
        
        if stage >= 2:
            offload_opt = config["zero_optimization"]["offload_optimizer"]["device"]
            print(f"Optimizer Offload: {offload_opt}")
            
        if stage >= 3:
            offload_param = config["zero_optimization"]["offload_param"]["device"]
            print(f"Parameter Offload: {offload_param}")
            
    print("=======================================")