"""
Comprehensive Training Configuration System
Unified configuration management for all training components
"""

import os
import json
import yaml
import argparse
from typing import Dict, List, Any, Optional, Union, Type
from dataclasses import dataclass, field, asdict
from pathlib import Path
import logging
from datetime import datetime
import torch

# Import component configs
from .real_training_system import TrainingConfig, TrainingMetrics
from .distributed_training import DistributedConfig
from .model_parallelism import ModelParallelConfig
from .cluster_management import ClusterConfig
from .mixed_precision import PrecisionConfig
from .advanced_schedulers import SchedulerConfig

logger = logging.getLogger(__name__)

@dataclass
class TrainingEnvironment:
    """Training environment configuration"""
    # Environment settings
    environment: str = "development"  # development, staging, production
    seed: int = 42
    deterministic: bool = True
    cudnn_benchmark: bool = False
    
    # Device settings
    device: str = "auto"  # auto, cpu, cuda
    num_gpus: int = 1
    gpu_ids: List[int] = None
    
    # Parallelism settings
    distributed: bool = False
    data_parallel: bool = False
    model_parallel: bool = False
    pipeline_parallel: bool = False
    
    # Memory settings
    pin_memory: bool = True
    non_blocking: bool = True
    
    # Debug settings
    debug: bool = False
    profile: bool = False
    verbose: bool = False
    
    def __post_init__(self):
        if self.gpu_ids is None:
            self.gpu_ids = list(range(self.num_gpus))

@dataclass
class DataConfig:
    """Data configuration"""
    # Dataset settings
    dataset_name: str = "custom"
    dataset_path: str = "./data"
    train_file: str = "train.txt"
    val_file: str = "val.txt"
    test_file: str = "test.txt"
    
    # Data loading
    batch_size: int = 32
    eval_batch_size: int = 64
    num_workers: int = 4
    prefetch_factor: int = 2
    persistent_workers: bool = True
    
    # Data preprocessing
    max_length: int = 512
    truncation: bool = True
    padding: str = "longest"  # longest, max_length
    
    # Data augmentation
    augment: bool = False
    augmentation_prob: float = 0.1
    
    # Data splitting
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    
    # Caching
    cache_dir: str = "./cache"
    use_cache: bool = True

@dataclass
class ModelConfig:
    """Model configuration"""
    # Model architecture
    model_type: str = "transformer"  # transformer, llama, bert, gpt
    model_name: str = "custom"
    
    # Architecture parameters
    vocab_size: int = 50000
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    intermediate_size: int = 3072
    max_position_embeddings: int = 1024
    
    # Model variants
    model_variant: str = "base"  # base, large, xl
    
    # Pretrained settings
    pretrained_model: str = None
    pretrained_path: str = None
    load_pretrained: bool = False
    
    # Model initialization
    init_method: str = "normal"  # normal, xavier, kaiming
    init_std: float = 0.02
    
    # Model optimizations
    gradient_checkpointing: bool = True
    activation_checkpointing: bool = True
    use_flash_attention: bool = True
    
    # Model output
    output_dir: str = "./outputs"
    save_model: bool = True
    save_best_only: bool = True
    save_interval: int = 1000

@dataclass
class OptimizationConfig:
    """Optimization configuration"""
    # Optimizer
    optimizer: str = "adamw"  # adamw, adam, sgd, rmsprop
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    betas: List[float] = field(default_factory=lambda: [0.9, 0.999])
    eps: float = 1e-8
    
    # Learning rate scheduling
    scheduler: str = "cosine"  # cosine, onecycle, warmupcosine, linear
    warmup_steps: int = 1000
    total_steps: int = 100000
    min_lr: float = 1e-6
    
    # Gradient handling
    max_grad_norm: float = 1.0
    grad_clip: bool = True
    
    # Gradient accumulation
    accumulation_steps: int = 1
    sync_frequency: int = 1
    
    # Mixed precision
    fp16: bool = True
    bf16: bool = False
    fp8: bool = False
    loss_scale: float = 0.0
    
    # Optimization techniques
    use_ema: bool = False
    ema_decay: float = 0.999
    use_lookahead: bool = False
    lookahead_k: int = 5
    lookahead_alpha: float = 0.5

@dataclass
class CheckpointingConfig:
    """Checkpointing configuration"""
    # Checkpoint settings
    save_dir: str = "./checkpoints"
    save_interval: int = 1000
    save_total_limit: int = 5
    save_best_only: bool = True
    
    # Checkpoint format
    save_format: str = "pt"  # pt, safetensors, both
    save_optimizer: bool = True
    save_scheduler: bool = True
    save_rng: bool = True
    
    # Resume training
    resume_from_checkpoint: str = None
    resume_optimizer: bool = True
    resume_scheduler: bool = True
    
    # Checkpoint validation
    validate_checkpoint: bool = True
    checkpoint_timeout: int = 300

@dataclass
class LoggingConfig:
    """Logging configuration"""
    # Logging settings
    log_level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR
    log_dir: str = "./logs"
    log_file: str = "training.log"
    
    # Console logging
    console_log: bool = True
    console_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # File logging
    file_log: bool = True
    file_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
    
    # TensorBoard
    tensorboard_dir: str = "./tensorboard"
    tensorboard_log: bool = True
    tensorboard_interval: int = 10
    
    # WandB
    wandb_project: str = "bharat-fm"
    wandb_entity: str = None
    wandb_log: bool = False
    wandb_interval: int = 10
    
    # Metrics logging
    log_interval: int = 10
    eval_interval: int = 100
    save_interval: int = 1000
    
    # Progress bar
    progress_bar: bool = True
    progress_bar_format: str = "{l_bar}{bar:10}{r_bar}{bar:-10b}"

@dataclass
class EvaluationConfig:
    """Evaluation configuration"""
    # Evaluation settings
    eval_strategy: str = "steps"  # steps, epoch, both
    eval_steps: int = 100
    eval_epochs: int = 1
    
    # Metrics
    metrics: List[str] = field(default_factory=lambda: ["loss", "accuracy"])
    compute_metrics: bool = True
    
    # Evaluation dataset
    eval_batch_size: int = 64
    eval_num_workers: int = 4
    
    # Evaluation behavior
    do_eval: bool = True
    do_predict: bool = True
    
    # Evaluation optimization
    eval_accumulation_steps: int = 1
    eval_delay: float = 0.0
    
    # Save evaluation results
    save_eval_results: bool = True
    eval_results_dir: str = "./eval_results"

@dataclass
class BharatFMTrainingConfig:
    """Complete training configuration for Bharat-FM"""
    # Core configurations
    environment: TrainingEnvironment = field(default_factory=TrainingEnvironment)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    checkpointing: CheckpointingConfig = field(default_factory=CheckpointingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    
    # Advanced configurations
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    model_parallel: ModelParallelConfig = field(default_factory=ModelParallelConfig)
    cluster: ClusterConfig = field(default_factory=ClusterConfig)
    precision: PrecisionConfig = field(default_factory=PrecisionConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    
    # Metadata
    config_version: str = "1.0.0"
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def __post_init__(self):
        """Post-initialization setup"""
        # Update scheduler config with optimization config
        self.scheduler.max_lr = self.optimization.learning_rate
        self.scheduler.min_lr = self.optimization.min_lr
        self.scheduler.total_steps = self.optimization.total_steps
        self.scheduler.warmup_steps = self.optimization.warmup_steps
        
        # Update precision config
        if self.optimization.fp16:
            self.precision.precision = "fp16"
        elif self.optimization.bf16:
            self.precision.precision = "bf16"
        elif self.optimization.fp8:
            self.precision.precision = "fp8"
        else:
            self.precision.precision = "fp32"
        
        # Update distributed config
        self.distributed.mixed_precision = self.precision.precision
        
        # Update timestamp
        self.updated_at = datetime.utcnow().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return asdict(self)
    
    def save(self, path: str):
        """Save configuration to file"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if path.suffix.lower() == '.json':
            with open(path, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
        elif path.suffix.lower() in ['.yaml', '.yml']:
            with open(path, 'w') as f:
                yaml.dump(self.to_dict(), f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
        
        logger.info(f"Configuration saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'BharatFMTrainingConfig':
        """Load configuration from file"""
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        if path.suffix.lower() == '.json':
            with open(path, 'r') as f:
                config_dict = json.load(f)
        elif path.suffix.lower() in ['.yaml', '.yml']:
            with open(path, 'r') as f:
                config_dict = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
        
        # Create configuration object
        config = cls.from_dict(config_dict)
        
        logger.info(f"Configuration loaded from {path}")
        return config
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'BharatFMTrainingConfig':
        """Create configuration from dictionary"""
        # Extract nested configurations
        nested_configs = {}
        for key in ['environment', 'data', 'model', 'optimization', 'checkpointing', 
                    'logging', 'evaluation', 'distributed', 'model_parallel', 
                    'cluster', 'precision', 'scheduler']:
            if key in config_dict:
                nested_configs[key] = config_dict.pop(key)
        
        # Create main configuration
        config = cls(**config_dict)
        
        # Update nested configurations
        for key, value in nested_configs.items():
            if hasattr(config, key):
                if key == 'environment':
                    config.environment = TrainingEnvironment(**value)
                elif key == 'data':
                    config.data = DataConfig(**value)
                elif key == 'model':
                    config.model = ModelConfig(**value)
                elif key == 'optimization':
                    config.optimization = OptimizationConfig(**value)
                elif key == 'checkpointing':
                    config.checkpointing = CheckpointingConfig(**value)
                elif key == 'logging':
                    config.logging = LoggingConfig(**value)
                elif key == 'evaluation':
                    config.evaluation = EvaluationConfig(**value)
                elif key == 'distributed':
                    config.distributed = DistributedConfig(**value)
                elif key == 'model_parallel':
                    config.model_parallel = ModelParallelConfig(**value)
                elif key == 'cluster':
                    config.cluster = ClusterConfig(**value)
                elif key == 'precision':
                    config.precision = PrecisionConfig(**value)
                elif key == 'scheduler':
                    config.scheduler = SchedulerConfig(**value)
        
        return config
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of errors"""
        errors = []
        
        # Validate environment
        if self.environment.num_gpus <= 0:
            errors.append("Number of GPUs must be positive")
        
        if self.environment.device == "cuda" and not torch.cuda.is_available():
            errors.append("CUDA device requested but CUDA is not available")
        
        # Validate data
        if self.data.batch_size <= 0:
            errors.append("Batch size must be positive")
        
        if self.data.train_split + self.data.val_split + self.data.test_split != 1.0:
            errors.append("Data splits must sum to 1.0")
        
        # Validate model
        if self.model.vocab_size <= 0:
            errors.append("Vocabulary size must be positive")
        
        if self.model.hidden_size <= 0:
            errors.append("Hidden size must be positive")
        
        # Validate optimization
        if self.optimization.learning_rate <= 0:
            errors.append("Learning rate must be positive")
        
        if self.optimization.weight_decay < 0:
            errors.append("Weight decay must be non-negative")
        
        # Validate checkpointing
        if self.checkpointing.save_interval <= 0:
            errors.append("Save interval must be positive")
        
        # Validate logging
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR"]
        if self.logging.log_level not in valid_log_levels:
            errors.append(f"Log level must be one of {valid_log_levels}")
        
        return errors
    
    def setup_environment(self):
        """Setup training environment based on configuration"""
        # Set random seed
        if self.environment.seed is not None:
            torch.manual_seed(self.environment.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.environment.seed)
            import numpy as np
            np.random.seed(self.environment.seed)
            import random
            random.seed(self.environment.seed)
        
        # Set deterministic behavior
        if self.environment.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        elif self.environment.cudnn_benchmark:
            torch.backends.cudnn.benchmark = True
        
        # Setup device
        if self.environment.device == "auto":
            self.environment.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Setup logging
        self._setup_logging()
        
        logger.info("Training environment setup complete")
    
    def _setup_logging(self):
        """Setup logging based on configuration"""
        import logging
        
        # Set log level
        log_level = getattr(logging, self.logging.log_level.upper())
        logging.basicConfig(level=log_level)
        
        # Create logger
        logger = logging.getLogger(__name__)
        
        # Setup console handler
        if self.logging.console_log:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(log_level)
            console_formatter = logging.Formatter(self.logging.console_format)
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
        
        # Setup file handler
        if self.logging.file_log:
            log_file = Path(self.logging.log_dir) / self.logging.log_file
            log_file.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(log_level)
            file_formatter = logging.Formatter(self.logging.file_format)
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
        
        logger.info("Logging setup complete")
    
    def get_effective_batch_size(self) -> int:
        """Get effective batch size considering gradient accumulation"""
        return (self.data.batch_size * 
                self.optimization.accumulation_steps * 
                (self.distributed.world_size if hasattr(self.distributed, 'world_size') else 1))
    
    def get_total_training_steps(self, num_epochs: int = None) -> int:
        """Get total training steps"""
        if num_epochs is None:
            return self.optimization.total_steps
        else:
            # Estimate steps per epoch
            # This is approximate - actual steps depend on dataset size
            estimated_steps_per_epoch = 1000  # Placeholder
            return num_epochs * estimated_steps_per_epoch


class ConfigManager:
    """Configuration manager for training"""
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path
        self.config = None
        
        if config_path:
            self.load_config(config_path)
    
    def load_config(self, config_path: str):
        """Load configuration from file"""
        self.config_path = config_path
        self.config = BharatFMTrainingConfig.load(config_path)
        
        # Validate configuration
        errors = self.config.validate()
        if errors:
            logger.error(f"Configuration validation failed: {errors}")
            raise ValueError(f"Invalid configuration: {errors}")
        
        # Setup environment
        self.config.setup_environment()
    
    def save_config(self, config_path: str = None):
        """Save configuration to file"""
        if self.config is None:
            raise ValueError("No configuration to save")
        
        save_path = config_path or self.config_path
        if save_path is None:
            raise ValueError("No config path specified")
        
        self.config.save(save_path)
    
    def get_config(self) -> BharatFMTrainingConfig:
        """Get current configuration"""
        if self.config is None:
            raise ValueError("No configuration loaded")
        return self.config
    
    def update_config(self, updates: Dict[str, Any]):
        """Update configuration with new values"""
        if self.config is None:
            raise ValueError("No configuration loaded")
        
        # Convert to dict, update, and recreate
        config_dict = self.config.to_dict()
        self._deep_update(config_dict, updates)
        self.config = BharatFMTrainingConfig.from_dict(config_dict)
        
        # Re-validate
        errors = self.config.validate()
        if errors:
            logger.error(f"Configuration validation failed after update: {errors}")
            raise ValueError(f"Invalid configuration after update: {errors}")
    
    def _deep_update(self, base_dict: Dict, update_dict: Dict):
        """Deep update dictionary"""
        for key, value in update_dict.items():
            if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def create_arg_parser(self) -> argparse.ArgumentParser:
        """Create argument parser for configuration"""
        parser = argparse.ArgumentParser(description="Bharat-FM Training")
        
        # Configuration file
        parser.add_argument("--config", type=str, help="Path to configuration file")
        
        # Override common parameters
        parser.add_argument("--model_name", type=str, help="Model name")
        parser.add_argument("--batch_size", type=int, help="Batch size")
        parser.add_argument("--learning_rate", type=float, help="Learning rate")
        parser.add_argument("--num_epochs", type=int, help="Number of epochs")
        parser.add_argument("--seed", type=int, help="Random seed")
        parser.add_argument("--device", type=str, help="Device to use")
        parser.add_argument("--output_dir", type=str, help="Output directory")
        
        # Training modes
        parser.add_argument("--mode", type=str, choices=["train", "eval", "predict"], 
                           default="train", help="Training mode")
        parser.add_argument("--resume", type=str, help="Resume from checkpoint")
        
        # Debug options
        parser.add_argument("--debug", action="store_true", help="Enable debug mode")
        parser.add_argument("--profile", action="store_true", help="Enable profiling")
        parser.add_argument("--verbose", action="store_true", help="Verbose output")
        
        return parser
    
    def parse_args_and_update_config(self, args: List[str] = None):
        """Parse command line arguments and update configuration"""
        parser = self.create_arg_parser()
        parsed_args = parser.parse_args(args)
        
        if parsed_args.config:
            self.load_config(parsed_args.config)
        
        # Update configuration with command line arguments
        updates = {}
        
        if parsed_args.model_name:
            updates.setdefault("model", {})["model_name"] = parsed_args.model_name
        
        if parsed_args.batch_size:
            updates.setdefault("data", {})["batch_size"] = parsed_args.batch_size
        
        if parsed_args.learning_rate:
            updates.setdefault("optimization", {})["learning_rate"] = parsed_args.learning_rate
        
        if parsed_args.seed:
            updates.setdefault("environment", {})["seed"] = parsed_args.seed
        
        if parsed_args.device:
            updates.setdefault("environment", {})["device"] = parsed_args.device
        
        if parsed_args.output_dir:
            updates.setdefault("model", {})["output_dir"] = parsed_args.output_dir
        
        if parsed_args.debug:
            updates.setdefault("environment", {})["debug"] = True
        
        if parsed_args.profile:
            updates.setdefault("environment", {})["profile"] = True
        
        if parsed_args.verbose:
            updates.setdefault("environment", {})["verbose"] = True
        
        if updates:
            self.update_config(updates)
        
        return parsed_args


def create_default_config() -> BharatFMTrainingConfig:
    """Create default training configuration"""
    return BharatFMTrainingConfig()


def create_config_from_template(template_name: str) -> BharatFMTrainingConfig:
    """Create configuration from template"""
    templates = {
        "small": {
            "model": {
                "hidden_size": 512,
                "num_layers": 6,
                "num_heads": 8,
                "intermediate_size": 2048
            },
            "data": {
                "batch_size": 16
            },
            "optimization": {
                "learning_rate": 5e-4
            }
        },
        "medium": {
            "model": {
                "hidden_size": 768,
                "num_layers": 12,
                "num_heads": 12,
                "intermediate_size": 3072
            },
            "data": {
                "batch_size": 32
            },
            "optimization": {
                "learning_rate": 1e-4
            }
        },
        "large": {
            "model": {
                "hidden_size": 1024,
                "num_layers": 24,
                "num_heads": 16,
                "intermediate_size": 4096
            },
            "data": {
                "batch_size": 16
            },
            "optimization": {
                "learning_rate": 5e-5
            }
        }
    }
    
    if template_name not in templates:
        raise ValueError(f"Unknown template: {template_name}")
    
    config = create_default_config()
    config_dict = config.to_dict()
    template_dict = templates[template_name]
    
    # Deep update with template
    def deep_update(base_dict, update_dict):
        for key, value in update_dict.items():
            if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
                deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    deep_update(config_dict, template_dict)
    
    return BharatFMTrainingConfig.from_dict(config_dict)


def main():
    """Main function for configuration testing"""
    
    # Create default configuration
    config = create_default_config()
    print("Default configuration created")
    
    # Save configuration
    config.save("default_config.json")
    print("Configuration saved to default_config.json")
    
    # Load configuration
    loaded_config = BharatFMTrainingConfig.load("default_config.json")
    print("Configuration loaded successfully")
    
    # Test validation
    errors = loaded_config.validate()
    if errors:
        print(f"Validation errors: {errors}")
    else:
        print("Configuration is valid")
    
    # Test templates
    for template_name in ["small", "medium", "large"]:
        template_config = create_config_from_template(template_name)
        print(f"\n{template_name.upper()} template:")
        print(f"  Hidden size: {template_config.model.hidden_size}")
        print(f"  Num layers: {template_config.model.num_layers}")
        print(f"  Batch size: {template_config.data.batch_size}")
        print(f"  Learning rate: {template_config.optimization.learning_rate}")


if __name__ == "__main__":
    main()