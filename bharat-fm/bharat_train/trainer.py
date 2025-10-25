"""
Main trainer class for BharatFM models
"""

import os
import json
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
from datetime import datetime
import logging
from transformers import get_linear_schedule_with_warmup
from transformers.trainer_utils import set_seed

from ..model import get_model_config
from ..data import IndicDataset, DataCleaner, TextNormalizer


class TrainingConfig:
    """Configuration for training BharatFM models"""
    
    def __init__(
        self,
        # Model configuration
        model_name: str = "bharat-base",
        model_type: str = "glm",
        
        # Training parameters
        batch_size: int = 32,
        learning_rate: float = 2e-5,
        num_epochs: int = 10,
        max_steps: int = 50000,
        warmup_steps: int = 1000,
        weight_decay: float = 0.01,
        gradient_accumulation_steps: int = 1,
        
        # Data configuration
        dataset_path: str = "./datasets",
        languages: List[str] = None,
        max_length: int = 512,
        
        # Distributed training
        distributed: bool = False,
        backend: str = "nccl",
        
        # DeepSpeed configuration
        use_deepspeed: bool = False,
        deepspeed_config: Optional[str] = None,
        
        # FSDP configuration
        use_fsdp: bool = False,
        fsdp_config: Optional[Dict] = None,
        
        # Mixed precision
        fp16: bool = False,
        bf16: bool = True,
        
        # Logging and checkpointing
        output_dir: str = "./outputs",
        logging_steps: int = 100,
        save_steps: int = 1000,
        eval_steps: int = 1000,
        save_total_limit: int = 3,
        
        # Bharat-specific settings
        enable_indic_attention: bool = True,
        multilingual_training: bool = True,
        domain_adapters: List[str] = None,
    ):
        self.model_name = model_name
        self.model_type = model_type
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.dataset_path = dataset_path
        self.languages = languages or ["hi", "en", "bn", "ta", "te", "mr", "gu", "kn", "ml", "pa"]
        self.max_length = max_length
        self.distributed = distributed
        self.backend = backend
        self.use_deepspeed = use_deepspeed
        self.deepspeed_config = deepspeed_config
        self.use_fsdp = use_fsdp
        self.fsdp_config = fsdp_config
        self.fp16 = fp16
        self.bf16 = bf16
        self.output_dir = output_dir
        self.logging_steps = logging_steps
        self.save_steps = save_steps
        self.eval_steps = eval_steps
        self.save_total_limit = save_total_limit
        self.enable_indic_attention = enable_indic_attention
        self.multilingual_training = multilingual_training
        self.domain_adapters = domain_adapters or ["general"]
        
    def to_dict(self) -> Dict:
        """Convert config to dictionary"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
        
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'TrainingConfig':
        """Create config from dictionary"""
        return cls(**config_dict)
        
    def save(self, path: str):
        """Save config to file"""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
            
    @classmethod
    def load(cls, path: str) -> 'TrainingConfig':
        """Load config from file"""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


class BharatTrainer:
    """Main trainer class for BharatFM models"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.train_dataloader = None
        self.eval_dataloader = None
        self.global_step = 0
        self.epoch = 0
        
        # Setup logging
        self.setup_logging()
        
        # Setup distributed training
        if config.distributed:
            self.setup_distributed()
            
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        config.save(self.output_dir / "training_config.json")
        
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.config.output_dir + '/training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def setup_distributed(self):
        """Setup distributed training"""
        if not dist.is_initialized():
            dist.init_process_group(backend=self.config.backend)
            
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        
        # Set device for distributed training
        torch.cuda.set_device(self.local_rank)
        
        self.logger.info(f"Distributed training initialized: rank={self.rank}, world_size={self.world_size}")
        
    def setup_model(self):
        """Setup model and move to device"""
        # Get model configuration
        model_config = get_model_config(self.config.model_name)
        
        # Update model config with training config
        model_config.enable_indic_attention = self.config.enable_indic_attention
        model_config.multilingual_head = self.config.multilingual_training
        model_config.domain_adapters = self.config.domain_adapters
        
        # Create model
        if self.config.model_type == "glm":
            from ..model import GLMForCausalLM
            self.model = GLMForCausalLM(model_config)
        elif self.config.model_type == "llama":
            from ..model import LlamaForCausalLM
            self.model = LlamaForCausalLM(model_config)
        elif self.config.model_type == "moe":
            from ..model import MoEForCausalLM
            self.model = MoEForCausalLM(model_config)
        else:
            raise ValueError(f"Unknown model type: {self.config.model_type}")
            
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Setup distributed model wrapping
        if self.config.distributed:
            if self.config.use_fsdp:
                self.model = self.setup_fsdp(self.model)
            else:
                self.model = torch.nn.parallel.DistributedDataParallel(
                    self.model,
                    device_ids=[self.local_rank],
                    output_device=self.local_rank
                )
                
        self.logger.info(f"Model setup complete: {self.config.model_name} ({self.config.model_type})")
        
    def setup_fsdp(self, model: nn.Module) -> nn.Module:
        """Setup FSDP wrapping"""
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import MixedPrecision
        
        # Mixed precision configuration
        mixed_precision = None
        if self.config.fp16:
            mixed_precision = MixedPrecision(param_dtype=torch.float16, reduce_dtype=torch.float16, buffer_dtype=torch.float16)
        elif self.config.bf16:
            mixed_precision = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16)
            
        # FSDP configuration
        fsdp_config = self.config.fsdp_config or {}
        
        return FSDP(
            model,
            mixed_precision=mixed_precision,
            **fsdp_config
        )
        
    def setup_optimizer(self):
        """Setup optimizer and scheduler"""
        # Separate parameters for weight decay
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        
        # Create optimizer
        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.config.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        
        # Create scheduler
        if self.config.max_steps > 0:
            num_training_steps = self.config.max_steps
        else:
            num_training_steps = len(self.train_dataloader) * self.config.num_epochs // self.config.gradient_accumulation_steps
            
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=num_training_steps,
        )
        
        self.logger.info("Optimizer and scheduler setup complete")
        
    def setup_data(self):
        """Setup data loaders"""
        # Load dataset
        dataset = IndicDataset(
            name="bharat_training",
            languages=self.config.languages,
            data_path=self.config.dataset_path
        )
        
        # Load from available sources
        if os.path.exists(os.path.join(self.config.dataset_path, "training_data.json")):
            dataset.load_from_json(os.path.join(self.config.dataset_path, "training_data.json"))
        else:
            # For demo, create synthetic data
            self.logger.warning("No training data found, using synthetic data")
            import datasets
            synthetic_data = [
                {"text": "भारत एक महान देश है।", "language": "hi"},
                {"text": "India is a great country.", "language": "en"},
                {"text": "ভারত একটি মহান দেশ।", "language": "bn"},
            ]
            dataset.dataset = datasets.Dataset.from_list(synthetic_data)
            
        # Data preprocessing
        cleaner = DataCleaner(self.config.languages)
        normalizer = TextNormalizer()
        
        # Apply preprocessing
        if hasattr(dataset.dataset, 'to_pandas'):
            df = dataset.dataset.to_pandas()
            df = cleaner.clean_dataset(df, text_column="text")
            df = normalizer.normalize_dataset(df, text_column="text", language_column="language")
            dataset.dataset = datasets.Dataset.from_pandas(df)
            
        # Create data loader
        if self.config.distributed:
            sampler = DistributedSampler(
                dataset.dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True
            )
        else:
            sampler = None
            
        def collate_fn(batch):
            # Simple collate function
            texts = [item["text"] for item in batch]
            languages = [item.get("language", "en") for item in batch]
            
            # Tokenize (simplified - in practice, use proper tokenizer)
            input_ids = torch.randint(0, 50000, (len(texts), self.config.max_length))
            attention_mask = torch.ones(len(texts), self.config.max_length)
            labels = input_ids.clone()
            
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "languages": languages
            }
            
        self.train_dataloader = DataLoader(
            dataset.dataset,
            batch_size=self.config.batch_size,
            sampler=sampler,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True,
        )
        
        self.logger.info(f"Data setup complete: {len(dataset.dataset)} samples")
        
    def train(self):
        """Main training loop"""
        self.logger.info("Starting training...")
        
        # Setup components
        self.setup_model()
        self.setup_data()
        self.setup_optimizer()
        
        # Training loop
        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            self.train_epoch()
            
            # Save checkpoint
            if self.rank == 0:
                self.save_checkpoint()
                
        self.logger.info("Training complete!")
        
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        
        if self.config.distributed:
            self.train_dataloader.sampler.set_epoch(self.epoch)
            
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(self.train_dataloader):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(**batch)
            loss = outputs["loss"]
            
            # Backward pass
            loss = loss / self.config.gradient_accumulation_steps
            loss.backward()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Optimizer step
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.config.use_deepspeed:
                    # DeepSpeed handles gradient clipping
                    pass
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                self.global_step += 1
                
                # Logging
                if self.global_step % self.config.logging_steps == 0 and self.rank == 0:
                    avg_loss = total_loss / num_batches
                    self.logger.info(f"Step {self.global_step}: Loss = {avg_loss:.4f}")
                    
                    # Reset counters
                    total_loss = 0.0
                    num_batches = 0
                    
                # Save checkpoint
                if self.global_step % self.config.save_steps == 0 and self.rank == 0:
                    self.save_checkpoint()
                    
                # Evaluation
                if self.global_step % self.config.eval_steps == 0 and self.rank == 0:
                    self.evaluate()
                    
            # Check max steps
            if self.config.max_steps > 0 and self.global_step >= self.config.max_steps:
                self.logger.info(f"Reached max steps: {self.config.max_steps}")
                return
                
    def evaluate(self):
        """Evaluate model"""
        self.logger.info("Starting evaluation...")
        self.model.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.eval_dataloader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs["loss"]
                
                total_loss += loss.item()
                num_batches += 1
                
        avg_loss = total_loss / num_batches
        self.logger.info(f"Evaluation complete: Loss = {avg_loss:.4f}")
        
        # Log metrics
        if self.rank == 0:
            metrics = {
                "eval_loss": avg_loss,
                "step": self.global_step,
                "epoch": self.epoch
            }
            self.log_metrics(metrics)
            
    def save_checkpoint(self):
        """Save model checkpoint"""
        checkpoint_dir = self.output_dir / f"checkpoint-{self.global_step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model state
        if self.config.use_fsdp:
            # FSDP requires special handling
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            from torch.distributed.fsdp import StateDictType, FullStateDictConfig
            
            full_state_dict_config = FullStateDictConfig(rank0_only=True, offload_to_cpu=True)
            with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, full_state_dict_config):
                cpu_state_dict = self.model.state_dict()
                if self.rank == 0:
                    torch.save(cpu_state_dict, checkpoint_dir / "model.pt")
        else:
            if self.rank == 0:
                torch.save(self.model.state_dict(), checkpoint_dir / "model.pt")
                
        # Save optimizer and scheduler states
        if self.rank == 0:
            torch.save(self.optimizer.state_dict(), checkpoint_dir / "optimizer.pt")
            torch.save(self.scheduler.state_dict(), checkpoint_dir / "scheduler.pt")
            
            # Save training state
            training_state = {
                "global_step": self.global_step,
                "epoch": self.epoch,
                "config": self.config.to_dict()
            }
            with open(checkpoint_dir / "training_state.json", 'w') as f:
                json.dump(training_state, f, indent=2)
                
        self.logger.info(f"Checkpoint saved: {checkpoint_dir}")
        
    def log_metrics(self, metrics: Dict[str, float]):
        """Log training metrics"""
        timestamp = datetime.now().isoformat()
        
        log_entry = {
            "timestamp": timestamp,
            "step": self.global_step,
            "epoch": self.epoch,
            **metrics
        }
        
        # Save to file
        metrics_file = self.output_dir / "metrics.jsonl"
        with open(metrics_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
            
        self.logger.info(f"Metrics logged: {log_entry}")
        
    def cleanup(self):
        """Cleanup resources"""
        if self.config.distributed:
            dist.destroy_process_group()
            
        self.logger.info("Training cleanup complete")