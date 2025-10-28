"""
Advanced Distributed Training Infrastructure for Bharat-FM
Multi-GPU support, model parallelism, and distributed optimization
"""

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
import torch.cuda.amp as amp
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
import json
import time
import argparse
import socket
import subprocess
from pathlib import Path
import numpy as np
from collections import defaultdict

# Import existing components
from .real_training_system import TrainingConfig, TrainingMetrics, TextDataset

logger = logging.getLogger(__name__)

@dataclass
class DistributedConfig:
    """Configuration for distributed training"""
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    master_addr: str = "localhost"
    master_port: str = "12355"
    backend: str = "nccl"  # nccl, gloo
    
    # Multi-GPU settings
    num_gpus: int = 1
    gpu_ids: List[int] = None
    
    # Model parallelism settings
    model_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    
    # Optimization settings
    mixed_precision: str = "fp16"  # fp16, fp32, fp8, bf16
    gradient_accumulation_steps: int = 1
    gradient_clipping: float = 1.0
    
    # Learning rate settings
    lr_finder: bool = False
    lr_scheduler_type: str = "cosine"  # cosine, onecycle, warmupcosine, linear
    max_lr: float = 1e-3
    min_lr: float = 1e-6
    warmup_steps: int = 1000
    total_steps: int = 10000
    
    # Cluster settings
    cluster_type: str = "local"  # local, kubernetes, slurm
    job_id: Optional[str] = None
    
    # Monitoring
    enable_tensorboard: bool = True
    log_interval: int = 10
    eval_interval: int = 100
    save_interval: int = 1000
    
    def __post_init__(self):
        if self.gpu_ids is None:
            self.gpu_ids = list(range(self.num_gpus))

class DistributedTrainer:
    """Advanced distributed trainer with multi-GPU support"""
    
    def __init__(self, 
                 training_config: TrainingConfig,
                 distributed_config: DistributedConfig):
        self.training_config = training_config
        self.distributed_config = distributed_config
        
        # Distributed training state
        self.world_size = distributed_config.world_size
        self.rank = distributed_config.rank
        self.local_rank = distributed_config.local_rank
        
        # Model and optimizer
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_metric = float('inf')
        
        # Mixed precision
        self.amp_dtype = self._get_amp_dtype()
        
        # Logging
        self.writer = None
        self.metrics_history = []
        
        # Initialize distributed training
        self._init_distributed()
        
    def _get_amp_dtype(self):
        """Get automatic mixed precision dtype"""
        precision = self.distributed_config.mixed_precision.lower()
        if precision == "fp16":
            return torch.float16
        elif precision == "bf16":
            return torch.bfloat16
        elif precision == "fp8":
            return torch.float8_e5m2 if hasattr(torch, 'float8_e5m2') else torch.float16
        else:
            return torch.float32
    
    def _init_distributed(self):
        """Initialize distributed training environment"""
        if self.world_size > 1:
            # Set environment variables
            os.environ['MASTER_ADDR'] = self.distributed_config.master_addr
            os.environ['MASTER_PORT'] = self.distributed_config.master_port
            os.environ['WORLD_SIZE'] = str(self.world_size)
            os.environ['RANK'] = str(self.rank)
            os.environ['LOCAL_RANK'] = str(self.local_rank)
            
            # Initialize process group
            dist.init_process_group(
                backend=self.distributed_config.backend,
                world_size=self.world_size,
                rank=self.rank
            )
            
            # Set device
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device(f"cuda:{self.local_rank}")
            
            logger.info(f"Initialized distributed training: rank={self.rank}/{self.world_size}, "
                       f"local_rank={self.local_rank}, device={self.device}")
        else:
            # Single GPU or CPU training
            if torch.cuda.is_available():
                self.device = torch.device("cuda:0")
            else:
                self.device = torch.device("cpu")
            logger.info(f"Initialized single device training: device={self.device}")
        
        # Initialize mixed precision scaler
        if self.distributed_config.mixed_precision in ["fp16", "bf16", "fp8"]:
            self.scaler = amp.GradScaler()
    
    def setup_model(self, model: nn.Module):
        """Setup model for distributed training"""
        self.model = model.to(self.device)
        
        # Wrap model for distributed training
        if self.world_size > 1:
            self.model = DDP(self.model, device_ids=[self.local_rank])
        
        logger.info(f"Model setup complete. Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def setup_optimizer(self, optimizer_class: torch.optim.Optimizer, **kwargs):
        """Setup optimizer for distributed training"""
        if self.model is None:
            raise RuntimeError("Model must be setup before optimizer")
        
        # Adjust learning rate for distributed training
        lr = kwargs.get('lr', self.training_config.learning_rate)
        if self.world_size > 1:
            # Linear scaling rule for distributed training
            lr = lr * self.world_size
            kwargs['lr'] = lr
        
        self.optimizer = optimizer_class(self.model.parameters(), **kwargs)
        logger.info(f"Optimizer setup complete. Learning rate: {lr}")
    
    def setup_scheduler(self, scheduler_type: str = None, **kwargs):
        """Setup learning rate scheduler"""
        if self.optimizer is None:
            raise RuntimeError("Optimizer must be setup before scheduler")
        
        scheduler_type = scheduler_type or self.distributed_config.lr_scheduler_type
        
        if scheduler_type == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.distributed_config.total_steps,
                eta_min=self.distributed_config.min_lr
            )
        elif scheduler_type == "onecycle":
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.distributed_config.max_lr,
                total_steps=self.distributed_config.total_steps,
                pct_start=0.3,
                anneal_strategy='cos'
            )
        elif scheduler_type == "warmupcosine":
            self.scheduler = WarmupCosineScheduler(
                self.optimizer,
                warmup_steps=self.distributed_config.warmup_steps,
                total_steps=self.distributed_config.total_steps,
                min_lr=self.distributed_config.min_lr
            )
        elif scheduler_type == "linear":
            self.scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=0.1,
                total_iters=self.distributed_config.total_steps
            )
        else:
            self.scheduler = None
        
        logger.info(f"Scheduler setup complete. Type: {scheduler_type}")
    
    def setup_tensorboard(self, log_dir: str = "./logs"):
        """Setup TensorBoard logging"""
        if self.rank == 0 and self.distributed_config.enable_tensorboard:
            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir)
            logger.info(f"TensorBoard logging setup: {log_dir}")
    
    def train_epoch(self, dataloader: DataLoader, epoch: int) -> TrainingMetrics:
        """Train for one epoch with distributed support"""
        self.model.train()
        epoch_start_time = time.time()
        
        total_loss = 0
        total_samples = 0
        correct_predictions = 0
        
        # Initialize gradient accumulation
        accumulated_loss = 0
        accumulation_steps = 0
        
        progress_bar = enumerate(dataloader)
        if self.rank == 0:
            progress_bar = tqdm(progress_bar, desc=f"Epoch {epoch}", total=len(dataloader))
        
        for step, batch in progress_bar:
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass with mixed precision
            with amp.autocast(dtype=self.amp_dtype):
                outputs = self.model(**batch)
                loss = self._compute_loss(outputs, batch)
                
                # Scale loss for gradient accumulation
                if self.distributed_config.gradient_accumulation_steps > 1:
                    loss = loss / self.distributed_config.gradient_accumulation_steps
            
            # Backward pass
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation
            accumulated_loss += loss.item()
            accumulation_steps += 1
            
            # Optimizer step (with accumulation)
            if accumulation_steps >= self.distributed_config.gradient_accumulation_steps:
                # Gradient clipping
                if self.distributed_config.gradient_clipping > 0:
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                    
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.distributed_config.gradient_clipping
                    )
                
                # Optimizer step
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                # Scheduler step
                if self.scheduler is not None:
                    self.scheduler.step()
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Update metrics
                total_loss += accumulated_loss
                total_samples += batch.get('input_ids', batch.get('inputs', torch.tensor(0))).size(0)
                
                # Calculate accuracy if possible
                if hasattr(outputs, 'argmax'):
                    predictions = outputs.argmax(dim=-1)
                    targets = batch.get('labels', batch.get('targets', torch.tensor(0)))
                    if predictions.shape == targets.shape:
                        correct_predictions += (predictions == targets).sum().item()
                
                # Reset accumulation
                accumulated_loss = 0
                accumulation_steps = 0
                
                # Update global step
                self.global_step += 1
                
                # Logging
                if self.rank == 0 and self.global_step % self.distributed_config.log_interval == 0:
                    self._log_step_metrics(loss.item(), epoch)
                
                # Evaluation
                if self.rank == 0 and self.global_step % self.distributed_config.eval_interval == 0:
                    self._run_evaluation()
                
                # Save checkpoint
                if self.rank == 0 and self.global_step % self.distributed_config.save_interval == 0:
                    self._save_checkpoint(epoch)
        
        # Synchronize metrics across processes
        if self.world_size > 1:
            total_loss = self._all_reduce_tensor(torch.tensor(total_loss)).item()
            total_samples = self._all_reduce_tensor(torch.tensor(total_samples)).item()
            correct_predictions = self._all_reduce_tensor(torch.tensor(correct_predictions)).item()
        
        # Calculate epoch metrics
        avg_loss = total_loss / max(len(dataloader), 1)
        accuracy = correct_predictions / max(total_samples, 1)
        epoch_time = time.time() - epoch_start_time
        
        metrics = TrainingMetrics(
            epoch=epoch,
            step=self.global_step,
            train_loss=avg_loss,
            train_accuracy=accuracy,
            learning_rate=self.optimizer.param_groups[0]['lr'] if self.optimizer else 0,
            throughput=total_samples / epoch_time,
            memory_usage=self._get_memory_usage(),
            timestamp=datetime.utcnow()
        )
        
        self.metrics_history.append(metrics)
        self.epoch = epoch
        
        return metrics
    
    def _compute_loss(self, outputs: torch.Tensor, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute loss based on model outputs"""
        if 'labels' in batch:
            return F.cross_entropy(outputs.view(-1, outputs.size(-1)), batch['labels'].view(-1))
        elif 'targets' in batch:
            return F.cross_entropy(outputs.view(-1, outputs.size(-1)), batch['targets'].view(-1))
        else:
            # Default to mean squared error for regression tasks
            target = batch.get('input_ids', batch.get('inputs', torch.tensor(0)))
            return F.mse_loss(outputs, target.float())
    
    def _all_reduce_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """All reduce tensor across distributed processes"""
        if self.world_size > 1:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            tensor /= self.world_size
        return tensor
    
    def _log_step_metrics(self, loss: float, epoch: int):
        """Log step metrics to TensorBoard and console"""
        lr = self.optimizer.param_groups[0]['lr'] if self.optimizer else 0
        
        # TensorBoard logging
        if self.writer:
            self.writer.add_scalar('Loss/train', loss, self.global_step)
            self.writer.add_scalar('Learning_Rate', lr, self.global_step)
            self.writer.add_scalar('Memory/GB', self._get_memory_usage(), self.global_step)
        
        # Console logging
        logger.info(f"Step {self.global_step}: Loss={loss:.4f}, LR={lr:.2e}, "
                   f"Memory={self._get_memory_usage():.2f}GB")
    
    def _run_evaluation(self):
        """Run evaluation on validation set"""
        # Placeholder for evaluation logic
        logger.info(f"Running evaluation at step {self.global_step}")
    
    def _save_checkpoint(self, epoch: int):
        """Save model checkpoint"""
        if self.rank == 0:
            checkpoint = {
                'epoch': epoch,
                'global_step': self.global_step,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
                'config': asdict(self.training_config),
                'metrics': [asdict(m) for m in self.metrics_history]
            }
            
            checkpoint_path = f"checkpoint_step_{self.global_step}.pt"
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def _get_memory_usage(self) -> float:
        """Get current GPU memory usage in GB"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024**3
        return 0.0
    
    def cleanup(self):
        """Cleanup distributed training resources"""
        if self.writer:
            self.writer.close()
        
        if self.world_size > 1:
            dist.destroy_process_group()
        
        logger.info("Distributed training cleanup complete")


class WarmupCosineScheduler:
    """Warmup + Cosine Annealing Learning Rate Scheduler"""
    
    def __init__(self, 
                 optimizer: torch.optim.Optimizer,
                 warmup_steps: int,
                 total_steps: int,
                 min_lr: float = 1e-6):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        self.current_step = 0
    
    def step(self):
        """Update learning rate"""
        self.current_step += 1
        
        if self.current_step < self.warmup_steps:
            # Warmup phase
            lr = self.base_lr * (self.current_step / self.warmup_steps)
        else:
            # Cosine annealing phase
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def get_last_lr(self) -> List[float]:
        """Get last learning rate"""
        return [self.optimizer.param_groups[0]['lr']]


class LearningRateFinder:
    """Learning rate finder for optimal LR selection"""
    
    def __init__(self, 
                 model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 criterion: nn.Module,
                 device: torch.device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        
        self.lr_history = []
        self.loss_history = []
    
    def range_test(self, 
                   dataloader: DataLoader,
                   start_lr: float = 1e-8,
                   end_lr: float = 10.0,
                   num_iter: int = 100,
                   smooth_f: float = 0.05):
        """Run learning rate range test"""
        lrs = np.logspace(np.log10(start_lr), np.log10(end_lr), num_iter)
        
        self.model.train()
        best_loss = float('inf')
        
        for i, lr in enumerate(lrs):
            # Update learning rate
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            
            # Get batch
            try:
                batch = next(iter(dataloader))
            except StopIteration:
                break
            
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(**batch)
            loss = self._compute_loss(outputs, batch)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Smooth loss
            if i == 0:
                smooth_loss = loss.item()
            else:
                smooth_loss = smooth_f * loss.item() + (1 - smooth_f) * smooth_loss
            
            # Store history
            self.lr_history.append(lr)
            self.loss_history.append(smooth_loss)
            
            # Stop if loss explodes
            if smooth_loss > 4 * best_loss:
                break
            
            if smooth_loss < best_loss:
                best_loss = smooth_loss
            
            if i % 10 == 0:
                logger.info(f"LR Test: LR={lr:.2e}, Loss={smooth_loss:.4f}")
        
        return self.lr_history, self.loss_history
    
    def _compute_loss(self, outputs: torch.Tensor, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute loss"""
        if 'labels' in batch:
            return self.criterion(outputs.view(-1, outputs.size(-1)), batch['labels'].view(-1))
        elif 'targets' in batch:
            return self.criterion(outputs.view(-1, outputs.size(-1)), batch['targets'].view(-1))
        else:
            target = batch.get('input_ids', batch.get('inputs', torch.tensor(0)))
            return self.criterion(outputs, target.float())
    
    def plot_lr_finder(self, save_path: str = "lr_finder.png"):
        """Plot learning rate finder results"""
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.lr_history, self.loss_history)
        plt.xscale('log')
        plt.xlabel('Learning Rate')
        plt.ylabel('Loss')
        plt.title('Learning Rate Finder')
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()
        
        logger.info(f"LR finder plot saved: {save_path}")


def create_distributed_sampler(dataset, world_size: int, rank: int, shuffle: bool = True):
    """Create distributed sampler for dataset"""
    if world_size > 1:
        return DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle
        )
    else:
        return None


def setup_distributed_environment():
    """Setup distributed training environment from environment variables"""
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    
    return world_size, rank, local_rank


def main():
    """Main function for distributed training"""
    parser = argparse.ArgumentParser(description='Distributed Training for Bharat-FM')
    parser.add_argument('--config', type=str, required=True, help='Training config file')
    parser.add_argument('--distributed_config', type=str, required=True, help='Distributed config file')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # Load configurations
    training_config = TrainingConfig(**json.load(open(args.config)))
    distributed_config = DistributedConfig(**json.load(open(args.distributed_config)))
    
    # Setup distributed environment
    world_size, rank, local_rank = setup_distributed_environment()
    distributed_config.world_size = world_size
    distributed_config.rank = rank
    distributed_config.local_rank = local_rank
    
    # Create distributed trainer
    trainer = DistributedTrainer(training_config, distributed_config)
    
    try:
        # Training logic here
        logger.info("Starting distributed training...")
        # ... training implementation ...
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    finally:
        trainer.cleanup()


if __name__ == "__main__":
    main()