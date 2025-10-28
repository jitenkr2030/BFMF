"""
Advanced Learning Rate Schedulers and Gradient Accumulation
Cosine, OneCycle, WarmupCosine, and custom schedulers with gradient accumulation
"""

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
import numpy as np
import math
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass
import logging
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

@dataclass
class SchedulerConfig:
    """Configuration for learning rate schedulers"""
    scheduler_type: str = "cosine"  # cosine, onecycle, warmupcosine, linear, cosine_with_restarts, polynomial
    
    # Common parameters
    max_lr: float = 1e-3
    min_lr: float = 1e-6
    total_steps: int = 10000
    warmup_steps: int = 1000
    
    # OneCycle specific
    pct_start: float = 0.3
    anneal_strategy: str = "cos"
    div_factor: float = 25.0
    final_div_factor: float = 10000.0
    
    # Cosine with restarts
    restart_period: int = 1000
    restart_mult: float = 1.0
    
    # Polynomial
    power: float = 1.0
    
    # Gradient accumulation
    accumulation_steps: int = 1
    sync_frequency: int = 1
    
    # Cycle parameters
    cycle_length: int = 1000
    cycle_mult: float = 1.0
    cycle_decay: float = 1.0


class GradientAccumulator:
    """Advanced gradient accumulation with synchronization"""
    
    def __init__(self, 
                 accumulation_steps: int = 1,
                 sync_frequency: int = 1,
                 gradient_clipping: float = 0.0):
        self.accumulation_steps = accumulation_steps
        self.sync_frequency = sync_frequency
        self.gradient_clipping = gradient_clipping
        
        # State
        self.accumulated_steps = 0
        self.total_steps = 0
        self.gradient_buffer = None
        self.loss_buffer = []
        
        # Synchronization
        self.sync_counter = 0
        
    def accumulate_gradients(self, 
                            model: torch.nn.Module,
                            loss: torch.Tensor,
                            scale_factor: float = 1.0) -> bool:
        """Accumulate gradients, return True if should step optimizer"""
        
        # Scale loss for accumulation
        scaled_loss = loss / self.accumulation_steps
        
        # Backward pass
        scaled_loss.backward()
        
        # Store loss for averaging
        self.loss_buffer.append(loss.item())
        
        # Update counters
        self.accumulated_steps += 1
        self.total_steps += 1
        
        # Check if should step optimizer
        if self.accumulated_steps >= self.accumulation_steps:
            # Apply gradient clipping if specified
            if self.gradient_clipping > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    self.gradient_clipping
                )
            
            # Reset accumulation
            self.accumulated_steps = 0
            
            # Check if should synchronize
            self.sync_counter += 1
            should_sync = self.sync_counter >= self.sync_frequency
            if should_sync:
                self.sync_counter = 0
            
            return True, should_sync
        
        return False, False
    
    def get_average_loss(self) -> float:
        """Get average loss over accumulation window"""
        if not self.loss_buffer:
            return 0.0
        avg_loss = sum(self.loss_buffer) / len(self.loss_buffer)
        self.loss_buffer.clear()
        return avg_loss
    
    def reset(self):
        """Reset accumulator state"""
        self.accumulated_steps = 0
        self.sync_counter = 0
        self.loss_buffer.clear()


class WarmupCosineScheduler(_LRScheduler):
    """Warmup + Cosine Annealing Scheduler"""
    
    def __init__(self, 
                 optimizer: torch.optim.Optimizer,
                 warmup_steps: int,
                 total_steps: int,
                 min_lr: float = 1e-6,
                 last_epoch: int = -1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if not self._get_lr_called_within_step:
            logger.warning("To get the last learning rate computed by the scheduler, "
                         "please use `get_last_lr()`.")
        
        step = self.last_epoch
        
        if step < self.warmup_steps:
            # Warmup phase
            lr = [base_lr * (step / max(1, self.warmup_steps)) for base_lr in self.base_lrs]
        else:
            # Cosine annealing phase
            progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            lr = [self.min_lr + (base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
                  for base_lr in self.base_lrs]
        
        return lr


class CosineWithRestartsScheduler(_LRScheduler):
    """Cosine Annealing with Restarts"""
    
    def __init__(self, 
                 optimizer: torch.optim.Optimizer,
                 restart_period: int,
                 restart_mult: float = 1.0,
                 min_lr: float = 1e-6,
                 last_epoch: int = -1):
        self.restart_period = restart_period
        self.restart_mult = restart_mult
        self.min_lr = min_lr
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.current_restart_period = restart_period
        
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if not self._get_lr_called_within_step:
            logger.warning("To get the last learning rate computed by the scheduler, "
                         "please use `get_last_lr()`.")
        
        step = self.last_epoch
        
        # Check if we need to restart
        if step > 0 and step % self.current_restart_period == 0:
            self.current_restart_period = int(self.current_restart_period * self.restart_mult)
            step = 0
        
        # Cosine annealing within current period
        progress = step / max(1, self.current_restart_period)
        lr = [self.min_lr + (base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
              for base_lr in self.base_lrs]
        
        return lr


class PolynomialScheduler(_LRScheduler):
    """Polynomial Learning Rate Decay"""
    
    def __init__(self, 
                 optimizer: torch.optim.Optimizer,
                 total_steps: int,
                 power: float = 1.0,
                 min_lr: float = 1e-6,
                 last_epoch: int = -1):
        self.total_steps = total_steps
        self.power = power
        self.min_lr = min_lr
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if not self._get_lr_called_within_step:
            logger.warning("To get the last learning rate computed by the scheduler, "
                         "please use `get_last_lr()`.")
        
        step = min(self.last_epoch, self.total_steps)
        progress = step / max(1, self.total_steps)
        
        lr = [self.min_lr + (base_lr - self.min_lr) * (1 - progress) ** self.power
              for base_lr in self.base_lrs]
        
        return lr


class CyclicScheduler(_LRScheduler):
    """Cyclic Learning Rate Scheduler"""
    
    def __init__(self, 
                 optimizer: torch.optim.Optimizer,
                 cycle_length: int,
                 max_lr: float,
                 min_lr: float,
                 cycle_mult: float = 1.0,
                 cycle_decay: float = 1.0,
                 mode: str = "triangular",
                 last_epoch: int = -1):
        self.cycle_length = cycle_length
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.cycle_mult = cycle_mult
        self.cycle_decay = cycle_decay
        self.mode = mode
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        
        # Cycle tracking
        self.current_cycle = 0
        self.current_cycle_length = cycle_length
        self.cycle_step = 0
        
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if not self._get_lr_called_within_step:
            logger.warning("To get the last learning rate computed by the scheduler, "
                         "please use `get_last_lr()`.")
        
        # Update cycle state
        if self.cycle_step >= self.current_cycle_length:
            self.cycle_step = 0
            self.current_cycle += 1
            self.current_cycle_length = int(self.current_cycle_length * self.cycle_mult)
        
        # Calculate progress within current cycle
        progress = self.cycle_step / max(1, self.current_cycle_length - 1)
        
        # Apply cycle decay
        effective_max_lr = self.max_lr * (self.cycle_decay ** self.current_cycle)
        effective_min_lr = self.min_lr * (self.cycle_decay ** self.current_cycle)
        
        # Calculate learning rate based on mode
        if self.mode == "triangular":
            lr = effective_min_lr + (effective_max_lr - effective_min_lr) * progress
        elif self.mode == "triangular2":
            lr = effective_min_lr + (effective_max_lr - effective_min_lr) * abs(2 * progress - 1)
        elif self.mode == "exp_range":
            lr = effective_min_lr + (effective_max_lr - effective_min_lr) * (0.99994 ** (self.cycle_step))
        else:  # cosine
            lr = effective_min_lr + (effective_max_lr - effective_min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
        
        # Update step
        self.cycle_step += 1
        
        return [lr for _ in self.base_lrs]


class OneCycleScheduler(_LRScheduler):
    """One Cycle Learning Rate Scheduler with momentum"""
    
    def __init__(self, 
                 optimizer: torch.optim.Optimizer,
                 max_lr: float,
                 total_steps: int,
                 pct_start: float = 0.3,
                 anneal_strategy: str = "cos",
                 div_factor: float = 25.0,
                 final_div_factor: float = 10000.0,
                 last_epoch: int = -1):
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.anneal_strategy = anneal_strategy
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor
        
        self.initial_lr = max_lr / div_factor
        self.min_lr = max_lr / final_div_factor
        
        # Calculate phase steps
        self.warmup_steps = int(total_steps * pct_start)
        self.anneal_steps = total_steps - self.warmup_steps
        
        # Momentum tracking (if optimizer supports it)
        self.initial_momentum = None
        self.max_momentum = None
        self.min_momentum = None
        
        for group in optimizer.param_groups:
            if 'momentum' in group:
                if self.initial_momentum is None:
                    self.initial_momentum = group['momentum']
                    self.max_momentum = group['momentum']
                    self.min_momentum = group['momentum'] * 0.8
                group['momentum'] = self.max_momentum
        
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if not self._get_lr_called_within_step:
            logger.warning("To get the last learning rate computed by the scheduler, "
                         "please use `get_last_lr()`.")
        
        step = self.last_epoch
        
        if step < self.warmup_steps:
            # Warmup phase
            progress = step / max(1, self.warmup_steps)
            lr = [self.initial_lr + (self.max_lr - self.initial_lr) * progress
                  for _ in self.base_lrs]
            
            # Update momentum (inverse of learning rate)
            if self.initial_momentum is not None:
                momentum = self.max_momentum - (self.max_momentum - self.min_momentum) * progress
                for group in self.optimizer.param_groups:
                    group['momentum'] = momentum
        
        else:
            # Annealing phase
            step = step - self.warmup_steps
            progress = step / max(1, self.anneal_steps)
            
            if self.anneal_strategy == "cos":
                lr = [self.min_lr + (self.max_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
                      for _ in self.base_lrs]
            else:  # linear
                lr = [self.max_lr - (self.max_lr - self.min_lr) * progress
                      for _ in self.base_lrs]
            
            # Update momentum
            if self.initial_momentum is not None:
                momentum = self.min_momentum + (self.max_momentum - self.min_momentum) * progress
                for group in self.optimizer.param_groups:
                    group['momentum'] = momentum
        
        return lr


class LearningRateFinder:
    """Advanced Learning Rate Finder with visualization"""
    
    def __init__(self, 
                 model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 criterion: torch.nn.Module,
                 device: torch.device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        
        # Results storage
        self.lr_history = []
        self.loss_history = []
        self.smooth_loss_history = []
        
        # Configuration
        self.start_lr = 1e-8
        self.end_lr = 10.0
        self.num_iter = 100
        self.smooth_f = 0.05
        self.beta = 0.98
        
        # Best LR tracking
        self.best_lr = None
        self.best_loss = float('inf')
    
    def range_test(self, 
                   dataloader,
                   start_lr: float = None,
                   end_lr: float = None,
                   num_iter: int = None,
                   smooth_f: float = None,
                   beta: float = None):
        """Run learning rate range test"""
        
        # Update configuration
        self.start_lr = start_lr or self.start_lr
        self.end_lr = end_lr or self.end_lr
        self.num_iter = num_iter or self.num_iter
        self.smooth_f = smooth_f or self.smooth_f
        self.beta = beta or self.beta
        
        # Reset history
        self.lr_history = []
        self.loss_history = []
        self.smooth_loss_history = []
        
        # Create learning rate schedule
        lrs = np.logspace(np.log10(self.start_lr), np.log10(self.end_lr), self.num_iter)
        
        # Set model to training mode
        self.model.train()
        
        # Store initial state
        initial_state = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        
        best_loss = float('inf')
        smooth_loss = 0.0
        
        try:
            for i, lr in enumerate(lrs):
                # Update learning rate
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
                
                # Get batch
                try:
                    batch = next(iter(dataloader))
                except StopIteration:
                    logger.warning("Dataloader exhausted, stopping LR finder")
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
                    smooth_loss = self.beta * smooth_loss + (1 - self.beta) * loss.item()
                
                # Store results
                self.lr_history.append(lr)
                self.loss_history.append(loss.item())
                self.smooth_loss_history.append(smooth_loss)
                
                # Track best loss
                if smooth_loss < best_loss:
                    best_loss = smooth_loss
                    self.best_lr = lr
                
                # Stop if loss explodes
                if smooth_loss > 4 * best_loss and i > 10:
                    logger.info(f"Loss exploded, stopping LR finder at iteration {i}")
                    break
                
                # Log progress
                if i % 10 == 0:
                    logger.info(f"LR Test: LR={lr:.2e}, Loss={loss.item():.4f}, Smooth Loss={smooth_loss:.4f}")
        
        finally:
            # Restore initial state
            self.model.load_state_dict(initial_state['model'])
            self.optimizer.load_state_dict(initial_state['optimizer'])
        
        return self.lr_history, self.loss_history, self.smooth_loss_history
    
    def _compute_loss(self, outputs, batch):
        """Compute loss"""
        if 'labels' in batch:
            return self.criterion(outputs.view(-1, outputs.size(-1)), batch['labels'].view(-1))
        elif 'targets' in batch:
            return self.criterion(outputs.view(-1, outputs.size(-1)), batch['targets'].view(-1))
        else:
            target = batch.get('input_ids', batch.get('inputs', torch.tensor(0)))
            return self.criterion(outputs, target.float())
    
    def plot_lr_finder(self, save_path: str = "lr_finder.png", show_plot: bool = True):
        """Plot learning rate finder results"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Raw loss vs LR
        ax1.plot(self.lr_history, self.loss_history, 'b-', alpha=0.6, label='Raw Loss')
        ax1.plot(self.lr_history, self.smooth_loss_history, 'r-', linewidth=2, label='Smooth Loss')
        ax1.set_xscale('log')
        ax1.set_xlabel('Learning Rate')
        ax1.set_ylabel('Loss')
        ax1.set_title('Learning Rate Finder')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Mark best LR
        if self.best_lr:
            ax1.axvline(x=self.best_lr, color='green', linestyle='--', alpha=0.7, label=f'Best LR: {self.best_lr:.2e}')
            ax1.legend()
        
        # Plot 2: Gradient of loss vs LR
        if len(self.lr_history) > 1:
            # Calculate gradient
            lr_log = np.log10(self.lr_history)
            loss_grad = np.gradient(self.smooth_loss_history)
            
            ax2.plot(lr_log, loss_grad, 'g-', linewidth=2)
            ax2.set_xlabel('Log10(Learning Rate)')
            ax2.set_ylabel('Loss Gradient')
            ax2.set_title('Loss Gradient vs Log LR')
            ax2.grid(True, alpha=0.3)
            
            # Find steepest gradient
            if len(loss_grad) > 0:
                steepest_idx = np.argmin(loss_grad)
                steepest_lr = self.lr_history[steepest_idx]
                ax2.axvline(x=np.log10(steepest_lr), color='red', linestyle='--', alpha=0.7, 
                           label=f'Steepest: {steepest_lr:.2e}')
                ax2.legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        logger.info(f"LR finder plot saved to {save_path}")
    
    def get_suggested_lr(self, method: str = "steepest") -> float:
        """Get suggested learning rate based on method"""
        if method == "steepest":
            # Find steepest gradient
            if len(self.smooth_loss_history) > 1:
                lr_log = np.log10(self.lr_history)
                loss_grad = np.gradient(self.smooth_loss_history)
                steepest_idx = np.argmin(loss_grad)
                return self.lr_history[steepest_idx]
        
        elif method == "best":
            return self.best_lr
        
        elif method == "10x_below_min":
            # 10x below the minimum loss
            if self.smooth_loss_history:
                min_idx = np.argmin(self.smooth_loss_history)
                if min_idx > 0:
                    return self.lr_history[min_idx - 1]
        
        return self.best_lr or 1e-3


class SchedulerFactory:
    """Factory for creating learning rate schedulers"""
    
    @staticmethod
    def create_scheduler(scheduler_type: str,
                        optimizer: torch.optim.Optimizer,
                        config: SchedulerConfig) -> _LRScheduler:
        """Create scheduler based on type and configuration"""
        
        if scheduler_type.lower() == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config.total_steps,
                eta_min=config.min_lr
            )
        
        elif scheduler_type.lower() == "warmupcosine":
            return WarmupCosineScheduler(
                optimizer,
                warmup_steps=config.warmup_steps,
                total_steps=config.total_steps,
                min_lr=config.min_lr
            )
        
        elif scheduler_type.lower() == "onecycle":
            return OneCycleScheduler(
                optimizer,
                max_lr=config.max_lr,
                total_steps=config.total_steps,
                pct_start=config.pct_start,
                anneal_strategy=config.anneal_strategy,
                div_factor=config.div_factor,
                final_div_factor=config.final_div_factor
            )
        
        elif scheduler_type.lower() == "cosine_with_restarts":
            return CosineWithRestartsScheduler(
                optimizer,
                restart_period=config.restart_period,
                restart_mult=config.restart_mult,
                min_lr=config.min_lr
            )
        
        elif scheduler_type.lower() == "polynomial":
            return PolynomialScheduler(
                optimizer,
                total_steps=config.total_steps,
                power=config.power,
                min_lr=config.min_lr
            )
        
        elif scheduler_type.lower() == "cyclic":
            return CyclicScheduler(
                optimizer,
                cycle_length=config.cycle_length,
                max_lr=config.max_lr,
                min_lr=config.min_lr,
                cycle_mult=config.cycle_mult,
                cycle_decay=config.cycle_decay
            )
        
        elif scheduler_type.lower() == "linear":
            return torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=config.total_steps
            )
        
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    @staticmethod
    def create_gradient_accumulator(config: SchedulerConfig) -> GradientAccumulator:
        """Create gradient accumulator"""
        return GradientAccumulator(
            accumulation_steps=config.accumulation_steps,
            sync_frequency=config.sync_frequency
        )


def create_scheduler_config(scheduler_type: str = "cosine", **kwargs) -> SchedulerConfig:
    """Create scheduler configuration"""
    config_dict = {
        "scheduler_type": scheduler_type,
        **kwargs
    }
    
    return SchedulerConfig(**config_dict)


def benchmark_schedulers(model: torch.nn.Module,
                        optimizer: torch.optim.Optimizer,
                        scheduler_configs: List[SchedulerConfig],
                        num_steps: int = 1000) -> Dict[str, Dict[str, Any]]:
    """Benchmark different schedulers"""
    
    results = {}
    
    for config in scheduler_configs:
        scheduler_name = config.scheduler_type
        
        logger.info(f"Benchmarking {scheduler_name} scheduler...")
        
        # Create scheduler
        scheduler = SchedulerFactory.create_scheduler(scheduler_name, optimizer, config)
        
        # Create gradient accumulator
        accumulator = SchedulerFactory.create_gradient_accumulator(config)
        
        # Track learning rates
        lr_history = []
        
        # Simulate training steps
        for step in range(num_steps):
            # Get current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            lr_history.append(current_lr)
            
            # Step scheduler
            scheduler.step()
        
        # Store results
        results[scheduler_name] = {
            "lr_history": lr_history,
            "config": asdict(config),
            "min_lr": min(lr_history),
            "max_lr": max(lr_history),
            "avg_lr": sum(lr_history) / len(lr_history)
        }
        
        logger.info(f"{scheduler_name}: Min LR={min(lr_history):.2e}, Max LR={max(lr_history):.2e}")
    
    return results


def main():
    """Main function for testing schedulers"""
    
    # Create a simple model
    model = torch.nn.Linear(10, 5)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Test different schedulers
    scheduler_configs = [
        create_scheduler_config("cosine", total_steps=1000, min_lr=1e-6),
        create_scheduler_config("warmupcosine", total_steps=1000, warmup_steps=100, min_lr=1e-6),
        create_scheduler_config("onecycle", total_steps=1000, max_lr=1e-3),
        create_scheduler_config("cyclic", cycle_length=200, max_lr=1e-3, min_lr=1e-6),
    ]
    
    # Benchmark schedulers
    results = benchmark_schedulers(model, optimizer, scheduler_configs, num_steps=1000)
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    for scheduler_name, data in results.items():
        plt.plot(data['lr_history'], label=scheduler_name, linewidth=2)
    
    plt.xlabel('Training Step')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Scheduler Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.savefig('scheduler_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    logger.info("Scheduler comparison plot saved")


if __name__ == "__main__":
    main()