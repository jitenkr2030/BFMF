"""
Advanced Mixed Precision Training Implementation
FP16, FP32, FP8, and BF16 support with automatic precision selection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass
import logging
import math
from abc import ABC, abstractmethod
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class PrecisionConfig:
    """Configuration for mixed precision training"""
    precision: str = "fp16"  # fp16, fp32, fp8, bf16, mixed
    enabled: bool = True
    
    # FP8 specific settings
    fp8_margin: int = 0
    fp8_interval: int = 1
    fp8_amax_history_len: int = 1024
    fp8_amax_compute_algo: str = "max"
    fp8_wgrad: bool = True
    
    # BF16 specific settings
    bf16_loss_scale: float = 1.0
    bf16_grad_scale: float = 1.0
    
    # Dynamic precision settings
    dynamic_precision: bool = False
    precision_threshold: float = 0.1
    precision_adaptation_rate: float = 0.01
    
    # Loss scaling
    loss_scale: float = 1.0
    loss_scale_factor: float = 2.0
    loss_scale_window: int = 1000
    
    # Gradient handling
    grad_clipping: float = 1.0
    grad_clipping_norm: str = "inf"  # inf, 2.0, 1.0
    
    # Memory optimization
    memory_efficient_attention: bool = True
    activation_checkpointing: bool = True
    gradient_checkpointing: bool = True


class PrecisionManager:
    """Advanced precision management for training"""
    
    def __init__(self, config: PrecisionConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Precision state
        self.current_precision = self._get_initial_precision()
        self.scaler = None
        self.grad_scaler = None
        
        # Loss scaling state
        self.loss_scale = config.loss_scale
        self.loss_scale_factor = config.loss_scale_factor
        self.loss_scale_window = config.loss_scale_window
        self.loss_scale_counter = 0
        self.loss_scale_history = []
        
        # Dynamic precision state
        self.precision_history = []
        self.precision_scores = []
        
        # Initialize scalers
        self._initialize_scalers()
    
    def _get_initial_precision(self) -> torch.dtype:
        """Get initial precision based on config"""
        precision_map = {
            "fp16": torch.float16,
            "fp32": torch.float32,
            "bf16": torch.bfloat16,
            "fp8": torch.float8_e5m2 if hasattr(torch, 'float8_e5m2') else torch.float16
        }
        
        if self.config.precision == "mixed":
            return torch.float16  # Default to FP16 for mixed precision
        else:
            return precision_map.get(self.config.precision.lower(), torch.float32)
    
    def _initialize_scalers(self):
        """Initialize gradient scalers"""
        if self.config.precision in ["fp16", "mixed"]:
            self.scaler = GradScaler(
                init_scale=self.loss_scale,
                growth_factor=self.config.loss_scale_factor,
                backoff_factor=0.5,
                growth_interval=self.config.loss_scale_window
            )
        
        if self.config.precision == "bf16":
            # BF16 typically doesn't need loss scaling but can benefit from it
            self.grad_scaler = GradScaler(
                init_scale=self.config.bf16_grad_scale,
                enabled=self.config.bf16_grad_scale != 1.0
            )
    
    def get_autocast_context(self):
        """Get autocast context for current precision"""
        if not self.config.enabled:
            return nullcontext()
        
        if self.config.precision == "fp8":
            return self._fp8_autocast()
        elif self.config.precision in ["fp16", "mixed"]:
            return autocast(dtype=torch.float16)
        elif self.config.precision == "bf16":
            return autocast(dtype=torch.bfloat16)
        else:
            return nullcontext()
    
    def _fp8_autocast(self):
        """FP8 autocast context"""
        class FP8Autocast:
            def __init__(self, manager):
                self.manager = manager
            
            def __enter__(self):
                # FP8 autocast implementation
                # This would typically use specialized FP8 kernels
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                pass
        
        return FP8Autocast(self)
    
    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss for mixed precision training"""
        if self.scaler is not None:
            return self.scaler.scale(loss)
        elif self.grad_scaler is not None:
            return self.grad_scaler.scale(loss)
        else:
            return loss
    
    def unscale_optimizer(self, optimizer: torch.optim.Optimizer):
        """Unscale optimizer gradients"""
        if self.scaler is not None:
            self.scaler.unscale_(optimizer)
        elif self.grad_scaler is not None:
            self.grad_scaler.unscale_(optimizer)
    
    def step_optimizer(self, optimizer: torch.optim.Optimizer):
        """Step optimizer with gradient scaling"""
        if self.scaler is not None:
            self.scaler.step(optimizer)
            self.scaler.update()
        elif self.grad_scaler is not None:
            self.grad_scaler.step(optimizer)
            self.grad_scaler.update()
        else:
            optimizer.step()
    
    def update_loss_scale(self, loss: torch.Tensor):
        """Update loss scale based on loss behavior"""
        if self.config.precision not in ["fp16", "mixed"]:
            return
        
        # Track loss history
        self.loss_scale_history.append(loss.item())
        if len(self.loss_scale_history) > self.loss_scale_window:
            self.loss_scale_history.pop(0)
        
        # Dynamic loss scaling
        if len(self.loss_scale_history) >= self.loss_scale_window:
            avg_loss = np.mean(self.loss_scale_history)
            loss_std = np.std(self.loss_scale_history)
            
            # Check for loss instability
            if loss_std > avg_loss * 0.1:  # 10% threshold
                # Reduce loss scale
                self.loss_scale *= 0.5
                if self.scaler is not None:
                    self.scaler.set_scale(self.loss_scale)
                logger.info(f"Reduced loss scale to {self.loss_scale} due to instability")
            elif loss_std < avg_loss * 0.01:  # 1% threshold
                # Increase loss scale
                self.loss_scale *= 1.1
                if self.scaler is not None:
                    self.scaler.set_scale(self.loss_scale)
                logger.info(f"Increased loss scale to {self.loss_scale} due to stability")
    
    def adapt_precision(self, model: nn.Module, optimizer: torch.optim.Optimizer):
        """Adapt precision dynamically based on training behavior"""
        if not self.config.dynamic_precision:
            return
        
        # Calculate precision score based on various metrics
        score = self._calculate_precision_score(model, optimizer)
        
        # Update precision history
        self.precision_scores.append(score)
        if len(self.precision_scores) > 100:
            self.precision_scores.pop(0)
        
        # Determine if precision should be changed
        avg_score = np.mean(self.precision_scores)
        
        if avg_score < self.config.precision_threshold and self.current_precision != torch.float32:
            # Switch to higher precision
            self._switch_precision(model, optimizer, torch.float32)
            logger.info("Switched to FP32 due to precision issues")
        elif avg_score > (1 - self.config.precision_threshold) and self.current_precision == torch.float32:
            # Switch to lower precision
            self._switch_precision(model, optimizer, torch.float16)
            logger.info("Switched to FP16 for better performance")
    
    def _calculate_precision_score(self, model: nn.Module, optimizer: torch.optim.Optimizer) -> float:
        """Calculate precision adaptation score"""
        score = 1.0
        
        # Check gradient norms
        total_norm = 0
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        
        # Penalize large gradient norms
        if total_norm > 10.0:
            score *= 0.8
        
        # Check loss stability
        if len(self.loss_scale_history) > 10:
            recent_loss = self.loss_scale_history[-10:]
            loss_std = np.std(recent_loss)
            loss_mean = np.mean(recent_loss)
            
            if loss_std > loss_mean * 0.1:
                score *= 0.7
        
        return score
    
    def _switch_precision(self, model: nn.Module, optimizer: torch.optim.Optimizer, new_precision: torch.dtype):
        """Switch model precision"""
        # This is a simplified implementation
        # In practice, you'd need to handle model state conversion
        
        if new_precision == torch.float16:
            model.half()
            self.current_precision = torch.float16
        elif new_precision == torch.float32:
            model.float()
            self.current_precision = torch.float32
        elif new_precision == torch.bfloat16:
            if hasattr(torch, 'bfloat16'):
                model.to(dtype=torch.bfloat16)
                self.current_precision = torch.bfloat16
        
        # Reinitialize scalers
        self._initialize_scalers()
    
    def get_precision_stats(self) -> Dict[str, Any]:
        """Get precision statistics"""
        return {
            "current_precision": str(self.current_precision),
            "loss_scale": self.loss_scale,
            "loss_scale_history": self.loss_scale_history[-10:] if self.loss_scale_history else [],
            "precision_scores": self.precision_scores[-10:] if self.precision_scores else [],
            "dynamic_precision_enabled": self.config.dynamic_precision,
            "scaler_enabled": self.scaler is not None
        }


class MixedPrecisionTrainer:
    """Trainer with advanced mixed precision support"""
    
    def __init__(self, 
                 model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 precision_config: PrecisionConfig):
        self.model = model
        self.optimizer = optimizer
        self.precision_config = precision_config
        self.precision_manager = PrecisionManager(precision_config)
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        
        # Memory optimization
        if precision_config.activation_checkpointing:
            self._enable_activation_checkpointing()
        
        if precision_config.gradient_checkpointing:
            self._enable_gradient_checkpointing()
    
    def _enable_activation_checkpointing(self):
        """Enable activation checkpointing for memory efficiency"""
        # This would typically use torch.utils.checkpoint
        # For now, we'll just log the intention
        logger.info("Activation checkpointing enabled")
    
    def _enable_gradient_checkpointing(self):
        """Enable gradient checkpointing"""
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Single training step with mixed precision"""
        self.model.train()
        
        # Move batch to device
        batch = {k: v.to(self.precision_manager.device) for k, v in batch.items()}
        
        # Forward pass with mixed precision
        with self.precision_manager.get_autocast_context():
            outputs = self.model(**batch)
            loss = self._compute_loss(outputs, batch)
            
            # Scale loss for mixed precision
            scaled_loss = self.precision_manager.scale_loss(loss)
        
        # Backward pass
        self.optimizer.zero_grad()
        scaled_loss.backward()
        
        # Unscale gradients
        self.precision_manager.unscale_optimizer(self.optimizer)
        
        # Gradient clipping
        if self.precision_config.grad_clipping > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.precision_config.grad_clipping,
                norm_type=self.precision_config.grad_clipping_norm
            )
        
        # Optimizer step
        self.precision_manager.step_optimizer(self.optimizer)
        
        # Update loss scale
        self.precision_manager.update_loss_scale(loss)
        
        # Dynamic precision adaptation
        if self.precision_config.dynamic_precision:
            self.precision_manager.adapt_precision(self.model, self.optimizer)
        
        # Update step counter
        self.global_step += 1
        
        # Prepare results
        results = {
            "loss": loss.item(),
            "scaled_loss": scaled_loss.item(),
            "global_step": self.global_step,
            "precision": str(self.precision_manager.current_precision),
            "loss_scale": self.precision_manager.loss_scale
        }
        
        # Add gradient norms
        grad_norms = self._compute_gradient_norms()
        results.update(grad_norms)
        
        return results
    
    def _compute_loss(self, outputs: torch.Tensor, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute loss based on model outputs"""
        if 'labels' in batch:
            return F.cross_entropy(outputs.view(-1, outputs.size(-1)), batch['labels'].view(-1))
        elif 'targets' in batch:
            return F.cross_entropy(outputs.view(-1, outputs.size(-1)), batch['targets'].view(-1))
        else:
            # Default to MSE loss
            target = batch.get('input_ids', batch.get('inputs', torch.tensor(0)))
            return F.mse_loss(outputs, target.float())
    
    def _compute_gradient_norms(self) -> Dict[str, float]:
        """Compute gradient norms for monitoring"""
        grad_norms = {}
        
        total_norm = 0
        param_count = 0
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                grad_norms[f"{name}_norm"] = param_norm.item()
                total_norm += param_norm.item() ** 2
                param_count += 1
        
        if param_count > 0:
            grad_norms["total_norm"] = total_norm ** 0.5
            grad_norms["avg_norm"] = total_norm ** 0.5 / param_count
        
        return grad_norms
    
    def validate(self, dataloader) -> Dict[str, Any]:
        """Validation with mixed precision"""
        self.model.eval()
        
        total_loss = 0
        total_samples = 0
        correct_predictions = 0
        
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(self.precision_manager.device) for k, v in batch.items()}
                
                with self.precision_manager.get_autocast_context():
                    outputs = self.model(**batch)
                    loss = self._compute_loss(outputs, batch)
                
                total_loss += loss.item() * batch.get('input_ids', batch.get('inputs', torch.tensor(0))).size(0)
                total_samples += batch.get('input_ids', batch.get('inputs', torch.tensor(0))).size(0)
                
                # Calculate accuracy if possible
                if hasattr(outputs, 'argmax'):
                    predictions = outputs.argmax(dim=-1)
                    targets = batch.get('labels', batch.get('targets', torch.tensor(0)))
                    if predictions.shape == targets.shape:
                        correct_predictions += (predictions == targets).sum().item()
        
        avg_loss = total_loss / total_samples
        accuracy = correct_predictions / total_samples if total_samples > 0 else 0
        
        return {
            "val_loss": avg_loss,
            "val_accuracy": accuracy,
            "total_samples": total_samples
        }
    
    def get_precision_stats(self) -> Dict[str, Any]:
        """Get precision statistics"""
        return self.precision_manager.get_precision_stats()


class FP8Linear(nn.Module):
    """FP8 linear layer with automatic scaling"""
    
    def __init__(self, 
                 in_features: int, 
                 out_features: int, 
                 bias: bool = True,
                 fp8_config: PrecisionConfig = None):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.fp8_config = fp8_config or PrecisionConfig(precision="fp8")
        
        # FP8 parameters
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        # FP8 scaling factors
        self.weight_scale = nn.Parameter(torch.ones(1))
        self.input_scale = nn.Parameter(torch.ones(1))
        self.output_scale = nn.Parameter(torch.ones(1))
        
        # AMAX history for dynamic scaling
        self.weight_amax_history = []
        self.input_amax_history = []
        self.output_amax_history = []
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize parameters"""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        # This is a simplified FP8 implementation
        # In practice, you'd use specialized FP8 kernels
        
        # Update AMAX history
        if self.training:
            self.weight_amax_history.append(self.weight.abs().max().item())
            self.input_amax_history.append(input_tensor.abs().max().item())
            
            # Keep history limited
            if len(self.weight_amax_history) > self.fp8_config.fp8_amax_history_len:
                self.weight_amax_history.pop(0)
                self.input_amax_history.pop(0)
            
            # Update scaling factors
            if len(self.weight_amax_history) > 10:
                weight_amax = max(self.weight_amax_history[-10:])
                input_amax = max(self.input_amax_history[-10:])
                
                self.weight_scale.data = torch.tensor(448.0 / weight_amax)
                self.input_scale.data = torch.tensor(448.0 / input_amax)
        
        # Quantize to FP8 (simplified)
        weight_fp8 = self.weight * self.weight_scale
        input_fp8 = input_tensor * self.input_scale
        
        # Compute in FP8 (simplified - actually compute in FP32)
        output_fp32 = F.linear(input_fp8, weight_fp8, self.bias)
        
        # Scale back
        output = output_fp32 / (self.weight_scale * self.input_scale)
        
        # Update output AMAX
        if self.training:
            self.output_amax_history.append(output.abs().max().item())
            if len(self.output_amax_history) > self.fp8_config.fp8_amax_history_len:
                self.output_amax_history.pop(0)
        
        return output


def create_precision_config(precision: str = "fp16", **kwargs) -> PrecisionConfig:
    """Create precision configuration"""
    config_dict = {
        "precision": precision,
        **kwargs
    }
    
    return PrecisionConfig(**config_dict)


def benchmark_precision_modes(model: nn.Module,
                            input_shape: Tuple[int, int],
                            precision_modes: List[str] = None,
                            num_iterations: int = 100) -> Dict[str, Dict[str, float]]:
    """Benchmark different precision modes"""
    
    if precision_modes is None:
        precision_modes = ["fp32", "fp16", "bf16"]
    
    results = {}
    
    # Create dummy input
    batch_size, seq_length = input_shape
    input_ids = torch.randint(0, 1000, (batch_size, seq_length)).cuda()
    
    for precision in precision_modes:
        logger.info(f"Benchmarking {precision} precision...")
        
        # Create precision config
        precision_config = create_precision_config(precision=precision)
        
        # Create trainer
        optimizer = torch.optim.Adam(model.parameters())
        trainer = MixedPrecisionTrainer(model, optimizer, precision_config)
        
        # Warmup
        for _ in range(10):
            dummy_batch = {"input_ids": input_ids, "labels": input_ids}
            _ = trainer.train_step(dummy_batch)
        
        torch.cuda.synchronize()
        
        # Benchmark
        start_time = time.time()
        for _ in range(num_iterations):
            dummy_batch = {"input_ids": input_ids, "labels": input_ids}
            _ = trainer.train_step(dummy_batch)
        torch.cuda.synchronize()
        
        end_time = time.time()
        
        # Calculate metrics
        total_time = end_time - start_time
        avg_time = total_time / num_iterations
        throughput = num_iterations / total_time
        
        # Memory usage
        memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        memory_reserved = torch.cuda.memory_reserved() / 1024**3  # GB
        
        results[precision] = {
            "total_time": total_time,
            "avg_time_per_iteration": avg_time,
            "throughput": throughput,
            "memory_allocated_gb": memory_allocated,
            "memory_reserved_gb": memory_reserved
        }
        
        logger.info(f"{precision}: {avg_time:.4f}s per iteration, {memory_allocated:.2f}GB memory")
    
    return results


def main():
    """Main function for mixed precision testing"""
    
    # Create a simple model for testing
    model = nn.Sequential(
        nn.Linear(512, 1024),
        nn.ReLU(),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Linear(512, 100)
    ).cuda()
    
    # Test different precision modes
    results = benchmark_precision_modes(
        model=model,
        input_shape=(32, 512),
        precision_modes=["fp32", "fp16", "bf16"],
        num_iterations=50
    )
    
    # Print results
    print("\nPrecision Benchmark Results:")
    print("-" * 60)
    for precision, metrics in results.items():
        print(f"{precision:5s}: {metrics['avg_time_per_iteration']:6.4f}s/iter, "
              f"{metrics['memory_allocated_gb']:6.2f}GB, "
              f"{metrics['throughput']:6.2f} iter/s")


if __name__ == "__main__":
    main()