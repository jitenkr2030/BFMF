"""
Fine-tuning utilities for BharatFM models
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
import json
from pathlib import Path
import logging

from ..model import get_model_config
from .trainer import TrainingConfig


@dataclass
class LoRAConfig:
    """Configuration for LoRA fine-tuning"""
    
    # LoRA parameters
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: Optional[List[str]] = None
    bias: str = "none"
    
    # Training parameters
    learning_rate: float = 1e-4
    num_epochs: int = 3
    batch_size: int = 16
    warmup_steps: int = 100
    
    # Bharat-specific settings
    enable_indic_lora: bool = True
    language_specific_lora: bool = False
    domain_adapters: List[str] = None
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
        
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'LoRAConfig':
        """Create config from dictionary"""
        return cls(**config_dict)
        
    def save(self, path: str):
        """Save config to file"""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
            
    @classmethod
    def load(cls, path: str) -> 'LoRAConfig':
        """Load config from file"""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


class LoRALayer(nn.Module):
    """LoRA layer implementation"""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        merge_weights: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = nn.Dropout(lora_dropout)
        self.merged = False
        self.merge_weights = merge_weights
        
        # LoRA matrices
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        
        # Scaling factor
        self.scaling = lora_alpha / r
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.merged:
            return x @ self.weight
        
        # Original linear transformation
        result = x @ self.weight
        
        # LoRA transformation
        lora_result = (x @ self.lora_A.transpose(0, 1)) @ self.lora_B.transpose(0, 1)
        result += self.scaling * lora_result
        
        return result
        
    def merge(self):
        """Merge LoRA weights with original weights"""
        if not self.merged and self.merge_weights:
            self.weight.data += self.scaling * (self.lora_B @ self.lora_A)
            self.merged = True
            
    def unmerge(self):
        """Unmerge LoRA weights"""
        if self.merged and self.merge_weights:
            self.weight.data -= self.scaling * (self.lora_B @ self.lora_A)
            self.merged = False


class LoRALinear(nn.Module):
    """Linear layer with LoRA support"""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        bias: bool = True,
        **kwargs
    ):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.lora = LoRALayer(
            in_features=in_features,
            out_features=out_features,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lora(x)
        
    @property
    def weight(self):
        return self.linear.weight
        
    def merge_lora(self):
        """Merge LoRA weights"""
        self.lora.merge()
        
    def unmerge_lora(self):
        """Unmerge LoRA weights"""
        self.lora.unmerge()


class LoRAAttention(nn.Module):
    """Attention layer with LoRA support"""
    
    def __init__(
        self,
        original_attention: nn.Module,
        r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        target_modules: List[str] = None,
    ):
        super().__init__()
        self.original_attention = original_attention
        self.target_modules = target_modules or ["q_proj", "k_proj", "v_proj", "o_proj"]
        
        # Replace target modules with LoRA versions
        for name in self.target_modules:
            if hasattr(original_attention, name):
                original_layer = getattr(original_attention, name)
                
                # Create LoRA version
                lora_layer = LoRALinear(
                    in_features=original_layer.in_features,
                    out_features=original_layer.out_features,
                    r=r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    bias=original_layer.bias is not None,
                )
                
                # Copy weights
                lora_layer.linear.weight.data = original_layer.weight.data.clone()
                if original_layer.bias is not None:
                    lora_layer.linear.bias.data = original_layer.bias.data.clone()
                    
                setattr(self, name, lora_layer)
                
    def forward(self, *args, **kwargs):
        # Replace original layers with LoRA versions
        original_layers = {}
        for name in self.target_modules:
            if hasattr(self.original_attention, name):
                original_layers[name] = getattr(self.original_attention, name)
                setattr(self.original_attention, name, getattr(self, name))
                
        # Forward pass
        result = self.original_attention(*args, **kwargs)
        
        # Restore original layers
        for name, layer in original_layers.items():
            setattr(self.original_attention, name, layer)
            
        return result


class FineTuner:
    """Fine-tuning class for BharatFM models"""
    
    def __init__(
        self,
        base_model: str = "bharat-base",
        lora_config: Optional[LoRAConfig] = None,
        training_config: Optional[TrainingConfig] = None,
    ):
        self.base_model = base_model
        self.lora_config = lora_config or LoRAConfig()
        self.training_config = training_config or TrainingConfig()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Model and components
        self.model = None
        self.lora_layers = []
        
        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def setup_model(self):
        """Setup model with LoRA layers"""
        # Get model configuration
        model_config = get_model_config(self.base_model)
        
        # Create base model
        if model_config.model_type == "glm":
            from ..model import GLMForCausalLM
            self.model = GLMForCausalLM(model_config)
        elif model_config.model_type == "llama":
            from ..model import LlamaForCausalLM
            self.model = LlamaForCausalLM(model_config)
        elif model_config.model_type == "moe":
            from ..model import MoEForCausalLM
            self.model = MoEForCausalLM(model_config)
        else:
            raise ValueError(f"Unknown model type: {model_config.model_type}")
            
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Apply LoRA to target modules
        self.apply_lora()
        
        self.logger.info(f"Model setup complete: {self.base_model}")
        
    def apply_lora(self):
        """Apply LoRA to target modules"""
        target_modules = self.lora_config.target_modules
        
        if target_modules is None:
            # Default target modules for different model types
            if hasattr(self.model.config, 'model_type'):
                if self.model.config.model_type == "glm":
                    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
                elif self.model.config.model_type == "llama":
                    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
                elif self.model.config.model_type == "moe":
                    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
                else:
                    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
            else:
                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
                
        # Find and replace target modules
        for name, module in self.model.named_modules():
            for target in target_modules:
                if name.endswith(target):
                    # Get original layer
                    original_layer = getattr(module, target)
                    
                    # Create LoRA version
                    lora_layer = LoRALinear(
                        in_features=original_layer.in_features,
                        out_features=original_layer.out_features,
                        r=self.lora_config.r,
                        lora_alpha=self.lora_config.lora_alpha,
                        lora_dropout=self.lora_config.lora_dropout,
                        bias=original_layer.bias is not None,
                    )
                    
                    # Copy weights
                    lora_layer.linear.weight.data = original_layer.weight.data.clone()
                    if original_layer.bias is not None:
                        lora_layer.linear.bias.data = original_layer.bias.data.clone()
                        
                    # Replace layer
                    setattr(module, target, lora_layer)
                    
                    # Track LoRA layers
                    self.lora_layers.append(lora_layer)
                    
        # Freeze non-LoRA parameters
        self.freeze_non_lora_parameters()
        
        self.logger.info(f"Applied LoRA to {len(self.lora_layers)} layers")
        
    def freeze_non_lora_parameters(self):
        """Freeze all parameters except LoRA parameters"""
        for name, param in self.model.named_parameters():
            # Skip LoRA parameters
            if "lora_A" in name or "lora_B" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
                
    def get_trainable_parameters(self) -> List[torch.nn.Parameter]:
        """Get trainable parameters"""
        trainable_params = []
        for param in self.model.parameters():
            if param.requires_grad:
                trainable_params.append(param)
        return trainable_params
        
    def count_trainable_parameters(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
    def count_total_parameters(self) -> int:
        """Count total parameters"""
        return sum(p.numel() for p in self.model.parameters())
        
    def get_parameter_stats(self) -> Dict[str, int]:
        """Get parameter statistics"""
        trainable = self.count_trainable_parameters()
        total = self.count_total_parameters()
        percentage = (trainable / total) * 100
        
        return {
            "trainable_parameters": trainable,
            "total_parameters": total,
            "trainable_percentage": percentage
        }
        
    def setup_optimizer(self):
        """Setup optimizer for LoRA training"""
        trainable_params = self.get_trainable_parameters()
        
        # Separate parameters for weight decay
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for p in trainable_params if not any(nd in n for n, p in self.model.named_parameters() if p in trainable_params for nd in no_decay)],
                "weight_decay": self.training_config.weight_decay,
            },
            {
                "params": [p for p in trainable_params if any(nd in n for n, p in self.model.named_parameters() if p in trainable_params for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        
        # Create optimizer
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.lora_config.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        
        return optimizer
        
    def train(self, train_dataloader, eval_dataloader=None):
        """Train LoRA model"""
        self.logger.info("Starting LoRA training...")
        
        # Setup model
        self.setup_model()
        
        # Setup optimizer
        optimizer = self.setup_optimizer()
        
        # Print parameter statistics
        stats = self.get_parameter_stats()
        self.logger.info(f"Parameter statistics: {stats}")
        
        # Training loop
        self.model.train()
        
        for epoch in range(self.lora_config.num_epochs):
            epoch_loss = 0.0
            
            for batch_idx, batch in enumerate(train_dataloader):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(**batch)
                loss = outputs["loss"]
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # Optimizer step
                optimizer.step()
                optimizer.zero_grad()
                
                epoch_loss += loss.item()
                
                # Logging
                if batch_idx % 100 == 0:
                    avg_loss = epoch_loss / (batch_idx + 1)
                    self.logger.info(f"Epoch {epoch + 1}, Batch {batch_idx}: Loss = {avg_loss:.4f}")
                    
            # Epoch summary
            avg_epoch_loss = epoch_loss / len(train_dataloader)
            self.logger.info(f"Epoch {epoch + 1} complete: Average Loss = {avg_epoch_loss:.4f}")
            
            # Evaluation
            if eval_dataloader is not None:
                self.evaluate(eval_dataloader)
                
        self.logger.info("LoRA training complete!")
        
    def evaluate(self, eval_dataloader):
        """Evaluate LoRA model"""
        self.model.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in eval_dataloader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs["loss"]
                
                total_loss += loss.item()
                num_batches += 1
                
        avg_loss = total_loss / num_batches
        self.logger.info(f"Evaluation complete: Loss = {avg_loss:.4f}")
        
        return avg_loss
        
    def save_lora_weights(self, save_path: str):
        """Save LoRA weights"""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save LoRA weights
        lora_weights = {}
        for i, layer in enumerate(self.lora_layers):
            lora_weights[f"layer_{i}"] = {
                "lora_A": layer.lora.lora_A.data.cpu(),
                "lora_B": layer.lora.lora_B.data.cpu(),
                "scaling": layer.lora.scaling,
            }
            
        torch.save(lora_weights, save_path / "lora_weights.pt")
        
        # Save configuration
        self.lora_config.save(save_path / "lora_config.json")
        
        self.logger.info(f"LoRA weights saved to {save_path}")
        
    def load_lora_weights(self, load_path: str):
        """Load LoRA weights"""
        load_path = Path(load_path)
        
        # Load LoRA weights
        lora_weights = torch.load(load_path / "lora_weights.pt")
        
        # Load weights into layers
        for i, layer in enumerate(self.lora_layers):
            if f"layer_{i}" in lora_weights:
                weights = lora_weights[f"layer_{i}"]
                layer.lora.lora_A.data = weights["lora_A"].to(self.device)
                layer.lora.lora_B.data = weights["lora_B"].to(self.device)
                layer.lora.scaling = weights["scaling"]
                
        self.logger.info(f"LoRA weights loaded from {load_path}")
        
    def merge_lora_weights(self):
        """Merge LoRA weights with base model"""
        for layer in self.lora_layers:
            layer.merge_lora()
            
        self.logger.info("LoRA weights merged with base model")
        
    def unmerge_lora_weights(self):
        """Unmerge LoRA weights from base model"""
        for layer in self.lora_layers:
            layer.unmerge_lora()
            
        self.logger.info("LoRA weights unmerged from base model")