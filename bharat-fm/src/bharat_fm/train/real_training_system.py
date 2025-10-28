"""
Real Training System for Bharat-FM
Actual implementation with proper backpropagation, optimization, and training loops
"""

import os
import json
import time
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import get_linear_schedule_with_warmup
import numpy as np
from tqdm import tqdm

# Import our real neural networks
from ..core.real_neural_networks import (
    RealGPTStyleModel, 
    RealBERTStyleModel, 
    create_gpt_model, 
    create_bert_model
)

logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Configuration for training"""
    model_type: str = "gpt"  # "gpt" or "bert"
    vocab_size: int = 30000
    d_model: int = 768
    num_heads: int = 12
    num_layers: int = 12
    d_ff: int = 3072
    max_len: int = 1024
    dropout: float = 0.1
    
    # Training parameters
    batch_size: int = 8
    learning_rate: float = 5e-5
    num_epochs: int = 3
    warmup_steps: int = 1000
    weight_decay: float = 0.01
    gradient_clip_val: float = 1.0
    
    # Data parameters
    train_file: str = ""
    val_file: str = ""
    test_file: str = ""
    
    # Output parameters
    output_dir: str = "./training_output"
    save_steps: int = 1000
    eval_steps: int = 500
    logging_steps: int = 100
    
    # Hardware
    device: str = "auto"  # "auto", "cpu", "cuda"
    mixed_precision: bool = True

@dataclass
class TrainingMetrics:
    """Training metrics tracking"""
    epoch: int
    step: int
    train_loss: float
    val_loss: Optional[float] = None
    learning_rate: float = 0.0
    train_accuracy: Optional[float] = None
    val_accuracy: Optional[float] = None
    throughput: float = 0.0
    memory_usage: float = 0.0
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()

class TextDataset(Dataset):
    """Real dataset for text training"""
    
    def __init__(self, 
                 texts: List[str], 
                 tokenizer, 
                 max_length: int = 512,
                 model_type: str = "gpt"):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.model_type = model_type
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        if self.model_type == "gpt":
            # For GPT-style models, we need input_ids and labels (shifted)
            encoding = self.tokenizer(
                text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            input_ids = encoding["input_ids"].squeeze()
            attention_mask = encoding["attention_mask"].squeeze()
            
            # For language modeling, labels are shifted input_ids
            labels = input_ids.clone()
            labels[:-1] = input_ids[1:]  # Shift left
            labels[-1] = -100  # Ignore last token
            
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels
            }
        
        elif self.model_type == "bert":
            # For BERT-style models, we need masked language modeling
            encoding = self.tokenizer(
                text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            input_ids = encoding["input_ids"].squeeze()
            attention_mask = encoding["attention_mask"].squeeze()
            
            # Create random mask for MLM (15% masking)
            mask = torch.rand(input_ids.shape) < 0.15
            labels = input_ids.clone()
            
            # Don't mask special tokens
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) 
                for val in input_ids.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
            mask = mask & ~special_tokens_mask
            
            # Mask tokens
            input_ids[mask] = self.tokenizer.mask_token_id
            
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels
            }
        
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

class RealTrainingSystem:
    """Real training system with proper optimization"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.train_dataloader = None
        self.val_dataloader = None
        self.test_dataloader = None
        
        # Training state
        self.current_epoch = 0
        self.current_step = 0
        self.best_val_loss = float('inf')
        self.training_history: List[TrainingMetrics] = []
        
        # Device setup
        self.device = self._setup_device()
        
        # Mixed precision
        self.scaler = None
        if config.mixed_precision and self.device.type == 'cuda':
            self.scaler = torch.cuda.amp.GradScaler()
        
        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
    
    def _setup_device(self) -> torch.device:
        """Setup device for training"""
        if self.config.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(self.config.device)
        
        logger.info(f"Using device: {device}")
        return device
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_file = os.path.join(self.config.output_dir, "training.log")
        
        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(file_handler)
        
        logger.info(f"Training system initialized. Output directory: {self.config.output_dir}")
    
    def initialize_model(self):
        """Initialize the model"""
        logger.info(f"Initializing {self.config.model_type} model...")
        
        if self.config.model_type == "gpt":
            self.model = create_gpt_model(
                vocab_size=self.config.vocab_size,
                d_model=self.config.d_model,
                num_heads=self.config.num_heads,
                num_layers=self.config.num_layers
            )
        elif self.config.model_type == "bert":
            self.model = create_bert_model(
                vocab_size=self.config.vocab_size,
                d_model=self.config.d_model,
                num_heads=self.config.num_heads,
                num_layers=self.config.num_layers
            )
        else:
            raise ValueError(f"Unsupported model type: {self.config.model_type}")
        
        self.model.to(self.device)
        logger.info(f"Model initialized with {sum(p.numel() for p in self.model.parameters())} parameters")
    
    def initialize_optimizer(self):
        """Initialize optimizer and scheduler"""
        if self.model is None:
            raise RuntimeError("Model must be initialized before optimizer")
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Initialize scheduler
        total_steps = len(self.train_dataloader) * self.config.num_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=total_steps
        )
        
        # Initialize loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        
        logger.info("Optimizer and scheduler initialized")
    
    def load_data(self, train_texts: List[str], val_texts: List[str] = None, test_texts: List[str] = None):
        """Load training data"""
        logger.info("Loading training data...")
        
        # Create dummy tokenizer for now (in production, use real tokenizer)
        class DummyTokenizer:
            def __init__(self, vocab_size=30000):
                self.vocab_size = vocab_size
                self.mask_token_id = 4
            
            def __call__(self, text, max_length=512, padding="max_length", truncation=True, return_tensors="pt"):
                # Simple tokenization for demo
                tokens = list(text.lower()[:max_length])
                token_ids = [ord(c) % self.vocab_size for c in tokens]
                
                # Pad to max_length
                if len(token_ids) < max_length:
                    token_ids.extend([0] * (max_length - len(token_ids)))
                else:
                    token_ids = token_ids[:max_length]
                
                attention_mask = [1 if t != 0 else 0 for t in token_ids]
                
                return {
                    "input_ids": torch.tensor(token_ids),
                    "attention_mask": torch.tensor(attention_mask)
                }
            
            def get_special_tokens_mask(self, token_ids, already_has_special_tokens=True):
                return [0] * len(token_ids)
        
        tokenizer = DummyTokenizer(self.config.vocab_size)
        
        # Create datasets
        self.train_dataloader = DataLoader(
            TextDataset(train_texts, tokenizer, self.config.max_len, self.config.model_type),
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=2
        )
        
        if val_texts:
            self.val_dataloader = DataLoader(
                TextDataset(val_texts, tokenizer, self.config.max_len, self.config.model_type),
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=2
            )
        
        if test_texts:
            self.test_dataloader = DataLoader(
                TextDataset(test_texts, tokenizer, self.config.max_len, self.config.model_type),
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=2
            )
        
        logger.info(f"Data loaded - Train: {len(train_texts)}, Val: {len(val_texts) if val_texts else 0}, Test: {len(test_texts) if test_texts else 0}")
    
    def train_epoch(self) -> TrainingMetrics:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_tokens = 0
        
        epoch_start_time = time.time()
        
        progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass
            if self.config.mixed_precision and self.scaler:
                with torch.cuda.amp.autocast():
                    outputs = self.model(**batch)
                    loss = self.criterion(outputs.view(-1, outputs.size(-1)), batch["labels"].view(-1))
                
                # Backward pass with mixed precision
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config.gradient_clip_val > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_val)
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard training
                outputs = self.model(**batch)
                loss = self.criterion(outputs.view(-1, outputs.size(-1)), batch["labels"].view(-1))
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                if self.config.gradient_clip_val > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_val)
                
                # Optimizer step
                self.optimizer.step()
            
            # Scheduler step
            self.scheduler.step()
            self.optimizer.zero_grad()
            
            # Update metrics
            total_loss += loss.item()
            
            # Calculate accuracy (for non-masked tokens)
            if self.config.model_type == "bert":
                mask = batch["labels"] != -100
                predictions = torch.argmax(outputs, dim=-1)
                correct = (predictions[mask] == batch["labels"][mask]).sum().item()
                total_correct += correct
                total_tokens += mask.sum().item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': loss.item(),
                'lr': self.scheduler.get_last_lr()[0]
            })
            
            # Update step counter
            self.current_step += 1
            
            # Logging
            if self.current_step % self.config.logging_steps == 0:
                self._log_step_metrics(loss.item())
            
            # Evaluation
            if self.val_dataloader and self.current_step % self.config.eval_steps == 0:
                val_metrics = self.evaluate()
                self._log_validation_metrics(val_metrics)
            
            # Save checkpoint
            if self.current_step % self.config.save_steps == 0:
                self.save_checkpoint()
        
        # Calculate epoch metrics
        avg_loss = total_loss / len(self.train_dataloader)
        accuracy = total_correct / total_tokens if total_tokens > 0 else 0
        throughput = len(self.train_dataloader.dataset) / (time.time() - epoch_start_time)
        memory_usage = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
        
        metrics = TrainingMetrics(
            epoch=self.current_epoch,
            step=self.current_step,
            train_loss=avg_loss,
            learning_rate=self.scheduler.get_last_lr()[0],
            train_accuracy=accuracy,
            throughput=throughput,
            memory_usage=memory_usage
        )
        
        self.training_history.append(metrics)
        logger.info(f"Epoch {self.current_epoch} completed - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        
        return metrics
    
    def evaluate(self) -> TrainingMetrics:
        """Evaluate the model"""
        if not self.val_dataloader:
            return None
        
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Evaluating"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                outputs = self.model(**batch)
                loss = self.criterion(outputs.view(-1, outputs.size(-1)), batch["labels"].view(-1))
                
                total_loss += loss.item()
                
                # Calculate accuracy
                if self.config.model_type == "bert":
                    mask = batch["labels"] != -100
                    predictions = torch.argmax(outputs, dim=-1)
                    correct = (predictions[mask] == batch["labels"][mask]).sum().item()
                    total_correct += correct
                    total_tokens += mask.sum().item()
        
        avg_loss = total_loss / len(self.val_dataloader)
        accuracy = total_correct / total_tokens if total_tokens > 0 else 0
        
        metrics = TrainingMetrics(
            epoch=self.current_epoch,
            step=self.current_step,
            val_loss=avg_loss,
            val_accuracy=accuracy
        )
        
        return metrics
    
    def train(self):
        """Main training loop"""
        logger.info("Starting training...")
        
        # Initialize model and optimizer
        self.initialize_model()
        self.initialize_optimizer()
        
        # Training loop
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            
            # Train epoch
            train_metrics = self.train_epoch()
            
            # Evaluate
            if self.val_dataloader:
                val_metrics = self.evaluate()
                if val_metrics and val_metrics.val_loss < self.best_val_loss:
                    self.best_val_loss = val_metrics.val_loss
                    self.save_checkpoint(best=True)
            
            # Save epoch checkpoint
            self.save_checkpoint()
            
            # Log epoch summary
            self._log_epoch_summary(train_metrics)
        
        logger.info("Training completed!")
        self.save_final_model()
    
    def save_checkpoint(self, best: bool = False):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'step': self.current_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': asdict(self.config),
            'training_history': [asdict(m) for m in self.training_history]
        }
        
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        checkpoint_path = os.path.join(
            self.config.output_dir,
            f"checkpoint_{'best_' if best else ''}epoch_{self.current_epoch}_step_{self.current_step}.pt"
        )
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def save_final_model(self):
        """Save final model"""
        final_model_path = os.path.join(self.config.output_dir, "final_model.pt")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': asdict(self.config),
            'training_history': [asdict(m) for m in self.training_history]
        }, final_model_path)
        
        logger.info(f"Final model saved: {final_model_path}")
    
    def _log_step_metrics(self, loss: float):
        """Log step metrics"""
        logger.info(f"Step {self.current_step} - Loss: {loss:.4f}, LR: {self.scheduler.get_last_lr()[0]:.2e}")
    
    def _log_validation_metrics(self, metrics: TrainingMetrics):
        """Log validation metrics"""
        logger.info(f"Validation - Loss: {metrics.val_loss:.4f}, Accuracy: {metrics.val_accuracy:.4f}")
    
    def _log_epoch_summary(self, metrics: TrainingMetrics):
        """Log epoch summary"""
        logger.info(f"Epoch {self.current_epoch} Summary:")
        logger.info(f"  Train Loss: {metrics.train_loss:.4f}")
        logger.info(f"  Train Accuracy: {metrics.train_accuracy:.4f}")
        logger.info(f"  Throughput: {metrics.throughput:.2f} samples/sec")
        logger.info(f"  Memory Usage: {metrics.memory_usage:.2f} GB")
        
        if metrics.val_loss:
            logger.info(f"  Val Loss: {metrics.val_loss:.4f}")
            logger.info(f"  Val Accuracy: {metrics.val_accuracy:.4f}")


# Factory function for easy usage
def create_real_training_system(config: TrainingConfig) -> RealTrainingSystem:
    """Create a real training system"""
    return RealTrainingSystem(config)