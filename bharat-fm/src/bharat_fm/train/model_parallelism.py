"""
Model Parallelism Implementation for Large Models
Pipeline parallelism, tensor parallelism, and hybrid parallelism for Bharat-FM
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed import rpc
from torch.distributed.pipeline.sync import Pipe
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass
import logging
import math
import time
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

@dataclass
class ModelParallelConfig:
    """Configuration for model parallelism"""
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    data_parallel_size: int = 1
    
    # Tensor parallelism settings
    tensor_parallel_mode: str = "column"  # column, row, both
    attention_parallel: bool = True
    mlp_parallel: bool = True
    
    # Pipeline parallelism settings
    chunks: int = 1
    checkpoint: bool = True
    balance: List[int] = None
    
    # Communication settings
    overlap_comm: bool = True
    comm_backend: str = "nccl"
    
    # Memory optimization
    memory_efficient: bool = True
    offload_params: bool = False


class TensorParallelLinear(nn.Module):
    """Tensor parallel linear layer"""
    
    def __init__(self, 
                 in_features: int, 
                 out_features: int, 
                 parallel_mode: str = "column",
                 bias: bool = True,
                 gather_output: bool = True):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.parallel_mode = parallel_mode
        self.gather_output = gather_output
        
        # Get distributed info
        self.world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        self.rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        
        if self.world_size == 1:
            # Single GPU, no parallelism
            self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
            if bias:
                self.bias = nn.Parameter(torch.Tensor(out_features))
            else:
                self.register_parameter('bias', None)
        else:
            # Tensor parallel setup
            if parallel_mode == "column":
                # Column parallel: split output dimension
                self.output_size_per_partition = out_features // self.world_size
                self.weight = nn.Parameter(torch.Tensor(self.output_size_per_partition, in_features))
                if bias:
                    self.bias = nn.Parameter(torch.Tensor(self.output_size_per_partition))
                else:
                    self.register_parameter('bias', None)
            elif parallel_mode == "row":
                # Row parallel: split input dimension
                self.input_size_per_partition = in_features // self.world_size
                self.weight = nn.Parameter(torch.Tensor(out_features, self.input_size_per_partition))
                if bias and self.rank == 0:  # Only rank 0 has bias in row parallel
                    self.bias = nn.Parameter(torch.Tensor(out_features))
                else:
                    self.register_parameter('bias', None)
                    self.bias = None
            else:
                raise ValueError(f"Unknown parallel mode: {parallel_mode}")
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize parameters"""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        if self.world_size == 1:
            # Single GPU forward pass
            return F.linear(input_tensor, self.weight, self.bias)
        
        if self.parallel_mode == "column":
            # Column parallel forward
            output_parallel = F.linear(input_tensor, self.weight, self.bias)
            
            if self.gather_output:
                # Gather outputs from all processes
                output_list = [torch.empty_like(output_parallel) for _ in range(self.world_size)]
                torch.distributed.all_gather(output_list, output_parallel)
                output = torch.cat(output_list, dim=-1)
            else:
                output = output_parallel
            
            return output
        
        elif self.parallel_mode == "row":
            # Row parallel forward
            # First, split input and process local part
            input_parallel = self._split_input(input_tensor)
            output_parallel = F.linear(input_parallel, self.weight, bias=None)
            
            # All-reduce to combine results
            torch.distributed.all_reduce(output_parallel, op=torch.distributed.ReduceOp.SUM)
            
            # Add bias if exists
            if self.bias is not None:
                output = output_parallel + self.bias
            else:
                output = output_parallel
            
            return output
    
    def _split_input(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Split input tensor for row parallel"""
        split_size = input_tensor.size(-1) // self.world_size
        return input_tensor.split(split_size, dim=-1)[self.rank]


class TensorParallelAttention(nn.Module):
    """Tensor parallel multi-head attention"""
    
    def __init__(self, 
                 hidden_size: int, 
                 num_heads: int,
                 dropout: float = 0.1):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        # Tensor parallel projections
        self.query_proj = TensorParallelLinear(
            hidden_size, hidden_size, 
            parallel_mode="column", 
            gather_output=False
        )
        self.key_proj = TensorParallelLinear(
            hidden_size, hidden_size, 
            parallel_mode="column", 
            gather_output=False
        )
        self.value_proj = TensorParallelLinear(
            hidden_size, hidden_size, 
            parallel_mode="column", 
            gather_output=False
        )
        self.out_proj = TensorParallelLinear(
            hidden_size, hidden_size, 
            parallel_mode="row", 
            gather_output=True
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, 
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        batch_size, seq_length, _ = hidden_states.shape
        
        # Project queries, keys, values
        query_states = self.query_proj(hidden_states)
        key_states = self.key_proj(hidden_states)
        value_states = self.value_proj(hidden_states)
        
        # Reshape for multi-head attention
        query_states = query_states.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        attn_scores = torch.matmul(query_states, key_states.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply attention mask
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask
        
        # Softmax and dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Compute output
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_length, self.hidden_size)
        
        # Final projection
        output = self.out_proj(attn_output)
        
        return output


class TensorParallelMLP(nn.Module):
    """Tensor parallel MLP layer"""
    
    def __init__(self, 
                 hidden_size: int, 
                 intermediate_size: int,
                 activation: str = "gelu"):
        super().__init__()
        
        self.activation = activation
        
        # Tensor parallel layers
        self.dense_h_to_4h = TensorParallelLinear(
            hidden_size, intermediate_size,
            parallel_mode="column",
            gather_output=False
        )
        self.dense_4h_to_h = TensorParallelLinear(
            intermediate_size, hidden_size,
            parallel_mode="row",
            gather_output=True
        )
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # First linear transformation
        hidden_parallel = self.dense_h_to_4h(hidden_states)
        
        # Activation
        if self.activation == "gelu":
            hidden_parallel = F.gelu(hidden_parallel)
        elif self.activation == "relu":
            hidden_parallel = F.relu(hidden_parallel)
        elif self.activation == "silu":
            hidden_parallel = F.silu(hidden_parallel)
        else:
            hidden_parallel = F.gelu(hidden_parallel)
        
        # Second linear transformation
        output = self.dense_4h_to_h(hidden_parallel)
        
        return output


class PipelineParallelBlock(nn.Module):
    """Pipeline parallel transformer block"""
    
    def __init__(self, 
                 hidden_size: int,
                 num_heads: int,
                 intermediate_size: int,
                 dropout: float = 0.1,
                 layer_idx: int = 0):
        super().__init__()
        
        self.layer_idx = layer_idx
        
        # Attention layer
        self.attention = TensorParallelAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # MLP layer
        self.mlp = TensorParallelMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size
        )
        
        # Normalization layers
        self.input_layernorm = nn.LayerNorm(hidden_size)
        self.post_attention_layernorm = nn.LayerNorm(hidden_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual connection
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.attention(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states
        
        # MLP with residual connection
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states


class ModelParallelTransformer(nn.Module):
    """Model parallel transformer with tensor and pipeline parallelism"""
    
    def __init__(self, 
                 vocab_size: int,
                 hidden_size: int = 768,
                 num_heads: int = 12,
                 num_layers: int = 12,
                 intermediate_size: int = 3072,
                 max_seq_length: int = 1024,
                 dropout: float = 0.1,
                 config: ModelParallelConfig = None):
        super().__init__()
        
        self.config = config or ModelParallelConfig()
        
        # Embeddings (not parallelized)
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_seq_length, hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        # Create pipeline parallel layers
        self.layers = nn.ModuleList()
        
        if self.config.pipeline_parallel_size > 1:
            # Pipeline parallel: distribute layers across devices
            layers_per_stage = num_layers // self.config.pipeline_parallel_size
            
            for i in range(num_layers):
                layer = PipelineParallelBlock(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    intermediate_size=intermediate_size,
                    dropout=dropout,
                    layer_idx=i
                )
                self.layers.append(layer)
        else:
            # No pipeline parallelism, all layers on same device
            for i in range(num_layers):
                layer = PipelineParallelBlock(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    intermediate_size=intermediate_size,
                    dropout=dropout,
                    layer_idx=i
                )
                self.layers.append(layer)
        
        # Final layer norm
        self.final_layer_norm = nn.LayerNorm(hidden_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.zeros_(module.bias)
                nn.init.ones_(module.weight)
    
    def forward(self, 
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        batch_size, seq_length = input_ids.shape
        
        # Create position IDs if not provided
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        word_embeds = self.word_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)
        hidden_states = word_embeds + position_embeds
        hidden_states = self.dropout(hidden_states)
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_length, device=input_ids.device)
        
        # Extend attention mask for multi-head attention
        if attention_mask.dim() == 2:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = attention_mask.expand(batch_size, 1, seq_length, seq_length)
            attention_mask = attention_mask.to(dtype=next(self.parameters()).dtype)
            attention_mask = (1.0 - attention_mask) * -10000.0
        
        # Process through layers
        if self.config.pipeline_parallel_size > 1:
            # Pipeline parallel execution
            hidden_states = self._pipeline_forward(hidden_states, attention_mask)
        else:
            # Sequential execution
            for layer in self.layers:
                hidden_states = layer(hidden_states)
        
        # Final layer norm
        hidden_states = self.final_layer_norm(hidden_states)
        
        return hidden_states
    
    def _pipeline_forward(self, 
                         hidden_states: torch.Tensor,
                         attention_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass with pipeline parallelism"""
        # This is a simplified pipeline parallel implementation
        # In practice, you'd use torch.distributed.pipeline.sync.Pipe
        
        # Get current rank
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        
        # Calculate which layers this rank is responsible for
        layers_per_stage = len(self.layers) // self.config.pipeline_parallel_size
        start_layer = rank * layers_per_stage
        end_layer = start_layer + layers_per_stage
        
        # Process assigned layers
        for i in range(start_layer, end_layer):
            hidden_states = self.layers[i](hidden_states)
        
        # In real pipeline parallelism, you'd communicate between stages here
        # For now, we'll just return the processed hidden states
        
        return hidden_states


class HybridParallelModel(nn.Module):
    """Hybrid parallel model combining data, tensor, and pipeline parallelism"""
    
    def __init__(self, 
                 vocab_size: int,
                 hidden_size: int = 768,
                 num_heads: int = 12,
                 num_layers: int = 12,
                 intermediate_size: int = 3072,
                 config: ModelParallelConfig = None):
        super().__init__()
        
        self.config = config or ModelParallelConfig()
        
        # Create model parallel transformer
        self.transformer = ModelParallelTransformer(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_layers=num_layers,
            intermediate_size=intermediate_size,
            config=config
        )
        
        # Language model head
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        
        # Tie weights
        self.lm_head.weight = self.transformer.word_embeddings.weight
    
    def forward(self, 
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        
        # Get transformer outputs
        hidden_states = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids
        )
        
        # Language model logits
        logits = self.lm_head(hidden_states)
        
        outputs = {
            "hidden_states": hidden_states,
            "logits": logits
        }
        
        # Compute loss if labels provided
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            outputs["loss"] = loss
        
        return outputs


def create_model_parallel_model(vocab_size: int,
                               hidden_size: int = 768,
                               num_heads: int = 12,
                               num_layers: int = 12,
                               config: ModelParallelConfig = None) -> HybridParallelModel:
    """Create a model parallel model"""
    return HybridParallelModel(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_layers=num_layers,
        config=config
    )


def setup_model_parallel_environment():
    """Setup model parallel environment"""
    if not torch.distributed.is_initialized():
        # Initialize distributed process group
        torch.distributed.init_process_group(backend='nccl')
    
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    
    logger.info(f"Model parallel environment setup: rank={rank}, world_size={world_size}")
    
    return world_size, rank


def benchmark_model_parallel(model: nn.Module,
                           input_shape: Tuple[int, int],
                           num_iterations: int = 100) -> Dict[str, float]:
    """Benchmark model parallel performance"""
    model.eval()
    
    # Create dummy input
    batch_size, seq_length = input_shape
    input_ids = torch.randint(0, 1000, (batch_size, seq_length)).cuda()
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_ids)
    
    torch.cuda.synchronize()
    
    # Benchmark
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(input_ids)
    torch.cuda.synchronize()
    
    end_time = time.time()
    
    # Calculate metrics
    total_time = end_time - start_time
    avg_time = total_time / num_iterations
    throughput = num_iterations / total_time
    
    # Memory usage
    memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
    memory_reserved = torch.cuda.memory_reserved() / 1024**3  # GB
    
    return {
        "total_time": total_time,
        "avg_time_per_iteration": avg_time,
        "throughput": throughput,
        "memory_allocated_gb": memory_allocated,
        "memory_reserved_gb": memory_reserved
    }


def main():
    """Main function for model parallelism testing"""
    # Setup model parallel environment
    world_size, rank = setup_model_parallel_environment()
    
    # Create model parallel config
    config = ModelParallelConfig(
        tensor_parallel_size=2,
        pipeline_parallel_size=2,
        data_parallel_size=1,
        memory_efficient=True
    )
    
    # Create model
    model = create_model_parallel_model(
        vocab_size=50000,
        hidden_size=1024,
        num_heads=16,
        num_layers=24,
        config=config
    )
    
    # Move model to GPU
    model = model.cuda()
    
    # Benchmark
    if rank == 0:
        results = benchmark_model_parallel(model, (8, 512), 100)
        logger.info(f"Benchmark results: {results}")


if __name__ == "__main__":
    main()