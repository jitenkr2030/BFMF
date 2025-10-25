"""
GLM (General Language Model) implementation for BharatFM
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, List
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from .config import GLMConfig

logger = logging.get_logger(__name__)


class GLMAttention(nn.Module):
    """Multi-head attention with memory mechanism for GLM"""
    
    def __init__(self, config: GLMConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings
        
        # Multi-query attention support
        if config.multi_query_attention:
            self.num_key_value_heads = config.num_key_value_heads
            self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        else:
            self.num_key_value_heads = self.num_heads
            self.num_key_value_groups = 1
            
        # Linear layers
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        
        # Rotary embeddings
        if config.use_rotary_embeddings:
            self.rotary_emb = RotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=config.rope_theta if hasattr(config, 'rope_theta') else 10000
            )
        else:
            self.rotary_emb = None
            
        # Attention dropout
        self.attention_dropout = nn.Dropout(config.attention_dropout)
        
        # Memory attention
        if config.use_memory_attention:
            self.memory_length = config.memory_length
            self.memory_key = nn.Parameter(torch.zeros(self.num_key_value_heads, self.memory_length, self.head_dim))
            self.memory_value = nn.Parameter(torch.zeros(self.num_key_value_heads, self.memory_length, self.head_dim))
        else:
            self.memory_length = 0
            self.memory_key = None
            self.memory_value = None
            
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        batch_size, seq_length, _ = hidden_states.shape
        
        # Project queries, keys, values
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        query_states = query_states.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_length, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_length, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        # Apply rotary embeddings
        if self.rotary_emb is not None and position_ids is not None:
            query_states, key_states = self.rotary_emb(query_states, key_states, position_ids)
            
        # Add memory states if enabled
        if self.memory_key is not None and self.memory_value is not None:
            # Expand memory for batch
            memory_key = self.memory_key.unsqueeze(0).expand(batch_size, -1, -1, -1)
            memory_value = self.memory_value.unsqueeze(0).expand(batch_size, -1, -1, -1)
            
            # Concatenate with current states
            key_states = torch.cat([memory_key, key_states], dim=2)
            value_states = torch.cat([memory_value, value_states], dim=2)
            
        # Cache for generation
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
            
        past_key_value = (key_states, value_states) if use_cache else None
        
        # Repeat key/value heads if needed
        if self.num_key_value_groups > 1:
            key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
            value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)
            
        # Compute attention scores
        attn_weights = torch.matmul(query_states, key_states.transpose(-1, -2)) / (self.head_dim ** 0.5)
        
        # Apply attention mask
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
            
        # Softmax
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attention_dropout(attn_weights)
        
        # Compute output
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_length, self.hidden_size)
        
        # Final projection
        attn_output = self.o_proj(attn_output)
        
        return attn_output, attn_weights if output_attentions else None, past_key_value


class GLMMLP(nn.Module):
    """MLP layer for GLM"""
    
    def __init__(self, config: GLMConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU activation
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class GLMBlock(nn.Module):
    """Transformer block for GLM"""
    
    def __init__(self, config: GLMConfig):
        super().__init__()
        self.config = config
        
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.self_attention = GLMAttention(config)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = GLMMLP(config)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        
        # Self-attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        attn_output, attn_weights, present_key_value = self.self_attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        
        hidden_states = residual + attn_output
        
        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        outputs = (hidden_states,)
        
        if output_attentions:
            outputs += (attn_weights,)
            
        if use_cache:
            outputs += (present_key_value,)
            
        return outputs


class GLMPreTrainedModel(PreTrainedModel):
    """Base class for GLM models"""
    
    config_class = GLMConfig
    base_model_prefix = "glm"
    supports_gradient_checkpointing = True
    _no_split_modules = ["GLMBlock"]
    _skip_keys_device_placement = "past_key_values"
    
    def __init__(self, config: GLMConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        
    def _init_weights(self, module):
        """Initialize weights"""
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class GLMModel(GLMPreTrainedModel):
    """GLM model implementation"""
    
    def __init__(self, config: GLMConfig):
        super().__init__(config)
        self.config = config
        
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([GLMBlock(config) for _ in range(config.num_hidden_layers)])
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Initialize weights
        self.post_init()
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, dict]:
        
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else False
        output_hidden_states = output_hidden_states if output_hidden_states is not None else False
        return_dict = return_dict if return_dict is not None else True
        
        # Get embeddings
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
            
        batch_size, seq_length, _ = inputs_embeds.shape
        
        # Create position IDs if not provided
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=inputs_embeds.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
            
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), device=inputs_embeds.device)
            
        # Prepare past key values
        past_key_values = past_key_values or [None] * len(self.layers)
        
        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        next_decoder_cache = () if use_cache else None
        
        # Process through layers
        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
                
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values[i],
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
            
            hidden_states = layer_outputs[0]
            
            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)
                
            if output_attentions:
                all_self_attentions += (layer_outputs[1],)
                
        hidden_states = self.norm(hidden_states)
        
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
            
        if not return_dict:
            return tuple(
                v for v in [hidden_states, next_decoder_cache, all_hidden_states, all_self_attentions]
                if v is not None
            )
            
        return {
            "last_hidden_state": hidden_states,
            "past_key_values": next_decoder_cache,
            "hidden_states": all_hidden_states,
            "attentions": all_self_attentions,
        }


class GLMForCausalLM(GLMPreTrainedModel):
    """GLM for causal language modeling"""
    
    def __init__(self, config: GLMConfig):
        super().__init__(config)
        self.config = config
        
        self.model = GLMModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Initialize weights
        self.post_init()
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, dict]:
        
        return_dict = return_dict if return_dict is not None else True
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            # Shift labels for causal language modeling
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten for loss computation
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
            
        return {
            "loss": loss,
            "logits": logits,
            "past_key_values": outputs.get("past_key_values"),
            "hidden_states": outputs.get("hidden_states"),
            "attentions": outputs.get("attentions"),
        }


class RotaryEmbedding(nn.Module):
    """Rotary position embeddings"""
    
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        # Create frequency tensor
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # Cache for position embeddings
        self.max_cached_positions = max_position_embeddings
        self.cos_cached = None
        self.sin_cached = None
        
    def forward(self, q: torch.Tensor, k: torch.Tensor, position_ids: torch.Tensor):
        """Apply rotary embeddings to query and key tensors"""
        
        # Get sequence length
        seq_len = position_ids.max() + 1
        
        # Update cache if needed
        if seq_len > self.max_cached_positions:
            self._update_cos_sin_cache(seq_len)
            
        # Get cos and sin for current positions
        cos = self.cos_cached[position_ids].unsqueeze(1)
        sin = self.sin_cached[position_ids].unsqueeze(1)
        
        # Apply rotary embeddings
        q_rot = self._rotate_half(q)
        k_rot = self._rotate_half(k)
        
        q = q * cos + q_rot * sin
        k = k * cos + k_rot * sin
        
        return q, k
        
    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate half of the tensor"""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)
        
    def _update_cos_sin_cache(self, seq_len: int):
        """Update cos and sin cache"""
        # Create position tensor
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        
        # Compute frequencies
        freqs = torch.outer(t, self.inv_freq)
        
        # Compute cos and sin
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)
        
        # Cache
        self.cos_cached = cos
        self.sin_cached = sin
        self.max_cached_positions = seq_len