"""
Mixture of Experts (MoE) implementation for BharatFM
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, List
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from .config import MoEConfig

logger = logging.get_logger(__name__)


class MoERouter(nn.Module):
    """Router for Mixture of Experts"""
    
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts
        self.num_experts_per_token = config.num_experts_per_token
        
        # Router network
        self.gate = nn.Linear(self.hidden_size, self.num_experts, bias=False)
        
        # Load balancing loss
        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.router_z_loss_coef = config.router_z_loss_coef
        
    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the router
        
        Args:
            hidden_states: Input tensor [batch_size, seq_len, hidden_size]
            
        Returns:
            router_logits: Router logits [batch_size, seq_len, num_experts]
            expert_weights: Expert weights [batch_size, seq_len, num_experts_per_token]
            expert_indices: Expert indices [batch_size, seq_len, num_experts_per_token]
        """
        # Compute router logits
        router_logits = self.gate(hidden_states)
        
        # Apply softmax to get expert weights
        router_weights = F.softmax(router_logits, dim=-1, dtype=torch.float)
        
        # Select top-k experts
        expert_weights, expert_indices = torch.topk(
            router_weights, self.num_experts_per_token, dim=-1
        )
        
        # Normalize weights
        expert_weights = expert_weights / expert_weights.sum(dim=-1, keepdim=True)
        
        return router_logits, expert_weights, expert_indices


class MoEExpert(nn.Module):
    """Expert network for Mixture of Experts"""
    
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.expert_intermediate_size or config.intermediate_size
        
        # Expert layers
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU activation
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class MoELayer(nn.Module):
    """Mixture of Experts layer"""
    
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts
        self.num_experts_per_token = config.num_experts_per_token
        self.expert_capacity = config.expert_capacity
        
        # Router
        self.router = MoERouter(config)
        
        # Experts
        self.experts = nn.ModuleList([MoEExpert(config) for _ in range(self.num_experts)])
        
        # Load balancing
        self.use_load_balancing = config.use_load_balancing
        self.load_balancing_loss_coef = config.load_balancing_loss_coef
        
    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the MoE layer
        
        Args:
            hidden_states: Input tensor [batch_size, seq_len, hidden_size]
            
        Returns:
            output: Output tensor [batch_size, seq_len, hidden_size]
            aux_loss: Auxiliary loss for load balancing
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Route tokens to experts
        router_logits, expert_weights, expert_indices = self.router(hidden_states)
        
        # Reshape for expert processing
        hidden_states = hidden_states.view(-1, hidden_size)  # [batch_size * seq_len, hidden_size]
        expert_weights = expert_weights.view(-1, self.num_experts_per_token)  # [batch_size * seq_len, num_experts_per_token]
        expert_indices = expert_indices.view(-1, self.num_experts_per_token)  # [batch_size * seq_len, num_experts_per_token]
        
        # Initialize output and expert mask
        output = torch.zeros_like(hidden_states)
        expert_mask = torch.zeros(batch_size * seq_len, self.num_experts, device=hidden_states.device)
        
        # Process each expert
        for expert_idx in range(self.num_experts):
            # Find tokens assigned to this expert
            expert_tokens = (expert_indices == expert_idx).any(dim=-1)
            
            if not expert_tokens.any():
                continue
                
            # Get tokens for this expert
            expert_input = hidden_states[expert_tokens]
            
            # Process through expert
            expert_output = self.experts[expert_idx](expert_input)
            
            # Get weights for this expert
            expert_weight_indices = torch.where(expert_indices[expert_tokens] == expert_idx)
            weights = expert_weights[expert_tokens][expert_weight_indices]
            
            # Add to output
            output[expert_tokens] += expert_output * weights.unsqueeze(-1)
            
            # Update expert mask
            expert_mask[expert_tokens, expert_idx] = 1
            
        # Reshape output
        output = output.view(batch_size, seq_len, hidden_size)
        
        # Compute auxiliary losses
        aux_loss = self._compute_aux_loss(router_logits, expert_mask)
        
        return output, aux_loss
        
    def _compute_aux_loss(self, router_logits: torch.Tensor, expert_mask: torch.Tensor) -> torch.Tensor:
        """Compute auxiliary losses for load balancing"""
        batch_size, seq_len, num_experts = router_logits.shape
        
        # Router z-loss
        router_z_loss = torch.mean(torch.square(torch.logsumexp(router_logits, dim=-1)))
        router_z_loss = self.router.router_z_loss_coef * router_z_loss
        
        # Load balancing loss
        if self.use_load_balancing:
            # Expert utilization
            expert_utilization = expert_mask.sum(dim=0)  # [num_experts]
            expert_utilization = expert_utilization / (batch_size * seq_len)
            
            # Router probabilities
            router_probs = F.softmax(router_logits, dim=-1)
            expert_prob = router_probs.mean(dim=(0, 1))  # [num_experts]
            
            # Load balancing loss
            load_balancing_loss = self.load_balancing_loss_coef * (
                expert_utilization * expert_prob
            ).sum()
        else:
            load_balancing_loss = torch.tensor(0.0, device=router_logits.device)
            
        return router_z_loss + load_balancing_loss


class MoEAttention(nn.Module):
    """Multi-head attention for MoE model"""
    
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings
        
        # Linear layers
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        
        # Rotary embeddings
        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=10000
        )
        
        # Attention dropout
        self.attention_dropout = nn.Dropout(config.attention_dropout)
        
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
        key_states = key_states.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply rotary embeddings
        if position_ids is not None:
            query_states, key_states = self.rotary_emb(query_states, key_states, position_ids)
            
        # Cache for generation
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
            
        past_key_value = (key_states, value_states) if use_cache else None
        
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


class MoEBlock(nn.Module):
    """Transformer block for MoE model"""
    
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config
        
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.self_attention = MoEAttention(config)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.moe_layer = MoELayer(config)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
        
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
        
        # MoE layer
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        moe_output, aux_loss = self.moe_layer(hidden_states)
        hidden_states = residual + moe_output
        
        outputs = (hidden_states,)
        
        if output_attentions:
            outputs += (attn_weights,)
            
        if use_cache:
            outputs += (present_key_value,)
            
        return outputs + (aux_loss,)


class MoEPreTrainedModel(PreTrainedModel):
    """Base class for MoE models"""
    
    config_class = MoEConfig
    base_model_prefix = "moe"
    supports_gradient_checkpointing = True
    _no_split_modules = ["MoEBlock"]
    _skip_keys_device_placement = "past_key_values"
    
    def __init__(self, config: MoEConfig):
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
                module.weight.data[self.padding_idx].zero_()


class MoEModel(MoEPreTrainedModel):
    """MoE model implementation"""
    
    def __init__(self, config: MoEConfig):
        super().__init__(config)
        self.config = config
        
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([MoEBlock(config) for _ in range(config.num_hidden_layers)])
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
        total_aux_loss = 0.0
        
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
            aux_loss = layer_outputs[-1]
            total_aux_loss += aux_loss
            
            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)
                
            if output_attentions:
                all_self_attentions += (layer_outputs[1],)
                
        hidden_states = self.norm(hidden_states)
        
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
            
        if not return_dict:
            return tuple(
                v for v in [hidden_states, next_decoder_cache, all_hidden_states, all_self_attentions, total_aux_loss]
                if v is not None
            )
            
        return {
            "last_hidden_state": hidden_states,
            "past_key_values": next_decoder_cache,
            "hidden_states": all_hidden_states,
            "attentions": all_self_attentions,
            "aux_loss": total_aux_loss,
        }


class MoEForCausalLM(MoEPreTrainedModel):
    """MoE for causal language modeling"""
    
    def __init__(self, config: MoEConfig):
        super().__init__(config)
        self.config = config
        
        self.model = MoEModel(config)
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
            
            # Add auxiliary loss
            aux_loss = outputs.get("aux_loss", torch.tensor(0.0, device=loss.device))
            loss = loss + aux_loss
            
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
            
        return {
            "loss": loss,
            "logits": logits,
            "past_key_values": outputs.get("past_key_values"),
            "hidden_states": outputs.get("hidden_states"),
            "attentions": outputs.get("attentions"),
            "aux_loss": outputs.get("aux_loss"),
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