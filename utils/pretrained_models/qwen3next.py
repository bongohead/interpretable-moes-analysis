"""
Reversed engineered forward pass for Qwen3-Next
- Supports Qwen3-Next-* model
- See https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_next/modeling_qwen3_next.py
- This supports multiple device usage
"""
import torch
from transformers.masking_utils import create_causal_mask
from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextSparseMoeBlock
from ._pretrained_helpers import _move_device 

@torch.no_grad()
def run_qwen3next_return_topk(model, input_ids, attention_mask, return_hidden_states: bool = False):
    """
    Params:
        @model: A model of class `Qwen3NextForCausalLM`.
        @input_ids: A (B, N) tensor of input IDs on the same device as `model` (or will be moved to embeddings' device).
        @attention_mask: A (B, N) tensor of mask indicators (1=keep, 0=pad).
        @return_hidden_states: Whether to return optional diagnostics listed below.

    Returns:
        A dictionary with keys:
        - `logits`: (B, N, V) LM outputs
        - `all_topk_experts`: List (len = # MoE layers) of (BN, topk) expert IDs tensors
        - `all_topk_weights`: List (len = # MoE layers) of (BN, topk) expert weight tensors
        - `all_pre_mlp_hidden_states`: (optional) List (len = # MoE layers) of (BN, D) pre-MLP activations
        - `all_router_logits`: (optional) List (len = # MoE layers) of (BN, n_experts) router logits
        - `all_hidden_states`: (optional) List (len = # MoE layers) of (BN, D) post-layer activations
        - `all_expert_outputs`: (optional) List (len = # MoE layers) of (BN, topk, D) pre-weighting expert outputs
    """
    ##### Setup (anchor on the embeddings' device) #####
    emb_device = model.model.embed_tokens.weight.device
    input_ids = _move_device(input_ids, emb_device)
    attention_mask = _move_device(attention_mask, emb_device) if attention_mask is not None else None

    input_embeds = model.model.embed_tokens(input_ids)  # (B, N, D)
    B, N, D = input_embeds.shape

    cache_position = torch.arange(N, device=emb_device)
    position_ids = cache_position.unsqueeze(0)  # (1, N)
    causal_mask = create_causal_mask(model.model.config, input_embeds, attention_mask, cache_position, None, position_ids)

    # Dual attention masks: Linear-attention mask: either None (no padding suppression) or (B, N) + RoPE (compute once; copy per layer to the layer's device)
    linear_attn_mask = model.model._update_linear_attn_mask(attention_mask, cache_position)
    cos_global, sin_global = model.model.rotary_emb(input_embeds, position_ids)

    hidden_state = input_embeds

    all_topk_experts = []
    all_topk_weights = []
    all_pre_mlp_hidden_states = []
    all_router_logits = []
    all_hidden_states = []
    all_expert_outputs = []

    ##### Transformer Layers #####
    for layer in model.model.layers:

        # Device of this layers parameters (assuming all same)
        layer_dev = next(layer.parameters()).device

        # Move working tensors to this layer's device
        hidden_state = _move_device(hidden_state, layer_dev)
        position_ids = _move_device(position_ids, layer_dev)
        cache_position_layer = _move_device(cache_position, layer_dev)
        cos = _move_device(cos_global, layer_dev)
        sin = _move_device(sin_global, layer_dev)
        pos_emb = (cos, sin)

        # Pick the correct mask per layer type and move it
        if getattr(layer, "layer_type", "full_attention") == "linear_attention":
            layer_mask = _move_device(linear_attn_mask, layer_dev) if linear_attn_mask is not None else None  # (B,N) or None
        else:
            layer_mask = _move_device(causal_mask, layer_dev)  # (B,1,N,N)

        # SA 
        residual = hidden_state
        hidden_state = layer.input_layernorm(hidden_state)

        if getattr(layer, "layer_type", "full_attention") == "linear_attention":
            # Gated DeltaNet; no cache here (analyze full sequence)
            hidden_state = layer.linear_attn(hidden_states = hidden_state, cache_params = None, cache_position = cache_position_layer, attention_mask = layer_mask)
        else:
            # Full attention (gated attention internally)
            hidden_state, _ = layer.self_attn(hidden_states = hidden_state, attention_mask = layer_mask, position_ids = position_ids, cache_position = cache_position_layer, position_embeddings = pos_emb)
        hidden_state = residual + hidden_state

        ###### MoE Block
        residual = hidden_state
        hidden_state = layer.post_attention_layernorm(hidden_state)

        if isinstance(layer.mlp, Qwen3NextSparseMoeBlock):
            # Flatten tokens for routing
            batch_size, seq_len, hidden_dim = hidden_state.shape
            moe_hidden_state = hidden_state.view(-1, hidden_dim)  # (BN, D)

            # Router logits and top-k
            router_logits = layer.mlp.gate(moe_hidden_state)  # (BN, E)
            routing_prob = torch.nn.functional.softmax(router_logits, dim = 1, dtype = torch.float)  # stable softmax
            topk_weight, topk_ids = torch.topk(routing_prob, layer.mlp.top_k, dim = -1, sorted=True)
            if layer.mlp.norm_topk_prob:
                topk_weight = topk_weight / topk_weight.sum(dim = -1, keepdim=True)
            topk_weight = topk_weight.to(moe_hidden_state.dtype)

            # Accumulate routed experts on-device
            final_flat = torch.zeros_like(moe_hidden_state, dtype = moe_hidden_state.dtype)
            if return_hidden_states:
                layer_expert_outputs = moe_hidden_state.new_zeros((moe_hidden_state.size(0), layer.mlp.top_k, hidden_dim))

            num_exp = layer.mlp.num_experts
            expert_mask = torch.nn.functional.one_hot(topk_ids, num_classes=num_exp).permute(2, 0, 1)  # (E, BN, K)

            for e_idx in range(num_exp):
                token_idx, k_rank = torch.where(expert_mask[e_idx])
                if token_idx.numel() == 0:
                    continue
                expert_layer = layer.mlp.experts[e_idx]
                expert_in = moe_hidden_state[token_idx] # (hits, D)
                expert_out = expert_layer(expert_in) # (hits, D)
                weighted_out = expert_out * topk_weight[token_idx, k_rank].unsqueeze(-1)
                final_flat.index_add_(0, token_idx, weighted_out)
                if return_hidden_states:
                    layer_expert_outputs[token_idx, k_rank] = expert_out

            # SHARED EXPER 
            shared_out = layer.mlp.shared_expert(moe_hidden_state) # (BN, D)
            shared_gate = torch.sigmoid(layer.mlp.shared_expert_gate(moe_hidden_state))  # (BN, 1)
            final_flat = final_flat + shared_gate * shared_out

            # Reshape back to (B,S,D)
            hidden_state = final_flat.view(batch_size, seq_len, hidden_dim)

            # Stats
            all_topk_experts.append(topk_ids.detach().cpu())
            all_topk_weights.append(topk_weight.detach().cpu().to(torch.float32))
            if return_hidden_states:
                all_pre_mlp_hidden_states.append(moe_hidden_state.detach().cpu())
                all_router_logits.append(router_logits.detach().cpu())
                all_expert_outputs.append(layer_expert_outputs.detach().cpu())
        else:
            # Dense MLP path
            hidden_state = layer.mlp(hidden_state)

        # Post-MLP residual
        hidden_state = residual + hidden_state

        if return_hidden_states and isinstance(layer.mlp, Qwen3NextSparseMoeBlock):
            all_hidden_states.append(hidden_state.view(-1, hidden_state.shape[-1]).detach().cpu())

    #######
    hidden_state = _move_device(hidden_state, model.model.norm.weight.device)
    hidden_state = model.model.norm(hidden_state)
    hidden_state = _move_device(hidden_state, model.lm_head.weight.device)
    logits = model.lm_head(hidden_state).to(input_ids.device)

    return {
        'logits': logits,
        'all_topk_experts': all_topk_experts,
        'all_topk_weights': all_topk_weights,
        'all_pre_mlp_hidden_states': all_pre_mlp_hidden_states,
        'all_router_logits': all_router_logits,
        'all_hidden_states': all_hidden_states,
        'all_expert_outputs': all_expert_outputs
    }
