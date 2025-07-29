"""
Reversed engineered forward pass for GLM-4.5
- Supports GLM-4.5 and GLM-4.5-Air
- See https://github.com/huggingface/transformers/blob/main/src/transformers/models/glm4_moe/modeling_glm4_moe.py
"""
import torch
from transformers.masking_utils import create_causal_mask
from transformers.models.glm4_moe.modeling_glm4_moe import Glm4MoeMoE
from ._pretrained_helpers import _sort_gate_tensors

@torch.no_grad()
def run_glm4moe_return_topk(model, input_ids, attention_mask, return_hidden_states: bool = False):
    """
    Params:
        @model: A model of class `Glm4ForCausalLM`.
        @input_ids: A B x N tensor of inputs IDs on the same device as `model`.
        @attention_mask: A B x N tensor of mask indicators on the same device as `model`.
        @return_hidden_states: Boolean; whether to return hidden_states themselves.

    Returns:
        A dictionary with keys:
        - `logits`: The standard B x N x V LM output
        - `all_topk_experts`: A list of length equal to the number of MoE layers, with each element a BN x topk tensor of expert IDs
        - `all_topk_weights`: A list of length equal to the number of MoE layers, with each element a BN x topk tensor of expert weights
        - `all_pre_mlp_hidden_states`: If return_hidden_states, a list of length equal to the number of MoE layers, with each element a BN x D tensor of pre-MLP hidden states
        - `all_router_logits: If return_hidden_states, a list of length equal to the number of MoE layers, with each element a BN x n_experts tensor of router logits
        - `all_hidden_states`: If return_hidden_states, a list of length equal to the number of MoE layers, with each element a BN x D tensor of post-layer hidden states
        - `all_expert_outputs`: If return_hidden_states, a list of length equal to the number of MoE layers, with each element a BN x topk x D tensor of expert outputs (pre-weighting)
    """
    cfg = model.model.config

    # Helpers
    def to_device(t, dev):
        """
        Move tensor `t` to `dev` only if necessary
        """
        return t if t is None or t.device == dev else t.to(dev, non_blocking = True)

    ##### Setup #####
    # Pick the device that holds the token‑embedding weights as our starting point.
    emb_device = model.model.embed_tokens.weight.device
    input_ids = to_device(input_ids, emb_device)
    attention_mask = to_device(attention_mask, emb_device)

    input_embeds = model.model.embed_tokens(input_ids) # (B, N, D)
    B, N, D = input_embeds.shape

    cache_position = torch.arange(N, device = emb_device)
    position_ids = cache_position.unsqueeze(0) # (1, N)

    causal_mask = create_causal_mask(
        config = cfg,
        input_embeds = input_embeds,
        attention_mask = attention_mask,
        cache_position = cache_position,
        past_key_values = None,
        position_ids = position_ids,
    ) # (B, 1, N, N)

    # Compute ROPE embs (copy to each GPU as needed)
    position_embeddings = model.model.rotary_emb(input_embeds, position_ids)
    cos_global, sin_global = position_embeddings

    hidden_state = input_embeds

    # Diagnostic containers
    all_topk_experts, all_topk_weights = [], []
    all_pre_mlp_hidden_states, all_router_logits = [], []
    all_hidden_states, all_expert_outputs = [], []

    ##### Transformer Layers #####
    for layer in model.model.layers:

        # Device of this layers parameters (assuming all same)
        layer_dev = next(layer.parameters()).device

        # Move working tensors to that GPU (if not already there)
        hidden_state = to_device(hidden_state, layer_dev)
        causal_mask = to_device(causal_mask, layer_dev)
        position_ids = to_device(position_ids, layer_dev)
        cos = to_device(cos_global, layer_dev)
        sin = to_device(sin_global, layer_dev)
        pos_emb = (cos, sin)

        # SA 
        residual = hidden_state
        hidden_state = layer.input_layernorm(hidden_state)
        hidden_state, _ = layer.self_attn(
            hidden_states = hidden_state,
            attention_mask = causal_mask,
            position_ids = position_ids,
            cache_position = cache_position,
            position_embeddings = pos_emb,
        )
        hidden_state = residual + hidden_state

        # MoE
        residual = hidden_state
        hidden_state = layer.post_attention_layernorm(hidden_state)

        if isinstance(layer.mlp, Glm4MoeMoE):
            B_, S_, D_ = hidden_state.shape
            moe_hidden_state_flat = hidden_state.view(-1, D_) # (BN,D)

            topk_ids, topk_weight = layer.mlp.gate(hidden_state) # (BN, K)
            # Pre-sigmoid router logits for analysis
            router_logits = torch.nn.functional.linear(moe_hidden_state_flat.float(), layer.mlp.gate.weight.float()) # (BN, E)

            # manual pass through experts to capture outputs
            num_exp = layer.mlp.config.n_routed_experts
            K = layer.mlp.config.num_experts_per_tok
            final_flat = torch.zeros_like(moe_hidden_state_flat, dtype = topk_weight.dtype)

            if return_hidden_states:
                layer_expert_outputs = moe_hidden_state_flat.new_zeros((moe_hidden_state_flat.size(0), K, D_))

            expert_mask = torch.nn.functional.one_hot(topk_ids, num_classes = num_exp).permute(2, 0, 1)  # (E, BN, K)

            for e_idx in range(num_exp):
                token_idx, k_rank = torch.where(expert_mask[e_idx])
                if token_idx.numel() == 0:
                    continue
                expert_layer = layer.mlp.experts[e_idx]
                expert_input = moe_hidden_state_flat[token_idx]
                expert_out = expert_layer(expert_input) # (hits, D)
                weighted_out = expert_out * topk_weight[token_idx, k_rank].unsqueeze(-1)
                final_flat.index_add_(0, token_idx, weighted_out)
                if return_hidden_states:
                    layer_expert_outputs[token_idx, k_rank] = expert_out

            # Shared dense experts
            shared_out = layer.mlp.shared_experts(hidden_state) # (B,S,D)
            hidden_state = final_flat.view(B_, S_, D_).type(hidden_state.dtype) + shared_out

            # Store MoE stats
            topk_ids, topk_weight, layer_expert_outputs = _sort_gate_tensors(
                topk_ids.detach(),
                topk_weight.detach(),
                layer_expert_outputs.detach() if return_hidden_states else None
            )

            all_topk_experts.append(topk_ids.cpu())
            all_topk_weights.append(topk_weight.cpu().to(torch.float32))

            if return_hidden_states:
                all_pre_mlp_hidden_states.append(moe_hidden_state_flat.detach().cpu())
                all_router_logits.append(router_logits.detach().cpu())
                all_expert_outputs.append(layer_expert_outputs.cpu())

        else: # Dense layer
            hidden_state = layer.mlp(hidden_state)

        hidden_state = residual + hidden_state

        if return_hidden_states and isinstance(layer.mlp, Glm4MoeMoE):
            all_hidden_states.append(hidden_state.view(-1, D_).detach().cpu())

    ##### LM head  #####
    hidden_state = to_device(hidden_state, model.model.norm.weight.device)
    hidden_state = model.model.norm(hidden_state)
    hidden_state = to_device(hidden_state, model.lm_head.weight.device)
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
