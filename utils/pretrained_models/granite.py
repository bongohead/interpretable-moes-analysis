"""
Reversed engineered forward pass for IBM Granite-4.0-Tiny Models
- Supports Granite-4.0 Tiny Preview
"""
import torch
from ._pretrained_helpers import _sort_gate_tensors

@torch.no_grad()
def run_granite_return_topk(model, input_ids: torch.LongTensor, attention_mask: torch.Tensor, return_hidden_states: bool = False,):
    """
    Params:
        @model: A model of class `GraniteMoeHybridForCausalLM`.
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
    input_embeds = model.model.embed_tokens(input_ids) * model.config.embedding_multiplier

    cache_position = torch.arange(0, input_embeds.shape[1], device = input_embeds.device)
    position_ids = cache_position.unsqueeze(0)
    causal_mask = model.model._update_causal_mask(attention_mask, input_embeds, cache_position, None, output_attentions = False)
    mamba_mask = model.model._update_mamba_mask(attention_mask, cache_position)
    position_embeddings = model.model.rotary_emb(input_embeds, position_ids) if model.model.rotary_emb is not None else None

    hidden_state = input_embeds

    all_topk_experts = []
    all_topk_weights = []
    all_pre_mlp_hidden_states = []
    all_router_logits = []
    all_hidden_states = []
    all_expert_outputs = []

    for layer_ix, layer in enumerate(model.model.layers):
        # SA
        residual = hidden_state
        hidden_state = layer.input_layernorm(hidden_state)
        if layer.layer_type == 'mamba':
            hidden_state = layer.mamba(hidden_states = hidden_state, attention_mask = mamba_mask)
        else:
            hidden_state, _ = layer.self_attn(hidden_states = hidden_state, attention_mask = causal_mask, position_embeddings = position_embeddings)
        hidden_state = residual + hidden_state * layer.residual_multiplier
        residual = hidden_state
        hidden_state = layer.post_attention_layernorm(hidden_state)

        if return_hidden_states:
            all_pre_mlp_hidden_states.append(hidden_state.view(-1, hidden_state.shape[2]).detach().cpu())

        ####### GraniteMoeHybridMoE - below code replaces seltorch.nn.functional.block_sparse_moe(hidden_states) #######
        batch_size, seq_len, hidden_dim = hidden_state.shape
        tokens = batch_size * seq_len
        moe_hidden_state = hidden_state.view(tokens, hidden_dim)

        # 1️⃣  Router (identical to GraniteMoeHybridTopKGating)
        (
            index_sorted_experts, # (tokens*k,)
            batch_index, # (tokens*k,)   token ids grouped by expert
            batch_gates, # (tokens*k,)   softmax weights in same order
            expert_size, # python list   tokens routed to each expert
            router_logits, # (tokens, n_experts)
        ) = layer.block_sparse_moe.router(moe_hidden_state)

        k = layer.block_sparse_moe.router.top_k
        # index helpers to recover rank-within-top-k
        rank_in_topk = torch.remainder(index_sorted_experts, k)   # (tokens*k,)

        selected_experts = router_logits.topk(k, dim = 1).indices   # (tokens, k)
        routing_weights = torch.zeros(tokens, k, dtype = moe_hidden_state.dtype, device = moe_hidden_state.device) # (tokens, k)
        routing_weights[batch_index, rank_in_topk] = batch_gates # scatter
        routing_weights = routing_weights.view(batch_size * seq_len, k) # (tokens, k)

        if return_hidden_states:
            layer_expert_outputs = torch.zeros(tokens, k, hidden_dim, dtype = moe_hidden_state.dtype, device = moe_hidden_state.device)

        # 2. Gather inputs grouped by expert
        expert_inputs = moe_hidden_state[batch_index]                  # (tokens*k, D)

        # 3. Expert network
        hidden = layer.block_sparse_moe.input_linear(expert_inputs, expert_size)
        h_act, h_g  = hidden.chunk(2, dim = -1)
        hidden = layer.block_sparse_moe.activation(h_act) * h_g
        raw_outputs = layer.block_sparse_moe.output_linear(hidden, expert_size)  # (tokens*k, D)

        # keep a copy **before** weighting for diagnostics
        if return_hidden_states:
            layer_expert_outputs[batch_index, rank_in_topk] = raw_outputs.detach()

        # 4️. Apply routing prob and scatter-add once
        weighted = raw_outputs * batch_gates[:, None]
        final_flat = torch.zeros_like(moe_hidden_state) # (tokens, D)
        final_flat.index_add_(0, batch_index, weighted)

        final_hidden_states = final_flat.view(batch_size, seq_len, hidden_dim)
        #######
        hidden_state = residual + (final_hidden_states + layer.shared_mlp(hidden_state)) * layer.residual_multiplier

        all_topk_experts.append(selected_experts.detach().cpu())
        all_topk_weights.append(routing_weights.detach().cpu().to(torch.float32))

        if return_hidden_states:
            all_router_logits.append(router_logits.detach().cpu())
            all_hidden_states.append(hidden_state.view(-1, hidden_state.shape[2]).detach().cpu())
            all_expert_outputs.append(layer_expert_outputs.detach().cpu())

    hidden_state = model.model.norm(hidden_state)
    logits = model.lm_head(hidden_state) / model.config.logits_scaling

    return {
        'logits': logits,
        'all_topk_experts': all_topk_experts,
        'all_topk_weights': all_topk_weights,
        'all_pre_mlp_hidden_states': all_pre_mlp_hidden_states,
        'all_router_logits': all_router_logits,
        'all_hidden_states': all_hidden_states,
        'all_expert_outputs': all_expert_outputs
    }


