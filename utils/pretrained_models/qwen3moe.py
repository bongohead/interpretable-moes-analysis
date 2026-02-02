"""
Reversed engineered forward pass for Qwen
- Supports Qwen3-30B-A3B and Qwen3-235B-A22B
- See https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py
"""
import torch
from ._pretrained_helpers import _sort_gate_tensors

@torch.no_grad()
def run_qwen3moe_return_topk(model, input_ids, attention_mask, return_hidden_states = False):
    """
    Params:
        @model: A model of class `Qwen3MoeForCausalLM`.
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
    input_embeds = model.model.embed_tokens(input_ids)
    
    cache_position = torch.arange(0, input_embeds.shape[1], device = input_embeds.device)
    position_ids = cache_position.unsqueeze(0)
    causal_mask = model.model._update_causal_mask(attention_mask, input_embeds, cache_position, None, None)

    hidden_state = input_embeds
    position_embeddings = model.model.rotary_emb(hidden_state, position_ids)

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
        hidden_state, _ = layer.self_attn(hidden_states = hidden_state, attention_mask = causal_mask, position_ids = position_ids, position_embeddings = position_embeddings)
        hidden_state = residual + hidden_state
        residual = hidden_state
        hidden_state = layer.post_attention_layernorm(hidden_state)
        
        if return_hidden_states:
            all_pre_mlp_hidden_states.append(hidden_state.view(-1, hidden_state.shape[2]).detach().cpu())

        ####### Qwen3MoeSparseMoeBlock - below code replaces hidden_state = layer.mlp(hidden_state)
        batch_size, sequence_length, hidden_dim = hidden_state.shape
        moe_hidden_state = hidden_state.view(-1, hidden_dim)
        
        router_logits = torch.nn.functional.linear(moe_hidden_state, layer.mlp.gate.weight) # Size (BN, n_experts)
        router_logits = torch.nn.functional.softmax(router_logits, dim = 1, dtype = torch.float)
        routing_weights, selected_experts = torch.topk(router_logits, layer.mlp.gate.top_k, dim = -1, sorted = True)
        if layer.mlp.gate.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim = -1, keepdim = True)
        routing_weights = routing_weights.to(router_logits.dtype)

        final_hidden_states = torch.zeros((batch_size * sequence_length, hidden_dim), dtype = moe_hidden_state.dtype, device = moe_hidden_state.device)
                
        # One hot encode the selected experts to create an expert mask 
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes = layer.mlp.gate.num_experts).permute(2, 1, 0)
        
        if return_hidden_states:
            layer_expert_outputs = torch.zeros((batch_size * sequence_length, layer.mlp.gate.top_k, hidden_dim), dtype = moe_hidden_state.dtype, device = moe_hidden_state.device) # BN x topk x D

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(layer.mlp.gate.num_experts):
            idx, top_x = torch.where(expert_mask[expert_idx])
            # Index the correct hidden states and compute the expert hidden state for the current expert.
            current_state = moe_hidden_state[None, top_x].reshape(-1, hidden_dim)

            gate, up = torch.nn.functional.linear(current_state, layer.mlp.experts.gate_up_proj[expert_idx]).chunk(2, dim = -1)
            current_expert_output = torch.nn.functional.linear(layer.mlp.experts.act_fn(gate) * up, layer.mlp.experts.down_proj[expert_idx])

            current_hidden_states = current_expert_output * routing_weights[top_x, idx, None]
            # However `index_add_` only support torch tensors for indexing so we'll use the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(moe_hidden_state.dtype))

            if return_hidden_states:
                layer_expert_outputs[top_x, idx] = current_expert_output.to(layer_expert_outputs.dtype)

        final_hidden_states = (final_hidden_states).reshape(batch_size, sequence_length, hidden_dim)
        #######
        hidden_state = final_hidden_states
        hidden_state = residual + hidden_state

        all_topk_experts.append(selected_experts.detach().cpu())
        all_topk_weights.append(routing_weights.detach().cpu().to(torch.float32))

        if return_hidden_states:
            all_router_logits.append(router_logits.detach().cpu())
            all_hidden_states.append(hidden_state.view(-1, hidden_state.shape[2]).detach().cpu())
            all_expert_outputs.append(layer_expert_outputs.detach().cpu())

    hidden_state = model.model.norm(hidden_state)
    logits = model.lm_head(hidden_state)

    return {
        'logits': logits,
        'all_topk_experts': all_topk_experts,
        'all_topk_weights': all_topk_weights,
        'all_pre_mlp_hidden_states': all_pre_mlp_hidden_states,
        'all_router_logits': all_router_logits,
        'all_hidden_states': all_hidden_states,
        'all_expert_outputs': all_expert_outputs
    }
