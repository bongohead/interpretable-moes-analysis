"""
Reversed engineered forward pass for OlMoE
- See https://github.com/huggingface/transformers/blob/main/src/transformers/models/olmoe/modeling_olmoe.py
"""
import torch
from ._pretrained_helpers import _sort_gate_tensors

@torch.no_grad()
def run_olmoe_return_topk(model, input_ids, attention_mask, return_hidden_states = False):
    """
    Params:
        @model: A model of class `OlmoeForCausalLM`.
        @input_ids: A (B, N) tensor of inputs IDs on the same device as `model`.
        @attention_mask: A (B, N) tensor of mask indicators on the same device as `model`.
        @return_hidden_states: Boolean; whether to return optional outputs below.

    Returns:
        A dictionary with keys:
        - `logits`: (B, N, V) LM outputs
        - `all_topk_experts`: List (len = # MoE layers) of (BN, topk) expert IDs tensors
        - `all_topk_weights`: List (len = # MoE layers) of (BN, topk) expert weight tensors
        - `all_pre_mlp_hidden_states`: (optional) List (len = # MoE layers) of (BN, D) pre-MLP activations
        - `all_router_logits: (optional) List (len = # MoE layers) of (BN, n_experts) router *logits*
        - `all_hidden_states`: (optional) List (len = # MoE layers) of (BN, D) post-layer activations
        - `all_expert_outputs`: (optional) List (len = # MoE layers) of (BN, topk, D) pre-weighting expert outputs
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

        ####### OlMoESparseMoeBlock - below code replaces hidden_state = layer.mlp(hidden_state)
        batch_size, sequence_length, hidden_dim = hidden_state.shape
        moe_hidden_state = hidden_state.view(-1, hidden_dim)

        router_logits = torch.nn.functional.linear(moe_hidden_state, layer.mlp.gate.weight) # Size (BN, n_experts)
        router_logits = torch.nn.functional.softmax(router_logits, dim = 1, dtype = torch.float)
        routing_weights, selected_experts = torch.topk(router_logits, layer.mlp.gate.top_k, dim = -1, sorted = True)
        if layer.mlp.gate.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim = -1, keepdim = True)
        routing_weights = routing_weights.to(router_logits.dtype)

        final_hidden_states = torch.zeros((batch_size * sequence_length, hidden_dim), dtype = hidden_state.dtype, device = hidden_state.device)
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes = layer.mlp.experts.num_experts).permute(2, 1, 0)

        if return_hidden_states:
            layer_expert_outputs = torch.zeros((batch_size * sequence_length, layer.mlp.gate.top_k, hidden_dim), dtype = moe_hidden_state.dtype, device = moe_hidden_state.device) # BN x topk x D

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(layer.mlp.experts.num_experts):
            idx, top_x = torch.where(expert_mask[expert_idx])

            current_state = moe_hidden_state[None, top_x].reshape(-1, hidden_dim)
            gate, up = torch.nn.functional.linear(current_state, layer.mlp.experts.gate_up_proj[expert_idx]).chunk(2, dim = -1)
            current_expert_output = torch.nn.functional.linear(layer.mlp.experts.act_fn(gate) * up, layer.mlp.experts.down_proj[expert_idx])
            current_hidden_states = current_expert_output * routing_weights[top_x, idx, None]

            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(moe_hidden_state.dtype))

            if return_hidden_states:
                layer_expert_outputs[top_x, idx] = current_expert_output.to(layer_expert_outputs.dtype)

        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
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