"""
Reversed engineered forward pass for Qwen
- Supports Qwen1.5-MoE-A2.7B and Qwen2MoE
- See https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_moe/modeling_qwen2_moe.py
"""
import torch

@torch.no_grad()
def run_qwen2moe_return_topk(model, input_ids, attention_mask):
    """
    Params:
        @model: A model of class `Qwen2MoeForCausalLM`.
        @input_ids: A B x N tensor of inputs IDs on the same device as `model`.
        @attention_mask: A B x N tensor of mask indicators on the same device as `model`.

    Returns:
        A dictionary with keys:
        - `logits`: The standard B x N x V LM output
        - `all_topk_experts`: A list of length equal to the number of MoE layers, with each element a BN x topk tensor of expert IDs
        - `all_topk_weights`: A list of length equal to the number of MoE layers, with each element a BN x topk tensor of expert weights
    """
    input_embeds = model.model.embed_tokens(input_ids)
    
    cache_position = torch.arange(0, input_embeds.shape[1], device = input_embeds.device)
    position_ids = cache_position.unsqueeze(0)
    causal_mask = model.model._update_causal_mask(attention_mask, input_embeds, cache_position, None, None)

    hidden_state = input_embeds
    position_embeddings = model.model.rotary_emb(hidden_state, position_ids)

    all_topk_experts = []
    all_topk_weights = []
    for layer_ix, layer in enumerate(model.model.layers):
        # SA
        residual = hidden_state
        hidden_state = layer.input_layernorm(hidden_state)
        hidden_state, _, _ = layer.self_attn(hidden_states = hidden_state, attention_mask = causal_mask, position_ids = position_ids, position_embeddings = position_embeddings)
        hidden_state = residual + hidden_state
        residual = hidden_state
        hidden_state = layer.post_attention_layernorm(hidden_state)

        ####### Qwen2MoeSparseMoeBlock - below code replaces hidden_state = layer.mlp(hidden_state)
        batch_size, sequence_length, hidden_dim = hidden_state.shape
        moe_hidden_state = hidden_state.view(-1, hidden_dim)
        router_logits = layer.mlp.gate(moe_hidden_state) # Size (BN, n_experts)

        routing_weights = torch.nn.functional.softmax(router_logits, dim = 1, dtype = torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, layer.mlp.top_k, dim = -1, sorted = True)
        routing_weights = routing_weights.to(moe_hidden_state.dtype)

        final_hidden_states = torch.zeros((batch_size * sequence_length, hidden_dim), dtype = moe_hidden_state.dtype, device = moe_hidden_state.device)

        # One hot encode the selected experts to create an expert mask 
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes = layer.mlp.num_experts).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(layer.mlp.num_experts):
            expert_layer = layer.mlp.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])
            # Index the correct hidden states and compute the expert hidden state for the current expert.
            current_state = moe_hidden_state[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]
            # However `index_add_` only support torch tensors for indexing so we'll use the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(moe_hidden_state.dtype))

        shared_expert_output = layer.mlp.shared_expert(moe_hidden_state)
        shared_expert_output = torch.nn.functional.sigmoid(layer.mlp.shared_expert_gate(moe_hidden_state)) * shared_expert_output

        final_hidden_states = (final_hidden_states + shared_expert_output).reshape(batch_size, sequence_length, hidden_dim)
        #######
        hidden_state = final_hidden_states
        hidden_state = residual + hidden_state

        all_topk_experts.append(selected_experts.detach().cpu())
        all_topk_weights.append(routing_weights.detach().cpu().to(torch.float32))

    hidden_state = model.model.norm(hidden_state)
    logits = model.lm_head(hidden_state)
    return {'logits': logits, 'all_topk_experts': all_topk_experts, 'all_topk_weights': all_topk_weights}


@torch.no_grad()
def run_qwen2moe_with_ablation_return_topk(model, input_ids, attention_mask, layers_to_ablate = [], topk_to_ablate = [], renorm = False):
    """
    Params:
        @model: A model of class `Qwen2MoeForCausalLM`.
        @input_ids: A B x N tensor of inputs IDs on the same device as `model`.
        @attention_mask: A B x N tensor of mask indicators on the same device as `model`.
        @layers_to_ablate: A list of layer indices (0-indexed) for which experts will be ablated.
        @topk_to_ablate: A list of topk indices (0-indexed) for which experts will be ablated and replaced by zeros.
        @renorm: Whether to renormalize the sum of expert weights after ablation, to scale the post-ablation expert weight sum to the original expert weight sum.

    Returns:
        A dictionary with keys:
        - `logits`: The standard B x N x V LM output
        - `all_topk_experts`: A list of length equal to the number of MoE layers, with each element a BN x topk tensor of expert IDs. Returns tthe pre-ablation topk experts.
        - `all_topk_weights`: A list of length equal to the number of MoE layers, with each element a BN x topk tensor of expert weights. Returns the post-ablation weights.
    """
    input_embeds = model.model.embed_tokens(input_ids)
    
    cache_position = torch.arange(0, input_embeds.shape[1], device = input_embeds.device)
    position_ids = cache_position.unsqueeze(0)
    causal_mask = model.model._update_causal_mask(attention_mask, input_embeds, cache_position, None, None)

    hidden_state = input_embeds
    position_embeddings = model.model.rotary_emb(hidden_state, position_ids)

    all_topk_experts = []
    all_topk_weights = []
    for layer_ix, layer in enumerate(model.model.layers):
        # SA
        residual = hidden_state
        hidden_state = layer.input_layernorm(hidden_state)
        hidden_state, _, _ = layer.self_attn(hidden_states = hidden_state, attention_mask = causal_mask, position_ids = position_ids, position_embeddings = position_embeddings)
        hidden_state = residual + hidden_state
        residual = hidden_state
        hidden_state = layer.post_attention_layernorm(hidden_state)

        ####### Qwen2MoeSparseMoeBlock - below code replaces hidden_state = layer.mlp(hidden_state)
        batch_size, sequence_length, hidden_dim = hidden_state.shape
        moe_hidden_state = hidden_state.view(-1, hidden_dim)
        router_logits = layer.mlp.gate(moe_hidden_state) # Size (BN, n_experts)

        routing_weights = torch.nn.functional.softmax(router_logits, dim = 1, dtype = torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, layer.mlp.top_k, dim = -1, sorted = True)
        routing_weights = routing_weights.to(moe_hidden_state.dtype)

        #### ABLATION
        if layer_ix in layers_to_ablate:
            row_sum_before = routing_weights.sum(dim = -1, keepdim = True) # Shaype (BN, 1)            
            # For each rank in topk_to_ablate, zero out that column
            for rank in topk_to_ablate:
                routing_weights[:, rank] = 0.0
            if renorm:
                row_sum_after = routing_weights.sum(dim = -1, keepdim = True)
                scale_factor = row_sum_before / (row_sum_after + 1e-9)
                routing_weights *= scale_factor
        ####
        final_hidden_states = torch.zeros((batch_size * sequence_length, hidden_dim), dtype = moe_hidden_state.dtype, device = moe_hidden_state.device)

        # One hot encode the selected experts to create an expert mask 
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes = layer.mlp.num_experts).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(layer.mlp.num_experts):
            expert_layer = layer.mlp.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])
            # Index the correct hidden states and compute the expert hidden state for the current expert.
            current_state = moe_hidden_state[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]
            # However `index_add_` only support torch tensors for indexing so we'll use the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(moe_hidden_state.dtype))

        shared_expert_output = layer.mlp.shared_expert(moe_hidden_state)
        shared_expert_output = torch.nn.functional.sigmoid(layer.mlp.shared_expert_gate(moe_hidden_state)) * shared_expert_output

        final_hidden_states = (final_hidden_states + shared_expert_output).reshape(batch_size, sequence_length, hidden_dim)
        #######
        hidden_state = final_hidden_states
        hidden_state = residual + hidden_state

        all_topk_experts.append(selected_experts.detach().cpu())
        all_topk_weights.append(routing_weights.detach().cpu().to(torch.float32))

    hidden_state = model.model.norm(hidden_state)
    logits = model.lm_head(hidden_state)
    return {'logits': logits, 'all_topk_experts': all_topk_experts, 'all_topk_weights': all_topk_weights}