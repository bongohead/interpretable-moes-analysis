"""
Reversed engineered forward pass for Qwen
- Supports Qwen3-30B-A3B, Qwen3-235B-A22B
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

        router_logits = layer.mlp.gate(moe_hidden_state) # Size (BN, n_experts)
        routing_weights = torch.nn.functional.softmax(router_logits, dim = 1, dtype = torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, layer.mlp.top_k, dim = -1, sorted = True)
        if layer.mlp.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim = -1, keepdim = True)
        routing_weights = routing_weights.to(moe_hidden_state.dtype)

        final_hidden_states = torch.zeros((batch_size * sequence_length, hidden_dim), dtype = moe_hidden_state.dtype, device = moe_hidden_state.device)
        
        # One hot encode the selected experts to create an expert mask 
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes = layer.mlp.num_experts).permute(2, 1, 0)

        if return_hidden_states:
            layer_expert_outputs = torch.zeros((batch_size * sequence_length, layer.mlp.top_k, hidden_dim), dtype = moe_hidden_state.dtype, device = moe_hidden_state.device) # BN x topk x D

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(layer.mlp.num_experts):
            expert_layer = layer.mlp.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])
            # Index the correct hidden states and compute the expert hidden state for the current expert.
            current_state = moe_hidden_state[None, top_x].reshape(-1, hidden_dim)
            current_expert_output = expert_layer(current_state) 
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
            all_hidden_states.append(hidden_state.view(-1, hidden_state.shape[2]).detach().cpu())
            all_expert_outputs.append(layer_expert_outputs.detach().cpu())

    hidden_state = model.model.norm(hidden_state)
    logits = model.lm_head(hidden_state)

    return {
        'logits': logits,
        'all_topk_experts': all_topk_experts,
        'all_topk_weights': all_topk_weights,
        'all_pre_mlp_hidden_states': all_pre_mlp_hidden_states,
        'all_hidden_states': all_hidden_states,
        'all_expert_outputs': all_expert_outputs
    }

@torch.no_grad()
def run_qwen2moe_with_ablation_return_topk(model, input_ids, attention_mask, layers_to_ablate = [], topk_to_ablate = [], renorm = False, return_hidden_states = False):
    """
    Params:
        @model: A model of class `Qwen2MoeForCausalLM`.
        @input_ids: A B x N tensor of inputs IDs on the same device as `model`.
        @attention_mask: A B x N tensor of mask indicators on the same device as `model`.
        @layers_to_ablate: A list of layer indices (0-indexed) for which experts will be ablated.
        @topk_to_ablate: A list of topk indices (0-indexed) for which experts will be ablated and replaced by zeros.
        @renorm: Whether to renormalize the sum of expert weights after ablation, to scale the post-ablation expert weight sum to the original expert weight sum.
        @return_hidden_states: Boolean; whether to return hidden_states themselves.

    Returns:
        A dictionary with keys:
        - `logits`: The standard B x N x V LM output
        - `all_topk_experts`: A list of length equal to the number of MoE layers, with each element a BN x topk tensor of expert IDs. Returns tthe pre-ablation topk experts.
        - `all_topk_weights`: A list of length equal to the number of MoE layers, with each element a BN x topk tensor of expert weights. Returns the post-ablation weights.
        - `all_pre_mlp_hidden_states`: If return_hidden_states, a list of length equal to the number of MoE layers, with each element a BN x D tensor of pre-MLP hidden states
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

        router_logits = layer.mlp.gate(moe_hidden_state) # Size (BN, n_experts)
        routing_weights = torch.nn.functional.softmax(router_logits, dim = 1, dtype = torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, layer.mlp.top_k, dim = -1, sorted = True)
        if layer.mlp.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim = -1, keepdim = True)
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

        if return_hidden_states:
            layer_expert_outputs = torch.zeros((batch_size * sequence_length, layer.mlp.top_k, hidden_dim), dtype = moe_hidden_state.dtype, device = moe_hidden_state.device) # BN x topk x D

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(layer.mlp.num_experts):
            expert_layer = layer.mlp.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])
            # Index the correct hidden states and compute the expert hidden state for the current expert.
            current_state = moe_hidden_state[None, top_x].reshape(-1, hidden_dim)
            current_expert_output = expert_layer(current_state) 
            current_hidden_states = current_expert_output * routing_weights[top_x, idx, None]
            # However `index_add_` only support torch tensors for indexing so we'll use the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(moe_hidden_state.dtype))

            if return_hidden_states:
                layer_expert_outputs[top_x, idx] = current_expert_output.to(layer_expert_outputs.dtype)

        final_hidden_states = (final_hidden_states).reshape(batch_size, sequence_length, hidden_dim)
        #######
        hidden_state = final_hidden_states
        hidden_state = residual + hidden_state

        # Resort ids/weights/layer outputs post-ablation
        selected_experts, routing_weights, layer_expert_outputs = _sort_gate_tensors(
            selected_experts.detach(),
            routing_weights.detach(),
            layer_expert_outputs.detach() if return_hidden_states else None
        )

        all_topk_experts.append(selected_experts.cpu())
        all_topk_weights.append(routing_weights.cpu().to(torch.float32))

        if return_hidden_states:
            all_hidden_states.append(hidden_state.view(-1, hidden_state.shape[2]).detach().cpu())
            all_expert_outputs.append(layer_expert_outputs.cpu())

    hidden_state = model.model.norm(hidden_state)
    logits = model.lm_head(hidden_state)

    return {
        'logits': logits,
        'all_topk_experts': all_topk_experts,
        'all_topk_weights': all_topk_weights,
        'all_pre_mlp_hidden_states': all_pre_mlp_hidden_states,
        'all_hidden_states': all_hidden_states,
        'all_expert_outputs': all_expert_outputs
    }

@torch.no_grad()
def run_qwen3moe_with_ablation(model, input_ids, attention_mask, return_hidden_states = False, ablation_targets = {}, ablation_penalty = 1e6, history_k = 0):
    """
    Params:
        @model: A model of class `Qwen3MoeForCausalLM`.
        @input_ids: A B x N tensor of inputs IDs on the same device as `model`.
        @attention_mask: A B x N tensor of mask indicators on the same device as `model`.
        @return_hidden_states: Boolean; whether to return hidden_states themselves
        @ablation_targets: Dict specifying ablation rules, noting that both experts and layers are 0-indexed. For example:
         `{
            5: {
                (0,): 2, # Ablate expert #2 in layer #5 if previous expert was expert #0
                (1, 2): 10 # Ablate expert #10 in layer #5 if previous expert was expert #1 -> expert #2 (in previous 2 layers)
            },
            6: {
               ...
            }
         }`
        @ablation_penalty: Large positive value subtracted from logits for ablation..
        @history_k: Int; whyich of the top-k experts to track for path history (1-indexed)

    Returns:
        A dictionary with keys:
        - `logits`: The standard B x N x V LM output
        - `all_topk_experts`: A list of length equal to the number of MoE layers, with each element a BN x topk tensor of expert IDs
        - `all_topk_weights`: A list of length equal to the number of MoE layers, with each element a BN x topk tensor of expert weights
        - `token_path_history`: B x N x num_layers tensor showing the top-`history_k` expert chosen at each layer.
        - `all_pre_mlp_hidden_states`: If return_hidden_states, a list of length equal to the number of MoE layers, with each element a BN x D tensor of pre-MLP hidden states
        - `all_hidden_states`: If return_hidden_states, a list of length equal to the number of MoE layers, with each element a BN x D tensor of post-layer hidden states
        - `all_expert_outputs`: If return_hidden_states, a list of length equal to the number of MoE layers, with each element a BN x topk x D tensor of expert outputs (pre-weighting)
    """
    input_embeds = model.model.embed_tokens(input_ids)

    B, N, D = input_embeds.shape
    num_layers = len(model.model.layers)
    token_path_history = torch.full((B, N, num_layers), -1, dtype = torch.long, device = input_ids.device)

    cache_position = torch.arange(0, input_embeds.shape[1], device = input_embeds.device)
    position_ids = cache_position.unsqueeze(0)
    causal_mask = model.model._update_causal_mask(attention_mask, input_embeds, cache_position, None, None)
    
    hidden_state = input_embeds
    position_embeddings = model.model.rotary_emb(hidden_state, position_ids)

    all_topk_experts = []
    all_topk_weights = []
    all_pre_mlp_hidden_states = []
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

        # Calculate Original Logits (used for weights and original top-1 check)
        router_logits = layer.mlp.gate(moe_hidden_state) # Size (BN, n_experts)

        ### Ablation Logic Start ####
        modified_router_logits = router_logits.clone() # Start with original logits

        if layer_ix in ablation_targets:
            # Determine original top-1 choice BEFORE applying any penalties
            original_top1_indices_flat = torch.argmax(router_logits, dim=-1) # Shape (BN,)

            targets_for_layer = ablation_targets[layer_ix]
            for path_prefix_tuple, target_expert_to_ablate in targets_for_layer.items():
                prefix_len = len(path_prefix_tuple)
                if layer_ix >= prefix_len:
                    history_slice = token_path_history[:, :, layer_ix - prefix_len : layer_ix]
                    target_prefix = torch.tensor(path_prefix_tuple, device=history_slice.device).view(1, 1, prefix_len)
                    token_mask = torch.all(history_slice == target_prefix, dim=2)
                    token_mask_flat = token_mask.view(-1) # Shape (BN,)

                    # Create mask for tokens that ALSO originally chose the target expert as top-1
                    top1_matches_target_mask = (original_top1_indices_flat == target_expert_to_ablate) # Shape (BN,)

                    # Final mask: apply ablation only if path prefix matches AND original top-1 was the target
                    final_ablation_mask = token_mask_flat & top1_matches_target_mask # Shape (BN,)

                    # Apply penalty using the final mask
                    if torch.any(final_ablation_mask):
                         modified_router_logits[final_ablation_mask, target_expert_to_ablate] -= ablation_penalty
        ### Ablation Logic End ###

        # 3. Select Experts using *Modified* Logits
        _, selected_experts = torch.topk(modified_router_logits, layer.mlp.top_k, dim = -1, sorted = True) # BN x topk

        # --- Update history (using the selected top-k'th expert) ---
        current_history_k = min(history_k, selected_experts.shape[-1] - 1)
        history_expert_ids = selected_experts[:, current_history_k].view(B, N)
        if token_path_history.shape[2] > layer_ix:
             token_path_history[:, :, layer_ix] = history_expert_ids

        # 4. Calculate Weights using *Original* Logits and *Selected* Experts
        routing_weights = torch.nn.functional.softmax(router_logits, dim = 1, dtype = torch.float)
        routing_weights = torch.gather(routing_weights, 1, selected_experts)
        if layer.mlp.norm_topk_prob:
             routing_weights /= routing_weights.sum(dim = -1, keepdim = True)
        routing_weights = routing_weights.to(moe_hidden_state.dtype)

        # --- Expert Computation --- (same as before)
        final_hidden_states = torch.zeros((batch_size * sequence_length, hidden_dim), dtype = moe_hidden_state.dtype, device = moe_hidden_state.device)
        
        # One hot encode the selected experts to create an expert mask 
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes = layer.mlp.num_experts).permute(2, 1, 0)

        if return_hidden_states:
            layer_expert_outputs = torch.zeros((batch_size * sequence_length, layer.mlp.top_k, hidden_dim), dtype = moe_hidden_state.dtype, device = moe_hidden_state.device) # BN x topk x D

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(layer.mlp.num_experts):
            expert_layer = layer.mlp.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])
            # Index the correct hidden states and compute the expert hidden state for the current expert.
            current_state = moe_hidden_state[None, top_x].reshape(-1, hidden_dim)
            current_expert_output = expert_layer(current_state) 
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
            all_hidden_states.append(hidden_state.view(-1, hidden_state.shape[2]).detach().cpu())
            all_expert_outputs.append(layer_expert_outputs.detach().cpu())

    hidden_state = model.model.norm(hidden_state)
    logits = model.lm_head(hidden_state)

    return {
        'logits': logits,
        'all_topk_experts': all_topk_experts,
        'all_topk_weights': all_topk_weights,
        'token_path_history': token_path_history,
        'all_pre_mlp_hidden_states': all_pre_mlp_hidden_states,
        'all_hidden_states': all_hidden_states,
        'all_expert_outputs': all_expert_outputs
    }
