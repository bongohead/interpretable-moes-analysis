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
        hidden_state, _, _ = layer.self_attn(hidden_states = hidden_state, attention_mask = causal_mask, position_ids = position_ids, position_embeddings = position_embeddings)
        hidden_state = residual + hidden_state
        residual = hidden_state
        hidden_state = layer.post_attention_layernorm(hidden_state)
        if return_hidden_states:
            all_pre_mlp_hidden_states.append(hidden_state.view(-1, hidden_state.shape[2]).detach().cpu())

        ####### OlMoESparseMoeBlock - below code replaces hidden_state = layer.mlp(hidden_state)
        batch_size, sequence_length, hidden_dim = hidden_state.shape
        moe_hidden_state = hidden_state.view(-1, hidden_dim)
        router_logits = layer.mlp.gate(moe_hidden_state)

        routing_weights = torch.nn.functional.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, layer.mlp.top_k, dim=-1, sorted = True)
        routing_weights = routing_weights.to(moe_hidden_state.dtype)
        final_hidden_states = torch.zeros((batch_size * sequence_length, hidden_dim), dtype = hidden_state.dtype, device = hidden_state.device)
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes = layer.mlp.num_experts).permute(2, 1, 0)

        if return_hidden_states:
            layer_expert_outputs = torch.zeros((batch_size * sequence_length, layer.mlp.top_k, hidden_dim), dtype = moe_hidden_state.dtype, device = moe_hidden_state.device) # BN x topk x D

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(layer.mlp.num_experts):
            expert_layer = layer.mlp.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            current_state = moe_hidden_state[None, top_x].reshape(-1, hidden_dim)
            current_expert_output = expert_layer(current_state) 
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
def run_olmoe_return_topk_with_path_ablation(model, input_ids, attention_mask, ablation_targets = {}, ablate_if_in_topk: bool = False, ablation_penalty = 1e9):
    """
    Params:
        @model: A model of class `OlmoeForCausalLM`.
        @input_ids: A B x N tensor of inputs IDs on the same device as `model`.
        @attention_mask: A B x N tensor of mask indicators on the same device as `model`.
        @ablation_targets: Dict specifying ablation rules, noting that both experts and layers are 0-indexed. For example:
         `{
            5: [
                ((0,), 2), # Ablate expert #2 in layer #5 if previous expert was expert #0
                ((1, 2), 10) # Ablate expert #10 in layer #5 if previous expert was expert #1 -> expert #2 (in previous 2 layers)
            ],
            6: {
               ...
            }
         }`
        @ablate_if_in_topk: If True, ablate target expert if it appears anywhere in the original top-k choices for matching tokens.
         If False (default), ablate only if it was the original top-1 choice.
        @ablation_penalty: Large positive value subtracted from logits for ablation..

    Returns:
        A dictionary with keys:
        - `logits`: The standard B x N x V LM output
        - `all_topk_experts`: A list of length equal to the number of MoE layers, with each element a BN x topk tensor of expert IDs
        - `all_topk_weights`: A list of length equal to the number of MoE layers, with each element a BN x topk tensor of expert weights
        - `token_path_history`: B x N x num_layers tensor showing the top-1 expert chosen at each layer.
        - `num_ablations_applied`: Integer count of how many times the ablation penalty was applied.
    """
    input_embeds = model.model.embed_tokens(input_ids)
    hidden_state = input_embeds

    B, N, D = hidden_state.shape
    num_layers = len(model.model.layers)
    token_path_history = torch.full((B, N, num_layers), -1, dtype = torch.long, device = input_ids.device)

    cache_position = torch.arange(0, N, device = input_embeds.device)
    position_ids = cache_position.unsqueeze(0)
    causal_mask = model.model._update_causal_mask(attention_mask, input_embeds, cache_position, None, None)
    position_embeddings = model.model.rotary_emb(hidden_state, position_ids)

    all_topk_experts = []
    all_topk_weights = []
    num_ablations_applied = 0

    for layer_ix, layer in enumerate(model.model.layers):
        # SA
        residual = hidden_state
        hidden_state = layer.input_layernorm(hidden_state)
        hidden_state, _, _ = layer.self_attn(hidden_states = hidden_state, attention_mask = causal_mask, position_ids = position_ids, position_embeddings = position_embeddings)
        hidden_state = residual + hidden_state
        residual = hidden_state
        hidden_state = layer.post_attention_layernorm(hidden_state)

        ####### OlMoESparseMoeBlock - below code replaces hidden_state = layer.mlp(hidden_state)
        batch_size, sequence_length, hidden_dim = hidden_state.shape
        moe_hidden_state = hidden_state.view(-1, hidden_dim)
        router_logits = layer.mlp.gate(moe_hidden_state)

        ### Ablation Section
        modified_router_logits = router_logits.clone()

        if layer_ix in ablation_targets and layer_ix > 0: # Check if rules exist AND we have any history
            original_topk_indices = torch.topk(router_logits, top_k, dim=-1, sorted=False).indices # BN x topk

            rules_list_for_layer = ablation_targets[layer_ix]

            for path_prefix_tuple, target_expert_to_ablate in rules_list_for_layer:
                prefix_len = len(path_prefix_tuple)
                if layer_ix >= prefix_len:
                    history_slice = token_path_history[:, :, layer_ix - prefix_len : layer_ix]
                    target_prefix = torch.tensor(path_prefix_tuple, device=history_slice.device).view(1, 1, prefix_len)
                    token_mask = torch.all(history_slice == target_prefix, dim=2)
                    token_mask_flat = token_mask.view(-1) # Shape (BN,)

                    # Determine which tokens matching the path also had the target expert in their original top-k
                    if ablate_if_in_topk:
                        # Check if target expert is ANYWHERE in the original top-k for each token
                        # Shape (BN, top_k) -> (BN,)
                        target_in_original_topk_mask = torch.any(original_topk_indices == target_expert_to_ablate, dim=1)
                        # Final mask: path prefix matches AND target was in original top-k
                        final_ablation_mask = token_mask_flat & target_in_original_topk_mask
                    else: # Original behavior: ablate only if it was top-1
                        original_top1_indices_flat = original_topk_indices[:, 0] # Get top-1 index
                        top1_matches_target_mask = (original_top1_indices_flat == target_expert_to_ablate)
                        # Final mask: path prefix matches AND target was original top-1
                        final_ablation_mask = token_mask_flat & top1_matches_target_mask

                    # Apply penalty using the final mask for this rule
                    if torch.any(final_ablation_mask):
                         num_ablations_for_rule = final_ablation_mask.sum().item()
                         num_ablations_applied += num_ablations_for_rule
                         modified_router_logits[final_ablation_mask, target_expert_to_ablate] -= ablation_penalty

            # original_top1_indices_flat = torch.argmax(router_logits, dim = -1) # Shape (BN,)
            # rules_list_for_layer = ablation_targets[layer_ix] # List of (prefix, target_e) tuples

            # for path_prefix_tuple, target_expert_to_ablate in rules_list_for_layer:
            #     prefix_len = len(path_prefix_tuple)
            #     if layer_ix >= prefix_len: # Check if enough history exists for this specific rule
            #         history_slice = token_path_history[:, :, layer_ix - prefix_len : layer_ix]
            #         target_prefix = torch.tensor(path_prefix_tuple, device = history_slice.device).view(1, 1, prefix_len)
            #         token_mask = torch.all(history_slice == target_prefix, dim = 2)
            #         token_mask_flat = token_mask.view(-1)
            #         # Check if original top-1 was the target expert for the tokens matching the path
            #         top1_matches_target_mask = (original_top1_indices_flat == target_expert_to_ablate)
            #         # Final mask for this specific rule
            #         final_ablation_mask = token_mask_flat & top1_matches_target_mask

            #         if torch.any(final_ablation_mask):
            #              num_ablations_for_rule = final_ablation_mask.sum().item()
            #              num_ablations_applied += num_ablations_for_rule
            #              modified_router_logits[final_ablation_mask, target_expert_to_ablate] -= ablation_penalty

        # Select Experts using *modified* logits
        top_k = layer.mlp.top_k
        _, selected_experts = torch.topk(modified_router_logits, top_k, dim = -1, sorted = True) # BN x topk

        # --- Update history (using the top-1 expert, index 0) ---
        if top_k > 0: # Only update history if experts were selected
             history_expert_ids = selected_experts[:, 0].view(B, N) # Reshape to B x N
             if token_path_history.shape[2] > layer_ix:
                  token_path_history[:, :, layer_ix] = history_expert_ids
        ###

        routing_weights = torch.nn.functional.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights = torch.gather(routing_weights, 1, selected_experts)
        routing_weights = routing_weights.to(moe_hidden_state.dtype)
        final_hidden_states = torch.zeros((batch_size * sequence_length, hidden_dim), dtype = hidden_state.dtype, device = hidden_state.device)
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes = layer.mlp.num_experts).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(layer.mlp.num_experts):
            expert_layer = layer.mlp.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            current_state = moe_hidden_state[None, top_x].reshape(-1, hidden_dim)
            current_expert_output = expert_layer(current_state) 
            current_hidden_states = current_expert_output * routing_weights[top_x, idx, None]

            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(moe_hidden_state.dtype))

        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        #######

        hidden_state = final_hidden_states
        hidden_state = residual + hidden_state
        
        all_topk_experts.append(selected_experts.detach().cpu())
        all_topk_weights.append(routing_weights.detach().cpu().to(torch.float32))

    hidden_state = model.model.norm(hidden_state)
    logits = model.lm_head(hidden_state)

    return {
        'logits': logits,
        'all_topk_experts': all_topk_experts,
        'all_topk_weights': all_topk_weights,
        'token_path_history': token_path_history.detach().cpu(),
        'num_ablations_applied': num_ablations_applied
    }





@torch.no_grad()
def run_olmoe_with_ablation_return_topk(model, input_ids, attention_mask, layers_to_ablate = [], topk_to_ablate = [], renorm = False, return_hidden_states = False):
    """
    Params:
        @model: A model of class `OlmoeForCausalLM`.
        @input_ids: A B x N tensor of inputs IDs on the same device as `model`.
        @attention_mask: A B x N tensor of mask indicators on the same device as `model`.
        @layers_to_ablate: A list of layer indices (0-indexed) for which experts will be ablated.
        @topk_to_ablate: A list of topk indices (0-indexed) for which experts will be ablated and replaced by zeros.
        @renorm: Whether to renormalize the sum of expert weights after ablation, to scale the post-ablation expert weight sum to the original expert weight sum.
        @return_hidden_states: Boolean; whether to return hidden_states themselves.

    Returns:
        Note that returned values are SORTED.

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
        hidden_state, _, _ = layer.self_attn(hidden_states = hidden_state, attention_mask = causal_mask, position_ids = position_ids, position_embeddings = position_embeddings)
        hidden_state = residual + hidden_state
        residual = hidden_state
        hidden_state = layer.post_attention_layernorm(hidden_state)
        if return_hidden_states:
            all_pre_mlp_hidden_states.append(hidden_state.view(-1, hidden_state.shape[2]).detach().cpu())

        ####### OlMoESparseMoeBlock - below code replaces hidden_state = layer.mlp(hidden_state)
        batch_size, sequence_length, hidden_dim = hidden_state.shape
        moe_hidden_state = hidden_state.view(-1, hidden_dim)
        router_logits = layer.mlp.gate(moe_hidden_state)

        routing_weights = torch.nn.functional.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, layer.mlp.top_k, dim=-1, sorted = True)
        routing_weights = routing_weights.to(moe_hidden_state.dtype)
        final_hidden_states = torch.zeros((batch_size * sequence_length, hidden_dim), dtype = hidden_state.dtype, device = hidden_state.device)
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
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes = layer.mlp.num_experts).permute(2, 1, 0)

        if return_hidden_states:
            layer_expert_outputs = torch.zeros((batch_size * sequence_length, layer.mlp.top_k, hidden_dim), dtype = moe_hidden_state.dtype, device = moe_hidden_state.device) # BN x topk x D

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(layer.mlp.num_experts):
            expert_layer = layer.mlp.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            current_state = moe_hidden_state[None, top_x].reshape(-1, hidden_dim)
            current_expert_output = expert_layer(current_state) 
            current_hidden_states = current_expert_output * routing_weights[top_x, idx, None]

            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(moe_hidden_state.dtype))

            if return_hidden_states:
                layer_expert_outputs[top_x, idx] = current_expert_output.to(layer_expert_outputs.dtype)

        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
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