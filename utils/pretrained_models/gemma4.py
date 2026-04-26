"""
Reversed engineered forward pass for Gemma 4
- Supports google/gemma-4-26B-A4B-it (text-only path)
- See https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma4/modeling_gemma4.py
- A few notes:
    - all_pre_mlp_hidden_states returns the actual post-LN/scaled router input, which for Gemma 4 is different than the dense input and the expert inputs!
    - all_router_logits outputs router probs as this is a router obj output for Gemma 4, not logits
    - all_topk_weights and other orders are sorted by BEFORE post-expert scaling (which Gemma 4 has)
"""
import torch
import torch.nn.functional as F
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask

@torch.no_grad()
def run_gemma4_return_topk(model, input_ids, attention_mask, return_hidden_states = False):
    """
    Params:
        @model: A model of class `Gemma4ForConditionalGeneration` or `Gemma4ForCausalLM` (text-only usage).
        @input_ids: A (B, N) tensor of input IDs on the same device as `model`.
        @attention_mask: A (B, N) tensor of mask indicators on the same device as `model`.
        @return_hidden_states: Boolean; whether to return hidden_states themselves.

    Returns:
        A dictionary with keys:
        - `logits`: (B, N, V) LM outputs
        - `all_topk_experts`: List (len = # MoE layers) of (BN, topk) expert IDs tensors
        - `all_topk_weights`: List (len = # MoE layers) of (BN, topk) expert weight tensors
        - `all_pre_mlp_hidden_states`: (optional) List (len = # MoE layers) of (BN, D) pre-MLP activations
        - `all_router_logits: (optional) List (len = # MoE layers) of (BN, n_experts) router *probs*
        - `all_hidden_states`: (optional) List (len = # MoE layers) of (BN, D) post-layer activations
        - `all_expert_outputs`: (optional) List (len = # MoE layers) of (BN, topk, D) pre-weighting expert outputs
    """
    # Support both text-only LM and multimodal wrapper (text-only path)
    if hasattr(model, 'model') and hasattr(model.model, 'language_model'):
        text_model = model.model.language_model
        text_config = model.config.text_config if hasattr(model.config, 'text_config') else text_model.config
        lm_head = model.lm_head

        # Match Gemma4Model.forward() for text-only inputs: multimodal placeholders are embedded as PAD.
        image_mask, video_mask, audio_mask = model.model.get_placeholder_mask(input_ids)
        multimodal_mask = image_mask | video_mask | audio_mask
        llm_input_ids = input_ids.clone()
        llm_input_ids[multimodal_mask] = text_config.pad_token_id
        input_embeds = text_model.embed_tokens(llm_input_ids)
    else:
        text_model = model.model
        text_config = text_model.config
        lm_head = model.lm_head
        llm_input_ids = input_ids
        input_embeds = text_model.embed_tokens(input_ids)

    B, N, D = input_embeds.shape

    per_layer_inputs = None
    if text_model.hidden_size_per_layer_input:
        per_layer_inputs = text_model.get_per_layer_inputs(llm_input_ids, input_embeds)
        per_layer_inputs = text_model.project_per_layer_inputs(input_embeds, per_layer_inputs)

    position_ids = torch.arange(0, N, device = input_embeds.device).unsqueeze(0)

    # Build both masks; each layer picks one via `layer_types`
    mask_kwargs = {
        'config': text_config,
        'inputs_embeds': input_embeds,
        'attention_mask': attention_mask,
        'past_key_values': None,
        'position_ids': position_ids,
    }
    causal_mask_mapping = {
        'full_attention': create_causal_mask(**mask_kwargs),
        'sliding_attention': create_sliding_window_causal_mask(**mask_kwargs),
    }

    hidden_state = input_embeds
    position_embeddings = {}
    for layer_type in text_model.unique_layer_types:
        position_embeddings[layer_type] = text_model.rotary_emb(hidden_state, position_ids, layer_type)

    shared_kv_states = {}

    all_topk_experts = []
    all_topk_weights = []
    all_pre_mlp_hidden_states = []
    all_router_logits = []
    all_hidden_states = []
    all_expert_outputs = []

    for layer_ix, layer in enumerate(text_model.layers):
        # SA
        residual = hidden_state
        hidden_state = layer.input_layernorm(hidden_state)
        hidden_state, _ = layer.self_attn(
            hidden_states = hidden_state,
            position_embeddings = position_embeddings[text_config.layer_types[layer_ix]],
            attention_mask = causal_mask_mapping[text_config.layer_types[layer_ix]],
            shared_kv_states = shared_kv_states,
            position_ids = position_ids,
            past_key_values = None,
        )
        hidden_state = layer.post_attention_layernorm(hidden_state)
        hidden_state = residual + hidden_state

        # Dense FFN branch
        residual = hidden_state
        hidden_state = layer.pre_feedforward_layernorm(hidden_state)
        hidden_states_1 = layer.mlp(hidden_state)

        if not layer.enable_moe_block:
            hidden_state = layer.post_feedforward_layernorm(hidden_states_1)
            hidden_state = residual + hidden_state

            if text_model.hidden_size_per_layer_input:
                residual_per_layer = hidden_state
                per_layer_input = per_layer_inputs[:, :, layer_ix, :]
                hidden_state = layer.per_layer_input_gate(hidden_state)
                hidden_state = layer.act_fn(hidden_state)
                hidden_state = hidden_state * per_layer_input
                hidden_state = layer.per_layer_projection(hidden_state)
                hidden_state = layer.post_per_layer_input_norm(hidden_state)
                hidden_state = residual_per_layer + hidden_state

            hidden_state = hidden_state * layer.layer_scalar
            continue

        hidden_states_1 = layer.post_feedforward_layernorm_1(hidden_states_1)

        # MoE branch (uses the post-attention residual stream, not the dense-branch input)
        hidden_states_flat = residual.reshape(-1, residual.shape[-1])  # (BN, D)

        # Router path: matches Gemma4TextRouter.forward(), but exposes the actual router projection input.
        router_input = layer.router.norm(hidden_states_flat)
        router_input = router_input * layer.router.scale * layer.router.scalar_root_size

        router_logits_raw = layer.router.proj(router_input)
        router_probs = torch.nn.functional.softmax(router_logits_raw, dim = -1)

        # Top-k is sorted by unscaled router probabilities.
        top_k_probs, top_k_index = torch.topk(
            router_probs,
            k = layer.router.config.top_k_experts,
            dim = -1,
            sorted = True,
        )

        # Actual expert coefficients: top-k renorm, then per-expert scale.
        # Keep the original top-k order from router probabilities.
        top_k_weights = top_k_probs / top_k_probs.sum(dim = -1, keepdim = True)
        top_k_weights = top_k_weights * layer.router.per_expert_scale[top_k_index]

        if return_hidden_states:
            all_pre_mlp_hidden_states.append(router_input.detach().cpu())

        # Expert input path is separate from router input
        hidden_states_2 = layer.pre_feedforward_layernorm_2(hidden_states_flat)

        experts = layer.experts
        final_hidden_states = torch.zeros((B * N, D), dtype = hidden_states_2.dtype, device = hidden_states_2.device)

        if return_hidden_states:
            layer_expert_outputs = torch.zeros((B * N, text_config.top_k_experts, D), dtype = hidden_states_2.dtype, device = hidden_states_2.device)

        # Match Gemma4TextExperts.forward() eager path
        expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes = experts.num_experts).permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim = (-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == experts.num_experts:
                continue

            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states_2[token_idx]

            gate, up = F.linear(current_state, experts.gate_up_proj[expert_idx]).chunk(2, dim = -1)
            current_expert_output = experts.act_fn(gate) * up
            current_expert_output = F.linear(current_expert_output, experts.down_proj[expert_idx])

            current_hidden_states = current_expert_output * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

            if return_hidden_states:
                layer_expert_outputs[token_idx, top_k_pos] = current_expert_output.to(layer_expert_outputs.dtype)

        hidden_states_2 = final_hidden_states.view(B, N, D)
        hidden_states_2 = layer.post_feedforward_layernorm_2(hidden_states_2)

        # Combine dense + MoE branches
        hidden_state = hidden_states_1 + hidden_states_2
        hidden_state = layer.post_feedforward_layernorm(hidden_state)
        hidden_state = residual + hidden_state

        if text_model.hidden_size_per_layer_input:
            residual_per_layer = hidden_state
            per_layer_input = per_layer_inputs[:, :, layer_ix, :]
            hidden_state = layer.per_layer_input_gate(hidden_state)
            hidden_state = layer.act_fn(hidden_state)
            hidden_state = hidden_state * per_layer_input
            hidden_state = layer.per_layer_projection(hidden_state)
            hidden_state = layer.post_per_layer_input_norm(hidden_state)
            hidden_state = residual_per_layer + hidden_state

        hidden_state = hidden_state * layer.layer_scalar

        all_topk_experts.append(top_k_index.detach().cpu())
        all_topk_weights.append(top_k_weights.detach().cpu().to(torch.float32))

        if return_hidden_states:
            all_router_logits.append(router_probs.detach().cpu())
            all_hidden_states.append(hidden_state.view(-1, D).detach().cpu())
            all_expert_outputs.append(layer_expert_outputs.detach().cpu())

    hidden_state = text_model.norm(hidden_state)
    logits = lm_head(hidden_state)

    final_logit_softcapping = getattr(text_config, 'final_logit_softcapping', None)
    if final_logit_softcapping is not None:
        logits = logits / final_logit_softcapping
        logits = torch.tanh(logits)
        logits = logits * final_logit_softcapping

    return {
        'logits': logits,
        'all_topk_experts': all_topk_experts,
        'all_topk_weights': all_topk_weights,
        'all_pre_mlp_hidden_states': all_pre_mlp_hidden_states,
        'all_router_logits': all_router_logits,
        'all_hidden_states': all_hidden_states,
        'all_expert_outputs': all_expert_outputs
    }