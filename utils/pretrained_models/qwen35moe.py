"""
Reversed engineered forward pass for Qwen
- Supports Qwen-3.5 MoE (text-only)
- See https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py
"""
import torch
import torch.nn.functional as F
from transformers.masking_utils import create_causal_mask

@torch.no_grad()
def run_qwen35moe_return_topk(model, input_ids, attention_mask, return_hidden_states: bool = False):
    """
    Params:
        @model: A model of class `Qwen3_5MoeForCausalLM` or `Qwen3_5MoeForConditionalGeneration` (text-only usage).
        @input_ids: A (B, N) tensor of input IDs on the same device as `model`.
        @attention_mask: A (B, N) tensor of mask indicators on the same device as `model`.
        @return_hidden_states: Boolean; whether to return hidden_states themselves.

    Returns:
        A dictionary with keys:
        - `logits`: (B, N, V) LM outputs
        - `all_topk_experts`: List (len = # MoE layers) of (BN, topk) expert IDs tensors
        - `all_topk_weights`: List (len = # MoE layers) of (BN, topk) expert weight tensors
        - `all_pre_mlp_hidden_states`: (optional) List (len = # MoE layers) of (BN, D) pre-MLP activations
        - `all_router_logits: (optional) List (len = # MoE layers) of (BN, n_experts) router *logits*
        - `all_hidden_states`: (optional) List (len = # MoE layers) of (BN, D) post-layer activations
        - `all_expert_outputs`: (optional) List (len = # MoE layers) of (BN, topk, D) pre-weighting expert outputs ****(None if not in eager experts mode)
    """
    # Support both text-only LM and multimodal wrapper (text-only path)
    if hasattr(model, 'model') and hasattr(model.model, 'language_model'):
        text_model = model.model.language_model
        lm_head = model.lm_head
        config = model.config.text_config if hasattr(model.config, 'text_config') else text_model.config
    else:
        text_model = model.model
        lm_head = model.lm_head
        config = text_model.config

    input_embeds = text_model.embed_tokens(input_ids)
    B, N, D = input_embeds.shape

    # Qwen3.5 text uses 4x position ids: text + temporal + height + width
    position_ids = torch.arange(0, N, device = input_embeds.device).view(1, 1, -1).expand(4, B, -1)
    text_position_ids = position_ids[0]   # (B, N)
    rope_position_ids = position_ids[1:]  # (3, B, N)

    causal_mask = create_causal_mask(
        config = config,
        inputs_embeds = input_embeds,
        attention_mask = attention_mask,
        past_key_values = None,
        position_ids = text_position_ids,
    )

    linear_attn_mask = attention_mask
    if attention_mask is not None and torch.all(attention_mask == 1):
        linear_attn_mask = None

    hidden_state = input_embeds
    position_embeddings = text_model.rotary_emb(hidden_state, rope_position_ids)

    experts_impl = None
    for obj in (model, getattr(model, 'model', None), text_model, getattr(model, 'config', None), config):
        if obj is None:
            continue

        getter = getattr(obj, 'get_experts_implementation', None)
        if callable(getter):
            try:
                experts_impl = getter()
            except Exception:
                experts_impl = None
        if experts_impl is not None:
            break

        for attr in ('_experts_implementation_internal', '_experts_implementation', 'experts_implementation'):
            value = getattr(obj, attr, None)
            if value is not None:
                experts_impl = value
                break
        if experts_impl is not None:
            break

    is_eager_experts = experts_impl in [None, 'eager']

    all_topk_experts = []
    all_topk_weights = []
    all_pre_mlp_hidden_states = []
    all_router_logits = []
    all_hidden_states = []
    all_expert_outputs = []

    for layer_ix, layer in enumerate(text_model.layers):
        # Token mixer
        residual = hidden_state
        hidden_state = layer.input_layernorm(hidden_state)

        if layer.layer_type == 'linear_attention':
            attn_out = layer.linear_attn(
                hidden_states = hidden_state,
                cache_params = None,
                attention_mask = linear_attn_mask,
            )
            hidden_state = residual + attn_out
        else:
            attn_out, _ = layer.self_attn(
                hidden_states = hidden_state,
                attention_mask = causal_mask,
                position_ids = text_position_ids,
                past_key_values = None,
                position_embeddings = position_embeddings,
            )
            hidden_state = residual + attn_out

        # MoE MLP
        residual = hidden_state
        hidden_state = layer.post_attention_layernorm(hidden_state)

        if return_hidden_states:
            all_pre_mlp_hidden_states.append(hidden_state.view(-1, hidden_state.shape[2]).detach().cpu())

        BN = B * N
        moe_hidden_state = hidden_state.view(BN, D)

        # Router output in current HF is already post-softmax over experts
        router_probs, routing_weights, selected_experts = layer.mlp.gate(moe_hidden_state)

        if is_eager_experts:
            experts = layer.mlp.experts
            final_hidden_states = torch.zeros((BN, D), dtype = moe_hidden_state.dtype, device = moe_hidden_state.device)

            if return_hidden_states:
                layer_expert_outputs = torch.zeros((BN, layer.mlp.gate.top_k, D), dtype = moe_hidden_state.dtype, device = moe_hidden_state.device)

            expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes = experts.num_experts).permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim = (-1, -2)), 0).nonzero()

            for expert_idx in expert_hit:
                expert_idx = expert_idx[0]
                if expert_idx == experts.num_experts:
                    continue
                top_k_pos, token_idx = torch.where(expert_mask[expert_idx])

                current_state = moe_hidden_state[token_idx]
                gate, up = F.linear(current_state, experts.gate_up_proj[expert_idx]).chunk(2, dim = -1)
                current_expert_output = experts.act_fn(gate) * up
                current_expert_output = F.linear(current_expert_output, experts.down_proj[expert_idx])

                current_hidden_states = current_expert_output * routing_weights[token_idx, top_k_pos, None]
                final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

                if return_hidden_states:
                    layer_expert_outputs[token_idx, top_k_pos] = current_expert_output.to(layer_expert_outputs.dtype)

            shared_expert_output = layer.mlp.shared_expert(moe_hidden_state)
            shared_expert_output = torch.sigmoid(layer.mlp.shared_expert_gate(moe_hidden_state)) * shared_expert_output
            final_hidden_states = final_hidden_states + shared_expert_output.to(final_hidden_states.dtype)

            hidden_state = residual + final_hidden_states.view(B, N, D)

            if return_hidden_states:
                all_router_logits.append(router_probs.detach().cpu())
                all_hidden_states.append(hidden_state.view(BN, D).detach().cpu())
                all_expert_outputs.append(layer_expert_outputs.detach().cpu())
        else:
            mlp_out = layer.mlp(hidden_state)
            if isinstance(mlp_out, tuple):
                mlp_out = mlp_out[0]
            hidden_state = residual + mlp_out

            if return_hidden_states:
                all_router_logits.append(router_probs.detach().cpu())
                all_hidden_states.append(hidden_state.view(BN, D).detach().cpu())
                all_expert_outputs.append(None)

        all_topk_experts.append(selected_experts.detach().cpu())
        all_topk_weights.append(routing_weights.detach().cpu().to(torch.float32))

    hidden_state = text_model.norm(hidden_state)
    logits = lm_head(hidden_state)

    return {
        'logits': logits,
        'all_topk_experts': all_topk_experts,
        'all_topk_weights': all_topk_weights,
        'all_pre_mlp_hidden_states': all_pre_mlp_hidden_states,
        'all_router_logits': all_router_logits,
        'all_hidden_states': all_hidden_states,
        'all_expert_outputs': all_expert_outputs
    }