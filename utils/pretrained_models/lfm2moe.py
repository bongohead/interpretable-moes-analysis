"""
Reversed engineered forward pass for LFM2-MoE
- Supports LiquidAI/LFM2-8B-A1B style models
- See https://github.com/huggingface/transformers/blob/main/src/transformers/models/lfm2_moe/modeling_lfm2_moe.py
"""
import torch
import torch.nn.functional as F
from transformers.masking_utils import create_causal_mask

@torch.no_grad()
def run_lfm2moe_return_topk(model, input_ids, attention_mask, return_hidden_states = False):
    """
    Params:
        @model: A model of class `Lfm2MoeForCausalLM`.
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
    text_model = model.model
    config = text_model.config

    input_embeds = text_model.embed_tokens(input_ids)
    B, N, D = input_embeds.shape

    position_ids = torch.arange(0, N, device = input_embeds.device).unsqueeze(0)

    causal_mask = create_causal_mask(
        config = config,
        inputs_embeds = input_embeds,
        attention_mask = attention_mask,
        past_key_values = None,
        position_ids = position_ids,
    )

    # LFM2 skips masking for decoding stage only; for prompt forward with seq_len > 1 we keep the 2D padding mask.
    linear_attention = attention_mask if input_embeds.shape[1] != 1 else None

    hidden_state = input_embeds
    position_embeddings = text_model.pos_emb(hidden_state, position_ids = position_ids)

    experts_impl = None
    for obj in (model, text_model, getattr(model, 'config', None), config):
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
        operator_input = layer.operator_norm(hidden_state)

        if layer.is_attention_layer:
            operator_output, _ = layer.self_attn(
                hidden_states = operator_input,
                position_embeddings = position_embeddings,
                attention_mask = causal_mask,
                position_ids = position_ids,
                past_key_values = None,
            )
        else:
            operator_output = layer.conv(
                hidden_states = operator_input,
                past_key_values = None,
                attention_mask = linear_attention,
            )

        hidden_state = residual + operator_output

        # Feed-forward
        residual = hidden_state
        ffn_input = layer.ffn_norm(hidden_state)

        is_moe = hasattr(layer.feed_forward, 'gate') and hasattr(layer.feed_forward, 'experts')
        if not is_moe:
            ff_out = layer.feed_forward(ffn_input)
            hidden_state = residual + ff_out
            continue

        if return_hidden_states:
            all_pre_mlp_hidden_states.append(ffn_input.view(-1, D).detach().cpu())

        BN = B * N
        moe_hidden_state = ffn_input.view(BN, D)

        router_logits = layer.feed_forward.gate(moe_hidden_state)  # (BN, E), raw pre-sigmoid logits
        selected_experts, routing_weights = layer.feed_forward.route_tokens_to_experts(router_logits)

        if is_eager_experts:
            experts = layer.feed_forward.experts
            final_hidden_states = torch.zeros((BN, D), dtype = moe_hidden_state.dtype, device = moe_hidden_state.device)

            if return_hidden_states:
                layer_expert_outputs = torch.zeros((BN, layer.feed_forward.top_k, D), dtype = moe_hidden_state.dtype, device = moe_hidden_state.device)

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

            ff_out = final_hidden_states.view(B, N, D)
        else:
            ff_out = layer.feed_forward(ffn_input)
            if isinstance(ff_out, tuple):
                ff_out = ff_out[0]

        hidden_state = residual + ff_out

        all_topk_experts.append(selected_experts.detach().cpu())
        all_topk_weights.append(routing_weights.detach().cpu().to(torch.float32))

        if return_hidden_states:
            all_router_logits.append(router_logits.detach().cpu())
            all_hidden_states.append(hidden_state.view(BN, D).detach().cpu())
            all_expert_outputs.append(layer_expert_outputs.detach().cpu() if is_eager_experts else None)

    hidden_state = text_model.embedding_norm(hidden_state)
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