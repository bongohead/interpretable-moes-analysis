"""
Reversed engineered forward pass for Hunyuan-A13B models.
"""
import torch
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask, _prepare_4d_causal_attention_mask_for_sdpa
from ._pretrained_helpers import _sort_gate_tensors

@torch.no_grad()
def run_hunyuan_return_topk(model, input_ids, attention_mask, return_hidden_states: bool = False):
    """
    Params:
        @model: A model of class `HunYuanMoEV1ForCausalLM`.
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
    B, N = input_ids.shape
    input_embeds = model.model.embed_tokens(input_ids)

    position_ids = torch.arange(0, N, device = input_embeds.device).unsqueeze(0)
    # causal_mask = _prepare_4d_causal_attention_mask(attention_mask, (B, N), input_embeds, 0,)
    if model.model._use_flash_attention_2:
        causal_mask = (attention_mask if (attention_mask is not None and 0 in attention_mask) else None) # 2‑D or None
    elif model.model._use_sdpa:
        causal_mask = _prepare_4d_causal_attention_mask_for_sdpa(attention_mask, (B, N), input_embeds, 0)
    else:
        causal_mask = _prepare_4d_causal_attention_mask(attention_mask, (B, N), input_embeds, 0)

    hidden_state = input_embeds
    all_topk_experts = []
    all_topk_weights = []
    all_pre_mlp_hidden_states = []
    all_router_logits = []
    all_hidden_states = []
    all_expert_outputs = []

    for layer_ix, layer in enumerate(model.model.layers):
        _, _, D = hidden_state.shape
        # SA
        residual = hidden_state
        hidden_state = layer.input_layernorm(hidden_state)
        hidden_state, *_ = layer.self_attn(hidden_states = hidden_state, attention_mask = causal_mask, position_ids = position_ids)
        hidden_state = residual + hidden_state
        residual = hidden_state
        hidden_state = layer.post_attention_layernorm(hidden_state)

        if hasattr(layer.mlp, 'gate') and return_hidden_states:
            all_pre_mlp_hidden_states.append(hidden_state.view(-1, hidden_state.shape[2]).detach().cpu())

        ####### HunYuanMoE - below code replaces hidden_state = layer.mlp(hidden_state) #######
        if not hasattr(layer.mlp, 'gate'):
            hidden_state = layer.mlp(hidden_state)
            hidden_state = residual + hidden_state
            continue

        # Shared dense MLP
        if model.config.use_mixed_mlp_moe:
            shared_out = layer.mlp.shared_mlp(hidden_state) # (B, N, D)

        # ---- GATE ---------------------------------------------------------
        (_, _), combine_w, dispatch_m, _ = layer.mlp.gate(hidden_state)
        S, E, C = combine_w.shape # S = BN, E = n_experts, C = capacity slots per expert (Hunyuan has max cap limit)
        # Capacity is filled in the order tokens appear in the BN list, so early tokens have first dibs on every expert C's list

        # softmax-normalised weights per token / expert
        token_exp_w = combine_w.sum(dim = 2) # (S, E)
        top_k = model.config.moe_topk[layer.layer_idx] if isinstance(model.config.moe_topk, list) else model.config.moe_topk
        routing_weights, selected_experts = torch.topk(token_exp_w, k=top_k, dim=1)

        if return_hidden_states:
            router_logits = layer.mlp.gate.wg(hidden_state.reshape(-1, D).float())

        # ---- DISPATCH -----------------------------------------------------
        hidden_flat = hidden_state.reshape(-1, D) # (S, D)
        disp_inp = torch.einsum("sec,sm->ecm", dispatch_m.to(hidden_state.dtype), hidden_flat)  # (E, C, D)
        exp_chunks = disp_inp.chunk(E, dim=0)
        exp_outputs = torch.cat([exp(chunk) for exp, chunk in zip(layer.mlp.experts, exp_chunks)], dim=0)                                                       # [E,C,D]
        combined_out  = torch.einsum("sec,ecm->sm", combine_w.to(hidden_state.dtype), exp_outputs).reshape(B, N, D) # (B, N, D)

        hidden_state = shared_out + combined_out if model.config.use_mixed_mlp_moe else combined_out
        hidden_state = residual + hidden_state

        # Storage
        all_topk_experts.append(selected_experts.detach().cpu())
        all_topk_weights.append(routing_weights.detach().cpu().to(torch.float32))

        if return_hidden_states:
            all_router_logits.append(router_logits.detach().cpu())
            all_hidden_states.append(hidden_state.reshape(-1, D).detach().cpu())

            # Gather exper outputs (pre-weighting) 
            layer_expert_outputs = torch.zeros((S, top_k, D), dtype = hidden_flat.dtype, device = hidden_flat.device)
            rank_in_tok = torch.zeros(S, dtype = torch.int64, device = hidden_flat.device)
            t_idx, e_idx, c_idx = dispatch_m.nonzero(as_tuple = True)
            for t, e, c in zip(t_idx.tolist(), e_idx.tolist(), c_idx.tolist()):
                kpos = rank_in_tok[t].item()
                if kpos < top_k:
                    layer_expert_outputs[t, kpos] = exp_outputs[e, c]
                    rank_in_tok[t] += 1
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

