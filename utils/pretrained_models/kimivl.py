"""
Reversed engineered forward pass for Kimi-VL models
- Supports Deepseekv3-based Kimi VL models, such as https://huggingface.co/collections/moonshotai/kimi-vl-a3b-67f67b6ac91d3b03d382dd85
- See https://huggingface.co/moonshotai/Kimi-VL-A3B-Instruct/blob/main/modeling_kimi_vl.py
- This is VERY similar to the original Deepseek v3. Only changes are an initial section to join image data into the input embeddings, and switching `model`
   to `model.language_model`.
"""
import torch
from ._pretrained_helpers import _sort_gate_tensors
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

@torch.no_grad()
def run_kimivl_return_topk(model, input_ids, attention_mask, pixel_values = None, image_grid_hws = None, return_hidden_states = False):
    """
    Params:
        @model: A model of class `KimiVLForConditionalGeneration`.
        @input_ids: A B x N tensor of inputs IDs on the same device as `model`.
        @attention_mask: A B x N tensor of mask indicators on the same device as `model`.
        @pixel_values:  A L x 3 x P x P tensor of image processor vision patches, only pass for image inputs.
        @image_grid_hws: A L_img x 2 tensor with (H, W) grids for each encoded image, only pass for image inputs.
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
    lang_model = model.language_model
    B, N = input_ids.shape[:2]
    position_ids = torch.arange(0, N, dtype=torch.long, device = model.device).unsqueeze(0)
    inputs_embeds = lang_model.model.embed_tokens(input_ids)

    if pixel_values is not None and pixel_values.numel() > 0: # Vision
        pixel_values = pixel_values.to(model.vision_tower.dtype)
        image_feats = model._extract_image_features(pixel_values, image_grid_hws)
        # Merge vision into input embeds
        inputs_embeds = inputs_embeds.to(image_feats.dtype)
        inputs_embeds = model._merge_with_image_features(inputs_embeds, input_ids, image_feats)

    if lang_model.model._use_flash_attention_2:
        attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
    else:
        attention_mask = _prepare_4d_causal_attention_mask(attention_mask, (B, N), inputs_embeds, 0,)

    hidden_state = inputs_embeds
    all_topk_experts = []
    all_topk_weights = []
    all_pre_mlp_hidden_states = []
    all_router_logits = []
    all_hidden_states = []
    all_expert_outputs = []

    for layer_ix, layer in enumerate(lang_model.model.layers):
        # layer_outputs = layer(hidden_state, attention_mask = attention_mask, position_ids = position_ids,)
        residual = hidden_state
        hidden_state = layer.input_layernorm(hidden_state)
        # Self Attention
        hidden_state, self_attn_weights, present_key_value = layer.self_attn(hidden_states = hidden_state, attention_mask = attention_mask, position_ids = position_ids)
        hidden_state = residual + hidden_state
        # Fully Connected
        residual = hidden_state
        hidden_state = layer.post_attention_layernorm(hidden_state)
        # Return hidden states only for MoE layers
        if 'DeepseekV3MLP' not in str(type(layer.mlp)) and return_hidden_states:
            all_pre_mlp_hidden_states.append(hidden_state.view(-1, hidden_state.shape[2]).detach().cpu())

        ## MLP
        if 'DeepseekV3MLP' in str(type(layer.mlp)):
            hidden_state = layer.mlp(hidden_state)
        else:
            identity = hidden_state
            orig_shape = hidden_state.shape
            ### Start MoeGate - originally topk_idx, topk_weight = layer.mlp.gate(hidden_state)
            bsz, seq_len, h = hidden_state.shape
            moe_hidden_state = hidden_state.view(-1, h)
            logits = torch.nn.functional.linear(moe_hidden_state.type(torch.float32), layer.mlp.gate.weight.type(torch.float32), None)
            scores = logits.sigmoid()
            
            scores_for_choice = scores.view(bsz * seq_len, -1) + layer.mlp.gate.e_score_correction_bias.unsqueeze(0)
            group_scores = (scores_for_choice.view(bsz * seq_len, layer.mlp.gate.n_group, -1).topk(2, dim=-1)[0].sum(dim = -1))  # [n, n_group]
            group_idx = torch.topk(group_scores, k = layer.mlp.gate.topk_group, dim=-1, sorted = False)[1]  # [n, top_k_group]
            group_mask = torch.zeros_like(group_scores)  # [n, n_group]
            group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
            score_mask = (group_mask.unsqueeze(-1).expand(bsz * seq_len, layer.mlp.gate.n_group, layer.mlp.gate.n_routed_experts // layer.mlp.gate.n_group).reshape(bsz * seq_len, -1))  # [n, e]
            tmp_scores = scores_for_choice.masked_fill(~score_mask.bool(), 0.0)  # [n, e]
            _, topk_idx = torch.topk(tmp_scores, k=layer.mlp.gate.top_k, dim=-1, sorted=False)
            topk_weight = scores.gather(1, topk_idx)
            if layer.mlp.gate.top_k > 1 and layer.mlp.gate.norm_topk_prob:
                denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
                topk_weight = topk_weight / denominator

            topk_weight = topk_weight * layer.mlp.gate.routed_scaling_factor
            ### End MoeGate 
            hidden_state = hidden_state.view(-1, hidden_state.shape[-1])
            ### Start moe_infer - replaces layer.mlp.moe_infer(hidden_state, topk_idx, topk_weight).view(*orig_shape)
            x = hidden_state
            topk_ids = topk_idx
            cnts = topk_ids.new_zeros((topk_ids.shape[0], len(layer.mlp.experts)))
            cnts.scatter_(1, topk_ids, 1)
            tokens_per_expert = cnts.sum(dim=0)
            idxs = topk_ids.view(-1).argsort()
            sorted_tokens = x[idxs // topk_ids.shape[1]]
            tokens_per_expert = tokens_per_expert.cpu().numpy()
            outputs = []
            start_idx = 0
            for i, num_tokens in enumerate(tokens_per_expert):
                end_idx = start_idx + num_tokens
                if num_tokens == 0:
                    continue
                expert = layer.mlp.experts[i + layer.mlp.ep_rank * layer.mlp.experts_per_rank]
                tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
                expert_out = expert(tokens_for_this_expert)
                outputs.append(expert_out)
                start_idx = end_idx
            outs = torch.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty(0)
            new_x = torch.empty_like(outs)
            new_x[idxs] = outs
            # --- CAPTURE RAW EXPERT OUTPUTS ---
            layer_expert_outputs = None
            if return_hidden_states:
                # Reshape new_x to (BN, topk, D)
                layer_expert_outputs = new_x.view(*topk_idx.shape, -1)
            # --- END CAPTURE ---
            final_out = (new_x.view(*topk_ids.shape, -1).type(topk_weight.dtype).mul_(topk_weight.unsqueeze(dim=-1)).sum(dim=1).type(new_x.dtype))
            ### End moe_infer
            y = final_out.view(*orig_shape)
            if layer.mlp.config.n_shared_experts is not None:
                y = y + layer.mlp.shared_experts(identity)
            hidden_state = y

        hidden_state = residual + hidden_state

        if 'DeepseekV3MLP' not in str(type(layer.mlp)):
            topk_ids, topk_weight, layer_expert_outputs = _sort_gate_tensors(
                topk_ids.detach(),
                topk_weight.detach(),
                layer_expert_outputs.detach() if return_hidden_states else None
            )
            all_topk_experts.append(topk_ids.cpu())
            all_topk_weights.append(topk_weight.cpu().to(torch.float32))
        
            if return_hidden_states:
                all_router_logits.append(logits.detach().cpu())
                all_hidden_states.append(hidden_state.view(-1, hidden_state.shape[2]).detach().cpu())
                all_expert_outputs.append(layer_expert_outputs.cpu())

    hidden_state = lang_model.model.norm(hidden_state)
    logits = lang_model.lm_head(hidden_state).float()

    return {
        'logits': logits,
        'all_topk_experts': all_topk_experts,
        'all_topk_weights': all_topk_weights,
        'all_pre_mlp_hidden_states': all_pre_mlp_hidden_states,
        'all_router_logits': all_router_logits,
        'all_hidden_states': all_hidden_states,
        'all_expert_outputs': all_expert_outputs
    }