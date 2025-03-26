"""
Reversed engineered forward pass for Qwen
- Supports Deepseek-v3/R1 and Moonlight
- See https://huggingface.co/deepseek-ai/DeepSeek-R1/blob/main/modeling_deepseek.py
"""
import torch
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

@torch.no_grad()
def run_dsv3_return_topk(model, input_ids, attention_mask):
    """
    Params:
        @model: A model of class `DeepseekV3ForCausalLM`.
        @input_ids: A B x N tensor of inputs IDs on the same device as `model`.
        @attention_mask: A B x N tensor of mask indicators on the same device as `model`.

    Returns:
        A dictionary with keys:
        - `logits`: The standard B x N x V LM output
        - `all_topk_experts`: A list of length equal to the number of MoE layers, with each element a BN x topk tensor of expert IDs
        - `all_topk_weights`: A list of length equal to the number of MoE layers, with each element a BN x topk tensor of expert weights
    """
    B, N = input_ids.shape[:2]
    position_ids = torch.arange(0, N, dtype=torch.long, device = model.device).unsqueeze(0)
    inputs_embeds = model.model.embed_tokens(input_ids)
    
    if model.model._use_flash_attention_2:
        attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
    else:
        attention_mask = _prepare_4d_causal_attention_mask(attention_mask, (B, N), inputs_embeds, 0,)

    hidden_state = inputs_embeds
    all_topk_experts = []
    all_topk_weights = []
    for layer_ix, layer in enumerate(model.model.layers):
        # layer_outputs = layer(hidden_state, attention_mask = attention_mask, position_ids = position_ids,)
        residual = hidden_state
        hidden_state = layer.input_layernorm(hidden_state)
        # Self Attention
        hidden_state, self_attn_weights, present_key_value = layer.self_attn(hidden_states = hidden_state, attention_mask = attention_mask, position_ids = position_ids)
        hidden_state = residual + hidden_state
        # Fully Connected
        residual = hidden_state
        hidden_state = layer.post_attention_layernorm(hidden_state)
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
            _, topk_idx = torch.topk(tmp_scores, k=layer.mlp.gate.top_k, dim=-1, sorted=True)
            topk_weight = scores.gather(1, topk_idx)
            if layer.mlp.gate.top_k > 1 and layer.mlp.gate.norm_topk_prob:
                denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
                topk_weight = topk_weight / denominator
            else:
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
            final_out = (new_x.view(*topk_ids.shape, -1).type(topk_weight.dtype).mul_(topk_weight.unsqueeze(dim=-1)).sum(dim=1).type(new_x.dtype))
            ### End moe_infer
            y = final_out.view(*orig_shape)
            if layer.mlp.config.n_shared_experts is not None:
                y = y + layer.mlp.shared_experts(identity)
            hidden_state = y

            all_topk_experts.append(topk_ids)
            all_topk_weights.append(topk_weight)

        hidden_state = residual + hidden_state

    hidden_state = model.model.norm(hidden_state)
    logits = model.lm_head(hidden_state).float()
    return {'logits': logits, 'all_topk_experts': all_topk_experts, 'all_topk_weights': all_topk_weights}

@torch.no_grad()
def run_dsv3_return_topk(model, input_ids, attention_mask, layers_to_ablate = [], topk_to_ablate = [], renorm = False):
    """
    Params:
        @model: A model of class `DeepseekV3ForCausalLM`.
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
    B, N = input_ids.shape[:2]
    position_ids = torch.arange(0, N, dtype=torch.long, device = model.device).unsqueeze(0)
    inputs_embeds = model.model.embed_tokens(input_ids)
    
    if model.model._use_flash_attention_2:
        attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
    else:
        attention_mask = _prepare_4d_causal_attention_mask(attention_mask, (B, N), inputs_embeds, 0,)

    hidden_state = inputs_embeds
    all_topk_experts = []
    all_topk_weights = []
    for layer_ix, layer in enumerate(model.model.layers):
        # layer_outputs = layer(hidden_state, attention_mask = attention_mask, position_ids = position_ids,)
        residual = hidden_state
        hidden_state = layer.input_layernorm(hidden_state)
        # Self Attention
        hidden_state, self_attn_weights, present_key_value = layer.self_attn(hidden_states = hidden_state, attention_mask = attention_mask, position_ids = position_ids)
        hidden_state = residual + hidden_state
        # Fully Connected
        residual = hidden_state
        hidden_state = layer.post_attention_layernorm(hidden_state)
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
            _, topk_idx = torch.topk(tmp_scores, k=layer.mlp.gate.top_k, dim=-1, sorted=True)
            topk_weight = scores.gather(1, topk_idx)
            if layer.mlp.gate.top_k > 1 and layer.mlp.gate.norm_topk_prob:
                denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
                topk_weight = topk_weight / denominator
            else:
                topk_weight = topk_weight * layer.mlp.gate.routed_scaling_factor
            ### End MoeGate 
            ######################## ABLATION
            # shape: topk_weight is [B*N, top_k]
            if layer_ix in layers_to_ablate:
                # (A) Sort the topk dimension locally to find which columns correspond to the rank-ordered experts (note shape of topk_weight: [BN, k])
                sorted_w, sorted_idx = topk_weight.sort(dim=-1, descending=True)
                # sorted_w[:,0] is the largest weight in each row, sorted_idx[:,0] gives the original column index for that largest weight.
                row_sum_before = topk_weight.sum(dim=-1, keepdim=True)

                # (B) For each rank in topk_to_ablate, zero out that column in topk_weight
                for rank in topk_to_ablate:
                    columns_to_ablate = sorted_idx[:, rank]  # columns_to_ablate is [BN], each entry is the "original column" that corresponds to rank `rank` in sorted order
                    # Now zero out topk_weight[row, col]
                    for row_i in range(topk_weight.shape[0]):
                        col_j = columns_to_ablate[row_i].item()
                        topk_weight[row_i, col_j] = 0.0

                # Re-scale the remaining top-k weights to keep sum the same
                if renorm:
                    row_sum_after = topk_weight.sum(dim=-1, keepdim=True)
                    scale_factor = row_sum_before / (row_sum_after + 1e-9)
                    topk_weight *= scale_factor
            ######################## The rest is unchanged
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
            final_out = (new_x.view(*topk_ids.shape, -1).type(topk_weight.dtype).mul_(topk_weight.unsqueeze(dim=-1)).sum(dim=1).type(new_x.dtype))
            ### End moe_infer
            y = final_out.view(*orig_shape)
            if layer.mlp.config.n_shared_experts is not None:
                y = y + layer.mlp.shared_experts(identity)
            hidden_state = y

            all_topk_experts.append(topk_ids)
            all_topk_weights.append(topk_weight)

        hidden_state = residual + hidden_state

    hidden_state = model.model.norm(hidden_state)
    logits = model.lm_head(hidden_state).float()
    return {'logits': logits, 'all_topk_experts': all_topk_experts, 'all_topk_weights': all_topk_weights}