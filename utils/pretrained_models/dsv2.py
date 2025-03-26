"""
Reversed engineered forward pass for Qwen
- Supports Deepseek-v2 and Deepseek-v2-Lite
- See https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite/blob/main/modeling_deepseek.py
"""
import torch
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

@torch.no_grad()
def run_dsv2_return_topk(model, input_ids, attention_mask):
    """
    Params:
        @model: A model of class `DeepseekV2ForCausalLM`.
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
        if 'DeepseekV2MLP' in str(type(layer.mlp)):
            hidden_state = layer.mlp(hidden_state)
        else:
            identity = hidden_state
            orig_shape = hidden_state.shape
            ### Start MoeGate - originally topk_idx, topk_weight, aux_loss = layer.mlp.gate(hidden_state)
            bsz, seq_len, h = hidden_state.shape
            moe_hidden_state = hidden_state.view(-1, h)
            logits = torch.nn.functional.linear(moe_hidden_state.type(torch.float32), layer.mlp.gate.weight.type(torch.float32), None)
            scores = logits.softmax(dim=-1, dtype=torch.float32)
            topk_weight, topk_idx = torch.topk(scores, k=layer.mlp.gate.top_k, dim=-1, sorted=False)
            if layer.mlp.gate.top_k > 1 and layer.mlp.gate.norm_topk_prob:
                denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
                topk_weight = topk_weight / denominator
            else:
                topk_weight = topk_weight * layer.mlp.gate.routed_scaling_factor
            #### End MoeGate
            hidden_state = hidden_state.view(-1, hidden_state.shape[-1])
            ### Start moe_infer
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
    logits = model.lm_head(hidden_state)
    return {'logits': logits, 'all_topk_experts': all_topk_experts, 'all_topk_weights': all_topk_weights}


@torch.no_grad()
def run_dsv2_return_topk(model, input_ids, attention_mask, layers_to_ablate = [], topk_to_ablate = [], renorm = False):
    """
    Params:
        @model: A model of class `DeepseekV2ForCausalLM`.
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
        if 'DeepseekV2MLP' in str(type(layer.mlp)):
            hidden_state = layer.mlp(hidden_state)
        else:
            identity = hidden_state
            orig_shape = hidden_state.shape
            ### Start MoeGate - originally topk_idx, topk_weight, aux_loss = layer.mlp.gate(hidden_state)
            bsz, seq_len, h = hidden_state.shape
            moe_hidden_state = hidden_state.view(-1, h)
            logits = torch.nn.functional.linear(moe_hidden_state.type(torch.float32), layer.mlp.gate.weight.type(torch.float32), None)
            scores = logits.softmax(dim=-1, dtype=torch.float32)
            topk_weight, topk_idx = torch.topk(scores, k=layer.mlp.gate.top_k, dim=-1, sorted=False)
            if layer.mlp.gate.top_k > 1 and layer.mlp.gate.norm_topk_prob:
                denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
                topk_weight = topk_weight / denominator
            else:
                topk_weight = topk_weight * layer.mlp.gate.routed_scaling_factor
            #### End MoeGate
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
            ### Start moe_infer
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
    logits = model.lm_head(hidden_state)
    return {'logits': logits, 'all_topk_experts': all_topk_experts, 'all_topk_weights': all_topk_weights}
