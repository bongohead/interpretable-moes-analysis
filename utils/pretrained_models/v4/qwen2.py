"""
Reversed engineered forward pass for Qwen
- Supports Qwen2.5
- See https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2/modeling_qwen2.py
"""
import torch

@torch.no_grad()
def run_qwen2(model, input_ids, attention_mask, return_hidden_states = False):
    """
    Params:
        @model: A model of class `Qwen2MoeForCausalLM`.
        @input_ids: A B x N tensor of inputs IDs on the same device as `model`.
        @attention_mask: A B x N tensor of mask indicators on the same device as `model`.
        @return_hidden_states: Boolean; whether to return hidden_states themselves.

    Returns:
        A dictionary with keys:
        - `logits`: The standard B x N x V LM output
        - `all_pre_mlp_hidden_states`: If return_hidden_states, a list of length equal to the number of MoE layers, with each element a BN x D tensor of pre-MLP hidden states
        - `all_hidden_states`: If return_hidden_states, a list of length equal to the number of MoE layers, with each element a BN x D tensor of post-layer hidden states
    """
    input_embeds = model.model.embed_tokens(input_ids)
    
    cache_position = torch.arange(0, input_embeds.shape[1], device = input_embeds.device)
    position_ids = cache_position.unsqueeze(0)
    causal_mask = model.model._update_causal_mask(attention_mask, input_embeds, cache_position, None, None)

    hidden_state = input_embeds
    position_embeddings = model.model.rotary_emb(hidden_state, position_ids)

    all_pre_mlp_hidden_states = []
    all_hidden_states = []
        
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
        
        hidden_state = layer.mlp(hidden_state)
        hidden_state = residual + hidden_state

        if return_hidden_states:
            all_hidden_states.append(hidden_state.view(-1, hidden_state.shape[2]).detach().cpu())

    hidden_state = model.model.norm(hidden_state)
    logits = model.lm_head(hidden_state)
    return {'logits': logits, 'all_pre_mlp_hidden_states': all_pre_mlp_hidden_states, 'all_hidden_states': all_hidden_states}