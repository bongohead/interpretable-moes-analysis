import pandas as pd
import torch

@torch.no_grad()
def convert_topk_to_df(all_topk_experts, input_ids):
    """
    Converts all_topk_experts into a pandas dataframe for later analysis.

    Params:
        @all_topk_experts: A tuple of n_layers length, with each element a tensor size (BN, topk) containing the expert IDs selected
        @input_ids: A 

    Returns:
        A dataframe at `sequence_ix` x `token_ix` x `layer_ix`, with columns:
        - `sequence_ix`: The index of the batch sub-samples.
        - `token_ix`: The token index of a single sequence.
        - `layer_ix`: The layer index.
        - `token_id`: The token ID at that `sequence_ix` x `token_ix`.
        - `expert_1`, `expert_2`, ..., `expert_[topk]`:  The routing of that token at `sequence_ix` x `token_ix` x `layer_ix`.

    Example:
        prompt = 'Hello'
        inputs = tokenizer(prompt, return_tensors = 'pt').to(main_device)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        with torch.no_grad():
            output = model(input_ids, attention_mask)

        topk_to_df(output['all_topk_experts'], input_ids)
    """
    data = []
    
    B, N = input_ids.shape
    top_k = all_topk_experts[0].shape[1]
    
    # Flatten input_ids to match all_topk_experts shape
    flat_input_ids = input_ids.reshape(-1).cpu().numpy()
    
    for layer_ix, layer_experts in enumerate(all_topk_experts):
        layer_experts_np = layer_experts.cpu().numpy()
        
        # For each token position
        for token_pos in range(B * N):
            # Get batch and token indices
            sequence_ix = token_pos // N
            token_ix = token_pos % N
            
            # Get experts for this token
            experts = layer_experts_np[token_pos]
            
            # Create a row with all info
            row = {"sequence_ix": sequence_ix, "token_ix": token_ix, "token_id": flat_input_ids[token_pos], "layer_ix": layer_ix}
            
            # Add each expert
            for k in range(top_k):
                row[f"expert_{k+1}"] = experts[k]
                
            data.append(row)
    
    df = pd.DataFrame(data)
    return df
