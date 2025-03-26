import pandas as pd
import torch

@torch.no_grad()
def convert_outputs_to_df(input_ids: torch.Tensor, attention_mask: torch.Tensor, output_logits: torch.Tensor) -> pd.DataFrame:
    """
    Converts batch output logits into a pandas dataframe for later analysis. Skips positions where attention_mask == 0 (i.e., padding).

    Params:
        @input_ids: A tensor of input IDs of size B x N
        @attention_mask: A tensor of 1 for real tokens, 0 for padding, of size B x N
        @output_logits: A B x N x V tensor of output logits

    Returns:
        A dataframe at `sequence_ix` x `token_ix` level, excluding masked tokens, with columns:
        - `sequence_ix`: Which sample in the batch.
        - `token_ix`: Token index within that sample.
        - `token_id`: The input token ID at that `sequence_ix` x `token_ix`.
        - `output_id`: Argmax of the output logits (the predicted token) associated with that `sequence_ix` x `token_ix`.
        - `output_prob`: The probability (softmax) of the output.

    Example:
        prompt = 'Hello'
        inputs = tokenizer(prompt, return_tensors = 'pt').to(main_device)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        with torch.no_grad():
            output = model(input_ids, attention_mask)

        convert_outputs_to_df(input_ids, attention_mask, output['logits'])
    """
    output_logits = output_logits.detach().cpu().to(torch.float32)
    B, N, _ = output_logits.shape

    # Get the argmax (top-1 prediction) at each [B, N] position
    top_ids = output_logits.argmax(dim = -1)  # [B, N]

    # Convert logits to probabilities, then find max along vocab dimension
    probs = torch.nn.functional.softmax(output_logits, dim = -1) # [B, N, V]
    top_probs = probs.max(dim = -1).values # [B, N]

    # Flatten everything so we can mask out padding in a single step
    # We'll keep track of (sequence_ix, token_ix) as well.
    flat_seq_ix = torch.arange(B).unsqueeze(1).expand(B, N).reshape(-1) # [B*N]
    flat_token_ix = torch.arange(N).unsqueeze(0).expand(B, N).reshape(-1) # [B*N]

    flat_input_ids = input_ids.reshape(-1).cpu() # [B*N]
    flat_attention = attention_mask.reshape(-1).cpu() # [B*N]
    flat_top_ids = top_ids.reshape(-1) # [B*N]
    flat_top_probs = top_probs.reshape(-1) # [B*N]

    # Build a mask for positions with attention=1
    valid_positions = (flat_attention == 1)

    # Construct a DataFrame only for the valid positions
    data = {
        "sequence_ix": flat_seq_ix[valid_positions].numpy(),
        "token_ix": flat_token_ix[valid_positions].numpy(),
        "token_id": flat_input_ids[valid_positions].numpy(),
        "output_id": flat_top_ids[valid_positions].numpy(),
        "output_prob": flat_top_probs[valid_positions].numpy(),
    }
    df = pd.DataFrame(data)
    return df