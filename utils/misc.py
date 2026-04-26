import torch

def check_max_seq_len(tokenizer, strings: list[str]) -> int:
    """
    Check max token length across a list of strings
    """
    max_seqlen = int(tokenizer(strings, padding = True, truncation = False, return_tensors = 'pt')['attention_mask'].sum(dim = 1).max().item())
    return max_seqlen