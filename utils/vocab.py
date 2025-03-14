import csv

def export_vocab_as_csv(tokenizer, output_file: str):
    """
    Export a tokenizer vocabulary file as a CSV.

    Params:
        @tokenizer: A standard HF tokenizer object

    Example:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('allenai/OLMoE-1B-7B-0924', add_eos_token = False, add_bos_token = False)
        export_vocab_as_csv('olmoe.csv')
    """
    # Get the vocabulary as a dictionary of token -> token_id
    vocab = tokenizer.get_vocab()

    # Sort by token_id for a consistent output
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])

    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
    # Write header
    writer.writerow(['token_id', 'token', 'display_form'])

    # Write each token and its ID
    for token, token_id in sorted_vocab:
        # Create a more readable representation
        if token.startswith('Ġ'):  # This is a common prefix in GPT-2 tokenizer for space
            display_form = f" {token[1:]}"
        elif token.startswith('Ċ'):  # Common for newline
            display_form = "[NEWLINE]"
        elif token.startswith('▁'):  # Space in some other tokenizers
            display_form = f" {token[1:]}"
        elif len(token) == 1 and ord(token) < 32:  # Control characters
            display_form = f"[CTRL-{ord(token)}]"
        elif token.isprintable():
            display_form = token
        else:
            # Show as hex for unprintable characters
            display_form = ''.join(f'\\x{ord(c):02x}' for c in token)

    writer.writerow([token_id, token, display_form])

    print(f"CSV file has been created: {output_file}")