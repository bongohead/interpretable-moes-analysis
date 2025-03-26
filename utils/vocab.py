import csv

def export_vocab_as_csv(tokenizer, output_file: str):
    """
    Export a tokenizer vocabulary file as a CSV.

    Params:
        @tokenizer: A standard HF tokenizer object
        @output_file: The path to output into

    Example:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('allenai/OLMoE-1B-7B-0924', add_eos_token = False, add_bos_token = False)
        export_vocab_as_csv(tokenizer, 'olmoe.csv')
    """
    # Get the vocabulary as a dictionary of token -> token_id
    vocab = tokenizer.get_vocab()
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])

    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['token_id', 'raw_token', 'token'])

        for raw_token, token_id in sorted_vocab:
            cleaned = raw_token
            cleaned = cleaned.replace("Ġ", " ")
            cleaned = cleaned.replace("▁", " ")
            cleaned = cleaned.replace("Ċ", "\n")

            # Now handle control characters or unprintable stuff - we'll say "unprintable" if ANY char is not .isprintable().
            if not all(ch.isprintable() or ch.isspace() for ch in cleaned):
                # Convert any unprintable to hex escapes:
                cleaned_hex = []
                for ch in cleaned:
                    if ch.isprintable() or ch.isspace():
                        cleaned_hex.append(ch)
                    else:
                        cleaned_hex.append(f'\\x{ord(ch):02x}')
                cleaned = ''.join(cleaned_hex)

            writer.writerow([token_id, raw_token, cleaned])

        print(f"CSV file has been created: {output_file}")