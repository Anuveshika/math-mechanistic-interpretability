def align_steps_to_tokens(text, steps, tokenizer):
    """
    Returns a list of (token_indices, step_text)
    """
    encoding = tokenizer(text, return_offsets_mapping=True)
    offsets = encoding["offset_mapping"]

    aligned = []
    for step in steps:
        start = text.find(step)
        end = start + len(step)

        token_ids = [
            i for i, (s, e) in enumerate(offsets)
            if s >= start and e <= end
        ]
        aligned.append((token_ids, step))

    return aligned
