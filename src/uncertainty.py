""" 
This script contains the efficient uncertainty estimation strategies

- per token entropy
author: Claudio Fanconi
"""

import torch
import torch.nn.functional as F


def per_token_entropy(
    model,
    tokenizer,
    prompts,
    generated_responses,
    device="cuda",
    normalize_by_length=True,
    max_length=512,
):
    """
    Computes the *average token-level entropy* over the generated portion only.
    Returns a 1D tensor [batch_size] with the average entropy for each example.
    """
    assert len(prompts) == len(generated_responses), "Mismatch in batch sizes!"

    batch_full_texts = []
    prompt_lengths = []
    gen_lengths = []

    for prompt, full_text in zip(prompts, generated_responses):
        gen_resp = full_text[len(prompt) :]
        prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
        gen_ids = tokenizer(gen_resp, add_special_tokens=False).input_ids

        prompt_len = len(prompt_ids)
        gen_len = len(gen_ids)

        batch_full_texts.append(full_text)
        prompt_lengths.append(prompt_len)
        gen_lengths.append(gen_len)

    inputs = tokenizer(
        batch_full_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    ).to(model.device)

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits.to(device)

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous().to(device)
    shift_mask = attention_mask[:, 1:].contiguous().to(device)

    # We'll compute the per-token entropy:
    # entropy = - sum_{v} p(v) log p(v), with p(v) = softmax(logits)
    # We'll average over tokens in the generated region.

    # 1. log_probs: [batch, seq_len-1, vocab_size]
    log_probs = F.log_softmax(shift_logits, dim=-1)
    probs = log_probs.exp()

    # 2. token_entropy: [batch, seq_len-1]
    #    for each token, - sum_{v} p_{v} log p_{v}
    token_entropy = -(probs * log_probs).sum(dim=-1)

    batch_entropies = torch.zeros(shift_labels.size(0), device=device)

    for i in range(shift_labels.size(0)):
        start_idx = prompt_lengths[i]
        end_idx = prompt_lengths[i] + gen_lengths[i]

        gen_start_in_shift = max(start_idx - 1, 0)
        gen_end_in_shift = max(end_idx - 1, 0)

        seq_len_i = shift_labels.size(1)
        gen_start_in_shift = min(gen_start_in_shift, seq_len_i)
        gen_end_in_shift = min(gen_end_in_shift, seq_len_i)

        # sum of entropies in that region
        region_entropy = (token_entropy[i, :] * shift_mask[i, :])[
            gen_start_in_shift:gen_end_in_shift
        ].sum()

        if normalize_by_length:
            n_gen_tokens = (
                shift_mask[i, gen_start_in_shift:gen_end_in_shift] == 1
            ).sum()
            if n_gen_tokens > 0:
                region_entropy = region_entropy / n_gen_tokens

        batch_entropies[i] = region_entropy

    return batch_entropies
