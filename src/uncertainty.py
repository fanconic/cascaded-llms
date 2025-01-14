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

    for prompt, gen_resp in zip(prompts, generated_responses):
        full_text = prompt + gen_resp
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


def verdict_distribution_entropy(
    model,
    tokenizer,
    prompts,
    generated_responses,
    verdict_tokens=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
    device="cuda",
    max_length=512,
    normalize_by_length=True,
):
    """
    Computes an 'uncertainty entropy' from a single forward pass over a
    short 'verification prompt' that asks the model to choose one verdict
    from the specified 'verdict_tokens'.
    
    :param model: A (decoder) language model, e.g. GPT-style or an encoder-decoder
                  that returns `outputs.logits`.
    :param tokenizer: HuggingFace tokenizer for the given model.
    :param prompts: List[str]. The original user prompts/questions.
    :param generated_responses: List[str]. The model's answers to be verified.
    :param verdict_tokens: A list of short strings representing discrete verdicts.
                          IMPORTANT: Each string should ideally tokenize to a *single* ID.
    :param device: "cuda" or "cpu".
    :param max_length: For safety if sequences get long; used in tokenizer calls.
    
    :return: A torch.FloatTensor of shape [batch_size], containing
             the entropy of the distribution over verdict tokens.
    """
    assert len(prompts) == len(generated_responses), "Mismatch in batch sizes!"

    # Convert each verdict token to a single token ID (assuming each maps to exactly 1 ID).
    # If a verdict word splits into multiple tokens, you’ll need a more careful approach.
    verdict_ids = []
    for vt in verdict_tokens:
        ids = tokenizer.encode(vt, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(
                f"Verdict token '{vt}' did not map to exactly 1 ID. Got: {ids}"
            )
        verdict_ids.append(ids[0])

    batch_verification_prompts = []
    for (p, r) in zip(prompts, generated_responses):
        verification_text = (
            "Given the following question and the model's answer, please evaluate certainty.\n\n"
            f"Respond with a single token: {verdict_tokens} where 0 is the most uncertain, and 9 is the most certain\n\n"
            f"Question:\n{p}\n\n"
            f"model answer:\n{r}\n\n"
            f"How certain are you that above answer is correct? Respond with exactly one token from "
            f"{verdict_tokens}."
            "Answer: "
        )
        batch_verification_prompts.append(verification_text)

    # Tokenize in batch
    inputs = tokenizer(
        batch_verification_prompts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(device)

    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)
        last_logits = outputs.logits[:, -1, :]  # shape: [batch_size, vocab_size]

    verdict_logits = last_logits[:, verdict_ids]  # shape: [batch_size, len(verdict_tokens)]

    # Convert to distribution
    verdict_log_probs = F.log_softmax(verdict_logits, dim=-1)  # shape: [batch_size, N]
    verdict_probs = verdict_log_probs.exp()                    # shape: [batch_size, N]

    verdict_entropy = -(verdict_probs * verdict_log_probs).sum(dim=-1)

    return verdict_entropy