"""
This script contains the efficient verification strategies:
- surrogate token probabilities (batched)
- sequence probabilities (batched)
- verbalisation (batched)

Source: https://aclanthology.org/2024.acl-long.250.pdf
author: Claudio Fanconi + GPT
"""

import torch
import torch.nn.functional as F


def verbalisation(
    model,
    tokenizer,
    prompts,
    generated_responses,
    questions,
    device="cuda",
    yes_token="YES",
    no_token="NO",
    max_length=512,
    max_new_tokens=2,
    normalize_by_length=None,
    prompt_template="",
):
    """
    Batched version of the verbalisation approach.
    For each prompt+generated_response pair, we construct a short "verification prompt"
    that instructs the model to respond with a single token: yes_token or no_token.

    Then, we generate up to `max_new_tokens` tokens (often 1-2 tokens is enough).
    We parse the resulting text to see if it starts with yes_token or no_token.

    Returns: A list of floats [batch_size],
       where each entry is:
         1.0 if we detect `yes_token`,
         0.0 if we detect `no_token`,
         None if neither is found (you can handle that as you wish).
    """
    batch_size = len(prompts)
    # Build all verification prompts
    verify_prompts = []
    for question, candidate_answer in zip(prompts, generated_responses):
        text = (
            "Given the following medical question and the model's answer, please evaluate correctness.\n"
            f"Respond with a single token: {yes_token} or {no_token}\n"
            f"Question:\n{question}\n\n"
            f"model answer:\n{candidate_answer}\n\n"
            f"Is this answer correct: {yes_token} or {no_token}?\n"
            "Answer: "
        )
        verify_prompts.append(text)

    # Tokenize all prompts in one batch
    inputs = tokenizer(
        verify_prompts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    ).to(model.device)

    # Generate for each prompt
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode each model output
    decoded_outputs = [
        tokenizer.decode(output_seq, skip_special_tokens=True) for output_seq in outputs
    ]

    # Now parse each decoded text to see if it starts with "YES" or "NO"
    results = []
    for i, raw_text in enumerate(decoded_outputs):
        # The original prompt is verify_prompts[i]
        # We only want to see what's generated *after* the prompt
        prompt_len = len(verify_prompts[i])
        verification_text = raw_text[prompt_len:].strip().upper()

        if verification_text.startswith(yes_token.upper()):
            results.append(1.0)
        elif verification_text.startswith(no_token.upper()):
            results.append(0.0)
        else:
            results.append(0.0)

    return torch.Tensor(results)


def surrogate_token_probs(
    model,
    tokenizer,
    prompts,
    generated_responses,
    questions,
    device="cuda",
    yes_token="YES",
    no_token="NO",
    max_length=512,
    normalize_by_length=None,
    prompt_template="",
):
    """
    Batched version of surrogate token probability approach.

    For each (prompt, candidate_answer), we build a "verification prompt" that ends
    right before the next token we want the model to predict (YES or NO).
    Then we do a single forward pass to get the next-token logits, from which we read off
    p(YES) and p(NO). Finally, we compute pYES / (pYES + pNO).

    Returns: A list of floats [batch_size], each in [0,1], giving the probability
             that the model chooses yes_token over no_token.
             (If we can't find the token IDs, or p(no_token)=0 => handle edge cases.)
    """
    batch_size = len(prompts)

    # Build verification prompts for each item
    verify_prompts = []
    for question, candidate_answer in zip(questions, generated_responses):
        text = (
            "Given the following medical question and the model's answer, please evaluate correctness.\n"
            f"Respond with a single token: {yes_token} or {no_token}\n"
            f"Question:\n{question}\n\n"
            f"model answer:\n{candidate_answer}\n\n"
            f"Is this answer correct: {yes_token} or {no_token}?\n"
            "Answer: "
        )
        verify_prompts.append(text)

    # Tokenize all in a batch
    inputs = tokenizer(
        verify_prompts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    ).to(model.device)

    # Find IDs for yes_token/no_token
    yes_ids = tokenizer.encode(yes_token, add_special_tokens=False)
    no_ids = tokenizer.encode(no_token, add_special_tokens=False)
    yes_id = yes_ids[0] if yes_ids else None
    no_id = no_ids[0] if no_ids else None

    # Single forward pass
    with torch.no_grad():
        outputs = model(
            **inputs
        )  # outputs.logits shape: [batch_size, seq_len, vocab_size]

    logits = outputs.logits  # [batch, seq_len, vocab_size]

    # For each example i, the next token distribution is logits[i, seq_len-1]
    # We'll apply softmax to that distribution to get probabilities
    probs_list = []
    for i in range(batch_size):
        seq_len_i = inputs["input_ids"].size(
            1
        )  # because padding => same length for all
        # The distribution for the next token:
        next_token_logits = logits[i, seq_len_i - 1, :]  # shape [vocab_size]
        dist = F.softmax(next_token_logits, dim=-1)

        p_yes = dist[yes_id].item() if yes_id is not None else 0.0
        p_no = dist[no_id].item() if no_id is not None else 0.0

        denom = p_yes + p_no
        if denom > 0:
            p_yes_normalized = p_yes / denom
        else:
            # fallback if denom=0
            p_yes_normalized = 0.5  # or 1.0, or 0.0, or however you want to handle

        probs_list.append(p_yes_normalized)

    return torch.Tensor(probs_list)


def sequence_probability(
    model,
    tokenizer,
    prompts,
    generated_responses,
    questions,
    device="cuda",
    normalize_by_length=True,
    max_length=512,
    prompt_template="",
):
    """
    Computes the log probability (or average log probability) *only* over the generated
    portion (i.e., ignoring the prompt tokens), for a batch of (prompts, generated_responses).
    Returns a 1D tensor [batch_size] with the average log prob of the *generated* portion.
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

    input_ids = inputs["input_ids"]  # [batch, seq_len]
    attention_mask = inputs["attention_mask"]  # [batch, seq_len]

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits  # [batch, seq_len, vocab_size]

    # shift for causal LM
    shift_logits = logits[:, :-1, :].contiguous().to(device)
    shift_labels = input_ids[:, 1:].contiguous().to(device)
    shift_mask = attention_mask[:, 1:].contiguous().to(device)

    token_loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="none",
    )
    token_loss = token_loss.view(shift_labels.size())  # [batch, seq_len-1]

    batch_log_probs = torch.zeros(shift_labels.size(0), device=device)

    for i in range(shift_labels.size(0)):
        start_idx = prompt_lengths[i]
        end_idx = prompt_lengths[i] + gen_lengths[i]

        gen_start_in_shift = max(start_idx - 1, 0)
        gen_end_in_shift = max(end_idx - 1, 0)

        seq_len_i = shift_labels.size(1)
        gen_start_in_shift = min(gen_start_in_shift, seq_len_i)
        gen_end_in_shift = min(gen_end_in_shift, seq_len_i)

        gen_loss_i = (token_loss[i, :] * shift_mask[i, :])[
            gen_start_in_shift:gen_end_in_shift
        ].sum()
        seq_log_prob_i = -gen_loss_i

        if normalize_by_length:
            n_gen_tokens = (
                shift_mask[i, gen_start_in_shift:gen_end_in_shift] == 1
            ).sum()
            if n_gen_tokens > 0:
                seq_log_prob_i /= n_gen_tokens

        batch_log_probs[i] = seq_log_prob_i

    return torch.exp(batch_log_probs)
