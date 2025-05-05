import torch
import torch.nn.functional as F
from src.utils import calculate_costs, patch_dropout


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
        if gen_resp is None:
            gen_resp = "N/A"
        prompt = f"{prompt}\n\nAnswer: "
        full_text = f"{prompt}\n\nAnswer: {gen_resp}"
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
    for p, r in zip(prompts, generated_responses):
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

    verdict_logits = last_logits[
        :, verdict_ids
    ]  # shape: [batch_size, len(verdict_tokens)]

    # Convert to distribution
    verdict_log_probs = F.log_softmax(verdict_logits, dim=-1)  # shape: [batch_size, N]
    verdict_probs = verdict_log_probs.exp()  # shape: [batch_size, N]

    verdict_entropy = -(verdict_probs * verdict_log_probs).sum(dim=-1)

    return verdict_entropy




def surrogate_token_uncertainties(
    model,
    tokenizer,
    prompts,
    generated_responses,
    n=5,  # Number of repetitions for each prompt
    device="cuda",
    yes_token="YES",
    no_token="NO",
    max_length=512,
    normalize_by_length=None,
):
    """
    Computes the distribution of p(YES) for each question, repeated n times in a batch.

    Args:
        model: The LLM to evaluate.
        tokenizer: The tokenizer for the model.
        prompts: A list of questions (strings).
        generated_responses: A list of generated responses (strings).
        n: Number of times to repeat each question in the batch.
        device: Device to run the model on (e.g., "cuda").
        yes_token: The token representing "YES".
        no_token: The token representing "NO".
        max_length: Maximum sequence length for tokenization.
        normalize_by_length: Optional; if provided, normalize probabilities by length.

    Returns:
        torch.Tensor of shape [batch_size, n] containing p(YES) values for each repetition.
    """
    batch_size = len(prompts)

    # Build verification prompts for each item
    verify_prompts = []
    for question, candidate_answer in zip(prompts, generated_responses):
        text = (
            "Given the following question and the model's answer, please evaluate correctness.\n"
            f"Respond with a single token: {yes_token} or {no_token}\n\n"
            f"Question: {question}\n\n"
            f"Model Answer: {candidate_answer}\n\n"
            f"Is this answer correct: {yes_token} or {no_token}?\n\n"
            "Answer: "
        )
        verify_prompts.extend([text] * n)  # Repeat each question n times

    # Tokenize all in a batch
    inputs = tokenizer(
        verify_prompts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    ).to(device)

    # Find IDs for yes_token/no_token
    yes_ids = tokenizer.encode(yes_token, add_special_tokens=False)
    no_ids = tokenizer.encode(no_token, add_special_tokens=False)
    yes_id = yes_ids[0] if yes_ids else None
    no_id = no_ids[0] if no_ids else None

    patch_dropout(model, 0.1)

    # Process in a single forward pass
    with torch.no_grad():
        model.train()
        outputs = model(**inputs)
    logits = outputs.logits  # [batch_size * n, seq_len, vocab_size]

    patch_dropout(model, 0.0)
    model.eval()

    # For memory efficiency, compute probabilities one batch at a time
    seq_len = inputs["input_ids"].size(1)  # Length of each sequence
    next_token_logits = logits[:, seq_len - 1, :]  # [batch_size * n, vocab_size]
    next_token_probs = F.softmax(
        next_token_logits, dim=-1
    )  # [batch_size * n, vocab_size]

    # Extract probabilities for YES and NO tokens
    p_yes = (
        next_token_probs[:, yes_id]
        if yes_id is not None
        else torch.zeros(next_token_probs.size(0), device=device)
    )
    p_no = (
        next_token_probs[:, no_id]
        if no_id is not None
        else torch.zeros(next_token_probs.size(0), device=device)
    )

    # Normalize probabilities
    denom = p_yes + p_no
    p_yes_normalized = torch.where(
        denom > 0, p_yes / denom, torch.full_like(p_yes, 0.5)
    )  # Handle edge cases

    # Reshape to [batch_size, n]
    p_yes_distribution = p_yes_normalized.view(batch_size, n)
    p_yes_distribution = p_yes_distribution.mean(dim=1)
    entropy = -(p_yes_distribution * torch.log(p_yes_distribution + 1e-9))

    # Calculate verification costs based on token counts
    input_token_counts = [
        len(tokenizer.encode(verify_prompts[i])) * n
        for i in range(0, len(verify_prompts), n)
    ]
    output_token_counts = [0] * int(len(verify_prompts)/n)  # No generation, only inference

    # Extract model name from the model path or object
    model_name = getattr(
        model,
        "name_or_path",
        str(model).split("/")[-1] if "/" in str(model) else str(model),
    )

    # Calculate costs using the utility function
    # Calculate costs using the utility function
    uncertainty_costs = [
        calculate_costs(
            model_name=model_name,
            input_token_length=input_count,
            output_token_length=output_count,
            output_input_price_ratio=1.0,  # irrelevant, because of no generation
        )
        for input_count, output_count in zip(input_token_counts, output_token_counts)
    ]

    return entropy, uncertainty_costs


def coannotating_uncertainty_entropy(
    model,
    tokenizer,
    prompts,
    generated_responses,
    n=5,  # Number of repetitions for each prompt
    device="cuda",
    yes_token="YES",
    no_token="NO",
    max_length=512,
    normalize_by_length=None,
):
    """
    Computes the distribution of p(YES) for each question, repeated n times in a batch.

    Args:
        model: The LLM to evaluate.
        tokenizer: The tokenizer for the model.
        prompts: A list of questions (strings).
        generated_responses: A list of generated responses (strings).
        n: Number of times to repeat each question in the batch.
        device: Device to run the model on (e.g., "cuda").
        yes_token: The token representing "YES".
        no_token: The token representing "NO".
        max_length: Maximum sequence length for tokenization.
        normalize_by_length: Optional; if provided, normalize probabilities by length.

    Returns:
        torch.Tensor of shape [batch_size, n] containing p(YES) values for each repetition.
    """
    batch_size = len(prompts)

    # Build verification prompts for each item
    verify_prompts = []
    for question, candidate_answer in zip(prompts, generated_responses):
        text = (
            "Given the following question and the model's answer, please evaluate correctness.\n"
            f"Respond with a single token: {yes_token} or {no_token}\n\n"
            f"Question: {question}\n\n"
            f"Model Answer: {candidate_answer}\n\n"
            f"Please give a confidence score on a scale of 0.0 to 1.0 for this prediction.\n\n"
            "Answer: "
        )
        verify_prompts.extend([text] * n)  # Repeat each question n times

    # Tokenize all in a batch
    inputs = tokenizer(
        verify_prompts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    ).to(device)

    # Generate outputs
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=4,
            temperature=0.7,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Process generated outputs
    generated_texts = tokenizer.batch_decode(
        outputs.sequences[:, inputs.input_ids.shape[1] :], skip_special_tokens=True
    )

    # Parse the confidence scores
    confidence_scores = []
    for text in generated_texts:
        import re

        # Try to extract a float using regex
        match = re.search(r"(\d+\.\d+|\d+)", text.strip())
        if match:
            # Convert the matched string to float
            score = float(match.group(1))
            # Ensure the score is between 0 and 1
            score = max(0.0, min(1.0, score))
            confidence_scores.append(score)
        else:
            # If no float is found, default to NaN
            confidence_scores.append(torch.nan)

    # Convert to tensor and reshape
    confidence_tensor = torch.tensor(confidence_scores, device=device).view(
        batch_size, n
    )

    # Calculate mean confidence, ignoring NaN values
    mean_confidence = torch.nanmean(confidence_tensor, dim=1)

    # Calculate binary entropy
    entropy = -(confidence_tensor * torch.log(confidence_tensor + 1e-9)).sum(dim=1)
    return entropy
