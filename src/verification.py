import torch
import torch.nn.functional as F
from src.utils import calculate_costs, patch_dropout


def surrogate_token_probs(
    model,
    tokenizer,
    prompts,
    generated_responses,
    questions,
    yes_token="YES",
    no_token="NO",
    max_length=512,
    mc_dropout=False,
    n=1,
    dropout_proba=0.1,
    output_input_price_ratio=4.0
):
    """
    Batched version of surrogate token probability approach.

    For each (prompt, candidate_answer), we build a "verification prompt" that ends
    right before the next token we want the model to predict (YES or NO).
    Then we do a single forward pass to get the next-token logits, from which we read off
    p(YES) and p(NO). Finally, we compute pYES / (pYES + pNO).

    If mc_dropout=True, applies Monte Carlo dropout by running the model n times with
    dropout enabled, then averaging the results.

    Args:
        model: The LLM to evaluate.
        tokenizer: The tokenizer for the model.
        prompts: A list of questions (strings).
        generated_responses: A list of generated responses (strings).
        questions: The original questions.
        yes_token: The token representing "YES".
        no_token: The token representing "NO".
        max_length: Maximum sequence length for tokenization.
        mc_dropout: Whether to use Monte Carlo dropout.
        n: Number of repetitions for MC dropout (ignored if mc_dropout=False).
        dropout_proba: dropout probability

    Returns:
        Tuple of (probabilities, costs) where:
        - probabilities: torch.Tensor of shape [batch_size] containing p(YES) values
        - costs: List of calculated costs for the verification
    """
    batch_size = len(prompts)

    # Build verification prompts for each item
    verify_prompts = []
    for question, candidate_answer in zip(questions, generated_responses):
        text = (
            "Given the following question and the model's answer, please evaluate correctness.\n"
            f"Respond with a single token: {yes_token} or {no_token}\n\n"
            f"Question: {question}\n\n"
            f"Model Answer: {candidate_answer}\n\n"
            f"Is this answer correct: {yes_token} or {no_token}?\n\n"
            "Answer: "
        )
        if mc_dropout:
            verify_prompts.extend([text] * n)  # Repeat each question n times
        else:
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

    # Apply MC dropout if requested
    if mc_dropout:
        patch_dropout(model, dropout_proba)
        model.train()
    
    # Single forward pass
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits  # [batch, seq_len, vocab_size]
    
    # Reset dropout if we used it
    if mc_dropout:
        patch_dropout(model, 0.0)
        model.eval()

    # Get sequence length and compute next token probabilities
    seq_len = inputs["input_ids"].size(1)
    next_token_logits = logits[:, seq_len - 1, :]  # [batch_size or batch_size*n, vocab_size]
    next_token_probs = F.softmax(next_token_logits, dim=-1)

    # Extract probabilities for YES and NO tokens
    p_yes = (
        next_token_probs[:, yes_id]
        if yes_id is not None
        else torch.zeros(next_token_probs.size(0), device=model.device)
    )
    p_no = (
        next_token_probs[:, no_id]
        if no_id is not None
        else torch.zeros(next_token_probs.size(0), device=model.device)
    )

    # Normalize probabilities
    denom = p_yes + p_no
    p_yes_normalized = torch.where(
        denom > 0, p_yes / denom, torch.full_like(p_yes, 0.5)
    )  # Handle edge cases

    # For MC dropout, reshape and average
    if mc_dropout:
        p_yes_normalized = p_yes_normalized.view(batch_size, n).mean(dim=1)

    # Calculate verification costs based on token counts
    input_token_counts = [
        len(tokenizer.encode(verify_prompts[i])) * n
        for i in range(0, len(verify_prompts), n)
    ]
    output_token_counts = [0] * batch_size


    # Calculate costs using the utility function
    model_name_attr = getattr(model, "model_name", getattr(model, "name_or_path", "unknown"))
    verification_costs = [
        calculate_costs(
            model_name=model_name_attr,
            input_token_length=input_count,
            output_token_length=output_count,
            output_input_price_ratio=output_input_price_ratio,  # irrelevant, because of no generation
        )
        for input_count, output_count in zip(input_token_counts, output_token_counts)
    ]

    return p_yes_normalized, verification_costs
    

def self_verification(
    model,
    tokenizer,
    prompts,
    generated_responses,
    questions,
    n=1,  # Number of repetitions for each prompt
    max_length=512,
    max_new_tokens=4,
    output_input_price_ratio=4.0,
    mc_dropout=False,
):
    """
    Computes the distribution of p(YES) for each question, repeated n times in a batch.

    Args:
        model: The LLM to evaluate.
        tokenizer: The tokenizer for the model.
        prompts: A list of questions (strings).
        generated_responses: A list of generated responses (strings).
        n: Number of times to repeat each question in the batch.
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
    for question, candidate_answer in zip(questions, generated_responses):
        text = (
            "Given the following question and the model's answer, please evaluate correctness.\n"
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
    ).to(model.device)

    # Generate outputs
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
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
    confidence_tensor = torch.tensor(confidence_scores, device=model.device).view(
        batch_size, n
    )

    # Calculate mean confidence, ignoring NaN values
    mean_confidence = torch.nanmean(confidence_tensor, dim=1)
    
    
    # Calculate verification costs based on token counts
    input_token_counts = [
        len(tokenizer.encode(verify_prompts[i])) * n
        for i in range(0, len(verify_prompts), n)
    ]
    output_token_counts = [output_input_price_ratio] * batch_size

    # Calculate costs using the utility function
    model_name_attr = getattr(model, "model_name", getattr(model, "name_or_path", "unknown"))
    verification_costs = [
        calculate_costs(
            model_name=model_name_attr,
            input_token_length=input_count,
            output_token_length=output_count,
            output_input_price_ratio=output_input_price_ratio,  # irrelevant, because of no generation
        )
        for input_count, output_count in zip(input_token_counts, output_token_counts)
    ]

    return mean_confidence, verification_costs
