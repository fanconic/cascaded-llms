import scienceplots
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.nn.functional import kl_div, log_softmax
import torch.nn.functional as F
from tqdm import tqdm
import datetime
import numpy as np
import re
import random  # for the probabilistic gating

sns.set_style("whitegrid")
plt.style.use("science")
plt.rcParams["font.family"] = "sans-serif"

# Parameters
MAX_NEW_TOKENS = 128
SMALL_GEN_COST = 1.0
LARGE_INF_COST = 0.2
LARGE_GEN_COST = 5.0
EXPERT_COST = 10.0
SMALL_MODEL = "HuggingFaceTB/SmolLM-1.7B-Instruct"
LARGE_MODEL = (
    "/mnt/pdata/caf83/helm/me-llama/physionet.org/files/me-llama/1.0.0/MeLLaMA-13B-chat"
)
PROBABILITY_THRESHOLD = 1.0   # (Not used directly anymore for acceptance—see note below)
UNCERTAINTY_THRESHOLD = 0.3   # If average entropy is above this, we route to expert
BATCH_SIZE = 7
MAX_INPUT_LENGTH = 512
NUM_SAMPLES = 1000
now = datetime.datetime.now()
RUN_NAME = f"debug_{now.year}{now.month}{now.day}_{now.hour}{now.minute}"

# ---------------------------------------------------------------------------
# 1) HELPER FUNCTIONS FOR LOG-PROB & ENTROPY
# ---------------------------------------------------------------------------
def compute_generated_log_probs(
    model,
    tokenizer,
    prompts,
    generated_responses,
    device="cuda",
    normalize_by_length=True,
):
    """
    Computes the log probability (or average log probability) *only* over the generated
    portion (i.e., ignoring the prompt tokens).
    Returns a 1D tensor [batch_size] with the average log prob of the *generated* portion.
    """
    assert len(prompts) == len(generated_responses), "Mismatch in batch sizes!"

    batch_full_texts = []
    prompt_lengths = []  # number of tokens in the prompt portion
    gen_lengths = []     # number of tokens in the generated portion

    # 1. Build the combined sequences, record lengths
    for prompt, full_text in zip(prompts, generated_responses):
        gen_resp = full_text[len(prompt) :]
        prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
        gen_ids = tokenizer(gen_resp, add_special_tokens=False).input_ids

        prompt_len = len(prompt_ids)
        gen_len = len(gen_ids)

        batch_full_texts.append(full_text)
        prompt_lengths.append(prompt_len)
        gen_lengths.append(gen_len)

    # 2. Tokenize the *combined* sequences in a batch
    inputs = tokenizer(
        batch_full_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_INPUT_LENGTH,
    ).to(model.device)

    input_ids = inputs["input_ids"]         # [batch, seq_len]
    attention_mask = inputs["attention_mask"]  # [batch, seq_len]

    # 3. Forward pass
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits.to(device)  # [batch, seq_len, vocab_size]

    # 4. Shift for causal LM
    shift_logits = logits[:, :-1, :].contiguous()  # [batch, seq_len-1, vocab_size]
    shift_labels = input_ids[:, 1:].contiguous().to(device)  # [batch, seq_len-1]
    shift_mask = attention_mask[:, 1:].contiguous().to(device)  # [batch, seq_len-1]

    # 5. Compute token-level cross-entropy (returns per-token loss)
    token_loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="none",  # we want per-token
    )
    token_loss = token_loss.view(shift_labels.size())  # [batch, seq_len-1]

    # 6. Sum over generated tokens
    batch_log_probs = torch.zeros(shift_labels.size(0), device=device)

    for i in range(shift_labels.size(0)):
        start_idx = prompt_lengths[i]
        end_idx = prompt_lengths[i] + gen_lengths[i]

        gen_start_in_shift = max(start_idx - 1, 0)
        gen_end_in_shift = max(end_idx - 1, 0)

        seq_len_i = shift_labels.size(1)
        gen_start_in_shift = min(gen_start_in_shift, seq_len_i)
        gen_end_in_shift = min(gen_end_in_shift, seq_len_i)

        gen_loss_i = (token_loss[i, :] * shift_mask[i, :])[gen_start_in_shift:gen_end_in_shift].sum()
        seq_log_prob_i = -gen_loss_i  # sum of log probs in that region

        if normalize_by_length:
            n_gen_tokens = (shift_mask[i, gen_start_in_shift:gen_end_in_shift] == 1).sum()
            if n_gen_tokens > 0:
                seq_log_prob_i = seq_log_prob_i / n_gen_tokens

        batch_log_probs[i] = seq_log_prob_i

    return batch_log_probs


def compute_generated_entropy(
    model,
    tokenizer,
    prompts,
    generated_responses,
    device="cuda",
    normalize_by_length=True,
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
        max_length=MAX_INPUT_LENGTH,
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
        region_entropy = (token_entropy[i, :] * shift_mask[i, :])[gen_start_in_shift:gen_end_in_shift].sum()

        if normalize_by_length:
            n_gen_tokens = (shift_mask[i, gen_start_in_shift:gen_end_in_shift] == 1).sum()
            if n_gen_tokens > 0:
                region_entropy = region_entropy / n_gen_tokens

        batch_entropies[i] = region_entropy

    return batch_entropies


# ---------------------------------------------------------------------------
# 2) MAIN CLASS
# ---------------------------------------------------------------------------
class ClinicalHELM:
    def __init__(self):
        # Initialise models and tokenizers
        print("Loading Small Model")
        self.base_tokenizer = AutoTokenizer.from_pretrained(SMALL_MODEL)
        self.base_tokenizer.pad_token = self.base_tokenizer.eos_token
        self.base_tokenizer.padding_side = "left"
        self.base_model = AutoModelForCausalLM.from_pretrained(SMALL_MODEL).to("cuda:0")

        print("Loading Large Model")
        self.large_tokenizer = AutoTokenizer.from_pretrained(LARGE_MODEL)
        self.large_tokenizer.pad_token = self.large_tokenizer.eos_token
        self.large_tokenizer.padding_side = "left"
        self.large_model = AutoModelForCausalLM.from_pretrained(LARGE_MODEL).to("cuda:1")

        self.uncertainty_threshold = UNCERTAINTY_THRESHOLD

    def generate_response(self, model, tokenizer, prompts):
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_INPUT_LENGTH,
        ).to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                pad_token_id=tokenizer.pad_token_id,
            )
        responses = [
            tokenizer.decode(output, skip_special_tokens=True) for output in outputs
        ]
        return responses, inputs

    def decide_batch(self, prompts):
        """
        Probabilistic gating:
          1) If small-model entropy > threshold => expert (decision=2).
          2) Else accept the small model with probability alpha = ratio / (1 + ratio).
             If not accepted => check large model's entropy.
               - If large-model entropy > threshold => expert (2)
               - else => large model (1)
        """
        # Generate from small model & large model
        base_outputs, _ = self.generate_response(self.base_model, self.base_tokenizer, prompts)
        large_outputs, _ = self.generate_response(self.large_model, self.large_tokenizer, prompts)

        # 1. Compute log probs for the small model & the large model verifying the small model's text
        base_log_probs = compute_generated_log_probs(
            model=self.base_model,
            tokenizer=self.base_tokenizer,
            prompts=prompts,
            generated_responses=base_outputs,
            device="cuda:2",
            normalize_by_length=True,
        )
        large_log_probs_for_base = compute_generated_log_probs(
            model=self.large_model,
            tokenizer=self.large_tokenizer,
            prompts=prompts,
            generated_responses=base_outputs,
            device="cuda:2",
            normalize_by_length=True,
        )

        # 2. Compute ratio & acceptance probability
        difference = large_log_probs_for_base - base_log_probs
        ratio = torch.exp(difference)  # ratio = p_large / p_small
        # We define acceptance_prob = ratio / (1 + ratio)
        acceptance_prob = ratio / (1.0 + ratio)

        # 3. Compute entropies to gauge uncertainty
        base_uncertainties = compute_generated_entropy(
            model=self.base_model,
            tokenizer=self.base_tokenizer,
            prompts=prompts,
            generated_responses=base_outputs,
            device="cuda:2",
            normalize_by_length=True,
        )
        large_uncertainties = compute_generated_entropy(
            model=self.large_model,
            tokenizer=self.large_tokenizer,
            prompts=prompts,
            generated_responses=large_outputs,  # The large model's own output
            device="cuda:2",
            normalize_by_length=True,
        )

        decisions = []
        small_model_predictions = []
        large_model_predictions = []

        for i, prompt in enumerate(prompts):
            base_response = base_outputs[i]
            large_response = large_outputs[i]
            base_answer = base_response[len(prompt) :].strip()
            large_answer = large_response[len(prompt) :].strip()

            # We'll read off the ratio-based acceptance probability
            alpha_i = acceptance_prob[i].item()

            # Uncertainties
            small_uncert = base_uncertainties[i].item()
            big_uncert = large_uncertainties[i].item()

            small_model_predictions.append(base_answer)
            large_model_predictions.append(large_answer)

            # 1) If small model is too uncertain => send to expert
            if small_uncert > self.uncertainty_threshold:
                decisions.append(
                    {
                        "decision": 2,  # Expert
                        "response": None,
                        "base_response": base_response,
                        "large_response": large_response,
                        "base_log_probs": base_log_probs[i].item(),
                        "large_log_probs": large_log_probs_for_base[i].item(),
                        "prob_delta": ratio[i].item(),
                        "uncertainty": small_uncert,
                        "cost": SMALL_GEN_COST + LARGE_INF_COST + EXPERT_COST,
                    }
                )
                continue

            # 2) Probabilistic acceptance
            u = random.random()  # uniform(0,1)
            if u < alpha_i:
                # Accept small model
                decisions.append(
                    {
                        "decision": 0,  # small model
                        "response": base_response,
                        "base_response": base_response,
                        "large_response": large_response,
                        "base_log_probs": base_log_probs[i].item(),
                        "large_log_probs": large_log_probs_for_base[i].item(),
                        "prob_delta": ratio[i].item(),
                        "uncertainty": small_uncert,
                        "cost": SMALL_GEN_COST + LARGE_INF_COST,
                    }
                )
            else:
                # 3) If we use the large model, check if it's also too uncertain
                if big_uncert > self.uncertainty_threshold:
                    decisions.append(
                        {
                            "decision": 2,  # Expert
                            "response": None,
                            "base_response": base_response,
                            "large_response": large_response,
                            "base_log_probs": base_log_probs[i].item(),
                            "large_log_probs": large_log_probs_for_base[i].item(),
                            "prob_delta": ratio[i].item(),
                            "uncertainty": big_uncert,
                            "cost": SMALL_GEN_COST + LARGE_INF_COST + LARGE_GEN_COST + EXPERT_COST,
                        }
                    )
                else:
                    # Accept large model
                    decisions.append(
                        {
                            "decision": 1,  # large model
                            "response": large_response,
                            "base_response": base_response,
                            "large_response": large_response,
                            "base_log_probs": base_log_probs[i].item(),
                            "large_log_probs": large_log_probs_for_base[i].item(),
                            "prob_delta": ratio[i].item(),
                            "uncertainty": big_uncert,
                            "cost": SMALL_GEN_COST + LARGE_INF_COST + LARGE_GEN_COST,
                        }
                    )

        return decisions, small_model_predictions, large_model_predictions


# ---------------------------------------------------------------------------
# 3) UTILITY FOR EXTRACTING MC PREDICTIONS
# ---------------------------------------------------------------------------
def extract_predictions(response: str) -> str:
    """ Extract the single MC prediction from the response text.

    Args:
        response (str): Full response of the LLMs.

    Returns:
        str: Single character response (A, B, C, D, or E) of the MC question.
    """
    try:
        patterns = [
            r"The best answer is: (\w)",  # "The best answer is: A"
            r"The best answer is (\w)",  # "The best answer is A"
            r"\b([A-E])\b"              # single letter A-E
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response)
            if match:
                return match.group(1)

        return None
    except Exception as e:
        print(f"Error while extracting prediction: {e}")
        return None


# ---------------------------------------------------------------------------
# 4) MAIN
# ---------------------------------------------------------------------------
def main():
    # Load dataset
    dataset = load_dataset(
        "medalpaca/medical_meadow_medqa", split=f"train[:{NUM_SAMPLES}]"
    )

    helm = ClinicalHELM()

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    decisions, outputs, prob_deltas, uncerts, costs, labels, predictions = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    base_log_prob, large_log_prob = [], []
    all_small_predictions, all_large_predictions = [], []

    for batch in tqdm(dataloader):
        prompts = [
            f"{sample}\nPlease answer with only one of the multiple choice option in the brackets. The format should be in the format 'The best answer is: <ans>."
            for sample in batch["input"]
        ]
        batch_decisions, small_predictions, large_predictions = helm.decide_batch(prompts)

        all_small_predictions.extend(small_predictions)
        all_large_predictions.extend(large_predictions)

        for i, decision in enumerate(batch_decisions):
            decisions.append(decision["decision"])
            outputs.append(decision["response"])
            prob_deltas.append(decision["prob_delta"])
            uncerts.append(decision["uncertainty"])
            costs.append(decision["cost"])
            labels.append(batch["output"][i][0])

            if decision["response"] is not None:
                # get final predicted letter
                pred_letter = extract_predictions(
                    decision["response"][len(prompts[i]) :].strip()
                )
            else:
                pred_letter = None

            predictions.append(pred_letter)
            base_log_prob.append(decision["base_log_probs"])
            large_log_prob.append(decision["large_log_probs"])

    # Create DataFrame
    data = pd.DataFrame(
        {
            "decision": decisions,
            "output": outputs,
            "prob_delta": prob_deltas,
            "uncertainty": uncerts,
            "label": labels,
            "prediction": predictions,
            "cost": costs,
            "base_response": all_small_predictions,
            "base_prediction": [
                extract_predictions(pred) for pred in all_small_predictions
            ],
            "large_response": all_large_predictions,
            "large_prediction": [
                extract_predictions(pred) for pred in all_large_predictions
            ],
            "base_log_prob": base_log_prob,
            "large_log_prob": large_log_prob,
        }
    )

    # Calculate full dataset accuracy & costs for small/large-only
    small_model_correct = [
        label == extract_predictions(pred)
        for label, pred in zip(labels, all_small_predictions)
    ]
    large_model_correct = [
        label == extract_predictions(pred)
        for label, pred in zip(labels, all_large_predictions)
    ]

    accuracy_base = sum(small_model_correct) / len(small_model_correct)
    accuracy_base_err = np.array(small_model_correct).std() / np.sqrt(len(small_model_correct))
    accuracy_large = sum(large_model_correct) / len(large_model_correct)
    accuracy_large_err = np.array(large_model_correct).std() / np.sqrt(len(large_model_correct))

    cost_base = len(dataset) * SMALL_GEN_COST
    cost_large = len(dataset) * LARGE_GEN_COST

    # Dynamic decision performance
    data["correct"] = (data["prediction"] == data["label"]).astype(int)
    dynamic_accuracy = data["correct"].mean()
    dynamic_accuracy_err = data["correct"].std() / np.sqrt(len(data["correct"]))
    dynamic_cost = data["cost"].sum()

    # Plot: Accuracy vs Cost
    plt.figure(figsize=(10, 5))

    base_plot = plt.errorbar(
        cost_base,
        accuracy_base,
        yerr=accuracy_base_err,
        label="Base Model",
        fmt="o",
        capsize=5,
    )
    large_plot = plt.errorbar(
        cost_large,
        accuracy_large,
        yerr=accuracy_large_err,
        label="Large Model",
        fmt="o",
        capsize=5,
    )
    expert_plot = plt.errorbar(
        EXPERT_COST * len(data), 1.0, yerr=0.0, label="Expert", fmt="o", capsize=5
    )
    dynamic_plot = plt.errorbar(
        dynamic_cost,
        dynamic_accuracy,
        yerr=dynamic_accuracy_err,
        label="Dynamic Decisions",
        fmt="x",
        capsize=5,
    )

    base_color = base_plot[0].get_color()
    large_color = large_plot[0].get_color()
    expert_color = expert_plot[0].get_color()

    plt.plot(
        [cost_base, cost_large, EXPERT_COST * len(data)],
        [accuracy_base, accuracy_large, 1.0],
        linestyle="--",
        color="gray",
        alpha=0.7,
    )

    plt.xlabel("Cumulative Cost")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Cost Curve")
    plt.legend()
    plt.savefig(f"results/accuracy_vs_cost_{RUN_NAME}.pdf", bbox_inches="tight")

    # Decision distribution
    plt.figure(figsize=(8, 4))
    decision_counts = data["decision"].value_counts(sort=False).reindex([0, 1, 2], fill_value=0)
    decision_counts.plot(kind="bar", color=[base_color, large_color, expert_color], alpha=0.7)
    plt.xticks(ticks=[0, 1, 2], labels=["Base Model", "Large Model", "Expert"], rotation=0)
    plt.ylabel("Frequency")
    plt.title("Distribution of Decisions")
    plt.savefig(f"results/decision_distribution_{RUN_NAME}.pdf", bbox_inches="tight")

    # Save data
    data.to_csv(f"results/result_df_{RUN_NAME}.csv", index=False)


if __name__ == "__main__":
    main()