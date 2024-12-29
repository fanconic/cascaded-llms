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

sns.set_style("whitegrid")
#plt.style.use("science")
plt.rcParams["font.family"] = "sans-serif"

# Parameters
MAX_NEW_TOKENS = 128
SMALL_GEN_COST = 1.0
LARGE_INF_COST = 0.5
LARGE_GEN_COST = 5.0
EXPERT_COST = 10.0
SMALL_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
LARGE_MODEL = (
    "/mnt/pdata/caf83/helm/me-llama/physionet.org/files/me-llama/1.0.0/MeLLaMA-13B-chat"
)
LARGE_MODEL =  "meta-llama/Llama-3.1-8B-Instruct"
PROBABILITY_THRESHOLD = 1.0
UNCERTAINTY_THRESHOLD = 0.3
BATCH_SIZE = 7
MAX_INPUT_LENGTH = 512
NUM_SAMPLES = 70


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

        self.verification_threshold = PROBABILITY_THRESHOLD
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

    def compute_kl_divergence(
        self, base_logits, large_logits, base_tokens, large_tokens
    ):
        with torch.no_grad():
            base_probs = log_softmax(base_logits, dim=-1)
            large_probs = log_softmax(large_logits, dim=-1)
            kl_divergence = kl_div(
                base_probs, large_probs, reduction="batchmean", log_target=True
            )
            return kl_divergence.item()
        
    def compute_generated_log_probs(
        self,
        model,
        tokenizer,
        prompts,
        generated_responses,
        device="cuda",
        normalize_by_length=True
    ):
        """
        Computes the log probability (or average log probability) *only* over the generated
        portion (i.e., ignoring the prompt tokens).

        :param model: A language model (e.g., GPT-like) in huggingface/transformers style.
        :param tokenizer: The corresponding tokenizer.
        :param prompts: List[str], the user prompts for each example in the batch.
        :param generated_responses: List[str], the generated responses (small-model output) 
                                    for each example in the batch.
        :param device: Which device to run on.
        :param normalize_by_length: If True, we divide the final log prob by the number
                                    of generated tokens.
        :return: A 1D tensor [batch_size], where each entry is the log probability
                (or avg log probability) of the *generated* portion only.
        """
        assert len(prompts) == len(generated_responses), "Mismatch in batch sizes!"

        batch_full_texts = []
        prompt_lengths = []  # number of tokens in the prompt portion
        gen_lengths = []     # number of tokens in the generated portion

        # 1. Build the combined sequences, record lengths
        for prompt, full_text in zip(prompts, generated_responses):
            gen_resp = full_text[len(prompt):]
            prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
            gen_ids    = tokenizer(gen_resp, add_special_tokens=False).input_ids

            prompt_len = len(prompt_ids)
            gen_len    = len(gen_ids)

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

        # 4. Shift the logits (causal LM): we compare logits[:, t] to input_ids[:, t+1]
        shift_logits = logits[:, :-1, :].contiguous()      # [batch, seq_len-1, vocab_size]
        shift_labels = input_ids[:, 1:].contiguous().to(device)       # [batch, seq_len-1]
        shift_mask   = attention_mask[:, 1:].contiguous().to(device)   # [batch, seq_len-1]

        # 5. Compute token-level cross-entropy (returns per-token loss)
        token_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction='none'  # we want per-token
        )
        token_loss = token_loss.view(shift_labels.size())  # [batch, seq_len-1]

        # 6. Now we accumulate the loss *only* in the region of generated tokens
        #    We'll do this example-by-example, because each prompt/gen length is different.
        batch_log_probs = torch.zeros(shift_labels.size(0), device=device)

        for i in range(shift_labels.size(0)):
            start_idx = prompt_lengths[i]
            end_idx   = prompt_lengths[i] + gen_lengths[i]

            # For safety, clamp start_idx-1 to >=0
            gen_start_in_shift = max(start_idx - 1, 0)
            gen_end_in_shift   = max(end_idx - 1, 0)

            # clamp to the actual length of shift_labels
            seq_len_i = shift_labels.size(1)
            gen_start_in_shift = min(gen_start_in_shift, seq_len_i)
            gen_end_in_shift   = min(gen_end_in_shift, seq_len_i)

            # Now sum up the negative log-likelihood over that slice
            # token_loss[i, t] is = -log p(token t+1), so summation is total NLL
            gen_loss_i = (token_loss[i, :] * shift_mask[i, :])[gen_start_in_shift:gen_end_in_shift].sum()

            # Convert negative log-likelihood to log-prob
            seq_log_prob_i = -gen_loss_i  # sum of log probs in that region

            # 7. Normalization
            if normalize_by_length:
                n_gen_tokens = (shift_mask[i, gen_start_in_shift:gen_end_in_shift] == 1).sum()
                if n_gen_tokens > 0:
                    seq_log_prob_i = seq_log_prob_i / n_gen_tokens

            batch_log_probs[i] = seq_log_prob_i

        return batch_log_probs

    def decide_batch(self, prompts):
        # This part here is only implemenmt like this, in order to efficiently compare against using dynamic vs small vs large model
        # In Practice the inference part would not be batched, and the large model evaluation only called if the verification check fails
        base_outputs, _ = self.generate_response(
            self.base_model, self.base_tokenizer, prompts
        )
        large_outputs, _ = self.generate_response(
            self.large_model, self.large_tokenizer, prompts
        )

        # 2. Verify with the large model's log probs (we can still do it in one pass if desired)
        #    But here we'll show how to compute the base model's *own* log prob, or large model's prob, etc.
        base_log_probs = self.compute_generated_log_probs(
            model=self.base_model,
            tokenizer=self.base_tokenizer,
            prompts=prompts,
            generated_responses=base_outputs,
            device="cuda:2",
            normalize_by_length=True
        )

        # If you want the large model's perspective on the *same text*:
        large_log_probs = self.compute_generated_log_probs(
            model=self.large_model,
            tokenizer=self.large_tokenizer,
            prompts=prompts,
            generated_responses=base_outputs,  # note: we pass base_outputs as the text to verify
            device="cuda:2",
            normalize_by_length=True
        )

        # 3. Now you can define a difference or ratio
        # difference = log P_large - log P_base
        difference = large_log_probs - base_log_probs
        ratio = torch.exp(difference)

        decisions = []
        small_model_predictions = []
        large_model_predictions = []

        for i, prompt in enumerate(prompts):
            base_response = base_outputs[i]
            large_response = large_outputs[i]

            base_uncertainty = 0  # Placeholder for uncertainty calculation
            large_uncertainty = 0  # Placeholder for uncertainty calculation
            prob_delta = ratio[i].item()

            small_model_predictions.append(base_response[len(prompt) :].strip())
            large_model_predictions.append(large_response[len(prompt) :].strip())

            if prob_delta >= self.verification_threshold:
                if base_uncertainty < self.uncertainty_threshold:
                    decisions.append(
                        {
                            "decision": 0,
                            "response": base_response,
                            "base_response": base_response,
                            "large_response": large_response,
                            "prob_delta": prob_delta,
                            "uncertainty": base_uncertainty,
                            "cost": SMALL_GEN_COST + LARGE_INF_COST,
                        }
                    )
                else:
                    decisions.append(
                        {
                            "decision": 2,
                            "response": None,
                            "base_response": base_response,
                            "large_response": large_response,
                            "prob_delta": prob_delta,
                            "uncertainty": base_uncertainty,
                            "cost": SMALL_GEN_COST + LARGE_INF_COST + EXPERT_COST,
                        }
                    )
            else:
                if large_uncertainty < self.uncertainty_threshold:
                    decisions.append(
                        {
                            "decision": 1,
                            "response": large_response,
                            "base_response": base_response,
                            "large_response": large_response,
                            "prob_delta": prob_delta,
                            "uncertainty": large_uncertainty,
                            "cost": SMALL_GEN_COST + LARGE_INF_COST + LARGE_GEN_COST,
                        }
                    )
                else:
                    decisions.append(
                        {
                            "decision": 2,
                            "response": None,
                            "base_response": base_response,
                            "large_response": large_response,
                            "prob_delta": prob_delta,
                            "uncertainty": large_uncertainty,
                            "cost": SMALL_GEN_COST + LARGE_INF_COST + LARGE_GEN_COST + EXPERT_COST,
                        }
                    )

        return decisions, small_model_predictions, large_model_predictions


def extract_predictions(response: str) -> str:
    """ Extract the single MC prediction from the response text

    Args:
        response (str): Full response of the LLMs

    Returns:
        str: Single character response (A, B, C, D, or E) of the MC question
    """
    try:
        if response.startswith("The best answer is: "): #8B
            return response[len("The best answer is: ")]
        elif response.startswith("The best answer is "): # 1B
            return response[len("The best answer is ")]
        elif response[0] in {"A", "B", "C", "D", "E"}: #MEDLLama
            return response[0]
        else:
            return None
    except:
        return None


def main():
    # Load dataset
    dataset = load_dataset("medalpaca/medical_meadow_medqa", split=f"train[:{NUM_SAMPLES}]")

    helm = ClinicalHELM()

    # Parallel processing for decision-making
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
    all_small_predictions, all_large_predictions = [], []

    for batch in tqdm(dataloader):
        prompts = [
            f"{sample}\nPlease answer with only one of the multiple choice option in the brackets and give an explanation. The format should be in the format 'The best answer is: <ans>\n\nExplanation:'"
            for sample in batch['input']
        ]
        batch_decisions, small_predictions, large_predictions = helm.decide_batch(
            prompts
        )

        # Store small and large model predictions
        all_small_predictions.extend(small_predictions)
        all_large_predictions.extend(large_predictions)

        for i, decision in enumerate(batch_decisions):
            decisions.append(decision["decision"])
            outputs.append(decision["response"])
            prob_deltas.append(decision["prob_delta"])
            uncerts.append(decision["uncertainty"])
            costs.append(decision["cost"])
            labels.append(batch["output"][i][0])
            predictions.append(
                extract_predictions(decision["response"][len(prompts[i]) :].strip())
            )

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
        }
    )

    # Calculate full dataset accuracy and costs for small and large models
    small_model_correct = [
        label == extract_predictions(pred)
        for label, pred in zip(labels, all_small_predictions)
    ]
    large_model_correct = [
        label == extract_predictions(pred)
        for label, pred in zip(labels, all_large_predictions)
    ]

    accuracy_base = sum(small_model_correct) / len(small_model_correct)
    accuracy_large = sum(large_model_correct) / len(small_model_correct)

    cost_base = len(dataset) * SMALL_GEN_COST
    cost_large = len(dataset) * LARGE_GEN_COST

    # Calculate dynamic decision accuracy and costs
    data["correct"] = (data["prediction"] == data["label"]).astype(int)
    dynamic_accuracy = data["correct"].mean()
    dynamic_cost = data["cost"].sum()

    # Visualisations
    plt.figure(figsize=(10, 5))
    plt.scatter(cost_base, accuracy_base, label="Base Model Only", marker="o")
    plt.scatter(cost_large, accuracy_large, label="Large Model Only", marker="o")
    plt.scatter(dynamic_cost, dynamic_accuracy, label="Dynamic Decisions", marker="x")
    plt.scatter(EXPERT_COST * len(data), 1.0, label="Expert", marker="o")
    # Add dashed lines between base model, large model, and expert
    plt.plot([cost_base, cost_large, EXPERT_COST * len(data)], 
            [accuracy_base, accuracy_large, 1.0], 
            linestyle="--", color="gray")
    plt.xlabel("Cumulative Cost")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Cost for Small, Large Models, and Dynamic Decisions")
    plt.legend()
    plt.savefig("results/accuracy_vs_cost_comparison.pdf", bbox_inches="tight")
    
   
        
    # Decision distribution plot
    plt.figure(figsize=(8, 4))
    decision_counts = data["decision"].value_counts(sort=False).reindex([0, 1, 2], fill_value=0)
    decision_counts.plot(kind="bar", color=["blue", "orange", "red"], alpha=0.7)
    plt.xticks(
        ticks=[0, 1, 2], labels=["Base Model", "Large Model", "Expert"], rotation=0
    )
    plt.ylabel("Frequency")
    plt.title("Distribution of Decisions")
    plt.savefig("results/decision_distribution.pdf", bbox_inches="tight")
    
    data.to_csv("result_df.csv")


if __name__ == "__main__":
    main()
