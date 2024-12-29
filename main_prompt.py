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

sns.set_style("whitegrid")
plt.style.use("science")
plt.rcParams["font.family"] = "sans-serif"

# Parameters
MAX_NEW_TOKENS = 128
SMALL_GEN_COST = 1.0
LARGE_INF_COST = 0.2
LARGE_GEN_COST = 5.0
EXPERT_COST = 10.0
SMALL_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
LARGE_MODEL = (
    "/mnt/pdata/caf83/helm/me-llama/physionet.org/files/me-llama/1.0.0/MeLLaMA-13B-chat"
)
LARGE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
PROBABILITY_THRESHOLD = 0.5  # For "YES"/"NO", treat YES => 1.0, NO => 0.0
UNCERTAINTY_THRESHOLD = 0.3
BATCH_SIZE = 7
MAX_INPUT_LENGTH = 512
NUM_SAMPLES = 1000
now = datetime.datetime.now()
RUN_NAME = f"debug_{now.year}{now.month}{now.day}_{now.hour}{now.minute}"


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
        self.large_model = AutoModelForCausalLM.from_pretrained(LARGE_MODEL).to(
            "cuda:1"
        )

        self.verification_threshold = PROBABILITY_THRESHOLD
        self.uncertainty_threshold = UNCERTAINTY_THRESHOLD

    def generate_response(self, model, tokenizer, prompts):
        """
        Generate a full answer from the given model for each prompt in 'prompts'.
        """
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

    def verify_with_large_model(self, prompt, base_answer):
        """
        Prompts the large model to verify correctness of the small model's answer.
        We ask for a single token: 'YES' or 'NO'.

        Returns a float:
          1.0 if the large model responds with 'YES'
          0.0 if the large model responds with 'NO'
        (You could also return more nuanced scores if desired.)
        """
        verify_prompt = (
            f"Question:\n{prompt}\n\n"
            f"Small model's answer:\n{base_answer}\n\n"
            "Is the small model's answer correct? "
            "Please respond with a single token: YES or NO."
        )

        inputs = self.large_tokenizer(
            [verify_prompt],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_INPUT_LENGTH,
        ).to(self.large_model.device)

        # We want to generate a very short response, ideally just 1–2 tokens
        with torch.no_grad():
            outputs = self.large_model.generate(
                **inputs,
                max_new_tokens=2,    # Enough to capture "YES" or "NO"
                pad_token_id=self.large_tokenizer.pad_token_id,
            )

        raw_text = self.large_tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract just the new portion after the prompt. A quick approach:
        verification_text = raw_text[len(verify_prompt) :].strip()

        # You can refine the parsing, but here's a simple approach:
        verification_text = verification_text.upper()
        if "YES" in verification_text[:5]:
            return 1.0
        else:
            # If it doesn't explicitly say YES, treat it as NO
            return 0.0

    def decide_batch(self, prompts):
        """
        In this version, the large model does *not* compute log-probs of the base model's
        answers. Instead, we prompt the large model to produce a single token 'YES'/'NO'
        to indicate whether the base model's answer is correct.
        """
        # 1. Generate from the base (small) model
        base_outputs, _ = self.generate_response(
            self.base_model, self.base_tokenizer, prompts
        )

        # 2. Also generate from the large model (if we need it)
        #    We'll do it here for demonstration, but in practice might do it only if needed
        large_outputs, _ = self.generate_response(
            self.large_model, self.large_tokenizer, prompts
        )

        decisions = []
        small_model_predictions = []
        large_model_predictions = []

        for i, prompt in enumerate(prompts):
            base_response = base_outputs[i]
            large_response = large_outputs[i]
            # Optionally, the part after the prompt is the "answer"
            base_answer_only = base_response[len(prompt) :].strip()

            # 3. Let the large model verify correctness of the small model’s answer
            verification_score = self.verify_with_large_model(prompt, base_answer_only)

            # Optionally define some "uncertainty." We'll keep it at 0.0 for now
            base_uncertainty = 0.0
            large_uncertainty = 0.0

            # We interpret verification_score = 1.0 => 'YES', 0.0 => 'NO'
            # If verification_score >= 0.5 => accept small model
            prob_delta = verification_score

            small_model_predictions.append(base_answer_only)
            large_model_predictions.append(large_response[len(prompt) :].strip())

            # 4. Decide which route to pick, similar to your prior logic
            if prob_delta >= self.verification_threshold:
                # Accept small model
                if base_uncertainty < self.uncertainty_threshold:
                    decisions.append(
                        {
                            "decision": 0,  # 0 => base model
                            "response": base_response,
                            "base_response": base_response,
                            "large_response": large_response,
                            "verification_score": prob_delta,
                            "uncertainty": base_uncertainty,
                            "cost": SMALL_GEN_COST + LARGE_INF_COST,
                        }
                    )
                else:
                    decisions.append(
                        {
                            "decision": 2,  # 2 => expert
                            "response": None,
                            "base_response": base_response,
                            "large_response": large_response,
                            "verification_score": prob_delta,
                            "uncertainty": base_uncertainty,
                            "cost": SMALL_GEN_COST + LARGE_INF_COST + EXPERT_COST,
                        }
                    )
            else:
                # Use large model
                if large_uncertainty < self.uncertainty_threshold:
                    decisions.append(
                        {
                            "decision": 1,  # 1 => large model
                            "response": large_response,
                            "base_response": base_response,
                            "large_response": large_response,
                            "verification_score": prob_delta,
                            "uncertainty": large_uncertainty,
                            "cost": SMALL_GEN_COST + LARGE_INF_COST + LARGE_GEN_COST,
                        }
                    )
                else:
                    decisions.append(
                        {
                            "decision": 2,  # 2 => expert
                            "response": None,
                            "base_response": base_response,
                            "large_response": large_response,
                            "verification_score": prob_delta,
                            "uncertainty": large_uncertainty,
                            "cost": SMALL_GEN_COST
                            + LARGE_INF_COST
                            + LARGE_GEN_COST
                            + EXPERT_COST,
                        }
                    )

        return decisions, small_model_predictions, large_model_predictions


def extract_predictions(response: str) -> str:
    """ Extract the single MC prediction from the response text.

    Args:
        response (str): Full response of the LLMs.

    Returns:
        str: Single character response (A, B, C, D, or E) of the MC question.
    """
    try:
        # Patterns to search for
        patterns = [
            r"The best answer is: (\w)",  # Matches "The best answer is: A"
            r"The best answer is (\w)",  # Matches "The best answer is A"
            r"\b([A-E])\b"              # Matches any single character A-E
        ]
        
        # Search for the patterns in the response
        for pattern in patterns:
            match = re.search(pattern, response)
            if match:
                return match.group(1)  # Extract the first capture group (the predicted letter)

        # If no match is found, return None
        return None
    except Exception as e:
        print(f"Error while extracting prediction: {e}")
        return None


def main():
    # Load dataset
    dataset = load_dataset(
        "medalpaca/medical_meadow_medqa", split=f"train[:{NUM_SAMPLES}]"
    )

    helm = ClinicalHELM()

    # Parallel processing for decision-making
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    decisions, outputs, prob_deltas, uncerts, costs, labels, predictions, base_log_prob, large_log_prob = (
        [],
        [],
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
            f"{sample}\nPlease answer with only one of the multiple choice option in the brackets. The format should be in the format 'The best answer is: <ans>."
            for sample in batch["input"]
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
            prob_deltas.append(decision["verification_score"])
            uncerts.append(decision["uncertainty"])
            costs.append(decision["cost"])
            labels.append(batch["output"][i][0])
            # Extract final predicted letter (for MC question)
            if decision["response"] is not None:
                pred_letter = extract_predictions(
                    decision["response"][len(prompts[i]) :].strip()
                )
            else:
                pred_letter = None
            predictions.append(pred_letter)
            base_log_prob.append(None)   # Not used in this simplified version
            large_log_prob.append(None)  # Not used in this simplified version

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
            "large_log_prob": large_log_prob
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
    accuracy_base_err = np.array(small_model_correct).std() / np.sqrt(
        len(small_model_correct)
    )
    accuracy_large = sum(large_model_correct) / len(small_model_correct)
    accuracy_large_err = np.array(large_model_correct).std() / np.sqrt(
        len(large_model_correct)
    )

    cost_base = len(dataset) * SMALL_GEN_COST
    cost_large = len(dataset) * LARGE_GEN_COST

    # Calculate dynamic decision accuracy and costs
    data["correct"] = (data["prediction"] == data["label"]).astype(int)
    dynamic_accuracy = data["correct"].mean()
    dynamic_accuracy_err = data["correct"].std() / np.sqrt(len(data["correct"]))
    dynamic_cost = data["cost"].sum()

    # First plot: Accuracy vs Cost Curve
    plt.figure(figsize=(10, 5))

    # Plot each category and capture the colours from the auto-generated palette (with error bars)
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

    # Extract colours from the plotted objects
    base_color = base_plot[0].get_color()
    large_color = large_plot[0].get_color()
    expert_color = expert_plot[0].get_color()

    # Connect points with dashed lines
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
    plt.savefig(f"results_prompt/accuracy_vs_cost_{RUN_NAME}.pdf", bbox_inches="tight")

    # Second plot: Decision distribution
    plt.figure(figsize=(8, 4))
    decision_counts = (
        data["decision"].value_counts(sort=False).reindex([0, 1, 2], fill_value=0)
    )

    decision_counts.plot(
        kind="bar", 
        color=[base_color, large_color, expert_color], 
        alpha=0.7
    )
    plt.xticks(
        ticks=[0, 1, 2], labels=["Base Model", "Large Model", "Expert"], rotation=0
    )
    plt.ylabel("Frequency")
    plt.title("Distribution of Decisions")
    plt.savefig(f"results_prompt/decision_distribution_{RUN_NAME}.pdf", bbox_inches="tight")

    # Save data
    data.to_csv(f"results_prompt/result_df_{RUN_NAME}.csv", index=False)


if __name__ == "__main__":
    main()