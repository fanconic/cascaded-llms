import re
import os
import matplotlib.pyplot as plt
import seaborn as sns
import random
import torch
import numpy as np

sns.set_style("whitegrid")
plt.style.use("science")
plt.rcParams["font.family"] = "sans-serif"


def extract_predictions(response: str) -> str:
    """Extract the single prediction from the response text.

    Args:
        response (str): Full response of the LLMs.

    Returns:
        str: Single character or word response ("A-E", "1-5", "yes", "no"), or None if not found.
    """
    try:
        # Patterns to match different formats of predictions
        patterns = [
            r"The best answer is: (\w+)",  # "The best answer is: A" or "The best answer is: 1"
            r"The best answer is (\w+)",  # "The best answer is A" or "The best answer is 1"
        ]

        # Loop through the patterns to find a match
        for pattern in patterns:
            match = re.search(pattern, response.strip(), re.IGNORECASE)
            if match:
                prediction = match.group(
                    1
                ).lower()  # Convert to lowercase for "yes"/"no"

                # Validate the extracted prediction
                if prediction in "abcde" or prediction in {"yes", "no"}:
                    return (
                        prediction.lower()
                        if prediction in {"yes", "no"}
                        else prediction.upper()
                    )

        # Return None if no valid pattern is matched
        return None
    except Exception as e:
        print(f"Error while extracting prediction: {e}")
        return None


def plot_accuracy_vs_cost(
    run_dir,
    cost_base,
    accuracy_base,
    accuracy_base_err,
    cost_large,
    accuracy_large,
    accuracy_large_err,
    expert_cost,
    data_length,
    dynamic_cost,
    dynamic_accuracy,
    dynamic_accuracy_err,
):
    plt.figure(figsize=(5, 5))

    base_plot = plt.errorbar(
        cost_base / data_length,
        accuracy_base,
        yerr=accuracy_base_err,
        label="Base Model",
        fmt="o",
        capsize=5,
    )
    large_plot = plt.errorbar(
        cost_large / data_length,
        accuracy_large,
        yerr=accuracy_large_err,
        label="Large Model",
        fmt="o",
        capsize=5,
    )
    expert_plot = plt.errorbar(
        expert_cost, 1.0, yerr=0.0, label="Expert", fmt="o", capsize=5
    )
    dynamic_plot = plt.errorbar(
        dynamic_cost / data_length,
        dynamic_accuracy,
        yerr=dynamic_accuracy_err,
        label="Dynamic Decisions",
        fmt="x",
        capsize=5,
    )

    plt.plot(
        [cost_base / data_length, cost_large / data_length, expert_cost],
        [accuracy_base, accuracy_large, 1.0],
        linestyle="--",
        color="gray",
        alpha=0.7,
    )

    plt.xlabel("Total Cost")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Cost Curve")
    plt.legend()
    plt.savefig(os.path.join(run_dir, "accuracy_vs_cost.pdf"), bbox_inches="tight")


def plot_decision_distribution(run_dir, decisions, labels):
    plt.figure(figsize=(5, 5))
    decision_counts = decisions.value_counts(sort=False).reindex(
        range(len(labels)), fill_value=0
    )
    decision_counts.plot(kind="bar", alpha=0.7)
    plt.xticks(ticks=range(len(labels)), labels=labels, rotation=0)
    plt.ylabel("Frequency")
    plt.title("Distribution of Decisions")
    plt.savefig(os.path.join(run_dir, "decision_distribution.pdf"), bbox_inches="tight")


def plot_tau_M(run_dir, decisions):
    plt.figure(figsize=(5, 5))

    # Option 1: Rename columns before plotting
    decisions_renamed = decisions.copy()
    decisions_renamed.columns = [
        # r"$\tau_{\text{base}}$",
        # r"$\tau_{\text{large}}$",
        "$M$",
    ]
    decisions_renamed.plot(kind="line")

    plt.xlabel("Online time steps (t)")
    plt.ylabel("Value")
    plt.title("Development of Parameters")
    plt.savefig(os.path.join(run_dir, "tau_M_development.pdf"), bbox_inches="tight")


def set_seed(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
