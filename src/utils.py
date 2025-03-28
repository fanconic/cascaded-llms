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

    plt.xlabel("Cost per Sample")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Cost Curve")
    plt.legend()
    plt.savefig(os.path.join(run_dir, "accuracy_vs_cost.pdf"), bbox_inches="tight")


def plot_accuracy_vs_cost_D1(
    run_dir,
    cost_base,
    accuracy_base,
    accuracy_base_err,
    cost_large,
    accuracy_large,
    accuracy_large_err,
    data_length,
    dynamic_cost,
    dynamic_accuracy,
    dynamic_accuracy_err,
    delta_ibc,
):
    plt.figure(figsize=(3, 3))

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

    dynamic_plot = plt.errorbar(
        dynamic_cost / data_length,
        dynamic_accuracy,
        yerr=dynamic_accuracy_err,
        label=r"$\phi_{\text{MvM}}$",
        fmt="x",
        capsize=5,
    )

    plt.plot(
        [cost_base / data_length, cost_large / data_length],
        [accuracy_base, accuracy_large],
        linestyle="--",
        color="gray",
        alpha=0.7,
    )

    plt.xlabel("Cost per Sample")
    plt.ylabel("Accuracy")
    plt.title(rf"$\Delta$IBC = {delta_ibc:.2f}")
    plt.legend()
    plt.savefig(os.path.join(run_dir, "accuracy_vs_cost_d1.pdf"), bbox_inches="tight")


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
    plt.figure(figsize=(15, 5))  # Wider figure to accommodate subplots

    # Option 1: Rename columns before plotting
    decisions_renamed = decisions.copy()
    decisions_renamed.columns = [
        r"$\tau_{\text{base}}$",
        r"$\tau_{\text{large}}$",
        "$M$",
    ]

    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True)
    parameter_names = decisions_renamed.columns

    for i, ax in enumerate(axes):
        decisions_renamed.iloc[:, i].plot(kind="line", ax=ax)
        ax.set_title(parameter_names[i])
        ax.set_xlabel("Online time steps (t)")
        ax.set_ylabel("Value")

    plt.suptitle("Development of Parameters", y=1.02)
    plt.tight_layout()
    plt.savefig(
        os.path.join(run_dir, "tau_M_development_subplots.pdf"), bbox_inches="tight"
    )


def plot_risk_and_cumulative_risk(run_dir, decisions):
    """
    Plots system risk and cumulative system risk in two subplots next to each other.

    Args:
        run_dir (str): Directory to save the plot.
        decisions (DataFrame): DataFrame with system risk values over time.

    Saves:
        A plot with system risk and cumulative system risk as a PDF.
    """

    # Prepare the data
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    base_color = colors[0]  # First system color
    large_color = colors[1]  # Second system color
    static_color = colors[2]  # Third system color
    dynamic_color = colors[3]  # Fourth system color
    expert_color = colors[4]  # Fifth system color

    decisions_copy = decisions.copy()
    decisions_copy.columns = ["Dynamic", "Base", "Large", "Static", "Expert", "M"]
    # Compute cumulative risk relative to the Dynamic system
    decisions_copy["Cumulative Dynamic"] = decisions_copy["Dynamic"].cumsum() - decisions_copy["Dynamic"].cumsum()  # will be zero
    decisions_copy["Cumulative Base"] = decisions_copy["Base"].cumsum() - decisions_copy["Dynamic"].cumsum()
    decisions_copy["Cumulative Large"] = decisions_copy["Large"].cumsum() - decisions_copy["Dynamic"].cumsum()
    decisions_copy["Cumulative Static"] = decisions_copy["Static"].cumsum() - decisions_copy["Dynamic"].cumsum()
    decisions_copy["Cumulative Expert"] = decisions_copy["Expert"].cumsum() - decisions_copy["Dynamic"].cumsum()

    # Calculate confidence intervals (95%)
    # Using bootstrap method to estimate confidence intervals
    n_bootstrap = 1000
    alpha = 0.05  # 95% confidence interval
    
    # Function to calculate bootstrap confidence intervals
    def bootstrap_ci(data, n_bootstrap=n_bootstrap, alpha=alpha):
        bootstrap_samples = np.zeros((n_bootstrap, len(data)))
        for i in range(n_bootstrap):
            # Sample with replacement and sort the indices to preserve time order
            indices = np.sort(np.random.choice(len(data), size=len(data), replace=True))
            bootstrap_samples[i] = (
                np.cumsum(data.iloc[indices].values) - 
                np.cumsum(decisions_copy["Dynamic"].iloc[indices].values)
            )
        lower = np.percentile(bootstrap_samples, alpha/2 * 100, axis=0)
        upper = np.percentile(bootstrap_samples, (1 - alpha/2) * 100, axis=0)
        return lower, upper
    
    # Calculate CIs for each system
    base_lower, base_upper = bootstrap_ci(decisions_copy["Base"])
    large_lower, large_upper = bootstrap_ci(decisions_copy["Large"])
    static_lower, static_upper = bootstrap_ci(decisions_copy["Static"])
    expert_lower, expert_upper = bootstrap_ci(decisions_copy["Expert"])
    
    # Create the plot
    plt.figure(figsize=(4, 4))

    # Plot the cumulative system risk with confidence intervals
    plt.plot(
        decisions_copy.index,
        decisions_copy["Cumulative Base"],
        label="Base",
        color=base_color,
        alpha=0.8,
    )
    plt.fill_between(
        decisions_copy.index,
        base_lower,
        base_upper,
        color=base_color,
        alpha=0.2,
    )
    
    plt.plot(
        decisions_copy.index,
        decisions_copy["Cumulative Large"],
        label="Large",
        color=large_color,
        alpha=0.8,
    )
    plt.fill_between(
        decisions_copy.index,
        large_lower,
        large_upper,
        color=large_color,
        alpha=0.2,
    )
    
    plt.plot(
        decisions_copy.index,
        decisions_copy["Cumulative Static"],
        label="Static",
        color=static_color,
        alpha=0.8,
    )
    plt.fill_between(
        decisions_copy.index,
        static_lower,
        static_upper,
        color=static_color,
        alpha=0.2,
    )
    
    plt.plot(
        decisions_copy.index,
        decisions_copy["Cumulative Dynamic"],
        label="Dynamic",
        color=dynamic_color,
        linestyle="--",
        alpha=0.8,
    )
    
    plt.plot(
        decisions_copy.index,
        decisions_copy["Cumulative Expert"],
        label="Expert",
        color=expert_color,
        alpha=0.8,
    )
    plt.fill_between(
        decisions_copy.index,
        expert_lower,
        expert_upper,
        color=expert_color,
        alpha=0.2,
    )
    
    plt.xlabel("Online time steps (t)")
    plt.ylabel("Cumulative Regret")
    plt.legend()

    plt.tight_layout()
    os.makedirs(run_dir, exist_ok=True)  # Ensure the directory exists
    plt.savefig(
        os.path.join(run_dir, "cumulative_regret_over_time.pdf"), bbox_inches="tight"
    )
    plt.close()

    # Plot the system risk
    plt.figure(figsize=(4, 4))

    plt.plot(
        decisions_copy.index,
        np.ones(len(decisions_copy)) * decisions_copy["M"].iloc[0],
        label=r"Static $M$",
        linestyle="--",
        color=static_color,
    )
    plt.plot(
        decisions_copy.index,
        decisions_copy["M"],
        label=r"Dynamic $M$",
        color=dynamic_color,
    )
    plt.title(r"$M$ Parameter over Time")
    plt.xlabel("Online time steps (t)")
    plt.ylabel(r"$M$")
    plt.legend()

    # Adjust layout and save
    plt.tight_layout()
    os.makedirs(run_dir, exist_ok=True)  # Ensure the directory exists
    plt.savefig(os.path.join(run_dir, "M_over_time.pdf"), bbox_inches="tight")
    plt.close()


def set_seed(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
