import scienceplots
import os
import json
import matplotlib.pyplot as plt
from src.utils import plot_accuracy_vs_cost_D1  # Replace with actual import path
import seaborn as sns
import pandas as pd

sns.set_style("whitegrid")
plt.style.use("science")
plt.rcParams["font.family"] = "sans-serif"



# === Helper: plot into a given axis (adapted from your function) ===
def plot_accuracy_vs_cost_D1(ax, experiment_data_list, data_length):
    NAMES_DICT = {
        "self_verification_1_base": "$\\text{SV}_{\\text{base}}$ ($n=1$)",
        "self_verification_5_base": "$\\text{SV}_{\\text{base}}$ ($n=5$)",
        "surrogate_token_probs_1_base": "$\\text{STP}_{\\text{base}}$ ($n=1$)",
        "surrogate_token_probs_5_base": "$\\text{STP}_{\\text{base}}$ ($n=5$)",
        "self_verification_1_large": "$\\text{SV}_{\\text{large}}$ ($n=1$)",
        "self_verification_5_large": "$\\text{SV}_{\\text{large}}$ ($n=5$)",
        "surrogate_token_probs_1_large": "$\\text{STP}_{\\text{large}}$ ($n=1$)",
        "surrogate_token_probs_5_large": "$\\text{STP}_{\\text{large}}$ ($n=5$)",
    }

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    exp_styles = {
        "Base Model": {"color": colors[0], "marker": "o"},
        "Large Model": {"color": colors[1], "marker": "o"},
        "self_verification_1_base": {"color": colors[2], "marker": "x"},
        "self_verification_5_base": {"color": colors[2], "marker": "^"},
        "surrogate_token_probs_1_base": {"color": colors[3], "marker": "x"},
        "surrogate_token_probs_5_base": {"color": colors[3], "marker": "^"},
        "self_verification_1_large": {"color": colors[4], "marker": "x"},
        "self_verification_5_large": {"color": colors[4], "marker": "^"},
        "surrogate_token_probs_1_large": {"color": colors[5], "marker": "x"},
        "surrogate_token_probs_5_large": {"color": colors[5], "marker": "^"},
    }

    plot_handles = {}

    # All rows have same base/large metrics
    base_metrics = experiment_data_list[0][1]
    plot_handles["Base Model"] = ax.errorbar(
        base_metrics["cost_base"] / data_length,
        base_metrics["accuracy_base"],
        yerr=base_metrics["accuracy_base_err"],
        fmt=exp_styles["Base Model"]["marker"],
        color=exp_styles["Base Model"]["color"],
        capsize=5,
    )
    plot_handles["Large Model"] =  ax.errorbar(
        base_metrics["cost_large"] / data_length,
        base_metrics["accuracy_large"],
        yerr=base_metrics["accuracy_large_err"],
        fmt=exp_styles["Large Model"]["marker"],
        color=exp_styles["Large Model"]["color"],
        capsize=5,
    )
    ax.plot(
        [base_metrics["cost_base"] / data_length, base_metrics["cost_large"] / data_length],
        [base_metrics["accuracy_base"], base_metrics["accuracy_large"]],
        linestyle="--", color="gray", alpha=0.7
    )

    # Plot dynamic points
    for experiment_name, metrics in experiment_data_list:
        if experiment_name in exp_styles:
            style = exp_styles[experiment_name]
            h = ax.errorbar(
                metrics["dynamic_cost"] / data_length,
                metrics["dynamic_accuracy"],
                yerr=metrics["dynamic_accuracy_err"],
                fmt=style["marker"],
                color=style["color"],
                capsize=5
            )
            plot_handles[experiment_name] = h

    ax.set_xlabel("Cost per Sample")
    ax.set_ylabel("Accuracy")
    ax.grid(True, linestyle="--", alpha=0.4)
    return plot_handles


# === Load metrics.csv and plot ===
def load_experiment_data_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    
    # Keep only the 'all' subject for MMLU
    if "mmlu" in csv_path.lower() and "subject" in df.columns:
        df = df[df["subject"] == "all"]
    
    experiment_data_list = []
    for _, row in df.iterrows():
        metrics_dict = row.to_dict()
        experiment_data_list.append((row["experiment_name"], metrics_dict))
    return experiment_data_list


def plot_all_datasets(base_dir, datasets, output_path):
    # Pre-check whether MMLU has data
    mmlu_path = os.path.join(base_dir, "mmlu", "metrics_combined.csv")
    mmlu_has_data = False
    if os.path.exists(mmlu_path):
        df_mmlu = pd.read_csv(mmlu_path)
        if "subject" in df_mmlu.columns:
            df_mmlu = df_mmlu[df_mmlu["subject"] == "all"]
        mmlu_has_data = not df_mmlu.empty

    # --- Figure layout depending on whether MMLU has data ---
    if mmlu_has_data:
        fig = plt.figure(figsize=(11, 7))
        # Top row: 3 plots, bottom row: 2 centred
        axes_top = [fig.add_axes([0.07 + i * 0.31, 0.55, 0.26, 0.35]) for i in range(3)]
        axes_bottom = [fig.add_axes([0.22 + i * 0.31, 0.10, 0.26, 0.35]) for i in range(2)]
        axes = axes_top + axes_bottom
    else:
        fig = plt.figure(figsize=(11, 3.5))
        # Single row with 4 evenly spaced subplots
        axes = [fig.add_axes([0.07 + i * 0.23, 0.15, 0.20, 0.70]) for i in range(4)]
    all_handles = []
    all_labels = []

    plot_idx = 0  # keep track of how many plots we actually draw
    for dataset in datasets:
        dataset_dir = os.path.join(base_dir, dataset)
        csv_path = os.path.join(dataset_dir, "metrics_combined.csv")
        if not os.path.exists(csv_path):
            print(f"⚠️ Skipping {dataset}: no metrics_combined.csv found")
            continue

        if dataset == "mmlu" and not mmlu_has_data:
            print("⚠️ Skipping MMLU: no 'all' subject data available")
            continue

        if plot_idx >= len(axes):
            print(f"⚠️ No available axes for {dataset}, skipping.")
            continue

        experiment_data_list = load_experiment_data_from_csv(csv_path)
        ax = axes[plot_idx]
        plot_idx += 1

        handles = plot_accuracy_vs_cost_D1(ax, experiment_data_list, len(experiment_data_list))

        # Titles
        ax.set_title({
            "arc_easy": "(a) ARC2-Easy",
            "arc_challenge": "(b) ARC2-Challenge",
            "mmlu": "(c) MMLU",
            "medqa": "(d) MedQA",
            "medmcqa": "(e) MedMCQA",
        }[dataset])

        # Collect handles/labels for shared legend (from last dataset)
        for h in handles.values():
            if h not in all_handles:
                all_handles.append(h)
        if not all_labels:
            all_labels = list(handles.keys())

    # Remove unused subplot if fewer than 6
    for j in range(len(datasets), len(axes)):
        fig.delaxes(axes[j])

    # Shared legend (horizontal, bottom)
    fig.legend(
        all_handles,
        [ "Base Model", "Large Model",
         r"$\text{SV}_{\text{base}}$ ($n=1$)",
         r"$\text{SV}_{\text{base}}$ ($n=5$)",
         r"$\text{STP}_{\text{base}}$ ($n=1$)",
         r"$\text{STP}_{\text{base}}$ ($n=5$)",
         r"$\text{SV}_{\text{large}}$ ($n=1$)",
         r"$\text{SV}_{\text{large}}$ ($n=5$)",
         r"$\text{STP}_{\text{large}}$ ($n=1$)",
         r"$\text{STP}_{\text{large}}$ ($n=5$)",
        ],
        loc="lower center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, -0.03) if mmlu_has_data else (0.5, -0.15)
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"✅ Combined figure saved to: {output_path}")


if __name__ == "__main__":
    BASE_DIR = "precomputed_responses/qwen_3_7/uncalibrated"
    DATASETS = ["arc_easy", "arc_challenge", "mmlu", "medqa", "medmcqa",]
    OUTPUT_PATH = os.path.join(BASE_DIR, "accuracy_vs_cost_all.pdf")
    plot_all_datasets(BASE_DIR, DATASETS, OUTPUT_PATH)