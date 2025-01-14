import os
import datetime
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils import plot_accuracy_vs_cost, plot_decision_distribution, plot_tau_M
from src.online_sft import OnlineSFTTrainerLoRA
from src.config import ExperimentConfig
from src.factory import DecisionMakerFactory
from src.utils import extract_predictions
from src.preprocessor import get_preprocessor


class Experiment:
    """Handles experiment execution and data collection."""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.run_dir = self._setup_run_directory()
        self.dataset, self.dataset_length = self._load_dataset()
        self.preprocessor = get_preprocessor(cfg.dataset.name)

        exp_config = ExperimentConfig(
            base_model=cfg.base_model,
            large_model=cfg.large_model,
            verification_fn=cfg.verification_fn,
            uncertainty_fn=cfg.uncertainty_fn,
            max_input_length=cfg.max_input_length,
            max_new_tokens=cfg.max_new_tokens,
            device=cfg.device,
            precomputed=cfg.precomputed.enable,
        )

        cost_config = {
            "base_gen_cost": cfg.base_gen_cost,
            "large_gen_cost": cfg.large_gen_cost,
            "large_inf_cost": cfg.large_inf_cost,
            "expert_cost": cfg.expert_cost,
        }

        self.decision_system = DecisionMakerFactory.create_decision_maker(
            exp_config, cfg.online, cost_config
        )

        self.sft_trainers = self._initialize_sft() if cfg.sft.enable else None

        if self.cfg.precomputed.enable:
            self.precomputed_df = pd.read_csv(self.cfg.precomputed.path)

    def _load_dataset(self):
        if self.cfg.dataset.subset:
            dataset = load_dataset(
                self.cfg.dataset.name,
                self.cfg.dataset.subset,
                split=f"{self.cfg.dataset.split}[:{self.cfg.dataset.num_samples}]",
            )
        else:
            dataset = load_dataset(
                self.cfg.dataset.name,
                split=f"{self.cfg.dataset.split}[:{self.cfg.dataset.num_samples}]",
            )
        return dataset, len(dataset)

    def _setup_run_directory(self) -> str:
        now = datetime.datetime.now()
        run_name = f"run_{now.strftime('%Y%m%d_%H%M%S')}_{self.cfg.name_postfix}"
        run_dir = os.path.join(self.cfg.results_dir, run_name)
        os.makedirs(run_dir, exist_ok=True)

        config_file = os.path.join(run_dir, "config.yaml")
        with open(config_file, "w") as f:
            OmegaConf.save(config=self.cfg, f=f)
        return run_dir

    def run(self):
        """Execute the experiment and collect results."""
        # Apply preprocessing
        preprocessed_data = self.preprocessor.preprocess(self.dataset, self.cfg)
        dataloader = DataLoader(
            preprocessed_data, batch_size=self.cfg.batch_size, shuffle=False
        )

        results = self._collect_results(dataloader)
        self._analyze_and_save_results(results)

        if self.sft_trainers:
            base_trainer, large_trainer = self.sft_trainers
            base_trainer.finalize()
            large_trainer.finalize()

    def _collect_results(self, dataloader: DataLoader) -> Dict:
        collectors = {
            "decisions": [],
            "outputs": [],
            "acceptance_ratios": [],
            "uncerts": [],
            "costs": [],
            "labels": [],
            "predictions": [],
            "acceptance_probs": [],
            "base_uncerts": [],
            "large_uncerts": [],
            "base_prob": [],
            "large_prob": [],
            "all_small_predictions": [],
            "all_large_predictions": [],
            "tau_base": [],
            "tau_large": [],
            "M": [],
            "u": [],
        }

        precomputed_batch = None
        batch_idx = 0
        for batch in tqdm(dataloader):
            if self.cfg.precomputed.enable:
                precomputed_batch = self.precomputed_df.loc[
                    batch_idx : batch_idx + len(batch["answer"]) - 1,
                    [
                        "base_response",
                        "large_response",
                        "base_prob",
                        "large_prob",
                        "base_uncertainty",
                        "large_uncertainty",
                    ],
                ]
                batch_idx += len(batch["answer"])

            batch_decisions, small_predictions, large_predictions = (
                self.decision_system.decide_batch(
                    batch["prompts"],
                    batch["answer"],
                    batch["questions"],
                    precomputed=self.cfg.precomputed.enable,
                    precomputed_batch=precomputed_batch,
                )
            )

            self._process_batch_results(
                batch_decisions,
                small_predictions,
                large_predictions,
                batch,
                collectors,
            )

        return collectors

    def _process_batch_results(
        self,
        batch_decisions,
        small_predictions,
        large_predictions,
        batch,
        collectors,
    ):
        collectors["all_small_predictions"].extend(small_predictions)
        collectors["all_large_predictions"].extend(large_predictions)
        prompts = batch["prompts"]

        for i, decision in enumerate(batch_decisions):
            self._update_collectors(collectors, decision, batch["answer"][i])

            if self.sft_trainers:
                self._update_sft_trainers(
                    decision,
                    prompts[i],
                    large_predictions[i],
                    batch["output"][i][0],
                    collectors["acceptance_ratios"][-1],
                )

    def _update_collectors(self, collectors: Dict, decision: Dict, label: str) -> None:
        """Update result collectors with batch decision data."""
        collectors["decisions"].append(decision["decision"])
        collectors["outputs"].append(decision["response"])
        collectors["acceptance_ratios"].append(decision["acceptance_prob"])
        collectors["uncerts"].append(decision["uncertainty"])
        collectors["base_uncerts"].append(decision["base_uncertainty"])
        collectors["large_uncerts"].append(decision["large_uncertainty"])
        collectors["costs"].append(decision["cost"])
        collectors["labels"].append(label)
        collectors["tau_base"].append(decision["tau_base"])
        collectors["tau_large"].append(decision["tau_large"])
        collectors["M"].append(decision["M"])

        # Handle prediction extraction
        pred_letter = (
            extract_predictions(decision["response"].strip())
            if decision["response"]
            else None
        )
        collectors["predictions"].append(pred_letter)

        collectors["base_prob"].append(decision["base_probs"])
        collectors["large_prob"].append(decision["large_probs"])
        collectors["acceptance_probs"].append(decision["acceptance_prob"])
        collectors["u"].append(decision["u"])

    def _create_dataframe_dict(self, collectors: Dict) -> Dict:
        """Create dictionary for DataFrame creation from collectors."""
        return {
            "decision": collectors["decisions"],
            "output": collectors["outputs"],
            "acceptance_ratio": collectors["acceptance_ratios"],
            "uncertainty": collectors["uncerts"],
            "base_uncertainty": collectors["base_uncerts"],
            "large_uncertainty": collectors["large_uncerts"],
            "label": collectors["labels"],
            "prediction": collectors["predictions"],
            "cost": collectors["costs"],
            "base_response": collectors["all_small_predictions"],
            "base_prediction": [
                extract_predictions(pred)
                for pred in collectors["all_small_predictions"]
            ],
            "large_response": collectors["all_large_predictions"],
            "large_prediction": [
                extract_predictions(pred)
                for pred in collectors["all_large_predictions"]
            ],
            "base_prob": collectors["base_prob"],
            "large_prob": collectors["large_prob"],
            "acceptance_prob": collectors["acceptance_probs"],
            "tau_base": collectors["tau_base"],
            "tau_large": collectors["tau_large"],
            "M": collectors["M"],
            "u": collectors["u"],
        }

    def _analyze_and_save_results(self, results: Dict):
        data = pd.DataFrame(self._create_dataframe_dict(results))
        self._calculate_and_plot_metrics(data)
        data.to_csv(
            os.path.join(self.run_dir, f"results_{self.cfg.name_postfix}.csv"),
            index=False,
        )

    def _calculate_and_plot_metrics(self, data: pd.DataFrame) -> None:
        """Calculate accuracy metrics and generate plots."""
        # Calculate base and large model accuracies
        small_model_correct = [
            label == extract_predictions(pred)
            for label, pred in zip(data["label"], data["base_response"])
        ]
        large_model_correct = [
            label == extract_predictions(pred)
            for label, pred in zip(data["label"], data["large_response"])
        ]

        # Calculate accuracies and errors
        accuracy_base = np.mean(small_model_correct)
        accuracy_base_err = np.std(small_model_correct) / np.sqrt(
            len(small_model_correct)
        )
        accuracy_large = np.mean(large_model_correct)
        accuracy_large_err = np.std(large_model_correct) / np.sqrt(
            len(large_model_correct)
        )

        # Calculate costs
        cost_base = self.dataset_length * self.cfg.base_gen_cost
        cost_large = self.dataset_length * self.cfg.large_gen_cost

        # Calculate dynamic system metrics
        data["correct"] = (data["prediction"] == data["label"]).astype(int)
        dynamic_accuracy = data["correct"].mean()
        dynamic_accuracy_err = data["correct"].std() / np.sqrt(len(data["correct"]))
        dynamic_cost = data["cost"].sum()

        # Calculate Incremental Benefit-Cost (ICB - AutoMix)
        # We will have three delta ICB scores, base-large, base-expert, large-expert
        ibc_base2large = (accuracy_large - accuracy_base) / (cost_large - cost_base)
        ibc_base2model = (dynamic_accuracy - accuracy_base) / (dynamic_cost - cost_base)
        delta_ibc_base2large = (ibc_base2model - ibc_base2large) / ibc_base2large * 100

        ibc_base2lexpert = (1.0 - accuracy_base) / (
            self.cfg.expert_cost * len(data) - cost_base
        )
        delta_ibc_base2lexpert = (
            (ibc_base2model - ibc_base2lexpert) / ibc_base2lexpert * 100
        )

        ibc_large2expert = (1.0 - accuracy_large) / (
            self.cfg.expert_cost * len(data) - cost_large
        )
        ibc_large2model = (dynamic_accuracy - accuracy_large) / (
            dynamic_cost - cost_large
        )
        delta_ibc_large2expert = (
            (ibc_large2model - ibc_large2expert) / ibc_large2expert * 100
        )

        # Generate plots
        plot_accuracy_vs_cost(
            self.run_dir,
            cost_base,
            accuracy_base,
            accuracy_base_err,
            cost_large,
            accuracy_large,
            accuracy_large_err,
            self.cfg.expert_cost,
            len(data),
            dynamic_cost,
            dynamic_accuracy,
            dynamic_accuracy_err,
        )

        plot_decision_distribution(
            self.run_dir, data["decision"], ["Base Model", "Large Model", "Expert"]
        )

        plot_tau_M(
            self.run_dir,
            # data[["tau_base", "tau_large", "M"]],
            data[["M"]],
        )

        # Save metrics
        metrics = {
            "accuracy_base": accuracy_base,
            "accuracy_base_err": accuracy_base_err,
            "accuracy_large": accuracy_large,
            "accuracy_large_err": accuracy_large_err,
            "dynamic_accuracy": dynamic_accuracy,
            "dynamic_accuracy_err": dynamic_accuracy_err,
            "cost_base": cost_base,
            "cost_large": cost_large,
            "dynamic_cost": dynamic_cost,
            "dibc_base2large": delta_ibc_base2large,
            "dibc_base2lexpert": delta_ibc_base2lexpert,
            "dibc_large2expert": delta_ibc_large2expert,
        }

        # Save metrics to file
        pd.DataFrame([metrics]).to_csv(
            os.path.join(self.run_dir, "metrics.csv"), index=False
        )

        self._print_performance_summary(data=data, metrics=metrics)

        # Add decision matrix analysis
        outcomes = self._analyze_decision_outcomes(data)
        self._print_decision_matrix(outcomes)

        # Save outcomes to file
        pd.DataFrame([outcomes]).to_csv(
            os.path.join(self.run_dir, "decision_outcomes.csv"), index=False
        )

    def _print_performance_summary(self, data: pd.DataFrame, metrics: Dict) -> None:
        """Print a formatted table summarizing the performance of different agents."""

        # Calculate expert accuracy if there were any expert decisions
        expert_data = data[data["decision"] == 2]
        if len(expert_data) > 0:
            expert_correct = (expert_data["prediction"] == expert_data["label"]).astype(
                int
            )
            expert_accuracy = expert_correct.mean()
            expert_accuracy_err = expert_correct.std() / np.sqrt(len(expert_correct))
        else:
            expert_accuracy = float("nan")
            expert_accuracy_err = float("nan")

        # Calculate costs per sample for easier comparison
        cost_per_sample_base = metrics["cost_base"] / len(data)
        cost_per_sample_large = metrics["cost_large"] / len(data)
        cost_per_sample_dynamic = metrics["dynamic_cost"] / len(data)
        cost_per_sample_expert = self.cfg.expert_cost  # cost per expert query

        # Create formatted strings for each metric
        rows = [
            ["Model", "Accuracy", "Cost per Sample"],
            ["-----", "--------", "---------------"],
            [
                "Base Model",
                f"{metrics['accuracy_base']:.3f} ± {metrics['accuracy_base_err']:.3f}",
                f"{cost_per_sample_base:.2f}",
            ],
            [
                "Large Model",
                f"{metrics['accuracy_large']:.3f} ± {metrics['accuracy_large_err']:.3f}",
                f"{cost_per_sample_large:.2f}",
            ],
            [
                "Expert",
                (
                    f"{expert_accuracy:.3f} ± {expert_accuracy_err:.3f}"
                    if not np.isnan(expert_accuracy)
                    else "N/A"
                ),
                f"{cost_per_sample_expert:.2f}",
            ],
            [
                "Dynamic System",
                f"{metrics['dynamic_accuracy']:.3f} ± {metrics['dynamic_accuracy_err']:.3f}",
                f"{cost_per_sample_dynamic:.2f}",
            ],
        ]

        # Calculate column widths
        col_widths = [
            max(len(str(row[i])) for row in rows) for i in range(len(rows[0]))
        ]

        # Print the table
        print("\nPerformance Summary:")
        print("===================")

        for row in rows:
            formatted_row = [
                f"{str(item):<{width}}" for item, width in zip(row, col_widths)
            ]
            print("  ".join(formatted_row))

        # Add Incremental Cost-Benefit
        print("\nIncremental Cost-Benefit:")
        print("====================")

        print(f"∆IBC(Base -> Large):\t {metrics['dibc_base2large']:.1f}")
        print(f"∆IBC(Base -> Expert):\t {metrics['dibc_base2lexpert']:.1f}")
        print(f"∆IBC(Large -> Expert):\t {metrics['dibc_large2expert']:.1f}")

        # Add decision distribution information
        print("\nDecision Distribution:")
        print("====================")
        model_counts = data["decision"].value_counts().sort_index()
        total_samples = len(data)

        for decision, count in model_counts.items():
            model_name = ["Base Model", "Large Model", "Expert"][decision]
            percentage = (count / total_samples) * 100
            print(f"{model_name}:\t\t {count} samples ({percentage:.1f}%)")

    def _print_decision_matrix(self, outcomes: Dict[str, int]) -> None:
        """
        Print a formatted decision matrix showing all possible outcomes.
        """
        total = outcomes["SM_Path_Total"] + outcomes["LM_Path_Total"]

        print("\nDecision Outcome Matrix:")
        print("=======================")

        # Small Model Path Analysis
        print(
            f"\nSmall Model Path ({outcomes['SM_Path_Total']} samples, {outcomes['SM_Path_Total']/total*100:.1f}%):"
        )
        s2m_total = outcomes["SM_Necessary_Large"] + outcomes["SM_Unnecessary_Large"]
        print(
            f"├── Large Model Escalation ({s2m_total} samples, {s2m_total/total*100:.1f}%):"
        )
        if s2m_total > 0:
            print(
                f"│   ├── Necessary: {outcomes['SM_Necessary_Large']} ({outcomes['SM_Necessary_Large']/s2m_total*100:.1f}%)"
            )
            print(
                f"│   └── Unnecessary: {outcomes['SM_Unnecessary_Large']} ({outcomes['SM_Unnecessary_Large']/s2m_total*100:.1f}%)"
            )
        sm_total_direct = outcomes["SM_Direct_Correct"] + outcomes["SM_Direct_Wrong"]
        print(
            f"├── Direct Decisions ({sm_total_direct} samples, {sm_total_direct/total*100:.1f}%):"
        )
        if sm_total_direct > 0:
            print(
                f"│   ├── Correct: {outcomes['SM_Direct_Correct']} ({outcomes['SM_Direct_Correct']/sm_total_direct*100:.1f}%)"
            )
            print(
                f"│   └── Wrong: {outcomes['SM_Direct_Wrong']} ({outcomes['SM_Direct_Wrong']/sm_total_direct*100:.1f}%)"
            )
        sm_total_escalated = (
            outcomes["SM_Necessary_Expert"] + outcomes["SM_Unnecessary_Expert"]
        )
        print(
            f"└── Expert Escalations: ({sm_total_escalated} samples, {sm_total_escalated/total*100:.1f}%):"
        )
        if sm_total_escalated > 0:
            print(
                f"    ├── Necessary Escalated: {outcomes['SM_Necessary_Expert']} ({outcomes['SM_Necessary_Expert']/sm_total_escalated*100:.1f}%)"
            )
            print(
                f"    └── Unnecessary Escalations: {outcomes['SM_Unnecessary_Expert']} ({outcomes['SM_Unnecessary_Expert']/sm_total_escalated*100:.1f}%)"
            )

        # Large Model Path Analysis
        print(
            f"\nLarge Model Path ({outcomes['LM_Path_Total']} samples, {outcomes['LM_Path_Total']/total*100:.1f}%):"
        )
        lm_total_direct = outcomes["LM_Direct_Correct"] + outcomes["LM_Direct_Wrong"]
        print(
            f"├── Direct Decisions ({lm_total_direct} samples, {lm_total_direct/outcomes['LM_Path_Total']*100:.1f}%):"
        )
        if lm_total_direct > 0:
            print(
                f"│   ├── Correct: {outcomes['LM_Direct_Correct']} ({outcomes['LM_Direct_Correct']/lm_total_direct*100:.1f}%)"
            )
            print(
                f"│   └── Wrong: {outcomes['LM_Direct_Wrong']} ({outcomes['LM_Direct_Wrong']/lm_total_direct*100:.1f}%)"
            )
        lm_total_escalated = (
            outcomes["LM_Necessary_Expert"] + outcomes["LM_Unnecessary_Expert"]
        )
        print(
            f"└── Expert Escalations: ({lm_total_escalated} samples, {lm_total_escalated/outcomes['LM_Path_Total']*100:.1f}%):"
        )
        if lm_total_escalated > 0:
            print(
                f"    ├── Necessary Escalated: {outcomes['LM_Necessary_Expert']} ({outcomes['LM_Necessary_Expert']/lm_total_escalated*100:.1f}%)"
            )
            print(
                f"    └── Unnecessary Escalations: {outcomes['LM_Unnecessary_Expert']} ({outcomes['LM_Unnecessary_Expert']/lm_total_escalated*100:.1f}%)"
            )

        # Expert Analysis
        print(
            f"\nExpert Decisions ({outcomes['Expert_Total']} samples, {outcomes['Expert_Total']/total*100:.1f}% of total):"
        )
        necessary = outcomes["SM_Necessary_Expert"] + outcomes["LM_Necessary_Expert"]
        unnecessary = (
            outcomes["SM_Unnecessary_Expert"] + outcomes["LM_Unnecessary_Expert"]
        )
        if outcomes["Expert_Total"] > 0:
            print(
                f"├── Necessary: {necessary} ({necessary/outcomes['Expert_Total']*100:.1f}%)"
            )
            print(
                f"└── Unnecessary: {unnecessary} ({unnecessary/outcomes['Expert_Total']*100:.1f}%)"
            )

    def _analyze_decision_outcomes(self, data: pd.DataFrame) -> Dict[str, int]:
        """
        Analyze decision outcomes and classify them into categories.

        The analysis tracks scenarios including:
        - Small Model decision outcomes (confident & correct/wrong, uncertain -> expert)
        - Large Model decision outcomes (confident & correct/wrong, uncertain -> expert)
        - Expert outcomes (from SM path vs LM path, correct/wrong)

        Args:
            data: DataFrame containing decision outcomes with columns:
                'decision', 'prediction', 'label', 'uncertainty',
                'acceptance_prob', etc.

        Returns:
            Dictionary containing counts for each scenario
        """
        outcomes = {
            # Small Model Direct Decisions
            "SM_Direct_Correct": 0,  # Small model correct (no escalation)
            "SM_Direct_Wrong": 0,  # Small model wrong (no escalation)
            "SM_Necessary_Large": 0,
            "SM_Unnecessary_Large": 0,
            "SM_Necessary_Expert": 0,  # Small model wrong & uncertainty was high
            "SM_Unnecessary_Expert": 0,  # Small model would be correct but escalated
            # Large Model Direct Decisions
            "LM_Direct_Correct": 0,  # Large model correct (no escalation)
            "LM_Direct_Wrong": 0,  # Large model wrong (no escalation)
            "LM_Necessary_Expert": 0,  # Large model wrong & uncertainty was high
            "LM_Unnecessary_Expert": 0,  # Large model would be correct but escalated
            # Path Analysis
            "SM_Path_Total": 0,  # Total decisions through small model path
            "LM_Path_Total": 0,  # Total decisions through large model path
            "Expert_Total": 0,  # Total expert consultations
        }

        for _, row in data.iterrows():
            # Extract key values
            decision = row["decision"]
            is_correct = row["prediction"] == row["label"]
            base_would_be_correct = row["base_prediction"] == row["label"]
            large_would_be_correct = row["large_prediction"] == row["label"]

            # Determine which path was taken (based on acceptance_prob)
            took_sm_path = row["u"] < row["acceptance_prob"]

            if took_sm_path:
                outcomes["SM_Path_Total"] += 1

                if decision == 0:  # Used Small Model
                    if is_correct:
                        outcomes["SM_Direct_Correct"] += 1
                    else:
                        outcomes["SM_Direct_Wrong"] += 1

                elif decision == 2:  # Escalated to Expert
                    outcomes["Expert_Total"] += 1
                    if base_would_be_correct:
                        outcomes["SM_Unnecessary_Expert"] += 1
                    else:
                        outcomes["SM_Necessary_Expert"] += 1

            else:  # Large Model Path
                outcomes["LM_Path_Total"] += 1
                if base_would_be_correct:
                    outcomes["SM_Unnecessary_Large"] += 1
                else:
                    outcomes["SM_Necessary_Large"] += 1

                if decision == 1:  # Used Large Model
                    if is_correct:
                        outcomes["LM_Direct_Correct"] += 1
                    else:
                        outcomes["LM_Direct_Wrong"] += 1

                elif decision == 2:  # Escalated to Expert
                    outcomes["Expert_Total"] += 1
                    if large_would_be_correct:
                        outcomes["LM_Unnecessary_Expert"] += 1
                    else:
                        outcomes["LM_Necessary_Expert"] += 1

        return outcomes
