import os
import datetime
from typing import Dict

import numpy as np
import pandas as pd
from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils import (
    plot_accuracy_vs_cost,
    plot_decision_distribution,
    plot_tau_M,
    plot_accuracy_vs_cost_D1,
    plot_risk_and_cumulative_risk,
)
from src.config import ExperimentConfig
from src.factory import DecisionMakerFactory
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
            precomputed=cfg.precomputed,
            uncertainty_samples=cfg.uncertainty_samples,
            batch_size=cfg.batch_size,
        )

        self.decision_system = DecisionMakerFactory.create_decision_maker(
            exp_config, cfg.online, cfg.costs
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
            "questions": [],
            "decisions": [],
            "responses": [],
            "base_responses": [],
            "large_responses": [],
            "predictions": [],
            "base_predictions": [],
            "large_predictions": [],
            "labels": [],
            # MvM defferal
            "u": [],
            "M": [],
            "acceptance_ratios": [],
            "base_prob": [],
            "large_prob": [],
            # MvH defferal
            "uncerts": [],
            "base_uncerts": [],
            "large_uncerts": [],
            "tau_base": [],
            "tau_large": [],
            # Costs
            "costs": [],
            "base_gen_cost": [],
            "large_gen_cost": [],
            "base_inf_cost": [],
            "large_inf_cost": [],
            "base_uncert_cost": [],
            "large_uncert_cost": [],
            # System risks
            "system_risk": [],
            "system_risk_base": [],
            "system_risk_large": [],
            "system_risk_expert": [],
            "system_risk_static": [],
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
                        "base_prediction",
                        "large_prediction",
                        "base_gen_cost",
                        "large_gen_cost",
                        "base_uncert_cost",
                        "large_uncert_cost",
                        "base_inf_cost",
                        "large_inf_cost"
                    ],
                ]
                batch_idx += len(batch["answer"])

            batch_decisions = self.decision_system.decide_batch(
                batch["prompts"],
                batch["answer"],
                batch["questions"],
                precomputed_batch=precomputed_batch,
            )

            self._process_batch_results(
                batch_decisions,
                batch,
                collectors,
            )

        return collectors

    def _process_batch_results(
        self,
        batch_decisions,
        batch,
        collectors,
    ):
        for i, decision in enumerate(batch_decisions):
            self._update_collectors(collectors, decision, batch["answer"][i])

    def _update_collectors(self, collectors: Dict, decision: Dict, label: str) -> None:
        """Update result collectors with batch decision data."""
        collectors["questions"].append(decision["question"])
        collectors["decisions"].append(decision["decision"])
        collectors["responses"].append(decision["response"])
        collectors["base_responses"].append(decision["base_response"])
        collectors["large_responses"].append(decision["large_response"])
        collectors["predictions"].append(decision["prediction"])
        collectors["base_predictions"].append(decision["base_prediction"])
        collectors["large_predictions"].append(decision["large_prediction"])
        collectors["labels"].append(label)

        # MvM defferal
        collectors["u"].append(decision["u"])
        collectors["M"].append(decision["M"])
        collectors["acceptance_ratios"].append(decision["acceptance_ratios"])
        collectors["base_prob"].append(decision["base_probs"])
        collectors["large_prob"].append(decision["large_probs"])

        # MvH defferal
        collectors["uncerts"].append(decision["uncertainty"])
        collectors["base_uncerts"].append(decision["base_uncertainty"])
        collectors["large_uncerts"].append(decision["large_uncertainty"])
        collectors["tau_base"].append(decision["tau_base"])
        collectors["tau_large"].append(decision["tau_large"])

        # Costs
        collectors["costs"].append(decision["cost"])
        collectors["base_gen_cost"].append(decision["base_gen_cost"])
        collectors["large_gen_cost"].append(decision["large_gen_cost"])
        collectors["base_inf_cost"].append(decision["base_inf_cost"])
        collectors["large_inf_cost"].append(decision["large_inf_cost"])
        collectors["base_uncert_cost"].append(decision["base_uncert_cost"])
        collectors["large_uncert_cost"].append(decision["large_uncert_cost"])

        # System Risk
        collectors["system_risk"].append(decision["system_risk"])
        collectors["system_risk_base"].append(decision["system_risk_base"])
        collectors["system_risk_large"].append(decision["system_risk_large"])
        collectors["system_risk_expert"].append(decision["system_risk_expert"])
        collectors["system_risk_static"].append(decision["system_risk_static"])

    def _create_dataframe_dict(self, collectors: Dict) -> Dict:
        """Create dictionary for DataFrame creation from collectors."""
        return {
            "question": collectors["questions"],
            "decision": collectors["decisions"],
            "output": collectors["responses"],
            "prediction": collectors["predictions"],
            "label": collectors["labels"],
            "base_response": collectors["base_responses"],
            "base_prediction": collectors["base_predictions"],
            "large_response": collectors["large_responses"],
            "large_prediction": collectors["large_predictions"],
            "acceptance_ratios": collectors["acceptance_ratios"],
            "M": collectors["M"],
            "u": collectors["u"],
            "base_prob": collectors["base_prob"],
            "large_prob": collectors["large_prob"],
            "uncertainty": collectors["uncerts"],
            "base_uncertainty": collectors["base_uncerts"],
            "large_uncertainty": collectors["large_uncerts"],
            "tau_base": collectors["tau_base"],
            "tau_large": collectors["tau_large"],
            "cost": collectors["costs"],
            "base_gen_cost": collectors["base_gen_cost"],
            "large_gen_cost": collectors["large_gen_cost"],
            "base_inf_cost": collectors["base_inf_cost"],
            "large_inf_cost": collectors["large_inf_cost"],
            "base_uncert_cost": collectors["base_uncert_cost"],
            "large_uncert_cost": collectors["large_uncert_cost"],
            "system_risk": collectors["system_risk"],
            "system_risk_base": collectors["system_risk_base"],
            "system_risk_large": collectors["system_risk_large"],
            "system_risk_expert": collectors["system_risk_expert"],
            "system_risk_static": collectors["system_risk_static"],
        }

    def _analyze_and_save_results(self, results: Dict):
        # Create DataFrame from results
        data = pd.DataFrame(self._create_dataframe_dict(results))

        # # Calibrate confidence scores using a small subset of the data
        # calibration_size = min(200, len(data))  # Use at most 100 samples for calibration
        # calibration_subset = data.sample(n=calibration_size, random_state=42)

        # # Calibrate base model probabilities
        # base_calibration = self._calibrate_confidence_scores(
        #     calibration_subset["base_prob"].values,
        #     (calibration_subset["base_prediction"] == calibration_subset["label"]).values
        # )

        # # Calibrate large model probabilities
        # large_calibration = self._calibrate_confidence_scores(
        #     calibration_subset["large_prob"].values,
        #     (calibration_subset["large_prediction"] == calibration_subset["label"]).values
        # )

        # # Apply calibration to the full dataset
        # data["base_prob"] = self._apply_calibration(data["base_prob"].values, base_calibration)
        # data["large_prob"] = self._apply_calibration(data["large_prob"].values, large_calibration)

        # data["acceptance_ratios"] = data["large_prob"] / (data["base_prob"] * data["M"])
        # data["decision"] = data.apply(lambda x: 0 if x["acceptance_ratios"] > x["u"] else 1, axis=1)
        # data["cost"] = data.apply(lambda x: self.cfg.costs.base_gen_cost + self.cfg.costs.large_inf_cost if x["acceptance_ratios"] > x["u"] else self.cfg.costs.base_gen_cost + self.cfg.costs.large_inf_cost + self.cfg.costs.large_gen_cost, axis=1)

        self._calculate_and_plot_metrics(data)
        data.to_csv(
            os.path.join(self.run_dir, f"results_{self.cfg.name_postfix}.csv"),
            index=False,
        )

    def _calibrate_confidence_scores(self, confidence_scores, correctness):
        """
        Calibrate confidence scores using isotonic regression.

        Args:
            confidence_scores: Array of model confidence scores
            correctness: Binary array indicating whether predictions were correct

        Returns:
            Fitted sklearn.isotonic.IsotonicRegression model
        """
        from sklearn.isotonic import IsotonicRegression

        # Initialize and fit isotonic regression model
        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(confidence_scores, correctness)

        return ir

    def _apply_calibration(self, confidence_scores, calibration_model):
        """
        Apply calibration model to confidence scores.

        Args:
            confidence_scores: Array of model confidence scores
            calibration_model: Fitted calibration model

        Returns:
            Array of calibrated confidence scores
        """
        return calibration_model.predict(confidence_scores)

    def _calculate_and_plot_metrics(self, data: pd.DataFrame) -> None:
        """Calculate accuracy metrics and generate plots."""
        # Calculate base and large model accuracies
        small_model_correct = [
            label == pred for label, pred in zip(data["label"], data["base_prediction"])
        ]
        large_model_correct = [
            label == pred
            for label, pred in zip(data["label"], data["large_prediction"])
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
        cost_base = data["base_gen_cost"].sum()
        cost_large = data["large_gen_cost"].sum()

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
            self.cfg.costs.expert_cost * len(data) - cost_base
        )
        delta_ibc_base2lexpert = (
            (ibc_base2model - ibc_base2lexpert) / ibc_base2lexpert * 100
        )

        ibc_large2expert = (1.0 - accuracy_large) / (
            self.cfg.costs.expert_cost * len(data) - cost_large
        )
        ibc_large2model = (dynamic_accuracy - accuracy_large) / (
            dynamic_cost - cost_large
        )
        delta_ibc_large2expert = (
            (ibc_large2model - ibc_large2expert) / ibc_large2expert * 100
        )

        # System risk:
        system_risk = data["system_risk"].mean()
        system_risk_base = data["system_risk_base"].mean()
        system_risk_large = data["system_risk_large"].mean()
        system_risk_expert = data["system_risk_expert"].mean()
        system_risk_static = data["system_risk_static"].mean()

        # Generate plots
        plot_accuracy_vs_cost(
            self.run_dir,
            cost_base,
            accuracy_base,
            accuracy_base_err,
            cost_large,
            accuracy_large,
            accuracy_large_err,
            self.cfg.costs.expert_cost,
            len(data),
            dynamic_cost,
            dynamic_accuracy,
            dynamic_accuracy_err,
        )

        plot_accuracy_vs_cost_D1(
            self.run_dir,
            cost_base,
            accuracy_base,
            accuracy_base_err,
            cost_large,
            accuracy_large,
            accuracy_large_err,
            len(data),
            dynamic_cost,
            dynamic_accuracy,
            dynamic_accuracy_err,
            delta_ibc_base2large,
        )

        plot_decision_distribution(
            self.run_dir, data["decision"], ["Base Model", "Large Model", "Expert"]
        )

        plot_tau_M(
            self.run_dir,
            data[["tau_base", "tau_large", "M"]],
        )

        plot_risk_and_cumulative_risk(
            self.run_dir,
            data[
                [
                    "system_risk",
                    "system_risk_base",
                    "system_risk_large",
                    "system_risk_static",
                    "system_risk_expert",
                    "M",
                ]
            ],
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
            "dibc_base2expert": delta_ibc_base2lexpert,
            "dibc_large2expert": delta_ibc_large2expert,
            "system_risk": system_risk,
            "system_risk_base": system_risk_base,
            "system_risk_large": system_risk_large,
            "system_risk_expert": system_risk_expert,
            "system_risk_static": system_risk_static,
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
        """Print a formatted table summarizing the performance of different agents,
        plus deferral metrics for the Model-vs-Model (MvM) step.
        """

        # 1. Run the outcomes analysis to retrieve counts
        outcomes = self._analyze_decision_outcomes(data)

        # 2. Existing code to compute expert accuracy, cost per sample, etc.
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

        cost_per_sample_base = metrics["cost_base"] / len(data)
        cost_per_sample_large = metrics["cost_large"] / len(data)
        cost_per_sample_dynamic = metrics["dynamic_cost"] / len(data)
        cost_per_sample_expert = self.cfg.costs.expert_cost

        system_risk = metrics["system_risk"]
        system_risk_base = metrics["system_risk_base"]
        system_risk_large = metrics["system_risk_large"]
        system_risk_expert = metrics["system_risk_expert"]
        system_risk_static = metrics["system_risk_static"]

        # 3. Print your original performance table as before
        rows = [
            ["Model", "Accuracy", "Cost per Sample", "System Risk"],
            ["-----", "--------", "---------------", "-----------"],
            [
                "Base Model",
                f"{metrics['accuracy_base']:.3f} ± {metrics['accuracy_base_err']:.3f}",
                f"{cost_per_sample_base:.2f}",
                f"{system_risk_base:.2f}",
            ],
            [
                "Large Model",
                f"{metrics['accuracy_large']:.3f} ± {metrics['accuracy_large_err']:.3f}",
                f"{cost_per_sample_large:.2f}",
                f"{system_risk_large:.2f}",
            ],
            [
                "Expert",
                (
                    f"{expert_accuracy:.3f} ± {expert_accuracy_err:.3f}"
                    if not np.isnan(expert_accuracy)
                    else "N/A"
                ),
                f"{cost_per_sample_expert:.2f}",
                f"{system_risk_expert:.2f}",
            ],
            [
                "Static System",
                f"N/A",
                f"N/A",
                f"{system_risk_static:.2f}",
            ],
            [
                "Dynamic System",
                f"{metrics['dynamic_accuracy']:.3f} ± {metrics['dynamic_accuracy_err']:.3f}",
                f"{cost_per_sample_dynamic:.2f}",
                f"{system_risk:.2f}",
            ],
        ]

        col_widths = [
            max(len(str(row[i])) for row in rows) for i in range(len(rows[0]))
        ]

        print("\nPerformance Summary:")
        print("===================")
        for row in rows:
            formatted_row = [
                f"{str(item):<{width}}" for item, width in zip(row, col_widths)
            ]
            print("  ".join(formatted_row))

        print("\nIncremental Cost-Benefit:")
        print("====================")
        print(f"∆IBC(Base -> Large):\t {metrics['dibc_base2large']:.2f}")
        print(f"∆IBC(Base -> Expert):\t {metrics['dibc_base2expert']:.2f}")
        print(f"∆IBC(Large -> Expert):\t {metrics['dibc_large2expert']:.2f}")

        print("\nDecision Distribution:")
        print("====================")
        model_counts = data["decision"].value_counts().sort_index()
        total_samples = len(data)
        for decision, count in model_counts.items():
            model_name = ["Base Model", "Large Model", "Expert"][decision]
            percentage = (count / total_samples) * 100
            print(f"{model_name}:\t\t {count} samples ({percentage:.1f}%)")

        # 4. Compute deferral metrics for the MvM step
        self._print_mvm_deferral_metrics(outcomes)

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
        if outcomes["LM_Path_Total"] > 0:
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
        if outcomes["LM_Path_Total"] > 0:
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
                'acceptance_ratios', etc.

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
            "LM_corrected_wrong_pred": 0,
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

            # Determine which path was taken (based on acceptance_ratios)
            took_sm_path = row["u"] < row["acceptance_ratios"]

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
                    if is_correct:
                        outcomes["LM_corrected_wrong_pred"] += 1

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

    def _print_mvm_deferral_metrics(self, outcomes: Dict[str, int]) -> None:
        """
        Calculate and print:
        1) Deferral rate
        2) Unnecessary deferral rate
        3) Error recovery rate
        along with binomial standard errors.
        """

        print("\nMvM Deferral Metrics:")
        print("=====================")

        # -- 1) DEFERRAL RATE
        # fraction of times we escalated to large among base-model candidates
        total = outcomes["SM_Path_Total"] + outcomes["LM_Path_Total"]
        deferrals = outcomes["LM_Path_Total"]
        if total > 0:
            deferral_rate = deferrals / total
            # Binomial standard error:
            deferral_rate_err = np.sqrt(deferral_rate * (1 - deferral_rate) / total)
        else:
            deferral_rate = float("nan")
            deferral_rate_err = float("nan")

        print(
            f"Deferral Rate (Base->Large):\t {deferral_rate*100:.1f} $\pm$ {deferral_rate_err*100:.1f}"
        )

        # -- 2) UNNECESSARY DEFERRAL RATE
        # among times base model was correct and used, fraction that got escalated
        all_deferrals = (
            outcomes["SM_Necessary_Large"] + outcomes["SM_Unnecessary_Large"]
        )
        # (We ignore SM_Unnecessary_Expert if we only care about deferrals to large.)
        if all_deferrals > 0:
            unnecessary_deferral_rate = outcomes["SM_Unnecessary_Large"] / all_deferrals
            un_rate_err = np.sqrt(
                unnecessary_deferral_rate
                * (1 - unnecessary_deferral_rate)
                / all_deferrals
            )
        else:
            unnecessary_deferral_rate = float("nan")
            un_rate_err = float("nan")

        print(
            f"Unnecessary Deferral Rate:\t {unnecessary_deferral_rate*100:.1f} $\pm$ {un_rate_err*100:.1f}"
        )

        # -- 3) ERROR RECOVERY RATE
        # among times base model was wrong, fraction that got escalated
        if all_deferrals > 0:
            error_recovery_rate = outcomes["LM_corrected_wrong_pred"] / all_deferrals
            er_rate_err = np.sqrt(
                error_recovery_rate * (1 - error_recovery_rate) / all_deferrals
            )
        else:
            error_recovery_rate = float("nan")
            er_rate_err = float("nan")

        print(
            f"Error Recovery Rate:\t\t {error_recovery_rate*100:.1f} $\pm$ {er_rate_err*100:.1f}"
        )
