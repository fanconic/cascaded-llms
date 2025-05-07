import os
import datetime
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils import plot_accuracy_vs_cost_D1
from src.config import ExperimentConfig
from src.factory import DecisionMakerFactory
from src.preprocessor import get_preprocessor
from src.verification import self_verification, surrogate_token_probs

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizer,
)


class Experiment_first:
    """Handles experiment execution and data collection."""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.run_dir = self._setup_run_directory()
        self.dataset, self.dataset_length = self._load_dataset()
        self.preprocessor = get_preprocessor(cfg.dataset.name)

        self.use_larger_models = [False, True]
        self.number_of_repetition = [1, 5]
        self.verification_funcs = [self_verification, surrogate_token_probs]

        # Initialize models
        if not (
            self.cfg.precomputed.generation
            and self.cfg.precomputed.verification
            and self.cfg.precomputed.uncertainty
        ):
            self.base_model, self.base_tokenizer = self._initialize_model(
                self.cfg.base_model, "Small"
            )
            self.large_model, self.large_tokenizer = self._initialize_model(
                self.cfg.large_model, "Large"
            )
        else:
            print("Generations have been precomputed")
            self.base_model, self.base_tokenizer = None, None
            self.large_model, self.large_tokenizer = None, None

        self.decision_systems = []
        self.experiment_names = []
        for use_larger_model in self.use_larger_models:
            for func in self.verification_funcs:
                for n in self.number_of_repetition:
                    exp_config = ExperimentConfig(
                        base_model=cfg.base_model,
                        large_model=cfg.large_model,
                        verification_fn=func,
                        uncertainty_fn=func,
                        max_input_length=cfg.max_input_length,
                        max_new_tokens=cfg.max_new_tokens,
                        device=cfg.device,
                        precomputed=cfg.precomputed,
                        uncertainty_samples=n,
                        batch_size=cfg.batch_size,
                        use_larger_model=use_larger_model,
                    )

                    decision_system = DecisionMakerFactory.create_decision_maker(
                        self.base_model,
                        self.base_tokenizer,
                        self.large_model,
                        self.large_tokenizer,
                        exp_config,
                        cfg.online,
                        cfg.costs,
                    )
                    self.decision_systems.append(decision_system)
                    self.experiment_names.append(
                        f"{func.__name__}_{n}_{'large' if use_larger_model else 'base'}"
                    )

        if self.cfg.precomputed.enable:
            self.precomputed_dfs = {}
            for experiment_name in self.experiment_names:
                # Check if the path is a CSV file or a directory
                if self.cfg.precomputed.path.endswith(".csv"):
                    # If it's a CSV file, read it for each experiment name
                    precomputed_path = self.cfg.precomputed.path
                    if os.path.exists(precomputed_path):
                        self.precomputed_dfs[experiment_name] = pd.read_csv(
                            precomputed_path
                        )
                    else:
                        raise FileNotFoundError(
                            f"Precomputed results file not found at {precomputed_path}"
                        )
                else:
                    # If it's a directory, use the original log
                    # Extract name_postfix from the last folder in precomputed.path
                    last_folder = os.path.basename(
                        os.path.normpath(self.cfg.precomputed.path)
                    )
                    name_postfix = (
                        "_".join(last_folder.split("_")[3:])
                        if "_" in last_folder
                        else ""
                    )

                    precomputed_path = os.path.join(
                        self.cfg.precomputed.path,
                        f"results_{name_postfix}_{experiment_name}.csv",
                    )
                    if os.path.exists(precomputed_path):
                        self.precomputed_dfs[experiment_name] = pd.read_csv(
                            precomputed_path
                        )
                    else:
                        raise FileNotFoundError(
                            f"Precomputed results file not found for experiment {experiment_name} at {precomputed_path}"
                        )

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

        results = []
        for experiment_name, decision_system in zip(
            self.experiment_names, self.decision_systems
        ):
            print(f"Decision System: {experiment_name}")
            result = self._collect_results(
                dataloader,
                decision_system=decision_system,
                experiment_name=experiment_name,
            )
            results.append((experiment_name, result))

        self._analyze_and_save_results(results)

    def _initialize_model(
        self, model_path: str, model_name: str
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """Initialize a model and its tokenizer."""
        print(f"Loading {model_name} Model: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        model = AutoModelForCausalLM.from_pretrained(model_path).to(self.cfg.device)
        return model, tokenizer

    def _collect_results(
        self, dataloader: DataLoader, decision_system, experiment_name
    ) -> Dict:
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
        calibration_data = []
        base_calibration_model = None
        large_calibration_model = None
        large_on_small_calibration_model = None

        for batch in tqdm(dataloader):
            if self.cfg.precomputed.enable:
                precomputed_batch = self.precomputed_dfs[experiment_name].loc[
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
                        "large_inf_cost",
                    ],
                ]
                batch_idx += len(batch["answer"])

            batch_decisions = decision_system.decide_batch(
                batch["prompts"],
                batch["answer"],
                batch["questions"],
                precomputed_batch=precomputed_batch,
                base_calibration_model=base_calibration_model,
                large_on_small_calibration_model=large_on_small_calibration_model,
                large_calibration_model=large_calibration_model,
            )

            # Data is appended to the collectors
            self._process_batch_results(
                batch_decisions,
                batch,
                collectors,
            )

            # Collect data for calibration
            if (
                base_calibration_model is None
                and large_calibration_model is None
                and large_on_small_calibration_model is None
                and self.cfg.calibrate
            ):
                for i, decision in enumerate(batch_decisions):
                    calibration_data.append(
                        {
                            "base_prob": decision["base_probs"],
                            "large_prob": decision["large_probs"],
                            "large_uncert": decision["large_uncertainty"],
                            "base_correct": decision["base_prediction"]
                            == batch["answer"][i],
                            "large_correct": decision["large_prediction"]
                            == batch["answer"][i],
                        }
                    )

                # After collecting n samples, calibrate the probabilities
                if len(calibration_data) >= self.cfg.calibration_size:
                    import pandas as pd

                    calib_df = pd.DataFrame(calibration_data)

                    # Calibrate base model probabilities
                    base_calibration_model = self._calibrate_confidence_scores(
                        calib_df["base_prob"].values, calib_df["base_correct"].values
                    )

                    # Calibrate large model probabilities
                    large_on_small_calibration_model = (
                        self._calibrate_confidence_scores(
                            calib_df["large_prob"].values,
                            calib_df["base_correct"].values,
                        )
                    )

                    # Calibrate large model probabilities
                    large_calibration_model = self._calibrate_confidence_scores(
                        calib_df["large_uncert"].values,
                        calib_df["large_correct"].values,
                    )

                    print(
                        f"Calibration models created after {self.cfg.calibration_size} samples"
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

    def _analyze_and_save_results(self, results: List[Dict]):
        # Create DataFrame from results and collect all metrics
        all_metrics = []
        experiment_data_list = []

        for experiment_name, result in results:
            data = pd.DataFrame(self._create_dataframe_dict(result))

            # Cut away data that was used for calibration
            data = data.iloc[self.cfg.calibration_size :]

            # Calculate metrics for this experiment
            metrics = self._calculate_metrics(data)
            metrics["experiment_name"] = experiment_name
            all_metrics.append(metrics)
            experiment_data_list.append((experiment_name, metrics))

            # Save individual experiment results
            data.to_csv(
                os.path.join(
                    self.run_dir,
                    f"results_{self.cfg.name_postfix}_{experiment_name}.csv",
                ),
                index=False,
            )

        # Create combined plot
        plot_accuracy_vs_cost_D1(
            self.run_dir,
            experiment_data_list,
            len(data),  # Using length of last experiment's data
        )

        # Save all metrics to a single CSV file
        pd.DataFrame(all_metrics).to_csv(
            os.path.join(self.run_dir, f"metrics_combined.csv"), index=False
        )

    def _calculate_metrics(self, data: pd.DataFrame) -> Dict:
        """Calculate accuracy metrics for a single experiment."""
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

        return {
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
        }

    def _calibrate_confidence_scores(self, confidence_scores, correctness):
        """
        Calibrate confidence scores using Platt scaling.

        Args:
            confidence_scores: Array of model confidence scores
            correctness: Binary array indicating whether predictions were correct

        Returns:
            Fitted sklearn.linear_model.LogisticRegression model
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.isotonic import IsotonicRegression
        import numpy as np

        # Reshape confidence scores for sklearn --> Zelliger & Thomson transformation
        # Handle edge cases for p=0 and p=1
        confidence_scores = np.array(confidence_scores)

        # Replace NaN values with 0
        confidence_scores = np.nan_to_num(confidence_scores, nan=0.0)

        mask_high = confidence_scores >= 0.5
        mask_low = confidence_scores < 0.5

        # Avoid division by zero and log(0)
        confidence_scores = np.clip(confidence_scores, 1e-6, 1 - 1e-6)

        # Apply different transformations based on value
        transformed_scores = np.zeros_like(confidence_scores, dtype=float)
        transformed_scores[mask_high] = np.log(1 / (1 - confidence_scores[mask_high]))
        transformed_scores[mask_low] = np.log(2) - np.log(
            1 / confidence_scores[mask_low]
        )

        confidence_scores = transformed_scores
        confidence_scores_reshaped = np.array(confidence_scores).reshape(-1, 1)

        # Initialize and fit logistic regression model (Platt scaling)
        model = LogisticRegression(C=1.0, solver="lbfgs")
        model.fit(confidence_scores_reshaped, correctness)

        # Initialize and fit isotonic regression model
        # model = IsotonicRegression(out_of_bounds="clip")
        # model.fit(confidence_scores_reshaped, correctness)

        return model
