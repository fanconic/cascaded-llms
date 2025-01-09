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


class Experiment:
    """Handles experiment execution and data collection."""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.run_dir = self._setup_run_directory()
        self.dataset, self.dataset_length = self._load_dataset()

        exp_config = ExperimentConfig(
            base_model=cfg.base_model,
            large_model=cfg.large_model,
            verification_fn=cfg.verification_fn,
            uncertainty_fn=cfg.uncertainty_fn,
            max_input_length=cfg.max_input_length,
            max_new_tokens=cfg.max_new_tokens,
            device=cfg.device,
            prompt_template=cfg.prompt_template,
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

    def _setup_run_directory(self) -> str:
        now = datetime.datetime.now()
        run_name = f"run_{now.strftime('%Y%m%d_%H%M%S')}_{self.cfg.name_postfix}"
        run_dir = os.path.join(self.cfg.results_dir, run_name)
        os.makedirs(run_dir, exist_ok=True)

        config_file = os.path.join(run_dir, "config.yaml")
        with open(config_file, "w") as f:
            OmegaConf.save(config=self.cfg, f=f)
        return run_dir

    def _load_dataset(self) -> Tuple[DataLoader, int]:
        dataset = load_dataset(
            self.cfg.dataset, split=f"train[:{self.cfg.num_samples}]"
        )
        return DataLoader(dataset, batch_size=self.cfg.batch_size, shuffle=False), len(
            dataset
        )

    def _initialize_sft(self) -> Tuple[OnlineSFTTrainerLoRA, OnlineSFTTrainerLoRA]:
        base_trainer = OnlineSFTTrainerLoRA(
            base_model=self.decision_system.base_model,
            learning_rate=self.cfg.sft.learning_rate,
            max_buffer_size=self.cfg.sft.buffer_size,
            tokenizer_name=self.decision_system.base_tokenizer,
            lora_r=self.cfg.sft.lora_r,
            lora_alpha=self.cfg.sft.lora_alpha,
            lora_dropout=self.cfg.sft.lora_dropout,
            output_dir=os.path.join(self.run_dir, "base_lora_adapters"),
        )

        large_trainer = OnlineSFTTrainerLoRA(
            base_model=self.decision_system.large_model,
            learning_rate=self.cfg.sft.learning_rate,
            max_buffer_size=self.cfg.sft.buffer_size,
            tokenizer_name=self.decision_system.large_tokenizer,
            lora_r=self.cfg.sft.lora_r,
            lora_alpha=self.cfg.sft.lora_alpha,
            lora_dropout=self.cfg.sft.lora_dropout,
            output_dir=os.path.join(self.run_dir, "large_lora_adapters"),
        )

        return base_trainer, large_trainer

    def run(self):
        """Execute the experiment and collect results."""
        results = self._collect_results()
        self._analyze_and_save_results(results)

        if self.sft_trainers:
            base_trainer, large_trainer = self.sft_trainers
            base_trainer.finalize()
            large_trainer.finalize()

    def _collect_results(self) -> Dict:
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
        }

        for batch in tqdm(self.dataset):
            prompts = [
                f"{self.cfg.prompt_template}\n{sample}\nThe best answer is: "
                for sample in batch["input"]
            ]
            questions = [f"{sample}" for sample in batch["input"]]

            batch_decisions, small_predictions, large_predictions = (
                self.decision_system.decide_batch(prompts, batch["output"], questions)
            )

            self._process_batch_results(
                batch_decisions,
                small_predictions,
                large_predictions,
                batch,
                prompts,
                collectors,
            )

        return collectors

    def _process_batch_results(
        self,
        batch_decisions,
        small_predictions,
        large_predictions,
        batch,
        prompts,
        collectors,
    ):
        collectors["all_small_predictions"].extend(small_predictions)
        collectors["all_large_predictions"].extend(large_predictions)

        for i, decision in enumerate(batch_decisions):
            self._update_collectors(collectors, decision, batch["output"][i][0])

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
            for label, pred in zip(data['label'], data['base_response'])
        ]
        large_model_correct = [
            label == extract_predictions(pred)
            for label, pred in zip(data['label'], data['large_response'])
        ]

        # Calculate accuracies and errors
        accuracy_base = np.mean(small_model_correct)
        accuracy_base_err = np.std(small_model_correct) / np.sqrt(len(small_model_correct))
        accuracy_large = np.mean(large_model_correct)
        accuracy_large_err = np.std(large_model_correct) / np.sqrt(len(large_model_correct))

        # Calculate costs
        cost_base = self.dataset_length * self.cfg.base_gen_cost
        cost_large = self.dataset_length * self.cfg.large_gen_cost

        # Calculate dynamic system metrics
        data['correct'] = (data['prediction'] == data['label']).astype(int)
        dynamic_accuracy = data['correct'].mean()
        dynamic_accuracy_err = data['correct'].std() / np.sqrt(len(data['correct']))
        dynamic_cost = data['cost'].sum()

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
            self.run_dir,
            data['decision'],
            ['Base Model', 'Large Model', 'Expert']
        )

        # Save metrics
        metrics = {
            'accuracy_base': accuracy_base,
            'accuracy_base_err': accuracy_base_err,
            'accuracy_large': accuracy_large,
            'accuracy_large_err': accuracy_large_err,
            'dynamic_accuracy': dynamic_accuracy,
            'dynamic_accuracy_err': dynamic_accuracy_err,
            'cost_base': cost_base,
            'cost_large': cost_large,
            'dynamic_cost': dynamic_cost,
        }
        
        # Save metrics to file
        pd.DataFrame([metrics]).to_csv(
            os.path.join(self.run_dir, 'metrics.csv'), 
            index=False
        )
        
        self._print_performance_summary(data=data, metrics=metrics)
        
    def _print_performance_summary(self, data: pd.DataFrame, metrics: Dict) -> None:
        """Print a formatted table summarizing the performance of different agents."""
        
        # Calculate expert accuracy if there were any expert decisions
        expert_data = data[data['decision'] == 2]
        if len(expert_data) > 0:
            expert_correct = (expert_data['prediction'] == expert_data['label']).astype(int)
            expert_accuracy = expert_correct.mean()
            expert_accuracy_err = expert_correct.std() / np.sqrt(len(expert_correct))
        else:
            expert_accuracy = float('nan')
            expert_accuracy_err = float('nan')

        # Calculate costs per sample for easier comparison
        cost_per_sample_base = metrics['cost_base'] / len(data)
        cost_per_sample_large = metrics['cost_large'] / len(data)
        cost_per_sample_dynamic = metrics['dynamic_cost'] / len(data)
        cost_per_sample_expert = self.cfg.expert_cost  # cost per expert query

        # Create formatted strings for each metric
        rows = [
            ['Model', 'Accuracy', 'Cost per Sample'],
            ['-----', '--------', '---------------'],
            [
                'Base Model',
                f"{metrics['accuracy_base']:.3f} ± {metrics['accuracy_base_err']:.3f}",
                f"{cost_per_sample_base:.2f}"
            ],
            [
                'Large Model',
                f"{metrics['accuracy_large']:.3f} ± {metrics['accuracy_large_err']:.3f}",
                f"{cost_per_sample_large:.2f}"
            ],
            [
                'Expert',
                f"{expert_accuracy:.3f} ± {expert_accuracy_err:.3f}" if not np.isnan(expert_accuracy) else "N/A",
                f"{cost_per_sample_expert:.2f}"
            ],
            [
                'Dynamic System',
                f"{metrics['dynamic_accuracy']:.3f} ± {metrics['dynamic_accuracy_err']:.3f}",
                f"{cost_per_sample_dynamic:.2f}"
            ]
        ]

        # Calculate column widths
        col_widths = [
            max(len(str(row[i])) for row in rows)
            for i in range(len(rows[0]))
        ]

        # Print the table
        print("\nPerformance Summary:")
        print("===================")
        
        for row in rows:
            formatted_row = [
                f"{str(item):<{width}}"
                for item, width in zip(row, col_widths)
            ]
            print("  ".join(formatted_row))

        # Add decision distribution information
        print("\nDecision Distribution:")
        print("====================")
        model_counts = data['decision'].value_counts().sort_index()
        total_samples = len(data)
        
        for decision, count in model_counts.items():
            model_name = ['Base Model', 'Large Model', 'Expert'][decision]
            percentage = (count / total_samples) * 100
            print(f"{model_name}: {count} samples ({percentage:.1f}%)")
