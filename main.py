import os
import datetime
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf

from src.decision_maker import AIDecisionSystem
from src.uncertainty import per_token_entropy
from src.verification import verbalisation, sequence_probability, surrogate_token_probs
from src.online_sft import OnlineSFTTrainerLoRA
from src.utils import (
    extract_predictions,
    plot_accuracy_vs_cost,
    plot_decision_distribution,
)

# Function mapping
VERIFICATION_FN_MAPPING = {
    "surrogate_token_probs": surrogate_token_probs,
    "sequence_probability": sequence_probability,
    "verbalisation": verbalisation,
}

UNCERTAINTY_FN_MAPPING = {
    "per_token_entropy": per_token_entropy,
}


def set_seed(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    # Set seed for reproducibility
    set_seed(cfg.seed)

    # Create a unique run directory
    now = datetime.datetime.now()
    run_name = f"run_{now.strftime('%Y%m%d_%H%M%S')}_{cfg.name_postfix}"
    run_dir = os.path.join(cfg.results_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    # Save configuration file with the run
    config_file = os.path.join(run_dir, "config.yaml")
    with open(config_file, "w") as f:
        OmegaConf.save(config=cfg, f=f)

    # Load dataset
    dataset = load_dataset(cfg.dataset, split=f"train[:{cfg.num_samples}]")
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False)

    # Initialize decision system
    decision_system = AIDecisionSystem(
        base_model_path=cfg.base_model,
        large_model_path=cfg.large_model,
        uncertainty_threshold=cfg.uncertainty_threshold,
        verification_fn=VERIFICATION_FN_MAPPING[cfg.verification_fn],
        uncertainty_fn=UNCERTAINTY_FN_MAPPING[cfg.uncertainty_fn],
        max_input_lenght=cfg.max_input_length,
        max_new_tokens=cfg.max_new_tokens,
        base_gen_cost=cfg.base_gen_cost,
        large_gen_cost=cfg.large_gen_cost,
        large_inf_cost=cfg.large_inf_cost,
        expert_cost=cfg.expert_cost,
        device=cfg.device,
        prompt_template=cfg.prompt_template,
    )

    # Initialize the SFT trainer (example parameters)
    if cfg.sft.enable:
        sft_trainer_base = OnlineSFTTrainerLoRA(
            base_model=decision_system.base_model,  # The underlying base model from AIDecisionSystem
            learning_rate=cfg.sft.learning_rate,
            max_buffer_size=cfg.sft.buffer_size,
            tokenizer_name=decision_system.base_tokenizer,  # or a specific tokenizer path if needed
            lora_r=cfg.sft.lora_r,
            lora_alpha=cfg.sft.lora_alpha,
            lora_dropout=cfg.sft.lora_dropout,
            output_dir=os.path.join(run_dir, "base_lora_adapters"),  # save location
        )

        sft_trainer_large = OnlineSFTTrainerLoRA(
            base_model=decision_system.large_model,  # The underlying base model from AIDecisionSystem
            learning_rate=cfg.sft.learning_rate,
            max_buffer_size=cfg.sft.buffer_size,
            tokenizer_name=decision_system.large_tokenizer,  # or a specific tokenizer path if needed
            lora_r=cfg.sft.lora_r,
            lora_alpha=cfg.sft.lora_alpha,
            lora_dropout=cfg.sft.lora_dropout,
            output_dir=os.path.join(run_dir, "large_lora_adapters"),  # save location
        )

    # Initialize result placeholders
    (
        decisions,
        outputs,
        acceptance_ratios,
        uncerts,
        costs,
        labels,
        predictions,
        acceptance_probs,
        base_uncerts,
        large_uncerts,
    ) = ([], [], [], [], [], [], [], [], [], [])
    base_prob, large_prob = [], []
    all_small_predictions, all_large_predictions = [], []

    for batch in tqdm(dataloader):
        prompts = [
            f"{cfg.prompt_template}\n{sample}\nThe best answer is: "
            for sample in batch["input"]
        ]
        questions = [f"{sample}" for sample in batch["input"]]
        batch_decisions, small_predictions, large_predictions = (
            decision_system.decide_batch(prompts, batch["output"], questions)
        )

        all_small_predictions.extend(small_predictions)
        all_large_predictions.extend(large_predictions)

        for i, decision in enumerate(batch_decisions):
            decisions.append(decision["decision"])
            outputs.append(decision["response"])
            acceptance_ratios.append(decision["acceptance_ratio"])
            uncerts.append(decision["uncertainty"])
            base_uncerts.append(decision["base_uncertainty"])
            large_uncerts.append(decision["large_uncertainty"])
            costs.append(decision["cost"])
            labels.append(batch["output"][i][0])

            # Evaluate final predicted letter
            if decision["response"] is not None:
                pred_letter = extract_predictions(decision["response"].strip())
            else:
                pred_letter = None
            predictions.append(pred_letter)

            base_prob.append(decision["base_probs"])
            large_prob.append(decision["large_probs"])
            acceptance_probs.append(decision["acceptance_prob"])

            if cfg.sft.enable:
                # 1. If final decision used the large model response, add that as a pseudo-label for the base model
                if decision["decision"] == 1:
                    sft_trainer_base.add_example(
                        prompt_text=prompts[i],
                        label_text=large_predictions[
                            i
                        ],  # pseudo-label from large model
                        pseudo_label_weight=acceptance_ratios[i],
                    )

                # 2. If final decision used the expert path, store the ground-truth label for the base and large model
                elif decision["decision"] == 2:
                    sft_trainer_base.add_example(
                        prompt_text=prompts[i],
                        label_text=batch["output"][i][0],  # ground-truth
                    )

                    sft_trainer_large.add_example(
                        prompt_text=prompts[i],
                        label_text=batch["output"][i][0],  # ground-truth
                    )

    # After the loop, optionally finalize the SFT to use remaining buffer
    if cfg.sft.enable:
        sft_trainer_base.finalize()
        sft_trainer_large.finalize()

    # Create DataFrame
    data = pd.DataFrame(
        {
            "decision": decisions,
            "output": outputs,
            "acceptance_ratio": acceptance_ratios,
            "uncertainty": uncerts,
            "base_uncertainty": base_uncerts,
            "large_uncertainty": large_uncerts,
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
            "base_prob": base_prob,
            "large_prob": large_prob,
            "acceptance_prob": acceptance_probs,
        }
    )

    # Evaluate performance
    small_model_correct = [
        label == extract_predictions(pred)
        for label, pred in zip(labels, all_small_predictions)
    ]
    large_model_correct = [
        label == extract_predictions(pred)
        for label, pred in zip(labels, all_large_predictions)
    ]

    accuracy_base = np.mean(small_model_correct)
    accuracy_base_err = np.std(small_model_correct) / np.sqrt(len(small_model_correct))
    accuracy_large = np.mean(large_model_correct)
    accuracy_large_err = np.std(large_model_correct) / np.sqrt(len(large_model_correct))

    cost_base = len(dataset) * cfg.base_gen_cost
    cost_large = len(dataset) * cfg.large_gen_cost

    data["correct"] = (data["prediction"] == data["label"]).astype(int)
    dynamic_accuracy = data["correct"].mean()
    dynamic_accuracy_err = data["correct"].std() / np.sqrt(len(data["correct"]))
    dynamic_cost = data["cost"].sum()

    # Plot Accuracy vs Cost
    plot_accuracy_vs_cost(
        run_dir,
        cost_base,
        accuracy_base,
        accuracy_base_err,
        cost_large,
        accuracy_large,
        accuracy_large_err,
        cfg.expert_cost,
        len(data),
        dynamic_cost,
        dynamic_accuracy,
        dynamic_accuracy_err,
    )

    # Plot Decision Distribution
    plot_decision_distribution(
        run_dir, data["decision"], ["Base Model", "Large Model", "Expert"]
    )

    # Save data
    data.to_csv(os.path.join(run_dir, f"results_{cfg.name_postfix}.csv"), index=False)


if __name__ == "__main__":
    main()
