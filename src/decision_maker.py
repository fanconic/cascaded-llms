"""
Human-AI Decision Making System with Online Learning

This module implements a decision-making system that combines small and large language models
with human expert oversight and online parameter adaptation.
"""

import random
from typing import Callable, List, Dict, Tuple, Any, Optional

import torch
from torch import Tensor
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from src.config import ModelConfig, CostConfig, OnlineConfig


class AIDecisionSystem:
    """A system that combines small and large language models with human expertise and online learning."""

    def __init__(
        self,
        base_model_path: str,
        large_model_path: str,
        verification_fn: Callable[
            [
                PreTrainedModel,
                PreTrainedTokenizer,
                List[str],
                List[str],
                List[str],
                str,
                bool,
                str,
            ],
            Tensor,
        ],
        uncertainty_fn: Callable[
            [PreTrainedModel, PreTrainedTokenizer, List[str], List[str], str, bool],
            Tensor,
        ],
        model_config: ModelConfig,
        cost_config: CostConfig,
        online_config: OnlineConfig,
    ):
        self.config = model_config
        self.costs = cost_config

        # Online learning parameters
        self.online_enable = online_config.enable
        self.tau_base = online_config.initial_uncertainty_threshold_base
        self.tau_large = online_config.initial_uncertainty_threshold_large
        self.M = online_config.initial_M
        self.lr_tau = online_config.lr_tau
        self.lr_M = online_config.lr_M
        self.error_penalty = online_config.error_penalty

        # Store functions
        self.verification_fn = verification_fn
        self.uncertainty_fn = uncertainty_fn

        # Initialize models
        if not self.config.precomputed:
            self.base_model, self.base_tokenizer = self._initialize_model(
                base_model_path, "Small"
            )
            self.large_model, self.large_tokenizer = self._initialize_model(
                large_model_path, "Large"
            )
        else:
            print("Generations have been precomputed")

    def _initialize_model(
        self, model_path: str, model_name: str
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """Initialize a model and its tokenizer."""
        print(f"Loading {model_name} Model: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        model = AutoModelForCausalLM.from_pretrained(model_path).to(self.config.device)
        return model, tokenizer

    def generate_response(
        self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, prompts: List[str]
    ) -> List[str]:
        """Generate responses for given prompts using the specified model."""
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.config.max_input_length,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
            )

        return [
            tokenizer.decode(output, skip_special_tokens=True)[len(prompt) :]
            for output, prompt in zip(outputs, prompts)
        ]

    def _evaluate_models(
        self,
        prompts: List[str],
        questions: List[str],
        base_outputs: List[str],
        large_outputs: List[str],
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Evaluate model outputs and compute probabilities and uncertainties."""
        base_probs = self.verification_fn(
            model=self.base_model,
            tokenizer=self.base_tokenizer,
            prompts=prompts,
            questions=questions,
            generated_responses=base_outputs,
            device=self.config.device,
            normalize_by_length=True,
            prompt_template=self.config.prompt_template,
        )

        large_probs_for_base = self.verification_fn(
            model=self.large_model,
            tokenizer=self.large_tokenizer,
            prompts=prompts,
            questions=questions,
            generated_responses=base_outputs,
            device=self.config.device,
            normalize_by_length=True,
            prompt_template=self.config.prompt_template,
        )

        base_uncertainties = self.uncertainty_fn(
            model=self.base_model,
            tokenizer=self.base_tokenizer,
            prompts=prompts,
            generated_responses=base_outputs,
            device=self.config.device,
            normalize_by_length=True,
        )

        large_uncertainties = self.uncertainty_fn(
            model=self.large_model,
            tokenizer=self.large_tokenizer,
            prompts=prompts,
            generated_responses=large_outputs,
            device=self.config.device,
            normalize_by_length=True,
        )

        return base_probs, large_probs_for_base, base_uncertainties, large_uncertainties

    def _make_decision(
        self,
        acceptance_prob: float,
        base_uncert: float,
        large_uncert: float,
        base_response: str,
        large_response: str,
        expert_response: str,
    ) -> Tuple[int, str, float]:
        """Make a decision for a single prompt."""
        u = random.random()
        if u < acceptance_prob:
            if base_uncert > self.tau_base:
                return (
                    2,
                    expert_response,
                    self.costs.base_gen_cost
                    + self.costs.large_inf_cost
                    + self.costs.expert_cost,
                    base_uncert,
                    u,
                )
            return (
                0,
                base_response,
                self.costs.base_gen_cost + self.costs.large_inf_cost,
                base_uncert,
                u,
            )
        else:
            if large_uncert > self.tau_large:
                return (
                    2,
                    expert_response,
                    self.costs.base_gen_cost
                    + self.costs.large_inf_cost
                    + self.costs.large_gen_cost
                    + self.costs.expert_cost,
                    large_uncert,
                    u,
                )
            return (
                1,
                large_response,
                self.costs.base_gen_cost
                + self.costs.large_inf_cost
                + self.costs.large_gen_cost,
                large_uncert,
                u,
            )

    def _update_parameters(self, decision: int, is_correct: bool) -> None:
        """Update M and tau parameters based on decision outcome."""
        dM = 0.0
        dTau_base = 0.0
        dTau_large = 0.0

        if decision == 0:  # used small model
            if is_correct:
                dM -= 1.0
            else:
                dM += 1.0
        elif decision == 1:  # used large model
            if is_correct:
                dM += 1.0
        #     else:
        #         dTau_large -= 1.0
        # elif decision == 2:  # expert
        #     dTau_base -= 0.1
        #     dTau_large -= 0.1

        self.M = max(1e-3, self.M + self.lr_M * dM)
        self.tau_base = max(0.0, self.tau_base + self.lr_tau * dTau_base)
        self.tau_large = max(0.0, self.tau_large + self.lr_tau * dTau_large)

    def decide_batch(
        self,
        prompts: List[str],
        expert_responses: List[List[str]],
        questions: List[str],
        precomputed=False,
        precomputed_batch=None,
    ) -> Tuple[List[Dict[str, Any]], List[str], List[str]]:
        """Process a batch of prompts and make decisions with optional online learning."""

        if not precomputed and precomputed_batch is None:
            base_outputs = self.generate_response(
                self.base_model, self.base_tokenizer, prompts
            )
            large_outputs = self.generate_response(
                self.large_model, self.large_tokenizer, prompts
            )

            (
                base_probs,
                large_probs_for_base,
                base_uncertainties,
                large_uncertainties,
            ) = self._evaluate_models(prompts, questions, base_outputs, large_outputs)

        else:
            base_outputs = precomputed_batch["base_response"].astype(str).tolist()
            large_outputs = precomputed_batch["large_response"].astype(str).tolist()
            base_probs = torch.Tensor(precomputed_batch["base_prob"].values)
            large_probs_for_base = torch.Tensor(precomputed_batch["large_prob"].values)
            base_uncertainties = torch.Tensor(
                precomputed_batch["base_uncertainty"].values
            )
            large_uncertainties = torch.Tensor(
                precomputed_batch["large_uncertainty"].values
            )

        ratio = large_probs_for_base / (base_probs * self.M)
        acceptance_prob = torch.clip(ratio, min=0.0, max=1.0)

        decisions = []
        base_model_predictions = []
        large_model_predictions = []

        for i, _ in enumerate(prompts):
            base_response = base_outputs[i].strip()
            large_response = large_outputs[i].strip()

            decision, final_answer, cost, uncertainty, u = self._make_decision(
                acceptance_prob[i].item(),
                base_uncertainties[i].item(),
                large_uncertainties[i].item(),
                base_response,
                large_response,
                expert_responses[i],
            )

            is_correct = None
            if self.online_enable:
                is_correct = final_answer.strip() == expert_responses[i].strip()
                self._update_parameters(decision, is_correct)

            decisions.append(
                {
                    "decision": decision,
                    "response": final_answer,
                    "cost": cost,
                    "is_correct": is_correct,
                    "base_response": base_response,
                    "large_response": large_response,
                    "uncertainty": uncertainty,
                    "base_uncertainty": base_uncertainties[i].item(),
                    "large_uncertainty": large_uncertainties[i].item(),
                    "acceptance_prob": acceptance_prob[i].item(),
                    "base_probs": base_probs[i].item(),
                    "large_probs": large_probs_for_base[i].item(),
                    "tau_base": self.tau_base,
                    "tau_large": self.tau_large,
                    "M": self.M,
                    "u": u,
                }
            )

            base_model_predictions.append(base_response)
            large_model_predictions.append(large_response)

        return decisions, base_model_predictions, large_model_predictions
