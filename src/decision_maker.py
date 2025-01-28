"""
Human-AI Decision Making System with Online Learning

This module implements a decision-making system that combines small and large language models
with human expert oversight and online parameter adaptation.
"""

import random
from typing import Callable, List, Dict, Tuple, Any, Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from src.config import ModelConfig, CostConfig, OnlineConfig
from src.utils import extract_predictions
from scipy.optimize import minimize, differential_evolution
import numpy as np


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
        self.online_config = online_config

        self.initial_M = torch.tensor(
            [np.log(np.exp(online_config.initial_M) - 1)],
            dtype=torch.float32,
            device=self.config.device,
        )
        self.initial_tau_base = torch.tensor(
            [np.log(np.exp(online_config.initial_uncertainty_threshold_base) - 1)],
            dtype=torch.float32,
            device=self.config.device,
        )
        self.initial_tau_large = torch.tensor(
            [np.log(np.exp(online_config.initial_uncertainty_threshold_large) - 1)],
            dtype=torch.float32,
            device=self.config.device,
        )

        # Online learning parameters
        self.online_enable = online_config.enable
        self.M = torch.nn.Parameter(
            self.initial_M.clone(),
            requires_grad=True,  # Ensure gradients are tracked
        )

        self.tau_base = torch.nn.Parameter(
            self.initial_tau_base.clone(),
            requires_grad=True,  # Ensure gradients are tracked
        )

        self.tau_large = torch.nn.Parameter(
            self.initial_tau_large.clone(),
            requires_grad=False,  # Ensure gradients are tracked
        )

        self.optimizer = torch.optim.Adam([self.M], lr=online_config.lr_tau)

        # self.tau_base = torch.Tensor([online_config.initial_uncertainty_threshold_base]).to(self.config.device)
        # self.tau_large = torch.Tensor([online_config.initial_uncertainty_threshold_large]).to(self.config.device)
        # self.M = online_config.initial_M
        self.lr_tau = online_config.lr_tau
        self.lr_M = online_config.lr_M
        self.error_penalty = online_config.error_penalty

        # Replay buffer to store past decisions
        self.replay_buffer = []
        self.batch_size = 7  # self.config.batch_size

        # Store functions
        self.verification_fn = verification_fn
        self.uncertainty_fn = uncertainty_fn

        # Initialize models
        if not (
            self.config.precomputed.generation
            and self.config.precomputed.verification
            and self.config.precomputed.uncertainty
        ):
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
                do_sample=True,
                temperature=0.3,
                top_p=0.9,
            )

        return [
            tokenizer.decode(output, skip_special_tokens=True)[len(prompt) :]
            for output, prompt in zip(outputs, prompts)
        ]

    def _evaluate_models(
        self,
        prompts: List[str],
        questions: List[str],
        base_answer: List[str],
        large_answer: List[str],
        precomputed_batch=None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Evaluate model outputs and compute probabilities and uncertainties."""

        if not self.config.precomputed.verification:
            base_probs = self.verification_fn(
                model=self.base_model,
                tokenizer=self.base_tokenizer,
                prompts=prompts,
                questions=questions,
                generated_responses=base_answer,
                device=self.config.device,
                normalize_by_length=True,
                prompt_template=self.config.prompt_template,
            )

            large_probs_for_base = self.verification_fn(
                model=self.large_model,
                tokenizer=self.large_tokenizer,
                prompts=prompts,
                questions=questions,
                generated_responses=base_answer,
                device=self.config.device,
                normalize_by_length=True,
                prompt_template=self.config.prompt_template,
            )
        else:
            base_probs = torch.Tensor(precomputed_batch["base_prob"].values)
            large_probs_for_base = torch.Tensor(precomputed_batch["large_prob"].values)

        if not self.config.precomputed.uncertainty:
            base_uncertainties = self.uncertainty_fn(
                model=self.base_model,
                tokenizer=self.base_tokenizer,
                prompts=questions,
                generated_responses=base_answer,
                device=self.config.device,
                normalize_by_length=True,
                max_length=self.config.max_input_length,
                n=self.config.uncertainty_samples,
            )

            large_uncertainties = self.uncertainty_fn(
                model=self.large_model,
                tokenizer=self.large_tokenizer,
                prompts=questions,
                generated_responses=large_answer,
                device=self.config.device,
                normalize_by_length=True,
                max_length=self.config.max_input_length,
                n=self.config.uncertainty_samples,
            )
        else:
            base_uncertainties = torch.Tensor(
                precomputed_batch["base_uncertainty"].values
            )
            large_uncertainties = torch.Tensor(
                precomputed_batch["large_uncertainty"].values
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
            if base_uncert > F.softplus(self.tau_base):
                return (
                    2,
                    f"Expert Reponse: The best answer is {expert_response}",
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
            if large_uncert > F.softplus(self.tau_large):
                return (
                    2,
                    f"Expert Reponse: The best answer is {expert_response}",
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

    def update_parameters(self):
        """Update model parameters using minibatches from the replay buffer."""
        if len(self.replay_buffer) < self.batch_size:
            return  # Not enough data to update

        # self.M.data = torch.Tensor([np.log(np.exp(self.online_config.initial_uncertainty_threshold_large) - 1)]).to(self.config.device)
        # self.tau_base.data = torch.Tensor([np.log(np.exp(self.online_config.initial_uncertainty_threshold_large) - 1)]).to(self.config.device)
        # self.tau_large.data = torch.Tensor([np.log(np.exp(self.online_config.initial_uncertainty_threshold_large) - 1)]).to(self.config.device)

        for i in range(
            len(self.replay_buffer) // self.batch_size - 1
        ):  # Perform multiple gradient steps
            # Sample a random minibatch from the replay buffer
            minibatch = self.replay_buffer[
                i * self.batch_size : i * self.batch_size + self.batch_size
            ]

            total_loss = 0.0
            for decision in minibatch:
                # Extract relevant data for this decision
                base_probs = torch.tensor(decision["base_probs"]).to(self.config.device)
                large_probs_for_base = torch.tensor(decision["large_probs"]).to(
                    self.config.device
                )
                base_uncert = torch.tensor(decision["base_uncertainty"])
                large_uncert = torch.tensor(decision["large_uncertainty"])
                label = decision["label"]

                # Compute system risk
                system_risk = self._calculate_system_risk(
                    self.M,
                    self.tau_base,
                    self.tau_large,
                    large_probs_for_base,
                    base_probs,
                    base_uncert,
                    large_uncert,
                    decision["base_prediction"],
                    decision["large_prediction"],
                    label,
                ).to(self.config.device)

                total_loss += system_risk

            # Normalize the loss over the minibatch
            total_loss /= self.batch_size

            # Perform gradient update
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

    def _calculate_system_risk(
        self,
        M,
        tau_base,
        tau_large,
        large_probs_for_base,
        base_probs,
        base_uncertainties,
        large_uncertainties,
        base_predictions,
        large_predictions,
        expert_responses,
    ):
        # Calculate the system risk in this step
        # MvM
        ratio = large_probs_for_base / (base_probs * F.softplus(M))

        acceptance_prob = torch.clip(ratio, min=0.0, max=1.0)

        phi_mvm_0 = acceptance_prob
        phi_mvm_1 = 1 - acceptance_prob

        # MvH
        phi_mvh_0_base = torch.sigmoid(F.softplus(tau_base) - base_uncertainties)
        phi_mvh_2_base = 1 - phi_mvh_0_base
        phi_mvh_0_large = torch.sigmoid(F.softplus(tau_large) - large_uncertainties)
        phi_mvh_2_large = 1 - phi_mvh_0_large

        # phi_mvh_0_base = base_uncertainties < F.softplus(tau_base)
        # phi_mvh_2_base = base_uncertainties >= F.softplus(tau_base)
        # phi_mvh_0_large = large_uncertainties < F.softplus(tau_large)
        # phi_mvh_2_large = large_uncertainties >= F.softplus(tau_large)

        # 01 loss
        l_y_base = self.costs.mistake_cost * (base_predictions != expert_responses)
        l_y_large = self.costs.mistake_cost * (large_predictions != expert_responses)

        # Cumulative sums per distinct action
        c_0 = self.costs.base_gen_cost + self.costs.large_inf_cost
        c_1 = (
            self.costs.base_gen_cost
            + self.costs.large_inf_cost
            + self.costs.large_gen_cost
        )
        c_2 = (
            self.costs.base_gen_cost
            + self.costs.large_inf_cost
            + self.costs.expert_cost
        )
        c_3 = (
            self.costs.base_gen_cost
            + self.costs.large_inf_cost
            + self.costs.large_gen_cost
            + self.costs.expert_cost
        )

        # Condensed policies
        Phi_0 = phi_mvm_0 * phi_mvh_0_base
        Phi_1 = phi_mvm_1 * phi_mvh_0_large
        Phi_2 = phi_mvm_0 * phi_mvh_2_base
        Phi_3 = phi_mvm_1 * phi_mvh_2_large

        loss_part = Phi_0 * l_y_base + Phi_1 * l_y_large
        cost_part = Phi_0 * c_0 + Phi_1 * c_1 + Phi_2 * c_2 + Phi_3 * c_3
        system_risk = loss_part + cost_part

        return system_risk.to("cpu")

    def _compute_log_prob(self, acceptance_prob, base_uncert, large_uncert, decision):
        """Compute the log-probability of the chosen action."""
        if decision == 0:  # Base model chosen
            log_prob = torch.log(acceptance_prob) + torch.log(
                torch.sigmoid(self.tau_base - base_uncert)
            )
        elif decision == 1:  # Large model chosen
            log_prob = torch.log(1 - acceptance_prob) + torch.log(
                torch.sigmoid(self.tau_large - large_uncert)
            )
        elif decision == 2:  # Expert chosen
            if base_uncert > self.tau_base:
                log_prob = torch.log(acceptance_prob) + torch.log(
                    1 - torch.sigmoid(self.tau_base - base_uncert)
                )
            else:
                log_prob = torch.log(1 - acceptance_prob) + torch.log(
                    1 - torch.sigmoid(self.tau_large - large_uncert)
                )
        return log_prob

    def decide_batch(
        self,
        prompts: List[str],
        expert_responses: List[List[str]],
        questions: List[str],
        precomputed_batch=None,
    ) -> Tuple[List[Dict[str, Any]], List[str], List[str]]:
        """Process a batch of prompts and make decisions with optional online learning."""

        if not self.config.precomputed.generation:
            base_outputs = self.generate_response(
                self.base_model, self.base_tokenizer, prompts
            )
            large_outputs = self.generate_response(
                self.large_model, self.large_tokenizer, prompts
            )

            base_predictions = [
                extract_predictions(base_output) for base_output in base_outputs
            ]
            large_predictions = [
                extract_predictions(large_output) for large_output in large_outputs
            ]

        else:
            base_outputs = precomputed_batch["base_response"].astype(str).tolist()
            large_outputs = precomputed_batch["large_response"].astype(str).tolist()
            base_predictions = precomputed_batch["base_prediction"].astype(str).tolist()
            large_predictions = (
                precomputed_batch["large_prediction"].astype(str).tolist()
            )

        (
            base_probs,
            large_probs_for_base,
            base_uncertainties,
            large_uncertainties,
        ) = self._evaluate_models(
            prompts, questions, base_predictions, large_predictions, precomputed_batch
        )

        ratio = large_probs_for_base.to(self.config.device) / (
            base_probs.to(self.config.device) * F.softplus(self.M)
        )
        acceptance_prob = torch.clip(ratio, min=0.0, max=1.0)

        decisions = []

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

            system_risk = self._calculate_system_risk(
                self.M,
                self.tau_base,
                self.tau_large,
                large_probs_for_base[i],
                base_probs[i],
                base_uncertainties[i],
                large_uncertainties[i],
                base_predictions[i],
                large_predictions[i],
                expert_responses[i],
            )

            system_risk_static = self._calculate_system_risk(
                self.initial_M,
                self.initial_tau_base,
                self.initial_tau_base,
                large_probs_for_base[i],
                base_probs[i],
                base_uncertainties[i],
                large_uncertainties[i],
                base_predictions[i],
                large_predictions[i],
                expert_responses[i],
            )

            # 01 loss
            l_y_base = self.costs.mistake_cost * (
                base_predictions[i] != expert_responses[i]
            )
            l_y_large = self.costs.mistake_cost * (
                large_predictions[i] != expert_responses[i]
            )

            # Other system risk strategies
            system_risk_base = l_y_base + self.costs.base_gen_cost
            system_risk_large = l_y_large + self.costs.large_gen_cost
            system_risk_expert = self.costs.expert_cost

            decisions.append(
                {
                    "question": questions[i],
                    "decision": decision,
                    "response": final_answer,
                    "prediction": extract_predictions(final_answer),
                    "cost": cost,
                    "base_response": base_response,
                    "large_response": large_response,
                    "base_prediction": base_predictions[i],
                    "large_prediction": large_predictions[i],
                    "uncertainty": uncertainty,
                    "base_uncertainty": base_uncertainties[i].item(),
                    "large_uncertainty": large_uncertainties[i].item(),
                    "acceptance_ratios": acceptance_prob[i].item(),
                    "base_probs": base_probs[i].item(),
                    "large_probs": large_probs_for_base[i].item(),
                    "tau_base": F.softplus(self.tau_base).item(),
                    "tau_large": F.softplus(self.tau_large).item(),
                    "M": F.softplus(self.M).item(),
                    "u": u,
                    "system_risk": system_risk.item(),
                    "system_risk_base": system_risk_base,
                    "system_risk_large": system_risk_large,
                    "system_risk_expert": system_risk_expert,
                    "system_risk_static": system_risk_static,
                    "label": expert_responses[i],
                }
            )

        # Add decisions to the replay buffer
        self.replay_buffer.extend(decisions)

        # Update model parameters after batch processing
        if self.online_enable:
            self.update_parameters()

        return decisions
