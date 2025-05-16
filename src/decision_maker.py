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
from src.utils import extract_predictions, calculate_costs
import numpy as np


class AIDecisionSystem:
    """A system that combines small and large language models with human expertise and online learning."""

    def __init__(
        self,
        base_model,
        base_tokenizer,
        large_model,
        large_tokenizer,
        base_model_path: str,
        large_model_path: str,
        verification_fn: Callable,
        uncertainty_fn: Callable,
        model_config: ModelConfig,
        cost_config: CostConfig,
        online_config: OnlineConfig,
        use_larger_model: bool,
    ):
        self.base_model_path = base_model_path
        self.large_model_path = large_model_path
        self.config = model_config
        self.costs = cost_config
        self.online_config = online_config
        self.use_larger_model = use_larger_model

        self.initial_phi_base = torch.log(torch.tensor(
            [online_config.initial_phi_base/(1-online_config.initial_phi_base)],
            dtype=torch.float32,
            device=self.config.device,
        ))
        self.initial_xi_base = torch.log(torch.tensor(
            [online_config.initial_xi_base/(1-online_config.initial_xi_base)],
            dtype=torch.float32,
            device=self.config.device,
        ))
        self.initial_xi_large = torch.log(torch.tensor(
            [online_config.initial_xi_large/(1-online_config.initial_xi_large)],
            dtype=torch.float32,
            device=self.config.device,
        ))

        # Online learning parameters
        self.online_enable = online_config.enable
        self.phi_base = torch.nn.Parameter(
            self.initial_phi_base.clone(),
            requires_grad=True,  # Ensure gradients are tracked
        )

        self.xi_base = torch.nn.Parameter(
            self.initial_xi_base.clone(),
            requires_grad=True,  # Ensure gradients are tracked
        )

        self.xi_large = torch.nn.Parameter(
            self.initial_xi_large.clone(),
            requires_grad=True,  # Ensure gradients are tracked
        )
        
        self.xi_base_single = torch.nn.Parameter(
            self.initial_xi_large.clone(),
            requires_grad=True,  # Ensure gradients are tracked
        )

        self.xi_large_single = torch.nn.Parameter(
            self.initial_xi_large.clone(),
            requires_grad=True,  # Ensure gradients are tracked
        )

        self.optimizer = torch.optim.Adam(
            [self.phi_base, self.xi_base, self.xi_large], lr=online_config.lr_tau
        )
        
        self.optimizer_base = torch.optim.Adam(
            [self.xi_base_single], lr=online_config.lr_tau
        )
        
        self.optimizer_large = torch.optim.Adam(
            [self.xi_large_single], lr=online_config.lr_tau
        )
        
        self.temperature = 0.01
        self.lr_tau = online_config.lr_tau
        self.lr_M = online_config.lr_M

        # Replay buffer to store past decisions
        self.replay_buffer = []
        self.batch_size = 10 #self.config.batch_size

        # Store functions
        self.verification_fn = verification_fn
        self.uncertainty_fn = uncertainty_fn
        self.base_model = base_model
        self.base_tokenizer = base_tokenizer
        self.large_model = large_model
        self.large_tokenizer = large_tokenizer

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
        # Calculate generation costs based on token counts
        input_token_counts = [len(tokenizer.encode(prompt)) for prompt in prompts]
        output_token_counts = [
            len(output) - input_count
            for output, input_count in zip(outputs, input_token_counts)
        ]

        costs = [
            calculate_costs(
                model.name_or_path,
                input_token_count,
                output_token_count,
                self.costs.output_input_price_ratio,
            )
            for (input_token_count, output_token_count) in zip(
                input_token_counts, output_token_counts
            )
        ]
        generated_output = [
            tokenizer.decode(output, skip_special_tokens=True)[len(prompt) :]
            for output, prompt in zip(outputs, prompts)
        ]

        return generated_output, costs

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
            base_probs, base_inf_costs = self.verification_fn(
                model=self.base_model,
                tokenizer=self.base_tokenizer,
                prompts=prompts,
                generated_responses=base_answer,
                questions=questions,
                output_input_price_ratio=self.costs.output_input_price_ratio,
                n=self.config.uncertainty_samples,
                mc_dropout=self.config.uncertainty_samples > 1,
            )

            large_probs_for_base, large_inf_costs = self.verification_fn(
                model=self.large_model,
                tokenizer=self.large_tokenizer,
                prompts=prompts,
                questions=questions,
                generated_responses=base_answer,
                output_input_price_ratio=self.costs.output_input_price_ratio,
                n=self.config.uncertainty_samples,
                mc_dropout=self.config.uncertainty_samples > 1,
            )
        else:
            base_probs = torch.Tensor(precomputed_batch["base_prob"].values)
            large_probs_for_base = torch.Tensor(precomputed_batch["large_prob"].values)
            base_inf_costs = precomputed_batch["base_inf_cost"].tolist()
            large_inf_costs = precomputed_batch["base_inf_cost"].tolist()

        if not self.config.precomputed.uncertainty:
            base_uncertainties, base_uncert_costs = self.verification_fn(
                model=self.base_model,
                tokenizer=self.base_tokenizer,
                prompts=prompts,
                questions=questions,
                generated_responses=base_answer,
                output_input_price_ratio=self.costs.output_input_price_ratio,
                n=self.config.uncertainty_samples,
                mc_dropout=self.config.uncertainty_samples > 1,
            )

            large_uncertainties, large_uncert_costs = self.verification_fn(
                model=self.large_model,
                tokenizer=self.large_tokenizer,
                prompts=prompts,
                questions=questions,
                generated_responses=large_answer,
                output_input_price_ratio=self.costs.output_input_price_ratio,
                n=self.config.uncertainty_samples,
                mc_dropout=self.config.uncertainty_samples > 1,
            )

        else:
            base_uncertainties = torch.Tensor(
                precomputed_batch["base_uncertainty"].values
            )
            base_uncert_costs = precomputed_batch["base_uncert_cost"].tolist()
            large_uncertainties = torch.Tensor(
                precomputed_batch["large_uncertainty"].values
            )
            large_uncert_costs = precomputed_batch["large_uncert_cost"].tolist()

        return (
            base_probs,
            large_probs_for_base,
            base_uncertainties,
            large_uncertainties,
            base_inf_costs,
            large_inf_costs,
            base_uncert_costs,
            large_uncert_costs,
        )

    def _make_decision(
        self,
        base_prob: float,
        large_prob: float,
        base_uncert: float,
        large_uncert: float,
        base_response: str,
        large_response: str,
        expert_response: str,
        base_inf_cost: float,
        large_inf_cost: float,
        base_gen_cost: float,
        large_gen_cost: float,# probability of making an error
        base_uncert_cost: float,
        large_uncert_cost: float,
        use_larger_model: bool = True,
        base_sigma = None,
        large_sigma = None,
    ) -> Tuple[int, str, float]:
        """Make a decision for a single prompt."""

        if use_larger_model == "large":
            acceptance_probability = large_prob
            first_inf_cost = large_inf_cost
        elif use_larger_model == "base":
            acceptance_probability = base_prob
            first_inf_cost = base_inf_cost
        elif use_larger_model == "ensemble":
            acceptance_probability = 1/(1+np.exp(-(3.44 * base_prob + 7.39 * large_prob)))
            first_inf_cost = base_inf_cost + large_inf_cost
        else:
            raise NotImplementedError(f"Not implemented {use_larger_model}")

        u = random.random()
        # # Predict Model 1
        # if u < acceptance_probability:
        #     return (
        #         0,
        #         base_response,
        #         base_gen_cost 
        #         + first_inf_cost,
        #         base_uncert,
        #         u,
        #     )

        # # Escalate to Model 2
        # else:  
        #     return (
        #         1,
        #         large_response,
        #         base_gen_cost 
        #         + first_inf_cost 
        #         + large_gen_cost,
        #         large_uncert,
        #         u,
        #     )
        
        
        # Abstain Model 1
        if base_sigma > F.sigmoid(self.xi_base):
            return (
                2,
                f"Expert Reponse: The best answer is {expert_response}",
                base_gen_cost 
                + first_inf_cost,
                base_uncert,
                u,
            )
        
        # Predict Model 1
        elif acceptance_probability > F.sigmoid(self.phi_base):
            return (
                0,
                base_response,
                base_gen_cost 
                + first_inf_cost,
                base_uncert,
                u,
            )

        # Escalate to Model 2
        else:
            # Abstain Model 2
            if large_sigma > F.sigmoid(self.xi_large):
                return (
                    2,
                    f"Expert Reponse: The best answer is {expert_response}",
                    base_gen_cost
                    + large_inf_cost
                    + large_gen_cost
                    + large_uncert_cost,
                    large_uncert,
                    u,
                )
            
            # Predict Model 2
            else:  
                return (
                    1,
                    large_response,
                    base_gen_cost 
                    + first_inf_cost 
                    + large_gen_cost
                    + large_uncert_cost,
                    large_uncert,
                    u,
                )

    def update_parameters(self):
        """Update model parameters using minibatches from the replay buffer."""
        random.shuffle(self.replay_buffer)

        if len(self.replay_buffer) < self.batch_size:
            return  # Not enough data to update


        for i in range(
            len(self.replay_buffer) // self.batch_size - 1
        ):  # Perform multiple gradient steps
            # Sample a random minibatch from the replay buffer
            minibatch = self.replay_buffer[
                i * self.batch_size : i * self.batch_size + self.batch_size
            ]

            total_loss, total_loss_base, total_loss_large = 0.0, 0.0, 0.0
            for decision in minibatch:
                # Extract relevant data for this decision
                large_probs_for_base = torch.tensor(decision["large_probs_for_base_dist"]).to(
                    self.config.device
                )
                large_uncert = torch.tensor(decision["large_uncertainties_dist"]).to(
                    self.config.device
                )
                

                # Compute system risk
                system_risk, system_risk_base , system_risk_large = self._calculate_system_risk(
                    self.phi_base,
                    self.xi_base,
                    self.xi_large,
                    self.xi_base_single,
                    self.xi_large_single,
                    large_probs_for_base,
                    large_uncert,
                    base_gen_cost=decision["base_gen_cost"],
                    large_gen_cost=decision["large_gen_cost"],
                    large_inf_cost=decision["large_inf_cost"],
                    large_uncert_cost=decision["large_uncert_cost"],
                )
                
                system_risk = system_risk.to(self.config.device)
                system_risk_base = system_risk_base.to(self.config.device)
                system_risk_large = system_risk_large.to(self.config.device)

                total_loss += system_risk
                total_loss_base += system_risk_base
                total_loss_large += system_risk_large
                
            # Normalize the loss over the minibatch
            total_loss /= self.batch_size
            total_loss_base /= self.batch_size
            total_loss_large /= self.batch_size
            
            # Perform gradient update
            self.optimizer.zero_grad()
            self.optimizer_base.zero_grad()
            self.optimizer_large.zero_grad()
            
            total_loss.backward()
            total_loss_base.backward()
            total_loss_large.backward()
            
            self.optimizer.step()
            self.optimizer_base.step()
            self.optimizer_large.step()

    
    def _calculate_system_risk2(
        self,
        phi_base,
        xi_base,
        xi_large,
        large_probs_for_base,
        large_uncertainties,
        base_gen_cost,
        large_gen_cost,
        large_inf_cost,
        large_uncert_cost,
    ):
        
        # probability of making an error
        # accepted M1
        assert large_probs_for_base.shape[0] > 1
        assert large_uncertainties.shape[0] > 1
        
        # Use differentiable operations instead of comparisons
        # Replace boolean operations with differentiable alternatives
        sigmoid_xi_base = F.sigmoid(xi_base)
        sigmoid_xi_large = F.sigmoid(xi_large)
        sigmoid_phi_base = F.sigmoid(phi_base)
        
        # Use smooth approximations for the indicator functions
        abstain_M1_probs = torch.sigmoid((sigmoid_xi_base - large_probs_for_base) * 100)  # Steepness factor for sharper transition
        abstain_M1 = abstain_M1_probs.mean()
        
        abstain_M2_probs = torch.sigmoid((sigmoid_xi_large - large_uncertainties) * 100)
        abstain_M2 = abstain_M2_probs.mean()
        
        cost_M1 = base_gen_cost + large_inf_cost
        cost_M2 = large_gen_cost + large_uncert_cost
        
        # Use smooth approximation for correct_M1
        correct_M1_probs = torch.sigmoid((large_probs_for_base - sigmoid_phi_base) * 100)
        correct_M1 = correct_M1_probs.mean()
        
        # Deferred and not abstained - use smooth approximation
        deferred_probs = torch.sigmoid((sigmoid_phi_base - large_probs_for_base) * 100)
        not_abstained_probs = torch.sigmoid((large_probs_for_base - sigmoid_xi_base) * 100)
        proba_deferred_not_abstrained = (deferred_probs * not_abstained_probs).mean()
        
        # Accepted M2
        correct_M2 = 1 - abstain_M2
        correct_proba = correct_M1 + proba_deferred_not_abstrained * correct_M2
        
        # Expected costs
        expected_cost = cost_M1 + proba_deferred_not_abstrained * cost_M2
        
        # Probability of abstention
        abst_proba = abstain_M1 + proba_deferred_not_abstrained * abstain_M2
        
        # Cascade
        # + 
        cascade_system_risk = (1 - correct_proba) + self.costs.cost_lambda * expected_cost + self.costs.abst_lambda * abst_proba
        
        # Only small model risk
        base_system_risk = abstain_M1 + self.costs.cost_lambda * cost_M1 + self.costs.abst_lambda * abstain_M1
        
        # Only large model risk
        large_system_risk = abstain_M2 + self.costs.cost_lambda * cost_M2 + self.costs.abst_lambda * abstain_M2

        # Keep the gradients by not detaching when moving to CPU
        return cascade_system_risk, base_system_risk, large_system_risk
    
    
    def single_model_risk(
        self,
        Phi, 
        xi_raw, 
        c,
        slope=10.0):
        """
        phi         : (N,) calibrated P(correct) for the model
        xi_raw      : scalar tensor, raw threshold parameter in ℝ
        c           : scalar, cost of running the model once
        """
        xi = torch.sigmoid(xi_raw)          # map raw → (0,1)
        g  = torch.sigmoid                  # convenience
        sigma = Phi.std(0)
        m_abst = g(slope * (sigma - xi))
        # m_abst = g(slope * (xi - Phi))      # soft indicator Φ < ξ
        m_pred = 1.0 - m_abst               # Φ ≥ ξ

        P_abst = m_abst.mean()              # probability of abstention
        E_corr = (m_pred * Phi).mean()      # expected correctness
        E_cost = c                          # always pay this cost once

        loss = (1.0 - E_corr)  +  self.costs.cost_lambda * E_cost +  self.costs.abst_lambda * P_abst
        return loss
        
    
    def _calculate_system_risk(
        self,
        phi_base_raw,
        xi_base_raw, 
        xi_large_raw,
        xi_base_single_raw,
        xi_large_single_raw,
        Phi1,                     # (N,) calibrated probs from M1
        Phi2,                     # (N,) calibrated probs from M2
        base_gen_cost,
        large_inf_cost,
        large_gen_cost,
        large_uncert_cost,
    ):
        # --- squash thresholds to (0,1)
        xi_b  = torch.sigmoid(xi_base_raw)
        phi_b = torch.sigmoid(phi_base_raw)
        xi_l  = torch.sigmoid(xi_large_raw)

        k = 10.0     # slope for soft indicators
        g = torch.sigmoid

        # --- soft masks
        #m_abst1  = g(k * (xi_b - Phi1))                           # Φ1 < ξ_
        sigma1 = Phi1.std(0)
        m_abst1 = g(k * (sigma1 - xi_b))
        
        m_defer1 = g(k * (Phi1 - xi_b)) * g(k * (phi_b - Phi1))   # defer
        m_pred1  = g(k * (Phi1 - phi_b))                          # predict M1

        #m_abst2  = m_defer1 * g(k * (xi_l - Phi2))
        sigma2 = Phi2.std(0)
        m_abst2 = m_defer1 * g(k * (sigma2 - xi_l))
        m_pred2  = m_defer1 * (1 - g(k * (xi_l - Phi2)))

        # --- probabilities
        P_abst1 = m_abst1.mean()
        P_defer = m_defer1.mean()
        P_abst2 = m_abst2.mean()

        # --- expected correctness
        E_corr = (m_pred1 * Phi1).mean() + (m_pred2 * Phi2).mean()

        # --- expected cost
        c1 = base_gen_cost + large_inf_cost
        c2 = large_gen_cost + large_uncert_cost
        E_cost = c1 + P_defer * c2

        # --- cascade loss
        loss = (1 - E_corr) + self.costs.cost_lambda * E_cost + self.costs.abst_lambda * (P_abst1 + P_abst2)
            
        # --- Only small model risk
        base_system_risk = self.single_model_risk(Phi1, xi_base_single_raw, c1, slope=k)
        
        # --- Only large model risk
        large_system_risk = self.single_model_risk(Phi2, xi_large_single_raw, c2, slope=k)

        return loss, base_system_risk, large_system_risk


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
        base_calibration_model=None,
        large_on_small_calibration_model=None,
        large_calibration_model=None,
        skip_risk: bool = False
    ) -> Tuple[List[Dict[str, Any]], List[str], List[str]]:
        """Process a batch of prompts and make decisions with optional online learning."""

        if not self.config.precomputed.generation:
            base_outputs, base_gen_costs = self.generate_response(
                self.base_model, self.base_tokenizer, prompts
            )
            large_outputs, large_gen_costs = self.generate_response(
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
            base_gen_costs = precomputed_batch["base_gen_cost"].tolist()
            large_outputs = precomputed_batch["large_response"].astype(str).tolist()
            large_gen_costs = precomputed_batch["large_gen_cost"].tolist()
            base_predictions = precomputed_batch["base_prediction"].astype(str).tolist()
            large_predictions = (
                precomputed_batch["large_prediction"].astype(str).tolist()
            )

        (
            base_probs,
            large_probs_for_base,
            base_uncertainties,
            large_uncertainties,
            base_inf_costs,
            large_inf_costs,
            base_uncert_costs,
            large_uncert_costs,
        ) = self._evaluate_models(
            prompts, questions, base_predictions, large_predictions, precomputed_batch
        )

        # recalibrate if surrogate model is available:
        if base_calibration_model is not None:
            base_probs, base_probs_dist = self._apply_calibration_bayesian(base_probs, base_calibration_model)
            base_uncertainties, base_uncertainties_dist = self._apply_calibration_bayesian(
                base_uncertainties, base_calibration_model
            )
        if large_calibration_model is not None:
            large_uncertainties, large_uncertainties_dist = self._apply_calibration_bayesian(
                large_uncertainties, large_calibration_model
            )

        if large_on_small_calibration_model is not None:
            large_probs_for_base, large_probs_for_base_dist = self._apply_calibration_bayesian(
                large_probs_for_base, large_on_small_calibration_model
            )

        acceptance_prob = large_probs_for_base.to(self.config.device)

        decisions = []

        for i, _ in enumerate(prompts):
            base_response = base_outputs[i].strip()
            large_response = large_outputs[i].strip()

            decision, final_answer, cost, uncertainty, u = self._make_decision(
                base_prob=base_probs[i].item(),
                large_prob=large_probs_for_base[i].item(),
                base_uncert=base_uncertainties[i].item(),
                large_uncert=large_uncertainties[i].item(),
                base_response=base_response,
                large_response=large_response,
                expert_response=expert_responses[i],
                base_gen_cost=base_gen_costs[i],
                large_gen_cost=large_gen_costs[i],
                base_inf_cost=base_inf_costs[i],
                large_inf_cost=large_inf_costs[i],
                base_uncert_cost=base_uncert_costs[i],
                large_uncert_cost=large_uncert_costs[i],
                use_larger_model=self.config.use_larger_model,
                base_sigma = large_probs_for_base_dist[i].std(0) if large_on_small_calibration_model else 1 - large_probs_for_base[i].item(),
                large_sigma = large_uncertainties_dist[i].std(0) if large_calibration_model else 1 - large_uncertainties[i].item(),
            )

            
            if skip_risk:
                cascade_risk_dynamic, base_risk_dynamic, large_risk_dynamic = torch.Tensor([np.nan]), torch.Tensor([np.nan]), torch.Tensor([np.nan])
                cascade_risk_static, base_risk_static, large_risk_static = torch.Tensor([np.nan]), torch.Tensor([np.nan]), torch.Tensor([np.nan])
                
            else:
                cascade_risk_dynamic, base_risk_dynamic, large_risk_dynamic = self._calculate_system_risk(
                    self.phi_base,
                    self.xi_base,
                    self.xi_large,
                    self.xi_base_single,
                    self.xi_large_single,
                    large_probs_for_base_dist[i],
                    large_uncertainties_dist[i],
                    base_gen_cost=base_gen_costs[i],
                    large_gen_cost=large_gen_costs[i],
                    large_inf_cost=large_inf_costs[i],
                    large_uncert_cost=large_uncert_costs[i],
                )

                cascade_risk_static, base_risk_static, large_risk_static = self._calculate_system_risk(
                    self.initial_phi_base,
                    self.initial_xi_base,
                    self.initial_xi_large,
                    self.xi_base_single,
                    self.xi_large_single,
                    large_probs_for_base_dist[i],
                    large_uncertainties_dist[i],
                    base_gen_cost=base_gen_costs[i],
                    large_gen_cost=large_gen_costs[i],
                    large_inf_cost=large_inf_costs[i],
                    large_uncert_cost=large_uncert_costs[i],
                )

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
                    "tau_base": F.sigmoid(self.xi_base).item(),
                    "tau_large": F.sigmoid(self.xi_large).item(),
                    "M": F.sigmoid(self.phi_base).item(),
                    "u": u,
                    "system_risk": cascade_risk_dynamic.item(),
                    "system_risk_base": base_risk_static.item(),
                    "system_risk_large": large_risk_static.item(),
                    "system_risk_expert": 0.0,
                    "system_risk_static": cascade_risk_static.item(),
                    "label": expert_responses[i],
                    "base_gen_cost": base_gen_costs[i],
                    "large_gen_cost": large_gen_costs[i],
                    "base_inf_cost": base_inf_costs[i],
                    "large_inf_cost": large_inf_costs[i],
                    "base_uncert_cost": base_uncert_costs[i],
                    "large_uncert_cost": large_uncert_costs[i],
                    "large_probs_for_base_dist": large_probs_for_base_dist[i] if large_on_small_calibration_model is not None else np.nan,
                    "large_uncertainties_dist": large_uncertainties_dist[i] if large_calibration_model is not None else np.nan,
                }
            )

        # Add only abstained decisions to the replay buffer
        if not skip_risk:
            abstained_decisions = [d for d in decisions if d["decision"] == 2]
            self.replay_buffer.extend(abstained_decisions)
            #self.replay_buffer.extend(decisions)

        # Update model parameters after batch processing
        if self.online_enable:
            self.update_parameters()

        return decisions

    def _apply_calibration(self, confidence_scores, calibration_model):
        """
        Apply calibration model to confidence scores.

        Args:
            confidence_scores: Array of model confidence scores
            calibration_model: Fitted calibration model

        Returns:
            Array of calibrated confidence scores
        """
        import numpy as np

        
        confidence_scores = np.array(confidence_scores.cpu())
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
        
        # Extract model and scaler from the calibration model dictionary
        model = calibration_model["model"]
        scaler = calibration_model["scaler"]
        
        # Apply the same scaling as during training
        confidence_scores_scaled = scaler.transform(confidence_scores_reshaped)

        # Predict probabilities (calibrated scores)
        return torch.Tensor(
            model.predict_proba(confidence_scores_scaled)[:, 1]
        ).to(self.config.device)
    
    def _apply_calibration_mlp(self, confidence_scores, calibration_model):
        """
        Apply MLP calibration model to confidence scores.

        Args:
            confidence_scores: Array of model confidence scores
            calibration_model: Dictionary containing the fitted MLP model and scaler

        Returns:
            Array of calibrated confidence scores
        """
        import numpy as np
        import torch

        # Get the device of the input tensor for later use
        device = (
            confidence_scores.device
            if hasattr(confidence_scores, "device")
            else torch.device("cpu")
        )
        
        # Convert to numpy and handle NaN values
        confidence_scores = np.array(confidence_scores.cpu())
        confidence_scores = np.nan_to_num(confidence_scores, nan=0.0)
        
        # Reshape for sklearn
        confidence_scores_reshaped = confidence_scores.reshape(-1, 1)
        
        # Extract model and scaler from the calibration model dictionary
        model = calibration_model["model"]
        scaler = calibration_model["scaler"]
        
        # Apply the same scaling as during training
        confidence_scores_scaled = scaler.transform(confidence_scores_reshaped)
        
        # Predict probabilities (calibrated scores)
        return torch.Tensor(
            model.predict_proba(confidence_scores_scaled)[:, 1]
        ).to(device)
        
        
    
    
    def _apply_calibration_bayesian(self, confidence_scores, bayesian_model):
        """
        Apply Bayesian calibration model to confidence scores and estimate uncertainty.
        
        Args:
            confidence_scores: Array of model confidence scores
            bayesian_model: Dictionary containing the fitted PyMC model, trace, and data
            n_samples: Number of posterior samples to draw for uncertainty estimation
            
        Returns:
            Dictionary containing:
                - calibrated_scores: Mean of posterior predictive distribution
                - uncertainty: Standard deviation of posterior predictive distribution
        """
        import pymc as pm
    
        
        # Convert to numpy and preprocess
        confidence_scores = np.array(confidence_scores.cpu())
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
        
        X_new = confidence_scores
        
        # Extract model components
        model = bayesian_model["model"]
        trace = bayesian_model["trace"]
        
        # Generate posterior predictive samples
        with model:
            # Get posterior samples for parameters
            pm.set_data({"x": X_new })
            posterior_samples = pm.sample_posterior_predictive(
                trace, 
                var_names=["p"], 
                random_seed=42,
                progressbar=False
            )
            
            predictions = posterior_samples.posterior_predictive.p.values
            predictions = predictions.reshape(-1, X_new.shape[0])

        # Calculate mean and standard deviation across samples
        calibrated_mean = np.mean(predictions, axis=0)
        calibrated_std = np.std(predictions,  axis=0)
        
        # return {
        #     "calibrated_scores": torch.Tensor(calibrated_mean).to(device),
        #     "uncertainty": torch.Tensor(calibrated_std).to(device)
        # }
        
        return torch.Tensor(calibrated_mean).to(self.config.device), torch.Tensor(predictions).to(self.config.device).T #
