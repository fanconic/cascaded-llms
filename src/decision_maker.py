"""
This script implements the Human-AI decision making system

Author: CLaudio + GPTo1
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import random


class AIDecisionSystem:
    def __init__(
        self,
        base_model_path,
        large_model_path,
        uncertainty_threshold,
        verification_fn,
        uncertainty_fn,
        max_input_lenght,
        max_new_tokens,
        base_gen_cost,
        large_gen_cost,
        large_inf_cost,
        expert_cost,
    ):

        self.base_model_path = base_model_path
        self.large_model_path = large_model_path
        self.uncertainty_threshold = uncertainty_threshold
        self.verification_fn = verification_fn
        self.uncertainty_fn = uncertainty_fn
        self.max_input_lenght = max_input_lenght
        self.max_new_tokens = max_new_tokens
        self.base_gen_cost = base_gen_cost
        self.large_gen_cost = large_gen_cost
        self.large_inf_cost = large_inf_cost
        self.expert_cost = expert_cost

        # Initialise models and tokenizers
        print("Loading Small Model")
        self.base_tokenizer = AutoTokenizer.from_pretrained(self.base_model_path)
        self.base_tokenizer.pad_token = self.base_tokenizer.eos_token
        self.base_tokenizer.padding_side = "left"
        self.base_model = AutoModelForCausalLM.from_pretrained(self.base_model_path).to(
            "cuda:0"
        )

        print("Loading Large Model")
        self.large_tokenizer = AutoTokenizer.from_pretrained(self.large_model_path)
        self.large_tokenizer.pad_token = self.large_tokenizer.eos_token
        self.large_tokenizer.padding_side = "left"
        self.large_model = AutoModelForCausalLM.from_pretrained(
            self.large_model_path
        ).to("cuda:1")

    def generate_response(self, model, tokenizer, prompts):
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_input_lenght,
        ).to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
            )
        responses = [
            tokenizer.decode(output, skip_special_tokens=True) for output in outputs
        ]
        return responses, inputs

    def decide_batch(self, prompts):
        """
        Probabilistic gating:
          1) If small-model entropy > threshold => expert (decision=2).
          2) Else accept the small model with probability alpha = ratio / (1 + ratio).
             If not accepted => check large model's entropy.
               - If large-model entropy > threshold => expert (2)
               - else => large model (1)
        """
        # Generate from small model & large model
        base_outputs, _ = self.generate_response(
            self.base_model, self.base_tokenizer, prompts
        )
        large_outputs, _ = self.generate_response(
            self.large_model, self.large_tokenizer, prompts
        )

        # 1. Compute log probs for the small model & the large model verifying the small model's text
        base_log_probs = self.verification_fn(
            model=self.base_model,
            tokenizer=self.base_tokenizer,
            prompts=prompts,
            generated_responses=base_outputs,
            device="cuda:2",
            normalize_by_length=True,
        )
        large_log_probs_for_base = self.verification_fn(
            model=self.large_model,
            tokenizer=self.large_tokenizer,
            prompts=prompts,
            generated_responses=base_outputs,
            device="cuda:2",
            normalize_by_length=True,
        )

        # 2. Compute ratio & acceptance probability
        difference = large_log_probs_for_base - base_log_probs
        ratio = torch.exp(difference)  # ratio = p_large / p_small
        # We define acceptance_prob = ratio / (1 + ratio)
        acceptance_prob = ratio / (1.0 + ratio)

        # 3. Compute entropies to gauge uncertainty
        base_uncertainties = self.uncertainty_fn(
            model=self.base_model,
            tokenizer=self.base_tokenizer,
            prompts=prompts,
            generated_responses=base_outputs,
            device="cuda:2",
            normalize_by_length=True,
        )
        large_uncertainties = self.uncertainty_fn(
            model=self.large_model,
            tokenizer=self.large_tokenizer,
            prompts=prompts,
            generated_responses=large_outputs,  # The large model's own output
            device="cuda:2",
            normalize_by_length=True,
        )

        decisions = []
        small_model_predictions = []
        large_model_predictions = []

        for i, prompt in enumerate(prompts):
            base_response = base_outputs[i]
            large_response = large_outputs[i]
            base_answer = base_response[len(prompt) :].strip()
            large_answer = large_response[len(prompt) :].strip()

            # We'll read off the ratio-based acceptance probability
            alpha_i = acceptance_prob[i].item()

            # Uncertainties
            small_uncert = base_uncertainties[i].item()
            big_uncert = large_uncertainties[i].item()

            small_model_predictions.append(base_answer)
            large_model_predictions.append(large_answer)

            # Probabilistic acceptance
            u = random.random()  # uniform(0,1)
            if u < alpha_i:
                # If small model is too uncertain => send to expert
                if small_uncert > self.uncertainty_threshold:
                    decisions.append(
                        {
                            "decision": 2,  # Expert
                            "response": None,
                            "base_response": base_response,
                            "large_response": large_response,
                            "base_log_probs": base_log_probs[i].item(),
                            "large_log_probs": large_log_probs_for_base[i].item(),
                            "prob_delta": ratio[i].item(),
                            "uncertainty": small_uncert,
                            "cost": self.base_gen_cost
                            + self.large_inf_cost
                            + self.expert_cost,
                        }
                    )

                else:
                    # Accept small model
                    decisions.append(
                        {
                            "decision": 0,  # small model
                            "response": base_response,
                            "base_response": base_response,
                            "large_response": large_response,
                            "base_log_probs": base_log_probs[i].item(),
                            "large_log_probs": large_log_probs_for_base[i].item(),
                            "prob_delta": ratio[i].item(),
                            "uncertainty": small_uncert,
                            "cost": self.base_gen_cost + self.large_inf_cost,
                        }
                    )
            else:
                # If we use the large model, check if it's also too uncertain
                if big_uncert > self.uncertainty_threshold:
                    decisions.append(
                        {
                            "decision": 2,  # Expert
                            "response": None,
                            "base_response": base_response,
                            "large_response": large_response,
                            "base_log_probs": base_log_probs[i].item(),
                            "large_log_probs": large_log_probs_for_base[i].item(),
                            "prob_delta": ratio[i].item(),
                            "uncertainty": big_uncert,
                            "cost": self.base_gen_cost
                            + self.large_inf_cost
                            + self.large_gen_cost
                            + self.expert_cost,
                        }
                    )
                else:
                    # Accept large model
                    decisions.append(
                        {
                            "decision": 1,  # large model
                            "response": large_response,
                            "base_response": base_response,
                            "large_response": large_response,
                            "base_log_probs": base_log_probs[i].item(),
                            "large_log_probs": large_log_probs_for_base[i].item(),
                            "prob_delta": ratio[i].item(),
                            "uncertainty": big_uncert,
                            "cost": self.base_gen_cost
                            + self.large_inf_cost
                            + self.large_gen_cost,
                        }
                    )

        return decisions, small_model_predictions, large_model_predictions
