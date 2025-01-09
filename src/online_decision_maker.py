import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import random
from typing import Callable


class OnlineAIDecisionSystem:
    def __init__(
        self,
        base_model_path: str,
        large_model_path: str,
        verification_fn: Callable,
        uncertainty_fn: Callable,
        max_input_lenght: int = 512,
        max_new_tokens: int = 256,
        base_gen_cost: float = 1.0,
        large_gen_cost: float = 25.0,
        large_inf_cost: float = 1.0,
        expert_cost: float = 100.0,
        device: str = "cpu",
        prompt_template: str = "",
        # New parameters:
        initial_uncertainty_threshold_base: float = 5.0,
        initial_uncertainty_threshold_large: float = 5.0,
        initial_M: float = 1.0,
        lr_tau: float = 0.01,
        lr_M: float = 0.01,
        error_penalty: float = 10.0,  # e.g. penalty if final answer is wrong
    ):
        """
        Args:
            ...
            initial_uncertainty_threshold: initial value for tau.
            initial_M: initial value for M.
            lr_tau: 'learning rate' for updating tau.
            lr_M: 'learning rate' for updating M.
            error_penalty: how heavily you penalize an incorrect final answer
                           (used in the heuristic cost-based updates).
        """
        self.base_model_path = base_model_path
        self.large_model_path = large_model_path
        self.verification_fn = verification_fn
        self.uncertainty_fn = uncertainty_fn
        self.max_input_lenght = max_input_lenght
        self.max_new_tokens = max_new_tokens
        self.base_gen_cost = base_gen_cost
        self.large_gen_cost = large_gen_cost
        self.large_inf_cost = large_inf_cost
        self.expert_cost = expert_cost
        self.device = device
        self.prompt_template = prompt_template

        # Online-learned parameters:
        self.tau_base = initial_uncertainty_threshold_base
        self.tau_large = initial_uncertainty_threshold_large
        self.M = initial_M
        self.lr_tau = lr_tau
        self.lr_M = lr_M
        self.error_penalty = error_penalty

        print("Loading Small Model")
        self.base_tokenizer = AutoTokenizer.from_pretrained(self.base_model_path)
        if self.base_tokenizer.pad_token is None:
            self.base_tokenizer.pad_token = self.base_tokenizer.eos_token
        self.base_tokenizer.padding_side = "left"
        self.base_model = AutoModelForCausalLM.from_pretrained(self.base_model_path).to(
            self.device
        )

        print("Loading Large Model")
        self.large_tokenizer = AutoTokenizer.from_pretrained(self.large_model_path)
        if self.large_tokenizer.pad_token is None:
            self.large_tokenizer.pad_token = self.large_tokenizer.eos_token
        self.large_tokenizer.padding_side = "left"
        self.large_model = AutoModelForCausalLM.from_pretrained(
            self.large_model_path
        ).to(self.device)

    def generate_response(self, model, tokenizer, prompts):
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding="max_length",
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
            tokenizer.decode(output, skip_special_tokens=True)[len(prompt) :]
            for output, prompt in zip(outputs, prompts)
        ]
        return responses

    def decide_batch(self, prompts, expert_responses, questions, gold_labels=None):
        """
        Decide how to answer each prompt.
        Optionally pass in gold_labels to check correctness for each sample,
        enabling an online update to (self.M, self.tau).
        """
        # Generate from small model & large model
        base_outputs = self.generate_response(
            self.base_model, self.base_tokenizer, prompts
        )
        large_outputs = self.generate_response(
            self.large_model, self.large_tokenizer, prompts
        )

        # (1) Probability ratio
        base_probs = self.verification_fn(
            model=self.base_model,
            tokenizer=self.base_tokenizer,
            prompts=prompts,
            questions=questions,
            generated_responses=base_outputs,
            device=self.device,
            normalize_by_length=True,
            prompt_template=self.prompt_template,
        )
        large_probs_for_base = self.verification_fn(
            model=self.large_model,
            tokenizer=self.large_tokenizer,
            prompts=prompts,
            questions=questions,
            generated_responses=base_outputs,
            device=self.device,
            normalize_by_length=True,
            prompt_template=self.prompt_template,
        )

        ratio = large_probs_for_base / (base_probs * self.M)
        acceptance_prob = torch.clip(ratio, 0.0, 1.0)

        # (2) Uncertainties for each model's own output
        base_uncertainties = self.uncertainty_fn(
            model=self.base_model,
            tokenizer=self.base_tokenizer,
            prompts=prompts,
            generated_responses=base_outputs,
            device=self.device,
            normalize_by_length=True,
        )
        large_uncertainties = self.uncertainty_fn(
            model=self.large_model,
            tokenizer=self.large_tokenizer,
            prompts=prompts,
            generated_responses=large_outputs,
            device=self.device,
            normalize_by_length=True,
        )

        decisions = []
        base_model_predictions = []
        large_model_predictions = []
        # For potential online updates
        dM = 0.0
        dTau = 0.0

        for i, prompt in enumerate(prompts):
            base_response = base_outputs[i].strip()
            large_response = large_outputs[i].strip()

            alpha_i = acceptance_prob[i].item()
            base_uncert = base_uncertainties[i].item()
            large_uncert = large_uncertainties[i].item()

            base_model_predictions.append(base_response)
            large_model_predictions.append(large_response)

            u = random.random()
            final_decision = None
            cost_used = None
            final_answer = None

            if u < alpha_i:
                # Accept small model unless too uncertain
                if base_uncert > self.tau_base:
                    # Escalate to Expert
                    final_decision = 2
                    final_answer = expert_responses[i][0]
                    cost_used = (
                        self.base_gen_cost + self.large_inf_cost + self.expert_cost
                    )
                else:
                    # Accept small model
                    final_decision = 0
                    final_answer = base_response
                    cost_used = self.base_gen_cost + self.large_inf_cost
            else:
                # Use large model
                if large_uncert > self.tau_large:
                    # Escalate to Expert
                    final_decision = 2
                    final_answer = expert_responses[i][0]
                    cost_used = (
                        self.base_gen_cost
                        + self.large_inf_cost
                        + self.large_gen_cost
                        + self.expert_cost
                    )
                else:
                    # Accept large model
                    final_decision = 1
                    final_answer = large_response
                    cost_used = (
                        self.base_gen_cost + self.large_inf_cost + self.large_gen_cost
                    )

            # Evaluate correctness if gold_labels is provided
            is_correct = True
            if gold_labels is not None:
                # Simple check: final_answer vs gold_labels[i]
                # You might have a more sophisticated matching approach
                gold = gold_labels[i]
                is_correct = final_answer.strip() == gold.strip()

            # Append final info
            decisions.append(
                {
                    "decision": final_decision,
                    "response": final_answer,
                    "cost": cost_used,
                    "is_correct": is_correct,
                    "base_response": base_response,
                    "large_response": large_response,
                    "base_uncertainty": base_uncert,
                    "large_uncertainty": large_uncert,
                    "acceptance_prob": alpha_i,
                }
            )

            # --- Heuristic updates to M and tau ---
            if gold_labels is not None:
                # Let's define a simplified cost measure:
                # "Wrong" => error_penalty
                # plus the actual cost of the chosen route.
                total_cost = cost_used + (0 if is_correct else self.error_penalty)

                # Very naive partial derivatives or sign-based updates:
                if final_decision == 0:  # used small model
                    if not is_correct:
                        # We incorrectly accepted the small model => decrease M
                        # because ratio_i was too high => acceptance probability was too large
                        dM -= 1.0
                        # Also might want to lower tau so we escalate more often
                        dTau -= 0.5
                elif final_decision == 1:  # used large model
                    if is_correct:
                        # Possibly we wasted cost if the small model might have done fine
                        # This is ambiguous, but let's say we "increase M" so next time
                        # we are more likely to accept the small model.
                        dM += 0.5
                    else:
                        # Large model also got it wrong => uncertain => reduce tau
                        dTau -= 1.0
                elif final_decision == 2:  # expert
                    # If we ended up at the expert, maybe it's correct but very expensive
                    # If we want fewer expert calls, we could reduce tau or reduce M
                    # if it was due to small model's high uncertainty. Let's do:
                    dTau -= 0.1

        # After going through all samples in this batch, we do an update
        if gold_labels is not None:
            # Scale by the learning rate
            self.M = max(1e-3, self.M - self.lr_M * dM)  # keep M>0
            self.tau = max(0.0, self.tau - self.lr_tau * dTau)

        return decisions, base_model_predictions, large_model_predictions
