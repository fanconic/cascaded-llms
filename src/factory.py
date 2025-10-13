from typing import Dict
from src.decision_maker import AIDecisionSystem
from src.config import ExperimentConfig, OnlineConfig, ModelConfig, CostConfig


class DecisionMakerFactory:
    """Factory for creating appropriate decision maker instance."""

    @staticmethod
    def create_decision_maker(
        base_model,
        base_tokenizer,
        large_model,
        large_tokenizer,
        exp_config: ExperimentConfig,
        online_config: OnlineConfig,
        cost_config: Dict[str, float],
    ) -> AIDecisionSystem:

        model_config = ModelConfig(
            max_input_length=exp_config.max_input_length,
            max_new_tokens=exp_config.max_new_tokens,
            device=exp_config.device,
            precomputed=exp_config.precomputed,
            uncertainty_samples=exp_config.uncertainty_samples,
            batch_size=exp_config.batch_size,
            use_larger_model=exp_config.use_larger_model,
            experiment=exp_config.experiment
        )

        cost_config_obj = CostConfig(**cost_config)

        online_config_obj = OnlineConfig(
            enable=online_config.enable,
            initial_xi_base=online_config.initial_xi_base,
            initial_xi_large=online_config.initial_xi_large,
            initial_phi_base=online_config.initial_phi_base,
            lr_tau=online_config.lr_tau,
            lr_M=online_config.lr_M,
            error_penalty=online_config.error_penalty,
        )

        return AIDecisionSystem(
            base_model=base_model,
            base_tokenizer=base_tokenizer,
            large_model=large_model,
            large_tokenizer=large_tokenizer,
            base_model_path=exp_config.base_model,
            large_model_path=exp_config.large_model,
            verification_fn=exp_config.verification_fn,
            uncertainty_fn=exp_config.uncertainty_fn,
            model_config=model_config,
            cost_config=cost_config_obj,
            online_config=online_config_obj,
            use_larger_model=exp_config.use_larger_model,
        )
