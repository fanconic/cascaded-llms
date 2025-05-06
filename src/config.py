from dataclasses import dataclass
from typing import Dict, Callable
from src.verification import self_verification, surrogate_token_probs

VERIFICATION_FN_MAPPING = {
    "surrogate_token_probs": surrogate_token_probs,
    "self_verification": self_verification,
}

@dataclass
class ModelConfig:
    """Configuration for model initialization and generation."""

    precomputed: Dict
    max_input_length: int = 512
    max_new_tokens: int = 256
    device: str = "cpu"
    prompt_template: str = ""
    uncertainty_samples: int = 1
    batch_size: int = 5
    use_larger_model: bool = True


@dataclass
class CostConfig:
    """Configuration for different operation costs."""

    base_gen_cost: float
    large_gen_cost: float
    large_inf_cost: float
    expert_cost: float
    mistake_cost: float
    output_input_price_ratio: float


@dataclass
class ExperimentConfig:
    """Configuration for experiment setup."""

    base_model: str
    large_model: str
    verification_fn: Callable
    uncertainty_fn: Callable
    max_input_length: int
    max_new_tokens: int
    device: str
    use_larger_model: bool
    precomputed: Dict
    uncertainty_samples: int = 1
    batch_size: int = 5
    


@dataclass
class OnlineConfig:
    """Configuration for online learning."""

    enable: bool
    initial_uncertainty_threshold_base: float
    initial_uncertainty_threshold_large: float
    initial_M: float
    lr_tau: float
    lr_M: float
    error_penalty: float
