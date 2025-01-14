from dataclasses import dataclass
from src.uncertainty import per_token_entropy, verdict_distribution_entropy
from src.verification import verbalisation, sequence_probability, surrogate_token_probs

VERIFICATION_FN_MAPPING = {
    "surrogate_token_probs": surrogate_token_probs,
    "sequence_probability": sequence_probability,
    "verbalisation": verbalisation,
}

UNCERTAINTY_FN_MAPPING = {
    "per_token_entropy": per_token_entropy,
    "verdict_distribution_entropy": verdict_distribution_entropy
}


@dataclass
class ModelConfig:
    """Configuration for model initialization and generation."""

    max_input_length: int = 512
    max_new_tokens: int = 256
    device: str = "cpu"
    prompt_template: str = ""
    precomputed: bool = False


@dataclass
class CostConfig:
    """Configuration for different operation costs."""

    base_gen_cost: float
    large_gen_cost: float
    large_inf_cost: float
    expert_cost: float


@dataclass
class ExperimentConfig:
    """Configuration for experiment setup."""

    base_model: str
    large_model: str
    verification_fn: str
    uncertainty_fn: str
    max_input_length: int
    max_new_tokens: int
    device: str
    precomputed: bool

    def __post_init__(self):
        self.verification_fn = VERIFICATION_FN_MAPPING[self.verification_fn]
        self.uncertainty_fn = UNCERTAINTY_FN_MAPPING[self.uncertainty_fn]


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
