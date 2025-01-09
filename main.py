"""Main script for running AI decision making experiments with online/offline variants."""

import hydra
import scienceplots
from omegaconf import DictConfig
from src.experiment import Experiment
from src.utils import set_seed


@hydra.main(config_path="configs", config_name="debug", version_base="1.3")
def main(cfg: DictConfig):
    set_seed(cfg.seed)
    experiment = Experiment(cfg)
    experiment.run()


if __name__ == "__main__":
    main()
