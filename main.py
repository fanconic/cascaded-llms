"""Main script for running AI decision making experiments with online/offline variants."""

import hydra
import scienceplots
from omegaconf import DictConfig
from src.experiment_first import Experiment_first
from src.experiment_second import Experiment_second
from src.utils import set_seed


@hydra.main(config_path="configs", config_name="debug", version_base="1.3")
def main(cfg: DictConfig):
    set_seed(cfg.seed)
    if cfg.experiment == "first":
        experiment = Experiment_first(cfg)
    elif cfg.experiment == "second":
        experiment = Experiment_second(cfg)
    experiment.run()


if __name__ == "__main__":
    main()
