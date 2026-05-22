import hydra
from omegaconf import DictConfig


# config_path adds conf/ to the Hydra searchpath automatically
@hydra.main(config_path="conf", config_name="config", version_base=None)
def train(cfg: DictConfig) -> None:
    pass


if __name__ == "__main__":
    train()
