import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig


@hydra.main(config_path="conf", config_name="config", version_base=None)
def train(cfg: DictConfig) -> None:
    datamodule = instantiate(cfg.datamodule)
    datamodule.setup()

    model = instantiate(cfg.model)

    logger = instantiate(cfg.trainer.logger)
    logger.log_hyperparams(model.hparams)

    trainer = instantiate(cfg.trainer, logger=logger)
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    train()
