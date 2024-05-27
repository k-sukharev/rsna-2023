import logging

import hydra

from pathlib import Path

from hydra.utils import instantiate
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from omegaconf import DictConfig, OmegaConf


logger = logging.getLogger(__name__)


@rank_zero_only
def prepare_data(dm):
    dm.prepare_data()


@hydra.main(version_base=None, config_path='../configs', config_name='config')
def main(cfg : DictConfig) -> None:
    instantiate(
        cfg.seed_everything
    )
    logger.info('Config:\n%s', OmegaConf.to_yaml(cfg))

    working_dir = Path('.').resolve()
    logger.info('Working directory: %s', working_dir)
    logger.info('Instantiating datamodule')
    dm = instantiate(
        cfg.datamodule
    )
    logger.info('Preparing data')
    prepare_data(dm)
    dm.setup(stage='fit')

    logger.info('Instantiating model')
    model = instantiate(
        cfg.model
    )

    logger.info('Instantiating trainer')
    trainer = instantiate(
        cfg.trainer
    )
    logger.info('Training model')
    trainer.fit(model, datamodule=dm)

if __name__ == '__main__':
    main()
