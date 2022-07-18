import logging
from typing import cast

import torch
from cotrain import Component, Trainer, after

from aicode.pretrain.data import PretrainDataset

logger = logging.getLogger(__name__)


class PretrainDatasetRefresh(Component):
    only_main_process = True

    @after(Trainer.run_epoch)
    def refresh(self, trainer: Trainer) -> None:
        if trainer.current_epoch != 0:
            logger.info('Refresh dataset')
            dataset = trainer.train_dataloader.dataset
            dataset = cast(PretrainDataset, dataset)
            dataset.refresh()


class SaveModel(Component):
    only_main_process = True

    @after(Trainer.run_epoch)
    def save(self, trainer: Trainer) -> None:
        torch.save(
            trainer.raw_model,
            trainer.output_dir / f'model-epoch-{trainer.current_epoch}.pt',
        )
