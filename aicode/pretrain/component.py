import logging
from typing import cast

from cotrain import Trainer, after

from aicode.pretrain.data import PretrainDataset

logger = logging.getLogger(__name__)


class PretrainDatasetRefresh:
    @after(Trainer.run_epoch)
    def refresh(self, trainer: Trainer) -> None:
        if trainer.current_epoch != 0:
            logger.info('Refresh dataset')
            dataset = trainer.train_dataloader.dataset
            dataset = cast(PretrainDataset, dataset)
            dataset.refresh()
