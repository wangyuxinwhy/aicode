import logging
from pathlib import Path
from typing import Union

import torch
import yaml
from cotrain import Trainer, TrainerConfig
from cotrain.components import RichInspect, RichProgressBar
from cotrain.utils.torch import seed_all
from pydantic import BaseModel
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from yaml.loader import SafeLoader

from aicode.pretrain.component import PretrainDatasetRefresh
from aicode.pretrain.data import PretrainCollator, PretrainDataset
from aicode.pretrain.model import PretrainAiCodeModel
from aicode.utils import load_notebooks_from_disk, split_notebooks

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO
)
logger = logging.getLogger(__name__)


class DataConfig(BaseModel):
    data_path: Path
    batch_size: int = 64
    num_negtive_sampels: int = 2
    max_length: int = 256
    test_size: float = 0.05


class ExperimentConfig(BaseModel):
    data: DataConfig
    trainer: TrainerConfig
    pretrained_model_path: str = 'microsoft/codebert-base'
    lr: float = 5e-5
    seed: int = 42

    @classmethod
    def from_yaml(cls, yaml_file: Union[str, Path]):
        with open(yaml_file) as f:
            data = yaml.load(f, Loader=SafeLoader)
        data_config = DataConfig.parse_obj(data.pop('data'))
        trainer_config = TrainerConfig.parse_obj(data.pop('trainer'))
        return cls(data=data_config, trainer=trainer_config, **data)


def main(cfg: ExperimentConfig):
    seed_all(cfg.seed)
    logger.info(f'Start Training, config:\n {cfg}')

    # tokenzier
    tokenizer = AutoTokenizer.from_pretrained(cfg.pretrained_model_path)

    # data
    collator = PretrainCollator(tokenizer, cfg.data.max_length)
    logger.info('Loading notebooks from %s', cfg.data.data_path)
    notebooks = load_notebooks_from_disk(cfg.data.data_path)
    train_notebooks, valid_notebooks = split_notebooks(
        notebooks, test_size=cfg.data.test_size, random_seed=cfg.seed
    )
    train_dataset = PretrainDataset(
        train_notebooks, cfg.data.num_negtive_sampels
    )
    valid_dataset = PretrainDataset(
        valid_notebooks, cfg.data.num_negtive_sampels
    )
    train_dataloader = DataLoader(
        train_dataset,
        collate_fn=collator.collate_fn,
        batch_size=cfg.data.batch_size,
        shuffle=True,
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        collate_fn=collator.collate_fn,
        batch_size=cfg.data.batch_size,
    )

    # Model
    logger.info('Loading model from %s', cfg.pretrained_model_path)
    model = PretrainAiCodeModel(cfg.pretrained_model_path)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    # compoents
    components = [
        RichInspect(),
        RichProgressBar(),
        PretrainDatasetRefresh(),
    ]
    # trainer
    trainer = Trainer(
        model,
        optimizer,
        train_dataloader,
        valid_dataloader,
        components=components,
        config=cfg.trainer,
    )
    trainer.train()


if __name__ == '__main__':
    debug = True
    if debug:
        config = ExperimentConfig(
            data=DataConfig(data_path=Path('dataset/aicode-debug-100')),
            trainer=TrainerConfig(),
            pretrained_model_path='cross-encoder/ms-marco-TinyBERT-L-2',
        )
    else:
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument(
            'config', type=str, help='path to yaml config file'
        )
        args = parser.parse_args()
        config = ExperimentConfig.from_yaml(args.config)
    main(config)
