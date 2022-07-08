import logging
from pathlib import Path
from typing import Optional, Union

import torch
import yaml
from cotrain import Trainer, TrainerConfig
from cotrain.components import RichInspect, RichProgressBar
from cotrain.utils.torch import seed_all
from pydantic import BaseModel
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from yaml.loader import SafeLoader

from aicode.reg_with_code.component import Scorer
from aicode.reg_with_code.data import RegWithCodeCollator, RegWithCodeDataset
from aicode.reg_with_code.model import RegWithCodeModel
from aicode.utils import load_notebooks_from_disk, split_notebooks

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


class DataConfig(BaseModel):
    data_path: Path
    batch_size: int = 8
    num_code_per_md: int = 1
    max_samples: Optional[int] = None
    dynamic_sample: bool = False
    md_max_length: int = 128
    total_max_length: int = 512
    test_size: float = 0.1


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

    # Data
    collator = RegWithCodeCollator(
        tokenizer, cfg.data.md_max_length, cfg.data.total_max_length
    )
    logger.info('Loading notebooks from %s', cfg.data.data_path)
    notebooks = load_notebooks_from_disk(cfg.data.data_path)
    train_notebooks, valid_notebooks = split_notebooks(
        notebooks, test_size=cfg.data.test_size, random_seed=cfg.seed
    )
    train_dataset = RegWithCodeDataset(
        train_notebooks,
        cfg.data.num_code_per_md,
        cfg.data.dynamic_sample,
        cfg.data.max_samples,
    )
    valid_dataset = RegWithCodeDataset(
        valid_notebooks, cfg.data.num_code_per_md, False, None
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.data.batch_size,
        collate_fn=collator.collate_fn,
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=cfg.data.batch_size,
        collate_fn=collator.collate_fn,
    )

    # Model
    logger.info('Loading model from %s', cfg.pretrained_model_path)
    model = RegWithCodeModel(cfg.pretrained_model_path)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    # compoents
    components = [
        RichInspect(),
        RichProgressBar(),
        Scorer(),
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
    debug = False
    if debug:
        config = ExperimentConfig(
            data=DataConfig(data_path=Path('../dataset/aicode-debug-100')),
            trainer=TrainerConfig(),
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
