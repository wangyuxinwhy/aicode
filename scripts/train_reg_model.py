from pathlib import Path
from typing import Optional, Union

import torch
import yaml
from cotrain import Trainer, TrainerConfig
from cotrain.components import RichInspect, RichProgressBar
from pydantic import BaseModel
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from yaml.loader import SafeLoader

from aicode.reg_with_code.data import RegWithCodeCollator, RegWithCodeDataset
from aicode.reg_with_code.model import RegWithCodeModel
from aicode.utils import load_notebooks_from_disk


class DataConfig(BaseModel):
    batch_size: int = 8
    num_code_per_md: int = 1
    max_samples: Optional[int] = None
    md_max_length: int = 128
    total_max_length: int = 512


class ExperimentConfig(BaseModel):
    data: DataConfig
    trainer: TrainerConfig
    pretrained_model_path: str = 'microsoft/codebert-base'
    lr: float = 5e-5

    @classmethod
    def from_yaml(cls, yaml_file: Union[str, Path]):
        with open(yaml_file) as f:
            data = yaml.load(f, Loader=SafeLoader)
        data_config = DataConfig.parse_obj(data.pop('data'))
        trainer_config = TrainerConfig.parse_obj(data.pop('trainer'))
        return cls(data=data_config, trainer=trainer_config, **data)


def main(cfg: ExperimentConfig):

    # tokenzier
    tokenizer = AutoTokenizer.from_pretrained(cfg.pretrained_model_path)

    # Data
    collator = RegWithCodeCollator(
        tokenizer, cfg.data.md_max_length, cfg.data.total_max_length
    )
    notebooks = load_notebooks_from_disk('dataset/aicode-debug')
    train_dataset = RegWithCodeDataset(
        notebooks, cfg.data.num_code_per_md, cfg.data.max_samples
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.data.batch_size,
        collate_fn=collator.collate_fn,
    )

    # Model
    model = RegWithCodeModel(cfg.pretrained_model_path)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    # compoents
    components = [
        RichInspect(),
        RichProgressBar(),
    ]

    # trainer
    trainer = Trainer(
        model,
        optimizer,
        train_dataloader,
        components=components,
        config=cfg.trainer,
    )
    trainer.train()


if __name__ == '__main__':
    import argparse

    debug = True
    if debug:
        config = ExperimentConfig.from_yaml(
            '/Users/wangyuxin/workspace/kaggle/aicode/configs/debug.yaml'
        )
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            'config', type=str, help='path to yaml config file'
        )
        args = parser.parse_args()
        config = ExperimentConfig.from_yaml(args.config)
    main(config)