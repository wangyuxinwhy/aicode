from __future__ import annotations

from pathlib import Path
from typing import Union

import pandas as pd
from datasets import load_from_disk
from sklearn.model_selection import GroupShuffleSplit

from aicode.core import Notebook


def get_notebook_id_cell_order_map(file_path: Union[str, Path]):
    file_path = Path(file_path)
    df = pd.read_csv(file_path)
    _map: dict[str, list[str]] = {}
    for notebook_id, cell_order_str in zip(df['id'], df['cell_order']):
        _map[notebook_id] = cell_order_str.split(' ')
    return _map


def load_notebooks_from_disk(aicode_dataset_path: Union[str, Path]):
    return [
        Notebook.from_dict(i) for i in load_from_disk(str(aicode_dataset_path))  # type: ignore
    ]


def clean_code(code: str):
    return str(code).replace('\\n', '\n')


def split_notebooks(
    notebooks: list[Notebook], test_size: float = 0.1, random_seed: int = 42
):
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_seed
    )
    train_ind, val_ind = next(
        splitter.split(notebooks, groups=[i.ancestor_id for i in notebooks])
    )
    train_notebooks = [notebooks[i] for i in train_ind]
    val_notebooks = [notebooks[i] for i in val_ind]
    return train_notebooks, val_notebooks
