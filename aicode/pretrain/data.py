from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Union

import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from aicode.core import Cell, Notebook


@dataclass
class PretrainSample:
    cell: Cell
    other_cell: Cell
    is_next: bool
    is_same_group: bool


def generate_pretrain_sample_from_notebook(
    notebook: Notebook, num_negative_samples: int = 2
) -> list[PretrainSample]:
    pretrain_samples = []

    cell_groups: list[list[Cell]] = []
    temp_group: list[Cell] = []
    cell_id_group_idx_map: dict[str, int] = {}
    for cell, next_cell in zip(notebook.cells, notebook.cells[1:]):
        if cell.is_code and next_cell.is_markdown:
            temp_group.append(cell)
            cell_id_group_idx_map[cell.id] = len(cell_groups)
            cell_groups.append(temp_group)
            temp_group = []
        else:
            temp_group.append(cell)
            cell_id_group_idx_map[cell.id] = len(cell_groups)
    if temp_group:
        cell_groups.append(temp_group)

    for cell in notebook.cells[:-1]:
        next_cell = notebook.cells[cell.rank + 1]
        assert next_cell.rank == cell.rank + 1

        # next cell
        if not (cell.is_code and next_cell.is_markdown):
            pretrain_samples.append(
                PretrainSample(
                    cell=cell,
                    other_cell=next_cell,
                    is_next=True,
                    is_same_group=True,
                )
            )

        pretrain_samples.append(
            PretrainSample(
                cell=next_cell,
                other_cell=cell,
                is_next=False,
                is_same_group=True,
            )
        )

        # same group
        group_idx = cell_id_group_idx_map[cell.id]
        cell_group = cell_groups[group_idx]
        if len(cell_group) > 4:
            random_same_group_cell = random.choice(cell_groups[group_idx])
            while (random_same_group_cell.id == cell.id) or (
                random_same_group_cell.id == next_cell.id
            ):
                random_same_group_cell = random.choice(cell_groups[group_idx])
            pretrain_samples.append(
                PretrainSample(
                    cell=cell,
                    other_cell=random_same_group_cell,
                    is_next=False,
                    is_same_group=True,
                )
            )

        other_group_indexes = list(set(range(len(cell_groups))) - {group_idx})
        if not other_group_indexes:
            continue
        for _ in range(num_negative_samples):
            # other cell group
            other_cell_group_idx = random.choice(other_group_indexes)
            other_cell_group = cell_groups[other_cell_group_idx]
            random_other_group_cell = random.choice(other_cell_group)
            pretrain_samples.append(
                PretrainSample(
                    cell=cell,
                    other_cell=random_other_group_cell,
                    is_next=False,
                    is_same_group=False,
                )
            )
    return pretrain_samples


class PretrainDataset(Dataset[PretrainSample]):
    pretrain_samples: list[PretrainSample]

    def __init__(
        self, notebooks: list[Notebook], num_negative_samples: int = 2
    ) -> None:
        super().__init__()
        self.notebooks = notebooks
        self.num_negative_samples = num_negative_samples
        self.refresh()

    def generate_pretrain_samples(self) -> None:
        for notebook in tqdm(self.notebooks):
            self.pretrain_samples.extend(
                generate_pretrain_sample_from_notebook(
                    notebook, self.num_negative_samples
                )
            )

    def refresh(self) -> None:
        self.pretrain_samples = []
        self.generate_pretrain_samples()

    def __len__(self) -> int:
        return len(self.pretrain_samples)

    def __getitem__(self, idx: int) -> PretrainSample:
        return self.pretrain_samples[idx]


class PretrainCollator:
    def __init__(
        self,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        max_length: int = 256,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def collate_fn(self, samples: list[PretrainSample]):
        text = []
        for sample in samples:
            source = sample.cell.source
            if len(source) <= 300:
                text.append(source)
            else:
                text.append(source[:150] + source[-150:])
        text_pair = []
        for sample in samples:
            source = sample.other_cell.source
            if len(source) <= 300:
                text_pair.append(source)
            else:
                text_pair.append(source[:150] + source[-150:])
        encodes = self.tokenizer(
            text,
            text_pair,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_token_type_ids=True,
        )
        next_labels = torch.tensor(
            [sample.is_next for sample in samples], dtype=torch.long
        )
        same_group_labels = torch.tensor(
            [sample.is_same_group for sample in samples], dtype=torch.long
        )
        encodes['next_labels'] = next_labels
        encodes['same_group_labels'] = same_group_labels
        return encodes
