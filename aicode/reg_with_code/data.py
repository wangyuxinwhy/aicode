from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Iterable, Optional, Union

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from aicode.core import Notebook
from aicode.utils import clean_code

logger = logging.getLogger(__name__)


@dataclass
class RegSample:
    notebook_id: str
    cell_id: str
    markdown: str
    codes: list[str]
    rank: int
    num_cells: int
    num_codes: int
    num_markdowns: int


class RegWithCodeDataset(Dataset[RegSample]):
    def __init__(
        self,
        notebooks: Iterable[Notebook],
        num_code_per_md: int = 10,
        dynamic_sample: bool = False,
        max_samples: Optional[int] = None,
    ):
        self.notebooks = notebooks
        self.num_code_per_md = num_code_per_md
        self.dynamic_sample = dynamic_sample
        self.max_samples = max_samples
        self.notebook_id_codes_map = self.sample_code()
        self.reg_samples = self.generate_reg_samples()

    def sample_code(self) -> dict[str, list[str]]:
        logger.info('Sampling code from notebooks')
        notebook_id_codes_map = {}
        for notebook in self.notebooks:
            all_code_source = [
                cell.source for cell in notebook.cells if cell.is_code
            ]
            if self.dynamic_sample:
                codes = self.dynamic_sample_codes(
                    all_code_source, self.num_code_per_md
                )
            else:
                codes = self.sample_codes(
                    all_code_source, self.num_code_per_md
                )
            notebook_id_codes_map[notebook.id] = codes
        return notebook_id_codes_map

    def generate_reg_samples(self) -> list[RegSample]:
        logger.info('Generating reg samples...')
        reg_samples = []
        count = 0
        for notebook in self.notebooks:
            for cell in notebook.cells:
                num_cells = len(notebook.cells)
                num_codes = sum([i.is_code for i in notebook.cells])
                num_markdowns = num_cells - num_codes
                if cell.is_markdown:
                    reg_sample = RegSample(
                        notebook_id=notebook.id,
                        cell_id=cell.id,
                        markdown=cell.source,
                        codes=self.notebook_id_codes_map[notebook.id],
                        rank=cell.rank,
                        num_cells=num_cells,
                        num_codes=num_codes,
                        num_markdowns=num_markdowns,
                    )
                    reg_samples.append(reg_sample)
                    count += 1
                    if (self.max_samples is not None) and (
                        count >= self.max_samples
                    ):
                        return reg_samples
        return reg_samples

    def refresh(self):
        if self.dynamic_sample:
            self.notebook_id_codes_map = self.sample_code()
            self.reg_samples = self.generate_reg_samples()

    @staticmethod
    def sample_codes(codes: list[str], n: int) -> list[str]:
        codes = [clean_code(code) for code in codes]
        if n >= len(codes):
            return [code[:200] for code in codes]
        else:
            results = []
            step = len(codes) / n
            idx = 0
            while int(np.round(idx)) < len(codes):
                results.append(codes[int(np.round(idx))])
                idx += step
            if codes[-1] not in results:
                results[-1] = codes[-1]
            return results

    @staticmethod
    def dynamic_sample_codes(codes: list[str], n: int) -> list[str]:
        codes = [clean_code(code)[:200] for code in codes]
        if n >= len(codes):
            return [code for code in codes]
        else:
            results = []
            step = len(codes) // n
            idx = 0
            while (idx + step) < len(codes):
                sampled_idx = random.randint(idx, idx + step)
                results.append(codes[sampled_idx])
                idx += step
            if codes[-1] != results[-1]:
                results[-1] = codes[-1]
            return results

    def __getitem__(self, index: int) -> RegSample:
        return self.reg_samples[index]

    def __len__(self) -> int:
        return len(self.reg_samples)


class RegWithCodeCollator:
    def __init__(
        self,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        md_max_length: int = 128,
        total_max_length: int = 512,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.md_max_length = md_max_length
        self.total_max_length = total_max_length
        self.max_num_code_tokens = total_max_length - md_max_length - 3

    def collate_fn(
        self, reg_samples: list[RegSample]
    ) -> dict[str, torch.Tensor]:
        sample_tokens = []
        for reg_sample in reg_samples:
            sample_tokens.append(self.tokenize_reg_sample(reg_sample))
        max_len = max([len(tokens) for tokens in sample_tokens])
        # pad
        attention_mask = torch.zeros(
            (len(sample_tokens), max_len), dtype=torch.long
        )
        for idx, tokens in enumerate(sample_tokens):
            tokens.extend(
                [self.tokenizer.pad_token_id] * (max_len - len(tokens))
            )
            attention_mask[idx, : len(tokens)] = 1
        input_ids = torch.tensor(sample_tokens, dtype=torch.long)
        md_ratios = torch.tensor(
            [self.get_md_ratio(reg_sample) for reg_sample in reg_samples],
            dtype=torch.float,
        )
        scores = torch.tensor(
            [self.get_reg_score(reg_sample) for reg_sample in reg_samples],
            dtype=torch.float,
        )
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'md_ratios': md_ratios,
            'scores': scores,
        }

    @staticmethod
    def get_reg_score(reg_sample: RegSample) -> float:
        return reg_sample.rank / reg_sample.num_cells

    @staticmethod
    def get_md_ratio(reg_sample: RegSample) -> float:
        if reg_sample.num_markdowns == 0:
            return 0.0
        return reg_sample.num_markdowns / reg_sample.num_cells

    def tokenize_reg_sample(self, sample: RegSample) -> list[int]:
        markdown_tokens = self.tokenizer.encode(
            sample.markdown, add_special_tokens=False
        )[: self.md_max_length]

        _code_tokens: list[list[int]] = self.tokenizer.batch_encode_plus(
            sample.codes,
            return_attention_mask=False,
            return_token_type_ids=False,
            add_special_tokens=False,
        )[
            'input_ids'
        ]  # type: ignore

        code_tokens: list[int] = []
        for i in _code_tokens:
            code_tokens.extend(i)
        code_tokens = code_tokens[: self.max_num_code_tokens]

        return (
            [self.tokenizer.cls_token_id]  # type: ignore
            + markdown_tokens
            + [self.tokenizer.pad_token_id]
            + code_tokens
            + [self.tokenizer.sep_token_id]
        )   # type: ignore
