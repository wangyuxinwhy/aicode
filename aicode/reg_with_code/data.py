from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, Optional, Union

import numpy as np
import torch
from torch.utils.data import IterableDataset
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


class RegWithCodeDataset(IterableDataset):
    def __init__(
        self,
        notebooks: Iterable[Notebook],
        num_code_per_md: int,
        max_samples: Optional[int] = None,
    ):
        self.notebooks = notebooks
        self.num_code_per_md = num_code_per_md
        self.max_samples = max_samples or self.get_num_samples()

    def get_num_samples(self) -> int:
        logger.info('num_samples is not specified, try to calculate it')
        count = 0
        for notebook in self.notebooks:
            for cell in notebook.cells:
                if cell.is_markdown:
                    count += 1
        return count

    def generate_reg_samples(self):
        count = 0
        for notebook in self.notebooks:
            all_code_source = [
                cell.source for cell in notebook.cells if cell.is_code
            ]
            for cell in notebook.cells:
                if cell.is_markdown:
                    codes = self.sample_codes(
                        all_code_source, self.num_code_per_md
                    )
                    reg_sample = RegSample(
                        notebook_id=notebook.id,
                        cell_id=cell.id,
                        markdown=cell.source,
                        codes=codes,
                        rank=cell.rank,
                        num_cells=len(notebook.cells),
                        num_codes=len(all_code_source),
                        num_markdowns=len(notebook.cells)
                        - len(all_code_source),
                    )
                    yield reg_sample
                    count += 1
                    if count >= self.max_samples:
                        return

    def __iter__(self):
        return self.generate_reg_samples()

    def __len__(self):
        return self.max_samples

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
